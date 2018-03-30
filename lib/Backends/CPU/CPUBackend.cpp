// Copyright 2017 Facebook Inc.  All Rights Reserved.

#define DEBUG_TYPE "jit"
#include "CPUBackend.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/Instrs.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace glow;
using llvm::StringRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

static llvm::cl::opt<std::string> target("target", llvm::cl::desc("target"));

CPUBackend::CPUBackend(IRFunction *F)
    : F_(F), irgen_(F_, allocationsInfo_, "") {}

CPUBackend::~CPUBackend() {
  clear();
  alignedFree(heap_);
}

void CPUBackend::clear() { F_->clear(); }

//===----------------------------------------------------------------------===//
//                   Functions for executing code using JIT
//===----------------------------------------------------------------------===//

/// Emit the entry point for JIT called "jitmain". It simply calls the main
/// entry of the module with the constant concrete addresses of all the memory
/// areas. Since these addresses are constants, the LLVM optimizer will constant
/// propagate them into relative addressing computations and the like and
/// produce a very efficient code that uses absolute addressing whenever
/// possible.
void CPUBackend::emitJitMain() {
  llvm::Type *voidTy = llvm::Type::getVoidTy(irgen_.getLLVMContext());
  llvm::FunctionType *jitFuncTy = llvm::FunctionType::get(voidTy, {}, false);
  auto *func =
      llvm::Function::Create(jitFuncTy, llvm::Function::ExternalLinkage,
                             "jitmain", &irgen_.getModule());
  llvm::BasicBlock *entry_bb =
      llvm::BasicBlock::Create(irgen_.getLLVMContext(), "entry", func);
  llvm::IRBuilder<> builder(entry_bb);

  // Prepare arguments for the "main" function.
  llvm::SmallVector<llvm::Value *, 4> initFunctionCallArgs;
  // Get the integer type having the same size in bits as size_t.
  auto *sizeTType = builder.getIntNTy(sizeof(size_t) * 8);
  auto int8PtrTy = llvm::Type::getInt8PtrTy(irgen_.getLLVMContext());

  initFunctionCallArgs.push_back(builder.CreateIntToPtr(
      llvm::ConstantInt::get(
          sizeTType, reinterpret_cast<size_t>(
                         allocationsInfo_.baseConstantWeightVarsAddress_)),
      int8PtrTy));
  initFunctionCallArgs.push_back(builder.CreateIntToPtr(
      llvm::ConstantInt::get(
          sizeTType, reinterpret_cast<size_t>(
                         allocationsInfo_.baseMutableWeightVarsAddress_)),
      int8PtrTy));
  initFunctionCallArgs.push_back(builder.CreateIntToPtr(
      llvm::ConstantInt::get(
          sizeTType,
          reinterpret_cast<size_t>(allocationsInfo_.baseActivationsAddress_)),
      int8PtrTy));
  // Now form the offsets array and pass it as the last argument.
  auto offsetsArray =
      irgen_.emitConstOffsetsArray(irgen_.getBuilder(), allocationsInfo_);
  initFunctionCallArgs.push_back(offsetsArray);
  // Invoke the main entry with constant arguments and let LLVM optimizer make
  // use of it.
  auto *entryF = irgen_.getModule().getFunction(irgen_.getMainEntryName());
  entryF->setLinkage(llvm::Function::InternalLinkage);
  builder.CreateCall(entryF, initFunctionCallArgs);
  // Terminate the function.
  builder.CreateRetVoid();
}

void CPUBackend::performJITMemoryAllocation() {
  allocationsInfo_.clear();
  allocationsInfo_.numberValues(F_);
  allocationsInfo_.allocateActivations(F_);
  // Tell the allocateWeightVars to reuse existing addresses for weights.
  allocationsInfo_.allocateWeightVars(F_, true);

  // Allocate the heap to match the max memory usage for activations.
  if (allocationsInfo_.activationsMemSize_ > 0) {
    alignedFree(heap_);
    heap_ = alignedAlloc(allocationsInfo_.activationsMemSize_, TensorAlignment);
    allocationsInfo_.baseActivationsAddress_ = (uint8_t *)heap_;
  }
}

void CPUBackend::init() {
  irgen_.initTargetMachine(target.empty() ? "" : target.getValue(),
                           llvm::CodeModel::Model::Large);
  JIT_ = llvm::make_unique<llvm::orc::GlowJIT>(irgen_.getTargetMachine());
  irgen_.initCodeGen();
  // Perform the address assignment for activations and WeightVars.
  performJITMemoryAllocation();
  // Create the jitmain function to be invoked by JIT.
  emitJitMain();
  // Emit the code for the body of the entry function.
  irgen_.performCodeGen();
  // Hand over the module to JIT for the machine code generation.
  JIT_->addModule(irgen_.borrowModule());
}

void CPUBackend::doForwardPass() {
  auto sym = JIT_->findSymbol("jitmain");
  assert(sym && "Unable to JIT the code!");
  using JitFuncType = void (*)(void);
  auto address = sym.getAddress();
  if (address) {
    JitFuncType funcPtr = reinterpret_cast<JitFuncType>(address.get());
    funcPtr();
  } else {
    GLOW_ASSERT(false && "Error getting address.");
  }
}

//===----------------------------------------------------------------------===//
//                   Functions for saving bundles
//===----------------------------------------------------------------------===//

void CPUBackend::saveWeights(llvm::StringRef weightsFileName) {
  std::error_code EC;
  llvm::raw_fd_ostream weightsFile(weightsFileName, EC, llvm::sys::fs::F_None);
  GLOW_ASSERT(!EC &&
              "Could not open the output file for saving the bundle weights");
  // Serialize only constant weights.
  // Do not serialize mutable weights representing inputs and outputs, because
  // it should be configurable and set by the client.
  size_t pos = 0;
  size_t maxPos = 0;
  for (auto &v : F_->getGraph()->getParent()->getVars()) {
    auto *w = cast<WeightVar>(F_->getWeightForNode(v));
    if (v->getVisibilityKind() == Variable::VisibilityKind::Public)
      continue;
    auto numBytes = w->getType()->getSizeInBytes();
    auto payload = v->getPayload().getUnsafePtr();
    auto addr = allocationsInfo_.allocatedAddressed_[getOrigin(w)];
    if (addr < pos) {
      // The payload was written already. It aliases something we have seen
      // already.
      continue;
    }
    weightsFile.seek(addr);
    weightsFile.write(payload, numBytes);
    pos = addr + numBytes;
    maxPos = std::max(pos, maxPos);
  }
  // Make sure that the file is as long as the constantWeightVarsMemSize_.
  // This is needed to properly handle alignments.
  weightsFile.seek(maxPos);
  for (size_t endPos = irgen_.getAllocationsInfo().constantWeightVarsMemSize_;
       maxPos < endPos; maxPos++) {
    weightsFile.write(0);
  }
  weightsFile.close();
}

void CPUBackend::produceBundle(llvm::StringRef outputDir) {
  // Emit the config for the bundle.
  emitBundleConfig();

  auto &M = irgen_.getModule();
  auto bundleName = irgen_.getMainEntryName();
  auto bundleCodeOutput = (outputDir + "/" + bundleName + ".o").str();
  auto bundleWeightsOutput = (outputDir + "/" + bundleName + ".weights").str();
  DEBUG(llvm::outs() << "Producing a bundle:\n"
                     << "bundle name: " << bundleName << "\n"
                     << "bundle code: " << bundleCodeOutput << "\n"
                     << "bundle weights:" << bundleWeightsOutput << "\n");
  llvm::StringRef fileName = bundleCodeOutput;
  std::error_code EC;
  llvm::raw_fd_ostream outputFile(fileName, EC, llvm::sys::fs::F_None);
  GLOW_ASSERT(!EC &&
              "Could not open the output file for saving the bundle code");
  if (fileName.endswith(".bc")) {
    // Emit the bitcode file.
    llvm::WriteBitcodeToFile(&M, outputFile);
  } else if (fileName.endswith(".o")) {
    // Emit the object file.
    llvm::legacy::PassManager PM;
    auto &TM = irgen_.getTargetMachine();
    TM.addPassesToEmitFile(
        PM, outputFile, llvm::TargetMachine::CodeGenFileType::CGFT_ObjectFile);
    PM.run(M);
  }
  outputFile.close();
  // Output weights.
  saveWeights(bundleWeightsOutput);
}

/// Emit the entry function for the bundle. It simply calls the main entry of
/// the module and forwards its arguments to it. As the last argument it
/// provides the constant array of offsets. Since these offsets are constants,
/// the LLVM optimizer will constant propagate them into relative addressing
/// computations and the like and produce a very efficient code that uses
/// absolute addressing whenever possible.
void CPUBackend::emitBundleEntryFunction() {
  // The bundle entry point has the following API:
  // void entry(uint8_t *baseConstantWeightVars, uint8_t *baseInoutWeightVars,
  // uint8_t *baseActivations);
  llvm::Type *voidTy = llvm::Type::getVoidTy(irgen_.getLLVMContext());
  auto int8PtrTy = llvm::Type::getInt8PtrTy(irgen_.getLLVMContext());
  llvm::FunctionType *bundleFuncTy =
      llvm::FunctionType::get(voidTy, {int8PtrTy, int8PtrTy, int8PtrTy}, false);
  auto *func =
      llvm::Function::Create(bundleFuncTy, llvm::Function::ExternalLinkage,
                             irgen_.getMainEntryName(), &irgen_.getModule());
  llvm::BasicBlock *entry_bb =
      llvm::BasicBlock::Create(irgen_.getLLVMContext(), "entry", func);
  llvm::IRBuilder<> builder(entry_bb);

  // Prepare arguments for the "main" function.
  llvm::SmallVector<llvm::Value *, 4> initFunctionCallArgs;
  initFunctionCallArgs.push_back(func->args().begin());
  initFunctionCallArgs.push_back(func->args().begin() + 1);
  initFunctionCallArgs.push_back(func->args().begin() + 2);
  // Now form the offsets array and pass it as the last argument.
  auto offsetsArray = irgen_.emitConstOffsetsArray(builder, allocationsInfo_);
  initFunctionCallArgs.push_back(offsetsArray);
  // Invoke the main entry with constant arguments and let LLVM optimizer make
  // use of it.
  auto *entryF = irgen_.getModule().getFunction("main");
  entryF->setLinkage(llvm::Function::InternalLinkage);
  builder.CreateCall(entryF, initFunctionCallArgs);
  // Terminate the function.
  builder.CreateRetVoid();
  // Create the debug info for the bundle entry point function.
  irgen_.generateFunctionDebugInfo(func);
}

// Create a config for this network. It will be exposed to the clients,
// so that they know how much memory they need to allocate, etc.
// Config consists of 3 fields.
// struct BundleConfig {
//   size_t constantWeightVarsMemSize;
//   size_t mutableWeightVarsMemSize;
//   size_t activationsMemSize;
// };
void CPUBackend::emitBundleConfig() {
  // Get the integer type having the same size in bits as size_t.
  auto *SizeTType = irgen_.getBuilder().getIntNTy(sizeof(size_t) * 8);
  auto *bundleConfigTy = llvm::StructType::get(
      irgen_.getLLVMContext(), {SizeTType, SizeTType, SizeTType});
  auto config = new llvm::GlobalVariable(
      irgen_.getModule(), bundleConfigTy, /* isConst */ true,
      llvm::GlobalValue::LinkageTypes::ExternalLinkage, nullptr,
      irgen_.getMainEntryName() + "_config");
  config->setInitializer(llvm::ConstantStruct::get(
      bundleConfigTy,
      llvm::ConstantInt::get(
          SizeTType, irgen_.getAllocationsInfo().constantWeightVarsMemSize_),
      llvm::ConstantInt::get(
          SizeTType, irgen_.getAllocationsInfo().mutableWeightVarsMemSize_),
      llvm::ConstantInt::get(SizeTType,
                             irgen_.getAllocationsInfo().activationsMemSize_)));
}

void CPUBackend::performBundleMemoryAllocation() {
  allocationsInfo_.clear();
  allocationsInfo_.numberValues(F_);
  allocationsInfo_.allocateActivations(F_);
  // Tell the allocateWeightVars to not reuse any existing addresses for weights
  // and to assign new ones.
  allocationsInfo_.allocateWeightVars(F_, false);
}

void CPUBackend::save(llvm::StringRef outputDir) {
  // Object files generation works properly only in small mode.
  irgen_.initTargetMachine(target.empty() ? "" : target.getValue(),
                           llvm::CodeModel::Model::Small);
  irgen_.setMainEntryName(F_->getGraph()->getName());
  irgen_.setOutputDir(outputDir);
  irgen_.getTargetMachine();
  irgen_.initCodeGen();
  // Perform the address assignment for activations and WeightVars.
  performBundleMemoryAllocation();
  // Create the bundle entry function.
  emitBundleEntryFunction();
  // Emit the code for the body of the entry function.
  irgen_.performCodeGen();
  // Produce the bundle.
  produceBundle(outputDir);
}

bool CPUBackend::isOpSupported(Kinded::Kind opKind, ElemKind elementTy) const {
  // Check for quantization support.
  if (elementTy == ElemKind::Int8QTy) {
    switch (opKind) {
    case Kinded::Kind::AddNodeKind:
    case Kinded::Kind::BatchedAddNodeKind:
    case Kinded::Kind::ConcatNodeKind:
    case Kinded::Kind::ConvolutionNodeKind:
    case Kinded::Kind::DequantizeNodeKind:
    case Kinded::Kind::FullyConnectedNodeKind:
    case Kinded::Kind::MatMulNodeKind:
    case Kinded::Kind::MaxNodeKind:
    case Kinded::Kind::MinNodeKind:
    case Kinded::Kind::PoolAvgNodeKind:
    case Kinded::Kind::PoolMaxNodeKind:
    case Kinded::Kind::QuantizeNodeKind:
    case Kinded::Kind::ReluNodeKind:
    case Kinded::Kind::RescaleQuantizedNodeKind:
    case Kinded::Kind::ReshapeNodeKind:
    case Kinded::Kind::SubNodeKind:
    case Kinded::Kind::TransposeNodeKind:
      return true;
    default:
      return false;
    }
  }

  return true;
}
