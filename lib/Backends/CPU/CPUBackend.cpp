/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define DEBUG_TYPE "jit"

#include "CPUBackend.h"

#include "BundleSaver.h"

#include "glow/Graph/Graph.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Debug.h"

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
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

static llvm::cl::opt<std::string> target("target", llvm::cl::desc("target"));

namespace glow {
Backend *createCPUBackend(IRFunction *F) { return new CPUBackend(F); }
} // namespace glow

CPUBackend::CPUBackend(const IRFunction *F)
    : F_(F), irgen_(F_, allocationsInfo_, "") {}

CPUBackend::~CPUBackend() { alignedFree(heap_); }

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
  createCall(builder, entryF, initFunctionCallArgs);
  // Terminate the function.
  builder.CreateRetVoid();
  // Create the debug info for the entry point function.
  irgen_.generateFunctionDebugInfo(func);
}

void CPUBackend::performJITMemoryAllocation() {
  allocationsInfo_.clear();
  allocationsInfo_.numberValues(F_);
  allocationsInfo_.allocateActivations(F_);
  // Tell the allocateWeightVars to reuse existing addresses for weights.
  allocationsInfo_.allocateWeightVars(F_, true);
  allocationsInfo_.allocateTensorViews(F_);

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

void CPUBackend::save(llvm::StringRef outputDir) {
  std::string tgt = target.empty() ? "" : target.getValue();
  BundleSaver(F_).save(tgt, outputDir);
}

bool CPUBackend::isOpSupported(Kinded::Kind opKind, ElemKind elementTy) const {
  // Check for quantization support.
  if (elementTy == ElemKind::Int8QTy) {
    switch (opKind) {
    case Kinded::Kind::AddNodeKind:
    case Kinded::Kind::BatchedAddNodeKind:
    case Kinded::Kind::BatchedReduceAddNodeKind:
    case Kinded::Kind::CmpLTENodeKind:
    case Kinded::Kind::ConcatNodeKind:
    case Kinded::Kind::ConvolutionNodeKind:
    case Kinded::Kind::DequantizeNodeKind:
    case Kinded::Kind::DivNodeKind:
    case Kinded::Kind::FullyConnectedNodeKind:
    case Kinded::Kind::MatMulNodeKind:
    case Kinded::Kind::MaxNodeKind:
    case Kinded::Kind::MinNodeKind:
    case Kinded::Kind::MulNodeKind:
    case Kinded::Kind::PoolAvgNodeKind:
    case Kinded::Kind::PoolMaxNodeKind:
    case Kinded::Kind::QuantizeNodeKind:
    case Kinded::Kind::ReluNodeKind:
    case Kinded::Kind::RescaleQuantizedNodeKind:
    case Kinded::Kind::ReshapeNodeKind:
    case Kinded::Kind::SelectNodeKind:
    case Kinded::Kind::SigmoidNodeKind:
    case Kinded::Kind::SubNodeKind:
    case Kinded::Kind::TanhNodeKind:
    case Kinded::Kind::TopKNodeKind:
    case Kinded::Kind::TransposeNodeKind:
      return true;
    default:
      return false;
    }
  }

  return true;
}

bool CPUBackend::shouldLower(const Node *N) const {
  if (N->getKind() == Kinded::Kind::ConvolutionNodeKind)
    return false;
  return true;
}

llvm::CallInst *glow::createCall(llvm::IRBuilder<> &builder,
                                 llvm::Function *callee,
                                 llvm::ArrayRef<llvm::Value *> args) {
#ifndef NDEBUG
  llvm::FunctionType *FTy = callee->getFunctionType();
  assert((args.size() == FTy->getNumParams() ||
          (FTy->isVarArg() && args.size() > FTy->getNumParams())) &&
         "Calling a function with bad signature: wrong number of arguments.");

  for (unsigned i = 0; i != args.size(); ++i)
    assert((i >= FTy->getNumParams() ||
            FTy->getParamType(i) == args[i]->getType()) &&
           "Calling a function with a bad signature: argument type mismatch.");
#endif
  return builder.CreateCall(callee, args);
}
