// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "LLVMIRGen.h"

#include "CommandLine.h"

#include "glow/Graph/Graph.h"
#include "glow/IR/Instrs.h"

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

using namespace glow;
using llvm::StringRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

llvm::cl::OptionCategory CPUBackendCat("Glow CPU Backend Options");

static llvm::cl::opt<bool>
    dumpIR("dump-llvm-ir",
           llvm::cl::desc("Dump the LLVM-IR of the jitted code"),
           llvm::cl::init(false), llvm::cl::cat(CPUBackendCat));

static llvm::cl::opt<bool>
    dumpJitAsm("dump-llvm-asm",
               llvm::cl::desc("Dump the textual assembly of the jitted code"),
               llvm::cl::init(false), llvm::cl::cat(CPUBackendCat));

/// Generate the LLVM MAttr list of attributes.
static llvm::SmallVector<std::string, 0> getMachineAttributes() {
  llvm::SmallVector<std::string, 0> result;
  llvm::StringMap<bool> hostFeatures;
  if (llvm::sys::getHostCPUFeatures(hostFeatures)) {
    for (auto &feature : hostFeatures) {
      if (feature.second) {
        llvm::StringRef fn = feature.first();
        // Skip avx512 because LLVM does not support it well.
        if (fn.startswith("avx512")) {
          continue;
        }
        result.push_back(fn);
      }
    }
  }
  return result;
}

/// Returns the CPU hostname.
static llvm::StringRef getHostCpuName() {
  auto cpu_name = llvm::sys::getHostCPUName();
  // Skip avx512 because LLVM does not support it well.
  cpu_name.consume_back("-avx512");
  return cpu_name;
}

LLVMIRGen::LLVMIRGen(IRFunction *F, AllocationsInfo &allocationsInfo,
                     std::string mainEntryName)
    : F_(F), allocationsInfo_(allocationsInfo), mainEntryName_(mainEntryName) {}

void LLVMIRGen::initTargetMachine(StringRef T,
                                  llvm::CodeModel::Model codeModel) {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();

  if (T.empty())
    TM_.reset(llvm::EngineBuilder().setCodeModel(codeModel).selectTarget(
        llvm::Triple(), "", getHostCpuName(), getMachineAttributes()));
  else
    TM_.reset(llvm::EngineBuilder().setCodeModel(codeModel).selectTarget(
        llvm::Triple(T), "", "", llvm::SmallVector<std::string, 0>()));
}

std::string LLVMIRGen::getMainEntryName() const {
  StringRef name = mainEntryName_.empty() ? "main" : F_->getGraph()->getName();
  auto delimPos = name.rfind('/');
  if (delimPos != StringRef::npos)
    name = name.substr(delimPos + 1);
  return name;
}

void LLVMIRGen::setMainEntryName(std::string name) { mainEntryName_ = name; }

/// Load base addresses of different memory areas so that they can be easily
/// reused during codegen.
void LLVMIRGen::loadBaseAddresses(llvm::IRBuilder<> &builder) {
  auto *F = builder.GetInsertBlock()->getParent();

  // Load the base addresses at the beginning of the entry function once they
  // are set. They won't change after this point and all relative addressing
  // computations will simply use them.
  baseActivationsAddr_ = builder.CreatePtrToInt(F->args().begin() + 2,
                                                llvm::Type::getInt64Ty(ctx_));
  baseConstantWeightVarsAddr_ =
      builder.CreatePtrToInt(F->args().begin(), llvm::Type::getInt64Ty(ctx_));
  baseMutableWeightVarsAddr_ = builder.CreatePtrToInt(
      F->args().begin() + 1, llvm::Type::getInt64Ty(ctx_));
  offsetsArray_ = F->args().begin() + 3;
}

// Search for the standard library bitcode file on disk and load it into an
// LLVM module. We search for the standard library around the current executable
// and also in the current directory.
static std::unique_ptr<llvm::Module> loadStandardLibrary(llvm::LLVMContext *ctx,
                                                         StringRef filename) {
  using llvm::sys::path::append;
  using llvm::sys::path::parent_path;

  llvm::SMDiagnostic Err;
  auto mainExec =
      llvm::sys::fs::getMainExecutable(nullptr, (void *)&loadStandardLibrary);
  StringRef basePath = parent_path(mainExec);

  for (int i = 0; i < 3; i++) {
    llvm::SmallString<256> libPath(basePath);
    append(libPath, filename);
    if (llvm::sys::fs::exists(libPath)) {
      return llvm::parseIRFile(libPath, Err, *ctx);
    }

    basePath = parent_path(basePath);
  }

  return llvm::parseIRFile(filename, Err, *ctx);
}

void LLVMIRGen::initCodeGen() {
  // Load the jit library as a new module.
  llmodule_ = loadStandardLibrary(&ctx_, "libjit.bc");
  GLOW_ASSERT(llmodule_.get() && "Unable to load the JIT library.");

  // Assign the target information to the module.
  llmodule_->setDataLayout(getTargetMachine().createDataLayout());

  // Create the entry function into the LLVM module.
  auto int8PtrTy = llvm::Type::getInt8PtrTy(ctx_);
  auto sizeTPtrTy = llvm::Type::getIntNPtrTy(ctx_, sizeof(size_t) * 8);
  // The entry point has the following API:
  // void entry(uint8_t *baseConstantWeightVars, uint8_t
  // *baseInoutWeightVars, uint8_t *baseActivations, size_t *offsets);
  llvm::Type *voidTy = llvm::Type::getVoidTy(ctx_);
  llvm::FunctionType *jitFuncTy = llvm::FunctionType::get(
      voidTy, {int8PtrTy, int8PtrTy, int8PtrTy, sizeTPtrTy}, false);
  auto *func = llvm::Function::Create(
      jitFuncTy, llvm::Function::ExternalLinkage, "main", llmodule_.get());

  // Setup the entry basic block and initialize the IR builder.
  llvm::BasicBlock *entry_bb = llvm::BasicBlock::Create(ctx_, "entry", func);
  builder_ = llvm::make_unique<llvm::IRBuilder<>>(entry_bb);
}

void LLVMIRGen::performCodeGen() {
  auto *func = builder_->GetInsertBlock()->getParent();
  loadBaseAddresses(*builder_);

  // For each instruction in the module:
  for (auto &I : F_->getInstrs()) {
    generateLLVMIRForInstr(*builder_, I);
  }

  // Terminate the function.
  builder_->CreateRetVoid();

  assert(!llvm::verifyFunction(*func, &llvm::errs()) &&
         "Function verification failed");

  if (dumpIR) {
    llvm::outs() << "LLVM module before optimizations:\n";
    llmodule_->print(llvm::outs(), nullptr);
  }

  // Optimize the module.
  optimizeLLVMModule(func, getTargetMachine());

  // And pass the ownership to the JIT.

  if (dumpIR) {
    llvm::outs() << "LLVM module after optimizations:\n";
    llmodule_->print(llvm::outs(), nullptr);
  }

  if (dumpJitAsm) {
    llvm::SmallVector<char, 0> asmBuffer;
    llvm::raw_svector_ostream asmStream(asmBuffer);
    llvm::legacy::PassManager PM;
    getTargetMachine().addPassesToEmitFile(
        PM, asmStream, llvm::TargetMachine::CodeGenFileType::CGFT_AssemblyFile);
    PM.run(*llmodule_);
    llvm::outs() << asmStream.str();
  }
}

llvm::Value *LLVMIRGen::emitValueAddress(llvm::IRBuilder<> &builder,
                                         glow::Value *val, ElemKind ptrTy) {
  val = getOrigin(val);
  assert(allocationsInfo_.allocatedAddressed_.count(val) &&
         "Value address was not allocated");
  auto sizeTTy = builder.getIntNTy(sizeof(size_t) * 8);
  llvm::Type *T = nullptr;

  switch (ptrTy) {
  case ElemKind::FloatTy:
    T = llvm::Type::getFloatPtrTy(ctx_);
    break;
  case ElemKind::Int8QTy:
    T = llvm::Type::getInt8PtrTy(ctx_);
    break;
  case ElemKind::IndexTy:
    T = sizeTTy->getPointerTo();
    break;
  default:
    llvm_unreachable("Unimplemented");
    break;
  }

  assert(allocationsInfo_.valueNumbers_.count(val));
  auto &kindAndValue = allocationsInfo_.valueNumbers_[val];

  // Get the required base address.
  llvm::Value *baseAddrValue = nullptr;
  switch (kindAndValue.first) {
  case AllocationsInfo::ValueKind::Activation:
    baseAddrValue = baseActivationsAddr_;
    break;
  case AllocationsInfo::ValueKind::ConstantWeight:
    baseAddrValue = baseConstantWeightVarsAddr_;
    break;
  case AllocationsInfo::ValueKind::MutableWeight:
    baseAddrValue = baseMutableWeightVarsAddr_;
    break;
  }

  // Use relative addressing.
  // Get offset.
  auto valueIdx = llvm::ConstantInt::get(sizeTTy, kindAndValue.second);
  auto offsetAddr = builder.CreateGEP(sizeTTy, offsetsArray_, valueIdx);
  auto offsetValue = builder.CreateLoad(sizeTTy, offsetAddr);
  // Add offset to the base address.
  llvm::Value *addr = builder.CreateAdd(baseAddrValue, offsetValue);
  return builder.CreateIntToPtr(addr, T);
}

llvm::Value *
LLVMIRGen::emitConstOffsetsArray(llvm::IRBuilder<> &builder,
                                 const AllocationsInfo &allocationsInfo) {

  auto sizeTType = builder.getIntNTy(sizeof(size_t) * 8);
  std::vector<llvm::Constant *> elems(allocationsInfo.valueNumbers_.size());
  for (auto &I : allocationsInfo.valueNumbers_) {
    auto *V = I.first;
    auto offset = I.second.second;
    elems[offset] = llvm::ConstantInt::get(
        sizeTType, allocationsInfo.allocatedAddressed_.lookup(V));
  }
  auto *arr = llvm::ConstantArray::get(
      llvm::ArrayType::get(sizeTType, elems.size()), elems);
  // Ensure that the same casted global variable is used for the equivalent
  // const arrays. This is important for the later function specialization pass.
  // LLVM does not do it automatically for this code pattern involving global
  // variables. It also reduces the number of variables.
  auto &constArrayVar = constArrayPtrs_[arr];
  if (constArrayVar)
    return constArrayVar;

  auto *M = builder.GetInsertBlock()->getModule();

  auto *G = new llvm::GlobalVariable(*M, arr->getType(), true,
                                     llvm::GlobalValue::InternalLinkage, arr);
  constArrayVar = builder.CreateBitCast(G, sizeTType->getPointerTo());
  return constArrayVar;
}

llvm::Value *LLVMIRGen::emitConstArray(llvm::IRBuilder<> &builder,
                                       llvm::ArrayRef<size_t> vals) {
  auto SizeTType = builder.getIntNTy(sizeof(size_t) * 8);
  std::vector<llvm::Constant *> elems;
  for (auto I : vals) {
    elems.push_back(llvm::ConstantInt::get(SizeTType, I));
  }
  auto *arr = llvm::ConstantArray::get(
      llvm::ArrayType::get(SizeTType, elems.size()), elems);
  // Ensure that the same casted global variable is used for the equivalent
  // const arrays. This is important for the later function specialization pass.
  // LLVM does not do it automatically for this code pattern involving global
  // variables. It also reduces the number of variables.
  auto &constArrayVar = constArrayPtrs_[arr];
  if (constArrayVar)
    return constArrayVar;

  auto *M = builder.GetInsertBlock()->getModule();

  auto *G = new llvm::GlobalVariable(*M, arr->getType(), true,
                                     llvm::GlobalValue::InternalLinkage, arr);
  constArrayVar = builder.CreateBitCast(G, SizeTType->getPointerTo());
  return constArrayVar;
}

llvm::Value *LLVMIRGen::emitValueDims(llvm::IRBuilder<> &builder,
                                      glow::Value *val) {
  auto dims = val->dims();
  return emitConstArray(builder, dims);
}

llvm::Value *LLVMIRGen::emitValueSize(llvm::IRBuilder<> &builder,
                                      glow::Value *val) {
  return builder.getIntN(sizeof(size_t) * 8, val->getType()->size());
}

llvm::Value *LLVMIRGen::emitConst(llvm::IRBuilder<> &builder, float val) {
  return llvm::ConstantFP::get(llvm::Type::getFloatTy(ctx_), val);
}

llvm::Value *LLVMIRGen::emitConst(llvm::IRBuilder<> &builder, size_t val) {
  return builder.getIntN(sizeof(size_t) * 8, val);
}

llvm::Function *LLVMIRGen::getFunction(const std::string &name) {
  return llmodule_->getFunction(name);
}

void LLVMIRGen::generateLLVMIRForInstr(llvm::IRBuilder<> &builder,
                                       glow::Instruction *I) {
  switch (I->getKind()) {
  case Kinded::Kind::SplatInstKind: {
    SplatInst *SI = cast<SplatInst>(I);
    auto *addr = emitValueAddress(builder, SI->getDest(), ElemKind::FloatTy);
    auto cnt = emitValueSize(builder, SI->getDest());
    auto *val = emitConst(builder, SI->getValue());
    auto *F = getFunction("libjit_splat_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {addr, cnt, val});
    break;
  }

  case Kinded::Kind::ElementMaxInstKind: {
    ElementMaxInst *EM = cast<ElementMaxInst>(I);
    auto *destPtr = emitValueAddress(builder, EM->getDest(), ElemKind::FloatTy);
    auto *lhsPtr = emitValueAddress(builder, EM->getLHS(), ElemKind::FloatTy);
    auto *rhsPtr = emitValueAddress(builder, EM->getRHS(), ElemKind::FloatTy);
    auto cnt = emitValueSize(builder, EM->getDest());
    auto *F = getFunction("libjit_elementmax_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {destPtr, lhsPtr, rhsPtr, cnt});
    break;
  }

  case Kinded::Kind::ElementMinInstKind: {
    ElementMinInst *EM = cast<ElementMinInst>(I);
    auto *destPtr = emitValueAddress(builder, EM->getDest(), ElemKind::FloatTy);
    auto *lhsPtr = emitValueAddress(builder, EM->getLHS(), ElemKind::FloatTy);
    auto *rhsPtr = emitValueAddress(builder, EM->getRHS(), ElemKind::FloatTy);
    auto cnt = emitValueSize(builder, EM->getDest());
    auto *F = getFunction("libjit_elementmin_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {destPtr, lhsPtr, rhsPtr, cnt});
    break;
  }

  case Kinded::Kind::ElementSelectInstKind: {
    ElementSelectInst *ES = cast<ElementSelectInst>(I);
    auto *destPtr = emitValueAddress(builder, ES->getDest(), ElemKind::FloatTy);
    auto *condPtr = emitValueAddress(builder, ES->getCond(), ElemKind::FloatTy);
    auto *lhsPtr = emitValueAddress(builder, ES->getLHS(), ElemKind::FloatTy);
    auto *rhsPtr = emitValueAddress(builder, ES->getRHS(), ElemKind::FloatTy);
    auto cnt = emitValueSize(builder, ES->getDest());
    auto *F = getFunction("libjit_elementselect_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {destPtr, condPtr, lhsPtr, rhsPtr, cnt});
    break;
  }

  case Kinded::Kind::BatchedMatMulInstKind: {
    BatchedMatMulInst *BMM = cast<BatchedMatMulInst>(I);
    auto *destPtr =
        emitValueAddress(builder, BMM->getDest(), ElemKind::FloatTy);
    auto *lhsPtr = emitValueAddress(builder, BMM->getLHS(), ElemKind::FloatTy);
    auto *rhsPtr = emitValueAddress(builder, BMM->getRHS(), ElemKind::FloatTy);
    auto *F = getFunction("libjit_batchedmatmul_f");
    assert(F && "Unable to load the function");

    auto *destDims = emitValueDims(builder, BMM->getDest());
    auto *lhsDims = emitValueDims(builder, BMM->getLHS());
    auto *rhsDims = emitValueDims(builder, BMM->getRHS());

    builder.CreateCall(F,
                       {destPtr, lhsPtr, rhsPtr, destDims, lhsDims, rhsDims});
    break;
  }

  case Kinded::Kind::CopyInstKind: {
    CopyInst *CI = cast<CopyInst>(I);
    auto *destPtr = emitValueAddress(builder, CI->getDest(), ElemKind::Int8QTy);
    auto *srcPtr = emitValueAddress(builder, CI->getSrc(), ElemKind::Int8QTy);
    auto sizeInBytes = CI->getDest()->getType()->getSizeInBytes();
    auto *bytes = emitConst(builder, sizeInBytes);

    auto *F = getFunction("libjit_copy_buffer");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {destPtr, srcPtr, bytes});
    break;
  }

  case Kinded::Kind::BatchedAddInstKind: {
    BatchedAddInst *BA = cast<BatchedAddInst>(I);
    auto *destPtr = emitValueAddress(builder, BA->getDest(), ElemKind::FloatTy);
    auto *batchPtr =
        emitValueAddress(builder, BA->getBatch(), ElemKind::FloatTy);
    auto *slicePtr =
        emitValueAddress(builder, BA->getSlice(), ElemKind::FloatTy);

    auto bdim = flattenCdr(BA->getBatch()->dims());
    auto *numSlice = emitConst(builder, bdim.first);
    auto *sliceSize = emitConst(builder, bdim.second);

    auto *F = getFunction("libjit_batchedadd_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {destPtr, batchPtr, slicePtr, numSlice, sliceSize});
    break;
  }

  case Kinded::Kind::BatchedReduceAddInstKind: {
    BatchedReduceAddInst *BR = cast<BatchedReduceAddInst>(I);
    auto *destPtr = emitValueAddress(builder, BR->getDest(), ElemKind::FloatTy);
    auto *batchPtr =
        emitValueAddress(builder, BR->getBatch(), ElemKind::FloatTy);

    auto *destSize = emitConst(builder, BR->getDest()->getType()->size());
    auto bdim = flattenCdr(BR->getBatch()->dims());
    auto *numSlice = emitConst(builder, bdim.first);
    auto *sliceSize = emitConst(builder, bdim.second);

    auto *F = getFunction("libjit_batchedreduceadd_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {destPtr, batchPtr, destSize, numSlice, sliceSize});
    break;
  }

  case Kinded::Kind::ConvolutionInstKind: {
    ConvolutionInst *CI = cast<ConvolutionInst>(I);
    auto *destPtr = emitValueAddress(builder, CI->getDest(), ElemKind::FloatTy);
    auto *srcPtr = emitValueAddress(builder, CI->getSrc(), ElemKind::FloatTy);
    auto *filterPtr =
        emitValueAddress(builder, CI->getFilter(), ElemKind::FloatTy);
    auto *biasPtr = emitValueAddress(builder, CI->getBias(), ElemKind::FloatTy);

    auto *destDims = emitValueDims(builder, CI->getDest());
    auto *srcDims = emitValueDims(builder, CI->getSrc());
    auto *filterDims = emitValueDims(builder, CI->getFilter());
    auto *biasDims = emitValueDims(builder, CI->getBias());

    auto *kernel = emitConst(builder, CI->getKernel());
    auto *stride = emitConst(builder, CI->getStride());
    auto *pad = emitConst(builder, CI->getPad());

    const char *kernelName = "libjit_convolution_f";

    // Use a special version of the kernel for the case where K (the depth of
    // the convolution) is a multiple of 4.
    if ((CI->getDest()->dims()[3] % 4) == 0) {
      kernelName = "libjit_convolution_f_unroll_k4";
    }

    auto *F = getFunction(kernelName);
    assert(F && "Unable to load the function");
    builder.CreateCall(F,
                       {srcPtr, destPtr, filterPtr, biasPtr, srcDims, destDims,
                        filterDims, biasDims, kernel, stride, pad});
    break;
  }

  case Kinded::Kind::ConvolutionGradInstKind: {
    ConvolutionGradInst *CG = cast<ConvolutionGradInst>(I);
    auto *srcGradPtr =
        emitValueAddress(builder, CG->getSrcGrad(), ElemKind::FloatTy);
    auto *destGradPtr =
        emitValueAddress(builder, CG->getDestGrad(), ElemKind::FloatTy);
    auto *srcPtr = emitValueAddress(builder, CG->getSrc(), ElemKind::FloatTy);
    auto *kernelGradPtr =
        emitValueAddress(builder, CG->getFilterGrad(), ElemKind::FloatTy);
    auto *biasGradPtr =
        emitValueAddress(builder, CG->getBiasGrad(), ElemKind::FloatTy);
    auto *kernelPtr =
        emitValueAddress(builder, CG->getFilter(), ElemKind::FloatTy);
    auto *destGradDims = emitValueDims(builder, CG->getDestGrad());
    auto *srcDims = emitValueDims(builder, CG->getSrc());
    auto *kernelGradDims = emitValueDims(builder, CG->getFilterGrad());

    auto *kernel = emitConst(builder, CG->getKernel());
    auto *stride = emitConst(builder, CG->getStride());
    auto *pad = emitConst(builder, CG->getPad());

    auto *F = getFunction("libjit_convolution_grad_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {srcGradPtr, destGradPtr, srcPtr, kernelGradPtr,
                           biasGradPtr, kernelPtr, destGradDims, srcDims,
                           kernelGradDims, kernel, stride, pad});
    break;
  }

  case Kinded::Kind::LocalResponseNormalizationInstKind: {
    LocalResponseNormalizationInst *LRN =
        cast<LocalResponseNormalizationInst>(I);
    auto *destPtr =
        emitValueAddress(builder, LRN->getDest(), ElemKind::FloatTy);
    auto *srcPtr = emitValueAddress(builder, LRN->getSrc(), ElemKind::FloatTy);
    auto *scalePtr =
        emitValueAddress(builder, LRN->getScale(), ElemKind::FloatTy);
    auto *destDims = emitValueDims(builder, LRN->getDest());
    auto *srcDims = emitValueDims(builder, LRN->getSrc());

    auto *halfWindow = emitConst(builder, LRN->getHalfWindowSize());
    auto *alpha = emitConst(builder, LRN->getAlpha());
    auto *beta = emitConst(builder, LRN->getBeta());
    auto *k = emitConst(builder, LRN->getK());

    auto *F = getFunction("libjit_local_response_normalization_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {destPtr, srcPtr, scalePtr, destDims, srcDims,
                           halfWindow, alpha, beta, k});
    break;
  }

  case Kinded::Kind::LocalResponseNormalizationGradInstKind: {
    LocalResponseNormalizationGradInst *LRNG =
        llvm::cast<LocalResponseNormalizationGradInst>(I);
    auto *srcGradPtr =
        emitValueAddress(builder, LRNG->getSrcGrad(), ElemKind::FloatTy);
    auto *destGradPtr =
        emitValueAddress(builder, LRNG->getDestGrad(), ElemKind::FloatTy);
    auto *srcPtr = emitValueAddress(builder, LRNG->getSrc(), ElemKind::FloatTy);
    auto *destPtr =
        emitValueAddress(builder, LRNG->getDest(), ElemKind::FloatTy);
    auto *scalePtr =
        emitValueAddress(builder, LRNG->getScale(), ElemKind::FloatTy);
    auto *destDims = emitValueDims(builder, LRNG->getDest());

    auto *halfWindow = emitConst(builder, LRNG->getHalfWindowSize());
    auto *alpha = emitConst(builder, LRNG->getAlpha());
    auto *beta = emitConst(builder, LRNG->getBeta());

    auto *F = getFunction("libjit_local_response_normalization_grad_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {srcGradPtr, destGradPtr, srcPtr, destPtr, scalePtr,
                           destDims, halfWindow, alpha, beta});
    break;
  }

  case Kinded::Kind::PoolMaxInstKind: {
    PoolMaxInst *PM = cast<PoolMaxInst>(I);
    auto *destPtr = emitValueAddress(builder, PM->getDest(), ElemKind::FloatTy);
    auto *srcPtr = emitValueAddress(builder, PM->getSrc(), ElemKind::FloatTy);
    auto *destDims = emitValueDims(builder, PM->getDest());
    auto *srcDims = emitValueDims(builder, PM->getSrc());

    auto *kernel = emitConst(builder, PM->getKernel());
    auto *stride = emitConst(builder, PM->getStride());
    auto *pad = emitConst(builder, PM->getPad());

    auto *F = getFunction("libjit_pool_max_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(
        F, {srcPtr, destPtr, srcDims, destDims, kernel, stride, pad});
    break;
  }

  case Kinded::Kind::PoolMaxWithXYInstKind: {
    PoolMaxWithXYInst *PMXY = cast<PoolMaxWithXYInst>(I);
    auto *destPtr =
        emitValueAddress(builder, PMXY->getDest(), ElemKind::FloatTy);
    auto *srcPtr = emitValueAddress(builder, PMXY->getSrc(), ElemKind::FloatTy);
    auto *srcXYPtr =
        emitValueAddress(builder, PMXY->getSrcXY(), ElemKind::IndexTy);
    auto *destDims = emitValueDims(builder, PMXY->getDest());
    auto *srcDims = emitValueDims(builder, PMXY->getSrc());

    auto *kernel = emitConst(builder, PMXY->getKernel());
    auto *stride = emitConst(builder, PMXY->getStride());
    auto *pad = emitConst(builder, PMXY->getPad());

    auto *F = getFunction("libjit_pool_max_xy_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(
        F, {srcPtr, destPtr, srcXYPtr, srcDims, destDims, kernel, stride, pad});
    break;
  }

  case Kinded::Kind::PoolMaxWithXYGradInstKind: {
    PoolMaxWithXYGradInst *PMG = cast<PoolMaxWithXYGradInst>(I);
    auto *srcGradPtr =
        emitValueAddress(builder, PMG->getSrcGrad(), ElemKind::FloatTy);
    auto *destGradPtr =
        emitValueAddress(builder, PMG->getDestGrad(), ElemKind::FloatTy);
    auto *srcXYPtr =
        emitValueAddress(builder, PMG->getSrcXY(), ElemKind::IndexTy);
    auto *srcGradDims = emitValueDims(builder, PMG->getSrcGrad());
    auto *destDims = emitValueDims(builder, PMG->getDest());

    auto *F = getFunction("libjit_pool_max_xy_grad_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(
        F, {srcGradPtr, destGradPtr, srcXYPtr, srcGradDims, destDims});
    break;
  }

  case Kinded::Kind::PoolAvgInstKind: {
    PoolAvgInst *PM = cast<PoolAvgInst>(I);
    auto *destPtr = emitValueAddress(builder, PM->getDest(), ElemKind::FloatTy);
    auto *srcPtr = emitValueAddress(builder, PM->getSrc(), ElemKind::FloatTy);
    auto *destDims = emitValueDims(builder, PM->getDest());
    auto *srcDims = emitValueDims(builder, PM->getSrc());

    auto *kernel = emitConst(builder, PM->getKernel());
    auto *stride = emitConst(builder, PM->getStride());
    auto *pad = emitConst(builder, PM->getPad());

    auto *F = getFunction("libjit_pool_avg_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(
        F, {srcPtr, destPtr, srcDims, destDims, kernel, stride, pad});
    break;
  }

  case Kinded::Kind::PoolAvgGradInstKind: {
    PoolAvgGradInst *PAG = cast<PoolAvgGradInst>(I);
    auto *srcGradPtr =
        emitValueAddress(builder, PAG->getSrcGrad(), ElemKind::FloatTy);
    auto *destGradPtr =
        emitValueAddress(builder, PAG->getDestGrad(), ElemKind::FloatTy);
    auto *srcGradDims = emitValueDims(builder, PAG->getSrcGrad());
    auto *destDims = emitValueDims(builder, PAG->getDest());
    auto *kernel = emitConst(builder, PAG->getKernel());
    auto *stride = emitConst(builder, PAG->getStride());
    auto *pad = emitConst(builder, PAG->getPad());

    auto *F = getFunction("libjit_pool_avg_grad_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {srcGradPtr, destGradPtr, srcGradDims, destDims,
                           kernel, stride, pad});
    break;
  }

  case Kinded::Kind::QuantizeInstKind: {
    QuantizeInst *QI = cast<QuantizeInst>(I);
    auto *destPtr = emitValueAddress(builder, QI->getDest(), ElemKind::Int8QTy);
    auto *srcPtr = emitValueAddress(builder, QI->getSrc(), ElemKind::FloatTy);
    auto *numElem = emitConst(builder, QI->getDest()->getType()->size());
    auto *scale = emitConst(builder, QI->getDest()->getType()->getScale());
    // TODO(hegemanjwh2): Fix generated integer type for offset.
    auto *offset =
        emitConst(builder, (size_t)QI->getDest()->getType()->getOffset());

    auto *F = getFunction("libjit_quantize_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {destPtr, srcPtr, numElem, scale, offset});
    break;
  }

  case Kinded::Kind::DequantizeInstKind: {
    DequantizeInst *DQI = cast<DequantizeInst>(I);
    auto *destPtr =
        emitValueAddress(builder, DQI->getDest(), ElemKind::FloatTy);
    auto *srcPtr = emitValueAddress(builder, DQI->getSrc(), ElemKind::Int8QTy);
    auto *numElem = emitConst(builder, DQI->getDest()->getType()->size());
    auto *scale = emitConst(builder, DQI->getSrc()->getType()->getScale());
    // TODO(hegemanjwh2): Fix generated integer type for offset.
    auto *offset =
        emitConst(builder, (size_t)DQI->getSrc()->getType()->getOffset());

    auto *F = getFunction("libjit_dequantize_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {destPtr, srcPtr, numElem, scale, offset});
    break;
  }

  case Kinded::Kind::SGDInstKind: {
    SGDInst *SGD = cast<SGDInst>(I);
    auto *W = emitValueAddress(builder, SGD->getWeight(), ElemKind::FloatTy);
    auto *G = emitValueAddress(builder, SGD->getGradient(), ElemKind::FloatTy);
    auto *Gsum = emitValueAddress(builder, SGD->getGsum(), ElemKind::FloatTy);
    auto *l1Decay = emitConst(builder, SGD->getL1Decay());
    auto *l2Decay = emitConst(builder, SGD->getL2Decay());
    auto *learningRate = emitConst(builder, SGD->getLearningRate());
    auto *momentum = emitConst(builder, SGD->getMomentum());
    auto *batchSize = emitConst(builder, (size_t)SGD->getBatchSize());
    auto *Wsize = emitConst(builder, SGD->getWeight()->getType()->size());

    auto *F = getFunction("libjit_sgd_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {W, G, Gsum, l1Decay, l2Decay, learningRate, momentum,
                           batchSize, Wsize});
    break;
  }

  case Kinded::Kind::SoftMaxInstKind: {
    SoftMaxInst *SM = cast<SoftMaxInst>(I);
    auto *destPtr = emitValueAddress(builder, SM->getDest(), ElemKind::FloatTy);
    auto *srcPtr = emitValueAddress(builder, SM->getSrc(), ElemKind::FloatTy);
    auto *destDims = emitValueDims(builder, SM->getDest());
    auto *srcDims = emitValueDims(builder, SM->getSrc());

    auto *F = getFunction("libjit_softmax_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {srcPtr, destPtr, srcDims, destDims});
    break;
  }

  case Kinded::Kind::SoftMaxWithLossGradInstKind: {
    SoftMaxWithLossGradInst *SMG = cast<SoftMaxWithLossGradInst>(I);
    auto *srcGradPtr =
        emitValueAddress(builder, SMG->getSrcGrad(), ElemKind::FloatTy);
    auto *destPtr =
        emitValueAddress(builder, SMG->getOrigDest(), ElemKind::FloatTy);
    auto *selectedPtr =
        emitValueAddress(builder, SMG->getSelected(), ElemKind::IndexTy);
    auto *srcGradDims = emitValueDims(builder, SMG->getSrcGrad());
    auto *selectedDims = emitValueDims(builder, SMG->getSelected());
    auto *F = getFunction("libjit_softmaxgrad_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(
        F, {srcGradPtr, destPtr, selectedPtr, srcGradDims, selectedDims});
    break;
  }

  case Kinded::Kind::SigmoidInstKind: {
    SigmoidInst *SI = cast<SigmoidInst>(I);
    auto *destPtr = emitValueAddress(builder, SI->getDest(), ElemKind::FloatTy);
    auto *srcPtr = emitValueAddress(builder, SI->getSrc(), ElemKind::FloatTy);
    auto *numElemVal = emitConst(builder, SI->getDest()->getType()->size());
    auto *F = getFunction("libjit_sigmoid_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {srcPtr, destPtr, numElemVal});
    break;
  }

  case Kinded::Kind::TanhInstKind: {
    TanhInst *TI = cast<TanhInst>(I);
    auto *destPtr = emitValueAddress(builder, TI->getDest(), ElemKind::FloatTy);
    auto *srcPtr = emitValueAddress(builder, TI->getSrc(), ElemKind::FloatTy);
    auto *numElemVal = emitConst(builder, TI->getDest()->getType()->size());
    auto *F = getFunction("libjit_tanh_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {srcPtr, destPtr, numElemVal});
    break;
  }

  case Kinded::Kind::TransposeInstKind: {
    TransposeInst *TI = cast<TransposeInst>(I);
    auto *destPtr = emitValueAddress(builder, TI->getDest(), ElemKind::FloatTy);
    auto *srcPtr = emitValueAddress(builder, TI->getSrc(), ElemKind::FloatTy);
    auto *destDims = emitValueDims(builder, TI->getDest());
    auto *srcDims = emitValueDims(builder, TI->getSrc());

    // Convert the mask to size_t type.
    llvm::SmallVector<size_t, 6> shuffSizeT;
    for (auto D : TI->getShuffle()) {
      shuffSizeT.push_back((size_t)D);
    }

    auto *shuffle = emitConstArray(builder, shuffSizeT);
    auto *len = emitConst(builder, TI->getShuffle().size());

    auto *F = getFunction("libjit_transpose_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {srcPtr, destPtr, srcDims, destDims, shuffle, len});
    break;
  }

  case Kinded::Kind::IntrinsicInstKind: {
    IntrinsicInst *II = cast<IntrinsicInst>(I);
    if (II->getIdentifier().equals("jit.max0")) {
      auto *dest = II->getOperand(0).first;
      auto *src = II->getOperand(1).first;
      auto *destPtr = emitValueAddress(builder, dest, ElemKind::FloatTy);
      auto *lhsPtr = emitValueAddress(builder, src, ElemKind::FloatTy);
      auto cnt = emitValueSize(builder, dest);
      auto *F = getFunction("libjit_elementmax0_f");
      assert(F && "Unable to load the function");
      builder.CreateCall(F, {destPtr, lhsPtr, cnt});
      break;
    }

    llvm_unreachable("Unknown intrinsic");
  }

  case Kinded::Kind::ElementDivInstKind:
  case Kinded::Kind::ElementMulInstKind:
  case Kinded::Kind::ElementAddInstKind:
  case Kinded::Kind::ElementSubInstKind:
  case Kinded::Kind::ElementCmpLTEInstKind: {
    // Generate code for the op parameters.
    Value *dest;
    llvm::Value *destPtr, *lhsPtr, *rhsPtr;

    // Select the correct kernel from the library.
    const char *funcName = "";
    switch (I->getKind()) {
    case Kinded::Kind::ElementDivInstKind: {
      auto *tmpInst = cast<ElementDivInst>(I);
      dest = tmpInst->getDest();
      destPtr = emitValueAddress(builder, dest, ElemKind::FloatTy);
      lhsPtr = emitValueAddress(builder, tmpInst->getLHS(), ElemKind::FloatTy);
      rhsPtr = emitValueAddress(builder, tmpInst->getRHS(), ElemKind::FloatTy);
      funcName = "libjit_element_div_f";
      break;
    }
    case Kinded::Kind::ElementMulInstKind: {
      auto *tmpInst = cast<ElementMulInst>(I);
      dest = tmpInst->getDest();
      destPtr = emitValueAddress(builder, dest, ElemKind::FloatTy);
      lhsPtr = emitValueAddress(builder, tmpInst->getLHS(), ElemKind::FloatTy);
      rhsPtr = emitValueAddress(builder, tmpInst->getRHS(), ElemKind::FloatTy);
      funcName = "libjit_element_mul_f";
      break;
    }
    case Kinded::Kind::ElementAddInstKind: {
      auto *tmpInst = cast<ElementAddInst>(I);
      dest = tmpInst->getDest();
      destPtr = emitValueAddress(builder, dest, ElemKind::FloatTy);
      lhsPtr = emitValueAddress(builder, tmpInst->getLHS(), ElemKind::FloatTy);
      rhsPtr = emitValueAddress(builder, tmpInst->getRHS(), ElemKind::FloatTy);
      funcName = "libjit_element_add_f";
      break;
    }
    case Kinded::Kind::ElementSubInstKind: {
      auto *tmpInst = cast<ElementSubInst>(I);
      dest = tmpInst->getDest();
      destPtr = emitValueAddress(builder, dest, ElemKind::FloatTy);
      lhsPtr = emitValueAddress(builder, tmpInst->getLHS(), ElemKind::FloatTy);
      rhsPtr = emitValueAddress(builder, tmpInst->getRHS(), ElemKind::FloatTy);
      funcName = "libjit_element_sub_f";
      break;
    }
    case Kinded::Kind::ElementCmpLTEInstKind: {
      auto *tmpInst = cast<ElementCmpLTEInst>(I);
      dest = tmpInst->getDest();
      destPtr = emitValueAddress(builder, dest, ElemKind::FloatTy);
      lhsPtr = emitValueAddress(builder, tmpInst->getLHS(), ElemKind::FloatTy);
      rhsPtr = emitValueAddress(builder, tmpInst->getRHS(), ElemKind::FloatTy);
      funcName = "libjit_element_cmp_lte_f";
      break;
    }
    default:
      llvm_unreachable("Invalid node kind");
    }

    auto numElem = dest->getType()->size();
    auto *numElemVal = emitConst(builder, numElem);

    auto *F = getFunction(funcName);
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {destPtr, lhsPtr, rhsPtr, numElemVal});
    break;
  }

    // Alloc and Dealloc instructions are handled by the memory allocator.
  case Kinded::Kind::AllocActivationInstKind:
  case Kinded::Kind::DeallocActivationInstKind:
  case Kinded::Kind::TensorViewInstKind:
    break;
  case Kinded::Kind::InsertTensorInstKind: {
    InsertTensorInst *ITI = llvm::cast<InsertTensorInst>(I);
    auto dest = ITI->getDest();
    auto src = ITI->getSrc();
    auto offsets = ITI->getOffsets();
    auto *destPtr = emitValueAddress(builder, dest, ElemKind::FloatTy);
    auto *srcPtr = emitValueAddress(builder, src, ElemKind::FloatTy);
    auto *destDims = emitValueDims(builder, dest);
    auto *srcDims = emitValueDims(builder, src);
    auto *destDimsSize = emitConst(builder, dest->getType()->dims().size());
    auto *srcDimsSize = emitConst(builder, src->getType()->dims().size());
    auto *offsetsPtr = emitConstArray(builder, offsets);
    auto *offsetsArraySize = emitConst(builder, offsets.size());
    auto *F = getFunction("libjit_insert_tensor_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {destPtr, srcPtr, offsetsPtr, destDims, srcDims,
                           destDimsSize, srcDimsSize, offsetsArraySize});
    break;
  }
  case Kinded::Kind::ExtractTensorInstKind: {
    ExtractTensorInst *ITI = llvm::cast<ExtractTensorInst>(I);
    auto dest = ITI->getDest();
    auto src = ITI->getSrc();
    auto offsets = ITI->getOffsets();
    auto *destPtr = emitValueAddress(builder, dest, ElemKind::FloatTy);
    auto *srcPtr = emitValueAddress(builder, src, ElemKind::FloatTy);
    auto *destDims = emitValueDims(builder, dest);
    auto *srcDims = emitValueDims(builder, src);
    auto *destDimsSize = emitConst(builder, dest->getType()->dims().size());
    auto *srcDimsSize = emitConst(builder, src->getType()->dims().size());
    auto *offsetsPtr = emitConstArray(builder, offsets);
    auto *offsetsArraySize = emitConst(builder, offsets.size());
    auto *F = getFunction("libjit_extract_tensor_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {srcPtr, destPtr, offsetsPtr, srcDims, destDims,
                           srcDimsSize, destDimsSize, offsetsArraySize});
    break;
  }

  default:
    llvm_unreachable("ERROR: Cannot select the instruction.");
  }
}
