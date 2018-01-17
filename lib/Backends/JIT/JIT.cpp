// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "JIT.h"

#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"
#include "llvm/ADT/DenseMap.h"

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

using namespace glow;
using llvm::StringRef;
using llvm::isa;

/// Optimize the module that contain the function \p F.
static void optimizeLLVMModule(llvm::Function *F, llvm::TargetMachine &TM) {
  auto *M = F->getParent();

  // Make all of the functions except for 'main' internal and optimizable.
  auto preserveMain = [=](const llvm::GlobalValue &GV) {
    return GV.getName() == "main";
  };
  llvm::internalizeModule(*M, preserveMain);

  llvm::PassManagerBuilder PMB;
  PMB.OptLevel = 3;
  PMB.SizeLevel = 0;
  PMB.LoopVectorize = true;
  PMB.SLPVectorize = true;
  PMB.Inliner = llvm::createFunctionInliningPass();

  M->setTargetTriple(TM.getTargetTriple().normalize());
  M->setDataLayout(TM.createDataLayout());

  // Replace the target-specific machine code attributes that were attached by
  // the frontend.
  llvm::AttributeList AL;
  for (auto &FF : *M) {
    FF.setAttributes(AL);
  }

  llvm::legacy::FunctionPassManager FPM(F->getParent());
  llvm::legacy::PassManager PM;

  // Add internal analysis passes from the target machine.
  PM.add(createTargetTransformInfoWrapperPass(TM.getTargetIRAnalysis()));
  FPM.add(createTargetTransformInfoWrapperPass(TM.getTargetIRAnalysis()));

  PMB.populateFunctionPassManager(FPM);
  PMB.populateModulePassManager(PM);
  FPM.doInitialization();
  PM.run(*F->getParent());
  for (auto &FF : *M) {
    FPM.run(FF);
  }
  FPM.doFinalization();
  PM.run(*F->getParent());
  M->print(llvm::errs(), nullptr);
}

JITBackend::JITBackend(Module *M) : M_(M) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  JIT_ = llvm::make_unique<llvm::orc::GlowJIT>();
}

JITBackend::~JITBackend() {
  clear();
  llvm::llvm_shutdown();
}

void JITBackend::clear() { M_->clear(); }

llvm::Value *JITBackend::emitValueAddress(llvm::IRBuilder<> &builder,
                                          glow::Value *val, ElemKind ptrTy) {
  assert(allocatedAddressed_.count(val) && "Value address was not allocated");
  void *ptr = allocatedAddressed_[val];
  auto *offset = emitConst(builder, (size_t)ptr);

  llvm::Type *T = nullptr;

  switch (ptrTy) {
  case ElemKind::FloatTy:
    T = llvm::Type::getFloatTy(ctx_)->getPointerTo();
    break;
  case ElemKind::Int8Ty:
    T = llvm::Type::getInt8Ty(ctx_)->getPointerTo();
    break;
  case ElemKind::IndexTy:
    T = builder.getIntNTy(sizeof(size_t) * 8)->getPointerTo();
    break;
  default:
    llvm_unreachable("Unimplemented");
    break;
  }

  return builder.CreateIntToPtr(offset, T);
}

llvm::Value *JITBackend::emitValueDims(llvm::IRBuilder<> &builder,
                                       glow::Value *val) {
  auto dims = val->dims();
  auto SizeTType = builder.getIntNTy(sizeof(size_t) * 8);

  std::vector<llvm::Constant *> elems;
  for (auto I : dims) {
    elems.push_back(llvm::ConstantInt::get(SizeTType, I));
  }
  auto *arr = llvm::ConstantArray::get(
      llvm::ArrayType::get(SizeTType, elems.size()), elems);

  auto *M = builder.GetInsertBlock()->getModule();

  auto *G = new llvm::GlobalVariable(*M, arr->getType(), true,
                                     llvm::GlobalValue::CommonLinkage, arr);
  return builder.CreateBitCast(G, SizeTType->getPointerTo());
}

llvm::Value *JITBackend::emitValueSize(llvm::IRBuilder<> &builder,
                                       glow::Value *val) {
  return builder.getIntN(sizeof(size_t) * 8, val->getType()->size());
}

llvm::Value *JITBackend::emitConst(llvm::IRBuilder<> &builder, float val) {
  return llvm::ConstantFP::get(llvm::Type::getFloatTy(ctx_), val);
}

llvm::Value *JITBackend::emitConst(llvm::IRBuilder<> &builder, size_t val) {
  return builder.getIntN(sizeof(size_t) * 8, val);
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

void JITBackend::init() {
  // Load the jit library as a new module.
  llmodule_ = loadStandardLibrary(&ctx_, "libjit.bc");
  GLOW_ASSERT(llmodule_.get() && "Unable to load the JIT library.");

  // Assign the target information to the module.
  llmodule_->setDataLayout(JIT_->getTargetMachine().createDataLayout());

  // Create the 'main' function into the LLVM module.
  llvm::Type *void_type = llvm::Type::getVoidTy(ctx_);
  llvm::FunctionType *jit_func_type =
      llvm::FunctionType::get(void_type, {}, false);
  func_ = llvm::Function::Create(jit_func_type, llvm::Function::ExternalLinkage,
                                 "main", llmodule_.get());

  // Setup the entry basic block and initialize the IR builder.
  llvm::BasicBlock *entry_bb = llvm::BasicBlock::Create(ctx_, "entry", func_);
  llvm::IRBuilder<> builder(entry_bb);

  allocateActivationsAndWeights();

  // For each instruction in the module:
  for (auto &I : M_->getInstrs()) {

    switch (I->getKind()) {
    case Kinded::Kind::SplatInstKind: {
      SplatInst *SI = llvm::cast<SplatInst>(I);
      auto *addr = emitValueAddress(builder, SI->getDest(), ElemKind::FloatTy);
      auto cnt = emitValueSize(builder, SI->getDest());
      auto *val = emitConst(builder, SI->getValue());
      auto *F = llmodule_->getFunction("splat_f");
      assert(F && "Unable to load the function");
      builder.CreateCall(F, {addr, cnt, val});
      break;
    }
    case Kinded::Kind::ElementMaxInstKind: {
      ElementMaxInst *EM = llvm::cast<ElementMaxInst>(I);
      auto *destPtr =
          emitValueAddress(builder, EM->getDest(), ElemKind::FloatTy);
      auto *LHSPtr = emitValueAddress(builder, EM->getLHS(), ElemKind::FloatTy);
      auto *RHSPtr = emitValueAddress(builder, EM->getRHS(), ElemKind::FloatTy);
      auto cnt = emitValueSize(builder, EM->getDest());
      auto *F = llmodule_->getFunction("elementmax_f");
      assert(F && "Unable to load the function");
      builder.CreateCall(F, {destPtr, LHSPtr, RHSPtr, cnt});
      break;
    }
    case Kinded::Kind::BatchedMatMulInstKind: {
      BatchedMatMulInst *BMM = llvm::cast<BatchedMatMulInst>(I);
      auto *destPtr =
          emitValueAddress(builder, BMM->getDest(), ElemKind::FloatTy);
      auto *LHSPtr =
          emitValueAddress(builder, BMM->getLHS(), ElemKind::FloatTy);
      auto *RHSPtr =
          emitValueAddress(builder, BMM->getRHS(), ElemKind::FloatTy);

      auto *destDims = emitValueDims(builder, BMM->getDest());
      auto *LHSDims = emitValueDims(builder, BMM->getLHS());
      auto *RHSDims = emitValueDims(builder, BMM->getRHS());

      auto *F = llmodule_->getFunction("batchedmatmul_f");
      assert(F && "Unable to load the function");
      builder.CreateCall(F,
                         {destPtr, LHSPtr, RHSPtr, destDims, LHSDims, RHSDims});
      break;
    }

    case Kinded::Kind::CopyInstKind: {
      CopyInst *CI = llvm::cast<CopyInst>(I);
      auto *destPtr =
          emitValueAddress(builder, CI->getDest(), ElemKind::Int8Ty);
      auto *srcPtr = emitValueAddress(builder, CI->getSrc(), ElemKind::Int8Ty);
      auto sizeInBytes = CI->getDest()->getType()->getSizeInBytes();
      auto *bytes = emitConst(builder, sizeInBytes);

      auto *F = llmodule_->getFunction("copy_buffer");
      assert(F && "Unable to load the function");
      builder.CreateCall(F, {destPtr, srcPtr, bytes});
      break;
    }

    case Kinded::Kind::BatchedAddInstKind: {
      BatchedAddInst *BA = llvm::cast<BatchedAddInst>(I);
      auto *destPtr =
          emitValueAddress(builder, BA->getDest(), ElemKind::FloatTy);
      auto *batchPtr =
          emitValueAddress(builder, BA->getBatch(), ElemKind::FloatTy);
      auto *slicePtr =
          emitValueAddress(builder, BA->getSlice(), ElemKind::FloatTy);

      auto bdim = flattenCdr(BA->getBatch()->dims());
      auto *numSlice = emitConst(builder, bdim.first);
      auto *sliceSize = emitConst(builder, bdim.second);

      auto *F = llmodule_->getFunction("batchedadd_f");
      assert(F && "Unable to load the function");
      builder.CreateCall(F, {destPtr, batchPtr, slicePtr, numSlice, sliceSize});
      break;
    }

    case Kinded::Kind::ConvolutionInstKind: {
      ConvolutionInst *CI = llvm::cast<ConvolutionInst>(I);
      auto *destPtr =
          emitValueAddress(builder, CI->getDest(), ElemKind::FloatTy);
      auto *srcPtr = emitValueAddress(builder, CI->getSrc(), ElemKind::FloatTy);
      auto *filterPtr =
          emitValueAddress(builder, CI->getFilter(), ElemKind::FloatTy);
      auto *biasPtr =
          emitValueAddress(builder, CI->getBias(), ElemKind::FloatTy);

      auto *destDims = emitValueDims(builder, CI->getDest());
      auto *srcDims = emitValueDims(builder, CI->getSrc());
      auto *filterDims = emitValueDims(builder, CI->getFilter());
      auto *biasDims = emitValueDims(builder, CI->getBias());

      auto *kernel = emitConst(builder, CI->getKernel());
      auto *stride = emitConst(builder, CI->getStride());
      auto *pad = emitConst(builder, CI->getPad());

      auto *F = llmodule_->getFunction("convolution_f");
      assert(F && "Unable to load the function");
      builder.CreateCall(F,
                         {srcPtr, destPtr, filterPtr, biasPtr, srcDims,
                          destDims, filterDims, biasDims, kernel, pad, stride});
      break;
    }

    case Kinded::Kind::PoolMaxInstKind: {
      PoolMaxInst *PM = llvm::cast<PoolMaxInst>(I);
      auto *destPtr =
          emitValueAddress(builder, PM->getDest(), ElemKind::FloatTy);
      auto *srcPtr = emitValueAddress(builder, PM->getSrc(), ElemKind::FloatTy);
      auto *destDims = emitValueDims(builder, PM->getDest());
      auto *srcDims = emitValueDims(builder, PM->getSrc());

      auto *kernel = emitConst(builder, PM->getKernel());
      auto *stride = emitConst(builder, PM->getStride());
      auto *pad = emitConst(builder, PM->getPad());

      auto *F = llmodule_->getFunction("pool_max_f");
      assert(F && "Unable to load the function");
      builder.CreateCall(
          F, {srcPtr, destPtr, srcDims, destDims, kernel, pad, stride});
      break;
    }

    case Kinded::Kind::ElementDivInstKind:
    case Kinded::Kind::ElementMulInstKind:
    case Kinded::Kind::ElementAddInstKind:
    case Kinded::Kind::ElementSubInstKind:
    case Kinded::Kind::ElementCmpLTEInstKind: {
      // Generate code for the op parameters.
      auto *dest = I->getOperand(0).first;
      auto *lhs = I->getOperand(1).first;
      auto *rhs = I->getOperand(2).first;
      auto *destPtr = emitValueAddress(builder, dest, ElemKind::FloatTy);
      auto *lhsPtr = emitValueAddress(builder, lhs, ElemKind::FloatTy);
      auto *rhsPtr = emitValueAddress(builder, rhs, ElemKind::FloatTy);
      auto numElem = dest->getType()->size();
      auto *numElemVal = emitConst(builder, numElem);

      // Select the correct kernel from the library.
      const char *funcName = "";
      switch (I->getKind()) {
      case Kinded::Kind::ElementDivInstKind:
        funcName = "element_div_f";
        break;
      case Kinded::Kind::ElementMulInstKind:
        funcName = "element_mul_f";
        break;
      case Kinded::Kind::ElementAddInstKind:
        funcName = "element_add_f";
        break;
      case Kinded::Kind::ElementSubInstKind:
        funcName = "element_sub_f";
        break;
      case Kinded::Kind::ElementCmpLTEInstKind:
        funcName = "element_cmp_lte_f";
        break;
      default:
        llvm_unreachable("Invalid node kind");
      }

      auto *F = llmodule_->getFunction(funcName);
      assert(F && "Unable to load the function");
      builder.CreateCall(F, {destPtr, lhsPtr, rhsPtr, numElemVal});
      break;
    }

      // Alloc and Dealloc instructions are handled by the memory allocator.
    case Kinded::Kind::AllocActivationInstKind:
    case Kinded::Kind::DeallocActivationInstKind:
    case Kinded::Kind::TensorViewInstKind:
      break;

    default:
      llvm_unreachable("ERROR: Cannot select the instruction.");
    }
  }

  // Terminate the function.
  builder.CreateRetVoid();
  assert(!llvm::verifyFunction(*func_, &llvm::errs()) && "Verification failed");

  // Optimize the module.
  optimizeLLVMModule(func_, JIT_->getTargetMachine());
  llmodule_->print(llvm::errs(), nullptr);
  // And pass the ownership to the JIT.
  JIT_->addModule(std::move(llmodule_));
}

void JITBackend::doForwardPass(bool isTrain) {
  auto sym = JIT_->findSymbol("main");
  assert(sym && "Unable to JIT the code!");
  using JitFuncType = void (*)(void);
  JitFuncType funcPtr = reinterpret_cast<JitFuncType>(sym.getAddress().get());
  funcPtr();
}

void JITBackend::allocateActivationsAndWeights() {
  // Use a memory allocator with no upper bound on how much memory we can
  // allocate.
  MemoryAllocator allocator(0);
  allocatedAddressed_.clear();

  // Maps activations and views to some offset within the heap.
  llvm::DenseMap<Value *, size_t> activationAddr;

  // Register the addresses of the tensor payload.
  for (auto &v : M_->getGraph()->getVars()) {
    auto *w = M_->getWeightForNode(v);
    allocatedAddressed_[w] = v->getPayload().getUnsafePtr();
  }

  // Assign device-space addresses to the activations.
  for (auto &I : M_->getInstrs()) {
    if (auto *A = llvm::dyn_cast<AllocActivationInst>(I)) {
      auto numBytes = I->getType()->getSizeInBytes();
      size_t addr = allocator.allocate(numBytes);
      assert(!activationAddr.count(A) && "Allocation already made!");
      activationAddr[A] = addr;
      continue;
    }

    if (auto *TV = llvm::dyn_cast<TensorViewInst>(I)) {
      // If the source is heap allocated then add it to the 'heap' map.
      if (activationAddr.count(TV->getSrc())) {
        assert(!activationAddr.count(TV) && "Allocation already made!");
        assert(activationAddr.count(TV->getSrc()) && "Can't find TV source");
        activationAddr[TV] = activationAddr[TV->getSrc()];
      } else {
        assert(allocatedAddressed_.count(TV->getSrc()) &&
               "Can't find the Weight address");
        allocatedAddressed_[TV] = allocatedAddressed_[TV->getSrc()];
      }
      continue;
    }

    if (auto *D = llvm::dyn_cast<DeallocActivationInst>(I)) {
      auto *A = D->getAlloc();
      assert(activationAddr.count(A) && "Invalid deallocation!");
      allocator.deallocate(activationAddr[A]);
      continue;
    }
  }

  // Allocate the heap to match the max memory usage.
  heap_.resize(allocator.getMaxMemoryUsage());

  // Register specific addresses within the heap to activations.
  for (auto &A : activationAddr) {
    allocatedAddressed_[A.first] = &heap_[0] + A.second;
  }
}
