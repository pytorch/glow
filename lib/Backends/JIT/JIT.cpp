// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "JIT.h"

#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"
#include "llvm/ADT/DenseMap.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace glow;
using llvm::isa;

JITBackend::~JITBackend() { clear(); }

void JITBackend::clear() { M_->clear(); }

llvm::Value *JITBackend::emitValueAddress(llvm::IRBuilder<> &builder,
                                          glow::Value *val) {
  void *ptr = allocatedAddressed_[val];
  auto *offset = emitConst(builder, (size_t)ptr);
  return builder.CreateIntToPtr(offset,
                                llvm::Type::getInt8Ty(ctx_)->getPointerTo());
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

void JITBackend::init() {
  llvm::SMDiagnostic Err;
  // Load the jit library as a new module.
  llmodule_ = llvm::parseIRFile("libjit.bc", Err, ctx_);
  GLOW_ASSERT(llmodule_.get() && "Unable to load the JIT library.");

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
      auto *addr = emitValueAddress(builder, SI->getDest());
      auto cnt = emitValueSize(builder, SI->getDest());
      auto *val = emitConst(builder, SI->getValue());
      auto *F = llmodule_->getFunction("splat_f");
      assert(F && "Unable to load the function");
      builder.CreateCall(F, {addr, cnt, val});
      break;
    }
    case Kinded::Kind::ElementMaxInstKind: {
      ElementMaxInst *EM = llvm::cast<ElementMaxInst>(I);
      auto *destPtr = emitValueAddress(builder, EM->getDest());
      auto *LHSPtr = emitValueAddress(builder, EM->getLHS());
      auto *RHSPtr = emitValueAddress(builder, EM->getRHS());
      auto cnt = emitValueSize(builder, EM->getDest());
      auto *F = llmodule_->getFunction("elementmax_f");
      assert(F && "Unable to load the function");
      builder.CreateCall(F, {destPtr, LHSPtr, RHSPtr, cnt});
      break;
    }
      // Alloc and Dealloc instructions are handled by the memory allocator.
    case Kinded::Kind::AllocActivationInstKind:
    case Kinded::Kind::DeallocActivationInstKind:
      break;

    default:
      llvm_unreachable("ERROR: Cannot select the instruction.");
    }
  }

  // Terminate the function.
  builder.CreateRetVoid();
  assert(!llvm::verifyFunction(*func_, &llvm::errs()) && "Verification failed");
}

void JITBackend::doForwardPass(bool isTrain) {
  // We can't call dump() directly because of a bug in LLVM 5.0 that results in
  // a linkage error. Call print of errs() instead.
  llmodule_->print(llvm::errs(), nullptr);

  llvm_unreachable("Unimplemented.");
}

void JITBackend::allocateActivationsAndWeights() {
  // Use a memory allocator with no upper bound on how much memory we can
  // allocate.
  MemoryAllocator allocator(0);

  // Maps activations and views to some offset within the heap.
  llvm::DenseMap<Value *, size_t> activationAddr;

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
      assert(!activationAddr.count(TV) && "Allocation already made!");
      activationAddr[TV] = activationAddr[TV->getSrc()];
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
  allocatedAddressed_.clear();

  // Register specific addresses within the heap to activations.
  for (auto &A : activationAddr) {
    allocatedAddressed_[A.first] = &heap_[0] + A.second;
  }

  // Register the addresses of the tensor payload.
  for (auto &v : M_->getGraph()->getVars()) {
    auto *w = M_->getWeightForNode(v);
    allocatedAddressed_[w] = v->getPayload().getUnsafePtr();
  }
}
