// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "JIT.h"

#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"
#include "llvm/ADT/DenseMap.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace glow;
using llvm::isa;

JITBackend::~JITBackend() { clear(); }

void JITBackend::clear() { M_->clear(); }

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

      void *ptr = allocatedAddressed_[SI->getDest()];
      auto *addr = builder.CreateIntToPtr(builder.getInt64((size_t)ptr),
                                          llvm::Type::getInt8Ty(ctx_));

      auto cnt =
          builder.getIntN(sizeof(size_t) * 8, SI->getDest()->getType()->size());
      auto *val =
          llvm::ConstantFP::get(llvm::Type::getFloatTy(ctx_), SI->getValue());

      auto *F = llmodule_->getFunction("splat_f");
      assert(F && "Unable to load the function");

      builder.CreateCall(F, {addr, cnt, val});
      break;
    }
    case Kinded::Kind::ElementMaxInstKind: {
      ElementMaxInst *EM = llvm::cast<ElementMaxInst>(I);
      void *ptr0 = allocatedAddressed_[EM->getDest()];
      auto *addr0 = builder.CreateIntToPtr(builder.getInt64((size_t)ptr0),
                                           llvm::Type::getInt8Ty(ctx_));
      void *ptr1 = allocatedAddressed_[EM->getLHS()];
      auto *addr1 = builder.CreateIntToPtr(builder.getInt64((size_t)ptr1),
                                           llvm::Type::getInt8Ty(ctx_));
      void *ptr2 = allocatedAddressed_[EM->getRHS()];
      auto *addr2 = builder.CreateIntToPtr(builder.getInt64((size_t)ptr2),
                                           llvm::Type::getInt8Ty(ctx_));
      auto cnt =
          builder.getIntN(sizeof(size_t) * 8, EM->getDest()->getType()->size());

      auto *F = llmodule_->getFunction("elementmax_f");
      assert(F && "Unable to load the function");

      builder.CreateCall(F, {addr0, addr1, addr2, cnt});
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
}

void JITBackend::doForwardPass(bool isTrain) {
  // We can't call dump() directly because of a bug in LLVM 5.0 that results in
  // a linkage error. Call print of errs() instead.
  llmodule_->print(llvm::errs(), nullptr);

  llvm_unreachable("Unimplemented.");
}
void JITBackend::registerGraphTensor(const Value *v, Tensor *t) {}

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
