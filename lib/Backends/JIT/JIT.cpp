// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "JIT.h"

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/Support/Casting.h"

using namespace glow;
using llvm::isa;

JITBackend::~JITBackend() { clear(); }

void JITBackend::clear() { M_->clear(); }

void JITBackend::init() {
  // Notice that we can't use std::make_unique here because it's only available
  // in c++14.
  llmodule_ = std::unique_ptr<llvm::Module>(new llvm::Module("program", ctx_));

  llvm::Type *void_type = llvm::Type::getVoidTy(ctx_);
  llvm::FunctionType *jit_func_type =
      llvm::FunctionType::get(void_type, {}, false);
  func_ = llvm::Function::Create(jit_func_type, llvm::Function::ExternalLinkage,
                                 "main", llmodule_.get());

  llvm::BasicBlock *entry_bb = llvm::BasicBlock::Create(ctx_, "entry", func_);
  llvm::IRBuilder<> builder(entry_bb);
}

void JITBackend::doForwardPass(bool isTrain) {
  llvm_unreachable("Unimplemented.");
}
void JITBackend::registerGraphTensor(const Value *v, Tensor *t) {
  llvm_unreachable("Unimplemented.");
}
