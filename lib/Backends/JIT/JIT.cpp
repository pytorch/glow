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

void JITBackend::init() { llvm_unreachable("Unimplemented."); }
void JITBackend::doForwardPass(bool isTrain) {
  llvm_unreachable("Unimplemented.");
}
void JITBackend::registerGraphTensor(const Value *v, Tensor *t) {
  llvm_unreachable("Unimplemented.");
}
