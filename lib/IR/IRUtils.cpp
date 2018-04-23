// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/IR/IRUtils.h"
#include "glow/IR/Instrs.h"

#include "llvm/Support/Casting.h"

using namespace glow;
using llvm::dyn_cast;
using llvm::isa;

bool glow::isTensorView(glow::Value *v) { return isa<TensorViewInst>(v); }

Value *glow::getAllocationOrigin(Value *V) {
  while (true) {
    if (auto *AI = dyn_cast<AllocActivationInst>(V))
      return AI;
    if (auto *TVI = dyn_cast<TensorViewInst>(V)) {
      V = TVI->getSrc();
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

Value *glow::getOrigin(Value *V) {
  while (true) {
    auto *TVI = dyn_cast<TensorViewInst>(V);
    if (!TVI)
      return V;
    V = TVI->getSrc();
  }
  return V;
}
