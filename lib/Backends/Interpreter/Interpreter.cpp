// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "Interpreter.h"

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/Support/Casting.h"

using namespace glow;
using llvm::isa;

Interpreter::~Interpreter() { clear(); }

void Interpreter::clear() {
  // Delete the tensors that are owned by this module.
  for (auto p : tensors_) {
    delete p.second;
  }

  tensors_.clear();
  externalTensors_.clear();
}

void Interpreter::init() {
  for (auto &v : M_->getGraph()->getVars()) {
    auto *w = M_->getWeightForNode(v);
    assert(!externalTensors_.count(w) && "The tensor is already registered");
    externalTensors_[w] = &v->getPayload();
  }

  for (auto *W : M_->getWeights()) {
    getOrCreateTensor(W);
  }
}

Tensor *Interpreter::getTensor(const Value *v) const {
  auto ie = externalTensors_.find(v);
  if (ie != externalTensors_.end()) {
    return ie->second;
  }

  auto it = tensors_.find(v);
  assert(it != tensors_.end() && "Unknown key Value.");
  return it->second;
}

Handle<float> Interpreter::getWeightHandle(Value *v) const {
  return getTensor(v)->getHandle<>();
}

Tensor *Interpreter::getOrCreateTensor(const Value *v) {
  auto ie = externalTensors_.find(v);
  if (ie != externalTensors_.end()) {
    return ie->second;
  }

  // Pick the tensor.
  auto it = tensors_.find(v);
  if (it == tensors_.end()) {
    auto *T = new Tensor(v->getType());
    tensors_[v] = T;
    return T;
  }
  return it->second;
}

Tensor *Interpreter::getOrCreateUnownedTensor(const Value *v,
                                              const Value *src) {
  assert(isa<TensorViewInst>(v) && "Expected a tensor view");

  // Pick the tensor.
  auto it = tensors_.find(v);

  // Release unowned tensors before re-creating them.
  if (it != tensors_.end()) {
    deleteTensor(v);
  }

  auto *T = new Tensor();
  *T = getTensor(src)->getUnowned(v->dims());
  tensors_[v] = T;
  return T;
}

void Interpreter::deleteTensor(const Value *v) {
  auto it = tensors_.find(v);
  if (it == tensors_.end()) {
    return;
  }

  delete it->second;
  tensors_.erase(it);
}

void Interpreter::doForwardPass(bool isTrain) {
// Do the forward pass.
#define DEF_VALUE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME)                                                 \
  case Kinded::Kind::CLASS##Kind: {                                            \
    fwd##CLASS(isTrain, llvm::cast<CLASS>(I));                                 \
    break;                                                                     \
  }
  // Dispatch the interpreter on each instruction in the program:
  for (auto *I : M_->getInstrs()) {
    switch (I->getKind()) {
#include "AutoGenInstr.def"

    default:
      llvm_unreachable("Invalid instruction.");
    }
  }
}
