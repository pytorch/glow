// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Interpreter/Interpreter.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"
#include "glow/Support/Casting.h"

using namespace glow;

Interpreter::~Interpreter() { clear(); }

void Interpreter::clear() {
  // Delete the tensors that are owned by this module.
  for (auto p : tensors_) {
    delete p.second;
  }

  // Delete the attached gradients.
  for (auto &p : gradients_) {
    delete p.second;
  }

  tensors_.clear();
  gradients_.clear();
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

void Interpreter::registerGraphTensor(const Value *v, Tensor *t) {
  assert(!externalTensors_.count(v) && "The tensor is already registered");
  externalTensors_[v] = t;
}

Tensor *Interpreter::getOrCreateGradTensor(const Value *v) {
  auto *T = getTensor(v);
  auto it = gradients_.find(T);
  if (it != gradients_.end()) {
    return it->second;
  }

  // Create a new tensor, register it and return it.
  auto *N = new Tensor(T->getType());
  gradients_[T] = N;
  return N;
}

Handle<float> Interpreter::getWeightHandle(Value *v) const {
  return getTensor(v)->getHandle<>();
}

Handle<float> Interpreter::getGradHandle(Value *v) {
  return getOrCreateGradTensor(v)->getHandle<>();
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

void Interpreter::deleteTensor(const Value *v) {
  auto it = tensors_.find(v);
  if (it == tensors_.end()) {
    return;
  }

  delete it->second;
  tensors_.erase(it);
}

bool Interpreter::hasTensor(const Value *v) { return tensors_.count(v); }

void Interpreter::doForwardPass(bool isTrain) {
  // Do the forward pass.
#define DEF_VALUE(CLASS, NAME)
#define DEF_NODE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME)                                                 \
  case Kinded::Kind::CLASS##Kind: {                                            \
    fwd##CLASS(isTrain, cast<CLASS>(I));                                       \
    break;                                                                     \
  }
  // Dispatch the interpreter on each instruction in the program:
  for (auto *I : M_->getInstrs()) {
    switch (I->getKind()) {
#include "glow/IR/Instrs.def"
    default:
      glow_unreachable();
    }
  }
}

void Interpreter::doBackwardPass() {
  // Do the backward pass.
#define DEF_VALUE(CLASS, NAME)
#define DEF_NODE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME)                                                 \
  case Kinded::Kind::CLASS##Kind: {                                            \
    bwd##CLASS(cast<CLASS>(*it));                                              \
    break;                                                                     \
  }
  // Dispatch the interpreter on each instruction in the program, in reverse
  // order.
  auto &L = M_->getInstrs();
  for (auto it = L.rbegin(), e = L.rend(); it != e; it++) {
    switch ((*it)->getKind()) {
#include "glow/IR/Instrs.def"
    default:
      glow_unreachable();
    }
  }
}
