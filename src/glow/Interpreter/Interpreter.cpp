// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Interpreter/Interpreter.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/Support/Casting.h"

using namespace glow;

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

void Interpreter::registerGraphTensor(const Value *v, Tensor *t) {
  assert(!externalTensors_.count(v) && "The tensor is already registered");
  externalTensors_[v] = t;
}

Tensor *Interpreter::getTensor(const Variable *v) const {
  auto *W = M_->getWeightForNode(v);
  return getTensor(W);
}

Tensor *Interpreter::getGradTensor(const Variable *v) const {
  auto *W = M_->getWeightForNode(v);
  return getGradTensor(W);
}

Handle<float> Interpreter::getWeightHandle(Variable *v) const {
  auto *W = M_->getWeightForNode(v);
  return getWeightHandle(W);
}

Handle<float> Interpreter::getGradHandle(Variable *v) const {
  auto *W = M_->getWeightForNode(v);
  return getGradHandle(W);
}

Handle<float> Interpreter::getWeightHandle(Value *v) const {
  return getTensor(v)->getHandle<>();
}

Handle<float> Interpreter::getGradHandle(Value *v) const {
  return getGradTensor(v)->getHandle<>();
}

Tensor *Interpreter::getGradTensor(const Value *v) const {
  auto &map = M_->getGradientMap();
  auto it = map.find(v);
  assert(it != map.end() && "Gradient tensor unavailable");
  return getTensor(it->second);
}

bool Interpreter::hasGradTensor(const Value *v) const {
  return M_->getGradientMap().count(v);
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
      glow_unreachable();
    }
  }
}
