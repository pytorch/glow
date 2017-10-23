// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/ExecutionEngine/ExecutionEngine.h"

#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "glow/Interpreter/Interpreter.h"
#include "glow/Optimizer/Optimizer.h"

using namespace glow;

ExecutionEngine::ExecutionEngine()
    : G_(std::make_unique<Graph>()), M_(std::make_unique<Module>(&*G_)),
      IP_(std::make_unique<Interpreter>(&*M_)) {}

ExecutionEngine::~ExecutionEngine() = default;

void ExecutionEngine::infer(llvm::ArrayRef<Variable *> vars,
                            llvm::ArrayRef<Tensor *> inputs) {
  assert(!inputs.empty() && "No inputs");
  assert(inputs.size() == vars.size() &&
         "The number of inputs does not match the number of variables");

  // Update the input variables.
  for (int i = 0, e = vars.size(); i < e; i++) {
    auto *val = M_->getWeightForNode(vars[i]);
    loadValueFromTensor(val, inputs[i], 0);
  }

  IP_->doForwardPass(false);
}

void ExecutionEngine::train(size_t iterations, llvm::ArrayRef<Variable *> vars,
                            llvm::ArrayRef<Tensor *> inputs) {
  static size_t trainCounter = 0;

  assert(!inputs.empty() && "No inputs");
  assert(inputs.size() == vars.size() &&
         "The number of inputs does not match the number of variables");

  std::vector<Value *> weights;
  for (auto *v : vars) {
    weights.push_back(M_->getWeightForNode(v));
  }

  // This is the size of one batch (the number of samples in the batch).
  size_t batchSize = vars[0]->dims()[0];

  for (size_t i = 0; i < iterations; i++) {
    // Launch threads that update the different chunks in the batch:
    updateForwardBackward(weights, inputs, trainCounter + batchSize);

    trainCounter += batchSize;

    // The algorithm for merging the state from the different threads is
    /// described in the paper: Alex Krizhevsky [2014]
    // "One weird trick for parallelizing convolutional neural networks"
    learnGradient(batchSize);
  }
}

void ExecutionEngine::learnGradient(size_t batchSize) {
  for (auto *V : M_->getWeights()) {
    // Do not try to learn the values of input/output buffers.
    if (V->getKind() == WeightVar::MutabilityKind::Constant) {
      continue;
    }

    auto W = IP_->getTensor(V);
    auto G = IP_->getOrCreateGradTensor(V);

    // Handle weight update by learning the gradients into the weights.
    trainer_.train(W, G, batchSize);
  }
}

void ExecutionEngine::updateForwardBackward(llvm::ArrayRef<Value *> vars,
                                            llvm::ArrayRef<Tensor *> inputs,
                                            size_t sampleIdx) {
  // Update the input variables.
  for (int i = 0, e = vars.size(); i < e; i++) {
    loadValueFromTensor(vars[i], inputs[i], sampleIdx);
  }

  IP_->doForwardPass(true);

  IP_->doBackwardPass();
}

void ExecutionEngine::loadValueFromTensor(const Value *v, Tensor *input,
                                          size_t sampleIdx) {
  assert(v && "Invalid value");
  auto *t = IP_->getTensor(v);

  auto dim = input->dims();
  assert(t->dims().drop_front() == dim.drop_front() && "Invalid slice size");
  // Extract the n'th slice, that must be a tensor.
  size_t slc = sampleIdx % dim[0];
  t->copyConsecutiveSlices(input, slc);
}

void ExecutionEngine::compile(OptimizationMode mode) {
  // Wipe out the module and start a new compilation process.
  M_->clear();
  IP_->clear();
  ::glow::optimize(*G_, mode);
  M_->generateIR();
  ::glow::optimize(*M_, mode);

  for (auto &v : G_->getVars()) {
    auto *w = M_->getWeightForNode(v);
    IP_->registerGraphTensor(w, &v->getPayload());
  }

  for (auto *W : M_->getWeights()) {
    IP_->getOrCreateTensor(W);
  }
}

void ExecutionEngine::optimize(OptimizationMode mode) {
  ::glow::optimize(*M_, mode);
}

/// \returns a float-handle to the tensor that is stored at \p v.
Handle<float>
ExecutionEngine::getWeightHandle(Variable *v) const {
  auto val = M_->getWeightForNode(v);
  return IP_->getWeightHandle(val);
}

/// \returns a float-handle to the tensor that is stored at \p v.
Handle<float> ExecutionEngine::getGradHandle(Variable *v) {
  auto val = M_->getWeightForNode(v);
  return IP_->getGradHandle(val);
}
