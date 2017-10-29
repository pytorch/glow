// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/ExecutionEngine/ExecutionEngine.h"

#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "glow/Interpreter/Interpreter.h"
#include "glow/Optimizer/Optimizer.h"

using namespace glow;

ExecutionEngine::ExecutionEngine() {
  G_ = std::unique_ptr<Graph>(new Graph());
  M_ = std::unique_ptr<Module>(new Module(&*G_));
  IP_ = std::unique_ptr<Interpreter>(new Interpreter(&*M_));
}

ExecutionEngine::~ExecutionEngine() = default;

void ExecutionEngine::infer(llvm::ArrayRef<Variable *> vars,
                            llvm::ArrayRef<Tensor *> inputs) {
  assert(!inputs.empty() && "No inputs");
  assert(inputs.size() == vars.size() &&
         "The number of inputs does not match the number of variables");

  // Update the input variables.
  for (int i = 0, e = vars.size(); i < e; i++) {
    loadValueFromTensor(vars[i], inputs[i], 0);
  }

  IP_->doForwardPass(false);
}

void ExecutionEngine::train(size_t iterations, llvm::ArrayRef<Variable *> vars,
                            llvm::ArrayRef<Tensor *> inputs) {
  static size_t trainCounter = 0;

  assert(!inputs.empty() && "No inputs");
  assert(inputs.size() == vars.size() &&
         "The number of inputs does not match the number of variables");

  // This is the size of one batch (the number of samples in the batch).
  size_t batchSize = vars[0]->dims()[0];

  for (size_t i = 0; i < iterations; i++) {
    // Launch threads that update the different chunks in the batch:
    updateForwardBackward(vars, inputs, trainCounter + batchSize);

    trainCounter += batchSize;

    // The algorithm for merging the state from the different threads is
    /// described in the paper: Alex Krizhevsky [2014]
    // "One weird trick for parallelizing convolutional neural networks"
    learnGradient(batchSize);
  }
}

void ExecutionEngine::learnGradient(size_t batchSize) {
  for (auto *V : G_->getVars()) {
    // Do not try to learn the values of input/output buffers.
    if (V->getInitKind() == Variable::InitKind::Extern) {
      continue;
    }

    auto W = IP_->getTensor(V);
    auto G = IP_->getGradTensor(V);

    // Handle weight update by learning the gradients into the weights.
    trainer_.train(W, G, batchSize);
  }
}

void ExecutionEngine::updateForwardBackward(llvm::ArrayRef<Variable *> vars,
                                            llvm::ArrayRef<Tensor *> inputs,
                                            size_t sampleIdx) {
  // Update the input variables.
  for (int i = 0, e = vars.size(); i < e; i++) {
    loadValueFromTensor(vars[i], inputs[i], sampleIdx);
  }

  IP_->doForwardPass(true);
}

void ExecutionEngine::loadValueFromTensor(const Variable *v, Tensor *input,
                                          size_t sampleIdx) {
  assert(v && "Invalid value");
  auto *t = IP_->getTensor(v);

  auto dim = input->dims();
  assert(t->dims().drop_front() == dim.drop_front() && "Invalid slice size");
  // Extract the n'th slice, that must be a tensor.
  size_t slc = sampleIdx % dim[0];
  t->copyConsecutiveSlices(input, slc);
}

void ExecutionEngine::compile(CompilationMode mode) {
  // Wipe out the module and start a new compilation process.
  M_->clear();
  IP_->clear();
  ::glow::optimize(*G_, mode);
  M_->generateIR(mode);
  ::glow::optimize(*M_, mode);

  for (auto &v : G_->getVars()) {
    auto *w = M_->getWeightForNode(v);
    IP_->registerGraphTensor(w, &v->getPayload());
  }

  IP_->init();
}

/// \returns a float-handle to the tensor that is stored at \p v.
Handle<float> ExecutionEngine::getWeightHandle(Variable *v) const {
  return IP_->getWeightHandle(v);
}

/// \returns a float-handle to the tensor that is stored at \p v.
Handle<float> ExecutionEngine::getGradHandle(Variable *v) {
  return IP_->getGradHandle(v);
}
