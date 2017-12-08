// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/ExecutionEngine/ExecutionEngine.h"

#include "glow/Backends/Backend.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"

using namespace glow;

ExecutionEngine::ExecutionEngine(BackendKind backendKind) {
  G_ = std::unique_ptr<Graph>(new Graph());
  M_ = std::unique_ptr<Module>(new Module(&*G_));
  IP_ = std::unique_ptr<Backend>(createBackend(backendKind, &*M_));
}

ExecutionEngine::~ExecutionEngine() = default;

void ExecutionEngine::run(llvm::ArrayRef<Variable *> vars,
                          llvm::ArrayRef<Tensor *> inputs) {
  assert(!inputs.empty() && "No inputs");
  assert(inputs.size() == vars.size() &&
         "The number of inputs does not match the number of variables");

  // Update the input variables.
  for (int i = 0, e = vars.size(); i < e; i++) {
    loadValueFromTensor(vars[i], inputs[i]);
  }

  IP_->doForwardPass(false);
}

void ExecutionEngine::runBatch(size_t iterations,
                               llvm::ArrayRef<Variable *> vars,
                               llvm::ArrayRef<Tensor *> inputs) {
  static size_t trainCounter = 0;

  assert(!inputs.empty() && "No inputs");
  assert(inputs.size() == vars.size() &&
         "The number of inputs does not match the number of variables");

  // This is the size of one batch (the number of samples in the batch).
  size_t batchSize = vars[0]->getType()->dims()[0];

  for (size_t i = 0; i < iterations; i++) {
    // Launch threads that update the different chunks in the batch:
    updateForwardBackward(vars, inputs, trainCounter);

    trainCounter += batchSize;
  }
}

void ExecutionEngine::updateForwardBackward(llvm::ArrayRef<Variable *> vars,
                                            llvm::ArrayRef<Tensor *> inputs,
                                            size_t sampleIdx) {
  // Update the input variables.
  for (int i = 0, e = vars.size(); i < e; i++) {
    loadValueFromTensorSlice(vars[i], inputs[i], sampleIdx);
  }

  IP_->doForwardPass(true);
}

void ExecutionEngine::loadValueFromTensorSlice(const Variable *v, Tensor *input,
                                               size_t sampleIdx) {
  assert(v && "Invalid value");
  auto *t = IP_->getTensor(v);

  auto dim = input->dims();
  assert(t->dims().drop_front() == dim.drop_front() && "Invalid slice size");
  // Extract the n'th slice, that must be a tensor.
  size_t slc = sampleIdx % dim[0];
  t->copyConsecutiveSlices(input, slc);
}

void ExecutionEngine::loadValueFromTensor(const Variable *v, Tensor *input) {
  assert(v && "Invalid value");
  auto *t = IP_->getTensor(v);
  auto dim = input->dims();
  (void)dim;
  assert(t->dims() == dim && "Invalid slice size");
  t->copyFrom(input);
}

void ExecutionEngine::compile(CompilationMode mode) {
  // Wipe out the module and start a new compilation process.
  M_->clear();
  IP_->clear();

  if (mode != CompilationMode::Infer) {
    generateGradientNodes(*G_, getConfig(), mode);
  }

  ::glow::optimize(*G_, mode);
  M_->generateIR(mode);
  ::glow::optimize(*M_, mode);

  for (auto &v : G_->getVars()) {
    auto *w = M_->getWeightForNode(v);
    IP_->registerGraphTensor(w, &v->getPayload());
  }

  IP_->init();
}

Tensor *ExecutionEngine::getWeight(const Variable *v) const {
  return IP_->getTensor(v);
}
