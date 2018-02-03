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
  backendKind_ = backendKind;
  G_.reset(new Graph());
  M_.reset(new Module(&*G_));
  IP_.reset(createBackend(backendKind_, &*M_));
}

// Set the code generator kind to \p backendKind.
void ExecutionEngine::setBackend(BackendKind backendKind) {
  backendKind_ = backendKind;
  IP_.reset(createBackend(backendKind, &*M_));
}

void ExecutionEngine::reset() {
  if (M_)
    M_->clear();
  IP_.reset(createBackend(backendKind_, &*M_));
  if (G_)
    G_->resetState();
}

ExecutionEngine::~ExecutionEngine() = default;

void ExecutionEngine::run(llvm::ArrayRef<Variable *> vars,
                          llvm::ArrayRef<Tensor *> inputs) {
  assert(inputs.size() == vars.size() &&
         "The number of inputs does not match the number of variables");
  assert(!M_->getInstrs().empty() && "Running a Module with no instructions.");

  // Update the input variables.
  for (int i = 0, e = vars.size(); i < e; i++) {
    assert(vars[i]->getVisibilityKind() == Variable::VisibilityKind::Public &&
           "Trying to update a private variable");
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
  assert(!M_->getInstrs().empty() && "Running a Module with no instructions.");

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

void ExecutionEngine::loadValueFromTensorSlice(Variable *v, Tensor *input,
                                               size_t sampleIdx) {
  assert(v && "Invalid value");
  auto &t = v->getPayload();

  auto dim = input->dims();
  assert(t.dims().drop_front() == dim.drop_front() && "Invalid slice size");
  // Extract the n'th slice, that must be a tensor.
  size_t slc = sampleIdx % dim[0];
  t.copyConsecutiveSlices(input, slc);
}

void ExecutionEngine::loadValueFromTensor(Variable *v, Tensor *input) {
  assert(v && "Invalid value");
  auto &t = v->getPayload();
  auto dim = input->dims();
  (void)dim;
  assert(t.dims() == dim && "Invalid slice size");
  t.copyFrom(input);
}

void ExecutionEngine::compile(CompilationMode mode) {
  // Reset the engine and start a new compilation process.
  reset();

  if (mode != CompilationMode::Infer) {
    generateGradientNodes(*G_, getConfig(), mode);
  }

  // Optimized the graph.
  ::glow::optimize(*G_, mode);

  // Lower the graph into a sequence of low-level linear algebra operations.
  ::glow::lower(*G_, mode);

  // Optimized the graph again.
  ::glow::optimize(*G_, mode);

  // Allow the backend to transform the graph.
  if (IP_->transform(*G_)) {
    // Optimize the graph again after the backend transformation.
    // In particular, DCE is very likely to be useful.
    ::glow::optimize(*G_, mode);
  }

  // Generate IR from the graph.
  M_->generateIR(mode);

  // Optimize the generated IR.
  ::glow::optimize(*M_, mode);
  IP_->init();
}
