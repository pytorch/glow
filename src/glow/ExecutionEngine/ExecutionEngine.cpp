// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/ExecutionEngine/ExecutionEngine.h"

using namespace glow;

void ExecutionEngine::infer(llvm::ArrayRef<Variable *> vars,
                            llvm::ArrayRef<Tensor *> inputs) {
  assert(!inputs.empty() && "No inputs");
  assert(inputs.size() == vars.size() &&
         "The number of inputs does not match the number of variables");

  // Update the input variables.
  for (int i = 0, e = vars.size(); i < e; i++) {
    auto *val = M_.getWeightForNode(vars[i]);
    loadValueFromTensor(val, inputs[i], 0);
  }

  IP_.doForwardPass(false);
}

void ExecutionEngine::train(size_t iterations, llvm::ArrayRef<Variable *> vars,
                            llvm::ArrayRef<Tensor *> inputs) {
  static size_t trainCounter = 0;

  assert(!inputs.empty() && "No inputs");
  assert(inputs.size() == vars.size() &&
         "The number of inputs does not match the number of variables");

  std::vector<Value *> weights;
  for (auto *v : vars) {
    weights.push_back(M_.getWeightForNode(v));
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
  for (auto *V : M_.getWeights()) {
    // Do not try to learn the values of input/output buffers.
    if (V->getInitKind() == WeightVar::InitKind::Extern) {
      continue;
    }

    auto W = IP_.getTensor(V);
    auto G = IP_.getOrCreateGradTensor(V);

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

  IP_.doForwardPass(true);

  IP_.doBackwardPass();
}

void ExecutionEngine::loadValueFromTensor(const Value *v, Tensor *input,
                                          size_t sampleIdx) {
  assert(v && "Invalid value");
  auto *t = IP_.getTensor(v);

  auto dim = input->dims();
  assert(t->dims().drop_front() == dim.drop_front() && "Invalid slice size");
  // Extract the n'th slice, that must be a tensor.
  size_t slc = sampleIdx % dim[0];
  t->copyConsecutiveSlices(input, slc);
}

void ExecutionEngine::optimize(OptimizationMode mode) {
  ::glow::optimize(M_, mode);
}

Tensor *ExecutionEngine::getTensor(const Node *v) const {
  auto val = M_.getWeightForNode(v);
  assert(val && "Node does not have a registered IR value");
  return IP_.getTensor(val);
}

/// \returns a float-handle to the tensor that is stored at \p v.
Handle<FloatTy> ExecutionEngine::getWeightHandle(Variable *v) const {
  auto val = M_.getWeightForNode(v);
  return IP_.getWeightHandle(val);
}

/// \returns a float-handle to the tensor that is stored at \p v.
Handle<FloatTy> ExecutionEngine::getGradHandle(Variable *v) {
  auto val = M_.getWeightForNode(v);
  return IP_.getGradHandle(val);
}

/// Copies the content of the tensor \p t into the value \p v.
void ExecutionEngine::initValue(const Variable *v, const Tensor *t) {
  auto *N = M_.getWeightForNode(v);
  return IP_.initValue(N, t);
}

void ExecutionEngine::initVars() {
  for (auto *W : M_.getWeights()) {
    // Don't initialize tensors that are already initialized.
    if (IP_.hasTensor(W)) {
      continue;
    }

    auto *T = IP_.getOrCreateTensor(W);
    // The parameter to the instruction.
    auto val = W->getVal();

    switch (W->getInitKind()) {
    case WeightVar::InitKind::Extern:
      break;

    case WeightVar::InitKind::Broadcast: {
      switch (T->getElementType()) {
      case ElemKind::FloatTy: {
        T->getHandle<float>().clear(val);
        break;
      }
      case ElemKind::DoubleTy: {
        T->getHandle<double>().clear(val);
        break;
      }
      case ElemKind::Int8Ty: {
        T->getHandle<int8_t>().clear(val);
        break;
      };
      case ElemKind::Int32Ty: {
        T->getHandle<int32_t>().clear(val);
        break;
      }
      case ElemKind::IndexTy: {
        T->getHandle<size_t>().clear(val);
        break;
      }
      }
      break;
    }

    case WeightVar::InitKind::Xavier: {
      switch (T->getElementType()) {
      case ElemKind::FloatTy: {
        T->getHandle<float>().randomize(val);
        break;
      }
      case ElemKind::DoubleTy: {
        T->getHandle<double>().randomize(val);
        break;
      }
      case ElemKind::Int8Ty: {
        T->getHandle<int8_t>().randomize(val);
        break;
      };
      case ElemKind::Int32Ty: {
        T->getHandle<int32_t>().randomize(val);
        break;
      }
      case ElemKind::IndexTy: {
        T->getHandle<size_t>().randomize(val);
        break;
      }
      }
      break;
    }
    }
  }
}
