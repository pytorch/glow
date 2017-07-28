#include "noether/Network.h"
#include "noether/Image.h"
#include "noether/Nodes.h"
#include "noether/Tensor.h"

#include <iostream>
#include <unordered_set>
#include <thread>

using namespace noether;

Context::~Context() {
  for (auto t : trainables_) {
    delete t.second;
  }
  for (auto t : tensors_) {
    delete t.second;
  }
}

Handle<FloatTy> Context::getWeightHandle(const TensorToken *tok) {
  return getTrainable(tok)->getWeightHandle();
}

Handle<FloatTy> Context::getGradHandle(const TensorToken *tok) {
  return getTrainable(tok)->getGradHandle();
}

void Context::allocateTrainable(const TensorToken *tok, bool trainable,
                       ArrayRef<size_t> dims) {
  assert(!trainables_.count(tok) && "Token already allocated");
  trainables_[tok] = new TrainableData(trainable, dims);
}

TrainableData *Context::getTrainable(const TensorToken *tok) {
  assert(trainables_.count(tok) && "The token was not allocated");
  return trainables_[tok];
}

void Context::allocateTensor(const TensorToken *tok, ElemKind kind,
                    ArrayRef<size_t> dims) {
  assert(!tensors_.count(tok) && "Token already allocated");
  tensors_[tok] = new Tensor(kind, dims);
}

Tensor *Context::getTensor(const TensorToken *tok) {
  assert(tensors_.count(tok) && "The token was not allocated");
  return tensors_[tok];
}

Network::Network() {
  for (unsigned i = 0, e = std::thread::hardware_concurrency(); i < e; i++) {
    state_.emplace_back(i);
  }
}

Network::~Network() {
  /// Delete the nodes of the network.
  for (auto *node : networkNodes_) {
    delete node;
  }
}

ConvNode *Network::createConvNode(NodeBase *input, size_t outDepth,
                                  size_t filterSize, size_t stride,
                                  size_t pad) {
  return addNode(new ConvNode(this, input, outDepth, filterSize, stride, pad));
}

MaxPoolNode *Network::createMaxPoolNode(NodeBase *input, size_t filterSize,
                                        size_t stride, size_t pad) {
  return addNode(new MaxPoolNode(this, input, filterSize, stride, pad));
}

FullyConnectedNode *Network::createFullyConnectedNode(NodeBase *input,
                                                      size_t outDepth) {
  return addNode(new FullyConnectedNode(this, input, outDepth));
}

RELUNode *Network::createRELUNode(NodeBase *input) {
  return addNode(new RELUNode(this, input));
}

SigmoidNode *Network::createSigmoidNode(NodeBase *input) {
  return addNode(new SigmoidNode(this, input));
}

SoftMaxNode *Network::createSoftMaxNode(NodeBase *input) {
  return addNode(new SoftMaxNode(this, input));
}

RegressionNode *Network::createRegressionNode(NodeBase *input) {
  return addNode(new RegressionNode(this, input));
}

MaxNode *Network::createMaxNode(NodeBase *input) {
  return addNode(new MaxNode(this, input));
}

ArrayNode *Network::createArrayNode(ArrayRef<size_t> dims) {
  return addNode(new ArrayNode(this, dims));
}

namespace {

struct BackwardPass : NodeVisitor {
  Context *ctx_;
  BackwardPass(Context *ctx) : ctx_(ctx) {}
  virtual void pre(NodeBase *N) override { N->backward(ctx_); }
};

struct ForwardPass : NodeVisitor {
  Context *ctx_;
  ForwardPass(Context *ctx) : ctx_(ctx) {}
  virtual void post(NodeBase *N) override { N->forward(ctx_); }
};

struct PrinterPass : NodeVisitor {
  virtual void post(NodeBase *N) override { std::cout << N->getName() << "->"; }
};

} // namespace

void Network::updateForwardBackward(Context *ctx, NodeBase *root, size_t start,
                           size_t len, ArrayRef<NodeBase *> nodes,
                           ArrayRef<Tensor *> inputs, bool isBatch) {
  for (size_t idx = 0; idx < len; idx++) {
    /// Update the inputs:
    for (int i = 0, e = nodes.size(); i < e; i++) {
      if (isBatch) {
      nodes[i]->updateInputs(ctx, inputs[i], start + idx);
      } else {
        nodes[i]->updateInput(ctx, inputs[i]);
      }
    }

    // Forward scan:
    ForwardPass FP(ctx);
    root->visit(&FP);

    // Backward scan in reverse order:
    BackwardPass BP(ctx);
    root->visit(&BP);
  }
}

/// Train the network starting with the node \p root. Perform \p iterations
/// iterations in the training loop. Update the nodes in \p nodes with the
/// values \p inputs.
void Network::train(NodeBase *root, size_t iterations,
                    ArrayRef<NodeBase *> nodes, ArrayRef<Tensor *> inputs) {
  for (size_t i = 0; i < iterations; i++) {
    updateForwardBackward(&state_[0], root, trainCounter_, 1, nodes, inputs,
                          true);
    trainCounter_++;

    // Only update the gradient when we've reached the end of the batch.
    if (trainCounter_ % getConfig().batchSize)
      continue;

    for (auto &p : state_[0].trainables_) {
      // Update the weights.
      trainer_.train(p.second);
      // Clear the gradients for the next round of training.
      p.second->clearGradient();
    }
  }
}

/// Perform a single training iteration for one input. Update the nodes in \p
/// nodes with the values \p inputs.
void Network::train(NodeBase *root, ArrayRef<NodeBase *> nodes,
                    ArrayRef<Tensor *> inputs) {
  assert(nodes.size() == inputs.size() && "Mismatched argument list");

  updateForwardBackward(&state_[0], root, trainCounter_, 1, nodes, inputs,
                        false);

  trainCounter_++;

  // Only update the gradient when we've reached the end of the batch.
  if (trainCounter_ % getConfig().batchSize)
    return;

  for (auto &p : state_[0].trainables_) {
    // Update the weights.
    trainer_.train(p.second);
    // Clear the gradients for the next round of training.
    p.second->clearGradient();
  }
}

Tensor *Network::infer(NodeBase *root, ArrayRef<NodeBase *> nodes,
                    ArrayRef<Tensor *> inputs) {
  // Update all inputs.
  for (int i = 0, e = nodes.size(); i < e; i++) {
    nodes[i]->updateInput(&state_[0], inputs[i]);
  }

  // Forward scan.
  ForwardPass FP(&state_[0]);
  root->visit(&FP);

  return &root->getOutput(&state_[0])->weights_;
}

void Network::dump(NodeBase *root) {
  std::cout << "Network structure:";

  // Print all of the nodes in the network.
  PrinterPass FP;
  root->visit(&FP);

  std::cout << "\n";
}

