#include "noether/Network.h"
#include "noether/Image.h"
#include "noether/Nodes.h"
#include "noether/Tensor.h"

#include <iostream>
#include <unordered_set>
#include <thread>

using namespace noether;

Network::Network() : numThreads_(std::thread::hardware_concurrency()) { }

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

/// Train the network starting with the node \p root. Perform \p iterations
/// iterations in the training loop. Update the nodes in \p nodes with the
/// values \p inputs.
void Network::train(NodeBase *root, size_t iterations,
                    ArrayRef<NodeBase *> nodes, ArrayRef<Tensor *> inputs) {
  for (size_t i = 0; i < iterations; i++) {
    // Update all of the inputs of all of the relevant nodes:
    for (int i = 0, e = nodes.size(); i < e; i++) {
      nodes[i]->updateInputs(&ctx0_, inputs[i], trainCounter_);
    }

    // Forward scan.
    ForwardPass FP(&ctx0_);
    root->visit(&FP);

    // Backward scan in reverse order.
    BackwardPass BP(&ctx0_);
    root->visit(&BP);

    trainCounter_++;

    // Only update the gradient when we've reached the end of the batch.
    if (trainCounter_ % getConfig().batchSize)
      continue;

    for (auto &p : ctx0_.trainables_) {
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

  // Update all inputs.
  for (int i = 0, e = nodes.size(); i < e; i++) {
    nodes[i]->updateInput(&ctx0_, inputs[i]);
  }

  // Forward scan.
  ForwardPass FP(&ctx0_);
  root->visit(&FP);

  // Backward scan in reverse order.
  BackwardPass BP(&ctx0_);
  root->visit(&BP);

  trainCounter_++;

  // Only update the gradient when we've reached the end of the batch.
  if (trainCounter_ % getConfig().batchSize)
    return;

  for (auto &p : ctx0_.trainables_) {
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
    nodes[i]->updateInput(&ctx0_, inputs[i]);
  }

  // Forward scan.
  ForwardPass FP(&ctx0_);
  root->visit(&FP);

  return &root->getOutput(&ctx0_)->weights_;
}

void Network::dump(NodeBase *root) {
  std::cout << "Network structure:";

  // Print all of the nodes in the network.
  PrinterPass FP;
  root->visit(&FP);

  std::cout << "\n";
}

