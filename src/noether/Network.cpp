#include "noether/Network.h"
#include "noether/Image.h"
#include "noether/Nodes.h"
#include "noether/Tensor.h"

#include <iostream>
#include <unordered_set>

using namespace noether;

Network::Network() {}

Network::~Network() {
  /// Delete the gradient tensors.
  for (auto &tp : gradientTensors_) {
    delete tp.second;
  }

  /// Delete the nodes of the network.
  for (auto *node : networkNodes_) {
    delete node;
  }
}

ConvNode *Network::createConvNode(TrainableNode *input, size_t outDepth,
                                  size_t filterSize, size_t stride,
                                  size_t pad) {
  return addNode(new ConvNode(this, input, outDepth, filterSize, stride, pad));
}

MaxPoolNode *Network::createMaxPoolNode(TrainableNode *input, size_t filterSize,
                                        size_t stride, size_t pad) {
  return addNode(new MaxPoolNode(this, input, filterSize, stride, pad));
}

FullyConnectedNode *Network::createFullyConnectedNode(TrainableNode *input,
                                                      size_t outDepth) {
  return addNode(new FullyConnectedNode(this, input, outDepth));
}

RELUNode *Network::createRELUNode(TrainableNode *input) {
  return addNode(new RELUNode(this, input));
}

SigmoidNode *Network::createSigmoidNode(TrainableNode *input) {
  return addNode(new SigmoidNode(this, input));
}

SoftMaxNode *Network::createSoftMaxNode(TrainableNode *input) {
  return addNode(new SoftMaxNode(this, input));
}

RegressionNode *Network::createRegressionNode(TrainableNode *input) {
  return addNode(new RegressionNode(this, input));
}

MaxNode *Network::createMaxNode(TrainableNode *input) {
  return addNode(new MaxNode(this, input));
}

ArrayNode *Network::createArrayNode(ArrayRef<size_t> dims) {
  return addNode(new ArrayNode(this, dims));
}

namespace {

struct BackwardPass : NodeVisitor {
  virtual void pre(NodeBase *N) override { N->backward(); }
};

struct ForwardPass : NodeVisitor {
  virtual void post(NodeBase *N) override { N->forward(); }
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
      nodes[i]->updateInputs(inputs[i], trainCounter_);
    }

    // Forward scan.
    ForwardPass FP;
    root->visit(&FP);

    // Backward scan in reverse order.
    BackwardPass BP;
    root->visit(&BP);

    trainCounter_++;

    // Only update the gradient when we've reached the end of the batch.
    if (trainCounter_ % getConfig().batchSize)
      continue;

    for (auto &p : gradientTensors_) {
      // Update the weights.
      trainer_.train(p.first, p.second);
      // Clear the gradients for the next round of training.
      p.second->zero();
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
    nodes[i]->updateInput(inputs[i]);
  }

  // Forward scan.
  ForwardPass FP;
  root->visit(&FP);

  // Backward scan in reverse order.
  BackwardPass BP;
  root->visit(&BP);

  trainCounter_++;

  // Only update the gradient when we've reached the end of the batch.
  if (trainCounter_ % getConfig().batchSize)
    return;

  for (auto &p : gradientTensors_) {
    // Update the weights.
    trainer_.train(p.first, p.second);
    // Clear the gradients for the next round of training.
    p.second->zero();
  }
}

void Network::infer(NodeBase *root, ArrayRef<NodeBase *> nodes,
                    ArrayRef<Tensor *> inputs) {
  // Update all inputs.
  for (int i = 0, e = nodes.size(); i < e; i++) {
    nodes[i]->updateInput(inputs[i]);
  }

  // Forward scan.
  ForwardPass FP;
  root->visit(&FP);
}

void Network::dump(NodeBase *root) {
  std::cout << "Network structure:";

  // Print all of the nodes in the network.
  PrinterPass FP;
  root->visit(&FP);

  std::cout << "\n";
}


void Network::allocateGradientTensor(Tensor *weights) {
  assert(!gradientTensors_.count(weights) &&
         "Already allocated gradient tensor for this weight tensor");
  // Allocate and register a new tensor with the same type and dimensions as the
  // weights tensor.
  Tensor *T = new Tensor();
  T->reset(weights);
  gradientTensors_[weights] = T;
}

Tensor *Network::getGradientTensor(Context *ctx, Tensor *weights) {
  // At this point ignore the context.

  assert(gradientTensors_.count(weights) &&
         "Gradient tensor was not allocated!");
  return gradientTensors_[weights];


}
