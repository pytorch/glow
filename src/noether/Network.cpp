#include "noether/Network.h"
#include "noether/Image.h"
#include "noether/Nodes.h"
#include "noether/Tensor.h"

#include <iostream>
#include <unordered_set>

using namespace noether;

Network::Network() {}

Network::~Network() {
  for (auto *node : this->networkNodes_) {
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

void Network::registerDerivTensor(NodeBase *node, TrainableData *weights) {
  trainableBuffers_.push_back(weights);
}

namespace {

struct BackwardPass : NodeVisitor {
  virtual void pre(NodeBase *N) override { N->backward(); }
};

struct ForwardPass : NodeVisitor {
  virtual void post(NodeBase *N) override { N->forward(); }
};

struct UpdaterPass : NodeVisitor {
  size_t index{0};
  UpdaterPass(size_t index) : index(index) {}

  virtual void post(NodeBase *N) override {
    N->updateBoundInputs(index);
  }
};

struct PrinterPass : NodeVisitor {
  virtual void post(NodeBase *N) override { std::cout<< N->getName() << "->"; }
};


} // namespace

void Network::train(NodeBase *root, size_t iterations) {
  size_t numInputs = trainConf_.inputSize;

  for (size_t i = 0; i < iterations; i++) {
    // Ask all of the nodes to update the input for the specific input by index.
    UpdaterPass UP(trainCounter_ % numInputs);
    root->visit(&UP);

    // Forward scan.
    ForwardPass FP;
    root->visit(&FP);

    // Backward scan in reverse order.
    BackwardPass BP;
    root->visit(&BP);

    trainCounter_++;

    // Only update the gradient when we've reached the end of the batch.
    if (trainCounter_ % trainConf_.batchSize)
      continue;

    // Update the gradients.
    for (auto &buffer : trainableBuffers_) {
      buffer->train(trainConf_);
    }

    for (auto &buffer : trainableBuffers_) {
      buffer->clearGradient();
    }
  }
}

void Network::infer(NodeBase *root) {
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

  std::cout << "Buffers content:\n";

  for (auto &buffer : trainableBuffers_) {
    buffer->dump();
  }

  std::cout << "\n";
}
