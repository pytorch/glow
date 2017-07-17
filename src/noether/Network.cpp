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

ConvNode* Network::createConvNode(TrainableNode *input, size_t outDepth,
                                  size_t filterSize, size_t stride, size_t pad) {
  return addNode(new ConvNode(this, input, outDepth, filterSize, stride, pad));
}

MaxPoolNode *Network::createMaxPoolNode(TrainableNode *input, size_t filterSize,
                                        size_t stride, size_t pad) {
  return addNode(new MaxPoolNode(this, input, filterSize, stride, pad));
}

FullyConnectedNode* Network::createFullyConnectedNode(TrainableNode *input,
                                                      size_t outDepth) {
  return addNode(new FullyConnectedNode(this, input, outDepth));
}

RELUNode* Network::createRELUNode(TrainableNode *input) {
  return addNode(new RELUNode(this, input));
}

SigmoidNode* Network::createSigmoidNode(TrainableNode *input) {
  return addNode(new SigmoidNode(this, input));
}

SoftMaxNode* Network::createSoftMaxNode(TrainableNode *input) {
  return addNode(new SoftMaxNode(this, input));
}

RegressionNode* Network::createRegressionNode(TrainableNode *input) {
  return addNode(new RegressionNode(this, input));
}

MaxNode* Network::createMaxNode(TrainableNode *input) {
  return addNode(new MaxNode(this, input));
}

ArrayNode* Network::createArrayNode(size_t x, size_t y, size_t z) {
  return addNode(new ArrayNode(this, x, y, z));
}


void Network::sortNetwork(std::vector<NodeBase *> &order) {
  // TODO: add a cycle detector.
  // A list of nodes that were processed.
  std::unordered_set<NodeBase *> visited;
  // Our DFS stack.
  std::vector<NodeBase *> stack;

  // Create a pseudo edge to all nodes in the graph by pusing all of the
  // keys into our stack.
  for (auto &entry : deps_) {
    stack.push_back(entry.first);
  }

  while (!stack.empty()) {
    auto node = stack.back();

    if (visited.count(node)) {
      // We already finished handling this node. Proceed to the next node.
      stack.pop_back();
      continue;
    }

    bool pushed = false;

    // Process the dependencies of the node on the stack:
    if (deps_.count(node)) {
      // For all dependencies of this node:
      for (auto &dep : deps_[node]) {
        // Ignore nodes that we already finished.
        if (visited.count(dep))
          continue;
        stack.push_back(dep);
        pushed = true;
      }
    }

    // If we pushed new nodes to the stack then we need to handle them.
    // Do another iteration now.
    if (pushed) {
      continue;
    }

    // No new dependencies to process. We are done with this node.
    visited.insert(node);
    order.push_back(node);
    stack.pop_back();
  }
  assert(order.size() >= deps_.size() && "Invalid order");
}

void Network::addNodeDependency(NodeBase *node, NodeBase *dep) {
  deps_[node].push_back(dep);
}

void Network::registerDerivTensor(NodeBase *node, TrainableData *weights) {
  trainableBuffers_.push_back(weights);
}

void Network::train() {
  std::vector<NodeBase *> order;
  sortNetwork(order);

  // We clear the gradient here and not as part of the trainign process to ease
  // debugging by leaving the gradients around at the end of the scan.
  for (auto &buffer : trainableBuffers_) {
    buffer->clearGradient();
  }

  // Forward scan.
  for (unsigned i = 0, e = order.size(); i < e; i++) {
    order[i]->forward();
  }

  // Backward scan in reverse order.
  for (unsigned i = 0, e = order.size(); i < e; i++) {
    order[e - i - 1]->backward();
  }

  // Update the gradients.
  for (auto &buffer : trainableBuffers_) {
    buffer->train(trainConf_);
  }
}

void Network::infer() {
  std::vector<NodeBase *> order;
  sortNetwork(order);

  // Forward scan.
  for (unsigned i = 0, e = order.size(); i < e; i++) {
    order[i]->forward();
  }
}

void Network::dump() {
  std::vector<NodeBase *> order;
  sortNetwork(order);

  std::cout << "Network structure:";
  for (auto &node : order) {
    std::cout << node->getName() << " ";
  }
  std::cout << "\n";

  std::cout << "Buffers content:\n";

  for (auto &buffer : trainableBuffers_) {
    buffer->dump();
  }

  std::cout << "\n";
}
