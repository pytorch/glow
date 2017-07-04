#include "noether/Image.h"
#include "noether/Layers.h"
#include "noether/Network.h"
#include "noether/Tensor.h"

#include <iostream>
#include <unordered_set>

using namespace noether;

Network::Network() {}

void Network::sortNetwork(std::vector<LayerBase*> &order) {
  // TODO: add a cycle detector.
  // A list of nodes that were processed.
  std::unordered_set<LayerBase*> visited;
  // Our DFS stack.
  std::vector<LayerBase*> stack;

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

void Network::addLayerDependency(LayerBase *node, LayerBase *dep) {
  deps_[node].push_back(dep);
}

void Network::registerDerivTensor(LayerBase *node, TrainableData *weights) {
  trainableBuffers_.push_back(weights);
}

void Network::train() {
  std::vector<LayerBase*> order;
  sortNetwork(order);

  std::cout<<"Network structure:";
  for (auto &node : order) {
    std::cout<<node->getName()<<" ";
  }
  std::cout<<"\n";

  // Forward scan.
  for (unsigned i = 0, e = order.size(); i < e; i++) {
    order[i]->forward();
  }

  // Backward scan in reverse order.
  for (unsigned i = 0, e = order.size(); i < e; i++) {
    order[e - i - 1]->backward();
  }

  // Update the gradients.
  for (auto &buffer: trainableBuffers_) {
      buffer->train();
  }

}

void Network::infer() {
  std::vector<LayerBase*> order;
  sortNetwork(order);

  std::cout<<"Network structure:";
  for (auto &node : order) {
    std::cout<<node->getName()<<" ";
  }
  std::cout<<"\n";

  // Forward scan.
  for (unsigned i = 0, e = order.size(); i < e; i++) {
    order[i]->forward();
  }
}

