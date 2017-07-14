#ifndef NOETHER_NETWORK_H
#define NOETHER_NETWORK_H

#include "noether/Train.h"

#include <map>
#include <vector>

namespace noether {

class NodeBase;

class TrainableData;

class Network {
  /// The configuration used to train the network.
  TrainingConfig trainConf_;

  /// A list of dependencies.
  std::map<NodeBase *, std::vector<NodeBase *>> deps_;

  /// A list of buffers to train as part of the backwards prop pass.
  std::vector<TrainableData *> trainableBuffers_;

  /// Generate a topological order of the nodes in the network.
  void sortNetwork(std::vector<NodeBase *> &order);

public:
  Network();

  /// Provides access to the training configuration.
  TrainingConfig &getTrainingConfig() { return trainConf_; }

  /// Add \p dep as a dependency (prerequisite) for \p node.
  void addNodeDependency(NodeBase *node, NodeBase *dep);

  /// Registers the derivable data \p weights (weights and gradient) as
  /// belonging to the node \p node.
  void registerDerivTensor(NodeBase *node, TrainableData *weights);

  /// Train the network on a single input.
  void train();

  /// Infer data for a single input.
  void infer();

  /// Dump the textual representation of the network.
  void dump();
};
}

#endif // NOETHER_NETWORK_H
