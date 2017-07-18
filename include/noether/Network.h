#ifndef NOETHER_NETWORK_H
#define NOETHER_NETWORK_H

#include "noether/Nodes.h"
#include "noether/Train.h"

#include <map>
#include <vector>

namespace noether {

class NodeBase;

class TrainableData;

class Network {
  /// This variable counts the number of iterations that train() was called.
  /// It is mainly used to detect batch size boundries.
  size_t trainCounter_;

  /// The configuration used to train the network.
  TrainingConfig trainConf_;

  /// A list of dependencies.
  std::map<NodeBase *, std::vector<NodeBase *>> deps_;

  /// A list of buffers to train as part of the backwards prop pass.
  std::vector<TrainableData *> trainableBuffers_;

  /// Generate a topological order of the nodes in the network.
  void sortNetwork(std::vector<NodeBase *> &order);

  /// This is a list of nodes (operations) that make the network. The nodes are
  /// owned by the network.
  std::vector<TrainableNode *> networkNodes_;

  /// Registers the newly create operation node into the network.
  /// \returns the newly created node.
  template <class NodeTy> NodeTy *addNode(NodeTy *N) {
    networkNodes_.push_back(N);
    return N;
  }

public:
  /// Ctor.
  Network();

  /// Dtor.
  ~Network();

  /// @name Node Builders
  /// These methods create new operation nodes that are owned by the network.
  /// The parameters are documented in the node constructors.
  ///@{
  ConvNode *createConvNode(TrainableNode *input, size_t outDepth,
                           size_t filterSize, size_t stride, size_t pad);

  MaxPoolNode *createMaxPoolNode(TrainableNode *input, size_t filterSize,
                                 size_t stride, size_t pad);

  FullyConnectedNode *createFullyConnectedNode(TrainableNode *input,
                                               size_t outDepth);

  RELUNode *createRELUNode(TrainableNode *input);

  SigmoidNode *createSigmoidNode(TrainableNode *input);

  SoftMaxNode *createSoftMaxNode(TrainableNode *input);

  RegressionNode *createRegressionNode(TrainableNode *input);

  MaxNode *createMaxNode(TrainableNode *input);

  ArrayNode *createArrayNode(size_t x, size_t y, size_t z);
  ///@}

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
