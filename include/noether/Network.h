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
  size_t trainCounter_{};

  /// The configuration used to train the network.
  TrainingConfig trainConf_{};

  /// A list of buffers to train as part of the backwards prop pass.
  std::vector<TrainableData *> trainableBuffers_;

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

  ArrayNode *createArrayNode(ArrayRef<size_t> dims);
  ///@}

  /// Provides access to the training configuration.
  TrainingConfig &getTrainingConfig() { return trainConf_; }

  /// Registers the derivable data \p weights (weights and gradient) as
  /// belonging to the node \p node.
  void registerDerivTensor(NodeBase *node, TrainableData *weights);

  /// Train the network starting with the node \p root. Perform \p iterations
  /// iterations in the training loop. Update the nodes in \p nodes with the
  /// values \p inputs.
  void train(NodeBase *root, size_t iterations, ArrayRef<NodeBase *> nodes,
             ArrayRef<Tensor *> inputs);

  /// Perform a single training iteration for one input. Update the nodes in \p
  /// nodes with the values \p inputs.
  void train(NodeBase *root, ArrayRef<NodeBase *> nodes,
             ArrayRef<Tensor *> inputs);

  /// Infer data for a single input. Update the nodes in \p nodes with the
  /// values \p inputs.
  void infer(NodeBase *root, ArrayRef<NodeBase *> nodes,
             ArrayRef<Tensor *> inputs);

  /// Dump the textual representation of the network.
  void dump(NodeBase *root);
};
}

#endif // NOETHER_NETWORK_H
