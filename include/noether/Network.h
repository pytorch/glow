#ifndef NOETHER_NETWORK_H
#define NOETHER_NETWORK_H

#include "noether/Nodes.h"
#include "noether/Train.h"

#include <unordered_map>
#include <vector>

namespace noether {

class NodeBase;

class TrainableData;

/// This represents the execution context of the graph.
class Context {};

class Network {
  /// This variable counts the number of iterations that train() was called.
  /// It is mainly used to detect batch size boundries.
  size_t trainCounter_{};

  /// The trainer performs the SGD and contains the caches that are needed for
  /// training.
  Trainer trainer_{};

  /// This is a list of nodes (operations) that make the network. The nodes are
  /// owned by the network.
  std::vector<TrainableNode *> networkNodes_;

  /// Maps weight tensors into the corresponding gradient tensors.
  std::unordered_map<Tensor*, Tensor*> gradientTensors_;

  /// Registers the newly create operation node into the network.
  /// \returns the newly created node.
  template <class NodeTy> NodeTy *addNode(NodeTy *N) {
    networkNodes_.push_back(N);
    allocateGradientTensor(&N->output_);
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
  TrainingConfig &getConfig() { return trainer_.config; }

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

  /// Allocates a gradient Tensor that corresponds to the tensor \p weights.
  void allocateGradientTensor(Tensor *weights);

  /// \returns the allocated gradient Tensor that was allocated for \p weights
  /// in the training context \p ctx.
  Tensor *getGradientTensor(Context *ctx, Tensor *weights);
};
}

#endif // NOETHER_NETWORK_H
