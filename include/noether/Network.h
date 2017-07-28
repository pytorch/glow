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
class Context {
public:
  using TrainableMap = std::unordered_map<const TensorToken*, TrainableData*>;

  /// Represents the cell number, when performing concurrent training.
  unsigned cellId_;

  /// Maps weight tensors into the corresponding weights and gradient tensors.
  TrainableMap trainables_;

  /// Maps weight tensors into the corresponding weights and gradient tensors.
  std::unordered_map<const TensorToken*, Tensor*> tensors_;

  Context(unsigned cellId) : cellId_(cellId) {}

  ~Context();

  TrainableMap::iterator begin() { return trainables_.begin(); }
  TrainableMap::iterator end() { return trainables_.end(); }

  Handle<FloatTy> getWeightHandle(const TensorToken *tok);

  Handle<FloatTy> getGradHandle(const TensorToken *tok);

  /// Allocates a new tensor pair that's addressed by the token \p tok.
  void allocateTrainable(const TensorToken *tok, bool trainable,
                         ArrayRef<size_t> dims);

  /// \returns the allocated gradient and weight Tensor pair.
  TrainableData *getTrainable(const TensorToken *tok);

  /// Allocates a new tensor that's addressed by the token \p tok.
  void allocateTensor(const TensorToken *tok, ElemKind kind,
                      ArrayRef<size_t> dims);

  /// \returns the allocated Tensor.
  Tensor *getTensor(const TensorToken *tok);
};

class Network {
  /// This vector holds the network state. One Context for each parallelism
  /// unit.
  std::vector<Context> state_{};

  /// This variable counts the number of iterations that train() was called.
  /// It is mainly used to detect batch size boundries.
  size_t trainCounter_{};

  /// The trainer performs the SGD and contains the caches that are needed for
  /// training.
  Trainer trainer_{};

  /// This is a list of nodes (operations) that make the network. The nodes are
  /// owned by the network.
  std::vector<NodeBase *> networkNodes_;

  /// Registers the newly create operation node into the network.
  /// \returns the newly created node.
  template <class NodeTy> NodeTy *addNode(NodeTy *N) {
    for (auto &c : state_)
    N->init(&c);

    networkNodes_.push_back(N);
    return N;
  }

  void updateForwardBackward(Context *ctx, NodeBase *root, size_t start, size_t len,
                             ArrayRef<NodeBase *> nodes,
                             ArrayRef<Tensor *> inputs, bool isBatch);

  void learnGradient(Context *ctx);

public:
  /// Ctor.
  Network();

  /// Dtor.
  ~Network();

  /// @name Node Builders
  /// These methods create new operation nodes that are owned by the network.
  /// The parameters are documented in the node constructors.
  ///@{
  ConvNode *createConvNode(NodeBase *input, size_t outDepth,
                           size_t filterSize, size_t stride, size_t pad);

  MaxPoolNode *createMaxPoolNode(NodeBase *input, size_t filterSize,
                                 size_t stride, size_t pad);

  FullyConnectedNode *createFullyConnectedNode(NodeBase *input,
                                               size_t outDepth);

  RELUNode *createRELUNode(NodeBase *input);

  SigmoidNode *createSigmoidNode(NodeBase *input);

  SoftMaxNode *createSoftMaxNode(NodeBase *input);

  RegressionNode *createRegressionNode(NodeBase *input);

  MaxNode *createMaxNode(NodeBase *input);

  ArrayNode *createArrayNode(ArrayRef<size_t> dims);
  ///@}

  /// Provides access to the training configuration.
  TrainingConfig &getConfig() { return trainer_.config; }

  /// Train the network starting with the node \p root. Perform \p iterations
  /// of batch size in the training loop. Update the nodes in \p nodes with the
  /// values \p inputs.
  void train(NodeBase *root, size_t batches, ArrayRef<NodeBase *> nodes,
             ArrayRef<Tensor *> inputs);

  /// Perform a single training iteration for one input. Update the nodes in \p
  /// nodes with the values \p inputs.
  void train(NodeBase *root, ArrayRef<NodeBase *> nodes,
             ArrayRef<Tensor *> inputs);

  /// Infer data for a single input. Update the nodes in \p nodes with the
  /// values \p inputs.
  Tensor *infer(NodeBase *root, ArrayRef<NodeBase *> nodes,
             ArrayRef<Tensor *> inputs);

  /// Dump the textual representation of the network.
  void dump(NodeBase *root);
};
}

#endif // NOETHER_NETWORK_H
