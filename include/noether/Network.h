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
  using TensorMap = std::unordered_map<const TensorToken *, Tensor *>;
  using TensorPairTy = std::pair<const TensorToken *, const TensorToken *>;
  using TensorPairListTy = std::vector<TensorPairTy>;

  enum ShareKind {
    /// Marks tensors that are shared between different context values.
    kSharedTensor,
        /// Marks tensors that unique to the context and are not shared.
    kPrivateTensor,
  };

private:
  /// A pointer to the prime context, or nullptr, if this is the prime ctx.
  Context *primeCtx_;

  /// Maps tensor descriptors into the corresponding tensors.
  TensorMap tensors_;

  /// A list of pairs of weight/gradient tensors that are attached.
  TensorPairListTy pairs_;

public:

  Context(Context *primeCtx) : primeCtx_(primeCtx) {}

  ~Context();

  TensorMap::iterator begin() { return tensors_.begin(); }
  TensorMap::iterator end() { return tensors_.end(); }

  /// Returns the list of paired tensors.
  TensorPairListTy &getTensorPairs() { return pairs_; }

  /// Mark two tensors as being paired together.
  void addTensorPair(TensorPairTy p) { pairs_.push_back(p); }

  /// Allocates a new tensor that's addressed by the token \p tok.
  /// \returns the address of the allocated tensor, or nullptr, if this is a
  /// shared tensor that's owned by the prime context.
  Tensor *allocateTensor(const TensorToken *tok, ElemKind kind,
                      ArrayRef<size_t> dims,
                      ShareKind shared = ShareKind::kPrivateTensor);

  /// \returns the allocated Tensor.
  Tensor *getTensor(const TensorToken *tok);

  /// \returns True if the tensor is managed by the context.
  bool hasTensor(const TensorToken *tok);

  Handle<FloatTy> getHandle(const TensorToken *tok);
};

class Network {
  /// This vector holds the network state. One Context for each parallelism
  /// unit.
  std::vector<Context*> state_{};

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
      N->init(c);

    networkNodes_.push_back(N);
    return N;
  }

  void updateForwardBackward(Context *ctx, NodeBase *root, size_t start,
                             size_t len, ArrayRef<NodeBase *> nodes,
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
  ConvNode *createConvNode(NodeBase *input, size_t outDepth, size_t filterSize,
                           size_t stride, size_t pad);

  ConcatNode *createConcatNode(ArrayRef<NodeBase *>inputs,
                               unsigned dimension);

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

  /// Update the content of the internally stored tensor \p tok with \p t.
  void updateTensor(const TensorToken *tok, Tensor *t);
};
}

#endif // NOETHER_NETWORK_H
