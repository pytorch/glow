#ifndef GLOW_NETWORK_NETWORK_H
#define GLOW_NETWORK_NETWORK_H

#include "glow/Network/Nodes.h"
#include "glow/Network/Train.h"

#include <unordered_map>
#include <vector>

namespace glow {

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
  explicit Context(Context *primeCtx) : primeCtx_(primeCtx) {}

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
  std::vector<Context *> state_{};

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
    for (auto &c : state_) {
      N->init(c);
    }
    networkNodes_.push_back(N);
    return N;
  }

  /// Update the inputs for all variables \p vars with data from the inputs \p
  /// inputs at offset \p sampleIdx. Then perform a forward and backwards scan.
  void updateForwardBackward(Context *ctx, NodeBase *root, size_t sampleIdx,
                             ArrayRef<Variable *> vars,
                             ArrayRef<Tensor *> inputs);

  void learnGradient(Context *ctx, size_t batchSize);

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

  ConcatNode *createConcatNode(ArrayRef<NodeBase *> inputs, unsigned dimension);

  MaxPoolNode *createMaxPoolNode(NodeBase *input, MaxPoolNode::OpKind kind,
                                 size_t filterSize, size_t stride, size_t pad);

  FullyConnectedNode *createFullyConnectedNode(NodeBase *input,
                                               size_t outDepth);

  RELUNode *createRELUNode(NodeBase *input);

  SigmoidNode *createSigmoidNode(NodeBase *input);

  SoftMaxNode *createSoftMaxNode(NodeBase *input, NodeBase *selected);

  RegressionNode *createRegressionNode(NodeBase *input, NodeBase *expected);

  MaxNode *createMaxNode(NodeBase *input);

  Variable *createVariable(ArrayRef<size_t> dims, ElemKind elemTy);

  ReshapeNode *createReshapeNode(NodeBase *input, ArrayRef<size_t> shape);

  BatchNormalizationNode *createBatchNormalizationNode(NodeBase *input,
                                                       size_t channelIdx = 0,
                                                       FloatTy epsilon = 1e-5,
                                                       FloatTy momentum = 0.9);

  ArithmeticNode *createArithmeticNode(NodeBase *LHS, NodeBase *RHS,
                                       ArithmeticNode::OpKind op);
  ///@}

  /// Provides access to the training configuration.
  TrainingConfig &getConfig() { return trainer_.config; }

  /// Train the network starting with the node \p root. Perform \p iterations
  /// of batch size in the training loop. Update the nodes in \p nodes with the
  /// values \p inputs.
  void train(NodeBase *root, size_t numBatches, ArrayRef<Variable *> vars,
             ArrayRef<Tensor *> inputs);

  /// Perform a single training iteration for one input. Update the nodes in \p
  /// nodes with the values \p inputs.
  void train(NodeBase *root, ArrayRef<Variable *> vars,
             ArrayRef<Tensor *> inputs);

  /// Infer data for a single input. Update the nodes in \p nodes with the
  /// values \p inputs.
  Tensor *infer(NodeBase *root, ArrayRef<Variable *> vars,
                ArrayRef<Tensor *> inputs);

  /// Dump the textual representation of the network.
  void dump();

  /// Dump the graph representation of the network.
  void dumpGraph();

  /// Update the content of the internally stored tensor \p tok with \p t.
  void updateTensor(const TensorToken *tok, Tensor *t);

  /// \returns a pointer to the main context.
  Context *getMainContext() { return state_[0]; }
};
} // namespace glow

#endif // GLOW_NETWORK_NETWORK_H
