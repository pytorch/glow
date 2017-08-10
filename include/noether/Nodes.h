#ifndef NOETHER_NODES_H
#define NOETHER_NODES_H

#include "noether/Node.h"
#include "noether/Tensor.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace noether {

class ConvNode final : public NodeBase {
  NodeBase *input_;
  /// A list of convolution filter weights.
  TensorToken filtersW_;
  /// A list of convolution filters gradients.
  TensorToken filtersG_;

  /// The convolution bias weights.
  TensorToken biasW_;
  /// The convolution bias gradients.
  TensorToken biasG_;

  size_t filterSize_;
  size_t stride_;
  size_t pad_;
  size_t outDepth_;

  ConvNode(Network *N, NodeBase *input, size_t outDepth, size_t filterSize,
           size_t stride, size_t pad);

  friend Network;

public:
  void init(Context *ctx) const override;

  virtual void forward(Context *ctx, PassKind kind) const override;

  virtual void backward(Context *ctx) const override;

  virtual std::string getName() const override { return "ConvNode"; }

  virtual void visit(NodeVisitor *visitor) override;

  void updateWeights(Network *N_, Tensor *filter, Tensor *bias);
};

class MaxPoolNode final : public NodeBase {
  /// The input node.
  NodeBase *input_;
  /// The source coordinate for each element in the result pool. This is used
  /// to accelerate the gradient backward pass.
  TensorToken srcX_, srcY_;

  size_t filterSize_;
  size_t stride_;
  size_t pad_;

  MaxPoolNode(Network *N, NodeBase *input, size_t filterSize, size_t stride,
              size_t pad);

  friend Network;

public:
  void init(Context *ctx) const override;

  virtual void forward(Context *ctx, PassKind kind) const override;

  virtual void backward(Context *ctx) const override;

  virtual std::string getName() const override { return "MaxPoolNode"; }

  virtual void visit(NodeVisitor *visitor) override;
};

class FullyConnectedNode final : public NodeBase {
  /// A reference to the layer input.
  NodeBase *input_;
  /// A list of filter weights.
  TensorToken filtersW_;
  /// A list of filters gradients.
  TensorToken filtersG_;
  /// The bias weights.
  TensorToken biasW_;
  /// The bias gradients.
  TensorToken biasG_;


  size_t outDepth_;

  FullyConnectedNode(Network *N, NodeBase *input, size_t outDepth);

  friend Network;

public:
  void init(Context *ctx) const override;

  virtual void forward(Context *ctx, PassKind kind) const override;

  virtual void backward(Context *ctx) const override;

  virtual std::string getName() const override { return "FullyConnectedNode"; }

  virtual void visit(NodeVisitor *visitor) override;
};

class RELUNode final : public NodeBase {
  /// A reference to the layer input.
  NodeBase *input_;

  RELUNode(Network *N, NodeBase *input);

  friend Network;

public:
  void init(Context *ctx) const override;

  virtual void forward(Context *ctx, PassKind kind) const override;

  virtual void backward(Context *ctx) const override;

  virtual std::string getName() const override { return "RELUNode"; }

  virtual void visit(NodeVisitor *visitor) override;
};

class SigmoidNode final : public NodeBase {
  /// A reference to the layer input.
  NodeBase *input_;

  SigmoidNode(Network *N, NodeBase *input);

  friend Network;

public:
  void init(Context *ctx) const override;

  virtual void forward(Context *ctx, PassKind kind) const override;

  virtual void backward(Context *ctx) const override;

  virtual std::string getName() const override { return "SigmoidNode"; }

  virtual void visit(NodeVisitor *visitor) override;
};

class SoftMaxNode final : public NodeBase {
  /// A reference to the node input.
  NodeBase *input_;
  /// The selected one-hot value from the softmax function.
  TensorToken selected_;

  /// A temporary array for storing the subexpression (e ^ (a[i] - max)).
  TensorToken e_{};

  /// Ctor - \p is the input layer that must be of shape (1 x 1 x N).
  /// And \p selected that's the selected one-hot representation of the
  /// softmax function.
  SoftMaxNode(Network *N, NodeBase *input);

  friend Network;

public:
  virtual void updateInputs(Context *ctx, Tensor *batch,
                            size_t sampleIdx) override;

  virtual void updateInput(Context *ctx, Tensor *var) override;

  void init(Context *ctx) const override;

  virtual void forward(Context *ctx, PassKind kind) const override;

  virtual void backward(Context *ctx) const override;

  /// Marks the channel that the SoftMax needs to optimize for.
  void setSelected(Context *ctx, size_t selected);

  virtual std::string getName() const override { return "SoftMaxNode"; }

  virtual void visit(NodeVisitor *visitor) override;
};

class RegressionNode final : public NodeBase {
  /// A reference to the node input.
  NodeBase *input_;
  /// The expected input (also known as Y).
  TensorToken expected_{};

  /// Ctor - \p is the input layer that must be a simple vector.
  /// And \p expected (aka Y) is the expected input for the layer, that must
  /// be of the same shape as \p input.
  RegressionNode(Network *N, NodeBase *input);

  friend Network;

public:
  virtual void updateInputs(Context *ctx, Tensor *batch,
                            size_t sampleIdx) override;

  virtual void updateInput(Context *ctx, Tensor *var) override;

  void init(Context *ctx) const override;

  virtual void forward(Context *ctx, PassKind kind) const override;

  virtual void backward(Context *ctx) const override;

  virtual std::string getName() const override { return "RegressionNode"; }

  virtual void visit(NodeVisitor *visitor) override;
};

/// This node attempts to maximize the inputs by sending back a gradient signal
/// that encourages positive values. This is very useful for debugging.
class MaxNode final : public NodeBase {
  /// A reference to the node input.
  NodeBase *input_;

  MaxNode(Network *N, NodeBase *input);

  friend Network;

public:
  void init(Context *ctx) const override;

  virtual void forward(Context *ctx, PassKind kind) const override;

  virtual void backward(Context *ctx) const override;

  virtual std::string getName() const override { return "MaxNode"; }

  virtual void visit(NodeVisitor *visitor) override;
};

/// This is an abstraction over raw variable inputs.
class ArrayNode final : public NodeBase {
  ArrayNode(Network *N, ArrayRef<size_t> dims);

  friend Network;

  std::vector<size_t> dims_;

public:
  virtual std::string getName() const override { return "ArrayNode"; }

  void init(Context *ctx) const override;

  void forward(Context *ctx, PassKind kind) const override {}

  void backward(Context *ctx) const override {}

  virtual void updateInputs(Context *ctx, Tensor *batch,
                            size_t sampleIdx) override;

  virtual void updateInput(Context *ctx, Tensor *var) override;

  virtual void visit(NodeVisitor *visitor) override;
};
  
/// Concats a number of input tensors into a single tensor.
class ConcatNode final : public NodeBase {
  /// Pointers to incoming inputs.
  std::vector<NodeBase *>inputs_;
  /// Concat on this dimension.
  unsigned dimension_;

  ConcatNode(Network *N, ArrayRef<NodeBase *>inputs, unsigned dimension);

  friend Network;

public:
  void init(Context *ctx) const override;

  virtual void forward(Context *ctx, PassKind kind) const override;

  virtual void backward(Context *ctx) const override;

  virtual std::string getName() const override { return "ConcatNode"; }

  virtual void visit(NodeVisitor *visitor) override;
};

class BatchNormalizationNode final : public NodeBase {
  /// A reference to the node input.
  NodeBase *input_;

  /// Specifies which dimension splits the data into independent batches.
  const size_t channelIdx_;
  const FloatTy epsilon_;
  const FloatTy momentum_;

  TensorToken betaW_;
  TensorToken betaG_;
  TensorToken gammaW_;
  TensorToken gammaG_;

  TensorToken mean_;
  TensorToken variance_;

  /// Ctor - \p is the input layer that must be a simple vector.
  /// \p epsilon and \p momentum are the batch normalization parameters.
  BatchNormalizationNode(Network *N, NodeBase *input, size_t channelIdx,
                         FloatTy epsilon,
                         FloatTy momentum);

  friend Network;

public:

  void init(Context *ctx) const override;

  virtual void forward(Context *ctx, PassKind kind) const override;
  
  void forwardTrain(Context *ctx) const;

  void forwardInfer(Context *ctx) const;

  virtual void backward(Context *ctx) const override;

  virtual std::string getName() const override {
    return "BatchNormalizationNode";
  }

  virtual void visit(NodeVisitor *visitor) override;
};


} // namespace noether

#endif // NOETHER_NODES_H
