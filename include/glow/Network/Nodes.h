#ifndef GLOW_NODES_H
#define GLOW_NODES_H

#include "glow/Network/Node.h"
#include "glow/Network/Tensor.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace glow {

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

  void forward(Context *ctx, PassKind kind) const override;

  void backward(Context *ctx) const override;

  virtual std::string getName() const override { return "ConvNode"; }

  void visit(NodeVisitor *visitor) override;

  void updateWeights(Network *N_, Tensor *filter, Tensor *bias);
};

class MaxPoolNode final : public NodeBase {
public:
  /// Specifies the kind of pooling done by the operator.
  enum class OpKind {
    kMax,
    kAvg,
  };

private:
  /// The kind of pooling (max, avg, etc).
  OpKind kind_;

  /// The input node.
  NodeBase *input_;

  /// The source coordinate for each element in the max pool results buffer.
  /// These tensors are used to accelerate the gradient backward pass (in max
  /// pool mode).
  TensorToken srcX_, srcY_;

  size_t filterSize_;
  size_t stride_;
  size_t pad_;

  MaxPoolNode(Network *N, NodeBase *input, OpKind kind, size_t filterSize,
              size_t stride, size_t pad);

  friend Network;

public:
  void init(Context *ctx) const override;

  void forward(Context *ctx, PassKind kind) const override;

  void backward(Context *ctx) const override;

  void forwardMax(Context *ctx) const;

  void backwardMax(Context *ctx) const;

  void forwardAvg(Context *ctx) const;

  void backwardAvg(Context *ctx) const;

  virtual std::string getName() const override { return "MaxPoolNode"; }

  void visit(NodeVisitor *visitor) override;
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

  void forward(Context *ctx, PassKind kind) const override;

  void backward(Context *ctx) const override;

  virtual std::string getName() const override { return "FullyConnectedNode"; }

  void visit(NodeVisitor *visitor) override;
};

class RELUNode final : public NodeBase {
  /// A reference to the layer input.
  NodeBase *input_;

  RELUNode(Network *N, NodeBase *input);

  friend Network;

public:
  void init(Context *ctx) const override;

  void forward(Context *ctx, PassKind kind) const override;

  void backward(Context *ctx) const override;

  virtual std::string getName() const override { return "RELUNode"; }

  void visit(NodeVisitor *visitor) override;
};

class SigmoidNode final : public NodeBase {
  /// A reference to the layer input.
  NodeBase *input_;

  SigmoidNode(Network *N, NodeBase *input);

  friend Network;

public:
  void init(Context *ctx) const override;

  void forward(Context *ctx, PassKind kind) const override;

  void backward(Context *ctx) const override;

  virtual std::string getName() const override { return "SigmoidNode"; }

  void visit(NodeVisitor *visitor) override;
};

class SoftMaxNode final : public NodeBase {
  /// A reference to the node input.
  NodeBase *input_;
  /// The selected one-hot value from the softmax function.
  NodeBase *selected_;

  /// A temporary array for storing the subexpression (e ^ (a[i] - max)).
  TensorToken e_{};

  /// Ctor - \p is the input layer that must be of shape (1 x 1 x N).
  /// And \p selected that's the selected one-hot representation of the
  /// softmax function.
  SoftMaxNode(Network *N, NodeBase *input, NodeBase *selected);

  friend Network;

public:
  void init(Context *ctx) const override;

  void forward(Context *ctx, PassKind kind) const override;

  void backward(Context *ctx) const override;

  virtual std::string getName() const override { return "SoftMaxNode"; }

  void visit(NodeVisitor *visitor) override;
};

class RegressionNode final : public NodeBase {
  /// A reference to the node input.
  NodeBase *input_;
  /// The expected input (also known as Y).
  NodeBase *expected_{};

  /// Ctor - \p is the input layer that must be a simple vector.
  /// And \p expected (aka Y) is the expected input for the layer, that must
  /// be of the same shape as \p input.
  RegressionNode(Network *N, NodeBase *input, NodeBase *expected);

  friend Network;

public:
  void init(Context *ctx) const override;

  void forward(Context *ctx, PassKind kind) const override;

  void backward(Context *ctx) const override;

  virtual std::string getName() const override { return "RegressionNode"; }

  void visit(NodeVisitor *visitor) override;
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

  void forward(Context *ctx, PassKind kind) const override;

  void backward(Context *ctx) const override;

  virtual std::string getName() const override { return "MaxNode"; }

  void visit(NodeVisitor *visitor) override;
};

/// This is an abstraction over raw variable inputs.
class Variable final : public NodeBase {

  Variable(Network *N, ArrayRef<size_t> dims, ElemKind elemTy);

  std::vector<size_t> dims_;

  ElemKind elemTy_;

  friend Network;

public:
  virtual std::string getName() const override { return "ArrayNode"; }

  void init(Context *ctx) const override;

  void forward(Context *ctx, PassKind kind) const override {}

  void backward(Context *ctx) const override {}

  /// Update the input or expected output variables of the node with data from
  /// \p batch. Select inputs from the slice specified by \p payload.
  void updateInputs(Context *ctx, Tensor *batch, size_t sampleIdx);

  void visit(NodeVisitor *visitor) override;
};

class ReshapeNode final : public NodeBase {
  /// A reference to the node input.
  NodeBase *input_;

  /// Specifies the output shape.
  std::vector<size_t> shape_;

  /// Ctor - reshape the input into the new shape \p shape.
  ReshapeNode(Network *N, NodeBase *input, ArrayRef<size_t> shape);

  friend Network;

public:
  void init(Context *ctx) const override;

  void forward(Context *ctx, PassKind kind) const override;

  void backward(Context *ctx) const override;

  virtual std::string getName() const override { return "ReshapeNode"; }

  void visit(NodeVisitor *visitor) override;
};

/// Concats a number of input tensors into a single tensor.
class ConcatNode final : public NodeBase {
  /// Pointers to incoming inputs.
  std::vector<NodeBase *> inputs_;
  /// Concat on this dimension.
  unsigned dimension_;

  ConcatNode(Network *N, ArrayRef<NodeBase *> inputs, unsigned dimension);

  friend Network;

public:
  void init(Context *ctx) const override;

  void forward(Context *ctx, PassKind kind) const override;

  void backward(Context *ctx) const override;

  virtual std::string getName() const override { return "ConcatNode"; }

  void visit(NodeVisitor *visitor) override;
};

/// Performs batch normalization.
/// https://arxiv.org/abs/1502.03167
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
                         FloatTy epsilon, FloatTy momentum);

  friend Network;

public:
  void init(Context *ctx) const override;

  void forward(Context *ctx, PassKind kind) const override;

  void forwardTrain(Context *ctx) const;

  void forwardInfer(Context *ctx) const;

  void backward(Context *ctx) const override;

  virtual std::string getName() const override {
    return "BatchNormalizationNode";
  }

  void visit(NodeVisitor *visitor) override;
};

/// Performs per-element arithmetic operations.
class ArithmeticNode final : public NodeBase {
public:
  enum class OpKind {
    kAdd,
    kMul,
  };

private:
  /// A reference to the left hand side node input.
  NodeBase *LHS_;
  /// A reference to the right hand side node input.
  NodeBase *RHS_;
  /// Specifies the kind of the operation.
  const OpKind op_;

  /// Ctor - perform a per-element operation on the input tensors
  /// \p LHS, RHS. The operation is specified by \p kind.
  ArithmeticNode(Network *N, NodeBase *LHS, NodeBase *RHS, OpKind op);

  friend Network;

public:
  void init(Context *ctx) const override;

  void forward(Context *ctx, PassKind kind) const override;

  void backward(Context *ctx) const override;

  virtual std::string getName() const override { return "ArithmeticNode"; }

  void visit(NodeVisitor *visitor) override;
};

} // namespace glow

#endif // GLOW_NODES_H
