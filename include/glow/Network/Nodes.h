#ifndef GLOW_NETWORK_NODES_H
#define GLOW_NETWORK_NODES_H

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

  std::string getName() const override { return "ConvNode"; }

  std::string getDebugRepr(Context *ctx) const override;

  void visit(NodeBase *parent, NodeVisitor *visitor) override;

  void loadWeights(Network *N_, Tensor *filter, Tensor *bias);

  /// Calculate the size of the output tensor based on the convolution
  /// parameters.
  static std::pair<size_t, size_t> calculateOutputDims(size_t sx, size_t sy,
                                                       size_t pad,
                                                       size_t filterSize,
                                                       size_t stride) {
    size_t outsx = ((sx + pad * 2 - filterSize) / stride + 1);
    size_t outsy = ((sy + pad * 2 - filterSize) / stride + 1);
    return {outsx, outsy};
  }
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

  std::string getName() const override { return "MaxPoolNode"; }

  std::string getDebugRepr(Context *ctx) const override;

  void visit(NodeBase *parent, NodeVisitor *visitor) override;
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

  /// Updates the weights \p b, and bias \p b of the node.
  void loadWeights(Network *N_, Tensor *w, Tensor *bias);

  void forward(Context *ctx, PassKind kind) const override;

  void backward(Context *ctx) const override;

  std::string getName() const override { return "FullyConnectedNode"; }

  std::string getDebugRepr(Context *ctx) const override;

  void visit(NodeBase *parent, NodeVisitor *visitor) override;
};

class LRNNode final : public NodeBase {
  /// A reference to the layer input.
  NodeBase *input_;

  /// The number of neighbouring channels on each side to sum over
  size_t halfWindowSize_;

  /// The scaling parameter
  float alpha_;

  /// The exponent parameter
  float beta_;

  /// The offset parameter
  float k_;

  /// This tensor is used to accelerate the gradient backward pass
  TensorToken scale_;

  LRNNode(Network *N, NodeBase *input, size_t windowSize, float alpha,
          float beta, float k);

  friend Network;

public:
  void init(Context *ctx) const override;

  void forward(Context *ctx, PassKind kind) const override;

  void backward(Context *ctx) const override;

  std::string getName() const override { return "LRNNode"; }

  std::string getDebugRepr(Context *ctx) const override;

  void visit(NodeBase *parent, NodeVisitor *visitor) override;
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

  std::string getName() const override { return "RELUNode"; }

  std::string getDebugRepr(Context *ctx) const override;

  void visit(NodeBase *parent, NodeVisitor *visitor) override;
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

  std::string getName() const override { return "SigmoidNode"; }

  std::string getDebugRepr(Context *ctx) const override;

  void visit(NodeBase *parent, NodeVisitor *visitor) override;
};

class TanhNode final : public NodeBase {
  /// A reference to the layer input.
  NodeBase *input_;

  TanhNode(Network *N, NodeBase *input);

  friend Network;

public:
  void init(Context *ctx) const override;

  void forward(Context *ctx, PassKind kind) const override;

  void backward(Context *ctx) const override;

  std::string getName() const override { return "TanhNode"; }

  std::string getDebugRepr(Context *ctx) const override;

  void visit(NodeBase *parent, NodeVisitor *visitor) override;
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

  std::string getName() const override { return "SoftMaxNode"; }

  std::string getDebugRepr(Context *ctx) const override;

  void visit(NodeBase *parent, NodeVisitor *visitor) override;
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

  std::string getName() const override { return "RegressionNode"; }

  std::string getDebugRepr(Context *ctx) const override;

  void visit(NodeBase *parent, NodeVisitor *visitor) override;
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

  std::string getName() const override { return "MaxNode"; }

  std::string getDebugRepr(Context *ctx) const override;

  void visit(NodeBase *parent, NodeVisitor *visitor) override;
};

/// This is an abstraction over raw variable inputs.
class Variable final : public NodeBase {

  Variable(Network *N, ArrayRef<size_t> dims, ElemKind elemTy);

  std::vector<size_t> dims_;

  ElemKind elemTy_;

  friend Network;

public:
  std::string getName() const override { return "Variable"; }

  std::string getDebugRepr(Context *ctx) const override;

  void init(Context *ctx) const override;

  void forward(Context *ctx, PassKind kind) const override {}

  void backward(Context *ctx) const override {}

  /// Update the input or expected output variables of the node with data from
  /// \p batch. Select inputs from the slice specified by \p payload.
  void updateInputs(Context *ctx, Tensor *batch, size_t sampleIdx);

  void visit(NodeBase *parent, NodeVisitor *visitor) override;
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

  std::string getName() const override { return "ReshapeNode"; }

  std::string getDebugRepr(Context *ctx) const override;

  void visit(NodeBase *parent, NodeVisitor *visitor) override;
};

/// Transposes a tensor by shuffling the dimensions.
class TransposeNode final : public NodeBase {
  NodeBase *input_;

  std::vector<unsigned> shuffle_;
  std::vector<unsigned> reverseShuffle_;

  /// Ctor - change the order of the dimensions in the tensor.
  /// \p shuffle represents a list of indices that point to the index of the
  /// dimension in the original tensor.
  TransposeNode(Network *N, NodeBase *input, ArrayRef<unsigned> shuffle);

  friend Network;

public:
  void init(Context *ctx) const override;

  void forward(Context *ctx, PassKind kind) const override;

  void backward(Context *ctx) const override;

  std::string getName() const override { return "TransposeNode"; }

  std::string getDebugRepr(Context *ctx) const override;

  void visit(NodeBase *parent, NodeVisitor *visitor) override;
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

  std::string getName() const override { return "ConcatNode"; }

  std::string getDebugRepr(Context *ctx) const override;

  void visit(NodeBase *parent, NodeVisitor *visitor) override;
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

  // Beta is the bias:
  TensorToken betaW_;
  TensorToken betaG_;
  // Gamma is the scale:
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

  std::string getName() const override { return "BatchNormalizationNode"; }

  std::string getDebugRepr(Context *ctx) const override;

  void visit(NodeBase *parent, NodeVisitor *visitor) override;

  void loadWeights(Network *N_, Tensor *scale, Tensor *bias, Tensor *mean,
                   Tensor *var);
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

  std::string getName() const override { return "ArithmeticNode"; }

  std::string getDebugRepr(Context *ctx) const override;

  void visit(NodeBase *parent, NodeVisitor *visitor) override;
};

} // namespace glow

#endif // GLOW_NETWORK_NODES_H
