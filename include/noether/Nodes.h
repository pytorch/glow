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

class ConvNode final : public TrainableNode {
  TrainableNode *input_;
  /// A list of convolution filters.
  std::vector<TrainableData> filters_;
  /// The convolution bias.
  TrainableData bias_;

  size_t filterSize_;
  size_t stride_;
  size_t pad_;

  ConvNode(Network *N, TrainableNode *input, size_t outDepth, size_t filterSize,
           size_t stride, size_t pad);

  friend Network;

public:
  virtual void forward() override;

  virtual void backward() override;

  virtual std::string getName() const override { return "ConvNode"; }

  virtual void visit(NodeVisitor *visitor) override;
};

class MaxPoolNode final : public TrainableNode {
  /// The input node.
  TrainableNode *input_;
  /// The source coordinate for each element in the result pool. This is used
  /// to accelerate the gradient backward pass.
  Tensor<size_t> srcX_, srcY_;

  size_t filterSize_;
  size_t stride_;
  size_t pad_;

  MaxPoolNode(Network *N, TrainableNode *input, size_t filterSize,
              size_t stride, size_t pad);

  friend Network;

public:
  virtual void forward() override;

  virtual void backward() override;

  virtual std::string getName() const override { return "MaxPoolNode"; }

    virtual void visit(NodeVisitor *visitor) override;
};

class FullyConnectedNode final : public TrainableNode {
  /// A reference to the layer input.
  TrainableNode *input_;
  /// A list of filters.
  std::vector<TrainableData> filters_;
  /// The biases.
  TrainableData bias_;

  FullyConnectedNode(Network *N, TrainableNode *input, size_t outDepth);

  friend Network;

public:
  virtual void forward() override;

  virtual void backward() override;

  virtual std::string getName() const override { return "FullyConnectedNode"; }

  virtual void visit(NodeVisitor *visitor) override;
};

class RELUNode final : public TrainableNode {
  /// A reference to the layer input.
  TrainableNode *input_;

  RELUNode(Network *N, TrainableNode *input);

  friend Network;

public:
  virtual void forward() override;

  virtual void backward() override;

  virtual std::string getName() const override { return "RELUNode"; }

  virtual void visit(NodeVisitor *visitor) override;
};

class SigmoidNode final : public TrainableNode {
  /// A reference to the layer input.
  TrainableNode *input_;

  SigmoidNode(Network *N, TrainableNode *input);

  friend Network;

public:
  virtual void forward() override;

  virtual void backward() override;

  virtual std::string getName() const override { return "SigmoidNode"; }

  virtual void visit(NodeVisitor *visitor) override;
};

class SoftMaxNode final : public TrainableNode {
  /// A reference to the node input.
  TrainableNode *input_;
  /// The selected one-hot value from the softmax function.
  size_t selected_;

  /// A temporary array for storing the subexpression (e ^ (a[i] - max)).
  Tensor<FloatTy> e_;

  /// Ctor - \p is the input layer that must be of shape (1 x 1 x N).
  /// And \p selected that's the selected one-hot representation of the
  /// softmax function.
  SoftMaxNode(Network *N, TrainableNode *input);

  /// If set, the training procedure will update the content of the array node
  /// from this input source.
  Tensor<size_t> *boundInputSource_{nullptr};

  friend Network;

public:
  void bind(Tensor<size_t> *input) { boundInputSource_ = input; }

  virtual void updateBoundInputs(size_t sampleIdx) override {
    if (!boundInputSource_)
      return;
    selected_ = boundInputSource_->getHandle().at({sampleIdx});

    assert(selected_ < dims()[0] && "Invalid selected value");
  }

  virtual void forward() override;

  virtual void backward() override;

  /// \returns the index of the highest value.
  size_t maxArg() const;

  /// Marks the channel that the SoftMax needs to optimize for.
  void setSelected(size_t selected);

  virtual std::string getName() const override { return "SoftMaxNode"; }

  virtual void visit(NodeVisitor *visitor) override;
};

class RegressionNode final : public TrainableNode {
  /// A reference to the node input.
  TrainableNode *input_;
  /// The expected input (also known as Y).
  Tensor<FloatTy> expected_;

  /// If set, the training procedure will update the content of the array node
  /// from this input source.
  Tensor<FloatTy> *boundInputSource_{nullptr};

  /// Ctor - \p is the input layer that must be a simple vector.
  /// And \p expected (aka Y) is the expected input for the layer, that must
  /// be of the same shape as \p input.
  RegressionNode(Network *N, TrainableNode *input);

  friend Network;

public:
  void bind(Tensor<FloatTy> *input) {
    auto idim = input->dims();
    auto dim = dims();
    (void)dim;
    (void)idim;
    assert(idim == dim && "Invalid input size");
    boundInputSource_ = input;
  }

  virtual void updateBoundInputs(size_t sampleIdx) override {
    if (!boundInputSource_)
      return;

    assert(boundInputSource_->isInBounds({(unsigned)sampleIdx, 0, 0, 0}));
    expected_ = boundInputSource_->getHandle().extractSlice(sampleIdx);
  }

  /// \returns a reference to the expected result vector.
  Tensor<FloatTy> &getExpected() { return expected_; }

  virtual void forward() override;

  virtual void backward() override;

  virtual std::string getName() const override { return "RegressionNode"; }

  virtual void visit(NodeVisitor *visitor) override;
};

/// This node attempts to maximize the inputs by sending back a gradient signal
/// that encourages positive values. This is very useful for debugging.
class MaxNode final : public TrainableNode {
  /// A reference to the node input.
  TrainableNode *input_;

  MaxNode(Network *N, TrainableNode *input);

  friend Network;

public:
  virtual void forward() override;

  virtual void backward() override;

  virtual std::string getName() const override { return "MaxNode"; }

  virtual void visit(NodeVisitor *visitor) override;
};

/// This is an abstraction over raw variable inputs.
class ArrayNode final : public TrainableNode {
  ArrayNode(Network *N, ArrayRef<size_t> dims) : TrainableNode(N) {
    this->getOutput().reset(dims);
    // Do not change the output of this layer when training the network.
    this->getOutput().isTrainable_ = false;
  }

  /// If set, the training procedure will update the content of the array node
  /// from this input source.
  Tensor<FloatTy> *boundInputSource_{nullptr};

  friend Network;

public:
  void bind(Tensor<FloatTy> *input) {
    auto inDim = input->dims();
    auto dim = dims();
    (void)inDim;
    (void)dim;
    assert(dim.size() + 1 == inDim.size() && "Invalid tensors");
    boundInputSource_ = input;
  }

  virtual std::string getName() const override { return "ArrayNode"; }

  void forward() override {}

  void backward() override {}

  virtual void updateBoundInputs(size_t sampleIdx) override {
    if (!boundInputSource_)
      return;

    this->getOutput().weight_ = boundInputSource_->getHandle().extractSlice(sampleIdx);
  }

  virtual void visit(NodeVisitor *visitor) override;
};

} // namespace noether

#endif // NOETHER_NODES_H
