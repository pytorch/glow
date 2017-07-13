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
  std::vector<DerivData> filters_;
  /// The convolution bias.
  DerivData bias_;

  size_t filterSize_;
  size_t stride_;
  size_t pad_;

public:
  ConvNode(Network *N, TrainableNode *input, size_t outDepth, size_t filterSize,
           size_t stride, size_t pad);

  virtual void forward() override;

  virtual void backward() override;

  virtual std::string getName() const override { return "ConvNode"; }
};

class MaxPoolNode final : public TrainableNode {
  /// The input node.
  TrainableNode *input_;
  /// The source coordinate for each element in the result pool. This is used
  /// to accelerate the gradient backward pass.
  Array3D<size_t> srcX_, srcY_;

  size_t filterSize_;
  size_t stride_;
  size_t pad_;

public:
  MaxPoolNode(Network *N, TrainableNode *input, size_t filterSize,
              size_t stride, size_t pad);

  virtual void forward() override;

  virtual void backward() override;

  virtual std::string getName() const override { return "MaxPoolNode"; }
};

class FullyConnectedNode final : public TrainableNode {
  /// A reference to the layer input.
  TrainableNode *input_;
  /// A list of filters.
  std::vector<DerivData> filters_;
  /// The biases.
  DerivData bias_;

public:
  FullyConnectedNode(Network *N, TrainableNode *input, size_t outDepth);

  virtual void forward() override;

  virtual void backward() override;

  virtual std::string getName() const override { return "FullyConnectedNode"; }
};

class RELUNode final : public TrainableNode {
  /// A reference to the layer input.
  TrainableNode *input_;

public:
  RELUNode(Network *N, TrainableNode *input);

  virtual void forward() override;

  virtual void backward() override;

  virtual std::string getName() const override { return "RELUNode"; }
};

class SigmoidNode final : public TrainableNode {
  /// A reference to the layer input.
  TrainableNode *input_;

public:
  SigmoidNode(Network *N, TrainableNode *input);

  virtual void forward() override;

  virtual void backward() override;

  virtual std::string getName() const override { return "SigmoidNode"; }
};

class SoftMaxNode final : public TrainableNode {
  /// A reference to the node input.
  TrainableNode *input_;
  /// The selected one-hot value from the softmax function.
  size_t selected_;

  /// A temporary array for storing the subexpression (e ^ (a[i] - max)).
  Array3D<FloatTy> e_;

public:
  /// Ctor - \p is the input layer that must be of shape (1 x 1 x N).
  /// And \p selected that's the selected one-hot representation of the
  /// softmax function.
  SoftMaxNode(Network *N, TrainableNode *input, size_t selected = 0);

  virtual void forward() override;

  virtual void backward() override;

  /// \returns the index of the highest value.
  size_t maxArg() const;

  /// Marks the channel that the SoftMax needs to optimize for.
  void setSelected(size_t selected);

  FloatTy loss() {
    // The loss is the class' negative log likelihood.
    return -std::log(e_.at(0, 0, selected_));
  }

  virtual std::string getName() const override { return "SoftMaxNode"; }
};

class RegressionNode final : public TrainableNode {
  /// A reference to the node input.
  TrainableNode *input_;
  /// The expected input (also known as Y).
  Array3D<FloatTy> expected_;

public:
  /// Ctor - \p is the input layer that must be of shape (1 x 1 x N).
  /// And \p expected (aka Y) is the expected input for the layer, that must
  /// be of the same shape as \p input.
  RegressionNode(Network *N, TrainableNode *input);

  /// \returns a reference to the expected result vector.
  Array3D<FloatTy> &getExpected() { return expected_; }

  virtual void forward() override;

  virtual void backward() override;

  virtual std::string getName() const override { return "RegressionNode"; }
};

/// This node attempts to maximize the inputs by sending back a gradient signal
/// that encourages positive values. This is very useful for debugging.
class MaxNode final : public TrainableNode {
  /// A reference to the node input.
  TrainableNode *input_;

public:
  MaxNode(Network *N, TrainableNode *input);

  virtual void forward() override;

  virtual void backward() override;

  virtual std::string getName() const override { return "MaxNode"; }
};

/// This is an abstraction over raw variable inputs.
class ArrayNode final : public TrainableNode {
public:
  ArrayNode(Network *N, size_t x, size_t y, size_t z) : TrainableNode(N) {
    this->getOutput().reset(x, y, z);
    // Do not change the output of this layer when training the network.
    this->getOutput().isTrainable_ = false;
  }

  void loadRaw(FloatTy *ptr, size_t numElements) {
    this->getOutput().weight_.loadRaw(ptr, numElements);
  }

  virtual std::string getName() const override { return "ArrayNode"; }

  void forward() override {}

  void backward() override {}
};

} // namespace noether

#endif // NOETHER_NODES_H
