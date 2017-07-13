#ifndef NOETHER_NODE_H
#define NOETHER_NODE_H

#include "noether/Network.h"
#include "noether/Tensor.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace noether {

class TrainableData {
public:
  /// Perform a single iteration of the simple SGD algorithm for updating the
  /// weights of the program based on the gradients.
  virtual void train(const TrainingConfig &config) = 0;

  /// Print the textual representation of the buffer.
  virtual void dump() = 0;

  /// Zero out the gradient and prepare for the next round of learning.
  virtual void clearGradient() = 0;
};

/// A pair of some weights and it's derivative. The derivative (gradient) of the
/// weights is optionally initialized.
struct DerivData : public TrainableData {
  /// W - the weight.
  Array3D<FloatTy> weight_{};
  /// dW - the derivative of the weight.
  Array3D<FloatTy> gradient_{};
  /// gradient sum - this buffer is used by the SGD algorithm to store the
  /// previous gradient. The array
  Array3D<FloatTy> gsum_{};
  /// If this flag is set to false then the data is not modified during training
  /// We use this for preventing the trainer from changing the weights of the
  /// input buffers.
  bool isTrainable_{true};

  DerivData() = default;

  DerivData(size_t x, size_t y, size_t z) { reset(x, y, z); }

  /// \returns True if the coordinate is within the array.
  bool isInBounds(size_t x, size_t y, size_t z) const {
    return weight_.isInBounds(x, y, z);
  }

  /// \returns the dimension of the weight tensor.
  std::tuple<size_t, size_t, size_t> dims() const { return weight_.dims(); }

  /// \returns the number of elements in the tensor.
  size_t size() const { return weight_.size(); }

  /// Resets the weights and gradients.
  void reset(std::tuple<size_t, size_t, size_t> dim) {
    size_t x, y, z;
    std::tie(x, y, z) = dim;
    reset(x, y, z);
  }

  /// Resets the weights and gradients.
  void reset(size_t x, size_t y, size_t z) {
    weight_.reset(x, y, z);
    gradient_.reset(x, y, z);
  }

  virtual void dump() override {
    weight_.dump("W");
    if (gradient_.size())
      gradient_.dump("G", "\n");
    if (gsum_.size())
      gsum_.dump("Gsum", "\n");
  }

  virtual void clearGradient() override { gradient_.clear(); }

  virtual void train(const TrainingConfig &config) override {
    size_t batchSize = config.batchSize;
    float L1Decay = config.L1Decay;
    float L2Decay = config.L2Decay;
    float learningRate = config.learningRate;
    float momentum = config.momentum;

    // Do not change the weights of input layers that are marked as untrainable.
    if (!isTrainable_)
      return;

    /// If we are using the momentum technique then we need to allocate an array
    /// for the gradient sum.
    if (momentum > 0.0 && gsum_.size() == 0) {
      gsum_.reset(weight_.dims());
    }

    size_t inx, iny, inz;
    std::tie(inx, iny, inz) = dims();

    // For each weight/gradient pair:
    for (size_t x = 0; x < inx; x++) {
      for (size_t y = 0; y < iny; y++) {
        for (size_t z = 0; z < inz; z++) {
          // Do a simple SGD update:
          FloatTy L1Grad = L1Decay * (weight_.at(x, y, z) > 0 ? 1 : -1);
          FloatTy L2Grad = L2Decay * (weight_.at(x, y, z));
          FloatTy gij = (L2Grad + L1Grad + gradient_.at(x, y, z)) / batchSize;

          // Use the momentum to improve the gradient descent:
          // http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/
          if (momentum > 0.0) {
            // Momentum update:
            FloatTy dx = momentum * gsum_.at(x, y, z) - learningRate * gij;
            // Save this value for the next iteration:
            gsum_.at(x, y, z) = dx;
            // Apply the gradient.
            weight_.at(x, y, z) += dx;
          } else {
            // Use regular SGD:
            weight_.at(x, y, z) -= learningRate * gij;
          }
        }
      }
    }
  }

  /// Performs some checks to validate the correctness of the payload.
  void verify() {
    if (gradient_.size()) {
      assert(gradient_.size() == weight_.size() &&
             "Gradient tensor does not match weight tensor");
    }
    if (gsum_.size()) {
      assert(gsum_.size() == weight_.size() &&
             "Gradient tensor does not match weight tensor");
    }
  }
};

/// This is the non-templated part of the compute node.
class NodeBase {
public:
  /// \returns a descriptive name for the operation.
  virtual std::string getName() const = 0;

  /// Does the forward propagation.
  virtual void forward() = 0;

  /// Does the backwards propagation.
  virtual void backward() = 0;
};

/// Represents a node in the network compute graph.
class Node : public NodeBase {
protected:
  /// The filter output.
  DerivData output_;

public:
  Node(Network *N) { N->registerDerivTensor(this, &output_); }

  /// \returns the output of a node in the compute graph.
  DerivData &getOutput() { return output_; }

  /// \returns the dimension of the tensor.
  std::tuple<size_t, size_t, size_t> dims() const { return output_.dims(); }

  /// \returns the number of elements in the tensor.
  size_t size() const { return output_.size(); }
};
}

#endif // NOETHER_NODE_H
