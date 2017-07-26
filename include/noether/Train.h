#ifndef NOETHER_TRAIN_H
#define NOETHER_TRAIN_H

#include "noether/ADT.h"
#include "noether/Tensor.h"

#include <cstddef>
#include <cstdint>

namespace noether {

/// This is a list of parameters that the network trainers (such as sgd and
/// adam) use for training the network.
struct TrainingConfig {
  size_t batchSize{1};
  float L1Decay{0};
  float L2Decay{0};
  float learningRate{0.01};
  float momentum{0.0};
};

/// A pair of some weights and it's derivative. The derivative (gradient) of the
/// weights is optionally initialized.
class TrainableData {
public:
  /// W - the weight.
  Tensor weight_{};
  /// dW - the derivative of the weight.
  Tensor gradient_{};
  /// gradient sum - this buffer is used by the SGD algorithm to store the
  /// previous gradient. The array
  Tensor gsum_{};
  /// If this flag is set to false then the data is not modified during training
  /// We use this for preventing the trainer from changing the weights of the
  /// input buffers.
  bool isTrainable_{true};

  TrainableData() = default;

  TrainableData(ArrayRef<size_t> dims) { reset(dims); }

  /// \returns True if the coordinate is within the array.
  bool isInBounds(ArrayRef<size_t> dims) const {
    return weight_.isInBounds(dims);
  }

  /// \returns the dimension of the weight tensor.
  ArrayRef<size_t> dims() const { return weight_.dims(); }

  /// \returns the number of elements in the tensor.
  size_t size() const { return weight_.size(); }

  /// Resets the weights and gradients.
  void reset(ArrayRef<size_t> dims) {
    weight_.reset(ElemKind::FloatTy, dims);
    gradient_.reset(ElemKind::FloatTy, dims);
    gsum_.reset(ElemKind::FloatTy, {});
  }

  /// Print the textual representation of the buffer.
  void dump();

  /// Zero out the gradient and prepare for the next round of learning.
  void clearGradient() { gradient_.zero(); }

  /// Perform a single iteration of the simple SGD algorithm for updating the
  /// weights of the program based on the gradients.
  void train(const TrainingConfig &config);

  /// Performs some checks to validate the correctness of the payload.
  void verify();
};

} // namespace

#endif // NOETHER_TRAIN_H
