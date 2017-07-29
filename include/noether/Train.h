#ifndef NOETHER_TRAIN_H
#define NOETHER_TRAIN_H

#include "noether/ADT.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>


namespace noether {

class Tensor;

/// This is a list of parameters that the network trainers (such as sgd and
/// adam) use for training the network.
struct TrainingConfig {
  size_t batchSize{1};
  float L1Decay{0};
  float L2Decay{0};
  float learningRate{0.01};
  float momentum{0.0};
};

class Network;

/// The address of this token is used for keeping track of buffers inside the
/// context.
struct TensorToken {
    char ID;
};

/// A pair of some weights and it's derivative. The derivative (gradient) of the
/// weights is optionally initialized.
class TrainableData {
public:
  /// W - the weights.
  Tensor weights_{};
  /// dW - the derivative of the weights.
  Tensor gradients_{};
  /// If this flag is set to false then the data is not modified during the
  /// training process. We use this for preventing the trainer from changing the
  /// weights of the input buffers.
  bool isTrainable_{false};

  TrainableData(bool trainable) : isTrainable_(trainable) {}

  TrainableData(bool trainable, ArrayRef<size_t> dims) :
    isTrainable_(trainable) { reset(dims); }

  /// Resets the weights and gradients.
  void reset(ArrayRef<size_t> dims) {
    weights_.reset(ElemKind::FloatTy, dims);
    gradients_.reset(ElemKind::FloatTy, dims);
  }

  Handle<FloatTy> getWeightHandle() { return weights_.getHandle<FloatTy>(); }
  Handle<FloatTy> getGradHandle() { return gradients_.getHandle<FloatTy>(); }

  /// \returns the dimension of the weight tensor.
  ArrayRef<size_t> dims() const { return weights_.dims(); }

  /// \returns the number of elements in the tensor.
  size_t size() const { return weights_.size(); }

  void mergeGradients(TrainableData *other) {
    auto myGrad = getGradHandle();
    auto otherGrad = other->getGradHandle();
    (void) otherGrad; (void) myGrad;
    assert(myGrad.dims() == otherGrad.dims() && "Mismatching sizes");

    for (size_t i = 0, e = myGrad.size(); i < e; i++) {
      myGrad.raw(i) += otherGrad.raw(i);
    }
  }

  void copyWeights(TrainableData *other) {
    auto myW = getWeightHandle();
    auto otherW = other->getWeightHandle();
    (void) otherW; (void) myW;
    assert(myW.dims() == otherW.dims() && "Mismatching sizes");
    weights_ = other->weights_.clone();
  }

  /// Zero out the gradient and prepare for the next round of learning.
  void clearGradient() { gradients_.zero(); }
};

class Trainer {
public:
  /// Holds the training configuration.
  TrainingConfig config{};
  
private:
  /// A temporary data structure for holding the attached gsum buffers.
  /// This data strucure owns the attached tensors, which are the values of
  /// the map, not the keys.
  std::unordered_map<Tensor*, Tensor*> gsum_;

public:

  Trainer() = default;

  ~Trainer();

  /// Perform a single iteration of the simple SGD algorithm for updating the
  /// weights of the program based on the gradients.
  void train(TrainableData *trainable);
};

} // namespace

#endif // NOETHER_TRAIN_H
