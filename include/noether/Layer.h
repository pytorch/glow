#ifndef NOETHER_LAYER_H
#define NOETHER_LAYER_H

#include "noether/Tensor.h"

#include <cstddef>
#include <cstdint>
#include <string>


/// A pair of some weighted data and it's derivative.
/// The derivative (gradient) of the data is optionally initialized.
template <class ElemTy> struct DerivData {
  /// W - the weight.
  Array3D<ElemTy> weight_{};
  /// dW - the derivative of the weight.
  Array3D<ElemTy> gradient_{};

  DerivData() = default;

  DerivData(size_t x, size_t y, size_t z) {
    reset(x,y,z);
  }

  /// \returns True if the coordinate is within the array.
  bool isInBounds(size_t x, size_t y, size_t z) const {
    return weight_.isInBounds(x,y,z);
  }

  /// \returns the dimension of the weight tensor.
  std::tuple<size_t, size_t, size_t> dims() const {
    return weight_.dims();
  }

  /// \returns the number of elements in the tensor.
  size_t size() const { return weight_.size(); }

  /// Resets the weights and gradients.
  void reset(size_t x, size_t y, size_t z) {
      weight_.reset(x,y,z);
      gradient_.reset(x,y,z);
  }

  /// Performs some checks to validate the correctness of the payload.
  void verify() {
    if (gradient_.size()) {
      assert(gradient_.size() == weight_.size() &&
             "Gradient tensor does not match weight tensor");
    }
  }
};

/// Represents a node in the network compute graph.
template <class ElemTy> class Layer {
protected:
  /// The filter output.
  DerivData<ElemTy> output_;

public:
  /// \returns a descriptive name for the operation.
  virtual std::string getName() const = 0;

  /// \returns the output of a node in the compute graph.
  const DerivData<ElemTy> &getOutput() const { return output_; }

  /// \returns the dimension of the tensor.
  std::tuple<size_t, size_t, size_t> dims() const {
    return output_.dims();
  }

  /// \returns the number of elements in the tensor.
  size_t size() const { return output_.size(); }

  /// Does the forward propagation.
  virtual void forward() = 0;

  /// Does the backwards propagation.
  virtual void backward() = 0;
};

#endif // NOETHER_LAYER_H
