#ifndef NOETHER_NODE_H
#define NOETHER_NODE_H

#include "noether/Tensor.h"
#include "noether/Network.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <map>

namespace noether {

class TrainableData {
public:
  /// Perform a single iteration of the simple SGD algorithm for updating the
  /// weights of the program based on the gradients.
  virtual void train() = 0;

  /// Print the textual representation of the buffer.
  virtual void dump() = 0;
};

/// A pair of some weights and it's derivative. The derivative (gradient) of the
/// weights is optionally initialized.
template <class ElemTy> struct DerivData : public TrainableData {
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
  void reset(std::tuple<size_t, size_t, size_t> dim) {
    size_t x, y, z;
    std::tie(x, y, z) = dim;
    reset(x,y,z);
  }

  /// Resets the weights and gradients.
  void reset(size_t x, size_t y, size_t z) {
      weight_.reset(x,y,z);
      gradient_.reset(x,y,z);
  }

  virtual void dump () override {
    weight_.dump("W");
    gradient_.dump("G", "\n");
  }

  virtual void train () override {
    ElemTy batchSize = 1;
    ElemTy L1Decay = 0;
    ElemTy L2Decay = 0;
    ElemTy learningRate = 0.001;

    size_t inx, iny, inz;
    std::tie(inx, iny, inz) = dims();

    // For each weight/gradient pair:
    for (size_t x = 0; x < inx; x++) {
      for (size_t y = 0; y < iny; y++) {
        for (size_t z = 0; z < inz; z++) {
          // Do a simple SGD update:
          ElemTy L1Grad = L1Decay * (weight_.at(x, y, z) > 0 ? 1 : -1);
          ElemTy L2Grad = L2Decay * (weight_.at(x, y, z));
          ElemTy gij = (L2Grad + L1Grad + gradient_.at(x,y,z)) / batchSize;
          weight_.at(x,y,z) -= learningRate * gij;
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
template <class ElemTy> class Node : public NodeBase {
protected:
  /// The filter output.
  DerivData<ElemTy> output_;

public:
  Node(Network *N) { N->registerDerivTensor(this, &output_); }

  /// \returns the output of a node in the compute graph.
  DerivData<ElemTy> &getOutput() { return output_; }

  /// \returns the dimension of the tensor.
  std::tuple<size_t, size_t, size_t> dims() const {
    return output_.dims();
  }

  /// \returns the number of elements in the tensor.
  size_t size() const { return output_.size(); }
};

}

#endif // NOETHER_NODE_H
