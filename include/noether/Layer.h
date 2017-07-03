#ifndef NOETHER_LAYER_H
#define NOETHER_LAYER_H

#include "noether/Tensor.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <map>


namespace noether {

class LayerBase;

class TrainableData;

class Network {
  /// A list of dependencies.
  std::map<LayerBase*, std::vector<LayerBase*>> deps_;

  /// A list of buffers to train as part of the backwards prop pass.
  std::vector<TrainableData*> trainableBuffers_;

  /// Generate a topological order of the nodes in the network.
  void sortNetwork(std::vector<LayerBase*> &order);
public:
  Network();

  /// Add \p dep as a dependency (prerequisite) for \p layer.
  void addLayerDependency(LayerBase *node, LayerBase *dep);

  /// Registers the derivable data \p weights (weights and gradient) as
  /// belonging to the node \p node.
  void registerDerivTensor(LayerBase *node, TrainableData *weights);

  /// Train the network on a single input.
  void train();

  /// Infer data for a single input.
  void infer();
};


class TrainableData {
public:
  /// Perform a single iteration of the simple SGD algorithm for updating the
  /// weights of the program based on the gradients.
  virtual void train() = 0;
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
class LayerBase {
public:
  /// \returns a descriptive name for the operation.
  virtual std::string getName() const = 0;

  /// Does the forward propagation.
  virtual void forward() = 0;

  /// Does the backwards propagation.
  virtual void backward() = 0;
};

/// Represents a node in the network compute graph.
template <class ElemTy> class Layer : public LayerBase {
protected:
  /// The filter output.
  DerivData<ElemTy> output_;

public:
  Layer(Network *N) { N->registerDerivTensor(this, &output_); }

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

#endif // NOETHER_LAYER_H
