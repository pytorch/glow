#ifndef NOETHER_LAYER_H
#define NOETHER_LAYER_H

#include "noether/Tensor.h"

#include <cstddef>
#include <cstdint>
#include <string>

/// Represents a node in the network compute graph.
template <class ElemTy> class Layer {
public:
  /// \returns a descriptive name for the operation.
  virtual std::string getName() const = 0;

  /// \returns the output of a node in the compute graph.
  virtual const Array3D<ElemTy> &getOutput() const = 0;

  /// \returns the dimension of the tensor.
  std::tuple<size_t, size_t, size_t> dims() const { return getOutput().dims(); }

  /// \returns the number of elements in the tensor.
  size_t size() const { return getOutput().size(); }

  /// Does the forward propagation.
  virtual void forward() = 0;

  /// Does the backwards propagation.
  virtual void backward() = 0;
};

#endif // NOETHER_LAYER_H
