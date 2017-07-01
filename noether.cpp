#include <cstddef>
#include <cstdint>
#include <vector>
#include <iostream>
#include <string>
#include <cassert>

/// A 3D tensor.
template <class ElemTy>
class Array3D final {
  size_t sx_{0}, sy_{0}, sz_{0};
  ElemTy *data_{nullptr};

  /// \returns the offset of the element in the tensor.
  size_t getElementIdx(size_t x, size_t y, size_t z) {
    assert(x < sx_ && y < sy_ && z < sz_ && "Out of bounds");
    return (sx_ * y + x) * sz_ + z;
  }

public:
  /// \returns the dimention of the tensor.
  std::tuple<size_t, size_t, size_t> dims() { return {sx_, sy_, sz_}; }

  /// \returns the number of elements in the array.
  size_t size() { return sx_ * sy_ * sz_; }

  /// Initialize an empty tensor.
  Array3D() = default;

  /// Initialize a new tensor.
  Array3D(size_t x, size_t y, size_t z) : sx_(x), sy_(y), sz_(z) {
    data_ = new ElemTy[size()];
  }

  ~Array3D() { delete data_; }

  ElemTy &get(size_t x, size_t y, size_t z) {
    return data_[getElementIdx(x,y,z)];
  }
};

/// Represents a node in the network compute graph.
template <class ElemTy>
class Layer {

  /// \returns a descriptive name for the operation.
  virtual std::string getName() = 0;

  /// \returns the output of a node in the compute graph.
  virtual Array3D<ElemTy> &getOutput() = 0;

  /// Does the forward propagation.
  void forward() = 0;

  /// Does the backwards propagation.
  void backward() = 0;
};


int main() {
  Array3D<float> X(320,200,3);
  X.get(10u,10u,2u) = 2;

}
