#ifndef NOETHER_TENSOR_H
#define NOETHER_TENSOR_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <tuple>

/// A 3D tensor.
template <class ElemTy> class Array3D final {
  size_t sx_{0}, sy_{0}, sz_{0};
  ElemTy *data_{nullptr};

  /// \returns the offset of the element in the tensor.
  size_t getElementIdx(size_t x, size_t y, size_t z) const {
    assert(isInBounds(x, y, z) && "Out of bounds");
    return (sx_ * y + x) * sz_ + z;
  }

public:
  /// \returns True if the coordinate is within the array.
  bool isInBounds(size_t x, size_t y, size_t z) const {
    return x < sx_ && y < sy_ && z < sz_;
  }

  /// \returns the dimension of the tensor.
  std::tuple<size_t, size_t, size_t> dims() const {
    return std::make_tuple(sx_, sy_, sz_);
  }

  /// \returns the number of elements in the array.
  size_t size() const { return sx_ * sy_ * sz_; }

  /// Initialize an empty tensor.
  Array3D() = default;

  /// Initialize a new tensor.
  Array3D(size_t x, size_t y, size_t z) : sx_(x), sy_(y), sz_(z) {
    data_ = new ElemTy[size()];
  }

  /// Assigns a new shape to the tensor and allocates a new buffer.
  void reset(size_t x, size_t y, size_t z) {
    sx_ = x;
    sy_ = y;
    sz_ = z;
    delete data_;
    data_ = new ElemTy[size()];
  }

  ~Array3D() { delete data_; }

  ElemTy &get(size_t x, size_t y, size_t z) const {
    return data_[getElementIdx(x, y, z)];
  }
};

#endif // NOETHER_TENSOR_H
