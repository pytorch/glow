#ifndef NOETHER_TENSOR_H
#define NOETHER_TENSOR_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <tuple>

namespace noether {

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

  void clear(ElemTy value = 0) {
      std::fill(&data_[0], &data_[0] + size(), value);
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
    clear();
  }

  /// Copy ctor.
  Array3D (const Array3D& other) = delete;

  // Move ctor.
  Array3D (Array3D&& other) noexcept {
    data_ = other.data_;
    sx_ = other.sx_;
    sy_ = other.sy_;
    sz_ = other.sz_;
    other.data_ = nullptr;
    other.sx_ = 0;
    other.sy_ = 0;
    other.sz_ = 0;
  }

  Array3D& operator= (const Array3D& other) = delete;

  /// Move assignment operator.
  Array3D& operator= (Array3D&& other) noexcept {
    data_ = other.data_;
    sx_ = other.sx_;
    sy_ = other.sy_;
    sz_ = other.sz_;
    other.data_ = nullptr;
    other.sx_ = 0;
    other.sy_ = 0;
    other.sz_ = 0;
    return *this;
  }

  /// Assigns a new shape to the tensor and allocates a new buffer.
  void reset(std::tuple<size_t, size_t, size_t> dim) {
    size_t x, y, z;
    std::tie(x, y, z) = dim;
    reset(x,y,z);
  }

  /// Assigns a new shape to the tensor and allocates a new buffer.
  void reset(size_t x, size_t y, size_t z) {
    sx_ = x;
    sy_ = y;
    sz_ = z;
    delete[] data_;
    data_ = new ElemTy[size()];
    clear();
  }

  ~Array3D() { delete[] data_; }

  /// Access the flat memory inside the tensor.
  ElemTy &atDirectIndex(size_t index) {
    assert(index < size() && "Out of bounds");
    return data_[index];
  }

  ElemTy &at(size_t x, size_t y, size_t z) {
    return data_[getElementIdx(x, y, z)];
  }
};

}

#endif // NOETHER_TENSOR_H
