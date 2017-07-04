#ifndef NOETHER_TENSOR_H
#define NOETHER_TENSOR_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <tuple>

namespace noether {

const size_t numRandomVals = 32;
const float randomVals[] = {0.05864586968245285, -0.06103524975849629, -0.09327265435603066, 0.008119399801320745, 0.09787898774061532, 0.047409735345541805, -0.07098228945991622, -0.08767971806030812, 0.02123944856834319, 0.09972937485816817, 0.03533945628595544, -0.07968044005185564, -0.08054411181351735, 0.033985802485422045, 0.09982508582899008, 0.02264740153518888, -0.08697666299992027, -0.07199138215690071, 0.04613419740563074, 0.09816443667824695, 0.009556879880205291, -0.0927425858121898, -0.062172009034198415, 0.05747088989762946, 0.09477664549195447, -0.0037017891153869956, -0.09687676067314378, -0.05125875819089679, 0.06779641792802212, 0.0897213183429788, -0.01689532744039967, -0.09930644935469998};

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

  /// Fill the array with random data that's close to zero.
  void randomize() {
    for (size_t i = 0, e = size(); i < e; ++i) {
      data_[i] = randomVals[i % numRandomVals];
    }
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
