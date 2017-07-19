#ifndef NOETHER_TENSOR_H
#define NOETHER_TENSOR_H

#include "Config.h"

#include "noether/ADT.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>


namespace noether {

static const unsigned max_tensor_dimensions = 6;

struct Point3d {
  size_t x{0};
  size_t y{0};
  size_t z{0};
  Point3d(size_t x, size_t y, size_t z) : x(x), y(y), z(z) {}
  bool operator==(const Point3d &other) {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct Point4d {
  size_t w{0};
  size_t x{0};
  size_t y{0};
  size_t z{0};
  Point4d(size_t w, size_t x, size_t y, size_t z) : w(w), x(x), y(y), z(z) {}
  bool operator==(const Point4d &other) {
    return w == other.w && x == other.x && y == other.y && z == other.z;
  }
};

/// This is the default floating point type used for training.
using FloatTy = TRAINING_TENSOR_ELEMENT_TYPE;

const size_t numRandomVals = 32;
const double randomVals[] = {
    0.4414916532837977,  0.9746009401393961,  0.7644974627030178,
    0.27827514759492433, 0.2823586355251567,  0.2506456727980517,
    0.42572714689894375, 0.0336972057297974,  0.243959576298854,
    0.18966169453269455, 0.5286137478346916,  0.6222636988212916,
    0.2957554911353527,  0.620880111550055,   0.6401712669183511,
    0.7998884586830336,  0.5631086934294972,  0.2538295602954759,
    0.8832542169407436,  0.48075360869679007, 0.5971563343845181,
    0.15586842213087104, 0.11915981553150845, 0.6615953736728364,
    0.9890390729567575,  0.12803209414922834, 0.6681364032106246,
    0.9624442797641373,  0.9968422482663746,  0.9591749207255753,
    0.7215510646287491,  0.9200771645764693};

/// A 3D tensor.
template <class ElemTy> class Array3D final {
  size_t sx_{0}, sy_{0}, sz_{0};
  ElemTy *data_{nullptr};

  /// \returns the offset of the element in the tensor.
  size_t getElementIdx(size_t x, size_t y, size_t z) const {
    assert(isInBounds(x, y, z) && "Out of bounds");
    return (y * sx_ + x) * sz_ + z;
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
    static int offset = 0;
    double scale = std::sqrt(double(size()));
    for (size_t i = 0, e = size(); i < e; ++i) {
      data_[i] = (randomVals[(offset + i) % numRandomVals] - 0.5) / scale;
    }

    offset++;
  }

  /// \returns the dimension of the tensor.
  Point3d dims() const { return Point3d(sx_, sy_, sz_); }

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
  Array3D(const Array3D &other) = delete;

  // Move ctor.
  Array3D(Array3D &&other) noexcept {
    data_ = other.data_;
    sx_ = other.sx_;
    sy_ = other.sy_;
    sz_ = other.sz_;
    other.data_ = nullptr;
    other.sx_ = 0;
    other.sy_ = 0;
    other.sz_ = 0;
  }

  Array3D &operator=(const Array3D &other) = delete;

  /// Move assignment operator.
  Array3D &operator=(Array3D &&other) noexcept {
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
  void reset(Point3d dim) { reset(dim.x, dim.y, dim.z); }

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

  /// Add all of the elements in the array.
  ElemTy sum() { return std::accumulate(&data_[0], &data_[size()], ElemTy(0)); }

  ElemTy &at(size_t x, size_t y, size_t z) {
    return data_[getElementIdx(x, y, z)];
  }

  const ElemTy &at(size_t x, size_t y, size_t z) const {
    return data_[getElementIdx(x, y, z)];
  }

  static char valueToChar(ElemTy val) {
    char ch = ' ';
    if (val > 0.2)
      ch = '.';
    if (val > 0.4)
      ch = ',';
    if (val > 0.6)
      ch = ':';
    if (val > 0.8)
      ch = 'o';
    if (val > 1.0)
      ch = 'O';
    if (val > 1.5)
      ch = '0';
    if (val > 2.0)
      ch = '@';
    if (val < -0.1)
      ch = '-';
    if (val < -0.2)
      ch = '~';
    if (val < -0.4)
      ch = '=';
    if (val < -1.0)
      ch = '#';
    return ch;
  }

  void dumpAscii(const std::string &prefix = "", std::string suffix = "\n") {
    std::cout << prefix << "\n";
    for (size_t z = 0; z < sz_; z++) {
      std::cout << "Layer #" << z << "\n";
      for (size_t y = 0; y < sy_; y++) {
        for (size_t x = 0; x < sx_; x++) {
          auto val = at(x, y, z);
          std::cout << valueToChar(val);
        }
        std::cout << "\n";
      }
      std::cout << suffix;
    }
  }

  void dump(std::string title = "", std::string suffix = "") {
    ElemTy mx = *std::max_element(&data_[0], &data_[size()]);
    ElemTy mn = *std::min_element(&data_[0], &data_[size()]);

    std::cout << title << " max=" << mx << " min=" << mn << "[";
    for (int z = 0; z < sz_; z++) {
      std::cout << "[";
      for (int y = 0; y < sy_; y++) {
        std::cout << "[";
        for (int x = 0; x < sx_; x++) {
          std::cout << at(x, y, z) << " ";
        }
        std::cout << "]";
      }
      std::cout << "]";
    }
    std::cout << "]" << suffix;
  }
};

/// A 4D tensor.
template <class ElemTy> class Array4D final {
  size_t sw_{0}, sx_{0}, sy_{0}, sz_{0};
  ElemTy *data_{nullptr};

  /// \returns the offset of the element in the tensor.
  size_t getElementIdx(size_t w, size_t x, size_t y, size_t z) const {
    assert(isInBounds(w, x, y, z) && "Out of bounds");
    return (((w * sy_ + y) * sx_ + x) * sz_ + z);
  }

public:
  /// \returns True if the coordinate is within the array.
  bool isInBounds(size_t w, size_t x, size_t y, size_t z) const {
    return w < sw_ && x < sx_ && y < sy_ && z < sz_;
  }

  void clear(ElemTy value = 0) {
    std::fill(&data_[0], &data_[0] + size(), value);
  }

  /// Fill the array with random data that's close to zero.
  void randomize() {
    static int offset = 0;
    double scale = std::sqrt(double(size()));
    for (size_t i = 0, e = size(); i < e; ++i) {
      data_[i] = (randomVals[(offset + i) % numRandomVals] - 0.5) / scale;
    }

    offset++;
  }

  /// \returns the dimension of the tensor.
  Point4d dims() const { return {sw_, sx_, sy_, sz_}; }

  /// \returns the number of elements in the array.
  size_t size() const { return sw_ * sx_ * sy_ * sz_; }

  /// Initialize an empty tensor.
  Array4D() = default;

  /// Initialize a new tensor.
  Array4D(size_t w, size_t x, size_t y, size_t z)
      : sw_(w), sx_(x), sy_(y), sz_(z) {
    data_ = new ElemTy[size()];
    clear();
  }

  Array4D(const Array4D &other) = delete;
  Array4D &operator=(const Array4D &other) = delete;

  /// Assigns a new shape to the tensor and allocates a new buffer.
  void reset(Point4d dim) { reset(dim.w, dim.x, dim.y, dim.z); }

  /// Assigns a new shape to the tensor and allocates a new buffer.
  void reset(size_t w, size_t x, size_t y, size_t z) {
    sw_ = w;
    sx_ = x;
    sy_ = y;
    sz_ = z;
    delete[] data_;
    data_ = new ElemTy[size()];
    clear();
  }

  ~Array4D() { delete[] data_; }

  /// Extract a 3D array from the \p w slice.
  Array3D<ElemTy> extractSlice(size_t w) {
    Array3D<ElemTy> slice(sx_, sy_, sz_);

    for (int x = 0; x < sx_; x++)
      for (int y = 0; y < sy_; y++)
        for (int z = 0; z < sz_; z++)
          slice.at(x, y, z) = this->at(w, x, y, z);

    return slice;
  }

  ElemTy &at(size_t w, size_t x, size_t y, size_t z) {
    return data_[getElementIdx(w, x, y, z)];
  }

  const ElemTy &at(size_t w, size_t x, size_t y, size_t z) const {
    return data_[getElementIdx(w, x, y, z)];
  }
};

template <class ElemTy>
class Handle;

template <class ElemTy>
class Tensor final {
  /// Contains the dimentions (sizes) of the tensor. Ex: [sz, sy, sz].
  uint32_t sizes_[max_tensor_dimensions] = {0,};
  uint8_t numSizes_{0};

  /// A pointer to the tensor data.
  ElemTy *data_{nullptr};

public:
  /// \returns True if the coordinate is within the array.
  bool isInBounds(ArrayRef<uint32_t> indices) const {
    assert(numSizes_ == indices.size() && "Invalid number of indices");
    for (int i = 0, e = indices.size(); i < e; i++) {
      if (indices[i] > sizes_[i]) return false;
    }
    return true;
  }

  void clear(ElemTy value = 0) {
    std::fill(&data_[0], &data_[0] + size(), value);
  }

  /// Fill the array with random data that's close to zero.
  void randomize() {
    static int offset = 0;
    double scale = std::sqrt(double(size()));
    for (size_t i = 0, e = size(); i < e; ++i) {
      data_[i] = (randomVals[(offset + i) % numRandomVals] - 0.5) / scale;
    }

    offset++;
  }

  /// \returns the dimension of the tensor.
  ArrayRef<uint32_t> dims() const {
    return ArrayRef<uint32_t>(sizes_, numSizes_);
  }

  /// \returns the number of elements in the array.
  size_t size() const {
    size_t s = 1;
    for (int i = 0; i < numSizes_; i++) {
      s *= size_t(sizes_[i]);
    }
    return s;
  }

  /// Initialize an empty tensor.
  Tensor() = default;

  /// Initialize a new tensor.
  Tensor(ArrayRef<uint32_t> dims) : data_(nullptr) {
    reset(dims);
  }

  Tensor(const Tensor &other) = delete;
  Tensor &operator=(const Tensor &other) = delete;

  /// Assigns a new shape to the tensor and allocates a new buffer.
  void reset(ArrayRef<uint32_t> dims) {
    delete[] data_;

    assert(dims.size() < max_tensor_dimensions && "Too many indices");
    for (int i = 0, e = dims.size(); i < e; i++) { sizes_[i] = dims[i]; }
    numSizes_ = dims.size();

    data_ = new ElemTy[size()];
    clear();
  }

  ElemTy &at(size_t index) {
    assert(index < size() && "Out of bounds");
    return data_[index];
  }

  const ElemTy &at(size_t index) const {
    assert(index < size() && "Out of bounds");
    return data_[index];
  }

  ~Tensor() { delete[] data_; }

  // Move ctor.
  Tensor(Tensor &&other) noexcept {
    std::swap(data_, other.data_);
    for (int i = 0; i < max_tensor_dimensions; i++) {
      std::swap(sizes_[i], other.sizes_[i]);
    }
    std::swap(numSizes_, other.numSizes_);
  }

  /// Move assignment operator.
  Tensor &operator=(Tensor &&other) noexcept {
    std::swap(data_, other.data_);
    for (int i = 0; i < max_tensor_dimensions; i++) {
      std::swap(sizes_[i], other.sizes_[i]);
    }
    std::swap(numSizes_, other.numSizes_);
    return *this;
  }

  void dump(std::string title = "", std::string suffix = "") {
    ElemTy mx = *std::max_element(&data_[0], &data_[size()]);
    ElemTy mn = *std::min_element(&data_[0], &data_[size()]);

    std::cout << title << " max=" << mx << " min=" << mn << "[";

    std::cout << "[";
    for (size_t i = 0, e = std::min(400u, size()); i < e; i++) {
      std::cout << at(i) << " ";
    }
    std::cout << "]" << suffix;
  }

  /// \return a new handle that points and manages this tensor.
  Handle<ElemTy> getHandle();
};


template <class ElemTy>
class Handle final {
  Tensor<ElemTy> *tensor_{nullptr};

  /// Contains the multiplication of the sizes from current position to end.
  /// For example, for index (w,z,y,z):  [x * y * z, y * z, z, 1]
  uint32_t sizeIntegral[max_tensor_dimensions] = {0,};
  uint8_t numDims{0};

  /// Calculate the index for a specific element in the tensor.
  size_t getElementPtr(ArrayRef<uint32_t> indices) {
    assert(indices.size() == numDims && "Invalid number of indices");
    size_t index = 0;
    for (int i = 0, e = indices.size(); i < e; i++) {
      index += size_t(sizeIntegral[i]) * size_t(indices[i]);
    }

    return index;
  }

public:
  /// Construct a Tensor handle. \p rsizes is a list of reverse sizes
  /// Example: (sz, sy, sx, sw, ... )
  Handle(Tensor<ElemTy> *tensor) : tensor_(tensor) {
    ArrayRef<uint32_t> sizes = tensor->dims();
    numDims = sizes.size();
    assert(sizes.size() && "Size list must not be empty");
    uint32_t pi = 1;

    for (int i = numDims - 1; i >= 0; i--) {
      sizeIntegral[i] = pi;
      assert(sizes[i] > 0 && "invalid dim size");
      pi *= sizes[i];
    }

    assert(numDims < max_tensor_dimensions);
  }

  size_t size() { return sizeIntegral[numDims]; }

  bool isInBounds(ArrayRef<uint32_t> indices) const {
    return tensor_->isInBounds(indices);
  }

  void clear(ElemTy value = 0) {
    tensor_->clear(value);
  }

  ElemTy &at(ArrayRef<uint32_t> indices) {
    assert(tensor_->isInBounds(indices));
    return tensor_->at(getElementPtr(indices));
  }

  const ElemTy &at(ArrayRef<uint32_t> indices) const {
    assert(tensor_->isInBounds(indices));
    return tensor_->at(getElementPtr(indices));
  }

  /// Extract a smaller dimension tensor from a specific slice (that has to be
  /// the first dimention).
  Tensor<ElemTy> extractSlice(size_t idx) {
    ArrayRef<uint32_t> sizes = tensor_->dims();
    assert(sizes.size() > 1 && "Tensor has only one dimension");
    Tensor<ElemTy> slice(ArrayRef<uint32_t>(&sizes[1], sizes.size() - 1));

    // Extract the whole slice.
    auto *base = &tensor_->at(sizeIntegral[0] * idx);
    std::copy(base, base + sizeIntegral[0], &slice.at(0));

    return slice;
  }

  void randomize() {
    tensor_->randomize();
  }

  void dump(std::string title = "", std::string suffix = "") {
    tensor_->dump(title, suffix);
  }

  void dumpAscii(const std::string &prefix = "", std::string suffix = "\n") {
    auto d = tensor_->dims();
    std::cout << prefix << "\n";

    if (d.size() == 2) {
      for (uint32_t y = 0; y < d[1]; y++) {
        for (uint32_t x = 0; x < d[0]; x++) {
          auto val = at({x, y});
          std::cout << valueToChar(val);
        }
        std::cout << "\n";
      }
    } else if (d.size() == 3) {
      for (uint32_t z = 0; z < d[2]; z++) {
        std::cout << "\n";
        for (uint32_t y = 0; y < d[1]; y++) {
          for (uint32_t x = 0; x < d[0]; x++) {
            auto val = at({x, y, z});
            std::cout << valueToChar(val);
          }
          std::cout << "\n";
        }
      }
    } else {
      assert(false && "Invalid tensor size");
    }

    std::cout << suffix;
  }

};

template <class ElemTy>
Handle<ElemTy> Tensor<ElemTy>::getHandle() {
  return Handle<ElemTy>(this);
}

}

#endif // NOETHER_TENSOR_H
