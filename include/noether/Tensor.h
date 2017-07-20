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

constexpr unsigned max_tensor_dimensions = 6;

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

template <class ElemTy>
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

template <class ElemTy>
class Handle;

/// A class that represents a contiguous n-dimensional array (a tensor).
template <class ElemTy>
class Tensor final {
  /// Contains the dimentions (sizes) of the tensor. Ex: [sz, sy, sz, ...].
  size_t sizes_[max_tensor_dimensions] = {0,};
  unsigned char numSizes_{0};

  /// A pointer to the tensor data.
  ElemTy *data_{nullptr};

public:
  /// \returns True if the coordinate is within the array.
  bool isInBounds(ArrayRef<size_t> indices) const {
    assert(numSizes_ == indices.size() && "Invalid number of indices");
    for (unsigned i = 0u, e = indices.size(); i < e; i++) {
      if (indices[i] >= sizes_[i]) return false;
    }
    return true;
  }

  void clear(ElemTy value = 0) {
    std::fill(&data_[0], &data_[0] + size(), value);
  }

  /// Fill the array with random data that's close to zero.
  void randomize() {
    // This is a global variable, a singleton, that's used as an index into
    // the array with random numbers.
    static int offset = 0;

    double scale = std::sqrt(double(size()));
    for (size_t i = 0, e = size(); i < e; ++i) {
      data_[i] = (randomVals[(offset++) % numRandomVals] - 0.5) / scale;
    }

    offset++;
  }

  /// \returns the dimension of the tensor.
  ArrayRef<size_t> dims() const {
    return ArrayRef<size_t>(sizes_, numSizes_);
  }

  /// \returns the number of elements in the array.
  size_t size() const {
    if (!numSizes_)
      return 0;
    
    size_t s = 1;
    for (unsigned i = 0; i < numSizes_; i++) {
      s *= size_t(sizes_[i]);
    }

    return s;
  }

  /// Initialize an empty tensor.
  Tensor() = default;

  /// Initialize a new tensor.
  Tensor(ArrayRef<size_t> dims) : data_(nullptr) {
    reset(dims);
  }

  Tensor(const Tensor &other) = delete;
  Tensor &operator=(const Tensor &other) = delete;

  /// Assigns a new shape to the tensor and allocates a new buffer.
  void reset(ArrayRef<size_t> dims) {
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

  void dump(const std::string &title = "", const std::string &suffix = "") {
    ElemTy mx = *std::max_element(&data_[0], &data_[size()]);
    ElemTy mn = *std::min_element(&data_[0], &data_[size()]);

    std::cout << title << " max=" << mx << " min=" << mn << "[";

    std::cout << "[";
    for (size_t i = 0, e = std::min<size_t>(400, size()); i < e; i++) {
      std::cout << at(i) << " ";
    }
    std::cout << "]" << suffix;
  }

  /// \return a new handle that points and manages this tensor.
  Handle<ElemTy> getHandle();
};


/// A class that provides indexed access to a tensor. This class has value
/// semantics and it's copied around. One of the reasons for making this class
/// value semantics is to allow efficient index calculation that the compiler
/// can optimize (because stack allocated structures don't alias).
template <class ElemTy>
class Handle final {
  Tensor<ElemTy> *tensor_{nullptr};

  /// Contains the multiplication of the sizes from current position to end.
  /// For example, for index (w,z,y,z):  [x * y * z, y * z, z, 1]
  size_t sizeIntegral[max_tensor_dimensions] = {0,};
  uint8_t numDims{0};

  /// Calculate the index for a specific element in the tensor.
  size_t getElementPtr(ArrayRef<size_t> indices) {
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
    auto sizes = tensor->dims();
    numDims = sizes.size();
    assert(sizes.size() && "Size list must not be empty");

    size_t pi = 1;
    for (int i = numDims - 1; i >= 0; i--) {
      sizeIntegral[i] = pi;
      assert(sizes[i] > 0 && "invalid dim size");
      pi *= sizes[i];
    }

    assert(numDims < max_tensor_dimensions);
  }

  size_t size() { return tensor_->size(); }

  bool isInBounds(ArrayRef<size_t> indices) const {
    return tensor_->isInBounds(indices);
  }

  void clear(ElemTy value = 0) {
    tensor_->clear(value);
  }

  ElemTy &at(ArrayRef<size_t> indices) {
    assert(tensor_->isInBounds(indices));
    return tensor_->at(getElementPtr(indices));
  }

  const ElemTy &at(ArrayRef<size_t> indices) const {
    assert(tensor_->isInBounds(indices));
    assert(indices.size() == numDims && "Wrong number of indices");
    return tensor_->at(getElementPtr(indices));
  }

  /// \returns the element at offset \p idx without any size calculations.
  ElemTy &raw(size_t idx) { return tensor_->at(idx); }

  /// \returns the element at offset \p idx without any size calculations.
  const ElemTy &raw(size_t idx) const { return tensor_->at(idx); }

  /// Extract a smaller dimension tensor from a specific slice (that has to be
  /// the first dimention).
  Tensor<ElemTy> extractSlice(size_t idx) {
    auto sizes = tensor_->dims();
    assert(sizes.size() > 1 && "Tensor has only one dimension");
    Tensor<ElemTy> slice(ArrayRef<size_t>(&sizes[1], sizes.size() - 1));

    // Extract the whole slice.
    auto *base = &tensor_->at(sizeIntegral[0] * idx);
    std::copy(base, base + sizeIntegral[0], &slice.at(0));

    return slice;
  }

  void randomize() {
    tensor_->randomize();
  }

  void dump(const std::string &title = "", const std::string &suffix = "") {
    tensor_->dump(title, suffix);
  }

  void dumpAscii(const std::string &prefix = "",
                 const std::string &suffix = "\n") {
    auto d = tensor_->dims();
    std::cout << prefix << "\n";

    if (d.size() == 2) {
      for (size_t y = 0; y < d[1]; y++) {
        for (size_t x = 0; x < d[0]; x++) {
          auto val = at({x, y});
          std::cout << valueToChar(val);
        }
        std::cout << "\n";
      }
    } else if (d.size() == 3) {
      for (size_t z = 0; z < d[2]; z++) {
        std::cout << "\n";
        for (size_t y = 0; y < d[1]; y++) {
          for (size_t x = 0; x < d[0]; x++) {
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
