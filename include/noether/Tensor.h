#ifndef NOETHER_TENSOR_H
#define NOETHER_TENSOR_H

#include "Config.h"

#include "noether/ADT.h"
#include "noether/Random.h"

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

template <class ElemTy> static char valueToChar(ElemTy val) {
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

enum class ElemKind : unsigned char {
  FloatTy,
  DoubleTy,
  Int8Ty,
  Int32Ty,
  IndexTy,
};

template <class ElemTy> class Handle;

/// A class that represents a contiguous n-dimensional array (a tensor).
class Tensor final {
  /// A pointer to the tensor data.
  char *data_{nullptr};

  /// Contains the dimentions (sizes) of the tensor. Ex: [sx, sy, sz, ...].
  size_t sizes_[max_tensor_dimensions] = {
      0,
  };

  /// Contains the number of dimensions used by the tensor.
  unsigned char numSizes_{0};

  /// Specifies the element type of the tensor.
  ElemKind elementType_;

  template <class ElemTy> friend class Handle;

public:
  /// \returns true if the templated parameter \p ElemTy matches the type that's
  /// specified by the parameter \p Ty.
  template <class ElemTy> static bool isType(ElemKind Ty) {
    switch (Ty) {
    case ElemKind::FloatTy:
      return std::is_same<ElemTy, float>::value;
    case ElemKind::DoubleTy:
      return std::is_same<ElemTy, double>::value;
    case ElemKind::Int8Ty:
      return std::is_same<ElemTy, int8_t>::value;
    case ElemKind::Int32Ty:
      return std::is_same<ElemTy, int32_t>::value;
    case ElemKind::IndexTy:
      return std::is_same<ElemTy, size_t>::value;
    }
  }

  /// \return the size of the element \p Ty.
  static unsigned getElementSize(ElemKind Ty) {
    switch (Ty) {
    case ElemKind::FloatTy:
      return sizeof(float);
    case ElemKind::DoubleTy:
      return sizeof(double);
    case ElemKind::Int8Ty:
      return sizeof(int8_t);
    case ElemKind::Int32Ty:
      return sizeof(int32_t);
    case ElemKind::IndexTy:
      return sizeof(size_t);
    }
  }

  /// \return the element type of the tensor.
  ElemKind getElementType() const { return elementType_; }

  /// \returns True if the coordinate is within the array.
  bool isInBounds(ArrayRef<size_t> indices) const {
    assert(numSizes_ == indices.size() && "Invalid number of indices");
    for (unsigned i = 0u, e = indices.size(); i < e; i++) {
      if (indices[i] >= sizes_[i])
        return false;
    }
    return true;
  }

  /// Set the content of the tensor to zero.
  void zero() {
    std::fill(&data_[0], &data_[0] + size() * getElementSize(elementType_), 0);
  }

  /// \returns the shape of the tensor.
  ArrayRef<size_t> dims() const { return ArrayRef<size_t>(sizes_, numSizes_); }

  /// \returns the number of elements in the tensor.
  size_t size() const {
    if (!numSizes_)
      return 0;

    size_t s = 1;
    for (unsigned i = 0; i < numSizes_; i++) {
      s *= size_t(sizes_[i]);
    }

    return s;
  }

  /// \returns a pointer to the raw data, of type \p ElemTy.
  template <class ElemTy> ElemTy *getRawDataPointer() {
    assert(isType<ElemTy>(elementType_) && "Asking for the wrong ptr type.");
    return reinterpret_cast<ElemTy *>(data_);
  }

  /// Initialize an empty tensor.
  Tensor() {}

  /// Initialize from a list of float literals.
  Tensor(const std::initializer_list<double> &vec) {
    reset(ElemKind::FloatTy, {vec.size()});
    FloatTy *data = getRawDataPointer<FloatTy>();
    int i = 0;
    for (auto &f : vec) {
      data[i++] = f;
    }
  }

  /// Allocate and initialize a new tensor.
  Tensor(ElemKind elemTy, ArrayRef<size_t> dims)
      : data_(nullptr), elementType_(elemTy) {
    reset(elemTy, dims);
  }

  Tensor(const Tensor &other) = delete;
  Tensor &operator=(const Tensor &other) = delete;

  /// Reset the shape and type of this tensor to match the shape and type of
  /// \p other.
  void reset(const Tensor *other) { reset(other->getElementType(),
                                          other->dims()); }

  /// Assigns a new shape to the tensor and allocates a new buffer.
  void reset(ElemKind elemTy, ArrayRef<size_t> shape) {
    // If the new size is identical to the allocated size then there is no need
    // to re-allocate the buffer.
    if (elemTy == elementType_ && shape == this->dims()) {
      zero();
      return;
    }

    // Delete the old buffer, update the shape, and allocate a new one.
    delete[] data_;
    elementType_ = elemTy;

    assert(shape.size() < max_tensor_dimensions && "Too many indices");
    for (int i = 0, e = shape.size(); i < e; i++) {
      sizes_[i] = shape[i];
    }
    numSizes_ = shape.size();

    if (size()) {
      data_ = new char[size() * getElementSize(elementType_)];
      zero();
    }
  }

  ~Tensor() { delete[] data_; }

  // Move ctor.
  Tensor(Tensor &&other) noexcept {
    std::swap(data_, other.data_);
    for (int i = 0; i < max_tensor_dimensions; i++) {
      std::swap(sizes_[i], other.sizes_[i]);
    }
    std::swap(numSizes_, other.numSizes_);
    std::swap(elementType_, other.elementType_);
  }

  /// Move assignment operator.
  Tensor &operator=(Tensor &&other) noexcept {
    std::swap(data_, other.data_);
    for (int i = 0; i < max_tensor_dimensions; i++) {
      std::swap(sizes_[i], other.sizes_[i]);
    }
    std::swap(numSizes_, other.numSizes_);
    std::swap(elementType_, other.elementType_);
    return *this;
  }

  /// Update the content of the tensor from the tensor \p t.
  void copyFrom(const Tensor *t) {
    reset(t);
    size_t bufferSize = size() * getElementSize(elementType_);
    std::copy(&t->data_[0], &t->data_[bufferSize], data_);
  }

  /// Create a new copy of the current tensor.
  Tensor clone() const {
    Tensor slice;
    slice.copyFrom(this);
    return slice;
  }

  /// \return a new handle that points and manages this tensor.
  template <class ElemTy> Handle<ElemTy> getHandle();
};

/// A class that provides indexed access to a tensor. This class has value
/// semantics and it's copied around. One of the reasons for making this class
/// value semantics is to allow efficient index calculation that the compiler
/// can optimize (because stack allocated structures don't alias).
template <class ElemTy> class Handle final {
  /// A pointer to the tensor that this handle wraps.
  Tensor *tensor_{nullptr};

  /// Contains the multiplication of the sizes from current position to end.
  /// For example, for index (w,z,y,z):  [x * y * z, y * z, z, 1]
  size_t sizeIntegral[max_tensor_dimensions] = {
      0,
  };

  /// Saves the number of dimensions used in the tensor.
  uint8_t numDims{0};

  /// Create a new invalid handle. Notice that this method is private and may
  /// only be used by the static factory method below.
  Handle() {}

public:
  /// Allocate a new invalid handle.
  static Handle createInvalidHandle() { return Handle(); }

  /// \returns true if this Handle points to a valid tensor.
  bool isValid() { return tensor_; }

  /// Calculate the index for a specific element in the tensor. Notice that
  /// the list of indices may be incomplete.
  size_t getElementPtr(ArrayRef<size_t> indices) const {
    assert(indices.size() <= numDims && "Invalid number of indices");
    size_t index = 0;
    for (int i = 0, e = indices.size(); i < e; i++) {
      index += size_t(sizeIntegral[i]) * size_t(indices[i]);
    }

    return index;
  }

  ElemKind getElementType() const { return tensor_->getElementType(); }

  /// Construct a Tensor handle.
  Handle(Tensor *tensor) : tensor_(tensor) {
    auto sizes = tensor->dims();
    numDims = sizes.size();

    /// We allow handles that wrap uninitialized tensors.
    if (!numDims)
      return;

    size_t pi = 1;
    for (int i = numDims - 1; i >= 0; i--) {
      sizeIntegral[i] = pi;
      assert(sizes[i] > 0 && "invalid dim size");
      pi *= sizes[i];
    }

    assert(numDims < max_tensor_dimensions);
  }

  ArrayRef<size_t> dims() const { return tensor_->dims(); }

  size_t size() const { return tensor_->size(); }

  bool isInBounds(ArrayRef<size_t> indices) const {
    return tensor_->isInBounds(indices);
  }

  void clear(ElemTy value = 0) {
    ElemTy *data = tensor_->getRawDataPointer<ElemTy>();
    std::fill(&data[0], &data[0] + size(), value);
  }

  ElemTy &at(ArrayRef<size_t> indices) {
    assert(tensor_->isInBounds(indices));
    size_t index = getElementPtr(indices);
    assert(index < size() && "Out of bounds");
    ElemTy *data = tensor_->getRawDataPointer<ElemTy>();
    return data[index];
  }

  const ElemTy &at(ArrayRef<size_t> indices) const {
    assert(tensor_->isInBounds(indices));
    size_t index = getElementPtr(indices);
    assert(index < size() && "Out of bounds");
    ElemTy *data = tensor_->getRawDataPointer<ElemTy>();
    return data[index];
  }

  /// \returns the element at offset \p idx without any size calculations.
  ElemTy &raw(size_t index) {
    assert(index < size() && "Out of bounds");
    ElemTy *data = tensor_->getRawDataPointer<ElemTy>();
    return data[index];
  }

  /// \returns the element at offset \p idx without any size calculations.
  const ElemTy &raw(size_t index) const {
    assert(index < size() && "Out of bounds");
    ElemTy *data = tensor_->getRawDataPointer<ElemTy>();
    return data[index];
  }

  /// Extract a smaller dimension tensor from a specific slice (that has to be
  /// the first dimention).
  Tensor extractSlice(size_t idx) const {
    auto sizes = tensor_->dims();
    assert(sizes.size() > 1 && "Tensor has only one dimension");
    assert(idx < sizes[0] && "Invalid first index");
    auto elemTy = tensor_->getElementType();
    Tensor slice(elemTy, sizes.drop_front());

    // Extract the whole slice.
    size_t startIdx = sizeIntegral[0] * idx;
    ElemTy *base = tensor_->getRawDataPointer<ElemTy>() + startIdx;
    ElemTy *dest = slice.getRawDataPointer<ElemTy>();
    std::copy(base, base + sizeIntegral[0], dest);

    return slice;
  }

  /// Create a new copy of the current tensor.
  Tensor clone() const { return tensor_->clone(); }

  /// Update the content of the tensor from a literal list:
  void operator=(const std::initializer_list<ElemTy> &vec) {
    assert(numDims == 1 && size() == vec.size() && "Invalid input dimension.");
    size_t i = 0;
    for (auto &e : vec) {
      at({i++}) = e;
    }
  }

  void dumpAscii(const std::string &prefix = "",
                 const std::string &suffix = "\n") const {
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

  /// \returns the index of the highest value.
  size_t maxArg() {
    ElemTy max = at({0});
    size_t idx = 0;

    for (size_t i = 1, e = size(); i < e; i++) {
      ElemTy val = at({i});
      if (val > max) {
        max = val;
        idx = i;
      }
    }
    return idx;
  }

  void dump(const char *title = "", const char *suffix = "") const {
    ElemTy *data = tensor_->getRawDataPointer<ElemTy>();
    ElemTy mx = *std::max_element(&data[0], &data[size()]);
    ElemTy mn = *std::min_element(&data[0], &data[size()]);

    std::cout << title << "max=" << mx << " min=" << mn << " [";
    const unsigned maxNumElem = 100;

    for (size_t i = 0, e = std::min<size_t>(maxNumElem, size()); i < e; i++) {
      std::cout << raw(i) << " ";
    }
    if (size() > maxNumElem)
      std::cout << "...";
    std::cout << "]" << suffix;
  }

  /// Fill the array with random data that's close to zero using the
  /// Xavier method, based on the paper [Bengio and Glorot 2010].
  /// The parameter \p filterSize is the number of elements in the
  /// tensor (or the relevant slice).
  void randomize(size_t filterSize) {
    assert(filterSize > 0 && "invalid filter size");
    double scale = std::sqrt(3.0 / double(filterSize));
    for (size_t i = 0, e = size(); i < e; ++i) {
      raw(i) = (nextRand()) * scale;
    }
  }
};

template <class ElemTy> Handle<ElemTy> Tensor::getHandle() {
  assert(isType<ElemTy>(elementType_) && "Getting a handle to the wrong type.");
  return Handle<ElemTy>(this);
}
}

#endif // NOETHER_TENSOR_H
