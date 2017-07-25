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

const size_t numRandomVals = 256;
const double randomVals[] = {
-0.981991, -0.733214, -0.704263, -0.984158, 0.517615, -0.738007, 0.524421, -0.601964, 0.047695, 0.999512, -0.339222, -0.911430, -0.067872, -0.359755, 0.844661, -0.738533, 0.224015, 0.694785, -0.739039, 0.170157, 0.105868, -0.345744, -0.932887, -0.947770, -0.047427, -0.002140, 0.875544, -0.939250, 0.819466, -0.309533, 0.292451, -0.918860, 0.687559, 0.716084, -0.494552, 0.318470, -0.579904, 0.267810, -0.972332, 0.587063, -0.657390, 0.585997, 0.987321, -0.039642, -0.638447, 0.723706, -0.524794, -0.157153, -0.407816, 0.655535, -0.390811, -0.521940, -0.893004, -0.122017, 0.575254, 0.198286, -0.761496, -0.735506, -0.087803, -0.227935, -0.864146, 0.363330, -0.168149, -0.843409, -0.754643, -0.076368, -0.945290, 0.366013, -0.114093, 0.066547, 0.518849, 0.147862, -0.262949, -0.598044, -0.219493, 0.946378, -0.958793, -0.828576, -0.354654, 0.008741, 0.626689, -0.597730, -0.816853, -0.255521, -0.089036, 0.515779, 0.447339, 0.711988, 0.667613, -0.036305, 0.651188, -0.288772, 0.896038, 0.488395, 0.502128, -0.969253, -0.046955, -0.292526, -0.160472, 0.450525, -0.020671, 0.613926, -0.365147, 0.975952, 0.384599, -0.298788, -0.872368, -0.270895, -0.091759, 0.055893, -0.555349, -0.328804, -0.899422, -0.824385, 0.014023, -0.984444, -0.780202, -0.476256, -0.884158, 0.116227, 0.286294, -0.303481, -0.089657, 0.519086, -0.711514, -0.282537, 0.121333, -0.201142, -0.278549, -0.922929, 0.248745, -0.538505, 0.883216, 0.301459, -0.484838, 0.198640, 0.193514, 0.837423, 0.016131, -0.953761, -0.104244, -0.376533, 0.951303, 0.160106, -0.460908, 0.987657, -0.253311, 0.715590, -0.650328, 0.699711, 0.931105, 0.364898, -0.956418, 0.418641, -0.367462, -0.740716, 0.293547, -0.939628, -0.149344, 0.985460, -0.175155, 0.005300, -0.704140, 0.847257, -0.753456, -0.120007, 0.214978, -0.196294, 0.308593, 0.580077, -0.278203, -0.048437, -0.157467, 0.576055, 0.786530, -0.188660, -0.744840, -0.007893, 0.831526, -0.714723, -0.631638, -0.166030, -0.056025, 0.986427, 0.001773, -0.239409, 0.301672, 0.404480, -0.293915, -0.634783, -0.486739, 0.580556, 0.203404, 0.358966, -0.059268, -0.603391, 0.054441, -0.342792, -0.884390, -0.186063, -0.708729, 0.612555, 0.941979, 0.693281, 0.598256, 0.655929, -0.332426, -0.713050, -0.606292, 0.657182, -0.641860, 0.116920, -0.474681, 0.368251, -0.571900, -0.563118, -0.455302, 0.183313, -0.837671, 0.072575, -0.966025, -0.891857, -0.518136, 0.672081, -0.631185, 0.265990, -0.608464, 0.878815, 0.933892, -0.388463, 0.023277, 0.114464, 0.922902, -0.410735, -0.161594, 0.298198, 0.682150, -0.273686, 0.038156, -0.116367, -0.795731, -0.470632, -0.777406, -0.771270, 0.849493, -0.474714, 0.677850, -0.619856, 0.041747, 0.102378, -0.160826, -0.382957, -0.867881, 0.360130, -0.927582, -0.466619};

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

enum class ElemKind: unsigned char {
  FloatTy,
  DoubleTy,
  Int8Ty,
  Int32Ty,
  IndexTy,
};

template <class ElemTy>
class Handle;

/// A class that represents a contiguous n-dimensional array (a tensor).
class Tensor final {
  /// A pointer to the tensor data.
  char *data_{nullptr};

  /// Contains the dimentions (sizes) of the tensor. Ex: [sx, sy, sz, ...].
  size_t sizes_[max_tensor_dimensions] = {0,};

  /// Contains the number of dimensions used by the tensor.
  unsigned char numSizes_{0};

  /// Specifies the element type of the tensor.
  ElemKind elementType_;

template <class ElemTy>
friend class Handle;

public:
  /// \returns true if the templated parameter \p ElemTy matches the type that's
  /// specified by the parameter \p Ty.
  template <class ElemTy>
  static bool isType(ElemKind Ty) {
    switch (Ty) {
      case ElemKind::FloatTy: return std::is_same<ElemTy, float>::value;
      case ElemKind::DoubleTy: return std::is_same<ElemTy, double>::value;
      case ElemKind::Int8Ty: return std::is_same<ElemTy, int8_t>::value;
      case ElemKind::Int32Ty: return std::is_same<ElemTy, int32_t>::value;
      case ElemKind::IndexTy: return std::is_same<ElemTy, size_t>::value;
    }
  }

  /// \return the size of the element \p Ty.
  static unsigned getElementSize(ElemKind Ty) {
    switch (Ty) {
      case ElemKind::FloatTy: return sizeof(float);
      case ElemKind::DoubleTy: return sizeof(double);
      case ElemKind::Int8Ty: return sizeof(int8_t);
      case ElemKind::Int32Ty: return sizeof(int32_t);
      case ElemKind::IndexTy: return sizeof(size_t);

    }
  }

  /// \return the element type of the tensor.
  ElemKind getElementType() { return elementType_; }

  /// \returns True if the coordinate is within the array.
  bool isInBounds(ArrayRef<size_t> indices) const {
    assert(numSizes_ == indices.size() && "Invalid number of indices");
    for (unsigned i = 0u, e = indices.size(); i < e; i++) {
      if (indices[i] >= sizes_[i]) return false;
    }
    return true;
  }

  /// Set the content of the tensor to zero.
  void zero() {
    std::fill(&data_[0], &data_[0] + size() * getElementSize(elementType_), 0);
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

  /// \returns a pointer to the raw data, of type \p ElemTy.
  template <class ElemTy>
  ElemTy *getRawDataPointer() {
    assert(isType<ElemTy>(elementType_) && "Asking for the wrong ptr type.");
    return reinterpret_cast<ElemTy*>(data_);
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
  Tensor(ElemKind elemTy, ArrayRef<size_t> dims) : data_(nullptr),
  elementType_(elemTy) {
    reset(elemTy, dims);
  }

  Tensor(const Tensor &other) = delete;
  Tensor &operator=(const Tensor &other) = delete;

  /// Assigns a new shape to the tensor and allocates a new buffer.
  void reset(ElemKind elemTy, ArrayRef<size_t> dims) {
    delete[] data_;
    elementType_ = elemTy;

    assert(dims.size() < max_tensor_dimensions && "Too many indices");
    for (int i = 0, e = dims.size(); i < e; i++) { sizes_[i] = dims[i]; }
    numSizes_ = dims.size();

    data_ = new char[size() * getElementSize(elementType_)];
    zero();
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

  /// Create a new copy of the current tensor.
  Tensor clone() {
    Tensor slice(getElementType(), dims());

    // Extract the whole slice.
    size_t bufferSize = size() * getElementSize(elementType_);
    std::copy(&data_[0], &data_[bufferSize], slice.data_);
    return slice;
  }

  /// \return a new handle that points and manages this tensor.
  template <class ElemTy>
  Handle<ElemTy> getHandle();
};


/// A class that provides indexed access to a tensor. This class has value
/// semantics and it's copied around. One of the reasons for making this class
/// value semantics is to allow efficient index calculation that the compiler
/// can optimize (because stack allocated structures don't alias).
template <class ElemTy>
class Handle final {
  /// A pointer to the tensor that this handle wraps.
  Tensor *tensor_{nullptr};

  /// Contains the multiplication of the sizes from current position to end.
  /// For example, for index (w,z,y,z):  [x * y * z, y * z, z, 1]
  size_t sizeIntegral[max_tensor_dimensions] = {0,};

  /// Saves the number of dimensions used in the tensor.
  uint8_t numDims{0};

public:
  /// Calculate the index for a specific element in the tensor. Notice that
  /// the list of indices may be incomplete.
  size_t getElementPtr(ArrayRef<size_t> indices) {
    assert(indices.size() <= numDims && "Invalid number of indices");
    size_t index = 0;
    for (int i = 0, e = indices.size(); i < e; i++) {
      index += size_t(sizeIntegral[i]) * size_t(indices[i]);
    }

    return index;
  }

  /// Construct a Tensor handle.
  Handle(Tensor *tensor) : tensor_(tensor) {
    auto sizes = tensor->dims();
    numDims = sizes.size();

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
  Tensor extractSlice(size_t idx) {
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
  Tensor clone() {
    return tensor_->clone();
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

  void dump(const std::string &title = "", const std::string &suffix = "") {
    ElemTy *data = tensor_->getRawDataPointer<ElemTy>();
    ElemTy mx = *std::max_element(&data[0], &data[size()]);
    ElemTy mn = *std::min_element(&data[0], &data[size()]);

    std::cout << title << "max=" << mx << " min=" << mn << " [";

    for (size_t i = 0, e = std::min<size_t>(400, size()); i < e; i++) {
      std::cout << at(i) << " ";
    }
    std::cout << "]" << suffix;
  }

  /// Fill the array with random data that's close to zero using the
  /// Xavier method, based on the paper [Bengio and Glorot 2010].
  /// The parameter \p filterSize is the number of elements in the
  /// tensor (or the relevant slice).
  void randomize(size_t filterSize) {
    // This is a global variable, a singleton, that's used as an index into
    // the array with random numbers.
    static int offset = 0;

    double scale = std::sqrt(3.0/double(filterSize));
    for (size_t i = 0, e = size(); i < e; ++i) {
      raw(i) = (randomVals[(offset++) % numRandomVals]) * scale;
    }

    offset++;
  }

};

template <class ElemTy>
Handle<ElemTy> Tensor::getHandle() {
  assert(isType<ElemTy>(elementType_) && "Getting a handle to the wrong type.");
  return Handle<ElemTy>(this);
}

}

#endif // NOETHER_TENSOR_H
