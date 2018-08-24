/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GLOW_BASE_TENSOR_H
#define GLOW_BASE_TENSOR_H

#include <cassert>
#include <vector>

#include "glow/Base/Type.h"
#include "glow/Support/Compiler.h"
#include "glow/Support/Memory.h"
#include "glow/Support/Random.h"

#include "llvm/ADT/ArrayRef.h"

namespace glow {

//===----------------------------------------------------------------------===//
//                               Tensor
//===----------------------------------------------------------------------===//

template <class ElemTy> class Handle;

class Tensor;

void genericTranspose(Tensor *src, Tensor *dest,
                      llvm::ArrayRef<unsigned_t> shuffle);

/// Helper function that \returns a ShapeVector of those dimensions in \p
/// currDims expanded with dimension = 1 until the maximum tensor dimension is
/// reached. The number of elements in the input dims is the same as in the
/// returned dims. For example, input {2,1,4} would result in {2,1,4,1,1,1}.
ShapeVector expandDimsToMax(llvm::ArrayRef<size_t> currDims);

/// A class that represents a contiguous n-dimensional array (a tensor).
class Tensor final {
public:
  /// Specifies the kind initialization for the tensor.
  enum class InitKind {
    Zero,      // The tensor is initialized to zero.
    Broadcast, // Broadcast a single value to all elements.
    Xavier,    // Init the variable with random values using the Xavier method.
  };

private:
  /// A pointer to the tensor data.
  char *data_{nullptr};

  /// The type of the tensor.
  Type type_;

  /// If the tensor is unowned.
  bool isUnowned_{false};

  template <class ElemTy> friend class Handle;

  /// \returns a pointer to the tensor data buffer.
  char *getData() const { return data_; }

  /// \returns true if it is an unowned tensor.
  bool isUnowned() const { return isUnowned_; }

public:
  /// \returns the type of the tensor.
  const Type &getType() const { return type_; }

  /// \return the element type of the tensor.
  ElemKind getElementType() const { return type_.getElementType(); }

  /// \returns True if the coordinate is within the array.
  bool isInBounds(llvm::ArrayRef<size_t> indices) const {
    assert(type_.numSizes_ == indices.size() && "Invalid number of indices");
    for (size_t i = 0u, e = indices.size(); i < e; i++) {
      if (indices[i] >= type_.sizes_[i]) {
        return false;
      }
    }
    return true;
  }

  /// Set the content of the tensor to zero.
  void zero() {
    std::fill(&getData()[0], &getData()[0] + size() * type_.getElementSize(),
              0);
  }

  /// \returns the shape of the tensor.
  llvm::ArrayRef<size_t> dims() const { return type_.dims(); }

  /// \returns the number of elements in the tensor.
  size_t size() const { return type_.size(); }

  /// Initialize an empty tensor.
  Tensor() = default;

  /// Initialize from a list of float literals.
  Tensor(const std::initializer_list<float> &vec) {
    reset(ElemKind::FloatTy, {vec.size()});
    auto *data = getRawDataPointer<float>();
    int i = 0;
    for (auto &f : vec) {
      data[i++] = f;
    }
  }

  /// Allocate and initialize a new tensor.
  explicit Tensor(TypeRef ty) : data_(nullptr), type_(*ty), isUnowned_{false} {
    reset(*ty);
  }

  /// Allocate and initialize a new tensor.
  explicit Tensor(const Type &ty)
      : data_(nullptr), type_(ty), isUnowned_{false} {
    reset(ty);
  }

  /// Allocate and initialize a float new tensor.
  Tensor(ElemKind elemTy, llvm::ArrayRef<size_t> dims)
      : data_(nullptr), type_(elemTy, dims), isUnowned_{false} {
    reset(elemTy, dims);
  }

  /// Construct an unowned tensor provided an existing payload buffer.
  /// This constructor can be used when there is a need to work with
  /// "externally" managed payload buffers using Tensor APIs.
  Tensor(void *data, TypeRef ty)
      : data_(reinterpret_cast<char *>(data)), type_(*ty), isUnowned_{false} {
    // Mark as unowned.
    isUnowned_ = true;
  }

  /// Allocate and initialize a new integer tensor with \p scale and \p offset.
  Tensor(ElemKind elemTy, llvm::ArrayRef<size_t> dims, float scale,
         int32_t offset)
      : data_(nullptr), type_(elemTy, dims, scale, offset), isUnowned_{false} {
    reset(type_);
  }

  Tensor(const Tensor &other) = delete;
  Tensor &operator=(const Tensor &other) = delete;

  /// Initialize the content of the tensor using the \p init method. The value
  /// \p val is the initialization parameter. \p PRNG is used to generate
  /// random numbers.
  void init(InitKind init, float val, PseudoRNG &PRNG);

  /// \returns unowned tensor using the same data buffer as the current tensor
  /// but having different dimensions \p dims. \p offsets represents an optional
  /// offset into the tensor representing the location of the first element to
  /// start a subview from. The returned unonwed tensor is essentially a
  /// different view or subview on the same data.
  ///
  /// The lifetime of the returned unowned tensor should be always within
  /// the lifetime of its parent tensor, i.e. the unowned tensor should not
  /// outlive its parent tensor.
  Tensor getUnowned(llvm::ArrayRef<size_t> dims,
                    llvm::ArrayRef<size_t> offsets = {}) const {
    Tensor unownedTensor;

    auto *firstElemPtr = getData();
    if (offsets.size()) {
      assert(offsets.size() == this->dims().size() &&
             "Number of dims of tensor must equal number of dims in offsets");
      // Find the index of the first element and use it to find the pointer to
      // the first element.
      size_t index = 0, pi = 1;
      for (int i = this->dims().size() - 1; i >= 0; i--) {
        index += pi * offsets[i];
        pi *= this->dims()[i];
      }
      firstElemPtr = &firstElemPtr[index * type_.getElementSize()];
    }

    unownedTensor.data_ = firstElemPtr;
    unownedTensor.isUnowned_ = true;
    unownedTensor.type_ = Type::newShape(getType(), dims);
    if (offsets.size() == 0) {
      assert(size() == unownedTensor.size() && "The size of the unowned tensor "
                                               "should the same as the size of "
                                               "the original tensor");

    } else {
      assert(size() >= unownedTensor.size() && "The size of the unowned tensor "
                                               "should be no greater than the "
                                               "size of the original tensor");
    }
    return unownedTensor;
  }

  /// Reset the shape and type of this tensor to match the shape and type of
  /// \p other.
  void reset(const Tensor *other) { reset(other->getType()); }

  void reset(ElemKind elemTy, llvm::ArrayRef<size_t> shape) {
    Type t(elemTy, shape);
    reset(t);
  }

  void reset(ElemKind elemTy, llvm::ArrayRef<size_t> shape, float scale,
             int32_t offset) {
    Type t(elemTy, shape, scale, offset);
    reset(t);
  }

  /// Assigns a new shape to the tensor and allocates a new buffer.
  void reset(const Type &T) {
    // If the new size is identical to the allocated size then there is no need
    // to re-allocate the buffer.
    if (type_ == T && getData()) {
      zero();
      return;
    }

    // Delete the old buffer, update the shape, and allocate a new one.
    if (!isUnowned())
      alignedFree(getData());
    type_ = T;

    // We are allocating memory specifically for this tensor, thus, it owns it.
    isUnowned_ = false;

    // Note: zero-dimensional tensors have size 1.
    assert(size() > 0 && "Tensors must always have positive size.");
    size_t count = size() * type_.getElementSize();
    data_ = reinterpret_cast<char *>(alignedAlloc(count, TensorAlignment));
    zero();
  }

  ~Tensor() {
    if (!isUnowned()) {
      alignedFree(getData());
    }
  }

  // Move ctor.
  Tensor(Tensor &&other) noexcept {
    std::swap(data_, other.data_);
    std::swap(type_, other.type_);
    std::swap(isUnowned_, other.isUnowned_);
  }

  /// Move assignment operator.
  Tensor &operator=(Tensor &&other) noexcept {
    std::swap(data_, other.data_);
    std::swap(type_, other.type_);
    std::swap(isUnowned_, other.isUnowned_);
    return *this;
  }

  /// \returns true if the content of the other tensor \p other is identical to
  /// this one.
  bool isEqual(const Tensor &other, float allowedError = 0.0001) const {
    if (other.dims() != dims()) {
      return false;
    }

    switch (getElementType()) {
    case ElemKind::FloatTy:
      return isEqualImpl<float>(other, allowedError);
    case ElemKind::Int8QTy:
      assert(getType().getScale() == other.getType().getScale() &&
             "Scales must match.");
      assert(getType().getOffset() == other.getType().getOffset() &&
             "Offsets must match.");
      return isEqualImpl<int8_t>(other, allowedError);
    case ElemKind::Int32QTy:
      return isEqualImpl<int32_t>(other, allowedError);
    case ElemKind::Int64ITy:
      return isEqualImpl<int64_t>(other, allowedError);
    }

    // This is to make compiler happy. It can never reach this point as switch
    // always covers all possible values.
    llvm_unreachable("unreachable");
  }

  /// Update the content and type of the tensor from the tensor \p t.
  void assign(const Tensor *t) {
    assert(this != t && "Copying to self");
    reset(t);
    size_t bufferSize = size() * type_.getElementSize();
    std::copy(&t->getData()[0], &t->getData()[bufferSize], getData());
  }

  /// Update the raw data of the tensor from the tensor \p t.
  void copyRawFrom(const Tensor *t) {
    assert(this != t && "Copying to self");
    assert(size() == t->size());
    assert(getElementType() == t->getElementType() && "Invalid element type");
    size_t bufferSize = size() * type_.getElementSize();
    std::copy(&t->getData()[0], &t->getData()[bufferSize], getData());
  }

  /// Update the content of the tensor with a slice from tensor \p t. A slice
  /// is one index from the first dimension of the tensor.
  void copySlice(const Tensor *t, size_t slice) {
    auto dim = t->dims().slice(1);
    (void)dim;
    assert(dim == dims() && "Invalid slice size");
    assert(getElementType() == t->getElementType() && "Invalid element type");

    size_t bufferSize = size() * type_.getElementSize();
    std::copy(&t->getData()[bufferSize * slice],
              &t->getData()[bufferSize * (slice + 1)], getData());
  }

  /// Update the content of the tensor with a sequence of slices from the
  /// tensor \p t. A slice is one index from the first dimension of the tensor.
  /// The copying operation may overlap the end of the tensor \p t one or more
  /// times. This means that the data in the input tensor may be duplicated.
  void copyConsecutiveSlices(const Tensor *t, size_t startSliceIdx) {
    auto onceSliceDim = t->dims().slice(1);
    (void)onceSliceDim;
    assert(onceSliceDim == dims().slice(1) && "Invalid slice size");
    assert(getElementType() == t->getElementType() && "Invalid element type");
    assert(dims().size() > 1 && "Tensor must contain at least two dimensions");

    size_t numSlicesInInput = t->dims()[0];
    size_t numElementsInSlice = size() / dims()[0];
    size_t bufferSize = numElementsInSlice * type_.getElementSize();

    // For each outer slice in the current tensor:
    for (size_t n = 0, e = dims()[0]; n < e; n++) {
      size_t startIdx = (startSliceIdx + n) % numSlicesInInput;
      std::copy(&t->getData()[bufferSize * startIdx],
                &t->getData()[bufferSize * (startIdx + 1)],
                &getData()[bufferSize * n]);
    }
  }

  /// Transpose the tensor \p src into the empty tensor \p dest. Shuffle the
  /// axis based on the list \p shuffle, where each element is the src index.
  void transpose(Tensor *dest, llvm::ArrayRef<unsigned_t> shuffle) {
    genericTranspose(this, dest, shuffle);
  }

  /// Create a new copy of the current tensor.
  Tensor clone() const {
    Tensor slice;
    slice.assign(this);
    return slice;
  }

  /// Return the raw unsafe pointer to the tensor payload.
  char *getUnsafePtr() const { return getData(); }

  /// \return a new handle that points and manages this tensor.
  template <class ElemTy = float> Handle<ElemTy> getHandle();

private:
  /// \returns a pointer to the raw data, of type \p ElemTy.
  template <class ElemTy> ElemTy *getRawDataPointer() {
    assert(type_.isType<ElemTy>() && "Asking for the wrong ptr type.");
    return reinterpret_cast<ElemTy *>(data_);
  }

  /// \returns a const pointer to the raw data, of type \p ElemTy.
  template <class ElemTy> const ElemTy *getRawDataPointer() const {
    assert(type_.isType<ElemTy>() && "Asking for the wrong ptr type.");
    return reinterpret_cast<const ElemTy *>(data_);
  }

  template <class ElemTy>
  bool isEqualImpl(const Tensor &other, float allowedError) const {
    auto const *myData = getRawDataPointer<ElemTy>();
    auto const *otherData = other.getRawDataPointer<ElemTy>();
    for (size_t i = 0, e = size(); i < e; i++) {
      double delta = myData[i] - otherData[i];
      if (std::abs(delta) > allowedError) {
        return false;
      }
    }
    return true;
  }
};

//===----------------------------------------------------------------------===//
//                    Tensor Handle
//===----------------------------------------------------------------------===//

void dumpAsciiImpl(Tensor *T, llvm::raw_ostream &os);
void dumpAsciiImpl(Tensor *T);

void dumpImpl(Tensor *T, llvm::raw_ostream &os);
void dumpImpl(Tensor *T);

/// A class that provides indexed access to a tensor. This class has value
/// semantics and it's copied around. One of the reasons for making this class
/// value semantics is to allow efficient index calculation that the compiler
/// can optimize (because stack allocated structures don't alias).
template <class ElemTy> class Handle final {
  /// A pointer to the tensor that this handle wraps.
  Tensor *tensor_{nullptr};

  /// Contains the multiplication of the sizes from current position to end.
  /// For example, for index (w,z,y,z):  [x * y * z, y * z, z, 1]
  size_t sizeIntegral_[max_tensor_dimensions] = {
      0,
  };

  size_t sizes_[max_tensor_dimensions] = {
      0,
  };

  /// Saves the number of dimensions used in the tensor.
  uint8_t numDims_{0};

  /// Create a new invalid handle. Notice that this method is private and may
  /// only be used by the static factory method below.
  Handle() = default;

public:
  /// Allocate a new invalid handle.
  static Handle createInvalidHandle() { return Handle(); }

  /// \returns true if this Handle points to a valid tensor.
  bool isValid() const { return tensor_; }

  /// Calculate the index for a specific element in the tensor. Notice that
  /// the list of indices may be incomplete.
  size_t getElementPtr(llvm::ArrayRef<size_t> indices) const {
    assert(indices.size() <= numDims_ && "Invalid number of indices");
    // The loop below can be rewritten using std::inner_product. Unfortunately
    // std::inner_product does not optimize very well and loops that use this
    // method don't get vectorized. Don't change this loop without benchmarking
    // the program on a few compilers.
    size_t index = 0;
    for (size_t i = 0, e = indices.size(); i < e; i++) {
      index += size_t(sizeIntegral_[i]) * size_t(indices[i]);
    }

    return index;
  }

  /// \returns the value of the n'th dimension \p dim, for the raw index \p idx.
  size_t getDimForPtr(size_t dim, size_t idx) const {
    assert(dim < numDims_ && "Invalid dimension");
    auto R = idx / sizeIntegral_[dim];
    return R % sizes_[dim];
  }

  /// \returns the type of the tensor.
  const Type &getType() const { return tensor_->getType(); }

  /// \returns the element type of the tensor.
  ElemKind getElementType() const { return tensor_->getElementType(); }

  /// Construct a Tensor handle.
  explicit Handle(Tensor *tensor) : tensor_(tensor) {
    auto sizes = tensor->dims();
    numDims_ = sizes.size();

    /// We allow handles that wrap uninitialized tensors.
    if (!numDims_) {
      return;
    }

    // Copy the sizes of the tensor.
    memcpy(sizes_, tensor_->type_.sizes_,
           max_tensor_dimensions * sizeof(sizes_[0]));

    size_t pi = 1;
    for (int i = numDims_ - 1; i >= 0; i--) {
      sizeIntegral_[i] = pi;
      assert(sizes_[i] > 0 && "invalid dim size");
      pi *= sizes_[i];
    }

    assert(numDims_ <= max_tensor_dimensions && "Too many dimensions.");
  }

  llvm::ArrayRef<size_t> dims() const {
    return llvm::ArrayRef<size_t>(sizes_, numDims_);
  }

  /// \returns the number of elements in the whole tensor.
  size_t size() const { return tensor_->size(); }

  bool isInBounds(llvm::ArrayRef<size_t> indices) const {
    return tensor_->isInBounds(indices);
  }

  void clear(ElemTy value = 0) {
    auto *data = tensor_->getRawDataPointer<ElemTy>();
    std::fill(&data[0], &data[0] + size(), value);
  }

  ElemTy &at(llvm::ArrayRef<size_t> indices) {
    assert(tensor_->isInBounds(indices));
    size_t index = getElementPtr(indices);
    assert(index < size() && "Out of bounds");
    auto *data = tensor_->getRawDataPointer<ElemTy>();
    return data[index];
  }

  const ElemTy &at(llvm::ArrayRef<size_t> indices) const {
    assert(tensor_->isInBounds(indices));
    size_t index = getElementPtr(indices);
    assert(index < size() && "Out of bounds");
    auto *data = tensor_->getRawDataPointer<ElemTy>();
    return data[index];
  }

  /// \returns the element at offset \p idx without any size calculations.
  ElemTy &raw(size_t index) {
    assert(index < size() && "Out of bounds");
    auto *data = tensor_->getRawDataPointer<ElemTy>();
    return data[index];
  }

  /// \returns the element at offset \p idx without any size calculations.
  const ElemTy &raw(size_t index) const {
    assert(index < size() && "Out of bounds");
    auto *data = tensor_->getRawDataPointer<ElemTy>();
    return data[index];
  }

  /// Extract a smaller dimension tensor from a specific slice (that has to be
  /// the first dimension).
  Tensor extractSlice(size_t idx) const {
    auto sizes = tensor_->dims();
    assert(sizes.size() > 1 && "Tensor must have at least two dimensions");
    assert(idx < sizes[0] && "Invalid first index");

    Tensor slice{Type::newShape(tensor_->getType(), sizes.slice(1))};

    // Extract the whole slice.
    size_t startIdx = sizeIntegral_[0] * idx;
    ElemTy *base = tensor_->getRawDataPointer<ElemTy>() + startIdx;
    auto *dest = slice.getRawDataPointer<ElemTy>();
    std::copy(base, base + sizeIntegral_[0], dest);

    return slice;
  }

  /// Create a new copy of the current tensor.
  Tensor clone() const { return tensor_->clone(); }

  /// Update the content of the tensor from a literal list:
  void operator=(const std::initializer_list<ElemTy> &vec) {
    assert(size() == vec.size() && "Invalid input size.");
    size_t i = 0;
    for (auto &e : vec) {
      raw(i++) = e;
    }
  }

  void operator=(llvm::ArrayRef<ElemTy> array) {
    assert(size() == array.size() && "Invalid input size.");
    for (size_t i = 0, e = array.size(); i < e; ++i) {
      raw(i) = array[i];
    }
  }

  void dumpAscii(llvm::raw_ostream &os) const { dumpAsciiImpl(tensor_, os); }
  void dumpAscii() const { dumpAsciiImpl(tensor_); }

  /// \returns the raw indices of a min and max values from the tensor.
  /// In case of multiple min or max, the smallest index is returned.
  std::pair<size_t, size_t> minMaxArg() const {
    ElemTy max = raw(0);
    ElemTy min = raw(0);

    size_t maxIdx = 0;
    size_t minIdx = 0;

    for (size_t i = 1, e = size(); i < e; i++) {
      ElemTy val = raw(i);
      if (val > max) {
        max = val;
        maxIdx = i;
      } else if (val < min) {
        min = val;
        minIdx = i;
      }
    }

    return std::make_pair(minIdx, maxIdx);
  }

  /// \returns true if tensor contains only elements equal to zero.
  bool isZero() const {
    for (size_t i = 0, e = size(); i < e; i++) {
      if (raw(i) != 0)
        return false;
    }

    return true;
  }

  void dump(llvm::raw_ostream &os) const { dumpImpl(tensor_, os); }
  void dump() const { dumpImpl(tensor_); }

  /// Fill the array with random data that's close to zero using the
  /// Xavier method, based on the paper [Bengio and Glorot 2010].
  /// This type of initialization facilitates better training performance.
  /// The parameter \p filterSize is the number of "input" neurons in the
  /// tensor (or the relevant slice). For example, consider case of MatMul:
  /// NxM (\p input) * MxK (\p weights) == NxK (\p result)
  /// Correct \p filterSize for weights tensor is M, so that norm for each
  /// row of \p input equals to norm of corresponding row of \p result.
  void initXavier(size_t filterSize, PseudoRNG &PRNG) {
    assert(filterSize > 0 && "invalid filter size");
    double scale = std::sqrt(3.0 / double(filterSize));
    std::uniform_real_distribution<> dist(-scale, scale);
    for (size_t i = 0, e = size(); i < e; i++) {
      raw(i) = dist(PRNG);
    }
  }

  /// Fill the tensor with uniformly distributed values in the range
  /// [low .. high].
  template <typename T = ElemTy>
  typename std::enable_if<std::is_floating_point<T>::value>::type
  randomize(float low, float high, PseudoRNG &PRNG) {
    assert(low < high && "invalid range");
    std::uniform_real_distribution<ElemTy> dist(low, high);
    for (size_t i = 0, e = size(); i < e; i++) {
      raw(i) = dist(PRNG);
    }
  }

  /// Fill the tensor with uniformly distributed values in the range
  /// [low .. high].
  template <typename T = ElemTy>
  typename std::enable_if<std::is_integral<T>::value>::type
  randomize(int low, int high, PseudoRNG &PRNG) {
    assert(low < high && "invalid range");
    std::uniform_int_distribution<int> dist(low, high);
    for (size_t i = 0, e = size(); i < e; i++) {
      raw(i) = dist(PRNG);
    }
  }

  /// \returns the mean and variance of the tensor.
  std::pair<double, double> calculateMeanVariance() const {
    size_t n = size();
    assert(n > 1 && "Input must have at least 2 elements.");

    // Calculate mean.
    double mean = 0;
    for (size_t i = 0; i < n; i++) {
      mean += raw({i});
    }
    mean /= n;

    // Calculate variance.
    double var = 0;
    for (size_t i = 0; i < n; i++) {
      double t = raw({i}) - mean;
      var += t * t;
    }
    var /= (n - 1);

    return {mean, var};
  }

  /// Insert the tensor \p slice at location \p offset \p count times along the
  /// \p axis. This operation is equivalent to the operation of scanning the
  /// source tensor, and saving the value that is stored at coordinate {d_0,
  /// d_1, ... d_n} in the new tensor at {d_0 + O_0, d_1 + O_1, ... d_n + O_n},
  /// where O is the offset vector, assuming \p count = 1. For \p count > 1, the
  /// same Tensor is copied \p count times along the provided \p axis. The
  /// tensors must be of the right dimensions.
  void insertTensors(Handle<ElemTy> &slice, llvm::ArrayRef<size_t> offset,
                     size_t count = 1, size_t axis = 0) {
    auto sliceCoor = slice.dims().vec();
    auto fusedCoor = dims().vec();
    insertTensorsImpl(sliceCoor, fusedCoor, slice, true, offset, count, axis,
                      0);
  }

  /// Extract the tensor \p slice at location \p offset. This operation is
  /// equivalent to the operation of scanning the destination tensor, and
  /// copying into the cell at coordinate {d_0, d_1, ... d_n} a value from the
  /// tensor at {d_0 + O_0, d_1 + O_1, ... d_n + O_n}, where O is the offset
  /// vector. The tensors must be of the right dimensions.
  void extractTensors(Handle<ElemTy> &slice, llvm::ArrayRef<size_t> offset) {
    auto sliceCoor = slice.dims().vec();
    auto fusedCoor = dims().vec();
    insertTensorsImpl(sliceCoor, fusedCoor, slice, false, offset, /* count */ 1,
                      /* axis */ 0, 0);
  }

private:
  /// Concats or splits tensors.
  /// This method concats or extracts a slice from a tensor.
  /// \p sliceCoor and \p fusedCoor are temporary storage that the function uses
  /// to construct the coordinates to access the tensor. They must be
  /// initialized to be the size of the shape of the tensor. \p slice and \p
  /// fused are the tensors to concat or extract. \p offset is the offset of the
  /// slice to add or extract along the dimension \p offsetDim. \p d is the
  /// recursion depth parameter that's following the number of the axis. if \p
  /// isInsert is set then data is copied from \p slice to \p fused. Otherwise
  /// data is copied from \p fused to \p slice. \p count and \p axis are used in
  /// conjunction for inserting the same tensor \p count times along the \p
  /// axis.
  void insertTensorsImpl(llvm::MutableArrayRef<size_t> sliceCoor,
                         llvm::MutableArrayRef<size_t> fusedCoor,
                         Handle<ElemTy> &slice, bool isInsert,
                         llvm::ArrayRef<size_t> offset, size_t count,
                         size_t axis, unsigned d) {
    bool isDone = (d == slice.dims().size());

    if (isDone) {
      if (isInsert) {
        at(fusedCoor) = slice.at(sliceCoor);
      } else {
        slice.at(sliceCoor) = at(fusedCoor);
      }
      return;
    }

    // Only need to iterate over count if the current dimension d is equal to
    // the axis we're inserting over.
    const size_t countIters = (axis == d) ? count : 1;
    for (size_t c = 0; c < countIters; c++) {
      for (size_t i = 0, e = slice.dims()[d]; i < e; i++) {
        // Construct the coordinates for the slice and for the joint shape.
        // Add the 'offset' to the dimension that we concat the shapes on.
        sliceCoor[d] = i;
        // If this is the correct axis to insert multiple times then calcuate
        // the additional offset to use.
        const size_t countAxisOffset = (axis == d) ? c * slice.dims()[d] : 0;
        fusedCoor[d] = i + offset[d] + countAxisOffset;
        insertTensorsImpl(sliceCoor, fusedCoor, slice, isInsert, offset, count,
                          axis, d + 1);
      }
    }
  }
};

template <class ElemTy> Handle<ElemTy> Tensor::getHandle() {
  assert(type_.isType<ElemTy>() && "Getting a handle to the wrong type.");
  return Handle<ElemTy>(this);
}

} // namespace glow

#endif // GLOW_BASE_TENSOR_H
