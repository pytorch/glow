#ifndef GLOW_BASE_TENSOR_H
#define GLOW_BASE_TENSOR_H

#include <cassert>
#include <vector>

#include "glow/Base/Type.h"
#include "glow/Support/Compiler.h"
#include "glow/Support/Random.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/PointerUnion.h"

namespace glow {

//===----------------------------------------------------------------------===//
//                               Tensor
//===----------------------------------------------------------------------===//

template <class ElemTy> class Handle;

/// A class that represents a contiguous n-dimensional array (a tensor).
class Tensor final {
  /// A pointer to the tensor data and the unowned flag.
  llvm::PointerIntPair<void *, 1, bool> data_{nullptr};

  /// The type of the tensor.
  Type type_;

  template <class ElemTy> friend class Handle;

  /// \returns a pointer to the tensor data buffer.
  char *getData() const { return reinterpret_cast<char *>(data_.getPointer()); }

  /// \returns true if it is an unowned tensor.
  bool isUnowned() const { return data_.getInt(); }

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

  /// \returns a pointer to the raw data, of type \p ElemTy.
  template <class ElemTy> ElemTy *getRawDataPointer() {
    assert(type_.isType<ElemTy>() && "Asking for the wrong ptr type.");
    return reinterpret_cast<ElemTy *>(data_.getPointer());
  }

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
  explicit Tensor(TypeRef ty) : data_(nullptr), type_(*ty) { reset(*ty); }

  /// Allocate and initialize a new tensor.
  explicit Tensor(const Type &ty) : data_(nullptr), type_(ty) { reset(ty); }

  /// Allocate and initialize a new tensor.
  Tensor(ElemKind elemTy, llvm::ArrayRef<size_t> dims)
      : data_(nullptr), type_(elemTy, dims) {
    reset(elemTy, dims);
  }

  Tensor(const Tensor &other) = delete;
  Tensor &operator=(const Tensor &other) = delete;

  /// \returns unowned tensor using the same data buffer as the current tensor
  /// but having different dimensions. This is essentially a different
  /// view on the same data.
  ///
  /// The lifetime of the returned unowned tensor should be always within
  /// the lifetime of its parent tensor, i.e. the unowned tensor should not
  /// outlive its parent tensor.
  ///
  /// TODO: Add mechanims like RC (reference counting) to ensure that unowned
  /// tensors do not outlive their parent tensors.
  Tensor getUnowned(llvm::ArrayRef<size_t> dims) const {
    Tensor unownedTensor;
    unownedTensor.data_.setPointer(getData());
    unownedTensor.data_.setInt(1);
    unownedTensor.type_ = Type(getElementType(), dims);
    assert(size() == unownedTensor.size() &&
           "The size of the non-owned tensor should be "
           "the same as the size of the original "
           "tensor");
    return unownedTensor;
  }

  /// Reset the shape and type of this tensor to match the shape and type of
  /// \p other.
  void reset(const Tensor *other) {
    reset(other->getElementType(), other->dims());
  }

  void reset(ElemKind elemTy, llvm::ArrayRef<size_t> shape) {
    Type t(elemTy, shape);
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
      delete[] getData();
    type_ = T;

    if (size()) {
      data_.setPointer(new char[size() * type_.getElementSize()]);
      zero();
    }
  }

  ~Tensor() {
    if (!isUnowned())
      delete[] getData();
  }

  // Move ctor.
  Tensor(Tensor &&other) noexcept {
    std::swap(data_, other.data_);
    std::swap(type_, type_);
  }

  /// Move assignment operator.
  Tensor &operator=(Tensor &&other) noexcept {
    std::swap(data_, other.data_);
    std::swap(type_, other.type_);
    return *this;
  }

  /// Update the content of the tensor from the tensor \p t.
  void copyFrom(const Tensor *t) {
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

  /// Create a new copy of the current tensor.
  Tensor clone() const {
    Tensor slice;
    slice.copyFrom(this);
    return slice;
  }

  /// Return the raw unsafe pointer to the tensor payload.
  char *getUnsafePtr() const { return getData(); }

  /// \return a new handle that points and manages this tensor.
  template <class ElemTy = float> Handle<ElemTy> getHandle();
};

//===----------------------------------------------------------------------===//
//                    Tensor Handle
//===----------------------------------------------------------------------===//

void dumpAsciiImpl(Tensor *T);

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
  size_t sizeIntegral[max_tensor_dimensions] = {
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
      index += size_t(sizeIntegral[i]) * size_t(indices[i]);
    }

    return index;
  }

  /// \returns the value of the n'th dimension \p dim, for the raw index \p idx.
  size_t getDimForPtr(size_t dim, size_t idx) const {
    assert(dim < numDims_ && "Invalid dimension");
    auto R = idx / sizeIntegral[dim];
    return R % sizes_[dim];
  }

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
      sizeIntegral[i] = pi;
      assert(sizes_[i] > 0 && "invalid dim size");
      pi *= sizes_[i];
    }

    assert(numDims_ < max_tensor_dimensions);
  }

  llvm::ArrayRef<size_t> dims() const {
    return llvm::ArrayRef<size_t>(sizes_, numDims_);
  }

  /// \returns the number of elements in the whole tensor.
  size_t size() const { return tensor_->size(); }

  /// \returns the number of elements in a slice for a specific dimension.
  /// For a tensor of dimensions (w, x,y,z) the result for each value of \p
  /// dimIdx would be [x * y * z, y * z, z, 1]. This means that each element
  /// in the n-th dimension is made of tensors with n-1 dimensions, and this
  /// function returns the size of that tensor.
  size_t sliceSize(unsigned dimIdx) const {
    assert(dimIdx < max_tensor_dimensions);
    return sizeIntegral[dimIdx];
  }

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
    assert(sizes.size() > 1 && "Tensor has only one dimension");
    assert(idx < sizes[0] && "Invalid first index");
    auto elemTy = tensor_->getElementType();
    Tensor slice(elemTy, sizes.slice(1));

    // Extract the whole slice.
    size_t startIdx = sizeIntegral[0] * idx;
    ElemTy *base = tensor_->getRawDataPointer<ElemTy>() + startIdx;
    auto *dest = slice.getRawDataPointer<ElemTy>();
    std::copy(base, base + sizeIntegral[0], dest);

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

  void dumpAscii() const { dumpAsciiImpl(tensor_); }

  /// \returns the index of the highest value.
  /// \returns the lowest index of a max element if there are multiple
  /// elements with the same max value.
  size_t maxArg() const {
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

  /// \returns true if tensor contains only elements equal to zero.
  bool isZero() const {
    for (size_t i = 0, e = size(); i < e; ++i) {
      if (raw(i) != 0)
        return false;
    }

    return true;
  }

  void dump() const { dumpImpl(tensor_); }

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

  /// \returns the mean and variance of the tensor.
  std::pair<ElemTy, ElemTy> calculateMeanVariance() const {
    size_t n = size();
    assert(n > 1 && "Input must haev at least 2 elements.");

    // Calculate mean.
    ElemTy sum = 0;
    for (size_t i = 0; i < n; i++) {
      sum += raw({i});
    }

    ElemTy mean = sum / n;

    // Calculate variance.
    ElemTy sigma = 0;
    for (size_t i = 0; i < n; i++) {
      ElemTy t = (raw({i}) - mean);
      sigma += t * t;
    }

    ElemTy variance = sigma / (n - 1);
    return {mean, variance};
  }

  /// Transpose the tensor \p src into the empty tensor \p dest. Shuffle the
  /// axis based on the list \p shuffle, where each element is the src index.
  void transpose(Tensor *dest, llvm::ArrayRef<unsigned> shuffle) {
    assert(dims().size() == shuffle.size() && "Invalid dimensions");

    size_t newSizes[max_tensor_dimensions];

    // Generate the swizzled dimensions.
    auto origDims = dims();
    for (unsigned i = 0; i < numDims_; i++) {
      newSizes[i] = origDims[shuffle[i]];
    }

    // Resize the tensor to the transposed shape.
    dest->reset(getElementType(), llvm::ArrayRef<size_t>(newSizes, numDims_));

    size_t srcCoor[max_tensor_dimensions];
    size_t destCoor[max_tensor_dimensions];

    auto DH = dest->getHandle<ElemTy>();
    transposeImpl(DH, srcCoor, destCoor, shuffle, 0);
  }

  /// \returns true if the content of the other handle \p other is identical to
  /// this one.
  bool isEqual(Handle<ElemTy> &other, float allowedError = 0.0001) const {
    if (other.dims() != dims()) {
      return false;
    }

    ElemTy sumErr = 0;
    for (size_t i = 0, e = size(); i < e; i++) {
      ElemTy delta = raw(i) - other.raw(i);
      sumErr += delta * delta;
    }

    return sumErr < allowedError;
  }

  /// Insert the tensor \p slice at location \p offset. This operation is
  /// equivalent to the operation of scanning the source tensor, and saving
  /// the value that is stored at coordinate {d_0, d_1, ... d_n} in the new
  /// tensor at {d_0 + O_0, d_1 + O_1, ... d_n + O_n}, where O is the offset
  /// vector. The tensors must be of the right dimensions.
  void insertTensors(Handle<ElemTy> &slice, llvm::ArrayRef<size_t> offset) {
    auto sliceCoor = slice.dims().vec();
    auto fusedCoor = dims().vec();
    insertTensorsImpl(sliceCoor, fusedCoor, slice, true, offset, 0);
  }

  /// Extract the tensor \p slice at location \p offset. This operation is
  /// equivalent to the operation of scanning the destination tensor, and
  /// copying into the cell at coordinate {d_0, d_1, ... d_n} a value from the
  /// tensor at {d_0 + O_0, d_1 + O_1, ... d_n + O_n}, where O is the offset
  /// vector. The tensors must be of the right dimensions.
  void extractTensors(Handle<ElemTy> &slice, llvm::ArrayRef<size_t> offset) {
    auto sliceCoor = slice.dims().vec();
    auto fusedCoor = dims().vec();
    insertTensorsImpl(sliceCoor, fusedCoor, slice, false, offset, 0);
  }

  /// Broadcast the current Tensor to a new shape specified by \p otherDims and
  /// place it in \p dest. Values in the new dimension(s) are copied from the
  /// original Tensor. The \p axis defines the dimension to start broadcasting
  /// from. Compared to numpy's broadcasting, this only allows broadcasting one
  /// tensor Tensor A to some new shape specified by \p otherDims. For example,
  /// numpy allows broadcasting two Tensors of shapes (3,1) and (1,4) to both be
  /// (3,4), while this implementation does not.
  void broadcastToNewShape(Tensor *dest, llvm::ArrayRef<size_t> otherDims,
                           int axis = 0) {
    auto origDims = dims();
    assert(otherDims.size() >= origDims.size() &&
           "Dimensions to broadcast to must be equal or greater size.");
    assert(axis >= 0 && (axis + origDims.size()) <= otherDims.size() &&
           "Axis must >= 0, && axis + nDims of orig Tensor <= newShape nDims");

    Tensor intermediate;
    intermediate.copyFrom(this->tensor_);

    for (int i = 0; i < origDims.size(); i++) {
      const int otherIdx = i + axis;
      if (origDims[i] == otherDims[otherIdx]) {
        // Keep original dimensions; they are compatible.
      } else if (origDims[i] == 1) {
        // Broadcast this dimension to size from otherDims.
        Tensor tmp;
        const bool addingNewDim = false;
        intermediate.getHandle<ElemTy>().broadcastOneDimension(
            &tmp, otherDims[otherIdx], i, addingNewDim);
        intermediate.copyFrom(&tmp);
      } else {
        // Incompatible dimensions for broadcasting
        assert(false && "Cannot broadcast with these dimensions.");
      }
    }

    // Fill in the rest of the dimensions with the other Tensor's dimensions.
    for (int i = origDims.size(); i < otherDims.size() - axis; ++i) {
      const int otherIdx = i + axis;
      Tensor tmp;
      const bool addingNewDim = true;
      intermediate.getHandle<ElemTy>().broadcastOneDimension(
          &tmp, otherDims[otherIdx], i, addingNewDim);
      intermediate.copyFrom(&tmp);
    }
    dest->copyFrom(&intermediate);
  }

  /// Broadcast the current Handle's tensor in \p direction of length \p
  /// newDimLen into Tensor \p dest. If not \p addingNewDim then the dimension
  /// being extended should be size 1.
  void broadcastOneDimension(Tensor *dest, unsigned newDimLen,
                             unsigned direction, bool addingNewDim) {
    auto origDims = dims();
    if (addingNewDim) {
      assert(direction <= origDims.size() &&
             "Adding new dimension requires direction >= 0 && <= size]");
      assert(origDims.size() != max_tensor_dimensions &&
             "Cannot broadcast tensor already at max dimensions");
    } else {
      assert(direction <= origDims.size() &&
             "Extending existing dimension requires direction >= 0 && < size");
      assert(origDims[direction] == 1 &&
             "Can only extend an existing dimension if size == 1");
    }

    // Reset size of dest to accomodate new broadcast dimension.
    size_t newDims[max_tensor_dimensions];
    unsigned shift = 0;
    for (int i = 0; i < origDims.size(); ++i) {
      if (addingNewDim && (i == direction)) {
        shift = 1;
      }
      newDims[i + shift] = origDims[i];
    }
    newDims[direction] = newDimLen;
    const unsigned newDimsSize = origDims.size() + (addingNewDim ? 1 : 0);
    dest->reset(getElementType(), llvm::ArrayRef<size_t>(newDims, newDimsSize));

    size_t currNewIdxsArr[max_tensor_dimensions];
    auto currNewIdxs =
        llvm::MutableArrayRef<size_t>(currNewIdxsArr, newDimsSize);
    size_t currIdxsArr[max_tensor_dimensions] = {0};
    auto currIdxs = llvm::MutableArrayRef<size_t>(currIdxsArr, origDims.size());

    // Iterate over all locations in the original Tensor.
    do {
      // New indices using current from original Tensor, plus new dimension.
      unsigned shift = 0;
      for (int i = 0; i < origDims.size(); ++i) {
        if (addingNewDim && (i == direction)) {
          shift = 1;
        }
        currNewIdxs[i + shift] = currIdxs[i];
      }

      // Copy all values in the new broadcast direction dimension.
      for (currNewIdxs[direction] = 0; currNewIdxs[direction] < newDimLen;
           currNewIdxs[direction]++) {
        dest->getHandle<ElemTy>().at(currNewIdxs) = at(currIdxs);
      }
    } while (!incrementIndicesAndCheckFinished(currIdxs));
  }

private:
  /// This is a slow generic transpose. This method performs a single for loop
  /// over a single dimension, or if we've reached the last dimension perform a
  /// single copy of a single element.
  void transposeImpl(Handle<ElemTy> &dest, size_t *srcCoor, size_t *destCoor,
                     llvm::ArrayRef<unsigned> shuffle, unsigned depth) {
    if (depth == shuffle.size()) {
      auto srcIdx = llvm::ArrayRef<size_t>(srcCoor, depth);
      auto destIdx = llvm::ArrayRef<size_t>(destCoor, depth);
      dest.at(destIdx) = at(srcIdx);
      return;
    }

    // Iterate over one dimension and continue recursively to the next dim.
    for (size_t x = 0, e = dest.dims()[depth]; x < e; x++) {
      unsigned swizzledDepth = shuffle[depth];
      srcCoor[swizzledDepth] = x;
      destCoor[depth] = x;
      transposeImpl(dest, srcCoor, destCoor, shuffle, depth + 1);
    }
  }

  /// Concats or splits tensors.
  /// This method concats or extracts a slice from a tensor.
  /// \p sliceCoor and \p fusedCoor are temporary storage that the function uses
  /// to construct the coordinates to access the tensor. They must be
  /// initialized to be the size of the shape of the tensor. \p slice and \p
  /// fused are the tensors to concat or extract. \p offset is the offset of the
  /// slice to add or extract along the dimension \p offsetDim. \p d is the
  /// recursion depth parameter that's following the number of the axis. if \p
  /// isInsert is set then data is copied from \p slice to \p fused. Otherwise
  /// data is copied from \p fused to \p slice.
  void insertTensorsImpl(llvm::MutableArrayRef<size_t> sliceCoor,
                         llvm::MutableArrayRef<size_t> fusedCoor,
                         Handle<ElemTy> &slice, bool isInsert,
                         llvm::ArrayRef<size_t> offset, unsigned d) {
    bool isDone = (d == slice.dims().size());

    if (isDone) {
      if (isInsert) {
        at(fusedCoor) = slice.at(sliceCoor);
      } else {
        slice.at(sliceCoor) = at(fusedCoor);
      }
      return;
    }

    for (size_t i = 0, e = slice.dims()[d]; i < e; i++) {
      // Construct the coordinates for the slice and for the joint shape.
      // Add the 'offset' to the dimension that we concat the shapes on.
      sliceCoor[d] = i;
      fusedCoor[d] = i + offset[d];
      insertTensorsImpl(sliceCoor, fusedCoor, slice, isInsert, offset, d + 1);
    }
  }

  /// Takes an array of indices \p currIdxs and increments it with respect to
  /// the Tensor's dims(), allowing for iterating over all of a Tensor's
  /// elements without statically knowing its shape.
  bool
  incrementIndicesAndCheckFinished(llvm::MutableArrayRef<size_t> currIdxs) {
    auto origDims = dims();
    assert(origDims.size() == currIdxs.size() &&
           "Set of indices should have same shape as Tensor");

    for (unsigned i = 0; i < currIdxs.size(); i++) {
      currIdxs[i] += 1;
      if (currIdxs[i] == origDims[i]) {
        currIdxs[i] = 0;
      } else {
        return false;
      }
    }

    assert(currIdxs[origDims.size() - 1] == 0 &&
           "Should have overflowed highest index if complete");
    return true;
  }
};

template <class ElemTy> Handle<ElemTy> Tensor::getHandle() {
  assert(type_.isType<ElemTy>() && "Getting a handle to the wrong type.");
  return Handle<ElemTy>(this);
}

} // namespace glow

#endif // GLOW_BASE_TENSOR_H
