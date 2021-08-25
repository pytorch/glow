/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

#include <algorithm>
#include <cassert>
#include <vector>

#include "glow/Base/DeviceTensorTransferManager.h"
#include "glow/Base/Type.h"
#include "glow/Support/Compiler.h"
#include "glow/Support/Memory.h"
#include "glow/Support/Random.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

namespace glow {

//===----------------------------------------------------------------------===//
//                               Tensor
//===----------------------------------------------------------------------===//

template <class ElemTy> class Handle;

class Tensor;
class TensorPool;

void genericTranspose(const Tensor *src, Tensor *dest,
                      llvm::ArrayRef<unsigned_t> shuffle);

/// Helper function that \returns a ShapeVector of those dimensions in \p
/// currDims expanded with dimension = 1 until the maximum tensor dimension is
/// reached. The number of elements in the input dims is the same as in the
/// returned dims. For example, input {2,1,4} would result in {2,1,4,1,1,1}.
ShapeVector expandDimsToMax(llvm::ArrayRef<dim_t> currDims);

/// Helper function that \returns a ShapeVector obtained from \p dims by
/// reducing (setting to 1) the dimensions given by \p axes. If the flag
/// \p keepDims is also used then the reduced dimensions are kept, otherwise
/// are pruned. For example, given the dimensions [2,3,4] and axes [0,2] the
/// returned shape will be [1,3,1] for keepDims true and [3] for keepDims false.
ShapeVector reduceDims(llvm::ArrayRef<dim_t> dims,
                       llvm::ArrayRef<unsigned_t> axes, bool keepDims);

/// Helper function that \returns the transpose shuffle that would undo the
/// given \p shuffle so that if two transposes were composed with the given
/// shuffle and the result of this function, it would result in the identity
/// shuffle.
std::vector<unsigned_t> getInverseTranspose(llvm::ArrayRef<unsigned_t> shuffle);

namespace runtime {
class DeviceManager;
}

/// Holds information regarding whether this Tensor exists in a device-specific
/// form, either resident or specific for a device, and what device holds it.
class DeviceResidencyInfo final {
  enum class TensorResidency {
    Host,
    Device,
  };

  // A pointer to the device manager of the device on which the tensor
  // resides.
  DeviceTensorTransferManager *deviceManager_{nullptr};
  /// The residency status of the tensor.
  TensorResidency tensorResidency_{TensorResidency::Host};
  // A pointer to a context structure, containing the required info to access
  // tensor data and perform transfers.
  void *locationContext_{nullptr};

public:
  DeviceResidencyInfo()
      : deviceManager_(nullptr), tensorResidency_(TensorResidency::Host),
        locationContext_(nullptr) {}

  /// Move ctor.
  DeviceResidencyInfo(DeviceResidencyInfo &&other) = delete;

  /// Move assignment operator.
  DeviceResidencyInfo &operator=(DeviceResidencyInfo &&other) = delete;

  ~DeviceResidencyInfo() {
    // If a tensor is device resident, let its device manager free the device
    // buffer.
    if (isDeviceResident()) {
      deviceManager_->releaseDeviceTensor(locationContext_);
    }
  }

  /// Removes all device specific state.
  void clear() {
    deviceManager_ = nullptr;
    locationContext_ = nullptr;
    tensorResidency_ = TensorResidency::Host;
  }

  /// \returns true if this Tensor is resident or specific for a device.
  bool isDeviceResident() const {
    assert((tensorResidency_ == TensorResidency::Host || deviceManager_) &&
           "Device resident tensor must have an assigned device manager.");
    return tensorResidency_ == TensorResidency::Device;
  }

  /// \returns the DeviceManager this tensor is resident on, if any.
  DeviceTensorTransferManager *getDeviceManager() const {
    return deviceManager_;
  }

  /// \returns the device specific location context for a resident Tensor.
  void *getLocationContext() const { return locationContext_; }

  friend class Tensor;
};

/// A class that represents a contiguous n-dimensional array (a tensor).
class Tensor final {
public:
  /// Specifies the kind initialization for the tensor.
  enum class InitKind {
    Zero,      // The tensor is initialized to zero.
    Broadcast, // Broadcast a single value to all elements.
    Xavier,    // Init the tensor with random values using the Xavier method.
  };

private:
  /// A pointer to the tensor data.
  char *data_{nullptr};

  /// The type of the tensor.
  Type type_;

  /// If the tensor is unowned.
  bool isUnowned_{false};

  /// The TensorPool that is managing this Tensor (if any).
  TensorPool *tensorPool_{nullptr};

  /// The device residency info accosiated with the tensor.
  DeviceResidencyInfo *deviceResidency_{nullptr};

  /// If this tensor owns the DeviceResidencyInfo.
  bool ownsDeviceResidency_{false};

  /// Size in bytes of the unpadded region memory. This is useful  communicating
  /// the actual size of the data, this allows for copying only inputs and not
  /// padding to the device.
  size_t unpaddedSize_{0};

  template <class ElemTy> friend class Handle;

  /// \returns a pointer to the tensor data buffer.
  char *getData() const { return data_; }

public:
  /// \returns true if it is an unowned tensor.
  bool isUnowned() const { return isUnowned_; }

  /// \returns the number of allocated bytes pointed to by \ref data_.
  size_t getUnpaddedSizeInBytes() const { return unpaddedSize_; }

  /// \returns the number of real elements in a Tensor, not including extra
  /// padding, or not including number of elements that do not exist outside of
  /// a partial tensor shape. Note that Tensors cannot be both custom aligned
  /// and partial.
  size_t getRealNumElements() const {
    // If custom alignment then return size from the handle.
    if (size() < actualSize()) {
      return size();
    }
    // Else assume no custom alignment, so return number of elements based on
    // unpaddedSize_, i.e. accounts for partial Tensors.
    return unpaddedSize_ / type_.getElementSize();
  }

  /// \returns the type of the tensor.
  const Type &getType() const { return type_; }

  /// Set the type of the Tensor to \p t.
  void setType(const TypeRef t) {
    assert(type_.dims() == t->dims() && "New type must retain the same shape.");
    assert(((type_.getElementType() == t->getElementType() &&
             type_.size() == t->size()) ||
            type_.getSizeInBytes() == t->getSizeInBytes()) &&
           "New type must retain the same size in bytes.");
    type_ = *t;
  }

  /// \return the element type of the tensor.
  ElemKind getElementType() const { return type_.getElementType(); }

  /// \returns True if the coordinate is within the array.
  bool isInBounds(llvm::ArrayRef<dim_t> indices) const {
    assert(type_.numSizes_ == indices.size() && "Invalid number of indices");
    for (size_t i = 0u, e = indices.size(); i < e; i++) {
      if (indices[i] >= type_.sizes_[i]) {
        return false;
      }
    }
    return true;
  }

  /// Set the content of the tensor to zero. If \p resetFusedScalesOffsets, then
  /// fused scales/offsets will be set to 1.0/0.0 as well.
  void zero(bool resetFusedScalesOffsets = false) {
    assert(!isDeviceResident() && "Tensor must reside on host to access data.");
    size_t size = actualSize();
    // Quantized tensors should go to their offset.
    switch (type_.getElementType()) {
    case ElemKind::Int8QTy: {
      auto *data = reinterpret_cast<int8_t *>(getData());
      std::fill(&data[0], &data[0] + size, (int8_t)type_.getOffset());
      break;
    }
    case ElemKind::UInt8QTy: {
      auto *data = reinterpret_cast<uint8_t *>(getData());
      std::fill(&data[0], &data[0] + size, (uint8_t)type_.getOffset());
      break;
    }
    case ElemKind::Int16QTy: {
      auto *data = reinterpret_cast<int16_t *>(getData());
      std::fill(&data[0], &data[0] + size, (int16_t)type_.getOffset());
      break;
    }
    case ElemKind::Int32QTy: {
      auto *data = reinterpret_cast<int32_t *>(getData());
      std::fill(&data[0], &data[0] + size, (int32_t)type_.getOffset());
      break;
    }
#define FUSED_CASE(ELEM_KIND, DATA_TYPE)                                       \
  case ElemKind::ELEM_KIND: {                                                  \
    assert(dims().size() == 2 && "Fused tensor must be 2-dimensional.");       \
    assert(dims()[1] > sizeof(DATA_TYPE) &&                                    \
           "Fused tensor must have space for scale and offset.");              \
    const size_t dataWidth = dims()[1];                                        \
    const size_t alignedLength = type_.strides()[0];                           \
    auto *data = reinterpret_cast<uint8_t *>(getData());                       \
    for (size_t i = 0, e = dims()[0]; i < e; i++) {                            \
      uint8_t *scaleOffsetPtr =                                                \
          data + i * alignedLength + dataWidth - 2 * sizeof(DATA_TYPE);        \
      DATA_TYPE scale, offset;                                                 \
      if (resetFusedScalesOffsets) {                                           \
        /* Use these as defaults, and copy them into each row. */              \
        scale = 1.0;                                                           \
        offset = 0.0;                                                          \
        memcpy(scaleOffsetPtr, &scale, sizeof(DATA_TYPE));                     \
        memcpy(scaleOffsetPtr + sizeof(DATA_TYPE), &offset,                    \
               sizeof(DATA_TYPE));                                             \
      } else {                                                                 \
        memcpy(&scale, scaleOffsetPtr, sizeof(DATA_TYPE));                     \
        memcpy(&offset, scaleOffsetPtr + sizeof(DATA_TYPE),                    \
               sizeof(DATA_TYPE));                                             \
      }                                                                        \
      DCHECK_NE(static_cast<float>(scale), 0.0)                                \
          << "Disallow scale = 0.0 for Fused ElemKinds; causes div by zero.";  \
      float zero = nearbyintf(-1 * static_cast<float>(offset / scale));        \
      std::fill(data + i * alignedLength, scaleOffsetPtr,                      \
                static_cast<uint8_t>(zero));                                   \
    }                                                                          \
    break;                                                                     \
  }
      FUSED_CASE(UInt8FusedQTy, float);
      FUSED_CASE(UInt8FusedFP16QTy, float16_t);
#undef FUSED_CASE

    default:
      // Non-quantized tensors are set to 0.
      std::fill(&getData()[0], &getData()[0] + size * type_.getElementSize(),
                0);
      break;
    }
  }

  /// \returns the shape of the tensor.
  llvm::ArrayRef<dim_t> dims() const { return type_.dims(); }

  /// \returns the number of real meaningful elements in the tensor. Does not
  /// take strides into account.
  dim_t size() const { return type_.size(); }

  /// \returns the actual number of elements in the tensor taking striding into
  /// account. Since size() does not take striding into account, size() is
  /// always <= actualSize().
  dim_t actualSize() const { return type_.actualSize(); }

  /// \returns the number of bytes required to store the tensor based on its
  /// Type. Note that this includes the size required for padding.
  uint64_t getSizeInBytes() const { return type_.getSizeInBytes(); }

  /// \returns the TensorPool managing this object, or nullptr if it is
  /// unmanaged.
  TensorPool *getOwningPool() { return tensorPool_; }

  template <typename DataType>
  static Tensor fromData(ElemKind elemKind, llvm::ArrayRef<dim_t> dims,
                         const std::initializer_list<DataType> &data) {
    Tensor tensor(elemKind, dims);
    tensor.getHandle<DataType>() = data;
    return tensor;
  }

  template <typename DataType>
  static Tensor fromData(ElemKind elemKind, float scale, int32_t offset,
                         llvm::ArrayRef<dim_t> dims,
                         const std::initializer_list<DataType> &data) {
    Tensor tensor(elemKind, dims, scale, offset);
    tensor.getHandle<DataType>() = data;
    return tensor;
  }

  /// Initialize an empty tensor.
  Tensor() = default;

  /// Initialize from a list of float literals.
  Tensor(const std::initializer_list<float> &vec) {
    reset(ElemKind::FloatTy, {(dim_t)vec.size()});
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
  Tensor(ElemKind elemTy, llvm::ArrayRef<dim_t> dims)
      : data_(nullptr), type_(elemTy, dims), isUnowned_{false} {
    reset(elemTy, dims);
  }

  /// Construct an unowned tensor provided an existing payload buffer.
  /// This constructor can be used when there is a need to work with
  /// "externally" managed payload buffers using Tensor APIs. Additionally
  /// \p unpaddedSize can be set to indicate actual size of the inputs. If
  /// negative then it defaults back to the size of the input type.
  Tensor(void *data, TypeRef ty, ssize_t unpaddedSize = -1)
      : data_(reinterpret_cast<char *>(data)), type_(*ty) {
    // Mark as unowned.
    isUnowned_ = true;
    // We do want DeviceResidency however, since there is no owning Glow Tensor.
    resetDeviceInfo();
    if (unpaddedSize < 0) {
      unpaddedSize_ = type_.getSizeInBytes();
    } else {
      unpaddedSize_ = static_cast<size_t>(unpaddedSize);
    }
  }

  /// Allocate and initialize a new integer tensor with \p scale and \p offset.
  Tensor(ElemKind elemTy, llvm::ArrayRef<dim_t> dims, float scale,
         int32_t offset)
      : data_(nullptr), type_(elemTy, dims, scale, offset), isUnowned_{false} {
    reset(type_);
  }

  /// Allocate a new Tensor managed by the \p tensorPool.
  explicit Tensor(TypeRef ty, TensorPool *tensorPool)
      : data_(nullptr), type_(*ty), tensorPool_(tensorPool) {
    reset(*ty);
  }

  Tensor(const Tensor &other) = delete;
  Tensor &operator=(const Tensor &other) = delete;

  /// Initialize the content of the tensor using the \p init method. The value
  /// \p val is the initialization parameter. \p PRNG is used to generate random
  /// numbers. Note that if the tensor's kind is Fused, then the fused
  /// scaled/offsets will not be modified.
  void init(InitKind init, float val, PseudoRNG &PRNG);

  /// \returns an unowned tensor with the exact same dimensions as this.
  Tensor getUnowned() const { return getUnowned(dims()); }

  /// \returns unowned tensor using the same data buffer as the current tensor
  /// but having different dimensions \p dims. \p offsets represents an optional
  /// offset into the tensor representing the location of the first element to
  /// start a subview from. The returned unonwed tensor is essentially a
  /// different view or subview on the same data.
  ///
  /// The lifetime of the returned unowned tensor should be always within
  /// the lifetime of its parent tensor, i.e. the unowned tensor should not
  /// outlive its parent tensor.
  Tensor getUnowned(llvm::ArrayRef<dim_t> dims,
                    llvm::ArrayRef<dim_t> offsets = {}) const {
    Tensor unownedTensor;

    auto *firstElemPtr = getData();
    if (offsets.size()) {
      assert(offsets.size() == this->dims().size() &&
             "Number of dims of tensor must equal number of dims in offsets");
      // Find the index of the first element and use it to find the pointer to
      // the first element.
      size_t index = 0;
      for (size_t i = 0; i < this->dims().size(); i++) {
        index += type_.strides()[i] * offsets[i];
      }
      firstElemPtr = &firstElemPtr[index * type_.getElementSize()];
    }

    unownedTensor.data_ = firstElemPtr;
    unownedTensor.isUnowned_ = true;
    unownedTensor.type_ = Type::newShape(getType(), dims);
    unownedTensor.deviceResidency_ = deviceResidency_;

    // If the original base Tensor is padded, then we only allow the unowned
    // Tensor to be padded if there are no offsets. Otherwise assert that the
    // base Tensor is not padded, and set unpaddedSize to that of the new
    // unowned type.
    if (offsets.size() == 0) {
      unownedTensor.unpaddedSize_ = unpaddedSize_;
      assert(actualSize() == unownedTensor.actualSize() &&
             "The size of the unowned tensor "
             "should be the same as the size of "
             "the original tensor");

    } else {
      unownedTensor.unpaddedSize_ = unownedTensor.type_.getSizeInBytes();
      assert(getSizeInBytes() == getUnpaddedSizeInBytes() &&
             "Problematic to get unowned offsetted view of a padded tensor");
      assert(actualSize() >= unownedTensor.actualSize() &&
             "The size of the unowned tensor "
             "should be no greater than the "
             "size of the original tensor");
    }
    return unownedTensor;
  }

  /// This is the same as \ref getUnowned() but it produces an owned tensor
  /// instead. \returns owned tensor copied from the data buffer of the current
  /// tensor but having different dimensions \p dims. \p offsets represents an
  /// optional offset into the tensor representing the location of the first
  /// element to start a subview from.
  Tensor getOwnedSlice(llvm::ArrayRef<dim_t> dims,
                       llvm::ArrayRef<dim_t> offsets = {}) const {
    assert(!isDeviceResident() && "Tensor must reside on host to access data.");
    return getUnowned(dims, offsets).clone();
  }

  /// Reset the shape and type of this tensor to match the shape and type of
  /// \p other. The size of the buffer is set to \p unpaddedSize unless it is
  /// negative, which will instead default back to the number of bytes needed
  /// for the type of \p other.
  void reset(const Tensor *other, ssize_t unpaddedSize = -1) {
    reset(other->getType(), unpaddedSize);
  }

  void reset(ElemKind elemTy, llvm::ArrayRef<dim_t> shape) {
    Type t(elemTy, shape);
    reset(t);
  }

  void reset(ElemKind elemTy, llvm::ArrayRef<dim_t> shape, float scale,
             int32_t offset) {
    Type t(elemTy, shape, scale, offset);
    reset(t);
  }

  /// Assigns a new shape to the tensor and allocates a new buffer. The size of
  /// the buffer is set to \p unpaddedSize unless it is negative, which will
  /// instead default back to the number of bytes needed for \p T.
  void reset(const Type &T, ssize_t unpaddedSize = -1) {
    assert(!isDeviceResident() && "Tensor must reside on host to access data.");

    // If negative then fall back to the passed in Type's padded size.
    if (unpaddedSize < 0) {
      unpaddedSize = T.getSizeInBytes();
    }

    // If the new size is identical to the allocated size then there is no need
    // to re-allocate the buffer.
    const bool isOrigPadded =
        getSizeInBytes() != uint64_t(getUnpaddedSizeInBytes());
    const bool isNewPadded = T.getSizeInBytes() != size_t(unpaddedSize);
    const bool isBufReuseAllowed =
        (isOrigPadded == isNewPadded) &&
        (getUnpaddedSizeInBytes() == size_t(unpaddedSize));
    if (type_ == T && getData() && isBufReuseAllowed) {
#ifdef GLOW_DEBUG_TENSOR_INIT
      PseudoRNG rng;
      init(InitKind::Broadcast, GLOW_DEBUG_TENSOR_INIT, rng);
#endif
      resetDeviceInfo();
      return;
    }

    // Delete the old buffer, update the shape, and allocate a new one.
    if (!isUnowned())
      alignedFree(getData());
    type_ = T;

    // We are allocating memory specifically for this tensor, thus, it owns it.
    isUnowned_ = false;

    // We are allocating memory on the host so it is not device resident.
    resetDeviceInfo();

    // Note: zero-dimensional tensors (i.e. {}) have size 1. However, Tensors
    // may have 0 for some dimension, meaning they have size of 0, and so we do
    // not allocate anything for them.
    data_ = unpaddedSize == 0 ? nullptr
                              : reinterpret_cast<char *>(alignedAlloc(
                                    unpaddedSize, TensorAlignment));

    // Set unpaddedSize_ to the actual number of bytes.
    unpaddedSize_ = unpaddedSize;

    assert(!(size() < actualSize() &&
             getSizeInBytes() != getUnpaddedSizeInBytes()) &&
           "Custom aligned Tensors cannot also be partial");

#ifdef GLOW_DEBUG_TENSOR_INIT
    PseudoRNG rng;
    init(InitKind::Broadcast, GLOW_DEBUG_TENSOR_INIT, rng);
#endif
  }
  /// Releases the data buffer and sets the unOwned flag to true. This is useful
  /// for keeping metadata around but not the actual contents.
  void release() {
    if (!isUnowned()) {
      alignedFree(getData());
    }
    if (ownsDeviceResidency_) {
      delete deviceResidency_;
      ownsDeviceResidency_ = false;
    }

    isUnowned_ = true;
  }
  ~Tensor() {
    if (!isUnowned()) {
      alignedFree(getData());
    }

    if (ownsDeviceResidency_) {
      delete deviceResidency_;
      ownsDeviceResidency_ = false;
    }
  }

  // Move ctor.
  Tensor(Tensor &&other) noexcept {
    if (!isUnowned()) {
      alignedFree(getData());
    }
    if (ownsDeviceResidency_) {
      delete deviceResidency_;
    }
    data_ = other.data_;
    type_ = other.type_;
    isUnowned_ = other.isUnowned_;
    tensorPool_ = other.tensorPool_;
    unpaddedSize_ = other.unpaddedSize_;
    deviceResidency_ = other.deviceResidency_;
    ownsDeviceResidency_ = other.ownsDeviceResidency_;
    other.data_ = nullptr;
    other.isUnowned_ = true;
    other.tensorPool_ = nullptr;
    other.deviceResidency_ = nullptr;
    other.ownsDeviceResidency_ = false;
  }

  /// Move assignment operator.
  Tensor &operator=(Tensor &&other) {
    if (!isUnowned()) {
      alignedFree(getData());
    }
    if (ownsDeviceResidency_) {
      delete deviceResidency_;
    }
    data_ = other.data_;
    type_ = other.type_;
    isUnowned_ = other.isUnowned_;
    tensorPool_ = other.tensorPool_;
    unpaddedSize_ = other.unpaddedSize_;
    deviceResidency_ = other.deviceResidency_;
    ownsDeviceResidency_ = other.ownsDeviceResidency_;
    other.data_ = nullptr;
    other.isUnowned_ = true;
    other.tensorPool_ = nullptr;
    other.deviceResidency_ = nullptr;
    other.ownsDeviceResidency_ = false;
    return *this;
  }

  /// Dump a textual representation of the Tensor into provided output stream.
  void dump(llvm::raw_ostream &os) const;

  /// Dump a textual representation of the Tensor into default output stream.
  void dump() const;

  /// Dump a textual representation of a specific number of elements in the
  /// Tensor into provided output stream.
  void dump(llvm::raw_ostream &os, unsigned maxNumElem) const;

  /// Dump a textual representation of a specific number of elements in the
  /// Tensor into default output stream.
  void dump(unsigned maxNumElem) const;

  /// Dump a textual representation of the Tensor to std::string.
  std::string toString() const;

  /// Dump a textual representation of a specific number of elements in the
  /// Tensor to std::string.
  std::string toString(unsigned maxNumElem) const;

  /// Dump a textual representation of the shape of this Tensor to std::string.
  std::string getShapeToString() const;

  /// \returns true if the content of the other tensor \p other is identical to
  /// this one, given some \p allowedError. If \p verbose and the tensors are
  /// not equal, then we will log information about the mismatch (number of
  /// elements exceeding allowed error; maximum error and location found; etc.).
  bool isEqual(const Tensor &other, float allowedError = 0.0001,
               bool verbose = true) const {
    if (isDeviceResident()) {
      if (!other.isDeviceResident()) {
        if (verbose) {
          LOG(INFO) << "Tensors cannot be compared as they are not resident in "
                       "the same location.";
        }
        return false;
      }

      return getDeviceManager() == other.getDeviceManager() &&
             getLocationContext() == other.getLocationContext();
    }
    return isEqualImpl(other, /*isBitwise=*/false, allowedError, verbose);
  }

  /// \returns true if the content of the other tensor \p other is bitwise
  /// identical to this one.
  bool isBitwiseEqual(const Tensor &other, bool verbose = false) const {
    return isEqualImpl(other, /*isBitwise=*/true, /*allowedError=*/0.0,
                       verbose);
  }

  bool isEqualImpl(const Tensor &other, bool isBitwise, float allowedError,
                   bool verbose) const {
    if (other.dims() != dims()) {
      if (verbose) {
        LOG(INFO) << "Tensors are not equal as they have different shapes: "
                  << this->getShapeToString() << " vs. "
                  << other.getShapeToString();
      }
      return false;
    }

    // For now, make sure that either both or neither of the tensors have
    // UInt8FusedQTy or UInt8Fused16QTy. While it is possible for an Int8QTy
    // tensor to equal a fused tensor if the fused tensor has the same
    // scale/offset on all of its rows, and that scale/offset match that of the
    // Int8QTy, we do not support checking this for now.
    assert(((getElementType() == ElemKind::UInt8FusedQTy &&
             other.getElementType() == ElemKind::UInt8FusedQTy) ||
            (getElementType() == ElemKind::UInt8FusedFP16QTy &&
             other.getElementType() == ElemKind::UInt8FusedFP16QTy) ||
            (getElementType() != ElemKind::UInt8FusedFP16QTy &&
             other.getElementType() != ElemKind::UInt8FusedQTy)) &&
           "Fused ElemKinds only supports comparing against same ElemKind.");

    // Assert that the scale and offset match for the quantized types.
    switch (getElementType()) {
    default:
      break;
    case ElemKind::Int8QTy:
    case ElemKind::UInt8QTy:
    case ElemKind::Int16QTy:
    case ElemKind::Int32QTy:
      assert(getType().getScale() == other.getType().getScale() &&
             "Scales must match.");
      assert(getType().getOffset() == other.getType().getOffset() &&
             "Offsets must match.");
    }

    // Bitwise compare.
    if (isBitwise) {
      return isBitwiseEqualImpl(other, verbose);
    }

    switch (getElementType()) {
    case ElemKind::FloatTy:
      return isEqualImpl<float>(other, allowedError, verbose);
    case ElemKind::Float16Ty:
      return isEqualImpl<float16_t>(other, allowedError, verbose);
    case ElemKind::BFloat16Ty:
      return isEqualImpl<bfloat16_t>(other, allowedError, verbose);
    case ElemKind::Float64Ty:
      return isEqualImpl<double>(other, allowedError, verbose);
    case ElemKind::Int8QTy:
      return isEqualImpl<int8_t>(other, allowedError, verbose);
    case ElemKind::UInt8QTy:
      return isEqualImpl<uint8_t>(other, allowedError, verbose);
    case ElemKind::Int16QTy:
      return isEqualImpl<int16_t>(other, allowedError, verbose);
    case ElemKind::Int32QTy:
      return isEqualImpl<int32_t>(other, allowedError, verbose);
    case ElemKind::Int64QTy:
      return isEqualImpl<int64_t>(other, allowedError, verbose);
    case ElemKind::UInt8ITy:
      return isEqualImpl<uint8_t>(other, allowedError, verbose);
    case ElemKind::Int32ITy:
      return isEqualImpl<int32_t>(other, allowedError, verbose);
    case ElemKind::Int64ITy:
      return isEqualImpl<int64_t>(other, allowedError, verbose);
      // Note: We can use isEqualImpl() here because the scales/offsets will be
      // compared as if they were data, so we will return false if any rowwise
      // scale/offset do not match.
    case ElemKind::UInt8FusedQTy:
      return isEqualImpl<uint8_t>(other, allowedError, verbose);
    case ElemKind::UInt8FusedFP16QTy:
      return isEqualImpl<uint8_t>(other, allowedError, verbose);
    case ElemKind::UInt4FusedFP16QTy:
      return isEqualImpl<uint8_t>(other, allowedError, verbose);
    case ElemKind::UInt4FusedQTy:
      return isEqualImpl<uint8_t>(other, allowedError, verbose);
    case ElemKind::BoolTy:
      return isEqualImpl<bool>(other, allowedError, verbose);
    }

    // This is to make compiler happy. It can never reach this point as switch
    // always covers all possible values.
    llvm_unreachable("unreachable");
  }

  /// \returns whether this Tensor is tiled (repeated) along \p axis for the
  /// given tile size \p size. Some examples:
  /// - A Tensor with size [2, 3] equal to [[1,2,3],[1,2,3]] is tiled along
  ///   axis 0 for a tile size equal to 1.
  /// - A Tensor with size [2, 4] equal to [[1, 2, 1, 2],[3, 4, 3, 4]] is tiled
  ///   along axis 1 for a tile size equal to 2.
  /// When the tile size matches the dimensions size this function returns TRUE.
  /// If the \p fractional flag is optionally given that this function will also
  /// perform fractional tiling verification (default is FALSE). Some examples:
  /// - For a Tensor with size [5] equal to [1,2,3,1,2], axis 0 and tile size 3,
  ///   this function returns TRUE if \p fractional is TRUE and returns FALSE if
  ///   \p fractional is FALSE.
  bool isTiled(unsigned_t axis, dim_t size = 1, bool fractional = false) const;

  /// \returns whether this Tensor is tiled (repeated) along \p axes for the
  /// given tile sizes \p sizes. Some examples:
  /// - A Tensor with size [2, 4] equal to [[1,2,1,2],[1,2,1,2]] is tiled along
  ///   axes {0,1} for the tile sizes {1,2}.
  /// When the tile sizes match the dimension sizes this function returns TRUE.
  /// If the \p fractional flag is optionally given that this function will also
  /// perform fractional tiling verification (default is FALSE). Some examples:
  /// - For a Tensor with size [5] equal to [1,2,3,1,2], axes {0} and sizes {3},
  ///   this function returns TRUE if \p fractional is TRUE and returns FALSE if
  ///   \p fractional is FALSE.
  bool isTiled(llvm::ArrayRef<unsigned_t> axes, llvm::ArrayRef<dim_t> sizes,
               bool fractional = false) const;

  /// Update the content and type of the tensor from the tensor \p t.
  void assign(const Tensor *t) {
    assert(!isDeviceResident() && "Tensor must reside on host to access data.");
    assert(this != t && "Copying to self");
    const size_t bufferSize = t->getUnpaddedSizeInBytes();
    reset(t, bufferSize);
    std::copy(&t->getData()[0], &t->getData()[bufferSize], getData());
  }

  /// Update the raw data of the tensor from the tensor \p t.
  void copyRawFrom(const Tensor *t) {
    assert(!isDeviceResident() && "Tensor must reside on host to access data.");
    assert(this != t && "Copying to self");
    assert(actualSize() == t->actualSize());
    assert(getElementType() == t->getElementType() && "Invalid element type");
    assert(t->getUnpaddedSizeInBytes() == getUnpaddedSizeInBytes() &&
           "Do not support copying between different unpadded sized tensors");
    size_t bufferSize = type_.getSizeInBytes();
    std::copy(&t->getData()[0], &t->getData()[bufferSize], getData());
  }

  /// Update the raw data of the tensor from a raw buffer \p data.
  void copyRawFrom(const char *data) {
    assert(!isDeviceResident() && "Tensor must reside on host to access data.");
    assert(data && "Null data pointer!");
    assert(getData() != data && "Copying to self");
    size_t bufferSize = type_.getSizeInBytes();
    std::memcpy(getData(), data, bufferSize);
  }

  /// Update the content of the tensor with a slice from tensor \p t. A slice
  /// is one index from the first dimension of the tensor.
  void copySlice(const Tensor *t, size_t slice) {
    assert(!isDeviceResident() && "Tensor must reside on host to access data.");
    auto dim = t->dims().slice(1);
    (void)dim;
    assert(dim == dims() && "Invalid slice size");
    assert(getElementType() == t->getElementType() && "Invalid element type");

    size_t bufferSize = type_.getSizeInBytes();
    std::copy(&t->getData()[bufferSize * slice],
              &t->getData()[bufferSize * (slice + 1)], getData());
  }

  /// Update the content of the tensor with a sequence of slices from the
  /// tensor \p t. A slice is one index from the first dimension of the tensor.
  /// The copying operation may overlap the end of the tensor \p t one or more
  /// times. This means that the data in the input tensor may be duplicated.
  void copyConsecutiveSlices(const Tensor *t, size_t startSliceIdx) {
    assert(!isDeviceResident() && "Tensor must reside on host to access data.");
    auto onceSliceDim = t->dims().slice(1);
    (void)onceSliceDim;
    assert(onceSliceDim == dims().slice(1) && "Invalid slice size");
    assert(getElementType() == t->getElementType() && "Invalid element type");
    assert(dims().size() > 1 && "Tensor must contain at least two dimensions");

    size_t numSlicesInInput = t->dims()[0];
    size_t numElementsInSlice = actualSize() / dims()[0];
    size_t bufferSize = numElementsInSlice * type_.getElementSize();

    // For each outer slice in the current tensor:
    for (size_t n = 0, e = dims()[0]; n < e; n++) {
      size_t startIdx = (startSliceIdx + n) % numSlicesInInput;
      std::copy(&t->getData()[bufferSize * startIdx],
                &t->getData()[bufferSize * (startIdx + 1)],
                &getData()[bufferSize * n]);
    }
  }

  /// Convenience method to copy the content of \p t
  /// to this while both have different underlying types.
  /// This copy will read each element of \p t as SrcElemType
  /// and cast them to DestElemType in this.
  template <typename DestElemType, typename SrcElemType>
  void copyWithCast(const Tensor *t) {
    assert(!isDeviceResident() && "Tensor must reside on host to access data.");
    static_assert(!std::is_same<DestElemType, SrcElemType>::value,
                  "Use copyRawFrom instead");
    assert(this != t && "Copying to self");
    assert(getElementType() != t->getElementType() &&
           "Use copyRawFrom instead");
    assert(actualSize() == t->actualSize() && "Different sizes");
    const auto *src = t->getRawDataPointer<SrcElemType>();
    auto *dst = getRawDataPointer<DestElemType>();
    for (size_t idx = 0, end = actualSize(); idx != end; ++idx) {
      dst[idx] = DestElemType(src[idx]);
    }
  }

  /// Convert each element of this tensor to \p newTy. Calls into
  /// \ref getCopyConvertedToType() to do the conversion, and hence supports
  /// converting between whatever ElemKinds it supports.
  void convertToType(ElemKind newTy);

  /// \returns a copy of the Tensor but converted to \p newKind. Currently
  /// supports conversion for:
  /// - FloatTy to Float16Ty
  /// - FloatTy to BFloat16Ty
  /// - Float16Ty to FloatTy
  /// - BFloat16Ty to FloatTy
  /// - UInt8FusedQTy to UInt8FusedFP16QTy
  Tensor getCopyConvertedToType(ElemKind newKind) const;

  /// Transpose the tensor \p src into the empty tensor \p dest. Shuffle the
  /// axis based on the list \p shuffle, where each element is the src index.
  void transpose(Tensor *dest, llvm::ArrayRef<unsigned_t> shuffle) const {
    assert(!isDeviceResident() && "Tensor must reside on host to access data.");
    genericTranspose(this, dest, shuffle);
  }

  /// Create a new copy of the current tensor.
  Tensor clone() const {
    assert(!isDeviceResident() && "Tensor must reside on host to access data.");
    Tensor slice;
    slice.assign(this);
    return slice;
  }

  /// Return the raw unsafe pointer to the tensor payload.
  char *getUnsafePtr() const { return getData(); }

  /// \returns true if tensor data is stored on a device
  bool isDeviceResident() const {
    return deviceResidency_ && deviceResidency_->isDeviceResident();
  }

  /// Update device residency info with new device manager and context
  void moveToDevice(DeviceTensorTransferManager *deviceManager,
                    void *locationContext);

  /// If device resident, copy Tensor contents back to host memory and release
  /// associated device memory.
  void ensureOnHost();

  /// Updates contents of a device resident Tensor with the data from \p t
  /// without copying its contents to host.
  void copyRawToDevice(const Tensor *t);

  /// \returns the pointer to the device manager where the tensor resides.
  DeviceTensorTransferManager *getDeviceManager() const {
    assert(deviceResidency_ != nullptr && "DeviceResidencyInfo must exist");
    assert(deviceResidency_->isDeviceResident() &&
           "Tensor must be device resident");
    return deviceResidency_->getDeviceManager();
  }

  /// \returns the pointer to the location context of where the tensor resides.
  void *getLocationContext() const {
    assert(deviceResidency_ != nullptr && "DeviceResidencyInfo must exist");
    assert(deviceResidency_->isDeviceResident() &&
           "Tensor must be device resident");
    return deviceResidency_->getLocationContext();
  }

  void resetDeviceInfo() {
    if (deviceResidency_ && ownsDeviceResidency_) {
      deviceResidency_->clear();
      return;
    }

    deviceResidency_ = new DeviceResidencyInfo();
    ownsDeviceResidency_ = true;
  }

  /// Clears DeviceResidencyInfo.
  /// Note that this does not affect the associated DeviceManager or device
  /// memory.
  void clearDeviceResidency() {
    assert(deviceResidency_ != nullptr && "DeviceResidencyInfo must exist");
    assert(deviceResidency_->isDeviceResident() &&
           "Tensor must be device resident");
    deviceResidency_->clear();
  }

  /// \return a new handle that points and manages this tensor.
  template <class ElemTy = float> Handle<ElemTy> getHandle() &;

  template <class ElemTy = float> const Handle<ElemTy> getHandle() const &;

  /// If Tensor is rvalue, it is an error to get its Handle.
  template <class ElemTy = float> Handle<ElemTy> getHandle() && = delete;

private:
  /// \returns a pointer to the raw data, of type \p ElemTy.
  template <class ElemTy> ElemTy *getRawDataPointer() {
    assert(!isDeviceResident() && "Tensor must reside on host to access data.");
    assert(type_.isType<ElemTy>() && "Asking for the wrong ptr type.");
    return reinterpret_cast<ElemTy *>(data_);
  }

  /// \returns a const pointer to the raw data, of type \p ElemTy.
  template <class ElemTy> const ElemTy *getRawDataPointer() const {
    assert(!isDeviceResident() && "Tensor must reside on host to access data.");
    assert(type_.isType<ElemTy>() && "Asking for the wrong ptr type.");
    return reinterpret_cast<const ElemTy *>(data_);
  }

  template <class ElemTy>
  bool isEqualImpl(const Tensor &other, float allowedError,
                   bool verbose) const {
    assert(!isDeviceResident() && "Tensor must reside on host to access data.");
    auto thisHandle = getHandle<ElemTy>();
    auto otherHandle = other.getHandle<ElemTy>();
    double maxFoundError = 0.0;
    size_t numExceedingError = 0;
    size_t currIndex = 0;
    size_t maxFoundErrorIdx = 0;
    double maxRE = 0.0; // relative error.
    size_t maxREIdx = 0;
    for (auto thisHandleIt = thisHandle.begin(),
              otherHandleIt = otherHandle.begin();
         thisHandleIt != thisHandle.end() && otherHandleIt != otherHandle.end();
         ++thisHandleIt, ++otherHandleIt, ++currIndex) {
      double delta = *thisHandleIt - *otherHandleIt;
      delta = std::abs(delta);
      // Since any comparison with NAN returns false, we use a negated condition
      // so that this function correctly returns false when delta is NAN.
      if (!(delta <= allowedError)) {
        if (!verbose) {
          return false;
        }
        numExceedingError += 1;
        if (!(delta <= maxFoundError)) {
          maxFoundError = delta;
          maxFoundErrorIdx = currIndex;
        }
        double sum = *thisHandleIt + *otherHandleIt;
        double re = delta / std::abs(sum);
        if (!(re <= maxRE)) {
          maxRE = re;
          maxREIdx = currIndex;
        }
      }
    }
    auto thisHandleIt = thisHandle.begin();
    auto otherHandleIt = otherHandle.begin();
    if (numExceedingError != 0) {
      LOG(INFO) << "Tensors not equal: " << numExceedingError << " out of "
                << actualSize() << " elements exceeded allowed error threshold "
                << allowedError << ". Maximum error found was " << maxFoundError
                << " at index " << maxFoundErrorIdx << ": "
                << *(thisHandleIt.operator+(maxFoundErrorIdx)) << " vs. "
                << *(otherHandleIt.operator+(maxFoundErrorIdx));
      LOG(INFO) << "Maximum relative error found was: " << maxRE
                << " at index: " << maxREIdx << ": "
                << *(thisHandleIt.operator+(maxREIdx)) << " v.s. "
                << *(otherHandleIt.operator+(maxREIdx));
    }
    return numExceedingError == 0;
  }

  bool isBitwiseEqualImpl(const Tensor &other, bool verbose) const {
    assert(!isDeviceResident() && "Tensor must reside on host to access data.");
    auto const *myData = getUnsafePtr();
    auto const *otherData = other.getUnsafePtr();
    dim_t mismatchCount = 0;

    if (verbose) {
      for (size_t i = 0, e = getSizeInBytes(); i < e; i++) {
        if (myData[i] != otherData[i]) {
          ++mismatchCount;
        }
      }
      if (mismatchCount != 0) {
        LOG(INFO) << "Tensors not bitwise equal: " << mismatchCount
                  << " bytes out of " << getSizeInBytes() << " mismatched.";
      }
    } else {
      mismatchCount = memcmp(myData, otherData, getSizeInBytes());
    }

    return mismatchCount == 0;
  }
};

//===----------------------------------------------------------------------===//
//                    Tensor Handle
//===----------------------------------------------------------------------===//

constexpr unsigned MAX_DUMP_ELEMS = 100;

void dumpAsciiImpl(const Tensor *T, llvm::raw_ostream &os);
void dumpAsciiImpl(const Tensor *T);

void dumpImpl(const Tensor *T, llvm::raw_ostream &os,
              unsigned maxNumElem = MAX_DUMP_ELEMS);
void dumpImpl(const Tensor *T, unsigned maxNumElem);
void dumpImpl(const Tensor *T);

template <class ElemTy> class Handle;

/// A class that provides ability to iterate over a Handle<ElemTy>. Since it's
/// common to have both mutating and const iterators, this class has template
/// parameter IsConst, which is true to create const_iterator and false
/// otherwise.
template <class ElemTy, bool IsConst>
class HandleIterator
    : public std::iterator<std::random_access_iterator_tag, ElemTy> {
  using HandleTy = typename std::conditional_t<IsConst, const Handle<ElemTy> *,
                                               Handle<ElemTy> *>;
  using ElemTyRef =
      typename std::conditional_t<IsConst, const ElemTy &, ElemTy &>;

  /// At every given moment, the iterator maintains an index, which is used to
  /// access the Handle. When moving the iterator forward, the index is
  /// incremented. Only valid elements can be accessed.
  /// 0 <= idx_ <= handle_->size()
  HandleTy handle_;
  llvm::ArrayRef<dim_t> sizes_;
  dim_t idx_;
  /// Holds true if the underlying tensor has non-trivial alignment (i.e. not 1)
  bool isAligned_;

  HandleIterator() = default;

  HandleIterator(HandleTy handle) : handle_(handle) {
    sizes_ = handle->dims();
    isAligned_ = handle->size() < handle->actualSize();
  }

  static HandleIterator begin(HandleTy handle) {
    auto res = HandleIterator(handle);
    res.idx_ = 0;
    return res;
  }

  static HandleIterator end(HandleTy handle) {
    auto res = HandleIterator(handle);
    res.idx_ = res.handle_->getRealNumElements();
    return res;
  }

  friend class Handle<ElemTy>;

public:
  HandleIterator &operator++() {
    if (*this != handle_->end()) {
      idx_++;
    }
    return *this;
  }
  HandleIterator &operator--() {
    if (idx_) {
      idx_--;
    }
    return *this;
  }
  HandleIterator operator+(int n) const {
    auto res = HandleIterator(handle_);
    res.idx_ = std::max(static_cast<int>(idx_) + n, 0);
    res.idx_ = std::min(res.idx_, res.handle_->size());
    return res;
  }
  HandleIterator operator-(int n) const { return *this + (-n); }
  operator int() const { return idx_; }

  ElemTyRef operator*() {
    if (!isAligned_) {
      return handle_->raw(idx_);
    }
    std::vector<dim_t> indices(sizes_.size(), 0);
    size_t rem = idx_;
    for (int i = static_cast<int>(sizes_.size()) - 1; i >= 0; i--) {
      indices[i] = rem % sizes_[i];
      rem /= sizes_[i];
    }
    return handle_->at(indices);
  }

  bool operator==(const HandleIterator<ElemTy, IsConst> &other) const {
    return idx_ == other.idx_;
  }

  bool operator!=(const HandleIterator<ElemTy, IsConst> &other) const {
    return !(*this == other);
  }
};

/// Helper which \returns the flattened 1D offset given \p indices into a tensor
/// with \p strides.
inline size_t getFlattenedOffset(llvm::ArrayRef<dim_t> strides,
                                 llvm::ArrayRef<dim_t> indices) {
  assert(indices.size() <= strides.size() && "Invalid number of indices");
  // The loop below can be rewritten using std::inner_product. Unfortunately
  // std::inner_product does not optimize very well and loops that use this
  // method don't get vectorized. Don't change this loop without benchmarking
  // the program on a few compilers.
  size_t index = 0;
  for (size_t i = 0, e = indices.size(); i < e; i++) {
    index += size_t(strides[i]) * size_t(indices[i]);
  }

  return index;
}

/// Helper function which \returns true if a slice with the shape \p sliceShape
/// referenced from a larger tensor with the shape \p tensorShape is contiguous
/// in memory (assuming the tensor it is referenced from is contiguous). This
/// happens when the slice dimensions:
/// - Start with singleton dimensions (dimensions equal to 1).
/// - Continue with a partially extracted dimension (one maximum).
/// - End with fully extracted dimensions.
bool isSliceContiguous(llvm::ArrayRef<dim_t> sliceShape,
                       llvm::ArrayRef<dim_t> tensorShape);

/// A class that provides indexed access to a tensor. This class has value
/// semantics and it's copied around. One of the reasons for making this class
/// value semantics is to allow efficient index calculation that the compiler
/// can optimize (because stack allocated structures don't alias).
template <class ElemTy> class Handle final {
  /// A pointer to the tensor that this handle wraps.
  Tensor *tensor_{nullptr};

  /// Contains the multiplication of the sizes from current position to end.
  /// For example, for index (w,z,y,z):  [x * y * z, y * z, z, 1]
  dim_t sizeIntegral_[max_tensor_dimensions] = {
      0,
  };

  dim_t sizes_[max_tensor_dimensions] = {
      0,
  };

  /// Saves the number of dimensions used in the tensor.
  uint8_t numDims_{0};

  /// Remember end iterators. This is needed to speed up iterator increment,
  /// which has to check that iterator hasn't reached the end yet.
  HandleIterator<ElemTy, false> mutating_end_;
  HandleIterator<ElemTy, true> const_end_;

  /// Create a new invalid handle. Notice that this method is private and may
  /// only be used by the static factory method below.
  Handle() = default;

public:
  /// \returns an iterator to the first element of the tensor.
  HandleIterator<ElemTy, false> begin() {
    return HandleIterator<ElemTy, false>::begin(this);
  }
  HandleIterator<ElemTy, true> begin() const {
    return HandleIterator<ElemTy, true>::begin(this);
  }

  /// \returns an iterator referring to the past-the-end element.
  HandleIterator<ElemTy, false> end() { return mutating_end_; }
  HandleIterator<ElemTy, true> end() const { return const_end_; }

  /// Allocate a new invalid handle.
  static Handle createInvalidHandle() { return Handle(); }

  /// \returns true if this Handle points to a valid tensor.
  bool isValid() const { return tensor_; }

  /// Calculate the index for a specific element in the tensor. Notice that
  /// the list of indices may be incomplete. This method provides access to
  /// padding elements, meaning that it's possible to get an index pointing at
  /// data, added to meet alignment requirements.
  size_t getElementPtr(llvm::ArrayRef<dim_t> indices) const {
    return getFlattenedOffset(llvm::makeArrayRef(sizeIntegral_, numDims_),
                              indices);
  }

  /// \returns the value of the n'th dimension \p dim, for the index \p idx.
  /// 0 <= idx < size(), meaning that \p idx addresses a real data elements,
  /// not paddings.
  size_t getDimForPtr(size_t dim, size_t idx) const {
    assert(dim < numDims_ && "Invalid dimension");
    assert(idx < size() && "Invalid index");
    auto R = idx;
    for (size_t i = dim + 1; i < numDims_; i++) {
      R /= sizes_[i];
    }
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
    if (numDims_) {
      // Copy the sizes of the tensor.
      memcpy(sizes_, tensor_->type_.sizes_,
             max_tensor_dimensions * sizeof(sizes_[0]));
      // Copy the strides of the tensor.
      memcpy(sizeIntegral_, tensor_->type_.strides_,
             max_tensor_dimensions * sizeof(tensor_->type_.strides_[0]));
      assert(numDims_ <= max_tensor_dimensions && "Too many dimensions.");
    }

    mutating_end_ = HandleIterator<ElemTy, false>::end(this);
    const_end_ = HandleIterator<ElemTy, true>::end(this);
  }

  llvm::ArrayRef<dim_t> dims() const {
    return llvm::ArrayRef<dim_t>(sizes_, numDims_);
  }

  /// \returns the number of elements in the whole tensor.
  dim_t size() const { return tensor_->size(); }

  /// \returns the actual number of elements in the tensor taking striding into
  /// account. Since size() does not take striding into account, size() is
  /// always <= actualSize().
  dim_t actualSize() const { return tensor_->actualSize(); }

  /// \returns the unpadded size of the underlying \ref tensor_.
  size_t getUnpaddedSizeInBytes() const {
    return tensor_->getUnpaddedSizeInBytes();
  }

  /// \returns the number of unpadded elements in the underlying \ref tensor_.
  size_t getRealNumElements() const { return tensor_->getRealNumElements(); }

  bool isInBounds(llvm::ArrayRef<dim_t> indices) const {
    return tensor_->isInBounds(indices);
  }

  void clear(ElemTy value = 0) { std::fill(begin(), end(), value); }

  /// Returns reference to a meaningful data element. This method does not
  /// address padding elements.
  ElemTy &at(llvm::ArrayRef<dim_t> indices) {
    size_t index = getElementPtr(indices);
    auto *data = tensor_->getRawDataPointer<ElemTy>();
    return data[index];
  }

  const ElemTy &at(llvm::ArrayRef<dim_t> indices) const {
    size_t index = getElementPtr(indices);
    auto *data = tensor_->getRawDataPointer<ElemTy>();
    return data[index];
  }

  /// \returns the element at offset \p idx without any size calculations.
  /// The returned element can be a pad element.
  ElemTy &raw(size_t index) {
    auto *data = tensor_->getRawDataPointer<ElemTy>();
    return data[index];
  }

  /// \returns the element at offset \p idx without any size calculations.
  /// The returned element can be a pad element.
  const ElemTy &raw(size_t index) const {
    auto *data = tensor_->getRawDataPointer<ElemTy>();
    return data[index];
  }

  /// Extract a smaller dimension tensor from a specific slice (that has to be
  /// the first dimension).
  Tensor extractSlice(size_t idx) const {
    auto sizes = tensor_->dims();
    assert(sizes.size() > 1 && "Tensor must have at least two dimensions");
    assert(idx < sizes[0] && "Invalid first index");

    Tensor slice{Type::newShape(tensor_->getType(), sizes.slice(1),
                                tensor_->type_.strides().slice(1))};

    // Extract the whole slice.
    size_t startIdx = sizeIntegral_[0] * idx;
    ElemTy *base = tensor_->getRawDataPointer<ElemTy>() + startIdx;
    auto *dest = slice.getRawDataPointer<ElemTy>();
    std::copy(base, base + sizeIntegral_[0], dest);

    return slice;
  }

  /// Insert a smaller dimension tensor into a larger tensor at a specific
  /// first-dimension index.
  void insertSlice(const Tensor &slice, size_t idx) {
    auto dims = tensor_->dims();
    (void)dims;
    assert(getElementType() == slice.getElementType());
    assert(dims.size() > 1 && "Tensor must have at least two dimensions");
    assert(idx < dims[0] && "Invalid first index");

    auto sliceSize = sizeIntegral_[0];
    size_t startIdx = sliceSize * idx;
    ElemTy *base = &raw(startIdx);
    const ElemTy *slicePtr = slice.getRawDataPointer<float>();
    std::copy(slicePtr, slicePtr + sliceSize, base);
  }

  /// Create a new copy of the current tensor.
  Tensor clone() const { return tensor_->clone(); }

  /// Update the content of the tensor from a literal list:
  void operator=(const std::initializer_list<ElemTy> &vec) {
    assert(actualSize() == vec.size() && "Invalid input size.");
    size_t i = 0;
    for (auto &e : vec) {
      raw(i++) = e;
    }
  }

  void operator=(llvm::ArrayRef<ElemTy> array) {
    assert(actualSize() == array.size() && "Invalid input size.");
    std::copy(array.begin(), array.end(), &raw(0));
  }

  void dumpAscii(llvm::raw_ostream &os) const { dumpAsciiImpl(tensor_, os); }
  void dumpAscii() const { dumpAsciiImpl(tensor_); }

  /// \returns the raw indices of a min and max values from the tensor.
  /// In case of multiple min or max, the smallest index is returned.
  std::pair<dim_t, dim_t> minMaxArg() const {
    ElemTy max = raw(0);
    ElemTy min = raw(0);

    size_t maxIdx = 0;
    size_t minIdx = 0;

    for (size_t i = 1, e = actualSize(); i < e; i++) {
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
  /// \p allowedError represents the delta from zero that is allowed before
  /// returning false.
  bool isZero(float allowedError = 0.0) const {
#define RETURN_WHETHER_FUSED_IS_ZERO(DATA_TYPE)                                \
  assert(dims().size() == 2 && "Fused tensor must be 2-dimensional.");         \
  assert(dims()[1] > 2 * sizeof(DATA_TYPE) &&                                  \
         "Fused tensor must have space for scale/offset.");                    \
  const dim_t dataWidth = dims()[1];                                           \
  const dim_t alignedLength = tensor_->getType().strides()[0];                 \
  auto *data = reinterpret_cast<uint8_t *>(tensor_->getUnsafePtr());           \
  for (dim_t i = 0, e = dims()[0]; i < e; i++) {                               \
    uint8_t *scaleOffsetPtr =                                                  \
        data + i * alignedLength + dataWidth - 2 * sizeof(DATA_TYPE);          \
    DATA_TYPE scale, offset;                                                   \
    memcpy(&scale, scaleOffsetPtr, sizeof(DATA_TYPE));                         \
    memcpy(&offset, scaleOffsetPtr + sizeof(DATA_TYPE), sizeof(DATA_TYPE));    \
    for (dim_t j = 0, e = dataWidth - 2 * sizeof(DATA_TYPE); j < e; j++) {     \
      float currVal = (at({i, j}) * (float)scale) + (float)offset;             \
      if (std::abs(currVal) > allowedError) {                                  \
        return false;                                                          \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  return true;

    if (getElementType() == ElemKind::UInt8FusedQTy) {
      RETURN_WHETHER_FUSED_IS_ZERO(float);
    }
    if (getElementType() == ElemKind::UInt8FusedFP16QTy) {
      RETURN_WHETHER_FUSED_IS_ZERO(float16_t);
    }
#undef RETURN_WHETHER_FUSED_IS_ZERO

    int32_t trueZero = getType().isQuantizedType() ? getType().getOffset() : 0;
    return std::all_of(begin(), end(), [=](ElemTy e) { return e == trueZero; });
  }

  void dump(llvm::raw_ostream &os, unsigned maxNumElem = MAX_DUMP_ELEMS) const {
    dumpImpl(tensor_, os, maxNumElem);
  }
  void dump(unsigned maxNumElem) const { dumpImpl(tensor_, maxNumElem); }
  void dump() const { dumpImpl(tensor_, MAX_DUMP_ELEMS); }

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
    assert(getType().isFPType() &&
           "Only support floating point Xavier initialization.");
    double scale = std::sqrt(3.0 / double(filterSize));
    std::uniform_real_distribution<> dist(-scale, scale);
    for (auto &e : *this) {
      e = dist(PRNG);
    }
  }

  /// Fill the tensor with uniformly distributed values in the range
  /// [low .. high).
  template <typename T = ElemTy>
  typename std::enable_if<std::is_floating_point<T>::value>::type
  randomize(float low, float high, PseudoRNG &PRNG) {
    assert(low <= high && "invalid range");
    std::uniform_real_distribution<ElemTy> dist(low, high);
    for (auto &elem : *this) {
      elem = dist(PRNG);
    }
  }

  /// Fill the tensor with uniformly distributed values in the range
  /// [low .. high]. For quantized fused tensors leave scales/offsets unchanged.
  template <typename T = ElemTy>
  typename std::enable_if<std::is_integral<T>::value>::type
  randomize(int low, int high, PseudoRNG &PRNG) {
    assert(low <= high && "invalid range");
    assert(low >= std::numeric_limits<ElemTy>::lowest() &&
           high <= std::numeric_limits<ElemTy>::max() &&
           "Cannot initialize outside range of representable values.");
    std::uniform_int_distribution<long long> dist(low, high);
    switch (getElementType()) {
    default: {
      for (auto &elem : *this) {
        elem = dist(PRNG);
      }
      return;
    }

#define FUSED_CASE(ELEM_KIND, DATA_TYPE)                                       \
  case ElemKind::ELEM_KIND: {                                                  \
    assert(dims().size() == 2 && "Fused tensor must be 2-dimensional.");       \
    assert(dims()[1] > 2 * sizeof(DATA_TYPE) &&                                \
           "Fused tensor must have space for scale/offset.");                  \
    for (dim_t i = 0, e = dims()[0]; i < e; i++) {                             \
      for (dim_t j = 0, f = dims()[1] - 2 * sizeof(DATA_TYPE); j < f; j++) {   \
        at({i, j}) = dist(PRNG);                                               \
      }                                                                        \
    }                                                                          \
    return;                                                                    \
  }
      FUSED_CASE(UInt8FusedQTy, float);
      FUSED_CASE(UInt8FusedFP16QTy, float16_t);
#undef FUSED_CASE
    }
  }

  /// Fill the tensor with uniformly distributed values in the range
  /// [low .. high).
  template <typename T = ElemTy>
  typename std::enable_if<!std::is_floating_point<T>::value &&
                          !std::is_integral<T>::value>::type
  randomize(float low, float high, PseudoRNG &PRNG) {
    assert(low <= high && "invalid range");
    std::uniform_real_distribution<float> dist(low, high);
    for (auto &elem : *this) {
      elem = dist(PRNG);
    }
  }

  /// \returns the mean and variance of the tensor.
  std::pair<double, double> calculateMeanVariance() const {
    size_t n = actualSize();
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
  void insertTensors(Handle<ElemTy> &slice, llvm::ArrayRef<dim_t> offset,
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
  void extractTensors(Handle<ElemTy> &slice, llvm::ArrayRef<dim_t> offset) {
    auto sliceCoor = slice.dims().vec();
    auto fusedCoor = dims().vec();
    insertTensorsImpl(sliceCoor, fusedCoor, slice, false, offset, /* count */ 1,
                      /* axis */ 0, 0);
  }

  /// \returns a pair of the scale and offset from a row \p rowIdx of a
  /// FusedRowwiseQuantized Tensor.
  template <typename T>
  std::pair<T, T> getFusedScaleOffsetFromRow(dim_t rowIdx) {
    ElemTy *rowScaleOffsetPtr = getFusedRowScaleOffsetPtr<T>(rowIdx);
    T scale;
    T offset;
    memcpy(&scale, rowScaleOffsetPtr, sizeof(T));
    memcpy(&offset, rowScaleOffsetPtr + sizeof(T), sizeof(T));
    return std::make_pair(scale, offset);
  }

  /// Sets the \p scale and \p offset to a row \p rowIdx of a
  /// FusedRowwiseQuantized Tensor.
  template <typename T>
  void setFusedScaleOffsetInRow(dim_t rowIdx, T scale, T offset) {
    ElemTy *rowScaleOffsetPtr = getFusedRowScaleOffsetPtr<T>(rowIdx);
    T finalScale = static_cast<T>(scale);
    T finalOffset = static_cast<T>(offset);
    memcpy(rowScaleOffsetPtr, &finalScale, sizeof(T));
    memcpy(rowScaleOffsetPtr + sizeof(T), &finalOffset, sizeof(T));
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
  void insertTensorsImpl(llvm::MutableArrayRef<dim_t> sliceCoor,
                         llvm::MutableArrayRef<dim_t> fusedCoor,
                         Handle<ElemTy> &slice, bool isInsert,
                         llvm::ArrayRef<dim_t> offset, size_t count,
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
        // If this is the correct axis to insert multiple times then calculate
        // the additional offset to use.
        const size_t countAxisOffset = (axis == d) ? c * slice.dims()[d] : 0;
        fusedCoor[d] = i + offset[d] + countAxisOffset;
        insertTensorsImpl(sliceCoor, fusedCoor, slice, isInsert, offset, count,
                          axis, d + 1);
      }
    }
  }

  /// Given a Fused tensor, \returns a pointer to the scale and offset with type
  /// \p T of a row \p rowIdx.
  template <typename T> ElemTy *getFusedRowScaleOffsetPtr(dim_t rowIdx) {
    switch (getElementType()) {
    case ElemKind::UInt8FusedQTy:
    case ElemKind::UInt4FusedQTy: {
      constexpr auto isFloat = std::is_same<float, T>::value;
      DCHECK(isFloat) << "Expected float scale/offset";
      break;
    }
    case ElemKind::UInt4FusedFP16QTy:
    case ElemKind::UInt8FusedFP16QTy: {
      constexpr auto isFloat16 = std::is_same<float16_t, T>::value;
      DCHECK(isFloat16) << "Expected float16_t scale/offset";
      break;
    }
    default:
      llvm_unreachable("Must be used with Tensor of supported Fused ElemKind");
    }

    static_assert(std::is_same<uint8_t, ElemTy>::value,
                  "Handle of current Fused tensors expected to be uint8_t.");
    const dim_t colIdx = dims()[1] - 2 * sizeof(T);
    return &at({rowIdx, colIdx});
  }
};

template <class ElemTy> Handle<ElemTy> Tensor::getHandle() & {
  assert(!isDeviceResident() && "Tensor must reside on host to access data.");
  assert(type_.isType<ElemTy>() && "Getting a handle to the wrong type.");
  return Handle<ElemTy>(this);
}

template <class ElemTy> const Handle<ElemTy> Tensor::getHandle() const & {
  assert(!isDeviceResident() && "Tensor must reside on host to access data.");
  assert(type_.isType<ElemTy>() && "Getting a handle to the wrong type.");
  return Handle<ElemTy>(const_cast<Tensor *>(this));
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Tensor &t);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Tensor *t);
} // namespace glow

#endif // GLOW_BASE_TENSOR_H
