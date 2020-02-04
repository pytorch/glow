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
#ifndef GLOW_BASE_TYPE_H
#define GLOW_BASE_TYPE_H

#include "DimType.h"

#include "glow/Support/Compiler.h"
#include "glow/Support/Float16.h"
#include "glow/Support/Memory.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace llvm {
class raw_ostream;
}

namespace glow {

// UINT8_MIN is not defined in standard headers.
// Define it here for using these definitions consistently.
#define UINT8_MIN 0

struct Type;

using TypeRef = const Type *;

constexpr unsigned max_tensor_dimensions = 6;

/// This type is used to implement the Node and Instruction builder's
/// MemberType::Unsigned and MemberType::VectorUnsigned. Thus it should be used
/// when handling members of these classes, e.g. a convolution Node/Instr's
/// getGroup() (Unsigned), or getKernels() (UnsignedVector).
using unsigned_t = uint32_t;

using float16_t = float16;
static_assert(sizeof(float16_t) == 2, "Half precision should be 16-bit");

using ShapeVector = llvm::SmallVector<dim_t, max_tensor_dimensions>;

struct ShapeNHWC {
  dim_t n; // Number of samples
  dim_t h; // Height
  dim_t w; // Width
  dim_t c; // Number of Channels

  template <typename T> explicit ShapeNHWC(llvm::ArrayRef<T> shape) {
    assert(shape.size() == 4 && "Invalid shape");
    n = shape[0];
    h = shape[1];
    w = shape[2];
    c = shape[3];
  }

  ShapeNHWC(dim_t samples, dim_t height, dim_t width, dim_t channels)
      : n(samples), h(height), w(width), c(channels) {}

  bool equals(const ShapeNHWC &other) const {
    return n == other.n && h == other.h && w == other.w && c == other.c;
  }
};

struct ShapeNHWDC {
  dim_t n; // Number of samples
  dim_t h; // Height
  dim_t w; // Width
  dim_t d; // Depth
  dim_t c; // Number of Channels

  template <typename T> explicit ShapeNHWDC(llvm::ArrayRef<T> shape) {
    assert(shape.size() == 5 && "Invalid shape");
    n = shape[0];
    h = shape[1];
    w = shape[2];
    d = shape[3];
    c = shape[4];
  }

  ShapeNHWDC(size_t samples, size_t height, size_t width, size_t depth,
             size_t channels)
      : n(samples), h(height), w(width), d(depth), c(channels) {}

  bool equals(const ShapeNHWDC &other) const {
    return n == other.n && h == other.h && w == other.w && d == other.d &&
           c == other.c;
  }
};

struct ShapeNCHW {
  dim_t n; // Number of samples
  dim_t c; // Number of Channels
  dim_t h; // Height
  dim_t w; // Width

  explicit ShapeNCHW(llvm::ArrayRef<dim_t> shape) {
    assert(shape.size() == 4 && "Invalid shape");
    n = shape[0];
    c = shape[1];
    h = shape[2];
    w = shape[3];
  }

  ShapeNCHW(dim_t samples, dim_t channels, dim_t height, dim_t width)
      : n(samples), c(channels), h(height), w(width) {}

  bool equals(const ShapeNCHW &other) const {
    return n == other.n && h == other.h && w == other.w && c == other.c;
  }
};

struct PaddingTLBR {
  dim_t top;
  dim_t left;
  dim_t bottom;
  dim_t right;

  template <typename T> explicit PaddingTLBR(llvm::ArrayRef<T> pads) {
    assert(pads.size() == 4 && "Invalid padding");
    top = pads[0];
    left = pads[1];
    bottom = pads[2];
    right = pads[3];
  }

  bool equalPadding() const {
    return top == left && top == bottom && top == right;
  }
};

struct PaddingTLNBRF {
  dim_t top;
  dim_t left;
  dim_t near;
  dim_t bottom;
  dim_t right;
  dim_t far;

  template <typename T> explicit PaddingTLNBRF(llvm::ArrayRef<T> pads) {
    assert(pads.size() == 6 && "Invalid padding");
    top = pads[0];
    left = pads[1];
    near = pads[2];
    bottom = pads[3];
    right = pads[4];
    far = pads[5];
  }

  bool equalPadding() const {
    return top == left && top == bottom && top == right && top == near &&
           top == far;
  }
};

struct ShapeHW {
  dim_t height;
  dim_t width;

  template <typename T> explicit ShapeHW(llvm::ArrayRef<T> shape) {
    assert(shape.size() == 2 && "Invalid shape");
    height = shape[0];
    width = shape[1];
  }

  bool isSquare() const { return height == width; }
};

struct ShapeHWD {
  dim_t height;
  dim_t width;
  dim_t depth;

  template <typename T> explicit ShapeHWD(llvm::ArrayRef<T> shape) {
    assert(shape.size() == 3 && "Invalid shape");
    height = shape[0];
    width = shape[1];
    depth = shape[2];
  }

  bool isCube() const { return height == width && height == depth; }
};

/// Collapse a tensor shape into two sizes: the first n dimensions and the size
/// of the rest of the dimensions. For example, ([7, 3, 4, 2], 1) -> [7, 24]
inline std::pair<dim_t, dim_t> flattenCdr(llvm::ArrayRef<dim_t> dims,
                                          unsigned_t n = 1) {
  assert(1 <= n && n <= dims.size());
  size_t first = dims[0];
  for (unsigned_t i = 1; i < n; i++) {
    first *= dims[i];
  }
  size_t rest = 1;
  for (unsigned_t i = n; i < dims.size(); i++) {
    rest *= dims[i];
  }

  return {first, rest};
}

inline bool operator==(const ShapeNHWC &LHS, const ShapeNHWC &RHS) {
  return LHS.equals(RHS);
}

inline bool operator==(const ShapeNCHW &LHS, const ShapeNCHW &RHS) {
  return LHS.equals(RHS);
}

inline bool operator==(const ShapeNHWDC &LHS, const ShapeNHWDC &RHS) {
  return LHS.equals(RHS);
}

/// An enum representing the type used by the elements of a tensor. The types of
/// Handles for these tensors should match the element kind.
/// When adding new type, note that this enum definition must match with
/// ElemKind definition in Glow/lib/Backends/CPU/libjit/libjit.cpp
enum class ElemKind : unsigned char {
  // 32-bit float type (float)
  FloatTy,
  // 16-bit float type (half, fp16)
  Float16Ty,
  // 8-bit quantized type (int8_t)
  Int8QTy,
  // unsigned 8-bit quantized type (uint8_t)
  UInt8QTy,
  // 16-bit quantized type (int16_t)
  Int16QTy,
  // 32-bit quantized type (int32_t)
  Int32QTy,
  // 32-bit index type (int32_t)
  Int32ITy,
  // 64-bit index type (int64_t)
  Int64ITy,
  // 8-bit quantized type with fused scale/offset (uint8_t)
  UInt8FusedQTy,
  // 8-bit quantized type with fused FP16 scale/offset (uint8_t)
  UInt8FusedFP16QTy,
  // 4-bit quantized type with fused FP16 scale/offset (uint8_t, each byte
  // represents 2 4-bit quantized data)
  UInt4FusedFP16QTy,
  // Bool type (bool)
  BoolTy,
};

constexpr ElemKind IndexElemKind =
    (sizeof(dim_t) == 4) ? ElemKind::Int32ITy : ElemKind::Int64ITy;

/// \returns whether \p e is a quantized ElemKind.
inline bool isQuantizedElemKind(ElemKind e) {
  return e == ElemKind::Int8QTy || e == ElemKind::UInt8QTy ||
         e == ElemKind::Int16QTy || e == ElemKind::Int32QTy ||
         e == ElemKind::UInt8FusedQTy || e == ElemKind::UInt8FusedFP16QTy ||
         e == ElemKind::UInt4FusedFP16QTy;
}

/// \returns whether \p e is a fused quantized ElemKind.
inline bool isFusedQuantizedElemKind(ElemKind e) {
  return e == ElemKind::UInt8FusedQTy || e == ElemKind::UInt8FusedFP16QTy ||
         e == ElemKind::UInt4FusedFP16QTy;
}

/// \returns the scale and offset ElemKind used by the fused ElemKind \p e.
inline ElemKind getScaleOffsetElemKindFromFused(ElemKind e) {
  assert(isFusedQuantizedElemKind(e) && "Must pass Fused ElemKind.");
  if (e == ElemKind::UInt8FusedQTy) {
    return ElemKind::FloatTy;
  }
  return ElemKind::Float16Ty;
}

/// A class that represents a type of a tensor.
struct Type final {
  /// Contains the dimensions (sizes) of the tensor. Ex: [sx, sy, sz, ...].
  dim_t sizes_[max_tensor_dimensions] = {
      0,
  };
  /// Contains the strides for each dimension (in elements). The order should be
  /// the same as in sizes_. In more details, suppose that the tensor is laid
  /// out flat in memory, and some dimensions are aligned. strides_[i] is the
  /// number of elements that needs to be skipped in order to reach the next
  /// plane in the i-th dimension. For example, if the tensor has dimensions
  /// [3, 5, 10] and alignments [3, 32, 1], the strides will be [162, 32, 1].
  dim_t strides_[max_tensor_dimensions] = {
      0,
  };

  /// Contains the number of dimensions used by the tensor.
  unsigned char numSizes_{0};

  /// On quantized tensors, this represents the scale of the values.
  float scale_{0};
  /// On quantized tensors, this represents the offset of the values.
  int32_t offset_{0};

  /// Specifies the element type of the tensor.
  ElemKind elementType_{ElemKind::Int64ITy};

  /// Initialize a new quantized type with \p scale and \p offset.
  Type(ElemKind elemTy, llvm::ArrayRef<dim_t> dims, float scale, int32_t offset)
      : scale_(scale), offset_(offset), elementType_(elemTy) {
    assert(isQuantizedType() && "Only quantized types have a scale and offset");
    ShapeVector alignments(dims.size(), 1);
    initDims(dims, llvm::makeArrayRef(alignments));
  }

  /// Initialize a new non-quantized type.
  Type(ElemKind elemTy, llvm::ArrayRef<dim_t> dims) : elementType_(elemTy) {
    assert(!isQuantizedType() &&
           "Can't initialize quantized types without scale and offset");
    ShapeVector alignments(dims.size(), 1);
    initDims(dims, llvm::makeArrayRef(alignments));
  }

  /// Initialize a new quantized type with \p scale and \p offset.
  Type(ElemKind elemTy, llvm::ArrayRef<dim_t> dims,
       llvm::ArrayRef<dim_t> alignments, float scale, int32_t offset)
      : scale_(scale), offset_(offset), elementType_(elemTy) {
    assert(isQuantizedType() && "Only quantized types have a scale and offset");
    initDims(dims, alignments);
  }

  /// Initialize a new non-quantized type.
  Type(ElemKind elemTy, llvm::ArrayRef<dim_t> dims,
       llvm::ArrayRef<dim_t> alignments)
      : elementType_(elemTy) {
    assert(!isQuantizedType() &&
           "Can't initialize quantized types without scale and offset");
    initDims(dims, alignments);
  }

  /// Reshape existing type. This method takes care of quantized types.
  static Type newShape(const Type &T, llvm::ArrayRef<dim_t> dims) {
    if (T.isQuantizedType()) {
      return Type(T.getElementType(), dims, T.getScale(), T.getOffset());
    } else {
      return Type(T.getElementType(), dims);
    }
  }

  /// Reshape existing type and change alignments.
  static Type newShape(const Type &T, llvm::ArrayRef<dim_t> dims,
                       llvm::ArrayRef<dim_t> alignments) {
    if (T.isQuantizedType()) {
      return Type(T.getElementType(), dims, alignments, T.getScale(),
                  T.getOffset());
    } else {
      return Type(T.getElementType(), dims, alignments);
    }
  }

  /// An empty type.
  Type() = default;

  /// \returns true if \p other is the same type.
  bool isEqual(TypeRef other) const { return isEqual(*other); }

  /// \returns the scale of a quantized type.
  float getScale() const {
    assert(isQuantizedType() && "Can't get the scale of a non-quantized type");
    return scale_;
  }

  /// \returns the offset of a quantized type.
  int32_t getOffset() const {
    assert(isQuantizedType() && "Can't get the offset of a non-quantized type");
    return offset_;
  }

  /// \returns the floating point value range that covers a quantized type (min
  /// first, max second).
  std::pair<float, float> getQuantizedValueRange() const {
    assert(isQuantizedType() &&
           "Can't get the quantized value range of a non-quantized type");

    int64_t low = 0, high = 0;
    switch (elementType_) {
    case ElemKind::Int32QTy: {
      low = INT32_MIN;
      high = INT32_MAX;
      break;
    }
    case ElemKind::Int16QTy: {
      low = INT16_MIN;
      high = INT16_MAX;
      break;
    }
    case ElemKind::Int8QTy: {
      low = INT8_MIN;
      high = INT8_MAX;
      break;
    }
    case ElemKind::UInt8QTy: {
      low = UINT8_MIN;
      high = UINT8_MAX;
      break;
    }
    default:;
    }

    float lowFloat = (low - offset_) * scale_;
    float highFloat = (high - offset_) * scale_;
    return std::make_pair(lowFloat, highFloat);
  }

  /// \returns true if \p other is the same type. If \p allowDifferentShape then
  /// shapes will not be considered as part of the equal comparison.
  bool isEqual(const Type &other, bool allowDifferentShape = false) const {
    // Element type must be the same.
    if (elementType_ != other.elementType_) {
      return false;
    }
    // Must have the same number of sizes.
    if (numSizes_ != other.numSizes_) {
      return false;
    }
    // Sizes must be the same.
    if (!allowDifferentShape) {
      for (size_t i = 0; i < numSizes_; i++) {
        if (sizes_[i] != other.sizes_[i]) {
          return false;
        }
      }

      // Strides must be the same.
      for (size_t i = 0; i < numSizes_; i++) {
        if (strides_[i] != other.strides_[i]) {
          return false;
        }
      }
    }

    // Compare the scale and offset of integers.
    if (isQuantizedType()) {
      if (scale_ != other.scale_ || offset_ != other.offset_) {
        return false;
      }
    }

    return true;
  }

  /// \returns a hash value for this Type. Hashes for Ty1 and Ty2 are equal if
  /// Ty1.isEqual(Ty2).
  llvm::hash_code equals_hash() const {
    return llvm::hash_combine(
        elementType_, dims(),
        // hashing floats is tricky, fall back to std::hash
        std::hash<float>{}(scale_), offset_);
  }

  ElemKind getElementType() const { return elementType_; }

  /// \returns the shape of the tensor.
  llvm::ArrayRef<dim_t> dims() const { return {sizes_, numSizes_}; }

  /// \returns the strides of the tensor.
  llvm::ArrayRef<dim_t> strides() const { return {strides_, numSizes_}; }

  /// \returns the number of elements in the tensor.
  dim_t size() const {
    dim_t s = 1;
    for (unsigned char i = 0; i < numSizes_; i++) {
      s *= dim_t(sizes_[i]);
    }

    return s;
  }

  /// \returns the number of elements in a slice in the tensor. Calculate the
  /// size of the slice starting at \p startDim. For example, the tensor with
  /// the shape [10, 10, 3] and startDim 1 would have the size 30, because this
  /// is the size of the slice [10, 3] that starts at index 1.
  dim_t getSliceSize(unsigned char startDim) const {
    assert(startDim <= numSizes_ && "Invalid start dim");
    dim_t s = 1;
    for (unsigned char i = startDim; i < numSizes_; i++) {
      s *= dim_t(sizes_[i]);
    }
    return s;
  }

  /// \returns true if the templated parameter \p ElemTy matches this type.
  template <class ElemTy> bool isType() const {
    return isType<ElemTy>(elementType_);
  }

  /// \returns true if the templated parameter \p ElemTy matches the type that's
  /// specified by the parameter \p Ty.
  template <class ElemTy> static bool isType(ElemKind Ty) {
    switch (Ty) {
    case ElemKind::FloatTy:
      return std::is_same<ElemTy, float>::value;
    case ElemKind::Float16Ty:
      return std::is_same<ElemTy, float16_t>::value;
    case ElemKind::Int8QTy:
      return std::is_same<ElemTy, int8_t>::value;
    case ElemKind::UInt8QTy:
      return std::is_same<ElemTy, uint8_t>::value;
    case ElemKind::Int16QTy:
      return std::is_same<ElemTy, int16_t>::value;
    case ElemKind::Int32QTy:
      return std::is_same<ElemTy, int32_t>::value;
    case ElemKind::Int32ITy:
      return std::is_same<ElemTy, int32_t>::value;
    case ElemKind::Int64ITy:
      return std::is_same<ElemTy, int64_t>::value;
    case ElemKind::UInt8FusedQTy:
      return std::is_same<ElemTy, uint8_t>::value;
    case ElemKind::UInt8FusedFP16QTy:
      return std::is_same<ElemTy, uint8_t>::value;
    case ElemKind::UInt4FusedFP16QTy:
      return std::is_same<ElemTy, uint8_t>::value;
    case ElemKind::BoolTy:
      return std::is_same<ElemTy, bool>::value;
    }
    LOG(FATAL) << "Invalid type: " << getElementName(Ty).str();
  }

  /// \returns true if the type of this Tensor is one of the quantized types.
  bool isQuantizedType() const { return isQuantizedElemKind(elementType_); }

  /// \returns true if the type of this Tensor is one of the floating point
  /// types.
  bool isFPType() const {
    return getElementType() == ElemKind::FloatTy ||
           getElementType() == ElemKind::Float16Ty;
  }

  /// \return the size of the type element.
  unsigned getElementSize() const { return getElementSize(elementType_); }

  /// \returns the size in bytes for this Tensor.
  size_t getSizeInBytes() const {
    size_t s = getElementSize();
    for (unsigned char i = 0; i < numSizes_; i++) {
      s = std::max<dim_t>(s,
                          size_t(sizes_[i]) * getElementSize() * strides_[i]);
    }
    return s;
  }

  /// \returns the actual number of elements in the tensor taking striding into
  /// account. Since size() does not take striding into account, size() is
  /// always <= actualSize().
  size_t actualSize() const { return getSizeInBytes() / getElementSize(); }

  /// \return the size of the element \p Ty.
  static unsigned getElementSize(ElemKind Ty) {
    switch (Ty) {
    case ElemKind::FloatTy:
      return sizeof(float);
    case ElemKind::Float16Ty:
      return sizeof(float16_t);
    case ElemKind::Int8QTy:
      return sizeof(int8_t);
    case ElemKind::UInt8QTy:
      return sizeof(uint8_t);
    case ElemKind::Int16QTy:
      return sizeof(int16_t);
    case ElemKind::Int32QTy:
      return sizeof(int32_t);
    case ElemKind::Int32ITy:
      return sizeof(int32_t);
    case ElemKind::Int64ITy:
      return sizeof(int64_t);
    case ElemKind::UInt8FusedQTy:
      return sizeof(uint8_t);
    case ElemKind::UInt8FusedFP16QTy:
      return sizeof(uint8_t);
    case ElemKind::UInt4FusedFP16QTy:
      return sizeof(uint8_t);
    case ElemKind::BoolTy:
      return sizeof(bool);
    }
    LOG(FATAL) << "Invalid type: " << getElementName(Ty).str();
  }

  /// \return the textual name of the element.
  llvm::StringRef getElementName() const {
    return getElementName(elementType_);
  }

  /// \return the textual name of the element \p Ty.
  static llvm::StringRef getElementName(ElemKind Ty) {
    static const char *names[] = {
        "float",    "float16",      "i8",           "ui8",
        "i16",      "i32",          "index32",      "index64",
        "ui8fused", "ui8fusedfp16", "ui4fusedfp16", "bool",
    };
    return names[(int)Ty];
  }

  /// Given a string \p str containing the name of an ElemKind from
  /// Type::getElementName, returns the corresponding ElemKind or Error if a
  /// mapping couldn't be found.
  static ElemKind getElementKindFromName(llvm::StringRef str) {
    if (str == Type::getElementName(ElemKind::FloatTy)) {
      return ElemKind::FloatTy;
    } else if (str == Type::getElementName(ElemKind::Float16Ty)) {
      return ElemKind::Float16Ty;
    } else if (str == Type::getElementName(ElemKind::Int8QTy)) {
      return ElemKind::Int8QTy;
    } else if (str == Type::getElementName(ElemKind::UInt8QTy)) {
      return ElemKind::UInt8QTy;
    } else if (str == Type::getElementName(ElemKind::Int16QTy)) {
      return ElemKind::Int16QTy;
    } else if (str == Type::getElementName(ElemKind::Int32QTy)) {
      return ElemKind::Int32QTy;
    } else if (str == Type::getElementName(ElemKind::Int32ITy)) {
      return ElemKind::Int32ITy;
    } else if (str == Type::getElementName(ElemKind::Int64ITy)) {
      return ElemKind::Int64ITy;
    } else if (str == Type::getElementName(ElemKind::UInt8FusedQTy)) {
      return ElemKind::UInt8FusedQTy;
    } else if (str == Type::getElementName(ElemKind::UInt8FusedFP16QTy)) {
      return ElemKind::UInt8FusedFP16QTy;
    } else if (str == Type::getElementName(ElemKind::UInt4FusedFP16QTy)) {
      return ElemKind::UInt4FusedFP16QTy;
    } else if (str == Type::getElementName(ElemKind::BoolTy)) {
      return ElemKind::BoolTy;
    } else {
      LOG(DFATAL) << "Invalid ElemKind string: " << str.str();
      return ElemKind::FloatTy;
    }
  }

  /// Dump a textual representation of the Type into provided output stream.
  void dump(llvm::raw_ostream &out) const;

  /// Dump a textual representation of the Type into default output stream.
  void dump() const;

  /// Dump a textual representation of the Type to std::string.
  std::string toString() const;

private:
  /// Setup the internals of type that store the dimensions. This method is
  /// used by the constructor.
  /// \param dims of the tensor (in elements).
  /// \param alignments of the tensor (in bytes).
  void initDims(llvm::ArrayRef<dim_t> dims, llvm::ArrayRef<dim_t> alignments) {
    assert(dims.size() <= max_tensor_dimensions && "Too many dimensions.");
    assert(dims.size() == alignments.size() &&
           "The number of dimensions and alignments should be the same");
    // Update the tensor strides and sizes based on given dims and alignments.
    // Sizes are simply assigned to dims. And strides are computed as partial
    // product of dims, making sure that each dimension is aligned as required.
    numSizes_ = dims.size();
    if (numSizes_ > 0) {
      // Stride of the last dimension is always 1.
      assert(alignments[numSizes_ - 1] == 1 &&
             "Last dimension must always be aligned.");
      strides_[numSizes_ - 1] = 1;
      sizes_[numSizes_ - 1] = dims[numSizes_ - 1];
    }
    for (int i = numSizes_ - 2; i >= 0; i--) {
      dim_t alignment = alignments[i];
      if (alignment != 1) {
        assert(alignment % getElementSize() == 0 &&
               "Alignment should be a multiple of element size");
        alignment /= getElementSize();
      }
      // All the strides (except for last one) depend on the previous dimension.
      strides_[i] = alignedSize(dims[i + 1] * strides_[i + 1], alignment);
      assert(dims[i] > 0 && "Do not allow a dimension of zero.");
      sizes_[i] = dims[i];
    }
  }

  void initDims(llvm::ArrayRef<dim_t> dims) {
    assert(dims.size() <= max_tensor_dimensions && "Too many dimensions.");
    // Update the tensor sizes.
    for (size_t i = 0, e = dims.size(); i < e; i++) {
      assert(dims[i] > 0 && "Do not allow a dimension of zero.");
      sizes_[i] = dims[i];
    }
    numSizes_ = dims.size();
  }
};

inline bool operator==(const Type &LHS, const Type &RHS) {
  return LHS.isEqual(RHS);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Type &type);
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TypeRef &type);

} // namespace glow

#endif // GLOW_BASE_TYPE_H
