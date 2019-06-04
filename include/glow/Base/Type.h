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
#ifndef GLOW_BASE_TYPE_H
#define GLOW_BASE_TYPE_H

#include "glow/Support/Compiler.h"

#include "glow/Support/Float16.h"

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

using ShapeVector = llvm::SmallVector<size_t, max_tensor_dimensions>;

struct ShapeNHWC {
  size_t n; // Number of samples
  size_t h; // Height
  size_t w; // Width
  size_t c; // Number of Channels

  template <typename T> explicit ShapeNHWC(llvm::ArrayRef<T> shape) {
    assert(shape.size() == 4 && "Invalid shape");
    n = shape[0];
    h = shape[1];
    w = shape[2];
    c = shape[3];
  }

  static ShapeNHWC fromXYZ(llvm::ArrayRef<size_t> shape) {
    assert(shape.size() == 3 && "Invalid 3d shape");
    return ShapeNHWC(shape[0], shape[1], shape[2], 1);
  }

  static ShapeNHWC fromXY(llvm::ArrayRef<size_t> shape) {
    assert(shape.size() == 2 && "Invalid 2d shape");
    return ShapeNHWC(shape[0], shape[1], 1, 1);
  }

  static ShapeNHWC fromX(llvm::ArrayRef<size_t> shape) {
    assert(shape.size() == 1 && "Invalid 1d shape");
    return ShapeNHWC(shape[0], 1, 1, 1);
  }

  static ShapeNHWC empty() { return ShapeNHWC(0, 0, 0, 0); }

  explicit ShapeNHWC(size_t samples, size_t height, size_t width,
                     size_t channels)
      : n(samples), h(height), w(width), c(channels) {}

  bool equals(const ShapeNHWC &other) const {
    return n == other.n && h == other.h && w == other.w && c == other.c;
  }
};

struct ShapeNHWDC {
  size_t n; // Number of samples
  size_t h; // Height
  size_t w; // Width
  size_t d; // Depth
  size_t c; // Number of Channels

  template <typename T> explicit ShapeNHWDC(llvm::ArrayRef<T> shape) {
    assert(shape.size() == 5 && "Invalid shape");
    n = shape[0];
    h = shape[1];
    w = shape[2];
    d = shape[3];
    c = shape[4];
  }

  static ShapeNHWDC empty() { return ShapeNHWDC(0, 0, 0, 0, 0); }

  explicit ShapeNHWDC(size_t samples, size_t height, size_t width, size_t depth,
                      size_t channels)
      : n(samples), h(height), w(width), d(depth), c(channels) {}

  bool equals(const ShapeNHWDC &other) const {
    return n == other.n && h == other.h && w == other.w && d == other.d &&
           c == other.c;
  }
};

struct ShapeNCHW {
  size_t n; // Number of samples
  size_t c; // Number of Channels
  size_t h; // Height
  size_t w; // Width

  explicit ShapeNCHW(llvm::ArrayRef<size_t> shape) {
    assert(shape.size() == 4 && "Invalid shape");
    n = shape[0];
    c = shape[1];
    h = shape[2];
    w = shape[3];
  }

  static ShapeNCHW fromXYZ(llvm::ArrayRef<size_t> shape) {
    assert(shape.size() == 3 && "Invalid 3d shape");
    return ShapeNCHW(shape[0], 1, shape[1], shape[2]);
  }

  static ShapeNCHW fromXY(llvm::ArrayRef<size_t> shape) {
    assert(shape.size() == 2 && "Invalid 2d shape");
    return ShapeNCHW(shape[0], 1, shape[1], 1);
  }

  static ShapeNCHW empty() { return ShapeNCHW(0, 0, 0, 0); }

  explicit ShapeNCHW(size_t samples, size_t channels, size_t height,
                     size_t width)
      : n(samples), c(channels), h(height), w(width) {}

  bool equals(const ShapeNCHW &other) const {
    return n == other.n && h == other.h && w == other.w && c == other.c;
  }
};

struct PaddingTLBR {
  size_t top;
  size_t left;
  size_t bottom;
  size_t right;

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
  size_t top;
  size_t left;
  size_t near;
  size_t bottom;
  size_t right;
  size_t far;

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
  size_t height;
  size_t width;

  template <typename T> explicit ShapeHW(llvm::ArrayRef<T> shape) {
    assert(shape.size() == 2 && "Invalid shape");
    height = shape[0];
    width = shape[1];
  }

  bool isSquare() const { return height == width; }
};

struct ShapeHWD {
  size_t height;
  size_t width;
  size_t depth;

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
inline std::pair<size_t, size_t> flattenCdr(llvm::ArrayRef<size_t> dims,
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
  FloatTy,       // 32-bit float type (float)
  Float16Ty,     // 16-bit float type (half, fp16)
  Int8QTy,       // 8-bit quantized type (int8_t)
  UInt8QTy,      // unsigned 8-bit quantized type (uint8_t)
  Int16QTy,      // 16-bit quantized type (int16_t)
  Int32QTy,      // 32-bit quantized type (int32_t)
  Int32ITy,      // 32-bit index type (int32_t)
  Int64ITy,      // 64-bit index type (int64_t)
  UInt8FusedQTy, // 8-bit quantized type with fused scale/offset (uint8_t)
  BoolTy,        // Bool type (bool)
};

/// \returns whether \p e is a quantized ElemKind.
inline bool isQuantizedElemKind(ElemKind e) {
  return e == ElemKind::Int8QTy || e == ElemKind::UInt8QTy ||
         e == ElemKind::Int16QTy || e == ElemKind::Int32QTy ||
         e == ElemKind::UInt8FusedQTy;
}

/// A class that represents a type of a tensor.
struct Type final {
  /// Contains the dimensions (sizes) of the tensor. Ex: [sx, sy, sz, ...].
  size_t sizes_[max_tensor_dimensions] = {
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
  Type(ElemKind elemTy, llvm::ArrayRef<size_t> dims, float scale,
       int32_t offset)
      : scale_(scale), offset_(offset), elementType_(elemTy) {
    assert(isQuantizedType() && "Only quantized types have a scale and offset");
    initDims(dims);
  }

  /// Initialize a new non-quantized type.
  Type(ElemKind elemTy, llvm::ArrayRef<size_t> dims) : elementType_(elemTy) {
    assert(!isQuantizedType() &&
           "Can't initialize quantized types without scale and offset");
    initDims(dims);
  }

  /// Reshape existing type. This method takes care of quantized types.
  static Type newShape(const Type &T, llvm::ArrayRef<size_t> dims) {
    if (T.isQuantizedType()) {
      return Type(T.getElementType(), dims, T.getScale(), T.getOffset());
    } else {
      return Type(T.getElementType(), dims);
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

  /// \returns true if \p other is the same type.
  bool isEqual(const Type &other) const {
    // Element type must be the same.
    if (elementType_ != other.elementType_) {
      return false;
    }
    // Must have the same number of sizes.
    if (numSizes_ != other.numSizes_) {
      return false;
    }
    // Sizes must be the same.
    for (size_t i = 0; i < numSizes_; i++) {
      if (sizes_[i] != other.sizes_[i]) {
        return false;
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
  llvm::ArrayRef<size_t> dims() const { return {sizes_, numSizes_}; }

  /// \returns the number of elements in the tensor.
  size_t size() const {
    size_t s = 1;
    for (unsigned char i = 0; i < numSizes_; i++) {
      s *= size_t(sizes_[i]);
    }

    return s;
  }

  /// \returns the number of elements in a slice in the tensor. Calculate the
  /// size of the slice starting at \p startDim. For example, the tensor with
  /// the shape [10, 10, 3] and startDim 1 would have the size 30, because this
  /// is the size of the slice [10, 3] that starts at index 1.
  size_t getSliceSize(unsigned char startDim) const {
    assert(startDim <= numSizes_ && "Invalid start dim");
    size_t s = 1;
    for (unsigned char i = startDim; i < numSizes_; i++) {
      s *= size_t(sizes_[i]);
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
  size_t getSizeInBytes() const { return getElementSize() * size(); }

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
        "float", "float16", "i8",      "ui8",      "i16",
        "i32",   "index32", "index64", "ui8fused", "bool",
    };
    return names[(int)Ty];
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
  void initDims(llvm::ArrayRef<size_t> dims) {
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
