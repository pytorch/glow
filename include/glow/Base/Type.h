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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace llvm {
class raw_ostream;
}

namespace glow {
struct Type;

using TypeRef = const Type *;

constexpr unsigned max_tensor_dimensions = 6;

using ShapeVector = llvm::SmallVector<size_t, max_tensor_dimensions>;

struct ShapeNHWC {
  size_t n; // Number of samples
  size_t h; // Height
  size_t w; // Width
  size_t c; // Number of Channels

  explicit ShapeNHWC(llvm::ArrayRef<size_t> shape) {
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

/// Collapse a tensor shape into two sizes: the first dimension and the size of
/// the rest of the dimensions.
/// For example, [7, 3, 4, 2] -> [7, 24]
inline std::pair<size_t, size_t> flattenCdr(llvm::ArrayRef<size_t> dims) {
  assert(dims.size() > 1);
  size_t first = dims[0];
  size_t rest = dims[1];
  for (size_t i = 2; i < dims.size(); i++) {
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

enum class ElemKind : unsigned char {
  FloatTy,
  Int8QTy,
  Int32QTy,
  IndexTy,
};

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
  ElemKind elementType_{ElemKind::IndexTy};

  /// Initialize a new integer type with \p scale and \p offset.
  Type(ElemKind elemTy, llvm::ArrayRef<size_t> dims, float scale,
       int32_t offset)
      : scale_(scale), offset_(offset), elementType_(elemTy) {
    assert(isQuantizedType() && "Only Integer types have a scale and offset");
    initDims(dims);
  }

  /// Initialize a new float type.
  Type(ElemKind elemTy, llvm::ArrayRef<size_t> dims) : elementType_(elemTy) {
    assert(!isQuantizedType() &&
           "Can't initialize Integer types without scale and offset");
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

  float getScale() const {
    assert(isQuantizedType() && "Can't get the scale of a float type");
    return scale_;
  }

  int32_t getOffset() const {
    assert(isQuantizedType() && "Can't get the offset of a float type");
    return offset_;
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

  ElemKind getElementType() const { return elementType_; }

  /// \returns the shape of the tensor.
  llvm::ArrayRef<size_t> dims() const { return {sizes_, numSizes_}; }

  /// \returns the number of elements in the tensor.
  size_t size() const {
    if (!numSizes_) {
      return 0;
    }

    size_t s = 1;
    for (unsigned i = 0; i < numSizes_; i++) {
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
    case ElemKind::Int8QTy:
      return std::is_same<ElemTy, int8_t>::value;
    case ElemKind::Int32QTy:
      return std::is_same<ElemTy, int32_t>::value;
    case ElemKind::IndexTy:
      return std::is_same<ElemTy, size_t>::value;
    }
    GLOW_UNREACHABLE("Invalid type.");
  }

  /// \returns true if the type of this Tensor is one of the integer types.
  /// Notice that we don't consider IndexTy as an integer because we are not
  /// performing calculations on this type.
  bool isQuantizedType() const { return isType<int8_t>() || isType<int32_t>(); }

  /// \return the size of the type element.
  unsigned getElementSize() const { return getElementSize(elementType_); }

  /// \returns the size in bytes for this Tensor.
  size_t getSizeInBytes() const { return getElementSize() * size(); }

  /// \return the size of the element \p Ty.
  static unsigned getElementSize(ElemKind Ty) {
    switch (Ty) {
    case ElemKind::FloatTy:
      return sizeof(float);
    case ElemKind::Int8QTy:
      return sizeof(int8_t);
    case ElemKind::Int32QTy:
      return sizeof(int32_t);
    case ElemKind::IndexTy:
      return sizeof(size_t);
    }
    GLOW_UNREACHABLE("Invalid type.");
  }

  /// \return the textual name of the element.
  llvm::StringRef getElementName() const {
    return getElementName(elementType_);
  }

  /// \return the textual name of the element \p Ty.
  static llvm::StringRef getElementName(ElemKind Ty) {
    static const char *names[] = {
        "float",
        "i8",
        "i32",
        "index",
    };
    return names[(int)Ty];
  }

private:
  /// Setup the internals of type that store the dimensions. This method is used
  /// by the constructor.
  void initDims(llvm::ArrayRef<size_t> dims) {
    assert(dims.size() < max_tensor_dimensions && "Too many indices");
    // Update the tensor sizes.
    for (size_t i = 0, e = dims.size(); i < e; i++) {
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
