#ifndef GLOW_IR_TYPE_H
#define GLOW_IR_TYPE_H

#include "glow/Support/ADT.h"
#include "glow/Support/Compiler.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace glow {
struct Type;

using TypeRef = const Type *;

constexpr unsigned max_tensor_dimensions = 6;

struct ShapeNHWC {
  size_t n;
  size_t h;
  size_t w;
  size_t c;
  ShapeNHWC(ArrayRef<size_t> shape) {
    assert(shape.size() == 4 && "Invalid shape");
    n = shape[0];
    h = shape[1];
    w = shape[2];
    c = shape[3];
  }

  bool equals(const ShapeNHWC &other) const {
    return n == other.n && h == other.h && w == other.w && c == other.c;
  }
};

/// Colllapse a tensor shape into two sizes: the first dimension and the size of
/// the rest of the dimensions.
/// For example, [7, 3, 4, 2] -> [7, 24]
inline std::pair<size_t, size_t> flattenCdr(ArrayRef<size_t> dims) {
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

enum class ElemKind : unsigned char {
  FloatTy,
  DoubleTy,
  Int8Ty,
  Int32Ty,
  IndexTy,
};

/// A class that represents a type of a tensor.
struct Type final {
  /// Contains the dimentions (sizes) of the tensor. Ex: [sx, sy, sz, ...].
  size_t sizes_[max_tensor_dimensions] = {
      0,
  };

  /// Contains the number of dimensions used by the tensor.
  unsigned char numSizes_{0};

  /// Specifies the element type of the tensor.
  ElemKind elementType_;

  /// Initialize a new type.
  Type(ElemKind elemTy, ArrayRef<size_t> dims) : elementType_(elemTy) {
    assert(dims.size() < max_tensor_dimensions && "Too many indices");

    // Update the tensor sizes.
    for (size_t i = 0, e = dims.size(); i < e; i++) {
      sizes_[i] = dims[i];
    }
    numSizes_ = dims.size();
  }

  /// Print the textual representation of the type.
  std::string asString() const;

  /// An empty type.
  Type() : elementType_(ElemKind::IndexTy) { numSizes_ = 0; }

  /// \returns true if \p other is the same type.
  bool isEqual(TypeRef other) const { return isEqual(*other); }

  /// \returns true if \p other is the same type.
  bool isEqual(const Type &other) const {
    // Element type must be the same.
    if (elementType_ != other.elementType_)
      return false;
    // Must have the same number of sizes.
    if (numSizes_ != other.numSizes_)
      return false;
    // Sizes must be the same.
    for (size_t i = 0; i < numSizes_; i++) {
      if (sizes_[i] != other.sizes_[i])
        return false;
    }

    return true;
  }

  ElemKind getElementType() const { return elementType_; }

  /// \returns the shape of the tensor.
  ArrayRef<size_t> dims() const { return {sizes_, numSizes_}; }

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
    case ElemKind::DoubleTy:
      return std::is_same<ElemTy, double>::value;
    case ElemKind::Int8Ty:
      return std::is_same<ElemTy, int8_t>::value;
    case ElemKind::Int32Ty:
      return std::is_same<ElemTy, int32_t>::value;
    case ElemKind::IndexTy:
      return std::is_same<ElemTy, size_t>::value;
    }
    glow_unreachable();
  }

  /// \return the size of the type element.
  unsigned getElementSize() const { return getElementSize(elementType_); }

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
    glow_unreachable();
  }

  /// \return the textual name of the element.
  StringRef getElementName() const { return getElementName(elementType_); }

  /// \return the textual name of the element \p Ty.
  static StringRef getElementName(ElemKind Ty) {
    const char *names[] = {
        "float", "double", "i8", "i32", "index",
    };
    return names[(int)Ty];
  }
};

inline bool operator==(const Type &LHS, const Type &RHS) {
  return LHS.isEqual(RHS);
}

} // namespace glow

#endif // GLOW_IR_TYPE_H
