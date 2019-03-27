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

#include "glow/Base/Tensor.h"

#include "llvm/Support/NativeFormatting.h"
#include "llvm/Support/raw_ostream.h"

using namespace glow;

namespace {

/// This is a helper method that's used in the visualization of tensors.
template <class ElemTy> static char valueToChar(ElemTy input) {
  char ch = ' ';
  const double val = input;
  if (val > 0.2) {
    ch = '.';
  }
  if (val > 0.4) {
    ch = ',';
  }
  if (val > 0.6) {
    ch = ':';
  }
  if (val > 0.8) {
    ch = 'o';
  }
  if (val > 1.0) {
    ch = 'O';
  }
  if (val > 1.5) {
    ch = '0';
  }
  if (val > 2.0) {
    ch = '@';
  }
  if (val < -0.1) {
    ch = '-';
  }
  if (val < -0.2) {
    ch = '~';
  }
  if (val < -0.4) {
    ch = '=';
  }
  if (val < -1.0) {
    ch = '#';
  }
  return ch;
}

template <class ElemTy>
static void dumpGenericImpl(Handle<ElemTy> handle, llvm::raw_ostream &os,
                            unsigned maxNumElem) {
  auto shape = handle.dims();
  size_t numDims = shape.size();
  auto &Ty = handle.getType();

  // Check for 0-dimensional tensor.
  if (!numDims) {
    os << "[ Scalar containing: ";
    llvm::write_double(os, handle.raw(0), llvm::FloatStyle::Fixed, 3);
    os << " ]\n";
    return;
  }

  // Output shape.
  os << "shape: ( ";
  for (auto &d : shape) {
    os << d << " ";
  }
  os << ")\n";

  ElemTy mx = handle.raw(0);
  ElemTy mn = handle.raw(0);

  for (auto elem : handle) {
    mx = std::max(mx, elem);
    mn = std::min(mn, elem);
  }

  // Check for zero tensor.
  if (mn == ElemTy(.0) && mx == ElemTy(.0)) {
    os << "[ Zero tensor ]\n";
    return;
  }

  // Output max and min.
  os << "max: ";
  llvm::write_double(os, mx, llvm::FloatStyle::Fixed, 3);
  os << "  min: ";
  llvm::write_double(os, mn, llvm::FloatStyle::Fixed, 3);
  os << "\n";

  os << "[";

  for (size_t i = 0, e = std::min<size_t>(maxNumElem, handle.size()); i < e;
       i++) {

    // Print one open brace at the beginning of every row, slice, and tensor.
    for (size_t j = 0, e = numDims - 1; numDims > 1 && j < e; j++) {
      if (i % Ty.getSliceSize(j + 1) == 0) {
        // This iteration of outer loop is a new row, slice or tensor.
        os << "[";
      }
    }

    // Print the value at the current index.
    llvm::write_double(os, handle.raw(i), llvm::FloatStyle::Fixed, 3);

    // Print one closed brace at the end of every row, slice, or tensor.
    for (size_t j = 0, e = numDims - 1; numDims > 1 && j < e; j++) {
      size_t next_index = i + 1;
      if (next_index % Ty.getSliceSize(j + 1) == 0u) {
        os << "]";
      }
    }

    os << ", ";

    // Print one newline at the end of every row, slice, or tensor.
    for (size_t j = 0, e = numDims - 1; numDims > 1 && j < e; j++) {
      size_t next_index = i + 1;
      if (next_index % Ty.getSliceSize(j + 1) == 0u) {
        // Next iteration of outer loop will be a new row, slice or tensor.
        os << "\n";
      }
    }
  }

  if (handle.size() > maxNumElem) {
    os << "...";
  }

  os << "]\n";
}

template <class ElemTy>
static void dumpAsciiGenericImpl(Handle<ElemTy> handle, llvm::raw_ostream &os) {
  auto d = handle.dims();

  if (d.size() == 2) {
    for (size_t x = 0; x < d[0]; x++) {
      for (size_t y = 0; y < d[1]; y++) {
        auto val = handle.at({x, y});
        os << valueToChar(val);
      }
      os << "\n";
    }
  } else if (d.size() == 3) {
    // Print monochrome (one-color channel) tensors:
    if (d[2] == 1) {
      for (size_t x = 0; x < d[0]; x++) {
        for (size_t y = 0; y < d[1]; y++) {
          auto val = handle.at({x, y, 0});
          os << valueToChar(val);
        }
        os << "\n";
      }
    } else {
      for (size_t z = 0; z < d[2]; z++) {
        os << "\n";
        for (size_t x = 0; x < d[0]; x++) {
          for (size_t y = 0; y < d[1]; y++) {
            auto val = handle.at({x, y, z});
            os << valueToChar(val);
          }
          os << "\n";
        }
      }
    }

  } else {
    llvm_unreachable("Invalid tensor size");
  }
}

/// This is a slow generic transpose. This method performs a single for loop
/// over a single dimension, or if we've reached the last dimension perform a
/// single copy of a single element.
template <class ElemTy>
static void
transposeGenericImpl(const Handle<ElemTy> &src, Handle<ElemTy> &dest,
                     size_t *srcCoor, size_t *destCoor,
                     llvm::ArrayRef<unsigned_t> shuffle, unsigned depth = 0) {
  if (depth == shuffle.size()) {
    auto srcIdx = llvm::ArrayRef<size_t>(srcCoor, depth);
    auto destIdx = llvm::ArrayRef<size_t>(destCoor, depth);
    dest.at(destIdx) = src.at(srcIdx);
    return;
  }

  // Iterate over one dimension and continue recursively to the next dim.
  for (size_t x = 0, e = dest.dims()[depth]; x < e; x++) {
    unsigned_t swizzledDepth = shuffle[depth];
    srcCoor[swizzledDepth] = x;
    destCoor[depth] = x;
    transposeGenericImpl(src, dest, srcCoor, destCoor, shuffle, depth + 1);
  }
}

/// Faster function for transposing a tensor for important/common tensor
/// shapes. If a transpose successfully occurs, the function \returns true;
/// otherwise it \returns false, representing no transpose occurred and some
/// other transpose function (e.g. transposeGenericImpl) must be called. \p
/// dest is the tensor to transpose, and \p shuffle defines how to transpose.
template <class ElemTy>
static bool tryTransposeFastImpl(const Handle<ElemTy> &src,
                                 Handle<ElemTy> &dest,
                                 llvm::ArrayRef<unsigned_t> shuffle) {
  const size_t numDims = dest.dims().size();
  size_t srcCoorArr[max_tensor_dimensions];
  size_t destCoorArr[max_tensor_dimensions] = {0};
  auto srcCoor = llvm::ArrayRef<size_t>(srcCoorArr, numDims);
  auto destCoor = llvm::ArrayRef<size_t>(destCoorArr, numDims);

  /// This defines a single depth of the for loop used to iterate over the
  /// source and destination tensors for transposing.
#define TRANSPOSE_LOOP_LEVEL(DEPTH_)                                           \
  for (srcCoorArr[shuffle[DEPTH_]] = 0, destCoorArr[DEPTH_] = 0;               \
       destCoorArr[DEPTH_] < dest.dims()[DEPTH_];                              \
       srcCoorArr[shuffle[DEPTH_]]++, destCoorArr[DEPTH_]++)

  switch (numDims) {
  case 2:
    TRANSPOSE_LOOP_LEVEL(1) {
      TRANSPOSE_LOOP_LEVEL(0) { dest.at(destCoor) = src.at(srcCoor); }
    }
    return true;
  case 4:
    TRANSPOSE_LOOP_LEVEL(1) {
      TRANSPOSE_LOOP_LEVEL(2) {
        TRANSPOSE_LOOP_LEVEL(0) {
          TRANSPOSE_LOOP_LEVEL(3) { dest.at(destCoor) = src.at(srcCoor); }
        }
      }
    }
    return true;
  }
  return false;
}

template <class ElemTy>
static void transposeSelectImpl(const Handle<ElemTy> &src, Handle<ElemTy> &dest,
                                llvm::ArrayRef<unsigned_t> shuffle) {
  bool transposeOccurred = tryTransposeFastImpl(src, dest, shuffle);
  if (!transposeOccurred) {
    size_t srcCoor[max_tensor_dimensions];
    size_t destCoor[max_tensor_dimensions];
    transposeGenericImpl(src, dest, srcCoor, destCoor, shuffle);
  }
}
} // namespace

void glow::dumpAsciiImpl(const Tensor *T, llvm::raw_ostream &os) {
  switch (T->getElementType()) {
  case ElemKind::FloatTy:
    return dumpAsciiGenericImpl(T->getHandle<float>(), os);
  case ElemKind::Float16Ty:
    return dumpAsciiGenericImpl(T->getHandle<float16_t>(), os);
  case ElemKind::Int8QTy:
    return dumpAsciiGenericImpl(T->getHandle<int8_t>(), os);
  case ElemKind::Int16QTy:
    return dumpAsciiGenericImpl(T->getHandle<int16_t>(), os);
  case ElemKind::Int32QTy:
    return dumpAsciiGenericImpl(T->getHandle<int32_t>(), os);
  case ElemKind::Int32ITy:
    return dumpAsciiGenericImpl(T->getHandle<int32_t>(), os);
  case ElemKind::Int64ITy:
    return dumpAsciiGenericImpl(T->getHandle<int64_t>(), os);
  case ElemKind::UInt8FusedQTy:
    return dumpAsciiGenericImpl(T->getHandle<uint8_t>(), os);
  case ElemKind::BoolTy:
    return dumpAsciiGenericImpl(T->getHandle<bool>(), os);
  }
}

void glow::dumpAsciiImpl(const Tensor *T) { dumpAsciiImpl(T, llvm::outs()); }

void glow::dumpImpl(const Tensor *T, llvm::raw_ostream &os,
                    unsigned maxNumElem) {
  switch (T->getElementType()) {
  case ElemKind::FloatTy:
    return dumpGenericImpl(T->getHandle<float>(), os, maxNumElem);
  case ElemKind::Float16Ty:
    return dumpGenericImpl(T->getHandle<float16_t>(), os, maxNumElem);
  case ElemKind::Int8QTy:
    return dumpGenericImpl(T->getHandle<int8_t>(), os, maxNumElem);
  case ElemKind::Int16QTy:
    return dumpGenericImpl(T->getHandle<int16_t>(), os, maxNumElem);
  case ElemKind::Int32QTy:
    return dumpGenericImpl(T->getHandle<int32_t>(), os, maxNumElem);
  case ElemKind::Int32ITy:
    return dumpGenericImpl(T->getHandle<int32_t>(), os, maxNumElem);
  case ElemKind::Int64ITy:
    return dumpGenericImpl(T->getHandle<int64_t>(), os, maxNumElem);
  case ElemKind::UInt8FusedQTy:
    return dumpGenericImpl(T->getHandle<uint8_t>(), os, maxNumElem);
  case ElemKind::BoolTy:
    return dumpGenericImpl(T->getHandle<bool>(), os, maxNumElem);
  }
}

void glow::dumpImpl(const Tensor *T, unsigned maxNumElem) {
  dumpImpl(T, llvm::outs(), maxNumElem);
}

void glow::dumpImpl(const Tensor *T) { dumpImpl(T, llvm::outs()); }

void glow::genericTranspose(const Tensor *src, Tensor *dest,
                            llvm::ArrayRef<unsigned_t> shuffle) {
  assert(src->dims().size() == shuffle.size() && "Invalid dimensions");

  size_t newSizes[max_tensor_dimensions];

  // Generate the swizzled dimensions.
  auto origDims = src->dims();
  for (unsigned i = 0; i < origDims.size(); i++) {
    newSizes[i] = origDims[shuffle[i]];
  }

  // Resize the tensor to the transposed shape.
  auto destType = Type::newShape(src->getType(), {newSizes, origDims.size()});
  dest->reset(destType);

  switch (src->getElementType()) {
  case ElemKind::FloatTy: {
    auto srcH = src->getHandle<float>();
    auto destH = dest->getHandle<float>();
    transposeSelectImpl(srcH, destH, shuffle);
    return;
  }
  case ElemKind::Float16Ty: {
    auto srcH = src->getHandle<float16_t>();
    auto destH = dest->getHandle<float16_t>();
    transposeSelectImpl(srcH, destH, shuffle);
    return;
  }
  case ElemKind::Int8QTy: {
    auto srcH = src->getHandle<int8_t>();
    auto destH = dest->getHandle<int8_t>();
    transposeSelectImpl(srcH, destH, shuffle);
    return;
  }
  case ElemKind::Int16QTy: {
    auto srcH = src->getHandle<int16_t>();
    auto destH = dest->getHandle<int16_t>();
    transposeSelectImpl(srcH, destH, shuffle);
    return;
  }
  case ElemKind::Int32QTy: {
    auto srcH = src->getHandle<int32_t>();
    auto destH = dest->getHandle<int32_t>();
    transposeSelectImpl(srcH, destH, shuffle);
    return;
  }
  case ElemKind::Int32ITy: {
    auto srcH = src->getHandle<int32_t>();
    auto destH = dest->getHandle<int32_t>();
    transposeSelectImpl(srcH, destH, shuffle);
    return;
  }
  case ElemKind::Int64ITy: {
    auto srcH = src->getHandle<int64_t>();
    auto destH = dest->getHandle<int64_t>();
    transposeSelectImpl(srcH, destH, shuffle);
    return;
  }
  case ElemKind::UInt8FusedQTy: {
    llvm_unreachable("Transposing UInt8FusedQTy is unsupported.");
  }
  case ElemKind::BoolTy: {
    auto srcH = src->getHandle<bool>();
    auto destH = dest->getHandle<bool>();
    transposeSelectImpl(srcH, destH, shuffle);
    return;
  }
  }
}

ShapeVector glow::expandDimsToMax(llvm::ArrayRef<size_t> currDims) {
  ShapeVector newDims(currDims.begin(), currDims.end());
  for (size_t i = newDims.size(); i < max_tensor_dimensions; i++) {
    newDims.push_back(1);
  }
  return newDims;
}

void Tensor::init(InitKind init, float val, PseudoRNG &PRNG) {
  switch (init) {
  case InitKind::Zero:
    zero();
    break;

  case InitKind::Broadcast: {
    switch (getElementType()) {
    case ElemKind::FloatTy: {
      getHandle<float>().clear(val);
      break;
    }
    case ElemKind::Float16Ty: {
      getHandle<float16_t>().clear(float16_t(val));
      break;
    }
    case ElemKind::Int8QTy: {
      getHandle<int8_t>().clear(val);
      break;
    }
    case ElemKind::Int16QTy: {
      getHandle<int16_t>().clear(val);
      break;
    }
    case ElemKind::Int32QTy: {
      getHandle<int32_t>().clear(val);
      break;
    }
    case ElemKind::Int32ITy: {
      getHandle<int32_t>().clear(val);
      break;
    }
    case ElemKind::Int64ITy: {
      getHandle<int64_t>().clear(val);
      break;
    }
    case ElemKind::UInt8FusedQTy: {
      assert(dims().size() == 2 && "Fused tensor must be 2-dimensional.");
      assert(dims()[1] > 8 && "Fused tensor must have more than 8 columns.");
      auto H = getHandle<uint8_t>();
      for (size_t i = 0; i < dims()[0]; i++) {
        for (size_t j = 0, f = dims()[1] - 8; j < f; j++) {
          H.at({i, j}) = val;
        }
      }
      break;
    }
    case ElemKind::BoolTy: {
      getHandle<bool>().clear(val);
      break;
    }
    }
    break;
  }

  case InitKind::Xavier: {
    switch (getElementType()) {
    case ElemKind::FloatTy: {
      getHandle<float>().initXavier(val, PRNG);
      break;
    }
    case ElemKind::Float16Ty: {
      getHandle<float16_t>().initXavier(val, PRNG);
      break;
    }
    default: {
      llvm_unreachable("Undefined to Xavier-initialize non-Float Tensors.");
    }
    }
    break;
  }
  }
}

void Tensor::convertToType(ElemKind newTy) {
  Tensor tmp(newTy, dims());
  switch (newTy) {
  case ElemKind::Float16Ty:
    assert(getElementType() == ElemKind::FloatTy && "Cast not implemented");
    tmp.copyWithCast<float16_t, float>(this);
    break;
  case ElemKind::FloatTy:
    assert(getElementType() == ElemKind::Float16Ty && "Cast not implemented");
    tmp.copyWithCast<float, float16_t>(this);
    break;
  default:
    llvm_unreachable("Type not supported");
  }
  *this = std::move(tmp);
}
