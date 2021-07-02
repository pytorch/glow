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

#include "glow/Base/Tensor.h"

#include "glow/Base/Type.h"

#include "llvm/Support/NativeFormatting.h"
#include "llvm/Support/raw_ostream.h"
#include <glog/logging.h>

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

static void dumpShape(llvm::ArrayRef<dim_t> shape, llvm::raw_ostream &os) {
  os << "shape: ( ";
  for (auto &d : shape) {
    os << d << " ";
  }
  os << ")";
}

template <class ElemTy>
static void dumpGenericImpl(Handle<ElemTy> handle, llvm::raw_ostream &os,
                            unsigned maxNumElem) {
  auto shape = handle.dims();
  size_t numDims = shape.size();
  auto &Ty = handle.getType();

  constexpr unsigned numDigsFP = 5;
  const unsigned numDigs = std::is_integral<ElemTy>::value ? 0 : numDigsFP;

  // Check for 0-dimensional tensor.
  if (!numDims) {
    os << "[ Scalar containing: ";
    llvm::write_double(os, handle.raw(0), llvm::FloatStyle::Fixed, numDigs);
    os << " ]\n";
    return;
  }

  const size_t numRealElems = handle.getRealNumElements();

  // Output shape.
  dumpShape(shape, os);
  if (numRealElems < handle.size()) {
    os << " ; partial num elements: " << numRealElems;
  }
  os << "\n";

  // Output ElemKind.
  os << "elemkind: " << Ty.getElementName() << "\n";

  // Check for tensor of size 0.
  if (handle.getUnpaddedSizeInBytes() == 0) {
    os << "[ tensor has no elements ]\n";
    return;
  }

  ElemTy mx = handle.raw(0);
  ElemTy mn = handle.raw(0);
  double avg = 0.0f;

  for (auto elem : handle) {
    mx = std::max(mx, elem);
    mn = std::min(mn, elem);
    avg += (double)elem;
  }
  avg /= numRealElems;

  // Check for zero tensor.
  if (mn == ElemTy(.0) && mx == ElemTy(.0)) {
    os << "[ Zero tensor ]\n";
    return;
  }

  // Output max and min.
  os << "max: ";
  llvm::write_double(os, mx, llvm::FloatStyle::Fixed, numDigs);
  os << "  min: ";
  llvm::write_double(os, mn, llvm::FloatStyle::Fixed, numDigs);
  os << "  avg: ";
  llvm::write_double(os, avg, llvm::FloatStyle::Fixed, numDigsFP);
  os << "\n";

  os << "[";

  for (size_t i = 0, e = std::min<size_t>(maxNumElem, numRealElems); i < e;
       i++) {

    // Print one open brace at the beginning of every row, slice, and tensor.
    for (size_t j = 0, e = numDims - 1; numDims > 1 && j < e; j++) {
      if (i % Ty.getSliceSize(j + 1) == 0) {
        // This iteration of outer loop is a new row, slice or tensor.
        os << "[";
      }
    }

    // Print the value at the current index.
    llvm::write_double(os, handle.raw(i), llvm::FloatStyle::Fixed, numDigs);

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

  if (numRealElems > maxNumElem) {
    os << "...";
  }

  os << "]\n";

  os.flush();
}

template <class ElemTy>
static void dumpAsciiGenericImpl(Handle<ElemTy> handle, llvm::raw_ostream &os) {
  auto d = handle.dims();

  if (d.size() == 2) {
    for (dim_t x = 0; x < d[0]; x++) {
      for (dim_t y = 0; y < d[1]; y++) {
        auto val = handle.at({x, y});
        os << valueToChar(val);
      }
      os << "\n";
    }
  } else if (d.size() == 3) {
    // Print monochrome (one-color channel) tensors:
    if (d[2] == 1) {
      for (dim_t x = 0; x < d[0]; x++) {
        for (dim_t y = 0; y < d[1]; y++) {
          auto val = handle.at({x, y, 0});
          os << valueToChar(val);
        }
        os << "\n";
      }
    } else {
      for (dim_t z = 0; z < d[2]; z++) {
        os << "\n";
        for (dim_t x = 0; x < d[0]; x++) {
          for (dim_t y = 0; y < d[1]; y++) {
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

  os.flush();
}

/// This is a slow generic transpose. This method performs a single for loop
/// over a single dimension, or if we've reached the last dimension perform a
/// single copy of a single element.
template <class ElemTy>
static void
transposeGenericImpl(const Handle<ElemTy> &src, Handle<ElemTy> &dest,
                     dim_t *srcCoor, dim_t *destCoor,
                     llvm::ArrayRef<unsigned_t> shuffle, unsigned depth = 0) {
  if (depth == shuffle.size()) {
    auto srcIdx = llvm::ArrayRef<dim_t>(srcCoor, depth);
    auto destIdx = llvm::ArrayRef<dim_t>(destCoor, depth);
    dest.at(destIdx) = src.at(srcIdx);
    return;
  }

  // Iterate over one dimension and continue recursively to the next dim.
  for (dim_t x = 0, e = dest.dims()[depth]; x < e; x++) {
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
  const dim_t numDims = dest.dims().size();
  dim_t srcCoorArr[max_tensor_dimensions];
  dim_t destCoorArr[max_tensor_dimensions] = {0};
  auto srcCoor = llvm::ArrayRef<dim_t>(srcCoorArr, numDims);
  auto destCoor = llvm::ArrayRef<dim_t>(destCoorArr, numDims);

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
    dim_t srcCoor[max_tensor_dimensions];
    dim_t destCoor[max_tensor_dimensions];
    transposeGenericImpl(src, dest, srcCoor, destCoor, shuffle);
  }
}

template <class ElemTy>
static bool isTiledImpl(const Tensor *tensor, unsigned_t axis, dim_t size,
                        bool fractional) {
  assert(axis < tensor->dims().size() && "Axis parameter invalid!");
  assert(size <= tensor->dims()[axis] && "Size parameter invalid!");
  assert(size >= 1 && "Size parameter invalid!");

  // When the tile size matches the dimension size then we return true.
  // This is because a tensor can be considered a tiled version of itself.
  if (size == tensor->dims()[axis]) {
    return true;
  }

  // If fractional tiling verification is disabled and the dimension size
  // is NOT divisible by the tile size then we return false.
  if (!fractional && ((tensor->dims()[axis] % size) != 0)) {
    return false;
  }

  static_assert(max_tensor_dimensions == 6,
                "Implementation assumes max_tensor_dimensions = 6.");

  // Get tensor view with maximum number of dimensions.
  auto dimsMax = expandDimsToMax(tensor->dims());
  Tensor tensorMax = tensor->getUnowned(dimsMax);
  auto tensorH = tensorMax.getHandle<ElemTy>();
  for (dim_t idx0 = 0; idx0 < dimsMax[0]; ++idx0) {
    for (dim_t idx1 = 0; idx1 < dimsMax[1]; ++idx1) {
      for (dim_t idx2 = 0; idx2 < dimsMax[2]; ++idx2) {
        for (dim_t idx3 = 0; idx3 < dimsMax[3]; ++idx3) {
          for (dim_t idx4 = 0; idx4 < dimsMax[4]; ++idx4) {
            for (dim_t idx5 = 0; idx5 < dimsMax[5]; ++idx5) {
              std::vector<dim_t> idx = {idx0, idx1, idx2, idx3, idx4, idx5};
              std::vector<dim_t> idxWrapped = idx;
              idxWrapped[axis] = (idx[axis] % size);
              double delta = tensorH.at(idx) - tensorH.at(idxWrapped);
              // Since any comparison with NAN returns false, we use a negated
              // condition so that this function correctly returns false when
              // delta is NAN.
              if (!(delta == 0.0)) {
                return false;
              }
            }
          }
        }
      }
    }
  }
  return true;
}

/// \returns a tensor with UInt8FusedQTy from \p T, whose type should be
/// UInt4FusedFP16QTy, UInt4FusedQTy, or UInt8FusedFP16QTy.
template <class scaleOffsetTy = float16_t>
static Tensor convertToUInt8FusedQTy(const Tensor *T) {
  const ElemKind origKind = T->getElementType();
  // Supports UInt4FusedFP16QTy/UInt8FusedFP16QTy/UInt4FusedQTy -> UInt8FusedQTy
  DCHECK((origKind == ElemKind::UInt4FusedFP16QTy ||
          origKind == ElemKind::UInt4FusedQTy ||
          origKind == ElemKind::UInt8FusedFP16QTy) &&
         T->dims().size() == 2)
      << "UInt4FusedFP16QTy, UInt4FusedQTy or UInt8FusedFP16QTy must be 2 "
         "dimensional.";
  bool is4Bit = (origKind == ElemKind::UInt4FusedFP16QTy ||
                 origKind == ElemKind::UInt4FusedQTy);
  const dim_t dataCol = T->dims()[1] - 2 * sizeof(scaleOffsetTy);
  const dim_t numTotalRows = T->dims()[0];
  const dim_t numTotalColumns = dataCol * (is4Bit ? 2 : 1) + 2 * sizeof(float);
  Tensor tmp(ElemKind::UInt8FusedQTy, {numTotalRows, numTotalColumns}, 1.0, 0);
  auto srcH = T->getHandle<uint8_t>();
  auto dstH = tmp.getHandle<uint8_t>();
  for (dim_t row = 0; row < T->dims()[0]; row++) {
    // Copy scale and offset from src to dst.
    scaleOffsetTy scale, offset;
    std::tie(scale, offset) =
        srcH.getFusedScaleOffsetFromRow<scaleOffsetTy>(row);
    dstH.setFusedScaleOffsetInRow<float>(row, static_cast<float>(scale),
                                         static_cast<float>(offset));
    for (dim_t column = 0; column < dataCol; column++) {
      if (is4Bit) {
        auto src = srcH.at({row, column});
        // Even column in new data uses value from LSB 4-bit from src data.
        dstH.at({row, column * 2}) = src & 0x0F;
        // Odd column in new data uses value from MSB 4-bit from dst data.
        dstH.at({row, column * 2 + 1}) = (src >> 4) & 0x0F;
      } else {
        dstH.at({row, column}) = srcH.at({row, column});
      }
    }
  }
  return tmp;
}

/// \returns a tensor with UInt4FusedQTy from \p T, whose type should be
/// UInt4FusedFP16QTy.
static Tensor convertToUInt4FusedQTy(const Tensor *T) {
  const ElemKind origKind = T->getElementType();
  // Supports UInt4FusedFP16QTy -> UInt4FusedQTy.
  DCHECK(origKind == ElemKind::UInt4FusedFP16QTy && T->dims().size() == 2)
      << "UInt4FusedFP16QTy must be 2 dimensional.";
  const dim_t dataCol = T->dims()[1] - 2 * sizeof(float16_t);
  const dim_t numTotalRows = T->dims()[0];
  const dim_t numTotalColumns = dataCol + 2 * sizeof(float);
  Tensor tmp(ElemKind::UInt4FusedQTy, {numTotalRows, numTotalColumns}, 1.0, 0);
  auto srcH = T->getHandle<uint8_t>();
  auto dstH = tmp.getHandle<uint8_t>();
  for (dim_t row = 0; row < T->dims()[0]; row++) {
    // Copy scale and offset from src to dst.
    float16_t scale, offset;
    std::tie(scale, offset) = srcH.getFusedScaleOffsetFromRow<float16_t>(row);
    dstH.setFusedScaleOffsetInRow<float>(row, static_cast<float>(scale),
                                         static_cast<float>(offset));
    for (dim_t column = 0; column < dataCol; column++) {
      dstH.at({row, column}) = srcH.at({row, column});
    }
  }
  return tmp;
}
} // namespace

void glow::dumpAsciiImpl(const Tensor *T, llvm::raw_ostream &os) {
  switch (T->getElementType()) {
  case ElemKind::FloatTy:
    return dumpAsciiGenericImpl(T->getHandle<float>(), os);
  case ElemKind::Float16Ty:
    return dumpAsciiGenericImpl(T->getHandle<float16_t>(), os);
  case ElemKind::BFloat16Ty:
    return dumpAsciiGenericImpl(T->getHandle<bfloat16_t>(), os);
  case ElemKind::Float64Ty:
    return dumpAsciiGenericImpl(T->getHandle<double>(), os);
  case ElemKind::Int8QTy:
    return dumpAsciiGenericImpl(T->getHandle<int8_t>(), os);
  case ElemKind::UInt8QTy:
    return dumpAsciiGenericImpl(T->getHandle<uint8_t>(), os);
  case ElemKind::Int16QTy:
    return dumpAsciiGenericImpl(T->getHandle<int16_t>(), os);
  case ElemKind::Int32QTy:
    return dumpAsciiGenericImpl(T->getHandle<int32_t>(), os);
  case ElemKind::Int64QTy:
    return dumpAsciiGenericImpl(T->getHandle<int64_t>(), os);
  case ElemKind::UInt8ITy:
    return dumpAsciiGenericImpl(T->getHandle<uint8_t>(), os);
  case ElemKind::Int32ITy:
    return dumpAsciiGenericImpl(T->getHandle<int32_t>(), os);
  case ElemKind::Int64ITy:
    return dumpAsciiGenericImpl(T->getHandle<int64_t>(), os);
  case ElemKind::UInt8FusedQTy:
    return dumpAsciiGenericImpl(T->getHandle<uint8_t>(), os);
  case ElemKind::UInt8FusedFP16QTy:
    return dumpAsciiGenericImpl(T->getHandle<uint8_t>(), os);
  case ElemKind::UInt4FusedFP16QTy:
    return dumpAsciiGenericImpl(T->getHandle<uint8_t>(), os);
  case ElemKind::UInt4FusedQTy:
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
  case ElemKind::BFloat16Ty:
    return dumpGenericImpl(T->getHandle<bfloat16_t>(), os, maxNumElem);
  case ElemKind::Float64Ty:
    return dumpGenericImpl(T->getHandle<double>(), os, maxNumElem);
  case ElemKind::Int8QTy:
    return dumpGenericImpl(T->getHandle<int8_t>(), os, maxNumElem);
  case ElemKind::UInt8QTy:
    return dumpGenericImpl(T->getHandle<uint8_t>(), os, maxNumElem);
  case ElemKind::Int16QTy:
    return dumpGenericImpl(T->getHandle<int16_t>(), os, maxNumElem);
  case ElemKind::Int32QTy:
    return dumpGenericImpl(T->getHandle<int32_t>(), os, maxNumElem);
  case ElemKind::Int64QTy:
    return dumpGenericImpl(T->getHandle<int64_t>(), os, maxNumElem);
  case ElemKind::UInt8ITy:
    return dumpGenericImpl(T->getHandle<uint8_t>(), os, maxNumElem);
  case ElemKind::Int32ITy:
    return dumpGenericImpl(T->getHandle<int32_t>(), os, maxNumElem);
  case ElemKind::Int64ITy:
    return dumpGenericImpl(T->getHandle<int64_t>(), os, maxNumElem);
  case ElemKind::UInt8FusedQTy:
    return dumpGenericImpl(T->getHandle<uint8_t>(), os, maxNumElem);
  case ElemKind::UInt8FusedFP16QTy:
    return dumpGenericImpl(T->getHandle<uint8_t>(), os, maxNumElem);
  case ElemKind::UInt4FusedFP16QTy:
    return dumpGenericImpl(T->getHandle<uint8_t>(), os, maxNumElem);
  case ElemKind::UInt4FusedQTy:
    return dumpGenericImpl(T->getHandle<uint8_t>(), os, maxNumElem);
  case ElemKind::BoolTy:
    return dumpGenericImpl(T->getHandle<bool>(), os, maxNumElem);
  }
}

void glow::dumpImpl(const Tensor *T, unsigned maxNumElem) {
  dumpImpl(T, llvm::outs(), maxNumElem);
}

void glow::dumpImpl(const Tensor *T) { dumpImpl(T, llvm::outs()); }

// Dump functions.
void Tensor::dump(llvm::raw_ostream &os) const { dumpImpl(this, os); }

void Tensor::dump() const { dumpImpl(this, llvm::outs()); }

std::string Tensor::toString() const {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  dumpImpl(this, os);
  return os.str();
}

std::string Tensor::getShapeToString() const {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  dumpShape(dims(), os);
  return os.str();
}

void Tensor::dump(llvm::raw_ostream &os, unsigned maxNumElem) const {
  dumpImpl(this, os, maxNumElem);
}

void Tensor::dump(unsigned maxNumElem) const {
  dumpImpl(this, llvm::outs(), maxNumElem);
}

std::string Tensor::toString(unsigned maxNumElem) const {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  dumpImpl(this, os, maxNumElem);
  return os.str();
}

/// Dump a textual representation of a specific number of elements in the Tensor
/// to std::string.

void glow::genericTranspose(const Tensor *src, Tensor *dest,
                            llvm::ArrayRef<unsigned_t> shuffle) {
  DCHECK(src->dims().size() == shuffle.size())
      << "Invalid dimensions " << src->dims().size()
      << " != " << src->dims().size();

  dim_t newSizes[max_tensor_dimensions];

  // Generate the swizzled dimensions.
  auto origDims = src->dims();
  for (unsigned i = 0; i < origDims.size(); i++) {
    newSizes[i] = origDims[shuffle[i]];
  }

  // Resize the tensor to the transposed shape.
  auto destType = Type::newShape(src->getType(), {newSizes, origDims.size()});
  // genericTranspose function doesn't know how to set non-trivial strides and
  // alignments and it cannot figure out the correct ones as it can be
  // backend-specific. Therefore set the type to destType only if it is not set
  // properly by the caller yet.
  // Reset should be called anyways to allocate memory for the tensor.
  if (dest->dims() != destType.dims()) {
    dest->reset(destType);
  } else {
    dest->reset(dest->getType());
  }

  // fill with 0 for padding bytes.
  if (src->actualSize() != dest->actualSize()) {
    dest->zero();
  }

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
  case ElemKind::BFloat16Ty: {
    auto srcH = src->getHandle<bfloat16_t>();
    auto destH = dest->getHandle<bfloat16_t>();
    transposeSelectImpl(srcH, destH, shuffle);
    return;
  }
  case ElemKind::Float64Ty: {
    auto srcH = src->getHandle<double>();
    auto destH = dest->getHandle<double>();
    transposeSelectImpl(srcH, destH, shuffle);
    return;
  }
  case ElemKind::Int8QTy: {
    auto srcH = src->getHandle<int8_t>();
    auto destH = dest->getHandle<int8_t>();
    transposeSelectImpl(srcH, destH, shuffle);
    return;
  }
  case ElemKind::UInt8QTy: {
    auto srcH = src->getHandle<uint8_t>();
    auto destH = dest->getHandle<uint8_t>();
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
  case ElemKind::Int64QTy: {
    auto srcH = src->getHandle<int64_t>();
    auto destH = dest->getHandle<int64_t>();
    transposeSelectImpl(srcH, destH, shuffle);
    return;
  }
  case ElemKind::UInt8ITy: {
    auto srcH = src->getHandle<uint8_t>();
    auto destH = dest->getHandle<uint8_t>();
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
  case ElemKind::UInt8FusedFP16QTy: {
    llvm_unreachable("Transposing UInt8FusedFP16QTy is unsupported.");
  }
  case ElemKind::UInt4FusedFP16QTy: {
    llvm_unreachable("Transposing UInt4FusedFP16QTy is unsupported.");
  }
  case ElemKind::UInt4FusedQTy: {
    llvm_unreachable("Transposing UInt4FusedQTy is unsupported.");
  }
  case ElemKind::BoolTy: {
    auto srcH = src->getHandle<bool>();
    auto destH = dest->getHandle<bool>();
    transposeSelectImpl(srcH, destH, shuffle);
    return;
  }
  }
}

ShapeVector glow::expandDimsToMax(llvm::ArrayRef<dim_t> currDims) {
  ShapeVector newDims(currDims.begin(), currDims.end());
  for (size_t i = newDims.size(); i < max_tensor_dimensions; i++) {
    newDims.push_back(1);
  }
  return newDims;
}

ShapeVector glow::reduceDims(llvm::ArrayRef<dim_t> dims,
                             llvm::ArrayRef<unsigned_t> axes, bool keepDims) {
  ShapeVector newDims;
  for (unsigned_t dim = 0, end = dims.size(); dim < end; ++dim) {
    auto it = std::find(axes.begin(), axes.end(), dim);
    bool dimReduced = (it != axes.end());
    if (dimReduced) {
      if (keepDims) {
        newDims.push_back(1);
      } else {
        continue;
      }
    } else {
      newDims.push_back(dims[dim]);
    }
  }
  return newDims;
}

std::vector<unsigned_t>
glow::getInverseTranspose(llvm::ArrayRef<unsigned_t> shuffle) {
  std::vector<unsigned_t> unshuffle;
  // For each index, go find where it ended up in the shuffle
  for (auto i = 0; i < shuffle.size(); ++i) {
    for (auto j = 0; j < shuffle.size(); ++j) {
      if (shuffle[j] == i) {
        unshuffle.push_back(j);
        break;
      }
    }
  }
  return unshuffle;
}

void Tensor::init(InitKind init, float val, PseudoRNG &PRNG) {
  assert(!isDeviceResident() && "Tensor must reside on host to access data.");
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
    case ElemKind::BFloat16Ty: {
      getHandle<bfloat16_t>().clear(bfloat16_t(val));
      break;
    }
    case ElemKind::Float64Ty: {
      getHandle<double>().clear(val);
      break;
    }
    case ElemKind::Int8QTy: {
      getHandle<int8_t>().clear(val);
      break;
    }
    case ElemKind::UInt8QTy: {
      getHandle<uint8_t>().clear(val);
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
    case ElemKind::Int64QTy: {
      getHandle<int64_t>().clear(val);
      break;
    }
    case ElemKind::UInt8ITy: {
      getHandle<uint8_t>().clear(val);
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

#define FUSED_CASE(ELEM_KIND, DATA_TYPE)                                       \
  case ElemKind::ELEM_KIND: {                                                  \
    DCHECK(dims().size() == 2)                                                 \
        << "Fused tensor must be 2-dimensional but instead has "               \
        << dims().size() << " dimensions.";                                    \
    DCHECK(dims()[1] > 2 * sizeof(DATA_TYPE))                                  \
        << "Fused tensor must have space for scale/offset, but only has  "     \
        << dims()[1] << " columns.";                                           \
    auto H = getHandle<uint8_t>();                                             \
    for (dim_t i = 0; i < dims()[0]; i++) {                                    \
      for (dim_t j = 0, f = dims()[1] - 2 * sizeof(DATA_TYPE); j < f; j++) {   \
        H.at({i, j}) = val;                                                    \
      }                                                                        \
    }                                                                          \
    break;                                                                     \
  }
      FUSED_CASE(UInt8FusedQTy, float);
      FUSED_CASE(UInt4FusedQTy, float);
      FUSED_CASE(UInt8FusedFP16QTy, float16_t);
      FUSED_CASE(UInt4FusedFP16QTy, float16_t);
#undef FUSED_CASE

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
    case ElemKind::BFloat16Ty: {
      getHandle<bfloat16_t>().initXavier(val, PRNG);
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
  assert(!isDeviceResident() && "Tensor must reside on host to access data.");
  *this = this->getCopyConvertedToType(newTy);
}

Tensor Tensor::getCopyConvertedToType(ElemKind newKind) const {
  assert(!isDeviceResident() && "Tensor must reside on host to access data.");
  const ElemKind origKind = getElementType();
  DCHECK((origKind == ElemKind::FloatTy && newKind == ElemKind::Float16Ty) ||
         (origKind == ElemKind::FloatTy && newKind == ElemKind::BFloat16Ty) ||
         (origKind == ElemKind::FloatTy && newKind == ElemKind::Int32ITy) ||
         (origKind == ElemKind::FloatTy && newKind == ElemKind::Int64ITy) ||
         (origKind == ElemKind::Float16Ty && newKind == ElemKind::FloatTy) ||
         (origKind == ElemKind::BFloat16Ty && newKind == ElemKind::FloatTy) ||
         (origKind == ElemKind::Int64ITy && newKind == ElemKind::Int32ITy) ||
         (origKind == ElemKind::Int64ITy && newKind == ElemKind::FloatTy) ||
         (origKind == ElemKind::Int32ITy && newKind == ElemKind::Int64ITy) ||
         (origKind == ElemKind::Int32ITy && newKind == ElemKind::FloatTy) ||
         (origKind == ElemKind::UInt8FusedQTy &&
          newKind == ElemKind::UInt8FusedFP16QTy) ||
         (origKind == ElemKind::UInt8FusedFP16QTy &&
          newKind == ElemKind::UInt8FusedQTy) ||
         (origKind == ElemKind::UInt4FusedFP16QTy &&
          newKind == ElemKind::UInt8FusedQTy) ||
         (origKind == ElemKind::UInt4FusedFP16QTy &&
          newKind == ElemKind::UInt4FusedQTy) ||
         (origKind == ElemKind::UInt4FusedQTy &&
          newKind == ElemKind::UInt8FusedQTy))
      << "Conversion from " << Type::getElementName(origKind).str() << " to "
      << Type::getElementName(newKind).str() << " is not yet implemented";

  if (!isQuantizedElemKind(newKind)) {
    Tensor tmp(newKind, dims());
    switch (newKind) {
    case ElemKind::Float16Ty:
      tmp.copyWithCast<float16_t, float>(this);
      break;
    case ElemKind::BFloat16Ty:
      tmp.copyWithCast<bfloat16_t, float>(this);
      break;

    case ElemKind::FloatTy:
      if (getElementType() == ElemKind::Int32ITy) {
        tmp.copyWithCast<float, int32_t>(this);
      } else if (getElementType() == ElemKind::Int64ITy) {
        tmp.copyWithCast<float, int64_t>(this);
      } else if (getElementType() == ElemKind::Float16Ty) {
        tmp.copyWithCast<float, float16_t>(this);
      } else if (getElementType() == ElemKind::BFloat16Ty) {
        tmp.copyWithCast<float, bfloat16_t>(this);
      } else if (getElementType() == ElemKind::FloatTy) {
        tmp.copyRawFrom(this);
      } else {
        llvm_unreachable("Invalid conversion to FLOAT.");
      }
      break;

    case ElemKind::Int32ITy:
      if (getElementType() == ElemKind::Int64ITy) {
        tmp.copyWithCast<int32_t, int64_t>(this);
      } else if (getElementType() == ElemKind::FloatTy) {
        tmp.copyWithCast<int32_t, float>(this);
      } else {
        llvm_unreachable("Invalid conversion from FLOAT.");
      }
      break;
    case ElemKind::Int64ITy:
      if (getElementType() == ElemKind::Int32ITy) {
        tmp.copyWithCast<int64_t, int32_t>(this);
      } else {
        llvm_unreachable("Invalid conversion from FLOAT.");
      }
      break;

    default:
      llvm_unreachable("Type not supported");
    }
    return tmp;
  }

  // Handle Fused conversion.
  if ((origKind == ElemKind::UInt8FusedFP16QTy ||
       origKind == ElemKind::UInt4FusedFP16QTy) &&
      newKind == ElemKind::UInt8FusedQTy) {
    return convertToUInt8FusedQTy<float16_t>(this);
  }
  if (origKind == ElemKind::UInt4FusedQTy &&
      newKind == ElemKind::UInt8FusedQTy) {
    return convertToUInt8FusedQTy<float>(this);
  }
  if (origKind == ElemKind::UInt4FusedFP16QTy &&
      newKind == ElemKind::UInt4FusedQTy) {
    return convertToUInt4FusedQTy(this);
  }

  // Supports UInt8FusedQTy -> UInt8FusedFP16QTy.
  DCHECK(origKind == ElemKind::UInt8FusedQTy && dims().size() == 2)
      << "UInt8FusedQTy must be 2 dimensional.";
  Tensor tmp(newKind,
             {dims()[0], dims()[1] - 2 * ((dim_t)sizeof(float) -
                                          (dim_t)sizeof(float16_t))},
             1.0, 0);

  const size_t dstWidth = tmp.dims()[1];
  auto srcH = getHandle<uint8_t>();
  auto dstH = tmp.getHandle<uint8_t>();
  for (dim_t i = 0, e = dims()[0]; i < e; i++) {
    // Copy the scale/offset from src to dst.
    float scale, offset;
    std::tie(scale, offset) = srcH.getFusedScaleOffsetFromRow<float>(i);
    dstH.setFusedScaleOffsetInRow<float16_t>(i, static_cast<float16_t>(scale),
                                             static_cast<float16_t>(offset));

    // Copy over the row's uint8 data from src to dst; scales and offsets were
    // already copied over above.
    for (dim_t j = 0, f = dstWidth - 2 * sizeof(float16_t); j < f; j++) {
      dstH.at({i, j}) = srcH.at({i, j});
    }
  }
  return tmp;
}

namespace glow {
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Tensor &t) {
  t.dump(os);
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Tensor *t) {
  assert(t != nullptr && "Null Pointer.");
  t->dump(os);
  return os;
}

void Tensor::moveToDevice(DeviceTensorTransferManager *deviceManager,
                          void *locationContext) {
  if (deviceResidency_ == nullptr) {
    deviceResidency_ = new DeviceResidencyInfo();
  }
  deviceResidency_->deviceManager_ = deviceManager;
  deviceResidency_->locationContext_ = locationContext;
  deviceResidency_->tensorResidency_ =
      DeviceResidencyInfo::TensorResidency::Device;
}

void Tensor::ensureOnHost() {
  if (deviceResidency_ == nullptr) {
    // already on host.
    return;
  }
  if (deviceResidency_->isDeviceResident()) {
    deviceResidency_->deviceManager_->transferFromDevice(*this);
  }
  assert(!isDeviceResident());
}

void Tensor::copyRawToDevice(const Tensor *t) {
  assert(isDeviceResident());
  void *locationContext = deviceResidency_->locationContext_;
  DeviceTensorTransferManager *DM = deviceResidency_->deviceManager_;
  clearDeviceResidency();
  copyRawFrom(t);
  DM->transferToDevice(*this, locationContext);
}

bool Tensor::isTiled(unsigned_t axis, dim_t size, bool fractional) const {
  switch (getElementType()) {
  case ElemKind::FloatTy: {
    return isTiledImpl<float>(this, axis, size, fractional);
  }
  case ElemKind::Float16Ty: {
    return isTiledImpl<float16_t>(this, axis, size, fractional);
  }
  case ElemKind::Int8QTy: {
    return isTiledImpl<int8_t>(this, axis, size, fractional);
  }
  case ElemKind::UInt8QTy: {
    return isTiledImpl<uint8_t>(this, axis, size, fractional);
  }
  case ElemKind::Int16QTy: {
    return isTiledImpl<int16_t>(this, axis, size, fractional);
  }
  case ElemKind::Int32QTy: {
    return isTiledImpl<int32_t>(this, axis, size, fractional);
  }
  case ElemKind::Int32ITy: {
    return isTiledImpl<int32_t>(this, axis, size, fractional);
  }
  case ElemKind::Int64ITy: {
    return isTiledImpl<int64_t>(this, axis, size, fractional);
  }
  case ElemKind::BoolTy: {
    return isTiledImpl<bool>(this, axis, size, fractional);
  }
  default:
    llvm_unreachable("isTiled: Precision not supported!");
  }
}

bool Tensor::isTiled(llvm::ArrayRef<unsigned_t> axes,
                     llvm::ArrayRef<dim_t> sizes, bool fractional) const {
  assert(axes.size() == sizes.size() &&
         "Mismatch between axes and sizes length!");
  for (size_t idx = 0, end = axes.size(); idx < end; ++idx) {
    if (!isTiled(axes[idx], sizes[idx], fractional)) {
      return false;
    }
  }
  return true;
}

bool isSliceContiguous(llvm::ArrayRef<dim_t> sliceShape,
                       llvm::ArrayRef<dim_t> tensorShape) {
  assert(sliceShape.size() == tensorShape.size() &&
         "Array length mismatch for slice/tensor sizes!");
  // Search first non-singleton slice dimension. If all the dimensions are
  // singleton then by convention the first non-singleton dimension is the
  // slice size.
  size_t firstNonSingleDim = sliceShape.size();
  for (size_t dim = 0, dimEnd = sliceShape.size(); dim < dimEnd; ++dim) {
    if (sliceShape[dim] != 1) {
      firstNonSingleDim = dim;
      break;
    }
  }
  // First non-singleton slice dimension can be partially or fully extracted.
  // The following dimensions must be fully extracted.
  for (size_t dim = firstNonSingleDim + 1, dimEnd = sliceShape.size();
       dim < dimEnd; ++dim) {
    if (sliceShape[dim] != tensorShape[dim]) {
      return false;
    }
  }
  return true;
}

} // namespace glow
