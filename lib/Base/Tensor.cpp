// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Base/Tensor.h"

#include "llvm/Support/NativeFormatting.h"
#include "llvm/Support/raw_ostream.h"

using namespace glow;

namespace {

/// This is a helper method that's used in the visualization of tensors.
template <class ElemTy> static char valueToChar(ElemTy val) {
  char ch = ' ';
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

template <class ElemTy> static void dumpGenericImpl(Handle<ElemTy> handle) {
  auto shape = handle.dims();
  size_t numDims = shape.size();

  // Check for empty tensor.
  if (!numDims) {
    llvm::outs() << "[ Empty tensor ]\n";
    return;
  }

  // Output shape.
  llvm::outs() << "shape: ( ";
  for (auto &d : shape) {
    llvm::outs() << d << " ";
  }
  llvm::outs() << ")\n";

  ElemTy mx = handle.raw(0);
  ElemTy mn = handle.raw(0);

  for (size_t i = 0, e = handle.size(); i < e; i++) {
    mx = std::max(mx, handle.raw(i));
    mn = std::min(mn, handle.raw(i));
  }

  // Check for zero tensor.
  if (mn == .0 && mx == .0) {
    llvm::outs() << "[ Zero tensor ]\n";
    return;
  }

  // Output max and min.
  llvm::outs() << "max: ";
  llvm::write_double(llvm::outs(), mx, llvm::FloatStyle::Fixed, 3);
  llvm::outs() << "  min: ";
  llvm::write_double(llvm::outs(), mn, llvm::FloatStyle::Fixed, 3);
  llvm::outs() << "\n";

  const unsigned maxNumElem = 100;

  llvm::outs() << "[";

  for (size_t i = 0, e = std::min<size_t>(maxNumElem, handle.size()); i < e;
       i++) {

    // Print one open brace at the beginning of every row, slice, and tensor.
    for (size_t j = 0, e = numDims - 1; numDims > 1 && j < e; j++) {
      if (i % handle.sliceSize(j) == 0) {
        // This iteration of outer loop is a new row, slice or tensor.
        llvm::outs() << "[";
      }
    }

    // Print the value at the current index.
    llvm::write_double(llvm::outs(), handle.raw(i), llvm::FloatStyle::Fixed, 3);

    // Print one closed brace at the end of every row, slice, or tensor.
    for (size_t j = 0, e = numDims - 1; numDims > 1 && j < e; j++) {
      size_t next_index = i + 1;
      if (next_index % handle.sliceSize(j) == 0u) {
        llvm::outs() << "]";
      }
    }

    llvm::outs() << ", ";

    // Print one newline at the end of every row, slice, or tensor.
    for (size_t j = 0, e = numDims - 1; numDims > 1 && j < e; j++) {
      size_t next_index = i + 1;
      if (next_index % handle.sliceSize(j) == 0u) {
        // Next iteration of outer loop will be a new row, slice or tensor.
        llvm::outs() << "\n";
      }
    }
  }

  if (handle.size() > maxNumElem) {
    llvm::outs() << "...";
  }

  llvm::outs() << "]\n";
}

template <class ElemTy>
static void dumpAsciiGenericImpl(Handle<ElemTy> handle) {
  auto d = handle.dims();

  if (d.size() == 2) {
    for (size_t x = 0; x < d[0]; x++) {
      for (size_t y = 0; y < d[1]; y++) {
        auto val = handle.at({x, y});
        llvm::outs() << valueToChar(val);
      }
      llvm::outs() << "\n";
    }
  } else if (d.size() == 3) {
    // Print monochrome (one-color channel) tensors:
    if (d[2] == 1) {
      for (size_t x = 0; x < d[0]; x++) {
        for (size_t y = 0; y < d[1]; y++) {
          auto val = handle.at({x, y, 0});
          llvm::outs() << valueToChar(val);
        }
        llvm::outs() << "\n";
      }
    } else {
      for (size_t z = 0; z < d[2]; z++) {
        llvm::outs() << "\n";
        for (size_t x = 0; x < d[0]; x++) {
          for (size_t y = 0; y < d[1]; y++) {
            auto val = handle.at({x, y, z});
            llvm::outs() << valueToChar(val);
          }
          llvm::outs() << "\n";
        }
      }
    }

  } else {
    assert(false && "Invalid tensor size");
  }
}

/// This is a slow generic transpose. This method performs a single for loop
/// over a single dimension, or if we've reached the last dimension perform a
/// single copy of a single element.
template <class ElemTy>
static void transposeGenericImpl(Handle<ElemTy> &src, Handle<ElemTy> &dest,
                                 size_t *srcCoor, size_t *destCoor,
                                 llvm::ArrayRef<unsigned> shuffle,
                                 unsigned depth = 0) {
  if (depth == shuffle.size()) {
    auto srcIdx = llvm::ArrayRef<size_t>(srcCoor, depth);
    auto destIdx = llvm::ArrayRef<size_t>(destCoor, depth);
    dest.at(destIdx) = src.at(srcIdx);
    return;
  }

  // Iterate over one dimension and continue recursively to the next dim.
  for (size_t x = 0, e = dest.dims()[depth]; x < e; x++) {
    unsigned swizzledDepth = shuffle[depth];
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
static bool tryTransposeFastImpl(Handle<ElemTy> &src, Handle<ElemTy> &dest,
                                 llvm::ArrayRef<unsigned> shuffle) {
  const size_t numDims = dest.dims().size();
  size_t srcCoorArr[numDims];
  size_t destCoorArr[numDims];
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
static void transposeSelectImpl(Handle<ElemTy> &src, Handle<ElemTy> &dest,
                                llvm::ArrayRef<unsigned> shuffle) {
  bool transposeOccurred = tryTransposeFastImpl(src, dest, shuffle);
  if (!transposeOccurred) {
    size_t srcCoor[max_tensor_dimensions];
    size_t destCoor[max_tensor_dimensions];
    transposeGenericImpl(src, dest, srcCoor, destCoor, shuffle);
  }
}

/// Takes an array of indices \p currIdxs and increments it with respect to
/// the Tensor's dims(), allowing for iterating over all of a Tensor's
/// elements without statically knowing its shape.
static bool
incrementIndicesAndCheckFinished(llvm::MutableArrayRef<size_t> currIdxs,
                                 llvm::ArrayRef<size_t> origDims) {
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

/// Broadcast the current Handle's tensor in \p direction of length \p
/// newDimLen into Tensor \p dest. If not \p addingNewDim then the dimension
/// being extended should be size 1.
template <class ElemTy>
static void broadcastOneDimensionGeneric(Tensor *src, Tensor *dest,
                                         unsigned newDimLen, unsigned direction,
                                         bool addingNewDim) {
  auto origDims = src->dims();

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
  for (unsigned i = 0; i < origDims.size(); ++i) {
    if (addingNewDim && (i == direction)) {
      shift = 1;
    }
    newDims[i + shift] = origDims[i];
  }
  newDims[direction] = newDimLen;
  const unsigned newDimsSize = origDims.size() + (addingNewDim ? 1 : 0);
  auto &srcType = src->getType();
  if (srcType.isQuantizedType()) {
    dest->reset(src->getElementType(),
                llvm::ArrayRef<size_t>(newDims, newDimsSize),
                srcType.getScale(), srcType.getOffset());
  } else {
    dest->reset(src->getElementType(),
                llvm::ArrayRef<size_t>(newDims, newDimsSize));
  }

  size_t currNewIdxsArr[max_tensor_dimensions];
  auto currNewIdxs = llvm::MutableArrayRef<size_t>(currNewIdxsArr, newDimsSize);
  size_t currIdxsArr[max_tensor_dimensions] = {0};
  auto currIdxs = llvm::MutableArrayRef<size_t>(currIdxsArr, origDims.size());

  auto srcH = src->getHandle<ElemTy>();
  auto destH = dest->getHandle<ElemTy>();

  // Iterate over all locations in the original Tensor.
  do {
    // New indices using current from original Tensor, plus new dimension.
    unsigned shift = 0;
    for (unsigned i = 0; i < origDims.size(); ++i) {
      if (addingNewDim && (i == direction)) {
        shift = 1;
      }
      currNewIdxs[i + shift] = currIdxs[i];
    }

    // Copy all values in the new broadcast direction dimension.
    for (currNewIdxs[direction] = 0; currNewIdxs[direction] < newDimLen;
         currNewIdxs[direction]++) {
      destH.at(currNewIdxs) = srcH.at(currIdxs);
    }
  } while (!incrementIndicesAndCheckFinished(currIdxs, origDims));
}

template <class ElemTy>
static void broadcastToNewShapeGenericImpl(Tensor *src, Tensor *dest,
                                           llvm::ArrayRef<size_t> otherDims,
                                           unsigned axis) {
  auto origDims = src->dims();
  const int dimDifference = otherDims.size() - origDims.size();
  (void)dimDifference;
  assert(otherDims.size() >= origDims.size() &&
         "Dimensions to broadcast to must be equal or greater size.");
  assert(axis <= dimDifference &&
         "Axis + nDims of orig Tensor must be <= newShape nDims");

  Tensor intermediate;
  intermediate.copyFrom(src);

  // Iterate over the new shape; if the original shape had a dimension here
  // (when considering the axis) then verify the dimension either matches the
  // new shape (no action taken) or == 1 (broadcast in that direction). Else
  // the original shape had no dimensions here (after considering axis), so
  // add the new dimension and broadcast in that direction.
  for (size_t i = 0; i < otherDims.size(); i++) {
    if (i >= axis && i < origDims.size() + axis) {
      const int origIdx = i - axis;
      if (origDims[origIdx] == otherDims[i]) {
        // Keep original dimensions; they are compatible.
      } else if (origDims[origIdx] == 1) {
        // Broadcast this dimension to size from otherDims.
        Tensor tmp;
        const bool addingNewDim = false;
        broadcastOneDimensionGeneric<ElemTy>(&intermediate, &tmp, otherDims[i],
                                             i, addingNewDim);
        intermediate.copyFrom(&tmp);
      } else {
        // Incompatible dimensions for broadcasting
        assert(false && "Cannot broadcast with these dimensions.");
      }
    } else {
      Tensor tmp;
      const bool addingNewDim = true;
      broadcastOneDimensionGeneric<ElemTy>(&intermediate, &tmp, otherDims[i], i,
                                           addingNewDim);
      intermediate.copyFrom(&tmp);
    }
  }

  dest->copyFrom(&intermediate);
}
} // namespace

void glow::dumpAsciiImpl(Tensor *T) {
  switch (T->getElementType()) {
  case ElemKind::FloatTy:
    return dumpAsciiGenericImpl(T->getHandle<float>());
  case ElemKind::Int8QTy:
    return dumpAsciiGenericImpl(T->getHandle<int8_t>());
  case ElemKind::Int32QTy:
    return dumpAsciiGenericImpl(T->getHandle<int32_t>());
  case ElemKind::IndexTy:
    return dumpAsciiGenericImpl(T->getHandle<size_t>());
  }
}

void glow::dumpImpl(Tensor *T) {
  switch (T->getElementType()) {
  case ElemKind::FloatTy:
    return dumpGenericImpl(T->getHandle<float>());
  case ElemKind::Int8QTy:
    return dumpGenericImpl(T->getHandle<int8_t>());
  case ElemKind::Int32QTy:
    return dumpGenericImpl(T->getHandle<int32_t>());
  case ElemKind::IndexTy:
    return dumpGenericImpl(T->getHandle<size_t>());
  }
}

void glow::genericTranspose(Tensor *src, Tensor *dest,
                            llvm::ArrayRef<unsigned> shuffle) {
  assert(src->dims().size() == shuffle.size() && "Invalid dimensions");

  size_t newSizes[max_tensor_dimensions];

  // Generate the swizzled dimensions.
  auto origDims = src->dims();
  for (unsigned i = 0; i < origDims.size(); i++) {
    newSizes[i] = origDims[shuffle[i]];
  }

  // Resize the tensor to the transposed shape.
  auto destType =
      src->getType().isQuantizedType()
          ? Type(src->getElementType(), {newSizes, origDims.size()},
                 src->getType().getScale(), src->getType().getOffset())
          : Type(src->getElementType(), {newSizes, origDims.size()});

  dest->reset(destType);

  switch (src->getElementType()) {
  case ElemKind::FloatTy: {
    auto srcH = src->getHandle<float>();
    auto destH = dest->getHandle<float>();
    transposeSelectImpl(srcH, destH, shuffle);
    return;
  }
  case ElemKind::Int8QTy: {
    auto srcH = src->getHandle<int8_t>();
    auto destH = dest->getHandle<int8_t>();
    transposeSelectImpl(srcH, destH, shuffle);
    return;
  }
  case ElemKind::Int32QTy: {
    auto srcH = src->getHandle<int32_t>();
    auto destH = dest->getHandle<int32_t>();
    transposeSelectImpl(srcH, destH, shuffle);
    return;
  }
  case ElemKind::IndexTy: {
    auto srcH = src->getHandle<size_t>();
    auto destH = dest->getHandle<size_t>();
    transposeSelectImpl(srcH, destH, shuffle);
    return;
  }
  }
}

void glow::broadcastToNewShapeImpl(Tensor *src, Tensor *dest,
                                   llvm::ArrayRef<size_t> otherDims,
                                   unsigned axis) {
  switch (src->getElementType()) {
  case ElemKind::FloatTy: {
    broadcastToNewShapeGenericImpl<float>(src, dest, otherDims, axis);
    return;
  }
  case ElemKind::Int8QTy: {
    broadcastToNewShapeGenericImpl<int8_t>(src, dest, otherDims, axis);
    return;
  }
  case ElemKind::Int32QTy: {
    broadcastToNewShapeGenericImpl<int32_t>(src, dest, otherDims, axis);
    return;
  }
  case ElemKind::IndexTy: {
    broadcastToNewShapeGenericImpl<size_t>(src, dest, otherDims, axis);
    return;
  }
  }
}
