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

template <class ElemTy> void dumpGenericImpl(Handle<ElemTy> handle) {
  auto shape = handle.dims();
  size_t num_dims = shape.size();

  // Check for empty tensor.
  if (!num_dims) {
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
  if (!mn && !mx) {
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
    for (size_t j = 0, e = num_dims - 1; num_dims > 1 && j < e; j++) {
      if (i % handle.sliceSize(j) == 0) {
        // This iteration of outer loop is a new row, slice or tensor.
        llvm::outs() << "[";
      }
    }

    // Print the value at the current index.
    llvm::write_double(llvm::outs(), handle.raw(i), llvm::FloatStyle::Fixed, 3);

    // Print one closed brace at the end of every row, slice, or tensor.
    for (size_t j = 0, e = num_dims - 1; num_dims > 1 && j < e; j++) {
      size_t next_index = i + 1;
      if (next_index % handle.sliceSize(j) == 0) {
        llvm::outs() << "]";
      }
    }

    llvm::outs() << ", ";

    // Print one newline at the end of every row, slice, or tensor.
    for (size_t j = 0, e = num_dims - 1; num_dims > 1 && j < e; j++) {
      size_t next_index = i + 1;
      if (next_index % handle.sliceSize(j) == 0) {
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

template <class ElemTy> void dumpAsciiGenericImpl(Handle<ElemTy> handle) {
  auto d = handle.dims();

  if (d.size() == 2) {
    for (size_t y = 0; y < d[1]; y++) {
      for (size_t x = 0; x < d[0]; x++) {
        auto val = handle.at({x, y});
        llvm::outs() << valueToChar(val);
      }
      llvm::outs() << "\n";
    }
  } else if (d.size() == 3) {
    // Print monochrome (one-color channel) tensors:
    if (d[2] == 1) {
      for (size_t y = 0; y < d[1]; y++) {
        for (size_t x = 0; x < d[0]; x++) {
          auto val = handle.at({x, y, 0});
          llvm::outs() << valueToChar(val);
        }
        llvm::outs() << "\n";
      }
    } else {
      for (size_t z = 0; z < d[2]; z++) {
        llvm::outs() << "\n";
        for (size_t y = 0; y < d[1]; y++) {
          for (size_t x = 0; x < d[0]; x++) {
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

} // namespace

void glow::dumpAsciiImpl(Tensor *T) {
  switch (T->getElementType()) {
  case ElemKind::FloatTy:
    return dumpAsciiGenericImpl(T->getHandle<float>());
  case ElemKind::DoubleTy:
    return dumpAsciiGenericImpl(T->getHandle<double>());
  case ElemKind::Int8Ty:
    return dumpAsciiGenericImpl(T->getHandle<int8_t>());
  case ElemKind::Int32Ty:
    return dumpAsciiGenericImpl(T->getHandle<int32_t>());
  case ElemKind::IndexTy:
    return dumpAsciiGenericImpl(T->getHandle<size_t>());
  }
}

void glow::dumpImpl(Tensor *T) {
  switch (T->getElementType()) {
  case ElemKind::FloatTy:
    return dumpGenericImpl(T->getHandle<float>());
  case ElemKind::DoubleTy:
    return dumpGenericImpl(T->getHandle<double>());
  case ElemKind::Int8Ty:
    return dumpGenericImpl(T->getHandle<int8_t>());
  case ElemKind::Int32Ty:
    return dumpGenericImpl(T->getHandle<int32_t>());
  case ElemKind::IndexTy:
    return dumpGenericImpl(T->getHandle<size_t>());
  }
}
