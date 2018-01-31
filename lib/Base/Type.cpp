// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Base/Type.h"
#include "llvm/Support/NativeFormatting.h"
#include "llvm/Support/raw_ostream.h"

namespace glow {
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Type &type) {
  if (type.numSizes_ == 0) {
    return os << "<void>";
  }

  os << type.getElementName();

  if (type.isQuantizedType()) {
    os << "[scale:";
    llvm::write_double(os, type.getScale(), llvm::FloatStyle::Fixed, 3);
    os << " offset:";
    llvm::write_double(os, type.getOffset(), llvm::FloatStyle::Fixed, 3);
    os << ']';
  }

  os << '<';
  for (unsigned i = 0; i < type.numSizes_; ++i) {
    if (i) {
      os << " x ";
    }
    os << type.sizes_[i];
  }
  os << '>';

  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TypeRef &type) {
  if (!type) {
    return os << "<none>";
  }
  return os << *type;
}
} // namespace glow
