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

#include "glow/Base/Type.h"
#include "llvm/Support/NativeFormatting.h"
#include "llvm/Support/raw_ostream.h"

namespace glow {
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Type &type) {
  os << type.getElementName();

  if (type.isQuantizedType()) {
    os << "[S:";
    llvm::write_double(os, type.getScale(), llvm::FloatStyle::Fixed, 4);
    os << " O:";
    os << type.getOffset();
    os << ']';
    float low = (-128 - type.getOffset()) * type.getScale();
    float high = (127 - type.getOffset()) * type.getScale();
    os << "[";
    llvm::write_double(os, low, llvm::FloatStyle::Fixed, 3);
    os << ",";
    llvm::write_double(os, high, llvm::FloatStyle::Fixed, 3);
    os << "]";
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
