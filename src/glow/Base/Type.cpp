// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Base/Type.h"

#include <sstream>

using namespace glow;

namespace std {
std::string to_string(const glow::Type &type) {
  if (type.numSizes_ == 0) {
    return "<void>";
  }

  std::stringstream os;
  os << type.getElementName().str() << '<';
  for (unsigned i = 0; i < type.numSizes_; ++i) {
    if (i) {
      os << " x ";
    }
    os << type.sizes_[i];
  }
  os << '>';

  return os.str();
}
std::string to_string(const glow::TypeRef &type) {
  if (!type)
    return "<none>";
  return std::to_string(*type);
}

} // namespace std
