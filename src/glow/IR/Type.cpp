#include "glow/IR/Type.h"

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
}

