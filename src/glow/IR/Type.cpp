#include "glow/IR/Type.h"

#include <sstream>

using namespace glow;

std::string Type::asString() const {
  if (!numSizes_) {
    return "<void>";
  }

  std::stringstream sb;
  sb << getElementName().str() << "<";

  for (unsigned i = 0; i < numSizes_; i++) {
    if (i) {
      sb << " x ";
    }
    sb << sizes_[i];
  }

  sb << ">";
  return sb.str();
}
