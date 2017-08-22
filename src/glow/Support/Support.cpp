#include "glow/Support/Support.h"

#include <sstream>

using namespace glow;

std::string glow::pointerToString(void *ptr) {
  std::ostringstream oss;
  oss << ptr;
  return oss.str();
}
