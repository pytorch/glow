#ifndef GLOW_SUPPORT_SUPPORT_H
#define GLOW_SUPPORT_SUPPORT_H

#include "llvm/ADT/ArrayRef.h"

#include <chrono>
#include <iostream>
#include <sstream>

namespace glow {

/// A class for measuring the lifetime of some event
/// and for rate calculation.
class TimerGuard {
  int iterations_;
  std::chrono::time_point<std::chrono::system_clock> start;

public:
  TimerGuard(int iterations) : iterations_(iterations) {
    start = std::chrono::system_clock::now();
  }

  ~TimerGuard() {
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Rate: " << (iterations_ / elapsed_seconds.count())
              << "/sec\n";
  }
};

/// Convert the ptr \p ptr into an ascii representation in the format
/// "0xFFF...";
std::string pointerToString(void *ptr);

/// \returns the escaped content of string \p str.
/// The char '\n' becomes '\'+'n' and quotes are handled correctly.
std::string escapeDottyString(const std::string &str);

/// A helper class that builds a textual descriptor of a group of parameters.
struct DescriptionBuilder {
  DescriptionBuilder(const std::string &name);
  std::stringstream repr_;
  DescriptionBuilder &addDim(const std::string &name,
                             llvm::ArrayRef<size_t> dims);
  DescriptionBuilder &addParam(const std::string &name, size_t param);
  DescriptionBuilder &addParam(const std::string &name, double param);
  DescriptionBuilder &addParam(const std::string &name, std::string param);

  operator std::string() { return repr_.str(); }
};

} // namespace glow

#endif // GLOW_SUPPORT_SUPPORT_H
