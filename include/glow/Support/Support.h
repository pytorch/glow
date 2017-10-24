#ifndef GLOW_SUPPORT_SUPPORT_H
#define GLOW_SUPPORT_SUPPORT_H

#include "llvm/ADT/ArrayRef.h"

#include <chrono>
#include <iostream>
#include <sstream>

namespace std {
/// Convert the ptr \p ptr into an ascii representation in the format "0xFFF..."
std::string to_string(void *ptr);
/// Converts LLVM's StringRef to std::string.
std::string to_string(const llvm::StringRef sr);
/// Converts LLVM's ArrayRef to std::string.
template <typename E> std::string to_string(const llvm::ArrayRef<E> list) {
  std::ostringstream buffer;
  buffer << '[';
  // Print the array without a trailing comma.
  for (size_t i = 0, e = list.size(); i < e; i++) {
    if (i) {
      buffer << ", ";
    }
    buffer << std::to_string(list[i]);
  }
  buffer << ']';
  return buffer.str();
}
} // namespace std

namespace glow {

/// \returns the escaped content of string \p str.
/// The char '\n' becomes '\'+'n' and quotes are handled correctly.
std::string escapeDottyString(const std::string &str);

/// Add quotes to the string \p in.
inline std::string quote(const std::string &in) { return '"' + in + '"'; }

template <typename E> std::string listToString_impl(E v) {
  return std::to_string(v);
}

template <typename E, typename... Args>
std::string listToString_impl(E first, Args... args) {
  return std::to_string(first) + " " + listToString_impl(args...);
}

template <typename... Args> std::string listToString(Args... args) {
  return "[" + listToString_impl(args...) + "]";
}

/// A helper class that builds a textual descriptor of a group of parameters.
class DescriptionBuilder {
  std::stringstream repr_;

public:
  DescriptionBuilder(const std::string &name) { repr_ << name << '\n'; }

  DescriptionBuilder &addParam(const std::string &name, const char *value) {
    repr_ << name << " : " << value << '\n';
    return *this;
  }

  template <typename E>
  DescriptionBuilder &addParam(const std::string &name, llvm::ArrayRef<E> L) {
    repr_ << name << " : " << std::to_string(L) << '\n';
    return *this;
  }

  template <typename T_,
            typename = typename std::enable_if<std::is_scalar<T_>::value>::type>
  DescriptionBuilder &addParam(const std::string &name, T_ value) {
    repr_ << name << " : " << value << '\n';
    return *this;
  }

  template <typename T_, typename = typename std::enable_if<
                             !std::is_scalar<T_>::value>::type>
  DescriptionBuilder &addParam(const std::string &name, const T_ &value) {
    repr_ << name << " : " << std::to_string(value) << '\n';
    return *this;
  }

  operator std::string() const { return repr_.str(); }
};

} // namespace glow

#endif // GLOW_SUPPORT_SUPPORT_H
