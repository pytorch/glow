#ifndef GLOW_SUPPORT_SUPPORT_H
#define GLOW_SUPPORT_SUPPORT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <iostream>
#include <sstream>

namespace glow {

/// Convert the ptr \p ptr into an ascii representation in the format "0xFFF..."
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, void *ptr);

/// Stream LLVM's ArrayRef into the given output stream.
template <typename E>
llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const llvm::ArrayRef<E> list) {
  os << '[';
  // Print the array without a trailing comma.
  for (size_t i = 0, e = list.size(); i < e; i++) {
    if (i) {
      os << ", ";
    }
    os << list[i];
  }
  os << ']';

  return os;
}

/// \returns the escaped content of string \p str.
/// The char '\n' becomes '\'+'n' and quotes are handled correctly.
std::string escapeDottyString(const std::string &str);

/// Add quotes to the string \p in.
inline std::string quote(const std::string &in) { return '"' + in + '"'; }

/// \returns the content of the string \p in after conversion to lower case.
inline std::string tolower(const std::string &in) {
  std::string data = in;
  std::transform(data.begin(), data.end(), data.begin(), ::tolower);
  return data;
}

/// \returns the content of the string \p in after conversion to sentence case.
inline std::string tosentence(const std::string &in) {
  std::string data = in;
  if (data.length() > 0) {
    auto itr = ++data.begin();
    std::transform(data.begin(), itr, data.begin(), ::toupper);
    std::transform(itr, data.end(), itr, ::tolower);
  }
  return data;
}

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
  std::string buffer_;
  llvm::raw_string_ostream repr_;

public:
  explicit DescriptionBuilder(const char *name) : repr_(buffer_) {
    repr_ << name << '\n';
  }

  DescriptionBuilder &addParam(const std::string &name, const char *value) {
    repr_ << name << " : " << value << '\n';
    return *this;
  }

  template <typename E>
  DescriptionBuilder &addParam(const std::string &name, llvm::ArrayRef<E> L) {
    repr_ << name << " : " << L << '\n';
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
    repr_ << name << " : " << value << '\n';
    return *this;
  }

  operator std::string() { return repr_.str(); }
};

} // namespace glow

#endif // GLOW_SUPPORT_SUPPORT_H
