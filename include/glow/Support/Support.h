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
std::string to_string(llvm::StringRef sr);
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

/// \returns the content of the string \p in after conversion to lower case.
inline std::string tolower(const std::string &in) {
  std::string data = in;
  std::transform(data.begin(), data.end(), data.begin(), ::tolower);
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
  std::vector<std::vector<std::string>> rows_;

public:
  explicit DescriptionBuilder(const char *name): rows_(2) {
    rows_[1].push_back(name);
  }

  DescriptionBuilder &addParam(size_t row_n, const std::string &name, const char *value) {
    rows_[row_n].push_back(name + " : " + value);
    return *this;
  }

  template <typename E>
  DescriptionBuilder &addParam(size_t row_n, const std::string &name, llvm::ArrayRef<E> L) {
    rows_[row_n].push_back(name + " : " + std::to_string(L));
    return *this;
  }

  template <typename T_,
            typename = typename std::enable_if<std::is_scalar<T_>::value>::type>
  DescriptionBuilder &addParam(size_t row_n, const std::string &name, T_ value) {
    std::ostringstream line;
    line << name << " : " << value;
    rows_[row_n].push_back(line.str());
    return *this;
  }

  template <typename T_, typename = typename std::enable_if<
                             !std::is_scalar<T_>::value>::type>
  DescriptionBuilder &addParam(size_t row_n, const std::string &name, const T_ &value) {
    rows_[row_n].push_back(name + " : " + std::to_string(value));
    return *this;
  }

  std::string toGraphNodeString() const {
    std::vector<std::vector<std::string>> escaped_rows(2);
    for (size_t i = 0; i < 2; i++) {
      escaped_rows[i].resize(rows_[i].size());
      for (size_t j = 0; j < rows_[i].size(); j++)
        escaped_rows[i][j] = escapeDottyString(rows_[i][j]);
    }

    std::ostringstream result;
    result << "{";
    if (!escaped_rows[0].empty()) {
      result << "{<i0>" << escaped_rows[0][0];
      for (size_t i = 1; i < escaped_rows[0].size(); i++) {
        result << "|<i" << i << ">" << escaped_rows[0][i];
      }
      result << "}";
    }
    if (!escaped_rows[0].empty() && !escaped_rows[1].empty()) {
      result << "|";
    }
    if (!escaped_rows[1].empty()) {
      result << "{";
      for (const auto& elem : escaped_rows[1]) {
        result << elem << "\\l";
      }
      result << "}";
    }
    result << "}";
    return result.str();
  }

  operator std::string() const {
    std::ostringstream result;
    for (int i = 1; i >= 0; i--) {
      for (const auto& elem : rows_[i]) {
        result << elem << '\n';
      }
    }
    return result.str();
  }
};

} // namespace glow

#endif // GLOW_SUPPORT_SUPPORT_H
