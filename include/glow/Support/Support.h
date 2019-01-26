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

/// \returns output stream for stdout.
llvm::raw_ostream &outs();

/// \returns output stream for stderr.
llvm::raw_ostream &errs();

/// \returns output stream for debug messages.
llvm::raw_ostream &dbgs();

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

/// Print \p msg on the error stream.
void report(const char *msg);
inline void report(const std::string &str) { report(str.c_str()); }
inline void report(llvm::StringRef str) { report(str.data()); }

/// Printf-like formatting for std::string.
const std::string strFormat(const char *format, ...)
#ifndef _MSC_VER
    __attribute__((__format__(__printf__, 1, 2)));
#endif
;
} // namespace glow

#endif // GLOW_SUPPORT_SUPPORT_H
