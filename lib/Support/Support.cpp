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

#include "glow/Support/Support.h"
#include "llvm/Support/Debug.h"

#include <cctype>
#include <sstream>
#include <string>

namespace glow {
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, void *ptr) {
  std::ostringstream stringstream;
  stringstream << ptr;
  return os << stringstream.str();
}

llvm::raw_ostream &outs() { return llvm::outs(); }

llvm::raw_ostream &errs() { return llvm::errs(); }

llvm::raw_ostream &dbgs() { return llvm::dbgs(); }

std::string escapeDottyString(const std::string &str) {
  std::string out;
  out.reserve(str.capacity());
  for (unsigned char c : str) {
    if (std::isprint(c) && c != '\\' && c != '"' && c != '<' && c != '>') {
      out += c;
    } else {
      out += "\\";
      switch (c) {
      case '"':
        out += "\"";
        break;
      case '<':
        out += "<";
        break;
      case '>':
        out += ">";
        break;
      case '\\':
        out += "\\";
        break;
      case '\t':
        out += 't';
        break;
      case '\r':
        out += 'r';
        break;
      case '\n':
        // The marker '\l' means left-justify linebreak.
        out += 'l';
        break;
      default:
        char const *const hexdig = "0123456789ABCDEF";
        out += 'x';
        out += hexdig[c >> 4];
        out += hexdig[c & 0xF];
      }
    }
  }
  return out;
}
} // namespace glow
