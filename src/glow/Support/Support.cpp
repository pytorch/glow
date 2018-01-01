// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Support/Support.h"

#include <cctype>
#include <sstream>
#include <string>

namespace glow {
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, void *ptr) {
  std::ostringstream stringstream;
  stringstream << ptr;
  return os << stringstream.str();
}

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
