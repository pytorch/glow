/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <cstdarg>
#include <sstream>
#include <string>

namespace llvm {
namespace yaml {
template <> struct BlockScalarTraits<glow::MultiLineStr> {
  static void output(const glow::MultiLineStr &Value, void *Ctxt,
                     llvm::raw_ostream &OS) {
    OS << Value.str;
  }

  static StringRef input(StringRef Scalar, void *Ctxt,
                         glow::MultiLineStr &Value) {
    Value.str = Scalar.str();
    return StringRef();
  }
};

template <> struct MappingTraits<glow::DeviceConfigHelper> {
  static void mapping(IO &io, glow::DeviceConfigHelper &info) {
    io.mapRequired("name", info.name_);
    io.mapRequired("backendName", info.backendName_);
    io.mapRequired("parameters", info.parameters_);
  }
};

} // end namespace yaml
} // end namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(glow::DeviceConfigHelper);

LLVM_YAML_IS_STRING_MAP(std::string);

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

void report(const char *msg) { errs() << msg; }

const std::string strFormat(const char *format, ...) {
  // Initialize use of varargs.
  va_list vaArgs;
  va_start(vaArgs, format);

  // Create a copy of the varargs.
  va_list vaArgsCopy;
  va_copy(vaArgsCopy, vaArgs);
  // Compute the length of the output to be produced.
  // The vsnprintf call does not actually write anything, but properly computes
  // the amount of characters that would be written.
  const int len = vsnprintf(NULL, 0, format, vaArgsCopy);
  va_end(vaArgsCopy);

  // Create a formatted string without any risk of memory issues.
  std::vector<char> str(len + 1);
  std::vsnprintf(str.data(), str.size(), format, vaArgs);
  va_end(vaArgs);
  return std::string(str.data(), len);
}

/// Create a formatted string that should live until the end of the execution.
const std::string &staticStrFormat(const char *format, ...) {
  // The storage for strings that should live until the end of the execution.
  static std::vector<std::string> staticStrings;
  // Initialize use of varargs.
  va_list vaArgs;
  va_start(vaArgs, format);

  // Create a copy of the varargs.
  va_list vaArgsCopy;
  va_copy(vaArgsCopy, vaArgs);
  // Compute the length of the output to be produced.
  // The vsnprintf call does not actually write anything, but properly computes
  // the amount of characters that would be written.
  const int len = vsnprintf(NULL, 0, format, vaArgsCopy);
  va_end(vaArgsCopy);

  // Create a formatted string without any risk of memory issues.
  std::vector<char> str(len + 1);
  std::vsnprintf(str.data(), str.size(), format, vaArgs);
  va_end(vaArgs);
  staticStrings.emplace_back(std::string(str.data(), len));
  return staticStrings.back();
}

std::string legalizeName(llvm::StringRef name) {
  std::string legalName;

  // Legalize the name.
  for (const char c : name) {
    bool legal = isalpha(c) || isdigit(c) || c == '_';
    legalName.push_back(legal ? c : '_');
  }

  // Names must start with some alphabetic character or underscore and can't be
  // empty.
  if (legalName.empty() || isdigit(legalName[0])) {
    legalName = "A" + legalName;
  }
  return legalName;
}

/// \returns the color based on \p index which is used in dot file.
const char *getDotFileNodeColor(size_t index) {
  static const char *colorNames[] = {
      "AliceBlue",      "CadetBlue1",   "Coral",      "DarkOliveGreen1",
      "DarkSeaGreen1",  "GhostWhite",   "Khaki1",     "LavenderBlush1",
      "LemonChiffon1",  "LightSkyBlue", "MistyRose1", "MistyRose2",
      "PaleTurquoise2", "PeachPuff1",   "PowderBlue", "Salmon",
      "Thistle1",       "Thistle3",     "Wheat1",     "Yellow2",
  };
  unsigned arrayLen = sizeof(colorNames) / sizeof(colorNames[0]);
  return colorNames[index % arrayLen];
}

template <typename T> static T deserializeFromYaml(llvm::StringRef fileName) {
  T result;
  llvm::outs() << fileName << "\n";
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> text =
      llvm::MemoryBuffer::getFileAsStream(fileName);
  assert(!text.getError() && "Unable to open file");

  std::unique_ptr<llvm::MemoryBuffer> buffer = std::move(*text);
  llvm::yaml::Input yin(buffer->getBuffer());
  yin >> result;

  assert(!yin.error() && "Error reading yaml file");

  return result;
}

std::vector<DeviceConfigHelper>
deserializeDeviceConfigFromYaml(llvm::StringRef fileName) {
  return deserializeFromYaml<std::vector<DeviceConfigHelper>>(fileName);
}

std::map<std::string, std::string>
deserializeStrStrMapFromYaml(llvm::StringRef fileName) {
  return deserializeFromYaml<std::map<std::string, std::string>>(fileName);
}

Expected<int> getIntFromStr(llvm::StringRef input) {
  const char *start = input.data();
  char *end;
  int val = std::strtol(start, &end, 10);
  RETURN_ERR_IF_NOT(!(end == start || *end != '\0'),
                    "Integer was not properly specified.");
  return val;
}

} // namespace glow
