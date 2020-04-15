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

#include "glow/Quantization/Serialization.h"

#include "glow/Quantization/Base/Base.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include <glog/logging.h>

namespace llvm {
namespace yaml {

/// The default behavior of YAML is to serialize floating point numbers
/// using the "%g" format specifier which is not guaranteed to print all
/// the decimals. During a round-trip (serialize, deserialize) decimals
/// might be lost and hence precision is lost. Although this might not be
/// critical for some quantization schema, for "SymmetricWithPower2Scale"
/// the round-trip must preserve the exact representation of the floating
/// point scale which is a power of 2. The code below is a workaround to
/// overwrite the behavior of the YAML serializer to print all the digits.
struct FloatWrapper {
  float val_;
  FloatWrapper(float val) : val_(val) {}
};

template <> struct ScalarTraits<FloatWrapper> {
  static void output(const FloatWrapper &value, void *ctxt,
                     llvm::raw_ostream &out) {
    // Print number with all the digits and without trailing 0's
    char buffer[200];
    snprintf(buffer, sizeof(buffer), "%.126f", value.val_);
    int n = strlen(buffer) - 1;
    while ((n > 0) && (buffer[n] == '0') && (buffer[n - 1] != '.')) {
      buffer[n--] = '\0';
    }
    out << buffer;
  }
  static StringRef input(StringRef scalar, void *ctxt, FloatWrapper &value) {
    if (to_float(scalar, value.val_))
      return StringRef();
    return "invalid floating point number";
  }
  static QuotingType mustQuote(StringRef) { return QuotingType::None; }
};

/// Mapping for NodeProfilingInfo yaml serializer.
template <> struct MappingTraits<glow::NodeProfilingInfo> {
  struct FloatNormalized {
    FloatNormalized(IO &io) : val_(0.0) {}
    FloatNormalized(IO &, float &val) : val_(val) {}
    float denormalize(IO &) { return val_.val_; }
    FloatWrapper val_;
  };
  static void mapping(IO &io, glow::NodeProfilingInfo &info) {
    MappingNormalization<FloatNormalized, float> min(
        io, info.tensorProfilingParams_.min);
    MappingNormalization<FloatNormalized, float> max(
        io, info.tensorProfilingParams_.max);
    io.mapRequired("nodeOutputName", info.nodeOutputName_);
    io.mapRequired("min", min->val_);
    io.mapRequired("max", max->val_);
    io.mapRequired("histogram", info.tensorProfilingParams_.histogram);
  }
};

/// Mapping for NodeQuantizationInfo yaml serializer.
template <> struct MappingTraits<glow::NodeQuantizationInfo> {
  struct FloatNormalized {
    FloatNormalized(IO &io) : val_(0.0) {}
    FloatNormalized(IO &, float &val) : val_(val) {}
    float denormalize(IO &) { return val_.val_; }
    FloatWrapper val_;
  };
  static void mapping(IO &io, glow::NodeQuantizationInfo &info) {
    MappingNormalization<FloatNormalized, float> scale(
        io, info.tensorQuantizationParams_.scale);
    io.mapRequired("nodeOutputName", info.nodeOutputName_);
    io.mapRequired("scale", scale->val_);
    io.mapRequired("offset", info.tensorQuantizationParams_.offset);
  }
};

} // end namespace yaml
} // end namespace llvm

/// Yaml serializer for vector of NodeProfilingInfo.
LLVM_YAML_IS_SEQUENCE_VECTOR(glow::NodeProfilingInfo);

/// Yaml serializer for vector of NodeQuantizationInfo.
LLVM_YAML_IS_SEQUENCE_VECTOR(glow::NodeQuantizationInfo);

namespace glow {

void serializeProfilingInfosToYaml(
    llvm::StringRef fileName,
    llvm::ArrayRef<NodeProfilingInfo> profilingInfos) {
  std::error_code EC;
  llvm::raw_fd_ostream outputStream(fileName, EC, llvm::sys::fs::F_None);
  CHECK(!EC) << "Unable to create output stream";

  llvm::yaml::Output yout(outputStream);
  // LLVM_YAML_IS_SEQUENCE_VECTOR cannot serialize ArrayRef.
  // Explicitly use a separate vector to allow serialization.
  std::vector<NodeProfilingInfo> info = profilingInfos;
  yout << info;
}

std::vector<NodeProfilingInfo>
deserializeProfilingInfosFromYaml(llvm::StringRef fileName) {
  std::vector<NodeProfilingInfo> result;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> text =
      llvm::MemoryBuffer::getFileAsStream(fileName);
  CHECK(!text.getError()) << "Unable to open file with name: "
                          << fileName.str();

  std::unique_ptr<llvm::MemoryBuffer> buffer = std::move(*text);
  llvm::yaml::Input yin(buffer->getBuffer());
  yin >> result;

  CHECK(!yin.error()) << "Error reading yaml file";

  return result;
}

void serializeQuantizationInfosToYaml(
    llvm::StringRef fileName,
    llvm::ArrayRef<NodeQuantizationInfo> quantizationInfos) {
  std::error_code EC;
  llvm::raw_fd_ostream outputStream(fileName, EC, llvm::sys::fs::F_None);
  CHECK(!EC) << "Unable to create output stream";

  llvm::yaml::Output yout(outputStream);
  // LLVM_YAML_IS_SEQUENCE_VECTOR cannot serialize ArrayRef.
  // Explicitly use a separate vector to allow serialization.
  std::vector<NodeQuantizationInfo> info = quantizationInfos;
  yout << info;
}

std::vector<NodeQuantizationInfo>
deserializeQuantizationInfosFromYaml(llvm::StringRef fileName) {
  std::vector<NodeQuantizationInfo> result;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> text =
      llvm::MemoryBuffer::getFileAsStream(fileName);
  CHECK(!text.getError()) << "Unable to open file with name: "
                          << fileName.str();

  std::unique_ptr<llvm::MemoryBuffer> buffer = std::move(*text);
  llvm::yaml::Input yin(buffer->getBuffer());
  yin >> result;

  CHECK(!yin.error()) << "Error reading yaml file";

  return result;
}

} // namespace glow
