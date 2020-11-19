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
#include "glow/Support/Support.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include <glog/logging.h>

/// Yaml serializer for the Glow tools version.
LLVM_YAML_STRONG_TYPEDEF(std::string, YAMLGlowToolsVersion)

/// Yaml serializer for the graph hash code.
LLVM_YAML_STRONG_TYPEDEF(llvm::yaml::Hex64, YAMLGraphPreLowerHash)

/// Yaml serializer for vector of NodeProfilingInfo.
LLVM_YAML_IS_SEQUENCE_VECTOR(glow::NodeProfilingInfo);

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

/// Mapping for YAMLGlowToolsVersion yaml serializer.
template <> struct MappingTraits<YAMLGlowToolsVersion> {
  static void mapping(IO &io, YAMLGlowToolsVersion &ver) {
    io.mapRequired("GlowToolsVersion", ver.value);
  }
};

/// Mapping for YAMLGraphPreLowerHash yaml serializer.
template <> struct MappingTraits<YAMLGraphPreLowerHash> {
  static void mapping(IO &io, YAMLGraphPreLowerHash &hash) {
    io.mapRequired("GraphPreLowerHash", hash.value);
  }
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
    io.mapRequired("NodeOutputName", info.nodeOutputName_);
    io.mapRequired("Min", min->val_);
    io.mapRequired("Max", max->val_);
    io.mapRequired("Histogram", info.tensorProfilingParams_.histogram);
  }
};

} // end namespace yaml
} // end namespace llvm

namespace glow {

void serializeProfilingInfosToYaml(
    llvm::StringRef fileName, llvm::hash_code graphPreLowerHash,
    std::vector<NodeProfilingInfo> &profilingInfos) {

  // Open YAML output stream.
  std::error_code EC;
  llvm::raw_fd_ostream outputStream(fileName, EC, llvm::sys::fs::F_None);
  CHECK(!EC) << "Error opening YAML file '" << fileName.str() << "'!";
  llvm::yaml::Output yout(outputStream);

  // Write Glow tools version.
#ifdef GLOW_BUILD_DATE
  YAMLGlowToolsVersion yamlVersion = YAMLGlowToolsVersion(GLOW_BUILD_DATE);
#else
  YAMLGlowToolsVersion yamlVersion = YAMLGlowToolsVersion("");
#endif
  yout << yamlVersion;

  // Write graph hash.
  auto uint64Hash = static_cast<uint64_t>(graphPreLowerHash);
  YAMLGraphPreLowerHash yamlHash = llvm::yaml::Hex64(uint64Hash);
  yout << yamlHash;

  // Write profiling info.
  yout << profilingInfos;
}

bool deserializeProfilingInfosFromYaml(
    llvm::StringRef fileName, llvm::hash_code &graphPreLowerHash,
    std::vector<NodeProfilingInfo> &profilingInfos) {

  if (!llvm::sys::fs::exists(fileName)) {
    return false;
  }

  // Open YAML input stream.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> text =
      llvm::MemoryBuffer::getFileAsStream(fileName);
  CHECK(!text.getError()) << "Unable to open file with name: "
                          << fileName.str();
  std::unique_ptr<llvm::MemoryBuffer> buffer = std::move(*text);
  llvm::yaml::Input yin(buffer->getBuffer());

  // Error message in case of incorrect profile format.
  std::string profileErrMsg =
      strFormat("Error reading YAML file '%s'!", fileName.data());
#ifdef GLOW_BUILD_DATE
  profileErrMsg += strFormat(" Verify that the YAML file was generated with "
                             "the current version (%s) of the Glow tools!",
                             GLOW_BUILD_DATE);
#endif

  // Read Glow tools version.
  YAMLGlowToolsVersion yamlVersion;
  yin >> yamlVersion;
  CHECK(yin.nextDocument()) << profileErrMsg;

  // Read graph hash.
  YAMLGraphPreLowerHash hash;
  yin >> hash;
  graphPreLowerHash = llvm::hash_code(static_cast<size_t>(hash.value));
  CHECK(yin.nextDocument()) << profileErrMsg;

  // Read profiling info.
  yin >> profilingInfos;
  CHECK(!yin.error()) << profileErrMsg;
  return true;
}

} // namespace glow
