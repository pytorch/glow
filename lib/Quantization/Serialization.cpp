// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Quantization/Serialization.h"
#include "glow/Quantization/Quantization.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace yaml {

/// Mapping for NodeQuantizationInfo yaml serializer.
template <> struct MappingTraits<glow::NodeQuantizationInfo> {
  static void mapping(IO &io, glow::NodeQuantizationInfo &info) {
    io.mapRequired("nodeOutputName", info.nodeOutputName_);
    io.mapRequired("scale", info.tensorQuantizationParams_.scale_);
    io.mapRequired("offset", info.tensorQuantizationParams_.offset_);
  }
};

} // end namespace yaml
} // end namespace llvm

/// Yaml serializer for vector of NodeQuantizationInfo.
LLVM_YAML_IS_SEQUENCE_VECTOR(glow::NodeQuantizationInfo);

namespace glow {

void serializeToYaml(llvm::StringRef fileName,
                     llvm::ArrayRef<NodeQuantizationInfo> quantizationInfos) {
  std::error_code EC;
  llvm::raw_fd_ostream outputStream(fileName, EC, llvm::sys::fs::F_None);
  GLOW_ASSERT(!EC && "Unable to create output stream");

  llvm::yaml::Output yout(outputStream);
  // LLVM_YAML_IS_SEQUENCE_VECTOR cannot serialize ArrayRef.
  // Explicitly use a separate vector to allow serialization.
  std::vector<NodeQuantizationInfo> info = quantizationInfos;
  yout << info;
}

std::vector<NodeQuantizationInfo>
deserializeFromYaml(llvm::StringRef fileName) {
  std::vector<NodeQuantizationInfo> result;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> text =
      llvm::MemoryBuffer::getFileAsStream(fileName);
  GLOW_ASSERT(!text.getError() && "Unable to open file");

  std::unique_ptr<llvm::MemoryBuffer> buffer = std::move(*text);
  llvm::yaml::Input yin(buffer->getBuffer());
  yin >> result;

  GLOW_ASSERT(!yin.error() && "Error reading yaml file");

  return result;
}

} // namespace glow
