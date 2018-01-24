// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Quantization/Serialization.h"
#include "glow/Quantization/Quantization.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

/// Mapping for NodeQuantizationInfo yaml serializer.
template <> struct llvm::yaml::MappingTraits<glow::NodeQuantizationInfo> {
  static void mapping(IO &io, glow::NodeQuantizationInfo &info) {
    io.mapRequired("nodeOutputName", info.nodeName_);
    float scale = info.Scale();
    float offset = info.Offset();
    io.mapRequired("scale", scale);
    io.mapRequired("offset", offset);
  }
};

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

} // namespace glow
