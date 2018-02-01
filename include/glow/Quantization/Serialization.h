// Copyright 2017 Facebook Inc.  All Rights Reserved.

#ifndef GLOW_QUANTIZATION_SERIALIZATION_H
#define GLOW_QUANTIZATION_SERIALIZATION_H

#include "glow/Quantization/Quantization.h"

#include "llvm/ADT/APInt.h"

namespace glow {

/// Serialize \p quantizationInfos into the file named \p fileName.
void serializeToYaml(llvm::StringRef fileName,
                     llvm::ArrayRef<NodeQuantizationInfo> quantizationInfos);

/// Deserialize quantization infos from the file \p fileName.
std::vector<NodeQuantizationInfo> deserializeFromYaml(llvm::StringRef fileName);

} // namespace glow

#endif
