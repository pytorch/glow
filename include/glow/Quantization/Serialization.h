// Copyright 2017 Facebook Inc.  All Rights Reserved.

#ifndef GLOW_QUANTIZATION_SERIALIZATION_H
#define GLOW_QUANTIZATION_SERIALIZATION_H

#include "glow/Quantization/Quantization.h"

#include "llvm/ADT/APInt.h"

namespace glow {

void serializeToYaml(llvm::StringRef fileName,
                     llvm::ArrayRef<NodeQuantizationInfo> quantizationInfos);

} // namespace glow

#endif
