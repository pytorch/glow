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

#ifndef GLOW_QUANTIZATION_SERIALIZATION_H
#define GLOW_QUANTIZATION_SERIALIZATION_H

#include "glow/Quantization/Base/Base.h"

#include "llvm/ADT/APInt.h"

namespace glow {

/// Serialize \p profilingInfos into the file named \p fileName.
void serializeProfilingInfosToYaml(
    llvm::StringRef fileName, llvm::ArrayRef<NodeProfilingInfo> profilingInfos);

/// Deserialize profiling infos from the file \p fileName.
std::vector<NodeProfilingInfo>
deserializeProfilingInfosFromYaml(llvm::StringRef fileName);

/// Serialize \p quantizationInfos into the file named \p fileName.
void serializeQuantizationInfosToYaml(
    llvm::StringRef fileName,
    llvm::ArrayRef<NodeQuantizationInfo> quantizationInfos);

/// Deserialize quantization infos from the file \p fileName.
std::vector<NodeQuantizationInfo>
deserializeQuantizationInfosFromYaml(llvm::StringRef fileName);

} // namespace glow

#endif
