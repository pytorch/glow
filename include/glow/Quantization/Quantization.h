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

#ifndef GLOW_QUANTIZATION_QUANTIZATION_H
#define GLOW_QUANTIZATION_QUANTIZATION_H

#include "glow/Graph/Graph.h"
#include "glow/Quantization/Base/Base.h"

#include <string>
#include <tuple>
#include <vector>

namespace glow {

class Backend;

namespace quantization {

/// Generate NodeQuantizationInfo for all required nodes from function \p F
/// using the method specified by \p schema and target quantization precision \p
/// quantizationPrecision. Profiling values will be written into context \p
/// bindings. \p loweredMap maps from the NodeOutputName of a NodeValue which
/// was lowered to a vector of the original NodeOutputNames which it replaced;
/// this map is used to generate infos for the original unlowered NodeValues
/// which no longer exist in \p F.
std::vector<NodeQuantizationInfo> generateNodeQuantizationInfos(
    PlaceholderBindings &bindings, const Function *F,
    const LoweredInfoMap &loweredMap = {}, Schema schema = Schema::Asymmetric,
    ElemKind quantizationPrecision = ElemKind::Int8QTy);

/// Quantizes the function \p F into an unoptimized partially quantized function
/// based on configuration from \p quantConfig. This method converts to integer
/// as many nodes as permitted by the backend \p B. \p loweredMap contains info
/// about what nodes were lowered from what, to be used during quantization.
/// \p doNotQuantizeKinds lists kinds to not quantize, even if a profile was
/// gathered for them and the backend supports the quantized operation.
void quantizeFunction(Function *F, const QuantizationConfiguration &quantConfig,
                      const Backend &B, const LoweredInfoMap &loweredMap = {},
                      const KindSet &doNotQuantizeKinds = {});

} // namespace quantization
} // namespace glow

#endif // GLOW_QUANTIZATION_QUANTIZATION_H
