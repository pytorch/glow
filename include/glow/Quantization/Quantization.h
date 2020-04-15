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

/// Generate NodeProfilingInfo for all the required nodes from function \p F
/// using the profiling information stored in \p bindings obtained after
/// running the network in profiling mode. During the profiling phase all the
/// nodes are lowered. The map \p loweredMap maps the vector of NodeOutputNames
/// obtained after lowering a NodeValue to the NodeOutputName of the lowered
/// NodeValue. This map is used to replicate the profiling information for the
/// unlowered NodeValues such that during the actual quantization, depending
/// on the backend decision to lower some nodes or not, a profile will be found
/// for any NodeValue.
std::vector<NodeProfilingInfo>
generateNodeProfilingInfos(PlaceholderBindings &bindings, const Function *F,
                           const LoweredInfoMap &loweredMap = {});

/// Generate NodeQuantizationInfo for all the required nodes from function \p F
/// using the profiling information and the parameters from \p quantConfig. The
/// map \p loweredMap is the lowering map obtained during the quantization phase
/// and is used to find lowering patterns for the bias operands.
std::vector<NodeQuantizationInfo>
generateNodeQuantizationInfos(Function *F,
                              const QuantizationConfiguration &quantConfig,
                              const LoweredInfoMap &loweredMap = {});

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
