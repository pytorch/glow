/**
 * Copyright (c) 2017-present, Facebook, Inc.
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

/// Configuration for Quantization, passed into \ref quantizeFunction().
struct QuantizationConfiguration {
  /// Infos to use when determining scale and offset for all Nodes inside, and
  /// Placeholders and Constants referenced by, a Function being quantized.
  std::vector<NodeQuantizationInfo> infos{};

  /// Precision to use when quantizing a Function.
  ElemKind precision{ElemKind::Int8QTy};

  /// Schema to use when quantizing a Function.
  quantization::Schema schema{quantization::Schema::Asymmetric};

  /// Whether to use rowwise quantization when quantizing a Function.
  bool enableRowwise{false};

  /// New name for the quantized function. If no name is given then
  /// \ref quantizeFunction() will generate a name.
  std::string newFuncName{""};

  QuantizationConfiguration() = default;
  QuantizationConfiguration(llvm::ArrayRef<NodeQuantizationInfo> i)
      : infos(i) {}
};

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

/// Quantizes the function \p F into a new unoptimized partially quantized
/// function based on configuration from \p quantConfig. This method converts to
/// integer as many nodes as permitted by the backend \p B.
/// \p doNotQuantizeKinds lists kinds to not quantize, even if a profile was
/// gathered for them and the backend supports the quantized operation.  This
/// method clones original function \p F and caller is responsible for cleaning
/// up/erasing original function \p F if needed. \returns a new quantized
/// function.
Function *quantizeFunction(Function *F,
                           const QuantizationConfiguration &quantConfig,
                           const Backend &B,
                           const LoweredInfoMap &loweredMap = {},
                           const KindSet &doNotQuantizeKinds = {});

} // namespace quantization
} // namespace glow

#endif // GLOW_QUANTIZATION_QUANTIZATION_H
