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

#include <string>
#include <tuple>
#include <vector>

namespace glow {

class ExecutionEngine;

/// Main attributes of a quantized tensor.
/// Scale and Offset allow quantization of a float tensor and dequantization of
/// integer tensor back to float one.
struct TensorQuantizationParams {
  float scale_;
  int32_t offset_;
};

/// Tensor quantization parameters for a given node.
struct NodeQuantizationInfo {
  std::string nodeOutputName_;
  TensorQuantizationParams tensorQuantizationParams_;

  NodeQuantizationInfo() = default;
  NodeQuantizationInfo(const std::string &nodeOutputName,
                       const TensorQuantizationParams &tensorQuantizationParams)
      : nodeOutputName_(nodeOutputName),
        tensorQuantizationParams_(tensorQuantizationParams) {}

  float Scale() const { return tensorQuantizationParams_.scale_; }
  int32_t Offset() const { return tensorQuantizationParams_.offset_; }

  /// Get the full node output name based on the node name and output number.
  /// The following format is used: nodename:outputNumber
  static std::string generateNodeOutputName(const std::string &nodeName,
                                            unsigned outputNumber = 0) {
    return nodeName + ":" + std::to_string(outputNumber);
  }
};

/// A data structure that represents the 32-bit to 8-bit quantization
/// scaling operation. This data structure represents the transformation:
/// (((input >> pre) * scale) + rtn) >> post + offset.
struct QuantizationTransform32To8 {
  int pre_;
  int post_;
  int scale_;
  int offset_;

  /// Initializes the transformation based on the conversion formula (above).
  QuantizationTransform32To8(int pre, int post, int scale, int offset)
      : pre_(pre), post_(post), scale_(scale), offset_(offset) {}

  /// \returns the scaled integer.
  int32_t transform(int32_t input) {
    // The operation x >> y is rounded down to negative infinity. To get to
    // round-nearest we add (1 << (shift - 1)) to the value prior to shifting.
    int rtn = (1 << (post_ - 1));
    return ((((input >> pre_) * scale_) + rtn) >> post_) + offset_;
  }
};

namespace quantization {

/// Generate NodeQuantizationInfo for all required nodes from graph \p G.
std::vector<NodeQuantizationInfo>
generateNodeQuantizationInfos(const Function *F);

/// Convert the floating point quantization parameters \p scale and \p offset
/// into the integer sequence of:
/// result = ((input >> pre) * scale) >> post + offset.
/// This scales a 32-bit signed integer word into an 8-bit signed integer.
/// \returns transformation parameters.
QuantizationTransform32To8 quantizeScaleOffset32To8(float scale,
                                                    int32_t offset);

/// Converts floating point value to int8 based on the quantization
/// parameters \p TQP.
int8_t quantize(float input, const TensorQuantizationParams &TQP);

/// Converts int8 quantized value back to floating point number based on
/// the quantization parameters \p TQP.
float dequantize(int8_t input, const TensorQuantizationParams &TQP);

/// \returns the value \p in as clipped to the range of \p DestTy.
template <class SrcTy, class DestTy> DestTy clip(SrcTy in) {
  assert(sizeof(SrcTy) >= sizeof(DestTy) && "Invalid types");

  auto mx = std::numeric_limits<DestTy>::max();
  auto mn = std::numeric_limits<DestTy>::min();
  return std::max<SrcTy>(mn, std::min<SrcTy>(mx, in));
}

/// Converts floating point graph to a quantized one.
/// Note, if not all operators have a conversion support graph ends up being
/// hybrid.
void generateQuantizedGraph(
    const ExecutionEngine &EE, Function *F,
    llvm::ArrayRef<NodeQuantizationInfo> quantizationInfos);

} // namespace quantization

} // namespace glow

#endif
