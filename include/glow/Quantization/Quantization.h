// Copyright 2017 Facebook Inc.  All Rights Reserved.

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

/// A data structure that contains parameters for a float * 32-bit integer
/// multiplication operation, using only integer arithmetic.
struct MultTransformF32ToI32 {
  int pre_;
  int post_;
  int32_t m_;

  MultTransformF32ToI32(int pre, int post, int32_t m)
      : pre_(pre), post_(post), m_(m) {}

  int32_t transform(int32_t input) {
    // The operation x >> y is rounded down (toward negative infinity). We would
    // prefer a symmetric rounding strategy. To this extent we perform some
    // extra manipulation...
    int32_t a = (int32_t)((uint32_t)input >> 31);

    // TODO(hegemanjwh2): This branch is not good. It needs to go away. This
    // entire method should be branchless.
    if (pre_ > 0) {
      input += (a << (pre_ - 1));
    }

    // ((input >> pre_) * m_) >> post_
    return ((input >> pre_) * m_ + (a << (post_ - 1))) >> post_;
  }
};

namespace quantization {

/// Generate NodeQuantizationInfo for all required nodes from graph \p G.
std::vector<NodeQuantizationInfo>
generateNodeQuantizationInfos(const Function *F);

/// Compute the multiplication-transform parameters for performing a multiply
/// between a float and a 32-bit integer using only integer arithmetic.
MultTransformF32ToI32 computeMultTransformParams(float scale, int bits);

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
