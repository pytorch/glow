// Copyright 2017 Facebook Inc.  All Rights Reserved.

#ifndef GLOW_QUANTIZATION_QUANTIZATION_H
#define GLOW_QUANTIZATION_QUANTIZATION_H

#include "glow/Graph/Graph.h"

#include <string>
#include <tuple>
#include <vector>

namespace glow {

/// Main attributes of a quantized tensor.
/// Scale and Offset allow quantization of a float tensor
/// and dequantization of integer tensor back to float one.
struct TensorQuantizationParams {
  float scale_;
  float offset_;
};

/// Tensor quantization parameters for a given node.
struct NodeQuantizationInfo {
  std::string nodeName_;
  TensorQuantizationParams tensorQuantizationParams_;

  NodeQuantizationInfo() {}
  NodeQuantizationInfo(const std::string &nodeName,
                       const TensorQuantizationParams &tensorQuantizationParams)
      : nodeName_(nodeName),
        tensorQuantizationParams_(tensorQuantizationParams) {}

  float Scale() const { return tensorQuantizationParams_.scale_; }
  float Offset() const { return tensorQuantizationParams_.offset_; }
};

/// Generate NodeQuantizationInfo for all required nodes from graph \p G.
std::vector<NodeQuantizationInfo> generateNodeQuantizationInfos(const Graph &G);

/// A data structure that represents the 32-bit to 8-bit quantization
/// scaling operation. This data structure represents the transformation:
/// (((input >> pre) * scale) + rtn) >> post + offset.
struct QuantizationTransform32To8 {
  int pre_;
  int post_;
  int scale_;
  int offset_;

public:
  /// Initializes the transformation based on the conversion formula (above).
  QuantizationTransform32To8(int pre, int post, int scale, int offset)
      : pre_(pre), post_(post), scale_(scale), offset_(offset) {}

  /// \returns the value \p in as clipped to the range [-128..127].
  static int8_t clip(int32_t in) {
    auto mx = std::numeric_limits<int8_t>::max();
    auto mn = std::numeric_limits<int8_t>::min();
    return std::max<int32_t>(mn, std::min<int32_t>(mx, in));
  }

  /// \returns the scaled integer.
  int32_t transform(int32_t input) {
    // The operation x >> y is rounded down to negative infinity. To get to
    // round-nearest we add (1 << (shift - 1)) to the value prior to shifting.
    int rtn = (1 << (post_ - 1));
    return ((((input >> pre_) * scale_) + rtn) >> post_) + offset_;
  }
};

/// Convert the floating point quantization parameters \p scale and \p offset
/// into the integer sequence of:
/// result = ((input >> pre) * scale) >> post + offset.
/// This scales a 32-bit signed integer word into an 8-bit signed integer.
/// \returns transformation parameters.
QuantizationTransform32To8 quantizeScaleOffset32To8(float scale, float offset);

} // namespace glow

#endif
