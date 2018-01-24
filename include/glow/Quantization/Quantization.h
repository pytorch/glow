// Copyright 2017 Facebook Inc.  All Rights Reserved.

#ifndef GLOW_QUANTIZATION_QUANTIZATION_H
#define GLOW_QUANTIZATION_QUANTIZATION_H

#include "glow/Graph/Graph.h"

#include <string>
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

  float Scale() { return tensorQuantizationParams_.scale_; }
  float Offset() { return tensorQuantizationParams_.offset_; }
};

/// Generate NodeQuantizationInfo for all required nodes from graph \p G.
std::vector<NodeQuantizationInfo> generateNodeQuantizationInfos(const Graph &G);

} // namespace glow

#endif