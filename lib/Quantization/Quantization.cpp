// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Quantization/Quantization.h"

namespace glow {

std::vector<NodeQuantizationInfo>
generateNodeQuantizationInfos(const Graph &G) {
  std::vector<NodeQuantizationInfo> quantizationInfos;

  for (auto *node : G.getNodes()) {
    auto *QPN = llvm::dyn_cast<QuantizationProfileNode>(node);

    if (QPN) {
      auto CI = QPN->getComputationInfoVar()->getHandle<float>();
      float min = CI.raw(0);
      float max = CI.raw(1);

      NodeValue &observedNodeValue = node->getNthInput(0);
      unsigned resNum = observedNodeValue.getResNo();
      Node *observedNode = observedNodeValue.getNode();

      std::string nodeName =
          observedNode->getName().str() + ":" + std::to_string(resNum);

      // TODO: calculate TensorQuantizationParams based on the histogram,
      // min and max values.
      // Just dump min and max for now.
      TensorQuantizationParams TQP{min, max};
      quantizationInfos.emplace_back(nodeName, TQP);
    }
  }

  return quantizationInfos;
}

} // namespace glow
