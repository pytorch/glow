// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#pragma once

#include "glow/Graph/Graph.h"
#include "nnpi_transformer_types.h"

namespace glow {

/// Utilities used across DSP kernel injectors.
struct DSPInjectorUtils {
  static int GetNumElements(uint32_t *dims, int numDims);

  static NNPICustomDSPNode *createCustomEltwiseFP16_configurable(
      Function *F_, const std::string &name, const std::string &kernel_name,
      std::vector<NodeValue> input_nodes, int64_t IceRefCallback,
      NNPITileParams tileParams);

  static NNPICustomDSPNode *
  createEltwiseFP16(Function *F_, const std::string &name,
                    const std::string &kernel_name,
                    std::vector<NodeValue> input_nodes, int64_t IceRefCallback);
};

} // namespace glow
