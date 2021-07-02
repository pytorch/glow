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

#pragma once

#include "glow/Graph/Graph.h"
#include "nnpi_transformer_types.h"

namespace glow {

/// Utilities used across DSP kernel injectors.
struct DSPInjectorUtils {
  static int GetNumElements(uint32_t *dims, int numDims);

  static NNPICustomDSPNode *createCustomEltwise_configurable(
      Function *F_, const std::string &name, const std::string &kernel_name,
      std::vector<NodeValue> input_nodes, int64_t IceRefCallback,
      NNPITileParams tileParamsInput, NNPITileParams tileParamsOutput,
      const int itemsPerLoopIter, const ElemKind outputElemKind);

  static NNPICustomDSPNode *
  createEltwiseFP16(Function *F_, const std::string &name,
                    const std::string &kernel_name,
                    std::vector<NodeValue> input_nodes, int64_t IceRefCallback);

  static NNPICustomDSPNode *createEltwiseInt32Compare(
      Function *F_, const std::string &name, const std::string &kernel_name,
      std::vector<NodeValue> input_nodes, int64_t IceRefCallback);
};

} // namespace glow
