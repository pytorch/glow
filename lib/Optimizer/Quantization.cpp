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

#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"

#include <unordered_set>

using namespace glow;

/// Instrument function \p F with Quantization Profile nodes.
/// Nodes should only be added when function is compiled in
/// the Inference mode.
void glow::profileQuantization(Function *F) {
  // Iterate over all nodes in the graph and insert QuantizationProfile nodes
  // to observe tensor values from every node's output.
  std::unordered_set<NodeValue> nodesToInstrument;

  // Add Quantization Profile node to all of the floating point outputs.
  for (const auto &node : F->getNodes()) {
    for (unsigned i = 0, e = node->getNumResults(); i < e; ++i) {
      if (node->getNthResult(i).getElementType() != ElemKind::FloatTy) {
        continue;
      }
      nodesToInstrument.insert(node->getNthResult(i));
    }
  }

  // Add Quantization Profile node to all floating point vars.
  for (const auto &var : F->getParent()->getVars()) {
    if (var->getNthResult(0).getElementType() != ElemKind::FloatTy) {
      continue;
    }
    // Assuming varable has only a single output.
    nodesToInstrument.insert(var->getNthResult(0));
  }

  for (const auto &node : nodesToInstrument) {
    F->createQuantizationProfile("QuantizationProfile", node);
  }
}
