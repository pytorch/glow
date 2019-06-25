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
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

#include <unordered_set>

using namespace glow;

void glow::profileQuantization(PlaceholderBindings &bindings, Function *F) {
  // Iterate over all nodes in the graph and insert QuantizationProfile nodes
  // to observe tensor values from every node's output.
  std::unordered_set<NodeValue> nodesToInstrument;

  // Add Quantization Profile node to all of the floating point outputs.
  for (auto &node : F->getNodes()) {
    for (unsigned i = 0, e = node.getNumResults(); i < e; ++i) {
      if (node.getElementType(i) != ElemKind::FloatTy) {
        continue;
      }
      nodesToInstrument.insert(node.getNthResult(i));
    }
  }

  // Add Quantization Profile node to all floating point vars.
  for (const auto &var : F->getParent()->getConstants()) {
    if (var->getOutput().getElementType() != ElemKind::FloatTy) {
      continue;
    }
    nodesToInstrument.insert(var->getOutput());
  }

  // Add Quantization Profile node to all floating point placeholders.
  for (const auto &PH : F->getParent()->getPlaceholders()) {
    if (PH->getOutput().getElementType() != ElemKind::FloatTy) {
      continue;
    }

    /// Don't profile output nodes.
    if (!PH->getUsers().empty()) {
      auto *SN = llvm::dyn_cast<SaveNode>(PH->getUsers().begin()->getUser());
      if (SN) {
        continue;
      }
    }
    nodesToInstrument.insert(PH->getOutput());
  }

  for (const auto &NV : nodesToInstrument) {
    F->createQuantizationProfile(bindings,
                                 "QP_" + NV.getNode()->getName().str(), NV);
  }
}
