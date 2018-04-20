// Copyright 2017 Facebook Inc.  All Rights Reserved.

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
