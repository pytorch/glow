// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"

#include <unordered_set>

using namespace glow;

/// Instrument graph \p G with Quantization Profile nodes.
/// Nodes should only be added when Graph is compiled in
/// the Inference mode.
void glow::profileQuantization(Graph &G) {
  // Iterate over all nodes in the graph and insert QuantizationProfile nodes
  // to observe tensor values from every node's output.
  std::unordered_set<NodeValue> nodesToInstrument;

  for (const auto &node : G.getNodes()) {
    // Add Quantization Profile node to parent's output linked to the
    // i-th input of the current node.
    for (unsigned i = 0; i < node->getNumInputs(); ++i) {
      nodesToInstrument.insert(node->getInputNode(i));
    }
  }

  for (const auto &node : nodesToInstrument) {
    G.createQuantizationProfile("QuantizationProfile", node);
  }
}
