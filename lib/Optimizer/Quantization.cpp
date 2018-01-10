// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"

using namespace glow;
using llvm::dyn_cast;

/// Profile quantization node should only be added when Graph is compiled in
/// Inference mode.
void glow::profileQuantization(Graph &G) {
  // Iterate over all nodes in the graph and insert QuantizationProfile nodes
  // to observe tensor values from every node's output.
  // Note, new nodes are inserted into the same list which is iterated.
  for (const auto &node : G.getNodes()) {
    // Skip QuantizationProfileNode nodes.
    if (auto *quantizationNode = dyn_cast<QuantizationProfileNode>(node)) {
      continue;
    }

    // Link quantization profile node to each output of the current node.
    for (unsigned i = 0; i < node->getNumRes(); ++i) {
      G.createQuantizationProfile("QuantizationProfile", NodeValue(node, i));
    }
  }
}