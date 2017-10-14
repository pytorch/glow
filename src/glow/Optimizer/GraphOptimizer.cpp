// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Optimizer/Optimizer.h"
#include "glow/Support/Casting.h"

#include <unordered_map>
#include <unordered_set>

using namespace glow;

/// Dead code elimination.
static void DCE(Graph &G) {
  auto &nodes = G.getNodes();

  // Remove unused nodes. Do not remove unused vars because they are the
  // interface to the user program.
  bool changedLocally = true;
  do {
    changedLocally = false;
    for (auto it = nodes.begin(), e = nodes.end(); it != e;) {
      bool used = (*it)->hasUsers();
      if (used || isa<ReturnNode>(*it)) {
        it++;
        continue;
      }

      delete *it;
      it = nodes.erase(it);
      changedLocally = true;
    }

  } while (changedLocally);
}

/// Dead code elimination.
static void SinkTranspose(Graph &G) {
  auto &nodes = G.getNodes();

  // For each node:
  for (auto it = nodes.begin(), e = nodes.end(); it != e; ++it) {
    // Sink Transpose below batch normalization nodes:
    if (auto *BN = dyn_cast<BatchNormalizationNode>(*it)) {
      auto *TR = dyn_cast<TransposeNode>(BN->getInput());
      if (!TR)
        continue;

      // Figure out where we transposed the channel index for batch
      // normalization.
      unsigned idx = BN->getChannelIdx();
      unsigned newChannelIdx = TR->getShuffle()[idx];

      auto *NewBN = G.createBatchNormalization(
          BN->getName(), TR->getInput(), BN->getBias(), BN->getScale(),
          BN->getMean(), BN->getVar(), newChannelIdx, BN->getEpsilon(),
          BN->getMomentum());
      auto *newTR = G.createTranspose(TR->getName(), NewBN, TR->getShuffle());

      BN->replaceAllUsesOfWith(newTR);
    }

    // Sink Transpose below batch RELU nodes.
    // TODO: support other similar activation functions, such as sigmoid, etc.
    if (auto *RL = dyn_cast<ReluNode>(*it)) {
      auto *TR = dyn_cast<TransposeNode>(RL->getInput());
      if (!TR)
        continue;

      auto *NRL = G.createRELU(RL->getName(), TR->getInput());
      auto *newTR = G.createTranspose(TR->getName(), NRL, TR->getShuffle());
      RL->replaceAllUsesOfWith(newTR);
    }
  }
}

void glow::optimize(Graph &G, OptimizationMode mode) {
  if (mode == OptimizationMode::None) {
    return;
  }

  // Sink transpose operations in an attempt to cancel them out.
  SinkTranspose(G);

  // Perform Dead Code Elimination.
  DCE(G);
}
