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
      if (used || isa<SaveNode>(*it)) {
        it++;
        continue;
      }

      delete *it;
      it = nodes.erase(it);
      changedLocally = true;
    }

  } while (changedLocally);
}

/// \returns true if the masks \p shuffle1 and shuffle2 are
/// the inverse of on another. Applying both masks should result in the identity
/// shuffle.
static bool isIdentityShuffle(llvm::ArrayRef<unsigned> shuffle1,
                              llvm::ArrayRef<unsigned> shuffle2) {

  if (shuffle1.size() != shuffle2.size()) {
    return false;
  }

  // Check if the combined masks are the identity mask.
  for (unsigned i = 0, e = shuffle1.size(); i < e; i++) {
    unsigned idx = shuffle2[shuffle1[i]];
    if (idx != i) {
      return false;
    }
  }
  return true;
}

/// Dead code elimination.
static void SinkTranspose(Graph &G) {
  auto &nodes = G.getNodes();

  // For each node:
  for (auto it = nodes.begin(), e = nodes.end(); it != e; ++it) {
    // Sink Transpose below batch normalization nodes:
    if (auto *BN = dyn_cast<BatchNormalizationNode>(*it)) {
      auto *TR = dyn_cast<TransposeNode>(BN->getInput());

      if (!TR) {
        continue;
      }

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
      continue;
    }

    // Sink Transpose below batch RELU nodes.
    // TODO: support other similar activation functions, such as sigmoid, etc.
    if (auto *RL = dyn_cast<ReluNode>(*it)) {
      auto *TR = dyn_cast<TransposeNode>(RL->getInput());

      if (!TR) {
        continue;
      }

      auto *NRL = G.createRELU(RL->getName(), TR->getInput());
      auto *newTR = G.createTranspose(TR->getName(), NRL, TR->getShuffle());
      RL->replaceAllUsesOfWith(newTR);
      continue;
    }

    // Merge consecutive Transpose operations.
    if (auto *TR1 = dyn_cast<TransposeNode>(*it)) {
      auto *TR2 = dyn_cast<TransposeNode>(TR1->getInput());

      if (!TR2) {
        continue;
      }

      auto mask1 = TR1->getShuffle();
      auto mask2 = TR2->getShuffle();
      assert(mask1.size() == mask2.size() && "Invalid mask size");

      // The two transposes are reversing one another. We can skip both of them
      // alltogether.
      if (isIdentityShuffle(mask1, mask2)) {
        TR1->replaceAllUsesOfWith(TR2->getInput());
        continue;
      }
    }

    // Sink Transpose below batch Arithmetic nodes.
    if (auto *AN = dyn_cast<ArithmeticNode>(*it)) {
      auto *LTR = dyn_cast<TransposeNode>(AN->getLHS());
      auto *RTR = dyn_cast<TransposeNode>(AN->getRHS());

      if (!LTR || !RTR) {
        continue;
      }
      // The masks of the transposes on both sizes must match.
      if (LTR->getShuffle() != RTR->getShuffle()) {
        continue;
      }

      auto *newAN = G.createArithmetic(AN->getName(), LTR->getInput(),
                                       RTR->getInput(), AN->getKind());
      auto *newTR = G.createTranspose(LTR->getName(), newAN, LTR->getShuffle());
      AN->replaceAllUsesOfWith(newTR);
    }

    // Sink Transpose below batch Arithmetic nodes.
    if (auto *CN = dyn_cast<ConcatNode>(*it)) {
      assert(CN->getInputs().size() > 1 && "Invalid number of concat operands");

      // Collect all of the transpose nodes and their inputs.
      std::vector<Node *> inputs;
      std::vector<TransposeNode *> transposes;
      for (auto &in : CN->getInputs()) {

        if (auto *II = dyn_cast<TransposeNode>(in.get())) {
          transposes.push_back(II);
          inputs.push_back(II->getInput());
          continue;
        }

        break;
      }

      // If some of the inputs were not transposes then bail out.
      if (CN->getInputs().size() != transposes.size()) {
        continue;
      }

      auto *first = transposes[0];
      auto firstMask = first->getShuffle();
      bool sameMask = true;
      for (auto *T : transposes) {
        if (T->getShuffle() != firstMask) {
          sameMask = false;
          break;
        }
      }

      // If the shuffle masks don't agree then bail out.
      if (!sameMask) {
        continue;
      }

      // Figure out where we transposed the channel index for batch
      // normalization.
      unsigned idx = CN->getDim();
      unsigned newChannelIdx = firstMask[idx];

      auto *newCN = G.createConcat(CN->getName(), inputs, newChannelIdx);
      auto *newTR = G.createTranspose(first->getName(), newCN, firstMask);
      CN->replaceAllUsesOfWith(newTR);
    }

  } // For all nodes in the graph.
}

/// Dead code elimination.
static void OptimizePool(Graph &G) {
  auto &nodes = G.getNodes();

  // For each node:
  for (auto it = nodes.begin(), e = nodes.end(); it != e; ++it) {
    // Swap the order of Relu->MaxPool, to perform the RELU operation on a
    // smaller tensor. This optimization is not a major performance win. The
    // RELU operation takes a small fraction of the time, and reordering the
    // nodes does not give us much. However, reordering the buffers allows us to
    // reuse the memory buffer of the pool operation and potentially save
    // memory.
    if (auto *PL = dyn_cast<PoolNode>(*it)) {
      auto *RL = dyn_cast<ReluNode>(PL->getInput());

      if (!RL) {
        continue;
      }

      // This optimization is only valid on max pooling.
      if (PL->getKind() != PoolNode::OpKind::Max) {
        continue;
      }

      // We don't want to increase the number of operations in the program, so
      // perform this transformation if the relu has a single user, which is the
      // pooling operation.
      if (!RL->hasOneUse()) {
        continue;
      }

      auto *NPL = G.createPool(PL->getName(), RL->getInput(), PL->getKind(),
                               PL->getKernel(), PL->getStride(), PL->getPad());
      auto *NRL = G.createRELU(RL->getName(), NPL);
      PL->replaceAllUsesOfWith(NRL);
      continue;
    }
  } // For all nodes in the graph.
}

void glow::optimize(Graph &G, OptimizationMode mode) {
  if (mode == OptimizationMode::None) {
    return;
  }

  // Sink transpose operations in an attempt to cancel them out.
  SinkTranspose(G);

  // Optimize the pooling operation.
  OptimizePool(G);

  // Perform Dead Code Elimination.
  DCE(G);
}
