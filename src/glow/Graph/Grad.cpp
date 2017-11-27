// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Base/Train.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/Support/Support.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace glow;

using llvm::cast;

void glow::generateGradientNodes(Graph &G, TrainingConfig &conf) {
  using Kind = glow::Kinded::Kind;
  UnownedNodeValueMap map;

  // A list of nodes to add to the graph.
  std::vector<Node *> toAppend;

  // Generate the gradient nodes for each one of the nodes in the module.
  auto &nodes = G.getNodes();
  for (auto it = nodes.rbegin(), e = nodes.rend(); it != e; it++) {
    Node *N = *it;

#define CONVERT_TO_GRAD_NODE(NodeKind)                                         \
  if (N->getKind() == Kind::NodeKind##Kind) {                                  \
    toAppend.push_back(cast<NodeKind>(N)->getGrad(map));                       \
    continue;                                                                  \
  }

    CONVERT_TO_GRAD_NODE(ConvolutionNode)
    CONVERT_TO_GRAD_NODE(PoolNode)
    CONVERT_TO_GRAD_NODE(FullyConnectedNode)
    CONVERT_TO_GRAD_NODE(BatchNormalizationNode)
    CONVERT_TO_GRAD_NODE(LocalResponseNormalizationNode)
    CONVERT_TO_GRAD_NODE(SoftMaxNode)
    CONVERT_TO_GRAD_NODE(RegressionNode)
    CONVERT_TO_GRAD_NODE(ArithmeticNode)
    CONVERT_TO_GRAD_NODE(ReluNode)
    CONVERT_TO_GRAD_NODE(SigmoidNode)
    CONVERT_TO_GRAD_NODE(TanhNode)
    CONVERT_TO_GRAD_NODE(ConvolutionNode)

    if (N->getKind() == Kind::SaveNodeKind) {
      // Swap the src and dest.
      auto *X =
          new ZeroNode(N->getName(), cast<SaveNode>(N)->getInput()->getType());
      toAppend.push_back(X);
      map.insert(cast<SaveNode>(N)->getInput(), X);
      map.insert(cast<SaveNode>(N)->getVariable(), X);
      continue;
    }

    if (N->getKind() == Kind::ReshapeNodeKind) {
      ReshapeNode *RN = cast<ReshapeNode>(N);
      NodeValue outputG = map.get(RN->getResult());
      NodeValue inputW = RN->getInput();

      // Swap the src and dest.
      auto *X = new ReshapeNode(N->getName(), inputW->getType(), outputG,
                                inputW->getType()->dims());
      toAppend.push_back(X);
      map.insert(RN->getResult(), X);
      break;
    }

    if (N->getKind() == Kind::TransposeNodeKind) {
      TransposeNode *TN = cast<TransposeNode>(N);
      NodeValue outputG = map.get(TN->getResult());
      NodeValue inputW = TN->getInput();

      // Generate the reverse shuffle.
      auto shuffle = TN->getShuffle();
      std::vector<unsigned> reverseShuffle(shuffle.begin(), shuffle.end());
      for (unsigned int i = 0; i < shuffle.size(); i++) {
        reverseShuffle[shuffle[i]] = i;
      }

      // Swap the src and dest.
      auto *X = new TransposeNode(N->getName(), inputW->getType(), outputG,
                                  reverseShuffle);
      map.insert(TN->getResult(), X);
      break;
    }

    if (N->getKind() == Kind::ConcatNodeKind) {
      auto *CC = cast<ConcatNode>(N);
      auto inputs = CC->getInputs();

      // We start extracting the shape at (0,0, ... ).
      std::vector<size_t> offsets(CC->dims().size(), 0);
      unsigned dim = CC->getDim();
      for (auto &N : inputs) {
        auto *X = new SliceNode("extract", N.getType(), N, offsets);
        // We are stacking the tensors along a specific dimension. This means
        // that we increase the size of the tensor along this dimension.
        offsets[dim] += N.dims()[dim];
        map.insert(N, X);
      }
      break;
    }

    assert(false);
    glow_unreachable();
  } // End of the for-each instr loop.

  for (auto &V : G.getVars()) {
    // Don't update nodes that are not in training mode.
    if (!V->isTraining()) {
      continue;
    }

    auto X = new SGDNode(V->getName(), map.get(V), V, conf.L1Decay,
                         conf.L2Decay, conf.learningRate, conf.momentum);
    toAppend.push_back(X);
  }

  // Add all of the new instructions.
  for (auto &I : toAppend) {
    nodes.push_back(I);
  }
}
