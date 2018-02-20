// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Grad.h"
#include "glow/Base/Train.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/Utils.h"
#include "glow/IR/IR.h"
#include "glow/Support/Support.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace glow;

using llvm::cast;
using llvm::isa;

void GraphGradMapper::addGradient(NodeValue activation, NodeValue grad) {
  if (map_.count(activation)) {
    auto curr = map_.get(activation);
    auto *sum = F_->createArithmetic("updateGrad", curr, grad,
                                     ArithmeticNode::Mode::Add);
    map_.insert(activation, sum);
    return;
  }

  map_.insert(activation, grad);
}

bool GraphGradMapper::hasGradient(NodeValue activation) {
  return map_.count(activation);
}

NodeValue GraphGradMapper::getGradient(NodeValue activation) {
  return map_.get(activation);
}

//===----------------------------------------------------------------------===//
//        Code for automatically generating the back propagation code.
//===----------------------------------------------------------------------===//

Function *glow::differentiate(Function *F, TrainingConfig &conf,
                              llvm::StringRef newFuncName,
                              bool onlyRecordGrads) {
  // Create a new name for the differentiated function, if none is given.
  std::string tmpName;
  if (newFuncName.empty()) {
    tmpName = std::string(F->getName()) + "_grad";
    newFuncName = tmpName;
  }

  // Clone the function.
  Function *G = F->clone(newFuncName);

  using Kind = glow::Kinded::Kind;
  GraphGradMapper map(G);

  // A list of nodes to add to the graph.
  std::vector<Node *> toAppend;
  // A list of vars to add to the graph.
  std::vector<Variable *> newVars;

  // Generate the gradient nodes for each one of the nodes in the function.

  PostOrderVisitor pov;
  for (auto &N : G->getParent()->getVars()) {
    N->visit(nullptr, &pov);
  }
  for (auto &N : G->getNodes()) {
    N->visit(nullptr, &pov);
  }

  auto nodes = pov.getPostOrder();

  for (auto it = nodes.rbegin(), e = nodes.rend(); it != e; it++) {
    Node *N = *it;
    if (isa<Variable>(N)) {
      continue;
    }

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
    CONVERT_TO_GRAD_NODE(CrossEntropyLossNode)
    CONVERT_TO_GRAD_NODE(RegressionNode)
    CONVERT_TO_GRAD_NODE(ArithmeticNode)
    CONVERT_TO_GRAD_NODE(ReluNode)
    CONVERT_TO_GRAD_NODE(SigmoidNode)
    CONVERT_TO_GRAD_NODE(TanhNode)

    if (N->getKind() == Kind::SaveNodeKind) {
      // Swap the src and dest. Send the Zero value as gradient for both sides.
      auto *X = new SplatNode(N->getName(),
                              cast<SaveNode>(N)->getInput()->getType(), 0);
      toAppend.push_back(X);
      map.addGradient(cast<SaveNode>(N)->getInput(), X);
      map.addGradient(cast<SaveNode>(N)->getVariable(), X);
      continue;
    }

    if (N->getKind() == Kind::ReshapeNodeKind) {
      ReshapeNode *RN = cast<ReshapeNode>(N);
      NodeValue outputG = map.getGradient(RN->getResult());
      NodeValue inputW = RN->getInput();

      // Swap the src and dest.
      auto *X = new ReshapeNode(N->getName(), inputW->getType(), outputG,
                                inputW->getType()->dims());
      toAppend.push_back(X);
      map.addGradient(RN->getInput(), X);
      continue;
    }

    if (N->getKind() == Kind::TransposeNodeKind) {
      TransposeNode *TN = cast<TransposeNode>(N);
      NodeValue outputG = map.getGradient(TN->getResult());
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
      toAppend.push_back(X);
      map.addGradient(TN->getInput(), X);
      continue;
    }

    if (N->getKind() == Kind::SliceNodeKind) {
      SliceNode *SN = cast<SliceNode>(N);
      auto *zero = new SplatNode("expand", SN->getInput()->getType(), 0);
      auto *insert = new InsertTensorNode("insert.slice.grad", zero,
                                          map.getGradient(SN->getResult()),
                                          SN->getStart());

      toAppend.push_back(zero);
      toAppend.push_back(insert);
      map.addGradient(SN->getInput(), insert);
      continue;
    }

    if (N->getKind() == Kind::ConcatNodeKind) {
      auto *CC = cast<ConcatNode>(N);
      auto inputs = CC->getInputs();
      NodeValue outputG = map.getGradient(CC->getResult());

      // We start extracting the shape at (0,0, ... ).
      std::vector<size_t> offsets(CC->dims().size(), 0);
      unsigned dim = CC->getDim();
      for (auto &N : inputs) {
        auto *X = new SliceNode("extract", N.getType(), outputG, offsets);
        toAppend.push_back(X);
        // We are stacking the tensors along a specific dimension. This means
        // that we increase the size of the tensor along this dimension.
        offsets[dim] += N.dims()[dim];
        map.addGradient(N, X);
      }
      continue;
    }

    if (N->getKind() == Kind::SplatNodeKind)
      // Constant nodes don't have inputs therefore don't need grad
      // calculations.
      continue;

    llvm_unreachable("Invalid instruction type.");
  } // End of the for-each instr loop.

  for (auto &V : G->getParent()->getVars()) {
    // In this special compilation mode we record the last gradient value
    // without performing the SGD update. This mode is used by the unit tests.
    if (onlyRecordGrads && map.hasGradient(V)) {
      std::string nodeName = "_grad_" + V->getName().str();
      // Save the gradient and return the destination variable.
      auto *saveNode = G->createSave(nodeName, map.getGradient(V));
      auto *GradV = llvm::dyn_cast<Variable>(saveNode->getOutput().getNode());
      G->getParent()->addGradientVariable(V, GradV);
    }

    // Don't update nodes that are not marked as trainable, or if no nodes are
    // being trained.
    if (!V->isTraining() || onlyRecordGrads) {
      continue;
    }

    TypeRef Ty = conf.momentum > 0 ? V->getType() : G->getParent()->getVoidTy();
    Variable *gsum = new Variable("gsum", Ty, Variable::VisibilityKind::Private,
                                  Variable::TrainKind::Broadcast, 0);

    newVars.push_back(gsum);

    auto X = new SGDNode(V->getName(), map.getGradient(V), V, gsum,
                         conf.L1Decay, conf.L2Decay, conf.learningRate,
                         conf.momentum, conf.batchSize);
    toAppend.push_back(X);
  }

  // Add all of the new variables and instructions.
  for (auto &I : toAppend) {
    G->addNode(I);
  }
  for (auto &I : newVars) {
    G->getParent()->addVar(I);
  }

  return G;
}
