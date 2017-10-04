// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IRBuilder.h"
#include "glow/Support/Casting.h"

#include <unordered_map>

using namespace glow;

//===----------------------------------------------------------------------===//
//                        IRGen visitor
//===----------------------------------------------------------------------===//

namespace {

/// A helper class for visiting and generating the dotty file from the graph.
struct IRGenVisitor : NodeVisitor {
  std::unordered_map<Node *, Value *> generatedNode;
  /// The module that we are building.
  Module &M_;
  /// The builder that adds instructions into the module.
  IRBuilder builder_;

public:
  bool shouldVisit(Node *parent, Node *N) override {
    // Don't revisit nodes that we've already processed.
    return !generatedNode.count(N);
  }

  IRGenVisitor(Module &M) : M_(M), builder_(M_) {}

  /// \returns the generated instruction for the node \p N.
  Value *valueForNode(Node *N) {
    auto it = generatedNode.find(N);
    assert(it != generatedNode.end() && "IR was not generated for the node");
    return it->second;
  }
  /// Saves the generated IR in \p v for the node \p N.
  void registerIR(Node *N, Value *v) {
    assert(!generatedNode.count(N) && "Already generated code for this node");
    generatedNode[N] = v;
  }

  void post(Node *parent, Node *N) override {
    switch (N->getKind()) {
    case glow::Kinded::Kind::AllocActivationInstKind:
    case glow::Kinded::Kind::DeallocActivationInstKind:
    case Kinded::Kind::CopyInstKind:
      assert("Invalid Node");
      break;
    case glow::Kinded::Kind::ConvolutionInstKind: {
      auto *C = cast<ConvolutionNode>(N);
      auto *in = valueForNode(C->getInput());
      auto *V = builder_.createConvOp(in, C->getDepth(), C->getKernel(),
                                      C->getStride(), C->getPad());
      registerIR(N, V);
      break;
    }
    case glow::Kinded::Kind::PoolInstKind: {
      auto *P = cast<PoolNode>(N);
      auto *in = valueForNode(P->getInput());
      auto *V = builder_.createPoolOp(in, P->getKind(), P->getKernel(),
                                      P->getStride(), P->getPad());
      registerIR(N, V);
      break;
    }
    case glow::Kinded::Kind::FullyConnectedInstKind: {
      auto *FC = cast<FullyConnectedNode>(N);
      auto *in = valueForNode(FC->getInput());
      auto *V = builder_.createFullyConnectedOp(in, FC->getDepth());
      registerIR(N, V);
      break;
    }
    case glow::Kinded::Kind::ReluInstKind: {
      auto *R = cast<ReluNode>(N);
      registerIR(N, builder_.createRELUOp(valueForNode(R->getInput())));
      break;
    }
    case glow::Kinded::Kind::SigmoidInstKind: {
      auto *S = cast<SigmoidNode>(N);
      registerIR(N, builder_.createSigmoidOp(valueForNode(S->getInput())));
      break;
    }
    case glow::Kinded::Kind::TanhInstKind: {
      auto *T = cast<TanhNode>(N);
      registerIR(N, builder_.createTanhOp(valueForNode(T->getInput())));
      break;
    }
    case glow::Kinded::Kind::SoftMaxInstKind: {
      auto *SM = cast<SoftMaxNode>(N);
      auto *in = valueForNode(SM->getInput());
      auto *select = valueForNode(SM->getSelected());
      auto *V = builder_.createSoftMaxOp(in, select);
      registerIR(N, V);
      break;
    }
    case glow::Kinded::Kind::RegressionInstKind: {
      auto *RR = cast<RegressionNode>(N);
      auto *in = valueForNode(RR->getInput());
      auto *expected = valueForNode(RR->getExpected());
      auto *V = builder_.createSoftMaxOp(in, expected);
      registerIR(N, V);
      break;
    }
    case glow::Kinded::Kind::TransposeInstKind: {
      auto *TT = cast<TransposeNode>(N);
      auto *in = valueForNode(TT->getInput());
      auto *V = builder_.createTransposeOp(in, TT->getShuffle());
      registerIR(N, V);
      break;
    }
    case glow::Kinded::Kind::ReshapeInstKind: {
      auto *RS = cast<ReshapeNode>(N);
      auto *in = valueForNode(RS->getInput());
      auto *V = builder_.createReshapeOp(in, RS->getDims());
      registerIR(N, V);
      break;
    }
    case glow::Kinded::Kind::ConcatInstKind: {
      auto *CC = cast<ConcatNode>(N);
      std::vector<Value *> vals;
      for (auto &in : CC->getInputs()) {
        vals.push_back(valueForNode(in));
      }
      auto *V = builder_.createConcatOp(vals, CC->getDim());
      registerIR(N, V);
      break;
    }
    case glow::Kinded::Kind::BatchNormalizationInstKind: {
      auto *BN = cast<BatchNormalizationNode>(N);
      auto *in = valueForNode(BN->getInput());
      auto *V = builder_.createBatchNormalizationOp(
          in, BN->getChannelIdx(), BN->getEpsilon(), BN->getMomentum());
      registerIR(N, V);
      break;
    }

    case glow::Kinded::Kind::LocalResponseNormalizationInstKind: {
      auto *LR = cast<LocalResponseNormalizationNode>(N);
      auto *in = valueForNode(LR->getInput());
      auto *V = builder_.createLocalResponseNormalizationOp(
          in, LR->gethalfWindowSize(), LR->getAlpha(), LR->getBeta(),
          LR->getK());
      registerIR(N, V);
      break;
    }
    case glow::Kinded::Kind::ArithmeticInstKind: {
      auto *AR = cast<ArithmeticNode>(N);
      auto *L = valueForNode(AR->getLHS());
      auto *R = valueForNode(AR->getRHS());
      auto *V = builder_.createArithmeticOp(L, R, AR->getKind());
      registerIR(N, V);
      break;
    }
    case glow::Kinded::Kind::WeightVarKind: {
      auto *V = cast<Variable>(N);
      auto *W = builder_.createWeightVar(V->getType(), V->getName(),
                                         V->getInitKind(), V->getVal());
      registerIR(N, W);
      break;
    }
    }
  }
};
} // namespace

void Graph::generateIR() {
  IRGenVisitor irgen(M_);

  for (auto &N : vars_) {
    N->visit(nullptr, &irgen);
  }

  for (auto &N : nodes_) {
    N->visit(nullptr, &irgen);
  }
}
