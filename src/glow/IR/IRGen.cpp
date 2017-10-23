// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
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
  using NodeToInstrTy = std::unordered_map<const Node *, Value *>;

  /// Holds the mapping between graph nodes to IR variables.
  NodeToInstrTy generatedNodes;
  /// The module that we are building.
  Module *M_;
  /// The builder that adds instructions into the module.
  IRBuilder builder_;

public:
  bool shouldVisit(Node *parent, Node *N) override {
    // Don't revisit nodes that we've already processed.
    return !generatedNodes.count(N);
  }

  explicit IRGenVisitor(Module *M) : M_(M), builder_(M_) {}

  /// \returns the generated instruction for the node \p N.
  Value *valueForNode(Node *N) {
    auto it = generatedNodes.find(N);
    assert(it != generatedNodes.end() && "IR was not generated for the node");
    return it->second;
  }
  /// Saves the generated IR in \p v for the node \p N.
  void registerIR(Node *N, Value *v) {
    assert(!generatedNodes.count(N) && "Already generated code for this node");
    assert((isa<AllocActivationInst>(v) || isa<WeightVar>(v)) &&
           "Value operand must be a memory location");
    generatedNodes[N] = v;
    // Register the fact that we've lowered this variable to the new weight.
    auto &map = M_->getVariableMap();
    map[N] = v;
  }

  void post(Node *parent, Node *N) override {
    switch (N->getKind()) {
    default:
      assert("Invalid Node");
      break;
    case glow::Kinded::Kind::ConvolutionNodeKind: {
      auto *C = cast<ConvolutionNode>(N);
      auto *in = valueForNode(C->getInput());
      auto *filter = valueForNode(C->getFilter());
      auto *bias = valueForNode(C->getBias());
      auto *V =
          builder_.createConvOp(in, filter, bias, C->getDepth(), C->getKernel(),
                                C->getStride(), C->getPad());
      V->setName(N->getName());
      registerIR(N, V->getDest());

      break;
    }
    case glow::Kinded::Kind::PoolNodeKind: {
      auto *P = cast<PoolNode>(N);
      auto *in = valueForNode(P->getInput());
      PoolInst::Mode Md =
          (P->getMode() == PoolNode::Mode::Max ? PoolInst::Mode::Max
                                               : PoolInst::Mode::Avg);
      auto *V = builder_.createPoolOp(in, Md, P->getKernel(), P->getStride(),
                                      P->getPad());
      V->setName(N->getName());
      registerIR(N, V->getDest());
      break;
    }
    case glow::Kinded::Kind::FullyConnectedNodeKind: {
      auto *FC = cast<FullyConnectedNode>(N);
      auto *in = valueForNode(FC->getInput());
      auto *filter = valueForNode(FC->getFilter());
      auto *bias = valueForNode(FC->getBias());
      auto *V =
          builder_.createFullyConnectedOp(in, filter, bias, FC->getDepth());
      V->setName(N->getName());
      registerIR(N, V->getDest());
      break;
    }
    case glow::Kinded::Kind::ReluNodeKind: {
      auto *R = cast<ReluNode>(N);
      auto *V = builder_.createRELUOp(valueForNode(R->getInput()));
      V->setName(N->getName());
      registerIR(N, V->getDest());

      break;
    }
    case glow::Kinded::Kind::SigmoidNodeKind: {
      auto *S = cast<SigmoidNode>(N);
      auto *V = builder_.createSigmoidOp(valueForNode(S->getInput()));
      V->setName(N->getName());
      registerIR(N, V->getDest());
      break;
    }
    case glow::Kinded::Kind::TanhNodeKind: {
      auto *T = cast<TanhNode>(N);
      auto *V = builder_.createTanhOp(valueForNode(T->getInput()));
      V->setName(N->getName());
      registerIR(N, V->getDest());
      break;
    }
    case glow::Kinded::Kind::SoftMaxNodeKind: {
      auto *SM = cast<SoftMaxNode>(N);
      auto *in = valueForNode(SM->getInput());
      auto *select = valueForNode(SM->getSelected());
      auto *V = builder_.createSoftMaxOp(in, select);
      V->setName(N->getName());
      registerIR(N, V->getDest());
      break;
    }
    case glow::Kinded::Kind::RegressionNodeKind: {
      auto *RR = cast<RegressionNode>(N);
      auto *in = valueForNode(RR->getInput());
      auto *expected = valueForNode(RR->getExpected());
      auto *V = builder_.createRegressionOp(in, expected);
      V->setName(N->getName());
      registerIR(N, V->getDest());
      break;
    }
    case glow::Kinded::Kind::TransposeNodeKind: {
      auto *TT = cast<TransposeNode>(N);
      auto *in = valueForNode(TT->getInput());
      auto *V = builder_.createTransposeOp(in, TT->getShuffle());
      V->setName(N->getName());
      registerIR(N, V->getDest());
      break;
    }
    case glow::Kinded::Kind::ReshapeNodeKind: {
      auto *RS = cast<ReshapeNode>(N);
      auto *in = valueForNode(RS->getInput());
      auto *V = builder_.createReshapeOp(in, RS->getDims());
      V->setName(N->getName());
      registerIR(N, V->getDest());
      break;
    }
    case glow::Kinded::Kind::ConcatNodeKind: {
      auto *CC = cast<ConcatNode>(N);
      auto *LHS = valueForNode(CC->getLHS());
      auto *RHS = valueForNode(CC->getRHS());
      auto *V = builder_.createConcatOp(LHS, RHS, CC->getDim());
      V->setName(N->getName());
      registerIR(N, V->getDest());
      break;
    }
    case glow::Kinded::Kind::BatchNormalizationNodeKind: {
      auto *BN = cast<BatchNormalizationNode>(N);
      auto *in = valueForNode(BN->getInput());
      auto *beta = valueForNode(BN->getBias());
      auto *gamma = valueForNode(BN->getScale());
      auto *mean = valueForNode(BN->getMean());
      auto *var = valueForNode(BN->getVar());

      auto *V = builder_.createBatchNormalizationOp(
          in, beta, gamma, mean, var, BN->getChannelIdx(), BN->getEpsilon(),
          BN->getMomentum());
      V->setName(N->getName());
      registerIR(N, V->getDest());
      break;
    }

    case glow::Kinded::Kind::LocalResponseNormalizationNodeKind: {
      auto *LR = cast<LocalResponseNormalizationNode>(N);
      auto *in = valueForNode(LR->getInput());
      auto *V = builder_.createLocalResponseNormalizationOp(
          in, LR->getHalfWindowSize(), LR->getAlpha(), LR->getBeta(),
          LR->getK());
      V->setName(N->getName());
      registerIR(N, V->getDest());
      break;
    }
    case glow::Kinded::Kind::ArithmeticNodeKind: {
      auto *AR = cast<ArithmeticNode>(N);
      auto *L = valueForNode(AR->getLHS());
      auto *R = valueForNode(AR->getRHS());

      ArithmeticInst::Mode Md = (AR->getMode() == ArithmeticNode::Mode::Add
                                     ? ArithmeticInst::Mode::Add
                                     : ArithmeticInst::Mode::Mul);

      auto *V = builder_.createArithmeticOp(L, R, Md);
      V->setName(N->getName());
      registerIR(N, V->getDest());
      break;
    }
    case glow::Kinded::Kind::SaveNodeKind: {
      auto *R = cast<SaveNode>(N);
      auto *src = valueForNode(R->getInput());
      auto *dest = valueForNode(R->getOutput());
      auto *V = builder_.createCopyInst(dest, src);
      V->setName(N->getName());
      break;
    }
    case glow::Kinded::Kind::VariableNodeKind: {
      using MK = WeightVar::MutabilityKind;
      auto *V = cast<Variable>(N);
      bool isConst = V->getInitKind() == Variable::InitKind::Extern;
      auto *W = builder_.createWeightVar(V->getType(), V->getName(),
                                         isConst ? MK::Constant : MK::Mutable);
      W->setName(N->getName());
      registerIR(N, W);
      break;
    }
    }
  }
};
} // namespace

void Module::generateIR() {
  IRGenVisitor irgen(this);

  for (auto &N : G_->getVars()) {
    N->visit(nullptr, &irgen);
  }

  for (auto &N : G_->getNodes()) {
    N->visit(nullptr, &irgen);
  }
}
