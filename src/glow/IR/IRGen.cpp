// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/Support/Casting.h"

#include <unordered_map>

using namespace glow;

//===----------------------------------------------------------------------===//
//              IRGen visitor - the code that generates the IR.
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
      Instruction *V = nullptr;
      if (P->getMode() == PoolNode::Mode::Max) {
        V = builder_.createPoolMaxOp(in, P->getKernel(), P->getStride(),
                                     P->getPad());
      } else {
        V = builder_.createPoolAvgOp(in, P->getKernel(), P->getStride(),
                                     P->getPad());
      }

      V->setName(N->getName());
      registerIR(N, V->getOperand(0).first);
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

      Instruction *V = nullptr;
      if (AR->getMode() == ArithmeticNode::Mode::Add) {
        V = builder_.createElementAddOp(L, R);
      } else {
        V = builder_.createElementMulOp(L, R);
      }

      V->setName(N->getName());
      registerIR(N, V->getOperand(0).first);
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

//===----------------------------------------------------------------------===//
//        Code for automatically generating the back propagation code.
//===----------------------------------------------------------------------===//

void generateBackwardPass(Module &M) {
  using Kind = glow::Kinded::Kind;
  auto &weightToGradMap = M.getGradientMap();
  auto &vars = M.getWeights();

  // Create a shadow variable for each one of the weights in the module.
  for (auto it = vars.begin(), e = vars.end(); it != e;) {
    WeightVar *I = *it;
    std::string newName = I->getName().str() + "_grad";
    auto *A = new WeightVar(newName, I->getType(),
                            WeightVar::MutabilityKind::Mutable);
    weightToGradMap[I] = A;
    auto curr = it;
    ++it;
    vars.insert(curr, A);
  }

  // A list of instructions to add to the module.
  std::vector<Instruction *> allocs;
  std::vector<Instruction *> toAppend;
  std::vector<Instruction *> deallocs;

  // Generate the gradient instructions for each one of the instructions in
  // the module.
  auto &instrs = M.getInstrs();
  for (auto it = instrs.begin(); it != instrs.end(); it++) {
    Instruction *I = *it;

    switch (I->getKind()) {
    case Kind::AllocActivationInstKind: {
      AllocActivationInst *AC = cast<AllocActivationInst>(I);
      auto *N = new AllocActivationInst(AC->getName(), AC->getType());
      allocs.push_back(N);
      weightToGradMap[I] = N;

      auto *D = new DeallocActivationInst(AC->getName(), N);
      deallocs.push_back(D);
      break;
    }
    case Kind::CopyInstKind: {
      CopyInst *CC = cast<CopyInst>(I);
      auto *N = new CopyInst(CC->getName(), weightToGradMap[CC->getSrc()],
                             weightToGradMap[CC->getDest()]);
      toAppend.push_back(N);
      break;
    }
    case Kind::ConvolutionInstKind: {
      toAppend.push_back(cast<ConvolutionInst>(I)->getGrad(weightToGradMap));
      break;
    }
    case Kind::PoolMaxInstKind: {
      toAppend.push_back(cast<PoolMaxInst>(I)->getGrad(weightToGradMap));
      break;
    }
    case Kind::PoolAvgInstKind: {
      toAppend.push_back(cast<PoolAvgInst>(I)->getGrad(weightToGradMap));
      break;
    }
    case Kind::FullyConnectedInstKind: {
      toAppend.push_back(cast<FullyConnectedInst>(I)->getGrad(weightToGradMap));
      break;
    }
    case Kind::BatchNormalizationInstKind: {
      toAppend.push_back(
          cast<BatchNormalizationInst>(I)->getGrad(weightToGradMap));
      break;
    }
    case Kind::LocalResponseNormalizationInstKind: {
      toAppend.push_back(
          cast<LocalResponseNormalizationInst>(I)->getGrad(weightToGradMap));
      break;
    }
    case Kind::SoftMaxInstKind: {
      toAppend.push_back(cast<SoftMaxInst>(I)->getGrad(weightToGradMap));
      break;
    }
    case Kind::RegressionInstKind: {
      toAppend.push_back(cast<RegressionInst>(I)->getGrad(weightToGradMap));

      break;
    }
    case Kind::ElementAddInstKind: {
      toAppend.push_back(cast<ElementAddInst>(I)->getGrad(weightToGradMap));

      break;
    }
    case Kind::ElementMulInstKind: {
      toAppend.push_back(cast<ElementMulInst>(I)->getGrad(weightToGradMap));

      break;
    }
    case Kind::ReluInstKind: {
      toAppend.push_back(cast<ReluInst>(I)->getGrad(weightToGradMap));

      break;
    }
    case Kind::SigmoidInstKind: {
      toAppend.push_back(cast<SigmoidInst>(I)->getGrad(weightToGradMap));

      break;
    }
    case Kind::TanhInstKind: {
      toAppend.push_back(cast<TanhInst>(I)->getGrad(weightToGradMap));
      break;
    }
    case Kind::ReshapeInstKind: {
      toAppend.push_back(cast<ReshapeInst>(I)->getGrad(weightToGradMap));
      break;
    }
    case Kind::TransposeInstKind: {
      toAppend.push_back(cast<TransposeInst>(I)->getGrad(weightToGradMap));
      break;
    }
    case Kind::ConcatInstKind: {
      toAppend.push_back(cast<ConcatInst>(I)->getGrad(weightToGradMap));
      break;
    }
    default:
      glow_unreachable();
    } // End of switch.
  }   // Eod of the for-each instr loop.

  for (auto &I : allocs) {
    instrs.push_back(I);
  }

  // Add all of the new instructions, in reverse.
  std::reverse(toAppend.begin(), toAppend.end());
  for (auto &I : toAppend) {
    instrs.push_back(I);
  }

  for (auto &I : deallocs) {
    instrs.push_back(I);
  }
}

} // namespace

void Module::generateIR(CompilationMode mode) {
  IRGenVisitor irgen(this);

  for (auto &N : G_->getVars()) {
    N->visit(nullptr, &irgen);
  }

  for (auto &N : G_->getNodes()) {
    N->visit(nullptr, &irgen);
  }

  if (mode == CompilationMode::Train) {
    generateBackwardPass(*this);
  }
}
