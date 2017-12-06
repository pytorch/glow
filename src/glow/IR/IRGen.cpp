// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"

#include "llvm/Support/Casting.h"

#include <unordered_map>
#include <unordered_set>

using namespace glow;

//===----------------------------------------------------------------------===//
//              IRGen visitor - the code that generates the IR.
//===----------------------------------------------------------------------===//

namespace {

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

/// A helper class for visiting and generating the dotty file from the graph.
struct IRGenVisitor : NodeWalker {
  using NodeValueToDestTy = std::unordered_map<NodeValue, Value *>;
  using NodeToInstrTy = std::unordered_map<Node *, Instruction *>;

  /// A set of visited nodes.
  std::unordered_set<Node *> visited;
  /// Holds the mapping between graph nodes to the destination buffers.
  NodeValueToDestTy generatedNodeDest_;
  /// Holds the mapping between graph nodes and the lowered instructions. This
  /// map is used by instructions that want to inspect the generated
  /// instructions. For example, gradient instructions that look at operands
  /// that do not exist at the graph level. Not all variables are representible.
  NodeToInstrTy nodeToInstr_;

  /// The module that we are building.
  Module *M_;
  /// The builder that adds instructions into the module.
  IRBuilder builder_;

public:
  bool shouldVisit(Node *parent, Node *N) override {
    // Don't revisit nodes that we've already processed.
    return !visited.count(N);
  }

  explicit IRGenVisitor(Module *M) : M_(M), builder_(M_) {}

  /// \returns the generated instruction for the node \p N.
  Value *valueForNode(NodeValue N) {
    if (auto *V = dyn_cast<Variable>(N)) {
      auto &map = M_->getVariableMap();
      return map[V];
    }
    auto it = generatedNodeDest_.find(N);
    assert(it != generatedNodeDest_.end() &&
           "IR was not generated for the node");
    return it->second;
  }
  /// Saves the generated IR in \p v for the node \p N.
  void registerIR(NodeValue N, Value *v) {
    if (auto *V = dyn_cast<Variable>(N)) {
      auto &map = M_->getVariableMap();
      map[V] = v;
      return;
    }
    assert(!generatedNodeDest_.count(N) &&
           "Already generated code for this node");
    assert(isa<AllocActivationInst>(v) && "The value must be an activation");
    generatedNodeDest_[N] = v;
  }

  void post(Node *parent, Node *N) override {
    visited.insert(N);
    switch (N->getKind()) {
    default:
      // Unkniwn node kind.
      glow_unreachable();
      assert(false);
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
    case glow::Kinded::Kind::ConvolutionGradNodeKind: {
      auto *CG = cast<ConvolutionGradNode>(N);

      auto *input = valueForNode(CG->getInput());
      auto *filter = valueForNode(CG->getFilter());
      auto *bias = valueForNode(CG->getBias());

      auto *outGrad = valueForNode(CG->getGradOfOriginalOutputNamedResult());

      auto *inG =
          builder_.createAllocActivationInst("conv.input.G", input->getType());
      auto *biasG =
          builder_.createAllocActivationInst("conv.bias.G", bias->getType());
      auto *filterG = builder_.createAllocActivationInst("conv.filter.G",
                                                         filter->getType());

      builder_.createConvolutionGradInst(
          N->getName(), input, filter, outGrad, inG, filterG, biasG,
          CG->getKernel(), CG->getStride(), CG->getPad(), CG->getDepth());

      registerIR(CG->getGradOfInputNamedInput(), inG);
      registerIR(CG->getGradOfInputNamedFilter(), filterG);
      registerIR(CG->getGradOfInputNamedBias(), biasG);
      break;
    }
    case glow::Kinded::Kind::PoolNodeKind: {
      auto *P = cast<PoolNode>(N);
      auto *in = valueForNode(P->getInput());
      Instruction *V = nullptr;
      if (P->getMode() == PoolNode::Mode::Max) {
        V = builder_.createPoolMaxOp(in, P->getKernel(), P->getStride(),
                                     P->getPad());
        nodeToInstr_[N] = V;
      } else {
        V = builder_.createPoolAvgOp(in, P->getKernel(), P->getStride(),
                                     P->getPad());
      }

      V->setName(N->getName());
      registerIR(N, V->getOperand(0).first);
      break;
    }

    case glow::Kinded::Kind::PoolGradNodeKind: {
      auto *PG = cast<PoolGradNode>(N);

      auto poolOut = PG->getOriginalOutputForResult();
      auto *outW = valueForNode(poolOut);
      auto *outG = valueForNode(PG->getGradOfOriginalOutputNamedResult());

      auto *inG = builder_.createAllocActivationInst("pool.outG",
                                                     PG->getInput()->getType());

      if (PG->getMode() == PoolGradNode::Mode::Max) {
        // Find the original pool instruction.
        assert(nodeToInstr_.count(poolOut) &&
               "Pool IRgen did not register itself");
        auto *PI = cast<PoolMaxInst>(nodeToInstr_[poolOut.getNode()]);

        builder_.createPoolMaxGradInst(N->getName(), outW, PI->getSrcXY(), outG,
                                       inG, PG->getKernel(), PG->getStride(),
                                       PG->getPad());
        registerIR(PG->getGradOfInputNamedInput(), inG);
        break;
      } else {
        builder_.createPoolAvgGradInst(N->getName(), outW, outG, inG,
                                       PG->getKernel(), PG->getStride(),
                                       PG->getPad());
        registerIR(PG->getGradOfInputNamedInput(), inG);
        break;
      }
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

    case glow::Kinded::Kind::FullyConnectedGradNodeKind: {
      auto *FCG = cast<FullyConnectedGradNode>(N);
      auto *inW = valueForNode(FCG->getInput());
      auto *filterW = valueForNode(FCG->getFilter());
      auto *outW = valueForNode(FCG->getGradOfOriginalOutputNamedOutput());
      auto biasX = FCG->getBias();

      auto *InG = builder_.createAllocActivationInst("inG", inW->getType());
      auto *FilterG =
          builder_.createAllocActivationInst("filterG", filterW->getType());
      auto *BiasG =
          builder_.createAllocActivationInst("biasG", biasX.getType());

      builder_.createFullyConnectedGradInst(N->getName(), inW, filterW, outW,
                                            InG, FilterG, BiasG,
                                            FCG->getDepth());

      registerIR(FCG->getGradOfInputNamedInput(), InG);
      registerIR(FCG->getGradOfInputNamedFilter(), FilterG);
      registerIR(FCG->getGradOfInputNamedBias(), BiasG);
      break;
    }

    case glow::Kinded::Kind::ReluNodeKind: {
      auto *R = cast<ReluNode>(N);
      auto *V = builder_.createRELUOp(valueForNode(R->getInput()));
      V->setName(N->getName());
      registerIR(N, V->getDest());
      break;
    }
    case glow::Kinded::Kind::ReluGradNodeKind: {
      auto *RG = cast<ReluGradNode>(N);
      auto *outGrad = valueForNode(RG->getGradOfOriginalOutputNamedResult());
      auto *DG = builder_.createAllocActivationInst("relu.inG.grad",
                                                    outGrad->getType());
      builder_.createReluGradInst(
          N->getName(), valueForNode(RG->getOriginalOutputForResult()), outGrad,
          DG);
      registerIR(N, DG);
      break;
    }
    case glow::Kinded::Kind::SigmoidNodeKind: {
      auto *S = cast<SigmoidNode>(N);
      auto *V = builder_.createSigmoidOp(valueForNode(S->getInput()));
      V->setName(N->getName());
      registerIR(N, V->getDest());
      break;
    }
    case glow::Kinded::Kind::SigmoidGradNodeKind: {
      auto *SG = cast<SigmoidGradNode>(N);
      auto *outGrad = valueForNode(SG->getGradOfOriginalOutputNamedResult());
      auto *DG = builder_.createAllocActivationInst("sigmoid.inG.grad",
                                                    outGrad->getType());
      builder_.createSigmoidGradInst(
          N->getName(), valueForNode(SG->getOriginalOutputForResult()), outGrad,
          DG);
      registerIR(N, DG);
      break;
    }
    case glow::Kinded::Kind::TanhNodeKind: {
      auto *T = cast<TanhNode>(N);
      auto *V = builder_.createTanhOp(valueForNode(T->getInput()));
      V->setName(N->getName());
      registerIR(N, V->getDest());
      break;
    }
    case glow::Kinded::Kind::TanhGradNodeKind: {
      auto *TG = cast<TanhGradNode>(N);
      auto *outGrad = valueForNode(TG->getGradOfOriginalOutputNamedResult());
      auto *DG = builder_.createAllocActivationInst("tanh.inG.grad",
                                                    outGrad->getType());
      builder_.createTanhGradInst(
          N->getName(), valueForNode(TG->getOriginalOutputForResult()), outGrad,
          DG);
      registerIR(N, DG);
      break;
    }
    case glow::Kinded::Kind::SoftMaxNodeKind: {
      auto *SM = cast<SoftMaxNode>(N);
      auto *in = valueForNode(SM->getInput());
      auto *select = valueForNode(SM->getSelected());
      auto *V = builder_.createSoftMaxOp(in, select);
      V->setName(N->getName());
      registerIR(N, V->getDest());
      nodeToInstr_[N] = V;
      break;
    }
    case glow::Kinded::Kind::SoftMaxGradNodeKind: {
      auto *SMG = cast<SoftMaxGradNode>(N);
      // Original inputs:
      auto *origIn = valueForNode(SMG->getInput());
      auto *origSelect = valueForNode(SMG->getSelected());
      // Values related to the output of the node.
      auto *outGrad = valueForNode(SMG->getGradOfOriginalOutputNamedResult());
      auto originalNodeResult = SMG->getOriginalOutputForResult();
      assert(nodeToInstr_.count(originalNodeResult.getNode()) &&
             "Unknown original node");
      auto *SM = cast<SoftMaxInst>(nodeToInstr_[originalNodeResult]);
      auto *srcGrad = builder_.createAllocActivationInst("softmax.res.grad",
                                                         outGrad->getType());

      auto *SMGI = builder_.createSoftMaxGradInst(
          N->getName(), origIn, SM->getE(), origSelect, srcGrad);

      registerIR(SMG->getGradOfInputNamedInput(), SMGI->getSrcGrad());
      break;
    }
    case glow::Kinded::Kind::RegressionNodeKind: {
      auto *RR = cast<RegressionNode>(N);
      auto *in = valueForNode(RR->getInput());
      auto *expected = valueForNode(RR->getExpected());
      auto *V = builder_.createRegressionOp(in, expected);
      nodeToInstr_[N] = V;
      V->setName(N->getName());
      registerIR(N, V->getDest());
      break;
    }
    case glow::Kinded::Kind::RegressionGradNodeKind: {
      auto *RG = cast<RegressionGradNode>(N);
      // Original inputs:
      auto *origIn = valueForNode(RG->getInput());
      auto *origExpected = valueForNode(RG->getExpected());
      // Values related to the output of the node.
      auto *srcGrad = builder_.createAllocActivationInst("regression.res.grad",
                                                         origIn->getType());
      auto *expGrad = builder_.createAllocActivationInst("expected.res.grad",
                                                         origIn->getType());

      builder_.createRegressionGradInst(N->getName(), origIn, origExpected,
                                        srcGrad, expGrad);

      registerIR(RG->getGradOfInputNamedInput(), srcGrad);
      registerIR(RG->getGradOfInputNamedExpected(), expGrad);
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

      auto *dest = builder_.createAllocActivationInst(
          CC->getName(), CC->getElementType(), CC->dims());
      builder_.createZeroInst(CC->getName(), dest);
      auto inputs = CC->getInputs();

      // We start inserting to the shape at (0,0, ... ).
      std::vector<size_t> offsets(CC->dims().size(), 0);
      unsigned dim = CC->getDim();

      for (int i = 0, e = inputs.size(); i < e; i++) {
        builder_.createInsertTensorInst(CC->getName(), dest,
                                        valueForNode(inputs[i]), offsets);
        // We are stacking the tensors along a specific dimension. This means
        // that we increase the size of the tensor along this dimension.
        offsets[dim] += inputs[i].dims()[dim];
      }
      registerIR(N, dest);
      break;
    }
    case glow::Kinded::Kind::SliceNodeKind: {
      auto *SL = cast<SliceNode>(N);
      auto start = SL->getStart();
      auto *in = valueForNode(SL->getInput());
      auto *dest = builder_.createAllocActivationInst(
          SL->getName(), SL->getElementType(), SL->dims());
      builder_.createExtractTensorInst(SL->getName(), dest, in, start);
      registerIR(N, dest);
      break;
    }
    case glow::Kinded::Kind::InsertTensorNodeKind: {
      auto *IT = cast<InsertTensorNode>(N);
      auto start = IT->getStart();
      auto *big = valueForNode(IT->getBig());
      auto *small = valueForNode(IT->getSmall());
      auto *dest =
          builder_.createAllocActivationInst(IT->getName(), IT->getType());
      builder_.createCopyInst("copy.insert", dest, big);
      builder_.createInsertTensorInst("insert", dest, small, start);
      registerIR(N, dest);
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
    case glow::Kinded::Kind::BatchNormalizationGradNodeKind: {
      auto *BN = cast<BatchNormalizationGradNode>(N);
      auto *in = valueForNode(BN->getInput());
      auto *beta = valueForNode(BN->getBias());
      auto *gamma = valueForNode(BN->getScale());
      auto *mean = valueForNode(BN->getMean());
      auto *var = valueForNode(BN->getVar());

      auto *outG = valueForNode(BN->getGradOfOriginalOutputNamedResult());

      auto *inG =
          builder_.createAllocActivationInst("bn.input.G", in->getType());
      auto *scaleG =
          builder_.createAllocActivationInst("bn.scale.G", gamma->getType());
      auto *biasG =
          builder_.createAllocActivationInst("bn.bias.G", beta->getType());

      builder_.createBatchNormalizationGradInst(
          N->getName(), in, gamma, mean, var, outG, inG, scaleG, biasG,
          BN->getChannelIdx(), BN->getEpsilon(), BN->getMomentum());
      registerIR(BN->getGradOfInputNamedInput(), inG);
      registerIR(BN->getGradOfInputNamedBias(), biasG);
      registerIR(BN->getGradOfInputNamedScale(), scaleG);
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
    case glow::Kinded::Kind::ArithmeticGradNodeKind: {
      auto *AR = cast<ArithmeticGradNode>(N);
      auto *L = valueForNode(AR->getLHS());
      auto *R = valueForNode(AR->getRHS());

      auto outG = valueForNode(AR->getOriginalOutputForResult());
      auto *LG = builder_.createAllocActivationInst("LG", L->getType());
      auto *RG = builder_.createAllocActivationInst("RG", L->getType());

      if (AR->getMode() == ArithmeticGradNode::Mode::Add) {
        builder_.createElementAddGradInst(N->getName(), outG, LG, RG);
      } else {
        builder_.createElementMulGradInst(N->getName(), L, R, outG, LG, RG);
      }

      registerIR(AR->getGradOfInputNamedLHS(), LG);
      registerIR(AR->getGradOfInputNamedRHS(), RG);
      break;
    }
    case glow::Kinded::Kind::SaveNodeKind: {
      auto *R = cast<SaveNode>(N);
      auto *src = valueForNode(R->getInput());
      auto *dest = valueForNode(R->getOutput());
      auto *V = builder_.createCopyInst("save", dest, src);
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
    case glow::Kinded::Kind::ZeroNodeKind: {
      auto *Z = cast<ZeroNode>(N);
      auto *AC = builder_.createAllocActivationInst(Z->getName(), Z->getType());
      builder_.createZeroInst(N->getName(), AC);
      registerIR(N, AC);
      break;
    }
    case glow::Kinded::Kind::SGDNodeKind: {
      auto *S = cast<SGDNode>(N);
      assert(S->getGradient().getType() == S->getWeight().getType());
      builder_.createSGDInst(N->getName(), valueForNode(S->getGradient()),
                             valueForNode(S->getWeight()),
                             valueForNode(S->getGsum()), S->getL1Decay(),
                             S->getL2Decay(), S->getLearningRate(),
                             S->getMomentum(), S->getBatchSize());
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
  for (auto I : instrs) {
    switch (I->getKind()) {
    case Kind::AllocActivationInstKind: {
      auto *AC = cast<AllocActivationInst>(I);
      auto *N = new AllocActivationInst(&M, AC->getName(), AC->getType());
      allocs.push_back(N);
      weightToGradMap[I] = N;

      auto *D = new DeallocActivationInst(&M, AC->getName(), N);
      deallocs.push_back(D);
      break;
    }
    case Kind::CopyInstKind: {
      auto *CC = cast<CopyInst>(I);
      auto *N = new CopyInst(&M, CC->getName(), weightToGradMap[CC->getSrc()],
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
      ReshapeInst *RI = cast<ReshapeInst>(I);
      Value *dest = weightToGradMap[RI->getDest()];
      Value *src = weightToGradMap[RI->getSrc()];
      // Swap the src and dest.
      toAppend.push_back(
          new ReshapeInst(&M, I->getName(), src, dest, src->dims()));
      break;
    }
    case Kind::TransposeInstKind: {
      TransposeInst *TI = cast<TransposeInst>(I);
      Value *dest = weightToGradMap[TI->getDest()];
      Value *src = weightToGradMap[TI->getSrc()];

      // Generate the reverse shuffle.
      auto shuffle = TI->getShuffle();
      std::vector<unsigned> reverseShuffle(shuffle.begin(), shuffle.end());
      for (unsigned int i = 0; i < shuffle.size(); i++) {
        reverseShuffle[shuffle[i]] = i;
      }

      // Swap the src and dest.
      toAppend.push_back(
          new TransposeInst(&M, I->getName(), src, dest, reverseShuffle));
      break;
    }
    case Kind::ZeroInstKind: {
      Value *src = weightToGradMap[I->getOperand(0).first];
      toAppend.push_back(new ZeroInst(&M, I->getName(), src));
      break;
    }
    case Kind::InsertTensorInstKind: {
      InsertTensorInst *ITI = cast<InsertTensorInst>(I);
      Value *dest = weightToGradMap[ITI->getDest()];
      Value *src = weightToGradMap[ITI->getSrc()];
      // Swap the src and dest.
      toAppend.push_back(new ExtractTensorInst(&M, I->getName(), src, dest,
                                               ITI->getOffsets()));
      break;
    }
    case Kind::ExtractTensorInstKind: {
      ExtractTensorInst *ETI = cast<ExtractTensorInst>(I);
      Value *dest = weightToGradMap[ETI->getDest()];
      Value *src = weightToGradMap[ETI->getSrc()];

      // Swap the src and dest.
      toAppend.push_back(
          new InsertTensorInst(&M, I->getName(), src, dest, ETI->getOffsets()));
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
