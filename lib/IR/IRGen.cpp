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
  std::unordered_set<Node *> visited_;
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
    return !visited_.count(N);
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
    visited_.insert(N);
    switch (N->getKind()) {
    default:
      // Unknown node kind.
      llvm_unreachable("Unhandled node kind");
      break;
    case glow::Kinded::Kind::IntrinsicNodeKind: {
      auto *II = cast<IntrinsicNode>(N);

      // A list of destination buffers.
      llvm::SmallVector<Value *, 4> dest;

      // Create the result buffers:
      for (int i = 0, e = II->getNumResults(); i < e; i++) {
        auto *out0 = builder_.createAllocActivationInst(
            "dest", NodeValue(II, i)->getType());
        dest.push_back(out0);
      }

      // Generate the intrinsic instruction.
      auto *instr =
          builder_.createIntrinsicInst(II->getName(), II->getIdentifier());

      // Add the results to the instruction.
      for (int i = 0, e = dest.size(); i < e; i++) {
        instr->pushOperand({dest[i], OperandKind::Out});
        registerIR(NodeValue(II, i), dest[i]);
      }

      // Add the inputs.
      for (int i = 0, e = II->glow::Node::getNumInputs(); i < e; i++) {
        auto *in = valueForNode(II->glow::Node::getNthInput(i));
        instr->pushOperand({in, OperandKind::In});
      }

      break;
    }

    case glow::Kinded::Kind::ConvolutionNodeKind: {
      auto *C = cast<ConvolutionNode>(N);
      auto *in = valueForNode(C->getInput());
      auto *filter = valueForNode(C->getFilter());
      auto *bias = valueForNode(C->getBias());
      Value *dest = builder_.createAllocActivationInst(
          "conv.res", C->getResult()->getType());

      auto *V = builder_.createConvolutionInst("conv", dest, in, filter, bias,
                                               C->getKernel(), C->getStride(),
                                               C->getPad(), C->getDepth());
      V->setName(N->getName());
      registerIR(N, V->getDest());
      break;
    }
    case glow::Kinded::Kind::ConvolutionQNodeKind: {
      auto *CQ = cast<ConvolutionQNode>(N);
      auto *in = valueForNode(CQ->getInput());
      auto *filter = valueForNode(CQ->getFilter());
      auto *bias = valueForNode(CQ->getBias());
      Value *dest = builder_.createAllocActivationInst(
          "convq.res", CQ->getResult()->getType());
      auto *V = builder_.createConvolutionQInst(
          "convq", dest, in, filter, bias, CQ->getKernel(), CQ->getStride(),
          CQ->getPad(), CQ->getDepth(), CQ->getScale(), CQ->getOffset());
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
        V = builder_.createPoolMaxWithXYOp(in, P->getKernel(), P->getStride(),
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
        auto *PI = cast<PoolMaxWithXYInst>(nodeToInstr_[poolOut.getNode()]);

        builder_.createPoolMaxWithXYGradInst(N->getName(), outW, PI->getSrcXY(),
                                             outG, inG, PG->getKernel(),
                                             PG->getStride(), PG->getPad());
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
    case glow::Kinded::Kind::BatchedMatMulNodeKind: {
      auto *BMM = cast<BatchedMatMulNode>(N);
      auto *lhs = valueForNode(BMM->getLHS());
      auto *rhs = valueForNode(BMM->getRHS());
      auto *dest = builder_.createAllocActivationInst(
          "bmm.res", BMM->getResult().getType());
      builder_.createBatchedMatMulInst("bmm", dest, lhs, rhs);
      registerIR(N, dest);
      break;
    }

    case glow::Kinded::Kind::BatchedReduceNodeKind: {
      auto *BR = cast<BatchedReduceNode>(N);
      auto *batch = valueForNode(BR->getBatch());
      auto *dest = builder_.createAllocActivationInst(
          "br.res", BR->getResult().getType());

      switch (BR->getMode()) {
      case BatchedReduceNode::Mode::Add: {
        builder_.createBatchedReduceAddInst(N->getName(), dest, batch);
        break;
      }
      }

      registerIR(N, dest);
      break;
    }

    case glow::Kinded::Kind::BatchedArithmeticNodeKind: {
      auto *BA = cast<BatchedArithmeticNode>(N);
      auto *batch = valueForNode(BA->getBatch());
      auto *sample = valueForNode(BA->getSlice());

      auto *dest = builder_.createAllocActivationInst(
          "br.res", BA->getResult().getType());

      switch (BA->getMode()) {
      case BatchedArithmeticNode::Mode::Add: {
        builder_.createBatchedAddInst(N->getName(), dest, batch, sample);
        break;
      }
      }

      registerIR(N, dest);
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
      auto *V = builder_.createSoftMaxOp(in);
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
      auto *origOut = valueForNode(originalNodeResult);
      auto *srcGrad = builder_.createAllocActivationInst("softmax.res.grad",
                                                         outGrad->getType());
      auto *SMGI = builder_.createSoftMaxGradInst(N->getName(), origOut, origIn,
                                                  origSelect, srcGrad);

      registerIR(SMG->getGradOfInputNamedInput(), SMGI->getSrcGrad());
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
    case glow::Kinded::Kind::BroadcastNodeKind: {
      auto *B = cast<BroadcastNode>(N);
      auto *in = valueForNode(B->getInput());
      auto *V = builder_.createBroadcastOp(in, B->getShape(), B->getAxis());
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
      builder_.createSplatInst(CC->getName(), dest, 0);
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

      auto *meanG =
          builder_.createAllocActivationInst("bn.mean.G", mean->getType());
      auto *varG =
          builder_.createAllocActivationInst("bn.var.G", var->getType());

      builder_.createSplatInst("bn.zero.mean.G", meanG, 0);
      builder_.createSplatInst("bn.zero.var.G", varG, 0);

      builder_.createBatchNormalizationGradInst(
          N->getName(), in, gamma, mean, var, outG, inG, scaleG, biasG,
          BN->getChannelIdx(), BN->getEpsilon(), BN->getMomentum());
      registerIR(BN->getGradOfInputNamedInput(), inG);
      registerIR(BN->getGradOfInputNamedBias(), biasG);
      registerIR(BN->getGradOfInputNamedScale(), scaleG);
      registerIR(BN->getGradOfInputNamedMean(), meanG);
      registerIR(BN->getGradOfInputNamedVar(), varG);
      break;
    }

    case glow::Kinded::Kind::LocalResponseNormalizationNodeKind: {
      auto *LR = cast<LocalResponseNormalizationNode>(N);
      auto *in = valueForNode(LR->getInput());
      auto *V = builder_.createLocalResponseNormalizationOp(
          in, LR->getHalfWindowSize(), LR->getAlpha(), LR->getBeta(),
          LR->getK());
      V->setName(N->getName());
      nodeToInstr_[N] = V;
      registerIR(N, V->getDest());
      break;
    }

    case glow::Kinded::Kind::LocalResponseNormalizationGradNodeKind: {
      auto *LRG = cast<LocalResponseNormalizationGradNode>(N);
      auto *origIn = valueForNode(LRG->getInput());

      auto originalNodeResult = LRG->getOriginalOutputForResult();
      assert(nodeToInstr_.count(originalNodeResult.getNode()) &&
             "Unknown original node");
      auto *LRI = cast<LocalResponseNormalizationInst>(
          nodeToInstr_[originalNodeResult]);

      auto *srcGrad =
          builder_.createAllocActivationInst("lrn.res.grad", origIn->getType());

      builder_.createLocalResponseNormalizationGradInst(
          N->getName(), valueForNode(LRG->getOriginalOutputForResult()),
          valueForNode(LRG->getInput()), LRI->getScale(),
          valueForNode(LRG->getGradOfOriginalOutputNamedResult()), srcGrad,
          LRG->getHalfWindowSize(), LRG->getAlpha(), LRG->getBeta(),
          LRG->getK());

      registerIR(LRG->getGradOfInputNamedInput(), srcGrad);
      break;
    }

    case glow::Kinded::Kind::ArithmeticNodeKind: {
      auto *AR = cast<ArithmeticNode>(N);
      auto *L = valueForNode(AR->getLHS());
      auto *R = valueForNode(AR->getRHS());

      Instruction *instruction = nullptr;
      switch (AR->getMode()) {
      case glow::ArithmeticNode::Mode::Add: {
        instruction = builder_.createElementAddOp(L, R);
        break;
      }
      case glow::ArithmeticNode::Mode::Mul: {
        instruction = builder_.createElementMulOp(L, R);
        break;
      }
      case glow::ArithmeticNode::Mode::Sub: {
        instruction = builder_.createElementSubOp(L, R);
        break;
      }
      case glow::ArithmeticNode::Mode::Div: {
        instruction = builder_.createElementDivOp(L, R);
        break;
      }
      case glow::ArithmeticNode::Mode::Max: {
        instruction = builder_.createElementMaxOp(L, R);
        break;
      }
      case glow::ArithmeticNode::Mode::Min: {
        instruction = builder_.createElementMinOp(L, R);
        break;
      }
      case glow::ArithmeticNode::Mode::CmpLTE: {
        instruction = builder_.createElementCmpLTEOp(L, R);
        break;
      }
      }
      instruction->setName(N->getName());
      registerIR(N, instruction->getOperand(0).first);
      break;
    }
    case glow::Kinded::Kind::SelectNodeKind: {
      auto *S = cast<SelectNode>(N);
      auto *cond = valueForNode(S->getCond());
      auto *lhs = valueForNode(S->getLHS());
      auto *rhs = valueForNode(S->getRHS());
      auto *V = builder_.createSelectOp(cond, lhs, rhs);
      registerIR(S->getResult(), V->getDest());
      V->setName(N->getName());
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
      auto *V = cast<Variable>(N);
      auto *W = builder_.createWeightVar(V->getType(), V->getName(),
                                         WeightVar::MutabilityKind::Mutable);
      W->setName(N->getName());
      registerIR(N, W);
      break;
    }
    case glow::Kinded::Kind::SplatNodeKind: {
      auto *Z = cast<SplatNode>(N);
      auto *AC = builder_.createAllocActivationInst(Z->getName(), Z->getType());
      builder_.createSplatInst(N->getName(), AC, Z->getValue());
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
    case glow::Kinded::Kind::QuantizationProfileNodeKind: {
      auto *QPN = cast<QuantizationProfileNode>(N);
      auto *inputTensor = valueForNode(QPN->getInput());
      auto *histogram = valueForNode(QPN->getHistogramVar());
      auto *computationInfo = valueForNode(QPN->getComputationInfoVar());
      builder_.createQuantizationProfileInst(QPN->getName(), inputTensor,
                                             histogram, computationInfo);
      break;
    }
    case glow::Kinded::Kind::TopKNodeKind: {
      auto *TKN = cast<TopKNode>(N);
      auto *inputTensor = valueForNode(TKN->getInput());
      auto k = TKN->getK();
      auto *V = builder_.createTopKOp(inputTensor, k);
      registerIR(TKN->getValues(), V->getValues());
      registerIR(TKN->getIndices(), V->getIndices());
      V->setName(N->getName());
      break;
    }
    case glow::Kinded::Kind::GatherNodeKind: {
      auto *GN = cast<GatherNode>(N);
      auto *dataTensor = valueForNode(GN->getData());
      auto *indicesTensor = valueForNode(GN->getIndices());
      auto *V = builder_.createGatherOp(dataTensor, indicesTensor);
      registerIR(GN->getResult(), V->getDest());
      V->setName(N->getName());
      break;
    }

    case glow::Kinded::Kind::TanhGradNodeKind:
    case glow::Kinded::Kind::SigmoidGradNodeKind:
    case glow::Kinded::Kind::ArithmeticGradNodeKind:
    case glow::Kinded::Kind::ReluNodeKind:
    case glow::Kinded::Kind::ReluGradNodeKind:
    case glow::Kinded::Kind::FullyConnectedNodeKind:
    case glow::Kinded::Kind::FullyConnectedGradNodeKind:
    case glow::Kinded::Kind::RegressionNodeKind:
    case glow::Kinded::Kind::RegressionGradNodeKind: {
      llvm_unreachable("Node should have been lowered to low-level nodes");
    }
    }
  }
};

} // namespace

void Module::generateIR(CompilationMode mode) {
  G_->advanceState(Graph::State::IRGenerated);
  G_->verify();
  // Schedule the nodes.
  NodesList ScheduledNodes;
  scheduleGraph(ScheduledNodes);
  IRGenVisitor irgen(this);

  for (auto &N : ScheduledNodes) {
    N->visit(nullptr, &irgen);
  }
}
