/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

/// Helper function that \returns the number of times the same consecutive
/// NodeValue in \p inputs is found, starting from index \p i.
static size_t getConsecutiveSameNodeCount(llvm::ArrayRef<NodeValue> inputs,
                                          const size_t i) {
  assert(i < inputs.size() && "Index must fit inside the size of the inputs.");
  for (size_t j = i, e = inputs.size(); j < e; j++) {
    if (inputs[i] != inputs[j]) {
      return j - i;
    }
  }
  return inputs.size() - i;
}

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

  /// The function that we are building.
  IRFunction *F_;
  /// The builder that adds instructions into the function.
  IRBuilder builder_;

public:
  bool shouldVisit(Node *parent, Node *N) override {
    // Don't revisit nodes that we've already processed.
    return !visited_.count(N);
  }

  explicit IRGenVisitor(IRFunction *M) : F_(M), builder_(F_) {}

  /// \returns the generated instruction for the node \p N.
  Value *valueForNode(NodeValue N) {
    if (auto *V = dyn_cast<Variable>(N)) {
      auto &map = F_->getVariableMap();
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
      auto &map = F_->getVariableMap();
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
      llvm_unreachable("Unhandled node; perhaps the node should have been "
                       "lowered, or the backend should have specified an IRGen "
                       "case for this node to a backend-specific Instr.");
      break;

      // Include all automatically generated cases:
#include "AutoGenIRGen.h"

    case glow::Kinded::Kind::ReshapeNodeKind: {
      auto *RN = cast<ReshapeNode>(N);

      auto *inVal = valueForNode(RN->getInput());
      std::vector<size_t> offsets(inVal->getType()->dims().size(), 0);
      auto *TVI = builder_.createTensorViewInst(
          "tensorview.reshape", inVal, RN->getResult().getType(), offsets);
      auto *dest = builder_.createAllocActivationInst(
          "copy.reshape.res", RN->getResult().getType());
      builder_.createCopyInst("copy.reshape", dest, TVI);
      registerIR(N, dest);
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
          CG->getKernel(), CG->getStride(), CG->getPads(), CG->getGroup());

      registerIR(CG->getGradOfInputNamedInput(), inG);
      registerIR(CG->getGradOfInputNamedFilter(), filterG);
      registerIR(CG->getGradOfInputNamedBias(), biasG);
      break;
    }
    case glow::Kinded::Kind::PoolMaxNodeKind: {
      auto *P = cast<PoolMaxNode>(N);
      auto *in = valueForNode(P->getInput());
      auto *V = builder_.createPoolMaxWithXYOp(in, P->getKernel(),
                                               P->getStride(), P->getPad());
      Value *dest = V->getDest();
      nodeToInstr_[N] = V;
      V->setName(N->getName());
      registerIR(N, dest);
      break;
    }
    case glow::Kinded::Kind::PoolMaxGradNodeKind: {
      auto *PG = cast<PoolMaxGradNode>(N);

      auto poolOut = PG->getOriginalOutputForResult();
      auto *outW = valueForNode(poolOut);
      auto *outG = valueForNode(PG->getGradOfOriginalOutputNamedResult());

      auto *inG = builder_.createAllocActivationInst("pool.outG",
                                                     PG->getInput().getType());

      // Find the original pool instruction.
      assert(nodeToInstr_.count(poolOut) &&
             "Pool IRgen did not register itself");
      auto *PI = cast<PoolMaxWithXYInst>(nodeToInstr_[poolOut.getNode()]);

      builder_.createPoolMaxWithXYGradInst(N->getName(), outW, PI->getSrcXY(),
                                           outG, inG, PG->getKernel(),
                                           PG->getStride(), PG->getPad());
      registerIR(PG->getGradOfInputNamedInput(), inG);
      break;
    }
    case glow::Kinded::Kind::PoolAvgGradNodeKind: {
      auto *PG = cast<PoolAvgGradNode>(N);

      auto poolOut = PG->getOriginalOutputForResult();
      auto *outW = valueForNode(poolOut);
      auto *outG = valueForNode(PG->getGradOfOriginalOutputNamedResult());

      auto *inG = builder_.createAllocActivationInst("pool.outG",
                                                     PG->getInput().getType());

      builder_.createPoolAvgGradInst(N->getName(), outW, outG, inG,
                                     PG->getKernel(), PG->getStride(),
                                     PG->getPad());
      registerIR(PG->getGradOfInputNamedInput(), inG);
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
    case glow::Kinded::Kind::CrossEntropyLossNodeKind: {
      auto *CELoss = cast<CrossEntropyLossNode>(N);
      auto *P = valueForNode(CELoss->getP());
      auto *Labels = valueForNode(CELoss->getLabels());
      auto *V = builder_.createCrossEntropyLossOp(P, Labels);
      V->setName(N->getName());
      registerIR(N, V->getCE());
      nodeToInstr_[N] = V;
      break;
    }
    case glow::Kinded::Kind::CrossEntropyLossGradNodeKind: {
      auto *CELossG = cast<CrossEntropyLossGradNode>(N);
      // Forward pass inputs.
      auto *P = valueForNode(CELossG->getP());
      auto *Y = valueForNode(CELossG->getLabels());
      // Backward pass gradient dL/dY.
      auto *dY = valueForNode(CELossG->getGradOfOriginalOutputNamedCE());
      auto *pGrad =
          builder_.createAllocActivationInst("celoss.p.grad", P->getType());
      auto *yGrad = builder_.createAllocActivationInst("celoss.labels.grad",
                                                       Y->getType());
      auto *CELossGI = builder_.createCrossEntropyLossGradInst(
          N->getName(), dY, P, Y, pGrad, yGrad);
      registerIR(CELossG->getGradOfInputNamedP(), CELossGI->getPgrad());
      registerIR(CELossG->getGradOfInputNamedLabels(),
                 CELossGI->getLabelsgrad());
      break;
    }
    case glow::Kinded::Kind::ConcatNodeKind: {
      auto *CC = cast<ConcatNode>(N);

      auto *dest = builder_.createAllocActivationInst(
          CC->getName(), CC->getResult().getType());
      builder_.createSplatInst(CC->getName(), dest, 0);
      auto inputs = CC->getInputs();

      // We start inserting to the shape at (0,0, ... ).
      std::vector<size_t> offsets(CC->getResult().dims().size(), 0);
      unsigned dim = CC->getDim();

      for (size_t i = 0, e = inputs.size(); i < e;) {
        // Look for a series of the same Node being concated consecutively many
        // times. We can wrap n such consecutive repeats into a single insert
        // with count n along the dim axis.
        const size_t consecutiveCount = getConsecutiveSameNodeCount(inputs, i);

        // Create the new InsertTensor instruction given the input node, along
        // with the number of times to insert the node and the axis (dim) we are
        // inserting in.
        builder_.createInsertTensorInst(CC->getName(), dest,
                                        valueForNode(inputs[i]), offsets,
                                        consecutiveCount, dim);

        // We are stacking the tensors along a specific dimension. This means
        // that we increase the size of the tensor along this dimension, count
        // times.
        offsets[dim] += inputs[i].dims()[dim] * consecutiveCount;

        // Increment i by the number of the same nodes that were found in a row,
        // which were all wrapped into a single InsertTensorInst.
        i += consecutiveCount;
      }
      registerIR(N, dest);
      break;
    }
    case glow::Kinded::Kind::SliceNodeKind: {
      auto *SL = cast<SliceNode>(N);
      auto start = SL->getStart();
      auto *in = valueForNode(SL->getInput());
      auto *dest = builder_.createAllocActivationInst(
          SL->getName(), SL->getResult().getType());
      builder_.createExtractTensorInst(SL->getName(), dest, in, start);
      registerIR(N, dest);
      break;
    }
    case glow::Kinded::Kind::InsertTensorNodeKind: {
      auto *IT = cast<InsertTensorNode>(N);
      auto start = IT->getStart();
      auto *big = valueForNode(IT->getBig());
      auto *small = valueForNode(IT->getSmall());
      auto *dest = builder_.createAllocActivationInst(
          IT->getName(), IT->getResult().getType());
      builder_.createCopyInst("copy.insert", dest, big);
      builder_.createInsertTensorInst("insert", dest, small, start,
                                      /* count */ 1, /* axis */ 0);

      registerIR(N, dest);
      break;
    }
    case glow::Kinded::Kind::ScatterAssignNodeKind: {
      auto *SAI = cast<ScatterAssignNode>(N);
      auto *dataTensor = valueForNode(SAI->getData());
      auto *indicesTensor = valueForNode(SAI->getIndices());
      auto *slicesTensor = valueForNode(SAI->getSlices());
      auto *dest = builder_.createAllocActivationInst(
          SAI->getName(), SAI->getResult().getType());
      builder_.createCopyInst("copy.scatterassign", dest, dataTensor);
      builder_.createScatterAssignInst("scatterassign", dest, indicesTensor,
                                       slicesTensor);
      registerIR(N, dest);
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
    case glow::Kinded::Kind::SaveNodeKind: {
      auto *R = cast<SaveNode>(N);
      auto *src = valueForNode(R->getInput());
      auto *dest = valueForNode(R->getOutput());
      auto *V = builder_.createCopyInst("save", dest, src);
      V->setName(N->getName());
      break;
    }
    case glow::Kinded::Kind::LoadNodeKind: {
      auto *load = cast<LoadNode>(N);
      auto *src = valueForNode(load->getVariable());
      auto *dest = builder_.createAllocActivationInst(
          load->getName(), load->getResult().getType());
      auto *copy = builder_.createCopyInst("load", dest, src);
      copy->setName(N->getName());
      registerIR(N, dest);
      break;
    }
    case glow::Kinded::Kind::VariableNodeKind: {
      auto *V = cast<Variable>(N);
      auto *W = builder_.createWeightVar(V->getType(), V->getName(),
                                         WeightVar::MutabilityKind::Mutable,
                                         V->getVisibilityKind());
      W->setName(N->getName());
      registerIR(N, W);
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
    }
  }
};

} // namespace

void IRFunction::generateIR() {
  G_->verify();
  // Schedule the nodes.
  NodesPtrList ScheduledNodes;
  scheduleGraph(ScheduledNodes);
  IRGenVisitor irgen(this);

  for (auto &N : ScheduledNodes) {
    N->visit(nullptr, &irgen);
  }
}
