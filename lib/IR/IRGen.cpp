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

#include "glow/IR/IRGen.h"
#include "llvm/Support/Casting.h"

#include <unordered_map>
#include <unordered_set>

//===----------------------------------------------------------------------===//
//              IRGen visitor - the code that generates the IR.
//===----------------------------------------------------------------------===//

using namespace glow;

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

/// Helper function that \returns the number of times the same consecutive
/// NodeValue in \p inputs is found, starting from index \p i.
static size_t getConsecutiveSameNodeCount(NodeValueArrayRef inputs,
                                          const size_t i) {
  assert(i < inputs.size() && "Index must fit inside the size of the inputs.");
  for (size_t j = i, e = inputs.size(); j < e; j++) {
    if (inputs[i] != inputs[j]) {
      return j - i;
    }
  }
  return inputs.size() - i;
}

bool IRGenVisitor::shouldVisit(Node *parent, Node *N) {
  // Don't revisit nodes that we've already processed.
  return !visited_.count(N);
}

/// \returns the generated instruction for the node \p N.
Value *IRGenVisitor::valueForNode(NodeValue N) {
  if (auto *V = dyn_cast<Storage>(N)) {
    auto &map = F_->getVariableMap();
    return map[V];
  }
  auto it = generatedNodeDest_.find(N);
  assert(it != generatedNodeDest_.end() && "IR was not generated for the node");
  return it->second;
}
/// Saves the generated IR in \p v for the node \p N.
void IRGenVisitor::registerIR(NodeValue N, Value *v) {
  if (auto *V = dyn_cast<Storage>(N)) {
    auto &map = F_->getVariableMap();
    map[V] = v;
    return;
  }
  assert(!generatedNodeDest_.count(N) &&
         "Already generated code for this node");
  assert(isa<AllocActivationInst>(v) && "The value must be an activation");
  generatedNodeDest_[N] = v;
}

/// Adds to Node \p N --> Instruction \p inst map.
void IRGenVisitor::setNodeToIR(Node *N, Instruction *inst) {
  nodeToInstr_[N] = inst;
}

/// Return Instruction that is mapped to Node \p N.
/// If mapping doesn't exists returns nullptr.
Instruction *IRGenVisitor::getNodeToIR(Node *N) {
  Instruction *retNode = nullptr;
  auto iterInst = nodeToInstr_.find(N);
  if (iterInst != nodeToInstr_.end())
    retNode = iterInst->second;

  return retNode;
}

void IRGenVisitor::post(Node *parent, Node *N) {
  visited_.insert(N);

  // Allows backend to generate their custom instrution IR.
  if (B_.generateInst(N, *this)) {
    return;
  }

  switch (N->getKind()) {
  default:
    llvm_unreachable("Unhandled node; perhaps the node should have been "
                     "lowered, or the backend should have specified an IRGen "
                     "case for this node to a backend-specific Instr.");
    break;

    // Include all automatically generated cases:
#include "glow/AutoGenIRGen.h"

  case glow::Kinded::Kind::ReshapeNodeKind: {
    auto *RN = cast<ReshapeNode>(N);

    auto *inVal = valueForNode(RN->getInput());
    std::vector<size_t> offsets(inVal->getType()->dims().size(), 0);
    auto *TVI = builder_.createTensorViewInst(
        "tensorview.reshape." + inVal->getName().str(), inVal,
        RN->getResult().getType(), offsets);
    auto *dest = builder_.createAllocActivationInst("copy.reshape.res",
                                                    RN->getResult().getType());
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
    auto *filterG =
        builder_.createAllocActivationInst("conv.filter.G", filter->getType());

    builder_.createConvolutionGradInst(
        N->getName(), input, filter, outGrad, inG, filterG, biasG,
        CG->getKernels(), CG->getStrides(), CG->getPads(), CG->getGroup(),
        CG->getDilation(), CG->getLayout(), CG->getFusedActivation());

    registerIR(CG->getGradOfInputNamedInput(), inG);
    registerIR(CG->getGradOfInputNamedFilter(), filterG);
    registerIR(CG->getGradOfInputNamedBias(), biasG);
    break;
  }
  case glow::Kinded::Kind::MaxPoolNodeKind: {
    auto *P = cast<MaxPoolNode>(N);
    auto *in = valueForNode(P->getInput());
    auto *V = builder_.createMaxPoolWithArgmaxOp(
        N->getName(), in, P->getKernels(), P->getStrides(), P->getPads(),
        P->getLayout());
    Value *dest = V->getDest();
    Value *argmax = V->getArgmax();
    nodeToInstr_[N] = V;
    registerIR(P->getResult(), dest);
    registerIR(P->getArgmax(), argmax);
    break;
  }
  case glow::Kinded::Kind::ArgMaxNodeKind: {
    auto *P = cast<ArgMaxNode>(N);
    auto *in = valueForNode(P->getInput());
    auto *V = builder_.createArgMaxOp(N->getName(), in, P->getAxis(),
                                      P->getKeepDims());
    registerIR(P->getArgmax(), V->getArgmax());
    break;
  }
  case glow::Kinded::Kind::MaxPoolGradNodeKind: {
    auto *PG = cast<MaxPoolGradNode>(N);

    auto poolOut = PG->getOriginalOutputForResult();
    auto *outW = valueForNode(poolOut);
    auto *outG = valueForNode(PG->getGradOfOriginalOutputNamedResult());

    auto *inG = builder_.createAllocActivationInst("pool.outG",
                                                   PG->getInput().getType());

    // Find the original pool instruction.
    assert(nodeToInstr_.count(poolOut) && "Pool IRgen did not register itself");
    auto *PI = cast<MaxPoolWithArgmaxInst>(nodeToInstr_[poolOut.getNode()]);

    builder_.createMaxPoolWithArgmaxGradInst(
        N->getName(), outW, PI->getArgmax(), outG, inG, PG->getKernels(),
        PG->getStrides(), PG->getPads(), PG->getLayout());
    registerIR(PG->getGradOfInputNamedInput(), inG);
    break;
  }
  case glow::Kinded::Kind::AvgPoolGradNodeKind: {
    auto *PG = cast<AvgPoolGradNode>(N);

    auto poolOut = PG->getOriginalOutputForResult();
    auto *outW = valueForNode(poolOut);
    auto *outG = valueForNode(PG->getGradOfOriginalOutputNamedResult());

    auto *inG = builder_.createAllocActivationInst("pool.outG",
                                                   PG->getInput().getType());

    builder_.createAvgPoolGradInst(N->getName(), outW, outG, inG,
                                   PG->getKernels(), PG->getStrides(),
                                   PG->getPads(), PG->getLayout());
    registerIR(PG->getGradOfInputNamedInput(), inG);
    break;
  }
  case glow::Kinded::Kind::AdaptiveAvgPoolGradNodeKind: {
    auto *PG = cast<AdaptiveAvgPoolGradNode>(N);

    auto poolOut = PG->getOriginalOutputForResult();
    auto *outW = valueForNode(poolOut);
    auto *outG = valueForNode(PG->getGradOfOriginalOutputNamedResult());

    auto *inG = builder_.createAllocActivationInst("pool.outG",
                                                   PG->getInput().getType());

    builder_.createAdaptiveAvgPoolGradInst(N->getName(), outW, outG, inG);
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
    auto *V = builder_.createCrossEntropyLossOp(N->getName(), P, Labels);
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
    auto *yGrad =
        builder_.createAllocActivationInst("celoss.labels.grad", Y->getType());
    auto *CELossGI = builder_.createCrossEntropyLossGradInst(
        N->getName(), dY, P, Y, pGrad, yGrad);
    registerIR(CELossG->getGradOfInputNamedP(), CELossGI->getPgrad());
    registerIR(CELossG->getGradOfInputNamedLabels(), CELossGI->getLabelsgrad());
    break;
  }
  case glow::Kinded::Kind::ConcatNodeKind: {
    auto *CC = cast<ConcatNode>(N);

    auto *dest = builder_.createAllocActivationInst(CC->getName(),
                                                    CC->getResult().getType());
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
    auto *dest = builder_.createAllocActivationInst(SL->getName(),
                                                    SL->getResult().getType());
    builder_.createExtractTensorInst(SL->getName(), dest, in, start);
    registerIR(N, dest);
    break;
  }
  case glow::Kinded::Kind::InsertTensorNodeKind: {
    auto *IT = cast<InsertTensorNode>(N);
    auto start = IT->getStart();
    auto count = IT->getCount();
    auto axis = IT->getAxis();
    auto *big = valueForNode(IT->getBig());
    auto *small = valueForNode(IT->getSmall());
    auto *dest = builder_.createAllocActivationInst(IT->getName(),
                                                    IT->getResult().getType());
    builder_.createCopyInst("copy.insert", dest, big);
    builder_.createInsertTensorInst("insert", dest, small, start, count, axis);

    registerIR(N, dest);
    break;
  }
  case glow::Kinded::Kind::ScatterDataNodeKind: {
    auto *SDI = cast<ScatterDataNode>(N);
    auto *dataTensor = valueForNode(SDI->getData());
    auto *indicesTensor = valueForNode(SDI->getIndices());
    auto *slicesTensor = valueForNode(SDI->getSlices());
    auto *dest = builder_.createAllocActivationInst(SDI->getName(),
                                                    SDI->getResult().getType());
    builder_.createCopyInst("copy.scatterdata", dest, dataTensor);
    builder_.createScatterDataInst("scatterdata", dest, indicesTensor,
                                   slicesTensor, SDI->getCumulative());
    registerIR(N, dest);
    break;
  }
  case glow::Kinded::Kind::LocalResponseNormalizationNodeKind: {
    auto *LR = cast<LocalResponseNormalizationNode>(N);
    auto *in = valueForNode(LR->getInput());
    auto *V = builder_.createLocalResponseNormalizationOp(
        N->getName(), in, LR->getHalfWindowSize(), LR->getAlpha(),
        LR->getBeta(), LR->getK());
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
    auto *LRI =
        cast<LocalResponseNormalizationInst>(nodeToInstr_[originalNodeResult]);

    auto *srcGrad =
        builder_.createAllocActivationInst("lrn.res.grad", origIn->getType());

    builder_.createLocalResponseNormalizationGradInst(
        N->getName(), valueForNode(LRG->getOriginalOutputForResult()),
        valueForNode(LRG->getInput()), LRI->getScale(),
        valueForNode(LRG->getGradOfOriginalOutputNamedResult()), srcGrad,
        LRG->getHalfWindowSize(), LRG->getAlpha(), LRG->getBeta(), LRG->getK());

    registerIR(LRG->getGradOfInputNamedInput(), srcGrad);
    break;
  }
  case glow::Kinded::Kind::SaveNodeKind: {
    auto *R = cast<SaveNode>(N);
    auto *src = valueForNode(R->getInput());
    auto *dest = valueForNode(R->getOutput());
    builder_.createCopyInst(N->getName(), dest, src);
    break;
  }
  case glow::Kinded::Kind::ConstantKind: {
    auto *V = cast<Constant>(N);
    auto *W = builder_.createWeightVar(V->getType(), V->getName(),
                                       WeightVar::MutabilityKind::Constant);
    registerIR(N, W);
    break;
  }
  case glow::Kinded::Kind::PlaceholderKind: {
    auto *P = cast<Placeholder>(N);
    auto *W = builder_.createWeightVar(P->getType(), P->getName(),
                                       WeightVar::MutabilityKind::Mutable);
    registerIR(N, W);
    break;
  }
  case glow::Kinded::Kind::QuantizationProfileNodeKind: {
    auto *QPN = cast<QuantizationProfileNode>(N);
    auto *inputTensor = valueForNode(QPN->getInput());
    auto *histogram = valueForNode(QPN->getHistogramPlaceholder());
    auto *computationInfo = valueForNode(QPN->getComputationInfoPlaceholder());
    builder_.createQuantizationProfileInst(QPN->getName(), inputTensor,
                                           histogram, computationInfo);
    break;
  }
  case glow::Kinded::Kind::TopKNodeKind: {
    auto *TKN = cast<TopKNode>(N);
    auto *inputTensor = valueForNode(TKN->getInput());
    auto k = TKN->getK();
    auto *V = builder_.createTopKOp(N->getName(), inputTensor, k);
    registerIR(TKN->getValues(), V->getValues());
    registerIR(TKN->getIndices(), V->getIndices());
    break;
  }
  case glow::Kinded::Kind::TraceEventNodeKind: {
    auto *TEN = cast<TraceEventNode>(N);
    auto *dataTensor = valueForNode(TEN->getData());
    builder_.createTraceEventInst(TEN->getName(), dataTensor, TEN->getIndex());
    break;
  }
  case glow::Kinded::Kind::SparseLengthsWeightedSumGradNodeKind: {
    auto *SLWSG = cast<SparseLengthsWeightedSumGradNode>(N);

    auto *data = valueForNode(SLWSG->getData());
    auto *weights = valueForNode(SLWSG->getWeights());
    auto *indices = valueForNode(SLWSG->getIndices());
    auto *lengths = valueForNode(SLWSG->getLengths());

    auto *destGrad = valueForNode(SLWSG->getGradOfOriginalOutputNamedResult());
    auto *dataGrad = builder_.createAllocActivationInst(
        "slws.data.G", SLWSG->getGradOfInputNamedData().getType());
    auto *weightsGrad = builder_.createAllocActivationInst(
        "slws.weights.G", SLWSG->getGradOfInputNamedWeights().getType());

    builder_.createSparseLengthsWeightedSumGradInst(N->getName(), data, weights,
                                                    indices, lengths, destGrad,
                                                    dataGrad, weightsGrad);

    registerIR(SLWSG->getGradOfInputNamedData(), dataGrad);
    registerIR(SLWSG->getGradOfInputNamedWeights(), weightsGrad);
    break;
  }
  }
}

void IRFunction::generateIR(const Backend &B) {
  assert(G_->verify() && "Invalid function");
  // Schedule the nodes.
  NodesPtrList ScheduledNodes;
  scheduleGraph(ScheduledNodes);
  IRGenVisitor irgen(this, B);

  for (auto &N : ScheduledNodes) {
    N->visit(nullptr, &irgen);
  }
}
