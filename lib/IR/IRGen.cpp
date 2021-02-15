/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

#include "llvm/ADT/StringExtras.h"
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

#define DECORATE_NODE_NAME(Node, ...)                                          \
  llvm::join_items("_", Node->getName(), __VA_ARGS__)

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
    std::vector<dim_t> offsets(inVal->getType()->dims().size(), 0);
    auto *TVI = builder_.createTensorViewInst(
        DECORATE_NODE_NAME(N, "tensorview"), inVal, RN->getResult().getType(),
        offsets);
    auto *dest = builder_.createAllocActivationInst(
        DECORATE_NODE_NAME(N, "res"), RN->getResult().getType());
    builder_.createCopyInst(DECORATE_NODE_NAME(N, "copy"), dest, TVI);
    registerIR(N, dest);
    break;
  }
  case glow::Kinded::Kind::ConvolutionGradNodeKind: {
    auto *CG = cast<ConvolutionGradNode>(N);

    auto *input = valueForNode(CG->getInput());
    auto *filter = valueForNode(CG->getFilter());
    auto *bias = valueForNode(CG->getBias());

    auto *outGrad = valueForNode(CG->getGradOfOriginalOutputNamedResult());

    auto *inG = builder_.createAllocActivationInst(
        DECORATE_NODE_NAME(N, "input", "grad"), input->getType());
    auto *biasG = builder_.createAllocActivationInst(
        DECORATE_NODE_NAME(N, "bias", "grad"), bias->getType());
    auto *filterG = builder_.createAllocActivationInst(
        DECORATE_NODE_NAME(N, "filter", "grad"), filter->getType());

    builder_.createConvolutionGradInst(
        N->getName(), input, filter, outGrad, inG, filterG, biasG,
        CG->getKernels(), CG->getStrides(), CG->getPads(), CG->getGroup(),
        CG->getDilation(), CG->getLayout(), CG->getFusedActivation(),
        CG->getFusedActivationArgs());

    registerIR(CG->getGradOfInputNamedInput(), inG);
    registerIR(CG->getGradOfInputNamedFilter(), filterG);
    registerIR(CG->getGradOfInputNamedBias(), biasG);
    break;
  }
  case glow::Kinded::Kind::MaxPoolNodeKind: {
    auto *P = cast<MaxPoolNode>(N);
    auto *in = valueForNode(P->getInput());
    auto argMax = P->getArgmax();
    auto *V = builder_.createMaxPoolWithArgmaxOp(
        N->getName(), in, P->getKernels(), P->getStrides(), P->getPads(),
        P->getLayout(), argMax.getElementType(), P->getFlattenIndices());
    Value *dest = V->getDest();
    Value *argmax = V->getArgmax();
    nodeToInstr_[N] = V;
    registerIR(P->getResult(), dest);
    registerIR(P->getArgmax(), argmax);
    break;
  }
  case glow::Kinded::Kind::MaxPoolGradNodeKind: {
    auto *PG = cast<MaxPoolGradNode>(N);

    auto poolIn = PG->getInput();
    auto poolOut = PG->getOriginalOutputForResult();
    auto *inW = valueForNode(poolIn);
    auto *outW = valueForNode(poolOut);
    auto *outG = valueForNode(PG->getGradOfOriginalOutputNamedResult());

    auto *inG = builder_.createAllocActivationInst(
        DECORATE_NODE_NAME(N, "outG"), PG->getInput().getType());

    // Find the original pool instruction.
    assert(nodeToInstr_.count(poolOut) && "Pool IRgen did not register itself");
    auto *PI = cast<MaxPoolWithArgmaxInst>(nodeToInstr_[poolOut.getNode()]);

    builder_.createMaxPoolWithArgmaxGradInst(
        N->getName(), outW, inW, PI->getArgmax(), outG, inG, PG->getKernels(),
        PG->getStrides(), PG->getPads(), PG->getLayout(),
        PG->getFlattenIndices());
    registerIR(PG->getGradOfInputNamedInput(), inG);
    break;
  }
  case glow::Kinded::Kind::AvgPoolGradNodeKind: {
    auto *PG = cast<AvgPoolGradNode>(N);

    auto poolIn = PG->getInput();
    auto poolOut = PG->getOriginalOutputForResult();
    auto *inW = valueForNode(poolIn);
    auto *outW = valueForNode(poolOut);
    auto *outG = valueForNode(PG->getGradOfOriginalOutputNamedResult());

    auto *inG = builder_.createAllocActivationInst(
        DECORATE_NODE_NAME(N, "outG"), PG->getInput().getType());

    builder_.createAvgPoolGradInst(
        N->getName(), outW, inW, outG, inG, PG->getKernels(), PG->getStrides(),
        PG->getPads(), PG->getLayout(), PG->getCountIncludePads());
    registerIR(PG->getGradOfInputNamedInput(), inG);
    break;
  }
  case glow::Kinded::Kind::AdaptiveAvgPoolGradNodeKind: {
    auto *PG = cast<AdaptiveAvgPoolGradNode>(N);

    auto poolOut = PG->getOriginalOutputForResult();
    auto *outW = valueForNode(poolOut);
    auto *outG = valueForNode(PG->getGradOfOriginalOutputNamedResult());

    auto *inG = builder_.createAllocActivationInst(
        DECORATE_NODE_NAME(N, "outG"), PG->getInput().getType());

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
    auto *srcGrad = builder_.createAllocActivationInst(
        DECORATE_NODE_NAME(N, "res"), outGrad->getType());
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
    auto *pGrad = builder_.createAllocActivationInst(
        DECORATE_NODE_NAME(N, "p", "grad"), P->getType());
    auto *yGrad = builder_.createAllocActivationInst(
        DECORATE_NODE_NAME(N, "labels", "grad"), Y->getType());
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
    // Mark the buffer as initialized, this is safe since the InsertTensors
    // below will fully overwrite the buffer.
    builder_.createTouchInst(CC->getName(), dest);
    auto inputs = CC->getInputs();

    // We start inserting to the shape at (0,0, ... ).
    std::vector<dim_t> offsets(CC->getResult().dims().size(), 0);
    unsigned dim = CC->getDim();

    for (size_t i = 0, e = inputs.size(); i < e;) {
      // Look for a series of the same Node being concated consecutively many
      // times. We can wrap n such consecutive repeats into a single insert
      // with count n along the dim axis.
      const size_t consecutiveCount = getConsecutiveSameNodeCount(inputs, i);

      // Create the new InsertTensor instruction given the input node, along
      // with the number of times to insert the node and the axis (dim) we are
      // inserting in.
      builder_.createInsertTensorInst(
          DECORATE_NODE_NAME(CC, inputs[i].getNode()->getName()), dest,
          valueForNode(inputs[i]), offsets, consecutiveCount, dim);

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
    builder_.createCopyInst(DECORATE_NODE_NAME(N, "copy"), dest, big);
    builder_.createInsertTensorInst(IT->getName(), dest, small, start, count,
                                    axis);

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
    builder_.createCopyInst(DECORATE_NODE_NAME(N, "copy"), dest, dataTensor);
    builder_.createScatterDataInst(SDI->getName(), dest, indicesTensor,
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

    auto *srcGrad = builder_.createAllocActivationInst(
        DECORATE_NODE_NAME(N, "res", "grad"), origIn->getType());

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
    auto *V = builder_.createTopKOp(N->getName(), inputTensor, k,
                                    TKN->getIndices().getElementType());
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
  case glow::Kinded::Kind::SparseLengthsSumGradNodeKind: {
    auto *SLSG = cast<SparseLengthsSumGradNode>(N);

    auto *data = valueForNode(SLSG->getData());
    auto *indices = valueForNode(SLSG->getIndices());
    auto *lengths = valueForNode(SLSG->getLengths());

    auto *destGrad = valueForNode(SLSG->getGradOfOriginalOutputNamedResult());
    auto *dataGrad = builder_.createAllocActivationInst(
        DECORATE_NODE_NAME(N, "dataG"),
        SLSG->getGradOfInputNamedData().getType());

    builder_.createSparseLengthsSumGradInst(
        N->getName(), data, indices, lengths, destGrad, dataGrad,
        SLSG->getLengthsMode(), SLSG->getAvgLength());

    registerIR(SLSG->getGradOfInputNamedData(), dataGrad);
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
        DECORATE_NODE_NAME(N, "dataG"),
        SLWSG->getGradOfInputNamedData().getType());
    auto *weightsGrad = builder_.createAllocActivationInst(
        DECORATE_NODE_NAME(N, "weightsG"),
        SLWSG->getGradOfInputNamedWeights().getType());

    builder_.createSparseLengthsWeightedSumGradInst(
        N->getName(), data, weights, indices, lengths, destGrad, dataGrad,
        weightsGrad, SLWSG->getLengthsMode(), SLWSG->getAvgLength());

    registerIR(SLWSG->getGradOfInputNamedData(), dataGrad);
    registerIR(SLWSG->getGradOfInputNamedWeights(), weightsGrad);
    break;
  }
  case glow::Kinded::Kind::BatchedPairwiseDotProductNodeKind: {
    auto *BPDPN = llvm::cast<BatchedPairwiseDotProductNode>(N);
    auto firstInput = BPDPN->getInputs()[0];

    std::string allocName = std::string(BPDPN->getName()) + ".res";
    auto *dest = builder_.createAllocActivationInst(
        allocName, BPDPN->getResult().getType());

    auto *inst = builder_.createBatchedPairwiseDotProductInst(
        BPDPN->getName(), dest, BPDPN->getInputs().size(),
        firstInput.getType()->dims()[1]);

    // First instruction operand is the buffer to write the dot products, the
    // rest are all inputs.
    for (auto &in : BPDPN->getInputs()) {
      inst->pushOperand({valueForNode(in), OperandKind::In});
    }

    registerIR(BPDPN->getResult(), dest);
    break;
  }

  case glow::Kinded::Kind::BatchedPairwiseDotProductGradNodeKind: {
    auto *BPDPGN = llvm::cast<BatchedPairwiseDotProductGradNode>(N);

    auto *in0 = valueForNode(BPDPGN->getOriginalInputs()[0]);
    auto *outputGrad = valueForNode(BPDPGN->getOutputGrad());

    // First, create alloc instructions for all of the gradients. This needs to
    // be done first so that these instructions precede the first use of the
    // buffers they create.
    std::vector<Value *> dests;
    for (unsigned i = 0, e = BPDPGN->getNumResults(); i < e; ++i) {
      NodeValue res = BPDPGN->getNthResult(i);
      std::string allocName =
          std::string(BPDPGN->getName()) + ".res." + std::to_string(i);
      auto *dest = builder_.createAllocActivationInst(allocName, res.getType());
      dests.emplace_back(dest);
    }

    auto *inst = builder_.createBatchedPairwiseDotProductGradInst(
        BPDPGN->getName(), outputGrad, BPDPGN->getOriginalInputs().size(),
        in0->dims()[1]);

    // Operands 1 -> numInputs are gradients.
    for (unsigned i = 0, e = BPDPGN->getNumResults(); i < e; ++i) {
      NodeValue res = BPDPGN->getNthResult(i);
      inst->pushOperand({dests[i], OperandKind::Out});
      registerIR(res, dests[i]);
    }

    // Operands numInputs + 1 -> 2 * numInputs are original inputs.
    for (auto &in : BPDPGN->getOriginalInputs()) {
      inst->pushOperand({valueForNode(in), OperandKind::In});
    }
    break;
  }
  case glow::Kinded::Kind::ExternalFunctionCallNodeKind: {
    auto *EFCN = llvm::cast<ExternalFunctionCallNode>(N);
    std::string externalCallType = std::string(EFCN->getName());
    std::string allocName = std::string(EFCN->getName()) + ".res";
    auto *dest = builder_.createAllocActivationInst(
        allocName, EFCN->getResult().getType());

    auto *inst = builder_.createExternalFunctionCallInst(
        EFCN->getName(), dest, EFCN->getFunctionName(), EFCN->getFunctionImpl(),
        EFCN->getFunctionKind());

    // First instruction operand is the buffer for the result, the
    // rest are all inputs.
    for (auto &in : EFCN->getInputs()) {
      inst->pushOperand({valueForNode(in), OperandKind::In});
    }
    registerIR(EFCN->getResult(), dest);
    break;
  }
  }
}

void IRFunction::generateIR(const Backend &B) {
  assert(G_->verify(&B) && "Invalid function");
  // Schedule the nodes.
  NodesPtrList ScheduledNodes;
  scheduleGraph(ScheduledNodes);
  IRGenVisitor irgen(this, B);

  for (auto &N : ScheduledNodes) {
    N->visit(nullptr, &irgen);
  }

  if (!B.verify(*this)) {
    EXIT_ON_ERR(
        MAKE_ERR(ErrorValue::ErrorCode::COMPILE_UNSUPPORTED_IR_AFTER_GENERATE,
                 "Unsupported instruction(s) found after generating IR " +
                     getName().str() + " for backend " + B.getBackendName()));
  }
}
