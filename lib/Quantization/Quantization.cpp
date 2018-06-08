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

#include "glow/Quantization/Quantization.h"

#include "glow/ExecutionEngine/ExecutionEngine.h"

#include <cmath>
#include <unordered_set>
#include <vector>

using llvm::cast;

namespace glow {
namespace quantization {

TensorQuantizationParams chooseQuantizationParams(float min, float max) {
  assert(min <= max && "min must not be bigger than max");

  // Given 8 bit precision.
  const int32_t qmin = std::numeric_limits<int8_t>::min();
  const int32_t qmax = std::numeric_limits<int8_t>::max();

  // We extend the [min, max] interval to ensure that it contains 0.
  // Otherwise, we would not meet the requirement that 0 be an exactly
  // representable value.
  min = std::min(min, 0.f);
  max = std::max(max, 0.f);

  double scale = (max - min) / ((double)qmax - qmin);

  // Dequantization uses the following formula scale * (X - offset), so
  // scale should not be equal to zero.
  // If scale is 0, we arbitrary adjust the scale to 0.1.
  if (scale == 0)
    scale = 0.1;

  assert(scale > 0 && "Scale must be non negative");

  // Zero-point computation.
  // First the initial floating-point computation. The zero-point can be
  // determined from solving an affine equation for any known pair
  // (real value, corresponding quantized value).
  // We know two such pairs: (rmin, qmin) and (rmax, qmax).
  // The arithmetic error on the zero point computed from either pair
  // will be roughly machine_epsilon * (sum of absolute values of terms)
  // so we want to use the variant that adds the smaller terms.
  double zeroPointFromMin = qmin - min / scale;
  double zeroPointFromMax = qmax - max / scale;
  double zeroPointFromMinError = std::abs(qmin) + std::abs(min / scale);
  double zeroPointFromMaxError = std::abs(qmax) + std::abs(max / scale);
  double initialZeroPoint = zeroPointFromMinError < zeroPointFromMaxError
                                ? zeroPointFromMin
                                : zeroPointFromMax;

  // For symmetric quantization, if min == -max, force the zero point to be 0.
  if (min == -max) {
    initialZeroPoint = 0;
  }

  // Now we need to nudge the zero point to be an integer (our zero points are
  // integer, and this is motivated by the requirement to be able to represent
  // the real value "0" exactly as a quantized value, which is required in
  // multiple places, for example in Im2col with SAME padding).
  int32_t nudgedZeroPoint = 0;
  if (initialZeroPoint < qmin) {
    nudgedZeroPoint = qmin;
  } else if (initialZeroPoint > qmax) {
    nudgedZeroPoint = qmax;
  } else {
    nudgedZeroPoint = static_cast<int32_t>(round(initialZeroPoint));
  }

  TensorQuantizationParams result{static_cast<float>(scale), nudgedZeroPoint};
  return result;
}

QuantizationTransform32To8 quantizeScaleOffset32To8(float scale,
                                                    int32_t offset) {
  // In this function we compute an efficient way to convert signed 32-bit
  // integers into signed 8-bit integers without the use of floating-point
  // multiplication. Instead, we represent the original calculation:
  //
  //    result = (x * scale + offset)
  //
  // as the following sequence of integer calculations:
  //
  //    ((x >> pre_scale  * integer_scale) >> post_scale) + offset
  //
  // This function converts the floating-point scale and offset values to the
  // constants in the integer formula.
  //
  // In this method we assume that any signed 32-bit integer in the input word
  // must be mapped into an 8-bit integer. If the scale factor is 2X, then the
  // number 1000 won't be a legal input because after scaling the result would
  // fall outside of the signed 8-bit range. Any 32-bit number that falls
  // outside of signed the 8-bit output integer will be clipped. This gives us
  // the ability to perform 32-bit arithmetic, as explained below.
  //
  // We can't accurately represent fraction scales (in the range zero to one),
  // because the lowest integer multiplication value is one. For example, the
  // scaling factor 0.25 must be represented as integer multiplication of either
  // zero or one, which would result in an highly inaccurate output.
  // Similarly, rounding the scaling factor of 1.6 to 2.0 would produce
  // inaccurate results because drop a significant part of the number.
  //
  // The solution here is to scale (increase in size) the signed integer scalar,
  // and divide the result by shifting it to the right hand side. For example,
  // the floating-point scalar 0.41 is multiplied by 32x (to 13.12, rounded to
  // 13). Then the signed 32-bit integer input is multiplied by 13, and then
  // shifted 5 times to the right (to shrink the result back). The output of
  // this calculation is (13.0 / 32), which is about ~0.4.
  //
  // This approach works well for some scale values. Notice that the modified
  // integer multiplication requires more bits because the intermediate result
  // is larger. Notice that it's always safe to promote the scalar value from a
  // fraction up to one. When you multiply by the integer value one, the
  // intermediate result does not overflow (does not require more bits).
  //
  // It is actually always safe to perform 16-bit post-multiplication
  // right-shifts. Let's consider two cases. If the value of the floating-point
  // scale is greater than 1.0 then we know that at most 8 of the 32-bits in the
  // input register are used, because the result must fit in 8-bits. The result
  // of 8-bit times 8-bit multiplication is 16-bits, which leaves another 16
  // bits that are unused. We can use these 16-bits to increase the size of the
  // integer scale, and shift the result, as described above, without
  // overflowing the register.
  // The second case is where the scalar value is smaller than 1.0.
  // Multiplication of any number by zero or one does not increase the number of
  // bits which are used by the number.
  //
  // Now, we need to consider another problem. In the previous section we
  // described how we scaled small fractions into a number that's close to one.
  // But scaling to around 1.0 is not accurate enough. Rounding a scale factor
  // like 0.6 to integer would give a very high error rate. Generally, we can't
  // increase the size of the integer multiplier without a limit because this
  // would overflow large values that are close to the upper signed 32-bit
  // limit.
  //
  // To solve the accuracy problem we need to continue to increase the size of
  // the integer scalar without overflowing the signed 32-bit register.
  // The solution here is to perform right-shift on the input, in addition to
  // the output. The idea here is that by performing the post-multiplication
  // right-shift we pick the high bits from the result of the multiplication,
  // and the low bits are ignored. This means that we can continue to increase
  // the size of the integer multiplier and continue to increase the accuracy of
  // the calculation by pre-shifting the 32-bit input. Shifting the input to the
  // right would flip some input bits to zero, but the accuracy loss would be
  // minimal.
  //
  // If the floating point scale factor small then it spans a small part of the
  // 32-bit word. For example, a scale factor of 0.125 (1/8) scales some range
  // into the signed 8-bit result. This range is 8 + 3 bits. This means that we
  // can shift as much as 32-11 bits without overflowing the register. This is
  // a net win because we get to increase the accuracy of the floating point
  // scale factor. For very small scale factors, the used range is very large
  // and can take up the whole 32-bit register, so overflow is a real problem.
  // Here we can use the post-shift value to estimate how many bits will be
  // discarded from the after the multiplication operation and figure out how
  // many bits we can take from the bottom of the input word by shifting it to
  // the right and add more precision to the integer scale multiplier.
  int preShift = 0;
  int postShift = 0;

  // Calculate the post-shift value. It's always safe to increase scale as long
  // as it's below one, and it's always legal to shift at least 15 bits for
  // small scale values.
  while (scale < 0.5 || (scale < 256 && postShift < 15)) {
    scale *= 2;
    postShift++;
  }

  // Calculate the pre-multiplication shift. Estimate how many bits we can take
  // from the input number and pass to the integer scale.
  while (scale < 255 && preShift < (postShift / 2)) {
    scale *= 2;
    preShift++;
  }

  return QuantizationTransform32To8(preShift, postShift, std::round(scale),
                                    offset);
}

std::vector<NodeQuantizationInfo>
generateNodeQuantizationInfos(const Function *F) {
  std::vector<NodeQuantizationInfo> quantizationInfos;

  for (auto &node : F->getNodes()) {
    auto *QPN = llvm::dyn_cast<QuantizationProfileNode>(&node);

    if (QPN) {
      auto CI = QPN->getComputationInfoVar()->getHandle<float>();
      auto histogram = QPN->getHistogramVar()->getHandle<float>();
      float min = CI.raw(0);
      float max = CI.raw(1);

      std::string fullOutputName = NodeQuantizationInfo::generateNodeOutputName(
          QPN->getProfiledNodeName(), QPN->getProfiledOutputNumber());

      // TODO: Ideally tensor quantization params should be calculated
      // based on the histogram distribution. Use simplistic approach for now.
      (void)histogram;
      TensorQuantizationParams TQP = chooseQuantizationParams(min, max);

      quantizationInfos.emplace_back(fullOutputName, TQP);
    }
  }

  return quantizationInfos;
}

int8_t quantize(float input, const TensorQuantizationParams &TQP) {
  float result = input / TQP.scale_ + TQP.offset_;
  return quantization::clip<int32_t, int8_t>(round(result));
}

float dequantize(int8_t input, const TensorQuantizationParams &TQP) {
  return TQP.scale_ * (input - TQP.offset_);
}

/// Quantize all inputs for \p node and return back pointers to the newly
/// created qunatization nodes.
static llvm::SmallVector<NodeValue, 6>
quantizeInputs(Function *F, Node *node,
               const std::unordered_map<std::string, TensorQuantizationParams>
                   &nodeToTQP) {
  llvm::SmallVector<NodeValue, 6> quantizedInputs;

  for (unsigned i = 0, e = node->getNumInputs(); i < e; ++i) {
    NodeValue &NV = node->getNthInput(i);

    // Do not quantize non floating point type, e.g., Index type.
    if (NV.getElementType() != ElemKind::FloatTy) {
      continue;
    }

    std::string nodeOutputName = NodeQuantizationInfo::generateNodeOutputName(
        NV->getName(), NV.getResNo());
    assert(nodeToTQP.find(nodeOutputName) != nodeToTQP.end() &&
           "Missing quantization params for a node");

    const TensorQuantizationParams &TQP =
        nodeToTQP.find(nodeOutputName)->second;
    auto QT = F->getParent()->uniqueType(ElemKind::Int8QTy, NV->dims(),
                                         TQP.scale_, TQP.offset_);

    Node *quantizeNode = F->createQuantize("quantize", NV, QT);
    quantizedInputs.push_back(quantizeNode);
  }

  return quantizedInputs;
}

/// \returns true when given \p node can be quantized.
/// Normally node's inputs must have floating point
/// type, but there are some special cases, e.g., Gather node.
static bool canBeQuantized(const Node *node) {
  // Handle special cases like Gather node when some of the inputs does not
  // need to be quantized while some does.
  switch (node->getKind()) {
  case Kinded::Kind::GatherNodeKind: {
    auto *gather = cast<GatherNode>(node);
    return gather->getData().getElementType() == ElemKind::FloatTy;
  }
  default:
    // Let the general procedure handle this node kind.
    break;
  }

  // Make sure that all inputs are floats.
  for (unsigned i = 0, e = node->getNumInputs(); i < e; ++i) {
    if (node->getNthInput(i).getElementType() != ElemKind::FloatTy) {
      return false;
    }
  }

  return true;
}

/// Quantize the \p node such that all floating point inputs and outputs
/// are quantized to int8 type with some scale and offset.
/// \returns Quantized node.
///
/// \param F Function which holds the non quantized \p node.
/// \param node Node to be quantized.
/// \param quantizedInputs Array of already quantized inputs to the result node.
/// \param qParams Tensor quantization parameters for all outputs of the
///        \p node.
static Node *quantizeNode(Function *F, Node *node,
                          llvm::MutableArrayRef<NodeValue> quantizedInputs,
                          llvm::ArrayRef<TensorQuantizationParams> qParams) {
  Node *quantizedNode{};

  switch (node->getKind()) {
  case Kinded::Kind::FullyConnectedNodeKind: {
    auto *FC = cast<FullyConnectedNode>(node);
    assert(quantizedInputs.size() == 3 && "Invalid number of inputs");
    assert(qParams.size() == 1 && "Invalid number of quantized outputs");
    auto QT =
        F->getParent()->uniqueType(ElemKind::Int8QTy, FC->getResult()->dims(),
                                   qParams[0].scale_, qParams[0].offset_);
    quantizedNode =
        F->createFullyConnected(FC->getName(), quantizedInputs[0],
                                quantizedInputs[1], quantizedInputs[2], QT);
    break;
  }

  case Kinded::Kind::ConvolutionNodeKind: {
    auto *CV = cast<ConvolutionNode>(node);
    assert(quantizedInputs.size() == 3 && "Invalid number of inputs");
    assert(qParams.size() == 1 && "Invalid number of quantized outputs");
    auto QT =
        F->getParent()->uniqueType(ElemKind::Int8QTy, CV->getResult()->dims(),
                                   qParams[0].scale_, qParams[0].offset_);
    quantizedNode =
        F->createConv(CV->getName(), quantizedInputs[0], quantizedInputs[1],
                      quantizedInputs[2], QT, CV->getKernel(), CV->getStride(),
                      CV->getPads(), CV->getGroup());
    break;
  }
  case Kinded::Kind::SliceNodeKind: {
    auto *S = cast<SliceNode>(node);
    assert(quantizedInputs.size() == 1 && "Invalid number of inputs");
    assert(qParams.size() == 1 && "Invalid number of quantized outputs");

    auto QT = F->getParent()->uniqueType(
        ElemKind::Int8QTy, S->getResult()->dims(),
        quantizedInputs[0]->getNthResult(0).getType()->getScale(),
        quantizedInputs[0]->getNthResult(0).getType()->getOffset());

    quantizedNode =
        F->createSlice(S->getName(), quantizedInputs[0], S->getStart(), QT);
    break;
  }
  case Kinded::Kind::ReluNodeKind: {
    auto *R = cast<ReluNode>(node);
    assert(quantizedInputs.size() == 1 && "Invalid number of inputs");
    assert(qParams.size() == 1 && "Invalid number of quantized outputs");

    quantizedNode = F->createRELU(R->getName(), quantizedInputs[0]);
    break;
  }
  case Kinded::Kind::TransposeNodeKind: {
    auto *T = cast<TransposeNode>(node);
    assert(quantizedInputs.size() == 1 && "Invalid number of inputs");
    assert(qParams.size() == 1 && "Invalid number of quantized outputs");
    quantizedNode =
        F->createTranspose(T->getName(), quantizedInputs[0], T->getShuffle());
    break;
  }
  case Kinded::Kind::ReshapeNodeKind: {
    auto *R = cast<ReshapeNode>(node);
    assert(quantizedInputs.size() == 1 && "Invalid number of inputs");
    assert(qParams.size() == 1 && "Invalid number of quantized outputs");
    quantizedNode =
        F->createReshape(R->getName(), quantizedInputs[0], R->getDims());
    break;
  }
  case Kinded::Kind::PoolMaxNodeKind: {
    auto *P = cast<PoolMaxNode>(node);
    assert(quantizedInputs.size() == 1 && "Invalid number of inputs");
    assert(qParams.size() == 1 && "Invalid number of quantized outputs");

    quantizedNode =
        F->createPoolMax(node->getName(), quantizedInputs[0], P->getKernel(),
                         P->getStride(), P->getPad());
    break;
  }
  case Kinded::Kind::PoolAvgNodeKind: {
    auto *P = cast<PoolAvgNode>(node);
    assert(quantizedInputs.size() == 1 && "Invalid number of inputs");
    assert(qParams.size() == 1 && "Invalid number of quantized outputs");

    quantizedNode =
        F->createPoolAvg(node->getName(), quantizedInputs[0], P->getKernel(),
                         P->getStride(), P->getPad());
    break;
  }
#define CASE_QUANTIZE_NODE(NODE_NAME_)                                         \
  case Kinded::Kind::NODE_NAME_##NodeKind: {                                   \
    auto *AN = cast<NODE_NAME_##Node>(node);                                   \
    assert(quantizedInputs.size() == 2 && "Invalid number of inputs");         \
    assert(qParams.size() == 1 && "Invalid number of quantized outputs");      \
    auto outTy =                                                               \
        F->getParent()->uniqueType(ElemKind::Int8QTy, AN->getResult().dims(),  \
                                   qParams[0].scale_, qParams[0].offset_);     \
    quantizedNode = F->create##NODE_NAME_(                                     \
        AN->getName(), outTy, quantizedInputs[0], quantizedInputs[1]);         \
    break;                                                                     \
  }
    CASE_QUANTIZE_NODE(Add);
    CASE_QUANTIZE_NODE(Mul);
    CASE_QUANTIZE_NODE(Sub);
    CASE_QUANTIZE_NODE(Max);
    CASE_QUANTIZE_NODE(Min);
#undef CASE_QUANTIZE_NODE
  case Kinded::Kind::ConcatNodeKind: {
    auto *C = cast<ConcatNode>(node);
    assert(qParams.size() == 1 && "Invalid number of quantized outputs");

    // Concat just moves tensors around, make sure that all tensors have the
    // same {S,O} params.
    for (size_t qi = 0, e = quantizedInputs.size(); qi < e; qi++) {
      auto argOutTy = F->getParent()->uniqueType(
          ElemKind::Int8QTy, quantizedInputs[qi]->dims(), qParams[0].scale_,
          qParams[0].offset_);

      quantizedInputs[qi] = F->createRescaleQuantized(
          quantizedInputs[qi]->getName(), quantizedInputs[qi], argOutTy);
    }

    auto outTy =
        F->getParent()->uniqueType(ElemKind::Int8QTy, C->getResult()->dims(),
                                   qParams[0].scale_, qParams[0].offset_);
    quantizedNode =
        F->createConcat(node->getName(), quantizedInputs, C->getDim(), outTy);
    break;
  }
  case Kinded::Kind::GatherNodeKind: {
    auto *gather = cast<GatherNode>(node);
    // Gather node has 2 inputs, but only one should be quantized.
    assert(quantizedInputs.size() == 1 && "Invalid number of inputs");
    assert(qParams.size() == 1 && "Invalid number of quantized outputs");

    quantizedNode = F->createGather(gather->getName(), quantizedInputs[0],
                                    gather->getIndices());

    break;
  }
  case Kinded::Kind::TopKNodeKind: {
    auto *topK = cast<TopKNode>(node);
    assert(quantizedInputs.size() == 1 && "Invalid number of inputs");
    // Two outputs but only one should be quantized.
    assert(qParams.size() == 1 && "Invalid number of quantized outputs");

    quantizedNode =
        F->createTopK(topK->getName(), quantizedInputs[0], topK->getK());
    break;
  }
  case Kinded::Kind::TanhNodeKind: {
    assert(quantizedInputs.size() == 1 && "Invalid number of inputs");

    // Quantized tanh operator expects input to be in a certain floating point
    // range. This operator works based on the precomputed table and has to
    // process input in a range of [-3.0, 3.0]. Tanh asymptotically approaches
    // +/-1.0 and is already +/-.995 at +/-3.0.
    // The output quantization parameters are chosen to represent the floating
    // point range of [-1.0, 1.0].
    auto inputQuantizationParams = chooseQuantizationParams(-3.0, 3.0);
    auto tanhInTy = F->getParent()->uniqueType(
        ElemKind::Int8QTy, quantizedInputs[0]->dims(),
        inputQuantizationParams.scale_, inputQuantizationParams.offset_);

    // Make sure input is clipped in [-3.0, 3.0] floating point range.
    auto *rescaleNode = F->createRescaleQuantized(quantizedInputs[0]->getName(),
                                                  quantizedInputs[0], tanhInTy);

    // Make sure output is clipped in [-1.0, 1.0] floating point range.
    auto outputQuantizationParams = chooseQuantizationParams(-1.0, 1.0);
    auto resultOutTy = F->getParent()->uniqueType(
        ElemKind::Int8QTy, rescaleNode->getResult().dims(),
        outputQuantizationParams.scale_, outputQuantizationParams.offset_);

    quantizedNode = F->createIntTanh(node->getName(), rescaleNode, resultOutTy);
    break;
  }
  case Kinded::Kind::SigmoidNodeKind: {
    assert(quantizedInputs.size() == 1 && "Invalid number of inputs");

    // Quantized sigmoid operator expects input to be in a certain floating
    // point range. This operator works based on the precomputed table and has
    // to process input in a range of [-6.0, 6.0]. Sigmoid asymptotically
    // approaches 0 at -inf and 1 at +inf. It has values of 0.00247262 and
    // 0.997527 at -6.0 and 6.0 correspondingly. The output quantization
    // parameters are chosen to represent the floating point range of [0, 1.0].
    auto inputQuantizationParams = chooseQuantizationParams(-6.0, 6.0);
    auto sigmoidInTy = F->getParent()->uniqueType(
        ElemKind::Int8QTy, quantizedInputs[0]->dims(),
        inputQuantizationParams.scale_, inputQuantizationParams.offset_);

    // Make sure input is clipped in [-6.0, 6.0] floating point range.
    auto *rescaleNode = F->createRescaleQuantized(
        quantizedInputs[0]->getName(), quantizedInputs[0], sigmoidInTy);

    // Make sure output is clipped in [0.0, 1.0] floating point range.
    auto outputQuantizationParams = chooseQuantizationParams(0.0, 1.0);
    auto resultOutTy = F->getParent()->uniqueType(
        ElemKind::Int8QTy, rescaleNode->getResult().dims(),
        outputQuantizationParams.scale_, outputQuantizationParams.offset_);

    quantizedNode =
        F->createIntSigmoid(node->getName(), rescaleNode, resultOutTy);

    break;
  }
  default:
    GLOW_UNREACHABLE("The node type is not supported for quantization");
  }

  return quantizedNode;
}

/// \returns Tensor quantization parameters for all eligible (floating point)
/// outputs of the \p node.
///
/// \param node Node for which quantization params are gathered.
/// \param nodeToTQP Tensor quantization parameters for all nodes.
static llvm::SmallVector<TensorQuantizationParams, 6> getQuantizationParameters(
    Node *node,
    std::unordered_map<std::string, TensorQuantizationParams> &nodeToTQP) {
  llvm::SmallVector<TensorQuantizationParams, 6> result;

  for (unsigned outNum = 0, e = node->getNumResults(); outNum < e; outNum++) {
    // Only floating point outputs were profiled.
    if (node->getNthResult(outNum).getElementType() != ElemKind::FloatTy) {
      continue;
    }

    const std::string nodeOutputName =
        NodeQuantizationInfo::generateNodeOutputName(node->getName(), outNum);
    assert(nodeToTQP.find(nodeOutputName) != nodeToTQP.end() &&
           "Missing quantization params for a node");
    result.push_back(nodeToTQP[nodeOutputName]);
  }

  return result;
}

/// Some of the nodes need special post processing after quantization.
/// For example, RELU node needs to have adjusted quantization parameters.
static Node *
postProcessQuantizedNode(Function *F, Node *quantizedNode,
                         llvm::ArrayRef<TensorQuantizationParams> qParams) {
  if (quantizedNode->getKind() == Kinded::Kind::ReluNodeKind ||
      quantizedNode->getKind() == Kinded::Kind::PoolMaxNodeKind ||
      quantizedNode->getKind() == Kinded::Kind::PoolAvgNodeKind ||
      quantizedNode->getKind() == Kinded::Kind::GatherNodeKind) {
    // These nodes do not change {S,O} of the output, they use the same
    // {S,O} as the input. Make sure that rescale is applied to comply with
    // the taken profile from the node.
    auto outTy =
        F->getParent()->uniqueType(ElemKind::Int8QTy, quantizedNode->dims(),
                                   qParams[0].scale_, qParams[0].offset_);
    return F->createRescaleQuantized(quantizedNode->getName(), quantizedNode,
                                     outTy);
  }
  return quantizedNode;
}

Function *
quantizeFunction(const ExecutionEngine &EE,
                 llvm::ArrayRef<NodeQuantizationInfo> quantizationInfos,
                 Function *F, llvm::StringRef newFuncName) {
  std::string tmpName;
  if (newFuncName.empty()) {
    tmpName = std::string(F->getName()) + "_quantized";
    newFuncName = tmpName;
  }

  Function *G = F->clone(newFuncName);

  // Build a mapping between node name and TensorQuantizatonParams.
  std::unordered_map<std::string, TensorQuantizationParams> nodeToTQP;
  for (const auto &quantizationInfo : quantizationInfos) {
    nodeToTQP.emplace(quantizationInfo.nodeOutputName_,
                      quantizationInfo.tensorQuantizationParams_);
  }

  // For every unprocessed node in the graph we keep the invariant of having
  // all inputs to be float typed.
  auto nodeIt = G->getNodes().end();
  auto stopIt = G->getNodes().begin();
  do {
    --nodeIt;
    Node *node = &*nodeIt;

    // Make sure that all inputs are floats and int8 operation is suppored by
    // the backend. Not all backends support particular quantized operation and
    // also we should not quantize Index type inputs.
    if (canBeQuantized(node) &&
        EE.isOpSupported(node->getKind(), ElemKind::Int8QTy)) {
      // 1) Quantize all of the inputs based on the profiles.
      //    Quantize only floating point inputs.
      auto quantizedInputs = quantizeInputs(G, node, nodeToTQP);

      auto qParams = getQuantizationParameters(node, nodeToTQP);

      // 2) Quantize the node.
      Node *quantizedNode = quantizeNode(G, node, quantizedInputs, qParams);
      quantizedNode = postProcessQuantizedNode(G, quantizedNode, qParams);
      assert(quantizedNode != nullptr && "Node must be quantized");

      // 3) Dequantize all outputs of the node so that invariant is kept.
      unsigned qParamIndex = 0;
      for (unsigned outNum = 0, e = quantizedNode->getNumResults(); outNum < e;
           outNum++) {
        // Dequantize only quantized outputs.
        // In case output was not quantized we still need to relink the node.
        if (quantizedNode->getNthResult(outNum).getElementType() !=
            ElemKind::Int8QTy) {
          node->getNthResult(outNum).replaceAllUsesOfWith(
              quantizedNode->getNthResult(outNum));
          continue;
        }

        auto *dequantized = G->createDequantize(
            "dequantize", quantizedNode->getNthResult(outNum));

        // 4) Replace all usages of the floating point node output by the
        // dequantized node to maintain the invariant.
        node->getNthResult(outNum).replaceAllUsesOfWith(dequantized);

        // 5) Make sure that TQP is not lost after addition of intermediate
        // dequantized node.
        nodeToTQP[NodeQuantizationInfo::generateNodeOutputName(
            dequantized->getName())] = qParams[qParamIndex++];
      }
    }
  } while (nodeIt != stopIt);

  return G;
}

} // namespace quantization
} // namespace glow
