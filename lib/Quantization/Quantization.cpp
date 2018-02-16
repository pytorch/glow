// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Quantization/Quantization.h"

#include <cmath>
#include <unordered_set>
#include <vector>

using llvm::cast;

namespace glow {

/// Calculate TensorQuantizationParams based on the clipped min and max float
/// values.
static TensorQuantizationParams chooseQuantizationParams(float min, float max) {
  assert(min <= max && "min must not be bigger than max");

  // Given 8 bit precision.
  const int32_t qmin = -128;
  const int32_t qmax = 127;

  double scale =
      (std::max(max, 0.f) - std::min(min, 0.f)) / ((double)qmax - qmin);

  // Dequantization uses the following formula scale * (X - offset), so
  // scale should not be equal to zero.
  // If scale is 0, we arbitrary adjust the scale to 0.1.
  if (scale == 0)
    scale = 0.1;

  assert(scale > 0 && "Scale must be non negative");

  // We extend the [min, max] interval to ensure that it contains 0.
  // Otherwise, we would not meet the requirement that 0 be an exactly
  // representable value.
  min = std::min(min, 0.f);
  max = std::max(max, 0.f);

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

  // For symmetric quantization (min == -max), we force zero_point to 128
  // to model signed integer (FIXME: this is a workaround that gemmlowp
  // doesn't support signed int AFAIK. Once we have an (efficient) gemm for
  // signed as well, we can just use signed int with zero_point = 0
  if (min == -max) {
    initialZeroPoint = (qmin + qmax) / 2 + 1;
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

std::vector<NodeQuantizationInfo>
generateNodeQuantizationInfos(const Function &G) {
  std::vector<NodeQuantizationInfo> quantizationInfos;

  for (auto *node : G.getNodes()) {
    auto *QPN = llvm::dyn_cast<QuantizationProfileNode>(node);

    if (QPN) {
      auto CI = QPN->getComputationInfoVar()->getHandle<float>();
      auto histogram = QPN->getHistogramVar()->getHandle<float>();
      float min = CI.raw(0);
      float max = CI.raw(1);

      std::string fullOutputName = NodeQuantizationInfo::generateNodeOutputName(
          QPN->getProfiledNodeName());

      // TODO: Ideally tensor quantization params should be calculated
      // based on the histogram distribution. Use simplistic approach for now.
      (void)histogram;
      TensorQuantizationParams TQP = chooseQuantizationParams(min, max);

      quantizationInfos.emplace_back(fullOutputName, TQP);
    }
  }

  return quantizationInfos;
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
  // as it's below one, and it's always legal to shift at least 16 bits,
  // because this won't overflow the calculation.
  while (scale < 0.5 || postShift < 15) {
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

int8_t quantize(float input, const TensorQuantizationParams &TQP) {
  float result = input / TQP.scale_ + TQP.offset_;
  return QuantizationTransform32To8::clip(round(result));
}

float dequantize(int8_t input, const TensorQuantizationParams &TQP) {
  return TQP.scale_ * (input - TQP.offset_);
}

/// \returns true if \p node should be quantized.
static bool shouldQuantize(const Node *node) {
  return llvm::isa<FullyConnectedNode>(node) ||
         llvm::isa<ConvolutionNode>(node);
}

/// Quantize all inputs for \p node and return back pointers to the newly
/// created qunatization nodes.
static llvm::SmallVector<Node *, 6>
quantizeInputs(Function &G, Node *node,
               const std::unordered_map<std::string, TensorQuantizationParams>
                   &nodeToTQP) {
  llvm::SmallVector<Node *, 6> quantizedInputs;

  for (unsigned i = 0, e = node->getNumInputs(); i < e; ++i) {
    NodeValue &NV = node->getNthInput(i);
    std::string nodeOutputName = NodeQuantizationInfo::generateNodeOutputName(
        NV->getName(), NV.getResNo());
    assert(nodeToTQP.find(nodeOutputName) != nodeToTQP.end() &&
           "Missing quantization params for a node");

    const TensorQuantizationParams &TQP =
        nodeToTQP.find(nodeOutputName)->second;
    auto QT = G.getParent().uniqueType(ElemKind::Int8QTy, NV->dims(),
                                       TQP.scale_, TQP.offset_);

    Node *quantizeNode = G.createQuantize("quantize", NV, QT);
    quantizedInputs.push_back(quantizeNode);
  }

  return quantizedInputs;
}

void generateQuantizedGraph(
    Function &G, llvm::ArrayRef<NodeQuantizationInfo> quantizationInfos) {
  // Build a mapping between node name and TensorQuantizatonParams.
  std::unordered_map<std::string, TensorQuantizationParams> nodeToTQP;
  for (const auto &quantizationInfo : quantizationInfos) {
    nodeToTQP.emplace(quantizationInfo.nodeOutputName_,
                      quantizationInfo.tensorQuantizationParams_);
  }

  std::unordered_set<Node *> addedNodes;
  // For every unprocessed node in the graph we keep the invariant of having
  // all inputs to be float typed.
  for (auto *node : G.getNodes()) {
    if (!shouldQuantize(node) || addedNodes.count(node)) {
      continue;
    }

    // 1) Quantize all of the inputs based on the profiles.
    llvm::SmallVector<Node *, 6> quantizedInputs =
        quantizeInputs(G, node, nodeToTQP);

    // 2) Quantize node itself.
    const std::string nodeOutputName =
        NodeQuantizationInfo::generateNodeOutputName(node->getName());
    assert(nodeToTQP.find(nodeOutputName) != nodeToTQP.end() &&
           "Missing quantization params for a node");
    const TensorQuantizationParams &TQP = nodeToTQP[nodeOutputName];

    Node *quantizedNode{};

    switch (node->getKind()) {
    case Kinded::Kind::FullyConnectedNodeKind: {
      auto *FC = cast<FullyConnectedNode>(node);
      assert(quantizedInputs.size() == 3 && "Invalid number of inputs");
      auto QT = G.getParent().uniqueType(
          ElemKind::Int8QTy, FC->getResult()->dims(), TQP.scale_, TQP.offset_);
      quantizedNode =
          G.createFullyConnected(FC->getName(), quantizedInputs[0],
                                 quantizedInputs[1], quantizedInputs[2], QT);
      break;
    }

    case Kinded::Kind::ConvolutionNodeKind: {
      auto *CV = cast<ConvolutionNode>(node);
      assert(quantizedInputs.size() == 3 && "Invalid number of inputs");
      auto QT = G.getParent().uniqueType(
          ElemKind::Int8QTy, CV->getResult()->dims(), TQP.scale_, TQP.offset_);
      quantizedNode =
          G.createConv(CV->getName(), quantizedInputs[0], quantizedInputs[1],
                       quantizedInputs[2], QT, CV->getDepth(), CV->getKernel(),
                       CV->getStride(), CV->getPad());
      break;
    }

    default:
      llvm_unreachable("The node type is not supported for quantization");
    }

    // Just insert newly quantized node into added set. Quantize and Dequantize
    // nodes are not eligible for quantization anyway.
    addedNodes.insert(quantizedNode);

    // 3) Dequantize output of the node so that invariant is kept.
    assert(quantizedNode != nullptr && "Node must be quantized");
    auto *dequantized = G.createDequantize("dequantize", quantizedNode);

    // Note, there is an assumption that converted node has only single output.
    // 4) Replace all usages of old floating point node by the dequantized
    // node to maintain the invariant.
    node->getNthResult(0).replaceAllUsesOfWith(dequantized);

    // 5) Make sure that TQP is not lost after addition of intermediate
    // dequantized node.
    nodeToTQP[NodeQuantizationInfo::generateNodeOutputName(
        dequantized->getName())] = TQP;
  }
}

} // namespace glow
