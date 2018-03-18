// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Quantization/Quantization.h"

#include "glow/ExecutionEngine/ExecutionEngine.h"

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

namespace quantization {

MultTransformF32ToI32 computeMultTransformParams(float scale, int bits) {
  int preShift = 0;
  int postShift = 0;
  int32_t m = 0;

  assert(scale > 0.0f && "scale is nonpositive");

  // We need to determine the mantissa in a floating-point-format-independent
  // way. To do this, upshift or downshift the scale as necessary until it is
  // equal to its mantissa -- i.e., until it is in [1.0, 2.0).
  while (scale >= 2.0f) {
    scale /= 2.0f;
    postShift--;
  }

  while (scale < 1.0f) {
    scale *= 2.0f;
    postShift++;
  }

  // Now scale is equal to its mantissa, and postShift is equal to the opposite
  // of the actual exponent of the original value of scale.

  int mBits = 1;

  // We want to ensure that mBits + bits == 31, as this helps to avoid the
  // problem of postShift being negative or zero.
  while (mBits + bits < 31) {
    scale *= 2.0f;
    mBits++;
    postShift++;
  }

  // At this point we have as much precision as possible, but it might not be
  // evenly balanced. The input variable bits is still equal to its original
  // value, and if that is greater than 16, then we may want to trade off some
  // input bits in favor of more precision in the mantissa -- if there is such
  // precision still to be acquired.
  float intPart;
  std::modf(scale, &intPart);
  while (bits > 16 && scale != intPart) {
    scale *= 2.0f;
    bits--;
    preShift++;
    std::modf(scale, &intPart);
  }

  m = (int32_t)intPart;

  if (postShift > 30) {
    preShift = 0;
    postShift = 1;
    m = 0;
  }

  return MultTransformF32ToI32(preShift, postShift, m);
}

std::vector<NodeQuantizationInfo>
generateNodeQuantizationInfos(const Function *F) {
  std::vector<NodeQuantizationInfo> quantizationInfos;

  for (auto *node : F->getNodes()) {
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

int8_t quantize(float input, const TensorQuantizationParams &TQP) {
  float result = input / TQP.scale_ + TQP.offset_;
  return quantization::clip<int32_t, int8_t>(round(result));
}

float dequantize(int8_t input, const TensorQuantizationParams &TQP) {
  return TQP.scale_ * (input - TQP.offset_);
}

/// Quantize all inputs for \p node and return back pointers to the newly
/// created qunatization nodes.
static llvm::SmallVector<Node *, 6>
quantizeInputs(Function *F, Node *node,
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
    auto QT = F->getParent()->uniqueType(ElemKind::Int8QTy, NV->dims(),
                                         TQP.scale_, TQP.offset_);

    Node *quantizeNode = F->createQuantize("quantize", NV, QT);
    quantizedInputs.push_back(quantizeNode);
  }

  return quantizedInputs;
}

void generateQuantizedGraph(
    const ExecutionEngine &EE, Function *F,
    llvm::ArrayRef<NodeQuantizationInfo> quantizationInfos) {
  if (F->getNodes().empty()) {
    return;
  }

  // Build a mapping between node name and TensorQuantizatonParams.
  std::unordered_map<std::string, TensorQuantizationParams> nodeToTQP;
  for (const auto &quantizationInfo : quantizationInfos) {
    nodeToTQP.emplace(quantizationInfo.nodeOutputName_,
                      quantizationInfo.tensorQuantizationParams_);
  }

  // For every unprocessed node in the graph we keep the invariant of having
  // all inputs to be float typed.
  auto nodeIt = F->getNodes().end();
  auto stopIt = F->getNodes().begin();
  do {
    --nodeIt;
    Node *node = *nodeIt;

    if (EE.isOpSupported(node->getKind(), ElemKind::Int8QTy)) {
      // 1) Quantize all of the inputs based on the profiles.
      llvm::SmallVector<Node *, 6> quantizedInputs =
          quantizeInputs(F, node, nodeToTQP);

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
        auto QT = F->getParent()->uniqueType(ElemKind::Int8QTy,
                                             FC->getResult()->dims(),
                                             TQP.scale_, TQP.offset_);
        quantizedNode =
            F->createFullyConnected(FC->getName(), quantizedInputs[0],
                                    quantizedInputs[1], quantizedInputs[2], QT);
        break;
      }

      case Kinded::Kind::ConvolutionNodeKind: {
        auto *CV = cast<ConvolutionNode>(node);
        assert(quantizedInputs.size() == 3 && "Invalid number of inputs");
        auto QT = F->getParent()->uniqueType(ElemKind::Int8QTy,
                                             CV->getResult()->dims(),
                                             TQP.scale_, TQP.offset_);
        quantizedNode =
            F->createConv(CV->getName(), quantizedInputs[0], quantizedInputs[1],
                          quantizedInputs[2], QT, CV->getDepth(),
                          CV->getKernel(), CV->getStride(), CV->getPad());
        break;
      }
      case Kinded::Kind::ReluNodeKind: {
        auto *R = cast<ReluNode>(node);
        assert(quantizedInputs.size() == 1 && "Invalid number of inputs");

        quantizedNode = F->createRELU(R->getName(), quantizedInputs[0]);
        break;
      }
      case Kinded::Kind::TransposeNodeKind: {
        auto *T = cast<TransposeNode>(node);
        assert(quantizedInputs.size() == 1 && "Invalid number of inputs");
        quantizedNode = F->createTranspose(T->getName(), quantizedInputs[0],
                                           T->getShuffle());
        break;
      }
      case Kinded::Kind::ReshapeNodeKind: {
        auto *R = cast<ReshapeNode>(node);
        assert(quantizedInputs.size() == 1 && "Invalid number of inputs");
        quantizedNode =
            F->createReshape(R->getName(), quantizedInputs[0], R->getDims());
        break;
      }
      case Kinded::Kind::PoolMaxNodeKind: {
        auto *P = cast<PoolMaxNode>(node);
        assert(quantizedInputs.size() == 1 && "Invalid number of inputs");

        quantizedNode =
            F->createPoolMax(node->getName(), quantizedInputs[0],
                             P->getKernel(), P->getStride(), P->getPad());
        break;
      }
      case Kinded::Kind::PoolAvgNodeKind: {
        auto *P = cast<PoolAvgNode>(node);
        assert(quantizedInputs.size() == 1 && "Invalid number of inputs");

        quantizedNode =
            F->createPoolAvg(node->getName(), quantizedInputs[0],
                             P->getKernel(), P->getStride(), P->getPad());
        break;
      }
#define CASE_QUANTIZE_NODE(NODE_NAME_)                                         \
  case Kinded::Kind::NODE_NAME_##NodeKind: {                                   \
    auto *AN = cast<NODE_NAME_##Node>(node);                                   \
    assert(quantizedInputs.size() == 2 && "Invalid number of inputs");         \
    quantizedNode = F->create##NODE_NAME_(AN->getName(), quantizedInputs[0],   \
                                          quantizedInputs[1]);                 \
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

        // Concat just moves tensors around, make sure that all tensors have the
        // same {S,O} params.
        for (size_t qi = 0, e = quantizedInputs.size(); qi < e; qi++) {
          auto argOutTy = F->getParent()->uniqueType(
              ElemKind::Int8QTy, quantizedInputs[qi]->dims(), TQP.scale_,
              TQP.offset_);

          quantizedInputs[qi] = F->createRescaleQuantized(
              quantizedInputs[qi]->getName(), quantizedInputs[qi], argOutTy);
        }

        auto outTy = F->getParent()->uniqueType(
            ElemKind::Int8QTy, C->getResult()->dims(), TQP.scale_, TQP.offset_);
        quantizedNode = F->createConcat(node->getName(), quantizedInputs,
                                        C->getDim(), outTy);
        break;
      }

      default:
        llvm_unreachable("The node type is not supported for quantization");
      }

      // Some of the quantized nodes need additional post processing.
      if (node->getKind() == Kinded::Kind::ReluNodeKind) {
        // Relu does not change {S,O} of the output, it uses the same {S,O} as
        // the input. Make sure that rescale is applied to comply with the taken
        // profile from RELU.
        auto outTy = F->getParent()->uniqueType(
            ElemKind::Int8QTy, quantizedNode->dims(), TQP.scale_, TQP.offset_);
        quantizedNode = F->createRescaleQuantized(quantizedNode->getName(),
                                                  quantizedNode, outTy);
      } else if (node->getKind() == Kinded::Kind::PoolMaxNodeKind ||
                 node->getKind() == Kinded::Kind::PoolAvgNodeKind) {
        auto outTy = F->getParent()->uniqueType(
            ElemKind::Int8QTy, quantizedNode->dims(), TQP.scale_, TQP.offset_);
        quantizedNode = F->createRescaleQuantized(quantizedNode->getName(),
                                                  quantizedNode, outTy);
      }

      // 3) Dequantize output of the node so that invariant is kept.
      assert(quantizedNode != nullptr && "Node must be quantized");
      auto *dequantized = F->createDequantize("dequantize", quantizedNode);

      // Note, there is an assumption that converted node has only single
      // output. 4) Replace all usages of old floating point node by the
      // dequantized node to maintain the invariant.
      node->getNthResult(0).replaceAllUsesOfWith(dequantized);

      // 5) Make sure that TQP is not lost after addition of intermediate
      // dequantized node.
      nodeToTQP[NodeQuantizationInfo::generateNodeOutputName(
          dequantized->getName())] = TQP;
    }
  } while (nodeIt != stopIt);
}

} // namespace quantization

} // namespace glow
