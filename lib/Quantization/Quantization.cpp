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

std::vector<NodeQuantizationInfo>
generateNodeQuantizationInfos(const Function *F, Schema schema) {
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
      TensorQuantizationParams TQP = chooseQuantizationParams(min, max, schema);

      quantizationInfos.emplace_back(fullOutputName, TQP);
    }
  }

  return quantizationInfos;
}

/// Quantize all inputs for \p node and return back pointers to the newly
/// created qunatization nodes.
static llvm::SmallVector<NodeValue, 6>
quantizeInputs(Function *F, Node *node,
               const std::unordered_map<std::string, TensorQuantizationParams>
                   &nodeToTQP) {
  llvm::SmallVector<NodeValue, 6> quantizedInputs;

  for (unsigned i = 0, e = node->getNumInputs(); i < e; ++i) {
    auto NV = node->getNthInput(i);

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
    auto QT = F->getParent()->uniqueType(ElemKind::Int8QTy, NV.dims(),
                                         TQP.scale, TQP.offset);

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
        F->getParent()->uniqueType(ElemKind::Int8QTy, FC->getResult().dims(),
                                   qParams[0].scale, qParams[0].offset);
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
        F->getParent()->uniqueType(ElemKind::Int8QTy, CV->getResult().dims(),
                                   qParams[0].scale, qParams[0].offset);
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

    auto QT =
        F->getParent()->uniqueType(ElemKind::Int8QTy, S->getResult().dims(),
                                   quantizedInputs[0].getType()->getScale(),
                                   quantizedInputs[0].getType()->getOffset());

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
                         P->getStride(), P->getPads());
    break;
  }
  case Kinded::Kind::PoolAvgNodeKind: {
    auto *P = cast<PoolAvgNode>(node);
    assert(quantizedInputs.size() == 1 && "Invalid number of inputs");
    assert(qParams.size() == 1 && "Invalid number of quantized outputs");

    quantizedNode =
        F->createPoolAvg(node->getName(), quantizedInputs[0], P->getKernel(),
                         P->getStride(), P->getPads());
    break;
  }
#define CASE_QUANTIZE_NODE(NODE_NAME_)                                         \
  case Kinded::Kind::NODE_NAME_##NodeKind: {                                   \
    auto *AN = cast<NODE_NAME_##Node>(node);                                   \
    assert(quantizedInputs.size() == 2 && "Invalid number of inputs");         \
    assert(qParams.size() == 1 && "Invalid number of quantized outputs");      \
    auto outTy =                                                               \
        F->getParent()->uniqueType(ElemKind::Int8QTy, AN->getResult().dims(),  \
                                   qParams[0].scale, qParams[0].offset);     \
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
          ElemKind::Int8QTy, quantizedInputs[qi].dims(), qParams[0].scale,
          qParams[0].offset);

      quantizedInputs[qi] = F->createRescaleQuantized(
          quantizedInputs[qi]->getName(), quantizedInputs[qi], argOutTy);
    }

    auto outTy =
        F->getParent()->uniqueType(ElemKind::Int8QTy, C->getResult().dims(),
                                   qParams[0].scale, qParams[0].offset);
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
    auto *TN = cast<TanhNode>(node);
    assert(quantizedInputs.size() == 1 && "Invalid number of inputs");
    assert(qParams.size() == 1 && "Invalid number of quantized outputs");

    // Note: This should either be lowered into an IntLookupTable, or
    // implemented via a backend-specific Node/Inst.
    quantizedNode = F->createTanh(TN->getName(), quantizedInputs[0]);
    break;
  }
  case Kinded::Kind::SigmoidNodeKind: {
    auto *SN = cast<SigmoidNode>(node);
    assert(quantizedInputs.size() == 1 && "Invalid number of inputs");
    assert(qParams.size() == 1 && "Invalid number of quantized outputs");

    // Note: This should either be lowered into an IntLookupTable, or
    // implemented via a backend-specific Node/Inst.
    quantizedNode = F->createSigmoid(SN->getName(), quantizedInputs[0]);
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
        F->getParent()->uniqueType(ElemKind::Int8QTy, quantizedNode->dims(0),
                                   qParams[0].scale, qParams[0].offset);
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

    // Make sure that all inputs are floats and int8 operation is supported by
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
