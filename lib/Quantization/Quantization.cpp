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

#include "glow/Converter/FunctionConverter.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"

#include <cmath>
#include <unordered_set>
#include <vector>

using llvm::cast;

namespace glow {
namespace quantization {

/// This class produces a quantized function based on a provided profile.
class FunctionQuantizer : public FunctionConverter {
protected:
  /// \see FunctionConverter::getTargetTypeForOutput.
  /// Get the quantized type for \p out if any.
  TypeRef getTargetTypeForOutput(const NodeValue &out) const override {
    if (out.getElementType() != ElemKind::FloatTy) {
      return nullptr;
    }

    const std::string nodeOutputName =
        NodeQuantizationInfo::generateNodeOutputName(out.getNode()->getName(),
                                                     out.getResNo());
    auto outTQPIt = nodeToTQP_.find(nodeOutputName);
    assert(outTQPIt != nodeToTQP_.end() &&
           "Missing quantization params for a node");

    const TensorQuantizationParams &TQP = outTQPIt->second;
    return mod_.uniqueType(ElemKind::Int8QTy, out.dims(), TQP.scale,
                           TQP.offset);
  }

  /// \see FunctionConverter::getTargetTypeForInput.
  /// Get the quantized type for the \p idx-th argument of \p use, if any.
  TypeRef getTargetTypeForInput(const Node &use, unsigned idx) const override {
    NodeValue val = use.getNthInput(idx);

    // Do not quantize non floating point type, e.g., Index type.
    if (val.getElementType() != ElemKind::FloatTy) {
      return nullptr;
    }

    std::string nodeOutputName = NodeQuantizationInfo::generateNodeOutputName(
        val.getNode()->getName(), val.getResNo());
    auto valTQPIt = nodeToTQP_.find(nodeOutputName);
    assert(valTQPIt != nodeToTQP_.end() &&
           "Missing quantization params for a node");

    const TensorQuantizationParams &TQP = valTQPIt->second;
    return mod_.uniqueType(ElemKind::Int8QTy, val.dims(), TQP.scale,
                           TQP.offset);
  }

  /// \see FunctionConverter::canConvert.
  /// Only convert nodes that use floating point types and that
  /// weren't specifically marked as to-ignore with doNotQuantizeKinds_.
  bool canConvert(const Node &node) const override {
    auto kind = node.getKind();
    if (!EE_.isOpSupported(kind, ElemKind::Int8QTy)) {
      return false;
    }

    if (doNotQuantizeKinds_.count(kind)) {
      return false;
    }

    // Handle special cases like Gather node where some of the inputs do not
    // need to be quantized while some do.
    switch (node.getKind()) {
    case Kinded::Kind::GatherNodeKind: {
      auto *gather = cast<GatherNode>(&node);
      return gather->getData().getElementType() == ElemKind::FloatTy;
    }
    case Kinded::Kind::SoftMaxNodeKind: {
      auto *SMN = cast<SoftMaxNode>(&node);
      return SMN->getInput().getElementType() == ElemKind::FloatTy;
    }
    default:
      // Let the general procedure handle this node kind.
      break;
    }

    // Make sure that all inputs are floats.
    for (unsigned i = 0, e = node.getNumInputs(); i < e; ++i) {
      if (node.getNthInput(i).getElementType() != ElemKind::FloatTy) {
        return false;
      }
    }

    return true;
  }

  /// Create either a QuantizeNode or DequantizeNode base on the \p destTy
  /// and the type of \p val.
  /// Basically, if \p val's type is floating point, this creates a
  /// QuantizeNode of \p val.
  /// If \p val's type is a quantized type, this creates a
  /// DequantizeNode of \p val.
  ///
  /// \pre One of t\p val's type and \p destTy must be FloatTy and
  ///      the other must be a quantized type.
  Node *createConversion(NodeValue &val, TypeRef destTy) override {
    if (destTy->isQuantizedType()) {
      assert(destTy->getElementType() == ElemKind::Int8QTy && "");
      return function_.createQuantize("quantize", val, destTy);
    }
    assert(destTy->getElementType() == ElemKind::FloatTy && "");
    return function_.createDequantize("quantize", val);
  }

  /// \see FunctionConverter::morphNode.
  /// This method does the final adjustment to the output types
  /// when the profile and the IR constraints do not agree.
  /// E.g., the profile of LocalResponseNormalizationNode may
  /// give a range that is different from the range its input.
  /// However, the IR expects that both the input and output of
  /// this node have the same type.
  Node &morphNode(Node &node) override {
    // FIXME: Right now, the TensorQuantizationParams only tracks one
    // type per NodeValue, whereas we would need one type for the output and
    // one for each user of that value. E.g., for
    // val = node
    // = use1 val
    // = use2 val
    //

    // We would want to track independently the types for the result of node,
    // the use of val in use1, and the use of val in use2, in respectively
    // outTy, inTy1, and inTy2, so that we can generate: outTy = node inTy1 =
    // cast outTy = use1 inTy1 inTy2 = cast outTy = use2 inTy2
    //
    //
    // But instead what we track only one type like this:
    // outTy = node
    // = use1 outTy
    // = use2 outTy
    //
    // However, sometimes outTy is not suitable for the input (we fix those in
    // postProcessing) and sometimes, the outTy is not suitable for the output
    // itself!
    // What this means basically is outTy encodes the inTy constraints whereas
    // they may disagree.
    // The following switch fixes outTy for the few nodes where inTy and outTy
    // can disagree.
    // This happens for cases where for instance, the quantized parameters for
    // the output have a different scale than the input, whereas the operation
    // itself doesn't allow that.
    //
    // E.g., the constraints for `outTy = op inTy` are `outTy == inTy`, but the
    // quantization profiling gave different types to outTy and inTy.
    switch (node.getKind()) {
    case Kinded::Kind::LocalResponseNormalizationNodeKind:
    case Kinded::Kind::SigmoidNodeKind:
    case Kinded::Kind::TanhNodeKind:
    case Kinded::Kind::TopKNodeKind:
    case Kinded::Kind::GatherNodeKind:
    case Kinded::Kind::MaxPoolNodeKind:
    case Kinded::Kind::AvgPoolNodeKind: {
      // The constraints on the IR says that the input type must
      // be the same as the output type.
      TypeRef inTy = node.getNthInput(0).getType();
      TypeRef fixedTy =
          mod_.uniqueType(ElemKind::Int8QTy, node.getNthResult(0).dims(),
                          inTy->getScale(), inTy->getOffset());

      node.setType(0, fixedTy);
      return node;
    }
    default:
      return node;
    }
  }

  /// Update the nodeToTQP_ for the added conversions for \p node.
  void postProcessing(Node &node) override {
    Node *quantizedNode = &node;
    switch (node.getKind()) {
    default:
      break;

    case Kinded::Kind::ConcatNodeKind: {
      auto *concat = cast<ConcatNode>(&node);
      TypeRef outputTy = concat->getType(0);
      assert(outputTy->isQuantizedType() && "Node hasn't been quantized yet?!");

      // Concat just moves tensors around, make sure that all tensors have the
      // same {S,O} params.
      unsigned idx = 0;
      for (NodeValue input : concat->getInputs()) {
        auto argOutTy =
            mod_.uniqueType(ElemKind::Int8QTy, input.dims(),
                            outputTy->getScale(), outputTy->getOffset());
        auto *rescale = function_.createRescaleQuantized(
            input.getNode()->getName(), input, argOutTy);
        concat->setNthInput(idx++, rescale);
      }

      break;
    }
    case Kinded::Kind::MaxPoolNodeKind:
    case Kinded::Kind::AvgPoolNodeKind:
    case Kinded::Kind::GatherNodeKind: {
      // These nodes do not change {S,O} of the output, they use the same
      // {S,O} as the input. Make sure that rescale is applied to comply with
      // the taken profile from the node.
      TypeRef outputTy = node.getType(0);
      assert(outputTy->isQuantizedType() && "Node hasn't been quantized yet?!");
      auto outTy = mod_.uniqueType(ElemKind::Int8QTy, outputTy->dims(),
                                   outputTy->getScale(), outputTy->getOffset());
      NodeValue val = node.getNthResult(0);
      // "node" should have only one use, the dequantize node.
      // Update this use.
      assert(
          val.hasOneUse() &&
          llvm::dyn_cast<DequantizeNode>((*val.getUsers().begin()).getUser()) &&
          "This node should only be used by the dequantize node");
      auto *dequantize =
          llvm::dyn_cast<DequantizeNode>((*val.getUsers().begin()).getUser());
      auto *rescale =
          function_.createRescaleQuantized(node.getName(), val, outTy);
      quantizedNode = rescale;
      dequantize->setNthInput(0, rescale);
      break;
    }
    }

    // Make sure that nodeToTQP_ is not lost after the addition of intermediate
    // dequantized nodes.
    for (unsigned outNum = 0, e = quantizedNode->getNumResults(); outNum != e;
         ++outNum) {
      NodeValue val = quantizedNode->getNthResult(outNum);
      if (!val.getType()->isQuantizedType()) {
        continue;
      }
      assert(
          val.hasOneUse() &&
          llvm::dyn_cast<DequantizeNode>((*val.getUsers().begin()).getUser()) &&
          "This node should only be used by the dequantize node");
      auto *dequantize =
          llvm::dyn_cast<DequantizeNode>((*val.getUsers().begin()).getUser());
      TypeRef outTy = val.getType();
      nodeToTQP_[NodeQuantizationInfo::generateNodeOutputName(
          dequantize->getName(), outNum)] = {outTy->getScale(),
                                             outTy->getOffset()};
    }
  }

private:
  /// Shortcut to the module of function_.
  Module &mod_;
  /// Execution engine used to check is a quantized operator is
  /// supported.
  const ExecutionEngine &EE_;
  /// Set of node kinds that should not be quantized.
  const KindSet &doNotQuantizeKinds_;
  /// Map the (name of a node, idx) to its quantization parameters.
  std::unordered_map<std::string, TensorQuantizationParams> nodeToTQP_;

public:
  /// Creates a function quantizer for \p F using the quantization
  /// parameters defined by \p quantizationInfos.
  /// \p EE and \p doNotQuantizeKinds are used to check which
  /// nodes shouldn't be converted.
  FunctionQuantizer(Function &F, const ExecutionEngine &EE,
                    llvm::ArrayRef<NodeQuantizationInfo> quantizationInfos,
                    const KindSet &doNotQuantizeKinds)
      : FunctionConverter(F), mod_(*F.getParent()), EE_(EE),
        doNotQuantizeKinds_(doNotQuantizeKinds) {
    // Build a mapping between node name and TensorQuantizatonParams.
    for (const auto &quantizationInfo : quantizationInfos) {
      nodeToTQP_.emplace(quantizationInfo.nodeOutputName_,
                         quantizationInfo.tensorQuantizationParams_);
    }
  }
};

std::vector<NodeQuantizationInfo>
generateNodeQuantizationInfos(Context &ctx, const Function *F, Schema schema) {
  std::vector<NodeQuantizationInfo> quantizationInfos;

  for (auto &node : F->getNodes()) {
    auto *QPN = llvm::dyn_cast<QuantizationProfileNode>(&node);

    if (QPN) {
      auto CI =
          ctx.get(QPN->getComputationInfoPlaceholder())->getHandle<float>();
      auto histogram =
          ctx.get(QPN->getHistogramPlaceholder())->getHandle<float>();
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

Function *
quantizeFunction(const ExecutionEngine &EE,
                 llvm::ArrayRef<NodeQuantizationInfo> quantizationInfos,
                 Function *F, llvm::StringRef newFuncName,
                 const KindSet &doNotQuantizeKinds) {
  std::string tmpName;
  if (newFuncName.empty()) {
    tmpName = std::string(F->getName()) + "_quantized";
    newFuncName = tmpName;
  }

  Function *G = F->clone(newFuncName);

  FunctionQuantizer quantizer(*G, EE, quantizationInfos, doNotQuantizeKinds);
  quantizer.convert();
  quantizer.optimizeConversions();

  return G;
}

} // namespace quantization
} // namespace glow
