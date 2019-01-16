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

namespace {

using namespace glow;
using namespace glow::quantization;

/// This class produces a quantized function based on a provided profile.
class FunctionQuantizer : public FunctionConverter {
protected:
  /// Get the type that \p out should have at the end of the conversion
  /// process regardless of what is its current type.
  /// This is similar to ::getTargetTypeForOutput except that \p out
  /// doesn't need to be a floating point type for this method to
  /// return the target type.
  /// The reason we need this method is because we may morph the type
  /// of \p out to match some IR constraints, but the final type still
  /// needs to be known to insert rescale nodes.
  TypeRef getTargetTypeForOutputImpl(const NodeValue &out) const {
    const std::string nodeOutputName =
        NodeQuantizationInfo::generateNodeOutputName(out.getNode()->getName(),
                                                     out.getResNo());
    auto outTQPIt = nodeToTQP_.find(nodeOutputName);
    assert(outTQPIt != nodeToTQP_.end() &&
           "Missing quantization params for a node");

    const TensorQuantizationParams &TQP = outTQPIt->second;
    return mod_.uniqueType(quantizationPrecision_, out.dims(), TQP.scale,
                           TQP.offset);
  }

  /// \see FunctionConverter::getTargetTypeForOutput.
  /// Get the quantized type for \p out if any.
  TypeRef getTargetTypeForOutput(const NodeValue &out) const override {
    if (out.getElementType() != ElemKind::FloatTy) {
      return nullptr;
    }
    return getTargetTypeForOutputImpl(out);
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
    // For bias of a conv op, it is quantized to int32.
    if (use.getKind() == glow::Kinded::Kind::ConvolutionNodeKind &&
        idx == ConvolutionNode::BiasIdx) {
      // For bias of a conv op, it is quantized to int32. Also, we should make
      // sure its scale should be (scale of input) * (scale of weights).
      auto convN = llvm::dyn_cast<ConvolutionNode>(&use);
      NodeValue input = convN->getInput();
      NodeValue weights = convN->getFilter();
      float scaleInput = input.getType()->getScale();
      float scaleWeights = weights.getType()->getScale();
      return mod_.uniqueType(ElemKind::Int32QTy, val.dims(),
                             scaleInput * scaleWeights, TQP.offset);
    } else if (use.getKind() == glow::Kinded::Kind::FullyConnectedNodeKind &&
               idx == FullyConnectedNode::BiasIdx) {
      auto fcN = llvm::dyn_cast<FullyConnectedNode>(&use);
      NodeValue input = fcN->getInput();
      NodeValue weights = fcN->getWeights();
      float scaleInput = input.getType()->getScale();
      float scaleWeights = weights.getType()->getScale();
      return mod_.uniqueType(ElemKind::Int32QTy, val.dims(),
                             scaleInput * scaleWeights, TQP.offset);
    } else {
      return mod_.uniqueType(quantizationPrecision_, val.dims(), TQP.scale,
                             TQP.offset);
    }
  }

  /// \see FunctionConverter::canConvert.
  /// Only convert nodes that use floating point types and that
  /// weren't specifically marked as to-ignore with doNotQuantizeKinds_.
  bool canConvert(const Node &node) const override {
    auto kind = node.getKind();
    if (!EE_.isOpSupported(kind, quantizationPrecision_)) {
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
    case Kinded::Kind::SparseLengthsWeightedSumNodeKind: {
      auto *SLS = cast<SparseLengthsWeightedSumNode>(&node);
      return (SLS->getData().getElementType() == ElemKind::FloatTy) &&
             (SLS->getWeights().getElementType() == ElemKind::FloatTy);
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

  /// Create either a QuantizeNode or DequantizeNode in \p function based on the
  /// \p destTy and the type of \p val.
  /// Basically, if \p val's type is floating point, this creates a
  /// QuantizeNode of \p val.
  /// If \p val's type is a quantized type, this creates a
  /// DequantizeNode of \p val.
  ///
  /// \pre One of t\p val's type and \p destTy must be FloatTy and
  ///      the other must be a quantized type.
  Node *createConversion(Function &function, NodeValue &val,
                         TypeRef destTy) override {
    if (destTy->isQuantizedType()) {
      return function_.createQuantize("quantize", val, destTy);
    }
    assert(destTy->getElementType() == ElemKind::FloatTy && "");
    return function.createDequantize("quantize", val);
  }

  /// All IRConstraint cases below assume that the input and output index that
  /// they are looking for the type is at idx 0. We statically assert that here
  /// along with the case.
  static constexpr unsigned IRConstraintInputIdx = 0;
  static constexpr unsigned IRConstraintResultIdx = 0;
#define IR_CONSTRAINT_CASE(NODE_NAME_, INPUT_NAME_, OUTPUT_NAME_)              \
  static_assert(                                                               \
      (NODE_NAME_##Node::INPUT_NAME_##Idx == IRConstraintInputIdx &&           \
       NODE_NAME_##Node::OUTPUT_NAME_##Idx == IRConstraintResultIdx),          \
      #NODE_NAME_ "Node format is unexpected.");                               \
  case Kinded::Kind::NODE_NAME_##NodeKind

  /// Macro to be put in a switch for all the nodes that have a constraint
  /// where the input and output type must be equals.
  /// Note: The last case of the macro doesn't have ':' so we can put it
  /// where the macro is inserted to keep the nice code formatting.
  // clang-format off
#define casesForNodesWithIRConstraint                                          \
  IR_CONSTRAINT_CASE(LocalResponseNormalization, Input, Result):               \
  IR_CONSTRAINT_CASE(Slice, Input, Result):                                    \
  IR_CONSTRAINT_CASE(Reshape, Input, Result):                                  \
  IR_CONSTRAINT_CASE(TopK, Input, Values):                                     \
  IR_CONSTRAINT_CASE(Gather, Data, Result):                                    \
  IR_CONSTRAINT_CASE(MaxPool, Input, Result)
  // clang-format on

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
      // Those cases need to be in sync with postProcessing, so we generate them
      // using macros.
    casesForNodesWithIRConstraint : {
      // The constraints on the IR says that the input type must
      // be the same as the output type.
      TypeRef inTy = node.getNthInput(IRConstraintInputIdx).getType();
      TypeRef fixedTy =
          mod_.uniqueType(quantizationPrecision_,
                          node.getNthResult(IRConstraintResultIdx).dims(),
                          inTy->getScale(), inTy->getOffset());

      node.setType(IRConstraintResultIdx, fixedTy);
      assert(!lastMorphedNodeWithTypeChanges &&
             "Missed one node to rescale in postprocessing");
      lastMorphedNodeWithTypeChanges = &node;
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
      TypeRef outputTy = concat->getType(ConcatNode::ResultIdx);
      assert(outputTy->isQuantizedType() && "Node hasn't been quantized yet?!");

      // Concat just moves tensors around, make sure that all tensors have the
      // same {S,O} params.
      unsigned idx = 0;
      for (NodeValue input : concat->getInputs()) {
        auto argOutTy =
            mod_.uniqueType(quantizationPrecision_, input.dims(),
                            outputTy->getScale(), outputTy->getOffset());
        auto *rescale = function_.createRescaleQuantized(
            input.getNode()->getName(), input, argOutTy);
        concat->setNthInput(idx++, rescale);
      }

      break;
    }
    casesForNodesWithIRConstraint : {
      // Check that the main loop hands us the node in the order we expect:
      // morph then postprocessing.
      // If the assert breaks, that means that morphNode and postprocessing
      // are out of sync (we probably miss a case in the switch).
      assert(lastMorphedNodeWithTypeChanges == &node &&
             "Mismatching last node changed");
      lastMorphedNodeWithTypeChanges = nullptr;
      // These nodes do not change {S,O} of the output, they use the same
      // {S,O} as the input. Make sure that rescale is applied to comply with
      // the taken profile from the node.
      TypeRef outputTy =
          getTargetTypeForOutputImpl(NodeValue(&node, IRConstraintResultIdx));
      assert(outputTy->isQuantizedType() && "Node hasn't been quantized yet?!");
      auto outTy = mod_.uniqueType(quantizationPrecision_, outputTy->dims(),
                                   outputTy->getScale(), outputTy->getOffset());
      NodeValue val = node.getNthResult(IRConstraintResultIdx);
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
      dequantize->setNthInput(DequantizeNode::InputIdx, rescale);
      break;
    }
    }

    assert(!lastMorphedNodeWithTypeChanges && "Type not fixed");
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

  void convertTensor(Tensor &tensor, TypeRef destTy) override {
    assert(tensor.getElementType() == ElemKind::FloatTy &&
           (destTy->getElementType() == ElemKind::Int8QTy ||
            destTy->getElementType() == ElemKind::Int16QTy) &&
           "Dequantization not implemented");

    tensor = quantizeTensor(tensor, {destTy->getScale(), destTy->getOffset()},
                            destTy->getElementType());
  }

private:
  /// Shortcut to the module of function_.
  Module &mod_;
  /// Execution engine used to check is a quantized operator is
  /// supported.
  const ExecutionEngine &EE_;
  /// Quantization precision.
  const ElemKind quantizationPrecision_;
  /// Set of node kinds that should not be quantized.
  const KindSet &doNotQuantizeKinds_;
  /// Map the (name of a node, idx) to its quantization parameters.
  std::unordered_map<std::string, TensorQuantizationParams> nodeToTQP_;
  /// For debug, keep track of the last node that we changed because of IR
  /// constraints.
  Node *lastMorphedNodeWithTypeChanges;

public:
  /// Creates a function quantizer for \p F using the quantization
  /// parameters defined by \p quantizationInfos and target quantization
  /// precision defined by \p quantizationPrecision.
  /// \p EE and \p doNotQuantizeKinds are used to check which
  /// nodes shouldn't be converted.
  FunctionQuantizer(Function &F, const ExecutionEngine &EE,
                    llvm::ArrayRef<NodeQuantizationInfo> quantizationInfos,
                    ElemKind quantizationPrecision,
                    const KindSet &doNotQuantizeKinds)
      : FunctionConverter(F), mod_(*F.getParent()), EE_(EE),
        quantizationPrecision_(quantizationPrecision),
        doNotQuantizeKinds_(doNotQuantizeKinds) {
    // Build a mapping between node name and TensorQuantizatonParams.
    for (const auto &quantizationInfo : quantizationInfos) {
      nodeToTQP_.emplace(quantizationInfo.nodeOutputName_,
                         quantizationInfo.tensorQuantizationParams_);
    }
    // Use for debug purposes.
    lastMorphedNodeWithTypeChanges = nullptr;
  }

  /// Traverse all nodes to find quantized FullyConnected nodes, and convert it
  /// to RowwiseQuantizedFullyConnected if the weights is constant.
  void enableRowwise() {
    auto nodeIt = function_.getNodes().end();
    auto stopIt = function_.getNodes().begin();
    do {
      --nodeIt;
      Node &node = *nodeIt;
      // After function "convert()" is called, one FullyConnectedNode is
      // converted into:
      // [fp32 input] [fp32 weights] [fp32 bias]
      //      |              |           |
      // [QuantizeNode] [QuantizeNode] [QuantizeNode]
      //      \              |           /
      // [            FullyConnectedNode            ]
      //                     |
      //              [DequantizeNode]
      // We need to find the above patern and convert it to:
      // [fp32 input]              [fp32 weights]            [fp32 bias]
      //      |                    /      |       \              |
      //      |         [int8 weights] [scales] [offsets]        |
      // [QuantizeNode]      |            |        |         [QuantizeNode]
      //      \              |            |        |             /
      // [         RowwiseQuantizedFullyConnectedNode            ]
      //                              |
      //                       [DequantizeNode]
      if (auto *Q = llvm::dyn_cast<DequantizeNode>(&node)) {
        if (auto *fcN = llvm::dyn_cast<FullyConnectedNode>(Q->getInput())) {
          NodeValue input = fcN->getInput();
          NodeValue weights = fcN->getWeights();
          NodeValue bias = fcN->getBias();
          NodeValue result = fcN->getResult();
          // Only convert quantized FullyConnected Node.
          if (input.getType()->isQuantizedType() &&
              llvm::isa<QuantizeNode>(weights.getNode()) &&
              bias.getType()->isQuantizedType() &&
              result.getType()->isQuantizedType()) {
            auto *wq = llvm::dyn_cast<QuantizeNode>(weights.getNode());
            // For RowwiseQuantizedFullyConnected, the weights need to be
            // constant.
            if (Constant *wc = llvm::dyn_cast<Constant>(wq->getInput())) {
              auto *fcq = function_.createRowwiseQuantizedFullyConnected(
                  "rowwiseqfc", input, wc, bias, result.getType(),
                  /* transposeWeight */ true);
              // Replace the usages of quantized FC node to
              // RowwiseQuantizedFullyConnected Node.
              result.replaceAllUsesOfWith(fcq->getResult());
            }
          }
        }
      }
    } while (nodeIt != stopIt);

    cleanUp();
    assert(function_.verify() && "Conversion led to invalid function");
  }
};

} // namespace

namespace glow {
namespace quantization {

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
                 ElemKind quantizationPrecision, Function *F,
                 llvm::StringRef newFuncName, const KindSet &doNotQuantizeKinds,
                 bool enableRowwise) {
  assert((quantizationPrecision == ElemKind::Int8QTy ||
          quantizationPrecision == ElemKind::Int16QTy) &&
         "Only Int8 and Int16 quantization supported");
  std::string tmpName;
  if (newFuncName.empty()) {
    tmpName = std::string(F->getName()) + "_quantized";
    newFuncName = tmpName;
  }

  Function *G = F->clone(newFuncName);

  FunctionQuantizer quantizer(*G, EE, quantizationInfos, quantizationPrecision,
                              doNotQuantizeKinds);
  quantizer.convert();
  if (enableRowwise) {
    quantizer.enableRowwise();
  }

  return G;
}

} // namespace quantization
} // namespace glow
