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

#include "glow/Quantization/Quantization.h"

#include "glow/Backend/Backend.h"
#include "glow/Converter/FunctionConverter.h"

#include <cmath>
#include <unordered_set>
#include <vector>

using llvm::cast;

namespace {

using namespace glow;
using namespace glow::quantization;

/// \returns whether BatchedAddNode \p baN was originally lowered from a
/// FullyConnectedNode based on the given \p loweredMap.
static bool isBAFromLoweredFC(const BatchedAddNode *baN,
                              const LoweredInfoMap &loweredMap) {
  // Look for the set of NodeNameAndKinds corresponding to the
  // BatchedAdd. If one exists, this means it was lowered.
  auto it = loweredMap.find(baN->getResult().generateNodeOutputName());
  if (it == loweredMap.end()) {
    return false;
  }

  // Look through the set looking to see if the BatchedAdd was lowered
  // from a FullyConnectedNode.
  auto &set = it->getValue();
  for (auto i = set.begin(), e = set.end(); i != e; ++i) {
    if (i->getKind() == glow::Kinded::Kind::FullyConnectedNodeKind) {
      return true;
    }
  }
  return false;
}

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
    auto outTQPIt = nodeToTQP_.find(out.generateNodeOutputName());
    assert(outTQPIt != nodeToTQP_.end() &&
           "Missing quantization params for a node");

    const TensorQuantizationParams &TQP = outTQPIt->second;
    return mod_.uniqueType(quantizationPrecision_, out.dims(), TQP.scale,
                           TQP.offset);
  }

  /// \see FunctionConverter::getTargetTypeForOutput.
  /// \returns quantized type for \p out if any; if not quantizable, then
  /// \returns the original type.
  TypeRef getTargetTypeForOutput(const NodeValue &out) const override {
    if (out.getElementType() != ElemKind::FloatTy) {
      return out.getType();
    }
    return getTargetTypeForOutputImpl(out);
  }

  /// \see FunctionConverter::getTargetTypeForInput.
  /// \returns the quantized type for the \p idx-th argument of \p use, if any;
  /// if not quantizable, then \returns the original type.
  TypeRef getTargetTypeForInput(const Node &use, unsigned idx) const override {
    NodeValue val = use.getNthInput(idx);

    // Do not quantize non floating point type, e.g., Index type.
    if (val.getElementType() != ElemKind::FloatTy) {
      return val.getType();
    }

    auto valTQPIt = nodeToTQP_.find(val.generateNodeOutputName());
    assert(valTQPIt != nodeToTQP_.end() &&
           "Missing quantization params for a node");

    const TensorQuantizationParams &TQP = valTQPIt->second;

    // Local lambda to specialize the bias quantization parameters.
    auto getBiasType = [&](TypeRef inputTy, TypeRef weightsTy) -> TypeRef {
      TensorQuantizationParams inputTQP = {inputTy->getScale(),
                                           inputTy->getOffset()};
      TensorQuantizationParams weightsTQP = {weightsTy->getScale(),
                                             weightsTy->getOffset()};
      auto biasTQP = specializeBiasQuantizationParams(
          TQP, inputTQP, weightsTQP, schema_, quantizationPrecisionBias_);
      return mod_.uniqueType(quantizationPrecisionBias_, val.dims(),
                             biasTQP.scale, biasTQP.offset);
    };

    // NOTE: For every node for which the bias is specialized add the similar
    // logic in the 'generateNodeQuantizationInfos' function.
    if (use.getKind() == glow::Kinded::Kind::ConvolutionNodeKind &&
        idx == ConvolutionNode::BiasIdx) {
      // Get the input and weights types. This ensures the types will be
      // quantized. This is often the case when calling into this function from
      // canConvert(), as we have not yet converted the inputs.
      return getBiasType(
          getTargetTypeForInput(use, ConvolutionNode::InputIdx),
          getTargetTypeForInput(use, ConvolutionNode::FilterIdx));
    } else if (use.getKind() == glow::Kinded::Kind::Convolution3DNodeKind &&
               idx == Convolution3DNode::BiasIdx) {
      // Get the input and weights types. This ensures the types will be
      // quantized. This is often the case when calling into this function from
      // canConvert(), as we have not yet converted the inputs.
      return getBiasType(
          getTargetTypeForInput(use, Convolution3DNode::InputIdx),
          getTargetTypeForInput(use, Convolution3DNode::FilterIdx));
    } else if (use.getKind() == glow::Kinded::Kind::ConvTransposeNodeKind &&
               idx == ConvTransposeNode::BiasIdx) {
      // Get the input and weights types. This ensures the types will be
      // quantized. This is often the case when calling into this function from
      // canConvert(), as we have not yet converted the inputs.
      return getBiasType(
          getTargetTypeForInput(use, ConvTransposeNode::InputIdx),
          getTargetTypeForInput(use, ConvTransposeNode::FilterIdx));
    } else if (use.getKind() == glow::Kinded::Kind::FullyConnectedNodeKind &&
               idx == FullyConnectedNode::BiasIdx) {
      // Return the original type if we don't want to convert FC biases.
      if (skipQuantizeFCBias_) {
        return val.getType();
      }
      // Get the input and weights types. This ensures the types will be
      // quantized. This is often the case when calling into this function from
      // canConvert(), as we have not yet converted the inputs.
      return getBiasType(
          getTargetTypeForInput(use, FullyConnectedNode::InputIdx),
          getTargetTypeForInput(use, FullyConnectedNode::WeightsIdx));
    } else if (use.getKind() == glow::Kinded::Kind::BatchedAddNodeKind &&
               idx == BatchedAddNode::SliceIdx) {
      // Check if this BatchedAdd was lowered from a FullyConnectedNode.
      const auto *baN = llvm::cast<BatchedAddNode>(&use);
      if (isBAFromLoweredFC(baN, loweredMap_)) {
        // If it came from a FullyConnected node then we need to backtrack to
        // the matrix multiplication to calculate the new scale for the batched
        // add slice. Slice must be a MatMul if this was lowered from a
        // FullyConnected. Batch may have already been quantized.
        NodeValue batch = baN->getBatch();
        assert(
            (llvm::isa<MatMulNode>(batch) || llvm::isa<QuantizeNode>(batch)) &&
            "Batch must be either a MatMul or a Quantize.");
        MatMulNode *MM = llvm::dyn_cast<MatMulNode>(batch);
        if (!MM) {
          QuantizeNode *QN = llvm::cast<QuantizeNode>(batch);
          assert(llvm::isa<MatMulNode>(QN->getInput()) &&
                 "MM must be input of BA if lowered from FC.");
          MM = llvm::cast<MatMulNode>(QN->getInput());
        }
        return getBiasType(getTargetTypeForOutput(MM->getLHS()),
                           getTargetTypeForOutput(MM->getRHS()));
      }
    }
    return mod_.uniqueType(quantizationPrecision_, val.dims(), TQP.scale,
                           TQP.offset);
  }

  /// Macro to be put in a switch for all nodes that may need to be replaced by
  /// a LookupTable if the backend doesn't support the quantized node directly.
#define CASES_FOR_INT_LOOKUP_TABLE_REPLACEMENT                                 \
  case Kinded::Kind::LogNodeKind:                                              \
  case Kinded::Kind::TanhNodeKind:                                             \
  case Kinded::Kind::SigmoidNodeKind

  /// \see FunctionConverter::canConvert.
  /// Only convert nodes that use floating point types and that
  /// weren't specifically marked as to-ignore with doNotQuantizeKinds_.
  bool canConvert(const Node &node) const override {
    // Check if the node is one that we never want to convert, e.g. SaveNode.
    if (!FunctionConverter::canConvert(node)) {
      return false;
    }

    // Check if the node kind should not be converted based on supplied kinds
    // informed to the converter.
    if (doNotQuantizeKinds_.count(node.getKind())) {
      return false;
    }

    // Gather the input and output types that we will have once we quantize the
    // node, and check if the backend supports such a node. Note that if a node
    // has float inputs or outputs then we must have quantization parameters for
    // them. For inputs and outputs without quantization parameters, we keep
    // their original element type.
    bool needsQuantization = false;
    std::vector<TypeRef> inputTypes, outputTypes;
    for (unsigned idx = 0, end = node.getNumInputs(); idx != end; ++idx) {
      NodeValue val = node.getNthInput(idx);
      if (val.getElementType() == ElemKind::FloatTy) {
        needsQuantization = true;
        if (!quantizationParamsExist(val)) {
          CHECK(!assertAllNodesQuantized_)
              << "Quantization parameters did not exist for an input NodeValue "
                 "that should have been quantized; input number "
              << idx << " of node:\n"
              << node.getDebugDesc();
          return false;
        }
      }
      TypeRef targetTy = getTargetTypeForInput(node, idx);
      inputTypes.push_back(targetTy);
    }
    for (unsigned idx = 0, end = node.getNumResults(); idx != end; ++idx) {
      NodeValue val = node.getNthResult(idx);
      if (val.getElementType() == ElemKind::FloatTy) {
        needsQuantization = true;
        if (!quantizationParamsExist(val)) {
          CHECK(!assertAllNodesQuantized_)
              << "Quantization parameters did not exist for a result of a Node "
                 "that should have been quantized; result number "
              << idx << " of node:\n"
              << node.getDebugDesc();
          return false;
        }
      }
      TypeRef targetTy = getTargetTypeForOutput(val);
      outputTypes.push_back(targetTy);
    }

    // If none of the inputs were FPType then there's no quantization to
    // perform, so we return that we cannot convert this node.
    if (!needsQuantization) {
      return false;
    }

    // Only convert the node if the backend supports the newly converted node.
    bool isOpSupported =
        B_.isOpSupported(NodeInfo(node.getKind(), inputTypes, outputTypes));

    // Some nodes are only supported as quantized via lookup tables. Here we
    // check if such nodes are supported without lookup tables; if so, then we
    // convert them.  Otherwise, return whether we can support them as lookup
    // tables instead, and they will be quantized as lookup tables.
    switch (node.getKind()) {
    CASES_FOR_INT_LOOKUP_TABLE_REPLACEMENT:
      if (!isOpSupported) {
        isOpSupported = B_.isOpSupported(NodeInfo(
            Kinded::Kind::IntLookupTableNodeKind, inputTypes, outputTypes));
      }
      break;
    default:
      break;
    }

    // Quantizer may be set up to die if a node is only skipped during
    // quantization because the backend does not support it as quantized.
    if (assertAllNodesQuantized_) {
      CHECK(isOpSupported) << B_.getBackendName()
                           << " Backend does not support node as quantized in "
                           << Type::getElementName(quantizationPrecision_).str()
                           << ":\n"
                           << node.getDebugDesc();
    }

    return isOpSupported;
  }

  /// Helper that \returns whether quantization parameters exist
  /// in \ref nodeToTQP_ given the name and result number of \p val.
  bool quantizationParamsExist(const NodeValue &val) const {
    auto valTQPIt = nodeToTQP_.find(val.generateNodeOutputName());
    return valTQPIt != nodeToTQP_.end();
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
  Node *createConversion(Function &function, const Node &node, NodeValue &val,
                         TypeRef destTy, bool /* isInput */) override {
    assert((&function == &function_) &&
           "Trying to add quantize/dequantize conversion to a function other "
           "than the function being quantized.");
    std::string nodeName = node.getName().str();
    if (destTy->isQuantizedType()) {
      return function.createQuantize(nodeName + "_quantize", val, destTy);
    }
    return function.createDequantize(nodeName + "_dequantize", val, destTy);
  }

  /// All IRConstraint cases below assume that the input and output index that
  /// they are looking for the type is at idx 0. We statically assert that here
  /// along with the case.
  static constexpr unsigned SingleMatchingInOutTypeInputIdx = 0;
  static constexpr unsigned SingleMatchingInOutTypeResultIdx = 0;
#define CASE_SINGLE_MATCHING_INOUT_TYPE(NODE_NAME_, INPUT_NAME_, OUTPUT_NAME_) \
  static_assert((NODE_NAME_##Node::INPUT_NAME_##Idx ==                         \
                     SingleMatchingInOutTypeInputIdx &&                        \
                 NODE_NAME_##Node::OUTPUT_NAME_##Idx ==                        \
                     SingleMatchingInOutTypeResultIdx),                        \
                #NODE_NAME_ "Node format is unexpected.");                     \
  case Kinded::Kind::NODE_NAME_##NodeKind

  /// Macro to be put in a switch for all the nodes that have a constraint
  /// where the input and output type must be equals.
  /// Note: The last case of the macro doesn't have ':' so we can put it
  /// where the macro is inserted to keep the nice code formatting.
  // clang-format off
#define CASES_FOR_SINGLE_MATCHING_IN_OUT_TYPE                                  \
  CASE_SINGLE_MATCHING_INOUT_TYPE(LocalResponseNormalization, Input, Result):  \
  CASE_SINGLE_MATCHING_INOUT_TYPE(Slice, Input, Result):                       \
  CASE_SINGLE_MATCHING_INOUT_TYPE(Reshape, Input, Result):                     \
  CASE_SINGLE_MATCHING_INOUT_TYPE(Transpose, Input, Result):                   \
  CASE_SINGLE_MATCHING_INOUT_TYPE(TopK, Input, Values):                        \
  CASE_SINGLE_MATCHING_INOUT_TYPE(Gather, Data, Result):                       \
  CASE_SINGLE_MATCHING_INOUT_TYPE(MaxPool, Input, Result):                     \
  CASE_SINGLE_MATCHING_INOUT_TYPE(ResizeNearest, Input, Result):               \
  CASE_SINGLE_MATCHING_INOUT_TYPE(ResizeBilinear, Input, Result):              \
  CASE_SINGLE_MATCHING_INOUT_TYPE(SpaceToDepth, Input, Result)
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
    CASES_FOR_SINGLE_MATCHING_IN_OUT_TYPE : {
      // The constraints on the IR says that the input type must
      // be the same as the output type.
      TypeRef inTy =
          node.getNthInput(SingleMatchingInOutTypeInputIdx).getType();
      TypeRef fixedTy = mod_.uniqueType(
          quantizationPrecision_,
          node.getNthResult(SingleMatchingInOutTypeResultIdx).dims(),
          inTy->getScale(), inTy->getOffset());

      node.setType(SingleMatchingInOutTypeResultIdx, fixedTy);
      assert(!lastMorphedNodeWithTypeChanges &&
             "Missed one node to rescale in postprocessing");
      lastMorphedNodeWithTypeChanges = &node;
      return node;
    }
    default:
      return node;
    }
  }

  /// Perform post processing for \p node. Handles special cases, e.g.
  /// requirements for input/output quantization parameters, converting to
  /// lookup tables, etc. Also updates nodeToTQP_ with the added dequantization
  /// nodes added for \p node.
  void postProcessing(Node &node) override {
    Node *quantizedNode = &node;
    switch (node.getKind()) {
    default:
      break;

      // Cases for nodes where all inputs should use the same scale/offset as
      // the output.
#define CASE_ALL_INS_MATCH_SINGLE_OUT(NODE_KIND_)                              \
  case Kinded::Kind::NODE_KIND_##NodeKind: {                                   \
    auto *N = cast<NODE_KIND_##Node>(&node);                                   \
    TypeRef outputTy = N->getResult().getType();                               \
    assert(outputTy->isQuantizedType() && "Node hasn't been quantized yet?!"); \
    unsigned idx = 0;                                                          \
    for (size_t i = 0, e = N->getNumInputs(); i < e; ++i) {                    \
      NodeValue input = N->getNthInput(i);                                     \
      auto argOutTy =                                                          \
          mod_.uniqueType(quantizationPrecision_, input.dims(),                \
                          outputTy->getScale(), outputTy->getOffset());        \
      auto *rescale = function_.createRescaleQuantized(                        \
          input.getNode()->getName(), input, argOutTy);                        \
      function_.getLogContext()->logNodeInputChange(*N, N->getNthInput(idx),   \
                                                    rescale);                  \
      N->setNthInput(idx++, rescale);                                          \
    }                                                                          \
    break;                                                                     \
  }
      CASE_ALL_INS_MATCH_SINGLE_OUT(Concat);
      CASE_ALL_INS_MATCH_SINGLE_OUT(InsertTensor);
#undef CASE_ALL_INS_MATCH_SINGLE_OUT

    CASES_FOR_SINGLE_MATCHING_IN_OUT_TYPE : {
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
      TypeRef outputTy = getTargetTypeForOutputImpl(
          NodeValue(&node, SingleMatchingInOutTypeResultIdx));
      assert(outputTy->isQuantizedType() && "Node hasn't been quantized yet?!");
      auto outTy = mod_.uniqueType(quantizationPrecision_, outputTy->dims(),
                                   outputTy->getScale(), outputTy->getOffset());
      NodeValue val = node.getNthResult(SingleMatchingInOutTypeResultIdx);
      // "val" may not have any users if the output goes unused, e.g. if we are
      // quantizing a TopKNode and only indices is used.
      if (val.getNumUsers() == 0) {
        break;
      }
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

      function_.getLogContext()->logNodeInputChange(
          *dequantize, dequantize->getNthInput(DequantizeNode::InputIdx),
          rescale);
      dequantize->setNthInput(DequantizeNode::InputIdx, rescale);
      break;
    }

    CASES_FOR_INT_LOOKUP_TABLE_REPLACEMENT : {
      // If these nodes aren't supported then we convert them to a lookup table.
      NodeInfo NI(node);
      if (B_.isOpSupported(NI)) {
        break;
      }
      assert(B_.isOpSupported(NodeInfo(Kinded::Kind::IntLookupTableNodeKind,
                                       NI.getInTypes(), NI.getOutTypes())) &&
             "Backend should support IntLookupTable at this point.");
      switch (node.getKind()) {
      case Kinded::Kind::LogNodeKind:
        quantizedNode = replaceQuantizedLogWithLookupTable(
            function_, llvm::cast<LogNode>(node), schema_);
        break;
      case Kinded::Kind::ExpNodeKind:
        quantizedNode = replaceQuantizedExpWithLookupTable(
            function_, llvm::cast<ExpNode>(node), schema_);
        break;
      case Kinded::Kind::TanhNodeKind:
        quantizedNode = replaceQuantizedTanhWithLookupTable(
            function_, llvm::cast<TanhNode>(node), schema_);
        break;
      case Kinded::Kind::SigmoidNodeKind:
        quantizedNode = replaceQuantizedSigmoidWithLookupTable(
            function_, llvm::cast<SigmoidNode>(node), schema_);
        break;
      default:
        llvm_unreachable("Unsupported case for converting to lookup table.");
      }
    }
    }
    assert(!lastMorphedNodeWithTypeChanges && "Type not fixed");

    // Update nodeToTQP_ since we've added in dequantized nodes to the output of
    // now-quantized nodes. This is necessary because later we try to quantize
    // nodes only if we have a quantized type for its operands (i.e. a profile
    // in nodeToTQP_). However its inputs may already have been quantized, which
    // means its inputs are replaced by a dequantize node, and no profile would
    // be found in nodeToTQP_ for the dequantize node_. Thus we add TQPs for
    // the dequantize node given the scale/offset it is dequantizing from.
    for (unsigned outNum = 0, e = quantizedNode->getNumResults(); outNum != e;
         ++outNum) {
      NodeValue val = quantizedNode->getNthResult(outNum);
      if (!val.getType()->isQuantizedType()) {
        continue;
      }
      // Not all float outputs will have a dequantize added to its quantized
      // output, as we may just be using some outputs of a quantized node
      // (e.g. when quantizing a TopK but only using the Indices output, no
      // dequantize node is added to Values).
      if (val.getNumUsers() == 0) {
        continue;
      }
      assert(
          val.hasOneUse() &&
          llvm::dyn_cast<DequantizeNode>((*val.getUsers().begin()).getUser()) &&
          "This node should only be used by the dequantize node");
      auto *dequantize =
          llvm::dyn_cast<DequantizeNode>((*val.getUsers().begin()).getUser());
      TypeRef outTy = val.getType();
      auto name = dequantize->getResult().generateNodeOutputName();
      nodeToTQP_[name] = {outTy->getScale(), outTy->getOffset()};
    }
  } // namespace

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
  /// Backend used to check is a quantized operator is supported.
  const Backend &B_;
  /// Quantization schema.
  quantization::Schema schema_;
  /// Quantization precision.
  const ElemKind quantizationPrecision_;
  /// Set of node kinds that should not be quantized.
  const KindSet &doNotQuantizeKinds_;
  /// Map the (name of a node, idx) to its quantization parameters.
  std::unordered_map<std::string, TensorQuantizationParams> nodeToTQP_;
  /// For debug, keep track of the last node that we changed because of IR
  /// constraints.
  Node *lastMorphedNodeWithTypeChanges;
  /// A map between quantization profiling names of NodeValues that were lowered
  /// from each other. Maps to a set of names of NodeValues and their NodeKinds
  /// that were replaced by the NodeValue (whose output name is the key) that
  /// replaced them.
  const LoweredInfoMap &loweredMap_;
  /// Used for debugging if we expect all nodes to be quantized by the
  /// quantizer.
  bool assertAllNodesQuantized_;
  /// Precision used for bias quantization for Convolution and FullyConnected.
  /// This allows specializing the bias quantization.
  const ElemKind quantizationPrecisionBias_;

  // If true, don't apply quantization to FC bias inputs.
  const bool skipQuantizeFCBias_;

public:
  /// Creates a function quantizer for function \p F using the quantization
  /// configuration \p quantConfig. This method quantizes as many nodes as
  /// permitted by the backend \p B. The map \p loweredMap contains info about
  /// what nodes were lowered from what, to be used during quantization.
  /// \p doNotQuantizeKinds lists kinds to not quantize, even if a profile was
  /// gathered for them and the backend supports the quantized operation.
  FunctionQuantizer(Function &F, const Backend &B,
                    const QuantizationConfiguration &quantConfig,
                    const KindSet &doNotQuantizeKinds,
                    const LoweredInfoMap &loweredMap)
      : FunctionConverter(F), mod_(*F.getParent()), B_(B),
        schema_(quantConfig.schema),
        quantizationPrecision_(quantConfig.precision),
        doNotQuantizeKinds_(doNotQuantizeKinds), loweredMap_(loweredMap),
        assertAllNodesQuantized_(quantConfig.assertAllNodesQuantized),
        quantizationPrecisionBias_(quantConfig.precisionBias),
        skipQuantizeFCBias_(quantConfig.skipQuantizeFCBias) {

    // Compute the TensorQuantizationParams using the profiling infos.
    auto quantizationInfos =
        generateNodeQuantizationInfos(&F, quantConfig, loweredMap);

    // Build a mapping between node name and TensorQuantizatonParams.
    for (const auto &quantizationInfo : quantizationInfos) {
      nodeToTQP_.emplace(quantizationInfo.nodeOutputName_,
                         quantizationInfo.tensorQuantizationParams_);
    }

    // Use for debug purposes.
    lastMorphedNodeWithTypeChanges = nullptr;
    (void)assertAllNodesQuantized_;
  }

  /// Traverse all nodes to find applicable quantized nodes, and convert them
  /// to RowwiseQuantized versions if required inputs are Constant.
  void enableRowwise() {
    auto nodeIt = function_.getNodes().end();
    auto stopIt = function_.getNodes().begin();
    do {
      --nodeIt;
      Node &node = *nodeIt;
      auto *Q = llvm::dyn_cast<DequantizeNode>(&node);
      if (!Q) {
        continue;
      }

      // ----------------------------------------------------------------------
      // After function "convert()" is called, one FullyConnectedNode is
      // converted into:
      // [fp32 input] [fp32 weights] [fp32 bias]
      //      |              |           |
      // [QuantizeNode] [QuantizeNode] [QuantizeNode (optional)]
      //      \              |           /
      // [            FullyConnectedNode            ]
      //                     |
      //              [DequantizeNode]
      // We need to find the above pattern and convert it to:
      // [fp32 input]              [fp32 weights]            [fp32 bias]
      //      |                    /      |       \              |
      //      |         [int8 weights] [scales] [offsets]        |
      // [QuantizeNode]      |            |        |         [QuantizeNode]
      //      \              |            |        |             /
      // [         RowwiseQuantizedFullyConnectedNode            ]
      //                              |
      //                       [DequantizeNode]
      // ----------------------------------------------------------------------
      bool foundFC = false;
      NodeValue input, weights, bias, result;
      if (auto *fcN = llvm::dyn_cast<FullyConnectedNode>(Q->getInput())) {
        foundFC = true;
        input = fcN->getInput();
        weights = fcN->getWeights();
        bias = fcN->getBias();
        result = fcN->getResult();
      } else if (const auto *baN =
                     llvm::dyn_cast<BatchedAddNode>(Q->getInput())) {
        if (isBAFromLoweredFC(baN, loweredMap_)) {
          foundFC = true;
          NodeValue batch = baN->getBatch();

          // All quantization has occurred at this point, but optimizations
          // haven't eliminated extra quantize/dequantize nodes. Look
          // backwards through them to find the MatMul of the FC.
          assert(llvm::isa<QuantizeNode>(batch));
          QuantizeNode *QN = llvm::cast<QuantizeNode>(batch);
          assert(llvm::isa<DequantizeNode>(QN->getInput()));
          DequantizeNode *DQN = llvm::cast<DequantizeNode>(QN->getInput());
          assert(llvm::isa<MatMulNode>(DQN->getInput()));
          MatMulNode *MM = llvm::cast<MatMulNode>(DQN->getInput());

          input = MM->getLHS();
          weights = MM->getRHS();
          bias = baN->getSlice();
          result = baN->getResult();
        }
      }
      if (foundFC) {
        // Only convert quantized FullyConnected Node (or its equivalent lowered
        // representation in MatMul + BatchedAdd form).
        if (input.getType()->isQuantizedType() &&
            llvm::isa<QuantizeNode>(weights.getNode()) &&
            result.getType()->isQuantizedType()) {
          auto *wq = llvm::dyn_cast<QuantizeNode>(weights.getNode());
          // For RowwiseQuantizedFullyConnected, the weights need to be
          // constant.
          if (Constant *wc = llvm::dyn_cast<Constant>(wq->getInput())) {
            auto *fcq = function_.createRowwiseQuantizedFullyConnected(
                "rowwiseqfc", input, wc, bias, result.getType(), schema_,
                /* transposeWeight */ true);
            // Replace usages of quantized FC node (or its equivalent lowered
            // representation MM + BA) to RowwiseQuantizedFullyConnectedNode.
            result.replaceAllUsesOfWith(fcq->getResult());
          }
        }
      }

      // Convert SLWS from normal version to fused rowwise-quantized version if
      // applicable. Data must be Constant for this to occur. We also will not
      // quantize the weights as we do for the default normal quantized SLWS, as
      // the rowwise version uses float weights.
      if (auto *SLWS =
              llvm::dyn_cast<SparseLengthsWeightedSumNode>(Q->getInput())) {
        NodeValue data = SLWS->getData();

        // It's possible we skipped quantizing this node due to
        // doNotQuantizeKinds, and so may not need to process it.
        auto *dataQN = llvm::dyn_cast<QuantizeNode>(data.getNode());
        if (!dataQN) {
          continue;
        }

        // Can only convert to rowwise-quantized version if the data input is
        // Constant.
        auto *dataC = llvm::dyn_cast<Constant>(dataQN->getInput());
        if (!dataC) {
          continue;
        }

        // Right now we quantize the weights input for SLWS. However, the
        // rowwise-quantized version does not, so we will skip the QN. At this
        // point we know the SLWS was quantized, so the weights input must be a
        // quantize node.
        auto *weightsQN = llvm::dyn_cast<QuantizeNode>(SLWS->getWeights());
        assert(weightsQN && "Weights should have been quantized");
        NodeValue weightsF = weightsQN->getInput();

        auto *FRWQSLWS =
            function_.createFusedRowwiseQuantizedSparseLengthsWeightedSum(
                SLWS->getName(), dataC->getPayloadMutable(), weightsF,
                SLWS->getIndices(), SLWS->getLengths(),
                /* fusedElemKind */ ElemKind::UInt8FusedQTy,
                /* useFP16Accumulation */ false, SLWS->getLengthsMode(),
                SLWS->getAvgLength());

        // Fused RWQSLWS stores the fused scales and offsets in trailing
        // columns. If the input was single dimensional then it adds extra
        // dimensions to both input and output. Therefore reshape back to the
        // expected output shape in case the input to the SLWS did not have a
        // second dimension but the fused version added one to insert columns.
        auto *RN = function_.createReshape("reshape", FRWQSLWS,
                                           SLWS->getResult().dims());

        // Replace the dequantize node of the original SLWS with the FRWQSLWS,
        // as its output is already in float.
        Q->getResult().replaceAllUsesOfWith(RN->getResult());
      }

    } while (nodeIt != stopIt);

    cleanUp();
    assert(function_.verify() && "Conversion led to invalid function");
  }

  /// Traverse all nodes to find applicable quantized nodes, and convert them
  /// to ChannelwiseQuantized versions if required inputs are Constant.
  void enableChannelwise() {
    auto nodeIt = function_.getNodes().end();
    auto stopIt = function_.getNodes().begin();
    do {
      --nodeIt;
      Node &node = *nodeIt;
      auto *Q = llvm::dyn_cast<DequantizeNode>(&node);
      if (!Q) {
        continue;
      }

      // ----------------------------------------------------------------------
      // After function "convert()" is called, one ConvolutionNode is
      // converted into:
      //  [fp32 input]  [fp32 filter]   [fp32 bias]
      //       |              |              |
      // [QuantizeNode] [QuantizeNode] [QuantizeNode]
      //        \             |            /
      // [             ConvolutionNode            ]
      //                     |
      //              [DequantizeNode]
      // We need to find the above pattern and convert it to:
      // [fp32 input]           [fp32 filter]                [fp32 bias]
      //      |                /        |     \            /       |     \
      //      |         [int8 filter|scales|offsets] [int8 bias|scales|offsets]
      // [QuantizeNode]       |         |      |         |         |      |
      //      \               |         |      |         /         /      /
      // [         ChannelwiseQuantizedConvolutionNode                        ]
      //                              |
      //                       [DequantizeNode]
      // ----------------------------------------------------------------------

      // Replace ConvolutionNode with ChannelwiseQuantizedConvolutionNode
      // if the filter and bias operands are constant. The node creation
      // function will be provided with the floating-point filter and
      // bias constants and will perform channel wise quantization.
      if (auto *convNode = llvm::dyn_cast<ConvolutionNode>(Q->getInput())) {

        NodeValue input = convNode->getInput();
        NodeValue filter = convNode->getFilter();
        NodeValue bias = convNode->getBias();
        NodeValue result = convNode->getResult();

        if (input.getType()->isQuantizedType() &&
            llvm::isa<QuantizeNode>(filter.getNode()) &&
            llvm::isa<QuantizeNode>(bias.getNode()) &&
            result.getType()->isQuantizedType()) {

          auto *filterQ = llvm::dyn_cast<QuantizeNode>(filter.getNode());
          Constant *filterC = llvm::dyn_cast<Constant>(filterQ->getInput());
          auto *biasQ = llvm::dyn_cast<QuantizeNode>(bias.getNode());
          Constant *biasC = llvm::dyn_cast<Constant>(biasQ->getInput());

          if (filterC && biasC) {
            // When the overall requested quantization schema is asymmetric
            // we use symmetric quantization schema for the channelwise filter
            // and bias in order to be closer to the TFLite quantization specs:
            // https://www.tensorflow.org/lite/performance/quantization_spec
            quantization::Schema quantSchema = schema_;
            if (quantSchema == quantization::Schema::Asymmetric) {
              quantSchema = quantization::Schema::Symmetric;
            }
            // Create per channel quantized Convolution.
            auto *convNodeCWQ = function_.createChannelwiseQuantizedConv(
                "ChannelwiseQuantizedConv", input, filterC, biasC,
                /* filterScales */ nullptr, /* filterOffsets */ nullptr,
                /* biasScales */ nullptr, /* biasOffsets */ nullptr,
                result.getType(), convNode->getKernels(),
                convNode->getStrides(), convNode->getPads(),
                convNode->getGroup(), convNode->getDilation(),
                /* quantizeFilter */ true, /* quantizeBias */ true, quantSchema,
                quantizationPrecision_, quantizationPrecisionBias_);
            convNodeCWQ->setFusedActivation(convNode->getFusedActivation());
            convNodeCWQ->setFusedActivationArgs(
                convNode->getFusedActivationArgs());
            result.replaceAllUsesOfWith(convNodeCWQ->getResult());
          }
        }
      }
    } while (nodeIt != stopIt);
    cleanUp();
    assert(function_.verify() && "Conversion led to invalid function");
  }
}; // namespace

} // namespace

namespace glow {
namespace quantization {

Node *replaceQuantizedLogWithLookupTable(Function &F, const LogNode &LN,
                                         Schema schema) {
  IntLookupTableNode *ILT = F.createIntLog(
      LN.getName().str() + ".log", LN.getInput(), LN.getResult().getType());
  LN.getResult().replaceAllUsesOfWith(ILT);
  return ILT;
}

Node *replaceQuantizedExpWithLookupTable(Function &F, const ExpNode &EN,
                                         Schema schema) {
  IntLookupTableNode *ELT = F.createIntExp(
      EN.getName().str() + ".exp", EN.getInput(), EN.getResult().getType());
  EN.getResult().replaceAllUsesOfWith(ELT);
  return ELT;
}

Node *replaceQuantizedTanhWithLookupTable(Function &F, const TanhNode &TN,
                                          Schema schema) {
  // Quantized tanh operator expects input to be in a certain floating point
  // range. This operator works based on the precomputed table and has to
  // process input in a range of [-3.0, 3.0]. Tanh asymptotically approaches
  // +/-1.0 and is already +/-.995 at +/-3.0.
  // The output quantization parameters are chosen to represent the floating
  // point range of [-1.0, 1.0].
  TypeRef inpTy = TN.getInput().getType();
  TypeRef outTy = TN.getResult().getType();
  auto inputQuantizationParams = glow::quantization::chooseQuantizationParams(
      {-3.0, 3.0}, schema, inpTy->getElementType());
  auto tanhInTy = F.getParent()->uniqueType(
      inpTy->getElementType(), TN.getResult().dims(),
      inputQuantizationParams.scale, inputQuantizationParams.offset);

  // Make sure input is clipped in [-3.0, 3.0] floating point range.
  auto *rescaleInputNode =
      F.createRescaleQuantized(TN.getName(), TN.getInput(), tanhInTy);

  // Make sure output is clipped in [-1.0, 1.0] floating point range.
  auto outputQuantizationParams = glow::quantization::chooseQuantizationParams(
      {-1.0, 1.0}, schema, outTy->getElementType());
  auto resultOutTy = F.getParent()->uniqueType(
      outTy->getElementType(), rescaleInputNode->getResult().dims(),
      outputQuantizationParams.scale, outputQuantizationParams.offset);

  // Note: The actual lookup table is created inside this call.
  auto *quantizedNode =
      F.createIntTanh(TN.getName(), rescaleInputNode, resultOutTy);

  auto *rescaleOutputNode = F.createRescaleQuantized(
      TN.getName(), quantizedNode, TN.getResult().getType());

  TN.getResult().replaceAllUsesOfWith(rescaleOutputNode);
  return rescaleOutputNode;
}

Node *replaceQuantizedSigmoidWithLookupTable(Function &F, const SigmoidNode &SN,
                                             Schema schema) {
  // Quantized sigmoid operator expects input to be in a certain floating
  // point range. This operator works based on the precomputed table and has
  // to process input in a range of [-6.0, 6.0]. Sigmoid asymptotically
  // approaches 0 at -inf and 1 at +inf. It has values of 0.00247262 and
  // 0.997527 at -6.0 and 6.0 correspondingly. The output quantization
  // parameters are chosen to represent the floating point range of [0, 1.0].
  TypeRef inpTy = SN.getInput().getType();
  TypeRef outTy = SN.getResult().getType();
  auto inputQuantizationParams = glow::quantization::chooseQuantizationParams(
      {-6.0, 6.0}, schema, inpTy->getElementType());
  auto sigmoidInTy = F.getParent()->uniqueType(
      inpTy->getElementType(), SN.getResult().dims(),
      inputQuantizationParams.scale, inputQuantizationParams.offset);

  // Make sure input is clipped in [-6.0, 6.0] floating point range.
  auto *rescaleInputNode =
      F.createRescaleQuantized(SN.getName(), SN.getInput(), sigmoidInTy);

  // Make sure output is clipped in [0.0, 1.0] floating point range.
  auto outputQuantizationParams = glow::quantization::chooseQuantizationParams(
      {0.0, 1.0}, schema, outTy->getElementType());
  auto resultOutTy = F.getParent()->uniqueType(
      outTy->getElementType(), rescaleInputNode->getResult().dims(),
      outputQuantizationParams.scale, outputQuantizationParams.offset);

  // Note: The actual lookup table is created inside this call.
  auto *quantizedNode =
      F.createIntSigmoid(SN.getName(), rescaleInputNode, resultOutTy);

  auto *rescaleOutputNode = F.createRescaleQuantized(
      SN.getName(), quantizedNode, SN.getResult().getType());

  SN.getResult().replaceAllUsesOfWith(rescaleOutputNode);

  return rescaleOutputNode->getResult();
}

/// Helper which, given the output name \p currName of some node, looks for
/// corresponding names in \p loweredMap which represent any names that this
/// node was lowered from. If any are found then they are inserted into \p
/// profilingInfos along with \p TPP.
static void
findAndInsertLoweredInfos(llvm::StringRef currName,
                          const LoweredInfoMap &loweredMap,
                          std::vector<NodeProfilingInfo> &profilingInfos,
                          const TensorProfilingParams &TPP) {
  auto currSetIt = loweredMap.find(currName);
  if (currSetIt == loweredMap.end()) {
    return;
  }

  // Get the set of names corresponding to currName. All names in the set are
  // names that were originally lowered into currName.
  auto &currSet = currSetIt->getValue();

  // For each of the names (currOrigName), insert them into profilingInfos,
  // and then recursively find and insert other names in case currOrigName was
  // also lowered from a previous node.
  for (auto i = currSet.begin(), e = currSet.end(); i != e; ++i) {
    llvm::StringRef currOrigName = i->getName();
    profilingInfos.emplace_back(currOrigName.str(), TPP);
    findAndInsertLoweredInfos(currOrigName, loweredMap, profilingInfos, TPP);
  }
}

std::vector<NodeProfilingInfo>
generateNodeProfilingInfos(PlaceholderBindings &bindings, const Function *F,
                           const LoweredInfoMap &loweredMap) {
  std::vector<NodeProfilingInfo> profilingInfos;
  for (auto &node : F->getNodes()) {
    auto *QPN = llvm::dyn_cast<QuantizationProfileNode>(&node);
    if (QPN) {

      // Extract the profiling information from the placeholders after running
      // the network in profiling mode.
      auto compInfoH = bindings.get(QPN->getComputationInfoPlaceholder())
                           ->getHandle<float>();
      auto *histogramT = bindings.get(QPN->getHistogramPlaceholder());
      float min = compInfoH.raw(0);
      float max = compInfoH.raw(1);

      // Generate a name to be used as profiling information identifier.
      std::string fullOutputName = NodeValue::generateNodeOutputName(
          QPN->getProfiledNodeName(), QPN->getProfiledOutputNumber());

      // Set TensorProfilingParams for this node output.
      TensorProfilingParams TPP(min, max, *histogramT);
      profilingInfos.emplace_back(fullOutputName, TPP);

      // If the NodeValue represented by fullOutputName was created via lowering
      // of another original NodeValue, then generate node profiling info for
      // the original NodeValue using the same profiling parameters.
      findAndInsertLoweredInfos(fullOutputName, loweredMap, profilingInfos,
                                TPP);
    }
  }
  return profilingInfos;
}

std::vector<NodeQuantizationInfo>
generateNodeQuantizationInfos(Function *F,
                              const QuantizationConfiguration &quantConfig,
                              const LoweredInfoMap &loweredMap) {
  std::vector<NodeQuantizationInfo> quantizationInfos;
  for (const auto &profilingInfo : quantConfig.infos) {
    // Get node value from node output name.
    std::string nodeOutputName = profilingInfo.nodeOutputName_;
    NodeValue nodeOutput = F->getNodeValueByName(nodeOutputName);

    // Skip if the node is not part of the graph.
    if (!nodeOutput.getNode()) {
      continue;
    }

    // Default quantization schema.
    Schema schema = quantConfig.schema;

    // Default target precision.
    ElemKind precision = quantConfig.precision;

    // Default calibration mode.
    Calibration calibration = quantConfig.calibration;

    // The TensorQuantizationParams must be computed using the target
    // precision used during the actual quantization. The code below
    // reflects the same logic as the one used in the function
    // FunctionQuantizer::getTargetTypeForInput for specializing the bias
    // quantization precision. Since bias quantization is sensitive we will
    // choose to use no calibration.
    for (const auto &use : nodeOutput.getUsers()) {
      const auto *user = use.getUser();
      if ((user->getKind() == glow::Kinded::Kind::ConvolutionNodeKind) &&
          (user->getNthInput(ConvolutionNode::BiasIdx) == nodeOutput)) {
        // Found bias for ConvolutionNode.
        precision = quantConfig.precisionBias;
        calibration = Calibration::None;
        continue;
      }
      if ((user->getKind() == glow::Kinded::Kind::Convolution3DNodeKind) &&
          (user->getNthInput(Convolution3DNode::BiasIdx) == nodeOutput)) {
        // Found bias for Convolution3DNode.
        precision = quantConfig.precisionBias;
        calibration = Calibration::None;
        continue;
      }
      if ((user->getKind() == glow::Kinded::Kind::ConvTransposeNodeKind) &&
          (user->getNthInput(ConvTransposeNode::BiasIdx) == nodeOutput)) {
        // Found bias for ConvTranspose.
        precision = quantConfig.precisionBias;
        calibration = Calibration::None;
        continue;
      }
      if ((user->getKind() == glow::Kinded::Kind::FullyConnectedNodeKind) &&
          (user->getNthInput(FullyConnectedNode::BiasIdx) == nodeOutput)) {
        // Found bias for FullyConnectedNode.
        precision = quantConfig.precisionBias;
        calibration = Calibration::None;
        continue;
      }
      if ((user->getKind() == glow::Kinded::Kind::BatchedAddNodeKind) &&
          (user->getNthInput(BatchedAddNode::SliceIdx) == nodeOutput)) {
        // Find out if this BatchAddNode was lowered from FullyConnectedNode.
        const auto *baN = llvm::cast<BatchedAddNode>(user);
        if (isBAFromLoweredFC(baN, loweredMap)) {
          // Found bias for lowered FullyConnectedNode.
          precision = quantConfig.precisionBias;
          calibration = Calibration::None;
          continue;
        }
      }
    }

    // Do not calibrate the quantization parameters for scalars.
    if (nodeOutput.getType()->size() == 1) {
      calibration = Calibration::None;
    }

    // Disable the quantization calibration for constant weights.
    if (!quantConfig.calibrateConstants &&
        llvm::isa<Constant>(nodeOutput.getNode())) {
      calibration = Calibration::None;
    }

    // Compute the TensorQuantizationParams using the profiling information
    // and the target precision and calibration.
    TensorProfilingParams TPP = profilingInfo.tensorProfilingParams_;
    TensorQuantizationParams TQP =
        chooseQuantizationParams(TPP, schema, precision, calibration);
    quantizationInfos.emplace_back(nodeOutputName, TQP);
  }

  return quantizationInfos;
}

void quantizeFunction(Function *F, const QuantizationConfiguration &quantConfig,
                      const Backend &B, const LoweredInfoMap &loweredMap,
                      const KindSet &doNotQuantizeKinds) {
  DCHECK(quantConfig.precision == ElemKind::Int8QTy ||
         quantConfig.precision == ElemKind::UInt8QTy ||
         quantConfig.precision == ElemKind::Int16QTy)
      << "Only Int8, UInt8, and Int16 quantization supported";

  FunctionQuantizer quantizer(*F, B, quantConfig, doNotQuantizeKinds,
                              loweredMap);
  quantizer.convert();

  // Enable rowwise quantization for FullyConnected node.
  if (quantConfig.enableRowwise) {
    quantizer.enableRowwise();
  }

  // Enable channelwise quantization for Convolution node.
  if (quantConfig.enableChannelwise) {
    quantizer.enableChannelwise();
  }
}

} // namespace quantization
} // namespace glow
