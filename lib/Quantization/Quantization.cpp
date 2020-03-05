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

/// Support quantized Log \p LN inside \p F by replacing it with an
/// IntLookupTable with a new mapping given the input and output quantization
/// parameters. \returns the new IntLookupTable created.
static Node *replaceQuantizedLogWithLookupTable(Function &F,
                                                const LogNode &LN) {
  TypeRef outTy = LN.getResult().getType();
  TypeRef inTy = LN.getInput().getType();

  auto inputRange = inTy->getQuantizedValueRange();
  (void)inputRange;
  assert(inputRange.first >= 0 &&
         "Input range must not be negative since this is input to log().");

  // Pass a function returning log here to create the mapping. Note that the
  // interval is always extended to include zero, so we check if the input is
  // zero and if so use log(float min), i.e. closest positive value to zero,
  // as -inf is unsupported to convert to int.
  auto logFun = [](float a) {
    return (a == 0.0) ? log(std::numeric_limits<float>::min()) : log(a);
  };
  std::vector<int8_t> mapping =
      glow::quantization::createMapping(inTy, outTy, logFun);

  // Create a new int lookup table with this newly calculated mapping to
  // implement this quantized log.
  IntLookupTableNode *ILT = F.createIntLookupTable(
      LN.getName().str() + ".log", LN.getInput(), mapping, outTy);

  LN.getResult().replaceAllUsesOfWith(ILT);
  return ILT;
}

/// Support quantized Tanh \p TN inside \p F by replacing it with an
/// IntLookupTable. \returns final node in the chain implementing the quantized
/// Tanh via the IntLookupTable.
static Node *replaceQuantizedTanhWithLookupTable(Function &F,
                                                 const TanhNode &TN) {
  // Quantized tanh operator expects input to be in a certain floating point
  // range. This operator works based on the precomputed table and has to
  // process input in a range of [-3.0, 3.0]. Tanh asymptotically approaches
  // +/-1.0 and is already +/-.995 at +/-3.0.
  // The output quantization parameters are chosen to represent the floating
  // point range of [-1.0, 1.0].
  auto inputQuantizationParams =
      glow::quantization::chooseQuantizationParams(-3.0, 3.0);
  auto tanhInTy = F.getParent()->uniqueType(
      ElemKind::Int8QTy, TN.getResult().dims(), inputQuantizationParams.scale,
      inputQuantizationParams.offset);

  // Make sure input is clipped in [-3.0, 3.0] floating point range.
  auto *rescaleInputNode =
      F.createRescaleQuantized(TN.getName(), TN.getInput(), tanhInTy);

  // Make sure output is clipped in [-1.0, 1.0] floating point range.
  auto outputQuantizationParams =
      glow::quantization::chooseQuantizationParams(-1.0, 1.0);
  auto resultOutTy = F.getParent()->uniqueType(
      ElemKind::Int8QTy, rescaleInputNode->getResult().dims(),
      outputQuantizationParams.scale, outputQuantizationParams.offset);

  // Note: The actual lookup table is created inside this call.
  auto *quantizedNode =
      F.createIntTanh(TN.getName(), rescaleInputNode, resultOutTy);

  auto *rescaleOutputNode = F.createRescaleQuantized(
      TN.getName(), quantizedNode, TN.getResult().getType());

  TN.getResult().replaceAllUsesOfWith(rescaleOutputNode);
  return rescaleOutputNode;
}

/// Support quantized Sigmoid \p SN inside \p F by replacing it with an
/// IntLookupTable. \returns final node in the chain implementing the quantized
/// Sigmoid via the IntLookupTable.
static Node *replaceQuantizedSigmoidWithLookupTable(Function &F,
                                                    const SigmoidNode &SN) {
  // Quantized sigmoid operator expects input to be in a certain floating
  // point range. This operator works based on the precomputed table and has
  // to process input in a range of [-6.0, 6.0]. Sigmoid asymptotically
  // approaches 0 at -inf and 1 at +inf. It has values of 0.00247262 and
  // 0.997527 at -6.0 and 6.0 correspondingly. The output quantization
  // parameters are chosen to represent the floating point range of [0, 1.0].
  auto inputQuantizationParams =
      glow::quantization::chooseQuantizationParams(-6.0, 6.0);
  auto sigmoidInTy = F.getParent()->uniqueType(
      ElemKind::Int8QTy, SN.getResult().dims(), inputQuantizationParams.scale,
      inputQuantizationParams.offset);

  // Make sure input is clipped in [-6.0, 6.0] floating point range.
  auto *rescaleInputNode =
      F.createRescaleQuantized(SN.getName(), SN.getInput(), sigmoidInTy);

  // Make sure output is clipped in [0.0, 1.0] floating point range.
  auto outputQuantizationParams =
      glow::quantization::chooseQuantizationParams(0.0, 1.0);
  auto resultOutTy = F.getParent()->uniqueType(
      ElemKind::Int8QTy, rescaleInputNode->getResult().dims(),
      outputQuantizationParams.scale, outputQuantizationParams.offset);

  // Note: The actual lookup table is created inside this call.
  auto *quantizedNode =
      F.createIntSigmoid(SN.getName(), rescaleInputNode, resultOutTy);

  auto *rescaleOutputNode = F.createRescaleQuantized(
      SN.getName(), quantizedNode, SN.getResult().getType());

  SN.getResult().replaceAllUsesOfWith(rescaleOutputNode);
  return rescaleOutputNode;
}

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

    // Bias quantization is specialized for Convolution and Fully Connected:
    // - for int32 bias quantization: since the dynamic range of int32 is
    //   large we can force symmetric quantization (offset = 0). This allows
    //   a faster implementation since no offset subtraction is required.
    // - for int8/int16 bias quantization: since the dynamic range is small we
    //   will keep the original offset.
    // - regardless of precision, we try to force the bias scale parameter to
    //   bias_scale = input_scale * weights_scale since this has a performance
    //   benefit by specializing the parameters to biasPre = 0, biasPost = 0,
    //   biasScale = 1. We must verify that by changing the bias scale we don`t
    //   saturate the data. This is also equivalent to forcing the effective
    //   scale applied at run-time (bias_scale / (input_scale * weights_scale))
    //   to be always greater than or equal to 1.0 which is a common constraint
    //   for the bias for most libraries with quantized implementations.
    auto getBiasType = [&](TypeRef inputTy, TypeRef weightsTy) -> TypeRef {
      // Choose bias offset. For int32 bias we always force offset 0 in order
      // to simplify the implementation since the dynamic range allows it.
      int32_t biasOffset = TQP.offset;
      if (quantizationPrecisionBias_ == ElemKind::Int32QTy) {
        biasOffset = 0;
      }
      // Choose bias scale. We try to force the bias scale value to the product
      // input_scale * weights_scale but only if the resulting scale is larger
      // (in order to avoid bias data saturation).
      float scaleInput = inputTy->getScale();
      float scaleWeights = weightsTy->getScale();
      float biasScale = TQP.scale;
      if (scaleInput * scaleWeights > TQP.scale) {
        biasScale = scaleInput * scaleWeights;
      }
      return mod_.uniqueType(quantizationPrecisionBias_, val.dims(), biasScale,
                             biasOffset);
    };

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
  Node *createConversion(Function &function, const Node & /* unused */,
                         NodeValue &val, TypeRef destTy,
                         bool /* isInput */) override {
    assert((&function == &function_) &&
           "Trying to add quantize/dequantize conversion to a function other "
           "than the function being quantized.");
    if (destTy->isQuantizedType()) {
      return function.createQuantize("quantize", val, destTy);
    }
    assert(destTy->getElementType() == ElemKind::FloatTy &&
           "Can't dequantize to any type except float.");
    return function.createDequantize("dequantize", val);
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
  CASE_SINGLE_MATCHING_INOUT_TYPE(TopK, Input, Values):                        \
  CASE_SINGLE_MATCHING_INOUT_TYPE(Gather, Data, Result):                       \
  CASE_SINGLE_MATCHING_INOUT_TYPE(MaxPool, Input, Result)
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
            function_, llvm::cast<LogNode>(node));
        break;
      case Kinded::Kind::TanhNodeKind:
        quantizedNode = replaceQuantizedTanhWithLookupTable(
            function_, llvm::cast<TanhNode>(node));
        break;
      case Kinded::Kind::SigmoidNodeKind:
        quantizedNode = replaceQuantizedSigmoidWithLookupTable(
            function_, llvm::cast<SigmoidNode>(node));
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

public:
  /// Creates a function quantizer for \p F using the quantization
  /// parameters defined by \p quantizationInfos and target quantization
  /// precision defined by \p quantizationPrecision.
  /// \p B and \p doNotQuantizeKinds are used to check which nodes shouldn't be
  /// converted. \p assertAllNodesQuantized is used as a debugging tool; if
  /// true then if the backend does not support a node as quantized for the
  /// given \p quantizationPrecision then the program will exit with an error.
  FunctionQuantizer(Function &F, const Backend &B, quantization::Schema schema,
                    llvm::ArrayRef<NodeQuantizationInfo> quantizationInfos,
                    ElemKind quantizationPrecision,
                    const KindSet &doNotQuantizeKinds,
                    const LoweredInfoMap &loweredMap,
                    bool assertAllNodesQuantized,
                    ElemKind quantizationPrecisionBias)
      : FunctionConverter(F), mod_(*F.getParent()), B_(B), schema_(schema),
        quantizationPrecision_(quantizationPrecision),
        doNotQuantizeKinds_(doNotQuantizeKinds), loweredMap_(loweredMap),
        assertAllNodesQuantized_(assertAllNodesQuantized),
        quantizationPrecisionBias_(quantizationPrecisionBias) {
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
            bias.getType()->isQuantizedType() &&
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
}; // namespace

} // namespace

namespace glow {
namespace quantization {

/// Helper which, given the output name \p currName of some node, looks for
/// corresponding names in \p loweredMap which represent any names that this
/// node was lowered from. If any are found then they are inserted into \p
/// quantizationInfos along with \p TQP.
static void
findAndInsertLoweredInfos(llvm::StringRef currName,
                          const LoweredInfoMap &loweredMap,
                          std::vector<NodeQuantizationInfo> &quantizationInfos,
                          const TensorQuantizationParams &TQP) {
  auto currSetIt = loweredMap.find(currName);
  if (currSetIt == loweredMap.end()) {
    return;
  }

  // Get the set of names corresponding to currName. All names in the set are
  // names that were originally lowered into currName.
  auto &currSet = currSetIt->getValue();

  // For each of the names (currOrigName), insert them into quantizationInfos,
  // and then recursively find and insert other names in case currOrigName was
  // also lowered from a previous node.
  for (auto i = currSet.begin(), e = currSet.end(); i != e; ++i) {
    llvm::StringRef currOrigName = i->getName();
    quantizationInfos.emplace_back(currOrigName, TQP);
    findAndInsertLoweredInfos(currOrigName, loweredMap, quantizationInfos, TQP);
  }
}

std::vector<NodeQuantizationInfo>
generateNodeQuantizationInfos(PlaceholderBindings &bindings, const Function *F,
                              const LoweredInfoMap &loweredMap, Schema schema,
                              ElemKind quantizationPrecision,
                              ElemKind quantizationPrecisionBias) {
  std::vector<NodeQuantizationInfo> quantizationInfos;

  for (auto &node : F->getNodes()) {
    auto *QPN = llvm::dyn_cast<QuantizationProfileNode>(&node);

    if (QPN) {
      auto CI = bindings.get(QPN->getComputationInfoPlaceholder())
                    ->getHandle<float>();
      auto histogram =
          bindings.get(QPN->getHistogramPlaceholder())->getHandle<float>();
      float min = CI.raw(0);
      float max = CI.raw(1);

      std::string fullOutputName = NodeValue::generateNodeOutputName(
          QPN->getProfiledNodeName(), QPN->getProfiledOutputNumber());

      ElemKind qPrec = quantizationPrecision;

      // During profiling, for a given node, the TQP params must be computed
      // according to the target precision used during actual quantization.
      // The code below reflects the same logic as the one used in the function
      // FunctionQuantizer::getTargetTypeForInput for specializing the bias
      // quantization precision.
      // TODO: For better clarity and to remove logic duplication this code
      // should be coupled tighter with the logic used in FunctionQuantizer.
      NodeValue profNode = QPN->getInput();
      for (const auto &use : profNode.getUsers()) {
        const auto *user = use.getUser();
        if ((user->getKind() == glow::Kinded::Kind::ConvolutionNodeKind) &&
            (user->getNthInput(ConvolutionNode::BiasIdx) == profNode)) {
          // Found bias for ConvolutionNode.
          qPrec = quantizationPrecisionBias;
          continue;
        }
        if ((user->getKind() == glow::Kinded::Kind::Convolution3DNodeKind) &&
            (user->getNthInput(Convolution3DNode::BiasIdx) == profNode)) {
          // Found bias for Convolution3DNode.
          qPrec = quantizationPrecisionBias;
          continue;
        }
        if ((user->getKind() == glow::Kinded::Kind::FullyConnectedNodeKind) &&
            (user->getNthInput(FullyConnectedNode::BiasIdx) == profNode)) {
          // Found bias for FullyConnectedNode.
          qPrec = quantizationPrecisionBias;
          continue;
        }
        if ((user->getKind() == glow::Kinded::Kind::BatchedAddNodeKind) &&
            (user->getNthInput(BatchedAddNode::SliceIdx) == profNode)) {
          // Find out if this BatchAddNode was lowered from FullyConnectedNode.
          const auto *baN = llvm::cast<BatchedAddNode>(user);
          if (isBAFromLoweredFC(baN, loweredMap)) {
            // Found bias for lowered FullyConnectedNode.
            qPrec = quantizationPrecisionBias;
            continue;
          }
        }
      }

      // TODO: Ideally tensor quantization params should be calculated
      // based on the histogram distribution. Use simplistic approach for now.
      (void)histogram;
      TensorQuantizationParams TQP =
          chooseQuantizationParams(min, max, schema, qPrec);

      quantizationInfos.emplace_back(fullOutputName, TQP);

      // If the NodeValue represented by fullOutputName was created via lowering
      // another original NodeValue, then generate node quantization info for
      // the original NodeValue using the same quantization parameters.
      findAndInsertLoweredInfos(fullOutputName, loweredMap, quantizationInfos,
                                TQP);
    }
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

  FunctionQuantizer quantizer(*F, B, quantConfig.schema, quantConfig.infos,
                              quantConfig.precision, doNotQuantizeKinds,
                              loweredMap, quantConfig.assertAllNodesQuantized,
                              quantConfig.precisionBias);
  quantizer.convert();
  if (quantConfig.enableRowwise) {
    quantizer.enableRowwise();
  }
}

} // namespace quantization
} // namespace glow
