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

#include "glow/Converter/TypeAToTypeBFunctionConverter.h"

#include "glow/Base/Tensor.h"
#include "glow/Graph/Graph.h"

using namespace glow;

TypeAToTypeBFunctionConverter::TypeAToTypeBFunctionConverter(
    Function &F, ElemKind fromKind, ElemKind toKind,
    const PrecisionConfiguration &precConfig)
    : FunctionConverter(F), mod_(*F.getParent()), dstKind_(toKind),
      srcKind_(fromKind), precConfig_(precConfig) {}

bool TypeAToTypeBFunctionConverter::canConvert(const Node &node) const {
  // For some ops, if we're converting to FP16/BFloat16 and the bias is FP32 and
  // the input is quantized then don't convert to FP16/BFloat16.
  if (srcKind_ == ElemKind::FloatTy &&
      (dstKind_ == ElemKind::Float16Ty || dstKind_ == ElemKind::BFloat16Ty)) {
#define QUANT_INPUT_FLOAT_BIAS_CASE(NODE_NAME_)                                \
  case glow::Kinded::Kind::NODE_NAME_##NodeKind: {                             \
    auto *N = llvm::cast<NODE_NAME_##Node>(&node);                             \
    if (N->getBias().getType()->getElementType() == ElemKind::FloatTy &&       \
        N->getInput().getType()->isQuantizedType()) {                          \
      return false;                                                            \
    }                                                                          \
    break;                                                                     \
  }

#define QUANT_OR_FP16_INPUT_FLOAT_BIAS_CASE(NODE_NAME_)                        \
  case glow::Kinded::Kind::NODE_NAME_##NodeKind: {                             \
    auto *N = llvm::cast<NODE_NAME_##Node>(&node);                             \
    if (N->getBias().getType()->getElementType() == ElemKind::FloatTy &&       \
        (N->getInput().getType()->isQuantizedType() ||                         \
         N->getInput().getType()->getElementType() == ElemKind::Float16Ty)) {  \
      return false;                                                            \
    }                                                                          \
    break;                                                                     \
  }

    switch (node.getKind()) {
      QUANT_INPUT_FLOAT_BIAS_CASE(FullyConnected);
      QUANT_INPUT_FLOAT_BIAS_CASE(RowwiseQuantizedFullyConnected);
      QUANT_INPUT_FLOAT_BIAS_CASE(Convolution);
      QUANT_INPUT_FLOAT_BIAS_CASE(ConvTranspose);
      QUANT_INPUT_FLOAT_BIAS_CASE(Convolution3D);
      QUANT_INPUT_FLOAT_BIAS_CASE(ChannelwiseQuantizedConvolution);
      QUANT_OR_FP16_INPUT_FLOAT_BIAS_CASE(BatchNormalization);
    default:
      break;
    }
#undef QUANT_INPUT_FLOAT_BIAS_CASE
  }

  const bool inSet = precConfig_.precisionModeKindSet.count(node.getKind());
  const bool allowConversion = precConfig_.useSetAsWhitelist ? inSet : !inSet;

  if (!allowConversion) {
    return false;
  }
  return FunctionConverter::canConvert(node);
}

TypeRef TypeAToTypeBFunctionConverter::getTargetTypeForOutput(
    const NodeValue &out) const {
  if (out.getType()->getElementType() != srcKind_) {
    return nullptr;
  }
  return mod_.uniqueType(dstKind_, out.dims());
}

TypeRef
TypeAToTypeBFunctionConverter::getTargetTypeForInput(const Node &use,
                                                     unsigned idx) const {
  return getTargetTypeForOutput(use.getNthInput(idx));
}

Node *TypeAToTypeBFunctionConverter::createConversion(Function &function,
                                                      const Node &node,
                                                      NodeValue &val,
                                                      TypeRef destTy,
                                                      bool isInput) {
  assert(((destTy->getElementType() == dstKind_ &&
           val.getType()->getElementType() == srcKind_) ||
          (destTy->getElementType() == srcKind_ &&
           val.getType()->getElementType() == dstKind_)) &&
         "Unexpected conversion type");

  bool needClip =
      ((dstKind_ == ElemKind::Float16Ty || dstKind_ == ElemKind::BFloat16Ty) &&
       precConfig_.clipFP16 && !(isInput && precConfig_.clipFP16SkipInputs));
  if (needClip) {
    switch (node.getKind()) {
    case Kinded::Kind::ConcatNodeKind:
    case Kinded::Kind::GatherNodeKind:
    case Kinded::Kind::ReshapeNodeKind:
    case Kinded::Kind::SliceNodeKind:
    case Kinded::Kind::TransposeNodeKind:
      needClip = false;
      break;
    case Kinded::Kind::SigmoidNodeKind:
    case Kinded::Kind::TanhNodeKind:
      needClip = isInput;
      break;
    case Kinded::Kind::ConvertToNodeKind:
      needClip = (llvm::dyn_cast<const ConvertToNode>(&node)
                          ->getInput()
                          .getElementType() == ElemKind::Float16Ty ||
                  llvm::dyn_cast<const ConvertToNode>(&node)
                          ->getInput()
                          .getElementType() == ElemKind::BFloat16Ty)
                     ? false
                     : true;
      break;
    default:
      break;
    }
  }
  if (needClip) {
    assert((destTy->getElementType() == ElemKind::Float16Ty ||
            val.getType()->getElementType() == ElemKind::Float16Ty ||
            destTy->getElementType() == ElemKind::BFloat16Ty ||
            val.getType()->getElementType() == ElemKind::BFloat16Ty) &&
           "Unexpected conversion type");
    // If the input is fp32 and output is fp16, then we want to do the convert
    // before the clip. This way the clip can execute in fp16 mode.
    if (destTy->getElementType() == ElemKind::Float16Ty &&
        val.getType()->getElementType() == ElemKind::FloatTy) {
      auto convert = function.createConvertTo(
          val.getNode()->getName().str() + "_converted", val, destTy);
      return function.createClipMinMaxFP16(
          val.getNode()->getName().str() + "_clipped", convert);
    } else if (destTy->getElementType() == ElemKind::BFloat16Ty &&
               val.getType()->getElementType() == ElemKind::FloatTy) {
      auto convert = function.createConvertTo(
          val.getNode()->getName().str() + "_converted", val, destTy);
      return function.createClipMinMaxBFloat16(
          val.getNode()->getName().str() + "_clipped", convert);
    } else {
      auto clip = function.createClipMinMaxFP16(
          val.getNode()->getName().str() + "_clipped", val);
      return function.createConvertTo(
          val.getNode()->getName().str() + "_converted", clip, destTy);
    }
  } else {
    return function.createConvertTo(
        val.getNode()->getName().str() + "_converted", val, destTy);
  }
}

void TypeAToTypeBFunctionConverter::convertTensor(Tensor &tensor,
                                                  TypeRef destTy) {
  assert(destTy->getElementType() == dstKind_);
  tensor.convertToType(dstKind_);
}

void convertAndClipStorageHelper(
    Storage &S, Function &F, bool clipFloat16,
    PrecisionConfiguration::Float16Format float16Format, ElemKind srcKind,
    ElemKind dstKind) {
  if (S.getOutput().getType()->getElementType() != srcKind) {
    return;
  }

  ConvertToNode *convertToFloat16 = F.createConvertTo(
      S.getName().str() + "convert_to", S.getOutput(), dstKind);

  NodeValue NV = convertToFloat16->getResult();
  if (clipFloat16) {
    switch (float16Format) {
    case PrecisionConfiguration::Float16Format::FP16:
      NV = F.createClipMinMaxFP16(S.getName().str() + "_clipped", NV)
               ->getResult();
      break;
    case PrecisionConfiguration::Float16Format::BFloat16:
      NV = F.createClipMinMaxBFloat16(S.getName().str() + "_clipped", NV)
               ->getResult();
      break;
    default:
      llvm_unreachable("Unknown float16 format");
    }
  }

  // We have to convert back to the srcKind now as the users currently must be
  // expecting FP32. The optimizer will remove if possible.
  NodeValue convertBack =
      F.createConvertTo(NV.getNode()->getName().str() + "convert_back", NV,
                        srcKind)
          ->getResult();

  // We need to specify to skip replacing convertToFloat16 here as otherwise we
  // will create a cycle in the graph.
  S.getOutput().replaceAllUsesOfWith(convertBack, &F, convertToFloat16);
}

void TypeAToTypeBFunctionConverter::convertAndClipStorage() {
  if (precConfig_.convertPlaceholdersToFP16) {
    for (Placeholder *PH : function_.findPlaceholders()) {
      // If the PH is not used as an input then we do not clip it.
      if (!isInput(PH, function_)) {
        continue;
      }
      convertAndClipStorageHelper(*PH, function_, precConfig_.clipFP16,
                                  precConfig_.float16Format, srcKind_,
                                  dstKind_);
    }
  }
  if (precConfig_.convertConstantsToFP16) {
    for (Constant *C : function_.findConstants()) {
      convertAndClipStorageHelper(*C, function_, precConfig_.clipFP16,
                                  precConfig_.float16Format, srcKind_,
                                  dstKind_);
    }
  }
}
