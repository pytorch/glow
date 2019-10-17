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
                                                      TypeRef destTy) {
  assert(((destTy->getElementType() == dstKind_ &&
           val.getType()->getElementType() == srcKind_) ||
          (destTy->getElementType() == srcKind_ &&
           val.getType()->getElementType() == dstKind_)) &&
         "Unexpected conversion type");

  bool needClip = precConfig_.clipFP16;
  if (needClip) {
    switch (node.getKind()) {
    case Kinded::Kind::ConcatNodeKind:
    case Kinded::Kind::GatherNodeKind:
    case Kinded::Kind::ReshapeNodeKind:
    case Kinded::Kind::SliceNodeKind:
    case Kinded::Kind::TransposeNodeKind:
      needClip = false;
      break;
    default:
      break;
    }
  }
  if (needClip) {
    assert((destTy->getElementType() == ElemKind::Float16Ty ||
            val.getType()->getElementType() == ElemKind::Float16Ty) &&
           "Unexpected conversion type");
    constexpr float float16Max = 65504.0f;
    constexpr float float16Min = -65504.0f;

    // If the input is fp32 and output is fp16, then we want to do the convert
    // before the clip. This way the clip can execute in fp16 mode.
    if (destTy->getElementType() == ElemKind::Float16Ty &&
        val.getType()->getElementType() == ElemKind::FloatTy) {
      auto convert =
          function.createConvertTo(val.getNode()->getName(), val, destTy);
      return function.createClip(val.getNode()->getName(), convert, float16Min,
                                 float16Max);
    } else {
      auto clip = function.createClip(val.getNode()->getName(), val, float16Min,
                                      float16Max);
      return function.createConvertTo(val.getNode()->getName(), clip, destTy);
    }
  } else {
    return function.createConvertTo(val.getNode()->getName(), val, destTy);
  }
}

void TypeAToTypeBFunctionConverter::convertTensor(Tensor &tensor,
                                                  TypeRef destTy) {
  assert(destTy->getElementType() == dstKind_);
  tensor.convertToType(dstKind_);
}
