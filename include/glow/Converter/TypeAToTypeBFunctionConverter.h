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
#ifndef GLOW_CONVERTER_TYPEATOTYPEBFUNCTIONCONVERTER_H
#define GLOW_CONVERTER_TYPEATOTYPEBFUNCTIONCONVERTER_H

#include "FunctionConverter.h"

#include "glow/Base/Traits.h" // For KindSet.
#include "glow/Base/Type.h"
#include "glow/Optimizer/GraphOptimizer/CompilationContext.h"

namespace glow {

class Function;
class Module;

/// This helper class converts values of typeA into values of typeB.
/// The nodes producing these values are morphed into the new type
/// and proper conversions from/to typeA and typeB are inserted.
class TypeAToTypeBFunctionConverter : public FunctionConverter {
protected:
  /// Module of the function to be converted.
  Module &mod_;
  /// Destination type of the conversions.
  ElemKind dstKind_;
  /// Source type of the conversions. I.e., the values of this
  /// element type are going to be converted.
  ElemKind srcKind_;
  /// Precision configuration used during conversion.
  const PrecisionConfiguration &precConfig_;

  /// If the element type of \p out is srcKind_ returns a similarly shaped type
  /// using dstKind_. Otherwise returns nullptr.
  /// \see FunctionConverter::getTargetTypeForOutput to know how this
  /// is used.
  TypeRef getTargetTypeForOutput(const NodeValue &out) const override;

  /// If the element type of the \p idx-th input of \p use is srcKind_
  /// returns a similarly shaped type using dstKind_. Otherwise returns nullptr.
  /// \see FunctionConverter::getTargetTypeForInput to know how this
  /// is used.
  TypeRef getTargetTypeForInput(const Node &use, unsigned idx) const override;

  /// Create a node in \p function that converts \p val to \p destTy, given
  /// context \p node. \p val and \p destTy must have the same shape.
  Node *createConversion(Function &function, const Node &node, NodeValue &val,
                         TypeRef destTy, bool isInput) override;

  /// Check if \p node can be converted.
  bool canConvert(const Node &node) const override;

  void convertTensor(Tensor &tensor, TypeRef destTy) override;

public:
  /// Create a type converter from \p fromKind to \p toKind for \p F given
  /// \p precConfig.
  TypeAToTypeBFunctionConverter(Function &F, ElemKind fromKind, ElemKind toKind,
                                const PrecisionConfiguration &precConfig);

  /// Convert and clip all Storage nodes used by the function.
  void convertAndClipStorage();
};
} // namespace glow
#endif
