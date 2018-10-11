/**
 * Copyright (c) 2018-present, Facebook, Inc.
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

namespace glow {

/// This helper class converts values of typeA into values of typeB.
/// The nodes producing these values are morphed into the new type
/// and proper conversions from/to typeA and typeB are inserted.
class TypeAToTypeBFunctionConverter : public FunctionConverter {
protected:
  /// Destination type of the conversions.
  ElemKind dstKind_;
  /// Source type of the conversions. I.e., the values of this
  /// element type are going to be converted.
  ElemKind srcKind_;
  /// Set of node kinds that should not be converted.
  KindSet doNotConvertKinds_;

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

  /// Create a node that converts \p val to \p destTy.
  /// \p val and \p destTy must have the same shape.
  Node *createConversion(NodeValue &val, TypeRef destTy) override;

  /// Check if \p node can be converted.
  bool canConvert(const Node &node) const override;

public:
  /// Create a type converter from \p fromKind to \p toKind for \p F.
  /// If \p doNotConvertKinds is not nullptr, the nodes which kind
  /// is in this set won't be converted.
  TypeAToTypeBFunctionConverter(Function &F, ElemKind fromKind, ElemKind toKind,
                                const KindSet *doNotConvertKinds = nullptr);
};
} // namespace glow
#endif
