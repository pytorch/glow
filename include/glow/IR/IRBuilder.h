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
#ifndef GLOW_IR_IRBUILDER_H
#define GLOW_IR_IRBUILDER_H

#include "glow/Base/Type.h"
#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace glow {

/// The IRBuilder constructs the IR in the function.
class IRBuilder {
  using MutabilityKind = WeightVar::MutabilityKind;

  /// The function that we are building.
  IRFunction *F_;

  /// \returns a unique legal name that's based on the string \p name.  Legal
  /// names are legal C identifiers in the form: "[a-zA-Z_][a-zA-Z0-9_]*".
  llvm::StringRef uniqueName(llvm::StringRef name) {
    return F_->uniqueName(name);
  }

public:
  explicit IRBuilder(IRFunction *F) : F_(F) {}

  ~IRBuilder();

  /// \returns the function of the current builder.
  IRFunction &getIRFunction() {
    assert(F_);
    return *F_;
  }

  /// @name High-level, operation-level IRBuilder.
  ///@{

  PoolMaxWithXYInst *createPoolMaxWithXYOp(Value *input, size_t kernel,
                                           size_t stride,
                                           llvm::ArrayRef<size_t> pads);

  PoolAvgInst *createPoolAvgOp(Value *input, size_t kernel, size_t stride,
                               llvm::ArrayRef<size_t> pads);

  CrossEntropyLossInst *createCrossEntropyLossOp(Value *P, Value *labels);

  TensorViewInst *createTensorView(ElemKind elemKind,
                                   llvm::ArrayRef<size_t> dims, Value *src,
                                   llvm::StringRef name,
                                   llvm::ArrayRef<size_t> offsets = {});

  LocalResponseNormalizationInst *
  createLocalResponseNormalizationOp(Value *input, size_t halfWindowSize = 2,
                                     float alpha = 1e-4, float beta = 0.75,
                                     float k = 2.0);

  TopKInst *createTopKOp(Value *input, size_t k);

  Value *createReturnOp(Value *input);

  ///@}

  /// @name Low-level, instruction-level IRBuilder.
  ///@{

  WeightVar *createWeightVar(TypeRef T, llvm::StringRef name = "",
                             MutabilityKind m = MutabilityKind::Mutable,
                             VisibilityKind v = VisibilityKind::Private);

  WeightVar *createWeightVar(ElemKind elemTy, llvm::ArrayRef<size_t> dims,
                             llvm::StringRef name = "",
                             MutabilityKind m = MutabilityKind::Mutable,
                             VisibilityKind v = VisibilityKind::Private);

  AllocActivationInst *createAllocActivationInst(llvm::StringRef name,
                                                 ElemKind elemTy,
                                                 llvm::ArrayRef<size_t> dims);

// Import the auto-generated instruction creation methods:
#include "AutoGenIRBuilder.h"

  ///@}

  /// Inserts the deallocation instructions for all 'alloc' instructions that
  /// need to be terminated.
  void deallocateActiveInstrs();
};

} // namespace glow

#endif // GLOW_IR_IRBUILDER_H
