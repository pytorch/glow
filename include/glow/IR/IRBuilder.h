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

  MaxPoolWithArgmaxInst *createMaxPoolWithArgmaxOp(
      llvm::StringRef name, Value *input, llvm::ArrayRef<unsigned_t> kernels,
      llvm::ArrayRef<unsigned_t> strides, llvm::ArrayRef<unsigned_t> pads,
      unsigned_t layout, ElemKind argMaxIndicesTy, bool flattenIndices = true);

  ArgMaxInst *createArgMaxOp(llvm::StringRef name, Value *input,
                             unsigned_t axis, bool keepDims,
                             ElemKind outIndicesTy);

  AvgPoolInst *createAvgPoolOp(Value *input, llvm::ArrayRef<unsigned_t> kernels,
                               llvm::ArrayRef<unsigned_t> strides,
                               llvm::ArrayRef<unsigned_t> pads,
                               unsigned_t layout, bool countIncludePads);

  CrossEntropyLossInst *createCrossEntropyLossOp(llvm::StringRef name, Value *P,
                                                 Value *labels);

  TensorViewInst *createTensorView(ElemKind elemKind,
                                   llvm::ArrayRef<dim_t> dims, Value *src,
                                   llvm::StringRef name,
                                   llvm::ArrayRef<dim_t> offsets = {});

  LocalResponseNormalizationInst *createLocalResponseNormalizationOp(
      llvm::StringRef name, Value *input, size_t halfWindowSize = 2,
      float alpha = 1e-4, float beta = 0.75, float k = 2.0);

  TopKInst *createTopKOp(llvm::StringRef name, Value *input, size_t k,
                         ElemKind outIndicesTy);

  Value *createReturnOp(Value *input);

  ///@}

  /// @name Low-level, instruction-level IRBuilder.
  ///@{

  WeightVar *createWeightVar(TypeRef T, llvm::StringRef name = "",
                             MutabilityKind m = MutabilityKind::Mutable);

  WeightVar *createWeightVar(ElemKind elemTy, llvm::ArrayRef<dim_t> dims,
                             llvm::StringRef name = "",
                             MutabilityKind m = MutabilityKind::Mutable);

  WeightVar *createWeightVar(ElemKind elemTy, llvm::ArrayRef<dim_t> dims,
                             float scale, int32_t offset,
                             llvm::StringRef name = "",
                             MutabilityKind m = MutabilityKind::Mutable);

  AllocActivationInst *createAllocActivationInst(llvm::StringRef name,
                                                 ElemKind elemTy,
                                                 llvm::ArrayRef<dim_t> dims);

// Import the auto-generated instruction creation methods:
#include "glow/AutoGenIRBuilder.h"

  ///@}

  /// Inserts the deallocation instructions for all 'alloc' instructions that
  /// need to be terminated.
  void deallocateActiveInstrs();
};

} // namespace glow

#endif // GLOW_IR_IRBUILDER_H
