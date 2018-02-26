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
                                           size_t stride, size_t pad);

  PoolAvgInst *createPoolAvgOp(Value *input, size_t kernel, size_t stride,
                               size_t pad);

  SigmoidInst *createSigmoidOp(Value *input);

  TanhInst *createTanhOp(Value *input);

  SoftMaxWithLossInst *createSoftMaxWithLossOp(Value *input, Value *labels);

  CrossEntropyLossInst *createCrossEntropyLossOp(Value *P, Value *labels);

  ReshapeInst *createReshapeOp(Value *input, llvm::ArrayRef<size_t> shape);

  TensorViewInst *createTensorView(ElemKind elemKind,
                                   llvm::ArrayRef<size_t> dims, Value *src,
                                   llvm::StringRef name);

  TransposeInst *createTransposeOp(Value *input,
                                   llvm::ArrayRef<unsigned> shuffle);

  BroadcastInst *createBroadcastOp(Value *input, llvm::ArrayRef<size_t> shape,
                                   unsigned axis);

  BatchNormalizationInst *createBatchNormalizationOp(Value *input,
                                                     size_t channelIdx = 0,
                                                     float epsilon = 1e-5,
                                                     float momentum = 0.9);

  BatchNormalizationInst *
  createBatchNormalizationOp(Value *input, Value *beta, Value *gamma,
                             Value *mean, Value *var, size_t channelIdx = 0,
                             float epsilon = 1e-5, float momentum = 0.9);

  LocalResponseNormalizationInst *
  createLocalResponseNormalizationOp(Value *input, size_t halfWindowSize = 2,
                                     float alpha = 1e-4, float beta = 0.75,
                                     float k = 2.0);

  ElementAddInst *createElementAddOp(Value *LHS, Value *RHS);

  ElementSubInst *createElementSubOp(Value *LHS, Value *RHS);

  ElementMulInst *createElementMulOp(Value *LHS, Value *RHS);

  ElementDivInst *createElementDivOp(Value *LHS, Value *RHS);

  ElementMaxInst *createElementMaxOp(Value *LHS, Value *RHS);

  ElementMinInst *createElementMinOp(Value *LHS, Value *RHS);

  ElementCmpLTEInst *createElementCmpLTEOp(Value *LHS, Value *RHS);

  ElementSelectInst *createSelectOp(Value *Cond, Value *LHS, Value *RHS);

  TopKInst *createTopKOp(Value *input, size_t k);

  GatherInst *createGatherOp(Value *data, Value *indices);

  Value *createReturnOp(Value *input);

  ///@}

  /// @name Low-level, instruction-level IRBuilder.
  ///@{

  WeightVar *createWeightVar(TypeRef T, llvm::StringRef name = "",
                             MutabilityKind k = MutabilityKind::Mutable);

  WeightVar *createWeightVar(ElemKind elemTy, llvm::ArrayRef<size_t> dims,
                             llvm::StringRef name = "",
                             MutabilityKind k = MutabilityKind::Mutable);

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
