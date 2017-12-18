#ifndef GLOW_IR_IRBUILDER_H
#define GLOW_IR_IRBUILDER_H

#include "glow/Base/Type.h"
#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace glow {

/// The IRBuilder create the IR in the module.
class IRBuilder {
  using MutabilityKind = WeightVar::MutabilityKind;

  /// The module that we are building.
  Module *M_;

public:
  explicit IRBuilder(Module *M) : M_(M) {}

  ~IRBuilder();

  /// \returns Module of the current builder.
  Module &getModule() {
    assert(M_);
    return *M_;
  }

  /// @name High-level, operation-level IRBuilder.
  ///@{

  ConvolutionInst *createConvOp(Value *input, size_t depth, size_t kernel,
                                size_t stride, size_t pad);

  ConvolutionInst *createConvOp(Value *input, Value *filter, Value *bias,
                                size_t depth, size_t kernel, size_t stride,
                                size_t pad);

  PoolMaxInst *createPoolMaxOp(Value *input, size_t kernel, size_t stride,
                               size_t pad);

  PoolAvgInst *createPoolAvgOp(Value *input, size_t kernel, size_t stride,
                               size_t pad);

  FullyConnectedInst *createFullyConnectedOp(Value *input, size_t outDepth);

  FullyConnectedInst *createFullyConnectedOp(Value *input, Value *filter,
                                             Value *bias, size_t outDepth);

  ReluInst *createRELUOp(Value *input);

  SigmoidInst *createSigmoidOp(Value *input);

  TanhInst *createTanhOp(Value *input);

  SoftMaxInst *createSoftMaxOp(Value *input, Value *selected);

  ReshapeInst *createReshapeOp(Value *input, llvm::ArrayRef<size_t> shape);

  TransposeInst *createTransposeOp(Value *input,
                                   llvm::ArrayRef<unsigned> shuffle);

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
                                                 llvm::ArrayRef<size_t> dims) {
    auto T = M_->getGraph()->uniqueType(elemTy, dims);
    return createAllocActivationInst(name, T);
  }

// Import the auto-generated instruction creation methods:
#include "AutoGenIRBuilder.h"

  ///@}

  /// Inserts the deallocation instructions for all 'alloc' instructions that
  /// need to be terminated.
  void deallocateActiveInstrs();
};

} // namespace glow

#endif // GLOW_IR_IRBUILDER_H
