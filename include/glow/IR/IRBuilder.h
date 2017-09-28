#ifndef GLOW_IR_IRBUILDER_H
#define GLOW_IR_IRBUILDER_H

#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "glow/IR/Type.h"

namespace glow {

/// The IRBuilder create the IR in the module.
class IRBuilder {
  using InitKind = WeightVar::InitKind;

  /// The module that we are building.
  Module &M_;

  /// A list of allocated buffers that need to be deallocated at the end of the
  /// program that we are constructing.
  std::vector<AllocActivationInst *> activeAllocs_;

public:
  IRBuilder(Module &M) : M_(M) {}

  ~IRBuilder();

  /// @name High-level, operation-level IRBuilder.
  ///@{

  ConvolutionInst *createConvOp(Value *input, size_t depth, size_t kernel,
                                size_t stride, size_t pad);

  PoolInst *createPoolOp(Value *input, PoolInst::OpKind kind, size_t kernel,
                         size_t stride, size_t pad);

  FullyConnectedInst *createFullyConnectedOp(Value *input, size_t outDepth);

  ReluInst *createRELUOp(Value *input);

  SigmoidInst *createSigmoidOp(Value *input);

  TanhInst *createTanhOp(Value *input);

  SoftMaxInst *createSoftMaxOp(Value *input, Value *selected);

  RegressionInst *createRegressionOp(Value *input, Value *expected);

  ReshapeInst *createReshapeOp(Value *input, ArrayRef<size_t> shape);

  TransposeInst *createTransposeOp(Value *input, ArrayRef<unsigned> shuffle);

  ConcatInst *createConcatOp(ArrayRef<Value *> inputs, unsigned dimension);

  BatchNormalizationInst *createBatchNormalizationOp(Value *input,
                                                     size_t channelIdx = 0,
                                                     float epsilon = 1e-5,
                                                     float momentum = 0.9);

  ArithmeticInst *createArithmeticOp(Value *LHS, Value *RHS,
                                     ArithmeticInst::OpKind op);

  ///@}

  /// @name Low-level, instruction-level IRBuilder.
  ///@{
  CopyInst *createCopyInst(Value *dest, Value *src);

  ConvolutionInst *createConvolutionInst(Value *dest, Value *src, Value *filter,
                                         Value *bias, size_t kernel,
                                         size_t stride, size_t pad,
                                         size_t depth);

  PoolInst *createPoolInst(Value *dest, Value *src, Value *srcXY,
                           PoolInst::OpKind kind, size_t kernel, size_t stride,
                           size_t pad);

  FullyConnectedInst *createFullyConnectedInst(Value *dest, Value *src,
                                               Value *filter, Value *bias,
                                               size_t depth);

  ReluInst *createReluInst(Value *dest, Value *src);

  SigmoidInst *createSigmoidInst(Value *dest, Value *src);

  TanhInst *createTanhInst(Value *dest, Value *src);

  SoftMaxInst *createSoftMaxInst(Value *dest, Value *src, Value *E,
                                 Value *selected);

  RegressionInst *createRegressionInst(Value *dest, Value *src,
                                       Value *expected);

  ReshapeInst *createReshapeInst(Value *dest, Value *src,
                                 ArrayRef<size_t> shape);

  TransposeInst *createTransposeInst(Value *dest, Value *src,
                                     ArrayRef<unsigned> shuffle);

  ConcatInst *createConcatInst(Value *dest, ArrayRef<Value *> src, size_t dim);

  BatchNormalizationInst *createBatchNormalizationInst(
      Value *dest, Value *src, Value *scale, Value *bias, Value *mean,
      Value *var, size_t channelIdx, float epsilon, float momentum);

  ArithmeticInst *createArithmeticInst(Value *dest, Value *LHS, Value *RHS,
                                       ArithmeticInst::OpKind kind);

  WeightVar *createWeightVar(TypeRef T, StringRef name = "",
                             InitKind initKind = InitKind::kExtern,
                             float val = 0);

  WeightVar *createWeightVar(ElemKind elemTy, ArrayRef<size_t> dims,
                             StringRef name = "",
                             InitKind initKind = InitKind::kExtern,
                             float val = 0);

  AllocActivationInst *createAllocActivationInst(TypeRef T,
                                                 StringRef name = "");
  AllocActivationInst *createAllocActivationInst(ElemKind elemTy,
                                                 ArrayRef<size_t> dims,
                                                 StringRef name = "");

  DeallocActivationInst *createDeallocActivationInst(Value *src);

  ///@}

  /// Inserts the deallocation instructions for all 'alloc' instructions that
  /// need to be terminated.
  void deallocateActiveInstrs();
};

} // namespace glow

#endif // GLOW_IR_IRBUILDER_H
