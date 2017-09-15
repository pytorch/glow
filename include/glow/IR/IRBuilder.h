#ifndef GLOW_IR_IRBUILDER_H
#define GLOW_IR_IRBUILDER_H

#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "glow/IR/Type.h"

namespace glow {

/// The IRBuilder create the IR in the module.
class IRBuilder {
  using InitKind = StaticVariable::InitKind;

  /// The module that we are building.
  Module &M_;

public:
  IRBuilder(Module &M) : M_(M) {}

  /// @name High-level, operation-level IRBuilder.
  ///@{

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

  SoftMaxInst *createSoftMaxInst(Value *dest, Value *src, Value *expected);

  RegressionInst *createRegressionInst(Value *dest, Value *src,
                                       Value *expected);

  TransposeInst *createTransposeInst(Value *dest, Value *src,
                                     ArrayRef<unsigned> shuffle);

  ConcatInst *createConcatInst(Value *dest, ArrayRef<Value *> src, size_t dim);

  BatchNormalizationInst *createBatchNormalizationInst(
      Value *dest, Value *src, Value *scale, Value *bias, Value *mean,
      Value *var, size_t channelIdx, float epsilon, float momentum);

  ArithmeticInst *createArithmeticInst(Value *dest, Value *LHS, Value *RHS,
                                       ArithmeticInst::OpKind kind);

  StaticVariable *createStaticVariable(ElemKind elemTy, ArrayRef<size_t> dims,
                                       InitKind mode = InitKind::kExtern,
                                       float val = 0);

  StaticVariable *createStaticVariable(TypeRef T,
                                       InitKind mode = InitKind::kExtern,
                                       float val = 0);
  ///@}
};

} // namespace glow

#endif // GLOW_IR_IRBUILDER_H
