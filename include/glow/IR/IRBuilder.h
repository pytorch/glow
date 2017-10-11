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

  ConvolutionInst *createConvOp(Value *input, Value *filter, Value *bias,
                                size_t depth, size_t kernel, size_t stride,
                                size_t pad);

  PoolInst *createPoolOp(Value *input, PoolInst::OpKind kind, size_t kernel,
                         size_t stride, size_t pad);

  FullyConnectedInst *createFullyConnectedOp(Value *input, size_t outDepth);

  FullyConnectedInst *createFullyConnectedOp(Value *input, Value *filter,
                                             Value *bias, size_t outDepth);

  ReluInst *createRELUOp(Value *input);

  SigmoidInst *createSigmoidOp(Value *input);

  TanhInst *createTanhOp(Value *input);

  SoftMaxInst *createSoftMaxOp(Value *input, Value *selected);

  RegressionInst *createRegressionOp(Value *input, Value *expected);

  ReshapeInst *createReshapeOp(Value *input, llvm::ArrayRef<size_t> shape);

  TransposeInst *createTransposeOp(Value *input,
                                   llvm::ArrayRef<unsigned> shuffle);

  ConcatInst *createConcatOp(llvm::ArrayRef<Value *> inputs,
                             unsigned dimension);

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

  ArithmeticInst *createArithmeticOp(Value *LHS, Value *RHS,
                                     ArithmeticInst::OpKind op);

  Value *createReturnOp(Value *input);

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
                                 llvm::ArrayRef<size_t> shape);

  TransposeInst *createTransposeInst(Value *dest, Value *src,
                                     llvm::ArrayRef<unsigned> shuffle);

  ConcatInst *createConcatInst(Value *dest, llvm::ArrayRef<Value *> src,
                               size_t dim);

  BatchNormalizationInst *createBatchNormalizationInst(
      Value *dest, Value *src, Value *scale, Value *bias, Value *mean,
      Value *var, size_t channelIdx, float epsilon, float momentum);

  LocalResponseNormalizationInst *
  createLocalResponseNormalizationInst(Value *dest, Value *src, Value *scale,
                                       size_t halfWindowSize, float alpha,
                                       float beta, float k);

  ArithmeticInst *createArithmeticInst(Value *dest, Value *LHS, Value *RHS,
                                       ArithmeticInst::OpKind kind);

  WeightVar *createWeightVar(TypeRef T, llvm::StringRef name = "",
                             InitKind initKind = InitKind::Extern,
                             float val = 0);

  WeightVar *createWeightVar(ElemKind elemTy, llvm::ArrayRef<size_t> dims,
                             llvm::StringRef name = "",
                             InitKind initKind = InitKind::Extern,
                             float val = 0);

  AllocActivationInst *createAllocActivationInst(TypeRef T,
                                                 llvm::StringRef name = "");
  AllocActivationInst *createAllocActivationInst(ElemKind elemTy,
                                                 llvm::ArrayRef<size_t> dims,
                                                 llvm::StringRef name = "");

  DeallocActivationInst *createDeallocActivationInst(Value *src);

  ///@}

  /// Inserts the deallocation instructions for all 'alloc' instructions that
  /// need to be terminated.
  void deallocateActiveInstrs();
};

} // namespace glow

#endif // GLOW_IR_IRBUILDER_H
