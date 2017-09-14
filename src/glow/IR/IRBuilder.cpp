#include "glow/IR/IRBuilder.h"

using namespace glow;

CopyInst *IRBuilder::createCopyInst(Value *dest, Value *src) {
  auto *A = new CopyInst(dest, src);
  M_.pushInstr(A);
  return A;
}

ConvolutionInst *IRBuilder::createConvolutionInst(Value *dest, Value *src,
                                                  Value *filter, Value *bias,
                                                  size_t kernel, size_t stride,
                                                  size_t pad, size_t depth) {
  auto *A =
      new ConvolutionInst(dest, src, filter, bias, kernel, stride, pad, depth);
  M_.pushInstr(A);
  return A;
}

PoolInst *IRBuilder::createPoolInst(Value *dest, Value *src, Value *srcXY,
                                    PoolInst::OpKind kind, size_t kernel,
                                    size_t stride, size_t pad) {
  auto *A = new PoolInst(dest, src, srcXY, kind, kernel, stride, pad);
  M_.pushInstr(A);
  return A;
}

FullyConnectedInst *IRBuilder::createFullyConnectedInst(Value *dest, Value *src,
                                                        Value *filter,
                                                        Value *bias,
                                                        size_t depth) {
  auto *A = new FullyConnectedInst(dest, src, filter, bias, depth);
  M_.pushInstr(A);
  return A;
}

ReluInst *IRBuilder::createReluInst(Value *dest, Value *src) {
  auto *A = new ReluInst(dest, src);
  M_.pushInstr(A);
  return A;
}

SigmoidInst *IRBuilder::createSigmoidInst(Value *dest, Value *src) {
  auto *A = new SigmoidInst(dest, src);
  M_.pushInstr(A);
  return A;
}

TanhInst *IRBuilder::createTanhInst(Value *dest, Value *src) {
  auto *A = new TanhInst(dest, src);
  M_.pushInstr(A);
  return A;
}

SoftMaxInst *IRBuilder::createSoftMaxInst(Value *dest, Value *src,
                                          Value *expected) {
  auto *A = new SoftMaxInst(dest, src, expected);
  M_.pushInstr(A);
  return A;
}

RegressionInst *IRBuilder::createRegressionInst(Value *dest, Value *src,
                                                Value *expected) {
  auto *A = new RegressionInst(dest, src, expected);
  M_.pushInstr(A);
  return A;
}

TransposeInst *IRBuilder::createTransposeInst(Value *dest, Value *src,
                                              ArrayRef<unsigned> shuffle) {
  auto *A = new TransposeInst(dest, src, shuffle);
  M_.pushInstr(A);
  return A;
}

ConcatInst *IRBuilder::createConcatInst(Value *dest, ArrayRef<Value *> src,
                                        size_t dim) {
  auto *A = new ConcatInst(dest, src, dim);
  M_.pushInstr(A);
  return A;
}

BatchNormalizationInst *IRBuilder::createBatchNormalizationInst(
    Value *dest, Value *src, Value *scale, Value *bias, Value *mean, Value *var,
    size_t channelIdx, float epsilon, float momentum) {
  auto *A = new BatchNormalizationInst(dest, src, scale, bias, mean, var,
                                       channelIdx, epsilon, momentum);
  M_.pushInstr(A);
  return A;
}

ArithmeticInst *IRBuilder::createArithmeticInst(Value *dest, Value *LHS,
                                                Value *RHS,
                                                ArithmeticInst::OpKind kind) {
  auto *A = new ArithmeticInst(dest, LHS, RHS, kind);
  M_.pushInstr(A);
  return A;
}

StaticVariable *IRBuilder::createStaticVariable(TypeRef T,
                                                StaticVariable::InitKind mode,
                                                float val) {
  auto *A = new StaticVariable(T, mode, val);
  M_.pushVar(A);
  return A;
}
