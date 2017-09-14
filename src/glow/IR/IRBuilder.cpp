#include "glow/IR/IRBuilder.h"

using namespace glow;

StaticVariable *IRBuilder::createStaticVariable(TypeRef T,
                                                StaticVariable::InitKind mode,
                                                float val) {
  auto *A = new StaticVariable(T, mode, val);
  M_.pushVar(A);
  return A;
}

AllocInst *IRBuilder::createAllocInst(TypeRef T) {
  auto *A = new AllocInst(T);
  M_.pushInstr(A);
  return A;
}
DeallocInst *IRBuilder::createDeallocInst(AllocInst *AT) {
  auto *A = new DeallocInst(AT);
  M_.pushInstr(A);
  return A;
}
CopyInst *IRBuilder::createCopyInst(Value *dest, Value *src) {
  auto *A = new CopyInst(dest, src);
  M_.pushInstr(A);
  return A;
}

ReluInst *IRBuilder::createReluInst(Value *dest, Value *src) {
  auto *A = new ReluInst(dest, src);
  M_.pushInstr(A);
  return A;
}

TransposeInst *IRBuilder::createTransposeInst(Value *dest, Value *src,
                                              ArrayRef<unsigned> shuffle) {
  auto *A = new TransposeInst(dest, src, shuffle);
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
