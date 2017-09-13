#include "glow/IR/IRBuilder.h"

using namespace glow;

AllocTensorInst *IRBuilder::createAllocTensorInst(TypeRef T) {
  auto *A = new AllocTensorInst(T);
  M_.pushInstr(A);
  return A;
}
DeallocTensorInst *IRBuilder::createDeallocTensorInst(AllocTensorInst *AT) {
  auto *A = new DeallocTensorInst(AT);
  M_.pushInstr(A);
  return A;
}
CopyTensorInst *IRBuilder::createCopyTensorInst(AllocTensorInst *dest,
                                                AllocTensorInst *src) {
  auto *A = new CopyTensorInst(dest, src);
  M_.pushInstr(A);
  return A;
}
ReturnInst *IRBuilder::createReturnInst(AllocTensorInst *src) {
  auto *A = new ReturnInst(src);
  M_.pushInstr(A);
  return A;
}
