#include "glow/IR/IRBuilder.h"

using namespace glow;

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
ReturnInst *IRBuilder::createReturnInst(Value *src) {
  auto *A = new ReturnInst(src);
  M_.pushInstr(A);
  return A;
}

ReluInst *IRBuilder::createReluInst(Value *dest, Value *src) {
  auto *A = new ReluInst(dest, src);
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
