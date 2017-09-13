#ifndef GLOW_IR_INSTRS_H
#define GLOW_IR_INSTRS_H

#include "glow/IR/IR.h"
#include "glow/IR/Type.h"

namespace glow {

class AllocTensorInst : public Instruction {
  TypeRef Ty_;

public:
  AllocTensorInst(TypeRef T) : Instruction(), Ty_(T) {}
  StringRef getInstrName() override { return "AllocTensor"; }
  std::string getInstrDesc() override { return Ty_->asString(); }
  TypeRef getType() { return Ty_; }
};

class DeallocTensorInst : public Instruction {
public:
  DeallocTensorInst(AllocTensorInst *A) : Instruction(A) {}
  StringRef getInstrName() override { return "DeallocTensor"; }
};

class CopyTensorInst : public Instruction {
public:
  CopyTensorInst(AllocTensorInst *dest, AllocTensorInst *src)
      : Instruction({dest, src}) {}
  StringRef getInstrName() override { return "CopyTensor"; }
};

class ReturnInst : public Instruction {
public:
  ReturnInst(AllocTensorInst *src) : Instruction(src) {}
  StringRef getInstrName() override { return "ReturnInst"; }
};

} // namespace glow

#endif // GLOW_IR_INSTRS_H
