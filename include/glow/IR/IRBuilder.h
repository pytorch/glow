#ifndef GLOW_IR_IRBUILDER_H
#define GLOW_IR_IRBUILDER_H

#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "glow/IR/Type.h"

namespace glow {

/// The IRBuilder create the IR in the module.
class IRBuilder {
  /// The module that we are building.
  Module &M_;

public:
  IRBuilder(Module &M) : M_(M) {}

  /// @name IRBuilder
  ///@{
  AllocInst *createAllocInst(TypeRef T);
  DeallocInst *createDeallocInst(AllocInst *AT);
  CopyInst *createCopyInst(Value *dest, Value *src);
  ReturnInst *createReturnInst(Value *src);
  ReluInst *createReluInst(Value *dest, Value *src);
  StaticVariable *createStaticVariable(TypeRef T, StaticVariable::InitKind mode,
                                       float val);
  ///@}
};

} // namespace glow

#endif // GLOW_IR_IRBUILDER_H
