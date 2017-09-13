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
  AllocTensorInst *createAllocTensorInst(TypeRef T);
  DeallocTensorInst *createDeallocTensorInst(AllocTensorInst *AT);
  CopyTensorInst *createCopyTensorInst(AllocTensorInst *dest,
                                       AllocTensorInst *src);
  ReturnInst *createReturnInst(AllocTensorInst *src);
  ///@}
};

} // namespace glow

#endif // GLOW_IR_IRBUILDER_H
