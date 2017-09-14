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
  StaticVariable *createStaticVariable(TypeRef T, StaticVariable::InitKind mode,
                                       float val);

  AllocInst *createAllocInst(TypeRef T);
  DeallocInst *createDeallocInst(AllocInst *AT);
  CopyInst *createCopyInst(Value *dest, Value *src);
  ReluInst *createReluInst(Value *dest, Value *src);
  TransposeInst *createTransposeInst(Value *dest, Value *src,
                                     ArrayRef<unsigned> shuffle);

  ConvolutionInst *createConvolutionInst(Value *dest, Value *src, Value *filter,
                                         Value *bias, size_t kernel,
                                         size_t stride, size_t pad,
                                         size_t depth);

  ///@}
};

} // namespace glow

#endif // GLOW_IR_IRBUILDER_H
