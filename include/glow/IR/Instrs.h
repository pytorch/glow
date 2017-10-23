#ifndef GLOW_IR_INSTRS_H
#define GLOW_IR_INSTRS_H

#include "glow/Base/Type.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/Support/Casting.h"

#include "llvm/ADT/ArrayRef.h"

namespace glow {

class AllocActivationInst;
class DeallocActivationInst;

class ConcatInst : public Instruction {
  /// We concat the tensors along this dimension.
  size_t dim_;

public:
  ConcatInst(Value *dest, llvm::ArrayRef<Value *> src, size_t dim)
      : Instruction(Kinded::Kind::ConcatInstKind, dest->getType(),
                    {{dest, OperandKind::Out}}),
        dim_(dim) {
    for (auto s : src) {
      pushOperand({s, OperandKind::In});
    }
  }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ConcatInstKind;
  }
  bool mayShareBuffers() const { return false; }
  std::string getExtraDesc() const;
  Value *getDest() const { return getOperand(0).first; }
  Value *getSrc() const { return getOperand(1).first; }
  size_t getDim() const { return dim_; }

  void verify() const;
};

class WeightVar : public Value {
public:
  enum class MutabilityKind {
    Constant, // A read-only region of memory.
    Mutable,  // A read/write region of memory.
  };

private:
  /// The mutability mode.
  MutabilityKind mut_;

public:
  WeightVar(TypeRef Ty, MutabilityKind mut)
      : Value(Ty, Kinded::Kind::WeightVarKind), mut_(mut) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::WeightVarKind;
  }

  static const char *getKindStr(MutabilityKind mut);

  const char *getKindStr() const;

  void setInitKind(MutabilityKind k) { mut_ = k; }
  MutabilityKind getKind() const { return mut_; }

  std::string getExtraDesc() const;
  void verify() const {}
};

} // namespace glow

#include "AutoGenInstr.h"

#endif // GLOW_IR_INSTRS_H
