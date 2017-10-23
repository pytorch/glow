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
