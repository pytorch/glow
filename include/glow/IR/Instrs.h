#ifndef GLOW_IR_INSTRS_H
#define GLOW_IR_INSTRS_H

#include "glow/Base/Type.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/Support/Casting.h"

#include "llvm/ADT/ArrayRef.h"

namespace glow {

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

  MutabilityKind getKind() const { return mut_; }
  void dump(std::ostream &os) const;
  void verify() const {}
};

} // namespace glow

// The rest of the nodes are auto-generated into this file:
#include "AutoGenInstr.h"

#endif // GLOW_IR_INSTRS_H
