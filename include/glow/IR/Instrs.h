#ifndef GLOW_IR_INSTRS_H
#define GLOW_IR_INSTRS_H

#include "glow/Base/Type.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"

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
  WeightVar(llvm::StringRef name, TypeRef Ty, MutabilityKind mut)
      : Value(name, Ty, Kinded::Kind::WeightVarKind), mut_(mut) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::WeightVarKind;
  }

  static const char *getMutabilityStr(MutabilityKind mut);

  const char *getMutabilityStr() const;

  MutabilityKind getMutability() const { return mut_; }

  void setMutability(MutabilityKind mut) { mut_ = mut; }

  void dump(llvm::raw_ostream &os) const;
  void verify() const {}
};

} // namespace glow

// The rest of the nodes are auto-generated into this file:
#include "AutoGenInstr.h"

#endif // GLOW_IR_INSTRS_H
