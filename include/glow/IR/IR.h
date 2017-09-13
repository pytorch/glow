#ifndef GLOW_IR_IR_H
#define GLOW_IR_IR_H

#include "glow/IR/Type.h"

#include <list>
#include <vector>

namespace glow {
class Instruction;
class Module;

using TypeRef = const Type *;

/// A Value is something that can be an operand for an instruction. It can be a
/// Tensor, an instruction, a constant, etc.
class Value {
public:
  using Use = std::pair<unsigned, Instruction *>;

private:
  /// A list of users. Notice that the same user may appear twice in the list.
  /// This is typically a very short list.
  std::list<Use> users_{};
  /// The type of the value.
  TypeRef T_;

public:
  Value(TypeRef T) : T_(T) {}

  /// \returns the type of the value.
  TypeRef getType() { return T_; }

  /// Removes the use \p U from the uselist.
  void removeUse(Use U);

  /// Adds the use \p U.
  void addUse(Use U);

  /// \returns True if the value has some users.
  bool hasUsers() { return users_.size(); }

  /// Returns true if the user \p I is in the list.
  bool hasUser(Instruction *I);

  /// Replace all of the uses of this value with \p v.
  void replaceAllUsesOfWith(Value *v);
};

/// This represents an instruction in our IR.
class Instruction : public Value {
  /// A list of operands that the instruction has. This is typically a very
  // short list.
  std::vector<Value *> ops_{};

  /// Adds a new operand \p v at the end of the operand list.
  void pushOperand(Value *v);

public:
  Instruction(TypeRef T, ArrayRef<Value *> ops = {}) : Value(T) {
    for (auto &v : ops) {
      pushOperand(v);
    }
  }

  /// Sets the ith operand at index \p idx to the value \p v.
  void setOperand(unsigned idx, Value *v);

  /// \returns the ith operand.
  Value *getOperand(unsigned idx);

  /// \returns the number of operands.
  unsigned getNumOperands() { return ops_.size(); }

  /// Check the correctness of the use-list.
  void verifyUseList();
};

/// A module that represents the compilation unit.
class Module {
  /// A uniqued list of types in the module. Types in this list can be compared
  /// by comparing their addresses.
  std::list<Type> types_{};
  /// A list of values that represent the non-instructions in the network.
  std::list<Instruction *> consts_{};
  /// A list of instruction that represent the network.
  std::list<Instruction *> instrs_{};

public:
  Module() = default;

  /// Return a pointer to a uniqued type \p t in the current module.
  TypeRef uniqueType(const Type &T);

  /// Verify the correctness of the module.
  void verify();
};

} // namespace glow

#endif // GLOW_IR_IR_H
