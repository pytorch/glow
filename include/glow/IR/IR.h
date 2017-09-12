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
  /// A list of users. Notice that the same user may appear twice in the list.
  /// This is typically a very short list.
  std::list<Instruction *> users_{};
  /// The type of the value.
  TypeRef T_;

public:
  Value(TypeRef T) : T_(T) {}

  /// Removes the user \p I from the list.
  void removeUser(Instruction *I);
  /// Adds the user \p I from the list.
  void addUser(Instruction *I);
  /// Returns true if the user \p I is in the list.
  bool hasUser(Instruction *I);
};

/// This represents an instruction in our IR.
class Instruction : Value {
  /// A list of operands that the instruction has. This is typically a very
  // short list.
  std::list<Instruction *> ops_{};

  Instruction(TypeRef T) : Value(T) {}

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
