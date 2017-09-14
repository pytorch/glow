#ifndef GLOW_IR_IR_H
#define GLOW_IR_IR_H

#include "glow/IR/Type.h"

#include <list>
#include <vector>

namespace glow {
class Instruction;
class Module;

using TypeRef = const Type *;

class Named {
  /// The name of the instruction. This is used for debugging.
  std::string name_{};

public:
  Named() = default;

  /// \returns the name of the instruction.
  const std::string &getName() { return name_; }

  /// \returns the name of the instruction.
  bool hasName() { return name_.size(); }

  /// Set the name of the instruction to \p name.
  void setName(const std::string &name) { name_ = name; }
};

/// A Value is something that can be an operand for an instruction. It can be a
/// Tensor, a mask, a constant, etc.
class Value : public Named {
public:
  using Use = std::pair<unsigned, Instruction *>;

  virtual ~Value() = default;

private:
  /// A list of users. Notice that the same user may appear twice in the list.
  /// This is typically a very short list.
  std::list<Use> users_{};

public:
  Value() = default;

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

  /// \returns the name of the value.
  virtual StringRef getValueName() { return "<bad>"; }

  /// \returns a description of the internal instruction parameters.
  virtual std::string getExtraDesc() { return ""; }
};

enum class OperandKind : unsigned char {
  kIn,
  kOut,
  kInOut,
};

inline const char *getOperandKindStr(OperandKind CC) {
  const char *names[] = {"@in", "@out", "@inout", nullptr};
  return names[(int)CC];
}

using Operand = std::pair<Value *, OperandKind>;

/// This represents an instruction in our IR.
class Instruction : public Value {
  /// A list of operands that the instruction has. This is typically a very
  /// short list.
  std::vector<Operand> ops_{};

  // Define/disallow default ctor, copy ctor and assignment operator.
  Instruction(const Instruction &I) = delete;
  Instruction &operator=(const Instruction &I) = delete;

protected:
  /// Adds a new operand \p v at the end of the operand list.
  void pushOperand(Operand op);

public:
  Instruction() : Value(){};

  Instruction(ArrayRef<Operand> ops) : Value() {
    for (auto &op : ops) {
      pushOperand(op);
    }
  }

  /// Sets the ith operand at index \p idx to the value \p v.
  void setOperand(unsigned idx, Value *v);

  /// \returns the ith operand.
  Operand getOperand(unsigned idx);

  /// \returns the number of operands.
  unsigned getNumOperands() { return ops_.size(); }

  /// Check the correctness of the use-list.
  void verifyUseList();
};

/// A module that represents the compilation unit.
class Module final {
  /// A uniqued list of types in the module. Types in this list can be compared
  /// by comparing their addresses.
  std::list<Type> types_{};
  /// A list of values that represent the non-instructions in the network.
  std::list<Value *> variables_{};
  /// A list of instruction that represent the network.
  std::list<Instruction *> instrs_{};

public:
  /// Add an instruction to the instr stream.
  void pushInstr(Instruction *I) { instrs_.push_back(I); }

  /// Add a value to the instr stream.
  void pushVar(Value *v) { variables_.push_back(v); }

  Module() = default;

  ~Module();

  /// Return a pointer to a uniqued type \p t in the current module.
  TypeRef uniqueType(const Type &T);

  /// Return a pointer to a uniqued type \p t in the current module.
  TypeRef uniqueType(ElemKind elemTy, ArrayRef<size_t> dims);

  /// Return the void type.
  TypeRef getVoidTy();

  /// Verify the correctness of the module.
  void verify();

  /// Dump a textual representation of the module.
  void dump();
};

} // namespace glow

#endif // GLOW_IR_IR_H
