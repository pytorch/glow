#ifndef GLOW_IR_IR_H
#define GLOW_IR_IR_H

#include "glow/IR/Type.h"
#include "glow/IR/UseDef.h"

#include <list>
#include <vector>

namespace glow {
class Instruction;
class Module;

/// This add the capability to name subclasses.
class Named {
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

/// Subclasses of this class have a type associated with them.
class Typed {
private:
  TypeRef Ty_{};

public:
  Typed(TypeRef Ty) : Ty_(Ty){};

  TypeRef getType() { return Ty_; }

  ArrayRef<size_t> dims() { return Ty_->dims(); }

  ElemKind getElementType() const { return Ty_->getElementType(); }

  bool isType(TypeRef T) { return Ty_ == T; }
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

/// Subclasses of Value have an enum that describe their kind.
class Kinded {
public:
  enum class Kind {
#define DEF_INSTR(CLASS, NAME) CLASS##Kind,
#define DEF_VALUE(CLASS, NAME) CLASS##Kind,
#include "glow/IR/Instrs.def"
#undef DEF_INSTR
#undef DEF_VALUE
  };

  const char *getKindName(Kind IK) {
    const char *names[] = {
#define DEF_INSTR(CLASS, NAME) #NAME,
#define DEF_VALUE(CLASS, NAME) #NAME,
#include "glow/IR/Instrs.def"
#undef DEF_INSTR
#undef DEF_VALUE
        nullptr};
    return names[(int)IK];
  }

private:
  /// The kind of the value.
  Kind kind_;

public:
  /// Ctor.
  Kinded(Kind vk) : kind_(vk) {}

  /// Returns the kind of the instruction.
  Kind getKind() const { return kind_; }

  const char *getKindName() { return getKindName(kind_); }
};

class Value : public Named,
              public UseDef<Instruction, Value>,
              public Typed,
              public Kinded {
public:
  Value(TypeRef T, Kinded::Kind k) : Named(), UseDef(), Typed(T), Kinded(k) {}
};

/// This represents an instruction in our IR.
class Instruction : public Named, public Kinded {
public:
  using Operand = std::pair<Value *, OperandKind>;

private:
  /// A list of operands that the instruction has. This is typically a very
  /// short list.
  std::vector<Operand> ops_{};

  // Define/disallow default ctor, copy ctor and assignment operator.
  Instruction(const Instruction &I) = delete;
  Instruction &operator=(const Instruction &I) = delete;

protected:
  /// Adds a new operand \p op at the end of the operand list.
  void pushOperand(Operand op);

public:
  Instruction(Kinded::Kind k) : Kinded(k) {}

  Instruction(Kinded::Kind k, ArrayRef<Operand> ops) : Kinded(k) {
    for (auto &op : ops) {
      pushOperand(op);
    }
  }

  /// When printing the instruction this method prints the extra metadata.
  std::string getExtraDesc() { return ""; }

  /// Sets the ith operand at index \p idx to the value \p v.
  void setOperand(unsigned idx, Value *v);

  /// \returns the ith operand.
  Operand getOperand(unsigned idx) const;

  /// \returns the number of operands.
  unsigned getNumOperands() const { return ops_.size(); }

  /// Check the correctness of the use-list.
  void verifyUseList();

  /// Verify the correctness of the instruction parameters.
  void verify();

  operator Value *() const { return getOperand(0).first; }
};

/// A module that represents the compilation unit.
class Module final {
public:
  using InstListTy = std::list<Instruction *>;
  using VarListTy = std::list<Value *>;

private:
  /// A uniqued list of types in the module. Types in this list can be compared
  /// by comparing their addresses.
  std::list<Type> types_{};
  /// A list of values that represent the non-instructions in the network.
  std::list<Value *> variables_{};
  /// A list of instruction that represent the network.
  InstListTy instrs_{};

  /// Give the instructions in the module a unique name.
  void nameInstructions();

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

  /// Dump a dotty graph that depicts the module.
  void dumpDAG();

  /// \returns the list of instructions.
  InstListTy &getInstrs() { return instrs_; }

  /// \returns the list of variables.
  VarListTy &getVars() { return variables_; }
};

} // namespace glow

#endif // GLOW_IR_IR_H
