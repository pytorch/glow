#ifndef GLOW_IR_IR_H
#define GLOW_IR_IR_H

#include "glow/Base/Traits.h"
#include "glow/Base/Type.h"
#include "glow/IR/UseDef.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <list>
#include <unordered_map>
#include <vector>

namespace glow {
class Instruction;
class Module;
class Graph;
class Value;

enum class OperandKind : unsigned char {
  In,
  Out,
  InOut,
};

inline const char *getOperandKindStr(OperandKind CC) {
  const char *names[] = {"@in", "@out", "@inout", nullptr};
  return names[(int)CC];
}

using InstructionOperand = std::pair<Value *, OperandKind>;

/// A 'Use' is a use-list representation of an instruction operand. It maps to a
/// specific operand in an instruction.
struct Use {
  /// The instruction.
  Instruction *use_;
  /// The index of the operand.
  unsigned idx_;

  bool operator==(const Use &other) const {
    return idx_ == other.idx_ && use_ == other.use_;
  }

  Use(unsigned idx, Instruction *use) : use_(use), idx_(idx) {}

  /// \returns the instruction that the use refers to.
  Instruction *get() const { return use_; }
  /// \returns true if this Use is for the instruction \p other.
  bool isSame(Instruction *other) const { return use_ == other; }
  /// Sets the operand to a new value.
  void setOperand(Value *other);
  /// \returns the operand of the user instruction.
  InstructionOperand getOperand();
};

class Value : public Named,
              public UseDef<Instruction, Value, Use>,
              public Typed,
              public Kinded {
public:
  Value(llvm::StringRef name, TypeRef T, Kinded::Kind k)
      : Named(name), Typed(T), Kinded(k) {}
};

/// This represents an instruction in our IR.
class Instruction : public Value {
public:
  using Operand = InstructionOperand;

private:
  /// Parent module.
  Module *M;

  /// A list of operands that the instruction has. This is typically a very
  /// short list.
  llvm::SmallVector<Operand, 6> ops_{};

  // Define/disallow default ctor, copy ctor and assignment operator.
  Instruction(const Instruction &I) = delete;
  Instruction &operator=(const Instruction &I) = delete;

protected:
  /// Adds a new operand \p op at the end of the operand list.
  void pushOperand(Operand op);

public:
  Instruction(Module &M, llvm::StringRef name, Kinded::Kind k, TypeRef Ty)
      : Value(name, Ty, k), M(&M) {}

  Instruction(Module &M, llvm::StringRef name, Kinded::Kind k, TypeRef Ty,
              llvm::ArrayRef<Operand> ops)
      : Value(name, Ty, k), M(&M) {
    for (auto &op : ops) {
      pushOperand(op);
    }
  }

  ~Instruction() {
    for (unsigned idx = 0, e = ops_.size(); idx < e; ++idx) {
      setOperand(idx, nullptr);
    }
  }

  /// \returns True if this instruction may reuse the memory buffer read by
  /// operand \p srcIdx for writing the result of the operand at \p dstIdx.
  bool isInplaceOp(unsigned dstIdx, unsigned srcIdx) const { return false; }

  /// Sets the ith operand at index \p idx to the value \p v.
  void setOperand(unsigned idx, Value *v);

  /// \returns the ith operand.
  Operand getOperand(unsigned idx) const;

  /// \returns the number of operands.
  unsigned getNumOperands() const { return ops_.size(); }

  /// Check the correctness of the use-list.
  void verifyUseList() const;

  /// Verify the correctness of the instruction parameters.
  void verify() const;

  /// The static dispatch version of isInplaceOp.
  static bool isInplaceOp(const Instruction *I, unsigned dstIdx,
                          unsigned srcIdx);

  /// \returns parent of current instruction.
  Module &getParent() const {
    assert(M);
    return *M;
  }

  /// Sets a parent for the current instruction.
  void setParent(Module *Mod) { M = Mod; }

  /// Erases instruction from its parent and destroy it.
  void eraseFromParent();

  /// Removes instruction from its parent, but does not destroy it.
  /// The instruction can be inserted elsewhere afterwards.
  void removeFromParent();

protected:
  /// Dump the operands of the instruction into the stream \p os.
  void dumpOperands(std::ostream &os) const;
};

class WeightVar;
class Value;
class Node;

/// A module that represents the compilation unit.
class Module final {
public:
  using GradientMap = std::unordered_map<const Value *, Value *>;
  using VariableMap = std::unordered_map<const Node *, Value *>;
  using InstListTy = std::list<Instruction *>;
  using WeightVarListTy = std::list<WeightVar *>;

private:
  /// A pointer to the graph structure. The Module does not own the graph.
  Graph *G_;
  /// Name of the module.
  llvm::StringRef name_;

  /// A list of weights. Weights are shared between all execution context.
  std::list<WeightVar *> weights_{};

  /// A list of instruction that represent the network.
  InstListTy instrs_{};

  /// Maps Variable nodes in the original graph to the weight values that
  /// represent them in the lower IR.
  VariableMap variableMap{};

  /// In training mode, this map maps some weight variable to the tensor that
  /// collects inputs for it.
  GradientMap gradientMap{};

  /// Assign the instructions in the module a unique name.
  void nameInstructions();

public:
  /// Add an instruction to the instr stream.
  void pushInstr(Instruction *I) { instrs_.push_back(I); }

  explicit Module(Graph *G);

  ~Module();

  /// Generate IR from the graph nodes. If the compilation mode is 'training'
  /// then this procedure will also generate the code for the backward pass.
  void generateIR(CompilationMode mode);

  /// Wipe out the content of the module. This allows the module to be used
  /// again for another round of code generation.
  void clear();

  /// \returns a reference to the original graph.
  Graph *getGraph() { return G_; }

  llvm::StringRef getName() const { return name_; }

  /// Verify the correctness of the module.
  void verify() const;

  /// Dump a textual representation of the module.
  void dump();

  /// Dump a dotty graph that depicts the module.
  void dumpDAG();

  /// \returns the variable map.
  VariableMap &getVariableMap() { return variableMap; }

  /// \returns the gradient map.
  GradientMap &getGradientMap() { return gradientMap; }

  /// \returns the weight that the variable \p v is lowered into, or null if the
  /// variable is unknown.
  Value *getWeightForNode(const Node *V) const;

  /// \returns the list of instructions.
  InstListTy &getInstrs() { return instrs_; }
  /// \returns the list of instructions.
  const InstListTy &getInstrs() const { return instrs_; }

  /// \returns the list of weights.
  WeightVarListTy &getWeights() { return weights_; }

  /// Erase the instruction from the module.
  void eraseInstruction(Instruction *I);

  /// Erase the instruction from the module.
  InstListTy::iterator eraseInstruction(InstListTy::iterator it);

  /// Remove the instruction from the module.
  void removeInstruction(Instruction *I);

  /// Remove the instruction from the module.
  InstListTy::iterator removeInstruction(InstListTy::iterator it);

  /// Inserts an instruction at the place described by \where.
  void insertInstruction(InstListTy::iterator where, Instruction *I);

  /// Inserts an instruction at the end of the instructions list.
  void insertInstruction(Instruction *I);
};

/// Iterator over inteructions.
using InstrIterator = Module::InstListTy::iterator;

} // namespace glow

#endif // GLOW_IR_IR_H
