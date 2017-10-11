#ifndef GLOW_IR_IR_H
#define GLOW_IR_IR_H

#include "glow/Base/Type.h"
#include "glow/IR/Traits.h"
#include "glow/IR/UseDef.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <list>
#include <unordered_map>
#include <vector>

namespace glow {
class Instruction;
class Module;
class Graph;

enum class OperandKind : unsigned char {
  In,
  Out,
  InOut,
};

inline const char *getOperandKindStr(OperandKind CC) {
  const char *names[] = {"@in", "@out", "@inout", nullptr};
  return names[(int)CC];
}

class Value : public Named,
              public UseDef<Instruction, Value>,
              public Typed,
              public Kinded {
public:
  Value(TypeRef T, Kinded::Kind k) : Named(), UseDef(), Typed(T), Kinded(k) {}
};

/// This represents an instruction in our IR.
class Instruction : public Value {
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
  Instruction(Kinded::Kind k, TypeRef Ty) : Value(Ty, k) {}

  Instruction(Kinded::Kind k, TypeRef Ty, llvm::ArrayRef<Operand> ops)
      : Value(Ty, k) {
    for (auto &op : ops) {
      pushOperand(op);
    }
  }

  /// \returns true if the @In arguments can share a buffer with the @Out
  /// arguments. This happens when the read and write access for the buffer are
  /// the same.
  bool mayShareBuffers() const { return true; }

  /// When printing the instruction this method prints the extra metadata.
  std::string getExtraDesc() const { return ""; }

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

  static bool mayShareBuffers(const Instruction *I);
};

class WeightVar;
class Value;
class Node;

/// A module that represents the compilation unit.
class Module final {
public:
  using VariableMap = std::unordered_map<const Node *, Value *>;
  using InstListTy = std::list<Instruction *>;
  using WeightVarListTy = std::list<WeightVar *>;

private:
  /// A reference to the graph structure.
  Graph &G_;

  /// A list of weights. Weights are shared between all execution context.
  std::list<WeightVar *> weights_{};

  /// A list of instruction that represent the network.
  InstListTy instrs_{};

  /// Maps Variable nodes in the original graph to the weight values that
  /// represent them in the lower IR.
  VariableMap variableMap{};

  /// Assign the instructions in the module a unique name.
  void nameInstructions();

public:
  /// Add an instruction to the instr stream.
  void pushInstr(Instruction *I) { instrs_.push_back(I); }

  explicit Module(Graph &G) : G_(G) {}

  ~Module();

  /// Generate IR from the graph nodes.
  void generateIR();

  /// \returns a reference to the original graph.
  Graph &getGraph() { return G_; }

  /// Verify the correctness of the module.
  void verify() const;

  /// Dump a textual representation of the module.
  void dump();

  /// Dump a dotty graph that depicts the module.
  void dumpDAG();

  /// \returns the variable map.
  VariableMap &getVariableMap() { return variableMap; }

  /// \returns the weight that the variable \p v is lowered into, or null if the
  /// variable is unknown.
  Value *getWeightForNode(const Node *V) const;

  /// \returns the list of instructions.
  InstListTy &getInstrs() { return instrs_; }

  /// \returns the list of weights.
  WeightVarListTy &getWeights() { return weights_; }
};

} // namespace glow

#endif // GLOW_IR_IR_H
