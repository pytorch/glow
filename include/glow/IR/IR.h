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
  InOut,
  Out,
};

inline const char *getOperandKindStr(OperandKind CC) {
  const char *names[] = {"@in", "@inout", "@out", nullptr};
  return names[(int)CC];
}

using InstructionOperand = std::pair<Value *, OperandKind>;
using ConstInstructionOperand = const std::pair<const Value *, OperandKind>;

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
  ConstInstructionOperand getOperand() const;
};

class Value : public Named,
              public UseDef<Instruction, Value, Use>,
              public Typed,
              public Kinded {
public:
  Value(llvm::StringRef name, TypeRef T, Kinded::Kind k)
      : Named(name), Typed(T), Kinded(k) {}

  void verifyUseList(const Module &M) const;

  /// Verify the correctness of the instruction parameters.
  void verify(const Module &M) const;

  /// Print value.
  void dump(llvm::raw_ostream &out) const;

  /// Print value using a default output stream.
  void dump() const;

  /// Print value in context.
  void dumpInContext(llvm::raw_ostream &out) const;

  /// Print value in context using a default output stream.
  void dumpInContext() const;
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

public:
  /// Adds a new operand \p op at the end of the operand list.
  void pushOperand(Operand op);

  Instruction(Module *M, llvm::StringRef name, Kinded::Kind k, TypeRef Ty)
      : Value(name, Ty, k), M(M) {}

  Instruction(Module *M, llvm::StringRef name, Kinded::Kind k, TypeRef Ty,
              llvm::ArrayRef<Operand> ops)
      : Value(name, Ty, k), M(M) {
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

  /// \returns the operands of the instruction.
  llvm::ArrayRef<Operand> getOperands() const { return ops_; }

  /// Check the correctness of the use-list.
  void verifyUseList() const;

  /// Verify the correctness of the instruction parameters.
  void verify() const;

  /// The static dispatch version of isInplaceOp.
  static bool isInplaceOp(const Instruction *I, unsigned dstIdx,
                          unsigned srcIdx);

  /// \returns parent of current instruction.
  Module *getParent() const { return M; }

  /// Sets a parent for the current instruction.
  void setParent(Module *Mod) { M = Mod; }

  /// Erases instruction from its parent and destroy it.
  void eraseFromParent();

  /// Removes instruction from its parent, but does not destroy it.
  /// The instruction can be inserted elsewhere afterwards.
  void removeFromParent();

  static bool classof(const Value *V);

  static bool classof(const Instruction *I) { return true; }

protected:
  /// Dump the operands of the instruction into the stream \p os.
  void dumpOperands(llvm::raw_ostream &os) const;
};

class WeightVar;
class Value;
class Node;

/// A module that represents the compilation unit.
class Module final {
public:
  using VariableMap = std::unordered_map<const Node *, Value *>;
  using InstListTy = std::list<Instruction *>;
  using InstrIterator = InstListTy::iterator;
  using InstrConstIterator = InstListTy::const_iterator;
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
  void dumpDAG(const char *dotFilename);

  /// Dump a dotty graph that depicts the module.
  void dumpDAG();

  /// \returns the variable map.
  VariableMap &getVariableMap() { return variableMap; }

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
  InstrIterator eraseInstruction(InstrIterator it);

  /// Remove the instruction from the module.
  void removeInstruction(Instruction *I);

  /// Remove the instruction from the module.
  InstrIterator removeInstruction(InstrIterator it);

  /// Inserts an instruction at the place described by \where.
  InstrIterator insertInstruction(InstrIterator where, Instruction *I);

  /// Moves an instruction belonging to a module to the place described by
  /// \where.
  InstrIterator moveInstruction(InstrIterator where, Instruction *I);

  /// Moves an instruction belonging to a module to the place described by
  /// \where.
  InstrIterator moveInstruction(const Instruction *where, Instruction *I);

  /// Inserts an instruction at the end of the instructions list.
  void insertInstruction(Instruction *I);

  /// \returns instruction's list iterator corresponding to the instruction.
  InstrIterator getInstrIterator(const Instruction *I);

  /// \returns instruction's list iterator corresponding to the instruction.
  InstrConstIterator getInstrIterator(const Instruction *I) const;
};

/// Iterator over inteructions.
using InstrIterator = Module::InstrIterator;
using InstrConstIterator = Module::InstrConstIterator;

/// A helper class used for instructions numbering.
class InstructionNumbering {
  using NumberedInstructionMap = std::vector<InstrIterator>;
  using InstructionNumbersMap = std::unordered_map<Instruction *, size_t>;
  /// Maps the number to an instruction.
  NumberedInstructionMap NumToInstr_;
  /// Maps an instruction to its number.
  InstructionNumbersMap InstrToNum_;
  Module &M_;

public:
  /// Virtual slot number to be used for instructions numbering. It helps to
  /// distinguish reads from writes and makes comparision of live intervals
  /// easier. LLVM used a similar approach for the linear scan register
  /// allocator.
  ///
  /// For an instruction with number N, its @in operands would be considered
  /// to be at (N+READ_SLOT), its @out operands would be at (N+WRITE_SLOT).
  enum SLOTS {
    READ_SLOT = 0,
    WRITE_SLOT = 2,
    MAX_SLOT = 4,
  };

  InstructionNumbering(Module &M);

  /// Return the instruction with a given number or
  /// M.getInstrs().end() if this instruction is not assigned any number.
  InstrIterator getInstr(size_t InstrNumber) const;

  /// Return the number of an instruction or a negative value if no number
  /// was assigned to this instruction.
  int64_t getInstrNumber(InstrIterator IT) const;

  /// Return the number of an instruction or a negative value if no number
  /// was assigned to this instruction.
  int64_t getInstrNumber(Instruction *I) const;

  /// \returns the base number of the instruction.
  /// It is the same for all slots of a given instruction.
  static int64_t getInstrBaseNumber(int64_t idx) {
    return idx / MAX_SLOT * MAX_SLOT;
  }

  /// \returns true if \p idx is the instruction number of the read slot of the
  /// instruction.
  static bool isReadSlotNumber(int64_t idx) {
    return idx % MAX_SLOT == READ_SLOT;
  }

  /// \returns true if \p idx is the instruction number of a write slot of the
  /// instruction.
  static bool isWriteSlotNumber(int64_t idx) {
    return idx % MAX_SLOT == WRITE_SLOT;
  }

  /// \returns the instruction number of a read slot of instruction with number
  /// \p idx.
  static int64_t getInstrReadSlotNumber(int64_t idx) {
    return getInstrBaseNumber(idx) + READ_SLOT;
  }

  /// \returns the instruction number of a write slot of instruction with number
  /// \p idx.
  static int64_t getInstrWriteSlotNumber(int64_t idx) {
    return getInstrBaseNumber(idx) + WRITE_SLOT;
  }

  /// Return the module
  Module &getModule() { return M_; }
};

/// Get the allocation corrsponding to th value \p V. It can look through
/// tensorview instructions. \returns found allocation or nullptr.
Value *getAllocationOrigin(Value *V);

/// \returns peels off the layers of tensorviews from a value \p V.
Value *getOrigin(Value *V);

} // namespace glow

#endif // GLOW_IR_IR_H
