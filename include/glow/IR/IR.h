/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GLOW_IR_IR_H
#define GLOW_IR_IR_H

#include "glow/Base/TaggedList.h"
#include "glow/Base/Traits.h"
#include "glow/Base/Type.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/UseDef.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <list>
#include <unordered_map>
#include <vector>
#if FACEBOOK_INTERNAL
namespace glow {
class FXIRWrapper;
}
#endif

namespace glow {
class Instruction;
class IRFunction;
class Function;
class Value;
class InstructionNumbering;

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
  /// \returns the instruction that the use refers to
  /// (for compatibility with UseDef::hasUser)
  Instruction *getUser() const { return use_; }
  /// \returns true if this Use is for the instruction \p other.
  bool isSame(Instruction *other) const { return use_ == other; }
  /// Sets the operand to a new value.
  void setOperand(Value *other);
  /// \returns the operand of the user instruction.
  InstructionOperand getOperand();
  ConstInstructionOperand getOperand() const;
};

class Value : public Named,
              public UseDef<Instruction, Use>,
              public Typed,
              public Kinded {
public:
  Value(llvm::StringRef name, TypeRef T, Kinded::Kind k)
      : Named(name), Typed(T), Kinded(k) {}

  bool verifyUseList(const InstructionNumbering &InstrNumbering) const;

  /// Verify the correctness of the instruction parameters.
  bool verify(const IRFunction &M) const;

  /// Dump a textual representation of the Value into provided output stream.
  void dump(llvm::raw_ostream &out) const;

  /// Dump a textual representation of the Value into default output stream.
  void dump() const;

  /// Dump a textual representation of the Value to std::string.
  std::string toString() const;

  /// Print value in context.
  void dumpInContext(llvm::raw_ostream &out) const;

  /// Print value in context using a default output stream.
  void dumpInContext() const;
};

/// This represents an instruction in our IR.
class Instruction : public Value, public TaggedListNode<Instruction> {
public:
  using Operand = InstructionOperand;

private:
  friend struct InstructionTraits;
  friend IRFunction;

  /// Parent function.
  IRFunction *F_;
  /// If a predicate is set this index points to the non-zero index of the
  /// predicate in the instruction list.
  unsigned predicateIndex_{0};

  /// A list of operands that the instruction has. This is typically a very
  /// short list.
  llvm::SmallVector<Operand, 6> ops_{};

  // Define/disallow default ctor, copy ctor and assignment operator.
  Instruction(const Instruction &I) = delete;
  Instruction &operator=(const Instruction &I) = delete;

  /// Destroy an instruction and deallocate its memory. This function is
  /// automatically invoked when the instruction is being deleted from the list
  /// of instructions.
  static void destroyInstruction(Instruction *I);

public:
  /// Prevent the destruction of a derived object via a base-class pointer.
  /// Use IRFunction::destroyInstruction instead.
  ~Instruction() {
    for (unsigned idx = 0, e = ops_.size(); idx < e; ++idx) {
      setOperand(idx, nullptr);
    }
  }

public:
  /// \returns the nullable predicate of the current node.
  Value *getPredicate() const;
  /// Assigns a nullable predicate to the current node.
  void setPredicate(Value *p);
  /// Checks if a predicate is assigned to the current node.
  bool hasPredicate() const;

  /// Adds a new operand \p op at the end of the operand list.
  void pushOperand(Operand op);

  Instruction(llvm::StringRef name, Kinded::Kind k, TypeRef Ty)
      : Value(name, Ty, k), F_(nullptr) {}

  Instruction(llvm::StringRef name, Kinded::Kind k, TypeRef Ty,
              llvm::ArrayRef<Operand> ops)
      : Value(name, Ty, k), F_(nullptr) {
    for (auto &op : ops) {
      pushOperand(op);
    }
  }

  /// Clone the current instruction.
  /// \returns a cloned instruction.
  Instruction *clone() const;

  /// \returns True if this instruction may reuse the memory buffer read by
  /// operand \p srcIdx for writing the result of the operand at \p dstIdx.
  bool isInplaceOp(unsigned dstIdx, unsigned srcIdx) const { return false; }

  /// \returns True if this instruction is not backend-specific.
  bool isCanonical() const;

  /// \returns True if this instruction is data parallel.
  bool isDataParallel() const;

  /// Sets the ith operand at index \p idx to the value \p v.
  void setOperand(unsigned idx, Value *v);

  /// \returns the ith operand.
  Operand getOperand(unsigned idx) const;

  /// \returns the number of operands.
  unsigned getNumOperands() const { return ops_.size(); }

  /// \returns the number of input operands (includes In and InOut operands).
  unsigned getNumInputs() const;

  /// \returns the number of output operands (includes Out and InOut operands).
  unsigned getNumOutputs() const;

  /// \returns the operands of the instruction.
  llvm::ArrayRef<Operand> getOperands() const { return ops_; }

  /// \returns the name of the operand.
  llvm::StringRef getOperandName(unsigned idx) const;

  /// Check the correctness of the use-list.
  bool verifyUseList(const InstructionNumbering &InstrNumbering) const;

  /// Verify the correctness of the instruction parameters.
  bool verify() const;

  /// The static dispatch version of isInplaceOp.
  static bool isInplaceOp(const Instruction *I, unsigned dstIdx,
                          unsigned srcIdx);

  /// \returns parent of current instruction.
  const IRFunction *getParent() const { return F_; }
  IRFunction *getParent() { return F_; }

  /// Sets a parent for the current instruction.
  void setParent(IRFunction *Mod) { F_ = Mod; }

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

/// Different stages of processing an IR instruction. Transitions between stages
/// always happen in the same order as they are defined below, i.e.
/// PROCESSING -> POSTPROCESSING
/// In particular, there is no possibility for any transition in the backwards
/// direction.
enum class IRInstructionProcessingStage {
  /// Instruction is being processed.
  PROCESSING,
  /// Instruction was just processed.
  POSTPROCESSING,
};

/// A function for processing instructions e.g. during a code generation or
/// during a code execution of instructions by backends. A processing function
/// takes an instruction \p I, a current instruction execution stage \p
/// executionStage and current context \p ctx as inputs, processes the
/// instruction and \returns true if the client should proceed to the next
/// execution stage or false if the client should proceed with the current
/// execution stage. The processing of the instruction will continue by the
/// caller (e.g. by a backend) from this stage. For example, if this processing
/// function has processed the instruction it may decide to return a stage
/// indicating that the processing has finished so that the caller does not
/// process it anymore. The passed \p ctx can be anything, e.g. a backend, an
/// ExecutionContext, etc.
using IRInstructionProcessingFn =
    std::function<bool(const Instruction *I,
                       IRInstructionProcessingStage executionStage, void *ctx)>;

/// The interface class for IR instruction handlers. Useful for intercepting IR
/// instructions processing.
class IRInstructionProcessingHandler {
public:
  IRInstructionProcessingHandler() = default;
  virtual ~IRInstructionProcessingHandler() = default;
  /// Set the handler to be used for IR instruction processing.
  virtual void
  setIRInstructionProcessingHandler(IRInstructionProcessingFn hook) {
    handler_ = hook;
  }
  /// \returns the handler to be used for IR instructions processing.
  virtual const IRInstructionProcessingFn &
  getIRInstructionProcessingHandler() const {
    return handler_;
  }

protected:
  /// The handler function to be used for instruction processing.
  glow::IRInstructionProcessingFn handler_;
};

//===----------------------------------------------------------------------===//
// TaggedListTraits for glow::Instruction
//===----------------------------------------------------------------------===//

struct InstructionTraits : public TaggedListTraits<Instruction> {
  static void deleteNode(Instruction *V) { Instruction::destroyInstruction(V); }

  void addNodeToList(Instruction *I);
  void removeNodeFromList(Instruction *I);

private:
  IRFunction *getContainingFunction();
  void createNode(const Instruction &);
};

class Backend;
class WeightVar;
class Value;
class Node;

/// A function that represents the compilation unit.
class IRFunction final : public IRContainer {
public:
  using VariableMap = llvm::MapVector<const Storage *, Value *>;
  using InstListTy = TaggedList<Instruction, InstructionTraits>;
  using InstrIterator = InstListTy::iterator;
  using InstrConstIterator = InstListTy::const_iterator;
  using WeightVarListTy = std::list<WeightVar *>;

private:
  /// A pointer to the graph structure. The function does not own the graph.
  IRContainer *G_{};

  /// A list of weights. Weights are shared between all execution context.
  WeightVarListTy weights_{};

  /// A list of instruction that represent the network.
  InstListTy instrs_{};

  /// Maps Variable nodes in the original graph to the weight values that
  /// represent them in the lower IR.
  VariableMap variableMap_{};

  /// A list of unique instruction names use by the function.
  llvm::StringSet<> stringTable_;

  /// Perform scheduling on the graph.
  /// \returns computed schedule in the \p Schedule parameter.
  void scheduleGraph(NodesPtrList &Schedule);

public:
  /// Add an instruction to the instr stream.
  void pushInstr(Instruction *I) { instrs_.push_back(I); }

  explicit IRFunction(IRContainer *G = nullptr);

  ~IRFunction();

  IRKind getIRKind() const override { return IRKind::GlowInstructionIRKind; };

  static bool classof(const IRContainer *I) {
    return I->getIRKind() == IRKind::GlowInstructionIRKind;
  }

  static bool classof(const IRFunction *I) { return true; }

  /// Generate IR from the graph nodes. If the compilation mode is 'training'
  /// then this procedure will also generate the code for the backward pass.
  /// It allows Backend \p B to custom translate from a Node to Instruction IR.
  void generateIR(const Backend &B);

  /// Wipe out the content of the function. This allows the function to be used
  /// again for another round of code generation.
  void clear();

  /// Clone the current IR function into a new function with the name \p newName
  /// in the same module. If \p map is non-null then the procedure records the
  /// mapping between the old node to the new node in \p map. If \p currToNewMap
  /// is non-null it is used as the initial state of the currToNew map inside
  /// the cloner.
  /// \returns a new function that is a copy of the current function.
  IRFunction *
  clone(llvm::StringRef newName,
        llvm::DenseMap<const Value *, Value *> *map = nullptr,
        llvm::DenseMap<const Value *, Value *> *currToNewMap = nullptr);

  /// Clone the current function into a user-provided function \p newF. The
  /// function \p newF is not automatically added to a module by the clone call.
  /// If \p map is non-null then the procedure records the mapping between the
  /// old node to the new node in \p map. If \p currToNewMap is non-null it is
  /// used as the initial state of the currToNew map inside the cloner. \returns
  /// a user-provided function \p newF that now contains a clone of the current
  /// function.
  IRFunction *
  clone(IRFunction *newF, llvm::DenseMap<const Value *, Value *> *map = nullptr,
        llvm::DenseMap<const Value *, Value *> *currToNewMap = nullptr) const;

  ///  \returns a reference to the high-level graph without down casting it to
  ///  children types like Function or FXIRWrapper.
  IRContainer *getRawGraph() { return G_; }

  /// \returns a reference to the high-level graph.
  Function *getGraph() {
    assert(llvm::isa<Function>(G_));
    return llvm::cast<Function>(G_);
  }

  /// \returns a reference to the high-level graph.
  const Function *getGraph() const {
    assert(llvm::isa<Function>(G_));
    return llvm::cast<Function>(G_);
  }

#if FACEBOOK_INTERNAL
  /// \returns a reference to the glow FX graph.
  FXIRWrapper *getFXGraph();

  /// \returns a reference to the glow FX graph.
  const FXIRWrapper *getFXGraph() const;
#endif

  Module *getParent() override { return G_->getParent(); }

  const Module *getParent() const override { return G_->getParent(); }

  /// Sets the high-level graph corresponding to this function.
  void setGraph(Function *F) { G_ = F; }

  /// \returns a unique legal name that's based on the string \p name.  Legal
  /// names are legal C identifiers in the form: "[a-zA-Z_][a-zA-Z0-9_]*".
  llvm::StringRef uniqueName(llvm::StringRef name) {
    return Module::uniqueName(name, stringTable_, stringTable_,
                              *getParent()->getOriginalNames());
  }

  /// Verify the correctness of the function.
  bool verify() const;

  /// Dump a textual representation of the IRFunction into default output
  /// stream.
  void dump() const;

  /// Dump a textual representation of the IRFunction to std::string.
  std::string toString() const;

  /// Dump a textual representation of the IRFunction into provided output
  /// stream.
  void dump(llvm::raw_ostream &OS) const;

  /// Dump a dotty graph that depicts the function.
  void dumpDAG(llvm::StringRef dotFilename) const;

  /// Dump a dotty graph that depicts the function.
  void dumpDAG(const char *dotFilename) const;

  /// Dump a dotty graph that depicts the function.
  void dumpDAG() const;

  /// \returns the variable map.
  VariableMap &getVariableMap() { return variableMap_; }

  /// \returns the variable map.
  const VariableMap &getVariableMap() const { return variableMap_; }

  /// Returns a list of constants associated with function.
  std::vector<const Constant *> findConstants() const;

  /// Returns a list of placeholders associated with the function.
  std::vector<const Placeholder *> findPlaceholders() const;

  /// \returns the weight that the variable \p v is lowered into, or null if the
  /// variable is unknown.
  Value *getWeightForNode(const Storage *V) const;

  /// \returns the list of instructions.
  InstListTy &getInstrs() { return instrs_; }

  /// \returns the list of instructions.
  const InstListTy &getInstrs() const { return instrs_; }

  /// \returns pointer to the class member for the instruction list.
  static InstListTy IRFunction::*getInstrsMemberPtr() {
    return &IRFunction::instrs_;
  }

  /// \returns the list of weights.
  WeightVarListTy &getWeights() { return weights_; }

  /// \returns the list of weights.
  const WeightVarListTy &getWeights() const { return weights_; }

  /// Erase the instruction from the function.
  void eraseInstruction(Instruction *I);

  /// Remove the instruction from the function.
  InstrIterator removeInstruction(Instruction *I);

  /// Inserts an instruction at the place described by \where.
  InstrIterator insertInstruction(Instruction *where, Instruction *I);

  /// Inserts an instruction at the end of the instructions list.
  void insertInstruction(Instruction *I);

  /// Moves an instruction belonging to a function before the place described by
  /// \where.
  InstrIterator moveInstruction(Instruction *where, Instruction *I);
};

/// Iterator over inteructions.
using InstrIterator = IRFunction::InstrIterator;
using InstrConstIterator = IRFunction::InstrConstIterator;

/// A helper class used for instructions numbering.
class InstructionNumbering {
  using NumberedInstructionMap = std::vector<const Instruction *>;
  using InstructionNumbersMap = std::unordered_map<const Instruction *, size_t>;
  /// Maps the number to an instruction.
  NumberedInstructionMap numToInstr_;
  /// Maps an instruction to its number.
  InstructionNumbersMap instrToNum_;

public:
  InstructionNumbering(const IRFunction &M);

  /// Return the instruction with a given number or
  /// M.getInstrs().end() if this instruction is not assigned any number.
  const Instruction *getInstr(size_t InstrNumber) const;

  /// Return the number of an instruction or a negative value if no number
  /// was assigned to this instruction.
  int64_t getInstrNumber(const Instruction *I) const;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Value &V);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Value *V);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const IRFunction &irf);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const IRFunction *irf);

/// IR instrumentation kind.
enum class InstrumentKind : unsigned char {
  /// Instrumentation before an instruction is executed.
  Before,
  /// Instrumentation after an instruction is executed.
  After,
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, InstrumentKind kind);

} // namespace glow

#endif // GLOW_IR_IR_H
