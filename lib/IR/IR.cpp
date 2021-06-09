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

#include "glow/IR/IR.h"
#include "glow/Graph/FXIRWrapper.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IRUtils.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Support.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_set>

using namespace glow;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

//===----------------------------------------------------------------------===//
//                       General IR operations
//===----------------------------------------------------------------------===//

bool Instruction::classof(const Value *V) {
#define DEF_VALUE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME)
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME)
#define DEF_INSTR_RANGE(CLASS, FIRST, LAST)                                    \
  constexpr auto First_##CLASS = Kinded::Kind::FIRST##Kind;                    \
  constexpr auto Last_##CLASS = Kinded::Kind::LAST##Kind;
#include "glow/AutoGenInstr.def"
  return V->getKind() >= First_Instruction && V->getKind() <= Last_Instruction;
}

void Use::setOperand(Value *other) { use_->setOperand(idx_, other); }

InstructionOperand Use::getOperand() { return use_->getOperand(idx_); }

ConstInstructionOperand Use::getOperand() const {
  return use_->getOperand(idx_);
}

Value *Instruction::getPredicate() const {
  assert(hasPredicate() && "No predicate is set");
  return getOperand(predicateIndex_).first;
}

void Instruction::setPredicate(Value *p) {
  // Push a new predicate.
  if (!hasPredicate()) {
    predicateIndex_ = getNumOperands();
    pushOperand({p, OperandKind::In});
  }

  setOperand(predicateIndex_, p);
}

bool Instruction::hasPredicate() const { return predicateIndex_ > 0; }

void Instruction::pushOperand(Operand op) {
  ops_.emplace_back(nullptr, op.second);
  setOperand(ops_.size() - 1, op.first);
}

void Instruction::setOperand(unsigned idx, Value *v) {
  auto *currVal = ops_[idx].first;

  if (currVal == v) {
    return;
  }

  if (currVal) {
    currVal->removeUse(Use(idx, this));
  }

  if (v) {
    ops_[idx].first = v;
    v->addUse(Use(idx, this));
  }
}

Instruction::Operand Instruction::getOperand(unsigned idx) const {
  assert(ops_.size() > idx && "Invalid operand");
  return ops_[idx];
}

unsigned Instruction::getNumInputs() const {
  unsigned numInputs = 0;
  for (const auto &op : ops_) {
    if (op.second != OperandKind::Out) {
      numInputs++;
    }
  }
  return numInputs;
}

unsigned Instruction::getNumOutputs() const {
  unsigned numOutputs = 0;
  for (const auto &op : ops_) {
    if (op.second != OperandKind::In) {
      numOutputs++;
    }
  }
  return numOutputs;
}

llvm::StringRef Instruction::getOperandName(unsigned idx) const {
  switch (getKind()) {
#define DEF_INSTR(CLASS, NAME)                                                 \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<const CLASS *>(this)->getOperandName(idx);
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME) DEF_INSTR(CLASS, NAME)
#define DEF_VALUE(CLASS, NAME)
#include "glow/AutoGenInstr.def"
  default:
    llvm_unreachable("Unhandled instruction");
  }
}

void Instruction::eraseFromParent() { getParent()->eraseInstruction(this); }

bool Instruction::verifyUseList(
    const InstructionNumbering &InstrNumbering) const {
  for (const auto &op : ops_) {
    auto *v = op.first;
    (void)v;
    assert(v && "Instruction operand must be a real value");
    assert(v->hasUser(this) && "Invalid use-list");
    v->verifyUseList(InstrNumbering);
  }
  return true;
}

bool Instruction::verify() const {
  // All operands of instructions must be either memory (AllocActivation or
  // WeightVar), or a TensorView of such.
  for (const auto &opPair : getOperands()) {
    const Value *op = opPair.first;
    (void)op;
    assert((isa<AllocActivationInst>(op) || isa<WeightVar>(op) ||
            isa<TensorViewInst>(op)) &&
           "Operands must be one of {AllocActivation, WeightVar, TensorView}");
  }

  // Perform Instruction-specific verification.
#define DEF_INSTR(CLASS, NAME)                                                 \
  if (auto *X = dyn_cast<const CLASS>(this))                                   \
    X->verify();
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME) DEF_INSTR(CLASS, NAME)
#define DEF_VALUE(CLASS, NAME)
#include "glow/AutoGenInstr.def"
  return true;
}

bool Value::verify(const IRFunction &M) const { return true; }

bool Value::verifyUseList(const InstructionNumbering &InstrNumbering) const {
  auto users = getUsers();
  for (const auto &use : users) {
    auto *I = use.get();
    (void)I;
    // Every instruction using this value should be in the instruction list.
    assert(InstrNumbering.getInstrNumber(I) != -1);
    // All uses must come after defs.
    assert(!isa<Instruction>(this) ||
           InstrNumbering.getInstrNumber(I) >
               InstrNumbering.getInstrNumber(cast<Instruction>(this)));
  }
  return true;
}

void Instruction::destroyInstruction(Instruction *I) {
  switch (I->getKind()) {
  default:
    llvm_unreachable("Unknown value kind");
    break;
#define DEF_INSTR(CLASS, NAME)                                                 \
  case Kinded::Kind::CLASS##Kind: {                                            \
    delete llvm::cast<CLASS>(I);                                               \
    break;                                                                     \
  }
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME) DEF_INSTR(CLASS, NAME)
#define DEF_VALUE(CLASS, NAME)
#include "glow/AutoGenInstr.def"
  }
}

void IRFunction::eraseInstruction(glow::Instruction *I) {
  assert(I->getParent() == this &&
         "Cannot erase an instruction not belonging to a function");
  instrs_.erase(I);
}

InstrIterator IRFunction::removeInstruction(glow::Instruction *I) {
  assert(I->getParent() == this &&
         "Cannot erase an instruction not beloning to a function");
  auto result = I->getIterator();
  ++result;
  instrs_.remove(I);
  return result;
}

void IRFunction::insertInstruction(glow::Instruction *I) {
  instrs_.push_back(I);
}

InstrIterator IRFunction::insertInstruction(glow::Instruction *where,
                                            glow::Instruction *I) {
  return instrs_.insert(where->getIterator(), I);
}

InstrIterator IRFunction::moveInstruction(Instruction *where,
                                          glow::Instruction *I) {
  I->getParent()->removeInstruction(I);
  return insertInstruction(where, I);
}

IRFunction::~IRFunction() { clear(); }

void IRFunction::clear() {
  // Remove the mapping between the graph nodes and the IR that we are deleting.
  variableMap_.clear();

  // Delete all of the instructions, in reverse order, to make sure that
  // we delete the users before the instructions.
  for (auto it = instrs_.rbegin(), e = instrs_.rend(); it != e;) {
    auto *curI = &*it;
    ++it;
    Instruction::destroyInstruction(curI);
  }

  // Delete all of the weights.
  for (auto &I : weights_) {
    delete I;
  }
  // TaggedList's destructor is going to destroy the InstList.
  instrs_.clearAndLeakNodesUnsafely();
  weights_.clear();

  G_ = nullptr;
}

static void LLVM_ATTRIBUTE_UNUSED verifyOperandsAccess(const Instruction *I) {
  if (llvm::isa<CopyInst>(I))
    return;
  for (size_t opIdx = 0, e = I->getNumOperands(); opIdx < e; ++opIdx) {
    auto op = I->getOperand(opIdx);
    auto opKind = op.second;
    auto opValue = op.first;
    // Check that an instruction never tries to update a constant argument.
    if (opKind != OperandKind::In) {
      if (auto *W = llvm::dyn_cast<WeightVar>(opValue)) {
        assert(W->getMutability() != WeightVar::MutabilityKind::Constant &&
               "Constant weights cannot be updated");
        (void)W;
      }
    }
    // If the same operand is used multiple times by an instruction,
    // check that it is a valid access pattern.
    for (size_t nextOpIdx = opIdx + 1; nextOpIdx < e; ++nextOpIdx) {
      auto nextOp = I->getOperand(nextOpIdx);
      auto nextOpKind = nextOp.second;
      auto nextOpValue = nextOp.first;
      // Bail if it is a different value.
      if (opValue != nextOpValue)
        continue;
      // It is OK to write into the same buffer if the instruction permits such
      // an inplace update.
      if (opKind == OperandKind::In && nextOpKind != OperandKind::In &&
          Instruction::isInplaceOp(I, nextOpIdx, opIdx))
        continue;
      if (opKind != OperandKind::In && nextOpKind == OperandKind::In &&
          Instruction::isInplaceOp(I, opIdx, nextOpIdx))
        continue;
      // If an operand is used as @out or @inout it cannot be used
      // for anything else.
      // It is OK to use the same operand as input multiple times.
      assert(opKind == OperandKind::In && nextOpKind == OperandKind::In &&
             "Conflicting uses of the same operand by the same instruction");
    }
  }
}

/// Verify that liveness constraints are satisfied.
/// There should be no uses of an allocation after
/// it was deallocated or before it is allocated.
static void verifyLiveness(const IRFunction &M) {
  // The live set stores allocations that are known to be live.
  std::unordered_map<const Value *, bool> liveBuffers;
  for (const auto &I : M.getInstrs()) {
    if (auto *AI = dyn_cast<AllocActivationInst>(&I)) {
      assert(liveBuffers.find(AI) == liveBuffers.end() &&
             "Redefinition of an existing allocation");
      liveBuffers.insert({AI, false});
      continue;
    }
    if (auto *DI = dyn_cast<DeallocActivationInst>(&I)) {
      assert(llvm::isa<AllocActivationInst>(DI->getSrc()) &&
             "Only allocations can be deallocated");
      assert(liveBuffers.find(DI->getSrc()) != liveBuffers.end() &&
             "Deallocation of an allocation that is not alive");
      liveBuffers.erase(DI->getSrc());
      continue;
    }
    // Do not consider tensorview definitions to be real uses of any
    // allocations.
    if (llvm::isa<TensorViewInst>(&I))
      continue;

    for (const auto &Op : I.getOperands()) {
      if (auto *AI = dyn_cast<AllocActivationInst>(getOrigin(Op.first))) {
        auto entry = liveBuffers.find(AI);
        assert(entry != liveBuffers.end() &&
               "Allocation should be alive when it is used");
        assert((Op.second == OperandKind::Out || entry->second) &&
               "@in and @inout operands should be initialized before their "
               "first use");
        // Remember that an allocation was initialized.
        if (Op.second != OperandKind::In)
          entry->second = true;
      }
    }
  }
}

bool IRFunction::verify() const {
  InstructionNumbering InstrNumbering(*this);
  assert(!instrs_.empty() && "Instruction list is empty!");
  llvm::StringSet<> nameSet;
  for (const auto &I : instrs_) {
    I.verifyUseList(InstrNumbering);
    verifyOperandsAccess(&I);
    I.verify();
    auto it = nameSet.insert(I.getName());
    (void)it;
    assert(it.second && "All Instruction and WeightVar names must be unique.");
  }

  verifyLiveness(*this);

  for (auto p : variableMap_) {
    (void)p;
    assert(p.first->getType() == p.second->getType() &&
           "Weight and variable must have the same type");
    p.second->verify(*this);
    p.second->verifyUseList(InstrNumbering);
    auto it = nameSet.insert(p.second->getName());
    (void)it;
    assert(it.second && "All Instruction and WeightVar names must be unique.");
  }
  return true;
}

Value *IRFunction::getWeightForNode(const Storage *V) const {
  auto it = variableMap_.find(V);
  if (it == variableMap_.end()) {
    return nullptr;
  }

  return it->second;
}

bool Instruction::isCanonical() const {
  switch (getKind()) {
  default:
    llvm_unreachable("Unknown value kind");
    break;
#define DEF_INSTR(CLASS, NAME)                                                 \
  case Kinded::Kind::CLASS##Kind: {                                            \
    auto *X = llvm::cast<const CLASS>(this);                                   \
    return X->isCanonical();                                                   \
    break;                                                                     \
  }
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME) DEF_INSTR(CLASS, NAME)
#define DEF_VALUE(CLASS, NAME)
#include "glow/AutoGenInstr.def"
  }
  return false;
}

bool Instruction::isDataParallel() const {
  switch (getKind()) {
  default:
    llvm_unreachable("Unknown value kind");
    break;
#define DEF_INSTR(CLASS, NAME)                                                 \
  case Kinded::Kind::CLASS##Kind: {                                            \
    auto *X = llvm::cast<const CLASS>(this);                                   \
    return X->isDataParallel();                                                \
    break;                                                                     \
  }
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME) DEF_INSTR(CLASS, NAME)
#define DEF_VALUE(CLASS, NAME)
#include "glow/AutoGenInstr.def"
  }
  return false;
}

Instruction *Instruction::clone() const {
  switch (getKind()) {
  default:
    llvm_unreachable("Unknown value kind");
    break;
#define DEF_INSTR(CLASS, NAME)                                                 \
  case Kinded::Kind::CLASS##Kind: {                                            \
    auto *X = llvm::cast<const CLASS>(this);                                   \
    return X->clone();                                                         \
    break;                                                                     \
  }
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME) DEF_INSTR(CLASS, NAME)
#define DEF_VALUE(CLASS, NAME)
#include "glow/AutoGenInstr.def"
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
//                    Instruction numbering
//===----------------------------------------------------------------------===//

InstructionNumbering::InstructionNumbering(const IRFunction &M) {
  auto &instrs = M.getInstrs();
  size_t instIdx = 0;
  for (const auto &I : instrs) {
    numToInstr_.push_back(&I);
    instrToNum_[&I] = instIdx;
    ++instIdx;
  }
}

int64_t InstructionNumbering::getInstrNumber(const Instruction *I) const {
  auto Result = instrToNum_.find(I);
  if (Result == instrToNum_.end())
    return -1;
  return (int64_t)Result->second;
}

const Instruction *InstructionNumbering::getInstr(size_t instrNumber) const {
  assert(instrNumber < numToInstr_.size());
  return numToInstr_[instrNumber];
}

//===----------------------------------------------------------------------===//
//                    IR printing and visualizing
//===----------------------------------------------------------------------===//

static void dumpIR(const Value *V, llvm::raw_ostream &out) {
  switch (V->getKind()) {
  default:
    llvm_unreachable("Unknown value kind");
    break;
#define DEF_INSTR(CLASS, NAME)                                                 \
  case Kinded::Kind::CLASS##Kind: {                                            \
    auto *X = llvm::cast<const CLASS>(V);                                      \
    X->dump(out);                                                              \
    break;                                                                     \
  }
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME) DEF_INSTR(CLASS, NAME)
#define DEF_VALUE(CLASS, NAME) DEF_INSTR(CLASS, NAME)
#include "glow/AutoGenInstr.def"
  }
}

static void dumpIRInContext(const Value *V, llvm::raw_ostream &out) {
  // Dump all operands.
  if (const auto *I = dyn_cast<const Instruction>(V)) {
    if (I->getNumOperands() > 0)
      out << "Operands:\n";
    for (const auto &Op : I->getOperands()) {
      out << "\t";
      Op.first->dump(out);
      out << "\n";
    }
  }
  out << "-> ";

  dumpIR(V, out);

  // Dump all uses.
  out << "\n";
  if (V->getNumUsers() > 0)
    out << "Users:\n";
  for (const Use &U : V->getUsers()) {
    out << "\t";
    U.get()->dump(out);
    out << "\n";
  }
}

std::vector<const Constant *> IRFunction::findConstants() const {
  std::vector<const Constant *> constants;
  for (auto &node : variableMap_) {
    if (const Constant *constant = dyn_cast<const Constant>(node.first)) {
      constants.push_back(constant);
    }
  }
  return constants;
}

std::vector<const Placeholder *> IRFunction::findPlaceholders() const {
  std::vector<const Placeholder *> placeholders;
  for (auto &node : variableMap_) {
    if (const Placeholder *PH = dyn_cast<const Placeholder>(node.first)) {
      placeholders.push_back(PH);
    }
  }
  return placeholders;
}

/// Dump the instruction numbers of all users of \p V.
static void dumpUsers(const Value *V, llvm::raw_ostream &out,
                      InstructionNumbering &InstrNumbering) {
  if (V->getNumUsers() == 0)
    return;
  out << " // Users: ";
  bool isFirst = true;
  for (auto U = V->getUsers().rbegin(), E = V->getUsers().rend(); U != E; ++U) {
    auto *I = U->get();
    if (!isFirst) {
      out << ", ";
    }

    out << getOperandKindStr(U->getOperand().second) << " ";

    auto instrNum = InstrNumbering.getInstrNumber(I);
    assert(instrNum >= 0);
    out << instrNum;
    isFirst = false;
  }
}

void Value::dump(llvm::raw_ostream &out) const { dumpIR(this, out); }

void Value::dump() const { dumpIR(this, llvm::outs()); }

std::string Value::toString() const {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  dump(os);
  return os.str();
}

void Value::dumpInContext(llvm::raw_ostream &out) const {
  dumpIRInContext(this, out);
}

void Value::dumpInContext() const { dumpInContext(llvm::outs()); }

bool Instruction::isInplaceOp(const Instruction *I, unsigned dstIdx,
                              unsigned srcIdx) {
#define DEF_INSTR(CLASS, NAME)                                                 \
  if (const auto *X = dyn_cast<const CLASS>(I))                                \
    return X->isInplaceOp(dstIdx, srcIdx);
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME) DEF_INSTR(CLASS, NAME)
#define DEF_VALUE(CLASS, NAME)
#include "glow/AutoGenInstr.def"

  llvm_unreachable("Invalid instruction kind.");
}

void Instruction::dumpOperands(llvm::raw_ostream &os) const {
  // Dump the predicate of the instruction:
  if (hasPredicate()) {
    Value *pred = getPredicate();
    os << "[ pred: " << pred->getName() << " ] ";
  }

  // Dump the operands of the instruction:
  for (size_t i = 0, e = getNumOperands(); i < e; i++) {
    auto op = getOperand(i);
    auto cc = getOperandKindStr(op.second);
    if (i) {
      os << ", ";
    }
    os << cc << " %" << op.first->getName().str();
  }
}

IRFunction::IRFunction(IRContainer *G)
    : IRContainer(G ? G->getName() : llvm::StringRef{}), G_(G) {}

#if FACEBOOK_INTERNAL
/// \returns a reference to the glow FX graph.
FXIRWrapper *IRFunction::getFXGraph() {
  assert(llvm::isa<FXIRWrapper>(G_));
  return llvm::cast<FXIRWrapper>(G_);
}

/// \returns a reference to the glow FX graph.
const FXIRWrapper *IRFunction::getFXGraph() const {
  assert(llvm::isa<FXIRWrapper>(G_));
  return llvm::cast<FXIRWrapper>(G_);
}
#endif

static bool hasResultValue(const Instruction *I) {
  return I->getKind() == Instruction::Kind::AllocActivationInstKind ||
         I->getKind() == Instruction::Kind::TensorViewInstKind;
}

void IRFunction::dump() const { dump(llvm::outs()); }

void IRFunction::dump(llvm::raw_ostream &OS) const {
  InstructionNumbering InstrNumbering(*this);
  // Print all of the variables:
  std::string s;
  llvm::raw_string_ostream sb{s};
  sb << "function " << getName().str() << "\n";

  size_t sizeInBytes = 0;
  sb << "declare {\n";
  for (auto it : weights_) {
    Value *V = it;
    sb << "  ";
    dumpIR(V, sb);
    sb << " // size: " << V->getSizeInBytes();
    dumpUsers(V, sb, InstrNumbering);
    sb << "\n";

    auto *T = V->getType();
    sizeInBytes += T->getElementSize() * T->size();
  }

  sb << "\n  ; size = " << sizeInBytes << " bytes\n";

  sb << "}\n\n";
  sb << "code {\n";

  // Print all of the instructions:
  for (const auto &I : instrs_) {
    sb << "  ";
    auto InstrNum = InstrNumbering.getInstrNumber(&I);
    assert(InstrNum >= 0);
    sb << InstrNum << " ";
    dumpIR(&I, sb);
    if (isa<AllocActivationInst>(&I))
      sb << " // size: " << I.getSizeInBytes();
    if (isa<DeallocActivationInst>(&I)) {
      sb << " // size: "
         << cast<DeallocActivationInst>(&I)
                ->getSrc()
                ->getType()
                ->getSizeInBytes();
    }
    if (hasResultValue(&I))
      dumpUsers(&I, sb, InstrNumbering);
    sb << "\n";
  }

  sb << "}\n";

  OS << sb.str();
  OS.flush();
}

std::string IRFunction::toString() const {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  dump(os);
  return os.str();
}

IRFunction *
IRFunction::clone(llvm::StringRef newName,
                  llvm::DenseMap<const Value *, Value *> *map,
                  llvm::DenseMap<const Value *, Value *> *currToNewMap) {
  // Create a new function.
  auto *newF = new IRFunction(getRawGraph());
  newF->setName(newName);
  return clone(newF, map, currToNewMap);
}

IRFunction *
IRFunction::clone(IRFunction *newF, llvm::DenseMap<const Value *, Value *> *map,
                  llvm::DenseMap<const Value *, Value *> *currToNewMap) const {

  llvm::DenseMap<const Value *, Value *> currToNew;
  // Initialize the map from a user-provided map.
  if (currToNewMap) {
    currToNew.insert(currToNewMap->begin(), currToNewMap->end());
  }
  // Clone weights.
  for (auto &w : getWeights()) {
    auto cloneWeight =
        new WeightVar(w->getName(), w->getType(), w->getMutability());
    newF->getWeights().emplace_back(cloneWeight);
    currToNew[w] = cloneWeight;
  }
  // Clone the variable map.
  for (auto &v : getVariableMap()) {
    assert(v.second && "The value should have been cloned already");
    newF->getVariableMap()[v.first] = currToNew[v.second];
  }
  // Clone instructions.
  for (const auto &I : getInstrs()) {
    auto *cloneI = I.clone();
    currToNew[&I] = cloneI;
    // Remap all operands.
    for (unsigned idx = 0, e = cloneI->getNumOperands(); idx < e; ++idx) {
      cloneI->setOperand(idx, currToNew[cloneI->getOperand(idx).first]);
    }
    newF->insertInstruction(cloneI);
  }
  // Record the node mapping into the external map.
  if (map) {
    assert(map->empty() && "The external map must be empty");
    for (auto it : currToNew) {
      map->insert(it);
    }
  }
  assert(newF->getInstrs().size() == getInstrs().size() && "Invalid func size");
  assert(newF->getWeights().size() == getWeights().size() &&
         "Invalid func size");
  newF->verify();
  return newF;
}

//===----------------------------------------------------------------------===//
// InstructionTraits<Instruction> Implementation
//===----------------------------------------------------------------------===//

// The trait object is embedded into a IRFunction.  Use dirty hacks to
// reconstruct the IRFunction from the 'self' pointer of the trait.
IRFunction *InstructionTraits::getContainingFunction() {
  size_t Offset(
      size_t(&((IRFunction *)nullptr->*IRFunction::getInstrsMemberPtr())));
  IRFunction::InstListTy *Anchor(static_cast<IRFunction::InstListTy *>(this));
  return reinterpret_cast<IRFunction *>(reinterpret_cast<char *>(Anchor) -
                                        Offset);
}

void InstructionTraits::addNodeToList(Instruction *I) {
  assert(I->getParent() == nullptr && "Already in a list!");
  I->setParent(getContainingFunction());
}

void InstructionTraits::removeNodeFromList(Instruction *I) {
  // When an instruction is removed from a function, clear the parent pointer.
  assert(I->getParent() && "Not in a list!");
  I->setParent(nullptr);
}

namespace glow {

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Value &V) {
  V.dump(os);
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Value *V) {
  assert(V != nullptr && "Null Pointer.");
  V->dump(os);
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const IRFunction &irf) {
  irf.dump(os);
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const IRFunction *irf) {
  assert(irf != nullptr && "Null Pointer.");
  irf->dump(os);
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, InstrumentKind kind) {
  switch (kind) {
  case InstrumentKind::Before:
    os << "Before";
    break;
  case InstrumentKind::After:
    os << "After";
    break;
  default:
    llvm_unreachable("Unknown instrumentation kind");
  }
  return os;
}
} // namespace glow
