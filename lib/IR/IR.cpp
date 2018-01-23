// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/IR/IR.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IRUtils.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Support.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_set>

using namespace glow;
using llvm::dyn_cast;
using llvm::isa;

//===----------------------------------------------------------------------===//
//                       General IR operations
//===----------------------------------------------------------------------===//

Value *glow::getAllocationOrigin(Value *V) {
  while (true) {
    if (auto *AI = dyn_cast<AllocActivationInst>(V))
      return AI;
    if (auto *TVI = dyn_cast<TensorViewInst>(V)) {
      V = TVI->getSrc();
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

Value *glow::getOrigin(Value *V) {
  while (true) {
    auto *TVI = dyn_cast<TensorViewInst>(V);
    if (!TVI)
      return V;
    V = TVI->getSrc();
  }
  return V;
}

bool glow::isTensorView(glow::Value *v) { return isa<TensorViewInst>(v); }

bool Instruction::classof(const Value *V) {
#define DEF_VALUE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME)
#define DEF_INSTR_RANGE(CLASS, FIRST, LAST)                                    \
  constexpr auto First_##CLASS = Kinded::Kind::FIRST##Kind;                    \
  constexpr auto Last_##CLASS = Kinded::Kind::LAST##Kind;
#include "AutoGenInstr.def"
  return V->getKind() >= First_Instruction && V->getKind() <= Last_Instruction;
}

void Use::setOperand(Value *other) { use_->setOperand(idx_, other); }

InstructionOperand Use::getOperand() { return use_->getOperand(idx_); }

ConstInstructionOperand Use::getOperand() const {
  return use_->getOperand(idx_);
}

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

void Instruction::eraseFromParent() { getParent()->eraseInstruction(this); }

void Instruction::verifyUseList() const {
  for (const auto &op : ops_) {
    auto *v = op.first;
    (void)v;
    assert(v && "Instruction operand must be a real value");
    assert(v->hasUser(this) && "Invalid use-list");
    v->verifyUseList(*getParent());
  }
}

void Instruction::verify() const {
#define DEF_INSTR(CLASS, NAME)                                                 \
  if (auto *X = dyn_cast<const CLASS>(this))                                   \
    X->verify();
#define DEF_VALUE(CLASS, NAME)
#include "AutoGenInstr.def"
}

void Value::verify(const Module &M) const {}

void Value::verifyUseList(const Module &M) const {
  auto Users = getUsers();
  auto Instrs = M.getInstrs();
  for (const auto &Use : Users) {
    auto *I = Use.get();
    (void)I;
    // Every instruction using this value should be in the instruction list.
    assert(std::find(Instrs.begin(), Instrs.end(), I) != Instrs.end());
  }
}

void Module::destroyInstruction(Instruction *I) {
  switch (I->getKind()) {
  default:
    llvm_unreachable("Unknown value kind");
    break;
#define DEF_INSTR(CLASS, NAME)                                                 \
  case Kinded::Kind::CLASS##Kind: {                                            \
    delete llvm::cast<CLASS>(I);                                               \
    break;                                                                     \
  }
#define DEF_VALUE(CLASS, NAME)
#include "AutoGenInstr.def"
  }
}

InstrIterator Module::eraseInstruction(InstrIterator it) {
  auto *I = *it;
  assert(std::find(instrs_.begin(), instrs_.end(), I) != instrs_.end() &&
         "Cannot erase an instruction not belonging to a module");
  destroyInstruction(I);
  auto result = instrs_.erase(it);
  assert(std::find(instrs_.begin(), instrs_.end(), I) == instrs_.end() &&
         "Instruction should be erased");
  return result;
}

void Module::eraseInstruction(glow::Instruction *I) {
  // find the instruction inside the module.
  auto it = std::find(instrs_.begin(), instrs_.end(), I);
  assert(it != instrs_.end() &&
         "Cannot erase an instruction not belonging to a module");
  eraseInstruction(it);
}

InstrIterator Module::removeInstruction(InstrIterator it) {
  auto *I = *it;
  (void)I;
  assert(std::find(instrs_.begin(), instrs_.end(), I) != instrs_.end() &&
         "Cannot remove an instruction not belonging to a module");
  auto result = instrs_.erase(it);
  assert(std::find(instrs_.begin(), instrs_.end(), I) == instrs_.end() &&
         "Instruction should be removed");
  return result;
}

void Module::removeInstruction(glow::Instruction *I) {
  // find the instruction inside the module.
  auto it = std::find(instrs_.begin(), instrs_.end(), I);
  assert(it != instrs_.end() &&
         "Cannot remove an instruction not belonging to a module");
  removeInstruction(it);
}

void Module::insertInstruction(glow::Instruction *I) {
  instrs_.push_back(I);
  I->setParent(this);
}

InstrIterator Module::insertInstruction(InstrIterator where,
                                        glow::Instruction *I) {
  I->setParent(this);
  return instrs_.insert(where, I);
}

InstrIterator Module::moveInstruction(InstrIterator where,
                                      glow::Instruction *I) {
  I->getParent()->removeInstruction(I);
  return insertInstruction(where, I);
}

InstrIterator Module::moveInstruction(const Instruction *where,
                                      glow::Instruction *I) {
  I->getParent()->removeInstruction(I);
  return insertInstruction(getInstrIterator(where), I);
}

InstrIterator Module::getInstrIterator(const Instruction *I) {
  auto it = std::find(instrs_.begin(), instrs_.end(), I);
  assert(it != instrs_.end() && "Instruction should be present");
  return it;
}

Module::InstListTy::const_iterator
Module::getInstrIterator(const Instruction *I) const {
  auto it = std::find(instrs_.begin(), instrs_.end(), I);
  assert(it != instrs_.end() && "Instruction should be present");
  return it;
}

Module::~Module() { clear(); }

void Module::clear() {
  // Remove the mapping between the graph nodes and the IR that we are deleting.
  variableMap.clear();

  // Delete all of the instructions, in reverse order, to make sure that
  // we delete the users before the instructions.
  for (auto it = instrs_.rbegin(), e = instrs_.rend(); it != e; ++it) {
    destroyInstruction(*it);
  }

  // Delete all of the weights.
  for (auto &I : weights_) {
    delete I;
  }
  instrs_.clear();
  weights_.clear();
}

static void LLVM_ATTRIBUTE_UNUSED verifyOperandsAccess(Instruction *I) {
  if (llvm::isa<CopyInst>(I))
    return;
  for (size_t opIdx = 0, e = I->getNumOperands(); opIdx < e; ++opIdx) {
    auto Op = I->getOperand(opIdx);
    auto OpKind = Op.second;
    auto OpValue = Op.first;
    // Check that an instruction never tries to update a constant argument.
    if (OpKind != OperandKind::In) {
      if (auto *W = llvm::dyn_cast<WeightVar>(OpValue)) {
        assert(W->getMutability() != WeightVar::MutabilityKind::Constant &&
               "Constant weights cannot be updated");
      }
    }
    // If the same operand is used multiple times by an instruction,
    // check that it is a valid access pattern.
    for (size_t nextOpIdx = opIdx + 1; nextOpIdx < e; ++nextOpIdx) {
      auto NextOp = I->getOperand(nextOpIdx);
      auto NextOpKind = NextOp.second;
      auto NextOpValue = NextOp.first;
      // Bail if it is a different value.
      if (OpValue != NextOpValue)
        continue;
      // It is OK to write into the same buffer if the instruction permits such
      // an inplace update.
      if (OpKind == OperandKind::In && NextOpKind != OperandKind::In &&
          Instruction::isInplaceOp(I, nextOpIdx, opIdx))
        continue;
      if (OpKind != OperandKind::In && NextOpKind == OperandKind::In &&
          Instruction::isInplaceOp(I, opIdx, nextOpIdx))
        continue;
      // If an operand is used as @out or @inout it cannot be used
      // for anything else.
      // It is OK to use the same operand as input multiple times.
      assert(OpKind == OperandKind::In && NextOpKind == OperandKind::In &&
             "Conflicting uses of the same operand by the same instruction");
    }
  }
}

/// Verify that liveness constraints are satisfied.
/// There should be no uses of an allocation after
/// it was deallocated or before it is allocated.
static void verifyLiveness(const Module &M) {
  // The live set stores allocations that are known to be live.
  std::unordered_map<Value *, bool> liveBuffers;
  for (auto *I : M.getInstrs()) {
    if (auto *AI = dyn_cast<AllocActivationInst>(I)) {
      assert(liveBuffers.find(AI) == liveBuffers.end() &&
             "Redefinition of an existing allocation");
      liveBuffers.insert({AI, false});
      continue;
    }
    if (auto *DI = dyn_cast<DeallocActivationInst>(I)) {
      assert(llvm::isa<AllocActivationInst>(DI->getSrc()) &&
             "Only allocations can be deallocated");
      assert(liveBuffers.find(DI->getSrc()) != liveBuffers.end() &&
             "Deallocation of an allocation that is not alive");
      liveBuffers.erase(DI->getSrc());
      continue;
    }
    // Do not consider tensorview definitions to be real uses of any
    // allocations.
    if (llvm::isa<TensorViewInst>(I))
      continue;

    for (const auto &Op : I->getOperands()) {
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

void Module::verify() const {
  assert(!instrs_.empty() && "Instruction list is empty!");
  for (auto it : instrs_) {
    it->verifyUseList();
    verifyOperandsAccess(it);
    it->verify();
  }

  verifyLiveness(*this);

  for (auto p : variableMap) {
    (void)p;
    assert(p.first->getType() == p.second->getType() &&
           "Weight and variable must have the same type");
    p.second->verify(*this);
    p.second->verifyUseList(*this);
  }
}

Value *Module::getWeightForNode(const Node *V) const {
  auto it = variableMap.find(V);
  if (it == variableMap.end()) {
    return nullptr;
  }

  return it->second;
}

//===----------------------------------------------------------------------===//
//                    Instruction numbering
//===----------------------------------------------------------------------===//

InstructionNumbering::InstructionNumbering(Module &M) : M_(M) {
  auto &instrs = M.getInstrs();
  size_t instIdx = 0;
  for (auto it = instrs.begin(), e = instrs.end(); it != e;
       instIdx += MAX_SLOT, ++it) {
    NumToInstr_.push_back(it);
    InstrToNum_[*it] = instIdx;
  }
}

int64_t InstructionNumbering::getInstrNumber(Instruction *I) const {
  auto Result = InstrToNum_.find(I);
  if (Result == InstrToNum_.end())
    return -1;
  return (int64_t)Result->second;
}

int64_t InstructionNumbering::getInstrNumber(InstrIterator IT) const {
  return getInstrNumber(*IT);
}

InstrIterator InstructionNumbering::getInstr(size_t InstrNumber) const {
  assert(InstrNumber / MAX_SLOT < NumToInstr_.size());
  return NumToInstr_[InstrNumber / MAX_SLOT];
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
#define DEF_VALUE(CLASS, NAME)                                                 \
  case Kinded::Kind::CLASS##Kind: {                                            \
    auto *X = llvm::cast<const CLASS>(V);                                      \
    X->dump(out);                                                              \
    break;                                                                     \
  }
#include "AutoGenInstr.def"
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

/// Dump the instruction numbers of all users of \p V.
static void dumpUsers(const Value *V, llvm::raw_ostream &out,
                      InstructionNumbering &IN) {
  if (V->getNumUsers() == 0)
    return;
  out << " // Users: ";
  bool IsFirst = true;
  for (auto U = V->getUsers().rbegin(), E = V->getUsers().rend(); U != E; ++U) {
    auto *I = U->get();
    if (!IsFirst) {
      out << ", ";
    }

    out << getOperandKindStr(U->getOperand().second) << " ";

    auto InstrNum = IN.getInstrNumber(I);
    assert(InstrNum >= 0);
    out << InstrNum;
    IsFirst = false;
  }
}

void Value::dump(llvm::raw_ostream &out) const { dumpIR(this, out); }

void Value::dump() const { dumpIR(this, llvm::outs()); }

void Value::dumpInContext(llvm::raw_ostream &out) const {
  dumpIRInContext(this, out);
}

void Value::dumpInContext() const { dumpInContext(llvm::outs()); }

bool Instruction::isInplaceOp(const Instruction *I, unsigned dstIdx,
                              unsigned srcIdx) {
#define DEF_INSTR(CLASS, NAME)                                                 \
  if (const auto *X = dyn_cast<const CLASS>(I))                                \
    return X->isInplaceOp(dstIdx, srcIdx);
#define DEF_VALUE(CLASS, NAME)
#include "AutoGenInstr.def"

  llvm_unreachable("Invalid instruction kind.");
}

void Instruction::dumpOperands(llvm::raw_ostream &os) const {
  for (size_t i = 0, e = getNumOperands(); i < e; i++) {
    auto op = getOperand(i);
    auto CC = getOperandKindStr(op.second);
    if (i) {
      os << ", ";
    }
    os << CC << " %" << op.first->getName().str();
  }
}

static void nameInstr(std::unordered_set<std::string> &usedNames, Named *named,
                      llvm::StringRef suggestion) {
  unsigned idx = 0;

  // Use the first few letters of the value as the initial name.
  if (!named->hasName()) {
    named->setName(suggestion.slice(0, 4));
  }

  std::string tempName = named->getName();

  while (!usedNames.insert(tempName).second) {
    tempName = named->getName().str() + std::to_string(idx++);
  }

  named->setName(tempName);
}

Module::Module(Graph *G) : G_(G), name_(G->getName()) {}

static bool hasResultValue(Instruction *I) {
  return I->getKind() == Instruction::Kind::AllocActivationInstKind ||
         I->getKind() == Instruction::Kind::TensorViewInstKind;
}

void Module::nameInstructions() {
  std::unordered_set<std::string> usedNames;
  for (auto &v : weights_) {
    nameInstr(usedNames, v, v->getKindName());
  }
  for (auto &v : instrs_) {
    nameInstr(usedNames, v, v->getKindName());
  }
}

void Module::dump() {
  nameInstructions();
  InstructionNumbering InstrNumbering(*this);
  // Print all of the variables:
  std::string s;
  llvm::raw_string_ostream sb{s};
  sb << "module " << G_->getName().str() << "\n";

  size_t sizeInBytes = 0;
  sb << "declare {\n";
  for (auto it : weights_) {
    Value *V = it;
    sb << "  ";
    dumpIR(V, sb);
    sb << " // size: " << V->getType()->getSizeInBytes();
    dumpUsers(V, sb, InstrNumbering);
    sb << "\n";

    auto *T = V->getType();
    sizeInBytes += T->getElementSize() * T->size();
  }

  sb << "\n  ; size = " << sizeInBytes << " bytes\n";

  sb << "}\n\n";
  sb << "program {\n";

  // Print all of the instructions:
  for (auto it : instrs_) {
    Instruction *II = it;
    sb << "  ";
    auto InstrNum = InstrNumbering.getInstrNumber(II);
    assert(InstrNum >= 0);
    sb << InstrNum << " ";
    dumpIR(II, sb);
    if (isa<AllocActivationInst>(II))
      sb << " // size: " << II->getType()->getSizeInBytes();
    if (isa<DeallocActivationInst>(II))
      sb << " // size: "
         << II->getOperand(0).first->getType()->getSizeInBytes();
    if (hasResultValue(II))
      dumpUsers(II, sb, InstrNumbering);
    sb << "\n";
  }

  sb << "}\n";

  llvm::outs() << sb.str();
}

static std::string getEscapedDottyType(const TypeRef &type) {
  std::string buffer;
  llvm::raw_string_ostream stream(buffer);
  stream << type;
  return escapeDottyString(stream.str());
}

static std::string getDottyDesc(const Value *v) {
  std::string buffer;
  llvm::raw_string_ostream stream(buffer);
  stream << v->getKindName() << " | " << v->getName() << " | "
         << getEscapedDottyType(v->getType());
  return stream.str();
}

static std::string getDottyDesc(const Instruction *II) {
  std::string buffer;
  llvm::raw_string_ostream stream(buffer);
  stream << II->getKindName();
  stream << "|" << getEscapedDottyType(II->getType()) << "|";

  // Print operands:
  for (int i = 0, e = II->getNumOperands(); i < e; i++) {
    const auto op = II->getOperand(i);
    if (i) {
      stream << "|";
    }
    stream << " <f" << i << ">";
    stream << op.first->getName();
  }

  return stream.str();
}

/// \returns the arrow property for the operand kind \p k. This method is used
/// for printing edges in the dotty printer.
static const char *getDottyArrowForCC(OperandKind k) {
  switch (k) {
  case glow::OperandKind::Out:
    return "forward";
    break;

  case glow::OperandKind::In:
    return "back";
    break;

  case glow::OperandKind::InOut:
    return "both";
    break;
  }
  llvm_unreachable("Invalid operand kind.");
}

void Module::dumpDAG() {
  std::string buffer;
  llvm::raw_string_ostream stream(buffer);
  stream << "dotty_ir_dump_" << this << ".dot";
  dumpDAG(stream.str().c_str());
}

/// Dump a dotty graph that depicts the module.
void Module::dumpDAG(const char *dotFilename) {
  std::string filename = dotFilename;
  llvm::outs() << "Writing dotty graph to: " << filename << '\n';

  std::string buffer;
  llvm::raw_string_ostream stream(buffer);

  stream << "digraph finite_state_machine {\n\trankdir=LR;\n";

  stream << "subgraph cluster_1 {";
  stream << "  style=invis;\n";

  for (auto &I : instrs_) {
    std::string desc = getDottyDesc(I);

    stream << '"' << I << "\"[\n";
    std::string repr = quote(desc);
    stream << "\tlabel = " << repr << "\n";
    stream << "\tshape = \"record\"\n";
    stream << "];\n\n";
  }
  stream << "}";

  stream << "subgraph cluster_0 {";
  stream << "  style=invis;\n";

  for (auto &v : weights_) {
    stream << '"' << v << "\"[\n";
    std::string desc = getDottyDesc(v);
    stream << "\tlabel = " << quote(desc) << "\n";
    stream << "\tshape = \"record\"\n";
    stream << "\tfillcolor=pink,style=filled\n";
    stream << "];\n\n";
  }
  stream << "}";

  stream << "subgraph cluster_1 {";
  stream << "  style=invis;\n";

  // Dump the use-def edges.
  for (auto &I : instrs_) {
    for (int i = 0, e = I->getNumOperands(); i < e; i++) {
      auto op = I->getOperand(i);
      stream << '"' << I << "\":f" << i << "->\"" << op.first
             << "\"[dir=" << getDottyArrowForCC(op.second) << "];\n";
    }
  }

  // Dump the order edges.
  Instruction *prev = nullptr;
  for (auto &I : instrs_) {
    if (prev) {
      stream << '"' << prev << "\"->\"" << I << "\"[color=\"blue\"];\n";
    }
    prev = I;
  }
  stream << "}";
  stream << "}";

  std::ofstream filestream(dotFilename);
  filestream << stream.str();
}
