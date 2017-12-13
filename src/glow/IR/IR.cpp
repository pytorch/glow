// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/IR/IR.h"
#include "glow/Graph/Graph.h"
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

//===----------------------------------------------------------------------===//
//                       General IR operations
//===----------------------------------------------------------------------===//

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
  for (auto Use : Users) {
    auto *I = Use.get();
    (void)I;
    // Every instruction using this value should be in the instruction list.
    assert(std::find(Instrs.begin(), Instrs.end(), I) != Instrs.end());
  }
}

InstrIterator Module::eraseInstruction(InstListTy::iterator it) {
  auto *I = *it;
  assert(std::find(instrs_.begin(), instrs_.end(), I) != instrs_.end() &&
         "Cannot erase an instruction not belonging to a module");
  delete I;
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

InstrIterator Module::removeInstruction(InstListTy::iterator it) {
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
  eraseInstruction(it);
}

void Module::insertInstruction(glow::Instruction *I) {
  instrs_.push_back(I);
  I->setParent(this);
}

void Module::insertInstruction(InstListTy::iterator where,
                               glow::Instruction *I) {
  instrs_.insert(where, I);
  I->setParent(this);
}

Module::~Module() { clear(); }

void Module::clear() {
  // Remove the mapping between the graph nodes and the IR that we are deleting.
  variableMap.clear();
  gradientMap.clear();

  // Delete all of the instructions, in reverse order, to make sure that
  // we delete the users before the instructions.
  for (auto it = instrs_.rbegin(), e = instrs_.rend(); it != e; ++it) {
    delete *it;
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

void Module::verify() const {
  assert(!instrs_.empty() && "Instruction list is empty!");
  for (auto it : instrs_) {
    it->verifyUseList();
    verifyOperandsAccess(it);
    it->verify();
  }

#if 0
  // gradientMap will soon be removed. Once we do it,
  // this whole check should be removed.
  for (auto p : gradientMap) {
    (void)p;
    assert(p.first->getType() == p.second->getType() &&
           "Weight and gradient must have the same type");
    p.second->verify(*this);
    p.second->verifyUseList(*this);
  }
#endif

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
  for (auto it = instrs.begin(), e = instrs.end(); it != e; ++instIdx, ++it) {
    NumToInstr_.push_back(it);
    InstrToNum_[*it] = instIdx;
  }
}

int64_t InstructionNumbering::getInstrNumber(Instruction *I) {
  auto Result = InstrToNum_.find(I);
  if (Result == InstrToNum_.end())
    return -1;
  return (int64_t)Result->second;
}

int64_t InstructionNumbering::getInstrNumber(InstrIterator IT) {
  return getInstrNumber(*IT);
}

InstrIterator InstructionNumbering::getInstr(size_t InstrNumber) {
  assert(InstrNumber < NumToInstr_.size());
  return NumToInstr_[InstrNumber];
}

//===----------------------------------------------------------------------===//
//                    IR printing and visualizing
//===----------------------------------------------------------------------===//

static void dumpIR(Value *V, llvm::raw_ostream &out) {
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

static void dumpIRInContext(Value *V, llvm::raw_ostream &out) {
  // Dump all operands.
  if (const auto *I = dyn_cast<const Instruction>(V)) {
    if (I->getNumOperands() > 0)
      out << "Operands:\n";
    for (auto &Op : I->getOperands()) {
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
  for (Use &U : V->getUsers()) {
    out << "\t";
    U.get()->dump(out);
    out << "\n";
  }
}

/// Dump the instruction numbers of all users of \p V.
static void dumpUsers(Value *V, llvm::raw_ostream &out,
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
    auto InstrNum = IN.getInstrNumber(I);
    assert(InstrNum >= 0);
    out << InstrNum;
    IsFirst = false;
  }
}

void Value::dump(llvm::raw_ostream &out) { dumpIR(this, out); }

void Value::dump() { dumpIR(this, llvm::outs()); }

void Value::dumpInContext(llvm::raw_ostream &out) {
  dumpIRInContext(this, out);
}

void Value::dumpInContext() { dumpInContext(llvm::outs()); }

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
  return I->getKind() == Instruction::Kind::AllocActivationInstKind;
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
    if (hasResultValue(II))
      dumpUsers(II, sb, InstrNumbering);
    sb << "\n";
  }

  sb << "}\n";

  llvm::outs() << sb.str();
}

static std::string getDottyDesc(const Value *v) {
  std::string sb;
  std::string name = v->getName();
  std::string valName = v->getKindName();
  sb += valName + " | " + name + " | " +
        escapeDottyString(std::to_string(*v->getType()));
  return sb;
}

static std::string getDottyDesc(const Instruction *II) {
  std::string sb;
  sb += II->getKindName();
  sb += "|" + escapeDottyString(std::to_string(II->getType())) + "|";

  // Print operands:
  for (int i = 0, e = II->getNumOperands(); i < e; i++) {
    auto op = II->getOperand(i);
    if (i) {
      sb += "|";
    }
    sb += " <f" + std::to_string(i) + ">";
    sb += op.first->getName();
  }

  return sb;
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

/// Dump a dotty graph that depicts the module.
void Module::dumpDAG() {
  std::string filename = "dotty_ir_dump_" + std::to_string(this) + ".dot";
  llvm::outs() << "Writing dotty graph to: " << filename << '\n';

  std::ofstream os;
  os.open(filename);

  os << "digraph finite_state_machine {\n\trankdir=LR;\n";

  os << "subgraph cluster_1 {";
  os << "  style=invis;\n";

  for (auto &I : instrs_) {
    std::string desc = getDottyDesc(I);

    os << quote(std::to_string(I)) << "[\n";
    std::string repr = quote(desc);
    os << "\tlabel = " << repr << "\n";
    os << "\tshape = \"record\"\n";
    os << "];\n\n";
  }
  os << "}";

  os << "subgraph cluster_0 {";
  os << "  style=invis;\n";

  for (auto &v : weights_) {
    os << quote(std::to_string(v)) + "[\n";
    std::string desc = getDottyDesc(v);
    os << "\tlabel = " << quote(desc) << "\n";
    os << "\tshape = \"record\"\n";
    os << "\tfillcolor=pink,style=filled\n";
    os << "];\n\n";
  }
  os << "}";

  os << "subgraph cluster_1 {";
  os << "  style=invis;\n";

  // Dump the use-def edges.
  for (auto &I : instrs_) {
    for (int i = 0, e = I->getNumOperands(); i < e; i++) {
      auto op = I->getOperand(i);
      std::string from = quote(std::to_string(I)) + ":f" + std::to_string(i);
      std::string to = quote(std::to_string(op.first));

      os << from + "->" << to << "[dir=" << getDottyArrowForCC(op.second)
         << "];\n";
    }
  }

  // Dump the order edges.
  Instruction *prev = nullptr;
  for (auto &I : instrs_) {
    if (prev) {
      std::string from = quote(std::to_string(prev));
      std::string to = quote(std::to_string(I));
      os << from << "->" << to << "[color=\"blue\"];\n";
    }
    prev = I;
  }
  os << "}";
  os << "}";
  os.close();
}
