#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Casting.h"
#include "glow/Support/Support.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_set>

using namespace glow;

TypeRef Module::uniqueType(ElemKind elemTy, ArrayRef<size_t> dims) {
  return uniqueType(Type(elemTy, dims));
}

TypeRef Module::uniqueType(const Type &T) {
  for (auto &tp : types_) {
    if (T.isEqual(tp))
      return &tp;
  }

  return &*types_.insert(types_.begin(), T);
}

TypeRef Module::getVoidTy() { return uniqueType(Type()); }

void Instruction::pushOperand(Operand op) {
  ops_.push_back({nullptr, op.second});
  setOperand(ops_.size() - 1, op.first);
}

void Instruction::setOperand(unsigned idx, Value *v) {
  auto *currVal = ops_[idx].first;

  if (currVal == v)
    return;

  if (currVal) {
    currVal->removeUse({idx, this});
  }

  if (v) {
    ops_[idx].first = v;
    v->addUse({idx, this});
  }
}

Instruction::Operand Instruction::getOperand(unsigned idx) const {
  assert(ops_.size() > idx && "Invalid operand");
  return ops_[idx];
}

void Instruction::verifyUseList() {
  for (int i = 0, e = ops_.size(); i < e; i++) {
    auto *v = ops_[i].first;
    (void)v;
    assert(v && "Instruction operand must be a real value");
    assert(v->hasUser(this) && "Invalid use-list");
  }
}

void Instruction::verify() {
#define DEF_INSTR(CLASS, NAME)                                                 \
  if (auto *X = dyn_cast<CLASS>(this))                                         \
    X->verify();
#define DEF_VALUE(CLASS, NAME)
#include "glow/IR/Instrs.def"
#undef DEF_INSTR
#undef DEF_VALUE
}

Module::~Module() {
  // Delete all of the instructions, in reverse order, to make sure that
  // we delete the users before the instructions.
  for (auto it = instrs_.rbegin(), e = instrs_.rend(); it != e; ++it) {
    delete *it;
  }

  // Delete all of the constants.
  for (auto &I : variables_) {
    delete I;
  }
}

void Module::verify() {
  for (auto it : instrs_) {
    it->verifyUseList();
    it->verify();
  }
}

static std::string getExtraDesc(Kinded *K) {
#define DEF_INSTR(CLASS, NAME)                                                 \
  if (auto *X = dyn_cast<CLASS>(K))                                            \
    return X->getExtraDesc();
#define DEF_VALUE(CLASS, NAME)                                                 \
  if (auto *X = dyn_cast<CLASS>(K))                                            \
    return X->getExtraDesc();
#include "glow/IR/Instrs.def"
#undef DEF_INSTRE
#undef DEF_VALUE

  glow_unreachable();
}

static std::string getDesc(Value *v) {
  std::string sb;
  auto name = v->getName();
  auto valName = v->getKindName();
  sb += "%" + name + " = " + valName + " ";
  sb += getExtraDesc(v);
  return sb;
}

static std::string getDesc(Instruction *II) {
  std::string sb;

  auto name = II->getName();
  auto instrName = II->getKindName();
  sb += "%" + name + " = " + instrName + " ";
  auto extraDesc = getExtraDesc(II);
  ;
  if (extraDesc.size()) {
    sb += extraDesc + " ";
  }

  // Print operands:
  for (int i = 0, e = II->getNumOperands(); i < e; i++) {
    auto op = II->getOperand(i);
    auto CC = getOperandKindStr(op.second);
    if (i) {
      sb += ", ";
    }
    sb += std::string(CC) + " %" + op.first->getName();
  }

  return sb;
}

void Module::nameInstructions() {
  std::unordered_set<std::string> usedNames;
  unsigned idx = 0;

  // Name all unnamed variables.
  for (auto it : variables_) {
    Value *V = it;

    if (!V->hasName())
      V->setName("v");

    while (!usedNames.insert(V->getName()).second) {
      V->setName(V->getName() + "." + std::to_string(idx++));
    }
  }

  idx = 0;

  // Name all unnamed instructions.
  for (auto it : instrs_) {
    Instruction *I = it;

    if (!I->hasName())
      I->setName("i");

    while (!usedNames.insert(I->getName()).second) {
      I->setName(I->getName() + "." + std::to_string(idx++));
    }
  }
}

void Module::dump() {
  nameInstructions();
  // Print all of the variables:
  std::stringstream sb;

  sb << "declare {\n";
  for (auto it : variables_) {
    Value *V = it;
    sb << "  " << getDesc(V) << "\n";
  }

  sb << "}\n\n";
  sb << "program {\n";

  // Print all of the instructions:
  for (auto it : instrs_) {
    Instruction *II = it;
    sb << "  " << getDesc(II) << "\n";
  }

  sb << "}\n";

  std::cout << sb.str();
}

static std::string quote(const std::string &in) { return '"' + in + '"'; }

static std::string getDottyDesc(Value *v) {
  std::string sb;
  auto name = v->getName();
  auto valName = v->getKindName();
  sb += name + " | " + valName + " ";
  sb += getExtraDesc(v);
  return sb;
}

static std::string getDottyDesc(Instruction *II) {
  std::string sb;

  auto name = II->getName();
  auto instrName = II->getKindName();
  sb += instrName;
  sb += "|";
  auto extraDesc = getExtraDesc(II);
  if (extraDesc.size()) {
    sb += extraDesc + "|";
  }

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
  case glow::OperandKind::kOut:
    return "forward";
    break;

  case glow::OperandKind::kIn:
    return "back";
    break;

  case glow::OperandKind::kInOut:
    return "both";
    break;
  }
}

/// Dump a dotty graph that depicts the module.
void Module::dumpDAG() {
  std::string filename = "dotty_network_dump_" + pointerToString(this) + ".dot";
  std::cout << "Writing dotty graph to: " << filename << '\n';

  std::string sb;
  sb += "digraph finite_state_machine {\n\trankdir=LR;\n";

  sb += "subgraph cluster_1 {";
  sb += "  style=invis;\n";

  for (auto &I : instrs_) {
    std::string desc = getDottyDesc(I);

    sb += quote(pointerToString(I)) + "[\n";
    std::string repr = quote(desc);
    sb += "\tlabel = " + repr + "\n";
    sb += "\tshape = \"record\"\n";
    sb += "];\n\n";
  }
  sb += "}";

  sb += "subgraph cluster_0 {";
  sb += "  style=invis;\n";
  for (auto &v : variables_) {
    sb += quote(pointerToString(v)) + "[\n";
    std::string desc = escapeDottyString(getDottyDesc(v));
    sb += "\tlabel = " + desc + "\n";
    sb += "\tshape = \"record\"\n";
    sb += "\tstyle=filled\n";
    sb += "];\n\n";
  }
  sb += "}";

  // Dump the use-def edges.
  for (auto &I : instrs_) {
    for (int i = 0, e = I->getNumOperands(); i < e; i++) {
      auto op = I->getOperand(i);
      std::string from = quote(pointerToString(I)) + ":f" + std::to_string(i);
      std::string to = quote(pointerToString(op.first));

      sb += from + "->" + to + "[dir=" + getDottyArrowForCC(op.second) + "];\n";
    }
  }

  sb += "}";

  std::ofstream myfile;
  myfile.open(filename);
  myfile << sb;
  myfile.close();
}
