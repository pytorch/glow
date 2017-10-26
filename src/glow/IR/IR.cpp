// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Casting.h"
#include "glow/Support/Support.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_set>

using namespace glow;

//===----------------------------------------------------------------------===//
//                       General IR operations
//===----------------------------------------------------------------------===//

void Use::setOperand(Value *other) { use_->setOperand(idx_, other); }

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

void Instruction::verifyUseList() const {
  for (const auto &op : ops_) {
    auto *v = op.first;
    (void)v;
    assert(v && "Instruction operand must be a real value");
    assert(v->hasUser(this) && "Invalid use-list");
  }
}

void Instruction::verify() const {
#define DEF_INSTR(CLASS, NAME)                                                 \
  if (auto *X = dyn_cast<const CLASS>(this))                                   \
    X->verify();
#define DEF_VALUE(CLASS, NAME)
#include "AutoGenInstr.def"
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

void Module::verify() const {
  assert(!instrs_.empty() && "Instruction list is empty!");
  for (auto it : instrs_) {
    it->verifyUseList();
    it->verify();
  }

  for (auto p : gradientMap) {
    (void)p;
    assert(p.first->getType() == p.second->getType() &&
           "Weight and gradient must have the same type");
  }

  for (auto p : variableMap) {
    (void)p;
    assert(p.first->getType() == p.second->getType() &&
           "Weight and variable must have the same type");
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
//                    IR printing and visualizing
//===----------------------------------------------------------------------===//

static void dumpIR(Value *V, std::ostream &out) {
#define DEF_INSTR(CLASS, NAME)                                                 \
  if (const auto *X = dyn_cast<const CLASS>(V))                                \
    return X->dump(out);
#define DEF_VALUE(CLASS, NAME)                                                 \
  if (const auto *X = dyn_cast<const CLASS>(V))                                \
    return X->dump(out);
#include "AutoGenInstr.def"
  glow_unreachable();
}

bool Instruction::mayShareBuffers(const Instruction *I) {
#define DEF_INSTR(CLASS, NAME)                                                 \
  if (const auto *X = dyn_cast<const CLASS>(I))                                \
    return X->mayShareBuffers();
#define DEF_VALUE(CLASS, NAME)
#include "AutoGenInstr.def"

  glow_unreachable();
}

void Instruction::dumpOperands(std::ostream &os) const {
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
  // Print all of the variables:
  std::stringstream sb;

  size_t sizeInBytes = 0;
  sb << "declare {\n";
  for (auto it : weights_) {
    Value *V = it;
    sb << "  ";
    dumpIR(V, sb);
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
    dumpIR(II, sb);
    sb << "\n";
  }

  sb << "}\n";

  std::cout << sb.str();
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
  sb += "|" + escapeDottyString(std::to_string(*II->getType())) + "|";

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

  glow_unreachable();
}

/// Dump a dotty graph that depicts the module.
void Module::dumpDAG() {
  std::string filename = "dotty_ir_dump_" + std::to_string(this) + ".dot";
  std::cout << "Writing dotty graph to: " << filename << '\n';

  std::string sb;
  sb += "digraph finite_state_machine {\n\trankdir=LR;\n";

  sb += "subgraph cluster_1 {";
  sb += "  style=invis;\n";

  for (auto &I : instrs_) {
    std::string desc = getDottyDesc(I);

    sb += quote(std::to_string(I)) + "[\n";
    std::string repr = quote(desc);
    sb += "\tlabel = " + repr + "\n";
    sb += "\tshape = \"record\"\n";
    sb += "];\n\n";
  }
  sb += "}";

  sb += "subgraph cluster_0 {";
  sb += "  style=invis;\n";

  for (auto &v : weights_) {
    sb += quote(std::to_string(v)) + "[\n";
    std::string desc = getDottyDesc(v);
    sb += "\tlabel = " + quote(desc) + "\n";
    sb += "\tshape = \"record\"\n";
    sb += "\tfillcolor=pink,style=filled\n";
    sb += "];\n\n";
  }
  sb += "}";

  sb += "subgraph cluster_1 {";
  sb += "  style=invis;\n";

  // Dump the use-def edges.
  for (auto &I : instrs_) {
    for (int i = 0, e = I->getNumOperands(); i < e; i++) {
      auto op = I->getOperand(i);
      std::string from = quote(std::to_string(I)) + ":f" + std::to_string(i);
      std::string to = quote(std::to_string(op.first));

      sb += from + "->" + to + "[dir=" + getDottyArrowForCC(op.second) + "];\n";
    }
  }

  // Dump the order edges.
  Instruction *prev = nullptr;
  for (auto &I : instrs_) {
    if (prev) {
      std::string from = quote(std::to_string(prev));
      std::string to = quote(std::to_string(I));
      sb += from + "->" + to + "[color=\"blue\"];\n";
    }
    prev = I;
  }
  sb += "}";
  sb += "}";

  std::ofstream myfile;
  myfile.open(filename);
  myfile << sb;
  myfile.close();
}
