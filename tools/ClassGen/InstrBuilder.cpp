// Copyright 2017 Facebook Inc.  All Rights Reserved.
#include "InstrBuilder.h"
#include "glow/Support/Compiler.h"

unsigned InstrBuilder::getOperandIndexByName(llvm::StringRef name) const {
  for (unsigned i = 0; i < operands_.size(); i++) {
    if (name == operands_[i].first) {
      return i;
    }
  }

  assert(false && "Can't find an operand with this name");
  glow_unreachable();
}

void InstrBuilder::emitCtor(std::ostream &os) const {
  os << "\t" << name_ << "Inst(llvm::StringRef name";

  // Constructor non-standard parameter list:
  for (const auto &op : extraParams_) {
    os << ", " << op.first << " " << op.second << " ";
  }

  // The operands of the instruction class:
  for (const auto &op : operands_) {
    os << ", Value *" << op.first;
  }

  // Extra class members:
  for (const auto &op : members_) {
    os << ", " << op.first << " " << op.second;
  }

  // Initialize the base clases:
  os << "):\n\t Instruction(name, Kinded::Kind::" << name_ << "InstKind, "
     << ty_ << ",{\n";

  // The operands of the instruction class:
  for (const auto &op : operands_) {
    os << "\t\t{" << op.first
       << ", OperandKind::" << getOperandKindStr(op.second) << "},\n";
  }

  os << "\t})";

  // Initialize the members:
  for (const auto &op : members_) {
    os << ", " << op.second << "_(" << op.second << ") ";
  }

  // Empty constructor body.
  os << " {}\n\n";
}

void InstrBuilder::emitInplaceMethod(std::ostream &os) const {
  os << "\tbool isInplaceOp(unsigned dstIdx, unsigned srcIdx) const {\n";
  if (!inplaceOperands_.empty()) {
    assert(inplaceOperands_.size() > 1 &&
           "We don't have a pair of inplace args");
    for (int i = 1, e = inplaceOperands_.size(); i < e; i++) {
      auto F0 = getOperandIndexByName(inplaceOperands_[0]);
      auto F1 = getOperandIndexByName(inplaceOperands_[i]);
      os << "\tif (" << F0 << " == dstIdx && " << F1
         << " == srcIdx) {return true;}\n";
    }
  }
  os << "\t\treturn false;\n";
  os << "}\n";
}

void InstrBuilder::emitClassMembers(std::ostream &os) const {
  // Emit class members:
  for (const auto &op : members_) {
    os << "\t" << op.first << " " << op.second << "_;\n";
  }
  os << "\n";
}

void InstrBuilder::emitSettersGetters(std::ostream &os) const {
  // Print the getters/setters.
  for (int i = 0, e = operands_.size(); i < e; i++) {
    auto &op = operands_[i];

    // Synthesize a user-defined operand getter.
    auto it = overrideGetter_.find(op.first);
    if (it != overrideGetter_.end()) {
      os << "\t" << it->second << "\n";
      continue;
    }

    // Synthesize the general getter.
    os << "\tValue *get" << op.first << "() const { return getOperand(" << i
       << ").first; }\n";
  }

  for (const auto &op : members_) {
    // Synthesize a user-defined member getter.
    auto it = overrideGetter_.find(op.second);
    if (it != overrideGetter_.end()) {
      os << "\t" << it->second << "\n";
      continue;
    }

    // Synthesize the general getter.
    os << "\t" << op.first + " get" << op.second << "() const { return "
       << op.second << "_; }\n";
  }
  // Synthesize the 'classof' method that enables the non-rtti polymorphism.
  os << "\n\tstatic bool classof(const Kinded *k) { return k->getKind() == "
        "Kinded::Kind::"
     << name_ << "InstKind; }\n";
}

void InstrBuilder::emitPrettyPrinter(std::ostream &os) const {
  os << "void " << name_ << "Inst::dump(std::ostream &os) const {\n";
  os << "\tos << '%' << (std::string) getName() << \" = \" << getKindName() << "
        "\" \";\n";
  os << "\tdumpOperands(os);\n";

  if (!members_.empty()) {
    os << "\tos << \" {\";\n";
    bool first = true;
    for (const auto &mem : members_) {
      os << "\tos << \"" << (first ? " " : ", ") << mem.second
         << ": \" <<  std::to_string("
         << "get" << mem.second << "());\n";
      first = false;
    }
    os << "\tos << '}';\n";
  }
  os << "\n}\n";
}

void InstrBuilder::emitClass(std::ostream &os) const {
  os << "namespace glow {\nclass " << name_
     << "Inst final : public Instruction {\n";

  emitClassMembers(os);

  os << "\tpublic:\n";

  emitCtor(os);

  emitSettersGetters(os);

  emitInplaceMethod(os);

  for (const auto &m : extraMethods_) {
    os << "\t" << m << "\n";
  }

  os << "\t void dump(std::ostream &os) const;\n";
  os << "\t void verify() const;\n";
  os << "};\n\n} // namespace glow\n";
}

void InstrBuilder::emitCppMethods(std::ostream &os) const {
  emitPrettyPrinter(os);
}

InstrBuilder::~InstrBuilder() {
  emitClass(hStream);
  emitCppMethods(cStream);
}

void InstrBuilder::addGradientInstr(
    llvm::ArrayRef<llvm::StringRef> originalFields,
    llvm::ArrayRef<llvm::StringRef> gradFields) {
  InstrBuilder GI(hStream, cStream, dStream, name_ + "Grad");

  // The new 'Grad' class will have all of the fields of the current class.
  GI.ty_ = ty_;
  GI.members_ = members_;
  GI.extraParams_ = extraParams_;
  GI.overrideGetter_ = overrideGetter_;
  GI.extraMethods_ = extraMethods_;

  // Add the operands that we'll use in the grad instruction.
  for (const auto &op : operands_) {
    for (const auto &field : originalFields) {
      if (field == op.first) {
        // We may only read from the original weight operands.
        GI.addOperand(op.first, OperandKind::In);
      }
    }
  }

  // Add the new 'grad' operands for the gradients.
  for (const auto &op : operands_) {
    for (const auto &field : gradFields) {
      if (field == op.first) {
        GI.addOperand(op.first + "Grad", negateOperandKind(op.second));
      }
    }
  }

  // Construct a factory method that builds the new grad instruction and add
  // it to the current non-grad instruction.
  std::stringstream ss;
  ss << name_ + "GradInst* getGrad(Module::GradientMap &map) const {\n";
  ss << "\t return new " + name_ + "GradInst(getName()";

  // Non-standard parameter list:
  for (const auto &op : extraParams_) {
    ss << ", " << op.second;
  }

  // The operands of the input class:
  for (const auto &op : operands_) {
    for (const auto &field : originalFields) {
      if (field == op.first) {
        ss << ", get" << op.first << "()";
      }
    }
  }

  // Add new operands for the gradients.
  for (const auto &op : operands_) {
    for (const auto &field : gradFields) {
      if (field == op.first) {
        ss << ", map[get" << op.first << "()]";
      }
    }
  }

  // Extra class members:
  for (const auto &op : members_) {
    ss << ", get" << op.second << "()";
  }

  ss << ");\n }";
  addExtraMethod(ss.str());
}
