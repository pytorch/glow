#include "InstrBuilder.h"

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
