#include "InstrBuilder.h"

void InstrBuilder::emitEnumModePrinters(std::ostream &os) const {
  os << "const char *" << name_ << "Instr::getModeStr(" << name_
     << "Instr::Mode m) {\n";
  os << "\tstatic const char *names[] = {";
  for (const auto &e : enum_) {
    os << "\"" << e << "\", ";
  }
  os << "nullptr};\n";
  os << "\treturn names[static_cast<int>(m)];\n";
  os << "}\n";
}

void InstrBuilder::emitCtor(std::ostream &os) const {
  os << "\t" << name_ << "Inst(llvm::StringRef name";

  // Constructor non-standard parameter list:
  for (const auto &op : extraParams_) {
    os << ", " << op.first << " " << op.second << " ";
  }

  // The enum 'Mode' parameter:
  if (!enum_.empty()) {
    os << ", Mode mode";
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
  os << "):\n\t Instruction(Kinded::Kind::" << name_ << "InstKind, " << ty_
     << ",{\n";

  // The operands of the instruction class:
  for (const auto &op : operands_) {
    os << "\t\t{" << op.first
       << ", OperandKind::" << getOperandKindStr(op.second) << "},\n";
  }

  os << "\t})";

  // Print the initialization list:
  if (!enum_.empty()) {
    os << ", mode_(mode)";
  }

  // Initialize the members:
  for (const auto &op : members_) {
    os << ", " << op.second << "_(" << op.second << ") ";
  }

  // Empty constructor body.
  os << " {}\n\n";
}

void InstrBuilder::emitClassMembers(std::ostream &os) const {
  // Emit the type of the enum (which is public).
  if (!enum_.empty()) {
    os << "\tpublic:\n\tenum class Mode {\n";
    for (const auto &E : enum_) {
      os << "\t  " << E << ",\n";
    }
    os << "\t};\n";

    os << "\tprivate:\n";
  }

  // Emit class members:
  if (!enum_.empty()) {
    os << "\tMode mode_;\n";
  }
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

  if (!enum_.empty()) {
    os << "\tMode getMode() const { return mode_; }\n";
  }
}

void InstrBuilder::emitPrettyPrinter(std::ostream &os) const {
  os << "void dump(std::ostream &os) const {\n";
  os << "\tos << '%' << getName() << \" = " << name_ << " \";\n";

  bool first = true;
  for (const auto &op : operands_) {
    if (!first) {
      os << "\tos << \", \";";
    }
    os << "\tos << \"@" << getOperandKindStr(op.second) << " %" << op.first
       << "\";\n";
    first = false;
  }

  for (const auto &mem : members_) {
    os << "\tos << \", " << mem.second << " \" <<  std::to_string("
       << mem.second << ")\";\n";
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

  if (!enum_.empty()) {
    os << "\tconst char *getModeStr() const { return getModeStr(mode_); "
          "}\n\tstatic const char *getModeStr(Mode m);\n";
  }
  os << "\t void dump(std::ostream &os) const;\n";
  os << "\t void verify() const;\n";
  os << "};\n\n} // namespace glow\n";
}

void InstrBuilder::emitCppMethods(std::ostream &os) const {
  emitPrettyPrinter(os);
  if (!enum_.empty()) {
    emitEnumModePrinters(os);
  }
}

InstrBuilder::~InstrBuilder() {
  emitClass(hStream);
  emitCppMethods(cStream);
}
