#include "NodeBuilder.h"

void NodeBuilder::emitEnumModePrinters(std::ostream &os) const {
  os << "const char *" << name_ << "Node::getModeStr(" << name_
     << "Node::Mode m) {\n";
  os << "\tstatic const char *names[] = {";
  for (const auto &e : enum_) {
    os << "\"" << e << "\", ";
  }
  os << "nullptr};\n";
  os << "\treturn names[static_cast<int>(m)];\n";
  os << "}\n";
}

void NodeBuilder::emitCtor(std::ostream &os) const {
  os << "\t" << name_ << "Node(llvm::StringRef name";

  // Constructor non-standard parameter list:
  for (const auto &op : extraParams_) {
    os << ", " << op.first << " " << op.second << " ";
  }

  // The enum 'Mode' parameter:
  if (!enum_.empty()) {
    os << ", Mode mode";
  }

  // The operands of the graph node:
  for (const auto &op : operands_) {
    os << ", Node *" << op;
  }

  // Extra class members:
  for (const auto &op : members_) {
    os << ", " << op.first << " " << op.second;
  }

  // Initialize the base clases:
  os << "):\n\t Node(Kinded::Kind::" << name_ << "NodeKind, " << ty_
     << ", name)";

  // Print the initialization list:
  if (!enum_.empty()) {
    os << ", mode_(mode)";
  }

  // Initialize the operands:
  for (const auto &op : operands_) {
    os << ", " << op << "_(" << op << ")";
  }

  // Initialize the members:
  for (const auto &op : members_) {
    os << ", " << op.second << "_(" << op.second << ") ";
  }

  // Empty constructor body.
  os << " {}\n\n";
}

void NodeBuilder::emitClassMembers(std::ostream &os) const {
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
  for (const auto &op : operands_) {
    os << "\tNodeOperand " << op << "_;\n";
  }
  for (const auto &op : members_) {
    os << "\t" << op.first << " " << op.second << "_;\n";
  }
  os << "\n";
}

void NodeBuilder::emitSettersGetters(std::ostream &os) const {
  // Print the getters/setters.
  for (const auto &op : operands_) {
    // Synthesize a user-defined operand getter.
    auto it = overrideGetter_.find(op);
    if (it != overrideGetter_.end()) {
      os << "\t" << it->second << "\n";
      continue;
    }

    // Synthesize the general getter.
    os << "\tNode *get" << op << "() const { return " << op << "_; }\n";
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
  os << "\nstatic bool classof(const Kinded *k) { return k->getKind() == "
        "Kinded::Kind::"
     << name_ << "NodeKind; }\n";

  if (!enum_.empty()) {
    os << "\tMode getMode() const { return mode_; }\n";
  }
}

void NodeBuilder::emitPrettyPrinter(std::ostream &os) const {
  os << "std::string " << name_
     << "Node::getDebugDesc() const {\n\t\tDescriptionBuilder "
        "db(getKindName());\n\t\tdb.addParam(\"name\", getName())\n";

  if (!enum_.empty()) {
    os << "\t\t.addParam(\"Mode\", getModeStr())\n";
  }

  for (const auto &op : operands_) {
    os << "\t\t.addParam(\"" << op << "\", *get" << op << "()->getType())\n";
  }

  for (const auto &mem : members_) {
    os << "\t\t.addParam(\"" << mem.second << "\", get" << mem.second
       << "())\n";
  }
  os << "\t\t.addParam(\"users\", getNumUsers());\n\t\treturn db;\n}\n";
}

void NodeBuilder::emitEquator(std::ostream &os) const {
  os << "bool " << name_ << "Node::isEqual(const " << name_
     << "Node &other) {\n\treturn true";

  if (!enum_.empty()) {
    os << " &&\n\t getMode() == other.getMode()";
  }

  for (const auto &op : operands_) {
    os << " &&\n\t " << op << "_ == other." << op << "_";
  }

  for (const auto &mem : members_) {
    os << " &&\n\t " << mem.second << "_ == other." << mem.second << "_";
  }

  os << ";\n }\n";
}

void NodeBuilder::emitVisitor(std::ostream &os) const {
  os << "void " << name_
     << "Node::visit(Node *parent, NodeVisitor *visitor) {\n\tif "
        "(!visitor->shouldVisit(parent, this)) { return; }\n";

  os << "\tvisitor->pre(parent, this);\n";
  for (const auto &op : operands_) {
    os << "\tget" << op << "()->visit(this, visitor);\n";
  }
  os << "\tvisitor->post(parent, this);\n";
  os << "}\n";
}

void NodeBuilder::emitNodeClass(std::ostream &os) const {
  os << "namespace glow {\nclass " << name_ << "Node final : public Node {\n";

  emitClassMembers(os);

  os << "\tpublic:\n";

  emitCtor(os);

  emitSettersGetters(os);

  os << "\tstd::string getDebugDesc() const override;\n";
  os << "\tbool isEqual(const " << name_ << "Node &other);\n";
  os << "\tvoid visit(Node *parent, NodeVisitor *visitor) override;\n";
  if (!enum_.empty()) {
    os << "\tconst char *getModeStr() const { return getModeStr(mode_); "
          "}\n\tstatic const char *getModeStr(Mode m);\n";
  }

  os << "};\n\n} // namespace glow\n";
}

void NodeBuilder::emitCppMethods(std::ostream &os) const {
  emitPrettyPrinter(os);
  emitVisitor(os);
  emitEquator(os);
  if (!enum_.empty()) {
    emitEnumModePrinters(os);
  }
}

NodeBuilder::~NodeBuilder() {
  emitNodeClass(hStream);
  emitCppMethods(cStream);
}
