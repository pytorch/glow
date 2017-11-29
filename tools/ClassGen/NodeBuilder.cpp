// Copyright 2017 Facebook Inc.  All Rights Reserved.
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
  for (const auto &op : nodeInputs_) {
    os << ", NodeValue " << op;
  }

  // Extra class members:
  for (const auto &op : members_) {
    os << ", " << getStorageTypename(op.first) << " " << op.second;
  }

  // Initialize the base clases:
  os << "):\n\t Node(Kinded::Kind::" << name_ << "NodeKind, name)";

  // Print the initialization list:
  if (!enum_.empty()) {
    os << ", mode_(mode)";
  }

  // Initialize the operands:
  for (const auto &op : nodeInputs_) {
    os << ", " << op << "_(" << op << ")";
  }

  // Initialize the members:
  for (const auto &op : members_) {
    os << ", " << op.second << "_(" << op.second << ") ";
  }

  // The constructor body:
  os << " {";
  for (auto &RT : nodeOutputs_) {
    os << "\taddResult(" << RT.first << ");\n";
  }
  os << "}\n\n";
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
  for (const auto &op : nodeInputs_) {
    os << "\tNodeValue " << op << "_;\n";
  }
  for (const auto &op : members_) {
    os << "\t" << getStorageTypename(op.first) << " " << op.second << "_;\n";
  }
  os << "\n";
}

void NodeBuilder::emitMemberGetter(std::ostream &os, MemberType type,
                                   const std::string &name) const {
  // Synthesize the general getter.
  auto returnTypeStr = getReturnTypename(type);
  os << "\t" << returnTypeStr << " get" << name << "() const { return " << name
     << "_; }\n";
}

void NodeBuilder::emitSettersGetters(std::ostream &os) const {
  // Print the getters/setters.
  for (const auto &inName : nodeInputs_) {
    os << "\tNodeValue get" << inName << "() const { return " << inName
       << "_; }\n";
  }

  unsigned idx = 0;
  for (const auto &op : nodeOutputs_) {
    os << "\tNodeValue get" << op.second << "() { return NodeValue(this,"
       << idx++ << "); }\n";
  }

  for (const auto &op : members_) {
    emitMemberGetter(os, op.first, op.second);
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

  for (const auto &op : nodeInputs_) {
    os << "\t\t.addParam(\"" << op << "\", *get" << op << "().getType())\n";
  }

  for (const auto &mem : members_) {
    // Don't try to print the node operands directly.
    if (mem.first == MemberType::VectorNodeValue)
      continue;

    os << "\t\t.addParam(\"" << mem.second << "\", get" << mem.second
       << "())\n";
  }
  os << "\t\t.addParam(\"users\", getNumUsers());";

  for (const auto &mem : members_) {
    if (mem.first != MemberType::VectorNodeValue)
      continue;

    os << " for (auto II : get" << mem.second << "()) { db.addParam(\""
       << mem.second << "\", *II->getType()); }";
  }

  os << "\n\t\treturn db;\n}\n";
}

void NodeBuilder::emitEquator(std::ostream &os) const {
  os << "bool " << name_ << "Node::isEqual(const " << name_
     << "Node &other) const {\n\treturn true";

  if (!enum_.empty()) {
    os << " &&\n\t getMode() == other.getMode()";
  }

  for (const auto &op : nodeInputs_) {
    os << " &&\n\t " << op << "_ == other." << op << "_";
  }

  for (const auto &mem : members_) {
    os << " &&\n\t " << mem.second << "_ == other." << mem.second << "_";
  }

  os << ";\n }\n";
}

static bool isVectorType(MemberType ty) {
  return ty == MemberType::VectorFloat || ty == MemberType::VectorNodeValue ||
         ty == MemberType::VectorSizeT || ty == MemberType::VectorUnsigned;
}

void NodeBuilder::emitHasher(std::ostream &os) const {
  os << "llvm::hash_code " << name_ << "Node::getHash()"
     << " const {\n\treturn llvm::hash_combine(";

  if (enum_.empty() && nodeInputs_.empty() && members_.empty()) {
    os << "0);\n }\n";
    return;
  }

  auto delim = "";
  if (!enum_.empty()) {
    os << delim << "\n\t getMode()";
    delim = ",";
  }

  for (const auto &mem : members_) {
    auto ty = mem.first;
    if (ty == MemberType::Float) {
      os << delim << "\n\t "
         << "toBinary(" << mem.second << "_"
         << ")";
    } else if (isVectorType(ty)) {
      os << delim << "\n\t "
         << "llvm::hash_combine_range(" << mem.second << "_"
         << ".begin(), " << mem.second << "_"
         << ".end()"
         << ")";
    } else {
      os << delim << "\n\t " << mem.second << "_";
    }
    delim = ",";
  }

  for (const auto &op : nodeInputs_) {
    os << delim << "\n\t " << op << "_";
    delim = ",";
  }

  os << ");\n }\n";
}
void NodeBuilder::emitVisitor(std::ostream &os) const {
  os << "void " << name_
     << "Node::visit(Node *parent, NodeWalker *visitor) {\n\tif "
        "(!visitor->shouldVisit(parent, this)) { return; }\n";

  os << "\tvisitor->pre(parent, this);\n";
  for (const auto &op : nodeInputs_) {
    os << "\tget" << op << "()->visit(this, visitor);\n";
  }

  for (const auto &op : members_) {
    if (op.first == MemberType::VectorNodeValue) {
      os << " for (auto &I : " << op.second
         << "_) { I->visit(this, visitor);}\n";
    }
  }

  os << "\tvisitor->post(parent, this);\n";
  os << "}\n";
}

void NodeBuilder::emitDocstring(std::ostream &os) const {
  std::istringstream stream(docstring_);
  std::string line;
  while (std::getline(stream, line)) {
    os << "/// " << line << "\n";
  }
}

void NodeBuilder::emitNodeClass(std::ostream &os) const {
  os << "namespace glow {\n";

  emitDocstring(os);

  os << "class " << name_ << "Node final : public Node {\n";

  emitClassMembers(os);

  os << "\tpublic:\n";

  emitCtor(os);

  emitSettersGetters(os);

  os << "\tstd::string getDebugDesc() const override;\n";
  os << "\tbool isEqual(const " << name_ << "Node &other) const;\n";
  os << "\tllvm::hash_code getHash() const;\n";
  os << "\tvoid visit(Node *parent, NodeWalker *visitor) override;\n";
  if (!enum_.empty()) {
    os << "\tconst char *getModeStr() const { return getModeStr(mode_); "
          "}\n\tstatic const char *getModeStr(Mode m);\n";
  }

  for (const auto &m : extraMethods_) {
    os << "\t" << m << "\n";
  }

  os << "};\n\n} // namespace glow\n";
}

void NodeBuilder::emitCppMethods(std::ostream &os) const {
  emitPrettyPrinter(os);
  emitVisitor(os);
  emitEquator(os);
  emitHasher(os);
  if (!enum_.empty()) {
    emitEnumModePrinters(os);
  }
}

void NodeBuilder::addGradient() {
  NodeBuilder GN(hStream, cStream, dStream, name_ + "Grad");

  // The new 'Grad' class will have all of the fields of the current class.
  GN.members_ = members_;
  GN.enum_ = enum_;

  // Add the inputs that we'll use in the grad instruction.
  for (const std::string &in : nodeInputs_) {
    GN.addInput(in);
  }

  for (const std::string &in : nodeInputs_) {
    GN.addResult(in + ".getType()", "GradOfInputNamed" + in);
  }

  for (const auto &out : nodeOutputs_) {
    GN.addInput("OriginalOutputFor" + out.second);
    GN.addInput("GradOfOriginalOutputNamed" + out.second);
  }

  // Construct a factory method that builds the new grad node and add
  // it to the current non-grad instruction.
  std::stringstream ss;
  ss << name_ + "GradNode* getGrad(UnownedNodeValueMap &map) {\n";
  ss << "\tauto * x = new " + name_ + "GradNode(getName()";

  if (enum_.size()) {
    ss << ", (" << name_ + "GradNode::Mode"
       << ")getMode()";
  }

  // Add the inputs that we'll use in the grad instruction.
  for (const std::string &in : nodeInputs_) {
    ss << ", get" << in << "()";
  }

  for (const auto &out : nodeOutputs_) {
    ss << ", get" << out.second << "()";
    ss << ", map.get(get" << out.second << "())";
  }

  // Extra class members:
  for (const auto &op : members_) {
    ss << ", get" << op.second << "()";
  }

  ss << ");\n";

  // Register the result of the new node as the gradients of the original node
  // inputs.
  for (const std::string &in : nodeInputs_) {
    ss << "\tmap.insert(get" << in << "(), x->getGradOfInputNamed" << in
       << "());\n";
  }
  ss << "\treturn x;\n";
  ss << " }";
  addExtraMethod(ss.str());
}

NodeBuilder::~NodeBuilder() {
  emitNodeClass(hStream);
  emitCppMethods(cStream);
}
