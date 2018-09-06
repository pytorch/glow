/**
 * Copyright (c) 2017-present, Facebook, Inc.
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

#include "NodeBuilder.h"

void NodeBuilder::emitEnumModePrinters(std::ostream &os) const {
  os << "\nconst char *" << name_ << "Node::getModeStr(" << name_
     << "Node::Mode m) {\n";
  os << "  static const char *names[] = {";
  for (const auto &e : enum_) {
    os << "\"" << e << "\", ";
  }
  os << "nullptr};\n";
  os << "  return names[static_cast<int>(m)];\n";
  os << "}\n";
}

void NodeBuilder::emitCtor(std::ostream &os) const {
  os << "  " << name_ << "Node(llvm::StringRef name";

  // Generate the external type parameters:
  for (const auto &paramName : ctorTypeParams_) {
    os << ", TypeRef " << paramName << " ";
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
    os << ", " << getCtorArgTypename(op.first) << " " << op.second;
  }

  // Initialize the base clases:
  os << ")\n      : Node(Kinded::Kind::" << name_ << "NodeKind, name)";

  // Print the initialization list:
  if (!enum_.empty()) {
    os << ", mode_(mode)";
  }

  // Initialize the operands:
  for (const auto &op : nodeInputs_) {
    os << ", " << op << "_("
       << "this, " << op << ")";
  }

  // Initialize the members:
  for (const auto &op : members_) {
    if (op.first != MemberType::VectorNodeValue) {
      os << ", " << op.second << "_(" << op.second << ")";
      continue;
    }
    continue;
    os << ", " << op.second << "_(" << op.second << ".begin(), " << op.second
       << ".end()"
       << ")";
  }

  // The constructor body:
  os << " {\n";
  for (auto &RT : nodeOutputs_) {
    os << "    addResult(" << RT.first << ");\n";
  }

  for (const auto &op : members_) {
    if (op.first != MemberType::VectorNodeValue) {
      continue;
    }
    os << "    " << op.second << "_.resize(" << op.second << ".size());\n";
    os << "    for (size_t idx = 0, e = " << op.second
       << ".size(); idx < e; ++idx) {\n"
       << "        " << op.second << "_[idx] = " << op.second << "[idx];\n"
       << "        " << op.second << "_[idx].setParent(this);\n"
       << "    }\n";
  }

  os << "  }\n";
}

void NodeBuilder::emitClassMembers(std::ostream &os) const {
  // Emit the type of the enum (which is public).
  if (!enum_.empty()) {
    os << " public:\n  enum class Mode {\n";
    for (const auto &E : enum_) {
      os << "    " << E << ",\n";
    }
    os << "  };\n\n private:\n";
  }

  // Emit class members:
  if (!enum_.empty()) {
    os << "  Mode mode_;\n";
  }
  for (const auto &op : nodeInputs_) {
    os << "  NodeHandle " << op << "_;\n";
  }
  for (const auto &op : members_) {
    os << "  " << getStorageTypename(op.first) << " " << op.second << "_;\n";
  }
}

void NodeBuilder::emitMemberGetter(std::ostream &os, MemberType type,
                                   const std::string &name) const {
  // Synthesize the general getter.
  auto returnTypeStr = getReturnTypename(type);
  os << "  " << returnTypeStr << " get" << name << "() const { return " << name
     << "_; }\n";
}

void NodeBuilder::emitSettersGetters(std::ostream &os) const {
  // Print the getters/setters.
  for (const auto &inName : nodeInputs_) {
    os << "  const NodeValue get" << inName << "() const { return " << inName
       << "_; }\n";
  }

  unsigned idx = 0;
  for (const auto &op : nodeOutputs_) {
    os << "  NodeValue get" << op.second << "() { return getNthResult(" << idx
       << "); }\n";
    os << "  const NodeValue get" << op.second
       << "() const { return getNthResult(" << idx << "); }\n";
    idx++;
  }

  for (const auto &op : members_) {
    emitMemberGetter(os, op.first, op.second);
  }

  // Synthesize the 'classof' method that enables the non-rtti polymorphism.
  os << "\n  static bool classof(const Kinded *k) {\n"
     << "    return k->getKind() == Kinded::Kind::" << name_ << "NodeKind;\n"
     << "  }\n\n";

  os << "\n  bool isOverwrittenNthInput(unsigned idx) const {\n";
  for (const auto &overwrittenInput : nodeOverwrittenInputs_) {
    os << "    if (idx == " << overwrittenInput << ") return true;\n";
  }
  os << "    return false;\n";
  os << "  }\n\n";

  if (!enum_.empty()) {
    os << "  Mode getMode() const { return mode_; }\n";
  }
}

void NodeBuilder::emitEdges(std::ostream &os) const {
  os << "\nunsigned " << name_ << "Node::getNumInputs() const {\n"
     << "  return " << nodeInputs_.size();
  for (const auto &op : members_) {
    if (op.first != MemberType::VectorNodeValue) {
      continue;
    }
    os << " + " << op.second << "_.size()";
  }
  os << ";\n}\n";

  os << "\nstd::string " << name_
     << "Node::getInputName(unsigned idx) const {\n";
  for (size_t i = 0; i < nodeInputs_.size(); i++) {
    os << "  if (idx == " << i << ") { return \"" << nodeInputs_[i]
       << "\"; }\n";
  }
  os << "  idx -= " << nodeInputs_.size() << ";\n";
  for (const auto &op : members_) {
    if (op.first != MemberType::VectorNodeValue) {
      continue;
    }
    os << "  if (idx < " << op.second << "_.size()) { return \"" << op.second
       << "\" + std::to_string(idx); }\n"
       << "  idx -= " << op.second << "_.size();\n";
  }
  os << "  llvm_unreachable(\"Invalid index\");\n}\n";

  os << "\nNodeValue " << name_ << "Node::getNthInput(unsigned idx) {\n";
  for (size_t i = 0; i < nodeInputs_.size(); i++) {
    os << "  if (idx == " << i << ") { return " << nodeInputs_[i] << "_; }\n";
  }
  os << "  idx -= " << nodeInputs_.size() << ";\n";
  for (const auto &op : members_) {
    if (op.first != MemberType::VectorNodeValue) {
      continue;
    }
    os << "  if (idx < " << op.second << "_.size()) { return " << op.second
       << "_[idx]; }\n  idx -= " << op.second << "_.size();\n";
  }
  os << "  llvm_unreachable(\"Invalid index\");\n}\n";

  os << "\nvoid " << name_
     << "Node::setNthInput(unsigned idx, NodeValue val) {\n";
  for (size_t i = 0; i < nodeInputs_.size(); i++) {
    os << "  if (idx == " << i << ") { " << nodeInputs_[i]
       << "_ = val; return; }\n";
  }
  os << "  idx -= " << nodeInputs_.size() << ";\n";
  for (const auto &op : members_) {
    if (op.first != MemberType::VectorNodeValue) {
      continue;
    }
    os << "  if (idx < " << op.second << "_.size()) { " << op.second
       << "_[idx] = val; return; }\n  idx -= " << op.second << "_.size();\n";
  }
  os << "  llvm_unreachable(\"Invalid index\");\n}\n";

  os << "\nllvm::StringRef " << name_
     << "Node::getOutputName(unsigned idx) const {\n";
  for (size_t i = 0; i < nodeOutputs_.size(); i++) {
    os << "  if (idx == " << i << ") { return \"" << nodeOutputs_[i].second
       << "\"; }\n";
  }
  os << "  llvm_unreachable(\"Invalid index\");\n}\n";
}

void NodeBuilder::emitPrettyPrinter(std::ostream &os) const {
  os << "\nstd::string " << name_ << "Node::getDebugDesc() const {\n"
     << "  DescriptionBuilder db(getKindName());\n"
     << "  db.addParam(\"name\", getName());\n";

  os << "  if (hasPredicate()) db.addParam(\"Predicate\", \"Yes\");\n";

  os << "  db\n";
  if (!enum_.empty()) {
    os << "    .addParam(\"Mode\", getModeStr())\n";
  }

  // Generate description for inputs.
  for (const auto &op : nodeInputs_) {
    os << "    .addParam(\"" << op << "\", *(get" << op << "().getType()))\n";
  }

  for (const auto &mem : members_) {
    // Don't try to print the node operands directly.
    if (mem.first == MemberType::VectorNodeValue) {
      continue;
    }

    os << "    .addParam(\"" << mem.second << "\", get" << mem.second
       << "())\n";
  }
  os << "    .addParam(\"users\", getNumUsers());\n";

  for (const auto &mem : members_) {
    if (mem.first != MemberType::VectorNodeValue) {
      continue;
    }

    // Make sure that inputs are properly indexed.
    os << "  unsigned mIndex = 0;\n";
    os << "  for (const auto &II : get" << mem.second << "()) {\n"
       << "    db.addParam(\"" << mem.second
       << "\"+std::to_string(mIndex++), *II.getType());\n"
       << "  }\n";
  }

  // Generate description for outputs.
  for (const auto &op : nodeOutputs_) {
    os << "db.addParam(\"" << op.second << "\", *(get" << op.second
       << "().getType()));\n";
  }

  os << "  return db;\n}\n";
}

void NodeBuilder::emitCloner(std::ostream &os) const {
  os << "\nNode* " << name_ << "Node::clone() const {\n";

  os << "  return new " << name_ << "Node(getName()";

  // Pass the external type arguments:
  for (const auto &paramName : ctorTypeParams_) {
    os << ", get" << paramName << "().getType()";
  }

  // The enum 'Mode' parameter:
  if (!enum_.empty()) {
    os << ", getMode()";
  }

  // The operands of the graph node:
  for (const auto &op : nodeInputs_) {
    os << ", get" << op << "()";
  }

  // Extra class members:
  for (const auto &op : members_) {
    os << ", get" << op.second << "()";
  }

  os << ");\n}\n";
}

void NodeBuilder::emitEquator(std::ostream &os) const {
  os << "\nbool " << name_ << "Node::isEqual(const " << name_
     << "Node &other) const {\n  return true";

  if (!enum_.empty()) {
    os << " &&\n      getMode() == other.getMode()";
  }

  for (const auto &op : nodeInputs_) {
    os << " &&\n      " << op << "_ == other." << op << "_";
  }

  for (const auto &mem : members_) {
    os << " &&\n      " << mem.second << "_ == other." << mem.second << "_";
  }

  for (int i = 0, e = nodeOutputs_.size(); i < e; i++) {
    os << " &&\n      getType(" << i << ") == other.getType(" << i << ")";
  }
  os << ";\n}\n";
}

static bool isVectorType(MemberType ty) {
  return ty == MemberType::VectorFloat || ty == MemberType::VectorNodeValue ||
         ty == MemberType::VectorSizeT || ty == MemberType::VectorUnsigned;
}

void NodeBuilder::emitHasher(std::ostream &os) const {
  os << "\nllvm::hash_code " << name_ << "Node::getHash() const {\n"
     << "  return llvm::hash_combine(";

  if (enum_.empty() && nodeInputs_.empty() && members_.empty()) {
    os << "0);\n }\n";
    return;
  }

  auto delim = "";
  if (!enum_.empty()) {
    os << delim << "\n      getMode()";
    delim = ",";
  }

  for (const auto &mem : members_) {
    auto ty = mem.first;
    if (ty == MemberType::Float) {
      os << delim << "\n      toBinary(" << mem.second << "_)";
    } else if (isVectorType(ty)) {
      os << delim << "\n      llvm::hash_combine_range(" << mem.second
         << "_.begin(), " << mem.second << "_.end())";
    } else {
      os << delim << "\n      " << mem.second << "_";
    }
    delim = ",";
  }

  for (const auto &op : nodeInputs_) {
    os << delim << "\n      " << op << "_";
    delim = ",";
  }

  os << ");\n}\n";
}
void NodeBuilder::emitVisitor(std::ostream &os) const {
  os << "\nvoid " << name_
     << "Node::visit(Node *parent, NodeWalker *visitor) {\n"
     << "  if (!visitor->shouldVisit(parent, this)) { return; }\n"
     << "  visitor->pre(parent, this);\n"
     << "if (hasPredicate())\n"
     << " getPredicate().getNode()->visit(this, visitor);\n";

  for (const auto &op : nodeInputs_) {
    os << "  get" << op << "().getNode()->visit(this, visitor);\n";
  }

  for (const auto &op : members_) {
    if (op.first == MemberType::VectorNodeValue) {
      os << "  for (auto &I : " << op.second
         << "_) { I.getNode()->visit(this, visitor); }\n";
    }
  }

  os << "  visitor->post(parent, this);\n}\n";
}

void NodeBuilder::emitDocstring(std::ostream &os) const {
  std::istringstream stream(docstring_);
  std::string line;
  while (std::getline(stream, line)) {
    os << "/// " << line << "\n";
  }
}

void NodeBuilder::emitNodeClass(std::ostream &os) const {
  os << "\nnamespace glow {\n";

  emitDocstring(os);

  os << "class " << name_ << "Node final : public Node {\n";

  emitClassMembers(os);

  os << "\n public:\n";

  emitCtor(os);
  emitSettersGetters(os);

  os << "  unsigned getNumInputs() const;\n"
     << "  std::string getInputName(unsigned idx) const;\n"
     << "  NodeValue getNthInput(unsigned idx);\n"
     << "  void setNthInput(unsigned idx, NodeValue val);\n"
     << "  llvm::StringRef getOutputName(unsigned idx) const;\n"
     << "  bool hasSideEffects() const { return " << hasSideEffects_ << "; }\n"
     << "  std::string getDebugDesc() const;\n"
     << "  bool isEqual(const " << name_ << "Node &other) const;\n"
     << "  llvm::hash_code getHash() const;\n"
     << "  void visit(Node *parent, NodeWalker *visitor);\n"
     << "  Node* clone() const;\n"
     << "  void verify() const;\n";

  if (!enum_.empty()) {
    os << "  const char *getModeStr() const { return getModeStr(mode_); }\n"
       << "  static const char *getModeStr(Mode m);\n";
  }

  for (const auto &m : extraMethods_) {
    os << "  " << m.first;
  }

  os << "};\n} // namespace glow\n";
}

void NodeBuilder::emitCppMethods(std::ostream &os) const {
  emitEdges(os);
  emitPrettyPrinter(os);
  emitVisitor(os);
  emitEquator(os);
  emitCloner(os);
  emitHasher(os);
  if (!enum_.empty()) {
    emitEnumModePrinters(os);
  }

  // Emit the "extra" method bodies.
  for (const auto &m : extraMethods_) {
    os << m.second;
  }
}

NodeBuilder &NodeBuilder::addGradient() {
  NodeBuilder GN(hStream, cStream, dStream, name_ + "Grad", isBackendSpecific_);

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

  std::string decl = name_ + "GradNode *getGrad(GraphGradMapper &builder);\n";
  std::stringstream ss;
  ss << "\n" + name_ + "GradNode *" + name_
     << "Node::getGrad(GraphGradMapper &builder) {\n"
     << "  auto *x = new " + name_ + "GradNode(getName()";

  if (enum_.size()) {
    ss << ", (" << name_ << "GradNode::Mode)getMode()";
  }

  // Add the inputs that we'll use in the grad instruction.
  for (const std::string &in : nodeInputs_) {
    ss << ", get" << in << "()";
  }

  for (const auto &out : nodeOutputs_) {
    ss << ", get" << out.second << "(), builder.getGradient(get" << out.second
       << "())";
  }

  // Extra class members:
  for (const auto &op : members_) {
    ss << ", get" << op.second << "()";
  }

  ss << ");\n";

  // Register the result of the new node as the gradients of the original node
  // inputs.
  for (const std::string &in : nodeInputs_) {
    ss << "  builder.addGradient(get" << in << "(), x->getGradOfInputNamed"
       << in << "());\n";
  }
  ss << "  return x;\n}\n";
  addExtraMethod(decl, ss.str());

  return *this;
}

NodeBuilder::~NodeBuilder() {
  emitNodeClass(hStream);
  emitCppMethods(cStream);
}
