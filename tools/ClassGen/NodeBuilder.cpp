/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

NodeBuilder &NodeBuilder::addMember(MemberType type, const std::string &name,
                                    bool addSetter) {
  MemberTypeInfo *typeInfo = nullptr;

  if (type == MemberType::TypeRef) {
    typeInfo = &kTypeRefTypeInfo;
  } else if (type == MemberType::Float) {
    typeInfo = &kFloatTypeInfo;
  } else if (type == MemberType::Unsigned) {
    typeInfo = &kUnsignedTypeInfo;
  } else if (type == MemberType::Boolean) {
    typeInfo = &kBooleanTypeInfo;
  } else if (type == MemberType::Int64) {
    typeInfo = &kInt64TypeInfo;
  } else if (type == MemberType::String) {
    typeInfo = &kStringTypeInfo;
  } else if (type == MemberType::VectorFloat) {
    typeInfo = &kVectorFloatTypeInfo;
  } else if (type == MemberType::VectorUnsigned) {
    typeInfo = &kVectorUnsignedTypeInfo;
  } else if (type == MemberType::VectorInt64) {
    typeInfo = &kVectorInt64TypeInfo;
  } else if (type == MemberType::VectorSigned) {
    typeInfo = &kVectorSignedTypeInfo;
  } else if (type == MemberType::VectorSizeT) {
    typeInfo = &kVectorSizeTTypeInfo;
  } else if (type == MemberType::VectorDimT) {
    typeInfo = &kVectorDimTTypeInfo;
  } else if (type == MemberType::VectorNodeValue) {
    typeInfo = &kVectorNodeValueTypeInfo;
  } else if (type == MemberType::Enum) {
    typeInfo = &kEnumTypeInfo;
  } else if (type == MemberType::UserDefinedType) {
    llvm_unreachable("addMember should be called with a MemberTypeInfo "
                     "parameter in this case");
  } else {
    llvm_unreachable("Type not recognized");
  }

  return addMember(*typeInfo, name, addSetter);
}

NodeBuilder &NodeBuilder::addFusedActivation() {
  return addMember(MEMBER_TYPE_INFO(glow::FusedActivation), "FusedActivation",
                   /* addSetter */ true)
      .addMember(MemberType::VectorFloat, "FusedActivationArgs",
                 /* addSetter */ true)
      .addExtraMethod("bool hasFusedActivation() const;",
                      "bool " + name_ +
                          "Node::hasFusedActivation() const { return "
                          "getFusedActivation() != FusedActivation::NONE; }");
}

void NodeBuilder::emitMemberForwardDecls(std::ostream &os) const {
  for (const auto &mem : members_) {
    const std::string &forwardDecl = (mem.first).forwardDecl;
    if (!forwardDecl.empty()) {
      os << forwardDecl << "\n";
    }
  }

  os << "\n";
}

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
  for (const auto &op : variableOutputMembers_) {
    // For outputs, the type is different llvm::ArrayRef<TypeRef> instead of
    // NodeValueArrayRef, so we hardcode it instead of calling
    // 'getCtorArgTypename'
    os << ", llvm::ArrayRef<TypeRef>" << " " << op.second;
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
    os << ", " << getCtorArgTypename(&op.first) << " " << op.second;
  }

  // Initialize the base clases:
  os << ")\n      : Node(Kinded::Kind::" << name_ << "NodeKind, name)";

  // Print the initialization list:
  if (!enum_.empty()) {
    os << ", mode_(mode)";
  }

  // Initialize the operands:
  for (const auto &op : nodeInputs_) {
    os << ", " << op << "_(" << "this, " << op << ")";
  }

  // Initialize the members:
  for (const auto &op : members_) {
    if ((op.first).type != MemberType::VectorNodeValue) {
      os << ", " << op.second << "_(" << op.second << ")";
      continue;
    }
    continue;
    os << ", " << op.second << "_(" << op.second << ".begin(), " << op.second
       << ".end()" << ")";
  }

  // The constructor body:
  os << " {\n";
  //  -> fix outputs.
  for (auto &RT : nodeOutputs_) {
    os << "    addResult(" << RT.first << ");\n";
  }
  //  -> variable outputs.
  for (const auto &op : variableOutputMembers_) {
    os << "    " << op.second << "Size_ = " << op.second << ".size();\n";
    os << "    for (size_t idx = 0, e = " << op.second
       << ".size(); idx < e; ++idx) {\n"
       << "        addResult(" << op.second << "[idx]);\n"
       << "    }\n";
  }

  for (const auto &op : members_) {
    if ((op.first).type != MemberType::VectorNodeValue) {
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
    os << "  " << getStorageTypename(&op.first) << " " << op.second << "_;\n";
  }

  for (const auto &op : variableOutputMembers_) {
    os << "  unsigned " << op.second << "Size_;\n";
  }
}

void NodeBuilder::emitMemberGetterSetter(std::ostream &os,
                                         const MemberTypeInfo *typeInfo,
                                         const std::string &name) const {
  // Synthesize the general getter.
  auto typeStr = getReturnTypename(typeInfo);
  os << "  " << typeStr << " get" << name << "() const { return " << name
     << "_; }\n";

  if (typeInfo->addSetter) {
    os << "  void set" << name << "(" << typeStr << " a) {" << name
       << "_ = a; }\n";
  }
}

void NodeBuilder::emitSettersGetters(std::ostream &os) const {
  // Print the getters/setters.
  for (const auto &inName : nodeInputs_) {
    os << "  const NodeValue get" << inName << "() const { return " << inName
       << "_; }\n";
  }

  // -> fixed outputs.
  unsigned idx = 0;
  for (const auto &op : nodeOutputs_) {
    os << "  // Methods for output " << op.second << "\n";
    os << "  NodeValue get" << op.second << "() { return getNthResult(" << idx
       << "); }\n";
    os << "  const NodeValue get" << op.second
       << "() const { return getNthResult(" << idx << "); }\n";
    idx++;
  }
  // -> variable outputs.
  if (variableOutputMembers_.size()) {
    std::string indexShift = std::to_string(nodeOutputs_.size());
    for (const auto &op : variableOutputMembers_) {
      os << "  // Methods for variable output " << op.second << "\n";
      // Returns all the values of the variable output.
      os << "  const std::vector<NodeValue> get" << op.second << "() const {\n"
         << "    unsigned_t indexShift = " << indexShift << ";\n"
         << "    std::vector<NodeValue> valueList;\n"
         << "    for (unsigned i = 0; i < " << op.second << "Size_; i++) {\n"
         << "      valueList.push_back(getNthResult(indexShift + i));\n"
         << "    }\n"
         << "    return valueList;\n"
         << "  }\n";
      // Returns the nth value of the variable output.
      os << "  const NodeValue get" << op.second << "Nth(unsigned i) const {\n"
         << "    return getNthResult(" << indexShift << " + i);\n"
         << "  }\n";
      // Returns the number of elements for the variable output.
      os << "  unsigned get" << op.second << "Size() const {\n"
         << "    return " << op.second << "Size_;\n"
         << "  }\n";
      indexShift += " + " + op.second + "Size_";
    }
  }

  for (const auto &op : members_) {
    emitMemberGetterSetter(os, &op.first, op.second);
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
    if ((op.first).type != MemberType::VectorNodeValue) {
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
    if ((op.first).type != MemberType::VectorNodeValue) {
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
    if ((op.first).type != MemberType::VectorNodeValue) {
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
    if ((op.first).type != MemberType::VectorNodeValue) {
      continue;
    }
    os << "  if (idx < " << op.second << "_.size()) { " << op.second
       << "_[idx] = val; return; }\n  idx -= " << op.second << "_.size();\n";
  }
  os << "  llvm_unreachable(\"Invalid index\");\n}\n";

  os << "\nstd::string " << name_
     << "Node::getOutputName(unsigned idx) const {\n";
  // Handle fixed outputs.
  for (size_t i = 0; i < nodeOutputs_.size(); i++) {
    os << "  if (idx == " << i << ") { return \"" << nodeOutputs_[i].second
       << "\"; }\n";
  }
  // Handle variable outputs.
  os << "  idx -= " << nodeOutputs_.size() << ";\n";
  for (const auto &op : variableOutputMembers_) {
    os << "  if (idx < " << op.second << "Size_) { \n"
       << "    // Pick automatic name for now\n"
       << "    return \"" << op.second << "\" + std::to_string(idx);\n"
       << "  }\n";
    os << "  idx -= " << op.second << "Size_;\n";
  }

  os << "  llvm_unreachable(\"Invalid index\");\n}\n";
}

void NodeBuilder::emitPrettyPrinter(std::ostream &os) const {
  os << "\nstd::string " << name_ << "Node::getDebugDesc() const {\n"
     << "  DescriptionBuilder db(getKindName());\n"
     << "  db.addParam(\"Name\", separateString(getName(), 100, \"\\n\"));\n";

  os << "  if (hasPredicate()) db.addParam(\"Predicate\", \"Yes\");\n";

  os << "  if (isFused()) db.addParam(\"Fused\", \"Yes\");\n";

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
    MemberType ty = (mem.first).type;
    if (ty == MemberType::VectorNodeValue) {
      continue;
    }

    if (ty == MemberType::Enum) {
      os << "    .addParam(\"" << mem.second << "\", static_cast<int>(get"
         << mem.second << "()))\n";
    } else {
      os << "    .addParam(\"" << mem.second << "\", get" << mem.second
         << "())\n";
    }
  }
  os << "    .addParam(\"Users\", getNumUsers());\n";

  for (const auto &mem : members_) {
    if ((mem.first).type != MemberType::VectorNodeValue) {
      continue;
    }

    // Make sure that inputs are properly indexed.
    os << "  {\n";
    os << "  unsigned mIndex = 0;\n";
    os << "  for (const auto &II : get" << mem.second << "()) {\n"
       << "    db.addParam(\"" << mem.second
       << "\"+std::to_string(mIndex++), *II.getType());\n"
       << "  }\n"
       << "  }\n";
  }

  // Generate description for outputs.
  // --> Fix outputs.
  for (const auto &op : nodeOutputs_) {
    os << "  db.addParam(\"" << op.second << "\", *(get" << op.second
       << "().getType()));\n";
  }
  // --> variable outputs.
  if (variableOutputMembers_.size()) {
    for (const auto &op : variableOutputMembers_) {
      os << "  for (unsigned i = 0; i < " << op.second << "Size_; i++) {\n"
         << "   db.addParam(\"" << op.second
         << "\" + std::to_string(i),"
         // Get the type of the ith element of the variable output
         << " *(get" << op.second << "Nth(i).getType()));\n"
         << "  }\n";
    }
  }

  os << "  return db;\n}\n";
}

void NodeBuilder::emitCloner(std::ostream &os) const {
  os << "\nNode* " << name_ << "Node::clone() const {\n";

  // Variable outputs
  if (variableOutputMembers_.size()) {
    os << "  unsigned_t variableOutIdx = " << nodeOutputs_.size() << ";\n"
       << "  llvm::ArrayRef<TypeRef> typeArray(types_);\n";
    for (const auto &op : variableOutputMembers_) {
      os << "  llvm::ArrayRef<TypeRef> resultTypesOf" << op.second
         << " = typeArray.slice(variableOutIdx, " << op.second << "Size_);\n"
         << "  variableOutIdx += " << op.second << "Size_;\n";
    }
  }

  os << "  return new " << name_ << "Node(getName()";

  // Pass the external type arguments:
  // -> fix outputs
  for (const auto &paramName : ctorTypeParams_) {
    os << ", get" << paramName << "().getType()";
  }
  // -> variable outputs
  for (const auto &op : variableOutputMembers_) {
    os << ", resultTypesOf" << op.second;
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

/// \returns true if a can be a part of a valid C/C++ identifier.
static bool isIdentifierChar(char c) { return (c == '_' || isalnum(c)); }

void NodeBuilder::emitEquator(std::ostream &os) const {
  os << "\nbool " << name_ << "Node::isEqual(const " << name_
     << "Node &other) const {\n";

  os << "  bool equal = true";

  if (!enum_.empty()) {
    os << " &&\n      getMode() == other.getMode()";
  }

  for (const auto &op : nodeInputs_) {
    os << " &&\n      " << op << "_ == other." << op << "_";
  }

  os << " &&\n      predicate_ == other.predicate_";

  for (const auto &mem : members_) {
    // Use custom user-defined comparator functions if available.
    std::string cmpFn = mem.first.cmpFn;
    if (cmpFn.empty() || !isIdentifierChar(cmpFn.at(0))) {
      if (cmpFn.empty()) {
        // Default comparator is ==.
        cmpFn = "==";
      }
      os << " &&\n      " << mem.second << "_ " << cmpFn << " other."
         << mem.second << "_";
    } else {
      os << " &&\n      " << cmpFn << "(" << mem.second << "_, other."
         << mem.second << "_)";
    }
  }

  // Fix outputs.
  for (int i = 0, e = nodeOutputs_.size(); i < e; i++) {
    os << " &&\n      getType(" << i << ") == other.getType(" << i << ")";
  }
  os << ";\n";

  // Variable outputs.
  if (variableOutputMembers_.size()) {
    os << "  equal = equal && (getNumResults() == other.getNumResults());\n"
       << "  for (unsigned i = " << nodeOutputs_.size()
       << "; i < getNumResults(); i++) {\n"
       << "    equal = equal && (getType(i) == other.getType(i));\n"
       << "  }\n";
  }
  os << "  return equal;\n}\n";
}

static bool isVectorType(MemberType ty) {
  return ty == MemberType::VectorFloat || ty == MemberType::VectorNodeValue ||
         ty == MemberType::VectorSizeT || ty == MemberType::VectorDimT ||
         ty == MemberType::VectorUnsigned || ty == MemberType::VectorInt64 ||
         ty == MemberType::VectorSigned;
}

static bool isFloatVectorType(MemberType ty) {
  return ty == MemberType::VectorFloat;
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
    auto ty = (mem.first).type;
    if (ty == MemberType::Float) {
      os << delim << "\n      toBinary(" << mem.second << "_)";
    } else if (isFloatVectorType(ty)) {
      os << delim
         << "\n      [](const std::vector<float>& floatVec) -> llvm::hash_code "
            "{\n        std::vector<size_t> sizeVec = toBinary(floatVec);\n    "
            "    return llvm::hash_combine_range(sizeVec.begin(), "
            "sizeVec.end());\n      }("
         << mem.second << "_)";
    } else if (isVectorType(ty)) {
      os << delim << "\n      llvm::hash_combine_range(" << mem.second
         << "_.begin(), " << mem.second << "_.end())";
    } else if (ty == MemberType::Enum) {
      os << delim << "\n      static_cast<int>(" << mem.second << "_)";
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
    if ((op.first).type == MemberType::VectorNodeValue) {
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

void NodeBuilder::emitIndicesEnum(std::ostream &os) const {
  os << "  enum InputIndices {\n";
  for (size_t i = 0; i < nodeInputs_.size(); i++) {
    os << "    ";
    os << nodeInputs_[i];
    os << "Idx = " << i << ",\n";
  }
  os << "  };\n\n";

  os << "  enum ResultIndices {\n";
  for (int i = 0, e = nodeOutputs_.size(); i < e; i++) {
    os << "    ";
    os << nodeOutputs_[i].second;
    os << "Idx = " << i << ",\n";
  }
  os << "  };\n\n";
}

void NodeBuilder::emitNodeClass(std::ostream &os) const {
  emitMemberForwardDecls(os);

  os << "\nnamespace glow {\n";

  emitDocstring(os);

  os << "class " << name_ << "Node final : public Node {\n";

  emitClassMembers(os);

  os << "\n public:\n";

  emitIndicesEnum(os);
  emitCtor(os);
  emitSettersGetters(os);

  os << "  unsigned getNumInputs() const;\n"
     << "  std::string getInputName(unsigned idx) const;\n"
     << "  NodeValue getNthInput(unsigned idx);\n"
     << "  void setNthInput(unsigned idx, NodeValue val);\n"
     << "  std::string getOutputName(unsigned idx) const;\n"
     << "  bool hasSideEffects() const { return " << hasSideEffects_ << "; }\n"
     << "  bool isCanonical() const { return " << !isBackendSpecific_ << "; }\n"
     << "  bool isDataParallel() const { return " << isDataParallel_ << "; }\n"
     << "  std::string getDebugDesc() const;\n"
     << "  bool isEqual(const " << name_ << "Node &other) const;\n"
     << "  llvm::hash_code getHash() const;\n"
     << "  void visit(Node *parent, NodeWalker *visitor);\n"
     << "  Node* clone() const;\n"
     << "  bool verify() const;\n";

  if (hasExtraResults_) {
    os << "  void addExtraResult(TypeRef T) { addResult(T); }\n";
  }

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
  if (!skipAutogenDebugDesc_) {
    emitPrettyPrinter(os);
  }
  if (!skipAutogenVisitor_) {
    emitVisitor(os);
  }
  emitEquator(os);
  emitCloner(os);
  if (!skipAutogenHasher_) {
    emitHasher(os);
  }
  if (!enum_.empty()) {
    emitEnumModePrinters(os);
  }

  // Emit the "extra" method bodies.
  for (const auto &m : extraMethods_) {
    os << m.second;
  }
}

bool NodeBuilder::hasCtorTypeParams(llvm::StringRef res) const {
  for (const std::string &s : ctorTypeParams_) {
    if (s == res) {
      return true;
    }
  }
  return false;
}

void NodeBuilder::emitImportMethods(std::ostream &os) const {
  os << "if (typeName == \"Glow_" << name_ << "\") {\n";

  // Load all the inputs.
  for (size_t i = 0, e = nodeInputs_.size(); i < e; i++) {
    auto &op = nodeInputs_[i];
    os << "  NodeValue " << op << ";\n";
    os << "  ASSIGN_VALUE_OR_RETURN_ERR(" << op
       << ", getNodeValueByName(op.input(" << i << ")));\n\n";
  }

  // Load all the output types.
  for (size_t i = 0, e = nodeOutputs_.size(); i < e; i++) {
    auto &op = nodeOutputs_[i];
    if (hasCtorTypeParams(op.second)) {
      os << "  TypeRef " << op.second << "OutTy;\n";
      os << "  ASSIGN_VALUE_OR_RETURN_ERR(" << op.second
         << "OutTy, loadTypeFromAttributes(" << std::to_string(i)
         << ", dict));\n\n";
    }
  }
  for (const auto &op : variableOutputMembers_) {
    auto ty = getCtorArgTypename(MemberType::Unsigned);
    os << "  " << ty << " " << op.second << "Size;\n"
       << "   std::vector<TypeRef> " << op.second << "OutTyList;\n"

       << "  ASSIGN_VALUE_OR_RETURN_ERR(" << op.second << "Size, loadAttribute<"
       << ty << ">(dict.at(\"" << op.second << "Size\"), *this));\n"
       << "  for (unsigned i = 0; i < " << op.second << "Size; i++) {\n"
       << "    TypeRef OutTy;\n"
       << "    ASSIGN_VALUE_OR_RETURN_ERR(OutTy, loadTypeFromAttributes(i, "
          "dict, \""
       << op.second << "_\"));\n"
       << "    " << op.second << "OutTyList.push_back(OutTy);\n"
       << "  }\n";
  }

  // Load the members.
  for (const auto &op : members_) {
    auto ty = getCtorArgTypename(&op.first);
    os << "  " << ty << " " << op.second << ";\n";
    os << "  ASSIGN_VALUE_OR_RETURN_ERR(" << op.second;
    os << ", loadAttribute<" << ty << ">(dict.at(\"" << op.second
       << "\"), *this));\n\n";
  }

  // We have all items needed to construct the node, so do so.
  const auto nodeName = name_ + "Node";
  os << "  " << nodeName << " *loadedNode = G_->addNode(new " << nodeName
     << "(opName";
  for (const auto &op : nodeOutputs_) {
    if (hasCtorTypeParams(op.second)) {
      os << ", " << op.second << "OutTy";
    }
  }
  for (const auto &op : variableOutputMembers_) {
    os << ", " << op.second << "OutTyList";
  }
  for (size_t i = 0, e = nodeInputs_.size(); i < e; i++) {
    auto &op = nodeInputs_[i];
    os << ", " << op;
  }
  for (const auto &op : members_) {
    os << ", " << op.second;
  }
  os << "));\n\n";

  // Now load a predicate if one exists.
  os << "  if (dict.count(\"Predicate\")) {\n";
  os << "    NodeValue Predicate;\n";
  os << "    ASSIGN_VALUE_OR_RETURN_ERR(Predicate, "
        "loadAttribute<NodeValue>(dict.at(\"Predicate\"), *this));\n";
  os << "    loadedNode->setPredicate(Predicate);\n";
  os << "  }\n\n";

  // Add the node to the Function and return it.
  os << "  RETURN_IF_ERR(addNodeAsOutput(op, loadedNode));\n";
  os << "  return loadedNode;\n";
  os << "}\n\n";
}

void NodeBuilder::emitExportMethods(std::ostream &os) const {
  os << "case glow::Kinded::Kind::" << name_ << "NodeKind: {\n";
  os << "  auto *N__ = llvm::cast<" << name_ << "Node>(node);\n";

  // Add the node. Note that Glow custom ops are prefixed with "Glow_"
  os << "  opProto = graph.add_node();\n";
  os << "  opProto->set_op_type(\"Glow_" << name_ << "\");\n";
  os << "  opProto->set_name(glow::legalizeName(N__->getName()));\n";

  // In order to manage variable inputs and outputs and export a clean ONNX
  // model, the export is done with a generic code working on the Node object,
  // the variability being well managed in the node getter function.
  os << "  for(unsigned i=0; i<node->getNumInputs(); i++) {\n";
  os << "    opProto->add_input(node->getNthInput(i).generateNodeOutputName(/* "
        "stripResNoFor0thInput */ true));\n";
  os << "    addTypeAttributes(opProto, N__, i, /* isInput */ true);\n";
  os << "  }\n";
  os << "  for(unsigned i=0; i<node->getNumResults(); i++) {\n";
  os << "    "
        "opProto->add_output(node->getNthResult(i).generateNodeOutputName(/* "
        "stripResNoFor0thInput */ true));\n";
  os << "    addTypeAttributes(opProto, N__, i, /* isInput */ false);\n";
  os << "  }\n";

  // Variadic parameters are always the last params. But in case there are
  // multiple variadic inputs or outputs, there is the need to know their index
  // and size in the input or output list. This information is exported as
  // extra attributes.
  bool hasVariadicInput = false;
  for (const auto &op : members_) {
    if ((op.first).type != MemberType::VectorNodeValue) {
      continue;
    }
    if (!hasVariadicInput) {
      os << "  // Handle variadic inputs\n";
      os << "  {\n";
      os << "    unsigned_t index = " << nodeInputs_.size() << ";\n";
      hasVariadicInput = true;
    }
    std::string sParamSize = "N__->get" + op.second + "().size()";
    os << "    addValueAttribute(opProto, \"" << op.second
       << "_Index\", index);\n";
    os << "    addValueAttribute(opProto, \"" << op.second << "Size\", "
       << sParamSize << ");\n";
    os << "    index += " + sParamSize + ";\n";
  }
  if (hasVariadicInput) {
    os << "  }\n";
  }
  if (variableOutputMembers_.size() > 0) {
    os << "  // Handle variadic outputs\n";
    os << "  {\n";
    os << "    unsigned_t index = " << nodeOutputs_.size() << ";\n";
    for (const auto &op : variableOutputMembers_) {
      std::string sParamSize = "N__->get" + op.second + "Size()";
      os << "    addValueAttribute(opProto, \"" << op.second
         << "_Index\", index);\n";
      os << "    addValueAttribute(opProto, \"" << op.second
         << "Size\", " + sParamSize + ");\n";
      os << "    index += " + sParamSize + ";\n";
    }
    os << "  }\n";
  }

  // Add any members the node has.
  for (const auto &op : members_) {
    os << "  addValueAttribute(opProto, \"" << op.second << "\", N__->get"
       << op.second << "());\n";

    // If the member is a VectorNodeValue then also add the types of the NVs.
    if (op.first.type == MemberType::VectorNodeValue) {
      os << "  for (unsigned i = 0, e = N__->get" << op.second
         << "().size(); i < e; i++) {\n";
      os << "    addTypeAttributes(opProto, N__->get" << op.second
         << "()[i], i, /* isInput */ true, \"" << op.second << "_\");\n";
      os << "  }\n";
    }
  }

  // Check if the node has a predicate and add it if so.
  os << "  if (N__->hasPredicate()) {\n";
  os << "    addValueAttribute(opProto, \"Predicate\",  "
        "N__->getPredicate().generateNodeOutputName(/* stripResNoFor0thInput "
        "*/ true));\n";
  os << "  }\n";

  os << "  break;\n";
  os << "}\n\n";
}

NodeBuilder &NodeBuilder::addGradient() {
  NodeBuilder GN(hStream, cStream, dStream, iStream, eStream, name_ + "Grad",
                 isBackendSpecific_);

  // The new 'Grad' class will have all of the fields of the current class.
  GN.members_ = members_;
  GN.enum_ = enum_;
  GN.isDataParallel_ = isDataParallel_;

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
     << "  auto *x = new " + name_ + "GradNode(getName().str() + \"_grad\"";

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
  if (!skipAutogenSerialization_) {
    emitImportMethods(iStream);
    emitExportMethods(eStream);
  }
}
