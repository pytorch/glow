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

#include "InstrBuilder.h"
#include "glow/Support/Compiler.h"

unsigned InstrBuilder::getOperandIndexByName(llvm::StringRef name) const {
  for (unsigned i = 0; i < operands_.size(); i++) {
    if (name == operands_[i].first) {
      return i;
    }
  }

  llvm_unreachable("Can't find an operand with this name");
}

void InstrBuilder::emitCtor(std::ostream &os) const {
  os << "  " << name_ << "Inst(llvm::StringRef name";

  // The operands of the instruction class:
  for (const auto &op : operands_) {
    os << ", Value *" << op.first;
  }

  // Extra class members:
  for (const auto &op : members_) {
    os << ", " << getStorageTypename(&op.first) << " " << op.second;
  }

  // Initialize the base clases:
  os << ")\n      : Instruction(name, Kinded::Kind::" << name_ << "InstKind, "
     << ty_ << ", {\n";

  // The operands of the instruction class:
  for (const auto &op : operands_) {
    os << "          {" << op.first
       << ", OperandKind::" << getOperandKindStr(op.second) << "},\n";
  }
  os << "      })";

  // Initialize the members:
  for (const auto &op : members_) {
    os << ", " << op.second << "_(" << op.second << ")";
  }

  // Empty constructor body.
  os << " {}\n\n";
}

void InstrBuilder::emitIRBuilderMethods(std::ostream &osH,
                                        std::ostream &osB) const {
  osH << "\n"
      << name_ << "Inst *create" << name_ << "Inst(llvm::StringRef name";
  osB << "\n"
      << name_ << "Inst *IRBuilder::create" << name_
      << "Inst(llvm::StringRef name";

  // The operands of the instruction class:
  for (const auto &op : operands_) {
    // Scratch operands are not exposed in the builder method interface
    // but only in the instruction constructor.
    if (op.second == OperandKind::Scratch) {
      continue;
    }
    osH << ", Value *" << op.first;
    osB << ", Value *" << op.first;
  }

  // Extra class members:
  for (const auto &op : members_) {
    osH << ", " << getStorageTypename(&op.first) << " " << op.second;
    osB << ", " << getStorageTypename(&op.first) << " " << op.second;
  }
  osH << ");\n";
  osB << ") {\n";

  // Create allocations for the scratch operands.
  for (const auto &op : operands_) {
    if (op.second == OperandKind::Scratch) {
      std::string allocSuffix = llvm::StringRef(op.first).lower();
      osB << "  std::string " << op.first << "Name = name.str() + \"."
          << allocSuffix << "\";\n";
      osB << "  auto *" << op.first << "Type = F_->getParent()"
          << "->uniqueType(ElemKind::Int8QTy, {1}, 0.0, 0);\n";
      osB << "  auto *" << op.first << " = createAllocActivationInst("
          << op.first << "Name, " << op.first << "Type);\n";
    }
  }

  // Initialize the base clases:
  osB << "  auto *A = new " << name_ << "Inst(uniqueName(name)";

  // The operands of the instruction class:
  for (const auto &op : operands_) {
    osB << ", " << op.first;
  }
  // Extra class members:
  for (const auto &op : members_) {
    osB << ", " << op.second;
  }
  osB << ");\n";

  // Modify allocation sizes based on the instruction requirements.
  // We allocate at least 1 byte since the memory allocator does not
  // handle properly allocation sizes of 0.
  for (const auto &op : operands_) {
    if (op.second == OperandKind::Scratch) {
      // A special case is when the instruction already has a member called
      // "<Operand>Size" for which we allow a different type than dim_t for
      // flexibility and hence we create a local cast here to dim_t.
      osB << "  dim_t " << op.first << "SizeVar = static_cast<dim_t>(A->get"
          << op.first << "Size());\n";
      osB << "  " << op.first << "SizeVar = " << op.first << "SizeVar > 0 ? "
          << op.first << "SizeVar : 1;\n";
      osB << "  auto *" << op.first << "TypeResized = F_->getParent()"
          << "->uniqueType(ElemKind::Int8QTy, {" << op.first
          << "SizeVar}, 0.0, 0);\n";
      osB << "  " << op.first << "->setType(" << op.first << "TypeResized);\n";
      osB << "  " << op.first << "->setTy(" << op.first << "TypeResized);\n";
    }
  }

  osB << "  F_->pushInstr(A);\n  return A;\n}\n";
}

void InstrBuilder::emitInplaceMethod(std::ostream &os) const {
  os << "\n  bool isInplaceOp(unsigned dstIdx, unsigned srcIdx) const {\n";
  if (!inplaceOperands_.empty()) {
    for (const auto &curInplaceOperands : inplaceOperands_) {
      assert(curInplaceOperands.size() > 1 &&
             "We don't have a pair of inplace args");
      for (int i = 1, e = curInplaceOperands.size(); i < e; i++) {
        auto F0 = getOperandIndexByName(curInplaceOperands[0]);
        auto F1 = getOperandIndexByName(curInplaceOperands[i]);
        os << "  if (" << F0 << " == dstIdx && " << F1
           << " == srcIdx) { return true; }\n";
      }
    }
  }
  os << "    return false;\n  }\n";
}

void InstrBuilder::emitCanonicalProperty(std::ostream &os) const {
  os << "\n  bool isCanonical() const {\n";
  os << "    return " << (isBackendSpecific_ ? "false" : "true") << ";\n  }\n";
}

void InstrBuilder::emitDataParallelProperty(std::ostream &os) const {
  os << "\n  bool isDataParallel() const {\n";
  os << "    return " << (isDataParallel_ ? "true" : "false") << ";\n  }\n";
}

void InstrBuilder::emitProperties(std::ostream &os) const {
  emitInplaceMethod(os);
  emitCanonicalProperty(os);
  emitDataParallelProperty(os);
}

void InstrBuilder::emitClassMembers(std::ostream &os) const {
  // Emit class members:
  for (const auto &op : members_) {
    os << "  " << getStorageTypename(&op.first) << " " << op.second << "_;\n";
  }
}

void InstrBuilder::emitOperandGetter(std::ostream &os, const std::string &name,
                                     int index) const {
  // Synthesize the general operand getter.
  os << "  Value *get" << name << "() const { return getOperand(" << index
     << ").first; }\n";
}

void InstrBuilder::emitMemberGetter(std::ostream &os,
                                    const MemberTypeInfo *type,
                                    const std::string &name) const {
  // Synthesize the general getter.
  auto returnTypeStr = getReturnTypename(type);
  os << "  " << returnTypeStr << " get" << name << "() const { return " << name
     << "_; }\n";
}

void InstrBuilder::emitSettersGetters(std::ostream &os) const {
  // Print the getters/setters.
  for (int i = 0, e = operands_.size(); i < e; i++) {
    auto &op = operands_[i];
    emitOperandGetter(os, op.first, i);
  }

  for (const auto &op : members_) {
    emitMemberGetter(os, &op.first, op.second);
  }

  // Print size getter declarations for scratch operands. The functions will be
  // manually implemented by the instruction creator.
  for (const auto &op : operands_) {
    if (op.second == OperandKind::Scratch) {
      // A special case is when the instruction already has a member called
      // "<Operand>Size" for which a getter was already emitted. We detect this
      // particular case and not emit the getter again.
      bool hasScratchSizeMember = false;
      for (const auto &memb : members_) {
        if (memb.second == (op.first + "Size")) {
          hasScratchSizeMember = true;
          break;
        }
      }
      if (!hasScratchSizeMember) {
        os << "  dim_t get" << op.first << "Size() const;\n";
      }
    }
  }

  // Synthesize the 'classof' method that enables the non-rtti polymorphism.
  os << "\n  static bool classof(const Kinded *k) {\n"
     << "    return k->getKind() == Kinded::Kind::" << name_ << "InstKind;\n"
     << "  }\n";
}

void InstrBuilder::emitPrettyPrinter(std::ostream &os) const {
  os << "\nvoid " << name_ << "Inst::dump(llvm::raw_ostream &os) const {\n";
  os << "  os << \"%\" << (std::string) getName() << \" = \" << getKindName()"
     << " << \" \";\n  dumpOperands(os);\n";

  if (!members_.empty()) {
    os << "  os << \" {\"\n";
    bool first = true;
    for (const auto &mem : members_) {
      os << "     << \"" << (first ? " " : ", ") << mem.second << ": \" << "
         << "get" << mem.second << "()\n";
      first = false;
    }
    os << "     << \"}\";\n";
  }
  os << "}\n";
}

void InstrBuilder::emitCloner(std::ostream &os) const {
  os << "\nInstruction* " << name_ << "Inst::clone() const {\n";

  os << "  return new " << name_ << "Inst(getName()";

  for (const auto &op : operands_) {
    os << ", get" << op.first << "()";
  }

  for (const auto &mem : members_) {
    os << ", get" << mem.second << "()";
  }

  os << ");\n}\n";
}

void InstrBuilder::emitGetOperandName(std::ostream &os) const {
  os << "\nllvm::StringRef " << name_
     << "Inst::getOperandName(unsigned idx) const {\n";
  for (size_t i = 0; i < operands_.size(); i++) {
    os << "  if (idx == " << i << ") { return \"" << operands_[i].first
       << "\"; }\n";
  }
  os << "  llvm_unreachable(\"Invalid index\");\n}\n";
}

std::string getOpElementType(const std::string &name) {
  const std::string elemKindPrefix = "ElemKind::";
  if (name.substr(0, elemKindPrefix.size()) == elemKindPrefix) {
    return name;
  }
  return "get" + name + "()->getElementType()";
}

void InstrBuilder::emitClass(std::ostream &os) const {
  os << "\nnamespace glow {\nclass " << name_
     << "Inst final : public Instruction {\n";

  emitClassMembers(os);

  os << "\n public:\n";

  emitCtor(os);
  emitSettersGetters(os);
  emitProperties(os);

  for (const auto &m : extraMethods_) {
    os << "\n  " << m.first << "\n";
  }

  os << "\n  Instruction* clone() const;\n";
  os << "\n  void dump(llvm::raw_ostream &os) const;\n";
  os << "\n  llvm::StringRef getOperandName(unsigned idx) const;\n";

  // If there is no auto-verification then we assume verification is manually
  // provided.
  if (autoVerificationPairs_.empty()) {
    os << "  void verify() const;\n";
  } else {
    os << "  void verify() const {\n";
    // Generate auto-verification checks for the current type of the node.
    for (auto &pair : autoVerificationPairs_) {
      switch (pair.first) {
      // Generates a check that two operands of an instruction are of the same
      // type.
      case VerifyKind::SameType: {
        for (size_t i = 1, e = pair.second.size(); i < e; i++) {
          os << "    assert(get" << pair.second[0] << "()->getType() == get"
             << pair.second[i] << "()->getType() && \"Invalid Type\");\n";
        }
        break;
      }
      // Generates a check that two operands of an instruction are of the same
      // shape.
      case VerifyKind::SameShape: {
        for (size_t i = 1, e = pair.second.size(); i < e; i++) {
          os << "    assert(get" << pair.second[0] << "()->dims().equals(get"
             << pair.second[i] << "()->dims()) && \"Invalid Shape\");\n";
        }
        break;
      }
      // Generates a check that two operands of an instruction have elements of
      // the same type.
      case VerifyKind::SameElementType: {
        auto firstOp = getOpElementType(pair.second[0]);
        for (size_t i = 1, e = pair.second.size(); i < e; i++) {
          os << "    assert(" << firstOp
             << " == " << getOpElementType(pair.second[i])
             << " && \"Invalid Element Type\");\n";
        }
        break;
      }
      // Generates a check that the type of an operand satisfies a specific
      // check performed by a predicate method on a type.
      case VerifyKind::TypeCheck: {
        for (size_t i = 1, e = pair.second.size(); i < e; i++) {
          os << "    assert(get" << pair.second[0] << "()->getType()->"
             << pair.second[i] << " && \"Invalid Type\");\n";
        }
        break;
      }
      // No verification check needs to be generated.
      case VerifyKind::NoVerify: {
        assert(autoVerificationPairs_.size() == 1);
        os << "    // Nothing to verify.\n";
        break;
      }
      default:
        llvm_unreachable("Unknown verification kind.");
        break;
      }
    }
    os << "  }\n";
  }
  os << "};\n} // namespace glow\n";
}

void InstrBuilder::emitCppMethods(std::ostream &os) const {
  emitPrettyPrinter(os);
  emitCloner(os);
  emitGetOperandName(os);
  // Emit the "extra" method bodies.
  for (const auto &m : extraMethods_) {
    os << "\n" << m.second << "\n";
  }
}

InstrBuilder::~InstrBuilder() {
  emitClass(headerStream);
  emitCppMethods(cppStream);
  emitIRBuilderMethods(builderHeaderStream, builderCppStream);
  emitAutoIRGen(irGenStream);
}

void InstrBuilder::addGradientInstr(
    llvm::ArrayRef<llvm::StringRef> originalFields,
    llvm::ArrayRef<llvm::StringRef> gradFields) {
  InstrBuilder GI(headerStream, cppStream, defStream, builderHeaderStream,
                  builderCppStream, irGenStream, name_ + "Grad",
                  isBackendSpecific_);

  // The new 'Grad' class will have all of the fields of the current class.
  GI.ty_ = ty_;
  GI.members_ = members_;
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

  // Copy over the auto-verify information, updating the operand names for
  // gradient.
  for (auto &verifPair : autoVerificationPairs_) {
    auto newPair = std::make_pair(verifPair.first, std::vector<std::string>());
    for (auto &opName : verifPair.second) {
      if (std::find(gradFields.begin(), gradFields.end(), opName) !=
          gradFields.end()) {
        newPair.second.push_back(opName + "Grad");
      }
      if (std::find(originalFields.begin(), originalFields.end(), opName) !=
          originalFields.end()) {
        newPair.second.push_back(opName);
      }
    }
    GI.autoVerificationPairs_.push_back(newPair);
  }
}

void InstrBuilder::emitAutoIRGen(std::ostream &os) const {
  if (autoIRGenNodeName.empty()) {
    return;
  }

  os << "case glow::Kinded::Kind::" << autoIRGenNodeName << "NodeKind: {\n";
  os << "  auto *CN__ = cast<" << autoIRGenNodeName << "Node>(N);\n";

  // Note: The convention is for Nodes to have 'Input's and 'Output's, and for
  // Instrs to have 'Src's and 'Dest's. Thus we map between the two below.

  // A list of pairs (nodeResultName, destOpName).
  llvm::SmallVector<std::pair<std::string, std::string>, 4>
      nodeResultNameToValueName;
  for (const auto &opPair : operands_) {
    // Skip the scratch operands for this instruction since they are not
    // registered as node operands.
    if (opPair.second == OperandKind::Scratch) {
      continue;
    }
    if (opPair.second == OperandKind::In) {
      // All inputs of a node were mapped to the glow::Values already.
      // So, just lookup for each input operand it's Value by using the
      // corresponding input of a node as a key.
      const std::string opNodeName =
          (opPair.first == "Src") ? "Input" : opPair.first;
      os << "  auto *" << opPair.first << " = valueForNode(CN__->get"
         << opNodeName << "());\n";
    } else if (opPair.second == OperandKind::Out) {
      // Remember for each output operand which result of a node produces it.
      auto destOpName = opPair.first;
      auto resNodeName = (destOpName == "Dest") ? "Result" : destOpName;
      nodeResultNameToValueName.emplace_back(
          std::make_pair(resNodeName, destOpName));
    }
  }

  assert(!nodeResultNameToValueName.empty() &&
         "Didn't find a result; Maybe using InOut which isn't yet supported");
  os << "  std::string allocName = std::string(N->getName()) + \".res\";\n";
  // Allocate activations for all output operands.
  for (auto &kv : nodeResultNameToValueName) {
    auto &nodeResultName = kv.first;
    auto &valueName = kv.second;
    // Create activation for the output operand with name valueName using the
    // type of the corresponding node result nodeResultName.
    os << "  auto *" << valueName
       << "__ = builder_.createAllocActivationInst(allocName,"
       << "CN__->get" << nodeResultName << "().getType());\n";
  }
  os << "  auto *V = builder_.create" << name_ << "Inst(N->getName()";

  // Pass down all the output operand Values as Instruction's constructor
  // arguments.
  for (auto &kv : nodeResultNameToValueName) {
    auto &valueName = kv.second;
    os << ", " << valueName << "__";
  }
  // Pass down all the input operand Values as Instruction's constructor
  // arguments.
  for (const auto &opPair : operands_) {
    if (opPair.second == OperandKind::In) {
      os << ", " << opPair.first;
    }
  }
  // Pass down all the additional members as Instruction's constructor
  // arguments.
  for (const auto &memPair : members_) {
    os << ", CN__->get" << memPair.second << "()";
  }
  os << ");\n";

  os << "  if (N->hasPredicate()) { "
        "V->setPredicate(valueForNode(N->getPredicate())); }\n";
  // Register which outputs of a node are mapped to which output operands of the
  // generated instruction.
  for (auto &kv : nodeResultNameToValueName) {
    auto &nodeResultName = kv.first;
    auto &valueName = kv.second;
    os << "  registerIR(CN__->get" << nodeResultName << "(), V->get"
       << valueName << "());\n";
  }
  os << "  nodeToInstr_[N] = V;\n";
  os << "  break;\n";
  os << "}\n";
}

InstrBuilder &InstrBuilder::addMember(MemberType type,
                                      const std::string &name) {
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

  return addMember(*typeInfo, name);
}

InstrBuilder &InstrBuilder::addFusedActivation() {
  // When adding a fused activation we add the activation type and a vector of
  // floating point parameters for parameterized activations (e.g. min and max
  // for Clip or alpha factor for LeakyRelu).
  return addMember(MEMBER_TYPE_INFO(glow::FusedActivation), "FusedActivation")
      .addMember(MemberType::VectorFloat, "FusedActivationArgs");
}
