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
    os << ", " << getStorageTypename(op.first) << " " << op.second;
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
    osH << ", Value *" << op.first;
    osB << ", Value *" << op.first;
  }

  // Extra class members:
  for (const auto &op : members_) {
    osH << ", " << getStorageTypename(op.first) << " " << op.second;
    osB << ", " << getStorageTypename(op.first) << " " << op.second;
  }

  osH << ");\n";

  // Initialize the base clases:
  osB << ") {\n";
  osB << "  auto *A = new " << name_ << "Inst(name ";

  // The operands of the instruction class:
  for (const auto &op : operands_) {
    osB << ", " << op.first;
  }
  // Extra class members:
  for (const auto &op : members_) {
    osB << ", " << op.second;
  }
  osB << ");\n";
  osB << "  F_->pushInstr(A);\n  return A;\n}\n";
}

void InstrBuilder::emitInplaceMethod(std::ostream &os) const {
  os << "\n  bool isInplaceOp(unsigned dstIdx, unsigned srcIdx) const {\n";
  if (!inplaceOperands_.empty()) {
    assert(inplaceOperands_.size() > 1 &&
           "We don't have a pair of inplace args");
    for (int i = 1, e = inplaceOperands_.size(); i < e; i++) {
      auto F0 = getOperandIndexByName(inplaceOperands_[0]);
      auto F1 = getOperandIndexByName(inplaceOperands_[i]);
      os << "  if (" << F0 << " == dstIdx && " << F1
         << " == srcIdx) { return true; }\n";
    }
  }
  os << "    return false;\n  }\n";
}

void InstrBuilder::emitDataParallelProperty(std::ostream &os) const {
  os << "\n  bool isDataParallel() const {\n";
  os << "    return " << (isDataParallel_ ? "true" : "false") << ";\n  }\n";
}

void InstrBuilder::emitProperties(std::ostream &os) const {
  emitInplaceMethod(os);
  emitDataParallelProperty(os);
}

void InstrBuilder::emitClassMembers(std::ostream &os) const {
  // Emit class members:
  for (const auto &op : members_) {
    os << "  " << getStorageTypename(op.first) << " " << op.second << "_;\n";
  }
}

void InstrBuilder::emitOperandGetter(std::ostream &os, const std::string &name,
                                     int index) const {
  // Synthesize the general operand getter.
  os << "  Value *get" << name << "() const { return getOperand(" << index
     << ").first; }\n";
}

void InstrBuilder::emitMemberGetter(std::ostream &os, MemberType type,
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
    emitMemberGetter(os, op.first, op.second);
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
    os << "  " << m.first << "\n";
  }

  os << "\n  void dump(llvm::raw_ostream &os) const;\n";

  // If there is no auto-verification then we assume verification is manually
  // provided.
  if (autoVerificationPairs_.empty()) {
    os << "  void verify() const;\n";
  } else {
    os << "  void verify() const {\n";
    for (auto &pair : autoVerificationPairs_) {
      switch (pair.first) {
      case VerifyKind::SameType: {
        for (size_t i = 1, e = pair.second.size(); i < e; i++) {
          os << "    assert(get" << pair.second[0] << "()->getType() == get"
             << pair.second[i] << "()->getType() && \"Invalid Type\");\n";
        }
        break;
      }
      case VerifyKind::SameShape: {
        for (size_t i = 1, e = pair.second.size(); i < e; i++) {
          os << "    assert(get" << pair.second[0] << "()->dims().equals(get"
             << pair.second[i] << "()->dims()) && \"Invalid Shape\");\n";
        }
        break;
      }
      case VerifyKind::SameElementType: {
        auto firstOp = getOpElementType(pair.second[0]);
        for (size_t i = 1, e = pair.second.size(); i < e; i++) {
          os << "    assert(" << firstOp
             << " == " << getOpElementType(pair.second[i])
             << " && \"Invalid Element Type\");\n";
        }
        break;
      }
      case VerifyKind::NoVerify: {
        assert(autoVerificationPairs_.size() == 1);
        os << "    // Nothing to verify.\n";
        break;
      }
      default:
        assert(false && "Unknown verification kind.");
        break;
      }
    }
    os << "  }\n";
  }
  os << "};\n} // namespace glow\n";
}

void InstrBuilder::emitCppMethods(std::ostream &os) const {
  emitPrettyPrinter(os);

  // Emit the "extra" method bodies.
  for (const auto &m : extraMethods_) {
    os << "  " << m.second << "\n";
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
  std::string destOpName = "";
  std::string resNodeName = "";
  for (const auto &opPair : operands_) {
    if (opPair.second == OperandKind::In) {
      const std::string opNodeName =
          (opPair.first == "Src") ? "Input" : opPair.first;
      os << "  auto *" << opPair.first << " = valueForNode(CN__->get"
         << opNodeName << "());\n";
    } else if (opPair.second == OperandKind::Out) {
      assert(resNodeName.empty() && destOpName.empty() &&
             "Must have multiple results; don't support autogen yet.");
      resNodeName = (opPair.first == "Dest") ? "Result" : opPair.first;
      destOpName = opPair.first;
    }
  }

  assert(!resNodeName.empty() && !destOpName.empty() &&
         "Didn't find a result; Maybe using InOut which isn't yet supported");
  os << "  std::string allocName = std::string(N->getName()) + \".res\";\n";
  os << "  auto *dest__ = builder_.createAllocActivationInst(allocName,"
     << "CN__->get" << resNodeName << "()->getType());\n";
  os << "  auto *V = builder_.create" << name_ << "Inst(\"" << autoIRGenNodeName
     << "\", dest__";
  for (const auto &opPair : operands_) {
    if (opPair.second == OperandKind::In) {
      os << ", " << opPair.first;
    }
  }
  for (const auto &memPair : members_) {
    os << ", CN__->get" << memPair.second << "()";
  }
  os << ");\n";

  os << "  V->setName(N->getName());\n";
  os << "  if (N->hasPredicate()) { "
        "V->setPredicate(valueForNode(N->getPredicate())); }";
  os << "  registerIR(N, V->get" << destOpName << "());\n";
  os << "  nodeToInstr_[N] = V;\n";
  os << "  break;\n";
  os << "}\n";
}
