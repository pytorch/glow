// Copyright 2017 Facebook, Inc. All Rights Reserved.

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
  os << "  " << name_ << "Inst(IRFunction *M, llvm::StringRef name";

  // The operands of the instruction class:
  for (const auto &op : operands_) {
    os << ", Value *" << op.first;
  }

  // Extra class members:
  for (const auto &op : members_) {
    os << ", " << getStorageTypename(op.first) << " " << op.second;
  }

  // Initialize the base clases:
  os << ")\n      : Instruction(M, name, Kinded::Kind::" << name_
     << "InstKind, " << ty_ << ", {\n";

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

void InstrBuilder::emitIRBuilderMethods(std::ostream &os) const {
  os << "\n" << name_ << "Inst *create" << name_ << "Inst(llvm::StringRef name";

  // The operands of the instruction class:
  for (const auto &op : operands_) {
    os << ", Value *" << op.first;
  }

  // Extra class members:
  for (const auto &op : members_) {
    os << ", " << getStorageTypename(op.first) << " " << op.second;
  }

  // Initialize the base clases:
  os << ") {\n";
  os << "  auto *A = new " << name_ << "Inst(&getIRFunction(), name ";

  // The operands of the instruction class:
  for (const auto &op : operands_) {
    os << ", " << op.first;
  }
  // Extra class members:
  for (const auto &op : members_) {
    os << ", " << op.second;
  }
  os << ");\n";
  os << "  F_->pushInstr(A);\n  return A;\n}\n";
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

void InstrBuilder::emitClass(std::ostream &os) const {
  os << "\nnamespace glow {\nclass " << name_
     << "Inst final : public Instruction {\n";

  emitClassMembers(os);

  os << "\n public:\n";

  emitCtor(os);
  emitSettersGetters(os);
  emitInplaceMethod(os);

  for (const auto &m : extraMethods_) {
    os << "  " << m.first << "\n";
  }

  os << "\n  void dump(llvm::raw_ostream &os) const;\n";
  os << "  void verify() const;\n";
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
  emitIRBuilderMethods(builderStream);
}

InstrBuilder &
InstrBuilder::addGradientInstr(llvm::ArrayRef<llvm::StringRef> originalFields,
                               llvm::ArrayRef<llvm::StringRef> gradFields) {
  InstrBuilder GI(headerStream, cppStream, defStream, builderStream,
                  irGenStream, name_ + "Grad", isBackendSpecific_);

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
  return *this;
}

InstrBuilder &InstrBuilder::autoIRGen(const std::string &name) {
  const std::string &nodeName = (name.empty() ? name_ : name);
  irGenStream << "case glow::Kinded::Kind::" << nodeName << "NodeKind: {\n";
  irGenStream << "  auto *CN__ = cast<" << nodeName << "Node>(N);\n";

  // Note: The convention is for Nodes to have 'Input's and 'Output's, and for
  // Instrs to have 'Src's and 'Dest's. Thus we map between the two below.
  std::string destOpName = "";
  std::string resNodeName = "";
  for (const auto &opPair : operands_) {
    if (opPair.second == OperandKind::In) {
      const std::string opNodeName =
          (opPair.first == "Src") ? "Input" : opPair.first;
      irGenStream << "  auto *" << opPair.first << " = valueForNode(CN__->get"
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

  irGenStream << "  auto *dest__ = builder_.createAllocActivationInst(\""
              << nodeName << ".res\", CN__->get" << resNodeName
              << "()->getType());\n";
  irGenStream << "  auto *V = builder_.create" << name_ << "Inst(\"" << nodeName
              << "\", dest__";
  for (const auto &opPair : operands_) {
    if (opPair.second == OperandKind::In) {
      irGenStream << ", " << opPair.first;
    }
  }
  for (const auto &memPair : members_) {
    irGenStream << ", CN__->get" << memPair.second << "()";
  }
  irGenStream << ");\n";

  irGenStream << "  V->setName(N->getName());\n";
  irGenStream << "  registerIR(N, V->get" << destOpName << "());\n";
  irGenStream << "  nodeToInstr_[N] = V;\n";
  irGenStream << "  break;\n";
  irGenStream << "}\n";

  return *this;
}
