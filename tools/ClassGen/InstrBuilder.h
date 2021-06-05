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
#ifndef GLOW_TOOLS_NODEGEN_INSTRBUILDER_H
#define GLOW_TOOLS_NODEGEN_INSTRBUILDER_H

#include "MemberType.h"
#include "glow/Support/Support.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

class Builder;

enum class OperandKind : unsigned char {
  In,
  Out,
  InOut,
  /// The 'Scratch' operand kind is similar to the 'Out' operand kind with the
  /// following distinctions:
  /// - is intended to be used as a temporary memory buffer for Instructions
  ///   to write some temporary data before writing the final results using
  ///   their 'Out' operands.
  /// - it is NOT intended to be consumed by other Instructions and hence it is
  ///   deallocated immediately after the Instruction execution.
  /// - it is only exposed in the Instruction constructor and NOT in the IR
  ///   builder method 'createXInst'. The intention is to have the IR builder
  ///   method manage the scratch allocation/deallocation automatically.
  /// - when defining a 'Scratch' operand named 'X', the instruction builder
  ///   automatically declares and uses a method 'getXSize()' as part of the
  ///   respective instruction which must be implemented by the developer in
  ///   order for the scratch size requirements (in bytes) to be provided.
  Scratch,
};

enum class VerifyKind : unsigned char {
  SameShape,
  SameType,
  SameElementType,
  TypeCheck,
  NoVerify,
};

inline OperandKind negateOperandKind(OperandKind CC) {
  switch (CC) {
  case OperandKind::In:
    return OperandKind::Out;
  case OperandKind::Out:
    return OperandKind::In;
  case OperandKind::InOut:
    return OperandKind::InOut;
  case OperandKind::Scratch:
    return OperandKind::Scratch;
  }
  llvm_unreachable("Invalid operand kind.");
}

inline const char *getOperandKindStr(OperandKind CC) {
  const char *names[] = {"In", "Out", "InOut", "Out", nullptr};
  return names[(int)CC];
}

class InstrBuilder {
  /// The type-initialization expression.
  std::string ty_{"nullptr"};
  /// The instruction name.
  std::string name_;
  /// The instruction operands.
  std::vector<std::pair<std::string, OperandKind>> operands_;
  /// A list of instruction members. Format: (type, name).
  std::vector<std::pair<MemberTypeInfo, std::string>> members_;
  /// Stores the decl and body of a new public method that will be added to the
  /// class.
  std::vector<std::pair<std::string, std::string>> extraMethods_;
  /// A list of list of operands that are declared as 'inplace' operands.
  /// Each list depicts the 'inplace' operands for one output.
  /// The output is the first element of the related list.
  std::list<std::vector<std::string>> inplaceOperands_;
  /// A list of (VerifyKind, {op1, op2, ...}) pairs. Each pair represents a
  /// specific kind of verification to apply on the list of operands.
  std::vector<std::pair<VerifyKind, std::vector<std::string>>>
      autoVerificationPairs_;
  /// If autoIRGen is used on this Instr, this is the name of the Node that
  /// generates to this Instr. If left empty then autoIRGen is not used.
  std::string autoIRGenNodeName;

  /// Header file stream.
  std::ofstream &headerStream;
  /// CPP file stream.
  std::ofstream &cppStream;
  /// Def-enum file stream.
  std::ofstream &defStream;
  /// The IRBuilder header stream.
  std::ofstream &builderHeaderStream;
  /// The IRBuilder CPP stream.
  std::ofstream &builderCppStream;
  /// The IRGen stream.
  std::ofstream &irGenStream;

  /// Specifies if this Instr is backend specific.
  bool isBackendSpecific_{false};

  /// Specifies if this Instr is data parallel.
  bool isDataParallel_{false};

  /// \returns the index of the operand with the name \p name. Aborts if no such
  /// name.
  unsigned getOperandIndexByName(llvm::StringRef name) const;

public:
  InstrBuilder(std::ofstream &H, std::ofstream &C, std::ofstream &D,
               std::ofstream &BH, std::ofstream &BC, std::ofstream &I,
               const std::string &name, bool isBackendSpecific)
      : name_(name), headerStream(H), cppStream(C), defStream(D),
        builderHeaderStream(BH), builderCppStream(BC), irGenStream(I),
        isBackendSpecific_(isBackendSpecific) {
    defStream << (isBackendSpecific_ ? "DEF_BACKEND_SPECIFIC_INSTR("
                                     : "DEF_INSTR(")
              << name << "Inst, " << glow::tolower(name) << ")\n";
  }

  /// Add an operand to the instruction. The name should start with a capital
  /// letter. For example: "Input".
  InstrBuilder &addOperand(const std::string &op, OperandKind k) {
    operands_.push_back({op, k});
    return *this;
  }

  /// Add a member to the instruction. Format: type, name.
  /// The name should start with a capital letter.
  /// For example: "Filter".
  InstrBuilder &addMember(const MemberType type, const std::string &name);

  /// Add a member to the node. Format type, name.
  /// The name should start with a capital letter.
  /// For example: "Filter".
  /// If MemberTypeInfo refers to an external user-defined type, this type T
  /// should satisfy the following requirements:
  ///   * There should be a hash function with a signature like `llvm::hash_code
  ///   hash_value(const T)` which takes T by value, by reference or as a
  ///   pointer, depending on the intended use.
  ///   * There should be a stream output operator with a signature like
  ///   `llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const T);`, which
  ///   takes T by value, by reference or as a pointer, depending on the
  ///   intended use.
  ///   * There should be a comparison operator `bool operator==(const T LHS,
  ///   const T RHS)` (or a custom comparator function mentioned in
  ///   MemberTypeInfo::cmpFn), which takes T by reference or by value depending
  ///   on the intended use.
  InstrBuilder &addMember(MemberTypeInfo typeInfo, const std::string &name) {
    members_.push_back({typeInfo, name});
    return *this;
  }

  /// Adds the body of a new public method to the class. \p decl is the
  /// decleration that goes in the header file. \p body is the implementation
  /// that goes in the cpp file.
  InstrBuilder &addExtraMethod(const std::string &decl,
                               const std::string &body) {
    extraMethods_.push_back(std::make_pair(decl, body));
    return *this;
  }

  /// Set the expression that initializes the type of the instruction.
  /// Example: 'LHS->getType()'.
  InstrBuilder &setType(const std::string &ty) {
    ty_ = ty;
    return *this;
  }

  /// Adds a list of inplace operands. The instruction may use the memory
  /// read by any of the operands in \p lst[1 .. n] for writing the result of
  /// the operand \p lst[0].
  InstrBuilder &inplaceOperand(llvm::ArrayRef<llvm::StringRef> lst) {
    assert(lst.size() > 1 && "Not enough operands");
    inplaceOperands_.emplace_back(lst.begin(), lst.end());
    // Check that the output parameter is described at most once.
    for (auto it = inplaceOperands_.begin(), end = inplaceOperands_.end();
         it != end; ++it) {
      for (auto secondIt = std::next(it); secondIt != end; ++secondIt) {
        assert(getOperandIndexByName((*it)[0]) !=
                   getOperandIndexByName((*secondIt)[0]) &&
               "Inplace operands for output appears more than once");
      }
    }
    return *this;
  }

  /// Constructs a new gradient instruction that is based on the current
  /// instruction that we are building.
  void addGradientInstr(llvm::ArrayRef<llvm::StringRef> originalFields,
                        llvm::ArrayRef<llvm::StringRef> gradFields);

  /// Turns on automatic IRGen generation for this instruction given the Node \p
  /// name (if empty, defaults to same name as Instr).
  InstrBuilder &autoIRGen(const std::string &name = "") {
    autoIRGenNodeName = (name.empty() ? name_ : name);
    return *this;
  }

  /// Automatically generates verification of type \p verif
  InstrBuilder &autoVerify(VerifyKind verif,
                           llvm::ArrayRef<std::string> operands = {""}) {
    if (verif != VerifyKind::NoVerify) {
      assert(operands.size() > 1 && "Must list 2 or more operands.");
    }
    auto newPair = std::make_pair(verif, std::vector<std::string>());
    newPair.second.insert(newPair.second.begin(), operands.begin(),
                          operands.end());
    autoVerificationPairs_.push_back(newPair);
    return *this;
  }

  InstrBuilder &dataParallel() {
    isDataParallel_ = true;
    return *this;
  }

  /// Helper to add a FusedActivation Member to this instruction.
  InstrBuilder &addFusedActivation();

  ~InstrBuilder();

private:
  /// Emit the class constructor.
  void emitCtor(std::ostream &os) const;

  /// Emit the methods that make the IRBuilder.
  void emitIRBuilderMethods(std::ostream &osH, std::ostream &osB) const;

  /// Emits the class members (the fields of the class).
  void emitClassMembers(std::ostream &os) const;

  /// Emits the method that calculates the inplace property.
  void emitInplaceMethod(std::ostream &os) const;

  /// Emits the property that returns true if the instruction is canonical.
  void emitCanonicalProperty(std::ostream &os) const;

  /// Emits the property that returns true if the instruction is data parallel.
  void emitDataParallelProperty(std::ostream &os) const;

  /// Emits the methods that are properties of the instructions.
  void emitProperties(std::ostream &os) const;

  /// Emit the getter for an operand.
  void emitOperandGetter(std::ostream &os, const std::string &name,
                         int index) const;

  /// Emit the getter for a accessible class member.
  void emitMemberGetter(std::ostream &os, const MemberTypeInfo *type,
                        const std::string &name) const;

  /// Emit setters/getters for each accessible class member.
  void emitSettersGetters(std::ostream &os) const;

  /// Emit the methods that print a textual summary.
  void emitPrettyPrinter(std::ostream &os) const;

  /// Emit the class definition.
  void emitClass(std::ostream &os) const;

  /// Emit the clone() method.
  void emitCloner(std::ostream &os) const;

  /// Emit the getOperandName() method.
  void emitGetOperandName(std::ostream &os) const;

  /// Emit the methods that go into the CPP file and implement the methods that
  /// were declared in the header file.
  void emitCppMethods(std::ostream &os) const;

  /// Adds a case to AutoIRGen for generating this Instr from a Node.
  void emitAutoIRGen(std::ostream &os) const;
};

class Builder {
  std::ofstream &headerStream;
  std::ofstream &cppStream;
  std::ofstream &defStream;
  std::ofstream &builderHeaderStream;
  std::ofstream &builderCppStream;
  std::ofstream &irGenStream;
  // First defined instruction.
  std::string firstInstr;
  // Last defined instruction.
  std::string lastInstr;

public:
  /// Create a new top-level builder that holds the output streams that
  /// point to the header file, cpp file and enum definition file, as well as
  /// the builder and IR Gen files.
  Builder(std::ofstream &H, std::ofstream &C, std::ofstream &D,
          std::ofstream &BH, std::ofstream &BC, std::ofstream &I)
      : headerStream(H), cppStream(C), defStream(D), builderHeaderStream(BH),
        builderCppStream(BC), irGenStream(I) {
    cppStream << "#include \"glow/IR/IR.h\"\n"
                 "#include \"glow/IR/Instrs.h\"\n"
                 "#include \"glow/Base/Type.h\"\n"
                 "#include \"glow/Support/Support.h\"\n\n"
                 "using namespace glow;\n";
    defStream
        << "#ifndef DEF_INSTR\n#error The macro DEF_INSTR was not declared.\n"
           "#endif\n"
           "#ifndef DEF_VALUE\n#error The macro DEF_VALUE was not declared.\n"
           "#endif\n"
           "#ifndef DEF_BACKEND_SPECIFIC_INSTR\n#error The macro "
           "DEF_BACKEND_SPECIFIC_INSTR "
           "was not declared.\n"
           "#endif\n"
           "#ifndef DEF_INSTR_RANGE\n"
           "#define DEF_INSTR_RANGE(ID, FIRST, LAST)\n"
           "#endif\n";

    builderCppStream << "#include \"glow/IR/IRBuilder.h\"\n"
                     << "#include \"glow/IR/IR.h\"\n"
                     << "#include \"glow/IR/Instrs.h\"\n\n"
                     << "using namespace glow;\n";
  }

  ~Builder() {
    defStream << "DEF_INSTR_RANGE(Instruction, " << firstInstr << "Inst"
              << ", " << lastInstr << "Inst"
              << ")\n";

    defStream << "#undef DEF_INSTR_RANGE\n"
                 "#undef DEF_INSTR\n"
                 "#undef DEF_BACKEND_SPECIFIC_INSTR\n"
                 "#undef DEF_VALUE";
  }

  /// Declare a new instruction and generate code for it.
  InstrBuilder newInstr(const std::string &name) {
    if (firstInstr.empty())
      firstInstr = name;
    lastInstr = name;
    const bool isBackendSpecific = false;
    return InstrBuilder(headerStream, cppStream, defStream, builderHeaderStream,
                        builderCppStream, irGenStream, name, isBackendSpecific);
  }

  /// Declare a new backend-specific instruction and generate code for it.
  InstrBuilder newBackendSpecificInstr(const std::string &name) {
    if (firstInstr.empty())
      firstInstr = name;
    lastInstr = name;
    const bool isBackendSpecific = true;
    return InstrBuilder(headerStream, cppStream, defStream, builderHeaderStream,
                        builderCppStream, irGenStream, name, isBackendSpecific);
  }

  /// Declare the instruction in the def file but don't generate code for it.
  void declareInstr(const std::string &name) {
    defStream << "DEF_INSTR(" << name << "Inst, " << glow::tolower(name)
              << ")\n";
  }

  /// Declare the value in the def file but don't generate code for it.
  void declareValue(const std::string &name) {
    defStream << "DEF_VALUE(" << name << ", " << glow::tolower(name) << ")\n";
  }

  /// Include backend-specific verification at the end of the auto-generated
  /// Instrs cpp file.
  void includeBackendSpecificVerification(const std::string &filename) {
    cppStream << "\n#include \"" << filename << "\"\n";
  }

  /// Include header into the auto-generated Instrs include file.
  void includeHeader(const std::string &filename) {
    headerStream << "\n#include \"" << filename << "\"\n";
  }
};

#endif // GLOW_TOOLS_NODEGEN_INSTRBUILDER_H
