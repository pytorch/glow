#ifndef GLOW_TOOLS_NODEGEN_INSTRBUILDER_H
#define GLOW_TOOLS_NODEGEN_INSTRBUILDER_H

#include "MemberType.h"
#include "glow/Support/Support.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

class Builder;

enum class OperandKind : unsigned char {
  In,
  Out,
  InOut,
};

enum class VerifyKind : unsigned char {
  SameShape,
  SameType,
  SameElementType,
};

inline OperandKind negateOperandKind(OperandKind CC) {
  switch (CC) {
  case OperandKind::In:
    return OperandKind::Out;
  case OperandKind::Out:
    return OperandKind::In;
  case OperandKind::InOut:
    return OperandKind::InOut;
  }
  llvm_unreachable("Invalid operand kind.");
}

inline const char *getOperandKindStr(OperandKind CC) {
  const char *names[] = {"In", "Out", "InOut", nullptr};
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
  std::vector<std::pair<MemberType, std::string>> members_;
  /// Stores the decl and body of a new public method that will be added to the
  /// class.
  std::vector<std::pair<std::string, std::string>> extraMethods_;
  /// A list of operands that are declared as 'inplace' operands.
  std::vector<std::string> inplaceOperands_;
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
  InstrBuilder &addMember(const MemberType type, const std::string &name) {
    members_.push_back({type, name});
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
    assert(!inplaceOperands_.size() && "Initializing field twice");
    inplaceOperands_.insert(inplaceOperands_.begin(), lst.begin(), lst.end());
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
                           llvm::ArrayRef<llvm::StringRef> operands) {
    assert(operands.size() > 1 && "Must list 2 or more operands.");
    auto newPair = std::make_pair(verif, std::vector<std::string>());
    newPair.second.insert(newPair.second.begin(), operands.begin(),
                          operands.end());
    autoVerificationPairs_.push_back(newPair);
    return *this;
  }

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

  /// Emit the getter for an operand.
  void emitOperandGetter(std::ostream &os, const std::string &name,
                         int index) const;

  /// Emit the getter for a accessible class member.
  void emitMemberGetter(std::ostream &os, MemberType type,
                        const std::string &name) const;

  /// Emit setters/getters for each accessible class member.
  void emitSettersGetters(std::ostream &os) const;

  /// Emit the methods that print a textual summary.
  void emitPrettyPrinter(std::ostream &os) const;

  /// Emit the class definition.
  void emitClass(std::ostream &os) const;

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
};

#endif // GLOW_TOOLS_NODEGEN_INSTRBUILDER_H
