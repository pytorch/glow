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

inline OperandKind negateOperandKind(OperandKind CC) {
  switch (CC) {
  case OperandKind::In:
    return OperandKind::Out;
  case OperandKind::Out:
    return OperandKind::In;
  case OperandKind::InOut:
    return OperandKind::InOut;
  }
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
  /// A list of extra parameters that are passed to the constructor.
  std::vector<std::pair<std::string, std::string>> extraParams_;
  /// A list of getters to override. Format (variable name, alternative getter).
  std::unordered_map<std::string, std::string> overrideGetter_;
  /// Stores the body of a new public method that will be added to the class.
  std::vector<std::string> extraMethods_;
  /// A list of operands that are declared as 'inplace' operands.
  std::vector<std::string> inplaceOperands_;

  /// Header file stream.
  std::ofstream &headerStream;
  /// CPP file stream.
  std::ofstream &cppStream;
  /// Def-enum file stream.
  std::ofstream &defStream;
  /// The IRBuilder stream.
  std::ofstream &builderStream;

  /// \returns the index of the operand with the name \p name. Aborts if no such
  /// name.
  unsigned getOperandIndexByName(llvm::StringRef name) const;

public:
  InstrBuilder(std::ofstream &H, std::ofstream &C, std::ofstream &D,
               std::ofstream &B, const std::string &name)
      : name_(name), headerStream(H), cppStream(C), defStream(D),
        builderStream(B) {
    defStream << "DEF_INSTR(" << name << "Inst, " << glow::tolower(name)
              << ")\n";
  }

  /// Add an operand to the instruction. The name should start with a capital
  /// letter. For example: "Input".
  InstrBuilder &addOperand(const std::string &op, OperandKind k) {
    operands_.push_back({op, k});
    return *this;
  }

  /// Adds two operands to the instruction: the operand and the gradient of the
  /// operand. This API is used for building instructions that perform the
  /// backward propagation pass.
  InstrBuilder &addOperandWithGrad(const std::string &op) {
    addOperand(op, OperandKind::InOut);
    addOperand(op + "Grad", OperandKind::InOut);
    return *this;
  }

  /// Add a member to the instruction. Format: type, name.
  /// The name should start with a capital letter.
  /// For example: "Filter".
  InstrBuilder &addMember(const MemberType type, const std::string &name) {
    members_.push_back({type, name});
    return *this;
  }

  /// Override the getter for variable \p var with the body \p body.
  InstrBuilder &overrideGetter(const std::string &var,
                               const std::string &body) {
    assert(!overrideGetter_.count(var) && "Variable already overridden");
    overrideGetter_[var] = body;
    return *this;
  }

  /// Adds the body of a new public method to the class.
  InstrBuilder &addExtraMethod(const std::string &body) {
    extraMethods_.push_back(body);
    return *this;
  }

  /// Set the expression that initializes the type of the instruction.
  /// Example: 'LHS->getType()'.
  InstrBuilder &setType(const std::string &ty) {
    ty_ = ty;
    return *this;
  }
  /// Add a parameter to the constructor. For example "TypeRef" "outTy".
  InstrBuilder &addExtraParam(const std::string &type,
                              const std::string &name) {
    extraParams_.push_back({type, name});
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

  /// Emit the class constructor.
  void emitCtor(std::ostream &os) const;

  /// Emit the methods that make the IRBuilder.
  void emitIRBuilderMethods(std::ostream &os) const;

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

  // Constructs a new gradient instruction that is based on the current
  // instruction that we are building.
  void addGradientInstr(llvm::ArrayRef<llvm::StringRef> originalFields,
                        llvm::ArrayRef<llvm::StringRef> gradFields);

  ~InstrBuilder();
};

class Builder {
  std::ofstream &headerStream;
  std::ofstream &cppStream;
  std::ofstream &defStream;
  std::ofstream &builderStream;

public:
  /// Create a new top-level builder that holds the three output streams that
  /// point to the header file, cpp file and enum definition file.
  Builder(std::ofstream &H, std::ofstream &C, std::ofstream &D,
          std::ofstream &B)
      : headerStream(H), cppStream(C), defStream(D), builderStream(B) {
    cppStream << "#include \"glow/IR/IR.h\"\n"
                 "#include \"glow/IR/Instrs.h\"\n"
                 "#include \"glow/Base/Type.h\"\n"
                 "#include \"glow/Support/Support.h\"\n\n"
                 "using namespace glow;\n";
    defStream
        << "#ifndef DEF_INSTR\n#error The macro DEF_INSTR was not declared.\n"
           "#endif\n#ifndef DEF_VALUE\n#error The macro DEF_VALUE was not "
           "declared.\n"
           "#endif\n";
  }

  ~Builder() { defStream << "#undef DEF_INSTR\n#undef DEF_VALUE"; }

  /// Declare a new instruction and generate code for it.
  InstrBuilder newInstr(const std::string &name) {
    return InstrBuilder(headerStream, cppStream, defStream, builderStream,
                        name);
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
};

#endif // GLOW_TOOLS_NODEGEN_INSTRBUILDER_H
