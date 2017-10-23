#ifndef GLOW_TOOLS_NODEGEN_INSTRBUILDER_H
#define GLOW_TOOLS_NODEGEN_INSTRBUILDER_H

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

inline const char *getOperandKindStr(OperandKind CC) {
  const char *names[] = {"In", "Out", "InOut", nullptr};
  return names[(int)CC];
}

class InstrBuilder {
  /// The type-initialization expression.
  std::string ty_;
  /// The instruction name.
  std::string name_;
  /// The instruction operands.
  std::vector<std::pair<std::string, OperandKind>> operands_;
  /// A list of instruction members. Format: (type, name).
  std::vector<std::pair<std::string, std::string>> members_;
  /// The instruction enum cases.
  std::vector<std::string> enum_;
  /// A list of extra parameters that are passed to the constructor.
  std::vector<std::pair<std::string, std::string>> extraParams_;
  /// A list of getters to override. Format (variable name, alternative getter).
  std::unordered_map<std::string, std::string> overrideGetter_;
  /// Header file stream.
  std::ofstream &hStream;
  /// CPP file stream.
  std::ofstream &cStream;
  /// Def file stream.
  std::ofstream &dStream;

public:
  InstrBuilder(std::ofstream &H, std::ofstream &C, std::ofstream &D,
               const std::string &name)
      : name_(name), hStream(H), cStream(C), dStream(D) {}

  /// Add an operand to the instruction. The name should start with a capital
  /// letter. For example: "Input".
  InstrBuilder &addOperand(const std::string &op, OperandKind k) {
    operands_.push_back({op, k});
    return *this;
  }
  /// Add a member to the instruction. Format: type, name.
  /// The name should start with a capital letter.
  /// For example: "Filter".
  InstrBuilder &addMember(const std::string &type, const std::string &name) {
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

  /// Add an field to the enum. The enum name should start with a capital
  /// letter. For example: "External".
  InstrBuilder &addEnumCase(const std::string &op) {
    enum_.push_back(op);
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

  /// Emits the methods that converts an enum case into a textual label.
  void emitEnumModePrinters(std::ostream &os) const;

  /// Emit the class constructor.
  void emitCtor(std::ostream &os) const;

  /// Emits the class members (the fields of the class).
  void emitClassMembers(std::ostream &os) const;

  /// Emit stters/getters for each accessible class member.
  void emitSettersGetters(std::ostream &os) const;

  /// Emit the methods that print a textual summary.
  void emitPrettyPrinter(std::ostream &os) const;

  /// Emit the class definition.
  void emitClass(std::ostream &os) const;

  /// Emit the methods that go into the CPP file and implement the methods that
  /// were declared in the header file.
  void emitCppMethods(std::ostream &os) const;

  ~InstrBuilder();
};

class Builder {
  std::ofstream &hStream;
  std::ofstream &cStream;
  std::ofstream &dStream;

public:
  /// Create a new top-level builder that holds the three output streams that
  /// point to the header file, cpp file and enum definition file.
  Builder(std::ofstream &H, std::ofstream &C, std::ofstream &D)
      : hStream(H), cStream(C), dStream(D) {
    cStream << "#include \"glow/IR/IR.h\"\n"
               "#include \"glow/Base/Type.h\"\n"
               "#include \"glow/Support/Support.h\"\n\n"
               "using namespace glow;\n";
    dStream
        << "#ifndef DEF_INSTR\n#error The macro DEF_INSTR was not declared.\n"
           "#endif\n";
  }

  ~Builder() { dStream << "#undef DEF_INSTR"; }

  /// Declare a new instruction and generate code for it.
  InstrBuilder newInstr(const std::string &name) {
    dStream << "DEF_INSTR(" << name << "Inst, " << name << ")\n";
    return InstrBuilder(hStream, cStream, dStream, name);
  }

  /// Declare the instruction in the def file but don't generate code for it.
  void declareInstr(const std::string &name) {
    dStream << "DEF_INSTR(" << name << "Inst, " << name << ")\n";
  }
};

#endif // GLOW_TOOLS_NODEGEN_INSTRBUILDER_H
