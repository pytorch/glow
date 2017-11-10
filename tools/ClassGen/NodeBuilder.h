#ifndef GLOW_TOOLS_NODEGEN_NODEBUILDER_H
#define GLOW_TOOLS_NODEGEN_NODEBUILDER_H

#include "MemberType.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

class Builder;

class NodeBuilder {
  /// The type-initialization expression.
  std::string ty_;
  /// The node name.
  std::string name_;
  /// The node operands.
  std::vector<std::string> operands_;
  /// A list of node members. Format: (type, name).
  std::vector<std::pair<MemberType, std::string>> members_;
  /// The node enum cases.
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
  /// Documentation string printed with the class definition.
  std::string docstring_;

public:
  NodeBuilder(std::ofstream &H, std::ofstream &C, std::ofstream &D,
              const std::string &name)
      : name_(name), hStream(H), cStream(C), dStream(D) {}

  /// Add an operand to the node. The name should start with a capital letter.
  /// For example: "Input".
  NodeBuilder &addOperand(const std::string &op) {
    operands_.push_back(op);
    return *this;
  }
  /// Add a member to the node. Format: type, name.
  /// The name should start with a capital letter.
  /// For example: "Filter".
  NodeBuilder &addMember(MemberType type, const std::string &name) {
    members_.push_back({type, name});
    return *this;
  }

  /// Override the getter for variable \p var with the body \p body.
  NodeBuilder &overrideGetter(const std::string &var, const std::string &body) {
    assert(!overrideGetter_.count(var) && "Variable already overridden");
    overrideGetter_[var] = body;
    return *this;
  }

  /// Add an field to the enum. The enum name should start with a capital
  /// letter. For example: "External".
  NodeBuilder &addEnumCase(const std::string &op) {
    enum_.push_back(op);
    return *this;
  }
  /// Set the expression that initializes the type of the node.
  /// Example: 'LHS->getType()'.
  NodeBuilder &setType(const std::string &ty) {
    ty_ = ty;
    return *this;
  }
  /// Add a parameter to the constructor. For example "TypeRef" "outTy".
  NodeBuilder &addExtraParam(const std::string &type, const std::string &name) {
    extraParams_.push_back({type, name});
    return *this;
  }

  /// Set the documentation string. Each line will be prepended with "/// ".
  NodeBuilder &setDocstring(const std::string &docstring) {
    docstring_ = docstring;
    return *this;
  }

  /// Emits the methods that converts an enum case into a textual label.
  void emitEnumModePrinters(std::ostream &os) const;

  /// Emit the Node class constructor.
  void emitCtor(std::ostream &os) const;

  /// Emits the class members (the fields of the class).
  void emitClassMembers(std::ostream &os) const;

  /// Emit the getter for an operand.
  void emitOperandGetter(std::ostream &os,
                         const std::string &name) const;

  /// Emit the getter for a accessible class member.
  void emitMemberGetter(std::ostream &os,
                        MemberType type,
                        const std::string &name) const;

  /// Emit setters/getters for each accessible class member.
  void emitSettersGetters(std::ostream &os) const;

  /// Emit the methods that print a textual summary of the node.
  void emitPrettyPrinter(std::ostream &os) const;

  /// Emit the isEqual method that performs node comparisons.
  void emitEquator(std::ostream &os) const;

  /// Emit the 'visit' method that implements node visitors.
  void emitVisitor(std::ostream &os) const;

  /// Emit the class-level documentation string, if any.
  void emitDocstring(std::ostream &os) const;

  /// Emit the class definition for the node.
  void emitNodeClass(std::ostream &os) const;

  /// Emit the methods that go into the CPP file and implement the methods that
  /// were declared in the header file.
  void emitCppMethods(std::ostream &os) const;

  ~NodeBuilder();
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
    cStream << "#include \"glow/Graph/Nodes.h\"\n"
               "#include \"glow/Base/Type.h\"\n"
               "#include \"glow/Support/Support.h\"\n\n"
               "using namespace glow;\n";
    dStream << "#ifndef DEF_NODE\n#error The macro DEF_NODE was not declared.\n"
               "#endif\n";
  }

  ~Builder() { dStream << "#undef DEF_NODE"; }

  /// Declare a new node and generate code for it.
  NodeBuilder newNode(const std::string &name) {
    dStream << "DEF_NODE(" << name << "Node, " << name << ")\n";
    return NodeBuilder(hStream, cStream, dStream, name);
  }

  /// Declare the node in the def file but don't generate code for it.
  void declareNode(const std::string &name) {
    dStream << "DEF_NODE(" << name << "Node, " << name << ")\n";
  }
};

#endif // GLOW_TOOLS_NODEGEN_NODEBUILDER_H
