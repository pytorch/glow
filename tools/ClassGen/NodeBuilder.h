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
#ifndef GLOW_TOOLS_NODEGEN_NODEBUILDER_H
#define GLOW_TOOLS_NODEGEN_NODEBUILDER_H

#include "MemberType.h"
#include "glow/Support/Support.h"

#include "llvm/ADT/ArrayRef.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

class Builder;
class NodeBuilder;

class EnumBuilder {
  struct EnumCase {
    // The name of the enum case (i.e. <enum type>::<enum case name>).
    std::string name;
    // The value of the enum case.
    int value;
    // The docstring for the enum case.
    std::string doc;
    // Whether or not the value for this case was explicitly defined.
    bool definedWithValue;
  };
  ;

  /// The name of the enum.
  std::string name_;
  /// The fully qualified name of the enum (i.e. with all enclosing namespaces).
  std::string fullyQualifiedName_;
  /// The containing namespaces for the enum.
  std::vector<std::string> namespaces_;
  /// The enum cases.
  std::vector<EnumCase> enumCases_;
  /// CPP file stream.
  std::ofstream &cStream;
  /// Specifies if this enum is backend specific. If true, a definition
  /// will not be generated.
  bool isBackendSpecific_{false};
  /// Documentation string printed with the class definition.
  std::string docstring_;
  /// An integer value that is guaranteed to be unused by the enum at all times.
  int unusedValue_{0};

public:
  /// Constructor.
  EnumBuilder(std::ofstream &C, const std::string &name,
              bool isBackendSpecific);

  /// Delete copy constructor to force moves and ensure one destructor call.
  EnumBuilder(const EnumBuilder &src) = delete;

  /// Set the documentation string. Each line will be prepended with "/// ".
  EnumBuilder &setDocstring(const std::string &docstring) {
    docstring_ = docstring;
    return *this;
  }

  /// Add a new enum case to the enumeration.
  EnumBuilder &addEnumCase(const std::string &name,
                           const std::string &doc = "");
  EnumBuilder &addEnumCaseWithValue(const std::string &name, const int value,
                                    const std::string &doc = "");

  /// Get a MemberTypeInfo object describing the enum being built by this
  /// builder instance.
  MemberTypeInfo getMemberTypeInfo() const;

  /// Destructor.
  ~EnumBuilder();

private:
  /// Emit the opening of the namespace that contains the enum.
  void emitNamespaceOpen(std::ostream &os) const;

  /// Emit the closing of the namespace that contains the enum.
  void emitNamespaceClose(std::ostream &os) const;

  /// Emit the class-level documentation string, if any.
  void emitDocstring(std::ostream &os) const;

  /// Emit the forward declaration of the enum.
  void emitForwardDecl(std::ostream &os) const;

  /// Return the fully qualified name of the enum.
  std::string getFullyQualifiedName() const { return fullyQualifiedName_; }

  /// Emit the cases of the enum.
  void emitEnumCases(std::ostream &os) const;

  /// Emit the definition of the enum.
  void emitEnumDefinition(std::ostream &os) const;
};

class NodeBuilder {
  /// The node name.
  std::string name_;
  /// The node operands.
  std::vector<std::string> nodeInputs_;
  /// A list of node inputs that are overwritten, i.e. are @out parameters
  /// essentially.
  std::vector<unsigned> nodeOverwrittenInputs_;
  /// Initializes the result types of the nodes. The first argument is the c++
  /// expression that computes the type. For example "X->getType()". The second
  /// argument is the name of the return type. Format: (type, name)
  std::vector<std::pair<std::string, std::string>> nodeOutputs_;
  /// A list of node members. Format: (type, name).
  std::vector<std::pair<const MemberTypeInfo *, std::string>> members_;
  /// The node enum cases.
  std::vector<std::string> enum_;
  /// A list of extra parameter that are declared in the node constructor. The
  /// arguments are used when creating the result types of the node.
  std::vector<std::string> ctorTypeParams_;
  /// Stores the decl and body of a new public method that will be added to the
  /// class.
  std::vector<std::pair<std::string, std::string>> extraMethods_;
  /// Header file stream.
  std::ofstream &hStream;
  /// CPP file stream.
  std::ofstream &cStream;
  /// Def file stream.
  std::ofstream &dStream;
  /// Documentation string printed with the class definition.
  std::string docstring_;
  /// Whether node has side effects. By default there are no side effects.
  bool hasSideEffects_{false};
  /// Specifies if this Node is backend specific.
  bool isBackendSpecific_{false};

public:
  NodeBuilder(std::ofstream &H, std::ofstream &C, std::ofstream &D,
              const std::string &name, bool isBackendSpecific)
      : name_(name), hStream(H), cStream(C), dStream(D),
        isBackendSpecific_(isBackendSpecific) {
    dStream << "DEF_NODE(" << name << "Node, " << name << ")\n";
  }

  /// Add an operand to the node. The name should start with a capital letter.
  /// For example: "Input".
  NodeBuilder &addInput(const std::string &op) {
    nodeInputs_.push_back(op);
    return *this;
  }
  /// Add a member to the node. Format: type, name.
  /// The name should start with a capital letter.
  /// For example: "Filter".
  NodeBuilder &addMember(MemberType type, const std::string &name);
  NodeBuilder &addMember(const MemberTypeInfo *typeInfo,
                         const std::string &name) {
    members_.push_back({typeInfo, name});
    return *this;
  }

  /// Adds the body of a new public method to the class. \p decl is the
  /// decleration that goes in the header file. \p body is the implementation
  /// that goes in the cpp file.
  NodeBuilder &addExtraMethod(const std::string &decl,
                              const std::string &body) {
    extraMethods_.push_back(std::make_pair(decl, body));
    return *this;
  }

  /// Add an field to the enum. The enum name should start with a capital
  /// letter. For example: "External".
  NodeBuilder &addEnumCase(const std::string &op) {
    enum_.push_back(op);
    return *this;
  }
  /// Set the expression that initializes a new return type for the node.
  /// Example: 'LHS->getType()', "Result".
  NodeBuilder &addResult(const std::string &ty,
                         const std::string &name = "Result") {
    nodeOutputs_.push_back({ty, name});
    return *this;
  }
  /// Add a TypeRef parameter to the constructor and use this argument to add
  /// a result type to the node.
  NodeBuilder &addResultFromCtorArg(const std::string &name = "Result") {
    ctorTypeParams_.push_back(name);
    nodeOutputs_.push_back({name, name});
    return *this;
  }

  /// Set the documentation string. Each line will be prepended with "/// ".
  NodeBuilder &setDocstring(const std::string &docstring) {
    docstring_ = docstring;
    return *this;
  }

  /// Set whether node has side effects.
  NodeBuilder &setHasSideEffects(bool hasSideEffects) {
    hasSideEffects_ = hasSideEffects;
    return *this;
  }

  NodeBuilder &addOverwrittenInput(const std::string &name) {
    // Find the index of the overwritten input.
    for (unsigned idx = 0, e = nodeInputs_.size(); idx < e; ++idx) {
      if (nodeInputs_[idx] == name) {
        nodeOverwrittenInputs_.push_back(idx);
        return *this;
      }
    }
    llvm_unreachable("Cannot register an overwritten input that is not a known "
                     "input of a node");
  }

  /// Constructs a new gradient node that is based on the current node that we
  /// are building. The gradient node will produce one gradient output for each
  /// input. The rule is that each output becomes an input (named "Output", to
  /// preserve the original name) and each input becomes a gradient output with
  /// the same name.
  NodeBuilder &addGradient();

  ~NodeBuilder();

private:
  /// Emit required forward declarations for node members.
  void emitMemberForwardDecls(std::ostream &os) const;

  /// Emits the methods that converts an enum case into a textual label.
  void emitEnumModePrinters(std::ostream &os) const;

  /// Emit the Node class constructor.
  void emitCtor(std::ostream &os) const;

  /// Emits the class members (the fields of the class).
  void emitClassMembers(std::ostream &os) const;

  /// Emit the getter for a accessible class member.
  void emitMemberGetter(std::ostream &os, const MemberTypeInfo *type,
                        const std::string &name) const;

  /// Emit setters/getters for each accessible class member.
  void emitSettersGetters(std::ostream &os) const;

  /// Emit getters for input/output names and input nodes.
  void emitEdges(std::ostream &os) const;

  /// Emit the methods that print a textual summary of the node.
  void emitPrettyPrinter(std::ostream &os) const;

  /// Emit the isEqual method that performs node comparisons.
  void emitEquator(std::ostream &os) const;

  /// Emit the clone() method copies the node.
  void emitCloner(std::ostream &os) const;

  /// Emit the getHash method that computes a hash of a node.
  void emitHasher(std::ostream &os) const;

  /// Emit the 'visit' method that implements node visitors.
  void emitVisitor(std::ostream &os) const;

  /// Emit the class-level documentation string, if any.
  void emitDocstring(std::ostream &os) const;

  /// Emit enums for each of the node's inputs and results indices.
  void emitIndicesEnum(std::ostream &os) const;

  /// Emit the class definition for the node.
  void emitNodeClass(std::ostream &os) const;

  /// Emit the methods that go into the CPP file and implement the methods that
  /// were declared in the header file.
  void emitCppMethods(std::ostream &os) const;
};

class Builder {
  std::ofstream &hStream;
  std::ofstream &cStream;
  std::ofstream &dStream;
  std::vector<std::shared_ptr<EnumBuilder>> enumBuilders_;

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
    const bool isBackendSpecific = false;
    return NodeBuilder(hStream, cStream, dStream, name, isBackendSpecific);
  }

  /// Declare a new backend specific node and generate code for it.
  NodeBuilder newBackendSpecificNode(const std::string &name) {
    const bool isBackendSpecific = true;
    return NodeBuilder(hStream, cStream, dStream, name, isBackendSpecific);
  }

  /// Declare a new enum and generate code for it.
  EnumBuilder &newEnum(const std::string &name) {
    const bool isBackendSpecific = false;
    auto eb = std::make_shared<EnumBuilder>(cStream, name, isBackendSpecific);
    enumBuilders_.emplace_back(eb);
    return *eb;
  }

  /// Declare a new backend specific enum. This will NOT generate a definition.
  EnumBuilder &newBackendSpecificEnum(const std::string &name) {
    const bool isBackendSpecific = true;
    auto eb = std::make_shared<EnumBuilder>(cStream, name, isBackendSpecific);
    enumBuilders_.emplace_back(eb);
    return *eb;
  }

  /// Declare the node in the def file but don't generate code for it.
  void declareNode(const std::string &name) {
    dStream << "DEF_NODE(" << name << ", " << name << ")\n";
  }

  /// Include backend-specific verification at the end of the auto-generated
  /// Nodes cpp file.
  void includeBackendSpecificVerification(const std::string &filename) {
    cStream << "\n#include \"" << filename << "\"\n";
  }
};

#endif // GLOW_TOOLS_NODEGEN_NODEBUILDER_H
