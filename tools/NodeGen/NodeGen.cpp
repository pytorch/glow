#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

class NodeBuilder {
  /// The type-initialization expression.
  std::string ty_;
  /// The node name.
  std::string name_;
  /// The node operands.
  std::vector<std::string> operands_;
  /// A list of node members. Format: (type, name).
  std::vector<std::pair<std::string, std::string>> members_;
  /// The node enum cases.
  std::vector<std::string> enum_;
  /// A list of extra parameters that are passed to the constructor.
  std::vector<std::pair<std::string, std::string>> extraParams_;
  /// A list of getters to override. Format (variable name, alternative getter).
  std::unordered_map<std::string, std::string> overrideGetter_;

public:
  NodeBuilder(const std::string &name) : name_(name) {}

  /// Add an operand to the node. The name should start with a capital letter.
  /// For example: "Input".
  NodeBuilder &addOperand(const std::string &op) {
    operands_.push_back(op);
    return *this;
  }
  /// Add a member to the node. Format: type, name.
  /// The name should start with a capital letter.
  /// For example: "Filter".
  NodeBuilder &addMember(const std::string &type, const std::string &name) {
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

  /// Emits the methods that converts an enum case into a textual label.
  void emitEnumModePrinters(std::ostream &os) {
    os << "const char *" << name_ << "Node::getModeStr(" << name_
       << "Node::Mode m) {\n";
    os << "\tconst char *names[] = {";
    for (auto &e : enum_) {
      os << "\"" << e << "\", ";
    }
    os << "nullptr};\n";
    os << "\treturn names[static_cast<int>(m)];\n";
    os << "}\n";
  }

  /// Emit the Node class constructor.
  void emitCtor(std::ostream &os) {
    os << "\t" << name_ << "Node(llvm::StringRef name";

    // Constructor non-standard parameter list:
    for (auto op : extraParams_) {
      os << ", " << op.first << " " << op.second << " ";
    }

    // The enum 'Mode' parameter:
    if (enum_.size()) {
      os << ", Mode mode";
    }

    // The operands of the graph node:
    for (auto op : operands_) {
      os << ", Node *" << op;
    }

    // Extra class members:
    for (auto op : members_) {
      os << ", " << op.first << " " << op.second;
    }

    // Initialize the base clases:
    os << "):\n\t Node(Kinded::Kind::" << name_ << "InstKind, " << ty_
       << ", name)";

    // Print the initialization list:
    if (enum_.size()) {
      os << ", mode_(mode)";
    }

    // Initialize the operands:
    for (auto op : operands_) {
      os << ", " << op << "_(" << op << ")";
    }

    // Initialize the members:
    for (auto op : members_) {
      os << ", " << op.second << "_(" << op.second << ") ";
    }

    // Empty constructor body.
    os << " {}\n\n";
  }

  /// Emits the class members (the fields of the class).
  void emitClassMembers(std::ostream &os) {

    // Emit the type of the enum (which is public).
    if (enum_.size()) {
      os << "\tpublic:\n\tenum class Mode {\n";
      for (auto E : enum_) {
        os << "\t  " << E << ",\n";
      }
      os << "\t};\n";

      os << "\tprivate:\n";
    }

    // Emit class members:
    if (enum_.size()) {
      os << "\tMode mode_;\n";
    }
    for (auto op : operands_) {
      os << "\tNodeOperand " << op << "_;\n";
    }
    for (auto op : members_) {
      os << "\t" << op.first << " " << op.second << "_;\n";
    }
    os << "\n";
  }

  /// Emit stters/getters for each accessible class member.
  void emitSettersGetters(std::ostream &os) {
    // Print the getters/setters.
    for (auto op : operands_) {
      // Synthesize a user-defined operand getter.
      auto it = overrideGetter_.find(op);
      if (it != overrideGetter_.end()) {
        os << "\t" << it->second << "\n";
        continue;
      }

      // Synthesize the general getter.
      os << "\tNode *get" << op << "() const { return " << op << "_; }\n";
    }

    for (auto op : members_) {
      // Synthesize a user-defined member getter.
      auto it = overrideGetter_.find(op.second);
      if (it != overrideGetter_.end()) {
        os << "\t" << it->second << "\n";
        continue;
      }

      // Synthesize the general getter.
      os << "\t" << op.first + " get" << op.second << "() const { return "
         << op.second << "_; }\n";
    }
    // Synthesize the 'classof' method that enables the non-rtti polymorphism.
    os << "\nstatic bool classof(const Kinded *k) { return k->getKind() == "
          "Kinded::Kind::"
       << name_ << "InstKind; }\n";

    if (enum_.size()) {
      os << "\tMode getMode() const { return mode_; }\n";
    }
  }

  /// Emit the methods that print a textual summary of the node.
  void emitPrettyPrinter(std::ostream &os) {
    os << "std::string " << name_
       << "Node::getDebugDesc() const {\n\t\tDescriptionBuilder "
          "db(getKindName());\n\t\tdb.addParam(\"name\", getName())\n";

    if (enum_.size()) {
      os << "\t\t.addParam(\"Mode\", getModeStr())\n";
    }

    for (auto op : operands_) {
      os << "\t\t.addParam(\"" << op << "\", *get" << op << "()->getType())\n";
    }

    for (auto mem : members_) {
      os << "\t\t.addParam(\"" << mem.second << "\", get" << mem.second
         << "())\n";
    }
    os << "\t\t.addParam(\"users\", getNumUsers());\n\t\treturn db;\n}\n";
  }

  /// Emit the isEqual method that performs node comparisons.
  void emitEquator(std::ostream &os) {
    os << "\tbool isEqual(const " << name_ << "Node &other) {\n\treturn true";

    if (enum_.size()) {
      os << " &&\n\t getMode() == other.getMode()";
    }

    for (auto op : operands_) {
      os << " &&\n\t " << op << "_ == other." << op << "_";
    }

    for (auto mem : members_) {
      os << " &&\n\t " << mem.second << "_ == other." << mem.second << "_";
    }

    os << ";\n }\n";
  }

  /// Emit the 'visit' method that implements node visitors.
  void emitVisitor(std::ostream &os) {
    os << "void " << name_
       << "Node::visit(Node *parent, NodeVisitor *visitor) {\n\tif "
          "(!visitor->shouldVisit(parent, this)) { return; }\n";

    os << "\tvisitor->pre(parent, this);\n";
    for (auto op : operands_) {
      os << "\tget" << op << "()->visit(this, visitor);\n";
    }
    os << "\tvisitor->post(parent, this);\n";
    os << "}\n";
  }

  /// Emit the class definition for the node.
  void emitNodeClass(std::ostream &os) {
    os << "namespace glow {\nclass " << name_ << "Node final : public Node {\n";

    emitClassMembers(os);

    os << "\tpublic:\n";

    emitCtor(os);

    emitSettersGetters(os);
    emitEquator(os);

    os << "\tstd::string getDebugDesc() const override;\n";
    os << "\tvoid visit(Node *parent, NodeVisitor *visitor) override;\n";
    if (enum_.size()) {
      os << "\tconst char *getModeStr() const { return getModeStr(mode_); "
            "}\n\tstatic const char *getModeStr(Mode m);\n";
    }

    os << "};\n\n} // namespace glow\n";
  }

  /// Emit the methods that go into the CPP file and implement the methods that
  /// were declared in the header file.
  void emitCppMethods(std::ostream &os) {
    emitPrettyPrinter(os);
    emitVisitor(os);
    if (enum_.size()) {
      emitEnumModePrinters(os);
    }
  }

  void done(std::ofstream &hFile, std::ofstream &cFile) {
    emitNodeClass(hFile);
    emitCppMethods(cFile);
  }
};

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " output.h output.cpp"
              << "\n";
    return -1;
  }

  std::cout << "Writing node descriptors into " << argv[1] << " and " << argv[2]
            << "\n";

  std::ofstream hFile;
  std::ofstream cFile;
  hFile.open(argv[1]);
  cFile.open(argv[2]);

  cFile << "#include \"glow/Graph/Nodes.h\"\n"
           "#include \"glow/Base/Type.h\"\n"
           "#include \"glow/IR/Instrs.h\"\n"
           "#include \"glow/Support/Support.h\"\n\n"
           "using namespace glow;\n";

  NodeBuilder("Convolution")
      .addOperand("Input")
      .addOperand("Filter")
      .addOperand("Bias")
      .addMember("size_t", "Kernel")
      .addMember("size_t", "Stride")
      .addMember("size_t", "Pad")
      .addMember("size_t", "Depth")
      .addExtraParam("TypeRef", "outTy")
      .setType("outTy")
      .done(hFile, cFile);

  NodeBuilder("Pool")
      .addEnumCase("Max")
      .addEnumCase("Avg")
      .addOperand("Input")
      .addMember("size_t", "Kernel")
      .addMember("size_t", "Stride")
      .addMember("size_t", "Pad")
      .addExtraParam("TypeRef", "outTy")
      .setType("outTy")
      .done(hFile, cFile);

  NodeBuilder("FullyConnected")
      .addOperand("Input")
      .addOperand("Filter")
      .addOperand("Bias")
      .addMember("size_t", "Depth")
      .addExtraParam("TypeRef", "outTy")
      .setType("outTy")
      .done(hFile, cFile);

  NodeBuilder("BatchNormalization")
      .addOperand("Input")
      .addOperand("Scale")
      .addOperand("Bias")
      .addOperand("Mean")
      .addOperand("Var")
      .addMember("size_t", "ChannelIdx")
      .addMember("float", "Epsilon")
      .addMember("float", "Momentum")
      .setType("Input->getType()")
      .done(hFile, cFile);

  NodeBuilder("SoftMax")
      .addOperand("Input")
      .addOperand("Selected")
      .setType("Input->getType()")
      .done(hFile, cFile);

  NodeBuilder("Regression")
      .addOperand("Input")
      .addOperand("Expected")
      .setType("Input->getType()")
      .done(hFile, cFile);

  NodeBuilder("LocalResponseNormalization")
      .addOperand("Input")
      .addOperand("Scale")
      .addMember("size_t", "HalfWindowSize")
      .addMember("float", "Alpha")
      .addMember("float", "Beta")
      .addMember("float", "K")
      .setType("Input->getType()")
      .done(hFile, cFile);

  NodeBuilder("Arithmetic")
      .addEnumCase("Add")
      .addEnumCase("Mul")
      .addOperand("LHS")
      .addOperand("RHS")
      .setType("LHS->getType()")
      .done(hFile, cFile);

  NodeBuilder("Relu")
      .addOperand("Input")
      .setType("Input->getType()")
      .done(hFile, cFile);
  NodeBuilder("Sigmoid")
      .addOperand("Input")
      .setType("Input->getType()")
      .done(hFile, cFile);
  NodeBuilder("Tanh")
      .addOperand("Input")
      .setType("Input->getType()")
      .done(hFile, cFile);

  NodeBuilder("Reshape")
      .addOperand("Input")
      .addMember("std::vector<size_t>", "Dims")
      .addExtraParam("TypeRef", "outTy")
      .setType("outTy")
      .overrideGetter(
          "Dims", "llvm::ArrayRef<size_t> getDims() const { return Dims_; }")
      .done(hFile, cFile);

  NodeBuilder("Transpose")
      .addOperand("Input")
      .addMember("std::vector<unsigned>", "Shuffle")
      .addExtraParam("TypeRef", "outTy")
      .setType("outTy")
      .overrideGetter(
          "Shuffle",
          "llvm::ArrayRef<unsigned> getShuffle() const { return Shuffle_; }")
      .done(hFile, cFile);

  NodeBuilder("Save")
      .addOperand("Input")
      .addOperand("Output")
      .setType("Input->getType()")
      .overrideGetter("Output", "Variable *getOutput() const { return "
                                "cast<Variable>(Output_.get()); };")
      .done(hFile, cFile);

  hFile.close();
  cFile.close();
}
