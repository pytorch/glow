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

  std::string getEnumModePrinters() {
    std::string sb;

    sb += "const char *" + name_ + "Node::getModeStr(" + name_ +
          "Node::Mode m) {\n";
    sb += "\tconst char *names[] = {";
    for (auto &e : enum_) {
      sb += "\"" + e + "\", ";
    }
    sb += "nullptr};\n";
    sb += "\treturn names[static_cast<int>(m)];\n";
    sb += "}\n";
    return sb;
  }

  std::string genCtor() {
    std::string sb;
    // Constructor parameter list:
    sb += "\t" + name_ + "Node(llvm::StringRef name";

    for (auto op : extraParams_) {
      sb += ", " + op.first + " " + op.second + " ";
    }

    if (enum_.size()) {
      sb += ", Mode mode";
    }

    for (auto op : operands_) {
      sb += ", Node *" + op;
    }

    for (auto op : members_) {
      sb += ", " + op.first + " " + op.second;
    }

    sb += "):\n\t Node(Kinded::Kind::" + name_ + "InstKind, " + ty_ + ", name)";

    // Print the initialization list:

    if (enum_.size()) {
      sb += ", mode_(mode)";
    }

    for (auto op : operands_) {
      sb += ", " + op + "_(" + op + ")";
    }

    for (auto op : members_) {
      sb += ", " + op.second + "_(" + op.second + ") ";
    }
    sb += " {}\n\n";
    return sb;
  }

  std::string genClassMembers() {
    std::string sb;
    if (enum_.size()) {
      sb += "\tpublic:\n";

      sb += "\tenum class Mode {\n";
      for (auto E : enum_) {
        sb += "\t  " + E + ",\n";
      }

      sb += "\t};\n";
      sb += "\tprivate:\n";
    }

    // Class members:
    if (enum_.size()) {
      sb += "\tMode mode_;\n";
    }
    for (auto op : operands_) {
      sb += "\tNodeOperand " + op + "_;\n";
    }
    for (auto op : members_) {
      sb += "\t" + op.first + " " + op.second + "_;\n";
    }

    sb += "\n";

    return sb;
  }

  std::string getSettersGetters() {
    std::string sb;

    // Print the getters/setters.
    for (auto op : operands_) {
      // Synthesize a user-defined getter.
      auto it = overrideGetter_.find(op);
      if (it != overrideGetter_.end()) {
        sb += "\t" + it->second + "\n";
        continue;
      }

      // Synthesize the general getter.
      sb += "\tNode *get" + op + "() const { return " + op + "_; }\n";
    }
    for (auto op : members_) {
      sb += "\t" + op.first + " get" + op.second + "() const { return " +
            op.second + "_; }\n";
    }
    sb += "\n";

    sb += "static bool classof(const Kinded *k) { return k->getKind() == "
          " Kinded::Kind::" +
          name_ + "InstKind; }\n";

    if (enum_.size()) {
      sb += "\tMode getMode() const { return mode_; }\n";
    }

    return sb;
  }

  std::string genPrettyPrinter() {
    std::string sb;
    sb += "std::string " + name_ + "Node::" + "getDebugDesc() const {\n";
    sb += "\t\tDescriptionBuilder db(getKindName());\n";
    sb += "\t\tdb.addParam(\"name\", getName())\n";

    if (enum_.size()) {
      sb += "\t\t.addParam(\"Mode\", getModeStr())\n";
    }

    for (auto op : operands_) {
      sb += "\t\t.addParam(\"" + op + "\", *get" + op + "()->getType())\n";
    }

    for (auto mem : members_) {
      sb += "\t\t.addParam(\"" + mem.second + "\", " + mem.second + "_)\n";
    }
    sb += "\t\t.addParam(\"users\", getNumUsers());\n";

    sb += "\t\treturn db;\n}\n";
    return sb;
  }

  std::string getEquator() {
    std::string sb;

    sb += "\tbool isEqual(const " + name_ + "Node &other) {\n";
    sb += "\treturn true";

    for (auto op : operands_) {
      sb += " &&\n\t " + op + "_ == other." + op + "_";
    }

    for (auto mem : members_) {
      sb += " &&\n\t " + mem.second + "_ == other." + mem.second + "_";
    }

    sb += ";\n }\n";
    return sb;
  }

  std::string getVisitor() {
    std::string sb;

    sb += "void " + name_ +
          "Node::" + "visit(Node *parent, NodeVisitor *visitor) {\n";
    sb += "\tif (!visitor->shouldVisit(parent, this)) { return; }\n";

    sb += "\tvisitor->pre(parent, this);\n";
    for (auto op : operands_) {
      sb += "\tget" + op + "()->visit(this, visitor);\n";
    }
    sb += "\tvisitor->post(parent, this);\n";
    sb += "}\n";
    return sb;
  }

  void done(std::ofstream &hFile, std::ofstream &cFile) {
    std::string hdr = "namespace glow {\n";
    hdr += "class " + name_ + "Node final : public Node {\n";

    hdr += genClassMembers();

    hdr += "\tpublic:\n";

    hdr += genCtor();

    hdr += getSettersGetters();
    hdr += getEquator();
    hdr += "\tstd::string getDebugDesc() const override;\n";
    hdr += "\tvoid visit(Node *parent, NodeVisitor *visitor) override;\n";
    if (enum_.size()) {
      hdr += "\tconst char *getModeStr() const { return getModeStr(mode_); }\n";
      hdr += "\tstatic const char *getModeStr(Mode m);\n";
    }

    hdr += "};\n\n";
    hdr += " } // namespace glow\n";
    hFile << hdr;

    std::string cpp;
    cpp += genPrettyPrinter();
    cpp += getVisitor();
    if (enum_.size()) {
      cpp += getEnumModePrinters();
    }
    cFile << cpp;
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
