#include <fstream>
#include <iostream>
#include <string>
#include <vector>

class NodeBuilder {
  /// The type-initialization expression.
  std::string ty_;
  /// The node name.
  std::string name_;
  /// The node operands.
  std::vector<std::string> operands_;
  /// The node members (type, name).
  std::vector<std::pair<std::string, std::string>> members_;
  /// The node enum cases.
  std::vector<std::string> enum_;
  /// A list of extra parameters;
  std::vector<std::pair<std::string, std::string>> extraParams_;

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
  void done(std::ofstream &OS) {
    std::string sb = "namespace glow {\n";
    sb += "class " + name_ + "Node final : public Node {\n";

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
    sb += "\tpublic:\n";

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

    // Print the getters/setters.
    for (auto op : operands_) {
      sb += "\tNode *get" + op + "() { return " + op + "_; }\n";
    }

    for (auto op : members_) {
      sb += "\t" + op.first + " get" + op.second + "() { return " + op.second +
            "_; }\n";
    }
    sb += "\n";

    if (enum_.size()) {
      sb += "\tMode getMode() const { return mode_; }\n";
    }

    sb += "\tstd::string getDebugDesc() const override;\n";
    sb += "\tvoid visit(Node *parent, NodeVisitor *visitor) override;\n";
    sb += "};\n\n";
    sb += " } // namespace glow\n";
    OS << sb;
  }
};

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " output.h"
              << "\n";
    return -1;
  }

  std::cout << "Writing into " << argv[1] << "\n";

  std::ofstream OS;
  OS.open(argv[1]);

  NodeBuilder("Convolution")
      .addOperand("Input")
      .addOperand("Filter")
      .addOperand("Bias")
      .addMember("size_t", "Kernel")
      .addMember("size_t", "Pad")
      .addMember("size_t", "Stride")
      .addMember("size_t", "Depth")
      .setType("Input->getType()")
      .done(OS);

  NodeBuilder("Pool")
      .addEnumCase("Max")
      .addEnumCase("Avg")
      .addOperand("Input")
      .addMember("size_t", "Kernel")
      .addMember("size_t", "Stride")
      .addMember("size_t", "Pad")
      .addExtraParam("TypeRef", "outTy")
      .setType("outTy")
      .done(OS);

  NodeBuilder("Relu").addOperand("Input").setType("Input->getType()").done(OS);
  NodeBuilder("Sigmoid")
      .addOperand("Input")
      .setType("Input->getType()")
      .done(OS);
  NodeBuilder("Tanh").addOperand("Input").setType("Input->getType()").done(OS);

  OS.close();
}
