#ifndef GLOW_GRAPH_NODES_H
#define GLOW_GRAPH_NODES_H

#include "glow/Base/Tensor.h"
#include "glow/Graph/Node.h"
#include "glow/Support/Casting.h"

#include "AutoGenNodes.h"

namespace glow {

class Variable final : public Node {
public:
  enum class InitKind {
    Extern,    // No initialization.
    Broadcast, // Broadcast a single value to all elements.
    Xavier,    // Init the tensor with random values using the Xavier method.
  };

private:
  /// The value to use during initialization. This can be the value to splat or
  /// a parameter to specify the range of the random values.
  float val_;
  /// The initialization mode.
  InitKind initKind_;
  /// The tensor payload that the variable holds.
  Tensor payload_;

  /// Initialize the content of the tensor. If an 'extern' is set then the user
  /// of the graph is responsible for updating the tensor externally.
  void initPayload();

public:
  Variable(llvm::StringRef name, TypeRef Ty, InitKind initKind, float val)
      : Node(Kinded::Kind::WeightVarKind, Ty, name), val_(val),
        initKind_(initKind) {
    initPayload();
  }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::WeightVarKind;
  }

  const char *getInitKindStr() const;

  static const char *getInitKindStr(InitKind kind);

  /// \returns the original initialization mode of the variable.
  InitKind getInitKind() const { return initKind_; }

  Tensor &getPayload() { return payload_; }

  template <class ElemTy> Handle<ElemTy> getHandle() {
    return getPayload().getHandle<ElemTy>();
  }

  void copyFrom(Tensor *t) { payload_.copyFrom(t); }

  std::string getDebugDesc() const override;

  void visit(Node *parent, NodeVisitor *visitor) override;
};

/// Calculate the size of the output tensor based on the convolution
/// parameters.
inline std::pair<size_t, size_t> calculateConvOutputDims(size_t sx, size_t sy,
                                                         size_t pad,
                                                         size_t filterSize,
                                                         size_t stride) {
  size_t outsx = ((sx + pad * 2 - filterSize) / stride + 1);
  size_t outsy = ((sy + pad * 2 - filterSize) / stride + 1);
  return {outsx, outsy};
}

class TransposeNode final : public Node {
  NodeOperand in_;
  std::vector<unsigned> shuffle_;

public:
  TransposeNode(Node *in, TypeRef outTy, llvm::StringRef name,
                llvm::ArrayRef<unsigned> shuffle)
      : Node(Kinded::Kind::TransposeInstKind, outTy, name), in_(in),
        shuffle_(shuffle.begin(), shuffle.end()) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::TransposeInstKind;
  }

  Node *getInput() const { return in_; }
  llvm::ArrayRef<unsigned> getShuffle() const { return shuffle_; }

  std::string getDebugDesc() const override;
  void visit(Node *parent, NodeVisitor *visitor) override;
};

class ReshapeNode final : public Node {
  NodeOperand in_;
  std::vector<size_t> dims_;

public:
  ReshapeNode(Node *in, llvm::StringRef name, TypeRef TR)
      : Node(Kinded::Kind::ReshapeInstKind, TR, name), in_(in),
        dims_(TR->dims().begin(), TR->dims().end()) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ReshapeInstKind;
  }

  Node *getInput() const { return in_; }
  llvm::ArrayRef<size_t> getDims() { return dims_; }

  std::string getDebugDesc() const override;
  void visit(Node *parent, NodeVisitor *visitor) override;
};

class ConcatNode final : public Node {
  /// The input nodes to concat.
  std::vector<NodeOperand> in_;
  /// We concat the tensors along this dimension.
  size_t dim_;

public:
  ConcatNode(llvm::ArrayRef<Node *> src, TypeRef outTy, llvm::StringRef name,
             size_t dim)
      : Node(Kinded::Kind::ConcatInstKind, outTy, name),
        in_(src.begin(), src.end()), dim_(dim) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ConcatInstKind;
  }
  llvm::ArrayRef<NodeOperand> getInputs() const { return in_; }
  size_t getDim() const { return dim_; }

  std::string getDebugDesc() const override;
  void visit(Node *parent, NodeVisitor *visitor) override;
};

class ArithmeticNode final : public Node {
public:
  /// Specifies the kind of pooling done by the operator.
  enum class OpKind {
    Add,
    Mul,
  };

private:
  NodeOperand LHS_;
  NodeOperand RHS_;
  OpKind kind_;
  const char *getKindStr() const;

public:
  ArithmeticNode(llvm::StringRef name, Node *LHS, Node *RHS, OpKind kind)
      : Node(Kinded::Kind::ArithmeticInstKind, LHS->getType(), name), LHS_(LHS),
        RHS_(RHS), kind_(kind) {}
  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ArithmeticInstKind;
  }
  Node *getLHS() const { return LHS_; }
  Node *getRHS() const { return RHS_; }
  OpKind getKind() const { return kind_; }

  std::string getDebugDesc() const override;
  void visit(Node *parent, NodeVisitor *visitor) override;
};

class LocalResponseNormalizationNode final : public Node {
  NodeOperand in_;
  NodeOperand scale_;
  /// The number of neighbouring channels on each side to sum over
  size_t halfWindowSize_;
  /// The scaling parameter
  float alpha_;
  /// The exponent parameter
  float beta_;
  /// The offset parameter
  float k_;

public:
  LocalResponseNormalizationNode(Node *in, llvm::StringRef name, Node *scale,
                                 size_t halfWindowSize, float alpha, float beta,
                                 float k)
      : Node(Kinded::Kind::LocalResponseNormalizationInstKind, in->getType(),
             name),
        in_(in), scale_(scale), halfWindowSize_(halfWindowSize), alpha_(alpha),
        beta_(beta), k_(k) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::LocalResponseNormalizationInstKind;
  }
  Node *getInput() const { return in_; }
  Node *getScale() const { return scale_; }

  size_t gethalfWindowSize() const { return halfWindowSize_; }
  float getAlpha() const { return alpha_; }
  float getBeta() const { return beta_; }
  float getK() const { return k_; }

  std::string getDebugDesc() const override;
  void visit(Node *parent, NodeVisitor *visitor) override;
};

class SaveNode final : public Node {
  NodeOperand in_;
  NodeOperand out_;

public:
  SaveNode(llvm::StringRef name, Node *input, Variable *output)
      : Node(Kinded::Kind::SaveInstKind, input->getType(), name), in_(input),
        out_(output) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SaveInstKind;
  }
  Node *getInput() const { return in_; }
  Variable *getOutput() const { return cast<Variable>(out_.get()); }

  std::string getDebugDesc() const override;
  void visit(Node *parent, NodeVisitor *visitor) override;
};

} // namespace glow

#endif // GLOW_GRAPH_NODES_H
