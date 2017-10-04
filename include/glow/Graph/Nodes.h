#ifndef GLOW_GRAPH_NODES_H
#define GLOW_GRAPH_NODES_H

#include "glow/Graph/Node.h"
#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"

namespace glow {

class Variable final : public Node {
  /// The value to use during initialization. This can be the value to splat or
  /// a parameter to specify the range of the random values.
  float val_;
  /// The initialization mode.
  WeightVar::InitKind initKind_;

public:
  Variable(llvm::StringRef name, TypeRef Ty, WeightVar::InitKind initKind,
           float val)
      : Node(Kinded::Kind::WeightVarKind, Ty, name), val_(val),
        initKind_(initKind) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::WeightVarKind;
  }

  WeightVar::InitKind getInitKind() const { return initKind_; }
  float getVal() const { return val_; }

  std::string getDebugDesc() const override;
  void visit(Node *parent, NodeVisitor *visitor) override;
};

class ConvolutionNode final : public Node {
  Node *in_;
  Node *filter_;
  Node *bias_;

  size_t kernel_;
  size_t stride_;
  size_t pad_;
  size_t depth_;

public:
  ConvolutionNode(Node *in, TypeRef outTy, llvm::StringRef name, Node *filter,
                  Node *bias, size_t kernel, size_t stride, size_t pad,
                  size_t depth)
      : Node(Kinded::Kind::ConvolutionInstKind, outTy, name), in_(in),
        filter_(filter), bias_(bias), kernel_(kernel), stride_(stride),
        pad_(pad), depth_(depth) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ConvolutionInstKind;
  }

  bool mayShareBuffers() const { return false; }

  Node *getInput() const { return in_; }
  Node *getFilter() const { return filter_; }
  Node *getBias() const { return bias_; }

  size_t getKernel() const { return kernel_; }
  size_t getStride() const { return stride_; }
  size_t getPad() const { return pad_; }
  size_t getDepth() const { return depth_; }

  std::string getDebugDesc() const override;
  void visit(Node *parent, NodeVisitor *visitor) override;
};

class PoolNode final : public Node {
  Node *in_;
  size_t kernel_;
  size_t stride_;
  size_t pad_;
  PoolInst::OpKind kind_;

public:
  PoolNode(Node *in, TypeRef outTy, llvm::StringRef name, PoolInst::OpKind kind,
           size_t kernel, size_t stride, size_t pad)
      : Node(Kinded::Kind::PoolInstKind, outTy, name), in_(in), kernel_(kernel),
        stride_(stride), pad_(pad), kind_(kind) {}
  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::PoolInstKind;
  }

  Node *getInput() const { return in_; }

  size_t getKernel() const { return kernel_; }
  size_t getStride() const { return stride_; }
  size_t getPad() const { return pad_; }
  PoolInst::OpKind getKind() const { return kind_; }

  std::string getDebugDesc() const override;
  void visit(Node *parent, NodeVisitor *visitor) override;
};

class FullyConnectedNode final : public Node {
  Node *in_;
  Node *filter_;
  Node *bias_;
  size_t depth_;

public:
  FullyConnectedNode(Node *in, TypeRef outTy, llvm::StringRef name,
                     Node *filter, Node *bias, size_t depth)
      : Node(Kinded::Kind::FullyConnectedInstKind, outTy, name), in_(in),
        filter_(filter), bias_(bias), depth_(depth) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::FullyConnectedInstKind;
  }

  bool mayShareBuffers() const { return false; }
  std::string getExtraDesc() const;
  Node *getInput() const { return in_; }
  Node *getFilter() const { return filter_; }
  Node *getBias() const { return bias_; }
  size_t getDepth() const { return depth_; }

  std::string getDebugDesc() const override;
  void visit(Node *parent, NodeVisitor *visitor) override;
};

class ReluNode final : public Node {
  Node *in_;

public:
  ReluNode(Node *in, llvm::StringRef name)
      : Node(Kinded::Kind::ReluInstKind, in->getType(), name), in_(in) {}
  Node *getInput() { return in_; }

  std::string getDebugDesc() const override;
  void visit(Node *parent, NodeVisitor *visitor) override;
};

class SigmoidNode final : public Node {
  Node *in_;

public:
  SigmoidNode(Node *in, llvm::StringRef name)
      : Node(Kinded::Kind::SigmoidInstKind, in->getType(), name), in_(in) {}

  Node *getInput() { return in_; }

  std::string getDebugDesc() const override;
  void visit(Node *parent, NodeVisitor *visitor) override;
};

class TanhNode final : public Node {
  Node *in_;

public:
  TanhNode(Node *in, llvm::StringRef name)
      : Node(Kinded::Kind::TanhInstKind, in->getType(), name), in_(in) {}

  Node *getInput() { return in_; }

  std::string getDebugDesc() const override;
  void visit(Node *parent, NodeVisitor *visitor) override;
};

class SoftMaxNode final : public Node {
  Node *in_;
  Node *selected_;

public:
  SoftMaxNode(Node *in, llvm::StringRef name, Node *selected)
      : Node(Kinded::Kind::SoftMaxInstKind, in->getType(), name), in_(in),
        selected_(selected) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SoftMaxInstKind;
  }
  Node *getInput() const { return in_; }
  Node *getSelected() const { return selected_; }

  std::string getDebugDesc() const override;
  void visit(Node *parent, NodeVisitor *visitor) override;
};

class RegressionNode final : public Node {
  Node *in_;
  Node *expected_;

public:
  RegressionNode(Node *in, llvm::StringRef name, Node *expected)
      : Node(Kinded::Kind::RegressionInstKind, in->getType(), name), in_(in),
        expected_(expected) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::RegressionInstKind;
  }
  Node *getInput() const { return in_; }
  Node *getExpected() const { return expected_; }

  std::string getDebugDesc() const override;
  void visit(Node *parent, NodeVisitor *visitor) override;
};

class TransposeNode final : public Node {
  Node *in_;
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
  Node *in_;
  std::vector<size_t> dims_;

public:
  ReshapeNode(Node *in, llvm::StringRef name, llvm::ArrayRef<size_t> dims)
      : Node(Kinded::Kind::ReshapeInstKind, in->getType(), name), in_(in),
        dims_(dims.begin(), dims.end()) {}

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
  std::vector<Node *> in_;
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
  llvm::ArrayRef<Node *> getInputs() const { return in_; }
  size_t getDim() const { return dim_; }

  std::string getDebugDesc() const override;
  void visit(Node *parent, NodeVisitor *visitor) override;
};

class BatchNormalizationNode final : public Node {
  Node *in_;
  Node *scale_;
  Node *bias_;
  Node *mean_;
  Node *var_;
  const size_t channelIdx_;
  const float epsilon_;
  const float momentum_;

public:
  BatchNormalizationNode(Node *in, llvm::StringRef name, Node *scale,
                         Node *bias, Node *mean, Node *var, size_t channelIdx,
                         float epsilon, float momentum)
      : Node(Kinded::Kind::BatchNormalizationInstKind, in->getType(), name),
        in_(in), scale_(scale), bias_(bias), mean_(mean), var_(var),
        channelIdx_(channelIdx), epsilon_(epsilon), momentum_(momentum) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::BatchNormalizationInstKind;
  }
  Node *getInput() const { return in_; }

  Node *getScale() const { return scale_; }
  Node *getBias() const { return bias_; }
  Node *getMean() const { return mean_; }
  Node *getVar() const { return var_; }

  size_t getChannelIdx() const { return channelIdx_; }
  float getEpsilon() const { return epsilon_; }
  float getMomentum() const { return momentum_; }

  std::string getDebugDesc() const override;
  void visit(Node *parent, NodeVisitor *visitor) override;
};

class ArithmeticNode final : public Node {
  Node *LHS_;
  Node *RHS_;
  ArithmeticInst::OpKind kind_;
  const char *getKindStr() const;

public:
  ArithmeticNode(llvm::StringRef name, Node *LHS, Node *RHS,
                 ArithmeticInst::OpKind kind)
      : Node(Kinded::Kind::ArithmeticInstKind, LHS->getType(), name), LHS_(LHS),
        RHS_(RHS), kind_(kind) {}
  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ArithmeticInstKind;
  }
  Node *getLHS() const { return LHS_; }
  Node *getRHS() const { return RHS_; }
  ArithmeticInst::OpKind getKind() const { return kind_; }

  std::string getDebugDesc() const override;
  void visit(Node *parent, NodeVisitor *visitor) override;
};

class LocalResponseNormalizationNode final : public Node {
  Node *in_;
  Node *scale_;
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
        in_(in), halfWindowSize_(halfWindowSize), alpha_(alpha), beta_(beta),
        k_(k) {}

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

} // namespace glow

#endif // GLOW_GRAPH_NODES_H
