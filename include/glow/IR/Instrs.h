#ifndef GLOW_IR_INSTRS_H
#define GLOW_IR_INSTRS_H

#include "glow/IR/IR.h"
#include "glow/IR/Type.h"
#include "glow/Support/Casting.h"

#include "llvm/ADT/ArrayRef.h"

namespace glow {

class AllocActivationInst : public Instruction {
public:
  AllocActivationInst(TypeRef Ty)
      : Instruction(Kinded::Kind::AllocActivationInstKind, Ty) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::AllocActivationInstKind;
  }

  std::string getExtraDesc() const;
  void verify() const;
};

class DeallocActivationInst : public Instruction {
public:
  DeallocActivationInst(Value *src)
      : Instruction(Kinded::Kind::DeallocActivationInstKind, src->getType(),
                    {{src, OperandKind::kOut}}) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::DeallocActivationInstKind;
  }

  void verify() const;

  AllocActivationInst *getAlloc() const {
    return cast<AllocActivationInst>(getOperand(0).first);
  }
};

class CopyInst : public Instruction {
public:
  CopyInst(Value *dest, Value *src)
      : Instruction(Kinded::Kind::CopyInstKind, dest->getType(),
                    {{dest, OperandKind::kOut}, {src, OperandKind::kIn}}) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::CopyInstKind;
  }
  Value *getDest() const { return getOperand(0).first; }
  Value *getSrc() const { return getOperand(1).first; }
  void verify() const;
};

class ConvolutionInst : public Instruction {
  size_t kernel_;
  size_t stride_;
  size_t pad_;
  size_t depth_;

public:
  ConvolutionInst(Value *dest, Value *src, Value *filter, Value *bias,
                  size_t kernel, size_t stride, size_t pad, size_t depth)
      : Instruction(Kinded::Kind::ConvolutionInstKind, dest->getType(),
                    {{dest, OperandKind::kOut},
                     {src, OperandKind::kIn},
                     {filter, OperandKind::kIn},
                     {bias, OperandKind::kIn}}),

        kernel_(kernel), stride_(stride), pad_(pad), depth_(depth) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ConvolutionInstKind;
  }

  bool mayShareBuffers() const { return false; }

  std::string getExtraDesc() const;
  Value *getDest() const { return getOperand(0).first; }
  Value *getSrc() const { return getOperand(1).first; }
  Value *getFilter() const { return getOperand(2).first; }
  Value *getBias() const { return getOperand(3).first; }

  size_t getKernel() const { return kernel_; }
  size_t getStride() const { return stride_; }
  size_t getPad() const { return pad_; }
  size_t getDepth() const { return depth_; }

  void verify() const;
};

class PoolInst : public Instruction {
public:
  /// Specifies the kind of pooling done by the operator.
  enum class OpKind {
    kMax,
    kAvg,
  };

private:
  size_t kernel_;
  size_t stride_;
  size_t pad_;
  OpKind kind_;

  const char *getKindStr() const;

public:
  PoolInst(Value *dest, Value *src, Value *srcXY, OpKind kind, size_t kernel,
           size_t stride, size_t pad)
      : Instruction(Kinded::Kind::PoolInstKind, dest->getType(),
                    {{dest, OperandKind::kOut},
                     {src, OperandKind::kIn},
                     {srcXY, OperandKind::kInOut}}),
        kernel_(kernel), stride_(stride), pad_(pad), kind_(kind) {}
  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::PoolInstKind;
  }

  bool mayShareBuffers() const { return false; }
  std::string getExtraDesc() const;
  Value *getDest() const { return getOperand(0).first; }
  Value *getSrc() const { return getOperand(1).first; }
  Value *srcXY() const { return getOperand(2).first; }
  size_t getKernel() const { return kernel_; }
  size_t getStride() const { return stride_; }
  size_t getPad() const { return pad_; }
  OpKind getKind() const { return kind_; }

  void verify() const;
};

class FullyConnectedInst : public Instruction {
  size_t depth_;

public:
  FullyConnectedInst(Value *dest, Value *src, Value *filter, Value *bias,
                     size_t depth)
      : Instruction(Kinded::Kind::FullyConnectedInstKind, dest->getType(),
                    {{dest, OperandKind::kOut},
                     {src, OperandKind::kIn},
                     {filter, OperandKind::kIn},
                     {bias, OperandKind::kIn}}),
        depth_(depth) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::FullyConnectedInstKind;
  }

  bool mayShareBuffers() const { return false; }
  std::string getExtraDesc() const;
  Value *getDest() const { return getOperand(0).first; }
  Value *getSrc() const { return getOperand(1).first; }
  Value *getFilter() const { return getOperand(2).first; }
  Value *getBias() const { return getOperand(3).first; }
  size_t getDepth() const { return depth_; }
  void verify() const;
};

class ReluInst : public Instruction {
public:
  ReluInst(Value *dest, Value *src)
      : Instruction(Kinded::Kind::ReluInstKind, dest->getType(),
                    {{dest, OperandKind::kOut}, {src, OperandKind::kIn}}) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ReluInstKind;
  }
  Value *getDest() const { return getOperand(0).first; }
  Value *getSrc() const { return getOperand(1).first; }
  void verify() const;
};

class SigmoidInst : public Instruction {
public:
  SigmoidInst(Value *dest, Value *src)
      : Instruction(Kinded::Kind::SigmoidInstKind, dest->getType(),
                    {{dest, OperandKind::kOut}, {src, OperandKind::kIn}}) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SigmoidInstKind;
  }
  Value *getDest() const { return getOperand(0).first; }
  Value *getSrc() const { return getOperand(1).first; }
  void verify() const;
};

class TanhInst : public Instruction {
public:
  TanhInst(Value *dest, Value *src)
      : Instruction(Kinded::Kind::TanhInstKind, dest->getType(),
                    {{dest, OperandKind::kOut}, {src, OperandKind::kIn}}) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::TanhInstKind;
  }
  Value *getDest() const { return getOperand(0).first; }
  Value *getSrc() const { return getOperand(1).first; }
  void verify() const;
};

class SoftMaxInst : public Instruction {
public:
  SoftMaxInst(Value *dest, Value *src, Value *E, Value *selected)
      : Instruction(Kinded::Kind::SoftMaxInstKind, dest->getType(),
                    {{dest, OperandKind::kOut},
                     {src, OperandKind::kIn},
                     {E, OperandKind::kInOut},
                     {selected, OperandKind::kIn}}) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SoftMaxInstKind;
  }
  Value *getDest() const { return getOperand(0).first; }
  Value *getSrc() const { return getOperand(1).first; }
  Value *getE() const { return getOperand(2).first; }
  Value *getSelected() const { return getOperand(3).first; }
  void verify() const;
};

class RegressionInst : public Instruction {
public:
  RegressionInst(Value *dest, Value *src, Value *expected)
      : Instruction(Kinded::Kind::RegressionInstKind, dest->getType(),
                    {{dest, OperandKind::kOut},
                     {src, OperandKind::kIn},
                     {expected, OperandKind::kIn}}) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::RegressionInstKind;
  }
  Value *getDest() const { return getOperand(0).first; }
  Value *getSrc() const { return getOperand(1).first; }
  Value *getExpected() const { return getOperand(2).first; }
  void verify() const;
};

class TransposeInst : public Instruction {
  std::vector<unsigned> shuffle_;

public:
  TransposeInst(Value *dest, Value *src, llvm::ArrayRef<unsigned> shuffle)
      : Instruction(Kinded::Kind::TransposeInstKind, dest->getType(),
                    {{dest, OperandKind::kOut}, {src, OperandKind::kIn}}),
        shuffle_(shuffle.begin(), shuffle.end()) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::TransposeInstKind;
  }

  bool mayShareBuffers() const { return false; }
  std::string getExtraDesc() const;
  Value *getDest() const { return getOperand(0).first; }
  Value *getSrc() const { return getOperand(1).first; }

  llvm::ArrayRef<unsigned> getShuffle() const { return shuffle_; }
  void verify() const;
};

class ReshapeInst : public Instruction {
  std::vector<size_t> dims_;

public:
  ReshapeInst(Value *dest, Value *src, llvm::ArrayRef<size_t> dims)
      : Instruction(Kinded::Kind::ReshapeInstKind, dest->getType(),
                    {{dest, OperandKind::kOut}, {src, OperandKind::kIn}}),
        dims_(dims.begin(), dims.end()) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ReshapeInstKind;
  }

  std::string getExtraDesc() const;
  Value *getDest() const { return getOperand(0).first; }
  Value *getSrc() const { return getOperand(1).first; }
  llvm::ArrayRef<size_t> getDims() { return dims_; }

  void verify() const;
};

class ConcatInst : public Instruction {
  /// We concat the tensors along this dimension.
  size_t dim_;

public:
  ConcatInst(Value *dest, llvm::ArrayRef<Value *> src, size_t dim)
      : Instruction(Kinded::Kind::ConcatInstKind, dest->getType(),
                    {{dest, OperandKind::kOut}}),
        dim_(dim) {
    for (auto s : src) {
      pushOperand({s, OperandKind::kIn});
    }
  }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ConcatInstKind;
  }
  bool mayShareBuffers() const { return false; }
  std::string getExtraDesc() const;
  Value *getDest() const { return getOperand(0).first; }
  Value *getSrc() const { return getOperand(1).first; }
  size_t getDim() const { return dim_; }

  void verify() const;
};

class BatchNormalizationInst : public Instruction {
  const size_t channelIdx_;
  const float epsilon_;
  const float momentum_;

public:
  BatchNormalizationInst(Value *dest, Value *src, Value *scale, Value *bias,
                         Value *mean, Value *var, size_t channelIdx,
                         float epsilon, float momentum)
      : Instruction(Kinded::Kind::BatchNormalizationInstKind, dest->getType(),
                    {{dest, OperandKind::kOut},
                     {src, OperandKind::kIn},
                     {scale, OperandKind::kIn},
                     {bias, OperandKind::kIn},
                     {mean, OperandKind::kInOut},
                     {var, OperandKind::kInOut}}),
        channelIdx_(channelIdx), epsilon_(epsilon), momentum_(momentum) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::BatchNormalizationInstKind;
  }
  std::string getExtraDesc() const;
  Value *getDest() const { return getOperand(0).first; }
  Value *getSrc() const { return getOperand(1).first; }

  Value *getScale() const { return getOperand(2).first; }
  Value *getBias() const { return getOperand(3).first; }
  Value *getMean() const { return getOperand(4).first; }
  Value *getVar() const { return getOperand(5).first; }

  size_t getChannelIdx() const { return channelIdx_; }
  float getEpsilon() const { return epsilon_; }
  float getMomentum() const { return momentum_; }
  void verify() const;
};

class ArithmeticInst : public Instruction {
public:
  /// Specifies the kind of pooling done by the operator.
  enum class OpKind {
    kAdd,
    kMul,
  };

private:
  OpKind kind_;
  const char *getKindStr() const;

public:
  ArithmeticInst(Value *dest, Value *LHS, Value *RHS, OpKind kind)
      : Instruction(Kinded::Kind::ArithmeticInstKind, dest->getType(),
                    {{dest, OperandKind::kOut},
                     {LHS, OperandKind::kIn},
                     {RHS, OperandKind::kIn}}),
        kind_(kind) {}
  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ArithmeticInstKind;
  }
  std::string getExtraDesc() const;
  Value *getDest() const { return getOperand(0).first; }
  Value *getLHS() const { return getOperand(1).first; }
  Value *getRHS() const { return getOperand(2).first; }
  OpKind getKind() const { return kind_; }

  void verify() const;
};

class LocalResponseNormalizationInst : public Instruction {
  /// The number of neighbouring channels on each side to sum over
  size_t halfWindowSize_;

  /// The scaling parameter
  float alpha_;

  /// The exponent parameter
  float beta_;

  /// The offset parameter
  float k_;

public:
  LocalResponseNormalizationInst(Value *dest, Value *src, Value *scale,
                                 size_t halfWindowSize, float alpha, float beta,
                                 float k)
      : Instruction(Kinded::Kind::LocalResponseNormalizationInstKind,
                    dest->getType(),
                    {{dest, OperandKind::kOut},
                     {src, OperandKind::kIn},
                     {scale, OperandKind::kInOut}}),
        halfWindowSize_(halfWindowSize), alpha_(alpha), beta_(beta), k_(k) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::LocalResponseNormalizationInstKind;
  }
  std::string getExtraDesc() const;
  Value *getDest() const { return getOperand(0).first; }
  Value *getSrc() const { return getOperand(1).first; }
  Value *getScale() const { return getOperand(2).first; }

  size_t gethalfWindowSize() const { return halfWindowSize_; }
  float getAlpha() const { return alpha_; }
  float getBeta() const { return beta_; }
  float getK() const { return k_; }
  void verify() const;
};

class WeightVar : public Value {
public:
  enum class InitKind {
    kExtern,    // No initialization.
    kBroadcast, // Broadcast a single value to all elements.
    kXavier,    // Init the tensor with random values using the Xavier method.
  };

private:
  /// The value to use during initialization. This can be the value to splat or
  /// a parameter to specify the range of the random values.
  float val_;

  /// The initialization mode.
  InitKind initKind_;

  const char *getInitKindStr() const;

public:
  WeightVar(TypeRef Ty, InitKind initKind, float val)
      : Value(Ty, Kinded::Kind::WeightVarKind), val_(val), initKind_(initKind) {
  }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::WeightVarKind;
  }

  InitKind getInitKind() const { return initKind_; }
  float getVal() const { return val_; }
  std::string getExtraDesc() const;
  void verify() const {}
};

} // namespace glow

#endif // GLOW_IR_INSTRS_H
