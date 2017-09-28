#ifndef GLOW_IR_INSTRS_H
#define GLOW_IR_INSTRS_H

#include "glow/IR/IR.h"
#include "glow/IR/Type.h"

namespace glow {

class AllocActivationInst : public Instruction {
public:
  AllocActivationInst(TypeRef Ty)
      : Instruction(Kinded::Kind::AllocActivationInstKind, Ty) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::AllocActivationInstKind;
  }

  std::string getExtraDesc();
  void verify();
};

class DeallocActivationInst : public Instruction {
public:
  DeallocActivationInst(Value *src)
      : Instruction(Kinded::Kind::DeallocActivationInstKind, src->getType(),
                    {{src, OperandKind::kIn}}) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::DeallocActivationInstKind;
  }

  void verify();
};

class CopyInst : public Instruction {
public:
  CopyInst(Value *dest, Value *src)
      : Instruction(Kinded::Kind::CopyInstKind, dest->getType(),
                    {{dest, OperandKind::kOut}, {src, OperandKind::kIn}}) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::CopyInstKind;
  }
  Value *getDest() { return getOperand(0).first; }
  Value *getSrc() { return getOperand(1).first; }
  void verify();
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
  std::string getExtraDesc();
  Value *getDest() { return getOperand(0).first; }
  Value *getSrc() { return getOperand(1).first; }
  Value *getFilter() { return getOperand(2).first; }
  Value *getBias() { return getOperand(3).first; }

  size_t getKernel() { return kernel_; }
  size_t getStride() { return stride_; }
  size_t getPad() { return pad_; }
  size_t getDepth() { return depth_; }

  void verify();
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

  const char *getKindStr();

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
  std::string getExtraDesc();
  Value *getDest() { return getOperand(0).first; }
  Value *getSrc() { return getOperand(1).first; }
  Value *srcXY() { return getOperand(2).first; }
  size_t getKernel() { return kernel_; }
  size_t getStride() { return stride_; }
  size_t getPad() { return pad_; }
  OpKind getKind() { return kind_; }

  void verify();
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
  std::string getExtraDesc();
  Value *getDest() { return getOperand(0).first; }
  Value *getSrc() { return getOperand(1).first; }
  Value *getFilter() { return getOperand(2).first; }
  Value *getBias() { return getOperand(3).first; }
  size_t getDepth() { return depth_; }
  void verify();
};

class ReluInst : public Instruction {
public:
  ReluInst(Value *dest, Value *src)
      : Instruction(Kinded::Kind::ReluInstKind, dest->getType(),
                    {{dest, OperandKind::kOut}, {src, OperandKind::kIn}}) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ReluInstKind;
  }
  Value *getDest() { return getOperand(0).first; }
  Value *getSrc() { return getOperand(1).first; }
  void verify();
};

class SigmoidInst : public Instruction {
public:
  SigmoidInst(Value *dest, Value *src)
      : Instruction(Kinded::Kind::SigmoidInstKind, dest->getType(),
                    {{dest, OperandKind::kOut}, {src, OperandKind::kIn}}) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::SigmoidInstKind;
  }
  Value *getDest() { return getOperand(0).first; }
  Value *getSrc() { return getOperand(1).first; }
  void verify();
};

class TanhInst : public Instruction {
public:
  TanhInst(Value *dest, Value *src)
      : Instruction(Kinded::Kind::TanhInstKind, dest->getType(),
                    {{dest, OperandKind::kOut}, {src, OperandKind::kIn}}) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::TanhInstKind;
  }
  Value *getDest() { return getOperand(0).first; }
  Value *getSrc() { return getOperand(1).first; }
  void verify();
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
  Value *getDest() { return getOperand(0).first; }
  Value *getSrc() { return getOperand(1).first; }
  Value *getE() { return getOperand(2).first; }
  Value *getSelected() { return getOperand(3).first; }
  void verify();
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
  Value *getDest() { return getOperand(0).first; }
  Value *getSrc() { return getOperand(1).first; }
  Value *getExpected() { return getOperand(2).first; }
  void verify();
};

class TransposeInst : public Instruction {
  std::vector<unsigned> shuffle_;

public:
  TransposeInst(Value *dest, Value *src, ArrayRef<unsigned> shuffle)
      : Instruction(Kinded::Kind::TransposeInstKind, dest->getType(),
                    {{dest, OperandKind::kOut}, {src, OperandKind::kIn}}),
        shuffle_(shuffle.begin(), shuffle.end()) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::TransposeInstKind;
  }
  std::string getExtraDesc();
  Value *getDest() { return getOperand(0).first; }
  Value *getSrc() { return getOperand(1).first; }

  ArrayRef<unsigned> getShuffle() { return shuffle_; }
  void verify();
};

class ReshapeInst : public Instruction {
  std::vector<size_t> dims_;

public:
  ReshapeInst(Value *dest, Value *src, ArrayRef<size_t> dims)
      : Instruction(Kinded::Kind::ReshapeInstKind, dest->getType(),
                    {{dest, OperandKind::kOut}, {src, OperandKind::kIn}}),
        dims_(dims.begin(), dims.end()) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ReshapeInstKind;
  }

  std::string getExtraDesc();
  Value *getDest() { return getOperand(0).first; }
  Value *getSrc() { return getOperand(1).first; }
  ArrayRef<size_t> getDims() { return dims_; }

  void verify();
};

class ConcatInst : public Instruction {
  /// We concat the tensors along this dimension.
  size_t dim_;

public:
  ConcatInst(Value *dest, ArrayRef<Value *> src, size_t dim)
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
  std::string getExtraDesc();
  Value *getDest() { return getOperand(0).first; }
  Value *getSrc() { return getOperand(1).first; }
  size_t getDim() { return dim_; }

  void verify();
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
  std::string getExtraDesc();
  Value *getDest() { return getOperand(0).first; }
  Value *getSrc() { return getOperand(1).first; }

  Value *getScale() { return getOperand(2).first; }
  Value *getBias() { return getOperand(3).first; }
  Value *getMean() { return getOperand(4).first; }
  Value *getVar() { return getOperand(5).first; }

  size_t getChannelIdx() { return channelIdx_; }
  float getEpsilon() { return epsilon_; }
  float getMomentum() { return momentum_; }
  void verify();
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
  const char *getKindStr();

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
  std::string getExtraDesc();
  Value *getDest() { return getOperand(0).first; }
  Value *getLHS() { return getOperand(1).first; }
  Value *getRHS() { return getOperand(2).first; }
  OpKind getKind() { return kind_; }

  void verify();
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

  const char *getInitKindStr();

public:
  WeightVar(TypeRef Ty, InitKind initKind, float val)
      : Value(Ty, Kinded::Kind::WeightVarKind), val_(val), initKind_(initKind) {
  }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::WeightVarKind;
  }

  InitKind getInitKind() { return initKind_; }
  float getVal() { return val_; }
  std::string getExtraDesc();
  void verify() {}
};

} // namespace glow

#endif // GLOW_IR_INSTRS_H
