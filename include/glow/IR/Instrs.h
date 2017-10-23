#ifndef GLOW_IR_INSTRS_H
#define GLOW_IR_INSTRS_H

#include "glow/Base/Type.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/Support/Casting.h"

#include "llvm/ADT/ArrayRef.h"

namespace glow {

class AllocActivationInst;
class DeallocActivationInst;

class ConcatInst : public Instruction {
  /// We concat the tensors along this dimension.
  size_t dim_;

public:
  ConcatInst(Value *dest, llvm::ArrayRef<Value *> src, size_t dim)
      : Instruction(Kinded::Kind::ConcatInstKind, dest->getType(),
                    {{dest, OperandKind::Out}}),
        dim_(dim) {
    for (auto s : src) {
      pushOperand({s, OperandKind::In});
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
                    {{dest, OperandKind::Out},
                     {src, OperandKind::In},
                     {scale, OperandKind::In},
                     {bias, OperandKind::In},
                     {mean, OperandKind::InOut},
                     {var, OperandKind::InOut}}),
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
  using OpKind = ArithmeticNode::Mode;

private:
  OpKind kind_;
  const char *getKindStr() const;

public:
  ArithmeticInst(Value *dest, Value *LHS, Value *RHS, OpKind kind)
      : Instruction(Kinded::Kind::ArithmeticInstKind, dest->getType(),
                    {{dest, OperandKind::Out},
                     {LHS, OperandKind::In},
                     {RHS, OperandKind::In}}),
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
                    {{dest, OperandKind::Out},
                     {src, OperandKind::In},
                     {scale, OperandKind::InOut}}),
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
  enum class MutabilityKind {
    Constant, // A read-only region of memory.
    Mutable,  // A read/write region of memory.
  };

private:
  /// The mutability mode.
  MutabilityKind mut_;

public:
  WeightVar(TypeRef Ty, MutabilityKind mut)
      : Value(Ty, Kinded::Kind::WeightVarKind), mut_(mut) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::WeightVarKind;
  }

  static const char *getKindStr(MutabilityKind mut);

  const char *getKindStr() const;

  void setInitKind(MutabilityKind k) { mut_ = k; }
  MutabilityKind getKind() const { return mut_; }

  std::string getExtraDesc() const;
  void verify() const {}
};

} // namespace glow

#include "AutoGenInstr.h"

#endif // GLOW_IR_INSTRS_H
