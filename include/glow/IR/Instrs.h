#ifndef GLOW_IR_INSTRS_H
#define GLOW_IR_INSTRS_H

#include "glow/IR/IR.h"
#include "glow/IR/Type.h"

namespace glow {

class CopyInst : public Instruction {
public:
  CopyInst(Value *dest, Value *src)
      : Instruction({{dest, OperandKind::kOut}, {src, OperandKind::kIn}}) {}
  StringRef getValueName() override { return "copy"; }
};

class ReluInst : public Instruction {
public:
  ReluInst(Value *dest, Value *src)
      : Instruction({{dest, OperandKind::kOut}, {src, OperandKind::kIn}}) {}
  StringRef getValueName() override { return "relu"; }
};

class TransposeInst : public Instruction {
  std::vector<unsigned> shuffle_;

public:
  TransposeInst(Value *dest, Value *src, ArrayRef<unsigned> shuffle)
      : Instruction({{dest, OperandKind::kOut}, {src, OperandKind::kIn}}),
        shuffle_(shuffle.begin(), shuffle.end()) {}
  StringRef getValueName() override { return "transpose"; }

  std::string getExtraDesc() override {
    std::string sb = " {";
    for (int i = 0; i < shuffle_.size(); i++) {
      if (i) {
        sb += ", ";
      }
      sb += std::to_string(shuffle_[i]);
    }
    return sb + "}";
  }
};

class ConvolutionInst : public Instruction {
  std::vector<unsigned> shuffle_;

  size_t kernel_;
  size_t stride_;
  size_t pad_;
  size_t depth_;

public:
  ConvolutionInst(Value *dest, Value *src, Value *filter, Value *bias,
                  size_t kernel, size_t stride, size_t pad, size_t depth)
      : Instruction({{dest, OperandKind::kOut},
                     {src, OperandKind::kIn},
                     {filter, OperandKind::kIn},
                     {bias, OperandKind::kIn}}),

        kernel_(kernel), stride_(stride), pad_(pad), depth_(depth) {}

  StringRef getValueName() override { return "convolution"; }

  std::string getExtraDesc() override {
    std::string sb = " {";
    sb += std::to_string(kernel_) + " ";
    sb += std::to_string(stride_) + " ";
    sb += std::to_string(pad_) + " ";
    sb += std::to_string(depth_);
    return sb + " }";
  }
};

class StaticVariable : public Value {
public:
  enum class InitKind {
    kExtern,    // No initialization.
    kBroadcast, // Broadcast a single value to all elements.
    kXavier,    // Init the tensor with random values using the Xavier method.
  };

  const char *getKindStr(InitKind K) {
    const char *names[] = {"extern", "broadcast", "xavier", nullptr};
    return names[(int)K];
  }

private:
  /// The type of the tensor to allocate.
  TypeRef Ty_;

  /// The value to use during initialization. This can be the value to splat or
  /// a parameter to specify the range of the random values.
  float val_;

  /// The initialization mode.
  InitKind mode_;

public:
  StaticVariable(TypeRef Ty, InitKind mode, float val)
      : Value(), Ty_(Ty), val_(val), mode_(mode) {}
  StringRef getValueName() override { return "static"; }
  InitKind getMode() { return mode_; }
  float getVal() { return val_; }
  TypeRef getType() { return Ty_; }

  std::string getExtraDesc() override {
    return Ty_->asString() + ", " + std::to_string(val_) + ", " +
           getKindStr(mode_);
  }
};

} // namespace glow

#endif // GLOW_IR_INSTRS_H
