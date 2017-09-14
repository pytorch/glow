#include "glow/IR/Instrs.h"
#include "glow/IR/IR.h"

using namespace glow;

template <typename E> std::string listToString_impl(E v) {
  return std::to_string(v);
}

template <typename E, typename... Args>
std::string listToString_impl(E first, Args... args) {
  return std::to_string(first) + " " + listToString_impl(args...);
}

template <typename... Args> std::string listToString(Args... args) {
  return "{" + listToString_impl(args...) + "}";
}

template <typename E> std::string arrayRefToString(ArrayRef<E> list) {
  std::string sb = "{";
  for (int i = 0, e = list.size(); i < e; i++) {
    if (i) {
      sb += ", ";
    }
    sb += std::to_string(list[i]);
  }
  return sb + "}";
}

std::string ConvolutionInst::getExtraDesc() {
  return listToString(kernel_, stride_, pad_, depth_);
}

const char *PoolInst::getKindStr() {
  const char *names[] = {"max", "avg", nullptr};
  return names[(int)kind_];
}

std::string PoolInst::getExtraDesc() {
  std::string sb = getKindStr();
  return sb += " " + listToString(kernel_, stride_, pad_);
}

std::string FullyConnectedInst::getExtraDesc() { return listToString(depth_); }

std::string TransposeInst::getExtraDesc() {
  return arrayRefToString<unsigned>(shuffle_);
}

std::string ReshapeInst::getExtraDesc() {
  return arrayRefToString<size_t>(dims_);
}

std::string ConcatInst::getExtraDesc() {
  return "{ " + std::to_string(dim_) + " }";
}

std::string BatchNormalizationInst::getExtraDesc() {
  return listToString(channelIdx_, epsilon_, momentum_);
}

const char *StaticVariable::getKindStr() {
  const char *names[] = {"extern", "broadcast", "xavier", nullptr};
  return names[(int)mode_];
}

std::string ArithmeticInst::getExtraDesc() { return getKindStr(); }

std::string StaticVariable::getExtraDesc() {
  return Ty_->asString() + ", " + std::to_string(val_) + ", " + getKindStr();
}
