#include "glow/IR/Instrs.h"
#include "glow/IR/IR.h"

#include "glow/Network/Nodes.h"

#include <cassert>

using namespace glow;

// Helper methods that are used to print the instruction parameters.
namespace {
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
} // namespace

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

const char *ArithmeticInst::getKindStr() {
  const char *names[] = {"add", "mul", nullptr};
  return names[(int)kind_];
}

std::string ArithmeticInst::getExtraDesc() { return getKindStr(); }

const char *StaticVariable::getKindStr() {
  const char *names[] = {"extern", "broadcast", "xavier", nullptr};
  return names[(int)mode_];
}

std::string StaticVariable::getExtraDesc() {
  auto sp = ", ";
  return getType()->asString() + sp + std::to_string(val_) + sp + getKindStr();
}

/// Check that the type of the first operand matches the type of the second
/// operand.
static void checkSameType(Instruction::Operand A, Instruction::Operand B) {
  assert(A.first->getType() == B.first->getType() && "Invalid type");
}

void CopyInst::verify() { checkSameType(getOperand(0), getOperand(1)); }
void ConvolutionInst::verify() {
  Value *dest = getOperand(0).first;
  Value *src = getOperand(1).first;
  Value *filter = getOperand(2).first;
  Value *bias = getOperand(3).first;

  ShapeNHWC idim = src->getType()->dims();
  ShapeNHWC odim = dest->getType()->dims();
  assert(idim.w >= kernel_ && idim.h >= kernel_ &&
         "buffer too small for selected stride");

  auto outSz =
      ConvNode::calculateOutputDims(idim.h, idim.w, pad_, kernel_, stride_);
  ShapeNHWC exp = ArrayRef<size_t>{idim.n, outSz.first, outSz.second, depth_};
  assert(exp == odim && "Invalid output dimensions");

  ArrayRef<size_t> filterDims = {depth_, kernel_, kernel_, idim.c};
  assert(filter->getType()->dims() == filterDims && "Invalid filter dims");

  ArrayRef<size_t> biasDims = {depth_};
  assert(bias->getType()->dims() == biasDims && "Invalid bias dims");
}

void PoolInst::verify() {
  Value *dest = getOperand(0).first;
  Value *src = getOperand(1).first;
  Value *srcXY = getOperand(2).first;

  ShapeNHWC idim = src->getType()->dims();
  ShapeNHWC odim = dest->getType()->dims();
  assert(idim.w >= kernel_ && idim.h >= kernel_ &&
         "buffer too small for selected stride");

  auto outSz =
      ConvNode::calculateOutputDims(idim.h, idim.w, pad_, kernel_, stride_);
  ShapeNHWC exp = ArrayRef<size_t>{idim.n, outSz.first, outSz.second, idim.c};
  assert(exp == odim && "Invalid output dimensions");

  // Allocate cache arrays that store the x and y coordinates of the incoming
  // gradient for each max element.
  if (kind_ == OpKind::kMax) {
    ArrayRef<size_t> exp = {idim.n, outSz.first, outSz.second, idim.c, 2};
    assert(srcXY->getType()->dims() == exp && "Invalid srcXY dims");
  }
}

void FullyConnectedInst::verify() {
  Value *dest = getOperand(0).first;
  Value *src = getOperand(1).first;
  Value *W = getOperand(2).first;
  Value *B = getOperand(3).first;
  auto idim = flattenCdr(src->dims());

  ArrayRef<size_t> exp = {idim.first, depth_};
  assert(dest->dims() == exp && "Invalid output shape");
  (void)exp;

  ArrayRef<size_t> expW = {depth_, idim.second};
  assert(W->dims() == expW && "Invalid output shape");
  (void)expW;

  ArrayRef<size_t> expB = {depth_};
  assert(B->dims() == expB && "Invalid output shape");
  (void)expB;
}

void ReluInst::verify() { checkSameType(getOperand(0), getOperand(1)); }
void SigmoidInst::verify() { checkSameType(getOperand(0), getOperand(1)); }
void TanhInst::verify() { checkSameType(getOperand(0), getOperand(1)); }

void SoftMaxInst::verify() { checkSameType(getOperand(0), getOperand(1)); }
void RegressionInst::verify() { checkSameType(getOperand(0), getOperand(1)); }

void TransposeInst::verify() {}
void ReshapeInst::verify() {}
void ConcatInst::verify() {}
void BatchNormalizationInst::verify() {}
void ArithmeticInst::verify() {
  checkSameType(getOperand(0), getOperand(1));
  checkSameType(getOperand(0), getOperand(2));
}
