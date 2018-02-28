// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/IR/Instrs.h"
#include "glow/IR/IR.h"
#include "glow/Support/Support.h"

#include "llvm/Support/Casting.h"

#include <cassert>

using namespace glow;
using llvm::cast;
using llvm::isa;

//===----------------------------------------------------------------------===//
//                      Instruction textual printers
//===----------------------------------------------------------------------===//

const char *WeightVar::getMutabilityStr(MutabilityKind kind) {
  const char *names[] = {"const", "mutable", nullptr};
  return names[static_cast<int>(kind)];
}

const char *WeightVar::getMutabilityStr() const {
  return getMutabilityStr(mut_);
}

void WeightVar::dump(llvm::raw_ostream &os) const {
  os << "%" << getName() << " = WeightVar ";
  os << *getType() << " " << getMutabilityStr();
}

//===----------------------------------------------------------------------===//
//                       Instruction verification
//===----------------------------------------------------------------------===//

/// Check that the type of the first operand matches the type of the second
/// operand.
static void checkSameType(Value *A, Value *B) {
  assert(A->getType() == B->getType() && "Invalid type");
}

static void checkType(Value *A, ElemKind expectedType) {
  assert(A->getElementType() == expectedType && "Invalid type");
}

static void checkSameShape(Value *A, Value *B) {
  assert(A->dims().equals(B->dims()) && "Dimensions mismatch");
}

void CopyInst::verify() const {
  auto *dest = getDest();
  auto *src = getSrc();
  checkSameType(dest, src);
  // The operands of the copy instruction must be variables.
  assert(isa<AllocActivationInst>(dest) || isa<WeightVar>(dest) ||
         isa<TensorViewInst>(dest));
  assert(isa<AllocActivationInst>(src) || isa<WeightVar>(src) ||
         isa<TensorViewInst>(src));
}

static void verifyConvolution(Value *src, Value *dest, Value *filter,
                              Value *bias, size_t kernel, size_t stride,
                              size_t pad, size_t depth) {
  assert(src->getElementType() == dest->getElementType() && "Invalid Type");
  assert(src->getElementType() == filter->getElementType() && "Invalid Type");
  assert(src->getElementType() == bias->getElementType() && "Invalid Type");

  ShapeNHWC idim(src->getType()->dims());
  ShapeNHWC odim(dest->getType()->dims());

  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  auto outSz = calculateConvOutputDims(idim.h, idim.w, kernel, stride, pad);
  ShapeNHWC exp(idim.n, outSz.first, outSz.second, depth);
  (void)exp;
  assert(exp == odim && "Invalid output dimensions");

  auto filterDims = {depth, kernel, kernel, idim.c};
  assert(filter->getType()->dims().equals(filterDims) && "Invalid filter dims");
  (void)filterDims;

  auto biasDims = {depth};
  assert(bias->getType()->dims().equals(biasDims) && "Invalid bias dims");
  (void)biasDims;
}

static void verifyPoolMaxWithXY(Value *src, Value *dest, Value *srcXY,
                                size_t kernel, size_t stride, size_t pad) {
  (void)srcXY;
  ShapeNHWC idim = ShapeNHWC(src->getType()->dims());
  ShapeNHWC odim = ShapeNHWC(dest->getType()->dims());
  (void)odim;
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  auto outSz = calculateConvOutputDims(idim.h, idim.w, kernel, stride, pad);
  ShapeNHWC exp(idim.n, outSz.first, outSz.second, idim.c);
  (void)exp;
  assert(exp == odim && "Unexpected output dimensions");

  // Allocate cache arrays that store the x and y coordinates of the incoming
  // gradient for each max element.
  auto E = {idim.n, outSz.first, outSz.second, idim.c, 2UL};
  assert(srcXY->getType()->dims().equals(E) && "Invalid srcXY dims");
  (void)E;
}

static void verifyPoolAvg(Value *src, Value *dest, size_t kernel, size_t stride,
                          size_t pad) {
  ShapeNHWC idim = ShapeNHWC(src->getType()->dims());
  ShapeNHWC odim = ShapeNHWC(dest->getType()->dims());
  (void)odim;
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  auto outSz = calculateConvOutputDims(idim.h, idim.w, kernel, stride, pad);
  ShapeNHWC exp(idim.n, outSz.first, outSz.second, idim.c);
  (void)exp;
  assert(exp == odim && "Unexpected output dimensions");
}

static void verifyBatchNormalization(Value *src, Value *dest, Value *scale,
                                     Value *bias, Value *mean, Value *var,
                                     size_t channel) {
  checkSameType(dest, src);

  // Figure out how many channels are in the tensor.
  size_t channels = dest->dims()[channel];

  auto exp = {channels};
  (void)exp;
  assert(scale->getType()->dims().equals(exp) && "Invalid bias dim");
  assert(bias->getType()->dims().equals(exp) && "Invalid scale dim");
  assert(mean->getType()->dims().equals(exp) && "Invalid mean dim");
  assert(var->getType()->dims().equals(exp) && "Invalid var dim");
}

void ConvolutionInst::verify() const {
  Value *dest = getDest();
  Value *src = getSrc();
  Value *filter = getFilter();
  Value *bias = getBias();

  verifyConvolution(src, dest, filter, bias, Kernel_, Stride_, Pad_, Depth_);
}

void ConvolutionGradInst::verify() const {
  Value *dest = getDestGrad();
  Value *src = getSrcGrad();
  Value *filter = getFilterGrad();
  Value *bias = getBiasGrad();

  verifyConvolution(src, dest, filter, bias, Kernel_, Stride_, Pad_, Depth_);
}

void PoolMaxInst::verify() const {
  Value *dest = getDest();
  Value *src = getSrc();
  ShapeNHWC idim = ShapeNHWC(src->getType()->dims());
  ShapeNHWC odim = ShapeNHWC(dest->getType()->dims());
  (void)odim;
  assert(idim.w >= Kernel_ && idim.h >= Kernel_ &&
         "buffer too small for selected stride");

  auto outSz = calculateConvOutputDims(idim.h, idim.w, Kernel_, Stride_, Pad_);
  ShapeNHWC exp(idim.n, outSz.first, outSz.second, idim.c);
  (void)exp;
  assert(exp == odim && "Unexpected output dimensions");
}

void PoolMaxWithXYInst::verify() const {
  Value *dest = getDest();
  Value *src = getSrc();
  Value *srcXY = getSrcXY();

  verifyPoolMaxWithXY(src, dest, srcXY, Kernel_, Stride_, Pad_);
}

void PoolMaxWithXYGradInst::verify() const {
  Value *dest = getDestGrad();
  Value *src = getSrcGrad();
  Value *srcXY = getSrcXY();

  verifyPoolMaxWithXY(src, dest, srcXY, Kernel_, Stride_, Pad_);
}

void PoolAvgInst::verify() const {
  Value *dest = getDest();
  Value *src = getSrc();

  verifyPoolAvg(src, dest, Kernel_, Stride_, Pad_);
}

void PoolAvgGradInst::verify() const {
  Value *dest = getDestGrad();
  Value *src = getSrcGrad();

  verifyPoolAvg(src, dest, Kernel_, Stride_, Pad_);
}

void BatchedMatMulInst::verify() const {
  Value *dest = getDest();
  Value *lhs = getLHS();
  Value *rhs = getRHS();

  auto LDims = lhs->dims();
  auto RDims = rhs->dims();
  auto DDims = dest->dims();
  (void)DDims;
  assert(DDims.size() == 3);
  auto elem = dest->getType()->getElementType();
  (void)elem;
  assert(lhs->getType()->getElementType() == elem);
  assert(rhs->getType()->getElementType() == elem);

  size_t N, X, Y;
  std::tie(N, X, Y) = calculateMatMulOutputDims(LDims, RDims);

  assert(N == DDims[0] && "Invalid matrix dims");
  assert(X == DDims[1] && "Invalid matrix dims");
  assert(Y == DDims[2] && "Invalid matrix dims");

  (void)N;
  (void)X;
  (void)Y;
}

void SigmoidInst::verify() const { checkSameType(getDest(), getSrc()); }

void TanhInst::verify() const { checkSameType(getDest(), getSrc()); }

void SoftMaxInst::verify() const {
  checkSameType(getDest(), getSrc());
  assert(getDest()->dims() == getSrc()->dims() && "Invalid shape");
}

void SoftMaxGradInst::verify() const {
  checkSameType(getOrigDest(), getOrigSrc());
  checkSameType(getOrigDest(), getSrcGrad());
  auto destShape = getOrigDest()->dims();
  assert(destShape == getOrigSrc()->dims() && "Invalid shape");
  assert(destShape == getSrcGrad()->dims() && "Invalid shape");
  (void)destShape;
}

void CrossEntropyLossInst::verify() const {
  assert(getP()->dims()[0] == getLabels()->dims()[0] && "Invalid shape");
}

void CrossEntropyLossGradInst::verify() const {
  assert(getPgrad()->dims()[0] == getLabels()->dims()[0] && "Invaild shape");
}

void ReshapeInst::verify() const {
  assert(getDest()->getType()->size() == getSrc()->getType()->size() &&
         "Reshape into a different size");
}

void TensorViewInst::verify() const {
  assert(getSrc()->getType()->size() == getType()->size() &&
         "TensorView view size should be the same as Src size");
  assert(getSrc()->getElementType() == getType()->getElementType() &&
         "TensorView view element type should be the same as Src type");
}

void TransposeInst::verify() const {
  auto *dest = getDest();
  auto *src = getSrc();
  (void)dest;
  llvm::SmallVector<size_t, 6> shape;

  auto dims = src->dims();
  for (size_t i = 0; i < dims.size(); i++) {
    shape.push_back(dims[Shuffle_[i]]);
  }

  assert(dest->dims().equals(shape) && "Invalid transpose dims");
}

void BroadcastInst::verify() const {
  auto *src = getSrc();
  auto *dest = getDest();
  auto shape = getShape();
  (void)src;
  (void)dest;
  (void)shape;

  assert(src->dims().size() <= dest->dims().size() &&
         "Source being broadcasted must have <= number dims of result shape.");
  assert(dest->dims().equals(shape) &&
         "New broadcasted shape does not match shape to broadcast to.");
}

void SplatInst::verify() const {}

void InsertTensorInst::verify() const {
  auto *dest = getDest();
  auto *src = getSrc();
  auto offsets = getOffsets();
  unsigned numDims = dest->dims().size();
  (void)numDims;
  (void)dest;
  (void)src;
  (void)offsets;
  assert(numDims == src->dims().size() && numDims == offsets.size() &&
         "Invalid number of dimensions");

  for (unsigned i = 0; i < numDims; i++) {
    assert(src->dims()[i] + offsets[i] <= dest->dims()[i] && "out of bounds");
  }
}

void ExtractTensorInst::verify() const {
  auto *dest = getDest();
  auto *src = getSrc();
  auto offsets = getOffsets();
  unsigned numDims = dest->dims().size();
  (void)numDims;
  (void)dest;
  (void)src;
  (void)offsets;
  assert(numDims == src->dims().size() && numDims == offsets.size() &&
         "Invalid number of dimensions");

  for (unsigned i = 0; i < numDims; i++) {
    assert(dest->dims()[i] + offsets[i] <= src->dims()[i] && "out of bounds");
  }
}

void BatchNormalizationInst::verify() const {
  Value *src = getSrc();
  Value *dest = getDest();
  Value *scale = getScale();
  Value *bias = getBias();
  Value *mean = getMean();
  Value *var = getVar();

  verifyBatchNormalization(src, dest, scale, bias, mean, var, ChannelIdx_);
}

void BatchNormalizationGradInst::verify() const {
  Value *src = getSrcGrad();
  Value *dest = getDestGrad();
  Value *scale = getScaleGrad();
  Value *bias = getBiasGrad();
  Value *mean = getMean();
  Value *var = getVar();

  verifyBatchNormalization(src, dest, scale, bias, mean, var, ChannelIdx_);
}

void LocalResponseNormalizationInst::verify() const {
  checkSameType(getDest(), getSrc());
  checkSameType(getDest(), getScale());
}

void LocalResponseNormalizationGradInst::verify() const {
  checkSameType(getDestGrad(), getSrcGrad());
  checkSameType(getDestGrad(), getScale());
}

void ElementAddInst::verify() const {
  checkSameShape(getDest(), getLHS());
  checkSameShape(getDest(), getRHS());
}

void ElementMulInst::verify() const {
  checkSameShape(getDest(), getLHS());
  checkSameShape(getDest(), getRHS());
}

void ElementSubInst::verify() const {
  checkSameType(getDest(), getLHS());
  checkSameType(getDest(), getRHS());
}

void BatchedAddInst::verify() const {
  auto batchShape = getBatch()->dims();
  auto rhsShape = getSlice()->dims();
  assert(batchShape.drop_front() == rhsShape && "Invalid shape");
  assert(getBatch()->dims() == getDest()->dims() && "Invalid dest type");
  (void)batchShape;
  (void)rhsShape;
  assert(getBatch()->getType()->getElementType() ==
             getSlice()->getType()->getElementType() &&
         "Mismatched element types");
}

void BatchedReduceAddInst::verify() const {
  assert(getBatch()->dims().size() > 1 && "Invalid shape");
}

void ElementDivInst::verify() const {
  checkSameShape(getDest(), getLHS());
  checkSameShape(getDest(), getRHS());
}

void ElementMaxInst::verify() const {
  checkSameShape(getDest(), getLHS());
  checkSameShape(getDest(), getRHS());
}

void ElementMinInst::verify() const {
  checkSameShape(getDest(), getLHS());
  checkSameShape(getDest(), getRHS());
}

void ElementCmpLTEInst::verify() const {
  checkSameShape(getDest(), getLHS());
  checkSameShape(getDest(), getRHS());
}

void ElementSelectInst::verify() const {
  checkSameShape(getDest(), getCond());
  checkSameShape(getDest(), getLHS());
  checkSameShape(getDest(), getRHS());
}

void AllocActivationInst::verify() const {
  unsigned numDealloc = 0;
  for (const Use &U : getUsers()) {
    numDealloc += isa<DeallocActivationInst>(U.get());
  }

  // Make sure that there is exactly one user is a deallocation.
  assert(numDealloc == 1 && "Invalid number of tensor deallocation");
}

void DeallocActivationInst::verify() const {
  // The operand of this instruction needs to be an AllocActivationInst.
  assert(isa<AllocActivationInst>(getSrc()) && "Invalid operand");
}

void QuantizationProfileInst::verify() const {
  assert(getInputTensor()->getElementType() == ElemKind::FloatTy &&
         "Floating point type is expected");

  assert(getComputationInfo()->dims().size() == 1 &&
         "Computation info should be 1 dimensional");
  assert(getComputationInfo()->dims()[0] == 2 &&
         "Computation info should contain Min and Max value only");
}

void QuantizeInst::verify() const {
  checkType(getDest(), ElemKind::Int8QTy);
  checkType(getSrc(), ElemKind::FloatTy);
  checkSameShape(getDest(), getSrc());
}

void DequantizeInst::verify() const {
  checkType(getDest(), ElemKind::FloatTy);
  checkType(getSrc(), ElemKind::Int8QTy);
  checkSameShape(getDest(), getSrc());
}

void RescaleQuantizedInst::verify() const {
  checkType(getDest(), ElemKind::Int8QTy);
  checkType(getSrc(), ElemKind::Int8QTy);
  checkSameShape(getDest(), getSrc());
}

void TopKInst::verify() const {
  assert(getValues()->getElementType() == ElemKind::FloatTy);
  assert(getInput()->getElementType() == ElemKind::FloatTy);
  assert(getValues()->dims() == getIndices()->dims());
}

void GatherInst::verify() const {
  assert(getDest()->getElementType() == getData()->getElementType());
  assert(getIndices()->getElementType() == ElemKind::IndexTy);
  assert(getDest()->dims().size() ==
         getData()->dims().size() + getIndices()->dims().size() - 1);
}

void IntrinsicInst::verify() const {
  assert(getName().size() && "Name must not be empty");
}

void DebugPrintInst::verify() const {
  // Nothing to verify.
}
