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
  os << "%" << (std::string)getName() << " = WeightVar ";
  os << std::to_string(*getType()) << " " << getMutabilityStr();
}

//===----------------------------------------------------------------------===//
//                       Instruction verification
//===----------------------------------------------------------------------===//

/// Check that the type of the first operand matches the type of the second
/// operand.
static void checkSameType(Instruction::Operand A, Instruction::Operand B) {
  assert(A.first->getType() == B.first->getType() && "Invalid type");
}

void CopyInst::verify() const {
  checkSameType(getOperand(0), getOperand(1));
  auto *op0 = getOperand(0).first;
  auto *op1 = getOperand(1).first;
  (void)op0;
  (void)op1;
  // The operands of the copy instruction must be variables.
  assert(isa<AllocActivationInst>(op0) || isa<WeightVar>(op0));
  assert(isa<AllocActivationInst>(op1) || isa<WeightVar>(op1));
}
void ConvolutionInst::verify() const {
  Value *dest = getOperand(0).first;
  Value *src = getOperand(1).first;
  Value *filter = getOperand(2).first;
  Value *bias = getOperand(3).first;
  (void)filter;
  (void)bias;

  ShapeNHWC idim(src->getType()->dims());
  ShapeNHWC odim(dest->getType()->dims());
  (void)odim;
  assert(idim.w >= Kernel_ && idim.h >= Kernel_ &&
         "buffer too small for selected stride");

  auto outSz = calculateConvOutputDims(idim.h, idim.w, Pad_, Kernel_, Stride_);
  ShapeNHWC exp(idim.n, outSz.first, outSz.second, Depth_);
  (void)exp;
  assert(exp == odim && "Invalid output dimensions");

  llvm::ArrayRef<size_t> filterDims = {Depth_, Kernel_, Kernel_, idim.c};
  assert(filter->getType()->dims() == filterDims && "Invalid filter dims");

  llvm::ArrayRef<size_t> biasDims = {Depth_};
  assert(bias->getType()->dims() == biasDims && "Invalid bias dims");
}

void PoolMaxInst::verify() const {
  Value *dest = getOperand(0).first;
  Value *src = getOperand(1).first;
  Value *srcXY = getOperand(2).first;
  (void)srcXY;
  ShapeNHWC idim = ShapeNHWC(src->getType()->dims());
  ShapeNHWC odim = ShapeNHWC(dest->getType()->dims());
  (void)odim;
  assert(idim.w >= Kernel_ && idim.h >= Kernel_ &&
         "buffer too small for selected stride");

  auto outSz = calculateConvOutputDims(idim.h, idim.w, Pad_, Kernel_, Stride_);
  ShapeNHWC exp(idim.n, outSz.first, outSz.second, idim.c);
  (void)exp;
  assert(exp == odim && "Unexpected output dimensions");

  // Allocate cache arrays that store the x and y coordinates of the incoming
  // gradient for each max element.
  llvm::ArrayRef<size_t> E = {idim.n, outSz.first, outSz.second, idim.c, 2};
  assert(srcXY->getType()->dims() == E && "Invalid srcXY dims");
}

void PoolAvgInst::verify() const {
  Value *dest = getOperand(0).first;
  Value *src = getOperand(1).first;
  ShapeNHWC idim = ShapeNHWC(src->getType()->dims());
  ShapeNHWC odim = ShapeNHWC(dest->getType()->dims());
  (void)odim;
  assert(idim.w >= Kernel_ && idim.h >= Kernel_ &&
         "buffer too small for selected stride");

  auto outSz = calculateConvOutputDims(idim.h, idim.w, Pad_, Kernel_, Stride_);
  ShapeNHWC exp(idim.n, outSz.first, outSz.second, idim.c);
  (void)exp;
  assert(exp == odim && "Unexpected output dimensions");
}

void FullyConnectedInst::verify() const {
  Value *dest = getOperand(0).first;
  Value *src = getOperand(1).first;
  Value *W = getOperand(2).first;
  Value *B = getOperand(3).first;
  (void)dest;
  (void)W;
  (void)B;
  auto idim = flattenCdr(src->dims());

  llvm::ArrayRef<size_t> exp = {idim.first, Depth_};
  assert(dest->dims() == exp && "Invalid output shape");
  (void)exp;

  llvm::ArrayRef<size_t> expW = {Depth_, idim.second};
  assert(W->dims() == expW && "Invalid output shape");
  (void)expW;

  llvm::ArrayRef<size_t> expB = {Depth_};
  assert(B->dims() == expB && "Invalid output shape");
  (void)expB;

  assert(src->dims().size() == 2 &&
         "Src of a FullyConnectedInst should be 2-dimensional");
  assert(dest->dims().size() == 2 &&
         "Dest of a FullyConnectedInst should be 2-dimensional");
}

void BatchedMatMulInst::verify() const {
  Value *dest = getDest();
  Value *lhs = getLHS();
  Value *rhs = getRHS();
  (void)dest;
  (void)lhs;
  (void)rhs;

  auto LDims = lhs->dims();
  auto RDims = rhs->dims();
  auto DDims = dest->dims();
  assert(LDims.size() == 3);
  assert(RDims.size() == 3);
  assert(DDims.size() == 3);
  auto elem = dest->getType()->getElementType();
  (void)elem;
  assert(lhs->getType()->getElementType() == elem);
  assert(rhs->getType()->getElementType() == elem);

  size_t a0 = LDims[0];
  size_t a1 = LDims[1];
  size_t a2 = LDims[2];

  size_t b0 = RDims[0];
  size_t b1 = RDims[1];
  size_t b2 = RDims[2];

  size_t c0 = DDims[0];
  size_t c1 = DDims[1];
  size_t c2 = DDims[2];

  assert(a0 == 1 || b0 == 1 ||
         a0 == b0 && "Batch size must be broadcasted or identical");

  // Select the batch size. If the left operand is broadcast (value 1), select
  // the RHS.
  size_t N = (a0 != 1 ? a0 : b0);
  assert(N == c0);

  assert(a1 == b2 && "Column of LHS is not equal to the row of RHS.");

  assert(a1 == b2 && "Column of A is not equal to the row of A.");
  assert(c1 == a2 && c2 == b1 && "Invalid size of output matrix");
  (void)a0;
  (void)a1;
  (void)a2;
  (void)b0;
  (void)b1;
  (void)b2;
  (void)c0;
  (void)c1;
  (void)c2;
}

void ReluInst::verify() const { checkSameType(getOperand(0), getOperand(1)); }
void SigmoidInst::verify() const {
  checkSameType(getOperand(0), getOperand(1));
}
void TanhInst::verify() const { checkSameType(getOperand(0), getOperand(1)); }
void SoftMaxInst::verify() const {
  checkSameType(getOperand(0), getOperand(1));
}

void ReshapeInst::verify() const {
  assert(getOperand(0).first->getType()->size() ==
             getOperand(1).first->getType()->size() &&
         "Reshape into a different size");
}

void TensorViewInst::verify() const {
  assert(getOperand(0).first->getType()->size() == getType()->size() &&
         "TensorView view size should be the same as Src size");
  assert(getOperand(0).first->getElementType() == getType()->getElementType() &&
         "TensorView view element type should be the same as Src size");
}

void TransposeInst::verify() const {
  auto *dest = getOperand(0).first;
  auto *src = getOperand(1).first;
  (void)dest;
  llvm::SmallVector<size_t, 6> shape;

  auto dims = src->dims();
  for (size_t i = 0; i < dims.size(); i++) {
    shape.push_back(dims[Shuffle_[i]]);
  }

  assert(dest->dims() == llvm::ArrayRef<size_t>(shape) &&
         "Invalid transpose dims");
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
  checkSameType(getOperand(0), getOperand(1));

  // Figure out how many channels are in the tensor.
  size_t channels = getOperand(0).first->dims()[ChannelIdx_];

  llvm::ArrayRef<size_t> exp = {channels};
  assert(getOperand(2).first->getType()->dims() == exp && "Invalid bias dim");
  assert(getOperand(3).first->getType()->dims() == exp && "Invalid scale dim");
  assert(getOperand(4).first->getType()->dims() == exp && "Invalid mean dim");
  assert(getOperand(5).first->getType()->dims() == exp && "Invalid var dim");
}

void LocalResponseNormalizationInst::verify() const {
  checkSameType(getOperand(0), getOperand(1));
  checkSameType(getOperand(0), getOperand(2));
}

void ElementAddInst::verify() const {
  checkSameType(getOperand(0), getOperand(1));
  checkSameType(getOperand(0), getOperand(2));
}

void ElementMulInst::verify() const {
  checkSameType(getOperand(0), getOperand(1));
  checkSameType(getOperand(0), getOperand(2));
}

void ElementSubInst::verify() const {
  checkSameType(getOperand(0), getOperand(1));
  checkSameType(getOperand(0), getOperand(2));
}

void BatchedAddInst::verify() const {
  auto batchShape = getBatch()->dims();
  auto rhsShape = getSlice()->dims();
  assert(batchShape.drop_front() == rhsShape && "Invalid shape");
  assert(getBatch()->dims() == getDest()->dims() && "Invalid dest type");
  (void)batchShape;
  (void)rhsShape;
}

void BatchedReduceAddInst::verify() const {
  assert(getBatch()->dims().size() > 1 && "Invalid shape");
}

void ElementDivInst::verify() const {
  checkSameType(getOperand(0), getOperand(1));
  checkSameType(getOperand(0), getOperand(2));
}

void AllocActivationInst::verify() const {
  unsigned numDealloc = 0;
  for (const Use &U : getUsers()) {
    numDealloc += isa<DeallocActivationInst>(U.get());
  }

  // Make sure that there is exactly one user is a deallocation.
  assert(numDealloc == 1 && "Invalid number of tensor deallocation");
}

void SGDInst::verify() const {
  if (Momentum_ > 0.0) {
    assert(getGradient()->getType() == getGsum()->getType() &&
           "Invalid gsum type");
  }

  assert(getGradient()->getType() == getWeight()->getType() &&
         "Invalid weight or gradient type");
}

void DeallocActivationInst::verify() const {
  // The operand of this instruction needs to be an AllocActivationInst.
  assert(isa<AllocActivationInst>(getOperand(0).first) && "Invalid operand");
}

// TODO: verify the gradient instructions.
#define NOVERIFY(ClassName)                                                    \
  void ClassName ::verify() const {}
NOVERIFY(ConvolutionGradInst)
NOVERIFY(PoolMaxGradInst)
NOVERIFY(PoolAvgGradInst)
NOVERIFY(FullyConnectedGradInst)
NOVERIFY(BatchNormalizationGradInst)
NOVERIFY(LocalResponseNormalizationGradInst)
NOVERIFY(SoftMaxGradInst)
NOVERIFY(ReluGradInst)
NOVERIFY(TanhGradInst)
NOVERIFY(SigmoidGradInst)
NOVERIFY(ElementAddGradInst)
NOVERIFY(ElementMulGradInst)
NOVERIFY(ElementSubGradInst)
NOVERIFY(DebugPrintInst)
NOVERIFY(ElementDivGradInst)
#undef NOVERIFY
