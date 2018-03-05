// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/IR/Instrs.h"
#include "glow/IR/IR.h"
#include "glow/Support/Support.h"
#include "glow/Verification/Verification.h"

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

void CopyInst::verify() const {
  auto *dest = getDest();
  auto *src = getSrc();
  checkSameType(*dest, *src);
  // The operands of the copy instruction must be variables.
  assert(isa<AllocActivationInst>(dest) || isa<WeightVar>(dest) ||
         isa<TensorViewInst>(dest));
  assert(isa<AllocActivationInst>(src) || isa<WeightVar>(src) ||
         isa<TensorViewInst>(src));
}

void ConvolutionInst::verify() const {
  verifyConvolution(*getSrc(), *getDest(), *getFilter(), *getBias(), Kernel_,
                    Stride_, Pad_, Depth_);
}

void ConvolutionGradInst::verify() const {
  verifyConvolution(*getSrcGrad(), *getDestGrad(), *getFilterGrad(),
                    *getBiasGrad(), Kernel_, Stride_, Pad_, Depth_);
}

void PoolMaxInst::verify() const {
  verifyPool(*getSrc(), *getDest(), Kernel_, Stride_, Pad_);
}

void PoolMaxWithXYInst::verify() const {
  verifyPoolMaxWithXY(*getSrc(), *getDest(), *getSrcXY(), Kernel_, Stride_,
                      Pad_);
}

void PoolMaxWithXYGradInst::verify() const {
  verifyPoolMaxWithXY(*getSrcGrad(), *getDestGrad(), *getSrcXY(), Kernel_,
                      Stride_, Pad_);
}

void PoolAvgInst::verify() const {
  verifyPool(*getSrc(), *getDest(), Kernel_, Stride_, Pad_);
}

void PoolAvgGradInst::verify() const {
  verifyPool(*getSrcGrad(), *getDestGrad(), Kernel_, Stride_, Pad_);
}

void BatchedMatMulInst::verify() const {
  verifyBatchedMatMul(*getDest(), *getLHS(), *getRHS());
}

void SigmoidInst::verify() const { verifySigmoid(*getSrc(), *getDest()); }

void TanhInst::verify() const { verifyTanh(*getSrc(), *getDest()); }

void SoftMaxInst::verify() const { verifySoftMax(*getSrc(), *getDest()); }

void SoftMaxGradInst::verify() const {
  verifySoftMaxGrad(*getOrigSrc(), *getOrigDest(), *getSrcGrad());
}

void CrossEntropyLossInst::verify() const {
  verifyCrossEntropyLoss(*getP(), *getCE(), *getLabels());
}

void CrossEntropyLossGradInst::verify() const {
  verifyCrossEntropyLoss(*getPgrad(), *getCEGrad(), *getLabelsgrad());
}

void ReshapeInst::verify() const { verifyReshape(*getSrc(), *getDest()); }

void TensorViewInst::verify() const { verifyTensorView(*getSrc(), getType()); }

void TransposeInst::verify() const {
  verifyTranspose(*getSrc(), *getDest(), getShuffle());
}

void BroadcastInst::verify() const {
  verifyBroadcast(*getSrc(), *getDest(), getShape());
}

void SplatInst::verify() const {}

void InsertTensorInst::verify() const {
  verifyInsertTensor(*getSrc(), *getDest(), getOffsets());
}

void ExtractTensorInst::verify() const {
  verifyExtractTensor(*getSrc(), *getDest(), getOffsets());
}

void BatchNormalizationInst::verify() const {
  verifyBatchNormalization(*getSrc(), *getDest(), *getScale(), *getBias(),
                           *getMean(), *getVar(), ChannelIdx_);
}

void BatchNormalizationGradInst::verify() const {
  verifyBatchNormalization(*getSrcGrad(), *getDestGrad(), *getScaleGrad(),
                           *getBiasGrad(), *getMean(), *getVar(), ChannelIdx_);
}

void LocalResponseNormalizationInst::verify() const {
  verifyLocalResponseNormalization(*getSrc(), *getDest(), *getScale());
}

void LocalResponseNormalizationGradInst::verify() const {
  verifyLocalResponseNormalization(*getSrcGrad(), *getDestGrad(), *getScale());
}

void BatchedAddInst::verify() const {
  verifyBatchedAdd(*getDest(), *getBatch(), *getSlice());
}

void BatchedReduceAddInst::verify() const {
  verifyBatchedReduceAdd(*getBatch());
}

#define VERIFY_ARITHMETIC(INST_NAME_)                                          \
  void INST_NAME_##Inst::verify() const {                                      \
    verifyArithmetic(*getLHS(), *getRHS(), *getDest());                        \
  }
VERIFY_ARITHMETIC(ElementAdd);
VERIFY_ARITHMETIC(ElementMul);
VERIFY_ARITHMETIC(ElementSub);
VERIFY_ARITHMETIC(ElementDiv);
VERIFY_ARITHMETIC(ElementMax);
VERIFY_ARITHMETIC(ElementMin);
VERIFY_ARITHMETIC(ElementCmpLTE);
#undef VERIFY_ARITHMETIC

void ElementSelectInst::verify() const {
  verifySelect(*getDest(), *getCond(), *getLHS(), *getRHS());
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
  verifyQuantizationProfile(*getInputTensor(), *getComputationInfo());
}

void QuantizeInst::verify() const { verifyQuantize(*getSrc(), *getDest()); }

void DequantizeInst::verify() const { verifyDequantize(*getSrc(), *getDest()); }

void RescaleQuantizedInst::verify() const {
  verifyRescaleQuantized(*getSrc(), *getDest());
}

void TopKInst::verify() const {
  verifyTopK(*getInput(), *getValues(), *getIndices());
}

void GatherInst::verify() const {
  verifyGather(*getDest(), *getData(), *getIndices());
}

void IntrinsicInst::verify() const { verifyIntrinsic(getName()); }

void DebugPrintInst::verify() const {
  // Nothing to verify.
}
