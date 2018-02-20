// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"

#include "glow/IR/IRBuilder.h"

using namespace glow;
using llvm::dyn_cast;
using llvm::isa;

IRBuilder::~IRBuilder() { deallocateActiveInstrs(); }

static bool hasDeallocas(AllocActivationInst *AA) {
  for (auto &U : AA->getUsers()) {
    if (isa<DeallocActivationInst>(U.get())) {
      return true;
    }
  }
  return false;
}

void IRBuilder::deallocateActiveInstrs() {
  auto &instrs = F_->getInstrs();
  // Inserts dealloc instructions for all instructions that don't have
  // 'dealloc' as one of their users.
  for (auto it = instrs.begin(), e = instrs.end(); it != e; ++it) {
    auto AA = dyn_cast<AllocActivationInst>(*it);
    if (!AA) {
      continue;
    }

    if (hasDeallocas(AA)) {
      continue;
    }

    createDeallocActivationInst("dealloc", AA);
  }
}

//===----------------------------------------------------------------------===//
//                        High level operators.
//===----------------------------------------------------------------------===//

PoolMaxInst *IRBuilder::createPoolMaxOp(Value *input, size_t kernel,
                                        size_t stride, size_t pad) {
  ShapeNHWC idim = ShapeNHWC(input->dims());
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  auto outSz = calculateConvOutputDims(idim.h, idim.w, kernel, stride, pad);

  Value *dest =
      createAllocActivationInst("pool.res", ElemKind::FloatTy,
                                {idim.n, outSz.first, outSz.second, idim.c});

  return createPoolMaxInst("pool", dest, input, kernel, stride, pad);
}

PoolMaxWithXYInst *IRBuilder::createPoolMaxWithXYOp(Value *input, size_t kernel,
                                                    size_t stride, size_t pad) {
  ShapeNHWC idim = ShapeNHWC(input->dims());
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  auto outSz = calculateConvOutputDims(idim.h, idim.w, kernel, stride, pad);

  // Allocate cache arrays that store the x and y coordinates of the incoming
  // gradient for each max element.
  Value *srcXY =
      createAllocActivationInst("srcXY", ElemKind::IndexTy,
                                {idim.n, outSz.first, outSz.second, idim.c, 2});
  Value *dest =
      createAllocActivationInst("pool.res", ElemKind::FloatTy,
                                {idim.n, outSz.first, outSz.second, idim.c});

  return createPoolMaxWithXYInst("pool", dest, input, srcXY, kernel, stride,
                                 pad);
}
PoolAvgInst *IRBuilder::createPoolAvgOp(Value *input, size_t kernel,
                                        size_t stride, size_t pad) {
  ShapeNHWC idim = ShapeNHWC(input->dims());
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  auto outSz = calculateConvOutputDims(idim.h, idim.w, kernel, stride, pad);

  Value *dest =
      createAllocActivationInst("pool.res", ElemKind::FloatTy,
                                {idim.n, outSz.first, outSz.second, idim.c});

  return createPoolAvgInst("pool", dest, input, kernel, stride, pad);
}

SigmoidInst *IRBuilder::createSigmoidOp(Value *input) {
  auto *res = createAllocActivationInst("sigmoid.res", input->getType());
  return createSigmoidInst("sigmoid", res, input);
}

TanhInst *IRBuilder::createTanhOp(Value *input) {
  auto *res = createAllocActivationInst("tanh.res", input->getType());
  return createTanhInst("tanh", res, input);
}

SoftMaxInst *IRBuilder::createSoftMaxOp(Value *input) {
  auto *res = createAllocActivationInst("softmax.res", input->getType());
  return createSoftMaxInst("softmax", res, input);
}

CrossEntropyLossInst *IRBuilder::createCrossEntropyLossOp(Value *p,
                                                          Value *labels) {
  auto *res = createAllocActivationInst("celoss.res", ElemKind::FloatTy, {1});
  return createCrossEntropyLossInst("celoss", p, labels, res);
}

ReshapeInst *IRBuilder::createReshapeOp(Value *input,
                                        llvm::ArrayRef<size_t> shape) {
  auto ty = F_->getGraph()->getParent()->uniqueTypeWithNewShape(
      input->getType(), shape);
  auto *res = createAllocActivationInst("reshape.res", ty);
  return createReshapeInst("reshape", res, input, shape);
}

/// Creates a tensorview instruction with the following parameters:
/// \param elemKind the type of elements in a tensor
/// \param dims dimensions of the view, such that the number of elements
/// in the view is the same as the number of elements in the source tensor
/// \p src
/// \param src the source tensor used to create the unowned tensor.
TensorViewInst *IRBuilder::createTensorView(ElemKind elemKind,
                                            llvm::ArrayRef<size_t> dims,
                                            Value *src, llvm::StringRef name) {
  auto ty =
      getIRFunction().getGraph()->getParent()->uniqueType(Type(elemKind, dims));
  return createTensorViewInst(name, src, ty);
}

TransposeInst *IRBuilder::createTransposeOp(Value *input,
                                            llvm::ArrayRef<unsigned> shuffle) {
  llvm::SmallVector<size_t, 6> shape;
  auto dims = input->dims();
  for (size_t i = 0; i < dims.size(); i++) {
    shape.push_back(dims[shuffle[i]]);
  }

  auto ty = F_->getGraph()->getParent()->uniqueTypeWithNewShape(
      input->getType(), shape);
  auto *res = createAllocActivationInst("transp.res", ty);
  return createTransposeInst("transp", res, input, shuffle);
}

BroadcastInst *IRBuilder::createBroadcastOp(Value *input,
                                            llvm::ArrayRef<size_t> shape,
                                            unsigned axis) {
  auto *res = createAllocActivationInst("broadcast.res",
                                        input->getElementType(), shape);
  return createBroadcastInst("broadcast", res, input, shape, axis);
}

BatchNormalizationInst *IRBuilder::createBatchNormalizationOp(
    Value *input, Value *beta, Value *gamma, Value *mean, Value *var,
    size_t channelIdx, float epsilon, float momentum) {
  // The output tensor is of the same shape as the input tensor.
  auto *dest = createAllocActivationInst("BN.res", input->getType());

  return createBatchNormalizationInst("BN", dest, input, gamma, beta, mean, var,
                                      channelIdx, epsilon, momentum);
}

LocalResponseNormalizationInst *IRBuilder::createLocalResponseNormalizationOp(
    Value *input, size_t halfWindowSize, float alpha, float beta, float k) {
  auto ty = input->getType();
  auto *scale = createAllocActivationInst("scale", ty);

  // The output tensor is of the same shape as the input tensor.
  auto *res = createAllocActivationInst("LRN.res", ty);
  return createLocalResponseNormalizationInst("LRN", res, input, scale,
                                              halfWindowSize, alpha, beta, k);
}

ElementAddInst *IRBuilder::createElementAddOp(Value *lhs, Value *rhs) {
  assert(lhs->dims() == rhs->dims() && "Invalid operand shapes");
  // The output tensor is of the same shape as the input tensor.
  auto *res = createAllocActivationInst("add.res", lhs->getType());
  return createElementAddInst("add", res, lhs, rhs);
}

ElementSubInst *IRBuilder::createElementSubOp(Value *lhs, Value *rhs) {
  assert(lhs->dims() == rhs->dims() && "Invalid operand shapes");
  // The output tensor is of the same shape as the input tensor.
  auto *res = createAllocActivationInst("sub.res", lhs->getType());
  return createElementSubInst("sub", res, lhs, rhs);
}

ElementMulInst *IRBuilder::createElementMulOp(Value *lhs, Value *rhs) {
  assert(lhs->dims() == rhs->dims() && "Invalid operand shapes");
  // The output tensor is of the same shape as the input tensor.
  auto *res = createAllocActivationInst("mul.res", lhs->getType());
  return createElementMulInst("mul", res, lhs, rhs);
}

ElementDivInst *IRBuilder::createElementDivOp(Value *lhs, Value *rhs) {
  assert(lhs->dims() == rhs->dims() &&
         "Input and Output dimensions are different");

  auto *res = createAllocActivationInst("div.res", lhs->getType());
  return createElementDivInst("div", res, lhs, rhs);
}

ElementMaxInst *IRBuilder::createElementMaxOp(Value *lhs, Value *rhs) {
  assert(lhs->dims() == rhs->dims() &&
         "Input and Output dimensions are different");

  auto *res = createAllocActivationInst("max.res", lhs->getType());
  return createElementMaxInst("max", res, lhs, rhs);
}

ElementMinInst *IRBuilder::createElementMinOp(Value *lhs, Value *rhs) {
  assert(lhs->dims() == rhs->dims() &&
         "Input and Output dimensions are different");

  auto *res = createAllocActivationInst("min.res", lhs->getType());
  return createElementMinInst("min", res, lhs, rhs);
}

ElementCmpLTEInst *IRBuilder::createElementCmpLTEOp(Value *lhs, Value *rhs) {
  assert(lhs->dims() == rhs->dims() &&
         "Input and Output dimensions are different");
  auto *res = createAllocActivationInst("cmp.lte.res", lhs->getType());
  return createElementCmpLTEInst("cmp.lte", res, lhs, rhs);
}

ElementSelectInst *IRBuilder::createSelectOp(Value *cond, Value *lhs,
                                             Value *rhs) {
  assert(lhs->dims() == rhs->dims() &&
         "Input and Output dimensions are different");
  assert(cond->dims() == rhs->dims() &&
         "Input and Output dimensions are different");
  auto *res = createAllocActivationInst("select.res", lhs->getType());
  return createElementSelectInst("select", res, cond, lhs, rhs);
}

TopKInst *IRBuilder::createTopKOp(Value *input, size_t k) {
  auto inDims = input->dims();
  assert(inDims.size() > 0);
  assert(k <= inDims.back());
  llvm::SmallVector<size_t, 6> outDims(inDims.begin(), inDims.end());
  outDims.back() = k;
  auto *values = createAllocActivationInst("topk.values",
                                           input->getElementType(), outDims);
  auto *indices =
      createAllocActivationInst("topk.indices", ElemKind::IndexTy, outDims);
  return createTopKInst("topk", values, indices, input, k);
}

Value *IRBuilder::createReturnOp(Value *input) {
  auto *W = createWeightVar(input->getType(), "result",
                            WeightVar::MutabilityKind::Mutable);
  createCopyInst("return", W, input);
  return W;
}

GatherInst *IRBuilder::createGatherOp(Value *data, Value *indices) {
  auto dDims = data->dims();
  auto iDims = indices->dims();
  assert(dDims.size() > 0);
  llvm::SmallVector<size_t, 6> outDims(iDims.begin(), iDims.end());
  outDims.insert(outDims.end(), dDims.begin() + 1, dDims.end());
  auto *res =
      createAllocActivationInst("gather.res", data->getElementType(), outDims);
  return createGatherInst("gather", res, data, indices);
}

//===----------------------------------------------------------------------===//
//                     Low level instructions.
//===----------------------------------------------------------------------===//

WeightVar *IRBuilder::createWeightVar(ElemKind elemTy,
                                      llvm::ArrayRef<size_t> dims,
                                      llvm::StringRef name,
                                      WeightVar::MutabilityKind k) {
  auto T = F_->getGraph()->getParent()->uniqueType(elemTy, dims);
  return createWeightVar(T, name, k);
}

WeightVar *IRBuilder::createWeightVar(TypeRef T, llvm::StringRef name,
                                      WeightVar::MutabilityKind k) {
  auto *A = new WeightVar(name, T, k);
  F_->getWeights().push_back(A);
  A->setName(name);
  return A;
}

AllocActivationInst *
IRBuilder::createAllocActivationInst(llvm::StringRef name, ElemKind elemTy,
                                     llvm::ArrayRef<size_t> dims) {
  auto T = F_->getGraph()->getParent()->uniqueType(elemTy, dims);
  return createAllocActivationInst(name, T);
}
