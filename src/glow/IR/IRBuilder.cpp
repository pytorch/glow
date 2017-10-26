// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"

#include "glow/IR/IRBuilder.h"

using namespace glow;

IRBuilder::~IRBuilder() { deallocateActiveInstrs(); }

void IRBuilder::deallocateActiveInstrs() {
  for (auto *A : activeAllocs_) {
    createDeallocActivationInst(A);
  }

  activeAllocs_.clear();
}

//===----------------------------------------------------------------------===//
//                        High level operators.
//===----------------------------------------------------------------------===//

ConvolutionInst *IRBuilder::createConvOp(Value *input, Value *filter,
                                         Value *bias, size_t depth,
                                         size_t kernel, size_t stride,
                                         size_t pad) {
  ShapeNHWC idim = ShapeNHWC(input->dims());
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  // Calculate the size and allocate the output buffer.
  auto outSz = calculateConvOutputDims(idim.h, idim.w, pad, kernel, stride);

  std::vector<size_t> outDims = {idim.n, outSz.first, outSz.second, depth};

  Value *dest = createAllocActivationInst(ElemKind::FloatTy, outDims);

  return createConvolutionInst(dest, input, filter, bias, kernel, stride, pad,
                               depth);
}

PoolMaxInst *IRBuilder::createPoolMaxOp(Value *input, size_t kernel,
                                        size_t stride, size_t pad) {
  ShapeNHWC idim = ShapeNHWC(input->dims());
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  auto outSz = calculateConvOutputDims(idim.h, idim.w, pad, kernel, stride);

  // Allocate cache arrays that store the x and y coordinates of the incoming
  // gradient for each max element.
  Value *srcXY = createAllocActivationInst(
      ElemKind::IndexTy, {idim.n, outSz.first, outSz.second, idim.c, 2},
      "srcXY");
  Value *dest = createAllocActivationInst(
      ElemKind::FloatTy, {idim.n, outSz.first, outSz.second, idim.c});

  return createPoolMaxInst(dest, input, srcXY, kernel, stride, pad);
}

PoolAvgInst *IRBuilder::createPoolAvgOp(Value *input, size_t kernel,
                                        size_t stride, size_t pad) {
  ShapeNHWC idim = ShapeNHWC(input->dims());
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  auto outSz = calculateConvOutputDims(idim.h, idim.w, pad, kernel, stride);

  Value *dest = createAllocActivationInst(
      ElemKind::FloatTy, {idim.n, outSz.first, outSz.second, idim.c});

  return createPoolAvgInst(dest, input, kernel, stride, pad);
}

FullyConnectedInst *IRBuilder::createFullyConnectedOp(Value *input,
                                                      Value *filter,
                                                      Value *bias,
                                                      size_t outDepth) {
  TypeRef T = input->getType();
  auto idim = flattenCdr(input->dims());

  auto *dest =
      createAllocActivationInst(T->getElementType(), {idim.first, outDepth});

  return createFullyConnectedInst(dest, input, filter, bias, outDepth);
}

ReluInst *IRBuilder::createRELUOp(Value *input) {
  auto *res = createAllocActivationInst(input->getType());
  return createReluInst(res, input);
}

SigmoidInst *IRBuilder::createSigmoidOp(Value *input) {
  auto *res = createAllocActivationInst(input->getType());
  return createSigmoidInst(res, input);
}

TanhInst *IRBuilder::createTanhOp(Value *input) {
  auto *res = createAllocActivationInst(input->getType());
  return createTanhInst(res, input);
}

SoftMaxInst *IRBuilder::createSoftMaxOp(Value *input, Value *selected) {
  auto *res = createAllocActivationInst(input->getType());
  auto *E = createAllocActivationInst(input->getType(), "e_cache");
  return createSoftMaxInst(res, input, E, selected);
}

RegressionInst *IRBuilder::createRegressionOp(Value *input, Value *expected) {
  auto *res = createAllocActivationInst(input->getType());
  return createRegressionInst(res, input, expected);
}

ReshapeInst *IRBuilder::createReshapeOp(Value *input,
                                        llvm::ArrayRef<size_t> shape) {
  auto *res = createAllocActivationInst(input->getElementType(), shape);
  return createReshapeInst(res, input, shape);
}

TransposeInst *IRBuilder::createTransposeOp(Value *input,
                                            llvm::ArrayRef<unsigned> shuffle) {
  std::vector<size_t> shape;
  auto dims = input->dims();
  for (size_t i = 0; i < dims.size(); i++) {
    shape.push_back(dims[shuffle[i]]);
  }

  auto *res = createAllocActivationInst(input->getElementType(), shape);
  return createTransposeInst(res, input, shuffle);
}

ConcatInst *IRBuilder::createConcatOp(Value *LHS, Value *RHS,
                                      unsigned dimension) {
  assert(LHS->getType() == RHS->getType() && "Invalid dims");
  auto inDim = LHS->dims();

  std::vector<size_t> shape(inDim.begin(), inDim.end());
  // We are stacking the tensors along a specific dimension. This means that we
  // increase the size of the tensor along this dimension.
  shape[dimension] *= 2;

  auto *res = createAllocActivationInst(LHS->getElementType(), shape);
  return createConcatInst(res, LHS, RHS, dimension);
}

BatchNormalizationInst *IRBuilder::createBatchNormalizationOp(
    Value *input, Value *beta, Value *gamma, Value *mean, Value *var,
    size_t channelIdx, float epsilon, float momentum) {
  // The output tensor is of the same shape as the input tensor.
  auto *dest = createAllocActivationInst(input->getType());

  return createBatchNormalizationInst(dest, input, gamma, beta, mean, var,
                                      channelIdx, epsilon, momentum);
}

LocalResponseNormalizationInst *IRBuilder::createLocalResponseNormalizationOp(
    Value *input, size_t halfWindowSize, float alpha, float beta, float k) {
  auto Ty = input->getType();
  auto *scale = createAllocActivationInst(Ty, "scale");

  // The output tensor is of the same shape as the input tensor.
  auto *res = createAllocActivationInst(Ty);
  return createLocalResponseNormalizationInst(input, res, scale, halfWindowSize,
                                              alpha, beta, k);
}

ElementAddInst *IRBuilder::createElementAddOp(Value *LHS, Value *RHS) {
  assert(LHS->dims() == RHS->dims() && "Invalid operand shapes");
  // The output tensor is of the same shape as the input tensor.
  auto *res = createAllocActivationInst(LHS->getType());
  return createElementAddInst(res, LHS, RHS);
}

ElementMulInst *IRBuilder::createElementMulOp(Value *LHS, Value *RHS) {
  assert(LHS->dims() == RHS->dims() && "Invalid operand shapes");
  // The output tensor is of the same shape as the input tensor.
  auto *res = createAllocActivationInst(LHS->getType());
  return createElementMulInst(res, LHS, RHS);
}

Value *IRBuilder::createReturnOp(Value *input) {
  auto *W = createWeightVar(input->getType(), "result",
                            WeightVar::MutabilityKind::Mutable);
  createCopyInst(W, input);
  return W;
}

//===----------------------------------------------------------------------===//
//                     Low level instructions.
//===----------------------------------------------------------------------===//

CopyInst *IRBuilder::createCopyInst(Value *dest, Value *src) {
  auto *A = new CopyInst("", dest, src);
  M_->pushInstr(A);
  return A;
}

ConvolutionInst *IRBuilder::createConvolutionInst(Value *dest, Value *src,
                                                  Value *filter, Value *bias,
                                                  size_t kernel, size_t stride,
                                                  size_t pad, size_t depth) {
  auto *A = new ConvolutionInst("", dest, src, filter, bias, kernel, stride,
                                pad, depth);
  M_->pushInstr(A);
  return A;
}

PoolMaxInst *IRBuilder::createPoolMaxInst(Value *dest, Value *src, Value *srcXY,
                                          size_t kernel, size_t stride,
                                          size_t pad) {
  auto *A = new PoolMaxInst("", dest, src, srcXY, kernel, stride, pad);
  M_->pushInstr(A);
  return A;
}

PoolAvgInst *IRBuilder::createPoolAvgInst(Value *dest, Value *src,
                                          size_t kernel, size_t stride,
                                          size_t pad) {
  auto *A = new PoolAvgInst("", dest, src, kernel, stride, pad);
  M_->pushInstr(A);
  return A;
}

FullyConnectedInst *IRBuilder::createFullyConnectedInst(Value *dest, Value *src,
                                                        Value *filter,
                                                        Value *bias,
                                                        size_t depth) {
  auto *A = new FullyConnectedInst("", dest, src, filter, bias, depth);
  M_->pushInstr(A);
  return A;
}

ReluInst *IRBuilder::createReluInst(Value *dest, Value *src) {
  auto *A = new ReluInst("", dest, src);
  M_->pushInstr(A);
  return A;
}

SigmoidInst *IRBuilder::createSigmoidInst(Value *dest, Value *src) {
  auto *A = new SigmoidInst("", dest, src);
  M_->pushInstr(A);
  return A;
}

TanhInst *IRBuilder::createTanhInst(Value *dest, Value *src) {
  auto *A = new TanhInst("", dest, src);
  M_->pushInstr(A);
  return A;
}

SoftMaxInst *IRBuilder::createSoftMaxInst(Value *dest, Value *src, Value *E,
                                          Value *selected) {
  auto *A = new SoftMaxInst("", dest, src, E, selected);
  M_->pushInstr(A);
  return A;
}

RegressionInst *IRBuilder::createRegressionInst(Value *dest, Value *src,
                                                Value *expected) {
  auto *A = new RegressionInst("", dest, src, expected);
  M_->pushInstr(A);
  return A;
}

ReshapeInst *IRBuilder::createReshapeInst(Value *dest, Value *src,
                                          llvm::ArrayRef<size_t> shape) {
  auto *A = new ReshapeInst("", dest, src, shape.vec());
  M_->pushInstr(A);
  return A;
}

TransposeInst *
IRBuilder::createTransposeInst(Value *dest, Value *src,
                               llvm::ArrayRef<unsigned> shuffle) {
  auto *A = new TransposeInst("", dest, src, shuffle.vec());
  M_->pushInstr(A);
  return A;
}

ConcatInst *IRBuilder::createConcatInst(Value *dest, Value *LHS, Value *RHS,
                                        size_t dim) {
  auto *A = new ConcatInst("", dest, LHS, RHS, dim);
  M_->pushInstr(A);
  return A;
}

BatchNormalizationInst *IRBuilder::createBatchNormalizationInst(
    Value *dest, Value *src, Value *scale, Value *bias, Value *mean, Value *var,
    size_t channelIdx, float epsilon, float momentum) {
  auto *A = new BatchNormalizationInst("", dest, src, scale, bias, mean, var,
                                       channelIdx, epsilon, momentum);
  M_->pushInstr(A);
  return A;
}

LocalResponseNormalizationInst *IRBuilder::createLocalResponseNormalizationInst(
    Value *dest, Value *src, Value *scale, size_t halfWindowSize, float alpha,
    float beta, float k) {
  auto *A = new LocalResponseNormalizationInst("", dest, src, scale,
                                               halfWindowSize, alpha, beta, k);
  M_->pushInstr(A);
  return A;
}

ElementAddInst *IRBuilder::createElementAddInst(Value *dest, Value *LHS,
                                                Value *RHS) {
  auto *A = new ElementAddInst("", dest, LHS, RHS);
  M_->pushInstr(A);
  return A;
}

ElementMulInst *IRBuilder::createElementMulInst(Value *dest, Value *LHS,
                                                Value *RHS) {
  auto *A = new ElementMulInst("", dest, LHS, RHS);
  M_->pushInstr(A);
  return A;
}

WeightVar *IRBuilder::createWeightVar(ElemKind elemTy,
                                      llvm::ArrayRef<size_t> dims,
                                      llvm::StringRef name,
                                      WeightVar::MutabilityKind k) {
  auto T = M_->getGraph()->uniqueType(elemTy, dims);
  return createWeightVar(T, name, k);
}

WeightVar *IRBuilder::createWeightVar(TypeRef T, llvm::StringRef name,
                                      WeightVar::MutabilityKind k) {
  auto *A = new WeightVar(name, T, k);
  M_->getWeights().push_back(A);
  A->setName(name);
  return A;
}

AllocActivationInst *
IRBuilder::createAllocActivationInst(TypeRef T, llvm::StringRef name) {
  auto *A = new AllocActivationInst(name, T);
  M_->pushInstr(A);
  // Add this instruction to the list of open allocations.
  activeAllocs_.push_back(A);
  return A;
}
AllocActivationInst *IRBuilder::createAllocActivationInst(
    ElemKind elemTy, llvm::ArrayRef<size_t> dims, llvm::StringRef name) {
  auto T = M_->getGraph()->uniqueType(elemTy, dims);
  return createAllocActivationInst(T, name);
}

DeallocActivationInst *IRBuilder::createDeallocActivationInst(Value *src) {
  auto *A = new DeallocActivationInst("", src);
  M_->pushInstr(A);
  return A;
}
