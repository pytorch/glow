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

ConvolutionInst *IRBuilder::createConvOp(Value *input, size_t depth,
                                         size_t kernel, size_t stride,
                                         size_t pad) {
  ShapeNHWC idim = ShapeNHWC(input->dims());
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  // Allocate the Filter and Bias tensors.
  std::vector<size_t> filterDim = {depth, kernel, kernel, idim.c};
  size_t fanIn = kernel * kernel * idim.c;
  Value *filter = createWeightVar(ElemKind::FloatTy, filterDim, "filter",
                                  InitKind::Xavier, fanIn);
  Value *bias = createWeightVar(ElemKind::FloatTy, {depth}, "bias",
                                InitKind::Broadcast, 0.1);

  return createConvOp(input, filter, bias, depth, kernel, stride, pad);
}

ConvolutionInst *IRBuilder::createConvOp(Value *input, Value *filter,
                                         Value *bias, size_t depth,
                                         size_t kernel, size_t stride,
                                         size_t pad) {
  ShapeNHWC idim = ShapeNHWC(input->dims());
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  // Calculate the size and allocate the output buffer.
  auto outSz =
      ConvolutionNode::calculateOutputDims(idim.h, idim.w, pad, kernel, stride);

  std::vector<size_t> outDims = {idim.n, outSz.first, outSz.second, depth};

  Value *dest = createAllocActivationInst(ElemKind::FloatTy, outDims);

  return createConvolutionInst(dest, input, filter, bias, kernel, stride, pad,
                               depth);
}

PoolInst *IRBuilder::createPoolOp(Value *input, PoolInst::OpKind kind,
                                  size_t kernel, size_t stride, size_t pad) {
  ShapeNHWC idim = ShapeNHWC(input->dims());
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  auto outSz =
      ConvolutionNode::calculateOutputDims(idim.h, idim.w, pad, kernel, stride);

  // Allocate cache arrays that store the x and y coordinates of the incoming
  // gradient for each max element.
  Value *srcXY;
  if (kind == PoolInst::OpKind::Max) {
    srcXY = createAllocActivationInst(
        ElemKind::IndexTy, {idim.n, outSz.first, outSz.second, idim.c, 2},
        "srcXY");
  } else {
    srcXY = createAllocActivationInst(ElemKind::IndexTy, {}, "srcXY");
  }

  Value *dest = createAllocActivationInst(
      ElemKind::FloatTy, {idim.n, outSz.first, outSz.second, idim.c});

  return createPoolInst(dest, input, srcXY, kind, kernel, stride, pad);
}

FullyConnectedInst *IRBuilder::createFullyConnectedOp(Value *input,
                                                      size_t outDepth) {
  TypeRef T = input->getType();
  auto idim = flattenCdr(input->dims());

  size_t fanIn = idim.second;

  auto *W = createWeightVar(T->getElementType(), {outDepth, idim.second},
                            "weights", InitKind::Xavier, fanIn);

  auto *B = createWeightVar(T->getElementType(), {outDepth}, "bias",
                            InitKind::Broadcast, 0.1);
  auto *dest =
      createAllocActivationInst(T->getElementType(), {idim.first, outDepth});

  return createFullyConnectedInst(dest, input, W, B, outDepth);
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
  auto *E = createWeightVar(input->getType(), "e_cache");
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

ConcatInst *IRBuilder::createConcatOp(llvm::ArrayRef<Value *> inputs,
                                      unsigned dimension) {
  auto inDim = inputs[0]->dims();
  for (auto in : inputs) {
    (void)in;
    assert(in->dims() == inDim && "Invalid input shape");
  }

  std::vector<size_t> shape(inDim.begin(), inDim.end());
  // We are stacking the tensors along a specific dimension. This means that we
  // increase the size of the tensor along this dimension.
  shape[dimension] *= inputs.size();

  auto *res = createAllocActivationInst(inputs[0]->getElementType(), shape);
  return createConcatInst(res, inputs, dimension);
}

BatchNormalizationInst *IRBuilder::createBatchNormalizationOp(Value *input,
                                                              size_t channelIdx,
                                                              float epsilon,
                                                              float momentum) {
  // Figure out how many channels are in the tensor.
  size_t channels = input->dims()[channelIdx];

  // Allocate the learnable parameters beta and gamma.
  auto *beta = createWeightVar(ElemKind::FloatTy, {channels}, "beta",
                               InitKind::Broadcast, 0.);
  auto *gamma = createWeightVar(ElemKind::FloatTy, {channels}, "gamma",
                                InitKind::Broadcast, 1.0);

  auto *mean = createAllocActivationInst(ElemKind::FloatTy, {channels}, "mean");

  auto *variance =
      createAllocActivationInst(ElemKind::FloatTy, {channels}, "variance");

  return createBatchNormalizationOp(input, beta, gamma, mean, variance,
                                    channelIdx, epsilon, momentum);
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

ArithmeticInst *IRBuilder::createArithmeticOp(Value *LHS, Value *RHS,
                                              ArithmeticInst::OpKind op) {
  assert(LHS->dims() == RHS->dims() && "Invalid operand shapes");
  // The output tensor is of the same shape as the input tensor.
  auto *res = createAllocActivationInst(LHS->getType());
  return createArithmeticInst(res, LHS, RHS, op);
}

Value *IRBuilder::createReturnOp(Value *input) {
  auto *W = createWeightVar(input->getType(), "result", InitKind::Extern);
  createCopyInst(W, input);
  return W;
}

//===----------------------------------------------------------------------===//
//                     Low level instructions.
//===----------------------------------------------------------------------===//

CopyInst *IRBuilder::createCopyInst(Value *dest, Value *src) {
  auto *A = new CopyInst(dest, src);
  M_->pushInstr(A);
  return A;
}

ConvolutionInst *IRBuilder::createConvolutionInst(Value *dest, Value *src,
                                                  Value *filter, Value *bias,
                                                  size_t kernel, size_t stride,
                                                  size_t pad, size_t depth) {
  auto *A =
      new ConvolutionInst(dest, src, filter, bias, kernel, stride, pad, depth);
  M_->pushInstr(A);
  return A;
}

PoolInst *IRBuilder::createPoolInst(Value *dest, Value *src, Value *srcXY,
                                    PoolInst::OpKind kind, size_t kernel,
                                    size_t stride, size_t pad) {
  auto *A = new PoolInst(dest, src, srcXY, kind, kernel, stride, pad);
  M_->pushInstr(A);
  return A;
}

FullyConnectedInst *IRBuilder::createFullyConnectedInst(Value *dest, Value *src,
                                                        Value *filter,
                                                        Value *bias,
                                                        size_t depth) {
  auto *A = new FullyConnectedInst(dest, src, filter, bias, depth);
  M_->pushInstr(A);
  return A;
}

ReluInst *IRBuilder::createReluInst(Value *dest, Value *src) {
  auto *A = new ReluInst(dest, src);
  M_->pushInstr(A);
  return A;
}

SigmoidInst *IRBuilder::createSigmoidInst(Value *dest, Value *src) {
  auto *A = new SigmoidInst(dest, src);
  M_->pushInstr(A);
  return A;
}

TanhInst *IRBuilder::createTanhInst(Value *dest, Value *src) {
  auto *A = new TanhInst(dest, src);
  M_->pushInstr(A);
  return A;
}

SoftMaxInst *IRBuilder::createSoftMaxInst(Value *dest, Value *src, Value *E,
                                          Value *selected) {
  auto *A = new SoftMaxInst(dest, src, E, selected);
  M_->pushInstr(A);
  return A;
}

RegressionInst *IRBuilder::createRegressionInst(Value *dest, Value *src,
                                                Value *expected) {
  auto *A = new RegressionInst(dest, src, expected);
  M_->pushInstr(A);
  return A;
}

ReshapeInst *IRBuilder::createReshapeInst(Value *dest, Value *src,
                                          llvm::ArrayRef<size_t> shape) {
  auto *A = new ReshapeInst(dest, src, shape);
  M_->pushInstr(A);
  return A;
}

TransposeInst *
IRBuilder::createTransposeInst(Value *dest, Value *src,
                               llvm::ArrayRef<unsigned> shuffle) {
  auto *A = new TransposeInst(dest, src, shuffle);
  M_->pushInstr(A);
  return A;
}

ConcatInst *IRBuilder::createConcatInst(Value *dest,
                                        llvm::ArrayRef<Value *> src,
                                        size_t dim) {
  auto *A = new ConcatInst(dest, src, dim);
  M_->pushInstr(A);
  return A;
}

BatchNormalizationInst *IRBuilder::createBatchNormalizationInst(
    Value *dest, Value *src, Value *scale, Value *bias, Value *mean, Value *var,
    size_t channelIdx, float epsilon, float momentum) {
  auto *A = new BatchNormalizationInst(dest, src, scale, bias, mean, var,
                                       channelIdx, epsilon, momentum);
  M_->pushInstr(A);
  return A;
}

LocalResponseNormalizationInst *IRBuilder::createLocalResponseNormalizationInst(
    Value *dest, Value *src, Value *scale, size_t halfWindowSize, float alpha,
    float beta, float k) {
  auto *A = new LocalResponseNormalizationInst(dest, src, scale, halfWindowSize,
                                               alpha, beta, k);
  M_->pushInstr(A);
  return A;
}

ArithmeticInst *IRBuilder::createArithmeticInst(Value *dest, Value *LHS,
                                                Value *RHS,
                                                ArithmeticInst::OpKind kind) {
  auto *A = new ArithmeticInst(dest, LHS, RHS, kind);
  M_->pushInstr(A);
  return A;
}

WeightVar *IRBuilder::createWeightVar(ElemKind elemTy,
                                      llvm::ArrayRef<size_t> dims,
                                      llvm::StringRef name, InitKind initKind,
                                      float val) {
  auto T = M_->getGraph()->uniqueType(elemTy, dims);
  return createWeightVar(T, name, initKind, val);
}

WeightVar *IRBuilder::createWeightVar(TypeRef T, llvm::StringRef name,
                                      InitKind initKind, float val) {
  auto *A = new WeightVar(T, initKind, val);
  M_->getWeights().push_back(A);
  A->setName(name);
  return A;
}

AllocActivationInst *
IRBuilder::createAllocActivationInst(TypeRef T, llvm::StringRef name) {
  auto *A = new AllocActivationInst(T);
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
  auto *A = new DeallocActivationInst(src);
  M_->pushInstr(A);
  return A;
}
