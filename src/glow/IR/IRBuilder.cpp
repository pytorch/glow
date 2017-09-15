#include "glow/IR/IRBuilder.h"

#include "glow/Network/Nodes.h"

using namespace glow;

//===----------------------------------------------------------------------===//
//                        High level operators.
//===----------------------------------------------------------------------===//

ConvolutionInst *IRBuilder::createConvOp(Value *input, size_t depth,
                                         size_t kernel, size_t stride,
                                         size_t pad) {
  ShapeNHWC idim = input->dims();
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  // Calculate the size and allocate the output buffer.
  auto outSz =
      ConvNode::calculateOutputDims(idim.h, idim.w, pad, kernel, stride);
  ArrayRef<size_t> outDims = {idim.n, outSz.first, outSz.second, depth};
  Value *dest = createStaticVariable(ElemKind::FloatTy, outDims);

  // Allocate the Filter and Bias tensors.
  ArrayRef<size_t> filterDim = {depth, kernel, kernel, idim.c};
  size_t fanIn = kernel * kernel * idim.c;
  Value *filter = createStaticVariable(ElemKind::FloatTy, filterDim,
                                       InitKind::kXavier, fanIn);
  Value *bias = createStaticVariable(ElemKind::FloatTy, {depth},
                                     InitKind::kBroadcast, 0.1);

  return createConvolutionInst(dest, input, filter, bias, kernel, stride, pad,
                               depth);
}

PoolInst *IRBuilder::createMaxPoolOp(Value *input, PoolInst::OpKind kind,
                                     size_t kernel, size_t stride, size_t pad) {
  ShapeNHWC idim = input->dims();
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  auto outSz =
      ConvNode::calculateOutputDims(idim.h, idim.w, pad, kernel, stride);
  ArrayRef<size_t> exp = {idim.n, outSz.first, outSz.second, idim.c};
  Value *dest = createStaticVariable(ElemKind::FloatTy, exp);

  // Allocate cache arrays that store the x and y coordinates of the incoming
  // gradient for each max element.
  Value *srcXY;
  if (kind == PoolInst::OpKind::kMax) {
    ArrayRef<size_t> exp = {idim.n, outSz.first, outSz.second, idim.c, 2};
    srcXY = createStaticVariable(ElemKind::IndexTy, exp);
  } else {
    srcXY = createStaticVariable(ElemKind::IndexTy, {});
  }

  return createPoolInst(dest, input, srcXY, kind, kernel, stride, pad);
}

FullyConnectedInst *IRBuilder::createFullyConnectedOp(Value *input,
                                                      size_t outDepth) {
  TypeRef T = input->getType();
  auto idim = flattenCdr(input->dims());

  size_t fanIn = idim.second;
  auto *Out = createStaticVariable(T->getElementType(), {idim.first, outDepth});

  auto *W = createStaticVariable(T->getElementType(), {outDepth, idim.second},
                                 InitKind::kXavier, fanIn);

  auto *B = createStaticVariable(T->getElementType(), {outDepth},
                                 InitKind::kBroadcast, 0.1);

  return createFullyConnectedInst(Out, input, W, B, outDepth);
}

ReluInst *IRBuilder::createRELUOp(Value *input) {
  auto *res = createStaticVariable(input->getType());
  return createReluInst(res, input);
}

SigmoidInst *IRBuilder::createSigmoidOp(Value *input) {
  auto *res = createStaticVariable(input->getType());
  return createSigmoidInst(res, input);
}

TanhInst *IRBuilder::createTanhOp(Value *input) {
  auto *res = createStaticVariable(input->getType());
  return createTanhInst(res, input);
}

SoftMaxInst *IRBuilder::createSoftMaxOp(Value *input, Value *selected) {
  auto *res = createStaticVariable(input->getType());
  return createSoftMaxInst(res, input, selected);
}

RegressionInst *IRBuilder::createRegressionOp(Value *input, Value *expected) {
  auto *res = createStaticVariable(input->getType());
  return createRegressionInst(res, input, expected);
}

ReshapeInst *IRBuilder::createReshapeOp(Value *input, ArrayRef<size_t> shape) {
  auto *res = createStaticVariable(input->getElementType(), shape);
  return createReshapeInst(res, input, shape);
}

TransposeInst *IRBuilder::createTransposeOp(Value *input,
                                            ArrayRef<unsigned> shuffle) {
  std::vector<size_t> shape;
  auto dims = input->dims();
  for (size_t i = 0; i < dims.size(); i++) {
    shape.push_back(dims[shuffle[i]]);
  }

  auto *res = createStaticVariable(input->getElementType(), shape);
  return createTransposeInst(res, input, shuffle);
}

ConcatInst *IRBuilder::createConcatOp(ArrayRef<Value *> inputs,
                                      unsigned dimension) {
  auto inDim = inputs[0]->dims();
  for (auto in : inputs) {
    assert(in->dims() == inDim && "Invalid input shape");
  }

  std::vector<size_t> shape(inDim.begin(), inDim.end());
  // We are stacking the tensors along a specific dimension. This means that we
  // increase the size of the tensor along this dimension.
  shape[dimension] *= inputs.size();

  auto *res = createStaticVariable(inputs[0]->getElementType(), shape);
  return createConcatInst(res, inputs, dimension);
}

BatchNormalizationInst *IRBuilder::createBatchNormalizationOp(Value *input,
                                                              size_t channelIdx,
                                                              float epsilon,
                                                              float momentum) {
  // The output tensor is of the same shape as the input tensor.
  auto *res = createStaticVariable(input->getType());

  // Figure out how many channels are in the tensor.
  size_t channels = input->dims()[channelIdx];

  // Allocate the learnable parameters beta and gamma.
  auto *beta = createStaticVariable(ElemKind::FloatTy, {channels},
                                    InitKind::kBroadcast, 0.);
  auto *gamma = createStaticVariable(ElemKind::FloatTy, {channels},
                                     InitKind::kBroadcast, 1.0);
  auto *mean = createStaticVariable(ElemKind::FloatTy, {channels});
  auto *variance = createStaticVariable(ElemKind::FloatTy, {channels});

  return createBatchNormalizationInst(res, input, gamma, beta, mean, variance,
                                      channelIdx, epsilon, momentum);
}

ArithmeticInst *IRBuilder::createArithmeticOp(Value *LHS, Value *RHS,
                                              ArithmeticInst::OpKind op) {
  assert(LHS->dims() == RHS->dims() && "Invalid operand shapes");
  // The output tensor is of the same shape as the input tensor.
  auto *res = createStaticVariable(LHS->getType());
  return createArithmeticInst(res, LHS, RHS, op);
}

//===----------------------------------------------------------------------===//
//                     Low level instructions.
//===----------------------------------------------------------------------===//

CopyInst *IRBuilder::createCopyInst(Value *dest, Value *src) {
  auto *A = new CopyInst(dest, src);
  M_.pushInstr(A);
  return A;
}

ConvolutionInst *IRBuilder::createConvolutionInst(Value *dest, Value *src,
                                                  Value *filter, Value *bias,
                                                  size_t kernel, size_t stride,
                                                  size_t pad, size_t depth) {
  auto *A =
      new ConvolutionInst(dest, src, filter, bias, kernel, stride, pad, depth);
  M_.pushInstr(A);
  return A;
}

PoolInst *IRBuilder::createPoolInst(Value *dest, Value *src, Value *srcXY,
                                    PoolInst::OpKind kind, size_t kernel,
                                    size_t stride, size_t pad) {
  auto *A = new PoolInst(dest, src, srcXY, kind, kernel, stride, pad);
  M_.pushInstr(A);
  return A;
}

FullyConnectedInst *IRBuilder::createFullyConnectedInst(Value *dest, Value *src,
                                                        Value *filter,
                                                        Value *bias,
                                                        size_t depth) {
  auto *A = new FullyConnectedInst(dest, src, filter, bias, depth);
  M_.pushInstr(A);
  return A;
}

ReluInst *IRBuilder::createReluInst(Value *dest, Value *src) {
  auto *A = new ReluInst(dest, src);
  M_.pushInstr(A);
  return A;
}

SigmoidInst *IRBuilder::createSigmoidInst(Value *dest, Value *src) {
  auto *A = new SigmoidInst(dest, src);
  M_.pushInstr(A);
  return A;
}

TanhInst *IRBuilder::createTanhInst(Value *dest, Value *src) {
  auto *A = new TanhInst(dest, src);
  M_.pushInstr(A);
  return A;
}

SoftMaxInst *IRBuilder::createSoftMaxInst(Value *dest, Value *src,
                                          Value *expected) {
  auto *A = new SoftMaxInst(dest, src, expected);
  M_.pushInstr(A);
  return A;
}

RegressionInst *IRBuilder::createRegressionInst(Value *dest, Value *src,
                                                Value *expected) {
  auto *A = new RegressionInst(dest, src, expected);
  M_.pushInstr(A);
  return A;
}

ReshapeInst *IRBuilder::createReshapeInst(Value *dest, Value *src,
                                          ArrayRef<size_t> shape) {
  auto *A = new ReshapeInst(dest, src, shape);
  M_.pushInstr(A);
  return A;
}

TransposeInst *IRBuilder::createTransposeInst(Value *dest, Value *src,
                                              ArrayRef<unsigned> shuffle) {
  auto *A = new TransposeInst(dest, src, shuffle);
  M_.pushInstr(A);
  return A;
}

ConcatInst *IRBuilder::createConcatInst(Value *dest, ArrayRef<Value *> src,
                                        size_t dim) {
  auto *A = new ConcatInst(dest, src, dim);
  M_.pushInstr(A);
  return A;
}

BatchNormalizationInst *IRBuilder::createBatchNormalizationInst(
    Value *dest, Value *src, Value *scale, Value *bias, Value *mean, Value *var,
    size_t channelIdx, float epsilon, float momentum) {
  auto *A = new BatchNormalizationInst(dest, src, scale, bias, mean, var,
                                       channelIdx, epsilon, momentum);
  M_.pushInstr(A);
  return A;
}

ArithmeticInst *IRBuilder::createArithmeticInst(Value *dest, Value *LHS,
                                                Value *RHS,
                                                ArithmeticInst::OpKind kind) {
  auto *A = new ArithmeticInst(dest, LHS, RHS, kind);
  M_.pushInstr(A);
  return A;
}

StaticVariable *IRBuilder::createStaticVariable(ElemKind elemTy,
                                                ArrayRef<size_t> dims,
                                                InitKind mode, float val) {
  auto T = M_.uniqueType(elemTy, dims);
  auto *A = new StaticVariable(T, mode, val);
  M_.pushVar(A);
  return A;
}

StaticVariable *IRBuilder::createStaticVariable(TypeRef T, InitKind mode,
                                                float val) {
  auto *A = new StaticVariable(T, mode, val);
  M_.pushVar(A);
  return A;
}
