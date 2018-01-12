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
  auto &instrs = M_->getInstrs();
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

ConvolutionInst *IRBuilder::createConvOp(Value *input, Value *filter,
                                         Value *bias, size_t depth,
                                         size_t kernel, size_t stride,
                                         size_t pad) {
  ShapeNHWC idim = ShapeNHWC(input->dims());
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  // Calculate the size and allocate the output buffer.
  auto outSz = calculateConvOutputDims(idim.h, idim.w, pad, kernel, stride);

  llvm::SmallVector<size_t, 4> outDims = {idim.n, outSz.first, outSz.second,
                                          depth};
  auto TR = M_->getGraph()->uniqueType(ElemKind::FloatTy, outDims);
  Value *dest = createAllocActivationInst("conv.res", TR);

  return createConvolutionInst("conv", dest, input, filter, bias, kernel,
                               stride, pad, depth);
}

PoolMaxInst *IRBuilder::createPoolMaxOp(Value *input, size_t kernel,
                                        size_t stride, size_t pad) {
  ShapeNHWC idim = ShapeNHWC(input->dims());
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  auto outSz = calculateConvOutputDims(idim.h, idim.w, pad, kernel, stride);

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

  auto outSz = calculateConvOutputDims(idim.h, idim.w, pad, kernel, stride);

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

  auto outSz = calculateConvOutputDims(idim.h, idim.w, pad, kernel, stride);

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

SoftMaxInst *IRBuilder::createSoftMaxOp(Value *input, Value *selected) {
  auto *res = createAllocActivationInst("softmax.res", input->getType());
  return createSoftMaxInst("softmax", res, input, selected);
}

SoftMaxWithEInst *IRBuilder::createSoftMaxWithEOp(Value *input,
                                                  Value *selected) {
  auto *res = createAllocActivationInst("softmax.res", input->getType());
  auto *E = createAllocActivationInst("e_cache", input->getType());
  // Initialize E, because it is an inout parameter.
  createSplatInst("zero", E, 0.0);
  return createSoftMaxWithEInst("softmax", res, input, E, selected);
}

ReshapeInst *IRBuilder::createReshapeOp(Value *input,
                                        llvm::ArrayRef<size_t> shape) {
  auto *res =
      createAllocActivationInst("reshape.res", input->getElementType(), shape);
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
  auto ty = getModule().getGraph()->uniqueType(Type(elemKind, dims));
  return createTensorViewInst(name, src, ty);
}

TransposeInst *IRBuilder::createTransposeOp(Value *input,
                                            llvm::ArrayRef<unsigned> shuffle) {
  llvm::SmallVector<size_t, 6> shape;
  auto dims = input->dims();
  for (size_t i = 0; i < dims.size(); i++) {
    shape.push_back(dims[shuffle[i]]);
  }

  auto *res =
      createAllocActivationInst("transp.res", input->getElementType(), shape);
  return createTransposeInst("transp", res, input, shuffle);
}

/*
ConcatInst *IRBuilder::createConcatOp(Value *LHS, Value *RHS,
                                      unsigned dimension) {
  assert(LHS->getType() == RHS->getType() && "Invalid dims");
  auto inDim = LHS->dims();

  llvm::SmallVector<size_t, 6> shape(inDim.begin(), inDim.end());
  // We are stacking the tensors along a specific dimension. This means that we
  // increase the size of the tensor along this dimension.
  shape[dimension] *= 2;

  auto *res =
      createAllocActivationInst("concat.res", LHS->getElementType(), shape);
  return createConcatInst("concat", res, LHS, RHS, dimension);
}*/

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
  auto Ty = input->getType();
  auto *scale = createAllocActivationInst("scale", Ty);

  // The output tensor is of the same shape as the input tensor.
  auto *res = createAllocActivationInst("LRN.res", Ty);
  return createLocalResponseNormalizationInst("LRN", res, input, scale,
                                              halfWindowSize, alpha, beta, k);
}

ElementAddInst *IRBuilder::createElementAddOp(Value *LHS, Value *RHS) {
  assert(LHS->dims() == RHS->dims() && "Invalid operand shapes");
  // The output tensor is of the same shape as the input tensor.
  auto *res = createAllocActivationInst("add.res", LHS->getType());
  return createElementAddInst("add", res, LHS, RHS);
}

ElementSubInst *IRBuilder::createElementSubOp(Value *LHS, Value *RHS) {
  assert(LHS->dims() == RHS->dims() && "Invalid operand shapes");
  // The output tensor is of the same shape as the input tensor.
  auto *res = createAllocActivationInst("sub.res", LHS->getType());
  return createElementSubInst("sub", res, LHS, RHS);
}

ElementMulInst *IRBuilder::createElementMulOp(Value *LHS, Value *RHS) {
  assert(LHS->dims() == RHS->dims() && "Invalid operand shapes");
  // The output tensor is of the same shape as the input tensor.
  auto *res = createAllocActivationInst("mul.res", LHS->getType());
  return createElementMulInst("mul", res, LHS, RHS);
}

ElementDivInst *IRBuilder::createElementDivOp(Value *LHS, Value *RHS) {
  assert(LHS->dims() == RHS->dims() &&
         "Input and Output dimensions are different");

  auto *res = createAllocActivationInst("div.res", LHS->getType());
  return createElementDivInst("div", res, LHS, RHS);
}

ElementMaxInst *IRBuilder::createElementMaxOp(Value *LHS, Value *RHS) {
  assert(LHS->dims() == RHS->dims() &&
         "Input and Output dimensions are different");

  auto *res = createAllocActivationInst("max.res", LHS->getType());
  return createElementMaxInst("max", res, LHS, RHS);
}

ElementMinInst *IRBuilder::createElementMinOp(Value *LHS, Value *RHS) {
  assert(LHS->dims() == RHS->dims() &&
         "Input and Output dimensions are different");

  auto *res = createAllocActivationInst("min.res", LHS->getType());
  return createElementMinInst("min", res, LHS, RHS);
}

ElementCmpLTEInst *IRBuilder::createElementCmpLTEOp(Value *LHS, Value *RHS) {
  assert(LHS->dims() == RHS->dims() &&
         "Input and Output dimensions are different");
  auto *res = createAllocActivationInst("cmp.lte.res", LHS->getType());
  return createElementCmpLTEInst("cmp.lte", res, LHS, RHS);
}

ElementSelectInst *IRBuilder::createSelectOp(Value *Cond, Value *LHS,
                                             Value *RHS) {
  assert(LHS->dims() == RHS->dims() &&
         "Input and Output dimensions are different");
  assert(Cond->dims() == RHS->dims() &&
         "Input and Output dimensions are different");
  auto *res = createAllocActivationInst("select.res", LHS->getType());
  return createElementSelectInst("select", res, Cond, LHS, RHS);
}

Value *IRBuilder::createReturnOp(Value *input) {
  auto *W = createWeightVar(input->getType(), "result",
                            WeightVar::MutabilityKind::Mutable);
  createCopyInst("return", W, input);
  return W;
}

//===----------------------------------------------------------------------===//
//                     Low level instructions.
//===----------------------------------------------------------------------===//

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
