/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "glow/Graph/Graph.h"

#include "glow/IR/IRBuilder.h"
#include "glow/IR/IRUtils.h"

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
  for (auto &I : instrs) {
    auto AA = dyn_cast<AllocActivationInst>(&I);
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

  auto outTy = F_->getGraph()->getParent()->uniqueTypeWithNewShape(
      input->getType(), {idim.n, outSz.first, outSz.second, idim.c});
  Value *dest = createAllocActivationInst("pool.res", outTy);

  return createPoolMaxWithXYInst("pool", dest, input, srcXY, kernel, stride,
                                 pad);
}
PoolAvgInst *IRBuilder::createPoolAvgOp(Value *input, size_t kernel,
                                        size_t stride, size_t pad) {
  ShapeNHWC idim = ShapeNHWC(input->dims());
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  auto outSz = calculateConvOutputDims(idim.h, idim.w, kernel, stride, pad);
  auto outTy = F_->getGraph()->getParent()->uniqueTypeWithNewShape(
      input->getType(), {idim.n, outSz.first, outSz.second, idim.c});
  Value *dest = createAllocActivationInst("pool.res", outTy);

  return createPoolAvgInst("pool", dest, input, kernel, stride, pad);
}

CrossEntropyLossInst *IRBuilder::createCrossEntropyLossOp(Value *p,
                                                          Value *labels) {
  auto *res = createAllocActivationInst("celoss.res", ElemKind::FloatTy, {1});
  return createCrossEntropyLossInst("celoss", p, labels, res);
}

/// Creates a tensorview instruction with the following parameters:
/// \param elemKind the type of elements in a tensor
/// \param dims dimensions of the view, such that the number of elements
/// in the view is the same as the number of elements in the source tensor
/// \p src
/// \param src the source tensor used to create the unowned tensor.
/// \param offsets is a vector of offsets into the Tensor for this view of the
/// Tensor.
TensorViewInst *IRBuilder::createTensorView(ElemKind elemKind,
                                            llvm::ArrayRef<size_t> dims,
                                            Value *src, llvm::StringRef name,
                                            llvm::ArrayRef<size_t> offsets) {
  auto ty =
      getIRFunction().getGraph()->getParent()->uniqueType(Type(elemKind, dims));
  return createTensorViewInst(
      name, src, ty,
      (offsets.size()
           ? offsets
           : llvm::ArrayRef<size_t>(std::vector<size_t>(dims.size(), 0))));
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

TopKInst *IRBuilder::createTopKOp(Value *input, size_t k) {
  auto inDims = input->dims();
  assert(inDims.size() > 0);
  assert(k <= inDims.back());
  ShapeVector outDims(inDims.begin(), inDims.end());
  outDims.back() = k;
  auto outTy = F_->getGraph()->getParent()->uniqueTypeWithNewShape(
      input->getType(), outDims);
  // Allocate enough scratch space to hold N values and N indices.
  auto *scratch = createAllocActivationInst("topk.scratch", ElemKind::IndexTy,
                                            {inDims.back() * 2});
  createSplatInst("topk.zero.scratch", scratch, 0);
  auto *values = createAllocActivationInst("topk.values", outTy);
  auto *indices =
      createAllocActivationInst("topk.indices", ElemKind::IndexTy, outDims);
  return createTopKInst("topk", values, indices, input, scratch, k);
}

Value *IRBuilder::createReturnOp(Value *input) {
  auto *W = createWeightVar(input->getType(), "result",
                            WeightVar::MutabilityKind::Mutable,
                            VisibilityKind::Public);
  createCopyInst("return", W, input);
  return W;
}

//===----------------------------------------------------------------------===//
//                     Low level instructions.
//===----------------------------------------------------------------------===//

WeightVar *IRBuilder::createWeightVar(ElemKind elemTy,
                                      llvm::ArrayRef<size_t> dims,
                                      llvm::StringRef name,
                                      WeightVar::MutabilityKind m,
                                      VisibilityKind v) {
  auto T = F_->getGraph()->getParent()->uniqueType(elemTy, dims);
  return createWeightVar(T, name, m, v);
}

WeightVar *IRBuilder::createWeightVar(TypeRef T, llvm::StringRef name,
                                      WeightVar::MutabilityKind m,
                                      VisibilityKind v) {
  assert(!(m == WeightVar::MutabilityKind::Constant &&
           v == VisibilityKind::Public) &&
         "Cannot have a Constant Public Variable.");
  auto *A = new WeightVar(name, T, m, v);
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
