/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

    createDeallocActivationInst("dealloc." + AA->getName().str(), AA);
  }
}

//===----------------------------------------------------------------------===//
//                        High level operators.
//===----------------------------------------------------------------------===//
MaxPoolWithArgmaxInst *IRBuilder::createMaxPoolWithArgmaxOp(
    llvm::StringRef name, Value *input, llvm::ArrayRef<unsigned_t> kernels,
    llvm::ArrayRef<unsigned_t> strides, llvm::ArrayRef<unsigned_t> pads,
    unsigned_t layout, ElemKind argMaxIndicesTy, bool flattenIndices) {
  TypeRef outTy{nullptr};
  Value *argmax{nullptr};

  if (layout == NHWC) {
    ShapeNHWC idim = ShapeNHWC(input->dims());

    auto outSz =
        calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides, pads);

    // Allocate storage for flattened NCHW index of max element.
    argmax =
        createAllocActivationInst(name.str() + ".argmax", argMaxIndicesTy,
                                  {idim.n, outSz.first, outSz.second, idim.c});

    outTy = F_->getGraph()->getParent()->uniqueTypeWithNewShape(
        input->getType(), {idim.n, outSz.first, outSz.second, idim.c});
  } else {
    ShapeNCHW idim(input->dims());

    auto outSz =
        calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides, pads);

    // Allocate storage for flattened NCHW index of max element.
    argmax =
        createAllocActivationInst(name.str() + ".argmax", argMaxIndicesTy,
                                  {idim.n, idim.c, outSz.first, outSz.second});

    outTy = F_->getGraph()->getParent()->uniqueTypeWithNewShape(
        input->getType(), {idim.n, idim.c, outSz.first, outSz.second});
  }

  Value *dest = createAllocActivationInst(name.str() + ".res", outTy);

  return createMaxPoolWithArgmaxInst(name, dest, input, argmax, kernels,
                                     strides, pads, layout, flattenIndices);
}

ArgMaxInst *IRBuilder::createArgMaxOp(llvm::StringRef name, Value *input,
                                      unsigned_t axis, bool keepDims,
                                      ElemKind outIndicesTy) {
  ShapeVector odim = reduceDims(input->dims(), {axis}, keepDims);
  Value *argmax =
      createAllocActivationInst(name.str() + ".argmax", outIndicesTy, odim);
  return createArgMaxInst(name, argmax, input, axis, keepDims);
}

AvgPoolInst *IRBuilder::createAvgPoolOp(Value *input,
                                        llvm::ArrayRef<unsigned_t> kernels,
                                        llvm::ArrayRef<unsigned_t> strides,
                                        llvm::ArrayRef<unsigned_t> pads,
                                        unsigned_t layout,
                                        bool countIncludePads) {

  TypeRef outTy;

  if (layout == NHWC) {
    ShapeNHWC idim(input->dims());
    auto outSz =
        calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides, pads);
    outTy = F_->getGraph()->getParent()->uniqueTypeWithNewShape(
        input->getType(), {idim.n, outSz.first, outSz.second, idim.c});
  } else {
    ShapeNCHW idim(input->dims());
    auto outSz =
        calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides, pads);
    outTy = F_->getGraph()->getParent()->uniqueTypeWithNewShape(
        input->getType(), {idim.n, idim.c, outSz.first, outSz.second});
  }

  Value *dest = createAllocActivationInst("pool.res", outTy);
  return createAvgPoolInst("pool", dest, input, kernels, strides, pads, layout,
                           countIncludePads);
}

CrossEntropyLossInst *IRBuilder::createCrossEntropyLossOp(llvm::StringRef name,
                                                          Value *p,
                                                          Value *labels) {
  auto *res =
      createAllocActivationInst(name.str() + ".res", ElemKind::FloatTy, {1});
  return createCrossEntropyLossInst(name, p, labels, res);
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
                                            llvm::ArrayRef<dim_t> dims,
                                            Value *src, llvm::StringRef name,
                                            llvm::ArrayRef<dim_t> offsets) {
  auto ty =
      getIRFunction().getGraph()->getParent()->uniqueType(Type(elemKind, dims));
  return createTensorViewInst(
      name, src, ty,
      (offsets.size()
           ? offsets
           : llvm::ArrayRef<dim_t>(std::vector<dim_t>(dims.size(), 0))));
}

LocalResponseNormalizationInst *IRBuilder::createLocalResponseNormalizationOp(
    llvm::StringRef name, Value *input, size_t halfWindowSize, float alpha,
    float beta, float k) {
  auto ty = input->getType();
  auto *scale = createAllocActivationInst(name.str() + ".scale", ty);

  // The output tensor is of the same shape as the input tensor.
  auto *res = createAllocActivationInst(name.str() + ".res", ty);
  return createLocalResponseNormalizationInst(name, res, input, scale,
                                              halfWindowSize, alpha, beta, k);
}

TopKInst *IRBuilder::createTopKOp(llvm::StringRef name, Value *input, size_t k,
                                  ElemKind outIndicesTy) {
  auto inDims = input->dims();
  assert(inDims.size() > 0);
  assert(k <= inDims.back());
  assert(outIndicesTy == ElemKind::Int32ITy ||
         outIndicesTy == ElemKind::Int64ITy);
  ShapeVector outDims(inDims.begin(), inDims.end());
  outDims.back() = k;
  auto outTy = F_->getGraph()->getParent()->uniqueTypeWithNewShape(
      input->getType(), outDims);
  auto *values = createAllocActivationInst(name.str() + ".values", outTy);
  auto *indices =
      createAllocActivationInst(name.str() + ".indices", outIndicesTy, outDims);
  return createTopKInst(name.str(), values, indices, input, k);
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
                                      llvm::ArrayRef<dim_t> dims,
                                      llvm::StringRef name,
                                      WeightVar::MutabilityKind m) {
  auto T = F_->getGraph()->getParent()->uniqueType(elemTy, dims);
  return createWeightVar(T, name, m);
}

WeightVar *IRBuilder::createWeightVar(TypeRef T, llvm::StringRef name,
                                      WeightVar::MutabilityKind m) {
  auto *A = new WeightVar(uniqueName(name), T, m);
  F_->getWeights().push_back(A);
  return A;
}

WeightVar *IRBuilder::createWeightVar(ElemKind elemTy,
                                      llvm::ArrayRef<dim_t> dims, float scale,
                                      int32_t offset, llvm::StringRef name,
                                      WeightVar::MutabilityKind m) {
  auto T = F_->getGraph()->getParent()->uniqueType(elemTy, dims, scale, offset);
  return createWeightVar(T, name, m);
}

AllocActivationInst *
IRBuilder::createAllocActivationInst(llvm::StringRef name, ElemKind elemTy,
                                     llvm::ArrayRef<dim_t> dims) {
  auto T = F_->getGraph()->getParent()->uniqueType(elemTy, dims);
  return createAllocActivationInst(name, T);
}
