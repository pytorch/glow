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
#ifdef GLOW_WITH_HABANA
#ifndef TOOLS_CLASSGEN_BACKENDS_HABANA_HABANASPECIFICNODESVERIFICATION_H
#define TOOLS_CLASSGEN_BACKENDS_HABANA_HABANASPECIFICNODESVERIFICATION_H

#include "glow/Graph/VerifierHelper.h"

namespace {
static bool verifyConvolution(NodeValue src, NodeValue dest, NodeValue filter,
                              NodeValue bias,
                              llvm::ArrayRef<unsigned_t> kernels,
                              llvm::ArrayRef<unsigned_t> strides,
                              llvm::ArrayRef<unsigned_t> pads,
                              unsigned_t group) {
  const Node *parent = dest.getNode();
  bool isValid = checkType(src, dest.getElementType(), parent);
  isValid &= checkType(src, filter.getElementType(), parent);
  // Non quantization type check.
  if (src.getElementType() == ElemKind::FloatTy) {
    isValid &= checkType(bias, ElemKind::FloatTy, parent);
  }
  // Quantization type check.
  if (src.getElementType() == ElemKind::Int8QTy) {
    isValid &= checkType(bias, ElemKind::Int32QTy, parent);
  }
  ShapeNHWC idim(src.getType()->dims());
  ShapeNHWC odim(dest.getType()->dims());
  PaddingTLBR pdim(pads);
  ShapeHW kdim(kernels);
  isValid &= expectCompareTrue("buffer height too small for selected stride",
                               idim.w + pdim.left + pdim.right, kdim.height,
                               parent, CompareOperatorGreaterEqual<size_t>());
  isValid &= expectCompareTrue("buffer width too small for selected stride",
                               idim.h + pdim.top + pdim.bottom, kdim.width,
                               parent, CompareOperatorGreaterEqual<size_t>());
  isValid &= expectCompareTrue("channels number must be divisible by groups",
                               idim.c % group, size_t(0), parent);

  auto outSz =
      calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides, pads);
  isValid &=
      expectCompareTrue("Invalid output dimension N", odim.n, idim.n, parent);
  isValid &= expectCompareTrue("Invalid output dimension H", odim.h,
                               outSz.first, parent);
  isValid &= expectCompareTrue("Invalid output dimension W", odim.w,
                               outSz.second, parent);
  isValid &= expectCompareTrue("Invalid output dimension C", odim.c % group,
                               size_t(0), parent);

  const size_t filterDims[] = {kdim.height, kdim.width, idim.c / (size_t)group,
                               odim.c};
  isValid &=
      expectCompareTrue("Invalid filter dimensions", filter.getType()->dims(),
                        llvm::makeArrayRef(filterDims), parent);
  const size_t biasDims[] = {odim.c};
  isValid &=
      expectCompareTrue("Invalid bias dimensions", bias.getType()->dims(),
                        llvm::makeArrayRef(biasDims), parent);
  return isValid;
}
} // namespace

bool HabanaFullyConnectedNode::verify() const { return true; }

bool HabanaConvolutionNode::verify() const {
  return verifyConvolution(getInput(), getResult(), getFilter(), getBias(),
                           Kernels_, Strides_, Pads_, Group_);
}

bool HabanaConvolutionAddNode::verify() const {
  bool isValid =
      verifyConvolution(getInput(), getResult(), getFilter(), getBias(),
                        Kernels_, Strides_, Pads_, Group_);
  isValid &= checkSameShape(getAddend(), getResult(), this);
  return isValid;
}

#endif // TOOLS_CLASSGEN_BACKENDS_HABANA_HABANASPECIFICNODESVERIFICATION_H
#endif // GLOW_WITH_HABANA
