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
#ifndef GLOW_BACKENDS_LAYOUTCONVERTER_H
#define GLOW_BACKENDS_LAYOUTCONVERTER_H

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"

namespace glow {

/// Convert regular convolution nodes (that use NHWC) into a backend-specific
/// convolution nodes using NCHW.
template <class NCHWConvNode>
Node *convertConvToNCHWConv(ConvolutionNode *CN, Function *F) {
  // Convert filter and input from NHWC (Glow's default) into NCHW.
  auto *NI = F->createTranspose("conv.input", CN->getInput(), NHWC2NCHW);
  auto *NF = F->createTranspose("conv.filter", CN->getFilter(), NHWC2NCHW);

  auto dimsNHWC = ShapeNHWC(CN->getResult().getType()->dims());
  auto dimsNCHW = {dimsNHWC.n, dimsNHWC.c, dimsNHWC.h, dimsNHWC.w};
  auto outTy = F->getParent()->uniqueType(ElemKind::FloatTy, dimsNCHW);

  auto *NC = F->addNode(new NCHWConvNode(CN->getName(), outTy, NI, NF,
                                         CN->getBias(), CN->getKernel(),
                                         CN->getStride(), CN->getPads()));
  auto NR = F->createTranspose("conv.result", NC, NCHW2NHWC);

  return NR;
}

/// Convert regular pool nodes (that use NHWC) into backend-specific nodes using
/// NCHW.
template <class PoolNode, class NCHWPoolNode>
Node *convertPoolToNCHWPool(PoolNode *PN, Function *F) {
  // Convert input from NHWC (Glow's default) into NCHW.
  auto *NI = F->createTranspose("conv.input", PN->getInput(), NHWC2NCHW);

  auto dimsNHWC = ShapeNHWC(PN->getResult().getType()->dims());
  auto dimsNCHW = {dimsNHWC.n, dimsNHWC.c, dimsNHWC.h, dimsNHWC.w};
  auto outTy = F->getParent()->uniqueType(ElemKind::FloatTy, dimsNCHW);

  auto *NPN =
      F->addNode(new NCHWPoolNode(PN->getName(), outTy, NI, PN->getKernel(),
                                  PN->getStride(), PN->getPads()));
  auto NR = F->createTranspose("poolmax.result", NPN, NCHW2NHWC);

  return NR;
}

} // namespace glow

#endif // GLOW_BACKENDS_LAYOUTCONVERTER_H
