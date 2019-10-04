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
inline Node *convertConvToNCHWConv(ConvolutionNode *CN, Function *F) {
  // Convert filter and input from NHWC (Glow's default) into NCHW.
  auto *NI = F->createTranspose("conv.input", CN->getInput(), NHWC2NCHW);
  auto *NF = F->createTranspose("conv.filter", CN->getFilter(), NHWC2NCHW);

  auto dimsNHWC = ShapeNHWC(CN->getResult().getType()->dims());
  auto dimsNCHW = {dimsNHWC.n, dimsNHWC.c, dimsNHWC.h, dimsNHWC.w};
  auto outTy = F->getParent()->uniqueTypeWithNewShape(CN->getResult().getType(),
                                                      dimsNCHW);

  auto *NC = F->addNode(new ConvolutionNode(
      CN->getName(), outTy, NI, NF, CN->getBias(), CN->getKernels(),
      CN->getStrides(), CN->getPads(), CN->getGroup(), CN->getDilation(), NCHW,
      CN->getFusedActivation()));
  auto *NR = F->createTranspose("conv.result", NC, NCHW2NHWC);

  return NR;
}

/// Convert regular max pool nodes (that use NHWC) into backend-specific nodes
/// using NCHW. \returns a pair containing the new MaxPool result and argmax
/// that the result and argmax of the original MaxPool should be replaced with.
inline std::pair<Node *, Node *> convertMaxPoolToNCHWPool(MaxPoolNode *PN,
                                                          Function *F) {
  // Convert input from NHWC (Glow's default) into NCHW.
  auto *NI = F->createTranspose("maxpool.input", PN->getInput(), NHWC2NCHW);

  auto dimsNHWC = ShapeNHWC(PN->getResult().getType()->dims());
  auto dimsNCHW = {dimsNHWC.n, dimsNHWC.c, dimsNHWC.h, dimsNHWC.w};
  auto outTy = F->getParent()->uniqueTypeWithNewShape(PN->getResult().getType(),
                                                      dimsNCHW);
  auto AMT = F->getParent()->uniqueTypeWithNewShape(PN->getArgmax().getType(),
                                                    dimsNCHW);

  auto *MPN = new MaxPoolNode(PN->getName(), outTy, AMT, NI, PN->getKernels(),
                              PN->getStrides(), PN->getPads(), NCHW);
  F->addNode(MPN);
  auto *NR = F->createTranspose("maxpool.result", MPN->getResult(), NCHW2NHWC);
  auto *NA = F->createTranspose("maxpool.argmax", MPN->getArgmax(), NCHW2NHWC);

  return std::make_pair(NR, NA);
}

inline Node *convertAvgPoolToNCHWPool(AvgPoolNode *PN, Function *F) {
  // Convert input from NHWC (Glow's default) into NCHW.
  auto *NI = F->createTranspose("maxpool.input", PN->getInput(), NHWC2NCHW);

  auto dimsNHWC = ShapeNHWC(PN->getResult().getType()->dims());
  auto dimsNCHW = {dimsNHWC.n, dimsNHWC.c, dimsNHWC.h, dimsNHWC.w};
  auto outTy = F->getParent()->uniqueTypeWithNewShape(PN->getResult().getType(),
                                                      dimsNCHW);

  auto *NPN =
      F->addNode(new AvgPoolNode(PN->getName(), outTy, NI, PN->getKernels(),
                                 PN->getStrides(), PN->getPads(), NCHW));
  auto *NR = F->createTranspose("avgpool.result", NPN->getResult(), NCHW2NHWC);

  return NR;
}

inline Node *convertAvgPoolGradToNCHWPool(AvgPoolGradNode *PGN, Function *F) {
  // Convert inputs from NHWC (Glow's default) into NCHW.
  auto *NI =
      F->createTranspose("avgpoolgrad.input", PGN->getInput(), NHWC2NCHW);
  auto *NO = F->createTranspose("avgpoolgrad.output",
                                PGN->getOriginalOutputForResult(), NHWC2NCHW);
  auto *NG =
      F->createTranspose("avgpoolgrad.outputgrad",
                         PGN->getGradOfOriginalOutputNamedResult(), NHWC2NCHW);

  auto *NPGN = F->addNode(
      new AvgPoolGradNode(PGN->getName(), NI, NO, NG, PGN->getKernels(),
                          PGN->getStrides(), PGN->getPads(), NCHW));
  auto *NR = F->createTranspose("avgpoolgrad.result",
                                NPGN->getGradOfInputNamedInput(), NCHW2NHWC);

  return NR;
}

} // namespace glow

#endif // GLOW_BACKENDS_LAYOUTCONVERTER_H
