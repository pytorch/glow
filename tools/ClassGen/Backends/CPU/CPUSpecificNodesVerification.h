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
#ifdef GLOW_WITH_CPU

#include "glow/Graph/VerifierHelper.h"

bool CPUMaxSplatNode::verify() const {
  return checkSameType(getInput(), getResult(), this);
}

bool CPUConvDKKC8Node::verify() const {
  ShapeNHWC idim(getInput().getType()->dims());
  ShapeNHWC odim(getResult().getType()->dims());
  auto outSz = calculateConvPoolOutputDims(idim.h, idim.w, getKernels(),
                                           getStrides(), getPads());
  ShapeNHWC exp(idim.n, outSz.first, outSz.second, getBias().dims()[0]);
  return expectCompareTrue("Invalid output dimensions", exp, odim, this);
}

#endif // GLOW_WITH_CPU
