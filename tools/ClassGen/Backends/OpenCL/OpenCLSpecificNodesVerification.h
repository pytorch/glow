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

#ifdef GLOW_WITH_OPENCL

void OCLConvolutionNode::verify() const {
  ShapeNCHW idim(getInput().getType()->dims());
  ShapeNCHW odim(getResult().getType()->dims());
  auto outSz = calculateConvPoolOutputDims(idim.h, idim.w, getKernel(),
                                           getStride(), getPads());
  ShapeNCHW exp(idim.n, getBias().dims()[0], outSz.first, outSz.second);
  (void)exp;
  assert(exp == odim && "Invalid output dimensions");
}

void OCLAvgPoolNode::verify() const {}

void OCLMaxPoolNode::verify() const {}
#endif // GLOW_WITH_OPENCL
