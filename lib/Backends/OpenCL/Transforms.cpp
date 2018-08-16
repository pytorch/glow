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

#include "OpenCL.h"

#include "glow/Backends/LayoutConverter.h"
#include "glow/Graph/Nodes.h"

#include "llvm/Support/Casting.h"

using llvm::dyn_cast;

using namespace glow;

/// Perform OpenCL specific post-lowering graph transformation.
bool OCLBackend::transformPostLowering(Function *F,
                                       CompilationMode mode) const {
  // NCHW transformation is not supported in training mode yet, because of some
  // issues with gradient nodes.
  if (mode == CompilationMode::Train)
    return false;

  bool changed = false;
  for (auto &node : F->getNodes()) {
    if (auto *CN = dyn_cast<ConvolutionNode>(&node)) {
      // TODO: OpenCL fast convolution kernel itself has some issue with group >
      // 1, which will be investigated later. So far, if the group > 1, we just
      // call the slow convolution kernel.
      if (CN->getGroup() > 1)
        continue;
      auto *NR = convertConvToNCHWConv<OCLConvolutionNode>(CN, F);
      NodeValue(&node, 0).replaceAllUsesOfWith(NR);
      changed = true;
      continue;
    }
    if (auto *PMN = dyn_cast<MaxPoolNode>(&node)) {
      auto *NR = convertPoolToNCHWPool<MaxPoolNode, OCLMaxPoolNode>(PMN, F);
      NodeValue(&node, 0).replaceAllUsesOfWith(NR);
      changed = true;
      continue;
    }
    if (auto *PAN = dyn_cast<AvgPoolNode>(&node)) {
      auto *NR = convertPoolToNCHWPool<AvgPoolNode, OCLAvgPoolNode>(PAN, F);
      NodeValue(&node, 0).replaceAllUsesOfWith(NR);
      changed = true;
      continue;
    }
  }
  return changed;
}
