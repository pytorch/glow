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

#include "PyTorchCommon.h"
#include "FusingOptimizer.h"
#include "PyTorchModelLoader.h"
#include <torch/csrc/jit/passes/graph_fuser.h>

namespace glow {

PyTorchLoaderSettings &getPyTorchLoaderSettings() {
  static PyTorchLoaderSettings settings;
  return settings;
}

const c10::Symbol &getGlowSymbol() {
  static c10::Symbol glowSymbol =
      at::Symbol::fromQualString("glow::FusionGroup");
  return glowSymbol;
}

void glowCustomFuse(std::shared_ptr<torch::jit::Graph> &g,
                    at::Symbol fuseSymbol) {
  // Fuse all linear operators
  // Currently PyTorch does not have good support for aten:addmm when fusing
  // Therefore we use some pattern to translate all aten::addmm to
  // aten::linear before we fuse the whole graph.
  FuseLinear(g);

  torch::jit::CustomFuseGraph(g, PyTorchModelLoader::isNodeSupported,
                              fuseSymbol);
}

} // namespace glow
