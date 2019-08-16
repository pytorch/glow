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
#include "CachingGraphRunner.h"
#include "GlowFuser.h"
#include "PyTorchModelLoader.h"

#include <torch/csrc/jit/operator_options.h>
#include <torch/csrc/jit/pass_manager.h>
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

  GlowCustomFuse(g, PyTorchModelLoader::isNodeSupported, fuseSymbol);
}

void registerGlowOp() {
  auto options = c10::OperatorOptions();
  options.setAliasAnalysis(at::AliasAnalysisKind::PURE);

  torch::jit::RegisterOperators op({torch::jit::Operator(
      getGlowSymbol(),
      [](const torch::jit::Node *node) {
        return [node](torch::jit::Stack &stack) {
          llvm::Error err =
              CachingGraphRunner::getGraphRunner()->runGraph(node, stack);
          if (static_cast<bool>(err)) {
            // PyTorch framework expects an exception been thrown here.
            throw std::invalid_argument(llvm::toString(std::move(err)));
          }
          return 0;
        };
      },
      options)});
}

void registerGlowFusionPass() {
  torch::jit::RegisterPass pass([](std::shared_ptr<torch::jit::Graph> &g) {
    if (getPyTorchLoaderSettings().fusionPassEnabled) {
      glow::glowCustomFuse(g, getGlowSymbol());
    }
  });
}

void registerGlowFusionOpAndPass(bool enableFusionPass) {
  registerGlowOp();
  registerGlowFusionPass();
  getPyTorchLoaderSettings().fusionPassEnabled = enableFusionPass;
}

} // namespace glow
