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

#include <pybind11/pybind11.h>

#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/operator_options.h>
#include <torch/csrc/jit/pass_manager.h>
#include <torch/csrc/jit/passes/graph_fuser.h>

#include "CachingGraphRunner.h"
#include "FusingOptimizer.h"
#include "PyTorchCommon.h"
#include "PyTorchModelLoader.h"

#include "glow/Graph/Graph.h"

namespace py = pybind11;

using namespace glow;

namespace {

/// Manages a CachingGraphRunner singleton.
CachingGraphRunner *getGraphRunner() {
  static auto runner_ =
      std::unique_ptr<CachingGraphRunner>(new CachingGraphRunner());
  return runner_.get();
}

/// Register the glow::FusionGroup operator.
void registerGlowOp() {
  auto options = c10::OperatorOptions();
  options.setAliasAnalysis(at::AliasAnalysisKind::PURE);

  torch::jit::RegisterOperators op({torch::jit::Operator(
      getGlowSymbol(),
      [](const torch::jit::Node *node) {
        return [node](torch::jit::Stack &stack) {
          llvm::Error err = getGraphRunner()->runGraph(node, stack);
          if (static_cast<bool>(err)) {
            // PyTorch framework expects an exception been thrown here.
            throw std::invalid_argument(llvm::toString(std::move(err)));
          }
          return 0;
        };
      },
      options)});
}

/// Register the pass that fuses parts of the graph into
/// a glow::FusionGroup
void registerPass() {
  torch::jit::RegisterPass pass([](std::shared_ptr<torch::jit::Graph> &g) {
    if (getPyTorchLoaderSettings().fusionPassEnabled) {

      // Fuse all linear operators
      // Currently PyTorch does not have good support for aten:addmm when fusing
      // Therefore we use some pattern to translate all aten::addmm to
      // aten::linear before we fuse the whole graph.
      FuseLinear(g);

      torch::jit::CustomFuseGraph(g, PyTorchModelLoader::isNodeSupported,
                                  getGlowSymbol());
    }
  });
}
} // namespace

/// The torch_glow pybind11 module.
PYBIND11_MODULE(_torch_glow, m) {
  registerGlowOp();
  registerPass();

  /// Enable compiling PyTorch subgraphs to Glow Functions.
  m.def("enableFusionPass",
        []() { getPyTorchLoaderSettings().fusionPassEnabled = true; });

  /// Disable compiling PyTorch subgraphs to Glow Functions.
  m.def("disableFusionPass",
        []() { getPyTorchLoaderSettings().fusionPassEnabled = false; });

  /// Enable freezing weights as Constants in PyTorch subgraphs loaded in Glow.
  m.def("enableWeightFreezing",
        []() { getPyTorchLoaderSettings().weightFreezingEnabled = true; });

  /// Disable freezing weights as Constants in PyTorch subgraphs loaded in Glow.
  m.def("disableWeightFreezing",
        []() { getPyTorchLoaderSettings().weightFreezingEnabled = false; });
}
