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

#ifndef GLOW_TORCH_GLOW_SRC_COMMON_H
#define GLOW_TORCH_GLOW_SRC_COMMON_H

#include <torch/csrc/jit/ir.h>

namespace glow {
/// Various settings to be used by code that loads PyTorch models. There should
/// only be one of these and it should be obtained by calling
/// getPyTorchLoaderSettings().
struct PyTorchLoaderSettings {
  /// Whether or not run the custom pass that fuses jit nodes into a glow node.
  bool fusionPassEnabled = false;

  /// The PyTorch symbol used to identify the Node that contains PyTorch
  /// subgraphs that are compiled for running on Glow.
  bool weightFreezingEnabled = true;

  /// Name of the Glow backend to use with CachingGraphRunner's HostManager.
  std::string glowBackendName = "Interpreter";
};

/// \returns the PyTorchLoaderSettings singleton to be used throughout Glow's
/// PyTorch model loading code.
PyTorchLoaderSettings &getPyTorchLoaderSettings();

/// \returns the PyTorch symbol to be used for the PyTorch node which represents
/// the subgraph that Glow will compile and run.
const c10::Symbol &getGlowSymbol();

/// Executes custom fuse pass for the given \p graph and \p fuseSymbol.
void glowCustomFuse(std::shared_ptr<torch::jit::Graph> &graph,
                    at::Symbol fuseSymbol);

/// Register the glow::FusionGroup operator.
void registerGlowOp();

/// Register the pass that fuses parts of the graph into a glow::FusionGroup.
void registerGlowFusionPass();

/// Convenience method to register the glow fusion op and pass. \p
/// enableFusionPass can be used to enable the glow fusion pass once it's
/// registered.
void registerGlowFusionOpAndPass(bool enableFusionPass = false);

} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_COMMON_H
