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

#ifndef GLOW_TORCH_GLOW_SRC_COMMON_H
#define GLOW_TORCH_GLOW_SRC_COMMON_H

#include "glow/Base/Tensor.h"
#include "glow/Base/Type.h"

#include "glow/Runtime/HostManager/HostManager.h"

#include <torch/csrc/jit/ir.h>

namespace glow {

extern bool GlowCompilePyTorchModule;
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

/// Given a PyTorch ScalarType \p ty, \returns a matching Glow ElemKind.
ElemKind scalarTypeToElemKind(c10::ScalarType ty);

/// Given a c10 typekind \p ty, \returns a matching Glow ElemKind.
ElemKind typeKindToElemKind(c10::TypeKind ty);

/// \returns the PyTorchLoaderSettings singleton to be used throughout Glow's
/// PyTorch model loading code.
PyTorchLoaderSettings &getPyTorchLoaderSettings();

/// \returns the HostManager singleton used to run all PyTorch graphs in Glow.
runtime::HostManager *getHostManager();

/// \returns the PyTorch symbol to be used for the PyTorch node which represents
/// the subgraph that Glow will compile and run.
const c10::Symbol &getGlowSymbol();

/// Executes custom fuse pass for the given \p graph and \p fuseSymbol.
void glowCustomFuse(std::shared_ptr<torch::jit::Graph> &graph,
                    at::Symbol fuseSymbol);

/// Register the glow::FusionGroup operator.
void registerGlowOp(const c10::Symbol &symbol);

/// Register the pass that fuses parts of the graph into a glow::FusionGroup. \p
/// enablePassFn is used to enable/disable the glow fusion pass once it's
/// registered.
void registerGlowFusionPass(std::function<bool()> enablePassFn);

/// Convenience method to register the glow fusion op and pass. \p
/// enablePassFn is used to enable/disable the glow fusion pass once it's
/// registered.
void registerGlowFusionOpAndPass(std::function<bool()> enablePassFn);

/// Given a PyTorch TensorType \p ptType, \returns a matching Glow Type.
glow::Type ptTypeToGlowType(const c10::TensorType &ptType);

/// Given a PyTorch Tensor \p ptTensor, \returns an unowned Glow Tensor with a
/// matching type backed by the same memory as ptTensor.
glow::Tensor ptTensorToGlowTensor(const at::Tensor &ptTensor);

/// Given a Glow Type \p glowType, \returns an empty PyTorch Tensor with a
/// matching type.
at::Tensor glowTypeToEmptyPTTensor(const glow::Type &glowType);

/// Fuse known sets of operators into compact ones.
void FuseKnownPatterns(std::shared_ptr<torch::jit::Graph> &graph);

} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_COMMON_H
