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

  /// Dump Glow dot graph to file after model loading is finished.
  bool dumpGlowDag = false;

  /// A list of symbols for nodes that will be ignored by the Glow fuser and
  /// thus will not be fused to Glow.
  std::unordered_set<torch::jit::Symbol> opBlacklist;

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
std::shared_ptr<runtime::HostManager> getHostManager();

/// \returns the PyTorch symbol to be used for the PyTorch node which represents
/// the subgraph that Glow will compile and run.
const c10::Symbol &getGlowSymbol();

/// Given a PyTorch TensorType \p ptType, \returns a matching Glow Type.
glow::Type ptTypeToGlowType(const c10::TensorType &ptType);

/// Given a PyTorch Tensor \p ptTensor, \returns an unowned Glow Tensor with a
/// matching type backed by the same memory as ptTensor.
glow::Tensor ptTensorToGlowTensor(const at::Tensor &ptTensor);

/// Given a Glow Type \p glowType, \returns an empty PyTorch Tensor with a
/// matching type.
at::Tensor glowTypeToEmptyPTTensor(const glow::Type &glowType);

} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_COMMON_H
