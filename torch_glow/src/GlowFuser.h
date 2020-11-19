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

#ifndef GLOW_TORCH_GLOW_SRC_GLOW_FUSER_H
#define GLOW_TORCH_GLOW_SRC_GLOW_FUSER_H

#include <torch/csrc/jit/ir/ir.h>

#include "PyTorchCommon.h"

namespace glow {
/// Registers Glow's default symbol on the first call to this function.
/// Later calls are NOP.
void registDefaultGlowFusionSymbolOnce();

/// Fuse nodes in \p graph that are supported by glow into a subgraph in a node
/// with symbol \p kind. NOTE: kind must be registered with jit before calling
/// this function.
void glowCustomFuse(std::shared_ptr<torch::jit::Graph> graph,
                    const PyTorchLoaderSettings &settings, at::Symbol kind);

/// Fuse nodes in \p graph that are supported by glow into a subgraph in Glow
/// fusion group nodes using the settings in \p settings.
void glowCustomFuse(std::shared_ptr<torch::jit::Graph> graph,
                    const PyTorchLoaderSettings &settings);

/// Fuse nodes in \p graph that have a kind in \p acceptableKinds into a
/// subgraph in Glow fusion group nodes.
void glowCustomFuseDebug(std::shared_ptr<torch::jit::Graph> graph,
                         const PyTorchLoaderSettings &settings,
                         std::vector<std::string> acceptableKinds);
} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_GLOW_FUSER_H
