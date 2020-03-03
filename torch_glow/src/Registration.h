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

#ifndef GLOW_TORCH_GLOW_SRC_REGISTRATION_H
#define GLOW_TORCH_GLOW_SRC_REGISTRATION_H

#include "CachingGraphRunner.h"

#include <torch/csrc/jit/ir/ir.h>

namespace glow {
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

/// Get the size of the global CachingGraphRunner map. For testing and debugging
/// purpose only.
size_t getGraphRunnerMapSize();

/// Store a CachingGraphRunner with a constucting functor \p graphRunnerBuilder
/// under a given \p key for later later use. This is so that a
/// CachingGraphRunner can be preloaded for a given graph and then stored until
/// the corresponding pt node is created for that graph.
std::shared_ptr<CachingGraphRunner>
setGraphRunnerForKey(const std::string &key,
                     std::function<std::shared_ptr<CachingGraphRunner>(void)>
                         graphRunnerBuilder);

/// Get a precreated CachingGraphRunner for a given \p key. \returns nullptr if
/// no CachingGraphRunner was registered for the given key.
std::shared_ptr<CachingGraphRunner>
getGraphRunnerForKey(const std::string &key);
} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_REGISTRATION_H
