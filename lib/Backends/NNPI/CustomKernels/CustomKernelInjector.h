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

#pragma once

#include "glow/Graph/Graph.h"

namespace glow {

/// Base class for callables that inject custom NNPI kernel nodes into a Glow
/// Function.
struct CustomKernelInjector {
  /// Given a Glow Function \p F and a node in that graph \p node, create a
  /// replacement for the node with a custom NNPI operator if possible. \returns
  /// the replacement node if one was created otherwise nullptr.
  /// NOTE: Injectors should return a replacement node but should not actually
  /// use the replacement node in the graph (for example by calling
  /// replaceAllUsesOfWith), this is to avoid modifying the graph in a way that
  /// will be visible to serialization.
  /// NOTE: for performance reasons implementations should return false quickly
  /// from tryInject if \p node is not relevent to the injector since all
  /// injectors are called on all nodes in every Function.
  virtual Node *tryInject(Function *F, Node *node) const = 0;
  virtual ~CustomKernelInjector() = default;
};
} // namespace glow
