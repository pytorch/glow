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
#ifndef GLOW_GRAPH_GRAD_H
#define GLOW_GRAPH_GRAD_H

#include "glow/Graph/Node.h"

#include <unordered_map>

namespace glow {

class Function;

/// A Helper class that manages the mapping between gradients and activations,
/// and helps to accumulate gradients into variables.
class GraphGradMapper {
  /// The graph that we mutate.
  Function *F_;
  /// Maps activation values to their gradient values.
  std::unordered_map<NodeValue, NodeValue> map_;

public:
  GraphGradMapper(Function *F) : F_(F) {}

  /// \register the node \p grad as the grad of \p activation. If the node is
  /// already registered then create an 'add' node that accumulates the gradient
  /// into the grad buffer.
  void addGradient(NodeValue activation, NodeValue grad);

  /// \returns the node that \p activation is mapped to.
  NodeValue getGradient(NodeValue activation);

  /// \returns True if the node \p activation is mapped.
  bool hasGradient(NodeValue activation);
};

} // namespace glow

#endif // GLOW_GRAPH_GRAD_H
