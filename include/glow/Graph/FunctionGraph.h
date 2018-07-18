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
#ifndef GLOW_GRAPH_FUNCTIONGRAPH_H
#define GLOW_GRAPH_FUNCTIONGRAPH_H

#include "glow/Graph/Graph.h"

#include <unordered_set>

namespace glow {

/// Data structure representing a set of interdependent Functions and the
/// dependences between them.
class FunctionGraph {
  /// List of functions in this graph.
  FunctionList functions;

  /// Set of variables representing dependences in the graph.
  std::unordered_set<Variable *> channels;

  /// Input varabiels of each function.
  std::unordered_map<Function *, VariablesList> inputs;

  /// Output variables of each function.
  std::unordered_map<Function *, VariablesList> outputs;

public:
  /// Ctor.
  FunctionGraph(FunctionList fs, std::unordered_set<Variable *> vs);

  /// Move ctor.
  FunctionGraph(FunctionGraph &&G) = default;

  /// FunctionGraph is non-copyable for efficiency.
  FunctionGraph(const FunctionGraph &G) = delete;

  /// Get list of functions in this graph.
  FunctionList &getFunctions() { return functions; }

  /// Get set of dependences in this graph.
  const std::unordered_set<Variable *> &getChannels() const { return channels; }

  /// Get input dependences for a function.
  const VariablesList &getInputs(Function *F) const { return inputs.at(F); }

  /// Get output dependences for a function.
  const VariablesList &getOutputs(Function *F) const { return outputs.at(F); }
};

} // namespace glow

#endif // GLOW_GRAPH_FUNCTIONGRAPH_H
