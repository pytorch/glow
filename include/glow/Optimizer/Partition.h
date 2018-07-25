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
#ifndef GLOW_OPTIMIZER_PARTITION_H
#define GLOW_OPTIMIZER_PARTITION_H

#include "glow/Graph/Graph.h"

#include <llvm/ADT/DenseMap.h>

namespace glow {

/// Data structure representing a set of interdependent Functions and the
/// dependences between them.
class FunctionGraph {
  /// Dependency map type.
  using Map = llvm::DenseMap<Function *, FunctionList>;

  /// Type of dependency list.
  using MappedType = Map::mapped_type;

  /// List of functions in this graph.
  FunctionList functions_;

  /// Input dependencies of each function.
  Map inputs_;

public:
  /// Ctor.
  FunctionGraph(const FunctionList &functions);

  /// Dtor.
  ~FunctionGraph() = default;

  /// Move ctor.
  FunctionGraph(FunctionGraph &&G) = default;
  FunctionGraph &operator=(FunctionGraph &&G) = default;

  /// FunctionGraph is non-copyable to ensure efficiency.
  FunctionGraph(const FunctionGraph &G) = delete;
  FunctionGraph &operator=(const FunctionGraph &G) = delete;

  /// Get list of functions in this graph.
  const FunctionList &getFunctions() const { return functions_; }

  /// Get input dependences for a function.
  const MappedType &getInputs(Function *F) const {
    auto it = inputs_.find(F);
    assert(it != inputs_.end() && "No inputs found for function in graph");
    return it->second;
  }

  /// Record that \p F depends on \p inputF.
  void add(Function *F, Function *inputF) { inputs_[F].push_back(inputF); }
};

/// Split an input Function into a FunctionGraph.
FunctionGraph partition(Function *F);

} // namespace glow

#endif // GLOW_OPTIMIZER_PARTITION_H
