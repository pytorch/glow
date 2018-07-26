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

/// Maps a set of functions to the set of functions it depends on.  The
/// execution of a function cannot begin until all its dependencies have
/// executed.
class FunctionDAG {
  /// Dependency map type.
  using Map = llvm::DenseMap<Function *, FunctionList>;

  /// Type of dependency list.
  using MappedType = Map::mapped_type;

  /// List of functions in this graph.
  FunctionList functions_;

  /// Input dependencies of each function.
  Map dependencies_;

public:
  /// Ctor.
  FunctionDAG(const FunctionList &functions);

  /// Dtor.
  ~FunctionDAG() = default;

  /// Move ctor.
  FunctionDAG(FunctionDAG &&G) = default;
  FunctionDAG &operator=(FunctionDAG &&G) = default;

  /// FunctionDAG is non-copyable to ensure efficiency.
  FunctionDAG(const FunctionDAG &G) = delete;
  FunctionDAG &operator=(const FunctionDAG &G) = delete;

  /// Get list of functions in this graph.
  const FunctionList &getFunctions() const { return functions_; }

  /// Get dependencies for a function.
  const MappedType &getDependencies(Function *F) const {
    auto it = dependencies_.find(F);
    assert(it != dependencies_.end() && "No dependencies found for function in graph");
    return it->second;
  }

  /// Record that \p F depends on \p inputF.
  void add(Function *F, Function *inputF) { dependencies_[F].push_back(inputF); }

  /// Verify that function graph is well-formed (acyclic, topologically sorted).
  bool verify() const;
};

/// Split an input Function into a FunctionDAG.
FunctionDAG partition(Function *F);

} // namespace glow

#endif // GLOW_OPTIMIZER_PARTITION_H
