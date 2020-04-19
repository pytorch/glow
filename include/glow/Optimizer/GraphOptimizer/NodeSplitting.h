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
#ifndef GLOW_OPTIMIZER_GRAPHOPTIMIZER_NODESPLITTING_H
#define GLOW_OPTIMIZER_GRAPHOPTIMIZER_NODESPLITTING_H

#include "glow/Base/Type.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Support/Error.h"

#include "llvm/ADT/ArrayRef.h"

#include <functional>

namespace glow {

/// Context provided by the node splitting procedure for the user to define node
/// splitting constraints. This context is related to one of the nodes which was
/// obtained during splitting.
struct SplitNodeContext {

  /// Original node which was used for splitting. When this context is provided
  /// the original node is still part of the graph so the user can also check
  /// graph connexion properties (like whether the original node is followed or
  /// or preceded by another node, etc).
  Node *origNode;

  /// Split node (potentially one of many) which was obtained by splitting the
  /// original node. When this context is provided, the split node is not (yet)
  /// part of the graph (orphan) so you are NOT allowed to check adjacent nodes
  /// but you are allowed to check node type, attributes or input/output types.
  Node *splitNode;

  /// Number of chunks the original node \ref origNode was split into, \ref
  /// splitNode being one of them.
  dim_t numChunks;
};

/// User defined constraint which verifies whether a given split configuration
/// given by the context \p ctx is allowed. \returns true if the configuration
/// is allowed (accepted) and false otherwise.
using SplitNodeConstraint = std::function<bool(const SplitNodeContext &ctx)>;

/// Function to split all the nodes from the function \p F for which there is
/// a split logic implemented using the given split node \p constraints. When
/// at least one of the constraints is not met the splitting is not done.
Error splitNodesWithConstraints(
    Function *F, const llvm::ArrayRef<SplitNodeConstraint> constraints);

/// Particular split node constraint which requires that the total amount of
/// memory of a split node (the size for all the inputs/outputs) be less than
/// a maximum value (in bytes).
inline SplitNodeConstraint getMaxMemSplitNodeConstraint(unsigned maxMem) {
  return [=](const SplitNodeContext &ctx) -> bool {
    return (ctx.splitNode->getTotMemSize() <= maxMem);
  };
}

/// Particular split node constraint which requires that a node is split exactly
/// in the given number of chunks. Note that this constraint will not be met if
/// the dimensions used for splitting are smaller than the number of chunks.
inline SplitNodeConstraint getNumChunksSplitNodeConstraint(dim_t numChunks) {
  return [=](const SplitNodeContext &ctx) -> bool {
    return (ctx.numChunks == numChunks);
  };
}
} // namespace glow

#endif // GLOW_OPTIMIZER_GRAPHOPTIMIZER_NODESPLITTING_H
