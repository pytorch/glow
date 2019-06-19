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
#ifndef GLOW_PARTITIONER_PARTITIONUTILS_H
#define GLOW_PARTITIONER_PARTITIONUTILS_H

#include "glow/Graph/Graph.h"
#include "llvm/ADT/DenseMap.h"

namespace glow {
using NodesSetTy = std::set<Node *>;

/// The memory usage of a subgraph (i.e. a list of nodes of a function).
struct GraphMemInfo {
  // The memory usage of all input nodes (whose predecessors are not included in
  // this subgraph) of this subgraph.
  uint64_t inMemSize;
  // The memory usage of all output nodes (whose successors are not included in
  // this subgraph) of this subgraph.
  uint64_t outMemSize;
  // The memory usage of all constants used in this subgraph.
  uint64_t constMemSize;

  GraphMemInfo() : inMemSize(0), outMemSize(0), constMemSize(0){};
  GraphMemInfo(uint64_t inMem, uint64_t outMem, uint64_t constMem)
      : inMemSize(inMem), outMemSize(outMem), constMemSize(constMem){};

  // Get the total memory size of each partition.
  uint64_t getTotalMemSize() const {
    return inMemSize + outMemSize + constMemSize;
  }

  bool equals(const GraphMemInfo &other) const {
    return inMemSize == other.inMemSize && outMemSize == other.outMemSize &&
           constMemSize == other.constMemSize;
  }
};

inline bool operator==(const GraphMemInfo &LHS, const GraphMemInfo &RHS) {
  return LHS.equals(RHS);
}

/// A list of <nodelist> with BFS order.
using BFSLevel = std::vector<std::vector<Node *>>;

/// Visit nodes if Function \p F in BFS order and return the nodes by levels
/// (the longest distance between one node and the root).
BFSLevel getBFSLevel(Function *F);

/// Given \p nodes, return a list of nodes who use any node in this set.
std::vector<Node *> getOutUsers(const NodesSetTy &nodes);

/// Given \p nodes, return a list of nodes who use only the nodes in this set or
/// constant.
std::vector<Node *> getOutUsersWithOnePredecessor(const NodesSetTy &nodes);

/// \returns the memory usage of the output caused by \p node who has users not
/// in the set \p nodes.
uint64_t getOutMemPerNode(const NodesSetTy &nodes, const Node *node);

/// Given nodes set \p currNodes and its memory usage info \p info, \returns the
/// new memory usage if \p newNode is added into \p currNodes.
GraphMemInfo updateGraphMemInfoByAddingNode(const NodesSetTy &currNodes,
                                            const GraphMemInfo &info,
                                            Node *newNode);

/// Return the memory usage of a given nodes set.
GraphMemInfo getGraphMemInfo(const NodesSetTy &nodes);

} // namespace glow
#endif // GLOW_PARTITIONER_PARTITIONUTILS_H
