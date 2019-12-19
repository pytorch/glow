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
#ifndef GLOW_PARTITIONER_PARTITIONUTILS_H
#define GLOW_PARTITIONER_PARTITIONUTILS_H

#include "glow/Graph/Graph.h"
#include "glow/Partitioner/PartitionerTypes.h"
#include "llvm/ADT/DenseMap.h"

namespace glow {
/// Visit nodes if Function \p F in BFS order and return the nodes by levels
/// (the longest distance between one node and the root).
BFSLevel getBFSLevel(Function *F);

/// Given \p nodes, return a list of nodes who use any node in this set.
std::vector<Node *> getOutUsers(const NodesSet &nodes);

/// Given \p nodes, return a list of nodes who use only the nodes in this set or
/// constant.
std::vector<Node *> getOutUsersWithOnePredecessor(const NodesSet &nodes);

/// \returns the memory usage of the output caused by \p node who has users not
/// in the set \p nodes.
uint64_t getOutMemPerNode(const NodesSet &nodes, const Node *node);

/// Given a node, \returns the NodeSet of inputs of this node.
NodesSet getInputs(const Node *node);

/// Return the estimated op computation time based on \p backendInfo.
float getNodeComputeTime(const Node *node, const BackendInfo &backendInfo);

/// Given a node, \returns the memory usage of its inputs (i.e. Storage input).
uint64_t getNodeMemUsage(const Node *node);

/// Given nodes set \p currNodes and its memory usage info \p info, \returns the
/// new memory usage if \p newNode is added into \p currNodes.
GraphMemInfo updateGraphMemInfoByAddingNode(const NodesSet &currNodes,
                                            const GraphMemInfo &info,
                                            Node *newNode);

/// Return the memory usage of a given nodes set.
GraphMemInfo getGraphMemInfo(const NodesSet &nodes);

/// Return the memory usage of \p func function.
GraphMemInfo getFunctionMemory(Function *func);

/// Parse a node name string (e.g. "Div,Add") \p names, \returns a set of
/// NodeKinds corresponding to the names in the string.
std::set<Kinded::Kind> generateNodeKindsSet(llvm::StringRef names);

/// Log the info of current partition \p partitions.
void logPartitionInfo(const NodeToFunctionMap &partitions);
} // namespace glow
#endif // GLOW_PARTITIONER_PARTITIONUTILS_H
