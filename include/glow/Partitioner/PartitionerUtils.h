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

/// Return the memory usage of a given nodes set with a given \p contextCount.
GraphMemInfo getGraphMemInfo(const NodesSet &nodes, unsigned contextCount);

/// Return the memory usage of \p func function.
GraphMemInfo getFunctionMemory(Function *func);

/// Parse a node name string (e.g. "Div,Add") \p names, \returns a set of
/// NodeKinds corresponding to the names in the string.
std::set<Kinded::Kind> generateNodeKindsSet(llvm::StringRef names);

/// Log the info of current partition \p partitions.
void logPartitionInfo(const NodeToFunctionMap &partitions);

/// Print numBytesInTable, deviceID, cost and cost/numBytesInTable.
/// Print item from [start to end), with start inclusively and end
/// exclusively. If verbose_only is true, we use VLOG(1), otherwise we use
/// LOG(INFO).
void printSlsTableInfo(std::vector<SLSTableInfo>::iterator start,
                       std::vector<SLSTableInfo>::iterator end,
                       bool verbose_only = true);
void printSlsTableInfo(std::vector<SLSTableInfo> &slsTables,
                       bool verbose_only = true);

/// Print deviceId, used_memory, free_memory, cost, node_size, cost/used_memory.
/// Used memeory is calculated using \p nodesets and \p contextCount. If
/// verbose_only is true, we use VLOG(1), otherwise we use LOG(INFO).
void printSlsDeviceInfo(const std::vector<SLSDeviceInfo> &slsDevices,
                        const std::vector<NodesSet> &nodesets,
                        const unsigned contextCount, bool verbose_only);

// Returns whether \p node is an SLS node
bool isSLSNode(const Node *node);

// Returns whether all inputs to \p node are of the kind \p kind
bool checkNodeInputsAllKind(const Node *node, glow::Kinded::Kind kind);

/// Loop through slsDevices, assign \p table to first available \p slsDevices
/// that can fit \p table.
/// \returns Error if we could not find one.
Error assignSlsTableToFirstAvailableDevice(
    SLSTableInfo &table, std::vector<SLSDeviceInfo> &slsDevices,
    std::vector<NodesSet> &nodesets,
    std::vector<std::unordered_set<NodeValue>> &frontierValues,
    const unsigned contextCount);

/// Assign \p slsTables to \p slsDevices by:
/// 1. Sort \p slsTables by size decreasing.
/// 2. Split \p slsTables into two parts: large tables, and small tables where
/// large tables have numBytesInTable >
/// glow::runtime::flags::BigTableThresholdBytes.
/// 3. For large tables, we sort tables by size, and then for each table we
/// assign it to the device with lowest size.
/// 4. For small tables, we sort tables by cost, and then for each table we
/// assign it to the device with lowest cost.
///
/// \returns Error if we could not find a feasible partitioning plan to fit all
/// slsTables into slsDevices.
/// In case of error, all the inputs will be restored to original values.
Error assignSlsTablesToDevices(
    std::vector<SLSTableInfo> &slsTables,
    std::vector<SLSDeviceInfo> &slsDevices,
    std::vector<std::unordered_set<NodeValue>> &frontierValues,
    const unsigned contextCount);

/// Assign \p slsTables to \p slsDevices by:
/// Sort \p slsTables by size, then for each sls table, assign to slsDevice with
/// lowest cost.
///
/// \returns Error if we could not find a feasible allocation plan to fit all
/// slsTables into slsDevices.
/// In case of error, all the inputs will be restored to original values.
Error assignSlsTablesToDevicesGreedy(
    std::vector<SLSTableInfo> &slsTables,
    std::vector<SLSDeviceInfo> &slsDevices,
    std::vector<std::unordered_set<NodeValue>> &frontierValues,
    const unsigned contextCount);
} // namespace glow
#endif // GLOW_PARTITIONER_PARTITIONUTILS_H
