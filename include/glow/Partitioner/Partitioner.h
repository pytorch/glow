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
#ifndef GLOW_PARTITIONER_PARTITIONER_H
#define GLOW_PARTITIONER_PARTITIONER_H

#include "glow/Graph/Graph.h"
#include "glow/Partitioner/PartitionerUtils.h"
#include "glow/Runtime/RuntimeTypes.h"

namespace glow {

using namespace runtime;

using MemUsageMapTy = std::unordered_map<Node *, uint64_t>;
using ComputeTimeMapTy = std::unordered_map<Node *, float>;
using NodesSetTy = std::set<Node *>;
using PartitionCostMapTy = llvm::DenseMap<Function *, GraphMemInfo>;

/// Helper structure for building a partition. Records 1) a mapping of nodes in
/// the original function to destination partitions, along with a list of the
/// newly-created functions; 2) a mapping of newly-created functions aalong with
/// a set of nodes sets.
using NodeToFunctionMapTy = llvm::DenseMap<Node *, Function *>;
using FunctionToNodesMapTy = llvm::DenseMap<Function *, NodesSetTy>;

class NodeToFunctionMap {

  /// Newly-created partitions.
  FunctionList functions_;

  /// Map of nodes in the original function to their target partition.
  NodeToFunctionMapTy nodeToFunction_;

  /// Map of sub-functions to their memory consumption.
  PartitionCostMapTy partitionCost_;

public:
  /// Create a new partition \p F.
  void createPartition(Function *F) { functions_.emplace_back(F); }

  /// Add a new Node->Function mapping.
  void add(Node *N, Function *F) { nodeToFunction_[N] = F; }

  /// Get list of functions contained in this map.
  const FunctionList &getPartitions() const { return functions_; }

  /// Map API.
  NodeToFunctionMapTy::iterator find(Node *N) {
    return nodeToFunction_.find(N);
  }
  NodeToFunctionMapTy::iterator begin() { return nodeToFunction_.begin(); }
  NodeToFunctionMapTy::iterator end() { return nodeToFunction_.end(); }

  Function *operator[](Node *n) { return nodeToFunction_[n]; }
  void deletePartition(Function *func) { functions_.remove(func); }

  /// Set the memory consumption \p cost for a partition \p func.
  void setGraphMemInfo(Function *func, GraphMemInfo cost) {
    partitionCost_[func] = cost;
  }

  /// Get the memory consumption for a partition \p func.
  GraphMemInfo getGraphMemInfo(Function *func) { return partitionCost_[func]; }
};

/// Given a module, partitions each of the its functions into multiple ones
/// based on memory constraints and minimizes the communication cost.
class Partitioner {
  /// The module that needs to be decomposed.
  Module *module_;

  /// The representative function used for partition. We choose the function who
  /// has the largest memory size.
  Function *F_;

  /// The cost model related to device.
  const std::vector<DeviceInfo> &deviceInfo_;

  /// The result of module partitioning.
  DAGListTy partitions_;

  /// Total memory (bytes) requested by one module.
  uint64_t memSize_;

  /// The map of each operator and the corresponding memory size.
  MemUsageMapTy memUsage_;

  /// The map of each operator and the compute runtime.
  ComputeTimeMapTy computeTime_;

  /// Get the representative function (the one with the largest input) and
  /// update the memSize.
  static Function *selectRepFunc(Module *parent, uint64_t &memSize);

  /// Get the minimal memory requirement for each op in the representive
  /// function.
  void initOpMemUsage();

  /// Inititalize the minimal compute time for each op in the function.
  void initOpComputeTime();

  /// Combine the partitions if necessary : if all outside uses of the nodes in
  /// /// partition1 is in partition2, and the sum of memory consumption of
  /// partition1 and partition2 is less than availableMemory, combine partition1
  /// and partition2.
  void partitionsCombine(NodeToFunctionMap &partitions,
                         FunctionToNodesMapTy &nodesSet,
                         uint64_t availableMemory);

  /// After getting the intial partitions, ajust the partitions to miminize
  /// communication and computation cost.
  void partitionsAdjust(NodeToFunctionMap &partitions,
                        uint64_t availableMemory);

  /// Assign nodes to partitions and return the mapping.
  NodeToFunctionMap selectPartitions(Function *F, uint64_t availableMemory);

  /// Adjust a logicalDevice ID to each DAGNode. It is possible that two
  /// sub-functions need to be assigned into 1 device due to the memory
  /// constraits.
  void adjustLogicalDeviceID(DAGNode *DAG, int num);

  /// Given the node-function mapping, do the actual partitioning.
  void doPartitioning(Function *F, NodeToFunctionMap &mapping);

public:
  /// \p parent is the module which contains the functions need to be divided.
  /// Here we assume that all the functions in one module belong to a same
  /// "Function Family", that is, without considerting the "dynamic stuff" (i.e.
  /// batch size, input/output shape of each op), all the functions are
  /// identical. The required memory and computation cost for each op can be
  /// found in Module. The \p devices provides the cost model related to
  /// devices.
  Partitioner(Module *parent, const std::vector<DeviceInfo> &devices);

  /// Decompose each function in a module and return a list of DAGNodes.
  DAGListTy &Partition();

  /// Get function for computeTime_
  ComputeTimeMapTy getComputeTime() const { return computeTime_; }
};
} // namespace glow
#endif // GLOW_PARTITIONER_PARTITIONER_H
