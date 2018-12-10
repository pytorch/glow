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
#include "glow/Runtime/RuntimeTypes.h"

#include "llvm/ADT/DenseMap.h"

#include <map>
#include <set>
#include <string>

namespace glow {

using namespace runtime;

using MemUsageMap = std::unordered_map<Node *, unsigned>;

/// Helper structure for building a partition. Records a mapping of nodes in the
/// original function to destination partitions, along with a list of the
/// newly-created functions.
class NodeToFunctionMap {
  using Map = llvm::DenseMap<Node *, Function *>;

  /// Newly-created partitions.
  FunctionList functions_;

  /// Map of nodes in the original function to their target partition.
  Map nodeToFunction_;

public:
  /// Create a new partition \p F.
  void createPartition(Function *F) { functions_.emplace_back(F); }

  /// Add a new Node->Function mapping.
  void add(Node *N, Function *F) { nodeToFunction_[N] = F; }

  /// Get list of functions contained in this map.
  const FunctionList &getPartitions() const { return functions_; }

  /// Map API.
  Map::iterator find(Node *N) { return nodeToFunction_.find(N); }
  Map::iterator begin() { return nodeToFunction_.begin(); }
  Map::iterator end() { return nodeToFunction_.end(); }
  Function *operator[](Node *n) { return nodeToFunction_[n]; }
};

/// The struct contains all the created DAGNodes. This DAGNodeList owns all the
/// DAGNodes, which cannot outlive the DAGNodeList. In addition, the DAGNodes
/// can only refer to the DAGNodes from the same DAGNodeList, and they can use
/// the raw pointers to refer to each other since they are in the same
/// DAGNodeList.
struct DAGNodeList {
  /// The root DAGNode pointer of each graph/function.
  std::vector<std::unique_ptr<DAGNode>> roots;
  /// The non-root DAGNode pointers.
  std::vector<std::unique_ptr<DAGNode>> nodes;
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
  DAGNodeList partitions_;

  /// Total memory (bytes) requested by one module.
  size_t memSize_;

  /// The map of each operator and the corresponding memory size.
  MemUsageMap memUsage_;

  /// Get the representative function (the one with the largest input) and
  /// update the memSize.
  static Function *selectRepFunc(Module *parent, size_t &memSize);

  /// Get the minimal memory requirement for each op in the representive
  /// function.
  void initOpMemUsage();

  /// Assign nodes to partitions and return the mapping.
  NodeToFunctionMap selectPartitions(Function *F, unsigned availableMemory);

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
  DAGNodeList &Partition();
};
} // namespace glow
#endif // GLOW_PARTITIONER_PARTITIONER_H
