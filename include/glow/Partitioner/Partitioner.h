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
#include "glow/Support/Error.h"

namespace glow {

using namespace runtime;

using MemUsageMapTy = std::unordered_map<Node *, uint64_t>;
using ComputeTimeMapTy = std::unordered_map<Node *, float>;
using PartitionCostMapTy = llvm::DenseMap<Function *, GraphMemInfo>;

/// Data structure that contains the info for each type of backend used for
/// partitioning.
struct BackendInfo {
  /// Num of the devices which has the same type of backend.
  size_t num = 0;
  /// The memory constraints for this backend.
  uint64_t memSize;
  /// Backend pointer.
  Backend *backend = nullptr;
};

/// Helper structure for building a partition. Records mapping of nodes in
/// the original function to destination partitions, along with a list of the
/// newly-created functions;
using NodeToFunctionMapTy = llvm::DenseMap<Node *, Function *>;

/// A mapping of newly-created functions along with a set of nodes sets. The
/// overloaded compare function to make sure the map is sorted by the key's
/// name(i.e. the function's name) which makes the optimization sequence is
/// consistent for each run.
struct FunctionNameComparator {
  bool operator()(const Function *lhs, const Function *rhs) const {
    return strcmp(lhs->getName().data(), rhs->getName().data()) < 0;
  }
};
using FunctionToNodesMapTy =
    std::map<Function *, NodesSetTy, FunctionNameComparator>;

using FunctionToBackendNameMapTy =
    std::map<Function *, std::string, FunctionNameComparator>;

class NodeToFunctionMap {

  /// Newly-created partitions.
  FunctionList functions_;

  /// Map of nodes in the original function to their target partition.
  NodeToFunctionMapTy nodeToFunction_;

  /// Map of the partitions to the backend which will be used for compiling
  /// this partition.
  FunctionToBackendNameMapTy functionToBackendName_;

  /// Map of sub-functions to their memory consumption.
  PartitionCostMapTy partitionCost_;

  /// Map of partitions and the logicalDeviceID. The partitions with the same
  /// logcialDeviceID will be assigned into the same physical device.
  std::map<Function *, std::vector<DeviceIDTy>> logicalDeviceIDMap_;

public:
  /// Create a new partition \p F, and map it with \p backendName.
  void createPartition(Function *F, llvm::StringRef backendName) {
    functions_.emplace_back(F);
    functionToBackendName_[F] = backendName;
  }

  std::string getPartitionBackendName(Function *F) const {
    DCHECK(functionToBackendName_.find(F) != functionToBackendName_.end())
        << "Unknown partition in Function: " << F->getName().str();
    return functionToBackendName_.find(F)->second;
  }

  /// Add a new Node->Function mapping.
  void add(Node *N, Function *F) { nodeToFunction_[N] = F; }

  /// Get list of functions contained in this map.
  const FunctionList &getPartitions() const { return functions_; }

  /// Get the list of logical device ID related to this function \p F.
  const std::vector<DeviceIDTy> getLogicalDeviceIDList(Function *F) const {
    if (logicalDeviceIDMap_.find(F) == logicalDeviceIDMap_.end()) {
      return {};
    }
    return logicalDeviceIDMap_.at(F);
  }

  void appendLogicalDeviceID(Function *F, DeviceIDTy id) {
    if (logicalDeviceIDMap_.find(F) == logicalDeviceIDMap_.end()) {
      logicalDeviceIDMap_.emplace(
          std::make_pair(F, std::vector<DeviceIDTy>{id}));
    } else {
      logicalDeviceIDMap_[F].push_back(id);
    }
  }
  /// attach \p map to current mapping.
  void insert(NodeToFunctionMap &map) {
    FunctionList flist = map.getPartitions();
    for (auto it = flist.begin(); it != flist.end(); ++it) {
      Function *func = *it;
      auto backendName = map.getPartitionBackendName(func);
      createPartition(func, backendName);
      GraphMemInfo cost = map.getGraphMemInfo(func);
      setGraphMemInfo(func, cost);
    }
    for (auto it = map.begin(); it != map.end(); ++it) {
      Node *n = it->first;
      Function *f = it->second;
      add(n, f);
    }
  }

  /// Map API.
  NodeToFunctionMapTy::iterator find(Node *N) {
    return nodeToFunction_.find(N);
  }
  NodeToFunctionMapTy::iterator begin() { return nodeToFunction_.begin(); }
  NodeToFunctionMapTy::iterator end() { return nodeToFunction_.end(); }

  Function *operator[](Node *n) { return nodeToFunction_[n]; }
  void deletePartition(Function *func) {
    functions_.remove(func);
    functionToBackendName_.erase(func);
    partitionCost_.erase(func);
  }

  /// Set the memory consumption \p cost for a partition \p func.
  void setGraphMemInfo(Function *func, GraphMemInfo cost) {
    partitionCost_[func] = cost;
  }

  /// Get the memory consumption for a partition \p func.
  GraphMemInfo getGraphMemInfo(Function *func) const {
    if (partitionCost_.find(func) == partitionCost_.end()) {
      return GraphMemInfo{};
    }
    return partitionCost_.find(func)->second;
  }
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
  std::vector<DeviceInfo> deviceInfo_;

  /// The backend pointers.
  std::vector<Backend *> backends_;

  /// The map between backend name and BackendInfo.
  std::map<std::string, BackendInfo> backendMap_;

  /// The map between partitions and the logicalDeviceID. The partitions with
  /// the same logicalDeviceID will be assigned into the same physical device.
  std::map<Function *, std::vector<DeviceIDTy>> logicalIDMap_;

  /// The number of logicalDevice IDs, i.e. the number of physical devices
  /// needed after partitions.
  DeviceIDTy logicalDeviceID_;

  /// The result of module partitioning.
  DAGListTy partitions_;

  /// Total memory (bytes) requested by one module.
  uint64_t memSize_;

  /// The map of each operator and the corresponding memory size.
  MemUsageMapTy memUsage_;

  /// The map of each operator and the compute runtime.
  ComputeTimeMapTy computeTime_;

  /// Flag to set if the Partitioner should attempt to saturate the host, and
  /// use all available devices.
  bool saturateHost_;

  // Flag to set if the funcitons in the module are areadly optimized. By
  // default, the optimization should be done in Partitioner due to
  // heterogeneous partition.
  bool optimized_;

  /// Get the representative function (the one with the largest input) and
  /// update the memSize.
  static Function *selectRepFunc(Module *parent, uint64_t &memSize);

  /// Get the minimal memory requirement for each op in the function \p F
  void initOpMemUsage(Function *F);

  /// Inititalize the minimal compute time for each op in the function \p F.
  void initOpComputeTime(Function *F);

  /// Combine the partitions if necessary : if all outside uses of the nodes in
  /// partition1 is in partition2, and the sum of memory consumption of
  /// partition1 and partition2 is less than availableMemory, combine partition1
  /// and partition2.
  void partitionsCombine(NodeToFunctionMap &partitions,
                         FunctionToNodesMapTy &nodesSet,
                         uint64_t availableMemory);

  /// After getting the initial partitions, adjust the partitions to minimize
  /// communication and computation cost.
  void partitionsAdjust(NodeToFunctionMap &partitions,
                        uint64_t availableMemory);

  /// Assign nodes to partitions grouped by \p backendName and return the
  /// mapping.
  NodeToFunctionMap selectPartitions(Function *F, uint64_t availableMemory,
                                     llvm::StringRef backendName);

  /// Assign a logicalDeviceID to each partition. It is possible that two
  /// partitions need to be assigned into 1 device due to the number of physical
  /// devices.
  DeviceIDTy assignLogicalDeviceID(NodeToFunctionMap &partitions);

  /// Check if \p partitions satisfies number of physical devices restriction.
  /// I.e. check if the number of logical devices is less than the given
  /// physical devices.
  llvm::Error
  logicalDevicesValidation(const NodeToFunctionMap &partitions) const;

  /// Check if the memory usage of each partition meets the physical device
  /// memory restriction.
  llvm::Error memoryUsageValidation(const NodeToFunctionMap &partitions) const;

  /// Duplicates all networks in the module order to saturate the Host.
  void saturateHost(unsigned logicalDeviceCount);

  FunctionToBackendNameMapTy
  backendBasedPartition(Function *F, std::vector<Backend *> &backends);

  /// Given the node-function mapping, do the actual partitioning. If \p saveDAG
  /// is true, the DAG will be saved into partitions_, which is the final
  /// partition result.
  void doPartitioning(llvm::StringRef funcName, std::vector<Function *>,
                      NodeToFunctionMap &mapping, bool saveDAG);

public:
  /// \p parent is the module which contains the functions need to be divided.
  /// Here we assume that all the functions in one module belong to a same
  /// "Function Family", that is, without considerting the "dynamic stuff" (i.e.
  /// batch size, input/output shape of each op), all the functions are
  /// identical. The required memory and computation cost for each op can be
  /// found in Module. The \p devices provides the cost model related to
  /// devices.
  Partitioner(Module *parent, const std::vector<DeviceInfo> &devices,
              bool saturateHost = false, bool optimized = false);

  /// Users can create Mock Backends and pass their points to test Graph
  /// Partitioning without actually register them in GLOW.
  Partitioner(Module *parent, const std::vector<DeviceInfo> &devices,
              const std::vector<Backend *> &backends, bool saturateHost = false,
              bool optimized = false);

  /// Get the map between the backend name and the concrete backend info (e.g.
  /// backend pointer, mem, number) used in this partiton. If there are backends
  /// need to be created, we use \p backendsHolder to hold them for memory
  /// purpose.
  void getBackendMap(std::map<std::string, BackendInfo> &backendMap,
                     std::vector<std::unique_ptr<Backend>> &backendsHolder,
                     std::vector<Backend *> &backends);

  /// If there is no need to do any partition, just generate the DAGNode based
  /// on current functions in this module for backend \p backendName found in \p
  /// backendMap. \p cctx is used during optimization of the Function. \returns
  /// whether there was an error encountered.
  llvm::Error
  createDAGWithoutPartition(llvm::StringRef backendName,
                            std::map<std::string, BackendInfo> &backendMap,
                            CompilationContext &cctx);

  /// Decompose each function in a module. Now we support partitioning a module
  /// among different type of devices. \p cctx is used during optimization of
  /// the Function. \returns whether there was an error encountered.
  llvm::Error Partition(CompilationContext &cctx);

  /// Get the partitions.
  DAGListTy &getPartitionResult() { return partitions_; }

  /// Dump the partition result to a dot file. Since now all functions belong to
  /// a function family and they have the same partition, we only dump the one
  /// function's partition.
  void dumpDAG(llvm::StringRef dotFilename) const;

  /// Get function for computeTime_
  ComputeTimeMapTy getComputeTime() const { return computeTime_; }

  /// Get function for memUsage_
  MemUsageMapTy getMemUsage() const { return memUsage_; }
};
} // namespace glow
#endif // GLOW_PARTITIONER_PARTITIONER_H
