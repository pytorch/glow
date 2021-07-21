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
#ifndef GLOW_PARTITIONER_PARTITIONERTYPES_H
#define GLOW_PARTITIONER_PARTITIONERTYPES_H

#include "glow/Graph/Graph.h"
#include "glow/Runtime/RuntimeTypes.h"

namespace glow {

using namespace runtime;

using NodesSet = std::set<Node *>;

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
  // The number of contexts reserved on the device, this affecting input/out
  // memory useage.
  unsigned contextCount;
  // Count of inputs to the graph.
  unsigned inputCount{0};
  // Count of inputs to the graph that are coming from a peer graph, i.e. are
  // the output of another graph, and not inputs to the original input model.
  unsigned inputFromPeerCount{0};
  // The memory usage of only deferred constants used in this subgraph.
  uint64_t deferredConstMemSize{0};

  GraphMemInfo()
      : inMemSize(0), outMemSize(0), constMemSize(0), contextCount(1){};
  GraphMemInfo(uint64_t inMem, uint64_t outMem, uint64_t constMem,
               unsigned count = 1)
      : inMemSize(inMem), outMemSize(outMem), constMemSize(constMem),
        contextCount(count){};

  /// Get the total memory size of each partition.
  uint64_t getTotalMemSize() const {
    return ((inMemSize + outMemSize) * contextCount) + constMemSize;
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

/// Data structure that contains the info for each type of backend used for
/// partitioning.
struct BackendInfo {
  /// Num of the devices which has the same type of backend.
  size_t num = 0;
  /// The memory constraints for this backend.
  uint64_t memSize;
  /// Maximum amount of input resources defaults to 0 if there is no limit.
  uint64_t inputCountMax{0};
  /// The following peakCompute, peakDramBw, peakSramBw, peakPCIeBw are from
  /// DeviceInfo_. Available SRAM capacity in bytes.
  uint64_t sramCapacity;
  /// Peak compute on device in ops/second. Assumes all ops are in int8.
  float peakCompute;
  /// Peak memory bandwidth from DRAM on device in bytes/second.
  float peakDramBw;
  /// Peak memory bandwidth from SRAM on device in bytes/second.
  float peakSramBw;
  /// Peak ingress/egress PCI-E bandwidth from device in bytes/second.
  float peakPCIeBw;
  /// Backend pointer.
  Backend *backend = nullptr;
  /// The non-supported nodes kind.
  std::set<Kinded::Kind> nonSupportedNodesKinds;
  /// The supported nodes kind.
  std::set<Kinded::Kind> supportedNodesKinds;
};

struct SLSTableInfo {
  Node *node;
  std::unordered_set<Node *> neighbors;
  std::unordered_set<NodeValue> frontier;
  uint64_t numBytesInTable;
  unsigned int deviceId;
  NodeValue slsResult;
  uint64_t cost;
};

struct SLSDeviceInfo {
  unsigned int deviceId;
  uint64_t memAvailableInBytes;
  size_t currentCost;
};

/// A mapping of newly-created functions along with a set of nodes sets. The
/// overloaded compare function to make sure the map is sorted by the key's
/// name(i.e. the function's name) which makes the optimization sequence is
/// consistent for each run.
struct FunctionNameComparator {
  bool operator()(const Function *lhs, const Function *rhs) const {
    return strcmp(lhs->getName().data(), rhs->getName().data()) < 0;
  }
};
using FunctionToNodesMap =
    std::map<Function *, NodesSet, FunctionNameComparator>;

using FunctionToBackendNameMap =
    std::map<Function *, std::string, FunctionNameComparator>;

class NodeToFunctionMap {
  /// Helper structure for building a partition. Records mapping of nodes in
  /// the original function to destination partitions, along with a list of the
  /// newly-created functions;
  using Map = llvm::DenseMap<Node *, Function *>;

  using PartitionCostMap = llvm::DenseMap<Function *, GraphMemInfo>;

  using BackendHintsMap = llvm::DenseMap<Function *, BackendHints>;

  using BackendSpecificOptsMap =
      llvm::DenseMap<Function *, BackendSpecificOptions>;

  /// Newly-created partitions.
  FunctionList functions_;

  /// Map of nodes in the original function to their target partition.
  Map nodeToFunction_;

  /// Map of the partitions to the backend which will be used for compiling
  /// this partition.
  FunctionToBackendNameMap functionToBackendName_;

  /// Map of sub-functions to their memory consumption.
  PartitionCostMap partitionCost_;

  /// BackendHints for this sub-function
  BackendHintsMap backendHints_;

  /// BackendSpecificOpts for this sub-function
  BackendSpecificOptsMap backendSpecificOpts_;

  /// Map of partitions and the logicalDeviceID. The partitions with the same
  /// logcialDeviceID will be assigned into the same physical device.
  std::map<Function *, std::vector<DeviceIDTy>> logicalDeviceIDMap_;

  /// Map of partitions and replication count.
  std::map<Function *, unsigned> replicationCountMap_;

public:
  /// Create a new partition \p F, and map it with \p backendName.
  void createPartition(Function *F, llvm::StringRef backendName) {
    functions_.emplace_back(F);
    functionToBackendName_[F] = backendName.str();
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

  void clearLogicalDeviceID() { logicalDeviceIDMap_.clear(); }

  void appendLogicalDeviceID(Function *F, DeviceIDTy id) {
    if (logicalDeviceIDMap_.find(F) == logicalDeviceIDMap_.end()) {
      logicalDeviceIDMap_.emplace(
          std::make_pair(F, std::vector<DeviceIDTy>{id}));
    } else {
      logicalDeviceIDMap_[F].push_back(id);
    }
  }

  void addReplicationCount(Function *F, unsigned count) {
    replicationCountMap_[F] = count;
  }

  unsigned getReplicationCount(Function *F) {
    auto it = replicationCountMap_.find(F);
    if (it == replicationCountMap_.end()) {
      return 1;
    } else {
      return it->second;
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
  Map::iterator find(Node *N) { return nodeToFunction_.find(N); }
  Map::iterator begin() { return nodeToFunction_.begin(); }
  Map::iterator end() { return nodeToFunction_.end(); }
  Function *operator[](Node *n) { return nodeToFunction_[n]; }

  void deletePartition(Function *func) {
    functions_.remove(func);
    functionToBackendName_.erase(func);
    partitionCost_.erase(func);
    backendHints_.erase(func);
    backendSpecificOpts_.erase(func);
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

  /// Set the backend hints for a partition \p func.
  void setBackendHints(Function *func, BackendHints hints) {
    backendHints_[func] = hints;
  }

  /// Get the backend hints for a partition \p func.
  BackendHints getBackendHints(Function *func) const {
    if (backendHints_.find(func) == backendHints_.end()) {
      return BackendHints{};
    }
    return backendHints_.find(func)->second;
  }

  /// Set the backend specific opts \p opts for a partition \p func.
  void setBackendSpecificOpts(Function *func,
                              const BackendSpecificOptions &opts) {
    backendSpecificOpts_[func] = opts;
  }

  /// Get the backend hints for a partition \p func.
  BackendSpecificOptions getBackendSpecificOpts(Function *func) const {
    if (backendSpecificOpts_.find(func) == backendSpecificOpts_.end()) {
      return BackendSpecificOptions{};
    }
    return backendSpecificOpts_.find(func)->second;
  }
};

} // namespace glow
#endif // GLOW_RUNTIME_PARTITIONERTYPES_H
