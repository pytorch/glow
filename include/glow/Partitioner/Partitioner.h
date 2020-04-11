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
#ifndef GLOW_PARTITIONER_PARTITIONER_H
#define GLOW_PARTITIONER_PARTITIONER_H

#include "glow/Partitioner/PartitionerBase.h"
#include "glow/Support/Error.h"

namespace glow {

using namespace runtime;

/// Given a module, partitions each of the its functions into multiple ones
/// based on memory constraints and minimizes the communication cost.
class Partitioner final : public PartitionerBase {
  /// The module that needs to be decomposed.
  Module *module_;

  /// The representative function used for partition. We choose the function who
  /// has the largest memory size.
  Function *F_;

  /// True if there are more than 1 type of backends.
  bool multiBackendNames_;

  /// The cost model related to device.
  std::vector<DeviceInfo> deviceInfo_;

  /// The backends created in Partitioner. Used for function optimization.
  std::vector<std::unique_ptr<Backend>> backendHolder_;

  /// The raw backend pointers.
  std::vector<Backend *> backends_;

  /// The map between backend name and BackendInfo.
  std::map<std::string, BackendInfo> backendMap_;

  /// The map between partitions and the logicalDeviceID. The partitions with
  /// the same logicalDeviceID will be assigned into the same physical device.
  std::map<Function *, std::vector<DeviceIDTy>> logicalIDMap_;

  /// The number of logicalDevice IDs, i.e. the number of physical devices
  /// needed after partitions.
  DeviceIDTy logicalDeviceID_;

  /// Total memory (bytes) requested by one module.
  uint64_t memSize_;

  /// Flag to set if the funcitons in the module are areadly optimized. By
  /// default, the optimization should be done in Partitioner due to
  /// heterogeneous partition.
  bool optimized_;

  /// The struct contain user-defined partition info.
  PartitionConfig partitionConfig_;

  /// Get the representative function (the one with the largest input) and
  /// update the memSize.
  static Function *selectRepFunc(Module *parent, uint64_t &memSize);

  /// Initialization. Called in class constructor.
  void init();

  /// Verify the generated functions in module, and \returns error if any
  /// function is invalid. Dump partition logs from \p partitions and \p
  /// mapping.
  Error finalize(const DAGListTy &partitions, const NodeToFunctionMap &mapping);

  /// After getting the initial partitions, adjust the partitions to minimize
  /// communication and computation cost.
  void partitionsAdjust(NodeToFunctionMap &partitions,
                        uint64_t availableMemory);

  /// Assign nodes to partitions grouped by \p backendName and return the
  /// mapping.
  NodeToFunctionMap selectPartitions(Function *F, uint64_t availableMemory,
                                     llvm::StringRef backendName);

  /// Duplicates \p partitions in the module order to saturate the Host. \p
  /// logicalDeviceCount is the number of logical devices used by the current
  /// partitions. For example: If a network is partitioned into two parts (\p
  /// logicalDeviceCount) and there are six devices this would duplicate the
  /// network three times.
  void saturateHost(unsigned logicalDeviceCount, const DAGListTy &partitions);

  /// Partition a function \p F based on backends \p backends. \returns the
  /// final partition result(or an err) and a map between partitions and backend
  /// names. \p cctx is used for functions optimization.
  Expected<DAGListTy>
  backendBasedPartition(FunctionToBackendNameMap &funcToBackend, Function *F,
                        std::vector<Backend *> &backends,
                        CompilationContext &cctx);

  /// If there is no need to do any partition, just generate the DAGNode based
  /// on current functions in this module for backend \p backendName found in \p
  /// backendMap. \p cctx is used for function optimization. \returns the
  /// partition result or an error.
  Expected<DAGListTy>
  createDAGWithoutPartition(llvm::StringRef backendName,
                            std::map<std::string, BackendInfo> &backendMap,
                            CompilationContext &cctx);

  /// Create the map between the backend name and the concrete backend info
  /// (e.g. backend pointer, mem, number) used in this partiton. If there are
  /// backends need to be created, we use \p backendsHolder to hold them for
  /// memory purpose.
  void genBackendMap(std::map<std::string, BackendInfo> &backendMap,
                     std::vector<std::unique_ptr<Backend>> &backendsHolder,
                     std::vector<Backend *> &backends);

  struct SLSTableInfo {
    Node *node;
    uint64_t numBytesInTable;
    size_t numElementsPerRowUpperBound;
    size_t numIndices;
    unsigned int deviceId;
    NodeValue slsClipResult;
  };

  struct SLSDeviceInfo {
    unsigned int deviceId;
    uint64_t memAvailableInBytes;
    size_t currentCost;
  };

  /// Helper function for SparseNN Partitioning scheme. Checks for each
  /// kind of SLS table and appends their metadata to the vector.
  template <typename SLSType>
  void appendSLSTable(Node &node, std::vector<SLSTableInfo> &slsTables);

  /// Helper function for SparseNN partitioning. Inserts concats into SLS
  /// partition and corresponding slices into non-SLS partitions
  void sparseNNInsertSplitConcat(Function *F,
                                 std::vector<SLSDeviceInfo> slsDevices,
                                 std::vector<SLSTableInfo> slsTables,
                                 PartitionConfig &partitionConfig);

  /// Returns info for the default device of the backend. If multiple devices,
  /// returns the first one.
  const DeviceInfo &getDeviceInfoForBackend(llvm::StringRef backendName);

public:
  /// \p parent is the module which contains the functions need to be divided.
  /// Here we assume that all the functions in one module belong to a same
  /// "Function Family", that is, without considerting the "dynamic stuff" (i.e.
  /// batch size, input/output shape of each op), all the functions are
  /// identical. The required memory and computation cost for each op can be
  /// found in Module.
  /// The \p devices provides the cost model related to devices.
  /// \p optimized is false by default, which means the functions in this module
  /// are not optimized. \p partitionConfig contains the user defined partition
  /// info.
  Partitioner(Module *parent, const std::vector<DeviceInfo> &devices,
              bool optimized = false,
              PartitionConfig partitionConfig = PartitionConfig());

  /// Users can create Mock Backends and pass their points to test Graph
  /// Partitioning without actually register them in GLOW.
  Partitioner(Module *parent, const std::vector<DeviceInfo> &devices,
              const std::vector<Backend *> &backends, bool optimized = false);

  /// Based on \p partitionConfig passed into Partitioner, do user-defined
  /// partition.
  Expected<DAGListTy>
  partitionFromConfig(const PartitionConfig &partitionConfig,
                      CompilationContext &cctx);

  /// Based on \p cctx, setup all data structures needed for a DAG.
  /// cctx.prepartitionedConfig contains the Functions which are already
  /// partitioned and connected via Placeholders.
  Expected<DAGListTy> setupPrepartitionedModule(CompilationContext &cctx);

  /// This partition approach is used in Glow Quantization Profiling flow. The
  /// backendBasedPartition is applied first in case there are heterogeneous
  /// backends. Then each sub-function will be compiled and run in CPU backend
  /// for profiling. \p cctx is used for function optimization. \returns the
  /// partition result or an error.
  Expected<DAGListTy> quantizationProfilingPartition(CompilationContext &cctx);

  /// This partition approch first do the partition based on backend types, and
  /// then based on cost models(memory usage and performance). \p cctx is used
  /// for function optimization. \returns the partition result or an error.
  Expected<DAGListTy> heterogeneousPartition(CompilationContext &cctx);

  /// This partition approach is an experimental one. It tries to balance the
  /// workloads of each accelerator/device in addition to respecting memory
  /// constraints. \p numDevices is the minimal number of partition. That is,
  /// after loadBalancedPartition, the network will be devided up into at lease
  /// \p numDevices sub-networks. Now it is overwritten inside of
  /// loadBalcnedPartition. But in the future, it can be manually defined by
  /// users.
  Expected<DAGListTy> loadBalancedPartition(CompilationContext &cctx,
                                            size_t numDevices = 0);

  // This partition approach is meant for SparseNN models. The SLS tables are
  // split across logical devices and the non-SLS nodes are assigned in a
  // round-robin fashion to all logical devices.
  Expected<DAGListTy> partitionSparseNN(CompilationContext &cctx);

  /// Decompose each function in a module. Given the parameters, this function
  /// will choose different partition approches supported in this class:
  /// heterogeneous partition, user-defined partition or quantization profiling.
  /// \p cctx is used for function optimization. \returns the partition result
  /// or an error.
  Expected<DAGListTy> partition(CompilationContext &cctx) override;
};
} // namespace glow
#endif // GLOW_PARTITIONER_PARTITIONER_H
