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

#include "glow/Partitioner/PartitionerTypes.h"
#include "glow/Support/Error.h"

namespace glow {

using namespace runtime;

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

  /// Flag to set if the Partitioner should attempt to saturate the host, and
  /// use all available devices.
  bool saturateHost_;

  /// Flag to set if the funcitons in the module are areadly optimized. By
  /// default, the optimization should be done in Partitioner due to
  /// heterogeneous partition.
  bool optimized_;

  /// The struct contain user-defined partition info.
  PartitionConfig partitionConfig_;

  /// Get the representative function (the one with the largest input) and
  /// update the memSize.
  static Function *selectRepFunc(Module *parent, uint64_t &memSize);

  /// After getting the initial partitions, adjust the partitions to minimize
  /// communication and computation cost.
  void partitionsAdjust(NodeToFunctionMap &partitions,
                        uint64_t availableMemory);

  /// Assign nodes to partitions grouped by \p backendName and return the
  /// mapping.
  NodeToFunctionMap selectPartitions(Function *F, uint64_t availableMemory,
                                     llvm::StringRef backendName);

  /// Duplicates all networks in the module order to saturate the Host.
  void saturateHost(unsigned logicalDeviceCount);

  FunctionToBackendNameMap
  backendBasedPartition(Function *F, std::vector<Backend *> &backends,
                        CompilationContext &cctx);

  /// Performs a load balancing optimization pass to optimize for load
  /// balance in addition to respecting memory constraints.
  llvm::Error loadBalancedPartitioning(Function *F, DeviceIDTy numDevices,
                                       uint64_t availableMemory,
                                       llvm::StringRef backendName,
                                       NodeToFunctionMap &mapping);

  /// Given the node-function mapping, do the actual partitioning. If \p saveDAG
  /// is true, the DAG will be saved into partitions_, which is the final
  /// partition result.
  void doPartitioning(llvm::StringRef funcName, std::vector<Function *>,
                      NodeToFunctionMap &mapping, bool saveDAG);

  /// If there is no need to do any partition, just generate the DAGNode based
  /// on current functions in this module for backend \p backendName found in \p
  /// backendMap. \p cctx is used during optimization of the Function. \returns
  /// whether there was an error encountered.
  llvm::Error
  createDAGWithoutPartition(llvm::StringRef backendName,
                            std::map<std::string, BackendInfo> &backendMap,
                            CompilationContext &cctx);

  /// Get the map between the backend name and the concrete backend info (e.g.
  /// backend pointer, mem, number) used in this partiton. If there are backends
  /// need to be created, we use \p backendsHolder to hold them for memory
  /// purpose.
  void getBackendMap(std::map<std::string, BackendInfo> &backendMap,
                     std::vector<std::unique_ptr<Backend>> &backendsHolder,
                     std::vector<Backend *> &backends);

public:
  /// \p parent is the module which contains the functions need to be divided.
  /// Here we assume that all the functions in one module belong to a same
  /// "Function Family", that is, without considerting the "dynamic stuff" (i.e.
  /// batch size, input/output shape of each op), all the functions are
  /// identical. The required memory and computation cost for each op can be
  /// found in Module.
  /// The \p devices provides the cost model related to devices.
  /// Saturating the host will be enabled if \p saturateHost is true.
  /// \p optimized is false by default, which means the functions in this module
  /// are not optimized. \p partitionConfig contains the user defined partition
  /// info.
  Partitioner(Module *parent, const std::vector<DeviceInfo> &devices,
              bool saturateHost = false, bool optimized = false,
              PartitionConfig partitionConfig = PartitionConfig());

  /// Users can create Mock Backends and pass their points to test Graph
  /// Partitioning without actually register them in GLOW.
  Partitioner(Module *parent, const std::vector<DeviceInfo> &devices,
              const std::vector<Backend *> &backends, bool saturateHost = false,
              bool optimized = false);

  /// Based on partitionConfig_ passed into Partitioner, do the user-defined
  /// partition.
  llvm::Error PartitionFromConfig();

  /// Decompose each function in a module. Now we support partitioning a module
  /// among different type of devices. \p cctx is used during optimization of
  /// the Function. \returns whether there was an error encountered.
  llvm::Error Partition(CompilationContext &cctx);

  /// This partition approach is used in Glow Quantization Profiling flow. The
  /// backendBasedPartition is applied first in case there are heterogeneous
  /// backends. Then each sub-function will be compiled and run in CPU backend
  /// for profiling.
  llvm::Error QuantizationProfilingPartition(CompilationContext &cctx,
                                             Function *F,
                                             std::vector<Backend *> backends);

  /// Get the final partitions.
  DAGListTy &getPartitionResult() { return partitions_; }

  /// Dump the partition result to a dot file. Since now all functions belong to
  /// a function family and they have the same partition, we only dump the one
  /// function's partition.
  void dumpDAG(llvm::StringRef dotFilename) const;
};
} // namespace glow
#endif // GLOW_PARTITIONER_PARTITIONER_H
