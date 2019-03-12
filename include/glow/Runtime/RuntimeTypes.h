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
#ifndef GLOW_RUNTIME_RUNTIMETYPES_H
#define GLOW_RUNTIME_RUNTIMETYPES_H

#include "glow/Backends/Backend.h"
#include "glow/Backends/BackendUtils.h"
#include "glow/Graph/Graph.h"
#include "glow/Support/Error.h"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace glow {

class ExecutionContext;

namespace runtime {

class DeviceManager;
using DeviceIDTy = size_t;
using RunIdentifierTy = size_t;

/// Map of DeviceIDTy -> DeviceManager.
using DeviceManagerMapTy = std::map<DeviceIDTy, std::unique_ptr<DeviceManager>>;

/// Callback type used by HostManager and DeviceManager, used to pass results of
/// an inference request back to the caller.
using ResultCBTy = std::function<void(runtime::RunIdentifierTy, llvm::Error,
                                      std::unique_ptr<ExecutionContext>)>;

/// Data structure that contains device constraint information for each device.
/// Used to communicate memory constraints and later costs to the Partitioner.
struct DeviceInfo {
  /// Available memory on device in bytes.
  size_t availableMemory;
  /// Available SRAM capacity in bytes.
  size_t sramCapacity;
  /// Peak compute on device in ops/second. Assumes all ops are in int8.
  /// TODO: distinguish between data types with different peak flops.
  float peakCompute;
  /// Peak memory bandwidth from DRAM on device in bytes/second.
  float peakDramBw;
  /// Peak memory bandwidth from SRAM on device in bytes/second.
  float peakSramBw;
  /// Peak ingress/egress PCI-E bandwidth from device in bytes/second.
  float peakPCIeBw;
};

/// Individual Node in the DAG for a given network. This contains all the
/// information needed to run the sub-network at inference time.
struct DAGNode {
  /// The children of this node, these are nodes that depend on the current
  /// node.
  std::vector<DAGNode *> children;
  /// Pointers to the parents of this node. This is used by the executor for
  /// determining if a given node has all dependencies met.
  std::vector<DAGNode *> parents;
  /// ID of the deviceManager that this network is assigned to.
  DeviceIDTy deviceID;
  /// The logicalDevice is an output of the Partitioner to indicate that two
  /// networks should be assigned to the same device.
  DeviceIDTy logicalDevice;
  /// Name assigned to the sub-network, this is the id that will be passed to
  /// the DeviceManager when requesting a run of the network.
  std::string name;
  /// Runtime bundle containing all the symbol information for this network at
  /// runtime.
  RuntimeBundle runtimeBundle;
};

/// This struct represents a DAG. The first element is the root of a DAG, and
/// the second one is a list of all rest nodes in this DAG.
using rootDAGNodeTy = std::unique_ptr<DAGNode>;
using nodesDAGNodeTy = std::vector<std::unique_ptr<DAGNode>>;
struct DAG {
  rootDAGNodeTy root;
  nodesDAGNodeTy nodes;
};

/// This list contains all the created DAGNodes from the Partitioner. The
/// contained DAGNodes can only refer to the DAGNodes from the same DAGListTy.
using DAGListTy = std::vector<DAG>;

/// This is the base class for DeviceManager configurations. Any specific
/// device can extend this class to contain information to identify
/// and configure the device manager. Additionally it needs to set it's kind_
/// member variable to it's correct BackendKind.
class DeviceConfig {
  const BackendKind backendKind_;

protected:
  DeviceConfig(BackendKind kind) : backendKind_(kind) {}

public:
  BackendKind getBackendKind() { return backendKind_; }
};

} // namespace runtime
} // namespace glow
#endif // GLOW_RUNTIME_RUNTIMETYPES_H
