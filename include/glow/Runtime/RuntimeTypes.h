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

#include "glow/Backends/BackendUtils.h"
#include "glow/Graph/Context.h"
#include "glow/Graph/Graph.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace glow {
namespace runtime {

using NetworkIDty = size_t;
using DeviceIDty = size_t;
using RunIdentifierTy = size_t;

/// Enum to communicate results when communicating with device at initialization
/// and runtime.
enum ResultCode { READY, EXECUTED, FAILED, CANCELLED };

/// Data structure that contains device constraint information for each device.
/// Used to communicate memory constraints and later costs to the Partitioner.
struct DeviceInfo {
  /// Available memory on device in bytes.
  uint64_t availableMemory;
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
  DeviceIDty deviceID;
  /// The logicalDevice is an output of the Partitioner to indicate that two
  /// networks should be assigned to the same device.
  DeviceIDty logicalDevice;
  /// Name assigned to the sub-network, this is the id that will be passed to
  /// the DeviceManager when requesting a run of the network.
  std::string name;
  /// Runtime bundle containing all the symbol information for this network at
  /// runtime.
  RuntimeBundle runtimeBundle;
};

} // namespace runtime
} // namespace glow
#endif // GLOW_RUNTIME_RUNTIMETYPES_H
