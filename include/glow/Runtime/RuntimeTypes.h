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

#include "glow/Graph/Graph.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace glow {
namespace runtime {

using NetworkID = size_t;
using DeviceID = size_t;

/// Enum to communicate results when communicating with device at initialization
/// and runtime.
enum ResultCode { READY, EXECUTED, FAILED, CANCELLED };

struct DeviceMemoryInfo {
  /// Available memory on device in bytes.
  uint64_t availableMemory;
  /// Total useable memory on device in bytes.
  uint64_t maximumUsableMemory;
};

/// Data structure that contains everything needed by the executor to execute a
/// network at runtime.
struct ExecutionDAG {
  /// Vector of networks to be run.
  std::vector<NetworkID> networks;
  /// Mapping of networkID to the deviceID where the network is loaded.
  std::unordered_map<NetworkID, DeviceID> devices;
  /// vector of root nodes, these have no dependencies and can be run first.
  std::vector<NetworkID> roots;
  /// Mapping of networkID to a vector of it's dependancies.
  std::unordered_map<NetworkID, std::vector<NetworkID>> dependencies;
  /// Mapping of networkID to a vector of networks that depend on it.
  std::unordered_map<NetworkID, std::vector<NetworkID>> dependents;
  /// Mapping of NetworkID to a vector of input placeholder names for the
  /// network.
  std::unordered_map<NetworkID, std::vector<std::string>> inputs;
  /// Mapping of NetworkID to a vector of output placeholder names for the
  /// network.
  std::unordered_map<NetworkID, std::vector<std::string>> outputs;
};

/// Data structure containing the output from the Partitioner. It is consumed by
/// the Provisioner and used to generate an executionDAG.
struct DependencyDAG {
  /// Vector of unique pointers to modules containing sub-networks.
  std::vector<std::unique_ptr<Module>> modules;
  /// Vector of root nodes.
  std::vector<Module *> roots;
  /// Mapping of Module * to a vector of it's dependancies.
  std::unordered_map<Module *, std::vector<Module *>> dependencies;
  /// Mapping of Module * to a vector of networks that depend on it.
  std::unordered_map<Module *, std::vector<Module *>> dependents;
};

} // namespace runtime
} // namespace glow
#endif // GLOW_RUNTIME_RUNTIMETYPES_H
