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
#ifndef GLOW_RUNTIME_HOSTMANAGERR_HOSTMANAGER_H
#define GLOW_RUNTIME_HOSTMANAGERR_HOSTMANAGER_H
#include "glow/Backends/Backend.h"
#include "glow/Backends/DeviceManager.h"
#include "glow/Graph/Graph.h"
#include "glow/Runtime/RuntimeTypes.h"

#include <atomic>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace glow {
namespace runtime {

/// Callback signalling the result of running a function.
struct DeviceConfig {
  std::string deviceName;
  BackendKind backendKind;
};

class Executor;

class Provisioner;

/// The HostManager serves as an entry point into the Runtime environment. It
/// provides an interface to add, run, and evict networks from the host. It
/// handles DeviceManager initialization, houses the Executor, and calls into
/// the Partitioner and Provisioner for network initialization.
class HostManager final {
  /// Count of current in-flight networks being run. Atomic to allow
  /// concurrency in runNetwork.
  std::atomic<size_t> activeRequestCount_{0};

  /// Count of total requests, this is used as a run ID. Atomic to allow
  /// concurrency in runNetwork.
  std::atomic<size_t> totalRequestCount_{0};

  /// Limit maximum count of networks run at once. Hardcoded for now this
  /// should be a configurable value. Above this limite the HostManager will
  /// refuse additional request and return Failed.
  const unsigned int activeRequestLimit_ = 20;

  /// A map from a networkName to the DAG that represents the root of a network.
  std::unordered_map<std::string, std::unique_ptr<DAGNode>> roots_;

  /// A map from a networkName to the DAG that represents the network.
  std::unordered_map<std::string, std::unique_ptr<DAGNode>> networks_;

  /// Mutex for networks_ and roots_ since runNetwork, addNetwork, and
  /// removeNetwork can all be called concurrently, a guard is needed.
  std::mutex networkLock_;

  /// A map of DeviceManagers by deviceID. An ordered map is used here to allow
  /// a stable iteration order over devices.
  std::map<DeviceIDTy, std::shared_ptr<DeviceManager>> devices_;

  /// Executor class, this handles dispatching execution requests to the
  /// appropriate device managers for an inference request.
  std::unique_ptr<Executor> executor_;

  /// Backend pointer. This allows the HostManager to optimize functions before
  /// they are passed to the Partitioner. It is just one since we are currently
  /// assuming a homogenous set of devices. This may get moved into the
  /// Partitioner at a later point.
  std::unique_ptr<Backend> backend_;

  /// The provisioner owns the compiledFunctions and handles loading functions
  /// onto the devices.
  std::unique_ptr<Provisioner> provisioner_;

public:
  /// Adds the network to the host and does the necessary setup work. This
  /// includes partitioning, provisioning, compiling and initializing
  /// backends. Additionally DAGs are created for each function and stored in
  /// networks_. Returns a result code to indicate success.
  ResultCode addNetwork(Module *M);

  /// Given \p networkName removes that network from the host. This also
  /// removes the network from any backends setup to execute it.
  void removeNetwork(llvm::StringRef networkName);

  /// Returns true if \p networkName is already added to the host.
  bool networkAdded(llvm::StringRef networkName);

  /// Removes all networks from the host, and stops execution on all devices.
  void clearHost();

  /// Runs the network specified by \p networkName using
  /// the provided \p context, returns a runIdentifier which refers to the
  /// specic inference request. Calls \p callback with the results when
  /// inference is done.
  /// Note: This method is intended to be thread-safe, it will be called
  /// concurrently from mutliple threads.
  /// Returns -1 if networkName not found or too many active requests.
  RunIdentifierTy runNetwork(llvm::StringRef networkName,
                             std::unique_ptr<Context> context,
                             ResultCBTy callback);
  HostManager(const std::vector<DeviceConfig> &configs);
  ~HostManager();
};
} // namespace runtime
} // namespace glow
#endif // GLOW_RUNTIME_HOSTMANAGERR_HOSTMANAGER_H
