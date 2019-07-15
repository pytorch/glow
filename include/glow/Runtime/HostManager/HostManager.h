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
#include "glow/Backend/Backend.h"
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

class Executor;

class Provisioner;

/// The HostManager serves as an entry point into the Runtime environment. It
/// provides an interface to add, run, and evict networks from the host. It
/// handles DeviceManager initialization, houses the Executor, and calls into
/// the Partitioner and Provisioner for network initialization.
class HostManager final {
  /// NetworkData contains data about each network in HostManager that is needed
  /// by the runtime.
  struct NetworkData {
    DAG dag;
    // Module that was used to create this network. Everything except
    // placeholders and types have been removed from it.
    std::shared_ptr<Module> module;

    /// use an atomic refcount rather than just store a shared_ptr for thread
    /// safety.
    std::atomic<size_t> refcount;
  };

  /// Count of current in-flight networks being run. Atomic to allow
  /// concurrency in runNetwork.
  std::atomic<size_t> activeRequestCount_{0};

  /// Count of total requests, this is used as a run ID. Atomic to allow
  /// concurrency in runNetwork.
  std::atomic<size_t> totalRequestCount_{0};

  /// Configuration parameters for this Runtime Host.
  const HostConfig config_{};

  /// A map from a networkName to a network, which is represented by struct DAG.
  std::unordered_map<std::string, NetworkData> networks_;

  /// Mutex for networks_ since runNetwork, addNetwork, and
  /// removeNetwork can all be called concurrently, a guard is needed.
  std::mutex networkLock_;

  /// A map of DeviceManagers by deviceID. An ordered map is used here to allow
  /// a stable iteration order over devices.
  DeviceManagerMapTy devices_;

  /// Executor class, this handles dispatching execution requests to the
  /// appropriate device managers for an inference request.
  std::unique_ptr<Executor> executor_;

  /// The provisioner owns the compiledFunctions and handles loading functions
  /// onto the devices.
  std::unique_ptr<Provisioner> provisioner_;

public:
  /// Default constructor.
  HostManager() = default;

  /// Constructor that takes configuration options.
  HostManager(const HostConfig &hostConfig);

  /// Constructor that takes a list of Devices to use.
  HostManager(std::vector<std::unique_ptr<DeviceConfig>> deviceConfigs);

  /// Constructor that takes both Devices and the configuration.
  HostManager(std::vector<std::unique_ptr<DeviceConfig>> deviceConfigs,
              const HostConfig &hostConfig);

  /// Adds the network to the host and does the necessary setup work. This
  /// includes partitioning, provisioning, compiling and initializing
  /// backends. Additionally DAGs are created for each function and stored in
  /// networks_. \returns an llvm::Error containing the results of the
  /// operation. This function consumes the \p module so any pointers to data
  /// contained within the module should be considered invalid. The function is
  /// optimized based on \p cctx. If \p saturateHost is set to true the
  /// HostManager will try to use all available devices on the host.
  llvm::Error addNetwork(std::unique_ptr<Module> module,
                         CompilationContext &cctx, bool saturateHost = false);

  /// Given \p networkName removes that network from the host. This also
  /// removes the network from any backends setup to execute it.
  /// \returns an llvm::Error indicating success or failure of the operation.
  llvm::Error removeNetwork(llvm::StringRef networkName);

  /// Returns true if \p networkName is already added to the host.
  bool networkAdded(llvm::StringRef networkName);

  /// Removes all networks from the host, and stops execution on all devices.
  llvm::Error clearHost();

  /// Runs the network specified by \p networkName using
  /// the provided \p context, returns a runIdentifier which refers to the
  /// specic inference request. Calls \p callback with the results when
  /// inference is done.
  /// Note: This method is intended to be thread-safe, it will be called
  /// concurrently from multiple threads.
  /// Returns -1 if networkName not found or too many active requests.
  RunIdentifierTy runNetwork(llvm::StringRef networkName,
                             std::unique_ptr<ExecutionContext> context,
                             ResultCBTy callback);

  /// A wrapper around runNetwork that provides a blocking interface for an
  /// inference request. Runs the network provided in \p networkName using \p
  /// bindings for placeholder bindings. \returns an llvm::Error indicating
  /// success or failure.
  llvm::Error runNetworkBlocking(llvm::StringRef networkName,
                                 PlaceholderBindings &bindings);
  /// Initialize the HostManager with the given \p configs creating one
  /// DeviceManager for each config listed.
  llvm::Error init(std::vector<std::unique_ptr<DeviceConfig>> configs);

  /// Get the network DAG for \p network if it exists.
  llvm::Expected<DAG &> getNetworkDAG(llvm::StringRef network);

  ~HostManager();
};
} // namespace runtime
} // namespace glow
#endif // GLOW_RUNTIME_HOSTMANAGERR_HOSTMANAGER_H
