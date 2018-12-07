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
#ifndef GLOW_RUNTIME_HOSTMANAGER_H
#define GLOW_RUNTIME_HOSTMANAGER_H
#include "glow/Graph/Graph.h"
#include "glow/Runtime/Executor.h"
#include "glow/Runtime/Partitioner.h"
#include "glow/Runtime/Provisioner.h"

#include <llvm/ADT/StringRef.h>

#include <atomic>
#include <unordered_map>
namespace glow {
class DeviceManager;
class Context;
enum ResultCode {
  READY,
  EXECUTED,
  FAILED,
  CANCELLED
}; // This will likely be defined in one common runtime place.
class HostManager final {
  /// Count of current networks being run.
  std::atomic<unsigned int> activeCount_;
  /// Count of networks initialized on this node.
  std::atomic<unsigned int> totalCount_;
  /// Limit maximum count of networks run at once. Hardcoded for now this should
  /// be a configurable value.
  const unsigned int activeLimit_ = 20;
  /// A map from a networkID to the DAG that represents the network.
  std::unordered_map<int, ExecutorFunctionDAG> networks_;
  /// A map of DeviceManagers by deviceID.
  std::unordered_map<int, DeviceManager> devices_;
  Executor executor_;
  Partitioner partitioner_;
  Provisioner provisioner_;

public:
  /// Adds the network to the host and does the necessary setup work. This
  /// includes partitioning, provisioning, compiling and initializing backends.
  /// Returns the networkID of the network.
  unsigned int addNetwork(Module *M);
  /// Given \p networkID removes that network from the host. This also removes
  /// the network from any backends setup to execute it.
  void removeNetwork(int networkID);
  /// Returns true if \p networkID is already added to the host.
  bool networkAdded(int networkID);
  /// Removes all networks from the host, and stops execution on all devices.
  void clearHost();
  /// Runs the network specified by \p networkID and \p functionName using the
  /// provided \p context returns true when results are copied into the context
  /// false if an error occurred or the count of current networks is above the
  /// threshold.
  ResultCode runNetwork(int networkID, llvm::StringRef functionName,
                        Context context);
  HostManager();
  ~HostManager();
};

} // namespace glow
#endif // GLOW_RUNTIME_HOSTMANAGER_H