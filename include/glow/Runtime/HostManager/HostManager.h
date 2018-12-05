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
#ifndef GLOW_RUNTIME_HOST_MANAGER_H
#define GLOW_RUNTIME_HOST_MANAGER_H
#include "glow/Graph/Graph.h"

#include <llvm/ADT/StringRef.h>

#include <unordered_map>
namespace glow {
class DependencyGraph;
class DeviceManager;
class Partitioner;
class Provisioner;
class Executor {}; // temporary the Executor header will need to be included
class Context;
class HostManager final {
  int activeCount_;
  int totalCount_;
  std::unordered_map<int, DependencyGraph> networks_;
  std::unordered_map<int, DeviceManager> devices_;
  Executor executor_;
  Partitioner partitioner_;
  Provisioner provisioner_;

public:
  /// Adds the network to the host and does the necessary setup work. This
  /// includes partitioning, provisioning, compiling and initializing backends.
  /// Returns the networkID of the network.
  int addNetwork(Function *F);
  /// Given \p networkID removes that network from the host. This also removes
  /// the network from any backends setup to execute it.
  void removeNetwork(int networkID);
  /// Returns true if \p networkID is already added to the host.
  bool networkAdded(int networkID);
  /// Removes all networks from the host, and stops execution on all devices.
  void clearHost();
  /// Runs the network specified by \p networkID and \p functionName using the
  /// provided \p context returns true when results are copied into the context
  /// false if an error occurred.
  bool runNetwork(int networkID, llvm::StringRef functionName, Context context);
  HostManager();
  ~HostManager();
};

} // namespace glow
#endif // GLOW_RUNTIME_HOST_MANAGER_H