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

#include "glow/Runtime/HostManager/HostManager.h"
#include "glow/Backends/DeviceManager.h"
#include "glow/Graph/Context.h"

#include <future>

using namespace glow;

HostManager::HostManager() {
  provisioner_ = Provisioner();
  partitioner_ = Partitioner();
}

HostManager::~HostManager() { clearHost(); }
unsigned int HostManager::addNetwork(Module *M) {
  totalCount_++;
  int networkID = totalCount_;
  auto dependencyGraph = partitioner_.partition(M);
  auto executionDAG = provisioner_.provision(dependencyGraph);
  networks_.emplace(networkID, executionDAG);
  return networkID;
}

void HostManager::removeNetwork(int networkID) {
  auto it = networks_.find(networkID);
  if (it == networks_.end()) {
    return;
  }
  auto network = it->second;
  networks_.erase(it);
  // walk DAG and remove from deviceManagers
  // assumes functions is a list of moduleID's and everything else uses that id
  // as the key.
  for (auto function : network.functions) {
    for (auto device : function.devices) {
      device.evictNetwork(function.id);
    }
  }
}

bool HostManager::networkAdded(int networkID) {
  return networks_.find(networkID) != networks_.end();
}

void HostManager::clearHost() {
  for (auto it : networks_) {
    removeNetwork(it->first);
  }
  for (auto it : devices_) {
    it->second.stop();
  }
  activeCount_ = 0;
  totalCount_ = 0;
}

ResultCode HostManager::runNetwork(int networkID, llvm::StringRef functionName,
                                   Context context) {
  if (networks_.find(networkID) == networks_.end()) {
    return FAILED;
  }
  if (activeCount_ > activeLimit_) {
    return FAILED;
  }
  ++activeCount_;
  std::promise<ResultCode> promise;
  auto result = promise.get_future();
  executor_.runNetwork(networks_[networkID], functionName, context,
                       [&promise](ResultCode id) { promise.set_value(id); });
  result.wait();
  --activeCount_;
  return result.get();
}