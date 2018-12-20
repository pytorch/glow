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

#include "glow/Runtime/Provisioner/Provisioner.h"
#include "glow/Graph/Graph.h"

#include <future>
#include <mutex>

using namespace glow;
using namespace runtime;
using DeviceID = unsigned int;
Provisioner::Provisioner() = default;

void addNetworkCallback(std::promise<bool> &promise, NetworkIDty id,
                        ResultCode result, std::mutex &cbMutex,
                        std::unordered_map<Module *, DeviceID> &networkIDs,
                        Module *moduleID, unsigned int networkCount) {
  cbMutex.lock();
  if (result == READY) {
    networkIDs.emplace(moduleID, id);
    if (networkIDs.size() == networkCount) {
      promise.set_value(true);
    }
  } else {
    promise.set_value(false);
  }
  cbMutex.unlock();
}

ResultCode
Provisioner::provision(dependencyDAG &networks, executionDAG &runDAG,
                       std::unordered_map<DeviceID, DeviceManager> &devices) {
  // Check that there is available space for provisioning.
  // This will be the planning phase, for the first pass we will just assign in
  // order. Later we will want to check if networks are already loaded.
  std::unordered_map<Module *, DeviceID> deviceAssignment;
  // Assuming number of devices > number of modules.
  if (networks.modules.size() > devices.size()) {
    return FAILED;
  }
  // Walk devices and networks.modules and pair them as assignments.
  for (auto itDevice = devices.begin(), itModule = networks.modules.begin();
       itDevice != devices.end() || itModule != networks.modules.end();) {
    // Pair Module and Device
    deviceAssignment.emplace(std::move(*itModule), itDevice->first, );
    ++itModule;
    ++itDevice;
  }
  // For each assignment:
  // Check that the device has space, if not fail.
  // Call addNetwork and pass in callback, on success add the network ID to
  // networkIDs. If a module fails to provision return failure, otherwise wait
  // until all modules are added then return success.
  std::mutex addNetworkMutex;
  std::unordered_map<Module *, NetworkIDty> networkIDs;
  std::promise<bool> addNetwork;
  unsigned int networkCount = deviceAssignment.size();
  auto ready = addNetwork.get_future();
  for (auto assignment : deviceAssignment) {
    deviceID = assignment->second;
    auto modulePtr = assignment->first.get();
    device = devices[deviceID];
    auto networkID = device.getNextDeviceNetworkID();
    device.addNetwork(networkID, std::move(assignment.second),
                      [&addNetworkMutex, &networkIDs, modulePtr, &addNetwork,
                       networkCount](NetworkIDty id, ResultCode result) {
                        addNetworkCallback(addNetwork, id, result,
                                           addNetworkMutex, networkIDs,
                                           modulePtr, networkCount);
                      });
  }
  auto result = ready.get();
  if (!ready) {
    return FAILED;
  } else {
    // Fill in the executionDAG.
    for (auto network : networkIDs) {
      auto networkID = network->second;
      auto moduleID = network->first;
      runDAG.networks.push_back(networkID);
      runDAG.devices.emplace(networkID, deviceAssignment[moduleID]);
      std::vector<NetworkIDty> dependencies;
      std::vector<NetworkIDty> dependents;
    }
    for (auto root : networks.roots) {
      runDAG.roots.push_back(networkIDs[root]);
    }
  }
};