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

ResultCode
Provisioner::provision(dependencyDAG &networks, executionDAG &runDAG,
                       std::unordered_map<DeviceID, DeviceManager> &devices) {
  // Check that there is available space for provisioning.
  // This will be the planning phase, for the first pass we will just assign in
  // order. Later we will want to check if networks are already loaded.
  std::unordered_map<DeviceID, Module *> deviceAssignment;
  // Assuming number of devices > number of modules.
  if (networks.modules.size() > devices.size()) {
    return FAILED;
  }
  // Walk devices and networks.modules and pair them as assignments.
  for (auto it_device = devices.begin(), it_module = networks.modules.begin();
       it_device != devices.end() || it_module != networks.modules.end();) {
    // Pair Module and Device
    deviceAssignment.emplace(it_device->first, std::move(*it_module));
    ++it_module;
    ++it_device;
  }
  // For each assignment:
  // Check that the device has space, if not fail.
  // Call addNetwork and pass in callback, on success add the deviceID to the
  // executionDAG If a module fails to provision return failure, otherwise wait
  // until all modules are added then return success.
  std::mutex provisioned;
  std::unordered_map<Module *, DeviceID> networkIDs;
  std::promise<bool> done;
  auto ready = done.get_future();
  for (auto assignment : deviceAssignment) {
    deviceID = assignment->first;
    modulePtr = assignment->second.get();
    device = devices[deviceID];
    device.addNetwork(std::move(assignment.second),
                      [&provisioned, &networkIDs, modulePtr, &done]() {
                        provisioned.lock();
                        provisioned.unlock();
                      })
  }
};