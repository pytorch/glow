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
#include "glow/Backends/BackendUtils.h"
#include "glow/Backends/CompiledFunction.h"
#include "glow/Graph/Graph.h"

#include <future>
#include <map>
#include <queue>

using namespace glow;
using namespace runtime;

Provisioner::Provisioner(DeviceManagerMapTy &devices) {
  for (auto &device : devices) {
    devices_.push_back(device.second.get());
  }
  auto backendKind = devices[0]->getBackendKind();
  backend_.reset(createBackend(backendKind));
}

void walkNetwork(DAGNode *node,
                 std::map<DeviceIDTy, std::vector<DAGNode *>> &logicalDevices) {

  auto it = logicalDevices.find(node->logicalDevice);
  if (it != logicalDevices.end()) {
    it->second.push_back(node);
  } else {
    logicalDevices.emplace(node->logicalDevice, std::vector<DAGNode *>{node});
  }
  for (auto &child : node->children) {
    walkNetwork(child, logicalDevices);
  }
}

bool sortMostMemory(const std::pair<DeviceIDTy, uint64_t> &a,
                    const std::pair<DeviceIDTy, uint64_t> &b) {
  return (a.second > b.second);
}
ResultCode
Provisioner::provision(std::vector<std::unique_ptr<DAGNode>> &networks,
                       Module &module) {
  // Walk the networks and group by logicalDeviceId.
  std::map<DeviceIDTy, std::vector<DAGNode *>> logicalDevices;

  for (auto &network : networks) {
    for (auto &child : network->children) {
      walkNetwork(child, logicalDevices);
    }
  }
  // Check if there are more logical devices than physical devices.
  if (logicalDevices.size() > devices_.size()) {
    return ResultCode::Failed;
  }

  std::vector<std::pair<DeviceIDTy, uint64_t>> logicalDeviceSize;
  std::map<DeviceIDTy, FunctionMapTy> functionMaps;
  // Compile functions and calculate required memory for each logical device.
  for (auto &device : logicalDevices) {
    uint64_t totalMemory = 0;
    FunctionMapTy functionMap;
    for (auto &node : device.second) {
      Function *function = module.getFunction(node->name);
      CompilationOptions compileOptions;
      compileOptions.collectConstants = false;
      auto compiled = backend_->compile(function, compileOptions);
      node->runtimeBundle = compiled->getRuntimeBundle();
      node->runtimeBundle.setInputsandOutputs();
      functionMap.emplace(node->name, compiled.get());
      functions_.emplace(node->name, std::move(compiled));
      totalMemory += node->runtimeBundle.getConstantWeightSize();
    }
    logicalDeviceSize.push_back(std::make_pair(device.first, totalMemory));
    functionMaps.emplace(device.first, functionMap);
  }
  // Sort by total size in descending order.
  std::sort(logicalDeviceSize.begin(), logicalDeviceSize.end(), sortMostMemory);

  // Get available memory for all devices.
  std::vector<std::pair<DeviceIDTy, uint64_t>> deviceMemory;
  for (unsigned i = 0; i < devices_.size(); i++) {
    uint64_t availableMemory = devices_[i]->getAvailableMemory();
    deviceMemory.push_back(std::make_pair(i, availableMemory));
  }
  // Sort by available memory in descending order.
  std::sort(deviceMemory.begin(), deviceMemory.end(), sortMostMemory);

  // Try to add functions to devices in order from largest to smallest.
  for (unsigned i = 0; i < logicalDeviceSize.size(); i++) {
    if (logicalDeviceSize[i].second * NETWORK_PADDING_FACTOR >=
        deviceMemory[i].second) {
      return ResultCode::Failed;
    }
    // Load functions on device.
    DeviceIDTy logicalID = logicalDeviceSize[i].first;
    DeviceIDTy deviceID = deviceMemory[i].first;
    std::promise<bool> addNetwork;
    auto ready = addNetwork.get_future();
    devices_[deviceID]->addNetwork(
        &module, functionMaps[logicalID],
        [&addNetwork](const Module *, ResultCode result) {
          addNetwork.set_value(result == ResultCode::Ready);
        });
    auto result = ready.get();
    if (!result) {
      return ResultCode::Failed;
    }
    // Set deviceID for each node added
    for (auto &node : logicalDevices[logicalID]) {
      node->deviceID = deviceID;
    }
  }
  return ResultCode::Ready;
};
