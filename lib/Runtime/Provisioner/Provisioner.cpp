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

bool sortMostMemory(const std::pair<DeviceIDTy, uint64_t> &a,
                    const std::pair<DeviceIDTy, uint64_t> &b) {
  return (a.second > b.second);
}

llvm::Error Provisioner::provision(DAGListTy &networks, Module &module) {
  // Walk the networks and group by logicalDeviceId.
  std::map<DeviceIDTy, std::vector<DAGNode *>> logicalDevices;

  for (auto &network : networks) {
    for (auto &node : network.nodes) {
      auto it = logicalDevices.find(node->logicalDevice);
      if (it != logicalDevices.end()) {
        it->second.push_back(node.get());
      } else {
        logicalDevices.emplace(node->logicalDevice,
                               std::vector<DAGNode *>{node.get()});
      }
    }
  }

  RETURN_ERR_IF_NOT(
      logicalDevices.size() <= devices_.size(),
      "Provisioner found more logical devices than physical devices.");

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
      node->runtimeBundle =
          llvm::make_unique<RuntimeBundle>(compiled->getRuntimeBundle());
      node->runtimeBundle->setInputsandOutputs();
      functionMap.emplace(node->name, compiled.get());
      functions_.emplace(node->name, std::move(compiled));
      totalMemory += node->runtimeBundle->getConstantWeightSize();
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
    RETURN_ERR_IF_NOT(logicalDeviceSize[i].second * NETWORK_PADDING_FACTOR <
                          deviceMemory[i].second,
                      "Not enough memory to provision functions onto devices");

    // Load functions on device.
    DeviceIDTy logicalID = logicalDeviceSize[i].first;
    DeviceIDTy deviceID = deviceMemory[i].first;
    llvm::Error addErr = llvm::Error::success();
    std::promise<void> addPromise;
    auto ready = addPromise.get_future();
    devices_[deviceID]->addNetwork(
        &module, functionMaps[logicalID],
        [&addErr, &addPromise](const Module *, llvm::Error err) {
          addErr = std::move(err);
          addPromise.set_value();
        });
    ready.wait();
    RETURN_IF_ERR(addErr);
    // Set deviceID for each node added
    for (auto &node : logicalDevices[logicalID]) {
      node->deviceID = deviceID;
    }
  }
  return llvm::Error::success();
};

void Provisioner::removeFunction(llvm::StringRef name) {
  functions_.erase(name);
}
