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

namespace {
// STL sorting algorithm cannot inline predicate if it got provided as a regular
// function.
// Template instantiation expands std::sort with predicate type as
// (bool)(const std::pair<DeviceIDTy, uint64_t> &,
//        const std::pair<DeviceIDTy, uint64_t> &).
// It means any regular function with the above signature will match
// the template instantiation, and compiler cannot inline the code of
// one of the possible functions.
// Declaring lambda, which has a unique type regardless its signature,
// forces compiler to instantiate the template with a provided unique type and
// correspondently compiler can inline the lambda code.
auto sortMostMemory = [](const std::pair<DeviceIDTy, uint64_t> &a,
                         const std::pair<DeviceIDTy, uint64_t> &b) -> bool {
  return a.second > b.second;
};
} // namespace

Provisioner::Provisioner(DeviceManagerMapTy &devices) {
  for (auto &device : devices) {
    devices_.push_back(device.second.get());
  }
  auto backendKind = devices[0]->getBackendKind();
  backend_.reset(createBackend(backendKind));
}

llvm::Error Provisioner::provision(DAGListTy &networks, Module &module) {
  // Walk the networks and group by logicalDeviceId.
  std::map<DeviceIDTy, std::vector<DAGNode *>> logicalDevices;
  // For each network visit all the partitions (nodes) and add the node to each
  // logical device it is assigned to.
  for (auto &network : networks) {
    for (auto &node : network.nodes) {
      for (auto logical : node->logicalDevices) {
        auto it = logicalDevices.find(logical);
        if (it != logicalDevices.end()) {
          it->second.push_back(node.get());
        } else {
          logicalDevices.emplace(logical, std::vector<DAGNode *>{node.get()});
        }
      }
    }
  }

  std::vector<std::pair<DeviceIDTy, uint64_t>> logicalDeviceSize;
  std::map<DeviceIDTy, FunctionMapTy> functionMaps;
  // Compile functions and calculate required memory for each logical device.
  for (auto &device : logicalDevices) {
    uint64_t totalMemory = 0;
    FunctionMapTy functionMap;
    for (auto &node : device.second) {
      // Only compile if we haven't compiled before. If we have previously
      // compiled the function reuse it.
      auto it = functions_.find(node->name);
      if (it == functions_.end()) {
        Function *function = module.getFunction(node->name);
        BackendOptions opts;
        // Set collectConstants to false, this is because the DeviceManager will
        // handle moving constants to the device, this way we can eliminate one
        // copy operation.
        opts.collectConstants = false;
        auto compiled = backend_->compile(function, opts);
        node->runtimeBundle =
            llvm::make_unique<RuntimeBundle>(compiled->getRuntimeBundle());
        functions_.emplace(node->name, std::move(compiled));
      }
      functionMap.emplace(node->name, functions_[node->name].get());
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
    RETURN_ERR_IF_NOT(logicalDeviceSize[i].second < deviceMemory[i].second,
                      llvm::formatv("Not enough memory to provision functions "
                                    "onto devices. Need {0} bytes, have {1}.",
                                    logicalDeviceSize[i].second,
                                    deviceMemory[i].second)
                          .str());

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
      node->deviceIDs.push_back(deviceID);
    }
  }
  return llvm::Error::success();
};

void Provisioner::removeFunction(llvm::StringRef name) {
  functions_.erase(name);
}
