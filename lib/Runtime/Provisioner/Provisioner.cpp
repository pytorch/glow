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
#include "glow/Backend/BackendUtils.h"
#include "glow/Backend/CompiledFunction.h"
#include "glow/Graph/Graph.h"
#include "glow/Support/Debug.h"

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
  llvm::SmallSet<std::string, 10> used;
  for (auto &device : devices) {
    devices_.push_back(device.second.get());
    auto backendName = device.second->getBackendName();
    if (used.count(backendName) == 0) {
      backends_.emplace_back(createBackend(backendName));
      used.insert(backendName);
    }
  }
}

llvm::Error Provisioner::provision(DAGListTy &networks, Module &module,
                                   CompilationContext &cctx) {
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
  if (cctx.backendOpts.collectConstants) {
    // @todo(nickg) must use VLOG_EVERY_N temporarily here due to old glog
    // version.
    VLOG_EVERY_N(1, 1000)
        << "Warning: collectConstants is set in a Runtime compile, "
           "ignoring it.";
  }

  // Set collectConstants to false, this is because the DeviceManager will
  // handle moving constants to the device, this way we can eliminate one
  // copy operation.
  cctx.backendOpts.collectConstants = false;

  std::vector<std::pair<DeviceIDTy, uint64_t>> logicalDeviceSize;
  std::map<DeviceIDTy, std::string> logicalDeviceBackendName;
  std::map<DeviceIDTy, FunctionMapTy> functionMaps;
  // Compile functions and calculate required memory for each logical device.
  for (auto &device : logicalDevices) {
    uint64_t totalMemory = 0;
    auto nodeBackendName = (device.second[0])->backendName;
    FunctionMapTy functionMap;
    for (auto &node : device.second) {
      // Only compile if we haven't compiled before. If we have previously
      // compiled the function reuse it.
      auto it = functions_.find(node->name);
      if (it == functions_.end()) {
        Function *function = module.getFunction(node->name);
        for (size_t i = 0, e = backends_.size(); i < e; i++) {
          if (backends_[i]->getBackendName() == nodeBackendName) {
            auto compiledOrErr =
                backends_[i]->compile(function, cctx.backendOpts);
            if (!compiledOrErr) {
              return compiledOrErr.takeError();
            }
            auto compiled = std::move(*compiledOrErr);
            node->runtimeBundle =
                llvm::make_unique<RuntimeBundle>(compiled->getRuntimeBundle());
            functions_.emplace(node->name, std::move(compiled));
            break;
          }
        }
      }
      functionMap.emplace(node->name, functions_[node->name].get());
      totalMemory += node->runtimeBundle->getConstantWeightSize();
    }
    logicalDeviceSize.push_back(std::make_pair(device.first, totalMemory));
    logicalDeviceBackendName[device.first] = nodeBackendName;
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
  std::map<std::string, size_t> startPos;
  for (unsigned i = 0; i < logicalDeviceSize.size(); i++) {
    std::string backendName =
        logicalDeviceBackendName[logicalDeviceSize[i].first];
    // Find the start point of each backendName device.
    if (startPos.find(backendName) == startPos.end()) {
      startPos[backendName] = 0;
    }
    for (size_t j = startPos[backendName]; j < deviceMemory.size(); j++) {
      DeviceIDTy deviceID = deviceMemory[j].first;
      if (devices_[deviceID]->getBackendName() == backendName) {
        startPos[backendName] = j + 1;
        RETURN_ERR_IF_NOT(
            logicalDeviceSize[i].second < deviceMemory[j].second,
            llvm::formatv("Not enough memory to provision functions "
                          "onto devices. Need {0} bytes, have {1}.",
                          logicalDeviceSize[i].second, deviceMemory[j].second)
                .str());

        // Load functions on device.
        DeviceIDTy logicalID = logicalDeviceSize[i].first;
        std::promise<void> addPromise;
        auto ready = addPromise.get_future();
        std::unique_ptr<llvm::Error> addErr;
        devices_[deviceID]->addNetwork(
            &module, functionMaps[logicalID],
            [&addErr, &addPromise](const Module *, llvm::Error err) {
              addErr = llvm::make_unique<llvm::Error>(std::move(err));
              addPromise.set_value();
            });
        ready.wait();
        RETURN_IF_ERR(*DCHECK_NOTNULL(addErr.get()));
        // Set deviceID for each node added
        for (auto &node : logicalDevices[logicalID]) {
          node->deviceIDs.push_back(deviceID);
        }
        break;
      }
    }
  }
  return llvm::Error::success();
};

void Provisioner::removeFunction(llvm::StringRef name) {
  functions_.erase(name);
}
