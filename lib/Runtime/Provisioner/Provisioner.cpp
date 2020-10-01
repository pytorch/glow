/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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
#include "glow/Runtime/DeferredWeightLoader.h"
#include "glow/Support/Debug.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"

#include <future>
#include <map>
#include <queue>

using namespace glow;
using namespace runtime;

namespace {
std::string getReplicatedName(std::string name, unsigned count) {
  return name + "_replicated" + std::to_string(count);
}
} // namespace

namespace glow {
extern bool GlowDumpCompilationLog;
extern bool GlowDumpBackendSpecificIRJSON;
} // namespace glow

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
  unsigned deviceMapping{0};
  for (auto &device : devices) {
    devices_.push_back(device.second.get());
    deviceMappings_.push_back(deviceMapping++);
    auto backendName = device.second->getBackendName();
    if (backends_.find(backendName) == backends_.end()) {
      std::unique_ptr<Backend> newBackend(createBackend(backendName));
      backends_.emplace(std::string(backendName), std::move(newBackend));
    }
  }
}

Error Provisioner::checkActiveNetworks(
    const DAGListTy &networks, std::vector<std::string> &localActiveNames) {

  std::lock_guard<std::mutex> networkLock(functionsLock_);
  for (auto &network : networks) {
    LOG(INFO) << "Checking for active networks when adding: "
              << network.root->name;

    for (auto &node : network.nodes) {
      //  Check to see if another thread is actively working on the same
      //  networks.
      if (activeFunctions_.find(node->name) != activeFunctions_.end()) {
        for (auto &name : localActiveNames) {
          activeFunctions_.erase(name);
        }
        return MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_NET_BUSY,
                        llvm::formatv("Cannot add the network {0}, as it is "
                                      "currently being provisioned.",
                                      node->name)
                            .str());
      }
      LOG(INFO) << "Adding partition name: " << node->name
                << " to activeFunctions_";
      localActiveNames.push_back(node->name);
      activeFunctions_.insert(node->name);
    }
  }
  return Error::success();
}

std::map<DeviceIDTy, std::vector<DAGNode *>>
Provisioner::generateLogicalDevices(const DAGListTy &networks) {
  // For each network visit all the partitions (nodes) and add the node to each
  // logical device it is assigned to.
  std::map<DeviceIDTy, std::vector<DAGNode *>> logicalDevices;
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
  return logicalDevices;
}

/// Helper method to calculate the size of each logical device, returns a
/// vector of deviceID size pairs sorted in descending order by size.
static std::vector<std::pair<DeviceIDTy, uint64_t>> calculateLogicalDeviceSize(
    const std::map<DeviceIDTy, std::vector<DAGNode *>> &devices) {
  std::vector<std::pair<DeviceIDTy, uint64_t>> logicalDeviceSize;
  for (auto &device : devices) {
    uint64_t sum{0};
    for (auto &node : device.second) {
      sum += node->size;
    }
    logicalDeviceSize.push_back(std::make_pair(device.first, sum));
  }
  // Sort by total size in descending order.
  std::sort(logicalDeviceSize.begin(), logicalDeviceSize.end(), sortMostMemory);
  return logicalDeviceSize;
}

Expected<std::map<DeviceIDTy, DeviceIDTy>>
Provisioner::generateDeviceAssignments(
    const std::vector<std::pair<DeviceIDTy, uint64_t>> &logicalDeviceSize,
    std::map<std::string, std::vector<std::pair<DeviceIDTy, uint64_t>>>
        &deviceMemoryMap,
    std::map<DeviceIDTy, std::vector<DAGNode *>> &logicalDevices) {
  // Generate assignments, logical DeviceID to physical DeviceID.
  std::map<DeviceIDTy, DeviceIDTy> deviceAssignment;
  // Setup iterators for each backend type, intialize them to 0.
  std::map<std::string, unsigned> positions;
  for (auto &device : deviceMemoryMap) {
    positions[device.first] = 0;
  }
  // Walk through the logical devices and assign them a physical device.
  // This approach will try to evenly spread networks across devices, we first
  // sort all devices by available space and then assign in descending order.
  // Once we reach the end we resort and start over. This goes until we are
  // unable to load a network at which point we sort one more time if the first
  // device has enough space we continue, otherwise we return an error.
  // This approach is to prevent many small networks from clumping on a single
  // device.
  for (auto logicalDevice : logicalDeviceSize) {
    // First check that there the requested backend kind is available.
    auto backendName = logicalDevices[logicalDevice.first][0]->backendName;
    if (deviceMemoryMap.find(backendName) == deviceMemoryMap.end()) {
      // Backend is unavailable return an error.
      return MAKE_ERR(
          ErrorValue::ErrorCode::RUNTIME_DEVICE_NOT_FOUND,
          llvm::formatv("Cannot add the network {0}, as the requested "
                        "backend: {1} is unavailable.",
                        logicalDevices[logicalDevice.first][0]->name,
                        backendName)
              .str());
    }

    auto currentPosition = positions[backendName];
    if (deviceMemoryMap[backendName][currentPosition].second >=
        logicalDevice.second) {
      // There is enough space, assign the logical device to this physical
      // device, increment the iterator and update the available memory.
      deviceAssignment.emplace(
          logicalDevice.first,
          deviceMemoryMap[backendName][currentPosition].first);
      deviceMemoryMap[backendName][currentPosition].second -=
          logicalDevice.second;

      // Check if we are at the end of the vector of devices.
      if (currentPosition == deviceMemoryMap[backendName].size() - 1) {
        // We are at the end of the vector of devices, re-sort and reset
        // position to 0.
        std::sort(deviceMemoryMap[backendName].begin(),
                  deviceMemoryMap[backendName].end(), sortMostMemory);
        positions[backendName] = 0;
      } else {
        // Increment current position by one.
        positions[backendName] = currentPosition + 1;
      }
    } else {
      // Before we assume failure we should re-sort the list to see if the
      // current largest amount of available space is enough to fit.
      std::sort(deviceMemoryMap[backendName].begin(),
                deviceMemoryMap[backendName].end(), sortMostMemory);
      if (deviceMemoryMap[backendName][0].second >= logicalDevice.second) {
        // There's a device that still has room, assign the network here.
        deviceAssignment.emplace(logicalDevice.first,
                                 deviceMemoryMap[backendName][0].first);
        deviceMemoryMap[backendName][0].second -= logicalDevice.second;

        // Since after sorting we were abel to add to device 0  set the current
        // position 1 we modulo with the number of devices in case there is only
        // 1 device.
        currentPosition = 1 % deviceMemoryMap[backendName].size();
        positions[backendName] = currentPosition;
      } else {
        // Return an error there is insufficient space for the logical device on
        // any available device.
        return MAKE_ERR(
            ErrorValue::ErrorCode::RUNTIME_OUT_OF_DEVICE_MEMORY,
            "Logical Device is too large to fit in available device memory.");
      }
    }
  }

  // Update nodes in logicalDevices with their assignments.
  for (auto &assignment : deviceAssignment) {
    for (auto &node : logicalDevices[assignment.first]) {
      node->deviceRuntimeInfos[deviceMappings_[assignment.second]] =
          DeviceRuntimeInfo();
    }
  }
  return deviceAssignment;
}

/// Updates \p allNodeInfo for all Nodes in \p newF given Nodes in \p oldF using
/// mapping \p oldToNewMap. Copies the final new info added from \p allNodeInfo
/// into \p origAllNodeInfo.
static Error propagateBackendSpecificNodeInfo(
    Function *oldF, Function *newF,
    const llvm::DenseMap<const Node *, Node *> &oldToNewMap,
    BackendSpecificNodeInfo &allNodeInfo,
    BackendSpecificNodeInfo &origAllNodeInfo) {
  // If there's no old node info for oldF then just return early.
  if (!allNodeInfo.count(oldF)) {
    return Error::success();
  }

  const auto &oldFunInfo = allNodeInfo[oldF];
  auto &newFunInfo = allNodeInfo[newF];

  for (const auto &oldNodeInfo : oldFunInfo) {
    const Node *oldN = oldNodeInfo.first;
    const auto oldToNewIt = oldToNewMap.find(oldN);
    RETURN_ERR_IF_NOT(oldToNewIt != oldToNewMap.end(), "No old node in map");
    const Node *newN = oldToNewIt->second;
    newFunInfo[newN] = oldNodeInfo.second;
  }

  // Copy into the original info for this Function, since allNodeInfo is from a
  // local copy of backendOpts we made in the caller to provision this Function.
  origAllNodeInfo[newF] = newFunInfo;

  return Error::success();
}

Error Provisioner::provision(DAGListTy &networks, Module &module,
                             CompilationContext &cctx) {

  // Check that the requested networks don't collide with the names of any other
  // networks being added.
  std::vector<std::string> localActiveNames;
  RETURN_IF_ERR(checkActiveNetworks(networks, localActiveNames));

  // Walk the networks and group by logicalDeviceId.
  auto logicalDevices = generateLogicalDevices(networks);

  if (cctx.backendOpts.collectConstants) {
    VLOG(1) << "Warning: collectConstants is set in a Runtime compile, "
               "ignoring it.";
  }
  if (cctx.backendOpts.backendHints.SRAMPrioritization.size() != 0 ||
      cctx.backendOpts.backendHints.executionUnits) {
    VLOG(1) << "Warning: backendHints is set in a Runtime compile, "
               "ignoring it.";
  }

  // Set collectConstants to false, this is because the DeviceManager will
  // handle moving constants to the device, this way we can eliminate one
  // copy operation.
  cctx.backendOpts.collectConstants = false;

  // Calculate the size of each logical device.
  auto logicalDeviceSize = calculateLogicalDeviceSize(logicalDevices);

  // Get available memory for all devices.
  std::vector<std::pair<DeviceIDTy, uint64_t>> deviceMemory;
  for (unsigned i = 0; i < devices_.size(); i++) {
    uint64_t availableMemory = devices_[i]->getAvailableMemory();
    deviceMemory.push_back(std::make_pair(i, availableMemory));
  }

  // Get available device memory, create a map of vectors for each backend kind
  std::map<std::string, std::vector<std::pair<DeviceIDTy, uint64_t>>>
      deviceMemoryMap;
  for (unsigned i = 0; i < devices_.size(); i++) {
    uint64_t availableMemory = devices_[i]->getAvailableMemory();

    deviceMemoryMap[devices_[i]->getBackendName()].push_back(
        std::make_pair(i, availableMemory));
  }

  // Sort all vectors in descending order of available memory.
  for (auto &sizes : deviceMemoryMap) {
    std::sort(sizes.second.begin(), sizes.second.end(), sortMostMemory);
  }

  // Generate assignments between physical and logical devices.
  auto deviceAssignments = generateDeviceAssignments(
      logicalDeviceSize, deviceMemoryMap, logicalDevices);

  // Check for errors.
  if (!deviceAssignments) {
    // If and error occured, clean up provisioning state and return
    // the error.
    cleanupProvision(localActiveNames, {});
    return deviceAssignments.takeError();
  }
  auto assignments = std::move(*deviceAssignments);

  // Container for duplicated functions and map tracking remaining installs for
  // a duplicated function.
  std::map<std::string, std::unique_ptr<CompiledFunction>> duplicatedFunctions;
  std::map<DAGNode *, unsigned> remainingDuplications;

  // Map from Placeholder* to DeviceManager, this is used for deferred weight
  // loading.
  std::unordered_map<Placeholder *, std::vector<unsigned>>
      placeholderToDeviceManager;
  if (cctx.deferredWeightLoader) {
    // Populate placeholdeToDeviceManager map.
    for (auto &assignment : assignments) {
      for (const auto &node : logicalDevices[assignment.first]) {
        Function *function = module.getFunction(node->name);
        for (auto PH : function->findPlaceholders()) {
          if (PH->isStatic()) {
            placeholderToDeviceManager[PH].push_back(assignment.second);
          }
        }
      }
    }
  } else {
    // Make sure there are no static placeholders.
    for (auto PH : module.getPlaceholders()) {
      if (PH->isStatic()) {
        return MAKE_ERR(
            ErrorValue::ErrorCode::RUNTIME_ERROR,
            llvm::formatv("Error Placholder: {0} is marked as static but no "
                          "deferredWeightLoader is provided.",
                          PH->getName())
                .str());
        ;
      }
    }
  }

  // Compile and load.
  // This is done one logical device at a time. All functions in a logical
  // device are compiled and then added to their assigned device. If a function
  // is in multiple logical devices it is stored so that it only needs to be
  // compiled once.
  std::map<DeviceIDTy, std::vector<std::string>> addedNetworks;
  for (auto &assignment : assignments) {
    auto logicalDevice = assignment.first;
    auto physicalDevice = assignment.second;
    auto deviceBackendName = logicalDevices[logicalDevice][0]->backendName;
    FunctionMapTy functionMap;
    // Container for the compiledFunctions for this logicalDevice.
    std::map<std::string, std::unique_ptr<CompiledFunction>> compiledFunctions;

    for (auto &node : logicalDevices[logicalDevice]) {
      // Check if this is a duplicated function that has already been compiled.
      if (duplicatedFunctions.find(node->name) != duplicatedFunctions.end()) {
        functionMap.emplace(node->name, duplicatedFunctions[node->name].get());
        // Add replications.
        for (unsigned i = 1; i < node->replicationCount; i++) {
          auto replicatedName = getReplicatedName(node->name, i);
          functionMap.emplace(replicatedName,
                              duplicatedFunctions[replicatedName].get());
        }

        remainingDuplications[node] -= 1;
      } else {
        // Compile and add to function map.
        auto options = cctx.backendOpts;
        options.backendHints = node->backendHints;
        // Insert all options loaded in the Partitioner alongside options
        // previously inserted, with Partitioner options taking precedence in
        // case of a collision of keys.
        for (auto &it : node->backendSpecificOpts) {
          options.backendSpecificOpts[it.first] = it.second;
        }
        Function *function = module.getFunction(node->name);
        if (backends_.find(deviceBackendName) == backends_.end()) {
          // Return error requested device type not found.
          return MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_DEVICE_NOT_FOUND,
                          "Unable to find device of type: " +
                              deviceBackendName);
        }

        std::unordered_map<std::string, std::unique_ptr<glow::CompiledFunction>>
            compiledReplications;
        // Before we compile clone the function so we can replicate it on the
        // device.
        for (unsigned i = 1; i < node->replicationCount; i++) {
          std::string replicatedName =
              getReplicatedName(function->getName(), i);
          llvm::DenseMap<const Node *, Node *> oldToNewMap;
          auto *clonedFunction = function->clone(replicatedName, &oldToNewMap);
          RETURN_IF_ERR(propagateBackendSpecificNodeInfo(
              function, clonedFunction, oldToNewMap,
              options.backendSpecificNodeInfo,
              cctx.backendOpts.backendSpecificNodeInfo));
          auto compiledOrErr2 =
              backends_[deviceBackendName]->compile(clonedFunction, options);
          if (!compiledOrErr2) {
            cleanupProvision(localActiveNames, {});
            return compiledOrErr2.takeError();
          }
          auto compiled2 = std::move(*compiledOrErr2);
          functionMap.emplace(replicatedName, compiled2.get());
          compiledReplications.emplace(replicatedName, std::move(compiled2));
        }

        auto compiledOrErr =
            backends_[deviceBackendName]->compile(function, options);

        // Note: This needs to come after compile above because compile may
        // modify the Function as well.
        if (cctx.dumpFinalGraph) {
          auto fname = strFormat(
              "%sfinal_graph_%s_%s.dot", cctx.dumpGraphPath.c_str(),
              deviceBackendName.c_str(), function->getName().str().c_str());
          LOG(INFO) << "Dumping final graph to " << fname;
          function->dumpDAG(fname);
        }

        if (GlowDumpCompilationLog) {
          llvm::SmallString<64> path;
          std::string prefix =
              llvm::formatv("{0}-{1}", cctx.compilationLogPrefix,
                            function->getName())
                  .str();
          auto tempFileRes =
              llvm::sys::fs::createTemporaryFile(prefix, "log", path);
          if (tempFileRes.value() != 0) {
            LOG(ERROR)
                << "Failed to create temp file for Glow compilation log: "
                << tempFileRes;
          }

          function->getLogContext()->dumpLog(path);
        }

        // Check to see if an error was encountered while compiling.
        if (!compiledOrErr) {
          // If and error occured, clean up provisioning state and return
          // the error.
          cleanupProvision(localActiveNames, {});
          return compiledOrErr.takeError();
        }
        auto compiled = std::move(*compiledOrErr);

        // Dump backend-specific IR
        if (GlowDumpBackendSpecificIRJSON) {
          compiled->dumpJSON(strFormat("%sbackend_specific_ir_%s.json",
                                       cctx.dumpGraphPath.c_str(),
                                       function->getName().str().c_str()));
        }

        node->runtimeBundle =
            glow::make_unique<RuntimeBundle>(compiled->getRuntimeBundle());

        functionMap.emplace(node->name, compiled.get());
        // If this function is in more than one logical device store it for
        // reuse.
        if (node->logicalDevices.size() > 1) {
          duplicatedFunctions.emplace(node->name, std::move(compiled));
          for (unsigned i = 1; i < node->replicationCount; i++) {
            std::string replicatedName =
                getReplicatedName(function->getName(), i);
            auto compiled2 = std::move(compiledReplications[replicatedName]);
            duplicatedFunctions.emplace(replicatedName, std::move(compiled2));
          }

          remainingDuplications[node] = node->logicalDevices.size() - 1;
        } else {
          compiledFunctions.emplace(node->name, std::move(compiled));
          for (unsigned i = 1; i < node->replicationCount; i++) {
            std::string replicatedName =
                getReplicatedName(function->getName(), i);
            auto compiled2 = std::move(compiledReplications[replicatedName]);
            compiledFunctions.emplace(replicatedName, std::move(compiled2));
          }
        }
      }
    }
    // Now that the functions are compiled add them to their assigned device
    // then cleanup.
    std::promise<void> addPromise;
    auto ready = addPromise.get_future();
    std::unique_ptr<Error> addErr;
    devices_[physicalDevice]->addNetwork(
        &module, functionMap,
        [&addErr, &addPromise](const Module *, Error err) {
          addErr = glow::make_unique<Error>(std::move(err));
          addPromise.set_value();
        });
    ready.wait();
    DCHECK_NOTNULL(addErr.get());
    if (*addErr.get()) {
      cleanupProvision(localActiveNames, addedNetworks);
      return std::move(*addErr.get());
    }
    // Add networks successfully loaded on device to addedNetworks, this way if
    // we fail later we can evict them.
    for (auto &node : logicalDevices[logicalDevice]) {
      addedNetworks[physicalDevice].push_back(node->name);
    }

    // Free up memory no longer needed by the compiledFunction.
    for (auto &func : compiledFunctions) {
      func.second->freeCompilationResources();
    }
    {
      // Move compiled functions from compiledFunctions to functions_.
      std::lock_guard<std::mutex> functionsLock(functionsLock_);
      for (auto &func : compiledFunctions) {
        functions_.emplace(func.first, std::move(func.second));
      }
      // Check if any of the duplicated functions can also be moved.
      for (auto iter = remainingDuplications.begin();
           iter != remainingDuplications.end();) {
        const auto &func = *iter;
        if (func.second == 0) {

          for (unsigned i = 1; i < func.first->replicationCount; i++) {
            std::string replicatedName = getReplicatedName(func.first->name, i);
            duplicatedFunctions[replicatedName]->freeCompilationResources();
            functions_.emplace(replicatedName,
                               std::move(duplicatedFunctions[replicatedName]));
            duplicatedFunctions.erase(replicatedName);
          }

          duplicatedFunctions[func.first->name]->freeCompilationResources();
          functions_.emplace(func.first->name,
                             std::move(duplicatedFunctions[func.first->name]));
          duplicatedFunctions.erase(func.first->name);
          iter = remainingDuplications.erase(iter);
        } else {
          ++iter;
        }
      }
    }
  }

  // If a deferredWeightLoader is provided, create a deferredWeightLoader and
  // load deferred weights.
  if (cctx.deferredWeightLoader) {
    const size_t totalNumDeferredWeights = placeholderToDeviceManager.size();
    LOG(INFO) << "Loading " << totalNumDeferredWeights << " deferred weights";

    auto startTime = std::chrono::steady_clock::now();
    auto loader = cctx.deferredWeightLoader;
    // Load the first weight.
    auto err = loader->loadNextWeight();
    if (err) {
      cleanupProvision(localActiveNames, addedNetworks);
      return err;
    }
    std::string weightName = loader->getName();
    // Load weights while there are weights to be loaded.
    unsigned int weightCount = 0;
    while (weightName != "") {
      LOG(INFO) << "Loading deferred weight (" << ++weightCount << " / "
                << totalNumDeferredWeights << "): " << weightName;
      const auto PH = module.getPlaceholderByNameSlow(weightName);
      if (!PH) {
        cleanupProvision(localActiveNames, addedNetworks);
        return MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_ERROR,
                        llvm::formatv("Error loading deferred weight. Name: "
                                      "{0} not found in module.",
                                      weightName)
                            .str());
      }
      // Convert the weight if needed.
      auto newTy = PH->getType();
      auto weight = loader->getTensor();
      auto oldKind = weight->getElementType();
      // Ensure we are working with a static PH.
      assert(PH->isStatic());
      if (!weight->getType().isEqual(newTy)) {
        ElemKind newK = newTy->getElementType();

        if (!isQuantizedElemKind(oldKind) && isQuantizedElemKind(newK)) {
          Tensor QT = quantization::quantizeTensor(
              *weight, {newTy->getScale(), newTy->getOffset()}, newK);
          weight->assign(&QT);
        } else {
          weight->convertToType(newK);
        }
      }
      // Transfer weight to all devices needed.
      std::list<Error> errors;
      std::list<std::future<void>> futures;
      for (const auto &device : placeholderToDeviceManager[PH]) {
        std::promise<void> transferPromise;
        errors.emplace_back(Error::empty());
        futures.emplace_back(transferPromise.get_future());
        devices_[device]->transferStaticPlaceholderToDevice(
            PH, weight,
            [&transferPromise, &error = errors.back()](Error err) mutable {
              error = std::move(err);
              transferPromise.set_value();
            });
      }

      for (auto &done : futures) {
        done.get();
      }

      for (auto &error : errors) {
        RETURN_IF_ERR(error);
      }

      err = loader->loadNextWeight();
      if (err) {
        cleanupProvision(localActiveNames, addedNetworks);
        return err;
      }
      weightName = loader->getName();
      // Remove PH from map, this way we can know that we've added all static
      // PH's
      placeholderToDeviceManager.erase(PH);
    }
    if (placeholderToDeviceManager.size()) {
      return MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_ERROR,
                      "Error not all static placeholders were initialized.");
    }

    std::chrono::duration<double> duration =
        std::chrono::steady_clock::now() - startTime;
    LOG(INFO) << "Done loading deferred weights in " << duration.count()
              << " seconds";
  }
  // Init alternate name states.
  for (auto &network : networks) {
    for (auto &node : network.nodes) {
      node->initAlternateState();
    }
  }
  cleanupProvision(localActiveNames, {}, false);
  return Error::success();
};

Backend &Provisioner::getBackend(llvm::StringRef backendName) const {
  assert(backends_.count(backendName) &&
         "No backend created by specified name.");
  return *backends_.at(backendName);
}

Expected<Backend *> Provisioner::getBackend() const {
  RETURN_ERR_IF_NOT(
      backends_.size() == 1,
      strFormat("Expected exactly 1 backend to be found but instead found %zu",
                backends_.size()));
  return backends_.begin()->second.get();
}

Error Provisioner::removeFunction(llvm::StringRef name) {
  std::lock_guard<std::mutex> functionsLock(functionsLock_);
  auto it = activeFunctions_.find(name);
  if (it != activeFunctions_.end()) {
    return MAKE_ERR(
        ErrorValue::ErrorCode::RUNTIME_NET_BUSY,
        llvm::formatv("Could not remove network: {0} as it is currently "
                      "being provisioned.",
                      name)
            .str());
  }
  functions_.erase(name);
  return Error::success();
}

Error Provisioner::evictFunction(llvm::StringRef name, DeviceManager *device) {
  std::promise<void> evictPromise;
  Error evictErr = Error::empty();
  auto done = evictPromise.get_future();
  device->evictNetwork(name,
                       [&evictPromise, &evictErr](std::string, Error err) {
                         evictErr = std::move(err);
                         evictPromise.set_value();
                       });
  done.get();
  return evictErr;
}

void Provisioner::cleanupProvision(
    llvm::ArrayRef<std::string> names,
    std::map<DeviceIDTy, std::vector<std::string>> const
        &currentNetworkResidency,
    bool failure) {
  std::lock_guard<std::mutex> functionLock(functionsLock_);
  for (auto &name : names) {
    LOG(INFO) << "Removing partition name: " << name
              << " from activeFunctions_\n";
    activeFunctions_.erase(name);
    if (failure) {
      // Remove any functions added before the failure.
      functions_.erase(name);
    }
  }
  if (failure) {
    // Remove any partitions added to devices.
    for (auto &device : currentNetworkResidency) {
      for (auto &network : device.second) {
        Error evictErr = evictFunction(network, devices_[device.first]);
        if (evictErr) {
          LOG(ERROR) << "Unable to evict network: " << network << "\n";
        }
      }
    }
  }
}
