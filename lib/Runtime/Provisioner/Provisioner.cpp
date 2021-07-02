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
#include "folly/String.h"
#include "glow/Backend/BackendUtils.h"
#include "glow/Backend/CompiledFunction.h"
#include "glow/Flags/Flags.h"
#include "glow/Graph/Graph.h"
#include "glow/Runtime/DeferredWeightLoader.h"
#include "glow/Support/Debug.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"

#include <folly/dynamic.h>
#include <future>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <vector>

using namespace glow;
using namespace runtime;

namespace {
std::string getReplicatedName(std::string name, unsigned count) {
  return name + "_replicated" + std::to_string(count);
}
} // namespace

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
    auto backendName = device.second->getBackendName().str();
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
#if FACEBOOK_INTERNAL
    LOG(INFO) << "Checking for active networks when adding: "
              << network.root->name;
#endif
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
#if FACEBOOK_INTERNAL
      LOG(INFO) << "Adding partition name: " << node->name
                << " to activeFunctions_";
#endif
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
    for (const auto *node : device.second) {
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
            strFormat(
                "Logical Device is too large to fit in available device "
                "memory. Largest device memory: %lu, logic device size: %lu",
                deviceMemoryMap[backendName][0].second, logicalDevice.second));
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

Error Provisioner::provisionNetwork(std::unique_ptr<Network> network) {
  VLOG(1) << "Started provisioner";
  DAGListTy &networks = network->networks;
  Module &module = network->module;
  CompilationContext &cctx = network->cctx;
  // Check that the requested networks don't collide with the names of any other
  // networks being added.
  std::vector<std::string> localActiveNames;
  RETURN_IF_ERR(checkActiveNetworks(networks, localActiveNames));

  // Mapping from function name to its compiled function. NB: compiledFunctions
  // will hold compiled function which might be used in clean up process by
  // cleanupGuard, hence this needs to be declared before cleanupGuard. We
  // probably should clean up the compiledFunctions logic to make this more
  // intuitive.
  llvm::StringMap<std::unique_ptr<CompiledFunction>> compiledFunctions;

  // If any error happens during the provison process, we will clean up the
  // compiled networks.
  std::map<DeviceIDTy, std::vector<std::string>> addedNetworks;
  ScopeGuard cleanupGuard([&localActiveNames, &addedNetworks, this]() {
    cleanupProvision(localActiveNames, addedNetworks);
  });

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

    deviceMemoryMap[devices_[i]->getBackendName().str()].push_back(
        std::make_pair(i, availableMemory));
  }

  // Sort all vectors in descending order of available memory.
  for (auto &sizes : deviceMemoryMap) {
    std::sort(sizes.second.begin(), sizes.second.end(), sortMostMemory);
  }

  // Generate assignments between physical and logical devices.
  auto deviceAssignments = generateDeviceAssignments(
      logicalDeviceSize, deviceMemoryMap, logicalDevices);

  VLOG(1) << "Before device assignment";
  // Check for errors.
  if (!deviceAssignments) {
    RETURN_ERR(deviceAssignments.takeError());
  }
  auto assignments = std::move(*deviceAssignments);

  VLOG(1) << "Before compile";

  // Stores function name and the remaining logical device count for that
  // function.
  llvm::StringMap<size_t> remainingDeviceCount;
  // Mapping from function name to its backend options.
  llvm::StringMap<BackendOptions> optsMap;

  // Compile and load.
  // This is done one logical device at a time. All functions in a logical
  // device are compiled and then added to their assigned device. If a function
  // is in multiple logical devices it is stored so that it only needs to be
  // compiled once.
  if (network->networkType == NetworkType::GLOW_NETWORK) {
    for (auto &assignment : assignments) {
      auto logicalDevice = assignment.first;
      auto physicalDevice = assignment.second;
      auto deviceBackendName = logicalDevices[logicalDevice][0]->backendName;

      if (backends_.find(deviceBackendName) == backends_.end()) {
        // Return error requested device type not found.
        return MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_DEVICE_NOT_FOUND,
                        "Unable to find device of type: " + deviceBackendName);
      }

      // Stores all the functions in a logical device.
      std::vector<glow::Function *> functionsToCompile;
      // Stores the compiled functions that will be added to physical device.
      FunctionMapTy functionMap;

      // Collect all the functions in a logical device.
      for (auto &node : logicalDevices[logicalDevice]) {
        // If the function name exist we don't need to compile it again.
        if (optsMap.count(node->name)) {
          remainingDeviceCount[node->name] -= 1;
          continue;
        }

        auto options = cctx.backendOpts;
        options.backendHints = node->backendHints;
        // Insert all options loaded in the Partitioner alongside options
        // previously inserted, with Partitioner options taking precedence in
        // case of a collision of keys.
        for (auto &it : node->backendSpecificOpts) {
          options.backendSpecificOpts[it.first] = it.second;
        }
        std::lock_guard<std::mutex> functionsLock(functionsLock_);
        Function *function = module.getFunction(node->name);

        functionsToCompile.push_back(function);
        optsMap.insert({function->getName(), options});
        functionReplicaCount_.emplace(node->name, node->replicationCount);
        remainingDeviceCount.insert(
            {node->name, node->logicalDevices.size() - 1});
      }

      // Compile all the functions in the logical device together.
      // We add a lock here because some backends are not threadsafe (CPU
      // backend).
      std::unique_lock<std::mutex> compileLock(functionsLock_);
      auto compiledOrErr = backends_[deviceBackendName]->compileFunctions(
          functionsToCompile, optsMap);
      VLOG(1) << "After compile";
      compileLock.unlock();

      // Dump graph and logs
      for (auto *function : functionsToCompile) {
        // Note: This needs to come after compile above because compile may
        // modify the Function as well.
        if (cctx.dumpFinalGraph) {
          auto fname = strFormat(
              "%sfinal_graph_%s_%s.dot", cctx.dumpGraphPath.c_str(),
              deviceBackendName.c_str(), function->getName().str().c_str());
          LOG(INFO) << "Dumping final graph to " << fname;
          function->dumpDAG(fname);
          // print stats of node
          std::map<std::string, int> opCounter;
          for (const auto &node : function->getNodes()) {
            opCounter[node.getKindName()]++;
          }
          std::ostringstream ss;
          ss << "Dump of Node stats for Function:\n";
          ss << folly::stringPrintf("%30s %13s \n", "NodeKind", "Count");
          for (const auto &p : opCounter) {
            ss << folly::stringPrintf("%30s %13d \n", p.first.c_str(),
                                      p.second);
          }
          LOG(INFO) << ss.str();
        }

        if (glow::flags::DumpCompilationLog) {
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
      }

      // If err return it, else store compiled functions into compiledFunctions.
      if (!compiledOrErr) {
        RETURN_ERR(compiledOrErr.takeError());
      }
      auto compiled = std::move(*compiledOrErr);
      for (auto &compiledFunction : compiled) {

        // Deserialize compiled function from cctx.nameToFunctions
        if (cctx.backendOpts.useDeserialize) {
          std::string name = compiledFunction.first().str();
          if (cctx.nameToFunctions.find(name) == cctx.nameToFunctions.end()) {
            return MAKE_ERR(
                ErrorValue::ErrorCode::UNKNOWN,
                "Cannot find compiled function when deserializing " + name);
          }
          RETURN_IF_ERR(compiledFunction.second->deserialize(
              *(cctx.nameToFunctions.find(name)->second)));
        }
        compiledFunctions.try_emplace(compiledFunction.first(),
                                      std::move(compiledFunction.second));
      }
      // Construnct functionMap for physical device.
      for (auto &node : logicalDevices[logicalDevice]) {
        RETURN_ERR_IF_NOT(compiledFunctions.count(node->name),
                          "Can't find corresponding compiled function " +
                              node->name);

        auto *compiledFunction = compiledFunctions[node->name].get();
        functionMap.emplace(node->name, compiledFunction);

        for (unsigned i = 1; i < node->replicationCount; i++) {
          auto replicatedName = getReplicatedName(node->name, i);
          functionMap.emplace(replicatedName, compiledFunction);
        }

        // Dump backend-specific IR
        if (glow::flags::DumpBackendSpecificIRJSON) {
          compiledFunction->dumpJSON(strFormat("%sbackend_specific_ir_%s.json",
                                               cctx.dumpGraphPath.c_str(),
                                               node->name.c_str()));
        }

        node->runtimeBundle = glow::make_unique<RuntimeBundle>(
            compiledFunction->getRuntimeBundle());
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
        return std::move(*addErr.get());
      }

      // Add networks successfully loaded on device to addedNetworks, this way
      // if we fail later we can evict them.
      for (const auto &func : functionMap) {
        addedNetworks[physicalDevice].push_back(func.first);
      }
      VLOG(1) << "Added networks";

      // Free up memory no longer needed by the compiledFunction.
      for (auto &node : logicalDevices[logicalDevice]) {
        // If the compiled function still needs to be added to other device,
        // don't free the resources.
        if (remainingDeviceCount[node->name] > 0) {
          continue;
        }

        // Free compilation resources. This need to be done after add network
        // and before move on to next logical device. If
        // DisableFreeCompilationResource is true, we will not free it here.
        // This is used in scenarios like model serialization.
        auto &funtionPtr = compiledFunctions[node->name];
        if (!glow::flags::DisableFreeCompilationResource) {
          funtionPtr->freeCompilationResources();
        }

        // Move compiled functions from compiledFunctions to functions_.
        {
          std::lock_guard<std::mutex> functionsLock(functionsLock_);
          functions_.emplace(node->name, std::move(funtionPtr));
        }

        compiledFunctions.erase(node->name);
      }
    }
  } else if (network->networkType == NetworkType::FX_NETWORK) {
#if FACEBOOK_INTERNAL
    // Container for duplicated functions and map tracking remaining installs
    // for a duplicated function.
    std::map<std::string, std::unique_ptr<CompiledFunction>>
        duplicatedFunctions;
    std::map<DAGNode *, unsigned> remainingDuplications;
    for (auto &assignment : assignments) {
      auto logicalDevice = assignment.first;
      auto physicalDevice = assignment.second;
      auto deviceBackendName = logicalDevices[logicalDevice][0]->backendName;
      FunctionMapTy functionMap;
      // Container for the compiledFunctions for this logicalDevice.
      std::map<std::string, std::unique_ptr<CompiledFunction>>
          compiledFunctions;

      for (auto &node : logicalDevices[logicalDevice]) {
        // Check if this is a duplicated function that has already been
        // compiled.
        if (duplicatedFunctions.find(node->name) != duplicatedFunctions.end()) {
          functionMap.emplace(node->name,
                              duplicatedFunctions[node->name].get());
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
          if (backends_.find(deviceBackendName) == backends_.end()) {
            // Return error requested device type not found.
            return MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_DEVICE_NOT_FOUND,
                            "Unable to find device of type: " +
                                deviceBackendName);
          }
          auto fxNetwork = static_cast<FXNetwork *>(network.get());
          auto compiledOrErr = backends_[deviceBackendName]->compileFX(
              fxNetwork->FXIR, node->name, fxNetwork->constants, options,
              &module);

          // Check to see if an error was encountered while compiling.
          if (!compiledOrErr) {
            // If an error occured return the error.
            RETURN_ERR(compiledOrErr.takeError());
          }
          auto compiled = std::move(*compiledOrErr);

          node->runtimeBundle =
              glow::make_unique<RuntimeBundle>(compiled->getRuntimeBundle());

          functionMap.emplace(node->name, compiled.get());
          // If this function is in more than one logical device store it for
          // reuse.
          if (node->logicalDevices.size() > 1) {
            duplicatedFunctions.emplace(node->name, std::move(compiled));
            remainingDuplications[node] = node->logicalDevices.size() - 1;
          } else {
            compiledFunctions.emplace(node->name, std::move(compiled));
          }
        }
      }
      VLOG(1) << "After compile";

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
        return std::move(*addErr.get());
      }
      // Add networks successfully loaded on device to addedNetworks, this way
      // if we fail later we can evict them.
      for (auto &node : logicalDevices[logicalDevice]) {
        addedNetworks[physicalDevice].push_back(node->name);
      }
      VLOG(1) << "Added networks";

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
            duplicatedFunctions[func.first->name]->freeCompilationResources();
            functions_.emplace(
                func.first->name,
                std::move(duplicatedFunctions[func.first->name]));
            duplicatedFunctions.erase(func.first->name);
            iter = remainingDuplications.erase(iter);
          } else {
            ++iter;
          }
        }
      }
    }
#endif
  }
  RETURN_ERR_IF_NOT(compiledFunctions.empty(),
                    "compiledFunctions should be empty because all compiled "
                    "functions should be moved to Provisioner::function_");

  // Map from Placeholder* to DeviceManager, this is used for deferred weight
  // loading.
  std::unordered_map<Placeholder *, std::vector<unsigned>>
      placeholderToDeviceManager;
  if (cctx.deferredWeightLoader) {
    // Populate placeholdeToDeviceManager map.
    for (auto &assignment : assignments) {
      for (const auto &node : logicalDevices[assignment.first]) {
        auto symbolTable = node->runtimeBundle->getSymbolTable();
        for (auto info : symbolTable) {
          if (info.second.symbolCategory ==
              glow::runtime::SymbolCategory::Placeholder) {
            auto PH = module.getPlaceholderByNameSlow(info.first);
            if (PH->isStatic()) {
              placeholderToDeviceManager[PH].push_back(assignment.second);
            }
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
      auto val = takeErrorValue(std::move(err));
      std::string msg = val->logToString();
      return MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_DEFERRED_WEIGHT_ERROR,
                      msg);
    }
    std::string weightName = loader->getName();
    // Load weights while there are weights to be loaded.
    unsigned int weightCount = 0;
    while (weightName != "") {
      LOG(INFO) << "Loading deferred weight (" << ++weightCount << " / "
                << totalNumDeferredWeights << "): " << weightName;
      const auto PH = module.getPlaceholderByNameSlow(weightName);
      if (!PH) {
        return MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_DEFERRED_WEIGHT_ERROR,
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
        auto val = takeErrorValue(std::move(err));
        std::string msg = val->logToString();
        return MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_DEFERRED_WEIGHT_ERROR,
                        msg);
      }
      weightName = loader->getName();
      // Remove PH from map, this way we can know that we've added all static
      // PH's
      placeholderToDeviceManager.erase(PH);
    }
    if (placeholderToDeviceManager.size()) {
      return MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_DEFERRED_WEIGHT_ERROR,
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

  cleanupGuard.dismiss();
  cleanupProvision(localActiveNames, {}, false);
  return Error::success();
};

Error Provisioner::provision(DAGListTy &networks, Module &module,
                             CompilationContext &cctx) {
  return provisionNetwork(
      glow::make_unique<GlowNetwork>(networks, module, cctx));
};

#if FACEBOOK_INTERNAL
Error Provisioner::provisionFX(DAGListTy &networks, Module &module,
                               const FXFunction &FXIR,
                               const llvm::StringMap<const void *> &constants,
                               CompilationContext &cctx) {
  return provisionNetwork(
      glow::make_unique<FXNetwork>(networks, module, cctx, FXIR, constants));
};
#endif

Backend &Provisioner::getBackend(llvm::StringRef backendName) const {
  assert(backends_.count(backendName.str()) &&
         "No backend created by specified name.");
  return *backends_.at(backendName.str());
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
  auto it = activeFunctions_.find(name.str());
  if (it != activeFunctions_.end()) {
    return MAKE_ERR(
        ErrorValue::ErrorCode::RUNTIME_NET_BUSY,
        llvm::formatv("Could not remove network: {0} as it is currently "
                      "being provisioned.",
                      name)
            .str());
  }
  functions_.erase(name.str());
  return Error::success();
}

Error Provisioner::evictFunction(llvm::StringRef name, DeviceManager *device,
                                 unsigned replicaCount) {
  std::promise<void> evictPromise;
  OneErrOnly evictErr;
  auto done = evictPromise.get_future();
  device->evictNetwork(name.str(),
                       [&evictPromise, &evictErr](std::string, Error err) {
                         evictErr.set(std::move(err));
                         evictPromise.set_value();
                       });
  done.get();

  // If we are evict a main function, evict its replications as well.
  if (replicaCount) {
    for (unsigned i = 1; i < replicaCount; i++) {
      auto replicaName = getReplicatedName(name.str(), i);
      std::promise<void> evictReplicaPromise;
      auto done = evictReplicaPromise.get_future();
      device->evictNetwork(replicaName, [&evictReplicaPromise,
                                         &evictErr](std::string, Error err) {
        evictErr.set(std::move(err));
        evictReplicaPromise.set_value();
      });

      done.get();
    }
  }

  return evictErr.get();
}

void Provisioner::cleanupProvision(
    llvm::ArrayRef<std::string> names,
    std::map<DeviceIDTy, std::vector<std::string>> const
        &currentNetworkResidency,
    bool failure) {
  std::lock_guard<std::mutex> functionLock(functionsLock_);
  if (failure) {
    // Remove any partitions added to devices.
    for (auto &device : currentNetworkResidency) {
      for (auto &network : device.second) {
#if FACEBOOK_INTERNAL
        LOG(INFO) << "Removing network " << network << " from device "
                  << device.first;
#endif
        auto replicaCountIdx = functionReplicaCount_.find(network);
        unsigned replicaCount = 0;
        if (replicaCountIdx != functionReplicaCount_.end()) {
          replicaCount = replicaCountIdx->second;
        }
        Error evictErr =
            evictFunction(network, devices_[device.first], replicaCount);
        if (evictErr) {
          LOG(ERROR) << "Unable to evict network: " << network << "\n";
        }
      }
    }
  }
  // After we've removed the functions from the deviceManagers now free the
  // compiledFunctions. We free after eviction to ensure the any reference the
  // DeviceManager has to the compiledFunctions stays valid until after
  // eviction.
  for (auto &name : names) {
    activeFunctions_.erase(name);
    if (failure) {
      // Remove any functions added before the failure.
      functions_.erase(name);
    }
  }
}

void Provisioner::cleanUpSerializedFunctionMap() {
  serializedFunctionMap_.clear();
}

// Get the hash as a string from a function's name
std::string getNameHash(std::string name) {
  return name.substr(name.find_last_of("_") + 1);
}

std::unique_ptr<
    std::unordered_map<std::string, std::unique_ptr<BlockStreamBase>>>
Provisioner::getAllSerializedFunctionsMap() {
  // Assume all functions in functions_ are using the same backend
  cleanUpSerializedFunctionMap();
  for (auto &kv : functions_) {
    std::string name = kv.first;
    auto data = kv.second->serialize();
    if (data != nullptr) {
      serializedFunctionMap_.emplace(
          std::make_pair(getNameHash(name), std::move(data)));
    }
  }
  return std::make_unique<
      std::unordered_map<std::string, std::unique_ptr<BlockStreamBase>>>(
      std::move(serializedFunctionMap_));
}
