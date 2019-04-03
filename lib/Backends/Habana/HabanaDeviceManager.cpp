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

#include "HabanaDeviceManager.h"

#include "llvm/Support/raw_ostream.h"

#include "synapse.h"

using namespace glow;
using namespace glow::runtime;

// TODO: A failed status probably shouldn't be an assert. We should
// fail gracefully.
#define chk(X) GLOW_ASSERT((X) == synSuccess)

namespace glow {
namespace runtime {
/// Factory function for creating a HabanaDeviceManager.
DeviceManager *
createHabanaDeviceManager(std::unique_ptr<DeviceConfig> config = nullptr) {
  return new HabanaDeviceManager(std::move(config));
}
} // namespace runtime
} // namespace glow

// It isn't clear what's threadsafe in Synapse.  Lock everything for now.
static std::mutex synapseLock;

// Initialization of static class variables.
unsigned HabanaDeviceManager::numActiveDevices_{0};
std::mutex HabanaDeviceManager::mtx_;

HabanaDeviceManager::HabanaDeviceManager(std::unique_ptr<DeviceConfig> config)
    : QueueBackedDeviceManager(BackendKind::Habana, std::move(config)) {
  std::lock_guard<std::mutex> lock(mtx_);

  // If this is the first HabanaDeviceManager to be created, initialize the
  // Synapse API.
  if (numActiveDevices_ == 0) {
    chk(synInitialize());
  }

  numActiveDevices_++;
}

HabanaDeviceManager::~HabanaDeviceManager() {
  std::lock_guard<std::mutex> lock(mtx_);
  numActiveDevices_--;

  // If this is the last HabanaDeviceManager to be destroyed, destroy the
  // Synapse API.
  if (numActiveDevices_ == 0) {
    chk(synDestroy());
  }
}

llvm::Error HabanaDeviceManager::init() {
  std::lock_guard<std::mutex> lock(synapseLock);
  // Acquire a device to work with for the lifetime of this instance.
  synStatus status = synAcquireDevice(&deviceId_, nullptr);

  if (status != synSuccess) {
    RETURN_ERR("Failed to acquire device");
  }

  // Fetch initial memory information.
  status = synGetMemInfo(deviceId_, &freeMemory_, &totalMemory_);

  if (status != synSuccess) {
    RETURN_ERR("Failed to get memory info");
  }

  return llvm::Error::success();
}

void HabanaDeviceManager::addNetworkImpl(const Module *module,
                                         FunctionMapTy functions,
                                         ReadyCBTy readyCB) {
  std::lock_guard<std::mutex> lock(synapseLock);
  for (const auto &func : functions) {
    // Check if a function with the same name has already been added.
    if (functions_.count(func.first) != 0) {
      llvm::errs() << "Failed to add network: already have a function called "
                   << func.first << ".\n";
      readyCB(module, MAKE_ERR("Failed to add network"));
      return;
    }

    uint64_t topologyId = 0;
    HabanaFunction *habanaFunction = static_cast<HabanaFunction *>(func.second);

    // Load the recipe (created during compilation) and store the resultant
    // topology ID. This is the reference that will be used lated to "activate"
    // this function and make it executable.
    synStatus status = synLoadRecipe(
        deviceId_, habanaFunction->getRecipeName().c_str(), &topologyId);

    if (status != synSuccess) {
      llvm::errs() << "Unable to load recipe "
                   << habanaFunction->getRecipeName() << " for function "
                   << func.first << ".\n";
      // TODO: Unload functions that were loaded successfully.
      readyCB(module, MAKE_ERR("Unable to load recipe"));
      return;
    }

    // Insert the function into functions_.
    bool inserted = false;
    std::tie(std::ignore, inserted) = functions_.insert(std::make_pair(
        func.first, HabanaFunctionMeta(deviceId_, topologyId, habanaFunction)));

    if (!inserted) {
      llvm::errs() << "Unable to add function " << func.first
                   << "to HabanaDeviceManager.\n";
      // TODO: Unload functions that were loaded successfully.
      readyCB(module, MAKE_ERR("Unable to add function"));
      return;
    }
  }

  // Update memory information after loading all the functions.
  chk(synGetMemInfo(deviceId_, &freeMemory_, &totalMemory_));
  readyCB(module, llvm::Error::success());
}

void HabanaDeviceManager::evictNetworkImpl(std::string functionName,
                                           EvictFunctionCBTy evictCB) {
  std::lock_guard<std::mutex> lock(synapseLock);

  // Check if a network with the given name exists on the device.
  if (functions_.count(functionName) == 0) {
    llvm::errs() << "Failed to evict network: function called " << functionName
                 << " was not added.\n";
    evictCB(functionName, MAKE_ERR("Failed to evict network"));
    return;
  }

  // Unload the topology ID corresponding to the function.
  synStatus status =
      synUnloadTopology(deviceId_, functions_[functionName].topologyId);

  if (status != synSuccess) {
    llvm::errs() << "Unable to unload function " << functionName << ".\n";
    evictCB(functionName, MAKE_ERR("Unable to unload function"));
    return;
  }

  // Erase the function from the functions_ map.
  auto numErased = functions_.erase(functionName);

  if (numErased == 0) {
    llvm::errs() << "Unable to evict function " << functionName
                 << "from HabanaDeviceManager.\n";
    evictCB(functionName, MAKE_ERR("Unable to evict function"));
  }

  // Update memory information after evicting the function.
  chk(synGetMemInfo(deviceId_, &freeMemory_, &totalMemory_));

  evictCB(functionName, llvm::Error::success());
}

void HabanaDeviceManager::runFunctionImpl(RunIdentifierTy runId,
                                          std::string functionName,
                                          std::unique_ptr<ExecutionContext> ctx,
                                          runtime::ResultCBTy resultCB) {
  std::lock_guard<std::mutex> lock(synapseLock);
  // Try to find the function with the given name in functions_.
  auto it = functions_.find(functionName);
  if (it == functions_.end()) {
    llvm::errs() << "Failed to run function: function called " << functionName
                 << " was not added.\n";
    resultCB(runId, MAKE_ERR("Failed to run function"), std::move(ctx));
    return;
  }

  HabanaFunctionMeta *meta = &it->second;
  uint64_t topologyId = meta->topologyId;
  HabanaFunction *function = meta->function;

  // Activate the topology ID.
  synStatus status = synActivateTopology(deviceId_, topologyId);

  if (status != synSuccess) {
    llvm::errs() << "Failed to activate topology for function " << functionName
                 << ".\n";
    resultCB(runId, MAKE_ERR("Failed to activate topology"), std::move(ctx));
    return;
  }

  // TODO: Make this return some error code?
  ctx->setDeviceBindings(llvm::make_unique<HabanaBindings>(deviceId_, meta));
  function->execute(ctx.get());
  resultCB(runId, llvm::Error::success(), std::move(ctx));
}

llvm::Error HabanaDeviceManager::stop(bool /*block*/) {
  std::lock_guard<std::mutex> lock(synapseLock);
  functions_.clear();
  synStatus status = synReleaseDevice(deviceId_);

  if (status != synSuccess) {
    return MAKE_ERR("Failed to release device");
  }

  return llvm::Error::success();
}

uint64_t HabanaDeviceManager::getMaximumMemory() const { return totalMemory_; }

uint64_t HabanaDeviceManager::getAvailableMemory() const { return freeMemory_; }

bool HabanaDeviceManager::isMemoryAvailable(uint64_t estimate) const {
  return estimate <= freeMemory_;
}
