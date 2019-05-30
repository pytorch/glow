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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include "synapse.h"

#include <glog/logging.h>
#include <limits>

using namespace glow;
using namespace glow::runtime;

namespace glow {
namespace runtime {

unsigned GlowHabanaMemory = 7 << 20; // 7 GB.

static llvm::cl::opt<unsigned, /* ExternalStorage */ true> GlowHabanaMemoryOpt(
    "glow-habana-memory",
    llvm::cl::desc("Amount of DRAM to allocate per Habana device in kilobytes"),
    llvm::cl::location(GlowHabanaMemory));

// TODO: A failed status probably shouldn't be an assert. We should
// fail gracefully.
#define chk(X) GLOW_ASSERT((X) == synSuccess)

/// Factory function for creating a HabanaDeviceManager.
DeviceManager *
createHabanaDeviceManager(std::unique_ptr<DeviceConfig> config = nullptr) {
  return new HabanaDeviceManager(std::move(config));
}
} // namespace runtime
} // namespace glow

// Initialization of static class variables.
unsigned HabanaDeviceManager::numActiveDevices_{0};
std::mutex HabanaDeviceManager::synapseMtx_;
std::atomic<RunIdentifierTy> HabanaDeviceManager::runIdentifier_;

HabanaDeviceManager::HabanaDeviceManager(std::unique_ptr<DeviceConfig> config,
                                         unsigned numRunners,
                                         unsigned numWaiters)
    : DeviceManager(BackendKind::Habana, std::move(config)),
      numRunners_(numRunners), numWaiters_(numWaiters) {}

HabanaDeviceManager::~HabanaDeviceManager() {
  // If a device was never successfully acquired, there's nothing to clean up.
  if (deviceId_ == INVALID_DEVICE) {
    return;
  }
  std::lock_guard<std::mutex> lock(synapseMtx_);
  numActiveDevices_--;

  // Explicitly clear this map to force synFree of the managed IOBuffers to
  // happen now, before we synReleaseDevice.  Otherwise synReleaseDevice will
  // free the buffers, and then the destructor will try to do it again.
  functions_.clear();
  chk(synReleaseDevice(deviceId_));

  // If this is the last HabanaDeviceManager to be destroyed, destroy the
  // Synapse API.
  if (numActiveDevices_ == 0) {
    chk(synDestroy());
  }
}

llvm::Error HabanaDeviceManager::init() {
  std::lock_guard<std::mutex> lock(synapseMtx_);

  // If this is the first HabanaDeviceManager to be created, initialize the
  // Synapse API.
  if (numActiveDevices_ == 0) {
    LOG(INFO) << "Using version " << synGetVersion();
    chk(synInitialize());
  }

  // Acquire a device to work with for the lifetime of this instance.
  synStatus status = synAcquireDevice(&deviceId_, nullptr);
  if (status != synSuccess) {
    RETURN_ERR("Failed to acquire device");
  }

  numActiveDevices_++;

  // Fetch initial memory information.
  RETURN_IF_ERR(updateMemoryUsage());

  // Create thread pools for running functions and waiting on function results.
  runPool_ = llvm::make_unique<ThreadPool>(numRunners_);
  waitPool_ = llvm::make_unique<ThreadPool>(numWaiters_);

  if (!runPool_ || !waitPool_) {
    RETURN_ERR("Failed to create HabanaDeviceManager thread pools");
  }

  return llvm::Error::success();
}

llvm::Error HabanaDeviceManager::updateMemoryUsage() {
  // TODO: Use synGetMemInfo once implemented.

  totalMemory_ = uint64_t{GlowHabanaMemory} * 1024;
  freeMemory_ = uint64_t{GlowHabanaMemory} * 1024;

  // Account for the size used by each function loaded on the card.
  for (const auto &pr : functions_) {
    const auto &functionMeta = pr.second;
    const auto &runtimeBundle = functionMeta.function->getRuntimeBundle();
    freeMemory_ -= runtimeBundle.getConstantWeightSize();
    freeMemory_ -= runtimeBundle.getMutableWeightSize();
  }

  return llvm::Error::success();
}

void HabanaDeviceManager::addNetwork(const Module *module,
                                     FunctionMapTy functions,
                                     ReadyCBTy readyCB) {
  std::unique_lock<std::mutex> lk(instanceMtx_);
  for (const auto &func : functions) {
    // Check if a function with the same name has already been added.
    if (functions_.count(func.first) != 0) {
      llvm::errs() << "Failed to add network: already have a function called "
                   << func.first << ".\n";
      lk.unlock();
      readyCB(module, MAKE_ERR("Failed to add network"));
      return;
    }

    uint64_t topologyId = 0;
    HabanaFunction *habanaFunction = static_cast<HabanaFunction *>(func.second);

    // Load the recipe (created during compilation) and store the resultant
    // topology ID. This is the reference that will be used lated to "activate"
    // this function and make it executable.
    synStatus status = synFail;

    {
      std::lock_guard<std::mutex> lock(synapseMtx_);
      status = synLoadRecipe(deviceId_, habanaFunction->getRecipeName().c_str(),
                             &topologyId);
    }

    if (status != synSuccess) {
      llvm::errs() << "Unable to load recipe "
                   << habanaFunction->getRecipeName() << " for function "
                   << func.first << ".\n";
      // TODO: Unload functions that were loaded successfully.
      lk.unlock();
      readyCB(module, MAKE_ERR("Unable to load recipe"));
      return;
    }

    // Insert the function into functions_.
    bool inserted = false;
    std::tie(std::ignore, inserted) = functions_.insert(std::make_pair(
        func.first,
        HabanaFunctionMeta{topologyId, habanaFunction,
                           llvm::make_unique<HabanaIOBufferPool>(
                               deviceId_, habanaFunction->getInputs(),
                               habanaFunction->getOutputs())}));

    if (!inserted) {
      llvm::errs() << "Unable to add function " << func.first
                   << "to HabanaDeviceManager.\n";
      // TODO: Unload functions that were loaded successfully.
      lk.unlock();
      readyCB(module, MAKE_ERR("Unable to add function"));
      return;
    }

    // Optimistically activate the topology if nothing else is loaded.
    cv_.wait(lk, [this] { return inflightRequests_ == 0; });
    chk(synActivateTopology(deviceId_, topologyId));
    activeTopo_ = topologyId;
  }

  lk.unlock();

  // Update memory information after loading all the functions.
  if (auto err = updateMemoryUsage()) {
    readyCB(module, std::move(err));
    return;
  }

  readyCB(module, llvm::Error::success());
}

void HabanaDeviceManager::evictNetwork(std::string functionName,
                                       EvictFunctionCBTy evictCB) {
  std::unique_lock<std::mutex> lk(instanceMtx_);

  // Check if a network with the given name exists on the device.
  if (functions_.count(functionName) == 0) {
    llvm::errs() << "Failed to evict network: function called " << functionName
                 << " was not added.\n";
    lk.unlock();
    evictCB(functionName, MAKE_ERR("Failed to evict network"));
    return;
  }

  // Unload the topology ID corresponding to the function.
  synStatus status = synFail;
  uint64_t topologyId = functions_[functionName].topologyId;

  {
    std::lock_guard<std::mutex> lock(synapseMtx_);
    status = synUnloadTopology(deviceId_, topologyId);
    if (topologyId == activeTopo_) {
      activeTopo_ = INVALID_TOPOLOGY;
    }
  }

  if (status != synSuccess) {
    llvm::errs() << "Unable to unload function " << functionName << ".\n";
    lk.unlock();
    evictCB(functionName, MAKE_ERR("Unable to unload function"));
    return;
  }

  // Erase the function from the functions_ map.
  auto numErased = functions_.erase(functionName);

  if (numErased == 0) {
    llvm::errs() << "Unable to evict function " << functionName
                 << "from HabanaDeviceManager.\n";
    lk.unlock();
    evictCB(functionName, MAKE_ERR("Unable to evict function"));
    return;
  }

  lk.unlock();

  // Update memory information after evicting the function.
  if (auto err = updateMemoryUsage()) {
    evictCB(functionName, std::move(err));
    return;
  }

  evictCB(functionName, llvm::Error::success());
}

void HabanaDeviceManager::runFunctionImpl(RunIdentifierTy runId,
                                          std::string functionName,
                                          std::unique_ptr<ExecutionContext> ctx,
                                          runtime::ResultCBTy resultCB) {
  TRACE_EVENT_SCOPE(ctx->getTraceContext(), "HabanaDM::runnerThread");
  // Try to find the function with the given name in functions_.
  uint64_t topologyId;
  HabanaFunction *function;
  HabanaIOBufferPool *ioBufferPool;
  {
    std::lock_guard<std::mutex> lock(instanceMtx_);
    auto it = functions_.find(functionName);
    if (it == functions_.end()) {
      llvm::errs() << "Failed to run function: function called " << functionName
                   << " was not added.\n";
      resultCB(runId, MAKE_ERR("Function not added"), std::move(ctx));
      return;
    }

    topologyId = (it->second).topologyId;
    function = (it->second).function;
    ioBufferPool = (it->second).ioBufferPool.get();
  }

  // If we need to switch topos, wait to drain the queue.
  {
    std::unique_lock<std::mutex> lock(instanceMtx_);
    if (topologyId != activeTopo_) {
      // FIXME: This can starve inactive topos.
      cv_.wait(lock, [this] { return inflightRequests_ == 0; });
      chk(synActivateTopology(deviceId_, topologyId));
      activeTopo_ = topologyId;
    }
    inflightRequests_++;
  }

  // Execute the function.
  auto deviceBindings =
      llvm::make_unique<HabanaBindings>(deviceId_, topologyId);
  deviceBindings->setIOBuffer(ioBufferPool->get());
  ctx->setDeviceBindings(std::move(deviceBindings));

  auto executeErr = function->execute(ctx.get());
  if (executeErr) {
    resultCB(runId, std::move(executeErr), std::move(ctx));
    return;
  }

  // Give the handle to the wait thread pool to wait on and call the callback
  // for.
  waitPool_->submit([this, runId, function, ioBufferPool,
                     functionName = std::move(functionName),
                     ctx = std::move(ctx),
                     resultCB = std::move(resultCB)]() mutable {
    TRACE_EVENT_SCOPE(ctx->getTraceContext(), "HabanaDM::waiterThread");
    TRACE_EVENT_BEGIN(ctx->getTraceContext(), "wait");
    auto &habanaHandle =
        static_cast<HabanaBindings *>(ctx->getDeviceBindings())->getHandle();
    bool ok = habanaHandle.wait();
    std::unique_ptr<HabanaIOBuffer> ioBuffer =
        static_cast<HabanaBindings *>(ctx->getDeviceBindings())->getIOBuffer();
    TRACE_EVENT_END(ctx->getTraceContext(), "wait");

    // Notify anything waiting for a topo switch.
    {
      std::lock_guard<std::mutex> lock(this->instanceMtx_);
      inflightRequests_--;
    }
    cv_.notify_one();

    if (!ok) {
      // Return the IO buffer to the IO buffer pool.
      ioBufferPool->put(std::move(ioBuffer));

      llvm::errs() << "Failed to execute function " << functionName << ".\n";
      resultCB(runId, MAKE_ERR("Failed to execute function"), std::move(ctx));
    } else {
      // Copy the execution outputs from the designated IO buffer back to the
      // PlaceholderBindings inside ctx.
      TRACE_EVENT_BEGIN(ctx->getTraceContext(), "copyOutputs");
      auto bindings = ctx->getPlaceholderBindings();
      for (const auto &ph : function->getOutputs()) {
        auto *tensor = bindings->get(ph);
        if (!tensor) {
          tensor = bindings->get(bindings->getPlaceholderByName(ph->getName()));
        }
        memcpy(tensor->getUnsafePtr(), ioBuffer->get(ph),
               ph->getType()->getSizeInBytes());
      }
      TRACE_EVENT_END(ctx->getTraceContext(), "copyOutputs");

      // Return the IO buffer to the IO buffer pool.
      ioBufferPool->put(std::move(ioBuffer));
      resultCB(runId, llvm::Error::success(), std::move(ctx));
    }
  });
}

RunIdentifierTy
HabanaDeviceManager::runFunction(std::string functionName,
                                 std::unique_ptr<ExecutionContext> ctx,
                                 runtime::ResultCBTy resultCB) {
  RunIdentifierTy runId = runIdentifier_++;
  runPool_->submit([this, runId, functionName = std::move(functionName),
                    ctx = std::move(ctx),
                    resultCB = std::move(resultCB)]() mutable {
    runFunctionImpl(runId, std::move(functionName), std::move(ctx),
                    std::move(resultCB));
  });
  return runId;
}

llvm::Error HabanaDeviceManager::stop(bool block) {
  runPool_->stop(block);
  waitPool_->stop(block);
  return llvm::Error::success();
}

uint64_t HabanaDeviceManager::getMaximumMemory() const { return totalMemory_; }

uint64_t HabanaDeviceManager::getAvailableMemory() const { return freeMemory_; }

bool HabanaDeviceManager::isMemoryAvailable(uint64_t estimate) const {
  return estimate <= freeMemory_;
}
