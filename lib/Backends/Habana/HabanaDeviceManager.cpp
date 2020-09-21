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

#include "HabanaDeviceManager.h"

#include "glow/Flags/Flags.h"
#include "glow/Runtime/StatsExporter.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "synapse.h"

#include <glog/logging.h>
#include <limits>

using namespace glow;
using namespace glow::runtime;

namespace glow {
namespace runtime {

static llvm::cl::opt<unsigned, /* ExternalStorage */ true> GlowHabanaMemoryOpt(
    "glow-habana-memory",
    llvm::cl::desc("Amount of DRAM to allocate per Habana device in kilobytes"),
    llvm::cl::location(GlowHabanaMemory));

DeviceManager *createHabanaDeviceManager(const DeviceConfig &config) {
  return new HabanaDeviceManager(config);
}
} // namespace runtime
} // namespace glow

// Initialization of static class variables.
unsigned HabanaDeviceManager::numActiveDevices_{0};
std::mutex HabanaDeviceManager::synapseMtx_;
std::atomic<RunIdentifierTy> HabanaDeviceManager::runIdentifier_;

HabanaDeviceManager::HabanaDeviceManager(const DeviceConfig &config,
                                         unsigned numRunners,
                                         unsigned numWaiters)
    : DeviceManager(config), numRunners_(numRunners), numWaiters_(numWaiters) {}

HabanaDeviceManager::~HabanaDeviceManager() {
  // If a device was never successfully acquired, there's nothing to clean up.
  if (deviceId_ == INVALID_DEVICE) {
    return;
  }
  std::lock_guard<std::mutex> lock(synapseMtx_);
  numActiveDevices_--;
  statsExporterRegistry_->incrementCounter(kDevicesUsedHabana, -1);

  // Explicitly clear this map to force synFree of the managed IOBuffers to
  // happen now, before we synReleaseDevice.  Otherwise synReleaseDevice will
  // free the buffers, and then the destructor will try to do it again.
  functions_.clear();
  chk_kill(synReleaseDevice(deviceId_));

  // If this is the last HabanaDeviceManager to be destroyed, destroy the
  // Synapse API.
  if (numActiveDevices_ == 0) {
    chk_kill(synDestroy());
  }
}

Error HabanaDeviceManager::init() {
  std::lock_guard<std::mutex> lock(synapseMtx_);

  // If this is the first HabanaDeviceManager to be created, initialize the
  // Synapse API.
  if (numActiveDevices_ == 0) {
    LOG(INFO) << "Using version " << synGetVersion();
    // This environment variable tells Synapse to allow enqueueing tensors that
    // are smaller than the declared size, which offers a significant savings
    // in PCI traffic for embedding lookups.
    setenv("IGNORE_ENQUEUE_SIZE_VALIDATION", "1", /*overwrite*/ 1);
    chk(synInitialize());
  }

  // Acquire a device to work with for the lifetime of this instance.
  synStatus status = synAcquireDevice(&deviceId_, nullptr);
  if (status != synSuccess) {
    RETURN_ERR("Failed to acquire device");
  }

  numActiveDevices_++;
  statsExporterRegistry_->incrementCounter(kDevicesUsedHabana);

  // Fetch initial memory information.
  RETURN_IF_ERR(updateMemoryUsage());

  // Create thread pools for running functions and waiting on function results.
  runPool_ = glow::make_unique<ThreadPool>(numRunners_);
  waitPool_ = glow::make_unique<ThreadPool>(numWaiters_);

  if (!runPool_ || !waitPool_) {
    RETURN_ERR("Failed to create HabanaDeviceManager thread pools");
  }

  return Error::success();
}

Error HabanaDeviceManager::updateMemoryUsage() {
  // TODO: Use synGetMemInfo once implemented.

  // Use GlowHabanaMemory if it is defined from GFLAGS or llvm params,
  // otherwise, fall back to what config says.
  uint64_t defaultMemory = 7 << 20;
  if (GlowHabanaMemory == defaultMemory && config_.getDeviceMemory() != 0) {
    totalMemory_ = config_.getDeviceMemory();
  } else {
    totalMemory_ = uint64_t{GlowHabanaMemory} * 1024;
  }
  freeMemory_ = totalMemory_;

  // Account for the size used by each function loaded on the card.
  for (const auto &pr : functions_) {
    const auto &functionMeta = pr.second;
    const auto &runtimeBundle = functionMeta.function->getRuntimeBundle();
    freeMemory_ -= runtimeBundle.getConstantWeightSize();
    freeMemory_ -= runtimeBundle.getMutableWeightSize();
  }

  return Error::success();
}

void HabanaDeviceManager::addNetwork(const Module *module,
                                     FunctionMapTy functions,
                                     ReadyCBTy readyCB) {
  DCHECK(readyCB != nullptr);

  std::unique_lock<std::mutex> lk(instanceMtx_);
  for (const auto &func : functions) {
    // Check if a function with the same name has already been added.
    if (functions_.count(func.first) != 0) {
      lk.unlock();
      readyCB(module,
              MAKE_ERR(strFormat(
                  "Failed to add network: already have a function called %s",
                  func.first.c_str())));
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

    if (auto err = chk_make_err(status)) {
      LOG(ERROR) << "Unable to load recipe " << habanaFunction->getRecipeName()
                 << " for function " << func.first << ".";
      // TODO: Unload functions that were loaded successfully.
      lk.unlock();
      readyCB(module, std::move(err));
      return;
    }

    // Insert the function into functions_.
    bool inserted = false;
    std::tie(std::ignore, inserted) = functions_.insert(std::make_pair(
        func.first,
        HabanaFunctionMeta{topologyId, habanaFunction,
                           glow::make_unique<HabanaIOBufferPool>(
                               deviceId_, habanaFunction->getInputs(),
                               habanaFunction->getOutputs())}));

    if (!inserted) {
      // TODO: Unload functions that were loaded successfully.
      lk.unlock();
      readyCB(module, MAKE_ERR(strFormat(
                          "Unable to add function %s to HabanaDeviceManager",
                          func.first.c_str())));
      return;
    }

    // Optimistically activate the topology if nothing else is loaded.
    cv_.wait(lk, [this] { return inflightRequests_ == 0; });
    if (auto err = chk_make_err(synActivateTopology(deviceId_, topologyId))) {
      lk.unlock();
      readyCB(module, std::move(err));
      return;
    }
    activeTopo_ = topologyId;
  }

  lk.unlock();

  // Update memory information after loading all the functions.
  if (auto err = updateMemoryUsage()) {
    readyCB(module, std::move(err));
    return;
  }

  readyCB(module, Error::success());
}

void HabanaDeviceManager::evictNetwork(std::string functionName,
                                       EvictFunctionCBTy evictCB) {
  DCHECK(evictCB != nullptr);

  std::unique_lock<std::mutex> lk(instanceMtx_);

  // Check if a network with the given name exists on the device.
  if (functions_.count(functionName) == 0) {
    lk.unlock();
    evictCB(functionName,
            MAKE_ERR(strFormat(
                "Failed to evict network: function called %s was not added",
                functionName.c_str())));
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

  if (auto err = chk_make_err(status)) {
    LOG(ERROR) << "Unable to unload function " << functionName;
    lk.unlock();
    evictCB(functionName, std::move(err));
    return;
  }

  // Erase the function from the functions_ map.
  auto numErased = functions_.erase(functionName);

  if (numErased == 0) {
    lk.unlock();
    evictCB(functionName,
            MAKE_ERR(strFormat(
                "Unable to evict function %s from HabanaDeviceManager",
                functionName.c_str())));
    return;
  }

  lk.unlock();

  // Update memory information after evicting the function.
  if (auto err = updateMemoryUsage()) {
    evictCB(functionName, std::move(err));
    return;
  }

  evictCB(functionName, Error::success());
}

void HabanaDeviceManager::runFunctionImpl(RunIdentifierTy runId,
                                          std::string functionName,
                                          std::unique_ptr<ExecutionContext> ctx,
                                          runtime::ResultCBTy resultCB) {
  DCHECK(resultCB != nullptr);

  TRACE_EVENT_SCOPE_NAMED(ctx->getTraceContext(), TraceLevel::RUNTIME,
                          "HabanaDM::runnerThread", trEvent);

  /// Habana DeviceManager doesn't support Device Resident Tensors.
  ctx->getPlaceholderBindings()->ensureOnHost();

  if (ctx->getTraceContext()) {
    ctx->getTraceContext()->setThreadName(
        llvm::formatv("Habana {0} (enqueue)", deviceId_).str());
  }
  // Try to find the function with the given name in functions_.
  uint64_t topologyId;
  HabanaFunction *function;
  HabanaIOBufferPool *ioBufferPool;
  {
    std::lock_guard<std::mutex> lock(instanceMtx_);
    auto it = functions_.find(functionName);
    if (it == functions_.end()) {
      resultCB(runId,
               MAKE_ERR(strFormat(
                   "Failed to run function: function called %s was not added",
                   functionName.c_str())),
               std::move(ctx));
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
      const auto activateTopoRes = synActivateTopology(deviceId_, topologyId);
      if (auto err = chk_make_err(activateTopoRes)) {
        LOG(ERROR) << "synActivateTopology failed with status "
                   << activateTopoRes;
        trEvent.addArg(
            "error", llvm::formatv("synActivateTopology failed with status {0}",
                                   activateTopoRes)
                         .str());
        TRACE_EVENT_SCOPE_END_NAMED(trEvent);
        resultCB(runId, std::move(err), std::move(ctx));
        return;
      }
      activeTopo_ = topologyId;
    }
    inflightRequests_++;
  }

  // Execute the function.
  auto deviceBindings =
      glow::make_unique<HabanaBindings>(deviceId_, topologyId);
  deviceBindings->setIOBuffer(ioBufferPool->get());
  ctx->setDeviceBindings(std::move(deviceBindings));

  auto executeErr = function->execute(ctx.get());
  if (executeErr) {
    trEvent.addArg("error", "execute() failed");
    TRACE_EVENT_SCOPE_END_NAMED(trEvent);
    resultCB(runId, std::move(executeErr), std::move(ctx));
    return;
  }

  // Give the handle to the wait thread pool to wait on and call the callback
  // for.
  waitPool_->submit([this, runId, function, ioBufferPool,
                     functionName = std::move(functionName),
                     ctx = std::move(ctx),
                     resultCB = std::move(resultCB)]() mutable {
    DCHECK(resultCB != nullptr);

    TRACE_EVENT_SCOPE(ctx->getTraceContext(), TraceLevel::RUNTIME,
                      "HabanaDM::waiterThread");
    if (ctx->getTraceContext()) {
      ctx->getTraceContext()->setThreadName(
          llvm::formatv("Habana {0} (waiter)", deviceId_).str());
    }

    TRACE_EVENT_BEGIN(ctx->getTraceContext(), TraceLevel::RUNTIME, "wait");
    auto &habanaHandle =
        static_cast<HabanaBindings *>(ctx->getDeviceBindings())->getHandle();
    bool ok = habanaHandle.wait();
    std::unique_ptr<HabanaIOBuffer> ioBuffer =
        static_cast<HabanaBindings *>(ctx->getDeviceBindings())->getIOBuffer();
    TRACE_EVENT_END(ctx->getTraceContext(), TraceLevel::RUNTIME, "wait");

    // Notify anything waiting for a topo switch.
    {
      std::lock_guard<std::mutex> lock(this->instanceMtx_);
      inflightRequests_--;
    }
    cv_.notify_one();

    if (!ok) {
      // Return the IO buffer to the IO buffer pool.
      ioBufferPool->put(std::move(ioBuffer));

      resultCB(runId,
               MAKE_ERR(strFormat("Failed to execute function %s",
                                  functionName.c_str())),
               std::move(ctx));
    } else {
      // Copy the execution outputs from the designated IO buffer back to the
      // PlaceholderBindings inside ctx.
      TRACE_EVENT_SCOPE_NAMED(ctx->getTraceContext(), TraceLevel::RUNTIME,
                              "copyOutputs", coEvent);
      auto bindings = ctx->getPlaceholderBindings();
      size_t tensors{0}, bytes{0};
      for (const auto &ph : function->getOutputs()) {
        auto *tensor = bindings->get(ph);
        if (!tensor) {
          tensor =
              bindings->get(bindings->getPlaceholderByNameSlow(ph->getName()));
        }
        tensors++;

        if (auto ioBufferDataOrErr = ioBuffer->get(ph)) {
          memcpy(tensor->getUnsafePtr(), *ioBufferDataOrErr,
                 ph->getType()->getSizeInBytes());
          bytes += ph->getType()->getSizeInBytes();
        } else {
          // Return the IO buffer to the IO buffer pool.
          ioBufferPool->put(std::move(ioBuffer));
          coEvent.addArg("tensors", std::to_string(tensors));
          coEvent.addArg("bytes", std::to_string(bytes));
          coEvent.addArg("missingTensor", ph->getName().str());
          TRACE_EVENT_SCOPE_END_NAMED(coEvent);
          resultCB(runId, ioBufferDataOrErr.takeError(), std::move(ctx));
          return;
        }
      }
      coEvent.addArg("tensors", std::to_string(tensors));
      coEvent.addArg("bytes", std::to_string(bytes));
      TRACE_EVENT_SCOPE_END_NAMED(coEvent);

      // Return the IO buffer to the IO buffer pool.
      ioBufferPool->put(std::move(ioBuffer));
      resultCB(runId, Error::success(), std::move(ctx));
    }
  });
}

RunIdentifierTy
HabanaDeviceManager::runFunction(std::string functionName,
                                 std::unique_ptr<ExecutionContext> ctx,
                                 runtime::ResultCBTy resultCB) {
  DCHECK(resultCB != nullptr);

  RunIdentifierTy runId = runIdentifier_++;
  runPool_->submit([this, runId, functionName = std::move(functionName),
                    ctx = std::move(ctx),
                    resultCB = std::move(resultCB)]() mutable {
    runFunctionImpl(runId, std::move(functionName), std::move(ctx),
                    std::move(resultCB));
  });
  return runId;
}

Error HabanaDeviceManager::stop(bool block) {
  runPool_->stop(block);
  waitPool_->stop(block);
  return Error::success();
}

uint64_t HabanaDeviceManager::getMaximumMemory() const { return totalMemory_; }

uint64_t HabanaDeviceManager::getAvailableMemory() const { return freeMemory_; }

bool HabanaDeviceManager::isMemoryAvailable(uint64_t estimate) const {
  return estimate <= freeMemory_;
}

DeviceInfo HabanaDeviceManager::getDeviceInfo() const {
  DeviceInfo info = DeviceInfo();
  info.sramCapacity = 50 * 1024 * 1024;
  info.peakCompute = 0.45 * 1024 * 1024 * 1024 * 1024;
  info.peakDramBw = 30.0 * 1024 * 1024 * 1024;
  info.peakSramBw = 1024.0 * 1024 * 1024 * 1024;
  info.peakPCIeBw = 16.0 * 1024 * 1024 * 1024;
  return info;
}
