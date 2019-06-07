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
#ifndef GLOW_BACKENDS_HABANA_HABANADEVICEMANAGER_H
#define GLOW_BACKENDS_HABANA_HABANADEVICEMANAGER_H

#include "Habana.h"
#include "glow/Backends/DeviceManager.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "glow/Support/ThreadPool.h"

#include "synapse.h"

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>

namespace glow {
namespace runtime {

/// This class implements the DeviceManager interface for
/// Habana devices.
class HabanaDeviceManager : public DeviceManager {
  using DeviceId = uint32_t;
  using TopologyId = uint64_t;

  static constexpr auto INVALID_DEVICE = std::numeric_limits<DeviceId>::max();
  static constexpr auto INVALID_TOPOLOGY =
      std::numeric_limits<TopologyId>::max();

  /// The ID of the device managed by this instance.
  DeviceId deviceId_{INVALID_DEVICE};
  /// The available memory on the device.
  uint64_t freeMemory_{0};
  /// The total memory on the device.
  uint64_t totalMemory_{0};
  /// Mutex for accessing instance members (activeTopologyId_, functions_,
  /// activeEnqueues_).
  std::mutex instanceMtx_;

  /// Thread pool for executing functions.
  std::unique_ptr<ThreadPool> runPool_;
  /// Thread pool for waiting on the results of executing functions.
  std::unique_ptr<ThreadPool> waitPool_;
  /// The default number of workers in run pool (overridable).
  constexpr static unsigned kNumRunners = 1;
  /// The default number of workers in wait pool (overridable).
  constexpr static unsigned kNumWaiters = 1;
  /// The number of workers in run pool.
  unsigned numRunners_{kNumRunners};
  /// The number of workers in wait pool.
  unsigned numWaiters_{kNumWaiters};

  /// Track active topology on this device.
  TopologyId activeTopo_{INVALID_TOPOLOGY};
  /// Number of requests in flight.  Used to block topo switching.
  unsigned inflightRequests_{0};
  /// Condition variable for signaling queue drain.
  std::condition_variable cv_;

  /// This struct wraps a topology ID with its corresponding HabanaFunction so
  /// that only one map is needed to keep track of both.
  struct HabanaFunctionMeta {
    /// The topology ID of the function. This is returned by the Synapse API
    /// after loading a recipe.
    uint64_t topologyId;
    /// The HabanaFunction corresponding to topologyId. This is needed in order
    /// to call HabanaFunction::executeOnDevice after loading and activating a
    /// topology.
    HabanaFunction *function;
    /// A pool of IO buffers to use during execution.
    std::unique_ptr<HabanaIOBufferPool> ioBufferPool;
  };

  /// A map from function name -> HabanaFunctionMeta. Its keys are the
  /// names of all functions added to the device manager.
  std::unordered_map<std::string, HabanaFunctionMeta> functions_;

  /// The total number of active Habana devices among all HabanaDeviceManager
  /// instances. This is used to determine which instance should
  /// initialize/destroy the Synapse API in the constructor/destructor.
  static unsigned numActiveDevices_;
  /// Mutex for guarding access to Synapse API.
  static std::mutex synapseMtx_;
  /// Identifier for next run.
  static std::atomic<RunIdentifierTy> runIdentifier_;

  /// Helper method for running a function. runFunction submits a lambda that
  /// calls this to runPool_ so that it can return immediately without taking up
  /// the calling thread for too long.
  void runFunctionImpl(RunIdentifierTy runId, std::string functionName,
                       std::unique_ptr<ExecutionContext> ctx,
                       runtime::ResultCBTy resultCB);

  /// Update the totalMemory_ and freeMemory_ counts for the device based once
  /// per-function memory estimates. This function is not thread safe and should
  /// only be invoked while holding synapseLock.
  llvm::Error updateMemoryUsage();

public:
  /// Constructor.
  HabanaDeviceManager(const DeviceConfig &config,
                      unsigned numRunners = kNumRunners,
                      unsigned numWaiters = kNumWaiters);

  /// Destructor.
  virtual ~HabanaDeviceManager();

  /// See DeviceManager and QueueBackedDeviceManager for the documentation of
  /// the interface below.
  llvm::Error init() override;

  void addNetwork(const Module *module, FunctionMapTy functions,
                  ReadyCBTy readyCB) override;

  void evictNetwork(std::string functionName,
                    EvictFunctionCBTy evictCB) override;

  RunIdentifierTy runFunction(std::string functionName,
                              std::unique_ptr<ExecutionContext> ctx,
                              runtime::ResultCBTy resultCB) override;

  llvm::Error stop(bool block) override;

  uint64_t getMaximumMemory() const override;
  uint64_t getAvailableMemory() const override;
  bool isMemoryAvailable(uint64_t estimate) const override;
};

} // namespace runtime
} // namespace glow

#endif // GLOW_BACKENDS_HABANADEVICEMANAGER_H
