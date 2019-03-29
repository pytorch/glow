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
#include "glow/Backends/QueueBackedDeviceManager.h"

#include "synapse.h"

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>

namespace glow {
namespace runtime {

/// This class implements the DeviceManager interface for
/// Habana devices.
class HabanaDeviceManager : public QueueBackedDeviceManager {
  /// The ID of the device managed by this instance.
  uint32_t deviceId_{0};
  /// The available memory on the device.
  uint64_t freeMemory_{0};
  /// The total memory on the device.
  uint64_t totalMemory_{0};

  /// A map from function name -> HabanaFunctionAndTopology. Its keys are the
  /// names of all functions added to the device manager.
  std::unordered_map<std::string, HabanaFunctionMeta> functions_;
  /// The total number of active Habana devices among all HabanaDeviceManager
  /// instances. This is used to determine which instance should
  /// initialize/destroy the Synapse API in the constructor/destructor.
  static unsigned numActiveDevices_;
  /// Mutex for guarding access to numActiveDevices_.
  static std::mutex mtx_;

public:
  /// Constructor.
  HabanaDeviceManager(std::unique_ptr<DeviceConfig> config = nullptr);

  /// Destructor.
  virtual ~HabanaDeviceManager();

  /// See DeviceManager and QueueBackedDeviceManager for the documentation of
  /// the interface below.
  llvm::Error init() override;

  void addNetworkImpl(const Module *module, FunctionMapTy functions,
                      ReadyCBTy readyCB) override;

  void evictNetworkImpl(std::string functionName,
                        EvictFunctionCBTy evictCB) override;

  void runFunctionImpl(RunIdentifierTy runId, std::string functionName,
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
