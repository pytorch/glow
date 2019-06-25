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
#ifndef GLOW_BACKENDS_CPU_CPUDEVICEMANAGER_H
#define GLOW_BACKENDS_CPU_CPUDEVICEMANAGER_H

#include "glow/Backends/QueueBackedDeviceManager.h"

namespace glow {
namespace runtime {

/// A class controlling a single CPU thread of execution driving the JIT
/// backend. Many CPUFunctions may be added, but only one inference is executed
/// at a time.
class CPUDeviceManager : public QueueBackedDeviceManager {
  /// Compiled function list by name.
  FunctionMapTy functions_;

  /// Maximum available memory on the device, for CPU devices fix to some
  /// constant.
  uint64_t maxMemoryBytes_{0};

  /// Amount of memory used by all models.
  uint64_t usedMemoryBytes_{0};

  /// Static memory cost of the CPU Function.
  /// This is very arbitrary for the CPU backend.
  const uint64_t functionCost_{1};

public:
  CPUDeviceManager(const DeviceConfig &config)
      : QueueBackedDeviceManager(config),
        maxMemoryBytes_(config_.getDeviceMemory(2000000000)) {}

  /// Returns the amount of memory in bytes available on the device when no
  /// models are loaded.
  uint64_t getMaximumMemory() const override;

  /// Returns the amount of memory in bytes currently availbe on the device.
  uint64_t getAvailableMemory() const override;

  /// Returns true if a function requiring the \p estimate size will fit on the
  /// device. This is not a promise as memory cost could vary due to alignment,
  /// etc.
  bool isMemoryAvailable(uint64_t estimate) const override;

protected:
  void addNetworkImpl(const Module *module, FunctionMapTy functions,
                      ReadyCBTy cb) override;
  void evictNetworkImpl(std::string functionName,
                        EvictFunctionCBTy evictCb) override;
  void runFunctionImpl(runtime::RunIdentifierTy id, std::string functionName,
                       std::unique_ptr<ExecutionContext> context,
                       ResultCBTy cb) override;
};

} // namespace runtime
} // namespace glow

#endif // GLOW_BACKENDS_CPU_CPUDEVICEMANAGER_H
