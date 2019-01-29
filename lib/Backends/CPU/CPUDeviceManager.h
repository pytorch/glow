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
#ifndef GLOW_BACKENDS_CPUDEVICEMANAGER_H
#define GLOW_BACKENDS_CPUDEVICEMANAGER_H

#include "glow/Backends/QueueBackedDeviceManager.h"

namespace glow {

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
  CPUDeviceManager(llvm::StringRef name = "unnamed", size_t maxMemory = 1000)
      : QueueBackedDeviceManager(BackendKind::CPU, name),
        maxMemoryBytes_(maxMemory) {}

  uint64_t getMaximumMemory() override;
  uint64_t getAvailableMemory() override;
  bool isMemoryAvailable(uint64_t estimate) override;

protected:
  void addNetworkImpl(const Module *module, FunctionMapTy functions,
                      ReadyCBTy cb) override;
  void evictNetworkImpl(llvm::StringRef functionName) override;
  void runFunctionImpl(runtime::RunIdentifierTy id, std::string functionName,
                       std::unique_ptr<Context> ctx, ResultCBTy cb) override;
};

} // namespace glow

#endif // GLOW_BACKENBDS_CPUDEVICEMANAGER_H
