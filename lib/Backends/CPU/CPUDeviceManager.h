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

#include "glow/Backends/Backend.h"
#include "glow/Backends/CompiledFunction.h"
#include "glow/Backends/DeviceManager.h"

namespace glow {

class CPUDeviceManager : public DeviceManager {
  /// Loaded module list.
  std::unordered_map<DeviceNetworkID, std::unique_ptr<Module>> modules_;
  /// Compiled function list.
  std::unordered_map<Function *, std::unique_ptr<CompiledFunction>> functions_;

  /// Maximum available memory on the device, for CPU devices fix to some
  /// constant.
  uint64_t maxMemoryBytes{0};
  /// Amount of memory used by all models.
  uint64_t usedMemoryBytes{0};

public:
  CPUDeviceManager(size_t MBsPerCore = 16000)
      : DeviceManager(
            std::unique_ptr<Backend>(createBackend(BackendKind::CPU))),
        maxMemoryBytes(MBsPerCore * 1024 * 1024) {}

  uint64_t getMaximumMemory() override;
  uint64_t getAvailableMemory() override;
  bool isMemoryAvailable(uint64_t estimate) override;

protected:
  void optimizeFunction(CompilationMode mode, Function *F);

  void addNetworkImpl(DeviceNetworkID id, std::unique_ptr<Module> module,
                      ReadyCB cb) override;
  void evictNetworkImpl(DeviceNetworkID id) override;
  void runFunctionImpl(DeviceNetworkID id, llvm::StringRef function,
                       std::unique_ptr<Context> ctx, ResultCB cb) override;
};

} // namespace glow

#endif // GLOW_BACKENBDS_CPUDEVICEMANAGER_H
