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
#ifndef GLOW_BACKENDS_INTERPRETER_INTERPRETERDEVICEMANAGER_H
#define GLOW_BACKENDS_INTERPRETER_INTERPRETERDEVICEMANAGER_H

#include "glow/Backends/QueueBackedDeviceManager.h"

namespace glow {
namespace runtime {

/// A class controlling a single "Interpreter Device", a thread of execution in
/// the IR-Interpreter. Many InterpreterFunctions may be added, but only one
/// inference is executed at a time.
class InterpreterDeviceManager : public QueueBackedDeviceManager {
  /// Compiled function list by name.
  FunctionMapTy functions_;

  /// Maximum available memory on the device, for local devices fix to some
  /// constant.
  uint64_t maxMemoryBytes_{0};

  /// Amount of memory used by all models.
  uint64_t usedMemoryBytes_{0};

  /// Static memory cost of the InterpreterFunction.
  /// This is very arbitrary for the Interpreter backend.
  const uint64_t functionCost_{1};

public:
  InterpreterDeviceManager(std::unique_ptr<DeviceConfig> config = nullptr,
                           size_t maxMemory = 2000000000)
      : QueueBackedDeviceManager(BackendKind::Interpreter, std::move(config)),
        maxMemoryBytes_(maxMemory) {}

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
                        EvictFunctionCBTy evictCB) override;
  void runFunctionImpl(runtime::RunIdentifierTy id, std::string functionName,
                       std::unique_ptr<ExecutionContext> context,
                       ResultCBTy cb) override;
};

} // namespace runtime
} // namespace glow

#endif // GLOW_BACKENBDS_INTERPRETER_INTERPRETERDEVICEMANAGER_H
