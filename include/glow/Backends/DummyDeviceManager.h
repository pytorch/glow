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
#ifndef GLOW_BACKENDS_INLINEDEVICEMANAGER_H
#define GLOW_BACKENDS_INLINEDEVICEMANAGER_H

#include "glow/Backends/DeviceManager.h"
#include "glow/Runtime/RuntimeTypes.h"

#include <atomic>

namespace glow {
namespace runtime {

/// The DummyDeviceManager is a simple DeviceManager implementation that
/// provides execution for backends that are in development. It is explicitly
/// not threadsafe and runs provided CompiledFunction's in the caller thread.
class DummyDeviceManager : public DeviceManager {
  /// Compiled function list by name.
  FunctionMapTy functions_;

public:
  DummyDeviceManager(const DeviceConfig &config) : DeviceManager(config) {}

  /// The DummyDeviceManager is a simple wrapper for testing, if you need
  /// memory guards you should implement a DeviceManager for your device.
  uint64_t getMaximumMemory() const override {
    return std::numeric_limits<uint64_t>::max();
  }
  uint64_t getAvailableMemory() const override {
    return std::numeric_limits<uint64_t>::max();
  }
  bool isMemoryAvailable(uint64_t) const override { return true; }

  /// Load the provided module into the device, readyCB will be called when
  /// ready to use.
  /// \p functions contains the list of functions to load, keyed by their name
  /// (as used in runFunction).
  void addNetwork(const Module *module, FunctionMapTy functions,
                  ReadyCBTy callback) override {
    for (const auto &func : functions) {
      if (functions_.count(func.first) != 0) {
        callback(
            module,
            MAKE_ERR(
                GlowErr::ErrorCode::RUNTIME_NET_NOT_FOUND,
                llvm::formatv("Function {0} not found", func.first).str()));
        return;
      }
    }

    for (const auto &func : functions) {
      if (func.second->getRuntimeBundle().getConstants() == nullptr) {
        func.second->getRuntimeBundle().collectConstants(module);
      }
      functions_.emplace(func.first, func.second);
    }

    // Fire the ready CB.
    callback(module, llvm::Error::success());
  }

  /// Remove (and delete) the provided function, freeing
  /// up space on the device.
  void evictNetwork(std::string functionName,
                    EvictFunctionCBTy evictCB) override {
    functions_.erase(functionName);
    evictCB(functionName, llvm::Error::success());
  }

  /// Execute the named Function in an already provided network on the device.
  /// functionName must match the name of a function already added.
  /// The ExecutionContext's PlaceholderBindings should have all Placeholders
  /// allocated. resultCB will be called with the ExecutionContext containing
  /// output tensors filled, and any generated TraceEvents.
  RunIdentifierTy runFunction(std::string functionName,
                              std::unique_ptr<ExecutionContext> context,
                              ResultCBTy callback) override {
    auto funcIt = functions_.find(functionName);
    if (funcIt == functions_.end()) {
      callback(
          0,
          MAKE_ERR(GlowErr::ErrorCode::RUNTIME_NET_NOT_FOUND,
                   llvm::formatv("Function {0} not found", functionName).str()),
          std::move(context));
      return 0;
    }

    CompiledFunction *func = funcIt->second;

    auto executeErr = func->execute(context.get());

    // Fire the resultCB.
    callback(0, std::move(executeErr), std::move(context));

    return 0;
  }
};

} // namespace runtime
} // namespace glow

#endif // GLOW_BACKENDS_INLINEDEVICEMANAGER_H
