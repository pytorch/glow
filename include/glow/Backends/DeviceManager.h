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
#ifndef GLOW_BACKENDS_DEVICEMANAGER_H
#define GLOW_BACKENDS_DEVICEMANAGER_H

#include "glow/Backend/Backend.h"
#include "glow/Backend/CompiledFunction.h"
#include "glow/ExecutionContext/ExecutionContext.h"
#include "glow/Graph/Graph.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "glow/Support/Error.h"

#include <functional>
#include <map>
#include <string>

namespace glow {
namespace runtime {

/// Callback signalling success/failure of evicting a function from a Device.
using EvictFunctionCBTy =
    std::function<void(std::string functionName, llvm::Error)>;

/// Callback signalling success/failure of loading a Module onto a device.
using ReadyCBTy = std::function<void(const Module *, llvm::Error)>;

/// Map of Function name -> CompiledFunction, used when loading a network onto a
/// device.
using FunctionMapTy = std::map<std::string, CompiledFunction *>;

/// Interface managing a specific instance of a device.
class DeviceManager {
protected:
  /// Configuration object for the device.
  DeviceConfig config_;

public:
  DeviceManager(const DeviceConfig &config) : config_(config) {}
  virtual ~DeviceManager() {}

  /// Create a device manager based on the device config \p config.
  static DeviceManager *createDeviceManager(const DeviceConfig &config);

  /// Initialize the device.
  virtual llvm::Error init() { return llvm::Error::success(); }

  /// Load the provided module into the device, readyCB will be called when
  /// ready to use.
  /// \p functions contains the list of functions to load, keyed by their name
  /// (as used in runFunction).
  virtual void addNetwork(const Module *module, FunctionMapTy functions,
                          ReadyCBTy readyCB) = 0;

  /// Remove (and delete) the provided function, freeing
  /// up space on the device. \p evictCB will be called when the operation
  /// is completed or attempted and failed.
  virtual void evictNetwork(std::string functionName,
                            EvictFunctionCBTy evictCB = [](std::string,
                                                           llvm::Error) {}) = 0;

  /// Execute the named Function in an already provided network on the device.
  /// functionName must match the name of a function already added.
  /// The ExecutionContext's PlaceholderBindings should have all Placeholders
  /// allocated. resultCB will be called with the ExecutionContext containing
  /// output tensors filled, and any generated TraceEvents.
  virtual runtime::RunIdentifierTy
  runFunction(std::string functionName,
              std::unique_ptr<ExecutionContext> context,
              runtime::ResultCBTy resultCB) = 0;

  /// Stops execution and shuts down the Device.
  virtual llvm::Error stop(bool block = true) {
    return llvm::Error::success();
  };

  /// \returns the name of backend that powers this Device.
  llvm::StringRef getBackendName() { return config_.backendName; }

  /// \returns the maximum memory (in bytes) available on the device.
  virtual uint64_t getMaximumMemory() const = 0;

  /// \returns the currently available memory (in bytes) available on the
  /// device, for provisioning new networks.
  virtual uint64_t getAvailableMemory() const = 0;

  /// \returns true if we expect a Module with the estimated constant size will
  /// fit on the device.
  virtual bool isMemoryAvailable(uint64_t estimate) const = 0;

  /// \returns the DeviceConfig which initialized this device.
  const DeviceConfig &getDeviceConfig() { return config_; }
};

} // namespace runtime
} // namespace glow

#endif // GLOW_BACKENDS_DEVICEMANAGER_H
