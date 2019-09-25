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

#include "glow/Backend/CompiledFunction.h"
#include "glow/ExecutionContext/ExecutionContext.h"
#include "glow/Graph/Graph.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "glow/Runtime/StatsExporter.h"
#include "glow/Support/Error.h"

#include <atomic>
#include <functional>
#include <map>
#include <string>

namespace glow {
namespace runtime {

/// Callback signalling success/failure of evicting a function from a Device.
using EvictFunctionCBTy = std::function<void(std::string functionName, Error)>;

/// Callback signalling success/failure of loading a Module onto a device.
using ReadyCBTy = std::function<void(const Module *, Error)>;

/// Map of Function name -> CompiledFunction, used when loading a network onto a
/// device.
using FunctionMapTy = std::map<std::string, CompiledFunction *>;

/// Interface managing a specific instance of a device.
class DeviceManager {
protected:
  /// Configuration object for the device.
  DeviceConfig config_;

  /// String for logging available memory for the device.
  const std::string availableMemoryKey_{"glow.device.available_memory.device"};

  /// String for logging used memory for the device.
  const std::string usedMemoryKey_{"glow.device.used_memory.device"};

  /// Maximum available memory on the device.
  std::atomic<uint64_t> maxMemoryBytes_{0};

  /// Amount of memory used by all models.
  std::atomic<uint64_t> usedMemoryBytes_{0};

  /// Helper method to export memory usage counters.
  void exportMemoryCounters() {
    Stats()->setCounter(availableMemoryKey_,
                        maxMemoryBytes_ - usedMemoryBytes_);
    Stats()->setCounter(usedMemoryKey_, usedMemoryBytes_);
  }

  /// Helper method to zero out memory counters, used when a device is freed.
  void zeroMemoryCounters() {
    Stats()->setCounter(availableMemoryKey_, 0);
    Stats()->setCounter(usedMemoryKey_, 0);
  }

public:
  DeviceManager(const DeviceConfig &config)
      : config_(config),
        availableMemoryKey_("glow.device.available_memory.device" +
                            std::to_string(config_.deviceID)),
        usedMemoryKey_("glow.device.used_memory.device" +
                       std::to_string(config_.deviceID)),
        maxMemoryBytes_(config_.getDeviceMemory(2000000000)) {}
  virtual ~DeviceManager() {}

  /// Create a device manager based on the device config \p config.
  static DeviceManager *createDeviceManager(const DeviceConfig &config);

  /// Query the system for the number of devices of a specified kind.
  static unsigned numDevices(llvm::StringRef backendName);

  /// Device discovery for a given backend kind. Returns a vector of configs for
  /// all found devices.
  static std::vector<std::unique_ptr<runtime::DeviceConfig>>
  generateDeviceConfigs(llvm::StringRef backendName);

  /// Initialize the device.
  virtual Error init() { return Error::success(); }

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
                            EvictFunctionCBTy evictCB = [](std::string, Error) {
                            }) = 0;

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
  virtual Error stop(bool block = true) { return Error::success(); };

  /// \returns the name of backend that powers this Device.
  llvm::StringRef getBackendName() { return config_.backendName; }

  /// \returns a string with \p name in parameters.
  llvm::StringRef getParamByName(llvm::StringRef name) const {
    auto it = config_.parameters.find(name);
    if (it != config_.parameters.end()) {
      return it->second;
    }
    return "";
  }

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

  /// \returns the DeviceInfo for this device containing peak limits for
  /// compute and bandwidths (used in partitioning).
  virtual DeviceInfo getDeviceInfo() const { return DeviceInfo(); }
};

} // namespace runtime
} // namespace glow

#endif // GLOW_BACKENDS_DEVICEMANAGER_H
