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
#ifndef GLOW_BACKENDS_DEVICEMANAGER_H
#define GLOW_BACKENDS_DEVICEMANAGER_H

#include "glow/Backend/CompiledFunction.h"
#include "glow/Base/DeviceTensorTransferManager.h"
#include "glow/ExecutionContext/ExecutionContext.h"
#include "glow/Graph/Graph.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "glow/Runtime/StatsExporter.h"
#include "glow/Support/Error.h"

#include <atomic>
#include <functional>
#include <map>
#include <mutex>
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
class DeviceManager : public DeviceTensorTransferManager {
protected:
  /// Configuration object for the device.
  DeviceConfig config_;

  /// Lock to protect allocations_ from being accessed concurrently. This can
  /// occur when multiple networks are added concurrently.
  std::mutex bufferLock_;

  /// String for logging available memory for the device.
  const std::string availableMemoryKey_{"glow.device.available_memory.device"};

  /// String for logging used memory for the device.
  const std::string usedMemoryKey_{"glow.device.used_memory.device"};

  /// Maximum available memory on the device.
  std::atomic<uint64_t> maxMemoryBytes_{0};

  /// Amount of memory used by all models.
  std::atomic<uint64_t> usedMemoryBytes_{0};

  /// Keeps the stats exporter registry object alive till destructor.
  std::shared_ptr<StatsExporterRegistry> statsExporterRegistry_;

  /// Set of all buffer allocations, these should all be freed when the device
  /// manager is destroyed.
  std::set<void *> allocations_;

  /// Helper method to export memory usage counters.
  void exportMemoryCounters() {
    statsExporterRegistry_->setCounter(availableMemoryKey_,
                                       maxMemoryBytes_ - usedMemoryBytes_);
    statsExporterRegistry_->setCounter(usedMemoryKey_, usedMemoryBytes_);
  }

  /// Helper method to zero out memory counters, used when a device is freed.
  void zeroMemoryCounters() {
    statsExporterRegistry_->setCounter(availableMemoryKey_, 0);
    statsExporterRegistry_->setCounter(usedMemoryKey_, 0);
  }

public:
  DeviceManager(const DeviceConfig &config)
      : config_(config),
        availableMemoryKey_("glow.device.available_memory.device" +
                            std::to_string(config_.deviceID)),
        usedMemoryKey_("glow.device.used_memory.device" +
                       std::to_string(config_.deviceID)),
        maxMemoryBytes_(config_.getDeviceMemory(2000000000)),
        statsExporterRegistry_(StatsExporterRegistry::Stats()) {}

  virtual ~DeviceManager() {
    // Free all allocated buffers.
    for (auto &buffer : allocations_) {
      alignedFree(buffer);
    }
  }

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

  /// \returns a pointer to a buffer of size \p size allocated on the host, that
  /// satistfies any requirements for pinning/alignment for transferring to/from
  /// the device. The lifetime of this buffer is managed by the device manager.
  virtual void *allocateDeviceIOBuffer(dim_t size) {
    std::lock_guard<std::mutex> lock(bufferLock_);
    void *buffer = alignedAlloc(size, TensorAlignment);
    allocations_.insert(buffer);
    return buffer;
  };

  /// Free all allocated buffers associated with /p PH.
  virtual void freeAllocatedDeviceIOBuffer(void *buffer) {
    std::lock_guard<std::mutex> lock(bufferLock_);
    auto it = allocations_.find(buffer);
    if (it != allocations_.end()) {
      alignedFree(buffer);
      allocations_.erase(it);
    }
  }

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

  /// Copies the contents of Tensor \p T to the device resource allocated to
  /// Placeholder \p PH. once finished calls \p resultCB with the result of the
  /// operation.
  virtual void
  transferStaticPlaceholderToDevice(Placeholder *PH, Tensor *T,
                                    std::function<void(Error)> resultCB) {
    resultCB(MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_ERROR,
                      "Unsupported feature, cannot copy Placeholder."));
  };

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

  /// Copies the contents of \p tensor from the host to the \p location
  /// address on this device. Updates the tensor residency info.
  virtual void
  transferToDevice(Tensor &tensor, void *locationContext,
                   std::function<void(Error)> resultCB = GLOW_DRT_DEFAULT_CB) {
    DCHECK("Not Implemented");
    resultCB(MAKE_ERR(ErrorValue::ErrorCode::DEVICE_FEATURE_NOT_SUPPORTED,
                      "Direct transfer not supported on this device"));
  }

  /// Copies the device buffer associated with \p tensor to the host.
  /// The tensor must be resident on this device. If \p release is true,
  /// frees the device memory. Updates the tensor residency info.
  virtual void transferFromDevice(
      Tensor &tensor, bool release = true,
      std::function<void(Error)> resultCB = GLOW_DRT_DEFAULT_CB) {
    DCHECK("Not Implemented");
    resultCB(MAKE_ERR(ErrorValue::ErrorCode::DEVICE_FEATURE_NOT_SUPPORTED,
                      "Direct transfer not supported on this device"));
  }

  /// Releases the device buffer associated with \p tensor.
  virtual bool releaseDeviceTensor(void *locationContext) {
    DCHECK("Not Implemented");
    return false;
  }

  /// Starts device tracing \returns Error if fails.
  virtual Error startDeviceTrace(TraceContext *traceContext) {
    return Error::success();
  }
  /// Stops device tracing \returns Error if fails.
  virtual Error stopDeviceTrace(TraceContext *traceContext) {
    return Error::success();
  }
};

} // namespace runtime
} // namespace glow

#endif // GLOW_BACKENDS_DEVICEMANAGER_H
