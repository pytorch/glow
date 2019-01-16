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

#include "glow/Backends/Backend.h"
#include "glow/Backends/CompiledFunction.h"
#include "glow/Graph/Context.h"
#include "glow/Graph/Graph.h"
#include "glow/Runtime/RuntimeTypes.h"

#include <functional>
#include <map>
#include <string>

namespace glow {

/// Callback signalling success/failure of loading a Module onto a device.
using ReadyCBTy = std::function<void(const Module *, runtime::ResultCode)>;
/// Callback signalling the result of running a function.
using ResultCBTy = std::function<void(
    runtime::RunIdentifierTy, runtime::ResultCode, std::unique_ptr<Context>)>;
/// Map of Function name -> CompiledFunction, used when loading a network onto a
/// device.
using FunctionMapTy = std::map<std::string, CompiledFunction *>;

/// Interface managing a specific instance of a device.
class DeviceManager {
protected:
  /// Type of Backend for this Device.
  BackendKind backend_;

public:
  DeviceManager(BackendKind backend) : backend_(backend) {}
  virtual ~DeviceManager() {}

  /// Initialize the device.
  virtual void init() {}

  /// Load the provided module into the device, readyCB will be called when
  /// ready to use.
  /// \p functions contains the list of functions to load, keyed by their name
  /// (as used in runFunction).
  virtual void addNetwork(const Module *module, FunctionMapTy functions,
                          ReadyCBTy readyCB) = 0;

  /// Remove (and delete) the provided network and all it's functions, freeing
  /// up space on the device.
  virtual void evictNetwork(const Module *module) = 0;

  /// Execute the named Function in an already provided network on the device.
  /// functionName must match the name of a function already added.
  /// Context should have all Placeholders allocated. resultCB will be called
  /// with the Context results filled.
  virtual runtime::RunIdentifierTy runFunction(std::string functionName,
                                               std::unique_ptr<Context> ctx,
                                               ResultCBTy resultCB) = 0;

  /// Stops execution and shuts down the Device.
  virtual void stop(bool block = true) {}

  /// \returns the type of Backend that powers this Device.
  BackendKind getBackendKind() { return backend_; }

  /// \returns the maximum memory (in bytes) available on the device.
  virtual uint64_t getMaximumMemory() = 0;

  /// \returns the currently available memory (in bytes) available on the
  /// device, for provisioning new networks.
  virtual uint64_t getAvailableMemory() = 0;

  /// \returns true if we expect a Module with the estimated constant size will
  /// fit on the device.
  virtual bool isMemoryAvailable(uint64_t estimate) = 0;
};

} // namespace glow

#endif // GLOW_BACKENDS_DEVICEMANAGER_H
