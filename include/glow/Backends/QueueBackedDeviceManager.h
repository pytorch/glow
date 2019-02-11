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
#ifndef GLOW_BACKENDS_QUEUEBACKEDDEVICEMANAGER_H
#define GLOW_BACKENDS_QUEUEBACKEDDEVICEMANAGER_H

#include "glow/Backends/DeviceManager.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "glow/Support/ThreadPool.h"

#include <atomic>

namespace glow {
namespace runtime {
class QueueBackedDeviceManager : public DeviceManager {
protected:
  /// Thread which interfaces with the device.
  ThreadPool workThread_;

  /// Identifier for next run.
  std::atomic<RunIdentifierTy> nextIdentifier_{1};

public:
  QueueBackedDeviceManager(BackendKind backend, llvm::StringRef name)
      : DeviceManager(backend, name), workThread_(1) {}

  virtual ~QueueBackedDeviceManager() {
    stop(true); // will join workThread_
  }

  /// Initialize the device.
  void init() override {}

  /// Load the provided module into the device, readyCB will be called when
  /// ready to use
  void addNetwork(const Module *module, FunctionMapTy functions,
                  ReadyCBTy callback) override {
    workThread_.submit([this, module, f = std::move(functions),
                        c = std::move(callback)]() mutable {
      addNetworkImpl(module, std::move(f), std::move(c));
    });
  }

  /// Remove (and delete) the provided network and all it's functions, freeing
  /// up space on the device.
  void evictNetwork(std::string functionName) override {
    workThread_.submit(
        [this, functionName] { evictNetworkImpl(functionName); });
  }

  /// Execute the named Function in an already provided network on the device.
  /// functionName must match the name of a function already added.
  /// Context should have all Placeholders allocated. resultCB will be called
  /// with the Context results filled.
  RunIdentifierTy runFunction(std::string functionName,
                              std::unique_ptr<Context> ctx,
                              ResultCBTy callback) override {

    RunIdentifierTy id = nextIdentifier_++;
    workThread_.submit([this, id, functionName = std::move(functionName),
                        ctx = std::move(ctx),
                        callback = std::move(callback)]() mutable {
      runFunctionImpl(id, std::move(functionName), std::move(ctx),
                      std::move(callback));
    });
    return id;
  }

  /// Stops execution and shuts down the Device.
  void stop(bool block = true) override { workThread_.stop(block); }

protected:
  /// Operator handling methods to be implemented in subclasses (i.e. per Device
  /// type)

  /// Load and compile the Module
  virtual void addNetworkImpl(const Module *, FunctionMapTy, ReadyCBTy) = 0;

  /// Remove the module and reclaim it's memory
  virtual void evictNetworkImpl(std::string functionName) = 0;

  /// Execute provided Function.
  virtual void runFunctionImpl(RunIdentifierTy, std::string,
                               std::unique_ptr<Context>, ResultCBTy) = 0;
};
} // namespace runtime
} // namespace glow

#endif // GLOW_BACKENDS_QUEUEBACKEDDEVICEMANAGER_H
