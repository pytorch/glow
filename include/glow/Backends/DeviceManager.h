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
#include "glow/Graph/Context.h"
#include "glow/Graph/Graph.h"
#include "glow/Support/ThreadPool.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace glow {

using DeviceNetworkID = size_t;

enum ResultCode { READY, EXECUTED, FAILED, CANCELLED };

using ReadyCB = std::function<void(DeviceNetworkID, ResultCode)>;
using ResultCB = std::function<void(ResultCode, std::unique_ptr<Context>)>;

class DeviceManager {
protected:
  /// Lookup for Modules.
  std::unordered_map<DeviceNetworkID, std::unique_ptr<Module>> modules_;
  /// The network execution backend.
  std::unique_ptr<Backend> backend_;
  /// Thread which interfaces with the device.
  ThreadPool workThread_;
  /// Next available network ID.
  std::atomic<DeviceNetworkID> nextNetworkID_{1};

public:
  DeviceManager(std::unique_ptr<Backend> backend);
  virtual ~DeviceManager();

  /// Initialize the device.
  void init();

  /// Load the provided module into the device, readyCB will be called when
  /// ready to use
  void addNetwork(DeviceNetworkID networkId, std::unique_ptr<Module> module,
                  ReadyCB readyCB);

  /// Remove (and delete) the provided network, freeing up space on the device.
  void evictNetwork(DeviceNetworkID networkId);

  /// Execute the named Function in an already provided network on the device.
  /// functionName must match the name of a function in the Module.
  /// Context should have all Placeholders allocated. resultCB will be called
  /// with the Context results filled.
  void runFunction(DeviceNetworkID moduleID, llvm::StringRef functionName,
                   std::unique_ptr<Context> ctx, ResultCB resultCB);

  /// Stops execution and shuts down the Device.
  void stop(bool block = true);

  /// \returns the maximum memory (in bytes) available on the device.
  virtual uint64_t getMaximumMemory() = 0;

  /// \returns the currently available memory (in MB) available on the device,
  /// for provisioning new networks.
  virtual uint64_t getAvailableMemory() = 0;

  /// \returns true if we expect a Module with the estimated constant size will
  /// fit on the device.
  virtual bool isMemoryAvailable(uint64_t estimate) = 0;

  /// Returns an available DeviceNetworkID to use for a new network.
  DeviceNetworkID getNextDeviceNetworkID() { return nextNetworkID_++; }

protected:
  /// Operator handling methods to be implemented in subclasses (i.e. per Device
  /// type)

  /// Load and compile the Module
  virtual void addNetworkImpl(DeviceNetworkID id,
                              std::unique_ptr<Module> module, ReadyCB cb) = 0;

  /// Remove the module and reclaim it's memory
  virtual void evictNetworkImpl(DeviceNetworkID id) = 0;

  /// Execute provided Function
  virtual void runFunctionImpl(DeviceNetworkID id, llvm::StringRef function,
                               std::unique_ptr<Context> ctx, ResultCB cb) = 0;
};

} // namespace glow

#endif // GLOW_BACKENDS_DEVICEMANAGER_H
