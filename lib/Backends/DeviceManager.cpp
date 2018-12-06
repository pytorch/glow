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

#include "glow/Backends/DeviceManager.h"

using namespace glow;

DeviceManager::DeviceManager(std::unique_ptr<Backend> backend)
    : backend_(std::move(backend)), workThread_(1) {}

DeviceManager::~DeviceManager() {
  stop(true); // will join workThread_
}

void DeviceManager::init() {}

void DeviceManager::addNetwork(DeviceNetworkID moduleID,
                               std::unique_ptr<Module> module,
                               ReadyCB callback) {
  workThread_.submit([this, moduleID, m = std::move(module),
                      c = std::move(callback)]() mutable {
    addNetworkImpl(moduleID, std::move(m), std::move(c));
  });
}

void DeviceManager::evictNetwork(DeviceNetworkID moduleID) {
  workThread_.submit([this, moduleID] { evictNetworkImpl(moduleID); });
}

void DeviceManager::runFunction(DeviceNetworkID moduleID,
                                llvm::StringRef functionName,
                                std::unique_ptr<Context> ctx,
                                ResultCB callback) {
  workThread_.submit([this, moduleID, functionName = std::move(functionName),
                      ctx = std::move(ctx),
                      callback = std::move(callback)]() mutable {
    runFunctionImpl(moduleID, std::move(functionName), std::move(ctx),
                    std::move(callback));
  });
}

void DeviceManager::stop(bool block) { workThread_.stop(block); }
