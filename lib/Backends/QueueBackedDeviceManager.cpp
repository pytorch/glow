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

#include "glow/Backends/QueueBackedDeviceManager.h"

using namespace glow;
using namespace glow::runtime;

QueueBackedDeviceManager::QueueBackedDeviceManager(BackendKind backend)
    : DeviceManager(backend), workThread_(1) {}

QueueBackedDeviceManager::~QueueBackedDeviceManager() {
  stop(true); // will join workThread_
}

void QueueBackedDeviceManager::init() {}

void QueueBackedDeviceManager::addNetwork(const Module *module,
                                          FunctionMapTy functions,
                                          ReadyCBTy callback) {
  workThread_.submit([this, module, f = std::move(functions),
                      c = std::move(callback)]() mutable {
    addNetworkImpl(module, std::move(f), std::move(c));
  });
}

void QueueBackedDeviceManager::evictNetwork(const Module *module) {
  workThread_.submit([this, module] { evictNetworkImpl(module); });
}

RunIdentifierTy
QueueBackedDeviceManager::runFunction(std::string functionName,
                                      std::unique_ptr<Context> ctx,
                                      ResultCBTy callback) {

  RunIdentifierTy id = nextIdentifier_++;
  workThread_.submit([this, id, functionName = std::move(functionName),
                      ctx = std::move(ctx),
                      callback = std::move(callback)]() mutable {
    runFunctionImpl(id, std::move(functionName), std::move(ctx),
                    std::move(callback));
  });
  return id;
}

void QueueBackedDeviceManager::stop(bool block) { workThread_.stop(block); }
