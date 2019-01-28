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
#include "CPUDeviceManager.h"

using namespace glow;
using namespace glow::runtime;

namespace glow {
DeviceManager *createCPUDeviceManager(llvm::StringRef name) {
  return new CPUDeviceManager(name);
}
} // namespace glow

uint64_t CPUDeviceManager::getMaximumMemory() { return maxMemoryBytes_; }

uint64_t CPUDeviceManager::getAvailableMemory() {
  return maxMemoryBytes_ - usedMemoryBytes_;
}

bool CPUDeviceManager::isMemoryAvailable(uint64_t estimate) {
  // No fuzz factor for the CPU device.
  return maxMemoryBytes_ >= (usedMemoryBytes_ + estimate);
}

void CPUDeviceManager::addNetworkImpl(const Module *module,
                                      FunctionMapTy functions,
                                      ReadyCBTy readyCB) {
  // First check for uniqueness of the function name.
  for (const auto &func : functions) {
    if (functions_.count(func.first) != 0) {
      readyCB(module, ResultCode::Failed);
      return;
    }
  }

  if (usedMemoryBytes_ + functionCost_ > maxMemoryBytes_) {
    readyCB(module, ResultCode::Failed);
    return;
  }

  // Add to the function name lookup map.
  for (const auto &func : functions) {
    func.second->getRuntimeBundle().collectConstants(module);
    functions_.emplace(func.first, func.second);
    usedMemoryBytes_ += functionCost_; // TODO:: static moduleSize
  }

  assert(usedMemoryBytes_ <= maxMemoryBytes_);

  // Fire the ready CB.
  readyCB(module, ResultCode::Ready);
}

void CPUDeviceManager::evictNetworkImpl(llvm::StringRef functionName) {
  if (functions_.erase(functionName)) {
    usedMemoryBytes_ -= functionCost_; // TODO: static moduleSize
  }
  assert(usedMemoryBytes_ >= 0);
}

void CPUDeviceManager::runFunctionImpl(RunIdentifierTy id, std::string function,
                                       std::unique_ptr<Context> ctx,
                                       ResultCBTy resultCB) {
  auto funcIt = functions_.find(function);
  if (funcIt == functions_.end()) {
    resultCB(id, ResultCode::Failed, std::move(ctx));
    return;
  }

  CompiledFunction *func = funcIt->second;

  // Run that function.
  func->execute(ctx.get());

  // Fire the resultCB.
  resultCB(id, ResultCode::Executed, std::move(ctx));
}
