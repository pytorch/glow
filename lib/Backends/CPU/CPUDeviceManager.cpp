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
#include "CPUFunction.h"

#include "llvm/Support/raw_ostream.h"

using namespace glow;
using namespace glow::runtime;

namespace glow {
namespace runtime {
DeviceManager *createCPUDeviceManager(llvm::StringRef name) {
  return new CPUDeviceManager(name);
}
} // namespace runtime
} // namespace glow

uint64_t CPUDeviceManager::getMaximumMemory() const { return maxMemoryBytes_; }

uint64_t CPUDeviceManager::getAvailableMemory() const {
  return maxMemoryBytes_ - usedMemoryBytes_;
}

bool CPUDeviceManager::isMemoryAvailable(uint64_t estimate) const {
  // No fuzz factor for the CPU device.
  return maxMemoryBytes_ >= (usedMemoryBytes_ + estimate);
}

void CPUDeviceManager::addNetworkImpl(const Module *module,
                                      FunctionMapTy functions,
                                      ReadyCBTy readyCB) {
  // First check for uniqueness of the function name.
  for (const auto &func : functions) {
    if (functions_.count(func.first) != 0) {
      llvm::errs() << "Failed to add network: already have a function called "
                   << func.first << ".\n";
      readyCB(module, ResultCode::Failed);
      return;
    }

    if (func.second->getCompileBackendKind() != BackendKind::CPU) {
      llvm::errs() << "Failed to add network: function " << func.first
                   << " is not a CPUFunction.\n";
      readyCB(module, ResultCode::Failed);
    }
  }

  if (usedMemoryBytes_ + functionCost_ > maxMemoryBytes_) {
    llvm::errs() << "Failed to add network: not enough memory.\n";
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

void CPUDeviceManager::evictNetworkImpl(std::string functionName) {
  if (functions_.erase(functionName)) {
    usedMemoryBytes_ -= functionCost_; // TODO: static moduleSize
  }
}

void CPUDeviceManager::runFunctionImpl(RunIdentifierTy id, std::string function,
                                       std::unique_ptr<Context> ctx,
                                       ResultCBTy resultCB) {
  auto funcIt = functions_.find(function);
  if (funcIt == functions_.end()) {
    llvm::errs() << "Failed to run function: name " << function
                 << " not found.\n";
    resultCB(id, ResultCode::Failed, std::move(ctx));
    return;
  }

  CompiledFunction *func = funcIt->second;

  // Run that function.
  func->execute(ctx.get());

  // Fire the resultCB.
  resultCB(id, ResultCode::Executed, std::move(ctx));
}
