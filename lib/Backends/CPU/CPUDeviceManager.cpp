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

uint64_t CPUDeviceManager::getMaximumMemory() { return maxMemoryBytes; }

uint64_t CPUDeviceManager::getAvailableMemory() {
  return maxMemoryBytes - usedMemoryBytes;
}

bool CPUDeviceManager::isMemoryAvailable(uint64_t estimate) {
  // No fuzz factor for the CPU device.
  return maxMemoryBytes >= (usedMemoryBytes + estimate);
}

void CPUDeviceManager::addNetworkImpl(const Module *module,
                                      FunctionMapTy functions,
                                      ReadyCBTy readyCB) {
  auto modIt = modules_.find(module);
  if (modIt != modules_.end()) {
    // Already have a module with this ID.
    // TODO: should we replace it?
    readyCB(module, Failed);
    return;
  }

  // TODO: we should update usedMemory but we don't currently have a nice way
  // to determine the memory used by the module. I'll come back to this, but for
  // now we'll guess (badly).
  size_t moduleSize = 200 * 1024 * 1024;

  if (usedMemoryBytes + moduleSize > maxMemoryBytes) {
    readyCB(module, Failed);
    return;
  }

  // Add to the function name lookup map.
  for (const auto &func : functions) {
    // TODO: collect constants here when available.
    functions_.emplace(func.first, func.second);
  }

  modules_.emplace_hint(modIt, module, std::move(functions));
  usedMemoryBytes += moduleSize;

  // Fire the ready CB.
  readyCB(module, Ready);
}

void CPUDeviceManager::evictNetworkImpl(const Module *module) {
  auto modIt = modules_.find(module);
  if (modIt == modules_.end()) {
    // Nothing to do.
    return;
  }

  FunctionMapTy moduleFuncs = std::move(modIt->second);
  for (const auto &func : moduleFuncs) {
    functions_.erase(func.first);
  }

  modules_.erase(modIt);
  usedMemoryBytes -= 200 * 1024 * 1024; // TODO: static moduleSize
  assert(usedMemoryBytes >= 0);
}

void CPUDeviceManager::runFunctionImpl(RunIdentifierTy id, std::string function,
                                       std::unique_ptr<Context> ctx,
                                       ResultCBTy resultCB) {
  auto funcIt = functions_.find(function);
  if (funcIt == functions_.end()) {
    resultCB(id, Failed, std::move(ctx));
    return;
  }

  CompiledFunction *func = funcIt->second;

  // Run that function.
  func->setupRuns();
  func->beforeRun(*ctx);
  func->execute();
  func->afterRun(*ctx);
  func->tearDownRuns();

  // Fire the resultCB.
  resultCB(id, Executed, std::move(ctx));
}
