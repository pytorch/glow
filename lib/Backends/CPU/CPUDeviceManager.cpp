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

uint64_t CPUDeviceManager::getMaximumMemory() { return maxMemoryBytes; }

uint64_t CPUDeviceManager::getAvailableMemory() {
  return maxMemoryBytes - usedMemoryBytes;
}

bool CPUDeviceManager::isMemoryAvailable(uint64_t estimate) {
  // No fuzz factor for the CPU device.
  return maxMemoryBytes >= (usedMemoryBytes + estimate);
}

// Lifted from ExcutionEngine.cpp
// TODO this could be a generic helper function somewhere?
void CPUDeviceManager::optimizeFunction(CompilationMode mode, Function *F) {
  // Verify the function pre-optimization/lowering.
  assert(F->verify() && "Function must be valid");

  // Optimize the graph.
  ::glow::optimize(F, mode);

  // Allow the backend to transform the graph prior to lowering.
  if (backend_->transformPreLowering(F, mode)) {
    // Optimize the graph again after the backend transformation.
    // In particular, DCE is very likely to be useful.
    ::glow::optimize(F, mode);
  }

  // Lower the graph into a sequence of low-level linear algebra operations.
  ::glow::lower(F, *backend_);

  // Optimize the graph again.
  ::glow::optimize(F, mode);

  // Allow the backend to transform the graph after lowering.
  if (backend_->transformPostLowering(F, mode)) {
    // Optimize the graph again after the backend transformation.
    // In particular, DCE is very likely to be useful.
    ::glow::optimize(F, mode);
  }
}

void CPUDeviceManager::addNetworkImpl(DeviceNetworkID id,
                                      std::unique_ptr<Module> module,
                                      ReadyCB readyCB) {
  auto modIt = modules_.find(id);
  if (modIt != modules_.end()) {
    // Already have a module with this ID.
    // TODO: should we replace it?
    readyCB(id, FAILED);
    return;
  }

  // TODO: we should update usedMemory but we don't currently have a nice way
  // to determine the meory used by the module. I'll come back to this, but for
  // now we'll guess (badly).
  size_t moduleSize = 200 * 1024 * 1024;

  if (usedMemoryBytes + moduleSize > maxMemoryBytes) {
    readyCB(id, FAILED);
    return;
  }

  // Compile the functions.
  auto &functions = module->getFunctions();
  for (auto *F : functions) {
    optimizeFunction(CompilationMode::Infer, F);
    functions_[F] = backend_->compile(F);
  }

  modules_.emplace_hint(modIt, id, std::move(module));
  usedMemoryBytes += moduleSize;

  // Fire the ready CB
  readyCB(id, READY);
}

void CPUDeviceManager::evictNetworkImpl(DeviceNetworkID id) {
  auto modIt = modules_.find(id);
  if (modIt == modules_.end()) {
    // nothing to do
    return;
  }

  std::unique_ptr<Module> module = std::move(modIt->second);
  auto &functions = module->getFunctions();
  for (auto *F : functions) {
    functions_.erase(F);
  }
  modules_.erase(modIt);
  usedMemoryBytes -= 200 * 1024 * 1024; // TODO: static moduleSize
  assert(usedMemoryBytes >= 0);
}

void CPUDeviceManager::runFunctionImpl(DeviceNetworkID id,
                                       llvm::StringRef function,
                                       std::unique_ptr<Context> ctx,
                                       ResultCB resultCB) {
  auto modIt = modules_.find(id);
  if (modIt == modules_.end()) {
    resultCB(FAILED, std::move(ctx));
    return;
  }

  auto *F = modIt->second->getFunction(function);
  if (!F) {
    resultCB(FAILED, std::move(ctx));
    return;
  }

  auto funcIt = functions_.find(F);
  if (funcIt == functions_.end()) {
    resultCB(FAILED, std::move(ctx));
    return;
  }

  // TODO: verify that context has been allocated already?
  ctx->allocate(modIt->second->getPlaceholders());
  funcIt->second->setupRuns();
  funcIt->second->beforeRun(*ctx);
  funcIt->second->execute();
  funcIt->second->afterRun(*ctx);
  funcIt->second->tearDownRuns();

  // fire the ResultCB
  resultCB(EXECUTED, std::move(ctx));
}
