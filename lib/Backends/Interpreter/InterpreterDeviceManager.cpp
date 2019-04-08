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
#include "InterpreterDeviceManager.h"
#include "Interpreter.h"

#include "llvm/Support/raw_ostream.h"

namespace glow {
namespace runtime {

DeviceManager *
createInterpreterDeviceManager(std::unique_ptr<DeviceConfig> config) {
  return new InterpreterDeviceManager(std::move(config));
}

uint64_t InterpreterDeviceManager::getMaximumMemory() const {
  return maxMemoryBytes_;
}

uint64_t InterpreterDeviceManager::getAvailableMemory() const {
  return maxMemoryBytes_ - usedMemoryBytes_;
}

bool InterpreterDeviceManager::isMemoryAvailable(uint64_t estimate) const {
  return maxMemoryBytes_ >= (usedMemoryBytes_ + estimate);
}

void InterpreterDeviceManager::addNetworkImpl(const Module *module,
                                              FunctionMapTy functions,
                                              ReadyCBTy readyCB) {
  // First check for uniqueness of the function name.
  for (const auto &func : functions) {
    if (functions_.count(func.first) != 0) {
      readyCB(
          module,
          MAKE_ERR(
              llvm::formatv(
                  "Failed to add network: already have a function called {0}",
                  func.first)
                  .str()));
      return;
    }

    if (func.second->getCompileBackendKind() != BackendKind::Interpreter) {
      readyCB(module, MAKE_ERR(llvm::formatv("Failed to add network: function "
                                             "{0} is not a InterpreterFunction",
                                             func.first)
                                   .str()));
      return;
    }
  }

  if (usedMemoryBytes_ + functionCost_ > maxMemoryBytes_) {
    readyCB(module, MAKE_ERR(GlowErr::ErrorCode::RUNTIME_OUT_OF_DEVICE_MEMORY,
                             "Failed to add network: not enough memory"));
    return;
  }

  // Add to the function name lookup map.
  for (const auto &func : functions) {
    if (func.second->getRuntimeBundle().getConstants() == nullptr) {
      func.second->getRuntimeBundle().collectConstants(module);
    }
    functions_.emplace(func.first, func.second);
    usedMemoryBytes_ += functionCost_; // TODO:: static moduleSize
  }

  assert(usedMemoryBytes_ <= maxMemoryBytes_);

  // Fire the ready CB.
  readyCB(module, llvm::Error::success());
}

void InterpreterDeviceManager::evictNetworkImpl(std::string functionName,
                                                EvictFunctionCBTy evictCB) {
  llvm::Error err = llvm::Error::success();

  if (functions_.erase(functionName)) {
    usedMemoryBytes_ -= functionCost_; // TODO: static moduleSize
  } else {
    err =
        MAKE_ERR(GlowErr::ErrorCode::RUNTIME_NET_NOT_FOUND,
                 llvm::formatv("Could not find function with name {0} to evict",
                               functionName)
                     .str());
  }

  if (evictCB) {
    evictCB(functionName, std::move(err));
  } else {
    llvm::errs() << llvm::toString(std::move(err));
  }
}

void InterpreterDeviceManager::runFunctionImpl(
    RunIdentifierTy id, std::string function,
    std::unique_ptr<ExecutionContext> context, ResultCBTy resultCB) {
  context->logTraceEvent("DM_run", "B");
  auto funcIt = functions_.find(function);
  if (funcIt == functions_.end()) {
    context->logTraceEvent("DM_run", "E", {{"reason", "function not found"}});
    resultCB(id,
             MAKE_ERR(GlowErr::ErrorCode::RUNTIME_NET_NOT_FOUND,
                      llvm::formatv("Function {0} not found", function).str()),
             std::move(context));
    return;
  }

  CompiledFunction *func = funcIt->second;

  // Run that function.
  func->execute(context.get());

  // End the TraceEvent early to avoid time in the CB.
  context->logTraceEvent("DM_run", "E");

  // Fire the resultCB.
  resultCB(id, llvm::Error::success(), std::move(context));
}

} // namespace runtime
} // namespace glow
