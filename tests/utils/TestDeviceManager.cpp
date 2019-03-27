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

#include "TestDeviceManager.h"

using namespace glow;
using namespace glow::runtime;

void TestDeviceManager::evictNetwork(std::string functionName,
                                     EvictFunctionCBTy evictCB) {
  // Erase the entry so that the same function name can be used to register
  // another result.
  llvm::Error err = llvm::Error::success();

  if (!resultMap_.erase(functionName)) {
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

void TestDeviceManager::doRunFunction(std::string functionName,
                                      std::shared_ptr<ExecutionContext> context,
                                      ResultCBTy resultCB) {
  RunIdentifierTy runId = 0;
  bool successResult = false;
  std::unique_ptr<ExecutionContext> resultContext = nullptr;

  // Retrieve the registered response for the function if there is one.
  if (context && resultCB && resultMap_.count(functionName)) {
    std::unique_ptr<RunFunctionResult> registeredResult =
        std::move(resultMap_[functionName]);

    // Check that context contains the expected Placeholder-Tensor mappings.
    std::unique_ptr<ExecutionContext> inputContext =
        std::move(registeredResult->inputContext);

    if (PlaceholderBindings::compare(context->getPlaceholderBindings(),
                                     inputContext->getPlaceholderBindings())) {
      // If bindings contains all expected mappings, overwrite the default
      // runId, result and resultContext with the registered
      // ones.
      runId = registeredResult->runId;
      successResult = registeredResult->success;
      resultContext = std::move(registeredResult->resultContext);
    }
  }

  if (successResult) {
    resultCB(runId, llvm::Error::success(), std::move(resultContext));
  } else {
    resultCB(runId, MAKE_ERR("An error occurred"), std::move(resultContext));
  }
}

runtime::RunIdentifierTy
TestDeviceManager::runFunction(std::string functionName,
                               std::unique_ptr<ExecutionContext> context,
                               ResultCBTy resultCB) {
  // Give the call to the thread pool to process to make the tests
  // multithreaded if needed.
  std::shared_ptr<ExecutionContext> sharedContext = std::move(context);
  this->threadPool_.submit([this, functionName, sharedContext, resultCB]() {
    this->doRunFunction(functionName, sharedContext, resultCB);
  });
  return 0;
}

bool TestDeviceManager::registerResult(
    const std::string &functionName, RunIdentifierTy runId, bool success,
    std::unique_ptr<ExecutionContext> inputContext,
    std::unique_ptr<ExecutionContext> resultContext) {
  bool registered = false;

  if (!resultMap_.count(functionName)) {
    // If the function name has not already been registered, insert it into
    // resultMap_.
    std::tie(std::ignore, registered) = resultMap_.insert(std::make_pair(
        functionName, llvm::make_unique<RunFunctionResult>(
                          runId, success, std::move(inputContext),
                          std::move(resultContext))));
  }

  return registered;
}
