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
#ifndef GLOW_TESTS_UTILS_TESTDEVICEMANAGER_H
#define GLOW_TESTS_UTILS_TESTDEVICEMANAGER_H

#include "glow/Backends/DeviceManager.h"
#include "glow/Support/ThreadPool.h"

#include <limits>
#include <memory>
#include <string>
#include <unordered_map>

namespace glow {
namespace runtime {

/// This is an implementation of DeviceManager tailored for testing Executor
/// implementations. registerResult() gives the caller the ability to
/// dictate precisely what a subsequent call to runFunction() should return.
/// registerResult() should be called before calling Executor::run() in each
/// test. The rest of the implementation of the DeviceManager interface exists
/// to satisfy the compiler.
class TestDeviceManager final : public DeviceManager {
public:
  /// Constructor.
  TestDeviceManager(unsigned numWorkers)
      : DeviceManager(BackendKind::Interpreter), threadPool_(numWorkers) {}

  /// The functions below are the interface for DeviceManager. See
  /// glow::DeviceManager for descriptions of what they do. Since this
  /// class exists only to help test Executor implementations, the only
  /// important function is runFunction().
  void addNetwork(const Module *module, FunctionMapTy functions,
                  ReadyCBTy readyCB) override {}

  /// Do not call this at the same time as registerResult().
  runtime::RunIdentifierTy
  runFunction(std::string functionName,
              std::unique_ptr<ExecutionContext> context,
              ResultCBTy resultCB) override;

  void evictNetwork(std::string functionName,
                    EvictFunctionCBTy evictCB) override;

  uint64_t getMaximumMemory() const override {
    return std::numeric_limits<uint64_t>::max();
  }

  uint64_t getAvailableMemory() const override {
    return std::numeric_limits<uint64_t>::max();
  }

  bool isMemoryAvailable(uint64_t /*estimate*/) const override { return true; }

  /// Look up the previously registered response for \p functionName and
  /// call \p resultCB with it after checking that \p context contains the
  /// expected Placeholder-Tensor mappings.
  void doRunFunction(std::string functionName,
                     std::shared_ptr<ExecutionContext> context,
                     ResultCBTy resultCB);

  /// Register a result that should be returned by the subsequent call to
  /// runFunction with the same \p functionName. The callback for that call
  /// to runFunction will be called with \p runId, \p success, and \p
  /// \p resultContext if the context passed in to runFunction
  /// matches \p inputContext. \returns true if registration was
  /// successful, false if not. Do not call this at the same time as
  /// runFunction().
  bool registerResult(const std::string &functionName, RunIdentifierTy runId,
                      bool success,
                      std::unique_ptr<ExecutionContext> inputContext,
                      std::unique_ptr<ExecutionContext> resultContext);

private:
  /// This struct wraps all of the data needed to reply to a runFunction() call.
  /// It exists so that that all of these things can be stored in one map.
  struct RunFunctionResult {
    /// The run ID that should be returned.
    RunIdentifierTy runId;
    /// If success then no error should be returned otherwise an Error should be
    /// returned.
    bool success;
    /// The expected input context for the invocation.
    std::unique_ptr<ExecutionContext> inputContext;
    /// The result context that should be returned.
    std::unique_ptr<ExecutionContext> resultContext;

    /// Constructor.
    RunFunctionResult(RunIdentifierTy run, bool successParam,
                      std::unique_ptr<ExecutionContext> inputcontext,
                      std::unique_ptr<ExecutionContext> resultcontext)
        : runId(run), success(successParam),
          inputContext(std::move(inputcontext)),
          resultContext(std::move(resultcontext)) {}
  };

  /// Map of function name -> RunFunctionResult instance containing the
  /// RunFunctionResult instance for the function.
  using TestDeviceManagerResultMapTy =
      std::unordered_map<std::string, std::unique_ptr<RunFunctionResult>>;

  /// Map for storing registered results.
  TestDeviceManagerResultMapTy resultMap_;
  /// Thread pool for executing runFunction() in a multithreaded fashion.
  ThreadPool threadPool_;
};

} // namespace runtime
} // namespace glow

#endif // GLOW_TESTS_UTILS_TESTDEVICEMANAGER_H
