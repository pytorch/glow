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

#include "ExecutorBenchmarkWrapper.h"

#include "glow/Runtime/Executor/Executor.h"

#include <future>
#include <memory>

using namespace glow;
using namespace glow::runtime;

bool ExecutorBenchmarkWrapper::run(benchmark::State &state) {
  // Pause timing until just before the call to Executor::run().
  state.PauseTiming();
  if (testRun_) {
    assert(!"Test has already been run!");
  }

  // Variables for storing runId actually returned by
  // Executor::run() via its callback.
  RunIdentifierTy executorRunId;
  std::unique_ptr<ExecutionContext> executorOutputContext;

  // Call Executor::run().
  std::promise<bool> promise;
  std::future<bool> future = promise.get_future();

  state.ResumeTiming();
  executor_->run(root_.get(), std::move(inputContext_), runId_,
                 [&promise, &executorRunId, &executorOutputContext](
                     RunIdentifierTy runId, llvm::Error err,
                     std::unique_ptr<ExecutionContext> context) {
                   executorRunId = runId;
                   executorOutputContext = std::move(context);
                   promise.set_value(errToBool(std::move(err)));
                 });

  bool runSuccess = !future.get();
  state.PauseTiming();

  // Check that the values returned in the Executor callback match
  // expectations.
  bool runIdsMatch = executorRunId == runId_;
  bool resultsMatch = runSuccess == expectSuccess_;

  bool bindingsMatch = PlaceholderBindings::compare(
      executorOutputContext->getPlaceholderBindings(),
      outputContext_->getPlaceholderBindings());

  // If the run failed, we shouldn't expect bindingsMatch to be true.
  bool testPassed =
      runIdsMatch && resultsMatch && (!runSuccess || bindingsMatch);

  testRun_ = true;

  return testPassed;
}
