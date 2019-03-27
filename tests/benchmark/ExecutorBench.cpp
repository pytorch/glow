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
#include "glow/Runtime/Executor/Executor.h"

#include "ExecutorBenchmarkWrapper.h"
#include "ExecutorTestBuilder.h"
#include "TestDeviceManager.h"

#include "benchmark/benchmark.h"

using namespace glow;
using namespace glow::runtime;

/// Fixture class for reusing Executor, ExecutorTestBuilder and DeviceManagerMap
/// setup across benchmarks.
class ThreadPoolExecutorBench : public benchmark::Fixture {
public:
  ThreadPoolExecutorBench()
      : executor_(std::shared_ptr<Executor>(
            createExecutor(deviceManagerMap_, ExecutorKind::ThreadPool))),
        benchmarkBuilder_(executor_, deviceManagerMap_) {}
  ~ThreadPoolExecutorBench() = default;

protected:
  /// The Executor being tested.
  std::shared_ptr<Executor> executor_;
  /// An ExecutorTestBuilder instance for creating benchmarks.
  ExecutorTestBuilder benchmarkBuilder_;
  /// DeviceManager map for initializing executor_.
  DeviceManagerMapTy deviceManagerMap_;
};

/// Benchmark for executing an empty DAG.
BENCHMARK_DEFINE_F(ThreadPoolExecutorBench, Empty)(benchmark::State &state) {
  constexpr RunIdentifierTy testRunId = 10;

  for (auto _ : state) {
    // Pause timing while setting up to run the benchmark.
    state.PauseTiming();

    // Make a PlaceholderBindings with one Placeholder in it to make sure
    // Executor::run() doesn't modify it when the root given to it is null. Make
    // two identical copies; one to give to Executor::run(), and another to
    // compare the returned PlaceholderBindings with.
    PseudoRNG rng;
    auto type = std::unique_ptr<Type>(new Type(ElemKind::FloatTy, {1, 2, 2}));
    auto placeholder = llvm::make_unique<Placeholder>("a", type.get(),
                                                      /*trainable=*/false);

    auto testContext = llvm::make_unique<ExecutionContext>();
    auto refContext = llvm::make_unique<ExecutionContext>();

    auto *tensor =
        testContext->getPlaceholderBindings()->allocate(placeholder.get());
    tensor->init(Tensor::InitKind::Xavier, 1.0, rng);
    refContext->getPlaceholderBindings()->insert(placeholder.get(),
                                                 tensor->clone());

    // Variables for storing runId actually returned by
    // Executor::run() via its callback.
    RunIdentifierTy executorRunId;
    std::unique_ptr<ExecutionContext> executorOutputContext;

    // Call Executor::run().
    llvm::Error runErr = llvm::Error::success();
    std::promise<void> promise;
    std::future<void> future = promise.get_future();

    // Start timing.
    state.ResumeTiming();
    executor_->run(nullptr, std::move(testContext), testRunId,
                   [&runErr, &promise, &executorRunId, &executorOutputContext](
                       RunIdentifierTy runId, llvm::Error err,
                       std::unique_ptr<ExecutionContext> context) {
                     executorRunId = runId;
                     executorOutputContext = std::move(context);
                     runErr = std::move(err);
                     promise.set_value();
                   });

    future.wait();

    // Pause timing again while checking correctness.
    state.PauseTiming();

    // Check that the returned ID and context are as expected.
    GLOW_ASSERT(!errToBool(std::move(runErr)));
    GLOW_ASSERT(executorRunId == testRunId);
    GLOW_ASSERT(PlaceholderBindings::compare(
        refContext->getPlaceholderBindings(),
        executorOutputContext->getPlaceholderBindings()));

    // Resuming timing after checking correctness to preserve loop invariant.
    state.ResumeTiming();
  }
}

/// Benchmark for executing a DAG with a single node.
BENCHMARK_DEFINE_F(ThreadPoolExecutorBench, SingleNode)
(benchmark::State &state) {
  constexpr RunIdentifierTy testRunId = 10;
  constexpr DeviceIDTy testDeviceId = 111;
  constexpr unsigned deviceManagerThreads = 1;

  for (auto _ : state) {
    // Pause timing while setting up the benchmark.
    state.PauseTiming();

    // Make a TestDeviceManager and insert into the DeviceManagerMap map (which
    // the ThreadPoolExecutor has a reference to) and the TestDeviceManager map
    // (which the ExecutorTestBuilder has a reference to).
    auto deviceManager =
        llvm::make_unique<TestDeviceManager>(deviceManagerThreads);
    deviceManagerMap_.emplace(testDeviceId, std::move(deviceManager));

    // Build the DAG. The DAG created below looks like this:
    /**
     *         root
     *          |
     *          v
     *         net
     **/

    benchmarkBuilder_.addNode("net", testDeviceId,
                              /*parents=*/{}, /*inputs=*/{"netInput"},
                              /*outputs=*/{"netOutput"}, testRunId,
                              /*success=*/true);

    std::unique_ptr<ExecutorBenchmarkWrapper> b =
        benchmarkBuilder_.emitTest<ExecutorBenchmarkWrapper>();
    GLOW_ASSERT(b);

    // Run the benchmark.
    state.ResumeTiming();
    bool ok = b->run(state);
    state.PauseTiming();

    // Check correctness.
    GLOW_ASSERT(ok);

    // Erase the entry for the TestDeviceManager in preparation for next
    // iteration.
    deviceManagerMap_.erase(testDeviceId);

    // Resume timing to preserve loop invariant.
    state.ResumeTiming();
  }
}

/// Benchmark for executing a straight-line DAG.
BENCHMARK_DEFINE_F(ThreadPoolExecutorBench, StraightLine)
(benchmark::State &state) {
  constexpr RunIdentifierTy testRunId = 10;
  constexpr DeviceIDTy testDeviceId = 111;
  constexpr unsigned deviceManagerThreads = 3;

  for (auto _ : state) {
    // Pause timing while setting up the benchmark.
    state.PauseTiming();

    // Make a TestDeviceManager and insert into the DeviceManagerMap map (which
    // the ThreadPoolExecutor has a reference to) and the TestDeviceManager map
    // (which the ExecutorTestBuilder has a reference to).
    auto deviceManager =
        llvm::make_unique<TestDeviceManager>(deviceManagerThreads);
    deviceManagerMap_.emplace(testDeviceId, std::move(deviceManager));

    // Build the DAG. The DAG created below looks like this:
    /**
     *         root
     *          |
     *          v
     *         net_0
     *          |
     *          v
     *         ...
     *          |
     *          v
     *       net_{e-1}
     **/
    benchmarkBuilder_.addNode("net_0", testDeviceId,
                              /*parents=*/{}, /*inputs=*/{"netInput_0"},
                              /*outputs=*/{"netOutput_0"}, testRunId,
                              /*success=*/true);

    for (int64_t i = 1, e = state.range(0); i < e; ++i) {
      benchmarkBuilder_.addNode(strFormat("net_%" PRId64, i), testDeviceId,
                                /*parents=*/{strFormat("net_%" PRId64, i - 1)},
                                /*inputs=*/{strFormat("netOutput_%" PRId64, i - 1)},
                                /*outputs=*/{strFormat("netOutput_%" PRId64, i)},
                                testRunId,
                                /*success=*/true);
    }

    std::unique_ptr<ExecutorBenchmarkWrapper> b =
        benchmarkBuilder_.emitTest<ExecutorBenchmarkWrapper>();
    GLOW_ASSERT(b);

    // Run the benchmark.
    state.ResumeTiming();
    bool ok = b->run(state);
    state.PauseTiming();

    // Check correctness.
    GLOW_ASSERT(ok);

    // Erase the entry for the TestDeviceManager in preparation for next
    // iteration.
    deviceManagerMap_.erase(testDeviceId);

    // Resume timing to preserve loop invariant.
    state.ResumeTiming();
  }
}

/// Benchmark for executing a DAG with many parallel nodes.
BENCHMARK_DEFINE_F(ThreadPoolExecutorBench, ManyParallelNodes)
(benchmark::State &state) {
  constexpr RunIdentifierTy testRunId = 10;
  constexpr DeviceIDTy testDeviceId = 111;
  constexpr unsigned deviceManagerThreads = 3;

  for (auto _ : state) {
    // Pause timing while setting up the benchmark.
    state.PauseTiming();

    // Make a TestDeviceManager and insert into the DeviceManagerMap map (which
    // the ThreadPoolExecutor has a reference to) and the TestDeviceManager map
    // (which the ExecutorTestBuilder has a reference to).
    auto deviceManager =
        llvm::make_unique<TestDeviceManager>(deviceManagerThreads);
    deviceManagerMap_.emplace(testDeviceId, std::move(deviceManager));

    // Build the DAG. The DAG created below looks like this:
    /**
     *    -------- root -----------
     *    |         |             |
     *    v         v             v
     *  net_0    net_1  ...  net_{e-1}
     **/
    for (int64_t i = 0, e = state.range(0); i < e; ++i) {
      benchmarkBuilder_.addNode(strFormat("net_%" PRId64, i), testDeviceId,
                                /*parents=*/{},
                                /*inputs=*/{strFormat("netInput_%" PRId64, i)},
                                /*outputs=*/{strFormat("netOutput_%" PRId64, i)},
                                testRunId,
                                /*success=*/true);
    }

    std::unique_ptr<ExecutorBenchmarkWrapper> b =
        benchmarkBuilder_.emitTest<ExecutorBenchmarkWrapper>();
    GLOW_ASSERT(b);

    // Run the benchmark.
    state.ResumeTiming();
    bool ok = b->run(state);
    state.PauseTiming();

    // Check correctness.
    GLOW_ASSERT(ok);

    // Erase the entry for the TestDeviceManager in preparation for next
    // iteration.
    deviceManagerMap_.erase(testDeviceId);

    // Resume timing to preserve loop invariant.
    state.ResumeTiming();
  }
}

/// Benchmark for executing a large and complicated DAG.
BENCHMARK_DEFINE_F(ThreadPoolExecutorBench, LargeDAG)
(benchmark::State &state) {
  constexpr RunIdentifierTy testRunId = 10;
  constexpr DeviceIDTy testDeviceId = 111;
  constexpr unsigned deviceManagerThreads = 3;

  for (auto _ : state) {
    // Pause timing while setting up the benchmark.
    state.PauseTiming();

    // Make a TestDeviceManager and insert into the DeviceManagerMap map (which
    // the ThreadPoolExecutor has a reference to) and the TestDeviceManager map
    // (which the ExecutorTestBuilder has a reference to).
    auto deviceManager =
        llvm::make_unique<TestDeviceManager>(deviceManagerThreads);
    deviceManagerMap_.emplace(testDeviceId, std::move(deviceManager));

    // Build the DAG. The DAG created below looks like this:
    /**
     *            root
     *           /   \
     *          v    v
     *      -- a     c
     *     |   |    / \
     *     |   v   |  v
     *     --> b   |  e
     *     |   |   |  |
     *     |   v   |  |
     *     --> d <-   |
     *     |   |      |
     *     |   v      v
     *     --> f ---> g
     **/

    benchmarkBuilder_.addNode("a", testDeviceId,
                              /*parents=*/{},
                              /*inputs=*/{"ai_0", "ai_1", "ai_2"},
                              /*outputs=*/{"ao_0", "ao_1", "ao_2"}, testRunId,
                              /*succcess=*/true);
    benchmarkBuilder_.addNode("b", testDeviceId,
                              /*parents=*/{"a"}, /*inputs=*/{"ao_0", "ao_1"},
                              /*outputs=*/{"bo_0", "bo_1", "bo_2"}, testRunId,
                              /*succcess=*/true);
    benchmarkBuilder_.addNode("c", testDeviceId,
                              /*parents=*/{},
                              /*inputs=*/{"ci_0", "ci_1", "ci_2"},
                              /*outputs=*/{"co_0", "co_1", "co_2"}, testRunId,
                              /*succcess=*/true);
    benchmarkBuilder_.addNode("d", testDeviceId,
                              /*parents=*/{"a", "b", "c"}, /*inputs=*/
                              {"ao_0", "ao_1", "ao_2", "bo_0", "bo_1", "bo_2",
                               "co_0", "co_1", "co_2"},
                              /*outputs=*/{"do_0", "do_1"}, testRunId,
                              /*succcess=*/true);
    benchmarkBuilder_.addNode("e", testDeviceId,
                              /*parents=*/{"c"},
                              /*inputs=*/{"co_0", "co_1", "co_2"},
                              /*outputs=*/{"eo_0", "eo_1", "eo_2"}, testRunId,
                              /*succcess=*/true);
    benchmarkBuilder_.addNode("f", testDeviceId,
                              /*parents=*/{"a", "d"},
                              /*inputs=*/{"ao_0", "do_0", "do_1"},
                              /*outputs=*/{"fo_0"}, testRunId,
                              /*succcess=*/true);
    benchmarkBuilder_.addNode("g", testDeviceId,
                              /*parents=*/{"e", "f"},
                              /*inputs=*/{"eo_0", "eo_1", "fo_0"},
                              /*outputs=*/{"go_0", "go_1"}, testRunId,
                              /*succcess=*/true);

    std::unique_ptr<ExecutorBenchmarkWrapper> b =
        benchmarkBuilder_.emitTest<ExecutorBenchmarkWrapper>();
    GLOW_ASSERT(b);

    // Run the benchmark.
    state.ResumeTiming();
    bool ok = b->run(state);
    state.PauseTiming();

    // Check correctness.
    GLOW_ASSERT(ok);

    // Erase the entry for the TestDeviceManager in preparation for next
    // iteration.
    deviceManagerMap_.erase(testDeviceId);

    // Resume timing to preserve loop invariant.
    state.ResumeTiming();
  }
}

/// Benchmark registration.
BENCHMARK_REGISTER_F(ThreadPoolExecutorBench, Empty)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(ThreadPoolExecutorBench, SingleNode)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(ThreadPoolExecutorBench, StraightLine)
    ->RangeMultiplier(2)
    ->Range(1, 64)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(ThreadPoolExecutorBench, ManyParallelNodes)
    ->RangeMultiplier(2)
    ->Range(1, 64)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(ThreadPoolExecutorBench, LargeDAG)
    ->Unit(benchmark::kMicrosecond);

/// Main.
BENCHMARK_MAIN();
