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

#include "glow/Runtime/Executor/Executor.h"
#include "RuntimeTestUtils.h"
#include "glow/Backends/DeviceManager.h"
#include "glow/Support/Support.h"
#include "glow/Support/ThreadPool.h"

#include "gtest/gtest.h"

#include <chrono>
#include <future>
#include <thread>
#include <unordered_set>

using namespace glow;
using namespace glow::runtime;

/// This test fixture provides ThreadPoolExecutor, ExecutorTestBuilder,
/// DeviceManagerMapTy instances to all tests.
class ThreadPoolExecutorTest : public ::testing::Test {
protected:
  ThreadPoolExecutorTest()
      : executor_(std::shared_ptr<Executor>(
            createExecutor(deviceManagerMap_, ExecutorKind::ThreadPool))),
        testBuilder_(executor_, deviceManagerMap_) {}
  ~ThreadPoolExecutorTest() = default;

  /// The Executor being tested.
  std::shared_ptr<Executor> executor_;
  /// An ExecutorTestBuilder instance for creating tests.
  ExecutorTestBuilder testBuilder_;
  /// DeviceManager map for initializing executor_.
  DeviceManagerMapTy deviceManagerMap_;
};

/// Tests that an empty DAG is handled correctly.
TEST_F(ThreadPoolExecutorTest, EmptyDAG) {
  constexpr RunIdentifierTy testRunId = 10;

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
  executor_->run(nullptr, std::move(testContext), testRunId,
                 [&runErr, &promise, &executorRunId, &executorOutputContext](
                     RunIdentifierTy runId, llvm::Error err,
                     std::unique_ptr<ExecutionContext> context) {
                   executorRunId = runId;
                   executorOutputContext = std::move(context);
                   runErr = std::move(err);
                   promise.set_value();
                 });

  EXPECT_FALSE(errToBool(std::move(runErr)));

  EXPECT_EQ(executorRunId, testRunId);

  EXPECT_TRUE(PlaceholderBindings::compare(
      refContext->getPlaceholderBindings(),
      executorOutputContext->getPlaceholderBindings()));
}

/// Tests that a single node can run correctly.
TEST_F(ThreadPoolExecutorTest, SingleNode) {
  constexpr RunIdentifierTy testRunId = 10;
  constexpr DeviceIDTy testDeviceId = 111;
  constexpr unsigned deviceManagerThreads = 1;

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

  testBuilder_.addNode("net", testDeviceId,
                       /*parents=*/{}, {"netInput"}, {"netOutput"}, testRunId,
                       true);

  ExecutorTest test = testBuilder_.emitTest();
  EXPECT_TRUE(test.run());
}

/// Tests that several instances of a single node DAG can be run in parallel.
TEST_F(ThreadPoolExecutorTest, ConcurrentSingleNode) {
  constexpr RunIdentifierTy baseTestRunId = 10;
  constexpr DeviceIDTy testDeviceId = 111;
  constexpr unsigned deviceManagerThreads = 3;
  unsigned numConcurrentRuns = 1000;

  // Make a TestDeviceManager and insert into the DeviceManagerMap map (which
  // the ThreadPoolExecutor has a reference to) and the TestDeviceManager map
  // (which the ExecutorTestBuilder has a reference to).
  auto deviceManager =
      llvm::make_unique<TestDeviceManager>(deviceManagerThreads);
  deviceManagerMap_.emplace(testDeviceId, std::move(deviceManager));

  // Mutex for accessing threadsReady and testsPassed.
  std::mutex mtx;
  // Condition variables for signalling between the test runner threads
  // and this thread. These are used to implement a barrier that ensures
  // all test runner threads have been created and are executing before any
  // are allowed to run a test (in order to try and increase the number of
  // threads that call Executor::run() at the same time).
  std::condition_variable driverCV, threadCV;
  // Counters for implementing the aforementioned barrier and tracking the
  // number of tests that pass.
  unsigned threadsReady = 0, testsPassed = 0;
  std::vector<std::thread> threads;
  for (unsigned i = 0; i < numConcurrentRuns; ++i) {
    // Build the DAG. The DAG created below looks like this:
    /**
     *         root
     *          |
     *          v
     *         net
     **/

    // The names must be distinct since the DeviceManager distinguishes based
    // on function name. The run IDs must also be distinct (hence the +i).
    testBuilder_.addNode(strFormat("net_%d", i), testDeviceId,
                         /*parents=*/{}, {"netInput"}, {"netOutput"},
                         baseTestRunId + i, true);
    ExecutorTest t = testBuilder_.emitTest();

    std::thread th([&mtx, &driverCV, &threadCV, &threadsReady, &testsPassed,
                    test = std::move(t), numConcurrentRuns]() mutable {
      std::unique_lock<std::mutex> lock(mtx);
      // Increment threadsReady to mark this thread as ready to run the test.
      threadsReady++;
      // If threadsReady == numConcurrentRuns, this thread is the last to be
      // initialized and execute, so signal the driver that all threads are
      // ready.
      if (threadsReady == numConcurrentRuns) {
        driverCV.notify_one();
      }
      // Wait for the driver's signal.
      threadCV.wait(lock);
      // Unlock the mutex to let all other threads run their tests concurrently.
      lock.unlock();
      bool passed = test.run();
      lock.lock();

      if (passed) {
        testsPassed++;
      }
    });
    threads.emplace_back(std::move(th));
  }

  std::unique_lock<std::mutex> lock(mtx);
  // If threadsReady != numConcurrentRuns, not all threads are ready to run
  // their tests. Wait until they are.
  if (threadsReady != numConcurrentRuns) {
    driverCV.wait(lock, [&threadsReady, numConcurrentRuns] {
      return threadsReady == numConcurrentRuns;
    });
  }
  // Wake up all test runners.
  threadCV.notify_all();
  lock.unlock();

  // Join all threads.
  for (unsigned i = 0; i < numConcurrentRuns; ++i) {
    threads[i].join();
  }

  // All tests should pass.
  EXPECT_EQ(testsPassed, numConcurrentRuns);
}

/// Tests that successive calls to ThreadPoolExecutor::run() with the same
/// runId don't succeed.
TEST_F(ThreadPoolExecutorTest, ConcurrentSingleNodeDuplicateRunId) {
  constexpr RunIdentifierTy testRunId = 10;
  constexpr DeviceIDTy testDeviceId = 111;
  constexpr unsigned deviceManagerThreads = 1;
  constexpr unsigned numConcurrentRuns = 100;

  // Make a TestDeviceManager and insert into the DeviceManagerMap map (which
  // the ThreadPoolExecutor has a reference to) and the TestDeviceManager map
  // (which the ExecutorTestBuilder has a reference to).
  auto deviceManager =
      llvm::make_unique<TestDeviceManager>(deviceManagerThreads);
  deviceManagerMap_.emplace(testDeviceId, std::move(deviceManager));

  std::atomic<unsigned> testsPassed{0};
  std::vector<std::thread> threads;
  std::vector<ExecutorTest> tests;

  // Build all tests.
  for (unsigned i = 0; i < numConcurrentRuns; ++i) {
    // Build the DAG. The DAG created below looks like this:
    /**
     *         root
     *          |
     *          v
     *         net
     **/

    testBuilder_.addNode(strFormat("net_%d", i), testDeviceId,
                         /*parents=*/{}, {"netInput"}, {"netOutput"}, testRunId,
                         true);
    tests.emplace_back(testBuilder_.emitTest());
  }

  // Run all tests.
  for (unsigned i = 0; i < numConcurrentRuns; ++i) {
    std::thread th([&testsPassed, test = std::move(tests[i])]() mutable {
      bool passed = test.run();
      if (passed) {
        testsPassed++;
      }
    });
    threads.emplace_back(std::move(th));
  }

  // Join all threads.
  for (unsigned i = 0; i < numConcurrentRuns; ++i) {
    threads[i].join();
  }

  // At least one test should pass. Depending on the interleaving, the
  // rest can all pass or all fail or anything in between.
  EXPECT_GE(testsPassed, 1);
}

/// Tests that a DAG with multiple nodes can run correctly.
TEST_F(ThreadPoolExecutorTest, MultiNode) {
  constexpr RunIdentifierTy testRunId = 10;
  constexpr DeviceIDTy testDeviceId = 111;
  constexpr unsigned deviceManagerThreads = 3;

  // Make a TestDeviceManager and insert into the DeviceManagerMap map (which
  // the ThreadPoolExecutor has a reference to) and the TestDeviceManager map
  // (which the ExecutorTestBuilder has a reference to).
  auto deviceManager =
      llvm::make_unique<TestDeviceManager>(deviceManagerThreads);
  deviceManagerMap_.emplace(testDeviceId, std::move(deviceManager));

  // Build the DAG. The DAG created below looks like this:
  /**
   *           root
   *         /      \
   *        v       v
   *      alpha    beta
   *        \       /
   *         v     v
   *          gamma
   *         /    \
   *        v     v
   *     delta   eps
   **/

  testBuilder_.addNode("alpha", testDeviceId,
                       /*parents=*/{}, /*inputs=*/{"alphaIn"},
                       /*outputs=*/{"alphaOut"}, testRunId, true);
  testBuilder_.addNode("beta", testDeviceId,
                       /*parents=*/{}, /*inputs=*/{"betaIn"},
                       /*outputs=*/{"betaOut"}, testRunId, true);
  testBuilder_.addNode("gamma", testDeviceId,
                       /*parents=*/{"alpha", "beta"},
                       /*inputs=*/{"alphaOut", "betaOut"},
                       /*outputs=*/{"deltaIn", "epsIn"}, testRunId, true);
  testBuilder_.addNode("delta", testDeviceId,
                       /*parents=*/{"gamma"}, /*inputs=*/{"deltaIn"},
                       /*outputs=*/{"deltaOut"}, testRunId, true);
  testBuilder_.addNode("eps", testDeviceId,
                       /*parents=*/{"gamma"}, /*inputs=*/{"epsIn"},
                       /*outputs=*/{"epsOut"}, testRunId, true);

  ExecutorTest test = testBuilder_.emitTest();
  EXPECT_TRUE(test.run());
}

/// Tests that a DAG with a node that fails can run correctly.
TEST_F(ThreadPoolExecutorTest, MultiNodeWithFailure) {
  constexpr RunIdentifierTy testRunId = 10;
  constexpr DeviceIDTy testDeviceId = 111;
  constexpr unsigned deviceManagerThreads = 3;

  // Make a TestDeviceManager and insert into the DeviceManagerMap map (which
  // the ThreadPoolExecutor has a reference to) and the TestDeviceManager map
  // (which the ExecutorTestBuilder has a reference to).
  auto deviceManager =
      llvm::make_unique<TestDeviceManager>(deviceManagerThreads);
  deviceManagerMap_.emplace(testDeviceId, std::move(deviceManager));

  // Build the DAG. The DAG created below looks like this:
  /**
   *             root
   *           /      \
   *          v       v
   *        alpha    delta
   *          |       |
   *          v       v
   *        beta     eps
   *          |       |
   *          v       v
   *        gamma    zeta
   **/

  testBuilder_.addNode("alpha", testDeviceId,
                       /*parents=*/{}, /*inputs=*/{"alphaIn"},
                       /*outputs=*/{"alphaOut"}, testRunId, true);
  testBuilder_.addNode("beta", testDeviceId,
                       /*parents=*/{"alpha"}, /*inputs=*/{"alphaOut"},
                       /*outputs=*/{"betaOut"}, testRunId, true);
  testBuilder_.addNode("gamma", testDeviceId,
                       /*parents=*/{"beta"},
                       /*inputs=*/{"betaOut"},
                       /*outputs=*/{"gammaOut"}, testRunId, true);
  testBuilder_.addNode("delta", testDeviceId,
                       /*parents=*/{}, /*inputs=*/{"deltaIn"},
                       /*outputs=*/{"deltaOut"}, testRunId, true);
  testBuilder_.addNode("eps", testDeviceId,
                       /*parents=*/{"delta"}, /*inputs=*/{"deltaOut"},
                       /*outputs=*/{"epsOut"}, testRunId, false);
  testBuilder_.addNode("zeta", testDeviceId,
                       /*parents=*/{"eps"}, /*inputs=*/{"epsOut"},
                       /*outputs=*/{"zetaOut"}, testRunId, true);

  ExecutorTest test = testBuilder_.emitTest();
  EXPECT_TRUE(test.run());
}

/// Tests that a DAG with nodes spread across multiple devices can run
/// correctly.
TEST_F(ThreadPoolExecutorTest, MultiNodeMultiDevice) {
  constexpr RunIdentifierTy testRunId = 10;
  constexpr DeviceIDTy testDeviceIdA = 111;
  constexpr DeviceIDTy testDeviceIdB = 112;
  constexpr DeviceIDTy testDeviceIdC = 113;
  constexpr unsigned deviceManagerThreads = 3;

  // Make TestDeviceManagers and insert them into the DeviceManagerMap map
  // (which the ThreadPoolExecutor has a reference to) and the TestDeviceManager
  // map (which the ExecutorTestBuilder has a reference to).
  for (DeviceIDTy deviceId : {testDeviceIdA, testDeviceIdB, testDeviceIdC}) {
    auto deviceManager =
        llvm::make_unique<TestDeviceManager>(deviceManagerThreads);
    deviceManagerMap_.emplace(deviceId, std::move(deviceManager));
  }

  // Build the DAG. The DAG created below looks like this:
  /**
   *           root
   *         /      \
   *        v       v
   *      alpha    beta
   *        \       /
   *         v     v
   *          gamma
   *         /    \
   *        v     v
   *     delta   eps
   **/

  testBuilder_.addNode("alpha", testDeviceIdA,
                       /*parents=*/{}, /*inputs=*/{"alphaIn"},
                       /*outputs=*/{"alphaOut"}, testRunId, true);
  testBuilder_.addNode("beta", testDeviceIdB,
                       /*parents=*/{}, /*inputs=*/{"betaIn"},
                       /*outputs=*/{"betaOut"}, testRunId, true);
  testBuilder_.addNode("gamma", testDeviceIdC,
                       /*parents=*/{"alpha", "beta"},
                       /*inputs=*/{"alphaOut", "betaOut"},
                       /*outputs=*/{"deltaIn", "epsIn"}, testRunId, true);
  testBuilder_.addNode("delta", testDeviceIdA,
                       /*parents=*/{"gamma"}, /*inputs=*/{"deltaIn"},
                       /*outputs=*/{"deltaOut"}, testRunId, true);
  testBuilder_.addNode("eps", testDeviceIdB,
                       /*parents=*/{"gamma"}, /*inputs=*/{"epsIn"},
                       /*outputs=*/{"epsOut"}, testRunId, true);

  ExecutorTest test = testBuilder_.emitTest();
  EXPECT_TRUE(test.run());
}

/// Tests that several instances of a DAG with multiple nodes can run correctly
/// in parallel.
TEST_F(ThreadPoolExecutorTest, ConcurrentMultiNode) {
  constexpr RunIdentifierTy baseTestRunId = 10;
  constexpr DeviceIDTy testDeviceId = 111;
  constexpr unsigned deviceManagerThreads = 3;
  unsigned numConcurrentRuns = 1000;

  // Make a TestDeviceManager and insert it into the DeviceManagerMap map
  // (which the ThreadPoolExecutor has a reference to) and the TestDeviceManager
  // map (which the ExecutorTestBuilder has a reference to).
  auto deviceManager =
      llvm::make_unique<TestDeviceManager>(deviceManagerThreads);
  deviceManagerMap_.emplace(testDeviceId, std::move(deviceManager));

  // Mutex for accessing threadsReady and testsPassed.
  std::mutex mtx;
  // Condition variables for signalling between the test runner threads
  // and this thread. These are used to implement a barrier that ensures
  // all test runner threads have been created and are executing before any
  // are allowed to run a test (in order to try and increase the number of
  // threads that call Executor::run() at the same time).
  std::condition_variable driverCV, threadCV;
  // Counters for implementing the aforementioned barrier and tracking the
  // number of tests that pass.
  unsigned threadsReady = 0, testsPassed = 0;
  std::vector<std::thread> threads;
  for (unsigned i = 0; i < numConcurrentRuns; ++i) {
    // Build the DAG. The DAG created below looks like this:
    /**
     *           root
     *         /      \
     *        v       v
     *      alpha    beta
     *        \       /
     *         v     v
     *          gamma
     *         /    \
     *        v     v
     *     delta   eps
     **/

    // The names must be distinct for each run since the DeviceManager
    // distinguishes based on function name.
    std::string alpha = strFormat("alpha_%d", i);
    std::string beta = strFormat("beta_%d", i);
    std::string gamma = strFormat("gamma_%d", i);
    std::string delta = strFormat("delta_%d", i);
    std::string eps = strFormat("eps_%d", i);

    // The run IDs must be distinct as well to distinguish all the concurrent
    // runs from each other.
    testBuilder_.addNode(alpha, testDeviceId,
                         /*parents=*/{}, /*inputs=*/{"alphaIn"},
                         /*outputs=*/{"alphaOut"}, baseTestRunId + i, true);
    testBuilder_.addNode(beta, testDeviceId,
                         /*parents=*/{}, /*inputs=*/{"betaIn"},
                         /*outputs=*/{"betaOut"}, baseTestRunId + i, true);
    testBuilder_.addNode(gamma, testDeviceId,
                         /*parents=*/{alpha, beta},
                         /*inputs=*/{"alphaOut", "betaOut"},
                         /*outputs=*/{"deltaIn", "epsIn"}, baseTestRunId + i,
                         true);
    testBuilder_.addNode(delta, testDeviceId,
                         /*parents=*/{gamma}, /*inputs=*/{"deltaIn"},
                         /*outputs=*/{"deltaOut"}, baseTestRunId + i, true);
    testBuilder_.addNode(eps, testDeviceId,
                         /*parents=*/{gamma}, /*inputs=*/{"epsIn"},
                         /*outputs=*/{"epsOut"}, baseTestRunId + i, true);

    ExecutorTest t = testBuilder_.emitTest();
    std::thread th([&mtx, &driverCV, &threadCV, &threadsReady, &testsPassed,
                    test = std::move(t), numConcurrentRuns]() mutable {
      std::unique_lock<std::mutex> lock(mtx);
      // Increment threadsReady to mark this thread as ready to run the test.
      threadsReady++;
      // If threadsReady == numConcurrentRuns, this thread is the last to be
      // initialized and execute, so signal the driver that all threads are
      // ready.
      if (threadsReady == numConcurrentRuns) {
        driverCV.notify_one();
      }
      // Wait for the driver's signal.
      threadCV.wait(lock);
      // Unlock the mutex to let all other threads run their tests concurrently.
      lock.unlock();
      bool passed = test.run();
      lock.lock();

      if (passed) {
        testsPassed++;
      }
    });
    threads.emplace_back(std::move(th));
  }

  std::unique_lock<std::mutex> lock(mtx);
  // If threadsReady != numConcurrentRuns, not all threads are ready to run
  // their tests. Wait until they are.
  if (threadsReady != numConcurrentRuns) {
    driverCV.wait(lock, [&threadsReady, numConcurrentRuns] {
      return threadsReady == numConcurrentRuns;
    });
  }
  // Wake up all test runners.
  threadCV.notify_all();
  lock.unlock();

  // Join all threads.
  for (unsigned i = 0; i < numConcurrentRuns; ++i) {
    threads[i].join();
  }

  // All tests should pass.
  EXPECT_EQ(testsPassed, numConcurrentRuns);
}
