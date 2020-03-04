/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

#include "glow/Runtime/HostManager/HostManager.h"
#include "glow/ExecutionContext/ExecutionContext.h"

#include "gtest/gtest.h"

#include <future>
#include <thread>

using namespace glow;
using namespace glow::runtime;
using DAGNodePairTy = std::pair<std::vector<std::unique_ptr<DAGNode>>,
                                std::vector<std::unique_ptr<DAGNode>>>;

class HostManagerTest : public ::testing::Test {};
std::unique_ptr<Module> setupModule(unsigned functionCount) {
  std::unique_ptr<Module> module = glow::make_unique<Module>();
  for (unsigned int i = 0; i < functionCount; i++) {
    Function *F = module->createFunction("function" + std::to_string(i));
    auto *X = module->createPlaceholder(ElemKind::FloatTy, {3},
                                        "X" + std::to_string(i), false);
    auto *pow = F->createPow("Pow" + std::to_string(i), X, 2.0);
    F->createSave("save" + std::to_string(i), pow);
  }
  return module;
}

std::unique_ptr<HostManager>
createHostManager(llvm::StringRef backendName,
                  HostConfig hostConfig = HostConfig()) {
  std::vector<std::unique_ptr<DeviceConfig>> configs;
  auto deviceConfig = glow::make_unique<DeviceConfig>(backendName);
  configs.push_back(std::move(deviceConfig));
  std::unique_ptr<HostManager> hostManager =
      glow::make_unique<HostManager>(std::move(configs), hostConfig);
  return hostManager;
}

Error addNetwork(HostManager *manager, std::string name) {
  std::unique_ptr<Module> module = glow::make_unique<Module>();
  Function *F = module->createFunction(name);
  auto *X =
      module->createPlaceholder(ElemKind::FloatTy, {3}, "X_" + name, false);
  auto *pow = F->createPow("Pow_" + name, X, 2.0);
  F->createSave("save" + name, pow);

  // Expect this to be an Error because multiple networks with the same name
  // have been added to HostManager
  CompilationContext cctx;
  return manager->addNetwork(std::move(module), cctx);
}

void addAndRemoveNetwork(HostManager *manager, unsigned int functionNumber) {
  std::string name = "function" + std::to_string(functionNumber);
  ERR_TO_BOOL(addNetwork(manager, name));
  // Removal can return an error if the network is in the process of being
  // added. That is fine we expect it in this test.
  ERR_TO_BOOL(manager->removeNetwork(name));
}

TEST_F(HostManagerTest, newHostManager) { createHostManager("CPU"); }

TEST_F(HostManagerTest, addNetwork) {
  auto module = setupModule(6);
  auto hostManager = createHostManager("CPU");
  CompilationContext cctx;
  ASSERT_FALSE(ERR_TO_BOOL(hostManager->addNetwork(std::move(module), cctx)));
}

TEST_F(HostManagerTest, queueOverflow) {
  std::unique_ptr<Module> module = glow::make_unique<Module>();

  Function *F = module->createFunction("main");
  auto *X = module->createPlaceholder(ElemKind::FloatTy, {10}, "X", false);
  auto *pow = F->createPow("Pow1", X, 2.0);
  pow = F->createPow("Pow1", pow, 2.0);
  auto *save = F->createSave("save", pow);
  std::vector<std::unique_ptr<ExecutionContext>> contexts;
  for (int i = 0; i < 100; ++i) {
    std::unique_ptr<ExecutionContext> context =
        glow::make_unique<ExecutionContext>();
    auto *XTensor = context->getPlaceholderBindings()->allocate(X);
    XTensor->getHandle() = {1., 2., 3., 1., 2., 3., 1., 2., 3., 1.};
    context->getPlaceholderBindings()->allocate(save->getPlaceholder());
    contexts.emplace_back(std::move(context));
  }

  HostConfig hostConfig;
  hostConfig.maxQueueSize = 1;
  hostConfig.maxActiveRequests = 1;
  auto hostManager = createHostManager("CPU", hostConfig);
  CompilationContext cctx;
  ASSERT_FALSE(ERR_TO_BOOL(hostManager->addNetwork(std::move(module), cctx)));

  std::vector<std::promise<void>> requests(100);
  std::list<std::future<void>> futures;
  for (auto &r : requests) {
    futures.emplace_back(r.get_future());
  }

  for (int i = 0; i < 100; ++i) {
    auto &context = contexts[i];
    auto &request = requests[i];
    hostManager->runNetwork(
        "main", std::move(context),
        [&request](RunIdentifierTy runID, Error err,
                   std::unique_ptr<ExecutionContext> context_) {
          TRACE_EVENT_SCOPE(context_->getTraceContext(), TraceLevel::RUNTIME,
                            "HostManager::runNetwork");
          ERR_TO_BOOL(std::move(err));
          request.set_value();
        });
  }

  for (auto &f : futures) {
    f.wait();
  }
}

TEST_F(HostManagerTest, runNetwork) {
  std::unique_ptr<Module> module = glow::make_unique<Module>();
  std::unique_ptr<ExecutionContext> context =
      glow::make_unique<ExecutionContext>();

  Function *F = module->createFunction("main");
  auto *X = module->createPlaceholder(ElemKind::FloatTy, {3}, "X", false);
  auto *XTensor = context->getPlaceholderBindings()->allocate(X);
  XTensor->getHandle() = {1., 2., 3.};
  auto *pow = F->createPow("Pow1", X, 2.0);
  auto *save = F->createSave("save", pow);
  auto *saveTensor =
      context->getPlaceholderBindings()->allocate(save->getPlaceholder());

  auto hostManager = createHostManager("CPU");
  CompilationContext cctx;
  ASSERT_FALSE(ERR_TO_BOOL(hostManager->addNetwork(std::move(module), cctx)));

  std::promise<void> runNetwork;
  auto ready = runNetwork.get_future();

  std::unique_ptr<Error> runErr;
  hostManager->runNetwork("main", std::move(context),
                          [&runNetwork, &saveTensor, &context, &runErr](
                              RunIdentifierTy runID, Error err,
                              std::unique_ptr<ExecutionContext> context_) {
                            auto HX = saveTensor->getHandle();
                            EXPECT_NEAR(HX.at({0}), 1, 1E-5);
                            EXPECT_NEAR(HX.at({1}), 4, 1E-5);
                            EXPECT_NEAR(HX.at({2}), 9, 1E-5);
                            context = std::move(context_);
                            runErr = glow::make_unique<Error>(std::move(err));
                            runNetwork.set_value();
                          });

  ready.wait();
  EXPECT_FALSE(ERR_TO_BOOL(std::move(*DCHECK_NOTNULL(runErr.get()))));

  // reset runErr
  runErr = nullptr;

  std::promise<void> newRun;
  ready = newRun.get_future();
  hostManager->runNetwork("main", std::move(context),
                          [&newRun, &saveTensor, &runErr](
                              RunIdentifierTy runID, Error err,
                              std::unique_ptr<ExecutionContext> context_) {
                            auto HX = saveTensor->getHandle();
                            EXPECT_NEAR(HX.at({0}), 1, 1E-5);
                            EXPECT_NEAR(HX.at({1}), 4, 1E-5);
                            EXPECT_NEAR(HX.at({2}), 9, 1E-5);
                            runErr = glow::make_unique<Error>(std::move(err));
                            newRun.set_value();
                          });

  ready.wait();
  EXPECT_FALSE(ERR_TO_BOOL(std::move(*DCHECK_NOTNULL(runErr.get()))));
}

/// Test that HostManager properly handles concurrent add/remove requests with
/// unique network names.
TEST_F(HostManagerTest, ConcurrentAddRemoveUnique) {
  constexpr auto numThreads = 6;
  constexpr auto numItersPerThread = 20;
  auto hostManager = createHostManager("CPU");
  std::atomic<unsigned> counter{0};
  std::vector<std::thread> threads;
  for (auto i = 0; i < numThreads; ++i) {
    threads.emplace_back([&]() {
      for (auto j = 0; j < numItersPerThread; ++j) {
        addAndRemoveNetwork(hostManager.get(), ++counter);
      }
    });
  }

  for (auto &t : threads) {
    t.join();
  }
}

/// Test that HostManager properly handles concurrent add/remove requests with a
/// duplicate network name.
TEST_F(HostManagerTest, ConcurrentAddRemoveDuplicate) {
  constexpr auto numThreads = 6;
  constexpr auto numItersPerThread = 20;
  auto hostManager = createHostManager("CPU");
  std::vector<std::thread> threads;
  for (auto i = 0; i < numThreads; ++i) {
    threads.emplace_back([&]() {
      for (auto j = 0; j < numItersPerThread; ++j) {
        addAndRemoveNetwork(hostManager.get(), 0);
      }
    });
  }

  for (auto &t : threads) {
    t.join();
  }
}

/// Test that the HostManager respects it's configuration parameters.
TEST_F(HostManagerTest, ConfigureHostManager) {
  HostConfig config;
  config.maxActiveRequests = 1;
  config.maxQueueSize = 0;
  auto hostManager = createHostManager("Interpreter", std::move(config));

  EXPECT_FALSE(ERR_TO_BOOL(addNetwork(hostManager.get(), "main")));

  auto context = glow::make_unique<ExecutionContext>();
  auto context2 = glow::make_unique<ExecutionContext>();

  std::unique_ptr<Error> runErr;

  std::shared_ptr<std::mutex> lock = std::make_shared<std::mutex>();
  std::unique_lock<std::mutex> guard(*lock);

  /// Don't care a about the first one.
  hostManager->runNetwork("main", std::move(context),
                          [lock](RunIdentifierTy runID, Error err,
                                 std::unique_ptr<ExecutionContext> context_) {
                            ERR_TO_BOOL(std::move(err));
                          });

  hostManager->runNetwork(
      "main", std::move(context2),
      [&runErr](RunIdentifierTy runID, Error err,
                std::unique_ptr<ExecutionContext> context_) {
        runErr = glow::make_unique<Error>(std::move(err));
      });
  guard.unlock();
  // Don't need a future, error CB called inline.
  EXPECT_TRUE(ERR_TO_BOOL(std::move(*DCHECK_NOTNULL(runErr.get()))));
}

/// Test that the HostManager properly enqueues requests.
TEST_F(HostManagerTest, QueueTest) {
  HostConfig config;
  // Setup the hostmanager to allow 1 active and 2 queued requests for a total
  // of 3 requests in the system.
  config.maxActiveRequests = 1;
  auto hostManager = createHostManager("Interpreter", std::move(config));

  EXPECT_FALSE(ERR_TO_BOOL(addNetwork(hostManager.get(), "main")));

  auto context = glow::make_unique<ExecutionContext>();
  auto context2 = glow::make_unique<ExecutionContext>();
  auto context3 = glow::make_unique<ExecutionContext>();
  auto context4 = glow::make_unique<ExecutionContext>();
  std::promise<unsigned> run1p, run2p, run3p, dispatched;
  auto dispatchDone = dispatched.get_future();
  auto run1f = run1p.get_future();
  auto run2f = run2p.get_future();
  auto run3f = run3p.get_future();
  std::atomic<unsigned> counter{0};

  // The first will go right to dispatch since there will be no inflight
  // requests.
  hostManager->runNetwork("main", std::move(context),
                          [&run1p, &counter, &dispatchDone](
                              RunIdentifierTy runID, Error err,
                              std::unique_ptr<ExecutionContext> context) {
                            EXIT_ON_ERR(std::move(err));
                            run1p.set_value(counter++);
                            dispatchDone.wait();
                          });
  // Set the priority of the second to 1.
  hostManager->runNetwork(
      "main", std::move(context2),
      [&run2p, &counter](RunIdentifierTy runID, Error err,
                         std::unique_ptr<ExecutionContext> context) {
        EXIT_ON_ERR(std::move(err));
        run2p.set_value(counter++);
      },
      1);

  // Set the priority of the run3 to 0 so it should be first in the queue
  // after run1.
  hostManager->runNetwork(
      "main", std::move(context3),
      [&run3p, &counter](RunIdentifierTy runID, Error err,
                         std::unique_ptr<ExecutionContext> context) {
        EXIT_ON_ERR(std::move(err));
        run3p.set_value(counter++);
      },
      0);
  /// Wait for all three to finish.
  dispatched.set_value(0);
  auto res1 = run1f.get();
  auto res2 = run2f.get();
  auto res3 = run3f.get();
  // Should expect them to finish in order: 1, 3, 2. Check atomic value
  EXPECT_GT(res3, res1);
  EXPECT_GT(res2, res3);
}
