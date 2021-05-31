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
#include "BackendTestUtils.h"

#include "glow/ExecutionContext/ExecutionContext.h"
#include "glow/Flags/Flags.h"
#include "glow/Runtime/HostManager/HostManager.h"

#include "gtest/gtest.h"

#include <future>
#include <thread>

using namespace glow;
using namespace glow::runtime;
using DAGNodePairTy = std::pair<std::vector<std::unique_ptr<DAGNode>>,
                                std::vector<std::unique_ptr<DAGNode>>>;

class HostManagerTest : public ::testing::TestWithParam<std::string> {
public:
  void SetUp() override { backendName_ = GetParam(); }
  std::string backendName_;
};

std::vector<std::unique_ptr<DeviceConfig>>
generateConfigs(std::string backendName, unsigned numConfigs = 1) {
  std::vector<std::unique_ptr<DeviceConfig>> configs;
  for (unsigned i = 0; i < numConfigs; i++) {
    auto deviceConfig = glow::make_unique<DeviceConfig>(backendName);
    deviceConfig->deviceID = i;
    configs.push_back(std::move(deviceConfig));
  }
  return configs;
}

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
  std::vector<std::unique_ptr<DeviceConfig>> configs =
      generateConfigs(std::string(backendName), 1);
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

TEST_P(HostManagerTest, newHostManager) {
  CHECK_IF_ENABLED();
  createHostManager(backendName_);
}

TEST_P(HostManagerTest, addNetwork) {
  CHECK_IF_ENABLED();
  auto module = setupModule(6);
  auto hostManager = createHostManager(backendName_);
  CompilationContext cctx;
  ASSERT_FALSE(ERR_TO_BOOL(hostManager->addNetwork(std::move(module), cctx)));
}

TEST_P(HostManagerTest, queueOverflow) {
  CHECK_IF_ENABLED();
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
  auto hostManager = createHostManager(backendName_, hostConfig);
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

TEST_P(HostManagerTest, runNetwork) {
  CHECK_IF_ENABLED();
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

  auto hostManager = createHostManager(backendName_);
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
TEST_P(HostManagerTest, ConcurrentAddRemoveUnique) {
  CHECK_IF_ENABLED();
  constexpr auto numThreads = 6;
  constexpr auto numItersPerThread = 20;
  auto hostManager = createHostManager(backendName_);
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
TEST_P(HostManagerTest, ConcurrentAddRemoveDuplicate) {
  CHECK_IF_ENABLED();
  constexpr auto numThreads = 6;
  constexpr auto numItersPerThread = 20;
  auto hostManager = createHostManager(backendName_);
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

/// Run several requests concurrently.
TEST_P(HostManagerTest, runNetworkConcurrent) {
  CHECK_IF_ENABLED();
  std::unique_ptr<Module> module = glow::make_unique<Module>();

  Function *F = module->createFunction("main");
  auto *X = module->createPlaceholder(ElemKind::FloatTy, {3}, "X", false);
  auto *pow = F->createPow("Pow1", X, 2.0);
  F->createSave("save", pow);
  auto *savePH = module->getPlaceholderByNameSlow("save");

  auto hostManager = createHostManager(backendName_);
  CompilationContext cctx;

  ASSERT_FALSE(ERR_TO_BOOL(hostManager->addNetwork(std::move(module), cctx)));

  std::vector<std::future<void>> ready;
  for (int i = 0; i < 50; i++) {
    auto runNetwork = std::make_shared<std::promise<void>>();
    ready.push_back(runNetwork->get_future());
    std::unique_ptr<ExecutionContext> context =
        glow::make_unique<ExecutionContext>();
    auto *XTensor = context->getPlaceholderBindings()->allocate(X);
    XTensor->getHandle() = {1., 2., 3.};
    auto *saveTensor = context->getPlaceholderBindings()->allocate(savePH);
    hostManager->runNetwork(
        "main", std::move(context),
        [runNetwork, saveTensor](RunIdentifierTy runID, Error err,
                                 std::unique_ptr<ExecutionContext> context_) {
          auto HX = saveTensor->getHandle();
          EXPECT_NEAR(HX.at({0}), 1, 1E-5);
          EXPECT_NEAR(HX.at({1}), 4, 1E-5);
          EXPECT_NEAR(HX.at({2}), 9, 1E-5);
          EXPECT_FALSE(std::move(err));
          runNetwork->set_value();
        });
  }

  for (auto &r : ready) {
    r.wait();
  }
}

TEST_P(HostManagerTest, testSaturateHost) {
  CHECK_IF_ENABLED();
  std::unique_ptr<Module> module = glow::make_unique<Module>();

  Function *F = module->createFunction("main");
  auto *X = module->createPlaceholder(ElemKind::FloatTy, {3}, "X", false);
  auto *pow = F->createPow("Pow1", X, 2.0);
  F->createSave("save", pow);
  auto *savePH = module->getPlaceholderByNameSlow("save");

  std::vector<std::unique_ptr<DeviceConfig>> configs =
      generateConfigs(backendName_, 2);
  std::unique_ptr<HostManager> hostManager =
      glow::make_unique<HostManager>(std::move(configs), HostConfig());

  CompilationContext cctx;
  cctx.saturateHost = true;
  ASSERT_FALSE(ERR_TO_BOOL(hostManager->addNetwork(std::move(module), cctx)));

  std::vector<std::future<void>> ready;
  for (int i = 0; i < 50; i++) {
    auto runNetwork = std::make_shared<std::promise<void>>();
    ready.push_back(runNetwork->get_future());
    std::unique_ptr<ExecutionContext> context =
        glow::make_unique<ExecutionContext>();
    auto *XTensor = context->getPlaceholderBindings()->allocate(X);
    XTensor->getHandle() = {1., 2., 3.};
    auto *saveTensor = context->getPlaceholderBindings()->allocate(savePH);
    hostManager->runNetwork(
        "main", std::move(context),
        [runNetwork, saveTensor](RunIdentifierTy, Error err,
                                 std::unique_ptr<ExecutionContext>) {
          auto HX = saveTensor->getHandle();
          EXPECT_NEAR(HX.at({0}), 1, 1E-5);
          EXPECT_NEAR(HX.at({1}), 4, 1E-5);
          EXPECT_NEAR(HX.at({2}), 9, 1E-5);
          EXPECT_FALSE(std::move(err));
          runNetwork->set_value();
        });
  }

  for (auto &r : ready) {
    r.wait();
  }
}

/// Test that the HostManager respects it's configuration parameters.
TEST_P(HostManagerTest, ConfigureHostManager) {
  CHECK_IF_ENABLED();
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
TEST_P(HostManagerTest, QueueTest) {
  CHECK_IF_ENABLED();
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

/// Test that the enabling partition replication through user defined
/// partitioning works.
TEST_P(HostManagerTest, testPartitionConfigReplication) {
  CHECK_IF_ENABLED();
  std::unique_ptr<Module> module = glow::make_unique<Module>();
  std::unique_ptr<ExecutionContext> context =
      glow::make_unique<ExecutionContext>();

  Function *F = module->createFunction("main");
  auto *X = module->createPlaceholder(ElemKind::FloatTy, {3}, "X", false);
  auto *XTensor = context->getPlaceholderBindings()->allocate(X);
  XTensor->getHandle() = {1., 2., 3.};
  auto *pow = F->createPow("Pow", X, 2.0);
  auto *save = F->createSave("save", pow);
  auto savePH = save->getPlaceholder();

  std::vector<std::unique_ptr<DeviceConfig>> configs =
      generateConfigs(backendName_, 2);
  std::unique_ptr<HostManager> hostManager =
      glow::make_unique<HostManager>(std::move(configs), HostConfig());
  CompilationContext cctx;

  // Setup forced partitioning.
  PartitionConfig partitionConfig;
  partitionConfig.funcName = "main";
  partitionConfig.numOfPartitions = 2;
  partitionConfig.backendNames = {backendName_, backendName_};
  partitionConfig.partitionNames = {"p0", "p1"};
  partitionConfig.nodeToPartition = {{"Pow", 0}, {"save", 3}};
  partitionConfig.logicalIDs = {{0}, {1}};
  partitionConfig.replicationCount[0] = 2;
  cctx.partitionConfig = &partitionConfig;

  ASSERT_FALSE(ERR_TO_BOOL(hostManager->addNetwork(std::move(module), cctx)));

  std::vector<std::future<void>> ready;
  for (int i = 0; i < 50; i++) {
    auto runNetwork = std::make_shared<std::promise<void>>();
    ready.push_back(runNetwork->get_future());
    std::unique_ptr<ExecutionContext> context =
        glow::make_unique<ExecutionContext>();
    auto *XTensor = context->getPlaceholderBindings()->allocate(X);
    XTensor->getHandle() = {1., 2., 3.};
    auto *saveTensor = context->getPlaceholderBindings()->allocate(savePH);
    hostManager->runNetwork(
        "main", std::move(context),
        [runNetwork, saveTensor](RunIdentifierTy runID, Error err,
                                 std::unique_ptr<ExecutionContext> context_) {
          auto HX = saveTensor->getHandle();
          EXPECT_NEAR(HX.at({0}), 1, 1E-5);
          EXPECT_NEAR(HX.at({1}), 4, 1E-5);
          EXPECT_NEAR(HX.at({2}), 9, 1E-5);
          EXPECT_FALSE(std::move(err));
          runNetwork->set_value();
        });
  }

  for (auto &r : ready) {
    r.wait();
  }
}

/// Test replication for a single partition network.
TEST_P(HostManagerTest, testSinglePartitionReplication) {
  CHECK_IF_ENABLED();
  std::unique_ptr<Module> module = glow::make_unique<Module>();
  std::unique_ptr<ExecutionContext> context =
      glow::make_unique<ExecutionContext>();

  Function *F = module->createFunction("main");
  auto *X = module->createPlaceholder(ElemKind::FloatTy, {3}, "X", false);
  auto *XTensor = context->getPlaceholderBindings()->allocate(X);
  XTensor->getHandle() = {1., 2., 3.};
  auto *pow = F->createPow("Pow1", X, 2.0);
  auto *save = F->createSave("save", pow);
  auto *savePH = save->getPlaceholder();

  auto hostManager = createHostManager(backendName_);
  CompilationContext cctx;
  cctx.replicationCount = 2;
  ASSERT_FALSE(ERR_TO_BOOL(hostManager->addNetwork(std::move(module), cctx)));

  std::vector<std::future<void>> ready;
  for (int i = 0; i < 50; i++) {
    auto runNetwork = std::make_shared<std::promise<void>>();
    ready.push_back(runNetwork->get_future());
    std::unique_ptr<ExecutionContext> context =
        glow::make_unique<ExecutionContext>();
    auto *XTensor = context->getPlaceholderBindings()->allocate(X);
    XTensor->getHandle() = {1., 2., 3.};
    auto *saveTensor = context->getPlaceholderBindings()->allocate(savePH);
    hostManager->runNetwork(
        "main", std::move(context),
        [runNetwork, saveTensor](RunIdentifierTy runID, Error err,
                                 std::unique_ptr<ExecutionContext> context_) {
          auto HX = saveTensor->getHandle();
          EXPECT_NEAR(HX.at({0}), 1, 1E-5);
          EXPECT_NEAR(HX.at({1}), 4, 1E-5);
          EXPECT_NEAR(HX.at({2}), 9, 1E-5);
          EXPECT_FALSE(std::move(err));
          runNetwork->set_value();
        });
  }

  for (auto &r : ready) {
    r.wait();
  }
}

// This test creates a network that is split into four partitions. P0,P1,P2,P3
// and three devices D0,D1,D2. P0 is loaded on D0, P1 and P2 are loaded on D2
// and P3 is loaded on D2. This test then enables both DRT and P2P
// optimizations. We then run the network twice to test the alternating static
// assignments.
TEST_P(HostManagerTest, testStaticAssignmentP2PandDRT) {
  CHECK_IF_ENABLED();
  std::unique_ptr<Module> module = glow::make_unique<Module>();
  std::unique_ptr<ExecutionContext> context =
      glow::make_unique<ExecutionContext>();

  Function *F = module->createFunction("main");
  auto *X = module->createPlaceholder(ElemKind::FloatTy, {3}, "X", false);
  auto *XTensor = context->getPlaceholderBindings()->allocate(X);
  XTensor->getHandle() = {1., 2., 3.};
  auto *pow = F->createPow("Pow1", X, 2.0);
  auto pow2 = F->createPow("Pow2", pow, 2.0);
  auto pow3 = F->createPow("Pow3", pow2, 1.0);
  auto *save = F->createSave("save", pow3);
  auto *saveTensor =
      context->getPlaceholderBindings()->allocate(save->getPlaceholder());

  std::vector<std::unique_ptr<DeviceConfig>> configs =
      generateConfigs(backendName_, 3);
  std::unique_ptr<HostManager> hostManager =
      glow::make_unique<HostManager>(std::move(configs), HostConfig());
  CompilationContext cctx;
  cctx.enableP2P = true;
  cctx.enableDRT = true;

  // Setup forced partitioning.
  PartitionConfig partitionConfig;
  partitionConfig.funcName = "main";
  partitionConfig.numOfPartitions = 4;
  partitionConfig.backendNames = {backendName_, backendName_, backendName_,
                                  backendName_};
  partitionConfig.partitionNames = {"p0", "p1", "p2", "p3"};
  partitionConfig.nodeToPartition = {
      {"Pow1", 0},    {"Pow2", 1},    {"Pow3", 2}, {"Pow1__1", 0},
      {"Pow2__1", 1}, {"Pow3__1", 2}, {"save", 3}, {"save_save", 3}};
  partitionConfig.logicalIDs = {{0}, {1}, {1}, {2}};
  cctx.partitionConfig = &partitionConfig;

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
                            EXPECT_NEAR(HX.at({1}), 16, 1E-5);
                            EXPECT_NEAR(HX.at({2}), 81, 1E-5);
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
                            EXPECT_NEAR(HX.at({1}), 16, 1E-5);
                            EXPECT_NEAR(HX.at({2}), 81, 1E-5);
                            runErr = glow::make_unique<Error>(std::move(err));
                            newRun.set_value();
                          });

  ready.wait();
  EXPECT_FALSE(ERR_TO_BOOL(std::move(*DCHECK_NOTNULL(runErr.get()))));
}

// This test creates a network that is split into four partitions. P0,P1,P2,P3
// and three devices D0,D1,D2. P0 is loaded on D0, P1 and P2 are loaded on D2
// and P3 is loaded on D2. This test then enables the DRT optimization without
// P2P. We then run the network twice to test the alternating static
// assignments.
TEST_P(HostManagerTest, testStaticAssignmentDeviceResidentTensorOnly) {
  CHECK_IF_ENABLED();
  std::unique_ptr<Module> module = glow::make_unique<Module>();
  std::unique_ptr<ExecutionContext> context =
      glow::make_unique<ExecutionContext>();

  Function *F = module->createFunction("main");
  auto *X = module->createPlaceholder(ElemKind::FloatTy, {3}, "X", false);
  auto *XTensor = context->getPlaceholderBindings()->allocate(X);
  XTensor->getHandle() = {1., 2., 3.};
  auto *pow = F->createPow("Pow1", X, 2.0);
  auto pow2 = F->createPow("Pow2", pow, 2.0);
  auto pow3 = F->createPow("Pow3", pow2, 1.0);
  auto *save = F->createSave("save", pow3);
  auto *saveTensor =
      context->getPlaceholderBindings()->allocate(save->getPlaceholder());

  std::vector<std::unique_ptr<DeviceConfig>> configs =
      generateConfigs(backendName_, 3);
  std::unique_ptr<HostManager> hostManager =
      glow::make_unique<HostManager>(std::move(configs), HostConfig());
  CompilationContext cctx;
  cctx.enableDRT = true;

  // Setup forced partitioning.
  PartitionConfig partitionConfig;
  partitionConfig.funcName = "main";
  partitionConfig.numOfPartitions = 4;
  partitionConfig.backendNames = {backendName_, backendName_, backendName_,
                                  backendName_};
  partitionConfig.partitionNames = {"p0", "p1", "p2", "p3"};
  partitionConfig.nodeToPartition = {
      {"Pow1", 0},    {"Pow2", 1},    {"Pow3", 2}, {"Pow1__1", 0},
      {"Pow2__1", 1}, {"Pow3__1", 2}, {"save", 3}, {"save_save", 3}};
  partitionConfig.logicalIDs = {{0}, {1}, {1}, {2}};
  cctx.partitionConfig = &partitionConfig;

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
                            EXPECT_NEAR(HX.at({1}), 16, 1E-5);
                            EXPECT_NEAR(HX.at({2}), 81, 1E-5);
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
                            EXPECT_NEAR(HX.at({1}), 16, 1E-5);
                            EXPECT_NEAR(HX.at({2}), 81, 1E-5);
                            runErr = glow::make_unique<Error>(std::move(err));
                            newRun.set_value();
                          });

  ready.wait();
  EXPECT_FALSE(ERR_TO_BOOL(std::move(*DCHECK_NOTNULL(runErr.get()))));
}

// This test creates a network that is split into four partitions. P0,P1,P2,P3
// and three devices D0,D1,D2. P0 is loaded on D0, P1 and P2 are loaded on D2
// and P3 is loaded on D2. This test then enables the P2P optimization without
// DRT. We then run the network twice to test the alternating static
// assignments.
TEST_P(HostManagerTest, testStaticAssignmentP2POnly) {
  CHECK_IF_ENABLED();
  std::unique_ptr<Module> module = glow::make_unique<Module>();
  std::unique_ptr<ExecutionContext> context =
      glow::make_unique<ExecutionContext>();

  Function *F = module->createFunction("main");
  auto *X = module->createPlaceholder(ElemKind::FloatTy, {3}, "X", false);
  auto *XTensor = context->getPlaceholderBindings()->allocate(X);
  XTensor->getHandle() = {1., 2., 3.};
  auto *pow = F->createPow("Pow1", X, 2.0);
  auto pow2 = F->createPow("Pow2", pow, 2.0);
  auto pow3 = F->createPow("Pow3", pow2, 1.0);
  auto *save = F->createSave("save", pow3);
  auto *saveTensor =
      context->getPlaceholderBindings()->allocate(save->getPlaceholder());

  std::vector<std::unique_ptr<DeviceConfig>> configs =
      generateConfigs(backendName_, 3);
  std::unique_ptr<HostManager> hostManager =
      glow::make_unique<HostManager>(std::move(configs), HostConfig());
  CompilationContext cctx;
  cctx.enableP2P = true;

  // Setup forced partitioning.
  PartitionConfig partitionConfig;
  partitionConfig.funcName = "main";
  partitionConfig.numOfPartitions = 4;
  partitionConfig.backendNames = {backendName_, backendName_, backendName_,
                                  backendName_};
  partitionConfig.partitionNames = {"p0", "p1", "p2", "p3"};
  partitionConfig.nodeToPartition = {
      {"Pow1", 0},    {"Pow2", 1},    {"Pow3", 2}, {"Pow1__1", 0},
      {"Pow2__1", 1}, {"Pow3__1", 2}, {"save", 3}, {"save_save", 3}};
  partitionConfig.logicalIDs = {{0}, {1}, {1}, {2}};
  cctx.partitionConfig = &partitionConfig;

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
                            EXPECT_NEAR(HX.at({1}), 16, 1E-5);
                            EXPECT_NEAR(HX.at({2}), 81, 1E-5);
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
                            EXPECT_NEAR(HX.at({1}), 16, 1E-5);
                            EXPECT_NEAR(HX.at({2}), 81, 1E-5);
                            runErr = glow::make_unique<Error>(std::move(err));
                            newRun.set_value();
                          });

  ready.wait();
  EXPECT_FALSE(ERR_TO_BOOL(std::move(*DCHECK_NOTNULL(runErr.get()))));
}

// This test creates a network that is split into two partitions. P0,P1. P0 is
// loaded on one device, P1 is loaded on two devices. This test then enables
// static assignment which allows for P2P testing. We then run the network
// multiple requests concurrently.
TEST_P(HostManagerTest, testStaticAssignmentP2PandDRTConcurrent) {
  CHECK_IF_ENABLED();
  std::unique_ptr<Module> module = glow::make_unique<Module>();

  Function *F = module->createFunction("main");
  auto *X = module->createPlaceholder(ElemKind::FloatTy, {3}, "X", false);
  auto *pow = F->createPow("Pow1", X, 2.0);
  F->createSave("save", pow);
  auto *savePH = module->getPlaceholderByNameSlow("save");

  std::vector<std::unique_ptr<DeviceConfig>> configs =
      generateConfigs(backendName_, 3);
  std::unique_ptr<HostManager> hostManager =
      glow::make_unique<HostManager>(std::move(configs), HostConfig());
  CompilationContext cctx;
  cctx.enableDRT = true;
  cctx.enableP2P = true;

  // Setup forced partitioning.
  PartitionConfig partitionConfig;
  partitionConfig.funcName = "main";
  partitionConfig.numOfPartitions = 2;
  partitionConfig.backendNames = {backendName_, backendName_};
  partitionConfig.partitionNames = {"p0", "p1"};
  partitionConfig.nodeToPartition = {{"Pow1", 0}, {"save", 1}};
  partitionConfig.logicalIDs = {{0}, {1, 2}};
  cctx.partitionConfig = &partitionConfig;

  ASSERT_FALSE(ERR_TO_BOOL(hostManager->addNetwork(std::move(module), cctx)));

  std::vector<std::future<void>> ready;
  for (int i = 0; i < 50; i++) {
    auto runNetwork = std::make_shared<std::promise<void>>();
    ready.push_back(runNetwork->get_future());
    std::unique_ptr<ExecutionContext> context =
        glow::make_unique<ExecutionContext>();
    auto *XTensor = context->getPlaceholderBindings()->allocate(X);
    XTensor->getHandle() = {1., 2., 3.};
    auto *saveTensor = context->getPlaceholderBindings()->allocate(savePH);
    hostManager->runNetwork(
        "main", std::move(context),
        [runNetwork, saveTensor](RunIdentifierTy runID, Error err,
                                 std::unique_ptr<ExecutionContext> context_) {
          auto HX = saveTensor->getHandle();
          EXPECT_NEAR(HX.at({0}), 1, 1E-5);
          EXPECT_NEAR(HX.at({1}), 4, 1E-5);
          EXPECT_NEAR(HX.at({2}), 9, 1E-5);
          EXPECT_FALSE(std::move(err));
          runNetwork->set_value();
        });
  }

  for (auto &r : ready) {
    r.wait();
  }
}

/// This tests that the HostMangaer registry works and is able to report what
/// devices a network is loaded on.
TEST_P(HostManagerTest, testHostManagerRegistry) {
  CHECK_IF_ENABLED();
  std::unique_ptr<Module> module = glow::make_unique<Module>();

  Function *F = module->createFunction("main");
  auto *X = module->createPlaceholder(ElemKind::FloatTy, {3}, "X", false);
  auto *pow = F->createPow("Pow1", X, 2.0);
  F->createSave("save", pow);
  module->getPlaceholderByNameSlow("save");

  std::unique_ptr<Module> module2 = glow::make_unique<Module>();

  Function *F2 = module2->createFunction("main2");
  auto *X2 = module2->createPlaceholder(ElemKind::FloatTy, {3}, "X2", false);
  auto *pow2 = F2->createPow("Pow2", X2, 2.0);
  F2->createSave("save2", pow2);
  module2->getPlaceholderByNameSlow("save2");

  std::vector<std::unique_ptr<DeviceConfig>> configs =
      generateConfigs(backendName_, 2);
  std::unique_ptr<HostManager> hostManager =
      glow::make_unique<HostManager>(std::move(configs), HostConfig());

  CompilationContext cctx;
  cctx.saturateHost = true;
  ASSERT_FALSE(ERR_TO_BOOL(hostManager->addNetwork(std::move(module), cctx)));
  cctx.saturateHost = false;
  ASSERT_FALSE(ERR_TO_BOOL(hostManager->addNetwork(std::move(module2), cctx)));
  glow::runtime::ManagerRegistry()->registerHostManager(hostManager.get());
  auto testHM = glow::runtime::ManagerRegistry()->getHostManager();
  auto loading = testHM->getDevicePartitionMapping("main");
  auto loading2 = testHM->getDevicePartitionMapping("main2");
  EXPECT_EQ(loading["main"].size(), 2);
  EXPECT_EQ(loading2["main2"].size(), 1);
}

TEST_P(HostManagerTest, testTimeout) {
  CHECK_IF_ENABLED();

  if (backendName_ == "NNPI") {
    // Skip this test if running on ICEREF, since we want to test the device
    // timeout.
    auto useInfAPI = getenv("USE_INF_API");
    if (!useInfAPI || strcmp(useInfAPI, "1")) {
      GTEST_SKIP();
    }
    // Set the timeout to very short so we fail intentionally.
    glow::runtime::flags::NNPITimeoutMs = 1;
  }

  std::unique_ptr<Module> module = glow::make_unique<Module>();
  std::unique_ptr<ExecutionContext> context =
      glow::make_unique<ExecutionContext>();

  Function *F = module->createFunction("main");
  auto *X = module->createPlaceholder(ElemKind::FloatTy, {3}, "X", false);
  auto *XTensor = context->getPlaceholderBindings()->allocate(X);
  XTensor->getHandle() = {1., 2., 3.};
  auto *pow = F->createPow("Poww", X, 2.0);
  for (unsigned i = 0; i < 1000; i++) {
    pow = F->createPow("pow" + std::to_string(i), pow, 1.0);
  }
  auto *save = F->createSave("save", pow);
  context->getPlaceholderBindings()->allocate(save->getPlaceholder());

  auto hostManager = createHostManager(backendName_);

  CompilationContext cctx;
  ASSERT_FALSE(ERR_TO_BOOL(hostManager->addNetwork(std::move(module), cctx)));

  std::promise<void> runNetwork;
  auto ready = runNetwork.get_future();

  std::unique_ptr<Error> runErr;
  hostManager->runNetwork("main", std::move(context),
                          [&runNetwork, &context, &runErr](
                              RunIdentifierTy runID, Error err,
                              std::unique_ptr<ExecutionContext> context_) {
                            context = std::move(context_);
                            runErr = glow::make_unique<Error>(std::move(err));
                            runNetwork.set_value();
                          });

  ready.wait();
  EXPECT_TRUE(ERR_TO_BOOL(std::move(*DCHECK_NOTNULL(runErr.get()))));
}

INSTANTIATE_BACKEND_TEST(HostManagerTest);
