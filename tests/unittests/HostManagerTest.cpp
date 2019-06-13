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
  std::unique_ptr<Module> module = llvm::make_unique<Module>();
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
  auto deviceConfig = llvm::make_unique<DeviceConfig>(backendName);
  configs.push_back(std::move(deviceConfig));
  std::unique_ptr<HostManager> hostManager =
      llvm::make_unique<HostManager>(std::move(configs), hostConfig);
  return hostManager;
}

llvm::Error addNetwork(HostManager *manager, std::string name) {
  std::unique_ptr<Module> module = llvm::make_unique<Module>();
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
  errToBool(addNetwork(manager, name));
  EXPECT_FALSE(errToBool(manager->removeNetwork(name)));
}

TEST_F(HostManagerTest, newHostManager) { createHostManager("CPU"); }

TEST_F(HostManagerTest, addNetwork) {
  auto module = setupModule(6);
  auto hostManager = createHostManager("CPU");
  CompilationContext cctx;
  ASSERT_FALSE(errToBool(hostManager->addNetwork(std::move(module), cctx)));
}

TEST_F(HostManagerTest, runNetwork) {
  std::unique_ptr<Module> module = llvm::make_unique<Module>();
  std::unique_ptr<ExecutionContext> context =
      llvm::make_unique<ExecutionContext>();

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
  ASSERT_FALSE(errToBool(hostManager->addNetwork(std::move(module), cctx)));

  std::promise<void> runNetwork;
  auto ready = runNetwork.get_future();

  std::unique_ptr<llvm::Error> runErr;
  hostManager->runNetwork("main", std::move(context),
                          [&runNetwork, &saveTensor, &context, &runErr](
                              RunIdentifierTy runID, llvm::Error err,
                              std::unique_ptr<ExecutionContext> context_) {
                            auto HX = saveTensor->getHandle();
                            EXPECT_NEAR(HX.at({0}), 1, 1E-5);
                            EXPECT_NEAR(HX.at({1}), 4, 1E-5);
                            EXPECT_NEAR(HX.at({2}), 9, 1E-5);
                            context = std::move(context_);
                            runErr =
                                llvm::make_unique<llvm::Error>(std::move(err));
                            runNetwork.set_value();
                          });

  ready.wait();
  EXPECT_FALSE(errToBool(std::move(*DCHECK_NOTNULL(runErr.get()))));

  // reset runErr
  runErr = nullptr;

  std::promise<void> newRun;
  ready = newRun.get_future();
  hostManager->runNetwork("main", std::move(context),
                          [&newRun, &saveTensor, &runErr](
                              RunIdentifierTy runID, llvm::Error err,
                              std::unique_ptr<ExecutionContext> context_) {
                            auto HX = saveTensor->getHandle();
                            EXPECT_NEAR(HX.at({0}), 1, 1E-5);
                            EXPECT_NEAR(HX.at({1}), 4, 1E-5);
                            EXPECT_NEAR(HX.at({2}), 9, 1E-5);
                            runErr =
                                llvm::make_unique<llvm::Error>(std::move(err));
                            newRun.set_value();
                          });

  ready.wait();
  EXPECT_FALSE(errToBool(std::move(*DCHECK_NOTNULL(runErr.get()))));
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
  auto hostManager = createHostManager("Interpreter", std::move(config));

  EXPECT_FALSE(errToBool(addNetwork(hostManager.get(), "main")));

  auto context = llvm::make_unique<ExecutionContext>();
  auto context2 = llvm::make_unique<ExecutionContext>();

  std::unique_ptr<llvm::Error> runErr;

  std::shared_ptr<std::mutex> lock = std::make_shared<std::mutex>();
  std::unique_lock<std::mutex> guard(*lock);

  /// Don't care a about the first one.
  hostManager->runNetwork("main", std::move(context),
                          [lock](RunIdentifierTy runID, llvm::Error err,
                                 std::unique_ptr<ExecutionContext> context_) {
                            errToBool(std::move(err));
                            std::unique_lock<std::mutex> guard(*lock);
                          });

  hostManager->runNetwork(
      "main", std::move(context2),
      [&runErr](RunIdentifierTy runID, llvm::Error err,
                std::unique_ptr<ExecutionContext> context_) {
        runErr = llvm::make_unique<llvm::Error>(std::move(err));
      });

  // Don't need a future, error CB called inline.
  EXPECT_TRUE(errToBool(std::move(*DCHECK_NOTNULL(runErr.get()))));
  guard.unlock();
}
