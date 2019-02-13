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
#include "glow/Graph/Context.h"

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

std::unique_ptr<HostManager> createHostManager(llvm::StringRef name,
                                               BackendKind kind) {
  std::vector<DeviceConfig> configs;
  auto config = DeviceConfig();
  config.deviceName = name;
  config.backendKind = kind;
  configs.push_back(config);
  std::unique_ptr<HostManager> hostManager =
      llvm::make_unique<HostManager>(configs);
  return hostManager;
}

void addAndRemoveNetwork(HostManager *manager, unsigned int functionNumber) {
  std::unique_ptr<Module> module = llvm::make_unique<Module>();
  Function *F =
      module->createFunction("function" + std::to_string(functionNumber));
  auto *X = module->createPlaceholder(
      ElemKind::FloatTy, {3}, "X" + std::to_string(functionNumber), false);
  auto *pow = F->createPow("Pow" + std::to_string(functionNumber), X, 2.0);
  F->createSave("save" + std::to_string(functionNumber), pow);

  manager->addNetwork(module.get());
  manager->removeNetwork("function" + std::to_string(functionNumber));
}

TEST_F(HostManagerTest, newHostManager) {
  createHostManager("CPU0", BackendKind::CPU);
}

TEST_F(HostManagerTest, addNetwork) {
  auto mod = setupModule(6);
  auto hostManager = createHostManager("CPU0", BackendKind::CPU);
  hostManager->addNetwork(mod.get());
}

TEST_F(HostManagerTest, runNetwork) {
  Module mod;
  std::unique_ptr<Context> ctx = llvm::make_unique<Context>();

  Function *F = mod.createFunction("main");
  auto *X = mod.createPlaceholder(ElemKind::FloatTy, {3}, "X", false);
  auto *XTensor = ctx->allocate(X);
  XTensor->getHandle() = {1., 2., 3.};
  auto *pow = F->createPow("Pow1", X, 2.0);
  auto *save = F->createSave("save", pow);
  auto *saveTensor = ctx->allocate(save->getPlaceholder());

  auto hostManager = createHostManager("CPU0", BackendKind::CPU);
  hostManager->addNetwork(&mod);
  std::promise<ResultCode> runNetwork;
  auto ready = runNetwork.get_future();
  hostManager->runNetwork(
      "main", std::move(ctx),
      [&runNetwork, &saveTensor, &ctx](RunIdentifierTy runID, ResultCode result,
                                       std::unique_ptr<Context> context) {
        auto HX = saveTensor->getHandle();
        EXPECT_NEAR(HX.at({0}), 1, 1E-5);
        EXPECT_NEAR(HX.at({1}), 4, 1E-5);
        EXPECT_NEAR(HX.at({2}), 9, 1E-5);
        ctx = std::move(context);
        runNetwork.set_value(result);
      });
  auto result = ready.get();
  EXPECT_EQ(result, ResultCode::Executed);

  std::promise<ResultCode> newRun;
  ready = newRun.get_future();
  hostManager->runNetwork(
      "main", std::move(ctx),
      [&newRun, &saveTensor](RunIdentifierTy runID, ResultCode result,
                             std::unique_ptr<Context> context) {
        auto HX = saveTensor->getHandle();
        EXPECT_NEAR(HX.at({0}), 1, 1E-5);
        EXPECT_NEAR(HX.at({1}), 4, 1E-5);
        EXPECT_NEAR(HX.at({2}), 9, 1E-5);
        newRun.set_value(result);
      });
  result = ready.get();
  EXPECT_EQ(result, ResultCode::Executed);
}
// This test is currently disabled, ASAN complains because the compiled function
// can be freed out from under the DeviceManager. There are plans to moved
// ownership of the compiledFunction to the deviceManager. Once that is done
// this won't be an issue. Or we can add a callback to evictNetwork.

// TEST_F(HostManagerTest, ConcurrentAddRemove) {
//   constexpr auto numThreads = 6;
//   constexpr auto numItersPerThread = 20;
//   auto hostManager = createHostManager("CPU0", BackendKind::CPU);
//   uint counter = 0;
//   std::vector<std::thread> threads;
//   for (auto i = 0; i < numThreads; ++i) {
//     threads.emplace_back([&]() {
//       for (auto j = 0; j < numItersPerThread; ++j) {
//         addAndRemoveNetwork(hostManager.get(), counter);
//         counter++;
//       }
//     });
//   }

//   for (auto &t : threads) {
//     t.join();
//   }
// }
