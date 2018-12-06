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

#include "CPUDeviceManager.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"

#include "gtest/gtest.h"

#include <chrono>
#include <future>

using namespace glow;
using namespace std::chrono_literals;

std::pair<std::promise<DeviceNetworkID>, std::future<DeviceNetworkID>>
getFutureHelper() {
  std::promise<DeviceNetworkID> promise;
  auto future = promise.get_future();
  return std::make_pair(std::move(promise), std::move(future));
}

void addNetworkCallbackHelper(std::promise<DeviceNetworkID> &promise,
                              DeviceNetworkID id, ResultCode result) {
  promise.set_value(result == READY ? id : 0);
}

TEST(CPUDeviceManagerTest, Basic) {
  std::unique_ptr<Module> module = std::make_unique<Module>();
  std::unique_ptr<Context> ctx = std::make_unique<Context>();

  Function *F = module->createFunction("main");

  auto *input = module->createPlaceholder(ElemKind::FloatTy, {1, 32, 32, 3},
                                          "input", false);

  auto *ex =
      module->createPlaceholder(ElemKind::Int64ITy, {1, 1}, "exp", false);

  auto *CV0 = F->createConv(*ctx, "conv1", input, 16, 5, 1, 2, 1);
  auto *RL0 = F->createRELU("relu1", CV0);
  auto *MP0 = F->createMaxPool("pool1", RL0, 2, 2, 0);

  auto *CV1 = F->createConv(*ctx, "conv2", MP0, 20, 5, 1, 2, 1);
  auto *RL1 = F->createRELU("relu2", CV1);
  auto *MP1 = F->createMaxPool("pool2", RL1, 2, 2, 0);

  auto *CV2 = F->createConv(*ctx, "conv3", MP1, 20, 5, 1, 2, 1);
  auto *RL2 = F->createRELU("relu3", CV2);
  auto *MP2 = F->createMaxPool("pool3", RL2, 2, 2, 0);

  auto *FCL1 = F->createFullyConnected(*ctx, "fc", MP2, 10);
  auto *RL3 = F->createRELU("relu4", FCL1);
  auto *SM = F->createSoftMax("sm", RL3, ex);
  auto *S = F->createSave("ret", SM);

  CPUDeviceManager cpuCoreDevice;
  cpuCoreDevice.init();

  std::promise<DeviceNetworkID> promise;
  std::future<DeviceNetworkID> future;
  std::tie(promise, future) = getFutureHelper();

  cpuCoreDevice.addNetwork(1, std::move(module),
                           [&promise](DeviceNetworkID id, ResultCode result) {
                             addNetworkCallbackHelper(promise, id, result);
                           });

  future.wait_for(2s);
  EXPECT_EQ(future.get(), 1);

  ctx->allocate(input);
  ctx->allocate(ex);
  ctx->allocate(S->getPlaceholder());

  Tensor inputs(ElemKind::FloatTy, {1, 32, 32, 3});
  updateInputPlaceholders(*ctx, {input}, {&inputs});

  std::tie(promise, future) = getFutureHelper();
  cpuCoreDevice.runFunction(
      1, "main", std::move(ctx),
      [&promise, &ctx](ResultCode result, std::unique_ptr<Context> ctx_) {
        if (result == EXECUTED) {
          ctx = std::move(ctx_);
          promise.set_value(1);
        } else {
          promise.set_exception(
              std::make_exception_ptr(std::runtime_error("not ready")));
        }
      });

  future.wait_for(2s);

  EXPECT_NE(ctx, nullptr);
}

std::unique_ptr<Module> makeBasicModule() {
  std::unique_ptr<Module> module = std::make_unique<Module>();
  std::unique_ptr<Context> ctx = std::make_unique<Context>();

  Function *F = module->createFunction("main");
  auto *input = module->createPlaceholder(ElemKind::FloatTy, {1, 32, 32, 3},
                                          "input", false);

  auto *FC = F->createFullyConnected(*ctx, "fc", input, 10);
  F->createSave("ret", FC);

  return module;
}

TEST(CPUDeviceManagerTest, availableMemory) {
  CPUDeviceManager cpuCoreDevice(200);
  cpuCoreDevice.init();

  uint64_t expectedBytes = 200 * 1024 * 1024;
  EXPECT_EQ(cpuCoreDevice.getMaximumMemory(), expectedBytes);
  EXPECT_EQ(cpuCoreDevice.getAvailableMemory(), expectedBytes);
  EXPECT_TRUE(cpuCoreDevice.isMemoryAvailable(expectedBytes));
  EXPECT_FALSE(cpuCoreDevice.isMemoryAvailable(expectedBytes + 1));

  auto module = makeBasicModule();

  std::promise<DeviceNetworkID> promise;
  std::future<DeviceNetworkID> future;

  std::tie(promise, future) = getFutureHelper();

  auto networkOne = cpuCoreDevice.getNextDeviceNetworkID();
  cpuCoreDevice.addNetwork(networkOne, std::move(module),
                           [&promise](DeviceNetworkID id, ResultCode result) {
                             addNetworkCallbackHelper(promise, id, result);
                           });

  future.wait_for(2s);
  EXPECT_EQ(future.get(), networkOne);

  EXPECT_EQ(cpuCoreDevice.getMaximumMemory(), expectedBytes);
  EXPECT_EQ(cpuCoreDevice.getAvailableMemory(), 0);
  EXPECT_FALSE(cpuCoreDevice.isMemoryAvailable(expectedBytes));
  EXPECT_FALSE(cpuCoreDevice.isMemoryAvailable(1));

  // Let's try again.
  module = makeBasicModule();
  std::tie(promise, future) = getFutureHelper();

  auto networkTwo = cpuCoreDevice.getNextDeviceNetworkID();
  cpuCoreDevice.addNetwork(networkTwo, std::move(module),
                           [&promise](DeviceNetworkID id, ResultCode result) {
                             addNetworkCallbackHelper(promise, id, result);
                           });

  future.wait_for(2s);
  auto returnedId = future.get();
  EXPECT_NE(returnedId, networkTwo);
  EXPECT_EQ(returnedId, 0);

  EXPECT_EQ(cpuCoreDevice.getMaximumMemory(), expectedBytes);
  EXPECT_EQ(cpuCoreDevice.getAvailableMemory(), 0);

  // Evict the first network.
  cpuCoreDevice.evictNetwork(1);

  module = makeBasicModule();
  std::tie(promise, future) = getFutureHelper();

  auto networkThree = cpuCoreDevice.getNextDeviceNetworkID();
  cpuCoreDevice.addNetwork(networkThree, std::move(module),
                           [&promise](DeviceNetworkID id, ResultCode result) {
                             addNetworkCallbackHelper(promise, id, result);
                           });

  future.wait_for(2s);
  EXPECT_EQ(future.get(), networkThree);

  EXPECT_EQ(cpuCoreDevice.getMaximumMemory(), expectedBytes);
  EXPECT_EQ(cpuCoreDevice.getAvailableMemory(), 0);
}
