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
#include "glow/Backends/DummyDeviceManager.h"

#include "../../lib/Backends/CPU/CPUDeviceManager.h"
#include "../../lib/Backends/Interpreter/InterpreterDeviceManager.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Runtime/RuntimeTypes.h"

#include "gtest/gtest.h"

#include <chrono>
#include <future>

using namespace glow;
using namespace glow::runtime;

class DeviceManagerTest : public ::testing::TestWithParam<std::string> {
public:
  void SetUp() override {
    backendName = GetParam();
    device.reset(DeviceManager::createDeviceManager(DeviceConfig(backendName)));
    ASSERT_TRUE(device.get());
    ASSERT_FALSE(errToBool(device->init()));
  }

  void TearDown() override { EXPECT_FALSE(errToBool(device->stop())); }

  std::string backendName;
  std::unique_ptr<DeviceManager> device{nullptr};
};

std::unique_ptr<Module> makeBasicModule(std::string functionName = "main") {
  std::unique_ptr<Module> module = llvm::make_unique<Module>();

  Function *F = module->createFunction(functionName);
  auto *input = module->createPlaceholder(ElemKind::FloatTy, {1},
                                          functionName + "_input", false);
  auto *output = module->createPlaceholder(ElemKind::FloatTy, {1},
                                           functionName + "_output", false);
  auto *p = F->createTanh("tanh2", input);
  F->createSave("ret", p, output);

  return module;
}

FunctionMapTy
compileFunctions(llvm::StringRef backendName, Module *module,
                 std::vector<std::unique_ptr<CompiledFunction>> &backing) {
  FunctionMapTy results;
  auto *backend = createBackend(backendName);
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  for (auto *F : module->getFunctions()) {
    EXIT_ON_ERR(::glow::optimizeFunction(F, *backend, cctx));
    auto f = EXIT_ON_ERR(backend->compile(F, cctx.backendOpts));
    backing.push_back(std::move(f));
    results.emplace(F->getName(), backing.back().get());
  }

  delete backend;
  return results;
}

template <typename ResultType>
std::pair<std::promise<ResultType>, std::future<ResultType>> getFutureHelper() {
  std::promise<ResultType> promise;
  auto future = promise.get_future();
  return std::make_pair(std::move(promise), std::move(future));
}

template <typename ResultType>
void callbackHelper(std::promise<ResultType> &promise, ResultType res,
                    llvm::Error err) {
  promise.set_value(!errToBool(std::move(err)) ? std::move(res) : ResultType());
}

TEST_P(DeviceManagerTest, Basic) {
  auto module = makeBasicModule();
  std::vector<std::unique_ptr<CompiledFunction>> backing;
  FunctionMapTy functions =
      compileFunctions(backendName, module.get(), backing);

  std::promise<const Module *> promise;
  std::future<const Module *> future;
  std::tie(promise, future) = getFutureHelper<const Module *>();

  device->addNetwork(module.get(), std::move(functions),
                     [&promise](const Module *module, llvm::Error err) {
                       callbackHelper(promise, module, std::move(err));
                     });

  future.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(future.get(), module.get());

  std::unique_ptr<ExecutionContext> context =
      llvm::make_unique<ExecutionContext>();
  context->getPlaceholderBindings()->allocate(module->getPlaceholders());

  Tensor input1(ElemKind::FloatTy, {1});
  Tensor output1(ElemKind::FloatTy, {1});
  input1.getHandle().clear(0.5);
  output1.getHandle().clear(std::tanh(0.5));

  updateInputPlaceholders(*context->getPlaceholderBindings(),
                          {module->getPlaceholderByName("main_input")},
                          {&input1});

  std::promise<std::unique_ptr<ExecutionContext>> runPromise;
  std::future<std::unique_ptr<ExecutionContext>> runFuture;

  std::tie(runPromise, runFuture) =
      getFutureHelper<std::unique_ptr<ExecutionContext>>();
  device->runFunction("main", std::move(context),
                      [&runPromise](RunIdentifierTy, llvm::Error err,
                                    std::unique_ptr<ExecutionContext> context) {
                        callbackHelper(runPromise, std::move(context),
                                       std::move(err));
                      });

  runFuture.wait_for(std::chrono::seconds(2));
  context = runFuture.get();
  ASSERT_TRUE(context);
  Tensor *result1 = context->getPlaceholderBindings()->get(
      module->getPlaceholderByName("main_output"));
  ASSERT_TRUE(result1);
  EXPECT_TRUE(result1->isEqual(output1));
}

// Test that the DeviceManager correctly supports virtual padding.
TEST_P(DeviceManagerTest, PartialTensorCopy) {
  // Temporarily disable this test for Habana.
  if (backendName == "Habana") {
    return;
  }
  std::unique_ptr<Module> module = llvm::make_unique<Module>();

  // Create function of batch size 2.
  Function *F = module->createFunction("main");
  auto *input =
      module->createPlaceholder(ElemKind::FloatTy, {2}, "main_input", false);
  auto *output =
      module->createPlaceholder(ElemKind::FloatTy, {2}, "main_output", false);
  auto *p = F->createTanh("tanh2", input);
  F->createSave("ret", p, output);

  std::vector<std::unique_ptr<CompiledFunction>> backing;
  FunctionMapTy functions =
      compileFunctions(backendName, module.get(), backing);

  std::promise<const Module *> promise;
  std::future<const Module *> future;
  std::tie(promise, future) = getFutureHelper<const Module *>();

  device->addNetwork(module.get(), std::move(functions),
                     [&promise](const Module *module, llvm::Error err) {
                       callbackHelper(promise, module, std::move(err));
                     });

  future.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(future.get(), module.get());

  std::unique_ptr<ExecutionContext> context =
      llvm::make_unique<ExecutionContext>();
  context->getPlaceholderBindings()->allocate(output);

  Tensor input1(ElemKind::FloatTy, {1});
  auto size = input->getType()->getSizeInBytes() / 2;
  Tensor *virtualPaddedInput =
      new Tensor(input1.getUnsafePtr(), input->getType(), size);

  Tensor output1(ElemKind::FloatTy, {1});
  input1.getHandle().clear(0.5);
  output1.getHandle().clear(std::tanh(0.5));

  context->getPlaceholderBindings()->insert(input, virtualPaddedInput);
  std::promise<std::unique_ptr<ExecutionContext>> runPromise;
  std::future<std::unique_ptr<ExecutionContext>> runFuture;

  std::tie(runPromise, runFuture) =
      getFutureHelper<std::unique_ptr<ExecutionContext>>();
  device->runFunction("main", std::move(context),
                      [&runPromise](RunIdentifierTy, llvm::Error err,
                                    std::unique_ptr<ExecutionContext> context) {
                        callbackHelper(runPromise, std::move(context),
                                       std::move(err));
                      });

  runFuture.wait_for(std::chrono::seconds(2));
  context = runFuture.get();
  ASSERT_TRUE(context);
  Tensor *result1 = context->getPlaceholderBindings()->get(
      module->getPlaceholderByName("main_output"));
  ASSERT_TRUE(result1);
  EXPECT_FLOAT_EQ(result1->getHandle().at({0}), std::tanh(0.5));
}

TEST_P(DeviceManagerTest, MultiRun) {
  auto module = makeBasicModule();
  std::vector<std::unique_ptr<CompiledFunction>> backing;
  FunctionMapTy functions =
      compileFunctions(backendName, module.get(), backing);

  std::promise<const Module *> promise;
  std::future<const Module *> future;
  std::tie(promise, future) = getFutureHelper<const Module *>();
  device->addNetwork(module.get(), std::move(functions),
                     [&promise](const Module *module, llvm::Error err) {
                       callbackHelper(promise, module, std::move(err));
                     });
  future.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(future.get(), module.get());

  std::unique_ptr<ExecutionContext> context1 =
      llvm::make_unique<ExecutionContext>();
  std::unique_ptr<ExecutionContext> context2 =
      llvm::make_unique<ExecutionContext>();
  context1->getPlaceholderBindings()->allocate(module->getPlaceholders());
  context2->getPlaceholderBindings()->allocate(module->getPlaceholders());

  Tensor input1(ElemKind::FloatTy, {1});
  Tensor input2(ElemKind::FloatTy, {1});
  input1.getHandle().clear(2.0f);
  input2.getHandle().clear(3.0f);

  Tensor output1(ElemKind::FloatTy, {1});
  Tensor output2(ElemKind::FloatTy, {1});
  output1.getHandle().clear(std::tanh(2.0f));
  output2.getHandle().clear(std::tanh(3.0f));

  updateInputPlaceholders(*context1->getPlaceholderBindings(),
                          {module->getPlaceholderByName("main_input")},
                          {&input1});
  updateInputPlaceholders(*context2->getPlaceholderBindings(),
                          {module->getPlaceholderByName("main_input")},
                          {&input2});

  std::promise<std::unique_ptr<ExecutionContext>> runP1, runP2;
  std::future<std::unique_ptr<ExecutionContext>> runF1, runF2;
  std::tie(runP1, runF1) = getFutureHelper<std::unique_ptr<ExecutionContext>>();
  std::tie(runP2, runF2) = getFutureHelper<std::unique_ptr<ExecutionContext>>();

  device->runFunction("main", std::move(context1),
                      [&runP1](RunIdentifierTy, llvm::Error err,
                               std::unique_ptr<ExecutionContext> context) {
                        callbackHelper(runP1, std::move(context),
                                       std::move(err));
                      });

  device->runFunction("main", std::move(context2),
                      [&runP2](RunIdentifierTy, llvm::Error err,
                               std::unique_ptr<ExecutionContext> context) {
                        callbackHelper(runP2, std::move(context),
                                       std::move(err));
                      });

  context1 = runF1.get();
  context2 = runF2.get();
  ASSERT_TRUE(context1);
  ASSERT_TRUE(context2);
  EXPECT_NE(context1, context2);

  Tensor *result1 = context1->getPlaceholderBindings()->get(
      module->getPlaceholderByName("main_output"));
  Tensor *result2 = context2->getPlaceholderBindings()->get(
      module->getPlaceholderByName("main_output"));
  ASSERT_TRUE(result1);
  ASSERT_TRUE(result2);
  EXPECT_TRUE(result1->isEqual(output1));
  EXPECT_TRUE(result2->isEqual(output2));
}

TEST_P(DeviceManagerTest, MultiFunction) {
  auto module = makeBasicModule("func1");

  std::unique_ptr<ExecutionContext> context1 =
      llvm::make_unique<ExecutionContext>();
  std::unique_ptr<ExecutionContext> context2 =
      llvm::make_unique<ExecutionContext>();
  context1->getPlaceholderBindings()->allocate(module->getPlaceholders());

  Function *F = module->createFunction("func2");
  auto *inP = module->getPlaceholderByName("func1_input");
  auto *outP =
      module->createPlaceholder(ElemKind::FloatTy, {1}, "func2_output", false);
  auto *p = F->createTanh("tanh2", inP);
  F->createSave("ret2", p, outP);
  // Add extra tanh and fcs to the second function, we do not care about it's
  // output but this makes the two functions have different memory requirements.
  auto *c = module->createConstant(ElemKind::FloatTy, {1}, "add_constant");
  auto *sideTan = F->createTanh("tanh_extra", c);
  auto *fc = F->createFullyConnected(*context2->getPlaceholderBindings(), "fc",
                                     sideTan, 1000);
  auto *fc2 = F->createFullyConnected(*context2->getPlaceholderBindings(),
                                      "fc2", fc, 1);
  auto res = F->createSave("side_save", fc2);

  context2->getPlaceholderBindings()->allocate(inP);
  context2->getPlaceholderBindings()->allocate(outP);
  context2->getPlaceholderBindings()->allocate(res->getPlaceholder());

  std::vector<std::unique_ptr<CompiledFunction>> backing;
  FunctionMapTy functions =
      compileFunctions(backendName, module.get(), backing);
  EXPECT_EQ(functions.size(), 2);

  std::promise<const Module *> promise;
  std::future<const Module *> future;
  std::tie(promise, future) = getFutureHelper<const Module *>();
  device->addNetwork(module.get(), std::move(functions),
                     [&promise](const Module *module, llvm::Error err) {
                       callbackHelper(promise, module, std::move(err));
                     });
  future.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(future.get(), module.get());

  Tensor input(ElemKind::FloatTy, {1});
  input.getHandle().clear(0.5f);
  Tensor output1(ElemKind::FloatTy, {1});
  output1.getHandle().clear(std::tanh(0.5f));
  Tensor output2(ElemKind::FloatTy, {1});
  output2.getHandle().clear(std::tanh(0.5f));

  updateInputPlaceholders(*context1->getPlaceholderBindings(),
                          {module->getPlaceholderByName("func1_input")},
                          {&input});
  updateInputPlaceholders(*context2->getPlaceholderBindings(),
                          {module->getPlaceholderByName("func1_input")},
                          {&input});

  std::promise<std::unique_ptr<ExecutionContext>> runP1, runP2;
  std::future<std::unique_ptr<ExecutionContext>> runF1, runF2;
  std::tie(runP1, runF1) = getFutureHelper<std::unique_ptr<ExecutionContext>>();
  std::tie(runP2, runF2) = getFutureHelper<std::unique_ptr<ExecutionContext>>();

  device->runFunction("func1", std::move(context1),
                      [&runP1](RunIdentifierTy, llvm::Error err,
                               std::unique_ptr<ExecutionContext> context) {
                        callbackHelper(runP1, std::move(context),
                                       std::move(err));
                      });

  device->runFunction("func2", std::move(context2),
                      [&runP2](RunIdentifierTy, llvm::Error err,
                               std::unique_ptr<ExecutionContext> context) {
                        callbackHelper(runP2, std::move(context),
                                       std::move(err));
                      });

  context1 = runF1.get();
  context2 = runF2.get();
  ASSERT_TRUE(context1);
  ASSERT_TRUE(context2);
  EXPECT_NE(context1, context2);

  Tensor *result1 = context1->getPlaceholderBindings()->get(
      module->getPlaceholderByName("func1_output"));
  Tensor *result2 = context2->getPlaceholderBindings()->get(
      module->getPlaceholderByName("func2_output"));
  ASSERT_TRUE(result1);
  ASSERT_TRUE(result2);
  EXPECT_TRUE(result1->isEqual(output1));
  EXPECT_TRUE(result2->isEqual(output2));
}

TEST_P(DeviceManagerTest, MultiModule) {
  auto module1 = makeBasicModule("func1");
  auto module2 = makeBasicModule("func2");

  std::vector<std::unique_ptr<CompiledFunction>> backing;
  FunctionMapTy functions1 =
      compileFunctions(backendName, module1.get(), backing);
  FunctionMapTy functions2 =
      compileFunctions(backendName, module2.get(), backing);

  std::promise<const Module *> promise;
  std::future<const Module *> future;
  std::tie(promise, future) = getFutureHelper<const Module *>();
  device->addNetwork(module1.get(), std::move(functions1),
                     [&promise](const Module *module, llvm::Error err) {
                       callbackHelper(promise, module, std::move(err));
                     });
  future.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(future.get(), module1.get());

  std::tie(promise, future) = getFutureHelper<const Module *>();
  device->addNetwork(module2.get(), std::move(functions2),
                     [&promise](const Module *module, llvm::Error err) {
                       callbackHelper(promise, module, std::move(err));
                     });
  future.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(future.get(), module2.get());

  std::unique_ptr<ExecutionContext> context1 =
      llvm::make_unique<ExecutionContext>();
  context1->getPlaceholderBindings()->allocate(module1->getPlaceholders());
  Tensor input(ElemKind::FloatTy, {1});
  input.getHandle().clear(0.5f);
  Tensor output(ElemKind::FloatTy, {1});
  output.getHandle().clear(std::tanh(0.5f));

  updateInputPlaceholders(*context1->getPlaceholderBindings(),
                          {module1->getPlaceholderByName("func1_input")},
                          {&input});

  std::unique_ptr<ExecutionContext> context2 =
      llvm::make_unique<ExecutionContext>();
  context2->getPlaceholderBindings()->allocate(module2->getPlaceholders());
  updateInputPlaceholders(*context2->getPlaceholderBindings(),
                          {module2->getPlaceholderByName("func2_input")},
                          {&input});

  std::promise<std::unique_ptr<ExecutionContext>> runP1, runP2;
  std::future<std::unique_ptr<ExecutionContext>> runF1, runF2;
  std::tie(runP1, runF1) = getFutureHelper<std::unique_ptr<ExecutionContext>>();
  std::tie(runP2, runF2) = getFutureHelper<std::unique_ptr<ExecutionContext>>();

  device->runFunction("func1", std::move(context1),
                      [&runP1](RunIdentifierTy, llvm::Error err,
                               std::unique_ptr<ExecutionContext> context) {
                        callbackHelper(runP1, std::move(context),
                                       std::move(err));
                      });

  device->runFunction("func2", std::move(context2),
                      [&runP2](RunIdentifierTy, llvm::Error err,
                               std::unique_ptr<ExecutionContext> context) {
                        callbackHelper(runP2, std::move(context),
                                       std::move(err));
                      });

  context1 = runF1.get();
  context2 = runF2.get();
  ASSERT_TRUE(context1);
  ASSERT_TRUE(context2);
  EXPECT_NE(context1, context2);

  Tensor *result1 = context1->getPlaceholderBindings()->get(
      module1->getPlaceholderByName("func1_output"));
  ASSERT_TRUE(result1);
  EXPECT_TRUE(result1->isEqual(output));

  Tensor *result2 = context2->getPlaceholderBindings()->get(
      module2->getPlaceholderByName("func2_output"));
  ASSERT_TRUE(result2);
  EXPECT_TRUE(result2->isEqual(output));
}

TEST_P(DeviceManagerTest, ReuseModule) {
  auto module = makeBasicModule("func1");

  std::unique_ptr<ExecutionContext> context1 =
      llvm::make_unique<ExecutionContext>();
  std::unique_ptr<ExecutionContext> context2 =
      llvm::make_unique<ExecutionContext>();
  context1->getPlaceholderBindings()->allocate(module->getPlaceholders());

  Function *F = module->createFunction("func2");
  auto *inP = module->getPlaceholderByName("func1_input");
  auto *outP =
      module->createPlaceholder(ElemKind::FloatTy, {1}, "func2_output", false);
  auto *p = F->createTanh("tanh2", inP);
  F->createSave("ret2", p, outP);

  context2->getPlaceholderBindings()->allocate(inP);
  context2->getPlaceholderBindings()->allocate(outP);

  std::vector<std::unique_ptr<CompiledFunction>> backing;
  FunctionMapTy functions =
      compileFunctions(backendName, module.get(), backing);
  EXPECT_EQ(functions.size(), 2);

  // Split the function map into two parts.
  FunctionMapTy functions2;
  functions2.emplace("func2", std::move(functions["func2"]));
  functions.erase("func2");
  EXPECT_EQ(functions.size(), 1);
  EXPECT_EQ(functions2.size(), 1);

  std::promise<const Module *> promise;
  std::future<const Module *> future;
  std::tie(promise, future) = getFutureHelper<const Module *>();
  device->addNetwork(module.get(), std::move(functions),
                     [&promise](const Module *module, llvm::Error err) {
                       callbackHelper(promise, module, std::move(err));
                     });
  future.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(future.get(), module.get());

  std::tie(promise, future) = getFutureHelper<const Module *>();
  device->addNetwork(module.get(), std::move(functions2),
                     [&promise](const Module *module, llvm::Error err) {
                       callbackHelper(promise, module, std::move(err));
                     });
  future.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(future.get(), module.get());

  Tensor input(ElemKind::FloatTy, {1});
  input.getHandle().clear(0.5f);
  Tensor output1(ElemKind::FloatTy, {1});
  output1.getHandle().clear(std::tanh(0.5f));
  Tensor output2(ElemKind::FloatTy, {1});
  output2.getHandle().clear(std::tanh(0.5f));

  updateInputPlaceholders(*context1->getPlaceholderBindings(),
                          {module->getPlaceholderByName("func1_input")},
                          {&input});
  updateInputPlaceholders(*context2->getPlaceholderBindings(),
                          {module->getPlaceholderByName("func1_input")},
                          {&input});

  std::promise<std::unique_ptr<ExecutionContext>> runP1, runP2;
  std::future<std::unique_ptr<ExecutionContext>> runF1, runF2;
  std::tie(runP1, runF1) = getFutureHelper<std::unique_ptr<ExecutionContext>>();
  std::tie(runP2, runF2) = getFutureHelper<std::unique_ptr<ExecutionContext>>();

  device->runFunction("func1", std::move(context1),
                      [&runP1](RunIdentifierTy, llvm::Error err,
                               std::unique_ptr<ExecutionContext> context) {
                        callbackHelper(runP1, std::move(context),
                                       std::move(err));
                      });

  device->runFunction("func2", std::move(context2),
                      [&runP2](RunIdentifierTy, llvm::Error err,
                               std::unique_ptr<ExecutionContext> context) {
                        callbackHelper(runP2, std::move(context),
                                       std::move(err));
                      });

  context1 = runF1.get();
  context2 = runF2.get();
  ASSERT_TRUE(context1);
  ASSERT_TRUE(context2);
  EXPECT_NE(context1, context2);

  Tensor *result1 = context1->getPlaceholderBindings()->get(
      module->getPlaceholderByName("func1_output"));
  ASSERT_TRUE(result1);
  EXPECT_TRUE(result1->isEqual(output1));

  Tensor *result2 = context2->getPlaceholderBindings()->get(
      module->getPlaceholderByName("func2_output"));
  ASSERT_TRUE(result2);
  EXPECT_TRUE(result2->isEqual(output2));
}

TEST(DeviceManagerTest, SetDeviceMemory) {
  // Test Interpreter.
  auto interpreterConfigEmpty = DeviceConfig("Interpreter");
  auto interpreterConfigFull = DeviceConfig("Interpreter");
  interpreterConfigFull.setDeviceMemory(32768);
  // Only deviceConfig setting.
  InterpreterDeviceManager interpreterDeviceSetByDeviceConfig(
      interpreterConfigFull);
  EXPECT_EQ(interpreterDeviceSetByDeviceConfig.getMaximumMemory(), 32768);
  // No setting at all, default memory size.
  InterpreterDeviceManager interpreterDeviceDefault(interpreterConfigEmpty);
  EXPECT_EQ(interpreterDeviceDefault.getMaximumMemory(), 2000000000);
}

#ifdef GLOW_WITH_CPU

TEST(DeviceManagerTest, AvailableMemory) {
  std::vector<std::unique_ptr<CompiledFunction>> backing;
  std::promise<const Module *> promise;
  std::future<const Module *> future;

  auto config = DeviceConfig("CPU");
  config.setDeviceMemory(1);
  CPUDeviceManager cpuCoreDevice(config);
  ASSERT_FALSE(errToBool(cpuCoreDevice.init()));

  uint64_t expectedBytes = 1;
  EXPECT_EQ(cpuCoreDevice.getMaximumMemory(), expectedBytes);
  EXPECT_EQ(cpuCoreDevice.getAvailableMemory(), expectedBytes);
  EXPECT_TRUE(cpuCoreDevice.isMemoryAvailable(expectedBytes));
  EXPECT_FALSE(cpuCoreDevice.isMemoryAvailable(expectedBytes + 1));

  auto module = makeBasicModule();
  std::tie(promise, future) = getFutureHelper<const Module *>();
  cpuCoreDevice.addNetwork(module.get(),
                           compileFunctions("CPU", module.get(), backing),
                           [&promise](const Module *module, llvm::Error err) {
                             callbackHelper(promise, module, std::move(err));
                           });

  future.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(future.get(), module.get());

  EXPECT_EQ(cpuCoreDevice.getMaximumMemory(), expectedBytes);
  EXPECT_EQ(cpuCoreDevice.getAvailableMemory(), 0);
  EXPECT_FALSE(cpuCoreDevice.isMemoryAvailable(expectedBytes));
  EXPECT_FALSE(cpuCoreDevice.isMemoryAvailable(1));

  // Let's try again.
  auto module2 = makeBasicModule();
  std::tie(promise, future) = getFutureHelper<const Module *>();
  cpuCoreDevice.addNetwork(module2.get(),
                           compileFunctions("CPU", module2.get(), backing),
                           [&promise](const Module *module, llvm::Error err) {
                             callbackHelper(promise, module, std::move(err));
                           });

  future.wait_for(std::chrono::seconds(2));
  auto *resultModule = future.get();
  EXPECT_NE(resultModule, module2.get());
  EXPECT_NE(resultModule, module.get());
  EXPECT_EQ(resultModule, nullptr);

  EXPECT_EQ(cpuCoreDevice.getMaximumMemory(), expectedBytes);
  EXPECT_EQ(cpuCoreDevice.getAvailableMemory(), 0);

  // Evict the first network.
  std::promise<std::string> evictPromise;
  std::future<std::string> evictFuture;
  std::tie(evictPromise, evictFuture) = getFutureHelper<std::string>();
  cpuCoreDevice.evictNetwork(
      "main", [&evictPromise](std::string functionName, llvm::Error err) {
        callbackHelper(evictPromise, functionName, std::move(err));
      });
  evictFuture.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(evictFuture.get(), "main");

  // And try again, this time with available space.
  std::tie(promise, future) = getFutureHelper<const Module *>();
  cpuCoreDevice.addNetwork(module2.get(),
                           compileFunctions("CPU", module2.get(), backing),
                           [&promise](const Module *module, llvm::Error err) {
                             callbackHelper(promise, module, std::move(err));
                           });

  future.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(future.get(), module2.get());

  EXPECT_EQ(cpuCoreDevice.getMaximumMemory(), expectedBytes);
  EXPECT_EQ(cpuCoreDevice.getAvailableMemory(), 0);

  EXPECT_FALSE(errToBool(cpuCoreDevice.stop()));

  // Test CPU DeviceConfig.
  auto cpuConfigEmpty = DeviceConfig("CPU");
  auto cpuConfigFull = DeviceConfig("CPU");
  cpuConfigFull.setDeviceMemory(32768);
  // Only deviceConfig setting.
  CPUDeviceManager cpuDeviceSetByDeviceConfig(cpuConfigFull);
  EXPECT_EQ(cpuDeviceSetByDeviceConfig.getMaximumMemory(), 32768);
  // No setting at all, default memory size.
  CPUDeviceManager cpuDeviceDefault(cpuConfigEmpty);
  EXPECT_EQ(cpuDeviceDefault.getMaximumMemory(), 2000000000);
}

TEST(DeviceManagerTest, DummyDeviceManager) {
  DummyDeviceManager deviceManager{DeviceConfig("Interpreter")};
  ASSERT_FALSE(errToBool(deviceManager.init()));

  auto module = makeBasicModule();
  std::vector<std::unique_ptr<CompiledFunction>> backing;
  FunctionMapTy functions =
      compileFunctions("Interpreter", module.get(), backing);

  std::promise<const Module *> promise;
  std::future<const Module *> future;
  std::tie(promise, future) = getFutureHelper<const Module *>();
  deviceManager.addNetwork(module.get(), std::move(functions),
                           [&promise](const Module *module, llvm::Error err) {
                             callbackHelper(promise, module, std::move(err));
                           });
  // no need to wait.
  EXPECT_EQ(future.get(), module.get());

  std::unique_ptr<ExecutionContext> context1 =
      llvm::make_unique<ExecutionContext>();
  context1->getPlaceholderBindings()->allocate(module->getPlaceholders());

  Tensor input1(ElemKind::FloatTy, {1});
  Tensor output1(ElemKind::FloatTy, {1});
  input1.getHandle().clear(0.5f);
  output1.getHandle().clear(std::tanh(0.5f));

  updateInputPlaceholders(*context1->getPlaceholderBindings(),
                          {module->getPlaceholderByName("main_input")},
                          {&input1});

  std::promise<std::unique_ptr<ExecutionContext>> runPromise;
  std::future<std::unique_ptr<ExecutionContext>> runFuture;

  std::tie(runPromise, runFuture) =
      getFutureHelper<std::unique_ptr<ExecutionContext>>();
  deviceManager.runFunction(
      "main", std::move(context1),
      [&runPromise](RunIdentifierTy, llvm::Error err,
                    std::unique_ptr<ExecutionContext> context) {
        callbackHelper(runPromise, std::move(context), std::move(err));
      });

  runFuture.wait_for(std::chrono::seconds(2));
  std::unique_ptr<ExecutionContext> context2 = runFuture.get();

  ASSERT_TRUE(context2);

  Tensor *result = context2->getPlaceholderBindings()->get(
      module->getPlaceholderByName("main_output"));
  ASSERT_TRUE(result);
  EXPECT_TRUE(result->isEqual(output1));

  EXPECT_FALSE(errToBool(deviceManager.stop()));
}

#endif // GLOW_WITH_CPU

INSTANTIATE_TEST_CASE_P(Interpreter, DeviceManagerTest,
                        ::testing::Values("Interpreter"));

#ifdef GLOW_WITH_CPU
INSTANTIATE_TEST_CASE_P(CPU, DeviceManagerTest, ::testing::Values("CPU"));
#endif // GLOW_WITH_CPU

#ifdef GLOW_WITH_OPENCL
INSTANTIATE_TEST_CASE_P(OpenCL, DeviceManagerTest, ::testing::Values("OpenCL"));
#endif // GLOW_WITH_OPENCL

#ifdef GLOW_WITH_HABANA
INSTANTIATE_TEST_CASE_P(Habana, DeviceManagerTest, ::testing::Values("Habana"));
#endif // GLOW_WITH_HABANA
