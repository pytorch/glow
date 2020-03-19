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

#include "glow/Backends/DeviceManager.h"
#include "glow/Backends/DummyDeviceManager.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Runtime/RuntimeTypes.h"

#include "gtest/gtest.h"

#include <chrono>
#include <future>

using namespace glow;
using namespace glow::runtime;

template <typename ResultType>
std::pair<std::promise<ResultType>, std::future<ResultType>> getFutureHelper() {
  std::promise<ResultType> promise;
  auto future = promise.get_future();
  return std::make_pair(std::move(promise), std::move(future));
}

template <typename ResultType>
void callbackHelper(std::promise<ResultType> &promise, ResultType res,
                    Error err) {
  promise.set_value(!ERR_TO_BOOL(std::move(err)) ? std::move(res)
                                                 : ResultType());
}

class DeviceManagerTest : public ::testing::TestWithParam<std::string> {
public:
  void SetUp() override {
    backendName = GetParam();
    DeviceConfig config(backendName);
    device.reset(DeviceManager::createDeviceManager(config));
    ASSERT_TRUE(device.get());
    ASSERT_FALSE(ERR_TO_BOOL(device->init()));
  }

  void TearDown() override { EXPECT_FALSE(ERR_TO_BOOL(device->stop())); }

  std::string backendName;
  std::unique_ptr<DeviceManager> device{nullptr};

  void addToDevice(Module *module, FunctionMapTy functions) {

    std::promise<const Module *> promise;
    std::future<const Module *> future;
    std::tie(promise, future) = getFutureHelper<const Module *>();

    device->addNetwork(module, std::move(functions),
                       [&promise](const Module *module, Error err) {
                         callbackHelper(promise, module, std::move(err));
                       });

    future.wait_for(std::chrono::seconds(2));
    EXPECT_EQ(future.get(), module);
  }

  std::unique_ptr<ExecutionContext>
  runFunction(std::string name, std::unique_ptr<ExecutionContext> context) {
    std::promise<std::unique_ptr<ExecutionContext>> runPromise;
    std::future<std::unique_ptr<ExecutionContext>> runFuture;

    std::tie(runPromise, runFuture) =
        getFutureHelper<std::unique_ptr<ExecutionContext>>();
    device->runFunction(
        name, std::move(context),
        [&runPromise](RunIdentifierTy, Error err,
                      std::unique_ptr<ExecutionContext> context) {
          callbackHelper(runPromise, std::move(context), std::move(err));
        });

    runFuture.wait_for(std::chrono::seconds(2));
    context = runFuture.get();
    return context;
  }
};

std::unique_ptr<Module> makeBasicModule(std::string functionName = "main") {
  std::unique_ptr<Module> module = glow::make_unique<Module>();

  Function *F = module->createFunction(functionName);
  auto *input = module->createPlaceholder(ElemKind::FloatTy, {1},
                                          functionName + "_input", false);
  auto *output = module->createPlaceholder(ElemKind::FloatTy, {1},
                                           functionName + "_output", false);
  auto *c =
      module->createConstant(ElemKind::FloatTy, {1}, functionName + "_const");
  auto *t = F->createTanh("tanh", input);
  auto *m = F->createMax("max", c, t);
  F->createSave("ret", m, output);

  c->getPayloadMutable().getHandle().clear(0.25f);
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

TEST_P(DeviceManagerTest, Basic) {
  auto module = makeBasicModule();
  std::vector<std::unique_ptr<CompiledFunction>> backing;
  FunctionMapTy functions =
      compileFunctions(backendName, module.get(), backing);

  std::unique_ptr<ExecutionContext> context =
      glow::make_unique<ExecutionContext>();
  context->getPlaceholderBindings()->allocate(module->getPlaceholders());

  addToDevice(module.get(), std::move(functions));

  Tensor input1(ElemKind::FloatTy, {1});
  Tensor output1(ElemKind::FloatTy, {1});
  input1.getHandle().clear(0.5);
  output1.getHandle().clear(std::max(std::tanh(0.5), 0.25));

  updateInputPlaceholders(*context->getPlaceholderBindings(),
                          {module->getPlaceholderByName("main_input")},
                          {&input1});

  context = runFunction("main", std::move(context));
  ASSERT_TRUE(context);
  // We must ensure results are on host since we're using DeviceManager
  // directly.
  context->getPlaceholderBindings()->ensureOnHost();
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
  std::unique_ptr<Module> module = glow::make_unique<Module>();

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
                     [&promise](const Module *module, Error err) {
                       callbackHelper(promise, module, std::move(err));
                     });

  future.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(future.get(), module.get());

  std::unique_ptr<ExecutionContext> context =
      glow::make_unique<ExecutionContext>();
  context->getPlaceholderBindings()->allocate(output);

  Tensor input1(ElemKind::FloatTy, {1});
  auto size = input->getType()->getSizeInBytes() / 2;
  Tensor *virtualPaddedInput =
      new Tensor(input1.getUnsafePtr(), input->getType(), size);

  Tensor output1(ElemKind::FloatTy, {1});
  input1.getHandle().clear(0.5);
  output1.getHandle().clear(std::max(std::tanh(0.5), 0.25));

  context->getPlaceholderBindings()->insert(input, virtualPaddedInput);
  std::promise<std::unique_ptr<ExecutionContext>> runPromise;
  std::future<std::unique_ptr<ExecutionContext>> runFuture;

  std::tie(runPromise, runFuture) =
      getFutureHelper<std::unique_ptr<ExecutionContext>>();
  device->runFunction("main", std::move(context),
                      [&runPromise](RunIdentifierTy, Error err,
                                    std::unique_ptr<ExecutionContext> context) {
                        callbackHelper(runPromise, std::move(context),
                                       std::move(err));
                      });

  runFuture.wait_for(std::chrono::seconds(2));
  context = runFuture.get();
  ASSERT_TRUE(context);
  // We must ensure results are on host since we're using DeviceManager
  // directly.
  context->getPlaceholderBindings()->ensureOnHost();

  Tensor *result1 = context->getPlaceholderBindings()->get(
      module->getPlaceholderByName("main_output"));

  ASSERT_TRUE(result1);
  EXPECT_FLOAT_EQ(result1->getHandle().at({0}), std::max(std::tanh(0.5), 0.25));
}

// Test that the DeviceManager correctly supports
// transferStaticPlaceholderToDevice
TEST_P(DeviceManagerTest, TransferStaticPlaceholderTest) {
  CHECK_IF_ENABLED();
  std::unique_ptr<Module> module = glow::make_unique<Module>();

  Function *F = module->createFunction("main");
  auto *input =
      module->createPlaceholder(ElemKind::FloatTy, {1}, "input", false);
  auto *staticPlaceholder = module->createPlaceholder(
      ElemKind::FloatTy, {1}, "static_placeholder", false);
  staticPlaceholder->setStatic(true);
  auto *output =
      module->createPlaceholder(ElemKind::FloatTy, {1}, "main_output", false);
  auto *p = F->createPow("pow", input, staticPlaceholder);
  F->createSave("ret", p, output);

  std::vector<std::unique_ptr<CompiledFunction>> backing;
  FunctionMapTy functions =
      compileFunctions(backendName, module.get(), backing);

  std::promise<const Module *> promise;
  std::future<const Module *> future;
  std::tie(promise, future) = getFutureHelper<const Module *>();

  device->addNetwork(module.get(), std::move(functions),
                     [&promise](const Module *module, Error err) {
                       callbackHelper(promise, module, std::move(err));
                     });

  future.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(future.get(), module.get());

  auto staticTensor = Tensor(staticPlaceholder->getType());
  staticTensor.getHandle().clear(3.0);
  std::promise<void> transferPromise;
  Error transferError = Error::empty();
  auto done = transferPromise.get_future();

  device->transferStaticPlaceholderToDevice(
      staticPlaceholder, &staticTensor,
      [&transferPromise, &transferError](Error err) {
        transferError = std::move(err);
        transferPromise.set_value();
      });
  EXPECT_FALSE(ERR_TO_BOOL(std::move(transferError)));
  std::unique_ptr<ExecutionContext> context =
      glow::make_unique<ExecutionContext>();
  context->getPlaceholderBindings()->allocate(output);

  Tensor input1(ElemKind::FloatTy, {1});

  Tensor output1(ElemKind::FloatTy, {1});
  input1.getHandle().clear(2.0);

  context->getPlaceholderBindings()->allocate(input);
  context->getPlaceholderBindings()->get(input)->getHandle().clear(2.0);
  std::promise<std::unique_ptr<ExecutionContext>> runPromise;
  std::future<std::unique_ptr<ExecutionContext>> runFuture;

  std::tie(runPromise, runFuture) =
      getFutureHelper<std::unique_ptr<ExecutionContext>>();
  device->runFunction("main", std::move(context),
                      [&runPromise](RunIdentifierTy, Error err,
                                    std::unique_ptr<ExecutionContext> context) {
                        callbackHelper(runPromise, std::move(context),
                                       std::move(err));
                      });

  runFuture.wait_for(std::chrono::seconds(2));
  context = runFuture.get();
  ASSERT_TRUE(context);
  // We must ensure results are on host since we're using DeviceManager
  // directly.
  context->getPlaceholderBindings()->ensureOnHost();

  Tensor *result = context->getPlaceholderBindings()->get(output);

  ASSERT_TRUE(result);
  EXPECT_NEAR(result->getHandle().at({0}), 8.0, 1E-5);
}

TEST_P(DeviceManagerTest, MultiRun) {
  CHECK_IF_ENABLED();
  auto module = makeBasicModule();
  std::vector<std::unique_ptr<CompiledFunction>> backing;
  FunctionMapTy functions =
      compileFunctions(backendName, module.get(), backing);

  addToDevice(module.get(), std::move(functions));

  std::unique_ptr<ExecutionContext> context1 =
      glow::make_unique<ExecutionContext>();
  std::unique_ptr<ExecutionContext> context2 =
      glow::make_unique<ExecutionContext>();
  context1->getPlaceholderBindings()->allocate(module->getPlaceholders());
  context2->getPlaceholderBindings()->allocate(module->getPlaceholders());

  Tensor input1(ElemKind::FloatTy, {1});
  Tensor input2(ElemKind::FloatTy, {1});
  input1.getHandle().clear(2.0f);
  input2.getHandle().clear(3.0f);

  Tensor output1(ElemKind::FloatTy, {1});
  Tensor output2(ElemKind::FloatTy, {1});
  output1.getHandle().clear(std::max(std::tanh(2.0f), 0.25f));
  output2.getHandle().clear(std::max(std::tanh(3.0f), 0.25f));

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
                      [&runP1](RunIdentifierTy, Error err,
                               std::unique_ptr<ExecutionContext> context) {
                        callbackHelper(runP1, std::move(context),
                                       std::move(err));
                      });

  device->runFunction("main", std::move(context2),
                      [&runP2](RunIdentifierTy, Error err,
                               std::unique_ptr<ExecutionContext> context) {
                        callbackHelper(runP2, std::move(context),
                                       std::move(err));
                      });

  context1 = runF1.get();
  context2 = runF2.get();
  ASSERT_TRUE(context1);
  ASSERT_TRUE(context2);
  EXPECT_NE(context1, context2);
  // We must ensure results are on host since we're using DeviceManager
  // directly.
  context1->getPlaceholderBindings()->ensureOnHost();
  context2->getPlaceholderBindings()->ensureOnHost();

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
  CHECK_IF_ENABLED();

  auto module = makeBasicModule("func1");

  std::unique_ptr<ExecutionContext> context1 =
      glow::make_unique<ExecutionContext>();
  std::unique_ptr<ExecutionContext> context2 =
      glow::make_unique<ExecutionContext>();
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
                     [&promise](const Module *module, Error err) {
                       callbackHelper(promise, module, std::move(err));
                     });
  future.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(future.get(), module.get());

  Tensor input(ElemKind::FloatTy, {1});
  input.getHandle().clear(0.5f);
  Tensor output1(ElemKind::FloatTy, {1});
  output1.getHandle().clear(std::max(std::tanh(0.5f), 0.25f));
  Tensor output2(ElemKind::FloatTy, {1});
  output2.getHandle().clear(std::max(std::tanh(0.5f), 0.25f));

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
                      [&runP1](RunIdentifierTy, Error err,
                               std::unique_ptr<ExecutionContext> context) {
                        callbackHelper(runP1, std::move(context),
                                       std::move(err));
                      });

  device->runFunction("func2", std::move(context2),
                      [&runP2](RunIdentifierTy, Error err,
                               std::unique_ptr<ExecutionContext> context) {
                        callbackHelper(runP2, std::move(context),
                                       std::move(err));
                      });

  context1 = runF1.get();
  context2 = runF2.get();
  ASSERT_TRUE(context1);
  ASSERT_TRUE(context2);
  EXPECT_NE(context1, context2);
  context1->getPlaceholderBindings()->ensureOnHost();
  context2->getPlaceholderBindings()->ensureOnHost();

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
                     [&promise](const Module *module, Error err) {
                       callbackHelper(promise, module, std::move(err));
                     });
  future.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(future.get(), module1.get());

  std::tie(promise, future) = getFutureHelper<const Module *>();
  device->addNetwork(module2.get(), std::move(functions2),
                     [&promise](const Module *module, Error err) {
                       callbackHelper(promise, module, std::move(err));
                     });
  future.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(future.get(), module2.get());

  std::unique_ptr<ExecutionContext> context1 =
      glow::make_unique<ExecutionContext>();
  context1->getPlaceholderBindings()->allocate(module1->getPlaceholders());
  Tensor input(ElemKind::FloatTy, {1});
  input.getHandle().clear(0.5f);
  Tensor output(ElemKind::FloatTy, {1});
  output.getHandle().clear(std::max(std::tanh(0.5f), 0.25f));

  updateInputPlaceholders(*context1->getPlaceholderBindings(),
                          {module1->getPlaceholderByName("func1_input")},
                          {&input});

  std::unique_ptr<ExecutionContext> context2 =
      glow::make_unique<ExecutionContext>();
  context2->getPlaceholderBindings()->allocate(module2->getPlaceholders());
  updateInputPlaceholders(*context2->getPlaceholderBindings(),
                          {module2->getPlaceholderByName("func2_input")},
                          {&input});

  std::promise<std::unique_ptr<ExecutionContext>> runP1, runP2;
  std::future<std::unique_ptr<ExecutionContext>> runF1, runF2;
  std::tie(runP1, runF1) = getFutureHelper<std::unique_ptr<ExecutionContext>>();
  std::tie(runP2, runF2) = getFutureHelper<std::unique_ptr<ExecutionContext>>();

  device->runFunction("func1", std::move(context1),
                      [&runP1](RunIdentifierTy, Error err,
                               std::unique_ptr<ExecutionContext> context) {
                        callbackHelper(runP1, std::move(context),
                                       std::move(err));
                      });

  device->runFunction("func2", std::move(context2),
                      [&runP2](RunIdentifierTy, Error err,
                               std::unique_ptr<ExecutionContext> context) {
                        callbackHelper(runP2, std::move(context),
                                       std::move(err));
                      });

  context1 = runF1.get();
  context2 = runF2.get();
  ASSERT_TRUE(context1);
  ASSERT_TRUE(context2);
  EXPECT_NE(context1, context2);
  context1->getPlaceholderBindings()->ensureOnHost();
  context2->getPlaceholderBindings()->ensureOnHost();

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
      glow::make_unique<ExecutionContext>();
  std::unique_ptr<ExecutionContext> context2 =
      glow::make_unique<ExecutionContext>();
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
                     [&promise](const Module *module, Error err) {
                       callbackHelper(promise, module, std::move(err));
                     });
  future.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(future.get(), module.get());

  std::tie(promise, future) = getFutureHelper<const Module *>();
  device->addNetwork(module.get(), std::move(functions2),
                     [&promise](const Module *module, Error err) {
                       callbackHelper(promise, module, std::move(err));
                     });
  future.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(future.get(), module.get());

  Tensor input(ElemKind::FloatTy, {1});
  input.getHandle().clear(0.5f);
  Tensor output1(ElemKind::FloatTy, {1});
  output1.getHandle().clear(std::max(std::tanh(0.5f), 0.25f));
  Tensor output2(ElemKind::FloatTy, {1});
  output2.getHandle().clear(std::max(std::tanh(0.5f), 0.25f));

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
                      [&runP1](RunIdentifierTy, Error err,
                               std::unique_ptr<ExecutionContext> context) {
                        callbackHelper(runP1, std::move(context),
                                       std::move(err));
                      });

  device->runFunction("func2", std::move(context2),
                      [&runP2](RunIdentifierTy, Error err,
                               std::unique_ptr<ExecutionContext> context) {
                        callbackHelper(runP2, std::move(context),
                                       std::move(err));
                      });

  context1 = runF1.get();
  context2 = runF2.get();
  ASSERT_TRUE(context1);
  ASSERT_TRUE(context2);
  EXPECT_NE(context1, context2);
  context1->getPlaceholderBindings()->ensureOnHost();
  context2->getPlaceholderBindings()->ensureOnHost();

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
  auto interpreterDeviceSetByDeviceConfig = std::unique_ptr<DeviceManager>(
      DeviceManager::createDeviceManager(interpreterConfigFull));
  EXPECT_EQ(interpreterDeviceSetByDeviceConfig->getMaximumMemory(), 32768);
  // No setting at all, default memory size.
  auto interpreterDeviceDefault = std::unique_ptr<DeviceManager>(
      DeviceManager::createDeviceManager(interpreterConfigEmpty));
  EXPECT_EQ(interpreterDeviceDefault->getMaximumMemory(), 2000000000);
}

TEST(DeviceManagerTest, AvailableMemory) {
  std::vector<std::unique_ptr<CompiledFunction>> backing;
  std::promise<const Module *> promise;
  std::future<const Module *> future;

  auto module = makeBasicModule();
  auto compiledFunctions = compileFunctions("CPU", module.get(), backing);

  uint64_t expectedBytes{0};
  for (const auto &f : backing) {
    expectedBytes += f->getRuntimeBundle().getConstantWeightSize();
  }

  auto config = DeviceConfig("CPU");
  config.setDeviceMemory(expectedBytes);
  auto cpuCoreDevice = std::unique_ptr<DeviceManager>(
      DeviceManager::createDeviceManager(config));
  ASSERT_FALSE(ERR_TO_BOOL(cpuCoreDevice->init()));

  EXPECT_EQ(cpuCoreDevice->getMaximumMemory(), expectedBytes);
  EXPECT_EQ(cpuCoreDevice->getAvailableMemory(), expectedBytes);
  EXPECT_TRUE(cpuCoreDevice->isMemoryAvailable(expectedBytes));
  EXPECT_FALSE(cpuCoreDevice->isMemoryAvailable(expectedBytes + 1));

  std::tie(promise, future) = getFutureHelper<const Module *>();
  cpuCoreDevice->addNetwork(module.get(), compiledFunctions,
                            [&promise](const Module *module, Error err) {
                              callbackHelper(promise, module, std::move(err));
                            });

  future.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(future.get(), module.get());

  EXPECT_EQ(cpuCoreDevice->getMaximumMemory(), expectedBytes);
  EXPECT_EQ(cpuCoreDevice->getAvailableMemory(), 0);
  EXPECT_FALSE(cpuCoreDevice->isMemoryAvailable(expectedBytes));
  EXPECT_FALSE(cpuCoreDevice->isMemoryAvailable(1));

  // Let's try again.
  auto module2 = makeBasicModule();
  std::tie(promise, future) = getFutureHelper<const Module *>();
  cpuCoreDevice->addNetwork(module2.get(),
                            compileFunctions("CPU", module2.get(), backing),
                            [&promise](const Module *module, Error err) {
                              callbackHelper(promise, module, std::move(err));
                            });

  future.wait_for(std::chrono::seconds(2));
  auto *resultModule = future.get();
  EXPECT_NE(resultModule, module2.get());
  EXPECT_NE(resultModule, module.get());
  EXPECT_EQ(resultModule, nullptr);

  EXPECT_EQ(cpuCoreDevice->getMaximumMemory(), expectedBytes);
  EXPECT_EQ(cpuCoreDevice->getAvailableMemory(), 0);

  // Evict the first network.
  std::promise<std::string> evictPromise;
  std::future<std::string> evictFuture;
  std::tie(evictPromise, evictFuture) = getFutureHelper<std::string>();
  cpuCoreDevice->evictNetwork(
      "main", [&evictPromise](std::string functionName, Error err) {
        callbackHelper(evictPromise, functionName, std::move(err));
      });
  evictFuture.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(evictFuture.get(), "main");

  // And try again, this time with available space.
  std::tie(promise, future) = getFutureHelper<const Module *>();
  cpuCoreDevice->addNetwork(module2.get(),
                            compileFunctions("CPU", module2.get(), backing),
                            [&promise](const Module *module, Error err) {
                              callbackHelper(promise, module, std::move(err));
                            });

  future.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(future.get(), module2.get());

  EXPECT_EQ(cpuCoreDevice->getMaximumMemory(), expectedBytes);
  EXPECT_EQ(cpuCoreDevice->getAvailableMemory(), 0);

  EXPECT_FALSE(ERR_TO_BOOL(cpuCoreDevice->stop()));

  // Test CPU DeviceConfig.
  auto cpuConfigEmpty = DeviceConfig("CPU");
  auto cpuConfigFull = DeviceConfig("CPU");
  cpuConfigFull.setDeviceMemory(32768);
  // Only deviceConfig setting.
  auto cpuDeviceSetByDeviceConfig = std::unique_ptr<DeviceManager>(
      DeviceManager::createDeviceManager(cpuConfigFull));
  EXPECT_EQ(cpuDeviceSetByDeviceConfig->getMaximumMemory(), 32768);
  // No setting at all, default memory size.
  auto cpuDeviceDefault = std::unique_ptr<DeviceManager>(
      DeviceManager::createDeviceManager(cpuConfigEmpty));
  EXPECT_EQ(cpuDeviceDefault->getMaximumMemory(), 2000000000);
}

TEST(DeviceManagerTest, DummyDeviceManager) {
  DummyDeviceManager deviceManager{DeviceConfig("Interpreter")};
  ASSERT_FALSE(ERR_TO_BOOL(deviceManager.init()));

  auto module = makeBasicModule();
  std::vector<std::unique_ptr<CompiledFunction>> backing;
  FunctionMapTy functions =
      compileFunctions("Interpreter", module.get(), backing);

  std::promise<const Module *> promise;
  std::future<const Module *> future;
  std::tie(promise, future) = getFutureHelper<const Module *>();
  deviceManager.addNetwork(module.get(), std::move(functions),
                           [&promise](const Module *module, Error err) {
                             callbackHelper(promise, module, std::move(err));
                           });
  // no need to wait.
  EXPECT_EQ(future.get(), module.get());

  std::unique_ptr<ExecutionContext> context1 =
      glow::make_unique<ExecutionContext>();
  context1->getPlaceholderBindings()->allocate(module->getPlaceholders());

  Tensor input1(ElemKind::FloatTy, {1});
  Tensor output1(ElemKind::FloatTy, {1});
  input1.getHandle().clear(0.5f);
  output1.getHandle().clear(std::max(std::tanh(0.5f), 0.25f));

  updateInputPlaceholders(*context1->getPlaceholderBindings(),
                          {module->getPlaceholderByName("main_input")},
                          {&input1});

  std::promise<std::unique_ptr<ExecutionContext>> runPromise;
  std::future<std::unique_ptr<ExecutionContext>> runFuture;

  std::tie(runPromise, runFuture) =
      getFutureHelper<std::unique_ptr<ExecutionContext>>();
  deviceManager.runFunction(
      "main", std::move(context1),
      [&runPromise](RunIdentifierTy, Error err,
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

  EXPECT_FALSE(ERR_TO_BOOL(deviceManager.stop()));
}

/// Check that the device can move data to and from the host.
/// Disable if your device does not support Device Resident Tensors.
TEST_P(DeviceManagerTest, DeviceResidentTensors) {
  CHECK_IF_ENABLED();
  Tensor T = {1.2f, 12.1f, 51.0f, 1515.2f};
  Tensor R = {1.2f, 12.1f, 51.0f, 1515.2f};

  ASSERT_FALSE(T.isDeviceResident());

  device->transferToDevice(T, nullptr);

  ASSERT_TRUE(T.isDeviceResident());

  device->transferFromDevice(T);

  ASSERT_FALSE(T.isDeviceResident());

  ASSERT_TRUE(T.isEqual(R));
}

/// A mock DeviceManager for use in Device Resident Tensor tests.
class MockDM : public DeviceManager {
public:
  MockDM() : DeviceManager(DeviceConfig("MockDM")) {}
  void addNetwork(const Module *module, FunctionMapTy functions,
                  ReadyCBTy readyCB) override {}

  void evictNetwork(std::string functionName,
                    EvictFunctionCBTy evictCB = [](std::string, Error) {
                    }) override {}

  runtime::RunIdentifierTy
  runFunction(std::string functionName,
              std::unique_ptr<ExecutionContext> context,
              runtime::ResultCBTy resultCB) override {
    return 0;
  }

  uint64_t getMaximumMemory() const override { return 0; }

  uint64_t getAvailableMemory() const override { return 0; }

  bool isMemoryAvailable(uint64_t estimate) const override { return 0; }

  void transferToDevice(Tensor &tensor, void *locationContext = nullptr,
                        std::function<void(Error)> resultCB = [](Error) {
                        }) override {
    if (locationContext == nullptr) {
      locationContext = malloc(tensor.getSizeInBytes());
    }
    memcpy(locationContext, tensor.getUnsafePtr(), tensor.getSizeInBytes());
    tensor.moveToDevice(this, locationContext);
  }

  void transferFromDevice(Tensor &tensor, bool release = true,
                          std::function<void(Error)> resultCB = [](Error) {
                          }) override {
    memcpy(tensor.getUnsafePtr(), tensor.getLocationContext(),
           tensor.getSizeInBytes());
    free(tensor.getLocationContext());
    tensor.clearDeviceResidency();
  }

  bool releaseDeviceTensor(void *locationContext) override { return true; }
};

TEST_P(DeviceManagerTest, CanHandleDeviceResidentTensors) {
  MockDM mockDM;

  auto module = makeBasicModule();
  std::vector<std::unique_ptr<CompiledFunction>> backing;
  FunctionMapTy functions =
      compileFunctions(backendName, module.get(), backing);

  addToDevice(module.get(), std::move(functions));

  std::unique_ptr<ExecutionContext> context =
      glow::make_unique<ExecutionContext>();
  context->getPlaceholderBindings()->allocate(module->getPlaceholders());

  Tensor input1(ElemKind::FloatTy, {1});
  Tensor output1(ElemKind::FloatTy, {1});
  input1.getHandle().clear(0.5);
  output1.getHandle().clear(std::max(std::tanh(0.5), 0.25));

  updateInputPlaceholders(*context->getPlaceholderBindings(),
                          {module->getPlaceholderByName("main_input")},
                          {&input1});

  mockDM.transferToDevice(*context->getPlaceholderBindings()->get(
      module->getPlaceholderByName("main_input")));

  context = runFunction("main", std::move(context));
  ASSERT_TRUE(context);
  Tensor *result1 = context->getPlaceholderBindings()->get(
      module->getPlaceholderByName("main_output"));
  ASSERT_TRUE(result1);
}

TEST_P(DeviceManagerTest, TensorCopyRawToDevice) {
  MockDM mockDM;

  Tensor input1(ElemKind::FloatTy, {10});
  Tensor input2(ElemKind::FloatTy, {10});

  input1.getHandle().clear(1);
  input2.getHandle().clear(2);

  float *deviceMemory = (float *)malloc(sizeof(float) * 10);
  mockDM.transferToDevice(input1, deviceMemory);

  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(deviceMemory[i], 1);
  }

  input1.copyRawToDevice(&input2);

  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(deviceMemory[i], 2);
  }

  mockDM.transferFromDevice(input1);
  auto inputHandle = input1.getHandle();
  for (unsigned i = 0; i < 10; ++i) {
    EXPECT_EQ(inputHandle.at({i}), 2);
  }
}

INSTANTIATE_BACKEND_TEST(DeviceManagerTest);
