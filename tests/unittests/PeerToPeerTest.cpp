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
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"
#include <future>

using namespace glow;
using namespace glow::runtime;

class PeerToPeerTestInterpreterOnly
    : public ::testing::TestWithParam<llvm::StringRef> {
public:
  void SetUp() override { backendName = GetParam(); }

  std::string backendName;
};

class PeerToPeerTestCPUOnly : public ::testing::TestWithParam<llvm::StringRef> {
public:
  void SetUp() override { backendName = GetParam(); }

  std::string backendName;
};

INSTANTIATE_TEST_CASE_P(Interpreter, PeerToPeerTestInterpreterOnly,
                        ::testing::Values("Interpreter"));
INSTANTIATE_TEST_CASE_P(CPU, PeerToPeerTestCPUOnly, ::testing::Values("CPU"));

std::unique_ptr<Module> createSenderModule(int64_t channel_id) {
  std::unique_ptr<Module> mod = llvm::make_unique<Module>();
  auto *F = mod->createFunction("send_func");
  auto *inputPH =
      mod->createPlaceholder(ElemKind::FloatTy, {1}, "send_input", false);
  auto *P = F->createPow("pow", inputPH, 2);
  TypeRef inputType = mod->uniqueType(ElemKind::FloatTy, {1});
  auto *remoteAddress =
      mod->createPlaceholder(ElemKind::AddressTy, {1}, "remoteAddress", false);
  auto *Send = F->createSend("send", P, remoteAddress, channel_id, inputType);
  F->createSave("send_save", Send);
  return mod;
}

std::unique_ptr<Module> createReceiverModule(int64_t channel_id) {
  std::unique_ptr<Module> mod = llvm::make_unique<Module>();
  auto *F0 = mod->createFunction("recv_ready_func");
  TypeRef inputType = mod->uniqueType(ElemKind::FloatTy, {1});
  auto *RR = F0->createRecvReady("recv_ready", channel_id, inputType);
  F0->createSave("recv_ready_save", RR);

  auto *F1 = mod->createFunction("recv_func");
  auto *R = F1->createRecv("recv", RR, channel_id, inputType);
  F1->createSave("recv_save", R);
  return mod;
}

std::unique_ptr<Module> createSenderModule2(int64_t channel_id) {
  std::unique_ptr<Module> mod = llvm::make_unique<Module>();
  auto *F = mod->createFunction("send_func");
  auto *inputPH =
      mod->createPlaceholder(ElemKind::FloatTy, {1}, "send_input", false);
  auto *P = F->createPow("pow", inputPH, 2);
  TypeRef inputType = mod->uniqueType(ElemKind::FloatTy, {1});
  auto *remoteAddress =
      mod->createPlaceholder(ElemKind::Int64ITy, {1}, "remoteAddress", false);
  auto *Send = F->createSend("send", P, remoteAddress, channel_id, inputType);
  F->createSave("send_save", Send);
  return mod;
}

std::unique_ptr<Module> createReceiverModule2(int64_t channel_id) {
  std::unique_ptr<Module> mod = llvm::make_unique<Module>();
  auto *F = mod->createFunction("recv_func");
  auto *inputPH =
      mod->createPlaceholder(ElemKind::FloatTy, {1}, "recv_input", false);
  inputPH->setIoPolicy(1);
  auto *P = F->createPow("pow", inputPH, 2);
  F->createSave("recv_save", P);
  return mod;
}

FunctionMapTy
compileFunctions(llvm::StringRef backendName, Module *mod,
                 std::vector<std::unique_ptr<CompiledFunction>> &backing) {
  FunctionMapTy results;
  auto *backend = createBackend(backendName);
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  cctx.optimizationOpts.enableConstantFolding = false;
  for (auto *F : mod->getFunctions()) {
    // EXIT_ON_ERR(::glow::optimizeFunction(F, *backend, cctx));
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

TEST_P(PeerToPeerTestCPUOnly, SendRecvSimple) {
  auto sender = std::unique_ptr<DeviceManager>(
      DeviceManager::createDeviceManager(DeviceConfig(backendName)));
  auto receiver = std::unique_ptr<DeviceManager>(
      DeviceManager::createDeviceManager(DeviceConfig(backendName)));
  ASSERT_FALSE(errToBool(sender->init()));
  ASSERT_FALSE(errToBool(receiver->init()));

  int64_t channelId = 8;
  auto sender_mod = createSenderModule2(channelId);
  auto receiver_mod = createReceiverModule2(channelId);

  // compile functions
  std::vector<std::unique_ptr<CompiledFunction>> receiverBacking;
  FunctionMapTy receiverFunctions =
      compileFunctions(backendName, receiver_mod.get(), receiverBacking);
  EXPECT_EQ(receiverBacking.size(), 1);

  std::vector<std::unique_ptr<CompiledFunction>> senderBacking;
  FunctionMapTy senderFunctions =
      compileFunctions(backendName, sender_mod.get(), senderBacking);
  EXPECT_EQ(senderBacking.size(), 1);

  // add networks
  std::promise<const Module *> recv_promise;
  std::future<const Module *> recv_future;
  std::tie(recv_promise, recv_future) = getFutureHelper<const Module *>();

  receiver->addNetwork(receiver_mod.get(), std::move(receiverFunctions),
                       [&recv_promise](const Module *module, llvm::Error err) {
                         callbackHelper(recv_promise, module, std::move(err));
                       });

  recv_future.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(recv_future.get(), receiver_mod.get());

  std::unique_ptr<ExecutionContext> recvContext =
      llvm::make_unique<ExecutionContext>();
  recvContext->getPlaceholderBindings()->allocate(
      receiver_mod->getPlaceholders());

  std::promise<const Module *> send_promise;
  std::future<const Module *> send_future;
  std::tie(send_promise, send_future) = getFutureHelper<const Module *>();

  sender->addNetwork(sender_mod.get(), std::move(senderFunctions),
                     [&send_promise](const Module *module, llvm::Error err) {
                       callbackHelper(send_promise, module, std::move(err));
                     });

  send_future.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(send_future.get(), sender_mod.get());

  std::unique_ptr<ExecutionContext> sendContext =
      llvm::make_unique<ExecutionContext>();
  sendContext->getPlaceholderBindings()->allocate(
      sender_mod->getPlaceholders());

  Tensor send_input(ElemKind::FloatTy, {1});
  send_input.getHandle().clear(2.0f);

  updateInputPlaceholders(*sendContext->getPlaceholderBindings(),
                          {sender_mod->getPlaceholderByName("send_input")},
                          {&send_input});

  llvm::Expected<int64_t> remoteAddressVal =
      receiver->getRemotePeerToPeerAddress(
          channelId, recvContext->getPlaceholderBindings());
  if (!remoteAddressVal) {
    llvm::outs() << "null address";
  }

  Tensor remoteAddressT(ElemKind::Int64ITy, {1});
  remoteAddressT.getHandle<uintptr_t>().clear(remoteAddressVal.get());
  updateInputPlaceholders(*sendContext->getPlaceholderBindings(),
                          {sender_mod->getPlaceholderByName("remoteAddress")},
                          {&remoteAddressT});

  // Run Send func
  std::promise<std::unique_ptr<ExecutionContext>> runPromise1;
  std::future<std::unique_ptr<ExecutionContext>> runFuture1;

  std::tie(runPromise1, runFuture1) =
      getFutureHelper<std::unique_ptr<ExecutionContext>>();
  sender->runFunction(
      "send_func", std::move(sendContext),
      [&runPromise1](RunIdentifierTy, llvm::Error err,
                     std::unique_ptr<ExecutionContext> context) {
        callbackHelper(runPromise1, std::move(context), std::move(err));
      });

  runFuture1.wait_for(std::chrono::seconds(2));
  sendContext = runFuture1.get();
  ASSERT_TRUE(sendContext);

  // Run Recv func
  std::promise<std::unique_ptr<ExecutionContext>> runPromise2;
  std::future<std::unique_ptr<ExecutionContext>> runFuture2;

  std::tie(runPromise2, runFuture2) =
      getFutureHelper<std::unique_ptr<ExecutionContext>>();
  receiver->runFunction(
      "recv_func", std::move(recvContext),
      [&runPromise2](RunIdentifierTy, llvm::Error err,
                     std::unique_ptr<ExecutionContext> context) {
        callbackHelper(runPromise2, std::move(context), std::move(err));
      });

  runFuture2.wait_for(std::chrono::seconds(2));
  recvContext = runFuture2.get();
  ASSERT_TRUE(recvContext);

  Tensor *result = recvContext->getPlaceholderBindings()->get(
      receiver_mod->getPlaceholderByName("recv_save"));
  ASSERT_TRUE(result);

  Tensor output_ref(ElemKind::FloatTy, {1});
  output_ref.getHandle().clear(4.0f * 4.0f);
  EXPECT_TRUE(result->isEqual(output_ref));
}

TEST_P(PeerToPeerTestInterpreterOnly, SendRecvSimple) {
  auto sender = std::unique_ptr<DeviceManager>(
      DeviceManager::createDeviceManager(DeviceConfig(backendName)));
  auto receiver = std::unique_ptr<DeviceManager>(
      DeviceManager::createDeviceManager(DeviceConfig(backendName)));
  ASSERT_FALSE(errToBool(sender->init()));
  ASSERT_FALSE(errToBool(receiver->init()));

  int64_t channelId = 8;
  auto sender_mod = createSenderModule(channelId);
  auto receiver_mod = createReceiverModule(channelId);

  // compile functions
  std::vector<std::unique_ptr<CompiledFunction>> receiverBacking;
  FunctionMapTy receiverFunctions =
      compileFunctions(backendName, receiver_mod.get(), receiverBacking);
  EXPECT_EQ(receiverBacking.size(), 2);

  std::vector<std::unique_ptr<CompiledFunction>> senderBacking;
  FunctionMapTy senderFunctions =
      compileFunctions(backendName, sender_mod.get(), senderBacking);
  EXPECT_EQ(senderBacking.size(), 1);

  // setup device-memory helper according to channel IDs
  EXIT_ON_ERR(sender->setupRemotePeerToPeer(channelId, receiver.get()));

  // add networks
  std::promise<const Module *> recv_promise;
  std::future<const Module *> recv_future;
  std::tie(recv_promise, recv_future) = getFutureHelper<const Module *>();

  receiver->addNetwork(receiver_mod.get(), std::move(receiverFunctions),
                       [&recv_promise](const Module *module, llvm::Error err) {
                         callbackHelper(recv_promise, module, std::move(err));
                       });

  recv_future.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(recv_future.get(), receiver_mod.get());

  std::promise<const Module *> send_promise;
  std::future<const Module *> send_future;
  std::tie(send_promise, send_future) = getFutureHelper<const Module *>();

  sender->addNetwork(sender_mod.get(), std::move(senderFunctions),
                     [&send_promise](const Module *module, llvm::Error err) {
                       callbackHelper(send_promise, module, std::move(err));
                     });

  send_future.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(send_future.get(), sender_mod.get());

  std::unique_ptr<ExecutionContext> sendContext =
      llvm::make_unique<ExecutionContext>();
  sendContext->getPlaceholderBindings()->allocate(
      sender_mod->getPlaceholders());

  std::unique_ptr<ExecutionContext> recvContext =
      llvm::make_unique<ExecutionContext>();
  recvContext->getPlaceholderBindings()->allocate(
      receiver_mod->getPlaceholders());

  Tensor send_input(ElemKind::FloatTy, {1});
  Tensor send_output_ref(ElemKind::FloatTy, {1});
  send_input.getHandle().clear(2.0f);
  send_output_ref.getHandle().clear(2.0f * 2.0f);

  updateInputPlaceholders(*sendContext->getPlaceholderBindings(),
                          {sender_mod->getPlaceholderByName("send_input")},
                          {&send_input});

  // Run RecvReady func
  std::promise<std::unique_ptr<ExecutionContext>> runPromise;
  std::future<std::unique_ptr<ExecutionContext>> runFuture;

  std::tie(runPromise, runFuture) =
      getFutureHelper<std::unique_ptr<ExecutionContext>>();
  receiver->runFunction(
      "recv_ready_func", std::move(recvContext),
      [&runPromise](RunIdentifierTy, llvm::Error err,
                    std::unique_ptr<ExecutionContext> context) {
        callbackHelper(runPromise, std::move(context), std::move(err));
      });

  runFuture.wait_for(std::chrono::seconds(2));
  recvContext = runFuture.get();
  ASSERT_TRUE(recvContext);

  // Run Send func
  std::promise<std::unique_ptr<ExecutionContext>> runPromise1;
  std::future<std::unique_ptr<ExecutionContext>> runFuture1;

  std::tie(runPromise1, runFuture1) =
      getFutureHelper<std::unique_ptr<ExecutionContext>>();
  sender->runFunction(
      "send_func", std::move(sendContext),
      [&runPromise1](RunIdentifierTy, llvm::Error err,
                     std::unique_ptr<ExecutionContext> context) {
        callbackHelper(runPromise1, std::move(context), std::move(err));
      });

  runFuture1.wait_for(std::chrono::seconds(2));
  sendContext = runFuture1.get();
  ASSERT_TRUE(sendContext);

  // Run Recv func
  std::promise<std::unique_ptr<ExecutionContext>> runPromise2;
  std::future<std::unique_ptr<ExecutionContext>> runFuture2;

  std::tie(runPromise2, runFuture2) =
      getFutureHelper<std::unique_ptr<ExecutionContext>>();
  receiver->runFunction(
      "recv_func", std::move(recvContext),
      [&runPromise2](RunIdentifierTy, llvm::Error err,
                     std::unique_ptr<ExecutionContext> context) {
        callbackHelper(runPromise2, std::move(context), std::move(err));
      });

  runFuture2.wait_for(std::chrono::seconds(2));
  recvContext = runFuture2.get();
  ASSERT_TRUE(recvContext);

  Tensor *result = recvContext->getPlaceholderBindings()->get(
      receiver_mod->getPlaceholderByName("recv_save"));
  ASSERT_TRUE(result);
  EXPECT_TRUE(result->isEqual(send_output_ref));
}
