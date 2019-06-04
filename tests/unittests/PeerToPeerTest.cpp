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

#include "../../lib/Backends/Interpreter/InterpreterDeviceManager.h"
#include "glow/Backends/DeviceManager.h"
#include "glow/Optimizer/Optimizer.h"

#include "gtest/gtest.h"

#include <chrono>
#include <future>

/// This macro asserts if \p lhsOrErr is an error, thus failing the containing
/// test, and assigns its value to \p lhs if not.
#define ASSIGN_VALUE_OR_ASSERT(rhs, lhsOrErr)                                  \
  do {                                                                         \
    auto lhsOrErrV = (lhsOrErr);                                               \
    static_assert(IsLLVMExpected<decltype(lhsOrErrV)>(),                       \
                  "Expected value to be a llvm::Expected");                    \
    ASSERT_FALSE(!lhsOrErrV);                                                  \
    rhs = std::move(lhsOrErrV.get());                                          \
  } while (0)

using namespace glow;
using namespace glow::runtime;

/// Fixture class for peer-to-peer communication tests.
class PeerToPeerTest : public ::testing::TestWithParam<BackendKind> {
public:
  void SetUp() override {
    // Ensure that the backend being tested supports peer-to-peer communication.
    auto backend = std::unique_ptr<Backend>(createBackend(backendKind));
    ASSERT_TRUE(backend->isPeerToPeerSupported());
  }

  /// The backend kind being tested.
  BackendKind backendKind{GetParam()};
};

/// Helper function to compile functions in a \p module and store the compiled
/// functions into \p backing. Only functions whose names appear in \p allow are
/// compiled.
FunctionMapTy
compileFunctions(BackendKind backendKind, Module *module,
                 std::vector<std::unique_ptr<CompiledFunction>> &backing,
                 llvm::ArrayRef<llvm::StringRef> allow = {}) {
  FunctionMapTy results;
  auto backend = std::unique_ptr<Backend>(createBackend(backendKind));
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  for (auto *F : module->getFunctions()) {
    bool skip = true;

    // Don't skip compilation if the function's name is in the allow list.
    for (const auto &a : allow) {
      if (F->getName() == a) {
        skip = false;
        break;
      }
    }

    if (skip) {
      continue;
    }

    EXIT_ON_ERR(::glow::optimizeFunction(F, *backend, cctx));
    auto f = EXIT_ON_ERR(backend->compile(F, cctx.backendOpts));
    backing.push_back(std::move(f));
    results.emplace(F->getName(), backing.back().get());
  }

  return results;
}

/// Helper function to create a future-promise pair of a specific type.
template <typename ResultType>
std::pair<std::promise<ResultType>, std::future<ResultType>> getFutureHelper() {
  std::promise<ResultType> promise;
  auto future = promise.get_future();
  return std::make_pair(std::move(promise), std::move(future));
}

/// Helper function to fulfill a promise.
template <typename ResultType>
void callbackHelper(std::promise<ResultType> &promise, ResultType res,
                    llvm::Error err) {
  promise.set_value(!errToBool(std::move(err)) ? std::move(res) : ResultType());
}

/// Tests peer-to-peer communication with a single sender and single receiver.
TEST_P(PeerToPeerTest, SingleSenderSingleReceiver) {
  // Create sender and receiver backends.
  auto sender = std::unique_ptr<DeviceManager>(
      DeviceManager::createDeviceManager(backendKind));
  auto receiver = std::unique_ptr<DeviceManager>(
      DeviceManager::createDeviceManager(backendKind));

  ASSERT_FALSE(errToBool(sender->init()));
  ASSERT_FALSE(errToBool(receiver->init()));

  auto module = llvm::make_unique<Module>();

  // Create the sender function. It implements the operation 1 + 2 and sends the
  // result (3) to the receiver function using a SendNode.
  Function *senderFn = module->createFunction("senderFn");
  auto *senderFnLHS = module->createConstant(ElemKind::FloatTy, {1}, "lhs");
  auto senderLHSH = senderFnLHS->getPayload().getHandle();
  senderLHSH = {1};
  auto *senderFnRHS = module->createConstant(ElemKind::FloatTy, {1}, "rhs");
  auto senderRHSH = senderFnRHS->getPayload().getHandle();
  senderRHSH = {2};
  auto *senderFnAdd = senderFn->createAdd("add", senderFnLHS, senderFnRHS);
  auto *address =
      module->createPlaceholder(ElemKind::AddrTy, {1}, "address", false);
  auto *send = senderFn->createSend("send", senderFnAdd, address);

  // Create the receiver function. It receives the result of the sender function
  // and adds 1 to it (the result should be 4).
  Function *receiverFn = module->createFunction("receiverFn");
  auto *recv = receiverFn->createReceive(
      "recv", senderFnAdd->getResult().getType(), address);
  auto *receiverFnRHS = module->createConstant(ElemKind::FloatTy, {1}, "rhs");
  auto receiverRHSH = receiverFnRHS->getPayload().getHandle();
  receiverRHSH = {1};
  auto *receiverFnAdd = receiverFn->createAdd("add", recv, receiverFnRHS);
  auto *save = receiverFn->createSave("save", receiverFnAdd);

  // Compile the sender and receiver functions.
  std::vector<std::unique_ptr<CompiledFunction>> senderBacking, receiverBacking;
  FunctionMapTy senderFunctions = compileFunctions(
      BackendKind::Interpreter, module.get(), senderBacking, {"senderFn"});
  FunctionMapTy receiverFunctions = compileFunctions(
      BackendKind::Interpreter, module.get(), receiverBacking, {"receiverFn"});

  // Add the sender and receiver functions to their respective devices.
  std::promise<const Module *> p1, p2;
  std::future<const Module *> f1, f2;

  std::tie(p1, f1) = getFutureHelper<const Module *>();
  sender->addNetwork(module.get(), senderFunctions,
                     [&p1](const Module *module, llvm::Error err) {
                       callbackHelper(p1, module, std::move(err));
                     });
  f1.wait();
  EXPECT_EQ(f1.get(), module.get());

  std::tie(p2, f2) = getFutureHelper<const Module *>();
  receiver->addNetwork(module.get(), receiverFunctions,
                       [&p2](const Module *module, llvm::Error err) {
                         callbackHelper(p2, module, std::move(err));
                       });
  f2.wait();
  EXPECT_EQ(f2.get(), module.get());

  // Create and set ExecutionContexts for the sender and receiver.
  auto senderCtx = llvm::make_unique<ExecutionContext>();
  auto receiverCtx = llvm::make_unique<ExecutionContext>();

  senderCtx->getPlaceholderBindings()->allocate(send->getPlaceholder());
  auto AH1 = senderCtx->getPlaceholderBindings()
                 ->allocate(address)
                 ->getHandle<uintptr_t>();

  // Get the receiver's address for the ReceiveNode Placeholder from the
  // receiver device and set the address input of the SendNode to this value.
  uintptr_t addr;
  ASSIGN_VALUE_OR_ASSERT(
      addr, receiver->getPlaceholderAddress(receiverFn->getName(),
                                            recv->getPlaceholder()));
  AH1 = {addr};

  receiverCtx->getPlaceholderBindings()->allocate(recv->getPlaceholder());
  auto AH2 = receiverCtx->getPlaceholderBindings()
                 ->allocate(address)
                 ->getHandle<uintptr_t>();

  // Set the address input of the ReceiveNode to the same address as the
  // SendNode.
  AH2 = {addr};
  receiverCtx->getPlaceholderBindings()->allocate(save->getPlaceholder());

  // Try to run this setup five times to make sure that peer-to-peer
  // communication works several times and state is managed correctly.
  for (size_t i = 0; i < 5; ++i) {
    // Clear the final result tensor to make sure an iteration does not reuse
    // data from the previous iteration.
    receiverCtx->getPlaceholderBindings()
        ->get(save->getPlaceholder())
        ->getHandle()
        .clear();

    // Run the sender and receiver functions.
    std::promise<std::unique_ptr<ExecutionContext>> runP1, runP2;
    std::future<std::unique_ptr<ExecutionContext>> runF1, runF2;
    std::tie(runP1, runF1) =
        getFutureHelper<std::unique_ptr<ExecutionContext>>();
    std::tie(runP2, runF2) =
        getFutureHelper<std::unique_ptr<ExecutionContext>>();

    sender->runFunction("senderFn", std::move(senderCtx),
                        [&runP1](RunIdentifierTy, llvm::Error err,
                                 std::unique_ptr<ExecutionContext> context) {
                          callbackHelper(runP1, std::move(context),
                                         std::move(err));
                        });

    receiver->runFunction("receiverFn", std::move(receiverCtx),
                          [&runP2](RunIdentifierTy, llvm::Error err,
                                   std::unique_ptr<ExecutionContext> context) {
                            callbackHelper(runP2, std::move(context),
                                           std::move(err));
                          });

    // Wait for the functions to finish running.
    senderCtx = runF1.get();
    receiverCtx = runF2.get();
    ASSERT_TRUE(senderCtx);
    ASSERT_TRUE(receiverCtx);
    EXPECT_NE(senderCtx, receiverCtx);

    // As mentioned earlier, the final result should be 4.
    auto RH = receiverCtx->getPlaceholderBindings()
                  ->get(save->getPlaceholder())
                  ->getHandle();
    ASSERT_EQ(RH.raw(0), 4);
  }
}

/// Tests peer-to-peer communication with a single sender and multiple
/// receivers.
TEST_P(PeerToPeerTest, SingleSenderMultipleReceivers) {
  // Create one sender and two receiver devices.
  auto sender = std::unique_ptr<DeviceManager>(
      DeviceManager::createDeviceManager(backendKind));
  auto receiver1 = std::unique_ptr<DeviceManager>(
      DeviceManager::createDeviceManager(backendKind));
  auto receiver2 = std::unique_ptr<DeviceManager>(
      DeviceManager::createDeviceManager(backendKind));

  ASSERT_FALSE(errToBool(sender->init()));
  ASSERT_FALSE(errToBool(receiver1->init()));
  ASSERT_FALSE(errToBool(receiver2->init()));

  auto module = llvm::make_unique<Module>();

  // Create the sender function. This function computes 1 + 2 and sends the
  // result to two receivers using two SendNodes.
  Function *senderFn = module->createFunction("senderFn");
  auto *senderFnLHS = module->createConstant(ElemKind::FloatTy, {1}, "lhs");
  auto senderLHSH = senderFnLHS->getPayload().getHandle();
  senderLHSH = {1};
  auto *senderFnRHS = module->createConstant(ElemKind::FloatTy, {1}, "rhs");
  auto senderRHSH = senderFnRHS->getPayload().getHandle();
  senderRHSH = {2};
  auto *senderFnAdd = senderFn->createAdd("add", senderFnLHS, senderFnRHS);
  auto *address1 =
      module->createPlaceholder(ElemKind::AddrTy, {1}, "address1", false);
  auto *address2 =
      module->createPlaceholder(ElemKind::AddrTy, {1}, "address2", false);
  auto *send1 = senderFn->createSend("send1", senderFnAdd, address1);
  auto *send2 = senderFn->createSend("send2", senderFnAdd, address2);

  // Create the first receiver function. This function adds 1 to the result
  // received from the sender function and saves the result (which should be 3).
  Function *receiver1Fn = module->createFunction("receiver1Fn");
  auto *receiver1Recv = receiver1Fn->createReceive(
      "recv", senderFnAdd->getResult().getType(), address1);
  auto *receiver1RHS = module->createConstant(ElemKind::FloatTy, {1}, "rhs");
  auto receiver1RHSH = receiver1RHS->getPayload().getHandle();
  receiver1RHSH = {1};
  auto *receiver1Add =
      receiver1Fn->createAdd("add", receiver1Recv, receiver1RHS);
  auto *receiver1Save = receiver1Fn->createSave("save", receiver1Add);

  // Create the first receiver function. This function adds 2 to the result
  // received from the sender function and saves the result (which should be 4).
  Function *receiver2Fn = module->createFunction("receiver2Fn");
  auto *receiver2Recv = receiver2Fn->createReceive(
      "recv", senderFnAdd->getResult().getType(), address2);
  auto *receiver2RHS = module->createConstant(ElemKind::FloatTy, {1}, "rhs");
  auto receiver2RHSH = receiver2RHS->getPayload().getHandle();
  receiver2RHSH = {2};
  auto *receiver2Add =
      receiver2Fn->createAdd("add", receiver2Recv, receiver2RHS);
  auto *receiver2Save = receiver2Fn->createSave("save", receiver2Add);

  // Compile the functions.
  std::vector<std::unique_ptr<CompiledFunction>> senderBacking,
      receiver1Backing, receiver2Backing;
  FunctionMapTy senderFunctions = compileFunctions(
      BackendKind::Interpreter, module.get(), senderBacking, {"senderFn"});
  FunctionMapTy receiver1Functions =
      compileFunctions(BackendKind::Interpreter, module.get(), receiver1Backing,
                       {"receiver1Fn"});
  FunctionMapTy receiver2Functions =
      compileFunctions(BackendKind::Interpreter, module.get(), receiver2Backing,
                       {"receiver2Fn"});

  // Add the functions to their respective devices.
  std::promise<const Module *> p1, p2, p3;
  std::future<const Module *> f1, f2, f3;

  std::tie(p1, f1) = getFutureHelper<const Module *>();
  sender->addNetwork(module.get(), senderFunctions,
                     [&p1](const Module *module, llvm::Error err) {
                       callbackHelper(p1, module, std::move(err));
                     });
  f1.wait();
  EXPECT_EQ(f1.get(), module.get());

  std::tie(p2, f2) = getFutureHelper<const Module *>();
  receiver1->addNetwork(module.get(), receiver1Functions,
                        [&p2](const Module *module, llvm::Error err) {
                          callbackHelper(p2, module, std::move(err));
                        });
  f2.wait();
  EXPECT_EQ(f2.get(), module.get());

  std::tie(p3, f3) = getFutureHelper<const Module *>();
  receiver2->addNetwork(module.get(), receiver2Functions,
                        [&p3](const Module *module, llvm::Error err) {
                          callbackHelper(p3, module, std::move(err));
                        });
  f3.wait();
  EXPECT_EQ(f3.get(), module.get());

  // Create and set ExecutionContexts for each function.
  auto senderCtx = llvm::make_unique<ExecutionContext>();
  auto receiver1Ctx = llvm::make_unique<ExecutionContext>();
  auto receiver2Ctx = llvm::make_unique<ExecutionContext>();

  senderCtx->getPlaceholderBindings()->allocate(send1->getPlaceholder());
  senderCtx->getPlaceholderBindings()->allocate(send2->getPlaceholder());
  auto AH1 = senderCtx->getPlaceholderBindings()
                 ->allocate(address1)
                 ->getHandle<uintptr_t>();
  auto AH2 = senderCtx->getPlaceholderBindings()
                 ->allocate(address2)
                 ->getHandle<uintptr_t>();

  // Get the addresses of the ReceiveNode Placeholders from the receiver devices
  // and set the address inputs of the sender's SendNodes to these addresses.
  uintptr_t addr1, addr2;
  ASSIGN_VALUE_OR_ASSERT(
      addr1, receiver1->getPlaceholderAddress(receiver1Fn->getName(),
                                              receiver1Recv->getPlaceholder()));
  ASSIGN_VALUE_OR_ASSERT(
      addr2, receiver2->getPlaceholderAddress(receiver2Fn->getName(),
                                              receiver2Recv->getPlaceholder()));
  AH1 = {addr1};
  AH2 = {addr2};

  receiver1Ctx->getPlaceholderBindings()->allocate(
      receiver1Recv->getPlaceholder());
  AH1 = receiver1Ctx->getPlaceholderBindings()
            ->allocate(address1)
            ->getHandle<uintptr_t>();
  // Set the address of this receiver's ReceiveNode to the same as the first
  // SendNode in the sender.
  AH1 = {addr1};
  receiver1Ctx->getPlaceholderBindings()->allocate(
      receiver1Save->getPlaceholder());

  receiver2Ctx->getPlaceholderBindings()->allocate(
      receiver2Recv->getPlaceholder());
  AH2 = receiver2Ctx->getPlaceholderBindings()
            ->allocate(address2)
            ->getHandle<uintptr_t>();
  // Set the address of this receiver's ReceiveNode to the same as the second
  // SendNode in the sender.
  AH2 = {addr2};
  receiver2Ctx->getPlaceholderBindings()->allocate(
      receiver2Save->getPlaceholder());

  // Run all three functions.
  std::promise<std::unique_ptr<ExecutionContext>> runP1, runP2, runP3;
  std::future<std::unique_ptr<ExecutionContext>> runF1, runF2, runF3;
  std::tie(runP1, runF1) = getFutureHelper<std::unique_ptr<ExecutionContext>>();
  std::tie(runP2, runF2) = getFutureHelper<std::unique_ptr<ExecutionContext>>();
  std::tie(runP3, runF3) = getFutureHelper<std::unique_ptr<ExecutionContext>>();

  sender->runFunction("senderFn", std::move(senderCtx),
                      [&runP1](RunIdentifierTy, llvm::Error err,
                               std::unique_ptr<ExecutionContext> context) {
                        callbackHelper(runP1, std::move(context),
                                       std::move(err));
                      });

  receiver1->runFunction("receiver1Fn", std::move(receiver1Ctx),
                         [&runP2](RunIdentifierTy, llvm::Error err,
                                  std::unique_ptr<ExecutionContext> context) {
                           callbackHelper(runP2, std::move(context),
                                          std::move(err));
                         });

  receiver2->runFunction("receiver2Fn", std::move(receiver2Ctx),
                         [&runP3](RunIdentifierTy, llvm::Error err,
                                  std::unique_ptr<ExecutionContext> context) {
                           callbackHelper(runP3, std::move(context),
                                          std::move(err));
                         });

  // Wait for the functions to finish.
  senderCtx = runF1.get();
  receiver1Ctx = runF2.get();
  receiver2Ctx = runF3.get();
  ASSERT_TRUE(senderCtx);
  ASSERT_TRUE(receiver1Ctx);
  ASSERT_TRUE(receiver2Ctx);
  EXPECT_NE(senderCtx, receiver1Ctx);
  EXPECT_NE(receiver1Ctx, receiver2Ctx);
  EXPECT_NE(senderCtx, receiver2Ctx);

  // The sender should send 3 to which the first receiver adds 1, which should
  // yield 4.
  auto RH1 = receiver1Ctx->getPlaceholderBindings()
                 ->get(receiver1Save->getPlaceholder())
                 ->getHandle();
  ASSERT_EQ(RH1.raw(0), 4);

  // The sender should send 3 to which the first receiver adds 2, which should
  // yield 5.
  auto RH2 = receiver2Ctx->getPlaceholderBindings()
                 ->get(receiver2Save->getPlaceholder())
                 ->getHandle();
  ASSERT_EQ(RH2.raw(0), 5);
}

/// Tests peer-to-peer communication with multiple senders and multiple
/// receivers.
TEST_P(PeerToPeerTest, MultipleSendersMultipleReceivers) {
  // Create two sender and two receiver devices.
  auto sender1 = std::unique_ptr<DeviceManager>(
      DeviceManager::createDeviceManager(backendKind));
  auto sender2 = std::unique_ptr<DeviceManager>(
      DeviceManager::createDeviceManager(backendKind));
  auto receiver1 = std::unique_ptr<DeviceManager>(
      DeviceManager::createDeviceManager(backendKind));
  auto receiver2 = std::unique_ptr<DeviceManager>(
      DeviceManager::createDeviceManager(backendKind));

  ASSERT_FALSE(errToBool(sender1->init()));
  ASSERT_FALSE(errToBool(sender2->init()));
  ASSERT_FALSE(errToBool(receiver1->init()));
  ASSERT_FALSE(errToBool(receiver2->init()));

  auto module = llvm::make_unique<Module>();

  // Create the first sender function. This function computes 1 + 2 and sends
  // the result to the first receiver.
  Function *sender1Fn = module->createFunction("sender1Fn");
  auto *sender1FnLHS = module->createConstant(ElemKind::FloatTy, {1}, "lhs");
  auto sender1LHSH = sender1FnLHS->getPayload().getHandle();
  sender1LHSH = {1};
  auto *sender1FnRHS = module->createConstant(ElemKind::FloatTy, {1}, "rhs");
  auto sender1RHSH = sender1FnRHS->getPayload().getHandle();
  sender1RHSH = {2};
  auto *sender1FnAdd = sender1Fn->createAdd("add", sender1FnLHS, sender1FnRHS);
  auto *address1 =
      module->createPlaceholder(ElemKind::AddrTy, {1}, "address1", false);
  auto *send1 = sender1Fn->createSend("send1", sender1FnAdd, address1);

  // Create the second sender function. This function computes 3 + 4 and sends
  // the result to the second receiver.
  Function *sender2Fn = module->createFunction("sender2Fn");
  auto *sender2FnLHS = module->createConstant(ElemKind::FloatTy, {1}, "lhs");
  auto sender2LHSH = sender2FnLHS->getPayload().getHandle();
  sender2LHSH = {3};
  auto *sender2FnRHS = module->createConstant(ElemKind::FloatTy, {1}, "rhs");
  auto sender2RHSH = sender2FnRHS->getPayload().getHandle();
  sender2RHSH = {4};
  auto *sender2FnAdd = sender2Fn->createAdd("add", sender2FnLHS, sender2FnRHS);
  auto *address2 =
      module->createPlaceholder(ElemKind::AddrTy, {1}, "address2", false);
  auto *send2 = sender2Fn->createSend("send2", sender2FnAdd, address2);

  // Create the first receiver function. This function recieves the result of
  // the first sender function, adds 1 to to it and saves the result.
  Function *receiver1Fn = module->createFunction("receiver1Fn");
  auto *receiver1Recv = receiver1Fn->createReceive(
      "recv", sender1FnAdd->getResult().getType(), address1);
  auto *receiver1RHS = module->createConstant(ElemKind::FloatTy, {1}, "rhs");
  auto receiver1RHSH = receiver1RHS->getPayload().getHandle();
  receiver1RHSH = {1};
  auto *receiver1Add =
      receiver1Fn->createAdd("add", receiver1Recv, receiver1RHS);
  auto *receiver1Save = receiver1Fn->createSave("save", receiver1Add);

  // Create the second receiver function. This function recieves the result of
  // the second sender function, adds 2 to to it and saves the result.
  Function *receiver2Fn = module->createFunction("receiver2Fn");
  auto *receiver2Recv = receiver2Fn->createReceive(
      "recv", sender2FnAdd->getResult().getType(), address2);
  auto *receiver2RHS = module->createConstant(ElemKind::FloatTy, {1}, "rhs");
  auto receiver2RHSH = receiver2RHS->getPayload().getHandle();
  receiver2RHSH = {2};
  auto *receiver2Add =
      receiver2Fn->createAdd("add", receiver2Recv, receiver2RHS);
  auto *receiver2Save = receiver2Fn->createSave("save", receiver2Add);

  // Compile all functions.
  std::vector<std::unique_ptr<CompiledFunction>> sender1Backing, sender2Backing,
      receiver1Backing, receiver2Backing;
  FunctionMapTy sender1Functions = compileFunctions(
      BackendKind::Interpreter, module.get(), sender1Backing, {"sender1Fn"});
  FunctionMapTy sender2Functions = compileFunctions(
      BackendKind::Interpreter, module.get(), sender2Backing, {"sender2Fn"});
  FunctionMapTy receiver1Functions =
      compileFunctions(BackendKind::Interpreter, module.get(), receiver1Backing,
                       {"receiver1Fn"});
  FunctionMapTy receiver2Functions =
      compileFunctions(BackendKind::Interpreter, module.get(), receiver2Backing,
                       {"receiver2Fn"});

  // Add all functions to their respective devices.
  std::promise<const Module *> p1, p2, p3, p4;
  std::future<const Module *> f1, f2, f3, f4;

  std::tie(p1, f1) = getFutureHelper<const Module *>();
  sender1->addNetwork(module.get(), sender1Functions,
                      [&p1](const Module *module, llvm::Error err) {
                        callbackHelper(p1, module, std::move(err));
                      });
  f1.wait();
  EXPECT_EQ(f1.get(), module.get());

  std::tie(p2, f2) = getFutureHelper<const Module *>();
  sender2->addNetwork(module.get(), sender2Functions,
                      [&p2](const Module *module, llvm::Error err) {
                        callbackHelper(p2, module, std::move(err));
                      });
  f2.wait();
  EXPECT_EQ(f2.get(), module.get());

  std::tie(p3, f3) = getFutureHelper<const Module *>();
  receiver1->addNetwork(module.get(), receiver1Functions,
                        [&p3](const Module *module, llvm::Error err) {
                          callbackHelper(p3, module, std::move(err));
                        });
  f3.wait();
  EXPECT_EQ(f3.get(), module.get());

  std::tie(p4, f4) = getFutureHelper<const Module *>();
  receiver2->addNetwork(module.get(), receiver2Functions,
                        [&p4](const Module *module, llvm::Error err) {
                          callbackHelper(p4, module, std::move(err));
                        });
  f4.wait();
  EXPECT_EQ(f4.get(), module.get());

  // Create and set ExecutionContexts for each function.
  auto sender1Ctx = llvm::make_unique<ExecutionContext>();
  auto sender2Ctx = llvm::make_unique<ExecutionContext>();
  auto receiver1Ctx = llvm::make_unique<ExecutionContext>();
  auto receiver2Ctx = llvm::make_unique<ExecutionContext>();

  sender1Ctx->getPlaceholderBindings()->allocate(send1->getPlaceholder());
  sender2Ctx->getPlaceholderBindings()->allocate(send2->getPlaceholder());

  // Get the addresses of the Placeholders of the ReceiveNodes in the two
  // receivers and assign those addresses to the address inputs of the SendNodes
  // in the corresponding senders.
  uintptr_t addr1, addr2;
  ASSIGN_VALUE_OR_ASSERT(
      addr1, receiver1->getPlaceholderAddress(receiver1Fn->getName(),
                                              receiver1Recv->getPlaceholder()));
  ASSIGN_VALUE_OR_ASSERT(
      addr2, receiver2->getPlaceholderAddress(receiver2Fn->getName(),
                                              receiver2Recv->getPlaceholder()));
  auto AH1 = sender1Ctx->getPlaceholderBindings()
                 ->allocate(address1)
                 ->getHandle<uintptr_t>();
  auto AH2 = sender2Ctx->getPlaceholderBindings()
                 ->allocate(address2)
                 ->getHandle<uintptr_t>();
  AH1 = {addr1};
  AH2 = {addr2};

  receiver1Ctx->getPlaceholderBindings()->allocate(
      receiver1Recv->getPlaceholder());
  AH1 = receiver1Ctx->getPlaceholderBindings()
            ->allocate(address1)
            ->getHandle<uintptr_t>();
  // Set the address of this receiver's ReceiveNode to the same as the first
  // sender's SendNode.
  AH1 = {addr1};
  receiver1Ctx->getPlaceholderBindings()->allocate(
      receiver1Save->getPlaceholder());

  receiver2Ctx->getPlaceholderBindings()->allocate(
      receiver2Recv->getPlaceholder());
  AH2 = receiver2Ctx->getPlaceholderBindings()
            ->allocate(address2)
            ->getHandle<uintptr_t>();
  // Set the address of this receiver's ReceiveNode to the same as the second
  // sender's SendNode.
  AH2 = {addr2};
  receiver2Ctx->getPlaceholderBindings()->allocate(
      receiver2Save->getPlaceholder());

  // Run all of the functions.
  std::promise<std::unique_ptr<ExecutionContext>> runP1, runP2, runP3, runP4;
  std::future<std::unique_ptr<ExecutionContext>> runF1, runF2, runF3, runF4;
  std::tie(runP1, runF1) = getFutureHelper<std::unique_ptr<ExecutionContext>>();
  std::tie(runP2, runF2) = getFutureHelper<std::unique_ptr<ExecutionContext>>();
  std::tie(runP3, runF3) = getFutureHelper<std::unique_ptr<ExecutionContext>>();
  std::tie(runP4, runF4) = getFutureHelper<std::unique_ptr<ExecutionContext>>();

  sender1->runFunction("sender1Fn", std::move(sender1Ctx),
                       [&runP1](RunIdentifierTy, llvm::Error err,
                                std::unique_ptr<ExecutionContext> context) {
                         callbackHelper(runP1, std::move(context),
                                        std::move(err));
                       });

  sender2->runFunction("sender2Fn", std::move(sender2Ctx),
                       [&runP2](RunIdentifierTy, llvm::Error err,
                                std::unique_ptr<ExecutionContext> context) {
                         callbackHelper(runP2, std::move(context),
                                        std::move(err));
                       });

  receiver1->runFunction("receiver1Fn", std::move(receiver1Ctx),
                         [&runP3](RunIdentifierTy, llvm::Error err,
                                  std::unique_ptr<ExecutionContext> context) {
                           callbackHelper(runP3, std::move(context),
                                          std::move(err));
                         });

  receiver2->runFunction("receiver2Fn", std::move(receiver2Ctx),
                         [&runP4](RunIdentifierTy, llvm::Error err,
                                  std::unique_ptr<ExecutionContext> context) {
                           callbackHelper(runP4, std::move(context),
                                          std::move(err));
                         });

  // Wait for all of the functions to complete.
  sender1Ctx = runF1.get();
  sender2Ctx = runF2.get();
  receiver1Ctx = runF3.get();
  receiver2Ctx = runF4.get();
  ASSERT_TRUE(sender1Ctx);
  ASSERT_TRUE(sender2Ctx);
  ASSERT_TRUE(receiver1Ctx);
  ASSERT_TRUE(receiver2Ctx);
  EXPECT_NE(sender1Ctx, sender2Ctx);
  EXPECT_NE(receiver1Ctx, receiver2Ctx);

  // The first sender computes and sends 1 + 2 and the receiver adds 1, so the
  // result should be 4.
  auto RH1 = receiver1Ctx->getPlaceholderBindings()
                 ->get(receiver1Save->getPlaceholder())
                 ->getHandle();
  ASSERT_EQ(RH1.raw(0), 4);

  // The first sender computes and sends 3 + 4 and the receiver adds 2, so the
  // result should be 9.
  auto RH2 = receiver2Ctx->getPlaceholderBindings()
                 ->get(receiver2Save->getPlaceholder())
                 ->getHandle();
  ASSERT_EQ(RH2.raw(0), 9);
}

/// Instantiate peer-to-peer tests for the Interpreter backend.
INSTANTIATE_TEST_CASE_P(Interpreter, PeerToPeerTest,
                        ::testing::Values(BackendKind::Interpreter));
