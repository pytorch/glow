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

/// This is an implementation of DeviceManager tailored for testing Executor
/// implementations. registerResult() gives the caller the ability to
/// dictate precisely what a subsequent call to runFunction() should return.
/// registerResult() should be called before calling Executor::run() in each
/// test. The rest of the implementation of the DeviceManager interface exists
/// to satisfy the compiler.
class TestDeviceManager final : public DeviceManager {
public:
  TestDeviceManager(unsigned numWorkers)
      : DeviceManager(BackendKind::Interpreter), threadPool_(numWorkers) {}

  /// The functions below are the interface for DeviceManager. See
  /// glow::DeviceManager for descriptions of what they do. Since this
  /// class exists only to help test Executor implementations, the only
  /// important function is runFunction().
  void init() override {}

  void addNetwork(const Module *module, FunctionMapTy functions,
                  ReadyCBTy readyCB) override {}

  void evictNetwork(llvm::StringRef functionName) override {
    // Erase the entry so that the same function name can be used to register
    // another result.
    resultMap_.erase(functionName);
  }

  /// Look up the previously registered response for \p functionName and
  /// call \p resultCB with it after checking that \ctx contains the
  /// expected Placeholder-Tensor mappings.
  void doRunFunction(std::string functionName, std::shared_ptr<Context> ctx,
                     ResultCBTy resultCB) {
    RunIdentifierTy runId = 0;
    ResultCode resultCode = ResultCode::Failed;
    std::unique_ptr<Context> resultContext = nullptr;

    // Retrieve the registered response for the function if there is one.
    if (ctx && resultCB && resultMap_.count(functionName)) {
      std::unique_ptr<RunFunctionResult> registeredResult =
          std::move(resultMap_[functionName]);

      // Check that ctx contains the expected Placeholder-Tensor mappings.
      std::unique_ptr<Context> inputContext =
          std::move(registeredResult->inputContext);

      if (Context::compare(ctx.get(), inputContext.get())) {
        // If ctx contains all expected mappings, overwrite the default
        // runId, resultCode and resultContext with the registered ones.
        runId = registeredResult->runId;
        resultCode = registeredResult->resultCode;
        resultContext = std::move(registeredResult->resultContext);
      }
    }

    resultCB(runId, resultCode, std::move(resultContext));
  }

  /// Do not call this at the same time as registerResult().
  runtime::RunIdentifierTy runFunction(std::string functionName,
                                       std::unique_ptr<Context> ctx,
                                       ResultCBTy resultCB) override {
    // Give the call to the thread pool to process to make the tests
    // multithreaded if needed.
    std::shared_ptr<Context> sharedCtx = std::move(ctx);
    this->threadPool_.submit([this, functionName, sharedCtx, resultCB]() {
      this->doRunFunction(functionName, sharedCtx, resultCB);
    });
    return 0;
  }

  void stop(bool /*block*/) override {}

  uint64_t getMaximumMemory() const override {
    return std::numeric_limits<uint64_t>::max();
  }

  uint64_t getAvailableMemory() const override {
    return std::numeric_limits<uint64_t>::max();
  }

  bool isMemoryAvailable(uint64_t /*estimate*/) const override { return true; }

  /// Register a result that should be returned by the subsequent call to
  /// runFunction with the same \p functionName. The callback for that call
  /// to runFunction will be called with \p runId, \p resultCode, and \p
  /// \p resultContext if the context passed in to runFunction matches \p
  /// inputContext. \returns true if registration was successful, false if not.
  /// Do not call this at the same time as runFunction().
  bool registerResult(const std::string &functionName, RunIdentifierTy runId,
                      ResultCode resultCode,
                      std::unique_ptr<Context> inputContext,
                      std::unique_ptr<Context> resultContext) {
    bool registered = false;

    if (!resultMap_.count(functionName)) {
      // If the function name has not already been registered, insert it into
      // resultMap_.
      std::tie(std::ignore, registered) = resultMap_.insert(std::make_pair(
          functionName, llvm::make_unique<RunFunctionResult>(
                            runId, resultCode, std::move(inputContext),
                            std::move(resultContext))));
    }

    return registered;
  }

private:
  /// This struct wraps all of the data needed to reply to a runFunction() call.
  /// It exists so that that all of these things can be stored in one map.
  struct RunFunctionResult {
    /// The run ID that should be returned.
    RunIdentifierTy runId;
    /// The result code that should be returned.
    ResultCode resultCode;
    /// The expected input Context for the invocation.
    std::unique_ptr<Context> inputContext;
    /// The result Context that should be returned.
    std::unique_ptr<Context> resultContext;

    /// Constructor.
    RunFunctionResult(RunIdentifierTy run, ResultCode result,
                      std::unique_ptr<Context> inputCtx,
                      std::unique_ptr<Context> resultCtx)
        : runId(run), resultCode(result), inputContext(std::move(inputCtx)),
          resultContext(std::move(resultCtx)) {}
  };

  /// Map of function name -> RunFunctionResult instance containing the
  /// RunFunctionResult instance for the function.
  using TestDeviceManagerResultMapTy =
      std::unordered_map<std::string, std::unique_ptr<RunFunctionResult>>;

  /// Map for storing registered results.
  TestDeviceManagerResultMapTy resultMap_;
  /// Thread pool for executing runFunction() in a multithreaded fashion.
  ThreadPool threadPool_;
};

using TestDeviceManagerMapTy =
    std::unordered_map<DeviceIDTy, std::shared_ptr<TestDeviceManager>>;
using PlaceholderNameMapTy =
    std::unordered_map<std::string, std::unique_ptr<Placeholder>>;
using DAGNodeNameMapTy =
    std::unordered_map<std::string, std::unique_ptr<DAGNode>>;

/// This class serves as an interface to a test created by ExecutorTestBuilder.
/// It also contains the resources necessary to run the test. Instances are
/// meant to be created only by ExecutorTestBuilder.
class ExecutorTest final {
public:
  /// Constructor.
  ExecutorTest(const std::shared_ptr<Executor> &executor,
               std::unique_ptr<DAGNode> root, std::unique_ptr<Type> type,
               DAGNodeNameMapTy nodes, PlaceholderNameMapTy placeholders,
               std::unique_ptr<Context> inputCtx,
               std::unique_ptr<Context> outputCtx, RunIdentifierTy runId,
               ResultCode resultCode)
      : executor_(executor), root_(std::move(root)), type_(std::move(type)),
        nodes_(std::move(nodes)), placeholders_(std::move(placeholders)),
        inputCtx_(std::move(inputCtx)), outputCtx_(std::move(outputCtx)),
        runId_(runId), resultCode_(resultCode), testRun_(false) {}

  /// Run the test.
  bool run() {
    if (testRun_) {
      assert(!"Test has already been run!");
    }

    // Variables for storing runId and resultCode actually returned by
    // Executor::run() via its callback.
    RunIdentifierTy executorRunId;
    ResultCode executorResultCode;
    std::unique_ptr<Context> executorOutputCtx;

    // Call Executor::run().
    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    executor_->run(root_.get(), std::move(inputCtx_), runId_,
                   [&promise, &executorRunId, &executorResultCode,
                    &executorOutputCtx](RunIdentifierTy runId, ResultCode code,
                                        std::unique_ptr<Context> context) {
                     executorRunId = runId;
                     executorResultCode = code;
                     executorOutputCtx = std::move(context);
                     promise.set_value();
                   });

    future.wait();

    // Check that the values returned in the Executor callback match
    // expectations.
    bool runIdsMatch = executorRunId == runId_;
    bool resultCodesMatch = executorResultCode == resultCode_;
    bool runFailed = executorResultCode == ResultCode::Failed;
    bool contextsMatch =
        Context::compare(executorOutputCtx.get(), outputCtx_.get());

    // If the run failed, we shouldn't expect contextsMatch to be true.
    bool testPassed =
        runIdsMatch && resultCodesMatch && (runFailed || contextsMatch);

    testRun_ = true;

    return testPassed;
  }

private:
  /// The Executor to run the test with.
  std::shared_ptr<Executor> executor_;
  /// The root node of the DAG being tested.
  std::unique_ptr<DAGNode> root_;
  /// The Type for all of the Placeholders that will be used during execution.
  std::unique_ptr<Type> type_;
  /// All nodes in the DAG.
  DAGNodeNameMapTy nodes_;
  /// All Placeholders that will be used during execution.
  PlaceholderNameMapTy placeholders_;
  /// The input Context that should be passed to Executor::run() when running
  /// the test.
  std::unique_ptr<Context> inputCtx_;
  /// The expected Context that the Executor should return.
  std::unique_ptr<Context> outputCtx_;
  /// The run ID that should be passed to Executor::run() when running
  /// the test.
  RunIdentifierTy runId_;
  /// The expected result code that the Executor should return.
  ResultCode resultCode_;
  /// Tracks whether or not the test has already been run.
  bool testRun_;
};

/// This class helps build tests for testing Executor implementations. It
/// presents a simple interface for executor DAG construction; nodes are added
/// by specifying its parents, device ID, and named inputs and outputs. This
/// builder class takes care of all of the work needed to actually run this DAG:
/// creation of Placeholders and Tensors for all inputs and outputs; creation of
/// input/output Context for each node to verify that each one receives the
/// correct input and produces the correct output; and registration with
/// the TestDeviceManager.
class ExecutorTestBuilder final {
public:
  /// Constructor. The exact value of type_ doesn't really matter since the
  /// important thing to test is that that Placeholder values are propagated
  /// between Contexts correctly.
  ExecutorTestBuilder(const std::shared_ptr<Executor> &executor,
                      const TestDeviceManagerMapTy &deviceManagers)
      : executor_(executor), root_(llvm::make_unique<DAGNode>()),
        ctx_(llvm::make_unique<Context>()),
        type_(
            std::unique_ptr<Type>(new Type(ElemKind::FloatTy, {32, 64, 128}))),
        resultCode_(ResultCode::Executed), deviceManagers_(deviceManagers) {}

  /// Add a node named \p name to the DAG with parents \p parents that runs on a
  /// device specified by \p deviceId. A RuntimeBundle is created for the node
  /// with runtime symbol information created from \p inputs and \p outputs.
  /// \p runId is the run ID for the node and \p resultCode is the desired
  /// execution status. If \p parents is empty, the new node is added as a child
  /// of the root.
  void addNode(const std::string &name, DeviceIDTy deviceId,
               llvm::ArrayRef<llvm::StringRef> parents,
               llvm::ArrayRef<llvm::StringRef> inputs,
               llvm::ArrayRef<llvm::StringRef> outputs, RunIdentifierTy runId,
               ResultCode resultCode) {
    auto newNode = llvm::make_unique<DAGNode>();
    auto *newNodeRawPtr = newNode.get();

    // If this is the first node being added, record the run ID for the graph.
    // Otherwise, make sure that the runId matches that of the previous nodes.
    if (nodes_.empty()) {
      runId_ = runId;
    } else {
      assert(runId == runId_ && "Node run ID does not match rest of graph!");
    }

    // If the result code for this node is ResultCode::Failed, set the expected
    // result code for the entire test to ResultCode::Failed.
    resultCode_ = resultCode == ResultCode::Failed ? resultCode : resultCode_;

    // Add parents to the list of parents in the new node and add the newNode
    // to the list of children in the parents. If the parent list is empty,
    // make the root the only parent. Also, update the set of known leaves
    // by removing any parents of the new node from it. This will be useful
    // later.
    if (!parents.empty()) {
      for (const auto &parent : parents) {
        auto it = nodes_.find(parent);
        if (it == nodes_.end()) {
          assert(!"Parent specified for node not found!");
        }
        DAGNode *parentPtr = (it->second).get();
        (newNode->parents).emplace_back(parentPtr);
        (parentPtr->children).emplace_back(newNodeRawPtr);
        leaves_.erase(parentPtr);
      }
    } else {
      (newNode->parents).emplace_back(root_.get());
      (root_->children).emplace_back(newNode.get());
    }

    // Iterate through inputs and outputs and:
    // 1) Create Placeholders and Tensors for inputs/output names that have not
    //    been mapped to a Placeholder yet.
    // 2) Assemble the input Context that the node is expected to be called with
    //    and the Context that the node should produce as output.
    // 3) Generate the symbol table for the new node by generating
    //    RuntimeSymbolInfo objects for each input and output.
    SymbolTableTy symbolTable;
    size_t offset = 0;

    auto nodeInputCtx = llvm::make_unique<Context>();
    auto nodeOutputCtx = llvm::make_unique<Context>();

    for (const auto &input : inputs) {
      insertSymbolIntoContext(input, nodeInputCtx.get());

      RuntimeSymbolInfo runtimeSymbolInfo{
          /*size=*/type_->getSizeInBytes(), /*offset=*/offset,
          /*type=*/*type_, /*input=*/true, /*output=*/false};
      symbolTable.insert(std::make_pair(input, runtimeSymbolInfo));
      offset += type_->getSizeInBytes();
    }

    for (const auto &output : outputs) {
      insertSymbolIntoContext(output, nodeOutputCtx.get());

      RuntimeSymbolInfo runtimeSymbolInfo{
          /*size=*/type_->getSizeInBytes(), /*offset=*/offset,
          /*type=*/*type_, /*input=*/false, /*output=*/true};
      symbolTable.insert(std::make_pair(output, runtimeSymbolInfo));
      offset += type_->getSizeInBytes();
    }

    // Set the name, device ID, and RuntimeBundle of the new node.
    newNode->name = name;
    newNode->deviceID = deviceId;
    newNode->runtimeBundle =
        RuntimeBundle(symbolTable, /*constWeight=*/0, /*mutableWeight=*/0,
                      /*activations=*/0);

    // Register node result with the appropriate DeviceManager.
    auto it = deviceManagers_.find(deviceId);

    if (it == deviceManagers_.end()) {
      assert(!"No test device manager found for this device ID");
    }

    std::shared_ptr<TestDeviceManager> deviceManager = it->second;
    bool registered = deviceManager->registerResult(name, runId, resultCode,
                                                    std::move(nodeInputCtx),
                                                    std::move(nodeOutputCtx));

    (void)registered;
    assert(registered && "Node registration was not successful");

    // Add the new node to nodes_ and leaves_.
    nodes_.insert(std::make_pair(name, std::move(newNode)));
    leaves_.insert(newNodeRawPtr);
  }

  /// Emit the test built so far and clear any state in the builder.
  ExecutorTest emitTest() {
    // Get the input and output symbol names for the whole DAG.
    std::vector<std::string> inputSymbols = gatherInputSymbols();
    std::vector<std::string> outputSymbols = gatherOutputSymbols();

    // Generate the input and output Contexts for the test. This input Context
    // contains the input Placeholders of all root nodes and output Placeholders
    // of all leaves (but backed by zero tensors). This is the Context that
    // needs to be passed to Executor::run() to run the test. The output Context
    // contains the same Placeholders as the input Context, but the leaves'
    // output Placeholders are mapped to their expected output Tensors. This
    // Context is used to verify that the one returned by the Executor is
    // correct.
    auto inputCtx = llvm::make_unique<Context>();
    auto outputCtx = llvm::make_unique<Context>();

    for (const auto &symbol : inputSymbols) {
      insertSymbolIntoContext(symbol, inputCtx.get());
      insertSymbolIntoContext(symbol, outputCtx.get());
    }

    for (const auto &symbol : outputSymbols) {
      auto *placeholder = ctx_->getPlaceholderByName(symbol);
      if (!placeholder) {
        assert(!"Placeholder for DAG output not found!");
      }
      inputCtx->allocate(placeholder)->zero();
      insertSymbolIntoContext(symbol, outputCtx.get());
    }

    // Create the test object.
    ExecutorTest test(executor_, std::move(root_), std::move(type_),
                      std::move(nodes_), std::move(placeholders_),
                      std::move(inputCtx), std::move(outputCtx), runId_,
                      resultCode_);

    // Reset builder state to allow a new test to be constructed with this
    // instance.
    root_ = llvm::make_unique<DAGNode>();
    ctx_->clear();
    type_ = std::unique_ptr<Type>(new Type(ElemKind::FloatTy, {1, 2, 2}));
    nodes_.clear();
    leaves_.clear();
    placeholders_.clear();
    resultCode_ = ResultCode::Executed;

    return test;
  }

private:
  /// Collect all input symbol names for the test. \returns a vector containing
  /// the names of all test input symbols.
  std::vector<std::string> gatherInputSymbols() const {
    std::vector<std::string> inputSymbols;

    // Input symbols for the entire test are the inputs of all nodes that have
    // no parents.
    for (const auto &node : root_->children) {
      const SymbolTableTy &symbolTable = (node->runtimeBundle).getSymbolTable();

      for (const auto &symbolPair : symbolTable) {
        const auto &symbolName = symbolPair.first;
        const auto &symbolInfo = symbolPair.second;

        if (symbolInfo.input) {
          inputSymbols.emplace_back(symbolName);
        }
      }
    }

    return inputSymbols;
  }

  /// Collect all output symbol names for the test. \returns a vector containing
  /// the names of all test output symbols.
  std::vector<std::string> gatherOutputSymbols() const {
    std::vector<std::string> outputSymbols;

    // Input symbols for the entire test are the outputs of all nodes that have
    // no children.
    for (const auto &node : leaves_) {
      const SymbolTableTy &symbolTable = (node->runtimeBundle).getSymbolTable();

      for (const auto &symbolPair : symbolTable) {
        const auto &symbolName = symbolPair.first;
        const auto &symbolInfo = symbolPair.second;

        if (symbolInfo.output) {
          outputSymbols.emplace_back(symbolName);
        }
      }
    }

    return outputSymbols;
  }

  /// Insert a Placeholder named \p name with type type_ into \p context
  /// and generate a random Tensor for it. If this Placeholder has already been
  /// mapped for the test being created, reuse the existing value.
  void insertSymbolIntoContext(llvm::StringRef name, Context *context) {
    auto it = placeholders_.find(name);

    if (it == placeholders_.end()) {
      // This is a new symbol. Create a Placeholder and an initialize and new
      // Tensor for it.
      auto placeholder = llvm::make_unique<Placeholder>(name, type_.get(),
                                                        /*trainable=*/false);
      auto *tensor = ctx_->allocate(placeholder.get());
      tensor->init(Tensor::InitKind::Xavier, 1.0, rng_);
      context->insert(placeholder.get(), tensor->clone());
      placeholders_[name] = std::move(placeholder);
    } else {
      // This is a symbol that already has an associated Placeholder and Tensor.
      // Copy that Tensor.
      auto *placeholder = (it->second).get();
      const auto *tensor = ctx_->get(placeholder);
      context->insert(placeholder, tensor->clone());
    }
  }

  /// The Executor being tested.
  std::shared_ptr<Executor> executor_;
  /// The root of the DAG being constructed.
  std::unique_ptr<DAGNode> root_;
  /// This Context holds all created and initialized Placeholders for the test.
  std::unique_ptr<Context> ctx_;
  /// The Type for all Placeholders and Tensors in the test. The exact value
  /// is not important; the main thing being tested is the propagation of
  /// Placeholders and Tensors as the DAG executes.
  std::unique_ptr<Type> type_;
  /// PRNG for filling Tensors.
  PseudoRNG rng_;
  /// The nodes in the DAG being constructed.
  DAGNodeNameMapTy nodes_;
  /// The leaves in the DAG being constructed. This helps collect output symbols
  /// during test emission.
  std::unordered_set<const DAGNode *> leaves_;
  /// All Placeholders in the test.
  PlaceholderNameMapTy placeholders_;
  /// The run ID for the DAG.
  RunIdentifierTy runId_;
  /// The expected result code for the DAG.
  ResultCode resultCode_;
  /// Map from DeviceIDTy -> TestDeviceManager. This enables the construction of
  /// tests with nodes spread across devices.
  const TestDeviceManagerMapTy &deviceManagers_;
};

/// This test fixture provides ThreadPoolExecutor, ExecutorTestBuilder,
/// DeviceManagerMapTy, and TestDeviceManagerMapTy instances to all tests.
class ThreadPoolExecutorTest : public ::testing::Test {
protected:
  ThreadPoolExecutorTest()
      : executor_(std::shared_ptr<Executor>(
            createExecutor(deviceManagerMap_, ExecutorKind::ThreadPool))),
        testBuilder_(executor_, testDeviceManagerMap_) {}
  ~ThreadPoolExecutorTest() = default;

  /// The Executor being tested.
  std::shared_ptr<Executor> executor_;
  /// An ExecutorTestBuilder instance for creating tests.
  ExecutorTestBuilder testBuilder_;
  /// DeviceManager map for initializing executor_.
  Executor::DeviceManagerMapTy deviceManagerMap_;
  /// TestDeviceManager map for initializing testBuilder_.
  TestDeviceManagerMapTy testDeviceManagerMap_;
};

/// Tests that an empty DAG is handled correctly.
TEST_F(ThreadPoolExecutorTest, EmptyDAG) {
  constexpr RunIdentifierTy testRunId = 10;

  // Make a Context with one Placeholder in it to make sure Executor::run()
  // doesn't modify it when the root given to it is null. Make two identical
  // copies; one to give to Executor::run(), and another to compare the
  // returned Context with.
  PseudoRNG rng;
  auto type = std::unique_ptr<Type>(new Type(ElemKind::FloatTy, {1, 2, 2}));
  auto placeholder = llvm::make_unique<Placeholder>("a", type.get(),
                                                    /*trainable=*/false);
  auto testCtx = llvm::make_unique<Context>();
  auto refCtx = llvm::make_unique<Context>();
  auto *tensor = testCtx->allocate(placeholder.get());
  tensor->init(Tensor::InitKind::Xavier, 1.0, rng);
  refCtx->insert(placeholder.get(), tensor->clone());

  // Variables for storing runId and resultCode actually returned by
  // Executor::run() via its callback.
  RunIdentifierTy executorRunId;
  ResultCode executorResultCode;
  std::unique_ptr<Context> executorOutputCtx;

  // Call Executor::run().
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  executor_->run(nullptr, std::move(testCtx), testRunId,
                 [&promise, &executorRunId, &executorResultCode,
                  &executorOutputCtx](RunIdentifierTy runId, ResultCode code,
                                      std::unique_ptr<Context> context) {
                   executorRunId = runId;
                   executorResultCode = code;
                   executorOutputCtx = std::move(context);
                   promise.set_value();
                 });

  future.wait();

  EXPECT_EQ(executorRunId, testRunId);
  EXPECT_EQ(executorResultCode, ResultCode::Executed);
  EXPECT_TRUE(Context::compare(refCtx.get(), executorOutputCtx.get()));
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
      std::make_shared<TestDeviceManager>(deviceManagerThreads);
  deviceManagerMap_.insert(std::make_pair(testDeviceId, deviceManager));
  testDeviceManagerMap_.insert(std::make_pair(testDeviceId, deviceManager));

  // Build the DAG. The DAG created below looks like this:
  /**
   *         root
   *          |
   *          v
   *         net
   **/

  testBuilder_.addNode("net", testDeviceId,
                       /*parents=*/{}, {"netInput"}, {"netOutput"}, testRunId,
                       ResultCode::Executed);

  ExecutorTest test = testBuilder_.emitTest();
  EXPECT_TRUE(test.run());
}

/// Tests that several instances of a single node DAG can be run in parallel.
TEST_F(ThreadPoolExecutorTest, ConcurrentSingleNode) {
  constexpr RunIdentifierTy baseTestRunId = 10;
  constexpr DeviceIDTy testDeviceId = 111;
  constexpr unsigned deviceManagerThreads = 3;
  constexpr unsigned numConcurrentRuns = 1000;

  // Make a TestDeviceManager and insert into the DeviceManagerMap map (which
  // the ThreadPoolExecutor has a reference to) and the TestDeviceManager map
  // (which the ExecutorTestBuilder has a reference to).
  auto deviceManager =
      std::make_shared<TestDeviceManager>(deviceManagerThreads);
  deviceManagerMap_.insert(std::make_pair(testDeviceId, deviceManager));
  testDeviceManagerMap_.insert(std::make_pair(testDeviceId, deviceManager));

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
                         baseTestRunId + i, ResultCode::Executed);
    ExecutorTest t = testBuilder_.emitTest();

    std::thread th([&mtx, &driverCV, &threadCV, &threadsReady, &testsPassed,
                    test = std::move(t)]() mutable {
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
    driverCV.wait(
        lock, [&threadsReady] { return threadsReady == numConcurrentRuns; });
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
/// runId returns ResultCode::Failed.
TEST_F(ThreadPoolExecutorTest, ConcurrentSingleNodeDuplicateRunId) {
  constexpr RunIdentifierTy testRunId = 10;
  constexpr DeviceIDTy testDeviceId = 111;
  constexpr unsigned deviceManagerThreads = 1;
  constexpr unsigned numConcurrentRuns = 100;

  // Make a TestDeviceManager and insert into the DeviceManagerMap map (which
  // the ThreadPoolExecutor has a reference to) and the TestDeviceManager map
  // (which the ExecutorTestBuilder has a reference to).
  auto deviceManager =
      std::make_shared<TestDeviceManager>(deviceManagerThreads);
  deviceManagerMap_.insert(std::make_pair(testDeviceId, deviceManager));
  testDeviceManagerMap_.insert(std::make_pair(testDeviceId, deviceManager));

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
                         ResultCode::Executed);
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
      std::make_shared<TestDeviceManager>(deviceManagerThreads);
  deviceManagerMap_.insert(std::make_pair(testDeviceId, deviceManager));
  testDeviceManagerMap_.insert(std::make_pair(testDeviceId, deviceManager));

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
                       /*outputs=*/{"alphaOut"}, testRunId,
                       ResultCode::Executed);
  testBuilder_.addNode("beta", testDeviceId,
                       /*parents=*/{}, /*inputs=*/{"betaIn"},
                       /*outputs=*/{"betaOut"}, testRunId,
                       ResultCode::Executed);
  testBuilder_.addNode("gamma", testDeviceId,
                       /*parents=*/{"alpha", "beta"},
                       /*inputs=*/{"alphaOut", "betaOut"},
                       /*outputs=*/{"deltaIn", "epsIn"}, testRunId,
                       ResultCode::Executed);
  testBuilder_.addNode("delta", testDeviceId,
                       /*parents=*/{"gamma"}, /*inputs=*/{"deltaIn"},
                       /*outputs=*/{"deltaOut"}, testRunId,
                       ResultCode::Executed);
  testBuilder_.addNode("eps", testDeviceId,
                       /*parents=*/{"gamma"}, /*inputs=*/{"epsIn"},
                       /*outputs=*/{"epsOut"}, testRunId, ResultCode::Executed);

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
      std::make_shared<TestDeviceManager>(deviceManagerThreads);
  deviceManagerMap_.insert(std::make_pair(testDeviceId, deviceManager));
  testDeviceManagerMap_.insert(std::make_pair(testDeviceId, deviceManager));

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
                       /*outputs=*/{"alphaOut"}, testRunId,
                       ResultCode::Executed);
  testBuilder_.addNode("beta", testDeviceId,
                       /*parents=*/{"alpha"}, /*inputs=*/{"alphaOut"},
                       /*outputs=*/{"betaOut"}, testRunId,
                       ResultCode::Executed);
  testBuilder_.addNode("gamma", testDeviceId,
                       /*parents=*/{"beta"},
                       /*inputs=*/{"betaOut"},
                       /*outputs=*/{"gammaOut"}, testRunId,
                       ResultCode::Executed);
  testBuilder_.addNode("delta", testDeviceId,
                       /*parents=*/{}, /*inputs=*/{"deltaIn"},
                       /*outputs=*/{"deltaOut"}, testRunId,
                       ResultCode::Executed);
  testBuilder_.addNode("eps", testDeviceId,
                       /*parents=*/{"delta"}, /*inputs=*/{"deltaOut"},
                       /*outputs=*/{"epsOut"}, testRunId, ResultCode::Failed);
  testBuilder_.addNode("zeta", testDeviceId,
                       /*parents=*/{"eps"}, /*inputs=*/{"epsOut"},
                       /*outputs=*/{"zetaOut"}, testRunId,
                       ResultCode::Executed);

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
        std::make_shared<TestDeviceManager>(deviceManagerThreads);
    deviceManagerMap_.insert(std::make_pair(deviceId, deviceManager));
    testDeviceManagerMap_.insert(std::make_pair(deviceId, deviceManager));
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
                       /*outputs=*/{"alphaOut"}, testRunId,
                       ResultCode::Executed);
  testBuilder_.addNode("beta", testDeviceIdB,
                       /*parents=*/{}, /*inputs=*/{"betaIn"},
                       /*outputs=*/{"betaOut"}, testRunId,
                       ResultCode::Executed);
  testBuilder_.addNode("gamma", testDeviceIdC,
                       /*parents=*/{"alpha", "beta"},
                       /*inputs=*/{"alphaOut", "betaOut"},
                       /*outputs=*/{"deltaIn", "epsIn"}, testRunId,
                       ResultCode::Executed);
  testBuilder_.addNode("delta", testDeviceIdA,
                       /*parents=*/{"gamma"}, /*inputs=*/{"deltaIn"},
                       /*outputs=*/{"deltaOut"}, testRunId,
                       ResultCode::Executed);
  testBuilder_.addNode("eps", testDeviceIdB,
                       /*parents=*/{"gamma"}, /*inputs=*/{"epsIn"},
                       /*outputs=*/{"epsOut"}, testRunId, ResultCode::Executed);

  ExecutorTest test = testBuilder_.emitTest();
  EXPECT_TRUE(test.run());
}

/// Tests that several instances of a DAG with multiple nodes can run correctly
/// in parallel.
TEST_F(ThreadPoolExecutorTest, ConcurrentMultiNode) {
  constexpr RunIdentifierTy baseTestRunId = 10;
  constexpr DeviceIDTy testDeviceId = 111;
  constexpr unsigned deviceManagerThreads = 3;
  constexpr unsigned numConcurrentRuns = 1000;

  // Make a TestDeviceManager and insert it into the DeviceManagerMap map
  // (which the ThreadPoolExecutor has a reference to) and the TestDeviceManager
  // map (which the ExecutorTestBuilder has a reference to).
  auto deviceManager =
      std::make_shared<TestDeviceManager>(deviceManagerThreads);
  deviceManagerMap_.insert(std::make_pair(testDeviceId, deviceManager));
  testDeviceManagerMap_.insert(std::make_pair(testDeviceId, deviceManager));

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
                         /*outputs=*/{"alphaOut"}, baseTestRunId + i,
                         ResultCode::Executed);
    testBuilder_.addNode(beta, testDeviceId,
                         /*parents=*/{}, /*inputs=*/{"betaIn"},
                         /*outputs=*/{"betaOut"}, baseTestRunId + i,
                         ResultCode::Executed);
    testBuilder_.addNode(gamma, testDeviceId,
                         /*parents=*/{alpha, beta},
                         /*inputs=*/{"alphaOut", "betaOut"},
                         /*outputs=*/{"deltaIn", "epsIn"}, baseTestRunId + i,
                         ResultCode::Executed);
    testBuilder_.addNode(delta, testDeviceId,
                         /*parents=*/{gamma}, /*inputs=*/{"deltaIn"},
                         /*outputs=*/{"deltaOut"}, baseTestRunId + i,
                         ResultCode::Executed);
    testBuilder_.addNode(eps, testDeviceId,
                         /*parents=*/{gamma}, /*inputs=*/{"epsIn"},
                         /*outputs=*/{"epsOut"}, baseTestRunId + i,
                         ResultCode::Executed);

    ExecutorTest t = testBuilder_.emitTest();
    std::thread th([&mtx, &driverCV, &threadCV, &threadsReady, &testsPassed,
                    test = std::move(t)]() mutable {
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
    driverCV.wait(
        lock, [&threadsReady] { return threadsReady == numConcurrentRuns; });
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
