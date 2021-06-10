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

#include "glow/Runtime/Executor/ThreadPoolExecutor.h"
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
class TestDeviceManager final : public runtime::DeviceManager {
public:
  TestDeviceManager(unsigned numWorkers, const DeviceConfig &deviceConfig)
      : DeviceManager(deviceConfig), threadPool_(numWorkers) {}

  /// The functions below are the interface for DeviceManager. See
  /// glow::DeviceManager for descriptions of what they do. Since this
  /// class exists only to help test Executor implementations, the only
  /// important function is runFunction().
  void addNetwork(const Module *module, FunctionMapTy functions,
                  ReadyCBTy readyCB) override {}

  void evictNetwork(std::string functionName,
                    EvictFunctionCBTy evictCB) override {
    // Erase the entry so that the same function name can be used to register
    // another result.

    if (!resultMap_.erase(functionName)) {
      evictCB(
          functionName,
          MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_NET_NOT_FOUND,
                   strFormat("Could not find function with name %s to evict",
                             functionName.c_str())));
      return;
    }
    evictCB(functionName, Error::success());
  }

  /// Look up the previously registered response for \p functionName and
  /// call \p resultCB with it after checking that \p context contains the
  /// expected Placeholder-Tensor mappings.
  void doRunFunction(std::string functionName,
                     std::unique_ptr<ExecutionContext> context,
                     ResultCBTy resultCB) {

    RunIdentifierTy runId = 0;
    bool successResult = false;

    // Retrieve the registered response for the function if there is one.
    if (context && resultCB && resultMap_.count(functionName)) {
      std::unique_ptr<RunFunctionResult> registeredResult =
          std::move(resultMap_[functionName]);

      // Check that context contains the expected Placeholder-Tensor mappings.
      std::unique_ptr<ExecutionContext> inputContext =
          std::move(registeredResult->inputContext);

      bool equalInputs = true;
      for (auto &p : inputContext->getPlaceholderBindings()->pairs()) {
        Tensor *CT = context->getPlaceholderBindings()->get(p.first);
        if (!CT) {
          equalInputs = false;
          break;
        }
        equalInputs &= p.second.isEqual(*CT, 0.0001, true);
      }

      if (equalInputs) {
        // If bindings contains all expected mappings, overwrite the default
        // runId, result and resultContext with the registered
        // ones.
        runId = registeredResult->runId;
        successResult = registeredResult->success;

        for (const auto &p :
             registeredResult->resultContext->getPlaceholderBindings()
                 ->pairs()) {
          context->getPlaceholderBindings()->get(p.first)->assign(&p.second);
        }
      }
    }

    if (successResult) {
      resultCB(runId, Error::success(), std::move(context));
    } else {
      resultCB(runId, MAKE_ERR("An error occurred"), std::move(context));
    }
  }

  /// Do not call this at the same time as registerResult().
  runtime::RunIdentifierTy
  runFunction(std::string functionName,
              std::unique_ptr<ExecutionContext> context,
              ResultCBTy resultCB) override {
    // Give the call to the thread pool to process to make the tests
    // multithreaded if needed.
    this->threadPool_.submit(
        [this, functionName, context = std::move(context), resultCB]() mutable {
          this->doRunFunction(functionName, std::move(context), resultCB);
        });
    return 0;
  }

  uint64_t getMaximumMemory() const override {
    return std::numeric_limits<uint64_t>::max();
  }

  uint64_t getAvailableMemory() const override {
    return std::numeric_limits<uint64_t>::max();
  }

  bool isMemoryAvailable(uint64_t /*estimate*/) const override { return true; }

  /// Register a result that should be returned by the subsequent call to
  /// runFunction with the same \p functionName. The callback for that call
  /// to runFunction will be called with \p runId, \p success, and \p
  /// \p resultContext if the context passed in to runFunction
  /// matches \p inputContext. \returns true if registration was
  /// successful, false if not. Do not call this at the same time as
  /// runFunction().
  bool registerResult(const std::string &functionName, RunIdentifierTy runId,
                      bool success,
                      std::unique_ptr<ExecutionContext> inputContext,
                      std::unique_ptr<ExecutionContext> resultContext) {
    bool registered = false;

    if (!resultMap_.count(functionName)) {
      // If the function name has not already been registered, insert it into
      // resultMap_.
      std::tie(std::ignore, registered) = resultMap_.insert(std::make_pair(
          functionName, glow::make_unique<RunFunctionResult>(
                            runId, success, std::move(inputContext),
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
    /// If success then no error should be returned otherwise an Error should be
    /// returned.
    bool success;
    /// The expected input context for the invocation.
    std::unique_ptr<ExecutionContext> inputContext;
    /// The result context that should be returned.
    std::unique_ptr<ExecutionContext> resultContext;

    /// Constructor.
    RunFunctionResult(RunIdentifierTy run, bool successParam,
                      std::unique_ptr<ExecutionContext> inputcontext,
                      std::unique_ptr<ExecutionContext> resultcontext)
        : runId(run), success(successParam),
          inputContext(std::move(inputcontext)),
          resultContext(std::move(resultcontext)) {}
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
               std::unique_ptr<DAGNode> root, std::unique_ptr<Module> module,
               std::unique_ptr<Type> type, DAGNodeNameMapTy nodes,
               PlaceholderNameMapTy placeholders,
               std::unique_ptr<ExecutionContext> inputContext,
               std::unique_ptr<ExecutionContext> outputContext,
               RunIdentifierTy runId, bool expectSuccess)
      : executor_(executor), root_(std::move(root)), module_(std::move(module)),
        type_(std::move(type)), nodes_(std::move(nodes)),
        placeholders_(std::move(placeholders)),
        inputContext_(std::move(inputContext)),
        outputContext_(std::move(outputContext)), runId_(runId),
        expectSuccess_(expectSuccess), testRun_(false) {
    root_->module = module_.get();
    // Create context pool.
    executor_->createPool(root_.get(), 1000, false, false);
  }

  /// Run the test.
  bool run() {
    if (testRun_) {
      assert(!"Test has already been run!");
    }

    // Variables for storing runId actually returned by
    // Executor::run() via its callback.
    RunIdentifierTy executorRunId;
    std::unique_ptr<ExecutionContext> executorOutputContext;

    // Call Executor::run().
    std::promise<bool> promise;
    std::future<bool> future = promise.get_future();
    executor_->run(root_.get(), std::move(inputContext_), runId_,
                   [&promise, &executorRunId, &executorOutputContext](
                       RunIdentifierTy runId, Error err,
                       std::unique_ptr<ExecutionContext> context) {
                     executorRunId = runId;
                     executorOutputContext = std::move(context);
                     promise.set_value(ERR_TO_BOOL(std::move(err)));
                   });

    bool runSuccess = !future.get();

    // Check that the values returned in the Executor callback match
    // expectations.
    bool runIdsMatch = executorRunId == runId_;
    bool resultsMatch = runSuccess == expectSuccess_;

    bool bindingsMatch = PlaceholderBindings::compare(
        executorOutputContext->getPlaceholderBindings(),
        outputContext_->getPlaceholderBindings());

    // If the run failed, we shouldn't expect bindingsMatch to be true.
    bool testPassed =
        runIdsMatch && resultsMatch && (!runSuccess || bindingsMatch);

    testRun_ = true;
    executor_->freePool(root_.get());
    return testPassed;
  }

private:
  /// The Executor to run the test with.
  std::shared_ptr<Executor> executor_;
  /// The root node of the DAG being tested.
  std::unique_ptr<DAGNode> root_;
  /// The Module containing the PHs.
  std::unique_ptr<Module> module_;
  /// The Type for all of the Placeholders that will be used during execution.
  std::unique_ptr<Type> type_;
  /// All nodes in the DAG.
  DAGNodeNameMapTy nodes_;
  /// All Placeholders that will be used during execution.
  PlaceholderNameMapTy placeholders_;
  /// The input ExecutionContext that should be passed to Executor::run()
  /// when running the test.
  std::unique_ptr<ExecutionContext> inputContext_;
  /// The expected ExecutionContext that the Executor should return.
  std::unique_ptr<ExecutionContext> outputContext_;
  /// The run ID that should be passed to Executor::run() when running
  /// the test.
  RunIdentifierTy runId_;
  /// The expected result that the Executor should return.
  bool expectSuccess_;
  /// Tracks whether or not the test has already been run.
  bool testRun_;
};

/// This class helps build tests for testing Executor implementations. It
/// presents a simple interface for executor DAG construction; nodes are added
/// by specifying its parents, device ID, and named inputs and outputs. This
/// builder class takes care of all of the work needed to actually run this DAG:
/// creation of Placeholders and Tensors for all inputs and outputs; creation of
/// input/output ExecutionContext for each node to verify that each one
/// receives the correct input and produces the correct output; and registration
/// with the TestDeviceManager.
class ExecutorTestBuilder final {
public:
  /// Constructor. The exact value of type_ doesn't really matter since the
  /// important thing to test is that that Placeholder values are propagated
  /// between ExecutionContexts correctly.
  ExecutorTestBuilder(const std::shared_ptr<Executor> &executor,
                      const DeviceManagerMapTy &deviceManagers)
      : executor_(executor), module_(glow::make_unique<Module>()),
        root_(glow::make_unique<DAGNode>()),
        bindings_(glow::make_unique<PlaceholderBindings>()),
        type_(
            std::unique_ptr<Type>(new Type(ElemKind::FloatTy, {32, 64, 128}))),
        success_(true), deviceManagers_(deviceManagers) {}

  /// Add a node named \p name to the DAG with parents \p parents that runs on a
  /// device specified by \p deviceId. A RuntimeBundle is created for the node
  /// with runtime symbol information created from \p inputs and \p outputs.
  /// \p runId is the run ID for the node and \p success is the desired
  /// execution status. If \p parents is empty, the new node is added as a child
  /// of the root.
  void addNode(const std::string &name, DeviceIDTy deviceId,
               llvm::ArrayRef<llvm::StringRef> parents,
               llvm::ArrayRef<llvm::StringRef> inputs,
               llvm::ArrayRef<llvm::StringRef> outputs, RunIdentifierTy runId,
               bool success) {
    auto newNode = glow::make_unique<DAGNode>();
    auto *newNodeRawPtr = newNode.get();

    // If this is the first node being added, record the run ID for the graph.
    // Otherwise, make sure that the runId matches that of the previous nodes.
    if (nodes_.empty()) {
      runId_ = runId;
    } else {
      assert(runId == runId_ && "Node run ID does not match rest of graph!");
    }

    // If the result for this node is false, set the expected
    // result for the entire test to false.
    success_ &= success;

    // Add parents to the list of parents in the new node and add the newNode
    // to the list of children in the parents. If the parent list is empty,
    // make the root the only parent. Also, update the set of known leaves
    // by removing any parents of the new node from it. This will be useful
    // later.
    if (!parents.empty()) {
      for (const auto &parent : parents) {
        auto it = nodes_.find(parent.str());
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
    // 2) Assemble the input ExecutionContexts that the node is expected to be
    //    called with
    //    and the ExecutionContexts that the node should produce as output.
    // 3) Generate the symbol table for the new node by generating
    //    RuntimeSymbolInfo objects for each input and output.
    SymbolTableTy symbolTable;
    size_t offset = 0;

    auto nodeInputContext = glow::make_unique<ExecutionContext>();
    auto nodeOutputContext = glow::make_unique<ExecutionContext>();

    auto nodeInputBindings = nodeInputContext->getPlaceholderBindings();
    auto nodeOutputBindings = nodeOutputContext->getPlaceholderBindings();

    for (const auto &input : inputs) {
      // Both input and output bindings should contain bindings for the inputs.
      insertSymbolIntoPlaceholderBindings(input, nodeInputBindings);
      insertSymbolIntoPlaceholderBindings(input, nodeOutputBindings);

      RuntimeSymbolInfo runtimeSymbolInfo;
      runtimeSymbolInfo.size = type_->getSizeInBytes();
      runtimeSymbolInfo.offset = offset;
      runtimeSymbolInfo.type = *type_;
      runtimeSymbolInfo.input = true;
      runtimeSymbolInfo.output = false;
      runtimeSymbolInfo.symbolCategory = SymbolCategory::Placeholder;
      symbolTable.insert(std::make_pair(input, runtimeSymbolInfo));
      offset += type_->getSizeInBytes();
    }

    for (const auto &output : outputs) {
      insertSymbolIntoPlaceholderBindings(output, nodeOutputBindings);

      RuntimeSymbolInfo runtimeSymbolInfo;
      runtimeSymbolInfo.size = type_->getSizeInBytes();
      runtimeSymbolInfo.offset = offset;
      runtimeSymbolInfo.type = *type_;
      runtimeSymbolInfo.input = false;
      runtimeSymbolInfo.output = true;
      runtimeSymbolInfo.symbolCategory = SymbolCategory::Placeholder;
      symbolTable.insert(std::make_pair(output, runtimeSymbolInfo));
      offset += type_->getSizeInBytes();
    }

    // Set the name, device ID, and RuntimeBundle of the new node.
    newNode->name = name;
    newNode->deviceRuntimeInfos[deviceId] = DeviceRuntimeInfo();

    newNode->runtimeBundle = glow::make_unique<RuntimeBundle>(
        symbolTable, /*constWeight=*/0, /*mutableWeight=*/0,
        /*activations=*/0);

    // Register node result with the appropriate DeviceManager.
    auto it = deviceManagers_.find(deviceId);

    if (it == deviceManagers_.end()) {
      assert(!"No test device manager found for this device ID");
    }

    auto *deviceManagerPtr = it->second.get();
    auto testDeviceManagerPtr =
        static_cast<TestDeviceManager *>(deviceManagerPtr);

    bool registered = testDeviceManagerPtr->registerResult(
        name, runId, success, std::move(nodeInputContext),
        std::move(nodeOutputContext));

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

    // Generate the input and output ExecutionContexts for the test. This
    // input ExecutionContexts contains the input Placeholders of all root
    // nodes and output Placeholders of all leaves (but backed by zero tensors).
    // This is the ExecutionContexts that needs to be passed to
    // Executor::run() to run the test. The output ExecutionContexts contains
    // the same Placeholders as the input ExecutionContexts, but the leaves'
    // output Placeholders are mapped to their expected output Tensors. This
    // ExecutionContext is used to verify that the one returned by the
    // Executor is correct.
    auto inputContext = glow::make_unique<ExecutionContext>();
    auto outputContext = glow::make_unique<ExecutionContext>();

    for (const auto &symbol : inputSymbols) {
      insertSymbolIntoPlaceholderBindings(
          symbol, inputContext->getPlaceholderBindings());
      insertSymbolIntoPlaceholderBindings(
          symbol, outputContext->getPlaceholderBindings());
    }

    for (const auto &symbol : outputSymbols) {
      auto *placeholder = bindings_->getPlaceholderByNameSlow(symbol);
      if (!placeholder) {
        assert(!"Placeholder for DAG output not found!");
      }
      insertSymbolIntoPlaceholderBindings(
          symbol, inputContext->getPlaceholderBindings());
      insertSymbolIntoPlaceholderBindings(
          symbol, outputContext->getPlaceholderBindings());
    }
    // Create the test object.
    ExecutorTest test(executor_, std::move(root_), std::move(module_),
                      std::move(type_), std::move(nodes_),
                      std::move(placeholders_), std::move(inputContext),
                      std::move(outputContext), runId_, success_);

    // Reset builder state to allow a new test to be constructed with this
    // instance.
    root_ = glow::make_unique<DAGNode>();
    module_ = glow::make_unique<Module>();
    bindings_->clear();
    type_ = std::unique_ptr<Type>(new Type(ElemKind::FloatTy, {1, 2, 2}));
    nodes_.clear();
    leaves_.clear();
    placeholders_.clear();
    success_ = true;

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
      const SymbolTableTy &symbolTable = node->runtimeBundle->getSymbolTable();

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
      const SymbolTableTy &symbolTable = node->runtimeBundle->getSymbolTable();

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

  /// Insert a Placeholder named \p name with type type_ into \p bindings
  /// and generate a random Tensor for it. If this Placeholder has already been
  /// mapped for the test being created, reuse the existing value.
  void insertSymbolIntoPlaceholderBindings(llvm::StringRef name,
                                           PlaceholderBindings *bindings) {
    auto ph = module_->getPlaceholderByNameSlow(name);

    if (!ph) {
      // This is a new symbol. Create a Placeholder and an initialize and new
      // Tensor for it.
      auto placeholder = module_->createPlaceholder(type_.get(), name, false);
      auto *tensor = bindings_->allocate(placeholder);
      tensor->init(Tensor::InitKind::Xavier, 1.0, rng_);
      bindings->insert(placeholder, tensor->clone());
    } else {
      // This is a symbol that already has an associated Placeholder and Tensor.
      // Copy that Tensor.
      const auto *tensor = bindings_->get(ph);
      bindings->insert(ph, tensor->clone());
    }
  }

  /// The Executor being tested.
  std::shared_ptr<Executor> executor_;
  /// Module for holding PHs
  std::unique_ptr<Module> module_;
  /// The root of the DAG being constructed.
  std::unique_ptr<DAGNode> root_;
  /// This PlaceholderBindings holds all created and initialized Placeholders
  /// for the test.
  std::unique_ptr<PlaceholderBindings> bindings_;
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
  /// The expected result for the DAG.
  bool success_;
  /// Map from DeviceIDTy -> TestDeviceManager. This enables the construction of
  /// tests with nodes spread across devices.
  const DeviceManagerMapTy &deviceManagers_;
};

/// This test fixture provides ThreadPoolExecutor, ExecutorTestBuilder,
/// DeviceManagerMapTy instances to all tests.
class ThreadPoolExecutorTest : public ::testing::Test {
protected:
  ThreadPoolExecutorTest()
      : executor_(std::make_shared<ThreadPoolExecutor>(deviceManagerMap_)),
        testBuilder_(executor_, deviceManagerMap_) {}
  ~ThreadPoolExecutorTest() = default;

  /// The Executor being tested.
  std::shared_ptr<ThreadPoolExecutor> executor_;
  /// An ExecutorTestBuilder instance for creating tests.
  ExecutorTestBuilder testBuilder_;
  /// DeviceManager map for initializing executor_.
  DeviceManagerMapTy deviceManagerMap_;
};

/// Tests that an empty DAG is handled correctly.
TEST_F(ThreadPoolExecutorTest, EmptyDAG) {
  constexpr RunIdentifierTy testRunId = 10;

  // Make a PlaceholderBindings with one Placeholder in it to make sure
  // Executor::run() doesn't modify it when the root given to it is null. Make
  // two identical copies; one to give to Executor::run(), and another to
  // compare the returned PlaceholderBindings with.
  PseudoRNG rng;
  auto type = std::unique_ptr<Type>(new Type(ElemKind::FloatTy, {1, 2, 2}));
  auto placeholder = glow::make_unique<Placeholder>(
      "a", type.get(), /*trainable=*/false, ANY_LAYOUT);

  auto testContext = glow::make_unique<ExecutionContext>();
  auto refContext = glow::make_unique<ExecutionContext>();

  auto *tensor =
      testContext->getPlaceholderBindings()->allocate(placeholder.get());
  tensor->init(Tensor::InitKind::Xavier, 1.0, rng);
  refContext->getPlaceholderBindings()->insert(placeholder.get(),
                                               tensor->clone());

  // Variables for storing runId actually returned by
  // Executor::run() via its callback.
  RunIdentifierTy executorRunId;
  std::unique_ptr<ExecutionContext> executorOutputContext;

  // Call Executor::run().
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  std::unique_ptr<Error> runErr;
  executor_->run(nullptr, std::move(testContext), testRunId,
                 [&runErr, &promise, &executorRunId, &executorOutputContext](
                     RunIdentifierTy runId, Error err,
                     std::unique_ptr<ExecutionContext> context) {
                   executorRunId = runId;
                   executorOutputContext = std::move(context);
                   runErr = glow::make_unique<Error>(std::move(err));
                   promise.set_value();
                 });

  EXPECT_FALSE(ERR_TO_BOOL(std::move(*DCHECK_NOTNULL(runErr.get()))));

  EXPECT_EQ(executorRunId, testRunId);

  EXPECT_TRUE(PlaceholderBindings::compare(
      refContext->getPlaceholderBindings(),
      executorOutputContext->getPlaceholderBindings()));
}

/// Tests that a single node can run correctly.
TEST_F(ThreadPoolExecutorTest, SingleNode) {
  constexpr RunIdentifierTy testRunId = 10;
  constexpr DeviceIDTy testDeviceId = 111;
  constexpr unsigned deviceManagerThreads = 1;

  // Make a TestDeviceManager and insert into the DeviceManagerMap map (which
  // the ThreadPoolExecutor has a reference to) and the TestDeviceManager map
  // (which the ExecutorTestBuilder has a reference to).
  auto deviceManager = glow::make_unique<TestDeviceManager>(
      deviceManagerThreads, DeviceConfig("Interpreter"));
  deviceManagerMap_.emplace(testDeviceId, std::move(deviceManager));

  // Build the DAG. The DAG created below looks like this:
  /**
   *         root
   *          |
   *          v
   *         net
   **/

  testBuilder_.addNode("net", testDeviceId,
                       /*parents=*/{}, {"netInput"}, {"netOutput"}, testRunId,
                       true);

  ExecutorTest test = testBuilder_.emitTest();
  EXPECT_TRUE(test.run());
}

/// Tests that several instances of a single node DAG can be run in parallel.
TEST_F(ThreadPoolExecutorTest, ConcurrentSingleNode) {
  constexpr RunIdentifierTy baseTestRunId = 10;
  constexpr DeviceIDTy testDeviceId = 111;
  constexpr unsigned deviceManagerThreads = 3;
  unsigned numConcurrentRuns = 100;

  // Make a TestDeviceManager and insert into the DeviceManagerMap map (which
  // the ThreadPoolExecutor has a reference to) and the TestDeviceManager map
  // (which the ExecutorTestBuilder has a reference to).
  auto deviceManager = glow::make_unique<TestDeviceManager>(
      deviceManagerThreads, DeviceConfig("Interpreter"));
  deviceManagerMap_.emplace(testDeviceId, std::move(deviceManager));

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
                         baseTestRunId + i, true);
    ExecutorTest t = testBuilder_.emitTest();

    std::thread th([&mtx, &driverCV, &threadCV, &threadsReady, &testsPassed,
                    test = std::move(t), numConcurrentRuns]() mutable {
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
    driverCV.wait(lock, [&threadsReady, numConcurrentRuns] {
      return threadsReady == numConcurrentRuns;
    });
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
/// runId don't succeed.
TEST_F(ThreadPoolExecutorTest, ConcurrentSingleNodeDuplicateRunId) {
  constexpr RunIdentifierTy testRunId = 10;
  constexpr DeviceIDTy testDeviceId = 111;
  constexpr unsigned deviceManagerThreads = 1;
  constexpr unsigned numConcurrentRuns = 100;

  // Make a TestDeviceManager and insert into the DeviceManagerMap map (which
  // the ThreadPoolExecutor has a reference to) and the TestDeviceManager map
  // (which the ExecutorTestBuilder has a reference to).
  auto deviceManager = glow::make_unique<TestDeviceManager>(
      deviceManagerThreads, DeviceConfig("Interpreter"));
  deviceManagerMap_.emplace(testDeviceId, std::move(deviceManager));

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
                         true);
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
  auto deviceManager = glow::make_unique<TestDeviceManager>(
      deviceManagerThreads, DeviceConfig("Interpreter"));
  deviceManagerMap_.emplace(testDeviceId, std::move(deviceManager));

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
                       /*outputs=*/{"alphaOut"}, testRunId, true);
  testBuilder_.addNode("beta", testDeviceId,
                       /*parents=*/{}, /*inputs=*/{"betaIn"},
                       /*outputs=*/{"betaOut"}, testRunId, true);
  testBuilder_.addNode("gamma", testDeviceId,
                       /*parents=*/{"alpha", "beta"},
                       /*inputs=*/{"alphaOut", "betaOut"},
                       /*outputs=*/{"deltaIn", "epsIn"}, testRunId, true);
  testBuilder_.addNode("delta", testDeviceId,
                       /*parents=*/{"gamma"}, /*inputs=*/{"deltaIn"},
                       /*outputs=*/{"deltaOut"}, testRunId, true);
  testBuilder_.addNode("eps", testDeviceId,
                       /*parents=*/{"gamma"}, /*inputs=*/{"epsIn"},
                       /*outputs=*/{"epsOut"}, testRunId, true);

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
  auto deviceManager = glow::make_unique<TestDeviceManager>(
      deviceManagerThreads, DeviceConfig("Interpreter"));
  deviceManagerMap_.emplace(testDeviceId, std::move(deviceManager));

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
                       /*outputs=*/{"alphaOut"}, testRunId, true);
  testBuilder_.addNode("beta", testDeviceId,
                       /*parents=*/{"alpha"}, /*inputs=*/{"alphaOut"},
                       /*outputs=*/{"betaOut"}, testRunId, true);
  testBuilder_.addNode("gamma", testDeviceId,
                       /*parents=*/{"beta"},
                       /*inputs=*/{"betaOut"},
                       /*outputs=*/{"gammaOut"}, testRunId, true);
  testBuilder_.addNode("delta", testDeviceId,
                       /*parents=*/{}, /*inputs=*/{"deltaIn"},
                       /*outputs=*/{"deltaOut"}, testRunId, true);
  testBuilder_.addNode("eps", testDeviceId,
                       /*parents=*/{"delta"}, /*inputs=*/{"deltaOut"},
                       /*outputs=*/{"epsOut"}, testRunId, false);
  testBuilder_.addNode("zeta", testDeviceId,
                       /*parents=*/{"eps"}, /*inputs=*/{"epsOut"},
                       /*outputs=*/{"zetaOut"}, testRunId, true);

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
    auto deviceManager = glow::make_unique<TestDeviceManager>(
        deviceManagerThreads, DeviceConfig("Interpreter"));
    deviceManagerMap_.emplace(deviceId, std::move(deviceManager));
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
                       /*outputs=*/{"alphaOut"}, testRunId, true);
  testBuilder_.addNode("beta", testDeviceIdB,
                       /*parents=*/{}, /*inputs=*/{"betaIn"},
                       /*outputs=*/{"betaOut"}, testRunId, true);
  testBuilder_.addNode("gamma", testDeviceIdC,
                       /*parents=*/{"alpha", "beta"},
                       /*inputs=*/{"alphaOut", "betaOut"},
                       /*outputs=*/{"deltaIn", "epsIn"}, testRunId, true);
  testBuilder_.addNode("delta", testDeviceIdA,
                       /*parents=*/{"gamma"}, /*inputs=*/{"deltaIn"},
                       /*outputs=*/{"deltaOut"}, testRunId, true);
  testBuilder_.addNode("eps", testDeviceIdB,
                       /*parents=*/{"gamma"}, /*inputs=*/{"epsIn"},
                       /*outputs=*/{"epsOut"}, testRunId, true);

  ExecutorTest test = testBuilder_.emitTest();
  EXPECT_TRUE(test.run());
}

/// Tests that several instances of a DAG with multiple nodes can run correctly
/// in parallel.
TEST_F(ThreadPoolExecutorTest, ConcurrentMultiNode) {
  constexpr RunIdentifierTy baseTestRunId = 10;
  constexpr DeviceIDTy testDeviceId = 111;
  constexpr unsigned deviceManagerThreads = 3;
  unsigned numConcurrentRuns = 100;

  // Make a TestDeviceManager and insert it into the DeviceManagerMap map
  // (which the ThreadPoolExecutor has a reference to) and the TestDeviceManager
  // map (which the ExecutorTestBuilder has a reference to).
  auto deviceManager = glow::make_unique<TestDeviceManager>(
      deviceManagerThreads, DeviceConfig("Interpreter"));
  deviceManagerMap_.emplace(testDeviceId, std::move(deviceManager));

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
                         /*outputs=*/{"alphaOut"}, baseTestRunId + i, true);
    testBuilder_.addNode(beta, testDeviceId,
                         /*parents=*/{}, /*inputs=*/{"betaIn"},
                         /*outputs=*/{"betaOut"}, baseTestRunId + i, true);
    testBuilder_.addNode(gamma, testDeviceId,
                         /*parents=*/{alpha, beta},
                         /*inputs=*/{"alphaOut", "betaOut"},
                         /*outputs=*/{"deltaIn", "epsIn"}, baseTestRunId + i,
                         true);
    testBuilder_.addNode(delta, testDeviceId,
                         /*parents=*/{gamma}, /*inputs=*/{"deltaIn"},
                         /*outputs=*/{"deltaOut"}, baseTestRunId + i, true);
    testBuilder_.addNode(eps, testDeviceId,
                         /*parents=*/{gamma}, /*inputs=*/{"epsIn"},
                         /*outputs=*/{"epsOut"}, baseTestRunId + i, true);

    ExecutorTest t = testBuilder_.emitTest();
    std::thread th([&mtx, &driverCV, &threadCV, &threadsReady, &testsPassed,
                    test = std::move(t), numConcurrentRuns]() mutable {
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
    driverCV.wait(lock, [&threadsReady, numConcurrentRuns] {
      return threadsReady == numConcurrentRuns;
    });
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
