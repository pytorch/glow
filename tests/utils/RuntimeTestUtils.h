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
#ifndef GLOW_UTILS_RUNTIMETESTUTILS_H
#define GLOW_UTILS_RUNTIMETESTUTILS_H

#include "glow/Backends/DeviceManager.h"
#include "glow/Runtime/Executor/Executor.h"
#include "glow/Support/ThreadPool.h"

#include <future>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace glow {
namespace runtime {

class Executor;

/// This is an implementation of DeviceManager tailored for testing Executor
/// implementations. registerResult() gives the caller the ability to
/// dictate precisely what a subsequent call to runFunction() should return.
/// registerResult() should be called before calling Executor::run() in each
/// test. The rest of the implementation of the DeviceManager interface exists
/// to satisfy the compiler.
class TestDeviceManager final : public DeviceManager {
public:
  /// Constructor.
  TestDeviceManager(unsigned numWorkers)
      : DeviceManager(BackendKind::Interpreter), threadPool_(numWorkers) {}

  /// The functions below are the interface for DeviceManager. See
  /// glow::DeviceManager for descriptions of what they do. Since this
  /// class exists only to help test Executor implementations, the only
  /// important function is runFunction().
  void addNetwork(const Module *module, FunctionMapTy functions,
                  ReadyCBTy readyCB) override {}

  /// Do not call this at the same time as registerResult().
  runtime::RunIdentifierTy
  runFunction(std::string functionName,
              std::unique_ptr<ExecutionContext> context,
              ResultCBTy resultCB) override;

  void evictNetwork(std::string functionName,
                    EvictFunctionCBTy evictCB) override;

  uint64_t getMaximumMemory() const override {
    return std::numeric_limits<uint64_t>::max();
  }

  uint64_t getAvailableMemory() const override {
    return std::numeric_limits<uint64_t>::max();
  }

  bool isMemoryAvailable(uint64_t /*estimate*/) const override { return true; }

  /// Look up the previously registered response for \p functionName and
  /// call \p resultCB with it after checking that \p context contains the
  /// expected Placeholder-Tensor mappings.
  void doRunFunction(std::string functionName,
                     std::shared_ptr<ExecutionContext> context,
                     ResultCBTy resultCB);

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
                      std::unique_ptr<ExecutionContext> resultContext);

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

/// This class serves as an interface to a test created by ExecutorTestBuilder.
/// It also contains the resources necessary to run the test. Instances are
/// meant to be created only by ExecutorTestBuilder.
class ExecutorTest final {
public:
  using PlaceholderNameMapTy =
      std::unordered_map<std::string, std::unique_ptr<Placeholder>>;
  using DAGNodeNameMapTy =
      std::unordered_map<std::string, std::unique_ptr<DAGNode>>;

  /// Constructor.
  ExecutorTest(const std::shared_ptr<Executor> &executor,
               std::unique_ptr<DAGNode> root, std::unique_ptr<Type> type,
               DAGNodeNameMapTy nodes, PlaceholderNameMapTy placeholders,
               std::unique_ptr<ExecutionContext> inputContext,
               std::unique_ptr<ExecutionContext> outputContext,
               RunIdentifierTy runId, bool expectSuccess)
      : executor_(executor), root_(std::move(root)), type_(std::move(type)),
        nodes_(std::move(nodes)), placeholders_(std::move(placeholders)),
        inputContext_(std::move(inputContext)),
        outputContext_(std::move(outputContext)), runId_(runId),
        expectSuccess_(expectSuccess), testRun_(false) {}

  /// Run the test.
  bool run();

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
  using PlaceholderNameMapTy =
      std::unordered_map<std::string, std::unique_ptr<Placeholder>>;
  using DAGNodeNameMapTy =
      std::unordered_map<std::string, std::unique_ptr<DAGNode>>;

  /// Constructor. The exact value of type_ doesn't really matter since the
  /// important thing to test is that that Placeholder values are propagated
  /// between ExecutionContexts correctly.
  ExecutorTestBuilder(const std::shared_ptr<Executor> &executor,
                      const DeviceManagerMapTy &deviceManagers)
      : executor_(executor), root_(llvm::make_unique<DAGNode>()),
        bindings_(llvm::make_unique<PlaceholderBindings>()),
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
               bool success);

  /// Emit the test built so far and clear any state in the builder.
  ExecutorTest emitTest();

private:
  /// Collect all input symbol names for the test. \returns a vector containing
  /// the names of all test input symbols.
  std::vector<std::string> gatherInputSymbols() const;

  /// Collect all output symbol names for the test. \returns a vector containing
  /// the names of all test output symbols.
  std::vector<std::string> gatherOutputSymbols() const;

  /// Insert a Placeholder named \p name with type type_ into \p bindings
  /// and generate a random Tensor for it. If this Placeholder has already been
  /// mapped for the test being created, reuse the existing value.
  void insertSymbolIntoPlaceholderBindings(llvm::StringRef name,
                                           PlaceholderBindings *bindings);

  /// The Executor being tested.
  std::shared_ptr<Executor> executor_;
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

} // namespace runtime
} // namespace glow

#endif // GLOW_UTILS_RUNTIMETESTUTILS_H
