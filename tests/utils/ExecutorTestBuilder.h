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
#ifndef GLOW_TESTS_UTILS_EXECUTORTESTBUILDER_H
#define GLOW_TESTS_UTILS_EXECUTORTESTBUILDER_H

#include "glow/Backends/DeviceManager.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace glow {
namespace runtime {

class Executor;
class ExecutorBenchmarkWrapper;
class ExecutorUnitTestWrapper;

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
  template <class TestType> std::unique_ptr<TestType> emitTest();

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

#endif // GLOW_TESTS_UTILS_EXECUTORTESTBUILDER_H
