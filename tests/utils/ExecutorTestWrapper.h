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
#ifndef GLOW_TESTS_UTILS_EXECUTORTESTWRAPPER_H
#define GLOW_TESTS_UTILS_EXECUTORTESTWRAPPER_H

#include "glow/Runtime/RuntimeTypes.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace glow {
namespace runtime {

class Executor;

/// This class serves as an interface to a test created by ExecutorTestBuilder.
/// It also contains the resources necessary to run the test.
class ExecutorTestWrapper {
protected:
  using PlaceholderNameMapTy =
      std::unordered_map<std::string, std::unique_ptr<Placeholder>>;
  using DAGNodeNameMapTy =
      std::unordered_map<std::string, std::unique_ptr<DAGNode>>;

  /// Constructor.
  ExecutorTestWrapper(const std::shared_ptr<Executor> &executor,
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

} // namespace runtime
} // namespace glow

#endif // GLOW_TESTS_UTILS_EXECUTORTESTWRAPPER_H
