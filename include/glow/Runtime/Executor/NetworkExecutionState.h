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
#ifndef GLOW_RUNTIME_EXECUTOR_NETWORKEXECUTIONSTATE_H
#define GLOW_RUNTIME_EXECUTOR_NETWORKEXECUTIONSTATE_H

#include "glow/Runtime/RuntimeTypes.h"
#include "glow/Support/TensorPool.h"
#include "glow/Support/ThreadPool.h"

#include <mutex>

namespace glow {
namespace runtime {

/// This class keeps track of the state of execution for a run (identified
/// by the runId).
class NetworkExecutionState final {
public:
  /// Constructor.
  explicit NetworkExecutionState(const DAGNode *root);

  const DAGNode *getRoot() { return root_; }

  /// Destructor.
  ~NetworkExecutionState();

  /// Does the BFS traversal and initializes the NetworkExecutionState. Takes in
  /// a map of all deviceManagers \p devices , and \p staticAssignment , a map
  /// between each node an a deviceManager. If this is an empty map no
  /// assignment is made.
  void init(const DeviceManagerMapTy &devices,
            std::unordered_map<DAGNode *, DeviceIDTy> &staticAssignment);

  /// Binds the state to a new run. This moves the result ctx and cb to be owned
  /// by the networkExecutionState for the duration of the run.
  void bind(std::unique_ptr<ExecutionContext> resultCtx, ResultCBTy cb,
            RunIdentifierTy runId);

  /// \returns a unique pointer to an input bindings for \p node. This should
  /// not be called at the same time as insertIntoNodeCtx().
  std::unique_ptr<ExecutionContext>
  getUniqueNodeContextPtr(const DAGNode *node);

  /// Returns the intermediateContext back to the networkExecutionState after
  /// completion so it can be re-used.
  void returnUniqueNodeContextPtr(const DAGNode *node,
                                  std::unique_ptr<ExecutionContext> ctx);

  /// Increment the count of inflight nodes by \p increment (default is 1).
  void incrementInflightNodes(unsigned increment = 1);

  /// Decrement the count of inflight nodes by the \p decrement (default is 1).
  /// \returns true if there are no nodes inflight after the decrement
  /// operation.
  bool decrementInflightNodes(unsigned decrement = 1);

  /// Increment the count of completed parent nodes for \p node. \returns
  /// true if all parents are done after the increment operation, false
  /// otherwise.
  bool incrementNodeParentsDone(const DAGNode *node, unsigned increment = 1);

  /// Move all events from the provided vector into the top level resultContxt.
  void insertIntoTraceContext(TraceContext *runCtx);

  /// \returns a unique pointer to the result bindings. This should not be
  /// called at the same time as getRawResultPlaceholderBindingsPtr() or
  /// insertIntoResultCtx().
  std::unique_ptr<ExecutionContext> getUniqueResultContextPtr();

  /// \returns a raw pointer to the result bindings. This should be not called
  /// at the same time as getUniqueResultPlaceholderBindingsPtr().
  ExecutionContext *getRawResultContextPtr() const;

  /// \returns the callback for this execution.
  ResultCBTy getCallback() { return cb_; }

  /// \returns the OneErrOnly Error container for the execution.
  OneErrOnly &getErrorContainer() { return errContainer_; }

  /// \returns the run ID for the execution.
  RunIdentifierTy getRunId() const { return runId_; }

  /// Whether or not this node has been initialized.
  bool initialized_{false};

private:
  /// The run identifier for this execution of a DAG.
  RunIdentifierTy runId_;

  /// The callback that should be called when execution is done.
  ResultCBTy cb_;

  /// The ExecutionContext object containing the results of the execution
  /// (i.e. the outputs of the DAGNodes that have no children).
  std::unique_ptr<ExecutionContext> resultCtx_;

  /// Counters for how many of each nodes parents are done. These are needed
  /// in order to determine when a node is ready to be executed.
  std::unordered_map<const DAGNode *, std::atomic<unsigned>> nodeParentsDone_;

  /// Count of current inflight nodes.
  std::atomic<unsigned> inflightNodes_;

  /// Value that is used to track if an Error was received.
  OneErrOnly errContainer_;

  /// Module for the network. This contains the PHs used by the functions in
  /// this network.
  Module *module_{nullptr};

  /// Root node of the DAG for this run.
  const DAGNode *root_;

  /// Map of all buffers allocated for intermediate contexts.
  std::unordered_map<Placeholder *, void *> buffers_;

  /// Map from buffer to device that allocated it, used at destruction to
  /// free buffers.
  std::unordered_map<void *, DeviceManager *> deviceAllocations_;

  /// Map of intermediate placeholder bindings that need to be pointed at
  /// resultCtx tensors.
  std::unordered_map<Placeholder *, std::vector<PlaceholderBindings *>>
      externalIntermediates_;
  /// Input contexts for all of the nodes. These are gradually
  /// populated as a node's parents finish.
  std::unordered_map<const DAGNode *, std::unique_ptr<ExecutionContext>>
      intermediateContexts_;
};

class NetworkExecutionStatePool {
public:
  NetworkExecutionState *getNextNetworkExecutionState() {
    std::lock_guard<std::mutex> lock(stateLock_);
    auto nextState = availableStates_.front();
    availableStates_.pop_front();
    return nextState;
  }

  void addNewState(std::unique_ptr<NetworkExecutionState> state);

  void returnNetworkExecutionState(NetworkExecutionState *state) {
    std::lock_guard<std::mutex> lock(stateLock_);
    availableStates_.push_back(state);
  }

private:
  std::vector<std::unique_ptr<NetworkExecutionState>> states_;
  std::deque<NetworkExecutionState *> availableStates_;
  std::mutex stateLock_;
};

} // namespace runtime
} // namespace glow

#endif // GLOW_RUNTIME_EXECUTOR_NetworkEXECUTIONSTATE_H
