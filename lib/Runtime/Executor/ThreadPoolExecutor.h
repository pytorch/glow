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
#ifndef GLOW_RUNTIME_THREAD_POOL_EXECUTOR_H
#define GLOW_RUNTIME_THREAD_POOL_EXECUTOR_H

#include <condition_variable>
#include <mutex>
#include <unordered_map>

#include "glow/Runtime/Executor/Executor.h"
#include "glow/Support/ThreadPool.h"

namespace glow {
namespace runtime {

/// This class implements a simple barrier with which to wait for all threads
/// to exit a certain section of code before proceeding.
class InflightBarrier final {
public:
  /// Decrement the count of threads in the barrier by \p decr.
  void decrement(unsigned decr = 1);

  /// Increment the count of threads in the barrier by \p incr.
  void increment(unsigned incr = 1);

  /// \returns the current count of the barrier.
  unsigned count();

  /// Wait for the barrier count to hit zero before continuing. This is
  /// potentially a blocking call.
  void wait();

private:
  /// Count of threads inside the barrier.
  unsigned count_{0};
  /// Mutex for accessing count_;
  std::mutex mtx_;
  /// Condition variable for implementing wait().
  std::condition_variable cv_;
};

/// This class keeps track of the state of execution for a run (identified
/// by the runId).
class ExecutionState final {
public:
  /// Constructor.
  explicit ExecutionState(RunIdentifierTy id, const DAGNode *root,
                          std::unique_ptr<Context> resultContext,
                          ResultCBTy doneCb);

  /// Insert the placeholder-tensor pair (\p P, \p T) into the input context
  /// for \p node. This should not be called at the same time as
  /// getUniqueNodeContextPtr().
  void insertIntoNodeCtx(const DAGNode *node, Placeholder *P, Tensor &&T);

  /// \returns a unique pointer to an input context for \p node. This should not
  /// be called at the same time as insertIntoNodeCtx().
  std::unique_ptr<Context> getUniqueNodeContextPtr(const DAGNode *node);

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

  /// Insert the placeholder-tensor pair (\p P, \p T) into the result context.
  /// This should not be called at the same time as getUniqueResultContextPtr().
  void insertIntoResultCtx(Placeholder *P, Tensor &&T);

  /// \returns a unique pointer to the result context. This should not be
  /// called at the same time as getRawResultContextPtr() or
  /// insertIntoResultCtx().
  std::unique_ptr<Context> getUniqueResultContextPtr();

  /// \returns a raw pointer to the result context. This should be not called
  /// at the same time as getUniqueResultContextPtr().
  Context *getRawResultContextPtr() const;

  /// \returns the callback for this execution.
  ResultCBTy getCallback() { return cb_; }

  /// \returns the result code for the execution.
  ResultCode getResultCode() const { return resultCode_; }

  /// Set result code for the execution.
  void setResultCode(const ResultCode resultCode);

  /// \returns the run ID for the execution.
  RunIdentifierTy getRunId() const { return runId_; }

private:
  /// The run identifier for this execution of a DAG.
  RunIdentifierTy runId_;
  /// The callback that should be called when execution is done.
  ResultCBTy cb_;
  /// The Context object containing the results of the execution (i.e. the
  /// outputs of the DAGNodes that have no children).
  std::unique_ptr<Context> resultCtx_;
  /// Counters for how many of each nodes parents are done. These are needed
  /// in order to determine when a node is ready to be executed.
  std::unordered_map<const DAGNode *, std::atomic<unsigned>> nodeParentsDone_;
  /// Input Contexts for all of the nodes. These are gradually populated as
  /// a node's parents finish.
  std::unordered_map<const DAGNode *, std::unique_ptr<Context>> inputCtxs_;
  /// The set of currently executing nodes. This is populated with the roots
  /// when a run starts, and does not become empty until execution finishes.
  std::atomic<unsigned> inflightNodes_;
  /// Flag that is used to track if a non-success error code was received.
  std::atomic<ResultCode> resultCode_;
  /// Mutex used by context insertion functions to make sure only one thread
  /// writes to a Context at a time.
  std::mutex contextMtx_;
};

/// This implementation of the Executor interface uses a thread pool to
/// handle and process multiple concurrent execution runs.
class ThreadPoolExecutor final : public Executor {
public:
  /// Constructor.
  explicit ThreadPoolExecutor(const DeviceManagerMapTy &deviceManagers,
                              unsigned numWorkers = kNumWorkers)
      : threadPool_(numWorkers), deviceManagers_(deviceManagers) {}

  /// See Executor::run. A particular invocation is specified completely by
  /// the triple (roots, context, runId).
  void run(const DAGNode *root, std::unique_ptr<Context> context,
           RunIdentifierTy runId, ResultCBTy cb) override;

  ~ThreadPoolExecutor() override { shutdown(); }

  void shutdown() override;

private:
  /// Propagate Placeholders from \p ctx into the final output Context for the
  /// run corresponding to \p executionState.
  void
  propagateOutputPlaceholders(std::shared_ptr<ExecutionState> executionState,
                              std::unique_ptr<Context> ctx);

  /// Propagate Placeholders needed by \p node from \p ctx into
  /// the Context for \p node within the run corresponding to \p executionState.
  void
  propagatePlaceholdersForNode(std::shared_ptr<ExecutionState> executionState,
                               const DAGNode *node, const Context *ctx);

  /// Execute the DAG node specified by \p node within the run corresponding to
  /// \p executionState.
  void executeDAGNode(std::shared_ptr<ExecutionState> executionState,
                      const DAGNode *node);

  /// Handle the result returned asynchronously by the DeviceManager.
  /// \p executionState is tracks the state of the run that the node that
  /// finished executing belongs to, \p is the resultCode returned by the
  /// DeviceManager, \p ctx is the Context that contains the outputs produced by
  /// \p node during the run.
  ///
  /// The main purpose of this function is to help move computation off of the
  /// DeviceManager thread pool on onto the one owned by this class.
  void handleDeviceManagerResult(std::shared_ptr<ExecutionState> executionState,
                                 ResultCode resultCode,
                                 std::unique_ptr<Context> ctx,
                                 const DAGNode *node);

  /// The default number of workers in the thread pool.
  constexpr static unsigned kNumWorkers = 3;
  /// The thread pool used to drive execution.
  ThreadPool threadPool_;
  /// Map of available DeviceManagers.
  const DeviceManagerMapTy &deviceManagers_;
  /// Map from run ID to the ExecutionState containing the state for the run.
  std::unordered_map<RunIdentifierTy, std::shared_ptr<ExecutionState>>
      executionStates_;
  /// Lock for executionStates_. This is needed to synchronize access to
  /// executionStateLocks_ so that multiple threads and can perform insertion
  /// and lookup concurrently.
  std::mutex executionStatesMutex_;
  /// Barrier for making sure all asynchronous requests made to the
  /// DeviceManager return before allowing destruction of the executor.
  InflightBarrier inflightBarrier_;
  /// Whether the executor is currently shutting down or not.
  std::atomic<bool> shuttingDown_{false};
};

} // namespace runtime
} // namespace glow
#endif
