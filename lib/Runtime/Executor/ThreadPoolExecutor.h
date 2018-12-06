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

#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "glow/Runtime/Executor/Executor.h"
#include "glow/Support/ThreadPool.h"

namespace glow {
namespace runtime {

/// This struct keeps track of the state of execution for a run (identified)
/// by the runId).
struct ExecutionState final {
public:
  using DoneCb = Executor::DoneCb;

  /// Constructor.
  explicit ExecutionState(RunIdentifierTy id, DoneCb doneCb)
      : runId(id), cb(doneCb), resultCtx(std::make_unique<Context>()) {}

  /// The runId corresponding to the run that this object is tracking.
  RunIdentifierTy runId;
  /// The callback that should be called when execution is done.
  DoneCb cb;
  /// The Context object containing the results of the execution (i.e. the
  /// outputs of the DAGNodes that have no children).
  std::unique_ptr<Context> resultCtx;
  /// Counters for how many of each nodes parents are done. These are needed
  /// in order to determine when a node is ready to be executed.
  std::unordered_map<DAGNode *, unsigned> nodeParentsDone;
  /// Input Contexts for all of the nodes. These are gradually populated as
  /// a node's parents finish.
  std::unordered_map<DAGNode *, std::unique_ptr<Context>> inputCtxs;
  /// The set of currently executing nodes. This is populated with the roots
  /// when a run starts, and does not become empty until execution finishes.
  std::unordered_set<DAGNode *> inflightNodes;
};

/// This implementation of the Executor interface uses a thread pool to
/// handle and process multiple concurrent execution runs.
class ThreadPoolExecutor final : public Executor {
public:
  using DoneCb = Executor::DoneCb;
  /// Constructor.
  explicit ThreadPoolExecutor(unsigned numWorkers = kNumWorkers)
      : threadPool_(numWorkers) {}

  /// Virtual destructor.
  virtual ~ThreadPoolExecutor() = default;

  /// See Executor::run. A particular invocation is specified completely by
  /// the triple (roots, context, runId).
  void run(std::vector<DAGNode *> roots, std::unique_ptr<Context> context,
           RunIdentifierTy runId, DoneCb cb) override;

private:
  /// Propagate Placeholders from \p ctx into the final output Context for the
  /// run corresponding to \p runId.
  void propagateOutputPlaceholders(RunIdentifierTy runId,
                                   std::unique_ptr<Context> ctx);
  /// Propagate Placeholders needed by \p node from \p ctx into
  /// the Context for \p node within the run specified by \p runId.
  void propagatePlaceholdersForNode(RunIdentifierTy runId, DAGNode *node,
                                    Context *ctx);
  /// Execute the DAG node specified by \p node within the run specified by
  /// \p runId.
  void executeDAGNode(RunIdentifierTy runId, DAGNode *node);
  /// Handle the result returned asynchronously by the DeviceManager.
  /// \p runId is the runId of the that finished executing,
  /// \p is the resultCode returned by the DeviceManager, \p ctx is the
  /// Context that contains the outputs produced by \p node during the run.
  ///
  /// The main purpose of this function is to help move computation off of the
  /// DeviceManager thread pool on onto the one owned by this class.
  void handleDeviceManagerResult(RunIdentifierTy runId, ResultCode resultCode,
                                 std::unique_ptr<Context> ctx, DAGNode *node);
  /// The default number of workers in the thread pool.
  constexpr static unsigned kNumWorkers = 3;
  /// The thread pool used to drive execution.
  ThreadPool threadPool_;
  /// Map from run ID to the ExecutionState containing the state for the run.
  std::unordered_map<RunIdentifierTy, std::shared_ptr<ExecutionState>>
      executionStates_;
  /// Locks for the execution state objects. These are used to ensure that
  /// the shared state contained in the ExecutionState object for a run is
  /// mutated by only one thread at a time.
  std::unordered_map<RunIdentifierTy, std::mutex> executionStateLocks_;
  /// Lock for executionStateLocks_. This is needed to synchronize access to
  /// executionStateLocks_ in order to add locks for new runs and delete locks
  /// for old ones.
  std::mutex executionStateLocksMtx_;
};

} // namespace runtime
} // namespace glow
#endif
