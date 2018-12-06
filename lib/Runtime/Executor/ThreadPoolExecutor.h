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

#include <list>
#include <mutex>
#include <queue>
#include <set>
#include <thread>
#include <tuple>

#include "glow/Runtime/Executor/Executor.h"

namespace glow {
namespace runtime {

/// This class represents a single work item of the TheadPoolExecutor. It
/// contains a DAG, an associated context, and some additional information to
/// track progress on executing the DAG.
class ThreadPoolExecutorWorkItem {
public:
  using DoneCb = Executor::DoneCb;

  /// This enum represents the overall status of the work item. The transitions
  /// are as follows:
  ///
  /// NONE ---> QUEUED ---> IN_PROGRESS ---> DONE
  ///                           |              ^
  ///                           |              |
  ///                           ---> FAILED ---
  enum class Status {
    // This work item has been created and not queued.
    NONE,
    // This work item has been inserted into the queue and can be worked on.
    QUEUED,
    // This work item is currently being worked on. Some pieces might be done.
    IN_PROGRESS,
    // This work item has failed, but the callback has not been called yet.
    FAILED,
    // This work item is done. It has either failed or succeeded, and its
    // callback has been called.
    DONE,
  };

  /// Constructor. \p dag is the DAG that this work item should run, and \p ctx
  /// is the context that the components should run with. \p cb is the callback
  /// to be called when the work item is done.
  explicit ThreadPoolExecutorWorkItem(ExecutorFunctionDAG *dag,
                                      ExecutorFunctionDAGContext *ctx,
                                      DoneCb cb);

  /// Destructor.
  ~ThreadPoolExecutorWorkItem() = default;

  /// \returns whether or not there is more work to be done on this work item.
  bool isMoreWork();

  /// \returns the next pair {function, context} that is ready for execution
  /// (i.e. all prerequisites have been fulfilled and it is not already
  /// being executed).
  std::tuple<Function *, Context *> getNext();

  /// Mark the work item as queued.
  void markQueued();

  /// Mark that the pair {\p function, \p context} have succeeded.
  void markSuccess(Function *function, Context *context);

  /// Mark the work item as failed.
  void markFailure();

private:
  /// Process the queue of updates buffered by markSuccess and update internal
  /// records accordingly.
  void processUpdates();

  /// The callback to call when the work item is done.
  DoneCb cb_;
  /// The DAG being executed.
  ExecutorFunctionDAG *dag_;
  /// The context with which the DAG should be executed.
  ExecutorFunctionDAGContext *ctx_;
  /// An iterator into the list of functions in the DAG.
  std::list<Function *>::const_iterator it_;
  /// The current status of the work item.
  Status status_;
  /// The Context object that holds the final results of execution.
  Context *result_;
  /// All functions that have finished executing.
  std::set<Function *> completedFunctions_;
  /// All functions that are currently executing.
  std::set<Function *> inflightFunctions_;
  /// All functions that have finished executing but have not been moved
  /// to completedFunctions_ yet.
  std::set<Function *> updateFunctions_;
  /// All contexts that resulted from finished executions whose contents have
  /// not yet been copied into the contexts of dependent graph components or
  /// into the result_ Context if applicable.
  std::set<Context *> updateContexts_;
  /// A mutex to guard accesses to class members.
  /// TODO: This can probably be eliminated by making certain members
  /// std::atomic.
  std::mutex mtx_;
};

/// This class implements the Executor interface by doing all of the work on a
/// thread pool. Each call to ThreadPoolExecutor::run() creates a stateful
/// work item that is ushered to completion though a series of state transitions
/// (see ThreadPoolExecutorWorkItem::Status) by the threads in the pool.
class ThreadPoolExecutor final : public Executor {
public:
  using DoneCb = Executor::DoneCb;
  /// Constructor.
  explicit ThreadPoolExecutor(unsigned numWorkers = kNumWorkers);

  /// Virtual destructor.
  virtual ~ThreadPoolExecutor();

  /// See Executor::run.
  void run(ExecutorFunctionDAG *functionDag, ExecutorFunctionDAGContext *ctx,
           DoneCb cb) override;

private:
  /// Main loop run by workers in the thread pool.
  void workerMain();
  /// Helper function for processing a work item.
  void processWorkItem(ThreadPoolExecutorWorkItem *workItem);
  /// The default number of workers in the thread pool.
  constexpr static unsigned kNumWorkers = 3;
  /// Thread pool workers.
  std::list<std::thread> workers_;
  /// Flag checked by the workers in between work items to determine
  /// whether they should stop and exit.
  std::atomic<bool> shouldStop_;
  /// Queue of work items.
  std::queue<ThreadPoolExecutorWorkItem *> workQueue_;
  /// Condition variable to signal to threads when work is added to
  /// the work queue.
  std::condition_variable queueNotEmpty_;
  /// Mutex to coordinate access to the work queue.
  std::mutex workQueueMtx_;
};

} // namespace runtime
} // namespace glow
#endif
