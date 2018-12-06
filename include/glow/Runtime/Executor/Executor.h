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
#ifndef GLOW_RUNTIME_EXECUTOR_H
#define GLOW_RUNTIME_EXECUTOR_H

#include "glow/Graph/Context.h"
#include "glow/Graph/Graph.h"
#include "glow/Runtime/RuntimeTypes.h"

namespace glow {
namespace runtime {

/// This enum lists the available executors.
enum class ExecutorKind {
  ThreadPool, // Executor backed by a thread pool.
};

/// This is an interface to an executor that can run and results the results of
/// a partitioned graph.
class Executor {
public:
  /// Virtual destructor.
  virtual ~Executor();
  using DoneCb = std::function<void(RunIdentifierTy, ResultCode,
                                    std::unique_ptr<Context>)>;

  /// Run the DAG specified by \p roots using \p context and call \cb with the
  /// results. \p runId is used to identify the run for logging and metrics
  /// purposes.
  /// cb will be called with a result code, the run ID and a Context containing
  /// placeholder-tensor mappings for the nodes in the DAG that have
  /// no postrequisites (i.e. the final results).
  virtual void run(std::vector<DAGNode *> roots,
                   std::unique_ptr<Context> context, RunIdentifierTy runId,
                   DoneCb cb) = 0;
};

/// Create a executor of kind \p kind.
Executor *createExecutor(ExecutorKind executorKind);

} // namespace runtime
} // namespace glow
#endif
