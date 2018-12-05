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

#include "glow/Runtime/RuntimeTypes.h"

#include <functional>
#include <map>
#include <memory>

namespace glow {

class Context;
class DeviceManager;

namespace runtime {

/// This enum lists the available executors.
enum class ExecutorKind {
  ThreadPool, // Executor backed by a thread pool.
};

/// This is an interface to an executor that can run and results the results of
/// a partitioned graph.
class Executor {
public:
  /// Map of DeviceIDTy -> DeviceManager, used when selecting a DeviceManager
  /// to use to execute a DAGNode.
  using DeviceManagerMapTy =
      std::map<DeviceIDTy, std::shared_ptr<DeviceManager>>;

  /// Callback signalling the success/failure of running a DAG. The arguments
  /// are the run identifier of the invocation, the execution result code and
  /// a Context containing outputs.
  using DoneCbTy = std::function<void(RunIdentifierTy, ResultCode,
                                      std::unique_ptr<Context>)>;
  /// Destructor.
  virtual ~Executor() = default;

  /// Run the DAG specified by \p root using \p context and call \cb with the
  /// results. \p runId is used to identify the run for logging and metrics
  /// purposes.
  /// cb will be called with a result code, the run ID and a Context containing
  /// placeholder-tensor mappings for the nodes in the DAG that have
  /// no postrequisites (i.e. the final results) in addition to the mappings
  /// present in \p context.
  virtual void run(const DAGNode *root, std::unique_ptr<Context> context,
                   RunIdentifierTy runId, DoneCbTy cb) = 0;
};

/// Create an executor of kind \p kind that will call into the DeviceManager
/// instances provided in \deviceManagers. \returns a pointer to the
/// executor.
Executor *createExecutor(const Executor::DeviceManagerMapTy &deviceManagers,
                         ExecutorKind executorKind = ExecutorKind::ThreadPool);

} // namespace runtime
} // namespace glow
#endif
