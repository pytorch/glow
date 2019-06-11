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

class PlaceholderBindings;

namespace runtime {

class DeviceManager;

/// This is an interface to an executor that can run and results the results of
/// a partitioned graph.
class Executor {
public:
  /// Destructor.
  virtual ~Executor() = default;

  /// Run the DAG specified by \p root using \p bindings and call \cb with the
  /// results. \p runId is used to identify the run for logging and metrics
  /// purposes.
  /// cb will be called with a result code, the run ID and a PlaceholderBindings
  /// containing placeholder-tensor mappings for the nodes in the DAG that have
  /// no postrequisites (i.e. the final results) in addition to the mappings
  /// present in \p bindings.
  virtual void run(const DAGNode *root,
                   std::unique_ptr<ExecutionContext> context,
                   RunIdentifierTy runId, ResultCBTy cb) = 0;

  /// Shutdown the Executor. Should block until all active requests are complete
  /// and prevent new requests from being initiated.
  virtual void shutdown() = 0;
};

} // namespace runtime
} // namespace glow
#endif
