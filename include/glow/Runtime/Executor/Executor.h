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

#include <map>
#include <set>
#include <string>

namespace glow {
namespace runtime {

/// Copied @nickg's ResultCode. I think we'll want a common one anyway.
enum class ResultCode {
  EXECUTED,
  FAILED,
  CANCELLED,
};

/// This enum lists the available executors.
enum class ExecutorKind {
  ThreadPool, // Executor backed by a thread pool.
};

/// This class contains the graph to be executed partitioned into subgraphs
/// that can be run on individual devices as well as extra information to help
/// manage execution.
struct ExecutorFunctionDAG {
  // The list of functions to run for this DAG, topologically sorted.
  std::list<Function *> functions;
  // All functions that output final results.
  std::set<Function *> endpoints;
  // Maps from a function to its prerequisites and postrequisites.
  std::map<Function *, std::list<Function *>> incoming;
  std::map<Function *, std::list<Function *>> outgoing;
  // Output placeholder names for each function.
  std::map<Function *, std::list<std::string>> outputs;
};

/// The class encapsulates the context required to run the given DAG.
struct ExecutorFunctionDAGContext {
  // Partioned contexts for each function.
  std::map<Function *, Context *> contexts;
};

/// This is an interface to an executor that can run and results the results of
/// a partitioned graph.
class Executor {
public:
  /// Virtual destructor.
  virtual ~Executor();
  using DoneCb = std::function<void(ResultCode, Context *)>;

  /// Run the DAG specified by \p functionDag using Placeholder values contained
  /// in \p ctx and call \cb with the results. cb will be called with a result
  /// code and a Context containing placeholder-tensor mappings for the
  /// Functions in \p functionDag that have no postrequisites (i.e. the final
  /// results).
  virtual void run(ExecutorFunctionDAG *functionDag,
                   ExecutorFunctionDAGContext *ctx, DoneCb cb) = 0;
};

/// Create a executor of kind \p kind.
Executor *createExecutor(ExecutorKind executorKind);

} // namespace runtime
} // namespace glow
#endif
