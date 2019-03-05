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
#ifndef GLOW_BACKENDS_COMPILEDFUNCTION_H
#define GLOW_BACKENDS_COMPILEDFUNCTION_H

#include "glow/Backends/BackendUtils.h"
#include "glow/Backends/TraceEvents.h"
#include "glow/Graph/Nodes.h"

#include <unordered_map>

namespace glow {

class Context;
enum class BackendKind;
/// Interface for executing a compiled function.
class CompiledFunction {
public:
  /// Default Ctor.
  CompiledFunction() = default;

  /// Ctor that accepts runtimeBundle.
  CompiledFunction(const runtime::RuntimeBundle &bundle);

  /// Dtor.
  virtual ~CompiledFunction();
  /// Execute the network and allocate Placeholder memory with given
  /// \p ctx providing mapping between Placeholder and populated tensor.
  virtual void execute(Context *ctx) = 0;

  /// Does any needed initialization work for the Backend.
  /// This includes device init constant memory allocation and copying to
  /// device. \deprecated
  virtual void setupRuns() { runsSetup_ = true; }

  /// Per run setup. Copy inputs to device. \deprecated
  virtual void beforeRun(const Context &ctx) {}

  /// Per run cleanup. Copy outputs from device. \deprecated
  virtual void afterRun(const Context &ctx) {}

  /// Final cleanup. Release memory, reset device. \deprecated
  virtual void tearDownRuns() { runsSetup_ = false; }

  /// Getter for the runtimeBundle.
  runtime::RuntimeBundle &getRuntimeBundle() { return runtimeBundle_; }

  /// Collects constants for runtime.
  virtual void collectConstants(Module *){};

  /// Setter for TraceEvent lookup. Note: does not enable tracing automatically.
  void setTraceInfo(TraceInfo &&info) { traceInfo_ = std::move(info); }

  /// Getter for the TraceEvent lookup.
  TraceInfo &getTraceInfo() { return traceInfo_; }
  const TraceInfo &getTraceInfo() const { return traceInfo_; }

  /// Read trace events out of this func and write them into /p ctx
  virtual void translateTraceEvents(Context *ctx) const {}

  /// \returns the Kind of Backend used to compile this function.
  virtual BackendKind getCompileBackendKind() const = 0;

protected:
  /// Flag to ensure setupRuns is only called once.
  bool runsSetup_{false};
  /// Contains symbol offsets and allocation sizes.
  runtime::RuntimeBundle runtimeBundle_;

  /// Information regarding runtime trace instrumentation present in this
  /// function.
  TraceInfo traceInfo_;
};
} // end namespace glow

#endif // GLOW_BACKENDS_COMPILEDFUNCTION_H
