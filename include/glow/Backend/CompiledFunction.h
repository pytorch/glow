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

#include "glow/Backend/BackendUtils.h"
#include "glow/ExecutionContext/ExecutionContext.h"
#include "glow/Graph/Nodes.h"
#include "glow/Support/Error.h"

namespace glow {

class PlaceholderBindings;
/// Interface for executing a compiled function.
class CompiledFunction {
public:
  /// Default Ctor.
  CompiledFunction() = delete;

  /// Ctor that accepts runtimeBundle.
  CompiledFunction(const runtime::RuntimeBundle &bundle);

  /// Dtor.
  virtual ~CompiledFunction();
  /// Execute the network and allocate Placeholder memory with given
  /// \p bindings providing mapping between Placeholder and populated tensor.
  /// \returns an llvm::Error if an error ocurred during execution.
  virtual llvm::Error execute(ExecutionContext *context) = 0;

  /// Getter for the runtimeBundle.
  runtime::RuntimeBundle &getRuntimeBundle() { return runtimeBundle_; }

  /// Collects constants for runtime.
  virtual void collectConstants(const Module *){};

  /// Setter for TraceEvent lookup. Note: does not enable tracing automatically.
  void setTraceInfo(TraceInfo &&info) { traceInfo_ = std::move(info); }

  /// Getter for the TraceEvent lookup.
  TraceInfo &getTraceInfo() { return traceInfo_; }
  const TraceInfo &getTraceInfo() const { return traceInfo_; }

  /// Read trace events out of this func and write them into /p bindings
  virtual void translateTraceEvents(ExecutionContext *bindings) const {}

  /// \returns the backend name used to compile this function.
  virtual std::string getCompileBackendName() const = 0;

protected:
  /// Contains symbol offsets and allocation sizes.
  runtime::RuntimeBundle runtimeBundle_;

  /// Information regarding runtime trace instrumentation present in this
  /// function.
  TraceInfo traceInfo_;
};
} // end namespace glow

#endif // GLOW_BACKENDS_COMPILEDFUNCTION_H
