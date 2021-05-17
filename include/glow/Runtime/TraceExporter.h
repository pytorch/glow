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
#ifndef GLOW_RUNTIME_TRACEEXPORTER_H
#define GLOW_RUNTIME_TRACEEXPORTER_H

#include "glow/ExecutionContext/TraceEvents.h"

#include <mutex>
#include <vector>

namespace glow {

/// Interface for exporting trace events.
//   TraceExporter provides two functionalities
//   1) shouldTrace() : a method to determine if the runtimeshould trace a
//     particular request. This can be used to windowed tracing on-demand on
//     a production system
//   2) exportTrace(..) : that passes collected trace events to be exported
//     in the target format and destination.
//
// The base implementation delegates to any subclass registered
// via `registerTraceExporter`.

class TraceExporter {
public:
  /// Dtor.
  virtual ~TraceExporter() = default;

  /// Determine if this request should be traced.
  virtual bool shouldTrace() = 0;

  /// Export events from the given TraceContext
  virtual void exportTrace(TraceContext *context) = 0;
};

/// Registry of TraceExporters.
class TraceExporterRegistry final {
public:
  /// Determine if this request should be traced.
  bool shouldTrace();

  /// Export events from the given TraceContext
  void exportTrace(TraceContext *tcontext);

  /// Register a TraceExporter.
  void registerTraceExporter(TraceExporter *exporter);

  /// Revoke a TraceExporter.
  void revokeTraceExporter(TraceExporter *exporter);

  /// Static singleton TraceExporter.
  static std::shared_ptr<TraceExporterRegistry> getInstance();

private:
  /// Registered TraceExporters.
  std::vector<TraceExporter *> exporters_;
  std::mutex mutex_;
};

} // namespace glow

#endif // GLOW_RUNTIME_TRACEEXPORTER_H
