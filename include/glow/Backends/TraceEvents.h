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
#ifndef GLOW_BACKENDS_TRACEEVENTS_H
#define GLOW_BACKENDS_TRACEEVENTS_H

#include "glow/Graph/Nodes.h"
#include "llvm/ADT/DenseMap.h"

#include <map>
#include <vector>

namespace glow {

class PlaceholderBindings;

/// An individual tracing event, such as the begin or end of an instruction.
/// Designed to match the Google Trace Event Format for Chrome:
// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
struct TraceEvent {
  /// Human readable name for the item, will be used to match up begin and end.
  std::string name;

  /// Time of the event, in milliseconds since epoch.
  uint64_t timestamp;

  /// Type of the event, a (usually) one char code (see Event Descriptions in
  /// the Trace Event Format spec). e.g. 'B' for begin event, 'E' for end event.
  std::string type;

  /// Thread Id for this event. All Events on the same tid will be shown on the
  /// same row of the trace.
  int tid;

  /// Arbitrary TraceEvent arguments (from spec).
  std::map<std::string, std::string> args;

  TraceEvent(llvm::StringRef n, uint64_t ts, llvm::StringRef c, int t)
      : name(n), timestamp(ts), type(c), tid(t) {}

  TraceEvent(llvm::StringRef n, uint64_t ts, llvm::StringRef c, int t,
             std::map<std::string, std::string> a)
      : name(n), timestamp(ts), type(c), tid(t), args(a) {}

  static void dumpTraceEvents(std::vector<TraceEvent> &events,
                              llvm::StringRef filename);

  static uint64_t now();
};

/// Tracing / Profiling events map for a CompiledFunction.
/// This class encodes information on how to read event metrics out of Tensors
/// and into TraceEvents, and is used by
/// CompiledFunction::translateTraceEvents().
struct TraceInfo {
  TraceInfo() = default;
  TraceInfo(bool e, size_t d) : enabled(e), dataSize(d) {}

  /// Whether tracing is enabled for this run.
  bool enabled{false};

  /// The size of each item in the backing Tensor.
  size_t dataSize{0};

  struct Event {
    size_t index;
    std::string name;
    std::string type;
  };

  std::map<Placeholder *, std::vector<Event>> events;

  void add(Placeholder *PH, size_t index, std::string name, std::string type) {
    events[PH].push_back({index, std::move(name), std::move(type)});
  }
};

/// The amount and type of TraceEvents that should appear in the trace.
enum class TraceLevel {
  NONE,     // No trace events.
  RUNTIME,  // Glow runtime events only.
  OPERATOR, // Backend operator instrumentation only.
  STANDARD, // Glow runtime events and backend operator events.
  DEBUG     // Full debug events with extra information.
};

/// A context for storing TraceEvents throughout a run (ie. between partitioned
/// CompiledFunctions).
class TraceContext {
  /// The list of materialized Events filled out with timestamp and metadata.
  std::vector<TraceEvent> traceEvents_;

  /// The detail level of tracing for this run.
  TraceLevel traceLevel_{TraceLevel::NONE};

  /// The thread (tid) used in the output tracing, allowing separation of events
  /// on different contexts.
  int traceThread_{0};

public:
  TraceContext(TraceLevel level, int thread)
      : traceLevel_(level), traceThread_(thread) {}
  /// \returns TraceEvents for the last run.
  std::vector<TraceEvent> &getTraceEvents() { return traceEvents_; }

  int getTraceThread() const { return traceThread_; }
  void setTraceThread(int tid) { traceThread_ = tid; }

  TraceLevel getTraceLevel() { return traceLevel_; }
  void setTraceLevel(TraceLevel level) { traceLevel_ = level; }

  void logTraceEvent(llvm::StringRef name, llvm::StringRef type = "i",
                     std::map<std::string, std::string> args = {});
};

/// Helper class which uses RAII for the start and end times of a TraceEvent.
/// At creation will create a "begin" TraceEvent and at destuction (or end())
/// will create an "end" TraceEvent.
class ScopedTraceBlock {
  /// The context to log to.
  TraceContext *context_;

  /// The name of the event.
  llvm::StringRef name_;

  /// Additional metadata associated with the event, which will be visible in
  /// the properties display of the event in the tracing visualizer.
  std::map<std::string, std::string> args_;

  /// Whether this event has already logged the "end" event, to avoid logging it
  /// twice.
  bool end_{false};

public:
  ScopedTraceBlock(TraceContext *context, llvm::StringRef name);
  ~ScopedTraceBlock();

  /// Adds an argument to the metadata for this object.
  ScopedTraceBlock &addArg(llvm::StringRef key, llvm::StringRef value);

  /// Triggers the "end" event before destruction of the object.
  void end();
};

} // namespace glow

#endif // GLOW_BACKENDS_TRACEEVENTS_H
