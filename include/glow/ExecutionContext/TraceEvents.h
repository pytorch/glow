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
#include <mutex>
#include <vector>

namespace glow {

class PlaceholderBindings;

/// An individual tracing event, such as the begin or end of an instruction.
/// Designed to match the Google Trace Event Format for Chrome:
/// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
struct TraceEvent {
  /// The amount and type of TraceEvents that should appear in the trace.
  enum TraceLevel {
    NONE = 0x00,     // No trace events.
    REQUEST = 0x01,  // Request timing events only.
    RUNTIME = 0x02,  // Glow runtime events only.
    OPERATOR = 0x04, // Backend operator instrumentation only.
    DEBUG = 0x08,    // Full debug events with extra information.
    STANDARD =
        RUNTIME | OPERATOR, // Glow runtime events and backend operator events.
  };

  /// Event Types.
  static constexpr auto BeginType = 'B';
  static constexpr auto EndType = 'E';
  static constexpr auto InstantType = 'I';
  static constexpr auto CompleteType = 'X';
  /// MetadataType is used for the thread name mapping.
  static constexpr auto MetadataType = 'M';

  /// Human readable name for the item, will be used to match up begin and end.
  std::string name;

  /// Time of the event, in microseconds since epoch.
  uint64_t timestamp;

  /// Type of the event, a (usually) one char code (see Event Descriptions in
  /// the Trace Event Format spec). e.g. 'B' for begin event, 'E' for end event.
  char type;

  /// Thread Id for this event. All Events on the same tid will be shown on the
  /// same row of the trace.
  int tid;

  /// Duration of the event (for Complete events).
  uint64_t duration{0};

  /// Arbitrary TraceEvent arguments (from spec).
  std::map<std::string, std::string> args;

  TraceEvent(llvm::StringRef n, uint64_t ts, char c, int t)
      : name(n), timestamp(ts), type(c), tid(t) {}

  TraceEvent(llvm::StringRef n, uint64_t ts, char c, int t,
             std::map<std::string, std::string> a)
      : name(n), timestamp(ts), type(c), tid(t), args(a) {}

  TraceEvent(llvm::StringRef n, uint64_t ts, uint64_t dur, int t,
             std::map<std::string, std::string> a = {})
      : name(n), timestamp(ts), type(CompleteType), tid(t), duration(dur),
        args(a) {}

  static void
  dumpTraceEvents(std::vector<TraceEvent> &events, llvm::StringRef filename,
                  const std::string &processName = "",
                  const std::map<int, std::string> &threadNames = {});

  // Return the current time in microseconds in the timestamp domain.
  static uint64_t now();

  // Returns a unique id associated with the current thread.
  static size_t getThreadId();
};

using TraceLevel = TraceEvent::TraceLevel;

/// Tracing / Profiling events map for a CompiledFunction.
/// This class encodes information on how to read event metrics out of Tensors
/// and into TraceEvents, and is used by
/// CompiledFunction::translateTraceEvents().
struct TraceInfo {
  TraceInfo() = default;
  TraceInfo(bool e, size_t d) : enabled(e), dataSize(d) {}

  /// Whether tracing is enabled for this run.
  bool enabled{false};

  /// Whether the function was auto instrumented.
  bool autoInstrumented{false};

  /// The size of each item in the backing Tensor.
  size_t dataSize{0};

  struct Event {
    size_t startIndex;
    size_t endIndex;
    std::string name;
    char type;

    // additional info per backend. May not be present.
    std::string context;
  };

  std::map<Placeholder *, std::vector<Event>> events;

  void add(Placeholder *PH, size_t index, std::string name, char type) {
    events[PH].push_back({index, 0, std::move(name), std::move(type), ""});
  }

  void add(Placeholder *PH, size_t index, std::string name, char type,
           std::string context) {
    events[PH].push_back(
        {index, 0, std::move(name), std::move(type), std::move(context)});
  }

  /// Add data for a Complete TraceEvent.
  void add(Placeholder *PH, size_t startIndex, size_t endIndex,
           std::string name, std::string context = "") {
    events[PH].push_back({startIndex, endIndex, std::move(name),
                          TraceEvent::CompleteType, std::move(context)});
  }
};

/// A context for storing TraceEvents throughout a run (ie. between
/// partitioned CompiledFunctions).
class TraceContext {
  /// The list of materialized Events filled out with timestamp and metadata.
  std::vector<TraceEvent> traceEvents_;

  /// Human readable name mapping for trace Threads.
  std::map<int, std::string> threadNames_;

  /// The detail level of tracing for this run.
  TraceLevel traceLevel_{TraceLevel::NONE};

  /// Lock around traceEvents_.
  std::mutex lock_;

public:
  TraceContext(TraceLevel level) : traceLevel_(level) {}

  /// \returns TraceEvents for the last run.
  std::vector<TraceEvent> &getTraceEvents() { return traceEvents_; }

  /// \returns TraceEvents for the last run.
  llvm::ArrayRef<TraceEvent> getTraceEvents() const { return traceEvents_; }

  /// \returns the level of verbosity allowed for TraceEvents.
  TraceLevel getTraceLevel() { return traceLevel_; }

  /// Sets the level of verbosity for TraceEvents.
  void setTraceLevel(TraceLevel level) { traceLevel_ = level; }

  /// \returns true if should log an event of the provided \p level.
  bool shouldLog(TraceLevel level) { return (traceLevel_ & level) != 0; }

  /// Logs a new TraceEvent at the current time with the given \p name, \p
  /// type and optionally additional attributes.
  void
  logTraceEvent(llvm::StringRef name, TraceLevel level,
                char type = TraceEvent::InstantType,
                std::map<std::string, std::string> additionalAttributes = {});

  // Logs a new TraceEvent at the provided \p timestamp, with the given \p
  // name, \p type and optionally additional attributes.
  void
  logTraceEvent(llvm::StringRef name, TraceLevel level, char type,
                uint64_t timestamp,
                std::map<std::string, std::string> additionalAttributes = {});

  /// Logs a new TraceEvent with the Complete event type, the start time is
  /// provided and uses the current time to determine duration.
  void logCompleteTraceEvent(
      llvm::StringRef name, TraceLevel level, uint64_t startTimestamp,
      std::map<std::string, std::string> additionalAttributes = {});

  /// Sets the human readable \p name for thread \tid.
  void setThreadName(int tid, llvm::StringRef name);

  /// Sets the human readable \p name for the current thread (by
  /// TraceEvent::getThreadId()).
  void setThreadName(llvm::StringRef name);

  /// \returns the list of human readable thread names.
  std::map<int, std::string> &getThreadNames() { return threadNames_; }

  /// Dumps all TraceEvents in json format to the given \p filename,
  /// optionally with a provided \p processName.
  void dump(llvm::StringRef filename, const std::string &processName = "");

  /// Moves all TraceEvents and thread names in \p other into this context.
  void merge(TraceContext *other);

  /// Moves all TraceEvents and thread names in \p other into this context.
  /// This version is destructive of the other TraceContext.
  void merge(std::unique_ptr<TraceContext> other) { merge(other.get()); }
};

/// These macros predicate the logging of a TraceEvent on a validity of the
/// given TraceContext.

/// Logs a new "Begin" event, beginning an event with duration.
#define TRACE_EVENT_BEGIN(ctx, level, name)                                    \
  if (ctx) {                                                                   \
    ctx->logTraceEvent(name, level, TraceEvent::BeginType);                    \
  }

/// Logs a new "End" event, ending an event with duration.
#define TRACE_EVENT_END(ctx, level, name)                                      \
  if (ctx) {                                                                   \
    ctx->logTraceEvent(name, level, TraceEvent::EndType);                      \
  }

/// Logs a new "Instant" event, which has an associated time, but no duration.
#define TRACE_EVENT_INSTANT(ctx, level, name)                                  \
  if (ctx) {                                                                   \
    ctx->logTraceEvent(name, level, TraceEvent::InstantType);                  \
  }

/// Logs a new TraceEvent with the provided type and timestamp.
#define TRACE_EVENT_LOG(ctx, level, name, type, ts)                            \
  if (ctx) {                                                                   \
    ctx->logTraceEvent(name, level, type, ts);                                 \
  }

/// Logs a new TraceEvent which begins and ends in the current scope block.
#define TRACE_EVENT_SCOPE(ctx, level, name)                                    \
  ScopedTraceBlock __event__(ctx, level, name);

/// Logs a new scoped TraceEvent with the provided name, allowing multiple
/// within the same scope.
#define TRACE_EVENT_SCOPE_NAMED(ctx, level, name, objName)                     \
  ScopedTraceBlock objName(ctx, level, name);

/// End a scoped TraceEvent before its scope exits.
#define TRACE_EVENT_SCOPE_END() __event__.end();

/// End a named scoped TraceEvent before its scope exits.
#define TRACE_EVENT_SCOPE_END_NAMED(name) name.end();

class ExecutionContext;

/// Helper class which uses RAII for the start and end times of a TraceEvent.
/// At creation will create a "begin" TraceEvent and at destuction (or end())
/// will create an "end" TraceEvent.
class ScopedTraceBlock {
  /// The context to log to.
  TraceContext *context_;

  /// The TraceLevel of the associated TraceEvent.
  TraceLevel level_;

  /// The name of the event.
  llvm::StringRef name_;

  /// Timestamp of the beginning of this event.
  uint64_t startTimestamp_;

  /// Additional metadata associated with the event, which will be visible in
  /// the properties display of the event in the tracing visualizer.
  std::map<std::string, std::string> args_;

  /// Whether this event has already logged the "end" event, to avoid logging
  /// it twice.
  bool end_{false};

public:
  ScopedTraceBlock(TraceContext *context, TraceLevel level,
                   llvm::StringRef name);

  ScopedTraceBlock(ExecutionContext *context, TraceLevel level,
                   llvm::StringRef name);
  ~ScopedTraceBlock();

  /// Adds an argument to the metadata for this object.
  ScopedTraceBlock &addArg(llvm::StringRef key, llvm::StringRef value);

  /// Triggers the "end" event before destruction of the object.
  void end();
};

} // namespace glow

#endif // GLOW_BACKENDS_TRACEEVENTS_H
