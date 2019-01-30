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
#ifndef GLOW_RUNTIME_TRACELOGGER_H
#define GLOW_RUNTIME_TRACELOGGER_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <vector>

namespace glow {
namespace runtime {

/// An individual tracing event, such as the begin or end of an instruction.
/// Designed to match the Google Trace Event Format for Chrome:
// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
struct TraceEvent {
  /// Human readable name for the item, will be used to match up begin and end.
  std::string name;

  /// Time of the event, in milliseconds since epoch.
  uint64_t timestamp;

  /// Type of the event, a one char code (see Event Descriptions in the Trace
  /// Event Format spec). e.g. 'B' for begin event, 'E' for end event.
  char type;

  /// Thread Id for this event. All Events on the same tid will be shown on the
  /// same row of the trace.
  int tid;

  TraceEvent(llvm::StringRef n, uint64_t ts, char c, int t)
      : name(n), timestamp(ts), type(c), tid(t) {}
};

/// Aggregator for a single thread of execution's TraceEvents.
/// All events on this TraceThread share a tid.
/// This abstraction is designed to to allow concurrent generation of events
/// without requiring synchronization, but this class is not thread safe.
class TraceThread {
  /// TraceEvents for this tid. Can be unordered.
  std::vector<TraceEvent> traceEvents_;

  /// The Thread Id of this TraceThread.
  int tid_;

  friend class TraceLogger;

public:
  TraceThread(int tid);

  /// Create and store a BEGIN (type 'B') TraceEvent with the \name and the
  /// current timestamp.
  void beginTraceEvent(llvm::StringRef name);

  /// Create and store an END (type 'E') TraceEvent with the \name and the
  /// current timestamp. This should match a previous beginTraceEvent.
  void endTraceEvent(llvm::StringRef name);
};

/// Aggregator for a single run's TraceEvents, i.e. for a single inference.
// The usage pattern should be something like:
//     1. getTraceThread() to get a logger per thread of execution.
//     2. log events on the traceThread()
//     3. when the execution is finished, returnTraceThread(traceThread) to
//        aggregate TraceEvents.
class TraceLogger {
  /// TraceEvents for this pid. Can be unordered.
  std::vector<TraceEvent> traceEvents_;

  /// Thread Id to use for the next TraceThread.
  int nextThreadId{0};

  /// The Process Id of this trace (unused currently).
  int pid_;

public:
  TraceLogger(int pid = 0);

  /// \returns a new TraceThread for the given tid. That TraceThread does not
  /// share state with the TraceLogger. Ia \p tid is not specified, will use the
  /// next sequential Thread Id.
  TraceThread getTraceThread(int tid = -1);

  /// Collects TraceEvents from the provided \p traceThread and moves them into
  /// the TraceLogger. The TraceThread can still be used, however it's event
  /// vector is cleared.
  void returnTraceThread(TraceThread &traceThread);
  void returnTraceThread(TraceThread &&traceThread);

  /// Writes a JSON file containing TraceEvents to the \p outputPath.
  void dumpTraceEvents(llvm::StringRef outputPath);
};

} // namespace runtime
} // namespace glow

#endif
