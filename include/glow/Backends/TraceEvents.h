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
};

/// Tracing / Profiling events map for a compiled function.
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

} // namespace glow

#endif // GLOW_BACKENDS_TRACEEVENTS_H
