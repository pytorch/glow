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

#include "glow/Runtime/TraceLogger.h"

#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <fstream>

using namespace std::chrono;

namespace glow {
namespace runtime {

uint64_t timestamp_now() {
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch())
      .count();
}

TraceThread::TraceThread(int tid) : tid_(tid) {}

void TraceThread::beginTraceEvent(llvm::StringRef name) {
  traceEvents_.push_back(TraceEvent(name, timestamp_now(), 'B', tid_));
}

void TraceThread::endTraceEvent(llvm::StringRef name) {
  traceEvents_.push_back(TraceEvent(name, timestamp_now(), 'E', tid_));
}

TraceLogger::TraceLogger(int pid) : pid_(pid) {}

TraceThread TraceLogger::getTraceThread(int tid) {
  if (tid < 0) {
    tid = nextThreadId++;
  }

  return TraceThread(tid);
}

void TraceLogger::returnTraceThread(TraceThread &traceThread) {
  traceEvents_.insert(traceEvents_.end(),
                      std::make_move_iterator(traceThread.traceEvents_.begin()),
                      std::make_move_iterator(traceThread.traceEvents_.end()));
}

void TraceLogger::returnTraceThread(TraceThread &&traceThread) {
  returnTraceThread(traceThread);
}

void TraceLogger::dumpTraceEvents(llvm::StringRef path) {
  llvm::errs() << "dumping " << traceEvents_.size() << " trace events.\n";

  std::ofstream file(path);
  file << "[\n";
  for (const auto &event : traceEvents_) {
    file << "{\"name\": \"" << event.name
         << "\", \"cat\": \"glow,interpreter\", \"ph\": \"" << event.type
         << "\", \"ts\": " << event.timestamp << ", \"pid\": " << pid_
         << ", \"tid\": " << event.tid << "},\n";
  }
  // Skip the ending bracket since that is allowed.
  file.close();
}

} // namespace runtime
} // namespace glow
