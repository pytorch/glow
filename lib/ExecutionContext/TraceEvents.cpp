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

#include "glow/ExecutionContext/TraceEvents.h"

#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <fstream>

namespace glow {

void TraceEvent::dumpTraceEvents(
    std::vector<TraceEvent> &events, llvm::StringRef filename,
    const std::string &processName,
    const std::map<int, std::string> &threadNames) {
  llvm::errs() << "dumping " << events.size() << " trace events to "
               << filename.str() << ".\n";

  auto process = processName.empty() ? "glow" : processName;

  std::ofstream file(filename);
  file << "[\n";
  for (const auto &event : events) {
    file << "{\"name\": \"" << event.name;
    file << "\", \"cat\": \"glow\",";
    file << "\"ph\": \"" << event.type;
    file << "\", \"ts\": " << event.timestamp;
    file << ", \"pid\": \"" << process << "\"";

    auto nameIt = threadNames.find(event.tid);
    if (nameIt != threadNames.end()) {
      file << ", \"tid\": \"" << nameIt->second << "\"";
    } else {
      file << ", \"tid\": " << event.tid;
    }

    if (event.type == CompleteType) {
      file << ", \"dur\": " << event.duration;
    }

    if (!event.args.empty()) {
      file << ", \"args\": {";
      bool first{true};
      for (auto &pair : event.args) {
        // Start with a comma unless it's the first item in the list.
        file << (first ? "" : ", ");
        first = false;
        file << "\"" << pair.first << "\" : \"" << pair.second << "\"";
      }
      file << "}";
    }
    file << "},\n";
  }
  // Skip the ending bracket since that is allowed.
  file.close();
}

uint64_t TraceEvent::now() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

size_t TraceEvent::getThreadId() {
  static std::atomic<std::size_t> thread_idx{0};
  thread_local std::size_t id = thread_idx++;
  return id;
}

void TraceContext::logTraceEvent(
    llvm::StringRef name, llvm::StringRef type,
    std::map<std::string, std::string> additionalAttributes) {
  logTraceEvent(name, type, TraceEvent::now(), std::move(additionalAttributes));
}

void TraceContext::logTraceEvent(
    llvm::StringRef name, llvm::StringRef type, uint64_t timestamp,
    std::map<std::string, std::string> additionalAttributes) {
  if (traceLevel_ == TraceLevel::NONE || traceLevel_ == TraceLevel::OPERATOR) {
    return;
  }
  TraceEvent ev(name, timestamp, type, TraceEvent::getThreadId(),
                std::move(additionalAttributes));
  {
    std::lock_guard<std::mutex> l(lock_);
    traceEvents_.push_back(std::move(ev));
  }
}

void TraceContext::logCompleteTraceEvent(
    llvm::StringRef name, uint64_t startTimestamp,
    std::map<std::string, std::string> additionalAttributes) {
  if (traceLevel_ == TraceLevel::NONE || traceLevel_ == TraceLevel::OPERATOR) {
    return;
  }

  TraceEvent ev(name, startTimestamp, TraceEvent::now() - startTimestamp,
                TraceEvent::getThreadId(), std::move(additionalAttributes));
  {
    std::lock_guard<std::mutex> l(lock_);
    traceEvents_.push_back(std::move(ev));
  }
}

void TraceContext::setThreadName(int tid, llvm::StringRef name) {
  threadNames_[tid] = name;
}

void TraceContext::dump(llvm::StringRef filename,
                        const std::string &processName) {
  TraceEvent::dumpTraceEvents(getTraceEvents(), filename,
                              std::move(processName), getThreadNames());
}

void TraceContext::merge(TraceContext *other) {
  std::lock_guard<std::mutex> l(lock_);
  auto &newEvents = other->getTraceEvents();
  std::move(newEvents.begin(), newEvents.end(),
            std::back_inserter(getTraceEvents()));
  auto &names = other->getThreadNames();
  threadNames_.insert(names.begin(), names.end());
  names.clear();
}

ScopedTraceBlock::ScopedTraceBlock(TraceContext *context, llvm::StringRef name)
    : context_(context), name_(name) {
  startTimestamp_ = TraceEvent::now();

  // A local memory fence to prevent the compiler reordering instructions to
  // before taking the start timestamp.
  std::atomic_signal_fence(std::memory_order_seq_cst);
}

ScopedTraceBlock::~ScopedTraceBlock() { end(); }

ScopedTraceBlock &ScopedTraceBlock::addArg(llvm::StringRef key,
                                           llvm::StringRef value) {
  args_[key] = value;
  return *this;
}

void ScopedTraceBlock::end() {
  /// A local memory fence to prevent the compiler reordering intructions to
  /// after the end timestamp.
  std::atomic_signal_fence(std::memory_order_seq_cst);

  if (!end_ && context_) {
    context_->logCompleteTraceEvent(name_, startTimestamp_, std::move(args_));
  }
  end_ = true;
}
} // namespace glow
