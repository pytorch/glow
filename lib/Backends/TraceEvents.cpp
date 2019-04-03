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

#include "glow/Backends/TraceEvents.h"

#include "llvm/Support/raw_ostream.h"

#include <fstream>

namespace glow {

void TraceEvent::dumpTraceEvents(
    std::vector<TraceEvent> &events, llvm::StringRef filename,
    const std::string &processName,
    const std::map<int, std::string> &threadNames) {
  llvm::errs() << "dumping " << events.size() << " trace events.\n";

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
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
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
  TraceEvent ev(name, timestamp, type, traceThread_,
                std::move(additionalAttributes));
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
  if (context_) {
    context_->logTraceEvent(name_, TraceEvent::BeginType, std::move(args_));
  }
}

ScopedTraceBlock::~ScopedTraceBlock() { end(); }

ScopedTraceBlock &ScopedTraceBlock::addArg(llvm::StringRef key,
                                           llvm::StringRef value) {
  args_[key] = value;
  return *this;
}

void ScopedTraceBlock::end() {
  if (!end_ && context_) {
    context_->logTraceEvent(name_, TraceEvent::EndType, std::move(args_));
  }
  end_ = true;
}
} // namespace glow
