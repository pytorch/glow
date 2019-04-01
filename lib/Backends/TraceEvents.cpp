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

void TraceEvent::dumpTraceEvents(std::vector<TraceEvent> &events,
                                 llvm::StringRef filename) {
  llvm::errs() << "dumping " << events.size() << " trace events.\n";

  std::ofstream file(filename);
  file << "[\n";
  for (const auto &event : events) {
    file << "{\"name\": \"" << event.name;
    file << "\", \"cat\": \"glow\",";
    file << "\"ph\": \"" << event.type;
    file << "\", \"ts\": " << event.timestamp;
    file << ", \"pid\": " << 0;
    file << ", \"tid\": " << event.tid;

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
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void TraceContext::logTraceEvent(llvm::StringRef name, llvm::StringRef type,
                                 std::map<std::string, std::string> args) {
  if (traceLevel_ == TraceLevel::NONE || traceLevel_ == TraceLevel::OPERATOR) {
    return;
  }
  TraceEvent ev(name, TraceEvent::now(), type, traceThread_, std::move(args));
  traceEvents_.push_back(std::move(ev));
}

ScopedTraceBlock::ScopedTraceBlock(TraceContext *context, llvm::StringRef name)
    : context_(context), name_(name) {
  if (context_) {
    context_->logTraceEvent(name_, "B", std::move(args_));
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
    context_->logTraceEvent(name_, "E", std::move(args_));
  }
  end_ = true;
}
} // namespace glow
