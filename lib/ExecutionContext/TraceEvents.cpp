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

#include "glow/ExecutionContext/TraceEvents.h"
#include "glow/ExecutionContext/ExecutionContext.h"
#include "glow/Support/ThreadPool.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <fstream>
#include <glog/logging.h>

namespace glow {

void writeMetadataHelper(llvm::raw_fd_ostream &file, llvm::StringRef type,
                         int id, llvm::StringRef name) {
  file << "{\"cat\": \"__metadata\", \"ph\":\"" << TraceEvent::MetadataType
       << "\", \"ts\":0, \"pid\":0, \"tid\":" << id << ", \"name\":\""
       << type.str() << "\", \"args\": {\"name\":\"" << name.str()
       << "\"} },\n";
}

void TraceEvent::dumpTraceEvents(
    std::list<TraceEvent> &events, llvm::StringRef filename,
    const std::string &processName,
    const std::map<int, std::string> &threadNames) {
  LOG(INFO) << "dumping " << events.size() << " trace events to "
            << filename.str();

  auto process = processName.empty() ? "glow" : processName;
  std::error_code EC;
  llvm::raw_fd_ostream file(filename, EC);

  // Print an error message if the output stream can't be created.
  if (EC) {
    LOG(ERROR) << "Unable to open file " << filename.str();
    return;
  }

  file << "[\n";
  /// Set up process name metadata.
  writeMetadataHelper(file, "process_name", 0,
                      processName.empty() ? "glow" : processName);

  /// And thread name metadata.
  for (const auto &nameMap : threadNames) {
    // Put thread name ahead of thread ID so chrome will group thread with the
    // same prefix together.
    writeMetadataHelper(
        file, "thread_name", nameMap.first,
        llvm::formatv("{1}: {0}", nameMap.first, nameMap.second).str());
  }

  bool first{true};
  for (const auto &event : events) {
    file << (first ? "" : ",\n");
    first = false;

    file << "{\"name\": \"" << event.name;
    file << "\", \"cat\": \"" << traceLevelToString(event.level) << "\",";
    file << "\"ph\": \"" << event.type;
    file << "\", \"ts\": " << event.timestamp;
    file << ", \"pid\": 0";
    file << ", \"tid\": " << event.tid;

    if (event.type == CompleteType) {
      file << ", \"dur\": " << event.duration;
    }

    if (event.id != -1) {
      file << ", \"id\": \"" << event.id << "\"";
    }

    if (!event.args.empty()) {
      file << ", \"args\": {";
      bool firstArg{true};
      for (auto &pair : event.args) {
        // Start with a comma unless it's the first item in the list.
        file << (firstArg ? "" : ", ");
        firstArg = false;
        file << "\"" << pair.first << "\" : \"" << pair.second << "\"";
      }
      file << "}";
    }
    file << "}";
  }
  file << "\n]";
  file.close();
}

uint64_t TraceEvent::now() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

llvm::StringRef TraceEvent::traceLevelToString(TraceLevel level) {
  switch (level) {
  case NONE:
    return "None";
  case REQUEST:
    return "Request";
  case RUNTIME:
    return "Runtime";
  case COPY:
    return "Copy";
  case OPERATOR:
    return "Operator";
  case DEBUG:
    return "Debug";
  case STANDARD:
    return "Standard";
  }

  return "Unknown";
}

void TraceContext::logTraceEvent(
    llvm::StringRef name, TraceLevel level, char type,
    std::map<std::string, std::string> additionalAttributes, size_t tid,
    int id) {
  logTraceEvent(name, level, type, TraceEvent::now(),
                std::move(additionalAttributes), tid, id);
}

void TraceContext::logTraceEvent(
    llvm::StringRef name, TraceLevel level, char type, uint64_t timestamp,
    std::map<std::string, std::string> additionalAttributes, size_t tid,
    int id) {
  if (!shouldLog(level)) {
    return;
  }

  TraceEvent ev(name, level, timestamp, type, tid,
                std::move(additionalAttributes), id);
  {
    std::lock_guard<std::mutex> l(lock_);
    traceEvents_.push_back(std::move(ev));
  }
}

void TraceContext::logTraceEvent(TraceEvent &&ev) {
  if (!shouldLog(ev.level)) {
    return;
  }
  std::lock_guard<std::mutex> l(lock_);
  traceEvents_.push_back(std::move(ev));
}

void TraceContext::logCompleteTraceEvent(
    llvm::StringRef name, TraceLevel level, uint64_t startTimestamp,
    std::map<std::string, std::string> additionalAttributes) {
  this->logCompleteTraceEvent(name, level, startTimestamp,
                              std::move(additionalAttributes),
                              threads::getThreadId());
}

void TraceContext::logCompleteTraceEvent(
    llvm::StringRef name, TraceLevel level, uint64_t startTimestamp,
    std::map<std::string, std::string> additionalAttributes, size_t tid) {
  if (!shouldLog(level)) {
    return;
  }

  TraceEvent ev(name, level, startTimestamp, TraceEvent::now() - startTimestamp,
                tid, std::move(additionalAttributes));
  {
    std::lock_guard<std::mutex> l(lock_);
    traceEvents_.push_back(std::move(ev));
  }
}

void TraceContext::setThreadName(int tid, llvm::StringRef name) {
  std::lock_guard<std::mutex> l(lock_);
  threadNames_[tid] = name.str();
}

void TraceContext::setThreadName(llvm::StringRef name) {
  setThreadName(threads::getThreadId(), name);
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
  newEvents.clear();
  auto &names = other->getThreadNames();
  threadNames_.insert(names.begin(), names.end());
  names.clear();
}

void TraceContext::copy(TraceContext *other) {
  std::lock_guard<std::mutex> l(lock_);
  auto &newEvents = other->getTraceEvents();
  std::copy(newEvents.begin(), newEvents.end(),
            std::back_inserter(getTraceEvents()));
  auto &names = other->getThreadNames();
  threadNames_.insert(names.begin(), names.end());
}

ScopedTraceBlock::ScopedTraceBlock(TraceContext *context, TraceLevel level,
                                   llvm::StringRef name)
    : context_(context), level_(level), name_(name) {
  startTimestamp_ = TraceEvent::now();

  // A local memory fence to prevent the compiler reordering instructions to
  // before taking the start timestamp.
  std::atomic_signal_fence(std::memory_order_seq_cst);
}

ScopedTraceBlock::ScopedTraceBlock(ExecutionContext *context, TraceLevel level,
                                   llvm::StringRef name)
    : ScopedTraceBlock(context ? context->getTraceContext() : nullptr, level,
                       name) {}

ScopedTraceBlock::~ScopedTraceBlock() { end(); }

ScopedTraceBlock &ScopedTraceBlock::addArg(llvm::StringRef key,
                                           llvm::StringRef value) {
  args_[key.str()] = value.str();
  return *this;
}

void ScopedTraceBlock::end() {
  /// A local memory fence to prevent the compiler reordering intructions to
  /// after the end timestamp.
  std::atomic_signal_fence(std::memory_order_seq_cst);

  if (!end_ && context_) {
    context_->logCompleteTraceEvent(name_, level_, startTimestamp_,
                                    std::move(args_));
  }
  end_ = true;
}
} // namespace glow
