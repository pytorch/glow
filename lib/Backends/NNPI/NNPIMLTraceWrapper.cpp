/*
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

#include "NNPIMLTraceWrapper.h"
#include "DebugMacros.h"
#include "nnpi_inference.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <time.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#define MAX_TRACE_BUFFER_SIZE (1024 * 1024 * 5)
#define TRACE_READ_BUFFER_SIZE (1024 * 10)

static inline uint64_t secondsToMicroseconds(double seconds) {
  return (uint64_t)(seconds * 1e6f);
}

static inline int64_t nanosecondsToMicrosecondsSigned(int64_t nanoseconds) {
  return nanoseconds / 1e3;
}

static uint64_t inline getNow() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

enum NNPITraceColumnIndex {
  NNPI_TRACE_PID_IDX = 0,
  NNPI_TRACE_CPU_IDX = 1,
  NNPI_TRACE_FLAG_IDX = 2,
  NNPI_TRACE_TIMESTAMP_IDX = 3,
  NNPI_TRACE_FUNCTION_IDX = 4,
  NNPI_TRACE_DETAILS_IDX = 5
};

class NNPITraceParser {
public:
  void parseLine(std::string line, NNPITraceEntry &entry) {
    size_t idx = 0;
    std::istringstream linestream(line);
    do {
      std::string part;
      linestream >> part;

      switch (idx) {
      case NNPI_TRACE_PID_IDX: {
        entry.processID = getPID(part);
        break;
      }
      case NNPI_TRACE_CPU_IDX: {
        entry.cpuID = getCPUID(part);
        break;
      }
      case NNPI_TRACE_FLAG_IDX: {
        getFlags(part, entry.flags_);
        break;
      }
      case NNPI_TRACE_TIMESTAMP_IDX: {
        entry.deviceUpTime = getOriginTime(part);
        entry.hostTime = entry.deviceUpTime;
        break;
      }
      case NNPI_TRACE_FUNCTION_IDX: {
        entry.traceType = getType(part);
        break;
      }
      case NNPI_TRACE_DETAILS_IDX: {
        // NNPI_TRACE_MARK lines (identified at NNPI_TRACE_FUNCTION_IDX column)
        // has a sub level function type.
        if (entry.traceType == NNPI_TRACE_MARK &&
            part[part.size() - 1] == ':') {
          entry.traceType = getType(part);
          break;
        }
        // Not NNPI_TRACE_MARK: consider as params.
      }
      default: // Params.
      {
        addParam(part, entry);
      }
      }
      idx++;
    } while (linestream);
  }

protected:
  uint32_t getPID(std::string part) {
    std::istringstream partSplitStream(part);
    std::string pid;
    while (std::getline(partSplitStream, pid, '-'))
      ;
    return std::stoi(pid);
  }

  uint32_t getCPUID(std::string part) {
    std::string cpuStr = part.substr(1, part.size() - 2);
    return std::stoi(cpuStr);
  }

  uint64_t getOriginTime(std::string part) {
    double dNumber = std::stod(part.substr(0, part.size() - 1));
    return secondsToMicroseconds(dNumber);
  }

  void getFlags(std::string part, char *flags) {
    if (part.size() != 4) {
      return;
    }
    part.copy(flags, 4);
  }

  NNPITraceType getType(std::string part) {
    if (part == "dma:") {
      return NNPI_TRACE_DMA;
    } else if (part == "copy:") {
      return NNPI_TRACE_COPY;
    } else if (part == "cmdlist:") {
      return NNPI_TRACE_CMDLIST;
    } else if (part == "icedrvExecuteNetwork:") {
      return NNPI_TRACE_NETEXEC;
    } else if (part == "runtime-subgraph:") {
      return NNPI_TRACE_SUBGRAPH;
    } else if (part == "infreq:") {
      return NNPI_TRACE_INFER;
    } else if (part == "clock_sync:") {
      return NNPI_TRACE_CLOCK_SYNC;
    } else if (part == "tracing_mark_write:") {
      return NNPI_TRACE_MARK;
    } else if (part == "vtune_time_sync:") {
      return NNPI_TARCE_TIME_SYNC;
    } else if (part == "runtime-infer-request:") {
      return NNPI_TRACE_RUNTIME_INFER;
    } else if (part == "icedrvScheduleJob:") {
      return NNPI_TRACE_ICED_SCHED_JOB;
    } else if (part == "icedrvCreateNetwork:") {
      return NNPI_TARCE_ICED_CREAT_NET;
    } else if (part == "icedrvNetworkResource:") {
      return NNPI_TARCE_ICED_NET_RES;
    } else if (part == "icedrvEventGeneration:") {
      return NNPI_TARCE_ICED_NET_GEN;
    } else if (part == "user_data:") {
      return NNPI_TARCE_USER_DATA;
    }
    return NNPI_TRACE_OTHER;
  }

  bool addParam(std::string part, NNPITraceEntry &entry) {
    std::string name;
    std::string value;
    std::istringstream partSplitStream(part);
    std::getline(partSplitStream, name, '=');
    std::getline(partSplitStream, value, '=');

    while (value[value.size() - 1] == ',') {
      value = value.substr(0, value.size() - 2);
    }
    entry.params[name] = value;
    return true;
  }
};

#define NNPI_SOFTWARE_EVENTS                                                   \
  "cmdlist,copy,cpylist_create,icedrvCreateContext,icedrvCreateNetwork,"       \
  "icedrvDestroyContext,icedrvDestroyNetwork,icedrvEventGeneration,"           \
  "icedrvExecuteNetwork,icedrvNetworkResource,icedrvScheduleJob,inf_net_"      \
  "subres,infreq,runtime_sw_events.runtime.infer,runtime_sw_events.runtime."   \
  "subgraph,user_data"

NNPITraceContext::NNPITraceContext(unsigned devID)
    : traceCtx_(0), devID_(devID), devIDSet_(false),
      events_(NNPI_SOFTWARE_EVENTS) {}

NNPITraceContext::~NNPITraceContext() { destroyInternalContext(); }

bool NNPITraceContext::startCapture(NNPIDeviceContext deviceContext) {
  if (!createInternalContext()) {
    LOG(WARNING) << "nnpi_trace: Failed to create trace device context.";
    return false;
  }
  nnpimlTraceOptions traceOptions;
  std::memset(&traceOptions, 0, sizeof(nnpimlTraceOptions));
  traceOptions.max_bytes = MAX_TRACE_BUFFER_SIZE;
  traceOptions.max_bytes_valid = true;

  nnpimlStatus mlStatus =
      nnpimlTraceStart(traceCtx_, devID_, &traceOptions, events_.c_str());
  if (mlStatus != NNPIML_SUCCESS) {
    LOG(WARNING) << "nnpi_trace: Failed to start trace, err=" << mlStatus;
    return false;
  }
  LOG_NNPI_INF_IF_ERROR(
      nnpiDeviceContextTraceUserData(deviceContext, "BG", getNow()),
      "Failed to inject trace timestamp - device trace may not be "
      "synchronized");
  return true;
}

bool NNPITraceContext::stopCapture(NNPIDeviceContext deviceContext) const {
  uint32_t outBytes, discardEvents;
  LOG_NNPI_INF_IF_ERROR(
      nnpiDeviceContextTraceUserData(deviceContext, "EN", getNow()),
      "Failed to inject trace timestamp - device trace may not be "
      "synchronized");
  nnpimlStatus mlStatus =
      nnpimlTraceStop(traceCtx_, devID_, &outBytes, &discardEvents);
  if (mlStatus != NNPIML_SUCCESS) {
    return false;
  }
  return true;
}

bool NNPITraceContext::readTraceOutput(std::stringstream &inputStream) {
  char readData[TRACE_READ_BUFFER_SIZE + 1];
  uint32_t size = TRACE_READ_BUFFER_SIZE;
  uint32_t actualSize = size;
  // Read trace bytes into stream.
  uint32_t offset = 0;
  while (actualSize >= size) {
    nnpimlStatus mlStatus =
        nnpimlTraceRead(traceCtx_, devID_, offset, size, readData, &actualSize);
    inputStream.write(readData, actualSize);
    offset += actualSize;
    if (mlStatus != NNPIML_SUCCESS) {
      // Failed to read trace.
      return false;
    }
  }
  return true;
}

bool NNPITraceContext::load() {
  entries_.clear();
  std::stringstream inputStream;

  if (!readTraceOutput(inputStream)) {
    destroyInternalContext();
    return false;
  }
  destroyInternalContext();

  // Handle stream.
  std::string line;
  NNPITraceParser parser;
  bool started = false;
  uint64_t glowStart = 0;
  uint64_t glowEnd = 0;
  uint64_t nnpiStart = 0;
  uint64_t nnpiEnd = 0;

  while (std::getline(inputStream, line)) {
    if (line.find("#", 0) == 0) {
      // Skip comment.
      continue;
    }
    NNPITraceEntry entry;
    parser.parseLine(line, entry);
    if (entry.traceType == NNPI_TARCE_USER_DATA) {
      if (!started && entry.params["key"] == "BG") {
        auto p = entry.params["user_data"];
        glowStart = std::stol(entry.params["user_data"]);
        nnpiStart = entry.deviceUpTime;
        started = true;
      } else if (entry.params["key"] == "EN") {
        auto p = entry.params["user_data"];
        glowEnd = std::stol(entry.params["user_data"]);
        nnpiEnd = entry.deviceUpTime;
        started = false;
      }
    }
    if (started) {
      entries_.push_back(entry);
    }
  }
  if (glowStart > 0 && glowEnd > 0 && nnpiStart > 0 && nnpiEnd > 0) {
    // Calculate host time function.
    double m = (double)(glowEnd - glowStart) / (double)(nnpiEnd - nnpiStart);
    int64_t C = glowStart - m * nnpiStart;
    // Update host time.
    for (NNPITraceEntry &entry : entries_) {
      entry.hostTime = entry.deviceUpTime * m + C;
    }
  } else {
    LOG(WARNING) << "Failed to synchronize glow and nnpi device traces.";
  }
  return true;
}

bool NNPITraceContext::destroyInternalContext() {
  if (traceCtx_ == 0) {
    return false;
  }
  nnpimlStatus mlStatus = nnpimlDestroyTraceContext(traceCtx_);
  traceCtx_ = 0;
  if (mlStatus != NNPIML_SUCCESS) {
    LOG(WARNING) << "nnpi_trace: Failed to stop device trace, err=" << mlStatus;
    traceCtx_ = 0;
    return false;
  }

  return true;
}

bool NNPITraceContext::createInternalContext() {
  if (traceCtx_ != 0) {
    return false;
  }
  devMask_ = 1UL << devID_;
  nnpimlStatus mlStatus =
      nnpimlCreateTraceContext(devMask_, &traceCtx_, &devMask_);
  if (mlStatus != NNPIML_SUCCESS) {
    LOG(WARNING) << "nnpi_trace: Failed to start device trace, err="
                 << mlStatus;
    traceCtx_ = 0;
    return false;
  }
  if (!(1UL << devID_ & devMask_)) {
    destroyInternalContext();
    LOG(WARNING) << "nnpi_trace: Cloud not open trace for device " << devID_;
    return false;
  }
  return true;
}