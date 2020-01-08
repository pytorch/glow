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
  NNPITraceParser(uint64_t timeDiff, uint64_t upRefTime)
      : timeDiff_(timeDiff), upRefTime_(upRefTime), refTimeDiff_(0.0){};

  int64_t getTimeDiff() {
    return abs(refTimeDiff_) > 0 ? refTimeDiff_ : timeDiff_;
  };

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
    setHostTime(entry);
  }

  void setHostTime(NNPITraceEntry &entry) {
    if (refTimeDiff_ == 0 && upRefTime_ != 0 &&
        entry.traceType == NNPI_TRACE_COPY) {
      if (entry.params.count("isC2H") > 0 && entry.params["isC2H"] == "0" &&
          entry.params.count("state") > 0 && entry.params["state"] == "q") {
        refTimeDiff_ = entry.deviceUpTime - upRefTime_;
        timeDiff_ = refTimeDiff_;
      }
    }
    if (refTimeDiff_ != 0) {
      entry.hostTime = (entry.deviceUpTime - refTimeDiff_);
    } else {
      entry.hostTime = (entry.deviceUpTime - timeDiff_);
    }
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
    } else if (part == "infreq:") {
      return NNPI_TRACE_INFER;
    } else if (part == "clock_sync:") {
      return NNPI_TRACE_CLOCK_SYNC;
    } else if (part == "tracing_mark_write:") {
      return NNPI_TRACE_MARK;
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
    if (refTimeDiff_ == 0 && entry.traceType == NNPI_TRACE_CLOCK_SYNC &&
        name == "clock_diff_in_nanosec") {
      // Nanoseconds to microseconds.
      timeDiff_ = nanosecondsToMicrosecondsSigned(std::stol(value));
    }
    return true;
  }

  int64_t timeDiff_;
  uint64_t upRefTime_;
  int64_t refTimeDiff_;
};

NNPITraceContext::NNPITraceContext(uint32_t eventsMask)
    : devID_(0), devIDSet_(false), events_("copy,infreq") {
  if (eventsMask) {
    events_ = "";
    if (eventsMask & NNPI_TRACE_DMA) {
      events_ += "dma,";
    }
    if (eventsMask & NNPI_TRACE_COPY) {
      events_ += "copy,";
    }
    if (eventsMask & NNPI_TRACE_INFER) {
      events_ += "infreq";
    }
  }
  createContext();
}

NNPITraceContext::~NNPITraceContext() {
  nnpimlDestroyTraceContext(traceCtx_);
  traceCtx_ = 0;
}

bool NNPITraceContext::startCapture() const {
  if (!(1UL << devID_ & devMask_)) {
    // Can't start for this device.
    return false;
  }
  nnpimlTraceOptions traceOptions;
  std::memset(&traceOptions, 0, sizeof(nnpimlTraceOptions));
  traceOptions.max_bytes = MAX_TRACE_BUFFER_SIZE;
  traceOptions.max_bytes_valid = true;
  nnpimlStatus mlStatus =
      nnpimlTraceStart(traceCtx_, devID_, &traceOptions, events_.c_str());
  if (mlStatus != NNPIML_SUCCESS) {
    LOG(WARNING) << "nnpi_trace: Failed to read trace file, err=" << mlStatus;
    return false;
  }
  return true;
}

bool NNPITraceContext::stopCapture() const {
  uint32_t outBytes, discardEvents;
  nnpimlStatus mlStatus =
      nnpimlTraceStop(traceCtx_, devID_, &outBytes, &discardEvents);
  if (mlStatus != NNPIML_SUCCESS) {
    return false;
  }
  return true;
}

bool NNPITraceContext::load() {
  entries_.clear();
  std::stringstream inputStream;
  uint32_t size = TRACE_READ_BUFFER_SIZE;
  uint32_t actualSize = size;
  char readData[TRACE_READ_BUFFER_SIZE + 1];

  // Read trace bytes into stream.
  while (actualSize >= size) {
    nnpimlStatus mlStatus =
        nnpimlTraceRead(traceCtx_, devID_, 0, size, readData, &actualSize);
    inputStream.write(readData, actualSize);
    if (mlStatus != NNPIML_SUCCESS) {
      // Failed to read trace.
      return false;
    }
  }

  // Handle stream.
  std::string line;
  NNPITraceParser parser(timeDiff_, upRefTime_);
  while (std::getline(inputStream, line)) {
    if (line.find("#", 0) == 0) {
      // Skip comment.
      continue;
    }
    NNPITraceEntry entry;

    parser.parseLine(line, entry);
    entries_.push_back(entry);
    if (timeDiff_ != parser.getTimeDiff()) {
      // On time diff updated update old entries.
      timeDiff_ = parser.getTimeDiff();
      for (std::vector<NNPITraceEntry>::iterator it =
               entries_.begin() + timeUpdatedIndex_;
           it != entries_.end() - 1; ++it) {
        parser.setHostTime(*it);
        timeUpdatedIndex_++;
      }
    }
  }
  return true;
}

bool NNPITraceContext::setDeviceID(uint32_t devID) {
  if (devIDSet_) {
    return false;
  }
  if (!(1UL << devID & devMask_)) {
    // Can't start for this device.
    return false;
  }
  devIDSet_ = true;
  devID_ = devID;
  return true;
}

bool NNPITraceContext::createContext() {
  nnpimlStatus mlStatus =
      nnpimlCreateTraceContext(UINT64_MAX, &traceCtx_, &devMask_);
  if (mlStatus != NNPIML_SUCCESS) {
    LOG(WARNING) << "nnpi_trace: Failed to read trace file, err=" << mlStatus;
    traceCtx_ = 0;
    return false;
  }
  return true;
}

void NNPITraceContext::markInputCopyStart(uint64_t uptime) {
  if (upRefTime_ == 0) {
    upRefTime_ = uptime;
  }
}
