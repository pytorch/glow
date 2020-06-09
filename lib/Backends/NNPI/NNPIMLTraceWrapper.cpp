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
#include "nnpi_ice_caps_hwtrace.h"
#include "nnpi_ice_caps_swtrace.h"
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

#define MAX_TRACE_BUFFER_SIZE (1024 * 1024 * 100)

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

static eIceCapsSwTraceEvent swEventTypes[] = {
    eIceCapsSwTraceEvent::ICE_CAPS_SW_EVENT_CMDLIST,
    eIceCapsSwTraceEvent::ICE_CAPS_SW_EVENT_COPY,
    eIceCapsSwTraceEvent::ICE_CAPS_SW_EVENT_CPYLIST_CREATE,
    eIceCapsSwTraceEvent::ICE_CAPS_SW_EVENT_ICE_DRV,
    eIceCapsSwTraceEvent::ICE_CAPS_SW_EVENT_INFR_SUBRES,
    eIceCapsSwTraceEvent::ICE_CAPS_SW_EVENT_INFR_CREATE,
    eIceCapsSwTraceEvent::ICE_CAPS_SW_EVENT_INFR_REQ,
    eIceCapsSwTraceEvent::ICE_CAPS_SW_EVENT_RUNTIME,
    eIceCapsSwTraceEvent::ICE_CAPS_SW_EVENT_USER_DATA};

NNPITraceContext::NNPITraceContext(unsigned devID)
    : capsSession_(0), devID_(devID), devIDSet_(false) {}

NNPITraceContext::~NNPITraceContext() { destroyInternalContext(); }

bool NNPITraceContext::startCapture(NNPIDeviceContext deviceContext,
                                    bool swTracess, bool hwTraces) {
  if (!createInternalContext(swTracess, hwTraces)) {
    LOG(WARNING) << "nnpi_trace: Failed to create trace device context.";
    return false;
  }

  nnpimlStatus mlStatus = nnpiIceCapsStart(capsSession_);
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
  LOG_NNPI_INF_IF_ERROR(
      nnpiDeviceContextTraceUserData(deviceContext, "EN", getNow()),
      "Failed to inject trace timestamp - device trace may not be "
      "synchronized");
  nnpimlStatus mlStatus = nnpiIceCapsStop(capsSession_);
  if (mlStatus != NNPIML_SUCCESS) {
    return false;
  }
  return true;
}

bool NNPITraceContext::readTraceOutput() {
  nnpimlStatus mlStatus = nnpiIceCapsRead(capsSession_);
  if (mlStatus != NNPIML_SUCCESS) {
    // Failed to read trace.
    LOG(WARNING) << "nnpi_trace: Failed to read traces from device, err="
                 << mlStatus;
    return false;
  }
  mlStatus = nnpiIceCapsParse(capsSession_);
  if (mlStatus != NNPIML_SUCCESS) {
    // Failed to read trace.
    LOG(WARNING) << "nnpi_trace: Failed to parse traces on device, err="
                 << mlStatus;
    return false;
  }

  mlStatus = nnpiIceCapsProcess(capsSession_);
  if (mlStatus != NNPIML_SUCCESS) {
    // Failed to read trace.
    LOG(WARNING) << "nnpi_trace: Failed to process traces on device, err="
                 << mlStatus;
    return false;
  }
  size_t entryCount = 0;
  mlStatus = nnpiIceCapsGetEntriesCount(capsSession_, &entryCount);
  if (mlStatus != NNPIML_SUCCESS) {
    // Failed to read trace.
    LOG(WARNING) << "nnpi_trace: Failed to read traces count, err=" << mlStatus;
    return false;
  }

  bool started = false;
  uint64_t glowStart = 0;
  uint64_t glowEnd = 0;
  uint64_t deviceStart = 0;
  uint64_t deviceEnd = 0;
  uint64_t hostStart = 0;
  uint64_t hostEnd = 0;
  for (size_t i = 0; i < entryCount; i++) {
    IceCapsEntry entry;
    NNPITraceEntry traceEntry;
    std::stringstream entryStrRep;
    mlStatus = nnpiIceCapsGetEntry(capsSession_, i, &entry);
    if (mlStatus != NNPIML_SUCCESS) {
      // Failed to read trace.
      LOG(WARNING) << "nnpi_trace: Failed to read trace entries, err="
                   << mlStatus;
      return false;
    }

    // Set parameters.
    traceEntry.params["name"] = entry.event_name;
    traceEntry.params["state"] = entry.state;
    traceEntry.hostTime = entry.timestamp;
    traceEntry.engineTime = entry.engine_timestamp;
    traceEntry.params["engine"] =
        ((entry.engine == eIceCapsEngine::ICE_CAPS_SW_TRACE)
             ? std::string("SW")
             : std::string("HW"));
    traceEntry.params["event_key"] = std::to_string(entry.event_key);
    traceEntry.params["device_id"] = std::to_string(entry.device_id);
    traceEntry.params["context_id"] = std::to_string(entry.context_id);
    traceEntry.params["network_id"] = std::to_string(entry.network_id);
    traceEntry.params["infer_id"] = std::to_string(entry.infer_id);
    traceEntry.params["ice_id"] = std::to_string(entry.ice_id);
    traceEntry.params["core_id"] = std::to_string(entry.core_id);
    traceEntry.params["network_name"] = entry.network_name;
    traceEntry.params["kernel_name"] = entry.kernel_name;
    traceEntry.params["opcode"] = entry.opcode;

    std::stringstream params;
    for (size_t p = 0; p < entry.params_count; p++) {
      IceCapsParam param;
      mlStatus = nnpiIceCapsGetEntryParam(capsSession_, i, p, &param);
      if (mlStatus != NNPIML_SUCCESS) {
        // Failed to read params.
        LOG(WARNING) << "nnpi_trace: Failed to read trace entry params, err="
                     << mlStatus;
        break;
      }
      traceEntry.params[param.name] = param.value;
      params << param.name << ":" << param.value << ", ";
    }

    if (entry.state == "created" || entry.state == "queued" ||
        entry.state == "req" || entry.state == "add") {
      entry.state = "q";
    } else if (entry.state == "executed" || entry.state == "cbs" ||
               entry.state == "start") {
      entry.state = "s";
    } else if (entry.state == "completed" || entry.state == "cbc") {
      entry.state = "c";
    }
    traceEntry.params["state"] = entry.state;
    entries_.push_back(traceEntry);
    if (entry.event_name == "user_data" &&
        traceEntry.params.count("user_data") > 0 &&
        traceEntry.params.count("key") > 0) {
      if (!started && traceEntry.params["key"] == "BG") {
        glowStart = std::stol(traceEntry.params["user_data"]);
        deviceStart = entry.engine_timestamp;
        hostStart = entry.timestamp;
        started = true;
      } else if (traceEntry.params["key"] == "EN") {
        glowEnd = std::stol(traceEntry.params["user_data"]);
        deviceEnd = entry.engine_timestamp;
        hostEnd = entry.timestamp;
      }
    }
  }
  // Sync clocks:
  if (glowStart > 0 && glowEnd > 0 && hostStart > 0 && hostEnd > 0 &&
      deviceStart > 0 && deviceEnd > 0) {
    // Calculate host time function for host time.
    double hostM =
        (double)(glowEnd - glowStart) / (double)(hostEnd - hostStart);
    double deviceM =
        (double)(glowEnd - glowStart) / (double)(deviceEnd - deviceStart);
    int64_t hostC = glowStart - hostM * hostStart;
    int64_t deviceC = glowStart - deviceM * deviceStart;
    // Update host time.
    for (NNPITraceEntry &entry : entries_) {
      entry.hostTime = entry.hostTime * hostM + hostC;
      entry.engineTime = entry.engineTime * deviceM + deviceC;
    }
  } else {
    LOG(WARNING) << "Failed to synchronize glow and nnpi device traces.";
  }
  return true;
}

bool NNPITraceContext::load() {
  entries_.clear();
  std::stringstream inputStream;

  if (!readTraceOutput()) {
    destroyInternalContext();
    return false;
  }
  destroyInternalContext();
  return true;
}

bool NNPITraceContext::destroyInternalContext() {
  if (capsSession_ == 0) {
    return false;
  }
  nnpimlStatus mlStatus = nnpiIceCapsCloseSession(capsSession_);
  capsSession_ = 0;
  if (mlStatus != NNPIML_SUCCESS) {
    LOG(WARNING) << "nnpi_trace: Failed to stop device trace session, err="
                 << mlStatus;
    capsSession_ = 0;
    return false;
  }

  return true;
}

bool NNPITraceContext::createInternalContext(bool swTraces, bool hwTraces) {
  if (capsSession_ != 0) {
    return false;
  }
  nnpimlStatus mlStatus = nnpiIceCapsOpenSession(&capsSession_);
  if (mlStatus != NNPIML_SUCCESS) {
    LOG(WARNING) << "nnpi_trace: Failed to trace session, err=" << mlStatus;
    capsSession_ = 0;
    return false;
  }
  devMask_ = 1UL << devID_;
  if (swTraces) {
    size_t swEventsCount = sizeof(swEventTypes) / sizeof(swEventTypes[0]);
    size_t idx = 0;
    IceCapsSwTraceConfig traceConfigs[1 + swEventsCount];
    traceConfigs[idx].traceOptions.config_type =
        eIceCapsSwTraceConfigType::ICE_CAPS_SWTRACE_OPTIONS;
    traceConfigs[idx].traceOptions.device_mask = devMask_;
    traceConfigs[idx].traceOptions.max_bytes = MAX_TRACE_BUFFER_SIZE;
    idx++;
    for (size_t i = 0; i < swEventsCount; i++) {
      traceConfigs[idx].traceEvent.config_type =
          eIceCapsSwTraceConfigType::ICE_CAPS_SWTRACE_EVENT;
      traceConfigs[idx].traceEvent.event = swEventTypes[i];
      idx++;
    }

    IceCapsConfig iceSWCapsConfig;
    iceSWCapsConfig.engine = eIceCapsEngine::ICE_CAPS_SW_TRACE;
    iceSWCapsConfig.size = sizeof(traceConfigs);
    iceSWCapsConfig.buffer = traceConfigs;
    mlStatus = nnpiIceCapsPrepare(capsSession_, &iceSWCapsConfig);
    if (mlStatus != NNPIML_SUCCESS) {
      LOG(WARNING)
          << "nnpi_trace: Failed to set device Software trace options, err="
          << mlStatus;
      destroyInternalContext();
      return false;
    }
  }
  if (hwTraces) {
    IceCapsHwTraceConfig traceConfigs[2];
    traceConfigs[0].traceOptions.config_type =
        eIceCapsHwTraceConfigType::ICE_CAPS_HWTRACE_OPTIONS;
    traceConfigs[0].traceOptions.device_mask = devMask_;
    traceConfigs[0].traceOptions.max_trace_size = MAX_TRACE_BUFFER_SIZE;
    traceConfigs[1].iceFilter.config_type =
        eIceCapsHwTraceConfigType::ICE_CAPS_HWTRACE_FILTER;
    traceConfigs[1].iceFilter.ice_mask = 0xFFF; // All ICEs.
    traceConfigs[1].iceFilter.filter_type =
        eIceCapsHwTraceFilter::ICE_CAPS_HWTRACE_CAPTURE_ALL;

    IceCapsConfig iceHWCapsConfig;
    iceHWCapsConfig.engine = eIceCapsEngine::ICE_CAPS_HW_TRACE;
    iceHWCapsConfig.size = sizeof(traceConfigs);
    iceHWCapsConfig.buffer = traceConfigs;

    mlStatus = nnpiIceCapsPrepare(capsSession_, &iceHWCapsConfig);
    if (mlStatus != NNPIML_SUCCESS) {
      LOG(WARNING)
          << "nnpi_trace: Failed to set device Hardware trace options, err="
          << mlStatus;
      destroyInternalContext();
      return false;
    }
  }

  return true;
}
