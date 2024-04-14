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

#include "NNPITracing.h"
#include "DebugMacros.h"
#include "NNPI.h"
#include "NNPIMLTraceWrapper.h"
#include <unordered_map>

using namespace glow;

std::map<std::string, int> NNPIDeviceTracing::activeAffinities_ = {};

NNPIDeviceTracing::NNPIDeviceTracing(unsigned deviceId) : deviceId_(deviceId) {
  traceCtx_ = glow::make_unique<NNPITraceContext>(deviceId_);
  deviceInfo_ =
      std::string("[Device #") + std::to_string(deviceId_) + std::string("] ");
}

bool NNPIDeviceTracing::start(TraceContext *traceContext,
                              NNPIDeviceContext deviceContext, bool swTraces,
                              bool hwTraces, uint32_t softwareBufferSizeMB,
                              uint32_t hardwareBufferSizeMB,
                              const std::string &dumpRawEventsPath) {
  if (!traceContext ||
      !traceContext->shouldLog(TraceEvent::TraceLevel::OPERATOR)) {
    return false;
  }
  if (started_.test_and_set()) {
    // Trace already started.
    return false;
  }
  bool isFirstToStart = NNPIDeviceTracing::isFirstToChangeCaptureStart(true);
  if (!traceCtx_->startCapture(deviceContext, swTraces, hwTraces,
                               softwareBufferSizeMB, hardwareBufferSizeMB,
                               dumpRawEventsPath)) {
    LOG(WARNING) << "Failed to start trace capture for device " << deviceId_
                 << " is first = " << (isFirstToStart);
    return false;
  }
  return true;
}

std::string NNPIDeviceTracing::getEntryName(NNPITraceEntry &entry) {
  std::string entryName = entry.params["name"];
  if (entryName.rfind("icedrv", 0) == 0) {
    entryName = entryName.substr(strlen("icedrv"));
  } else if (entryName.rfind("runtime-", 0) == 0) {
    entryName = entryName.substr(strlen("runtime-"));
  }
  if (entry.params.count("command") > 0) {
    entryName = entry.params["command"];
  }

  std::stringstream name;

  name << entryName;
  if (entry.params.count("isC2H") > 0) {
    if (entry.params["isC2H"] == "1") {
      name << " Card2Host";
    } else {
      name << " Host2Card";
    }
  }
  auto params = entry.params;
  if (entry.params.count("context_id") > 0) {
    name << " CTX 0x" << std::hex << std::stol(entry.params["context_id"]);
  }
  if (entry.params.count("ice_id") > 0) {
    name << " ICE_" << entry.params["ice_id"];
  }
  if (entry.params.count("core_id") > 0) {
    name << " CORE_" << entry.params["core_id"];
  }
  if (entry.params.count("network_id") > 0) {
    name << " Net " << entry.params["network_id"];
  }
  if (entry.params.count("network_name") > 0 &&
      entry.params["network_name"] != "NA") {
    name << " NetName " << entry.params["network_name"];
  }
  if (entry.params.count("subNetId") > 0) {
    name << " Subnet " << entry.params["subNetId"];
  }
  if (entry.params.count("infer_id") > 0) {
    name << " InfID " << entry.params["infer_id"];
  }
  if (entry.params.count("subGraphID") > 0) {
    name << " Subgraph " << entry.params["subGraphID"];
  }
  if (entry.params.count("agent") > 0) {
    name << " Agent " << entry.params["agent"];
  }
  if (entry.params.count("kernel_name") > 0 &&
      entry.params["kernel_name"] != "NA") {
    name << " Krnl " << entry.params["kernel_name"];
  }
  if (entry.params.count("userHandle") > 0) {
    name << " 0x" << std::hex << std::stol(entry.params["userHandle"]);
  }

  return name.str();
}

int NNPIDeviceTracing::getAffinityID(NNPITraceEntry &entry, std::string name,
                                     unsigned deviceId,
                                     TraceContext *traceContext) {
  // Need to be guarded when multiple devices are active.
  static std::mutex affinityMutext;
  std::lock_guard<std::mutex> lk(affinityMutext);

  // Start affinity at some high number to avoid collisions.
  int affinId = 10000;

  std::string contextId = entry.params["context_id"];
  std::stringstream affinityNameStuct;

  affinityNameStuct << "Device #" << deviceId;
  if (entry.params.count("ice_id")) {
    std::string iceId = entry.params["ice_id"];
    affinityNameStuct << " ICE #" << iceId;
  }

  // Add additional info to title.
  if (entry.params.count("opcode") > 0 && entry.params["opcode"] != "NA") {
    affinityNameStuct << " opcode " << entry.params["opcode"];
  }
  // Use the op name.
  affinityNameStuct << " " << name.substr(0, name.find(' '));
  if (entry.params.count("org_state") &&
      (entry.params["org_state"] == "q" ||
       entry.params["org_state"] == "queued")) {
    affinityNameStuct << " Queue";
  }

  if (activeAffinities_.count(affinityNameStuct.str()) <= 0) {
    affinId += activeAffinities_.size();
    activeAffinities_[affinityNameStuct.str()] = affinId;
    traceContext->setThreadName(affinId, affinityNameStuct.str());
  } else {
    affinId = activeAffinities_[affinityNameStuct.str()];
  }

  return affinId;
}

bool NNPIDeviceTracing::addTrace(
    NNPITraceEntry &entry, std::map<std::string, NNPITraceEntry> &inflight,
    TraceContext *traceContext) {
  std::stringstream entryLog;
  for (auto const &paramEntry : entry.params) {
    entryLog << paramEntry.first << ":" << paramEntry.second << " ,";
  }
  // Filter traces.
  if (entry.params.count("state") <= 0 || entry.params["state"] == "NA") {
    return false;
  }
  std::string name = getEntryName(entry);

  if (entry.params.count("state") <= 0) {
    return false;
  }

  std::string eventKey = name;
  if (entry.params.count("event_key") > 0) {
    eventKey += (std::string(" : ") + entry.params["event_key"]);
  }
  std::string state = entry.params["state"];

  // Calculate affinity - use the trace thread id to make sections in the
  // representation.
  int affinId =
      NNPIDeviceTracing::getAffinityID(entry, name, deviceId_, traceContext);
  if (affinId <= 0) {
    LOG(WARNING) << "Found unexpected affinity ID " << affinId << " for "
                 << name;
  }
  // Add events.
  if (state == "q") {
    traceContext->logTraceEvent(name, TraceLevel::OPERATOR,
                                TraceEvent::InstantType, entry.hostTime,
                                entry.params, affinId);
  } else if (state == "s" && inflight.count(eventKey) <= 0) {
    inflight[eventKey] = entry;
  } else if (state == "c" && inflight.count(eventKey) > 0) {
    // Add only complate events.
    if (entry.hostTime >= inflight[eventKey].hostTime) {
      traceContext->logTraceEvent(
          name, TraceLevel::OPERATOR, TraceEvent::BeginType,
          inflight[eventKey].hostTime, inflight[eventKey].params, affinId);
      traceContext->logTraceEvent(name, TraceLevel::OPERATOR,
                                  TraceEvent::EndType, entry.hostTime,
                                  entry.params, affinId);
    } else {
      LOG(WARNING) << "[INCOMPLETE EVENT] Found incomplete trace event "
                   << eventKey << ": start time " << inflight[eventKey].hostTime
                   << " end time " << entry.hostTime;
    }
    inflight.erase(eventKey);
  } else if (state == "po") {
    traceContext->logTraceEvent(name, TraceLevel::OPERATOR,
                                TraceEvent::InstantType, entry.hostTime,
                                entry.params, affinId);
  } else if (entry.params.count("engine") > 0 &&
             entry.params["engine"] != "HW") {
    // Notifies only software events that are incomplete since HW events are
    // much more likely to be lost.
    LOG(WARNING) << "[INCOMPLETE EVENT] " << " event key:" << eventKey
                 << " state:" << state
                 << " inflight: " << (inflight.count(eventKey) > 0)
                 << " time: " << entry.hostTime;
  }

  return true;
}

bool NNPIDeviceTracing::stopAndUpdate(TraceContext *traceContext,
                                      NNPIDeviceContext deviceContext) {
  if (traceContext == nullptr) {
    LOG(WARNING) << "Failed to stop trace capture trace context is null.";
    return false;
  }
  bool isFirstToStop = NNPIDeviceTracing::isFirstToChangeCaptureStart(false);
  if (!traceCtx_->stopCapture(deviceContext)) {
    LOG(WARNING) << "Failed to stop trace capture (first device stop ="
                 << isFirstToStop;
    return false;
  }

  if (!traceCtx_->load()) {
    LOG(WARNING) << "Failed to stop trace capture =" << isFirstToStop;
    return false;
  }
  traceContext->setThreadName("NNPI_Trace");
  std::map<std::string, NNPITraceEntry> inflight;
  for (auto entry : traceCtx_->getEntries()) {
    addTrace(entry, inflight, traceContext);
  }
  if (inflight.size() > 0) {
    LOG(WARNING) << "[INCOMPLETE EVENT] " << inflight.size()
                 << " events not logged (still in flight/incomplate)";
    for (const auto &event : inflight) {
      if (event.second.params.at("engine") != "HW") {
        // Notifies only software events that are incomplete since HW events are
        // much more likely to be lost.
        LOG(WARNING) << "[INCOMPLETE EVENT] " << event.first << " "
                     << event.second.params.at("name")
                     << " state:" << event.second.params.at("state");
      }
    }
  }
  started_.clear();
  return true;
}
