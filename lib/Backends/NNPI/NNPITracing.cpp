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

NNPIDeviceTracing::NNPIDeviceTracing(unsigned deviceID) {
  traceCtx_ = glow::make_unique<NNPITraceContext>(deviceID);
  deviceInfo_ =
      std::string("[Device #") + std::to_string(deviceID) + std::string("] ");
}

bool NNPIDeviceTracing::start(TraceContext *traceContext,
                              NNPIDeviceContext deviceContext) {
  if (!traceContext ||
      !traceContext->shouldLog(TraceEvent::TraceLevel::OPERATOR)) {
    return false;
  }
  if (started_.test_and_set()) {
    ASSERT_WITH_MSG(glowTraceCtx_ != traceContext,
                    "Trying to start tracing for an already started context.");
    // Trace already started.
    return false;
  }
  glowTraceCtx_ = traceContext;
  if (!traceCtx_->startCapture(deviceContext)) {
    LOG(WARNING) << "Failed to start trace capture";
    return false;
  }
  return true;
}

std::string NNPIDeviceTracing::getEntryName(NNPITraceEntry &entry) {
  std::stringstream name;
  name << deviceInfo_;
  switch (entry.traceType) {
  case NNPI_TRACE_UNKNOWN:
    name << "UnknownTrace";
    break;
  case NNPI_TRACE_DMA:
    name << "DMA";
    break;
  case NNPI_TRACE_INFER:
    name << "Infer";
    break;
  case NNPI_TRACE_COPY:
    name << "Copy";
    break;
  case NNPI_TRACE_MARK:
    name << "MarkTrace";
    break;
  case NNPI_TRACE_CLOCK_SYNC:
    name << "ClockSync";
    break;
  case NNPI_TRACE_CMDLIST:
    name << "CommandList";
    break;
  case NNPI_TRACE_NETEXEC:
    name << "NetExecute";
    break;
  case NNPI_TRACE_SUBGRAPH:
    name << "SubGraph";
    break;
  case NNPI_TRACE_RUNTIME_INFER:
    name << "RunTimeInf";
    break;
  case NNPI_TRACE_ICED_SCHED_JOB:
    name << "DSchedJob";
    break;
  case NNPI_TARCE_ICED_CREAT_NET:
    name << "DCreateNet";
    break;
  case NNPI_TARCE_ICED_NET_RES:
    name << "DNetRes";
    break;
  case NNPI_TARCE_ICED_NET_GEN:
    name << "DNetGen";
    break;
  default:
    name << "Othertrace";
  }
  if (entry.params.count("isC2H") > 0) {
    if (entry.params["isC2H"] == "1") {
      name << "-Card2Host";
    } else {
      name << "-Host2Card";
    }
  }
  auto params = entry.params;
  if (entry.params.count("iceId") > 0) {
    name << "-ICE_" << entry.params["iceId"];
  }
  if (entry.params.count("netID") > 0) {
    name << "-NET_" << entry.params["netID"];
  }
  if (entry.params.count("reqID") > 0) {
    name << "REQ_" << entry.params["reqID"];
  }
  if (entry.params.count("ctxID") > 0) {
    name << "-CTX_" << entry.params["ctxID"];
  }
  if (entry.params.count("subNetId") > 0) {
    name << "-SUBNET_" << entry.params["subNetId"];
  }
  if (entry.params.count("inferID") > 0) {
    name << "-INFR_" << entry.params["inferID"];
  }
  if (entry.params.count("subGraphID") > 0) {
    name << "-SUBGRAPH_" << entry.params["subGraphID"];
  }
  if (entry.params.count("agent") > 0) {
    name << "-AGENT_" << entry.params["agent"];
  }
  if (entry.params.count("copyID") > 0) {
    name << "-CPID_" << entry.params["copyID"];
  }
  if (entry.params.count("size") > 0) {
    name << "-SIZE_" << entry.params["size"];
  }
  return name.str();
}

bool NNPIDeviceTracing::addTrace(NNPITraceEntry &entry) {
  // Filter traces.
  switch (entry.traceType) {
  case NNPI_TRACE_INFER:
  case NNPI_TRACE_COPY:
  case NNPI_TRACE_CMDLIST:
  case NNPI_TRACE_NETEXEC:
  case NNPI_TRACE_SUBGRAPH:
  case NNPI_TRACE_RUNTIME_INFER:
  case NNPI_TRACE_ICED_SCHED_JOB:
  case NNPI_TARCE_ICED_CREAT_NET:
  case NNPI_TARCE_ICED_NET_RES:
  case NNPI_TARCE_ICED_NET_GEN:
    break;
  case NNPI_TRACE_UNKNOWN:
  case NNPI_TRACE_DMA:
  case NNPI_TRACE_MARK:
  case NNPI_TRACE_CLOCK_SYNC:
  case NNPI_TARCE_TIME_SYNC:
  case NNPI_TARCE_USER_DATA:
    return false;
  default:
    LOG(WARNING) << "Trying to add unsupported trace type:" << entry.traceType;
    return false;
  }

  std::string name = getEntryName(entry);

  if (entry.params.count("state") <= 0) {
    return false;
  }
  std::string state = entry.params["state"];

  if (state == "q" || state == "queued") {
    name += "-Queue";
    glowTraceCtx_->logTraceEvent(name, TraceLevel::OPERATOR,
                                 TraceEvent::InstantType, entry.hostTime, {});
  } else if (state == "s" || state == "cbs" || state == "executed") {
    glowTraceCtx_->logTraceEvent(name, TraceLevel::OPERATOR,
                                 TraceEvent::BeginType, entry.hostTime, {});
  } else if (state == "c" || state == "cbc" || state == "completed") {
    glowTraceCtx_->logTraceEvent(name, TraceLevel::OPERATOR,
                                 TraceEvent::EndType, entry.hostTime, {});
  } else if (state == "cbs") {
    glowTraceCtx_->logTraceEvent(name, TraceLevel::OPERATOR,
                                 TraceEvent::BeginType, entry.hostTime, {});
  } else if (state == "cbc") {
    glowTraceCtx_->logTraceEvent(name, TraceLevel::OPERATOR,
                                 TraceEvent::EndType, entry.hostTime, {});
  } else if (state == "cbnwc") {
    glowTraceCtx_->logTraceEvent(name, TraceLevel::OPERATOR,
                                 TraceEvent::InstantType, entry.hostTime, {});
  } else if (state == "req") {
    name += "-Req";
    glowTraceCtx_->logTraceEvent(name, TraceLevel::OPERATOR,
                                 TraceEvent::InstantType, entry.hostTime, {});
  }

  return true;
}

bool NNPIDeviceTracing::stopAndUpdate(TraceContext *traceContext,
                                      NNPIDeviceContext deviceContext) {
  if (glowTraceCtx_ !=
          nullptr && // For null glowTraceCtx assume global context (per device)
      (glowTraceCtx_ != traceContext)) {
    // Ignore stop from other contexts.
    return false;
  }
  if (!traceCtx_->stopCapture(deviceContext)) {
    LOG(WARNING) << "Failed to stop trace capture";
    return false;
  }

  if (!traceCtx_->load()) {
    LOG(WARNING) << "Failed to stop trace capture";
    return false;
  }
  traceContext->setThreadName("NNPI_Trace");
  for (auto entry : traceCtx_->getEntries()) {
    std::map<std::string, std::string> params = entry.params;
    addTrace(entry);
  }
  started_.clear();
  return true;
}
