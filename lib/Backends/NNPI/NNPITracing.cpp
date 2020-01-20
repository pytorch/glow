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

NNPIDeviceTracing::NNPIDeviceTracing(uint32_t deviceID) : deviceID_(deviceID) {
  traceCtx_ = std::make_unique<NNPITraceContext>(0);
}

void NNPIDeviceTracing::start(TraceContext *traceContext,
                              runtime::RunIdentifierTy runId) {
  if (!traceContext ||
      !traceContext->shouldLog(TraceEvent::TraceLevel::OPERATOR)) {
    return;
  }
  if (started_.test_and_set()) {
    ASSERT_WITH_MSG(glowTraceCtx_ != traceContext,
                    "Trying to start tracing for an already started context.");
    // Trace already started.
    return;
  }
  glowTraceCtx_ = traceContext;
  runId_ = runId;
  if (!traceCtx_->startCapture()) {
    LOG(WARNING) << "Failed to start trace capture";
  }
}

std::string NNPIDeviceTracing::getEntryName(NNPITraceEntry &entry) {
  std::stringstream name;
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
  if (entry.params.count("ctxID") > 0) {
    name << "-" << entry.params["ctxID"];
  }
  if (entry.params.count("copyID") > 0) {
    name << "-" << entry.params["copyID"];
  }
  if (entry.params.count("size") > 0) {
    name << "-" << entry.params["size"];
  }
  return name.str();
}

bool NNPIDeviceTracing::addTrace(NNPITraceEntry &entry) {
  // Filter traces.
  switch (entry.traceType) {
  case NNPI_TRACE_UNKNOWN:
    return false;
  case NNPI_TRACE_DMA:
    return false;
  case NNPI_TRACE_INFER:
    break;
  case NNPI_TRACE_COPY:
    break;
  case NNPI_TRACE_MARK:
    return false;
  case NNPI_TRACE_CLOCK_SYNC:
    return false;
  default:
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
                                 TraceEvent::InstantType, entry.hostTime,
                                 entry.params);
  } else if (state == "s" || state == "cbs" || state == "executed") {
    glowTraceCtx_->logTraceEvent(name, TraceLevel::OPERATOR,
                                 TraceEvent::BeginType, entry.hostTime,
                                 entry.params);
  } else if (state == "c" || state == "cbc" || state == "completed") {
    glowTraceCtx_->logTraceEvent(name, TraceLevel::OPERATOR,
                                 TraceEvent::EndType, entry.hostTime,
                                 entry.params);
  } else if (state == "cbs") {
    glowTraceCtx_->logTraceEvent(name, TraceLevel::OPERATOR,
                                 TraceEvent::BeginType, entry.hostTime,
                                 entry.params);
  } else if (state == "cbc") {
    glowTraceCtx_->logTraceEvent(name, TraceLevel::OPERATOR,
                                 TraceEvent::EndType, entry.hostTime,
                                 entry.params);
  } else if (state == "cbnwc") {
    glowTraceCtx_->logTraceEvent(name, TraceLevel::OPERATOR,
                                 TraceEvent::InstantType, entry.hostTime,
                                 entry.params);
  }
  return true;
}

void NNPIDeviceTracing::stopAndUpdate(TraceContext *traceContext,
                                      runtime::RunIdentifierTy runId) {
  if (glowTraceCtx_ != traceContext || runId_ != runId) {
    // Ignore stop from other contexts.
    return;
  }
  if (!traceCtx_->stopCapture()) {
    LOG(WARNING) << "Failed to stop trace capture";
    return;
  }

  if (!traceCtx_->load()) {
    LOG(WARNING) << "Failed to stop trace capture";
    return;
  }
  for (auto entry : traceCtx_->getEntries()) {
    std::map<std::string, std::string> params = entry.params;
    addTrace(entry);
  }
  started_.clear();
}

void NNPIDeviceTracing::startCopyTime() {
  if (traceCtx_) {
    traceCtx_->markInputCopyStart(TraceEvent::now());
  }
}
