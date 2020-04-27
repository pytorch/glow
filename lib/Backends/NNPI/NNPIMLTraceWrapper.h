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

#ifndef NNPI_NNPITRACING_ML_WRAPPER_H
#define NNPI_NNPITRACING_ML_WRAPPER_H

#include <map>
#include <nnpi_inference.h>
#include <nnpiml.h>
#include <vector>

enum NNPITraceType {
  NNPI_TRACE_UNKNOWN = 0x0000,
  NNPI_TRACE_DMA = 0x0001,
  NNPI_TRACE_INFER = 0x0002,
  NNPI_TRACE_COPY = 0x0004,
  NNPI_TRACE_MARK = 0x0008,
  NNPI_TRACE_CLOCK_SYNC = 0x0010,
  NNPI_TRACE_CMDLIST = 0x0020,
  NNPI_TRACE_NETEXEC = 0x0040,
  NNPI_TRACE_SUBGRAPH = 0x0080,
  NNPI_TARCE_TIME_SYNC = 0x0100,
  NNPI_TRACE_RUNTIME_INFER = 0x0200,
  NNPI_TRACE_ICED_SCHED_JOB = 0x0400,
  NNPI_TARCE_ICED_CREAT_NET = 0x0800,
  NNPI_TARCE_ICED_NET_RES = 0x1000,
  NNPI_TARCE_ICED_NET_GEN = 0x1001,
  NNPI_TARCE_USER_DATA = 0x4000,
  NNPI_TRACE_OTHER = 0x8000
};

struct NNPITraceEntry {
  uint64_t deviceUpTime{0};
  uint64_t hostTime{0};
  NNPITraceType traceType{NNPI_TRACE_UNKNOWN};
  uint32_t processID{0};
  uint32_t cpuID{0};
  char flags_[4];
  std::map<std::string, std::string> params;
};

/// Device trace api wrapper.
class NNPITraceContext {
public:
  NNPITraceContext(unsigned devID);
  virtual ~NNPITraceContext();
  /// Start capturing traces from the HW device.
  bool startCapture(NNPIDeviceContext deviceContext);
  /// Start capturing.
  bool stopCapture(NNPIDeviceContext deviceContext) const;
  /// Load traces (valid only after stopCapture()).
  bool load();
  /// Returns the number of traces captured and loaded (valid only after
  /// load()).
  size_t getTraceCount() const { return entries_.size(); }
  /// Read a loaded entry by index.
  NNPITraceEntry &getEntry(int index) { return entries_[index]; }
  /// Get the context device ID.
  uint32_t getDeviceID() const { return devID_; }
  /// Returns true if device ID was set, false otherwise.
  bool isDeviceIDSet() const { return devIDSet_; }
  /// Get a vector of the loaded entries (valid only after load()).
  std::vector<NNPITraceEntry> getEntries() const { return entries_; }

private:
  bool destroyInternalContext();
  bool createInternalContext();
  bool readTraceOutput(std::stringstream &inputStream);

  nnpimlTraceContext traceCtx_{0};
  uint64_t devMask_{0};
  unsigned devID_{0};
  bool devIDSet_{false};
  std::string events_;
  std::vector<NNPITraceEntry> entries_;
};

#endif // NNPI_NNPITRACING_ML_WRAPPER_H
