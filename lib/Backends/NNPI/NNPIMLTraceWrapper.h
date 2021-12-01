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
#include <nnpi_ice_caps.h>
#include <nnpi_inference.h>
#include <string>
#include <vector>

struct NNPITraceEntry {
  uint64_t engineTime{0};
  uint64_t hostTime{0};
  std::map<std::string, std::string> params;
};

/// Device trace api wrapper.
class NNPITraceContext {
public:
  NNPITraceContext(unsigned devID);
  virtual ~NNPITraceContext();
  /// Start capturing traces from the HW device.
  bool startCapture(NNPIDeviceContext deviceContext, bool swTraces,
                    bool hwTraces, uint32_t softwareBufferSizeMB,
                    uint32_t hardwareBufferSizeMB,
                    const std::string &dumpRawEventsPath);
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
  bool createInternalContext(bool swTraces, bool hwTraces,
                             uint32_t softwareBufferSizeMB,
                             uint32_t hardwareBufferSizeMB);
  bool readTraceOutput();

  IceCaps_t capsSession_{0};
  uint64_t devMask_{0};
  unsigned devID_{0};
  bool devIDSet_{false};
  std::vector<NNPITraceEntry> entries_;
  std::string dumpRawEventsPath_;
};

#endif // NNPI_NNPITRACING_ML_WRAPPER_H
