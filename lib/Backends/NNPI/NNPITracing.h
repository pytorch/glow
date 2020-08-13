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

#ifndef GLOW_BACKENBDS_NNPI_NNPITRACING_H
#define GLOW_BACKENBDS_NNPI_NNPITRACING_H

#include "NNPIMLTraceWrapper.h"
#include "glow/Runtime/RuntimeTypes.h"
#include <memory>
#include <unordered_map>

#define TRACING_BACKEND_EXECUTE "backend execute"
#define TRACING_PRE_PROCESS "pre process"
#define TRACING_INFERENCE "host inference"
#define TRACING_POST_PROCESS "post process"

namespace glow {

class NNPIDeviceTracing {
public:
  static std::shared_ptr<NNPIDeviceTracing> getForDevice(unsigned deviceId) {
    static std::unordered_map<uint32_t, std::shared_ptr<NNPIDeviceTracing>> map;
    static std::mutex mapSyncMutex;
    std::lock_guard<std::mutex> lk(mapSyncMutex);
    if (map.count(deviceId) <= 0) {
      // Stub to allow make_shared access to private constructor.
      struct EnabledShare : public NNPIDeviceTracing {
        EnabledShare(uint32_t deviceId) : NNPIDeviceTracing(deviceId) {}
      };
      map[deviceId] = std::make_shared<EnabledShare>(deviceId);
    }
    return map[deviceId];
  }

  static bool isFirstToChangeCaptureStart(bool startCapture) {
    static bool started = false;
    static std::mutex firstDevStartMutex;
    std::lock_guard<std::mutex> lk(firstDevStartMutex);
    if (started != startCapture) {
      // First to change state.
      started = startCapture;
      return true;
    }

    return false;
  }

  /// Dispose of tracing context.
  virtual ~NNPIDeviceTracing(){};

  /// Start recording events.
  bool start(TraceContext *traceContext, NNPIDeviceContext deviceContext,
             bool swTraces, bool hwTraces, uint32_t softwareBufferSizeMB,
             uint32_t hardwareBufferSizeMB,
             const std::string &dumpRawEventsPath);
  /// Stop recording, read and update trace context.
  bool stopAndUpdate(TraceContext *traceContext,
                     NNPIDeviceContext deviceContext);

protected:
  std::string getEntryName(NNPITraceEntry &entry);
  bool addTrace(NNPITraceEntry &entry,
                std::map<std::string, NNPITraceEntry> &inflight,
                TraceContext *traceContext);

  /// Affinity has to be in a global for all devices.
  static int getAffinityID(NNPITraceEntry &entry, std::string name,
                           unsigned deviceId, TraceContext *traceContext);

private:
  /// Per device tracing control.
  explicit NNPIDeviceTracing(unsigned deviceId);
  std::atomic_flag started_{false};
  /// NNPI Trace context.
  std::unique_ptr<NNPITraceContext> traceCtx_;
  /// Device id.
  unsigned deviceId_{0};
  /// Device id string prefix for event names.
  std::string deviceInfo_;

  /// Trace active affinities.
  static std::map<std::string, int> activeAffinities_;
};

} // namespace glow
#endif // GLOW_BACKENBDS_NNPI_NNPITRACING_H
