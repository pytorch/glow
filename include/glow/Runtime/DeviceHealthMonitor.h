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
#ifndef GLOW_RUNTIME_HEALTHMONITOR_H
#define GLOW_RUNTIME_HEALTHMONITOR_H

#include <memory>
#include <vector>

namespace glow {

/// Interface for exporting runtime statistics.  The base implementation
/// delegates to any subclass registered via `registerStatsExporter`.
class DeviceHealthMonitor {
public:
  /// Dtor.
  virtual ~DeviceHealthMonitor() = default;

  /// Start monitoring the device health
  virtual void start() = 0;
};

/// Registry of StatsExporters.
class DeviceHealthMonitorRegistry final {
public:
  /// Start all the device health monitors
  void start() {
    for (auto *monitor : monitors_) {
      monitor->start();
    }
  }

  /// Register a DeviceHealthMonitor.
  void registerDeviceHealthMonitor(DeviceHealthMonitor *monitor);

  /// Revoke a DeviceHealthMonitor.
  void revokeDeviceHealthMonitor(DeviceHealthMonitor *exporter);

  /// Static singleton DeviceHealthMonitor.
  static std::shared_ptr<DeviceHealthMonitorRegistry> Monitors();

private:
  /// Registered StatsExporters.
  std::vector<DeviceHealthMonitor *> monitors_;
};

} // namespace glow

#endif // GLOW_RUNTIME_HEALTHMONITOR_H
