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

#include "glow/Runtime/DeviceHealthMonitor.h"

#include <algorithm>

namespace glow {

void DeviceHealthMonitorRegistry::registerDeviceHealthMonitor(
    DeviceHealthMonitor *m) {
  monitors_.push_back(m);
}

void DeviceHealthMonitorRegistry::revokeDeviceHealthMonitor(
    DeviceHealthMonitor *m) {
  monitors_.erase(std::remove(monitors_.begin(), monitors_.end(), m),
                  monitors_.end());
}

std::shared_ptr<DeviceHealthMonitorRegistry>
DeviceHealthMonitorRegistry::Monitors() {
  static auto monitors = std::make_shared<DeviceHealthMonitorRegistry>();
  return monitors;
}

} // namespace glow
