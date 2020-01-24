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

#include "glow/Backends/DeviceManager.h"
#include "glow/Backends/DummyDeviceManager.h"
#include "glow/Support/Register.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <thread>

#include <glog/logging.h>

using namespace glow;
using namespace glow::runtime;

namespace glow {
namespace runtime {

DeviceManager *DeviceManager::createDeviceManager(const DeviceConfig &config) {
  std::unique_ptr<Backend> backend(
      FactoryRegistry<std::string, Backend>::get(config.backendName));

  if (backend == nullptr) {
    LOG(ERROR) << "There is no registered backend by name: "
               << config.backendName;
    LOG(ERROR) << "List of all registered backends: ";
    for (const auto &factory :
         FactoryRegistry<std::string, Backend>::factories()) {
      LOG(ERROR) << factory.first;
    }

    // As a fallback to make developing new backends easier we'll create a
    // DummyDeviceManager here, but this is not threadsafe and very simplistic.
    // Strongly recommended that you create a DeviceManager customized for your
    // device.
    LOG(ERROR) << "Warning: Creating a DummyDeviceManager.\n";
    return new DummyDeviceManager(config);
  }

  return backend->createDeviceManager(config);
}

unsigned DeviceManager::numDevices(llvm::StringRef backendName) {
  const auto &factories = FactoryRegistry<std::string, Backend>::factories();
  auto it = factories.find(backendName);
  if (it == factories.end()) {
    return 0;
  } else {
    return it->second->numDevices();
  }
}

std::vector<std::unique_ptr<runtime::DeviceConfig>>
DeviceManager::generateDeviceConfigs(llvm::StringRef backendName) {
  std::vector<std::unique_ptr<runtime::DeviceConfig>> configs;
  auto deviceCount = numDevices(backendName);
  for (unsigned i = 0; i < deviceCount; i++) {
    auto config = glow::make_unique<runtime::DeviceConfig>(backendName);
    config->deviceID = i;
    configs.push_back(std::move(config));
  }
  return configs;
}

} // namespace runtime
} // namespace glow
