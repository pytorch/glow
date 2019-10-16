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

#include <fstream>
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

    // As a fallback to make developing new Backends easier we'll create a
    // DummyDeviceManager here, but this is not threadsafe and very simplistic.
    // Strongly recommended that you create a DeviceManager customized for your
    // device.
    LOG(ERROR) << "Warning: Creating a DummyDeviceManager.\n";
    return new DummyDeviceManager(config);
  }

  return backend->createDeviceManager(config);
}

#if defined(GLOW_WITH_CPU)
unsigned numCPUDevices() { return std::thread::hardware_concurrency(); }
#else
unsigned numCPUDevices() { return 0; }
#endif

#if defined(GLOW_WITH_NNPI)
unsigned numNNPIDevices() {
  // TODO: unify with numHabanaDevices. copy-pasta with a different device name
  std::ifstream devices("/proc/bus/pci/devices");
  std::string device;
  unsigned count = 0;
  while (std::getline(devices, device)) {
    if (device.find("sph_pcie") != std::string::npos) {
      count++;
    }
  }
  if (count > 0) {
    return count;
  }
  // Todo
  return 1; // Fall back to emulator since GLOW_NNPI is set. This feels hacky.
}
#else
unsigned numNNPIDevices() { return 0; }
#endif

#if defined(GLOW_WITH_HABANA)
unsigned numHabanaDevices() {
  std::ifstream devices("/proc/bus/pci/devices");
  std::string device;
  unsigned count = 0;
  while (std::getline(devices, device)) {
    if (device.find("habanalabs") != std::string::npos) {
      count++;
    }
  }
  return count;
}
#else
unsigned numHabanaDevices() { return 0; }
#endif

#if defined(GLOW_WITH_OPENCL)
unsigned numOpenCLDevices() { return 1; }
#else
unsigned numOpenCLDevices() { return 0; }
#endif

unsigned DeviceManager::numDevices(llvm::StringRef backendName) {
  if (backendName == "Interpreter") {
    return std::thread::hardware_concurrency();
  }
  if (backendName == "CPU") {
    return numCPUDevices();
  }
  if (backendName == "Habana") {
    return numHabanaDevices();
  }
  if (backendName == "OpenCL") {
    return numOpenCLDevices();
  }
  if (backendName == "NNPI") {
    return numNNPIDevices();
  }
  return 0;
}

std::vector<std::unique_ptr<runtime::DeviceConfig>>
DeviceManager::generateDeviceConfigs(llvm::StringRef backendName) {
  std::vector<std::unique_ptr<runtime::DeviceConfig>> configs;
  auto deviceCount = numDevices(backendName);
  for (int i = 0; i < deviceCount; i++) {
    configs.push_back(llvm::make_unique<runtime::DeviceConfig>(backendName));
  }
  return configs;
}

} // namespace runtime
} // namespace glow
