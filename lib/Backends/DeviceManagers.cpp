/**
 * Copyright (c) 2017-present, Facebook, Inc.
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
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace glow;
using namespace glow::runtime;

namespace glow {
namespace runtime {
/// NOTE: Please add a declaration of a device-specific `create` method here
/// when you define a new DeviceManager.

/// Create a new instance of the interpreter Device.
DeviceManager *createInterpreterDeviceManager(const DeviceConfig &config);

#if defined(GLOW_WITH_CPU)
/// Create a new instance of the CPUBackend DeviceManager.
DeviceManager *createCPUDeviceManager(const DeviceConfig &config);
#else
DeviceManager *createCPUDeviceManager(const DeviceConfig &config) {
  (void)config;
  LOG(FATAL) << "Must compile with CPU support";
}
#endif

#if defined(GLOW_WITH_OPENCL)
/// Create a new instance of the OpenCL backend.
DeviceManager *createOCLDeviceManager(const DeviceConfig &config);
#else
DeviceManager *createOCLDeviceManager(const DeviceConfig &config) {
  (void)config;
  LOG(FATAL) << "Must compile with OpenCL support";
}
#endif

#if defined(GLOW_WITH_HABANA)
DeviceManager *createHabanaDeviceManager(const DeviceConfig &config);
#else
DeviceManager *createHabanaDeviceManager(const DeviceConfig &config) {
  (void)config;
  LOG(FATAL) << "Must compile with Habana support";
}
#endif
} // namespace runtime
} // namespace glow

DeviceManager *DeviceManager::createDeviceManager(const DeviceConfig &config) {
  if (config.backendName == "Interpreter") {
    return createInterpreterDeviceManager(config);
  } else if (config.backendName == "OpenCL") {
    return createOCLDeviceManager(config);
  } else if (config.backendName == "CPU") {
    return createCPUDeviceManager(config);
  } else if (config.backendName == "Habana") {
    return createHabanaDeviceManager(config);
  } else {
    // As a fallback to make developing new Backends easier we'll create a
    // DummyDeviceManager here, but this is not threadsafe and very simplistic.
    // Strongly recommended that you create a DeviceManager customized for your
    // device.
    LOG(ERROR) << "Warning: Creating a DummyDeviceManager.\n";
    return new DummyDeviceManager(config);
  }
}
