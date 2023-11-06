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
#include "../NNPIOptions.h"
#include "glow/Support/Support.h"
#include "nnpi_transformer_types.h"
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
namespace TestBlacklist {

enum NNPI_DEVICE_VERSION {
  NNPI_DEVICE_VERSION_1 = 0b001,
  NNPI_DEVICE_VERSION_2 = 0b010,
  NNPI_DEVICE_VERSION_3 = 0b100,
  NNPI_DEVICE_VERSION_ANY = 0b111,
};

enum NNPI_EXECUTION_ENGINE {
  NNPI_EXECUTION_ENGINE_SW = 0b01 << 16,
  NNPI_EXECUTION_ENGINE_HW = 0b10 << 16,
  NNPI_EXECUTION_ENGINE_ANY = 0b11 << 16,
};

const uint32_t AnyDeviceAnyEngine =
    NNPI_DEVICE_VERSION_ANY | NNPI_EXECUTION_ENGINE_ANY;
const uint32_t AnyDeviceHWEngine =
    AnyDeviceAnyEngine ^ NNPI_EXECUTION_ENGINE_SW;
const uint32_t AnyDeviceSWEngine =
    AnyDeviceAnyEngine ^ NNPI_EXECUTION_ENGINE_HW;
const uint32_t A0AnyEngine = NNPI_DEVICE_VERSION_1 | NNPI_EXECUTION_ENGINE_ANY;
const uint32_t A0B0DeviceAnyEngine =
    NNPI_DEVICE_VERSION_1 | NNPI_DEVICE_VERSION_2 | NNPI_EXECUTION_ENGINE_ANY;

/// Note: This function follows similar logic to the auto-device step detection
/// and setting used when compilation occurs.
inline uint32_t getCurrentDeviceVersion(bool inferOnDevice) {
  int devVer = -1;
  auto devVerStr = getenv("NNPI_DEVICE_VERSION");
  if (devVerStr) {
    auto devVerOrErr = glow::getIntFromStr(devVerStr);
    CHECK(devVerOrErr) << "Expected integer NNPI_DEVICE_VERSION: " << devVerStr;
    devVer = *devVerOrErr;
  }

  auto devVerOrErr = glow::NNPIOptions::getDeviceVersion(inferOnDevice, devVer);
  CHECK(devVerOrErr) << "Error when getting device version.";
  switch (*devVerOrErr) {
  case NNPI_1000_A:
    return NNPI_DEVICE_VERSION_1;
  case NNPI_1000_B:
    return NNPI_DEVICE_VERSION_2;
  case NNPI_1000_C:
    return NNPI_DEVICE_VERSION_3;
  default:
    LOG(FATAL) << "Error mapping device stepping to version";
  }
}

inline uint32_t getCurrentEngine() {
  auto useInfAPI = getenv("USE_INF_API");
  if (useInfAPI && !strcmp(useInfAPI, "1")) {
    return NNPI_EXECUTION_ENGINE_HW;
  }
  return NNPI_EXECUTION_ENGINE_SW;
}

typedef std::pair<std::string, uint32_t> TestSetup;

void prepareBlacklist(const std::vector<TestSetup> &testBlacklistedSetups,
                      std::set<std::string> &testBlacklist) {
  auto currentExecutionEngine = getCurrentEngine();
  auto currentDeviceVersion = getCurrentDeviceVersion(currentExecutionEngine ==
                                                      NNPI_EXECUTION_ENGINE_HW);

  auto currentSetup = currentDeviceVersion | currentExecutionEngine;
  for (const auto &testBlacklistedSetup : testBlacklistedSetups) {
    auto testname = testBlacklistedSetup.first;
    auto testSetup = testBlacklistedSetup.second;
    if ((testSetup & currentSetup) == currentSetup) {
      testBlacklist.insert(testname);
    }
  }
}
}; // namespace TestBlacklist
