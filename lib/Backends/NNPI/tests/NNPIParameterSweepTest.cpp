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

#include "TestBlacklist.h"
#include "tests/unittests/BackendTestUtils.h"

using namespace glow;

std::set<std::string> glow::backendTestBlacklist = {};

struct BlacklistInitializer {
  BlacklistInitializer() {
    std::vector<std::pair<std::string, uint32_t>> testBlacklistedSetups = {
        {"BatchMatMulTest_Int8/39", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Int8/43", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Int8/46", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Int8/51", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Int8/54", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Int8/59", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Int8/63", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Int8/67", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Int8/71", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Int8/75", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Int8/83", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Int8/87", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Int8/90", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Int8/91", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Int8/95", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Int8/139", TestBlacklist::AnyDeviceAnyEngine},
    };
    for (int i = 0; i < 80; i++) {
      testBlacklistedSetups.push_back(
          {"RWQSLWS_Float16_AccumFloat16/" + std::to_string(i),
           TestBlacklist::AnyDeviceAnyEngine});
    }
    for (int i = 0; i < 80; i++) {
      testBlacklistedSetups.push_back(
          {"FRWQSLWS_Float16_AccumFloat16/" + std::to_string(i),
           TestBlacklist::AnyDeviceAnyEngine});
    }

    auto useNnpiThreshold = getenv("USE_NNPI_THRESHOLD");
    if (useNnpiThreshold && strcmp(useNnpiThreshold, "1") == 0) {
      testBlacklistedSetups.clear();
    }

    TestBlacklist::prepareBlacklist(testBlacklistedSetups,
                                    backendTestBlacklist);
  }
} blacklistInitializer;
