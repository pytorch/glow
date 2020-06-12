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
        {"BatchMatMulTest_Float16/10", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/11", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/14", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/15", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/18", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/19", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/2", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/21", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/22", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/23", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/25", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/26", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/27", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/3", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/30", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/31", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/33", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/34", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/35", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/37", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/38", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/39", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/41", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/42", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/43", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/45", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/46", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/47", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/49", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/50", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/51", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/54", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/55", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/58", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/59", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/6", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/61", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/62", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/63", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/65", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/66", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/67", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/69", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/7", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/70", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/71", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/73", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/74", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/75", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/78", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/79", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/81", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/82", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/83", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/85", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/86", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/87", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/89", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/90", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/91", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/93", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/94", TestBlacklist::AnyDeviceAnyEngine},
        {"BatchMatMulTest_Float16/95", TestBlacklist::AnyDeviceAnyEngine},
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
        {"ConvTest_Float16/11", TestBlacklist::AnyDeviceAnyEngine},
        {"ConvTest_Float16/3", TestBlacklist::AnyDeviceAnyEngine},
        {"ConvTest_Float16/7", TestBlacklist::AnyDeviceAnyEngine},
        {"ConvTest_Float16/9", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/100", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/101", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/102", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/103", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/104", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/116", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/117", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/118", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/119", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/121", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/122", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/123", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/124", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/125", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/126", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/127", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/128", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/129", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/130", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/131", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/132", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/133", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/134", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/135", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/136", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/137", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/138", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/139", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/14", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/17", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/18", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/19", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/21", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/22", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/23", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/24", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/26", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/27", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/28", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/29", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/30", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/31", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/32", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/33", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/34", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/47", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/48", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/49", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/51", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/52", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/53", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/54", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/55", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/56", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/57", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/58", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/59", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/60", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/61", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/62", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/63", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/64", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/65", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/66", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/67", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/68", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/69", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/83", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/84", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/86", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/87", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/88", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/89", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/90", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/91", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/92", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/93", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/94", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/95", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/96", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/97", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/98", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Float16/99", TestBlacklist::AnyDeviceAnyEngine},
        {"FCTest_Int8/139", TestBlacklist::AnyDeviceAnyEngine},
    };

    std::vector<int> passingTestsRWQSLWS_Float16 = {
        43, 47, 48, 49, 53, 54, 57, 58, 59, 61, 62, 63, 64, 65,
        66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79};
    std::vector<int>::const_iterator passingTestIt =
        passingTestsRWQSLWS_Float16.begin();

    for (int i = 0; i < 80; i++) {
      if (i == *passingTestIt) {
        passingTestIt++;
      } else {
        testBlacklistedSetups.push_back(
            {"RWQSLWS_Float16_AccumFloat16/" + std::to_string(i),
             TestBlacklist::AnyDeviceAnyEngine});
      }
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
