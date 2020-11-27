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
#include "TestBlacklist.h"
#include "tests/unittests/BackendTestUtils.h"

using namespace glow;

std::set<std::string> glow::backendTestBlacklist = {};

struct BlacklistInitializer {
  BlacklistInitializer() {
    const std::vector<std::pair<std::string, uint32_t>> testBlacklistedSetups =
        {
            {"gradientCheckGatherVec/0", TestBlacklist::AnyDeviceAnyEngine},
            {"gradientCheckGatherDim/0", TestBlacklist::AnyDeviceAnyEngine},
            {"gradientCheckConv/0", TestBlacklist::AnyDeviceAnyEngine},
            {"gradientCheckDepthwiseConv/0", TestBlacklist::AnyDeviceAnyEngine},
            {"gradientCheckGroupConv/0", TestBlacklist::AnyDeviceAnyEngine},
            {"gradientCheckDilatedConv/0", TestBlacklist::AnyDeviceAnyEngine},
            {"gradientCheckAvgPool/0", TestBlacklist::AnyDeviceAnyEngine},
            {"gradientCheckAvgPoolCountExcludePads/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"gradientCheckMaxPool/0", TestBlacklist::AnyDeviceAnyEngine},
            {"gradientCheckAdaptiveAvgPool/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"gradientCheckBatchNorm/0", TestBlacklist::AnyDeviceAnyEngine},
            {"gradientCheckArithmeticDiv/0", TestBlacklist::AnyDeviceAnyEngine},
            {"gradientCheckArithmetic/0", TestBlacklist::AnyDeviceAnyEngine},
            {"gradientCheckFCConcatTanh/0", TestBlacklist::AnyDeviceAnyEngine},
            {"gradientCheckFC/0", TestBlacklist::AnyDeviceAnyEngine},
            {"gradientCheckTranspose/0", TestBlacklist::AnyDeviceAnyEngine},
            {"gradientCheckCrossEntropyLoss/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"gradientCheckBatchedPairwiseDotProduct/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"gradientCheckFC2/0", TestBlacklist::AnyDeviceAnyEngine},
        };
    TestBlacklist::prepareBlacklist(testBlacklistedSetups,
                                    backendTestBlacklist);
  }
} blacklistInitializer;
