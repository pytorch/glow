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
            {"AvgPoolGradTest/0", TestBlacklist::AnyDeviceAnyEngine},
            {"basicFCNet/0", TestBlacklist::AnyDeviceAnyEngine},
            {"basicFCNetQuantized/0", TestBlacklist::AnyDeviceAnyEngine},
            {"complexNet1/0", TestBlacklist::AnyDeviceAnyEngine},
            {"convDKKC8Test/0", TestBlacklist::AnyDeviceAnyEngine},
            {"convGradTest/0", TestBlacklist::AnyDeviceAnyEngine},
            {"convTest/0", TestBlacklist::AnyDeviceAnyEngine},
            {"groupConvTest/0", TestBlacklist::AnyDeviceAnyEngine},
            {"localResponseNormalizationGradTest/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"MaxPoolGradTest/0", TestBlacklist::AnyDeviceAnyEngine},
            {"nonSquareStrideConvTest/0", TestBlacklist::AnyDeviceAnyEngine},
            {"nonSquareKernelConvTest/0", TestBlacklist::AnyDeviceAnyEngine},
            {"nonSquarePaddingConvTest/0", TestBlacklist::AnyDeviceAnyEngine},
            {"quantizedConvTest/0", TestBlacklist::AnyDeviceAnyEngine},
            {"softmaxGradTest/0", TestBlacklist::AnyDeviceAnyEngine},
            {"convOps/0", TestBlacklist::AnyDeviceHWEngine},
            {"localResponseNormalizationTest/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"tinyResnet/0", TestBlacklist::AnyDeviceHWEngine},
            {"SymmetricQuantizedConvReluFusionTest/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"AsymmetricQuantizedConvReluFusionTest/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"intLookupTableInt16/0", TestBlacklist::AnyDeviceAnyEngine},
#if NNPI_MAJOR_VERSION == 1 && NNPI_MINOR_VERSION == 0
            {"intLookupTableInt8/0", TestBlacklist::AnyDeviceAnyEngine},
#else
            {"intLookupTableInt8/0", TestBlacklist::AnyDeviceSWEngine},
#endif
        };
    TestBlacklist::prepareBlacklist(testBlacklistedSetups,
                                    backendTestBlacklist);
  }
} blacklistInitializer;
