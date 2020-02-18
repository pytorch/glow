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
            {"learnSqrt2Placeholder/0", TestBlacklist::AnyDeviceAnyEngine},
            {"trainASimpleNetwork/0", TestBlacklist::AnyDeviceAnyEngine},
            {"simpleRegression/0", TestBlacklist::AnyDeviceAnyEngine},
            {"learnXor/0", TestBlacklist::AnyDeviceAnyEngine},
            {"learnLog/0", TestBlacklist::AnyDeviceAnyEngine},
            {"circle/0", TestBlacklist::AnyDeviceAnyEngine},
            {"learnSingleValueConcat/0", TestBlacklist::AnyDeviceAnyEngine},
            {"trainSimpleLinearRegression/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"classifyPlayerSport/0", TestBlacklist::AnyDeviceAnyEngine},
            {"learnSinus/0", TestBlacklist::AnyDeviceAnyEngine},
            {"learnSparseLengthsSumEmbeddings/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"nonLinearClassifier/0", TestBlacklist::AnyDeviceAnyEngine},
            {"convNetForImageRecognition/0", TestBlacklist::AnyDeviceAnyEngine},
            {"testFindPixelRegression/0", TestBlacklist::AnyDeviceAnyEngine},
            {"matrixRotationRecognition/0", TestBlacklist::AnyDeviceAnyEngine},
            {"learnSparseLengthsWeightedSumEmbeddings/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"learnSparseLengthsWeightedSumWeights/0",
             TestBlacklist::AnyDeviceAnyEngine},
        };
    TestBlacklist::prepareBlacklist(testBlacklistedSetups,
                                    backendTestBlacklist);
  }
} blacklistInitializer;
