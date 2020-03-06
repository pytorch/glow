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
#include "tests/unittests/BackendTestUtils.h"

using namespace glow;

std::set<std::string> glow::backendTestBlacklist = {
    "gradientCheckMatMul/0",
    "gradientCheckGatherVec/0",
    "gradientCheckGatherDim/0",
    "gradientCheckConv/0",
    "gradientCheckDepthwiseConv/0",
    "gradientCheckGroupConv/0",
    "gradientCheckDilatedConv/0",
    "gradientCheckAvgPool/0",
    "gradientCheckAdaptiveAvgPool/0",
    "gradientCheckBatchNorm/0",
    "gradientCheckArithmeticDiv/0",
    "gradientCheckArithmetic/0",
    "gradientCheckFCConcatTanh/0",
    "gradientCheckFC/0",
    "gradientCheckSigmoid/0",
    "gradientCheckRelu/0",
    "gradientCheckTranspose/0",
    "gradientCheckCrossEntropyLoss/0",
    "gradientCheckMaxPool/0",
    "gradientCheckTile/0",
    "gradientCheckBatchedPairwiseDotProduct/0",
    "gradientCheckFC2/0",
};
