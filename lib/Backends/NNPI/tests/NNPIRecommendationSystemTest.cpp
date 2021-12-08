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

#include "tests/unittests/BackendTestUtils.h"

using namespace glow;

std::set<std::string> glow::backendTestBlacklist = {
    "RecSys_FP32_Partitioned_Deferred/0",
    "RecSys_Partitioned_RWQuantizedFP16_SLWS_Deferred/0",
    "RecSys_SLS_Only_Deferred/0",
    "RecSys_RWQuantized_SLWS_FP16_Deferred/0",
    "RecSys_Partitioned_RWQuantized_SLWS_FP16_Deferred/0",
    "RecSys_Partitioned_RWQuantized_SLWS_Deferred/0",
    "RecSys_RWQuantized_SLWS_Deferred/0",
    "RecSys_RWQuantized_SLWS_FC_Deferred/0",
    "RecSys_Partitioned_RWQuantized_SLWS_FC_Deferred/0",
};

bool glow::useSymmetricRowwiseQuantFC = true;
