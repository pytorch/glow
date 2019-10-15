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
    "AvgPoolGradTest/0",
    "basicFCNet/0",
    "basicFCNetQuantized/0",
    "complexNet1/0",
    "convDKKC8Test/0",
    "convGradTest/0",
    "convTest/0",
    "groupConvTest/0",
    "intLookupTable/0",
    "localResponseNormalizationGradTest/0",
    "MaxPoolGradTest/0",
    "nonSquareKernelConvTest/0",
    "nonSquarePaddingConvTest/0",
    "nonSquareStrideConvTest/0",
    "quantizedConvTest/0",
    "softmaxGradTest/0",
};

struct EmulatorOnlyTests {
  EmulatorOnlyTests() {
    // If USE_INF_API is set, we are running on real hardware, and need
    // to blacklist additional testcases.
    auto useInfAPI = getenv("USE_INF_API");
    if (useInfAPI && !strcmp(useInfAPI, "1")) {
      // N.B.: This insertion is defined to come after the initialization of
      // backendTestBlacklist because they are ordered within the same
      // translation unit (this source file).  Otherwise this technique would
      // be subject to the static initialization order fiasco.
      backendTestBlacklist.insert({
          "convOps/0",
          "localResponseNormalizationTest/0",
          "tinyResnet/0",
      });
    }
  }
} emuTests;
