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
#include "TestBlacklist.h"
#include "tests/unittests/BackendTestUtils.h"

using namespace glow;

std::set<std::string> glow::backendTestBlacklist = {};

struct BlacklistInitializer {
  BlacklistInitializer() {
    const std::vector<std::pair<std::string, uint32_t>> testBlacklistedSetups =
        {{"testStaticAssignmentP2PandDRTConcurrent/0",
          TestBlacklist::AnyDeviceSWEngine},
         {"testStaticAssignmentDeviceResidentTensorOnly/0",
          TestBlacklist::AnyDeviceSWEngine},
         {"testStaticAssignmentP2PandDRT/0", TestBlacklist::AnyDeviceSWEngine},
         {"testStaticAssignmentP2POnly/0", TestBlacklist::AnyDeviceSWEngine},
         {"ConcurrentAddRemoveUnique/0", TestBlacklist::AnyDeviceSWEngine}};
    TestBlacklist::prepareBlacklist(testBlacklistedSetups,
                                    backendTestBlacklist);
  }
} blacklistInitializer;
