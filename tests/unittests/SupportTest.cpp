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

#include "glow/Support/Support.h"
#include "glow/Testing/StrCheck.h"
#include "gtest/gtest.h"

#ifndef GLOW_DATA_PATH
#define GLOW_DATA_PATH
#endif

using namespace glow;
using glow::StrCheck;

TEST(Support, strFormat) {
  // Check single-line formatted output.
  std::string str1 = strFormat("%s %d %c", "string1", 123, 'x');
  EXPECT_TRUE(StrCheck(str1).sameln("string1").sameln("123").sameln("x"));

  // Check multi-line formatted output.
  std::string str2 = strFormat("%s\n%d\n%c\n", "string2", 456, 'y');
  EXPECT_TRUE(StrCheck(str2).check("string2").check("456").check("y"));
  // Output is not a single line.
  EXPECT_FALSE(StrCheck(str2).sameln("string2").sameln("456").sameln("y"));
}

TEST(Support, legalizeName) {
  // Check that a name can't be empty.
  std::string str1 = legalizeName("");
  EXPECT_TRUE(str1.compare("A") == 0);

  // Check that a name must start with some alphabetic character or underscore.
  std::string str2 = legalizeName("1abc_/abc");
  EXPECT_TRUE(str2.compare("A1abc__abc") == 0);

  // Check that a legal name won't be converted.
  std::string str3 = legalizeName("abc_1aBc");
  EXPECT_TRUE(str3.compare("abc_1aBc") == 0);
}

/// Check the reading Device config from a yaml file.
TEST(Support, loadYamlFile) {
  std::string yamlFilename(GLOW_DATA_PATH "tests/runtime_test/cpuConfigs.yaml");
  std::vector<DeviceConfigHelper> lists;
  lists = deserializeDeviceConfigFromYaml(yamlFilename);
  EXPECT_EQ(lists.size(), 2);
  // Check the loading items.
  // The config file is:
  //---
  //- name:     Device1
  //  backendName: CPU
  //  parameters: |
  //  "platformID":"1"
  //    "deviceID" : "0"
  //    - name:     Device2
  //  backendName: CPU
  //  parameters: |
  //  "platformID":"1"
  //...
  EXPECT_EQ(lists[0].backendName_, "CPU");
  EXPECT_EQ(lists[0].name_, "Device1");
  EXPECT_EQ(lists[0].parameters_.str,
            "\"platformID\":\"1\"\n\"deviceID\" : \"0\"\n");
  EXPECT_EQ(lists[1].backendName_, "CPU");
  EXPECT_EQ(lists[1].name_, "Device2");
  EXPECT_EQ(lists[1].parameters_.str, "\"platformID\":\"1\"\n");
}

TEST(Support, loadStrStrMapYamlFile) {
  std::string yamlFilename(GLOW_DATA_PATH
                           "tests/runtime_test/backendSpecificOpts.yaml");
  auto map = deserializeStrStrMapFromYaml(yamlFilename);
  EXPECT_EQ(map.size(), 2);
  // Check the loading items.
  // The config file is:
  // ---
  // backendOption1: 'foo'
  // backendOption2: 'bar'
  // ...
  EXPECT_EQ(map["backendOption1"], "foo");
  EXPECT_EQ(map["backendOption2"], "bar");
}
