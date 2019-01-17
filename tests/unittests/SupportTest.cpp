/**
 * Copyright (c) 2018-present, Facebook, Inc.
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
