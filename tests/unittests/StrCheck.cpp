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

#include "glow/Testing/StrCheck.h"
#include "gtest/gtest.h"

using glow::StrCheck;

TEST(StrCheck, check) {
  EXPECT_TRUE(StrCheck("foo bar").check("oo").check("bar"));
  EXPECT_FALSE(StrCheck("foo bar").check("foo").check("oo"));
}

TEST(StrCheck, sameln) {
  EXPECT_TRUE(StrCheck("foo bar").sameln("foo").sameln("bar"));
  EXPECT_FALSE(StrCheck("foo\nbar").sameln("foo").sameln("bar"));
}

TEST(StrCheck, nextln) {
  EXPECT_FALSE(StrCheck("foo bar").check("foo").nextln("bar"));
  EXPECT_TRUE(StrCheck("foo\nbar").check("foo").nextln("bar"));
  EXPECT_FALSE(StrCheck("foo\n\nbar").check("foo").nextln("bar"));
}

TEST(StrCheck, no) {
  EXPECT_TRUE(StrCheck("foo bar").check("foo").no("baz"));
  EXPECT_FALSE(StrCheck("foo bar").check("foo").no("bar"));
  EXPECT_TRUE(StrCheck("foo bar").no("baz").check("bar"));
  EXPECT_FALSE(StrCheck("foo bar").no("foo").check("bar"));
  EXPECT_TRUE(StrCheck("foo bar").check("foo").no("o"));

  EXPECT_FALSE(StrCheck("foo bar\nbaz").check("foo").no("bar").nextln("baz"));
}
