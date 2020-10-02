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

// This file tests the basic functionality of the float16 type.
// This is by no mean a test to show the IEEE 754 compliance!

#include "glow/Support/Float16.h"

#include "gtest/gtest.h"

using namespace glow;

TEST(Float16, add) {
  float16 a = 2.0;
  float16 b = 1.0;
  EXPECT_EQ(a + b, float16(float(a) + float(b)));
}

TEST(Float16, addEqual) {
  float16 a = 46.2;
  float16 b = 13.66;
  float16 aPlusB = a + b;
  a += b;
  EXPECT_EQ(a, aPlusB);
}

TEST(Float16, sub) {
  float16 a = 2.0;
  float16 b = 0.5;
  EXPECT_EQ(a - b, float16(float(a) - float(b)));
}

TEST(Float16, minusEqual) {
  float16 a = -146.2;
  float16 b = 131.66;
  float16 aMinusB = a - b;
  a -= b;
  EXPECT_EQ(a, aMinusB);
}

TEST(Float16, mul) {
  float16 a = 3.5;
  float16 b = 3.0;
  EXPECT_EQ(a * b, float16(float(a) * float(b)));
}

TEST(Float16, div) {
  float16 a = 16.5;
  float16 b = -3.0;
  EXPECT_EQ(a / b, float16(float(a) / float(b)));
}

TEST(Float16, gt) {
  float16 a = 13.25;
  float16 b = 3.56;
  EXPECT_EQ(a > b, float(a) > float(b));
  EXPECT_TRUE(a > b);
}

TEST(Float16, lt) {
  float16 a = 123.75;
  float16 b = -12.6;
  EXPECT_EQ(a < b, float(a) < float(b));
  EXPECT_FALSE(a < b);
}

TEST(Float16, eq) {
  float16 a = -483.455;
  float16 b = 453.0;
  EXPECT_EQ(a == b, float(a) == float(b));
  EXPECT_FALSE(a == b);
  EXPECT_TRUE(a == a);
}

TEST(Float16, ge) {
  float16 a = -31.455;
  float16 b = 4543.4;
  EXPECT_EQ(a >= b, float(a) >= float(b));
  EXPECT_FALSE(a >= b);
}

TEST(Float16, le) {
  float16 a = 214.1;
  float16 b = 4543.4;
  EXPECT_EQ(a <= b, float(a) <= float(b));
  EXPECT_TRUE(a <= b);
}

template <typename T> static void testConvertTo() {
  float16 a = 19.3;
  T b = static_cast<T>(19.3);
  EXPECT_NEAR(T(a), b, 1);
}

#define TEST_CONVERT_TO(DEST_TYPE)                                             \
  TEST(Float16, CONVERT_TO_##DEST_TYPE) { testConvertTo<DEST_TYPE>(); }
TEST_CONVERT_TO(float)
TEST_CONVERT_TO(double)
TEST_CONVERT_TO(int32_t)
TEST_CONVERT_TO(int64_t)

#undef TEST_CONVERT_TO
