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

TEST(Float16, sub) {
  float16 a = 2.0;
  float16 b = 0.5;
  EXPECT_EQ(a - b, float16(float(a) - float(b)));
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
