/**
 * Copyright (c) 2017-present, Facebook, Inc.
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

#include "glow/Support/Random.h"

#include "gtest/gtest.h"

using namespace glow;

// Test that nextRandInt generates every number in the closed interval [lb, ub].
// Use enough trials that the probability of random failure is < 1.0e-9.
TEST(Utils, randomClosedInterval) {
  constexpr int lb = -3;
  constexpr int ub = 3;
  constexpr int trials = 200;

  for (int i = lb; i <= ub; i++) {
    int j = 0;
    for (; j < trials; j++) {
      if (nextRandInt(lb, ub) == i) {
        break;
      }
    }
    EXPECT_LT(j, trials);
  }
}

// Same as above, but with a PseudoRNG instance.
TEST(Utils, PRNGBasics) {
  PseudoRNG PRNG;
  constexpr int lb = -3;
  constexpr int ub = 3;
  constexpr int trials = 200;

  for (int i = lb; i <= ub; i++) {
    int j = 0;
    for (; j < trials; j++) {
      if (PRNG.nextRandInt(lb, ub) == i) {
        break;
      }
    }
    EXPECT_LT(j, trials);
  }
}

// Test that two default-constructed PseudoRNG objects do in fact generate
// identical sequences.
TEST(Utils, deterministicPRNG) {
  PseudoRNG genA, genB;
  std::uniform_int_distribution<int> dist(0, 100000);

  for (unsigned i = 0; i != 100; i++) {
    EXPECT_EQ(dist(genA), dist(genB));
  }
}
