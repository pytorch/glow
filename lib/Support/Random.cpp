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

#include <cassert>
#include <random>

namespace glow {

double nextRand() {
  static std::mt19937 generator;
  static std::uniform_real_distribution<> distribution(-1, 1);
  return distribution(generator);
}

int nextRandInt(int a, int b) {
  assert(a <= b && "Invalid bounds");
  double x = nextRand() + 1.0;
  int r = (b - a + 1) * x;
  return r / 2 + a;
}

} // namespace glow
