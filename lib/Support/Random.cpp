// Copyright 2017 Facebook Inc.  All Rights Reserved.

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
