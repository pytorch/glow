// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Support/Random.h"

#include <random>

namespace glow {

double nextRand() {
  static std::mt19937 generator;
  static std::uniform_real_distribution<> distribution(-1, 1);
  return distribution(generator);
}

int nextRandInt01() {
  static std::default_random_engine generator;
  static std::uniform_int_distribution<> distribution(0, 1);
  return distribution(generator);
}

} // namespace glow
