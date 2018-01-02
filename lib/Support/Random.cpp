// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Support/Random.h"

#include <random>

namespace glow {
double nextRand() {
  static std::mt19937 generator;
  static std::uniform_real_distribution<> distribution(-1, 1);
  return distribution(generator);
}
} // namespace glow
