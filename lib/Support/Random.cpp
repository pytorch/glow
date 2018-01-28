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

int nextRandInt(int n) {
  static std::default_random_engine generator;
  static std::uniform_int_distribution<> distribution(0);
  int max = distribution.max() / n;
  max *= n;
  int m = distribution(generator);
  while (m >= max) {
    m = distribution(generator);
  }
  return m % n;
}

} // namespace glow
