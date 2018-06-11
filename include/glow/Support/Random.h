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
#ifndef GLOW_SUPPORT_RANDOM_H
#define GLOW_SUPPORT_RANDOM_H

#include <random>

namespace glow {

/// A pseudo-random number generator.
///
/// A PseudoRNG generates a deterministic sequence of numbers controlled by the
/// initial seed. Use the various templates from the <random> standard library
/// header to draw numbers from a specific distribution.
class PseudoRNG {
public:
  /// Glow uses the Mersenne Twister engine.
  typedef std::mt19937 Engine;

private:
  Engine engine_;

public:
  /// Get a freshly initialized pseudo-random number generator.
  ///
  /// All the generators created by this default constructor will generate the
  /// same deterministic sequence of numbers, controlled by the
  /// "-pseudo-random-seed" command line option.
  PseudoRNG();

  /// \returns a pseudo-random floating point number from the half-open range
  /// [-1; 1).
  double nextRand();

  double nextRandReal(double a, double b);

  /// \returns the next uniform random integer in the closed interval [a, b].
  int nextRandInt(int a, int b) {
    return std::uniform_int_distribution<int>(a, b)(engine_);
  }

  /// This typedef and the methods below implement the standard interface for a
  /// random number generator.
  typedef Engine::result_type result_type;
  /// \returns the next pseudo-random number between min() and max().
  result_type operator()() { return engine_(); }
  /// \returns the smallest possible value generated.
  static constexpr result_type min() { return Engine::min(); }
  /// \returns the largest possible value generated.
  static constexpr result_type max() { return Engine::max(); }
};

} // namespace glow

#endif // GLOW_SUPPORT_RANDOM_H
