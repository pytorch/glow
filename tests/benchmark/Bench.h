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
#ifndef GLOW_TESTS_BENCHMARK_H
#define GLOW_TESTS_BENCHMARK_H

#include <algorithm>
#include <chrono>
#include <limits>

namespace glow {

/// Interface for benchmarks
class Benchmark {
public:
  virtual ~Benchmark() = default;
  virtual void setup() = 0;
  virtual void run() = 0;
  virtual void teardown() = 0;
};

/// Run a benchmark \p reps times and report the best execution time.
double bench(Benchmark *b, size_t reps) {
  double best = std::numeric_limits<double>::max();
  b->setup();
  for (size_t i = 0; i < reps; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    b->run();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();
    best = std::min(best, duration);
  }
  b->teardown();
  return best;
}

} // namespace glow

#endif // GLOW_TESTS_BENCHMARK_H
