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
#ifndef GLOW_TESTS_BENCHMARK_H
#define GLOW_TESTS_BENCHMARK_H

#include <algorithm>
#include <chrono>
#include <limits>
#include <tuple>
#include <vector>

namespace glow {

/// Interface for benchmarks
class Benchmark {
public:
  virtual ~Benchmark() = default;
  virtual void setup() = 0;
  virtual void run() = 0;
  virtual void teardown() = 0;
};

/// Run a benchmark \p reps times and return the execution times
std::vector<double> bench(Benchmark *b, size_t reps) {
  std::vector<double> times(reps);
  b->setup();
  for (size_t i = 0; i < reps; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    b->run();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();
    times[i] = duration;
  }
  b->teardown();
  return times;
}

std::vector<size_t> getBatchSizePerCore(size_t batchSize, size_t numCores) {
  std::vector<size_t> batchSizePerCore(numCores);
  for (size_t core = 0; core < numCores; core++) {
    size_t perCore = (batchSize + numCores - 1) / numCores;
    size_t startIdx = core * perCore;
    size_t endIdx = (core + 1) * perCore;
    if (startIdx > batchSize)
      startIdx = batchSize;
    if (endIdx > batchSize)
      endIdx = batchSize;
    batchSizePerCore[core] = (endIdx - startIdx);
  }
  return batchSizePerCore;
}

} // namespace glow

#endif // GLOW_TESTS_BENCHMARK_H
