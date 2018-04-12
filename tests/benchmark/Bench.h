#ifndef GLOW_TESTS_BENCHMARK_H
#define GLOW_TESTS_BENCHMARK_H

#include <algorithm>
#include <chrono>
#include <limits>

namespace glow {

/// Interface for benchmarks
class Benchmark {
public:
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
