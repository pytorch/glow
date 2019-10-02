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
#include <cstdlib>
#include <random>

#include "Bench.h"

using namespace glow;

extern "C" {
// Forward declare functions from libjit.
extern void libjit_matmul_f(float *c, const float *a, const float *b,
                            const size_t *cDims, const size_t *aDims,
                            const size_t *bDims);
}

/// Benchmark an (m x k) * (k x n) = (m x n) matrix multiplication.
class GemmBench : public Benchmark {
  /// Matrices.
  std::vector<float> a;
  std::vector<float> b;
  std::vector<float> c;

  /// Dimensions expressed in libjit's format.
  size_t aDims[2];
  size_t bDims[2];
  size_t cDims[2];

public:
  GemmBench(size_t m, size_t n, size_t k)
      : aDims{m, k}, bDims{k, n}, cDims{m, n} {}

  void setup() override {
    size_t m = cDims[0];
    size_t n = cDims[1];
    size_t k = aDims[1];
    a.resize(m * k);
    b.resize(k * n);
    c.resize(m * n);
    randomize(m, k, a.data(), k);
    randomize(k, n, b.data(), n);
    randomize(m, n, c.data(), n);
  }

  void run() override {
    libjit_matmul_f(c.data(), a.data(), b.data(), cDims, aDims, bDims);
  }

  void teardown() override {}

  double gflops() const { return 2.0 * cDims[0] * cDims[1] * aDims[1] / 1e9; }

private:
  void randomize(size_t m, size_t n, float *a, size_t lda) {
    std::mt19937 gen;
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        a[i * lda + j] = dis(gen);
      }
    }
  }
};

int main() {
  constexpr int reps = 100;
  printf("outX, outY, lhsX, lhsY, rhsX, rhsY, gflops/s, \n");

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int p = 0; p < 2; p++) {
        if (i == 1 && j == 1 && p == 1) {
          break;
        }
        for (size_t x = 32; x <= 1024; x += 32) {
          size_t m = i ? 32 : x;
          size_t n = j ? 32 : x;
          size_t k = p ? 32 : x;

          GemmBench b(m, n, k);
          auto time = bench(&b, reps);
          printf("%4zu, %-4zu,   %4zu, %-4zu,   %4zu,  %-4zu,   %5.2lf\n", m, n,
                 m, k, k, n, b.gflops() / time);
        }
      }
    }
  }
}
