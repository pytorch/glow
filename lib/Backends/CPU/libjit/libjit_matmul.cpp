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
#include "libjit_defs.h"

namespace {

/// Macros for accessing submatrices of a matmul using the leading dimension.
#define A(i, j) a[(i)*lda + (j)]
#define B(i, j) b[(i)*ldb + (j)]
#define C(i, j) c[(i)*ldc + (j)]

/// Naive gemm helper to handle oddly-sized matrices.
void libjit_matmul_odd(int m, int n, int k, const float *a, int lda,
                       const float *b, int ldb, float *c, int ldc) {
  for (int p = 0; p < k; p++) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        C(i, j) += A(i, p) * B(p, j);
      }
    }
  }
}

/// Compute a RAxRB block of C using a vectorized dot product, where RA is the
/// number of registers to load from matrix A, and RB is the number of registers
/// to load from matrix B.
template <unsigned int regsA, unsigned int regsB>
void libjit_matmul_dot(int k, const float *a, int lda, const float *b, int ldb,
                       float *c, int ldc) {
  float8 csum[regsA][regsB] = {{0.0}};
  for (int p = 0; p < k; p++) {

    // Perform the DOT product.
    for (int bi = 0; bi < regsB; bi++) {
      float8 bb = LoaduFloat8(&B(p, bi * 8));
      for (int ai = 0; ai < regsA; ai++) {
        float8 aa = BroadcastFloat8(A(ai, p));
        csum[ai][bi] += aa * bb;
      }
    }
  }

  // Accumulate the results into C.
  for (int ai = 0; ai < regsA; ai++) {
    for (int bi = 0; bi < regsB; bi++) {
      AdduFloat8(&C(ai, bi * 8), csum[ai][bi]);
    }
  }
}

/// Compute a portion of C one block at a time.  Handle ragged edges with calls
/// to a slow but general helper.
void libjit_matmul_inner(int m, int n, int k, const float *a, int lda,
                         const float *b, int ldb, float *c, int ldc) {
  constexpr int regsA = 3;
  constexpr int regsB = 4;

  constexpr int mc = regsA;
  constexpr int nr = regsB * 8;
  // The tiling scheme naturally divides the input matrices into 2 parts each;
  // one tiled section, and three "ragged" edges.
  //
  // --------------------    -------
  // | A00*B00 | A00*B01|    | A00 |   -------------
  // -------------------- += ------- * | B00 | B01 |
  // | A10*B00 | A10*B01|    | A10 |   -------------
  // --------------------    -------
  //
  // We can process this as 4 separate matrix multiplications.  A00*B00 is the
  // perfectly-tiled portion, which we handly with a 4x16 dot-product kernel.
  // The ragged edges are (ideally) less critical, so we handle them with a call
  // to a general matrix-multiplication for odd sizes.
  for (int j = 0; j < n - nr + 1; j += nr) {
    for (int i = 0; i < m - mc + 1; i += mc) {
      libjit_matmul_dot<regsA, regsB>(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j),
                                      ldc);
    }
  }
  int i = (m / mc) * mc;
  int j = (n / nr) * nr;
  if (i < m) {
    libjit_matmul_odd(m - i, j, k, &A(i, 0), lda, &B(0, 0), ldb, &C(i, 0), ldc);
  }
  if (j < n) {
    libjit_matmul_odd(i, n - j, k, &A(0, 0), lda, &B(0, j), ldb, &C(0, j), ldc);
  }
  if (i < m && j < n) {
    libjit_matmul_odd(m - i, n - j, k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j),
                      ldc);
  }
}

/// Tile A into mc * kc blocks, where mc and kc are chosen to approximately fit
/// the L2 cache on recent Intel processors (e.g., 256 KB for Skylake).  Stream
/// kc * n panels of B through memory to compute each mc * n block of C.
/// \p a is an \p m x \p k row-major matrix;
/// \p b is a \p k x \p n row-major matrix;
/// \p c is a \p m x \p n row-major matrix.
/// \p lda, \p ldb, and \p ldc are the leading dimensions of A, B, and C,
/// respectively.
void libjit_matmul_outer(int m, int n, int k, const float *a, int lda,
                         const float *b, int ldb, float *c, int ldc) {
  // TODO: Generalize these parameters for other cache sizes.
  constexpr int mc = 256;
  constexpr int kc = 128;
  for (int p = 0; p < k; p += kc) {
    int pb = MIN(k - p, kc);
    for (int i = 0; i < m; i += mc) {
      int ib = MIN(m - i, mc);
      libjit_matmul_inner(ib, n, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0),
                          ldc);
    }
  }
}

#undef C
#undef B
#undef A

} // namespace

extern "C" {

/// Performs the matrix multiplication c = a * b, where c, a, and b are
/// row-major matrices.
/// \p c is a m x n matrix, so \p cDims = {m, n}
/// \p a is a m x k matrix, so \p aDims = {m, k}
/// \p b is a k x n matrix, so \p bDims = {k, n}
void libjit_matmul_f(float *c, const float *a, const float *b,
                     const size_t *cDims, const size_t *aDims,
                     const size_t *bDims) {
  memset(c, 0, cDims[0] * cDims[1] * sizeof(float));
  // Call the matrix multiplication routine with appropriate dimensions and
  // leading dimensions. The "leading dimension" for a row-major matrix is equal
  // to the number of columns in the matrix.  For a, this is k; for b and c,
  // this is n.
  //
  // The matrix multiplication routine is heavily inspired by:
  // https://github.com/flame/how-to-optimize-gemm
  libjit_matmul_outer(cDims[0], cDims[1], aDims[1], a, aDims[1], b, bDims[1], c,
                      cDims[1]);
}

void libjit_matmul_i8(int8_t *outW, const int8_t *lhsW, const int8_t *rhsW,
                      const size_t *outWdims, const size_t *lhsWdims,
                      const size_t *rhsWdims, int32_t outOffset,
                      int32_t lhsOffset, int32_t rhsOffset, int32_t outPre,
                      int32_t outPost, int32_t outScale) {
  for (size_t x = 0; x < outWdims[0]; x++) {
    for (size_t y = 0; y < outWdims[1]; y++) {
      int32_t sum = 0;
      for (size_t i = 0; i < lhsWdims[1]; i++) {
        int32_t lhs = lhsW[libjit_getXY(lhsWdims, x, i)] - lhsOffset;
        int32_t rhs = rhsW[libjit_getXY(rhsWdims, i, y)] - rhsOffset;
        sum += lhs * rhs;
      }
      int32_t s = libjit_scale_i32i8(sum, outPre, outPost, outScale, outOffset);
      outW[libjit_getXY(outWdims, x, y)] = libjit_clip(s);
    }
  }
}
}
