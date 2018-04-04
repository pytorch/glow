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

/// Compute a 4x16 block of C using a vectorized dot product.
void libjit_matmul_dot4x16(int k, const float *a, int lda, const float *b,
                           int ldb, float *c, int ldc) {
  float8 ctmp07[4] = {0.0};
  float8 ctmp815[4] = {0.0};
  for (int p = 0; p < k; p++) {
    float8 a0p = BroadcastFloat8(A(0, p));
    float8 a1p = BroadcastFloat8(A(1, p));
    float8 a2p = BroadcastFloat8(A(2, p));
    float8 a3p = BroadcastFloat8(A(3, p));
    float8 bp0p7 = LoaduFloat8(&B(p, 0));
    float8 bp8p15 = LoaduFloat8(&B(p, 8));
    ctmp07[0] += a0p * bp0p7;
    ctmp07[1] += a1p * bp0p7;
    ctmp07[2] += a2p * bp0p7;
    ctmp07[3] += a3p * bp0p7;
    ctmp815[0] += a0p * bp8p15;
    ctmp815[1] += a1p * bp8p15;
    ctmp815[2] += a2p * bp8p15;
    ctmp815[3] += a3p * bp8p15;
  }
  AdduFloat8(&C(0, 0), ctmp07[0]);
  AdduFloat8(&C(1, 0), ctmp07[1]);
  AdduFloat8(&C(2, 0), ctmp07[2]);
  AdduFloat8(&C(3, 0), ctmp07[3]);
  AdduFloat8(&C(0, 8), ctmp815[0]);
  AdduFloat8(&C(1, 8), ctmp815[1]);
  AdduFloat8(&C(2, 8), ctmp815[2]);
  AdduFloat8(&C(3, 8), ctmp815[3]);
}

/// Compute a portion of C one 4*16 block at a time.  Handle ragged edges with
/// calls to a slow but general helper.
void libjit_matmul_inner(int m, int n, int k, const float *a, int lda,
                         const float *b, int ldb, float *c, int ldc) { 
  constexpr int mc = 4;
  constexpr int nr = 16;
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
      libjit_matmul_dot4x16(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
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

}
