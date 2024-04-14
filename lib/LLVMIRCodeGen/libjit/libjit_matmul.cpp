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
#include "libjit_defs.h"

namespace {

/// Macros for accessing submatrices of a matmul using the leading dimension.
#define A(i, j) a[(j) * lda + (i)]
#define B(i, j) b[(j) * ldb + (i)]
#define C(i, j) c[(j) * ldc + (i)]

/// Naive gemm helper to handle oddly-sized matrices.
void libjit_matmul_odd(int m, int n, int k, const float *a, int lda,
                       const float *b, int ldb, float *c, int ldc) {
  // The order of these loops is tuned for column-major matrices.
  for (int p = 0; p < k; p++) {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < m; i++) {
        C(i, j) += A(i, p) * B(p, j);
      }
    }
  }
}

/// Number of registers to use for rows of A in the dot-product kernel.
constexpr int regsA = 4;
/// Number of registers to use for columns of B in the dot-product kernel.
constexpr int regsB = 3;

/// Number of rows of A to process in the kernel.  Vector loads are used for A,
/// so we load eight times as many floats as we use registers.
constexpr int mr = regsA * 8;
/// Number of columns of B to process in the kernel.
constexpr int nr = regsB;

/// Blocking parameters for the outer kernel.  We multiply mc x kc blocks of A
/// with kc x nc panels of B (this approach is referred to as `gebp` in the
/// literature).  TODO: Generalize these parameters for other cache sizes.
constexpr int mc = 256;
constexpr int kc = 128;
constexpr int nc = 4096;

/// Compute a RAxRB block of C using a vectorized dot product, where RA is the
/// number of registers to load from matrix A, and RB is the number of registers
/// to load from matrix B.
template <size_t regsA, size_t regsB>
void libjit_matmul_dot(size_t k, const float *a, size_t lda, const float *b,
                       size_t ldb, float *c, size_t ldc) {
  float8 csum[regsA][regsB] = {{0.0}};
  for (size_t p = 0; p < k; p++) {
    // Perform the DOT product.
    for (size_t ai = 0; ai < regsA; ai++) {
      float8 aa = LoaduFloat8(&A(ai * 8, p));
      for (size_t bi = 0; bi < regsB; bi++) {
        float8 bb = BroadcastFloat8(B(p, bi));
        csum[ai][bi] += aa * bb;
      }
    }
  }

  // Accumulate the results into C.
  for (size_t bi = 0; bi < regsB; bi++) {
    for (size_t ai = 0; ai < regsA; ai++) {
      AdduFloat8(&C(ai * 8, bi), csum[ai][bi]);
    }
  }
}

/// Similar to libjit_matmul_dot, but assumes that \p a and \p b have been
/// packed using z-ordering.
template <size_t regsA, size_t regsB>
void libjit_matmul_zdot(size_t k, const float *a, size_t lda, const float *b,
                        size_t ldb, float *c, size_t ldc) {
  float8 csum[regsA][regsB] = {{0.0}};

  for (size_t p = 0; p < k; p++) {
    // Perform the DOT product.
    float8 *aptr = (float8 *)&A(0, p);
    for (size_t ai = 0; ai < regsA; ai++) {
      float8 aa = *aptr++;
      for (size_t bi = 0; bi < regsB; bi++) {
        float8 bb = BroadcastFloat8(*(b + bi));
        csum[ai][bi] += aa * bb;
      }
    }
    b += regsB;
  }

  // Accumulate the results into C.
  for (size_t bi = 0; bi < regsB; bi++) {
    for (size_t ai = 0; ai < regsA; ai++) {
      AdduFloat8(&C(ai * 8, bi), csum[ai][bi]);
    }
  }
}

/// Pack matrix \p a into matrix \p a_to using a z-ordering, so that the
/// dot-product kernel can stride sequentially through memory.
template <size_t regsA>
void pack_matrix_a(size_t m, size_t k, const float *a, size_t lda,
                   float *a_to) {
  for (int i = 0; i < int(m) - mr + 1; i += mr) {
    for (size_t j = 0; j < k; j++) {
      const float *a_ij_pntr = &A(i, j);
      for (size_t ai = 0; ai < regsA; ai++) {
        StoreuFloat8(a_to + 8 * ai, LoaduFloat8(a_ij_pntr + 8 * ai));
      }
      a_to += 8 * regsA;
    }
  }
}

/// Pack matrix \p b into matrix \p b_to using a z-ordering, so that the
/// dot-product kernel can stride sequentially through memory, rather than
/// reading from `regsB` separate columns.
template <size_t regsB>
void pack_matrix_b(size_t n, size_t k, const float *b, size_t ldb,
                   float *b_to) {
  for (int j = 0; j < int(n) - nr + 1; j += nr) {
    for (size_t i = 0; i < k; i++) {
      for (size_t bi = 0; bi < regsB; bi++) {
        *b_to++ = B(i, j + bi);
      }
    }
  }
}

/// Inner kernel for packed matrices.  The order of the M and N loops matters,
/// because packed matrices need to be more more sensitive to cache locality,
/// and N strides over the B matrix, which is very large and will blow out the
/// cache.
void libjit_matmul_inner_packed(int m, int n, int k, const float *packedA,
                                const float *packedB, float *c, int ldc) {
  for (int j = 0; j < n - nr + 1; j += nr) {
    for (int i = 0; i < m - mr + 1; i += mr) {
      libjit_matmul_zdot<regsA, regsB>(k, &packedA[i * k], mr, &packedB[j * k],
                                       k, &C(i, j), ldc);
    }
  }
}

/// Inner kernel for non-packed matrices.  In these cases N is small, so it
/// tends to be beneficial to retain locality in the A matrix.
void libjit_matmul_inner_unpacked(int m, int n, int k, const float *a, int lda,
                                  const float *b, int ldb, float *c, int ldc) {
  for (int i = 0; i < m - mr + 1; i += mr) {
    for (int j = 0; j < n - nr + 1; j += nr) {
      libjit_matmul_dot<regsA, regsB>(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j),
                                      ldc);
    }
  }
}

/// Compute a portion of C one block at a time.  Handle ragged edges with calls
/// to a slow but general helper.
template <bool pack>
void libjit_matmul_inner(int m, int n, int k, const float *a, int lda,
                         const float *b, int ldb, float *c, int ldc,
                         float *packedB) {
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
  float packedA[m * k] __attribute__((aligned(64)));
  if (pack) {
    pack_matrix_a<regsA>(m, k, &A(0, 0), lda, packedA);
  }

  if (pack) {
    libjit_matmul_inner_packed(m, n, k, packedA, packedB, c, ldc);
  } else {
    libjit_matmul_inner_unpacked(m, n, k, a, lda, b, ldb, c, ldc);
  }

  sdim_t i = (m / mr) * mr;
  sdim_t j = (n / nr) * nr;
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
/// \p a is an \p m x \p k column-major matrix;
/// \p b is a \p k x \p n column-major matrix;
/// \p c is a \p m x \p n column-major matrix.
/// \p lda, \p ldb, and \p ldc are the leading dimensions of A, B, and C,
/// respectively.
template <bool pack>
void __attribute__((noinline))
libjit_matmul_outer(dim_t m, dim_t n, dim_t k, const float *a, dim_t lda,
                    const float *b, dim_t ldb, float *c, dim_t ldc) {
  float *packedB = nullptr;
  if (pack) {
    libjit_aligned_malloc((void **)&packedB, 64, kc * nc);
  }

  for (dim_t p = 0; p < k; p += kc) {
    dim_t pb = MIN(k - p, kc);
    for (dim_t j = 0; j < n; j += nc) {
      dim_t jb = MIN(n - j, nc);
      if (pack) {
        pack_matrix_b<regsB>(jb, pb, &B(p, j), ldb, packedB);
      }
      for (dim_t i = 0; i < m; i += mc) {
        dim_t ib = MIN(m - i, mc);
        libjit_matmul_inner<pack>(ib, jb, pb, &A(i, p), lda, &B(p, j), ldb,
                                  &C(i, j), ldc, packedB);
      }
    }
  }

  if (pack) {
    libjit_aligned_free(packedB);
  }
}

#undef C
#undef B
#undef A

/// Generic template for FullyConnected. The template allows choosing the
/// element type and bias type.
template <typename ElemTy, typename BiasElemTy>
void libjit_fc_generic(ElemTy *outW, const ElemTy *inW, const ElemTy *weightsW,
                       const BiasElemTy *biasW, const dim_t *outWdims,
                       const dim_t *inWdims, const dim_t *weightsWdims,
                       const dim_t *biasWdims, int32_t outOffset,
                       int32_t inOffset, int32_t weightsOffset,
                       int32_t biasOffset, int32_t biasPre, int32_t biasPost,
                       int32_t biasScale, int32_t outPre, int32_t outPost,
                       int32_t outScale) {
  dim_t in_w = inWdims[1];
  dim_t out_h = outWdims[0];
  dim_t out_w = outWdims[1];
  for (size_t i = 0; i < out_h; i++) {
    for (size_t j = 0; j < out_w; j++) {
      int32_t sum = libjit_scale<int32_t>(biasW[j] - biasOffset, biasPre,
                                          biasPost, biasScale, 0);
      for (size_t k = 0; k < in_w; k++) {
        int32_t I = inW[libjit_getXY(inWdims, i, k)];
        int32_t W = weightsW[libjit_getXY(weightsWdims, k, j)];
        sum += (I - inOffset) * (W - weightsOffset);
      }
      int32_t scaledSum =
          libjit_scale<int32_t>(sum, outPre, outPost, outScale, outOffset);
      outW[libjit_getXY(outWdims, i, j)] = libjit_clip_i8(scaledSum);
    }
  }
}

/// Generic template for rowwise quantized FullyConnected. The template allows
/// choosing element type and bias type.
template <typename ElemTy, typename BiasElemTy>
void libjit_rowwise_quantized_fc_generic(
    ElemTy *outW, const ElemTy *inW, const ElemTy *weightsW,
    const BiasElemTy *biasW, const int32_t *weightsOffsets,
    const int32_t *biasPre, const int32_t *biasPost, const int32_t *biasScale,
    const int32_t *outPre, const int32_t *outPost, const int32_t *outScale,
    const dim_t *outWdims, const dim_t *inWdims, const dim_t *weightsWdims,
    const dim_t *biasWdims, dim_t rowNum, int32_t outOffset, int32_t inOffset,
    int32_t biasOffset) {
  dim_t in_w = inWdims[1];
  dim_t out_h = outWdims[0];
  dim_t out_w = outWdims[1];

  // In rowwise quantized FC, weights is not pretransposed : I * Tranpose(W) +
  // B. out(i, j) = in(i, 0) * weights(j, 0) + in(i, 1) * weights(j, 1) + ... +
  //                in(i, k) * weights(j, k) + bias(j);
  for (size_t i = 0; i < out_h; i++) {
    for (size_t j = 0; j < out_w; j++) {
      int32_t sum = 0;
      for (size_t k = 0; k < in_w; k++) {
        int32_t W = weightsW[libjit_getXY(weightsWdims, j, k)];
        int32_t I = inW[libjit_getXY(inWdims, i, k)];
        sum += (W - weightsOffsets[j]) * (I - inOffset);
      }
      int32_t B = libjit_scale<int32_t>(biasW[j] - biasOffset, biasPre[j],
                                        biasPost[j], biasScale[j], 0);
      sum += B;
      int32_t scaledSum = libjit_scale<int32_t>(sum, outPre[j], outPost[j],
                                                outScale[j], outOffset);
      outW[libjit_getXY(outWdims, i, j)] = libjit_clip_i8(scaledSum);
    }
  }
}
} // namespace

extern "C" {

/// Performs the matrix multiplication c = a * b, where c, a, and b are
/// row-major matrices.
/// \p c is a m x n matrix, so \p cDims = {m, n}
/// \p a is a m x k matrix, so \p aDims = {m, k}
/// \p b is a k x n matrix, so \p bDims = {k, n}
void libjit_matmul_f(float *c, const float *a, const float *b,
                     const dim_t *cDims, const dim_t *aDims,
                     const dim_t *bDims) {
  memset(c, 0, cDims[0] * cDims[1] * sizeof(float));
  // Call the matrix multiplication routine with appropriate dimensions and
  // leading dimensions. The "leading dimension" for a row-major matrix is equal
  // to the number of columns in the matrix.  For a, this is k; for b and c,
  // this is n.
  //
  // This "outer" helper assumes the matrices are given in column-major format
  // (the packing algorithm is more effective with column-major matrices), while
  // the input is row-major. So we compute C += B * A, which is equivalent.
  //
  // The matrix multiplication routine is heavily inspired by:
  // https://github.com/flame/how-to-optimize-gemm
  int m = cDims[1];
  int n = cDims[0];
  int k = aDims[1];

  // Use the unpacked version which does not use extra HEAP or STACK which
  // makes the memory usage predictable. This is very useful when building
  // bundles (AOT) for MCU targets where the HEAP and STACK are relatively
  // limited in size. By avoiding heap/stack usage the memory consumption
  // is controlled and perfectly known (e.g. printed in the bundle API).
  libjit_matmul_outer<false>(m, n, k, b, bDims[1], a, aDims[1], c, cDims[1]);
}

void libjit_matmul_i8(int8_t *outW, const int8_t *lhsW, const int8_t *rhsW,
                      const dim_t *outWdims, const dim_t *lhsWdims,
                      const dim_t *rhsWdims, int32_t outOffset,
                      int32_t lhsOffset, int32_t rhsOffset, int32_t outPre,
                      int32_t outPost, int32_t outScale) {
  for (dim_t x = 0; x < outWdims[0]; x++) {
    for (dim_t y = 0; y < outWdims[1]; y++) {
      int32_t sum = 0;
      for (dim_t i = 0; i < lhsWdims[1]; i++) {
        int32_t lhs = lhsW[libjit_getXY(lhsWdims, x, i)] - lhsOffset;
        int32_t rhs = rhsW[libjit_getXY(rhsWdims, i, y)] - rhsOffset;
        sum += lhs * rhs;
      }
      int32_t s =
          libjit_scale<int32_t>(sum, outPre, outPost, outScale, outOffset);
      outW[libjit_getXY(outWdims, x, y)] = libjit_clip_i8(s);
    }
  }
}

/// FullyConnected with float precision.
void libjit_fc_f(float *outW, const float *inW, const float *weightsW,
                 const float *biasW, const dim_t *outWdims,
                 const dim_t *inWdims, const dim_t *weightsWdims,
                 const dim_t *biasWdims) {
  dim_t in_w = inWdims[1];
  dim_t out_h = outWdims[0];
  dim_t out_w = outWdims[1];
  for (size_t i = 0; i < out_h; i++) {
    for (size_t j = 0; j < out_w; j++) {
      float sum = biasW[j];
      for (size_t k = 0; k < in_w; k++) {
        float I = inW[libjit_getXY(inWdims, i, k)];
        float W = weightsW[libjit_getXY(weightsWdims, k, j)];
        sum += I * W;
      }
      outW[libjit_getXY(outWdims, i, j)] = sum;
    }
  }
}

/// FullyConnected with int8 precision and int32 bias.
void libjit_fc_i8_i32(int8_t *outW, const int8_t *inW, const int8_t *weightsW,
                      const int32_t *biasW, const dim_t *outWdims,
                      const dim_t *inWdims, const dim_t *weightsWdims,
                      const dim_t *biasWdims, int32_t outOffset,
                      int32_t inOffset, int32_t weightsOffset,
                      int32_t biasOffset, int32_t biasPre, int32_t biasPost,
                      int32_t biasScale, int32_t outPre, int32_t outPost,
                      int32_t outScale) {
  libjit_fc_generic<int8_t, int32_t>(
      outW, inW, weightsW, biasW, outWdims, inWdims, weightsWdims, biasWdims,
      outOffset, inOffset, weightsOffset, biasOffset, biasPre, biasPost,
      biasScale, outPre, outPost, outScale);
}

/// FullyConnected with int8 precision and int8 bias.
void libjit_fc_i8_i8(int8_t *outW, const int8_t *inW, const int8_t *weightsW,
                     const int8_t *biasW, const dim_t *outWdims,
                     const dim_t *inWdims, const dim_t *weightsWdims,
                     const dim_t *biasWdims, int32_t outOffset,
                     int32_t inOffset, int32_t weightsOffset,
                     int32_t biasOffset, int32_t biasPre, int32_t biasPost,
                     int32_t biasScale, int32_t outPre, int32_t outPost,
                     int32_t outScale) {
  libjit_fc_generic<int8_t, int8_t>(
      outW, inW, weightsW, biasW, outWdims, inWdims, weightsWdims, biasWdims,
      outOffset, inOffset, weightsOffset, biasOffset, biasPre, biasPost,
      biasScale, outPre, outPost, outScale);
}

/// Rowwise quantized FullyConnected with int8 precision and int32 bias.
void libjit_rowwise_quantized_fc_i8_i32(
    int8_t *outW, const int8_t *inW, const int8_t *weightsW,
    const int32_t *biasW, const int32_t *weightsOffsets, const int32_t *biasPre,
    const int32_t *biasPost, const int32_t *biasScale, const int32_t *outPre,
    const int32_t *outPost, const int32_t *outScale, const dim_t *outWdims,
    const dim_t *inWdims, const dim_t *weightsWdims, const dim_t *biasWdims,
    dim_t rowNum, int32_t outOffset, int32_t inOffset, int32_t biasOffset) {
  libjit_rowwise_quantized_fc_generic<int8_t, int32_t>(
      outW, inW, weightsW, biasW, weightsOffsets, biasPre, biasPost, biasScale,
      outPre, outPost, outScale, outWdims, inWdims, weightsWdims, biasWdims,
      rowNum, outOffset, inOffset, biasOffset);
}

/// Rowwise quantized FullyConnected with int8 precision and int8 bias.
void libjit_rowwise_quantized_fc_i8_i8(
    int8_t *outW, const int8_t *inW, const int8_t *weightsW,
    const int8_t *biasW, const int32_t *weightsOffsets, const int32_t *biasPre,
    const int32_t *biasPost, const int32_t *biasScale, const int32_t *outPre,
    const int32_t *outPost, const int32_t *outScale, const dim_t *outWdims,
    const dim_t *inWdims, const dim_t *weightsWdims, const dim_t *biasWdims,
    dim_t rowNum, int32_t outOffset, int32_t inOffset, int32_t biasOffset) {
  libjit_rowwise_quantized_fc_generic<int8_t, int8_t>(
      outW, inW, weightsW, biasW, weightsOffsets, biasPre, biasPost, biasScale,
      outPre, outPost, outScale, outWdims, inWdims, weightsWdims, biasWdims,
      rowNum, outOffset, inOffset, biasOffset);
}
}
