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
#ifndef GLOW_LLVMIRCODEGEN_LIBJIT_LIBJIT_DEFS_H
#define GLOW_LLVMIRCODEGEN_LIBJIT_LIBJIT_DEFS_H

#include <cstdlib>
#include <stdint.h>
#include <string.h>

#include "libjit_dim_t.h"

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#if defined(__clang__)
using float4 = float __attribute__((ext_vector_type(4)));
using float8 = float __attribute__((ext_vector_type(8)));
#elif defined(__GNUC__) || defined(__GNUG__)
using float4 = float __attribute__((vector_size(16)));
using float8 = float __attribute__((vector_size(32)));
#endif

/// Loads a simd float8 value from \p ptr.
#define LoadFloat8(PTR) *((const float8 *)(PTR))

/// Stores the simd float8 value to \p ptr.
#define StoreFloat8(PTR, VAL) *((float8 *)(PTR)) = (VAL);

/// Accumulate (+=) the simd float8 value to \p ptr.
#define AddFloat8(PTR, VAL) *((float8 *)(PTR)) += (VAL);

/// Broadcast the input value to a float8.
#if defined(__clang__)
#define BroadcastFloat8(VAL) ((float8)(VAL))
#elif defined(__GNUC__) || defined(__GNUG__)
#define BroadcastFloat8(VAL) ((VAL) - (float8){0})
#endif

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define AT(tensor, dims, numDims, indices, numIndices)                         \
  tensor[get_element_ptr(tensor, dims, numDims, indices, numIndices)]

/// Perform an unaligned load of a float8 from a float pointer.
inline float8 LoaduFloat8(const float *p) {
  float8 res;
  memcpy(&res, p, sizeof(float8));
  return res;
}

/// Perform an unaligned store to a float pointer.
inline void StoreuFloat8(float *p, float8 v) { memcpy(p, &v, sizeof(float8)); }

/// Perform an unaligned addition to a float pointer.
inline void AdduFloat8(float *p, float8 v) {
  StoreuFloat8(p, LoaduFloat8(p) + v);
}

/// \returns the index of the element at x,y,z,w,q,r.
inline dim_t libjit_getXYZWQR(const dim_t *dims, dim_t x, dim_t y, dim_t z,
                              dim_t w, dim_t q, dim_t r) {
  return (x * dims[1] * dims[2] * dims[3] * dims[4] * dims[5]) +
         (y * dims[2] * dims[3] * dims[4] * dims[5]) +
         (z * dims[3] * dims[4] * dims[5]) + (w * dims[4] * dims[5]) +
         (q * dims[5]) + r;
}

/// \returns the index of the element at x,y,z,w,q.
inline dim_t libjit_getXYZWQ(const dim_t *dims, dim_t x, dim_t y, dim_t z,
                             dim_t w, dim_t q) {
  return (x * dims[1] * dims[2] * dims[3] * dims[4]) +
         (y * dims[2] * dims[3] * dims[4]) + (z * dims[3] * dims[4]) +
         (w * dims[4]) + q;
}

/// \returns the index of the element at x,y,z,w.
inline dim_t libjit_getXYZW(const dim_t *dims, dim_t x, dim_t y, dim_t z,
                            dim_t w) {
  return (x * dims[1] * dims[2] * dims[3]) + (y * dims[2] * dims[3]) +
         (z * dims[3]) + w;
}

/// \returns the index of the element at x,y,z.
inline dim_t libjit_getXYZ(const dim_t *dims, dim_t x, dim_t y, dim_t z) {
  return (x * dims[1] * dims[2]) + (y * dims[2]) + z;
}

/// \returns the index of the element at x,y.
inline dim_t libjit_getXY(const dim_t *dims, dim_t x, dim_t y) {
  return (x * dims[1]) + y;
}

inline int8_t libjit_clip(int32_t val) {
  return (int8_t)MIN(MAX(val, -128), 127);
}

/// Scales a 32-bit integer using the integer shift-mult-shift method.
/// See QuantizationTransform32To8 for more details.
inline int32_t libjit_scale_i32i8(int32_t input, int32_t pre, int32_t post,
                                  int32_t scale, int32_t offset) {
  // The operation x >> post is rounded down to negative infinity. To get to
  // round-nearest we add (1 << (post - 1)) to the value prior to shifting.
  // Rounding is performed only when shifting right (pos > 0).
  int rtn = (post > 0) ? (1 << (post - 1)) : 0;

  // NOTICE: If your tests are failing because of signed integer overflow then
  // this is a bug in the test and not in the program. You should make sure that
  // the inputs to the operations do not overflow. The semantics of the
  // quantization process is such that the result for values that fall out of
  // range is undefined. The conversion procedure will only tolerate a few bits
  // of overflow and the result will be clipped.
  return ((((input >> pre) * scale) + rtn) >> post) + offset;
}

/// Divides the 32-bit integer \p input with \p divider. The division is done
/// with rounding for better precision. Input can be both positive or negative.
/// Divider is assumed strictly positive.
inline int32_t libjit_div_round_i32(int32_t input, int32_t divider) {
  // Division rounding term which is added for positive input and subtracted
  // for negative input.
  int32_t rnd = (divider >> 1);
  return (input > 0) ? ((input + rnd) / divider) : ((input - rnd) / divider);
}

#ifdef _WIN32
#define libjit_aligned_malloc(p, a, s)                                         \
  (((*(p)) = _aligned_malloc((s), (a))), *(p) ? 0 : errno)
#define libjit_aligned_free(p) _aligned_free(p)
#else
#define libjit_aligned_malloc(p, a, s) posix_memalign(p, a, s)
#define libjit_aligned_free(p) free(p)
#endif

/// This function computes the minimum filter index based on the the minimum
/// input index \p inp_min.
static inline __attribute__((always_inline)) ssize_t
libjit_conv_flt_min(ssize_t inp_min) {
  return MAX(0, -inp_min);
}

/// This function computes the maximum filter index based on the the input size
/// \p inp_size, the filter size \p flt_size and the minimum input index
/// \p inp_min.
static inline __attribute__((always_inline)) ssize_t
libjit_conv_flt_max(ssize_t inp_size, ssize_t flt_size, ssize_t inp_min) {
  return MIN(flt_size, inp_size - inp_min);
}

/// This function computes the effective filter length given the minimum filter
/// index \p flt_min and the maximum filter index \p flt_max.
static inline __attribute__((always_inline)) ssize_t
libjit_conv_flt_len(ssize_t flt_min, ssize_t flt_max) {
  return MAX(0, flt_max - flt_min);
}

#endif // GLOW_LLVMIRCODEGEN_LIBJIT_LIBJIT_DEFS_H
