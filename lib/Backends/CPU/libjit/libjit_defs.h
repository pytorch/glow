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
#ifndef GLOW_BACKENDS_CPU_LIBJIT_DEFS_H
#define GLOW_BACKENDS_CPU_LIBJIT_DEFS_H

#include <stdint.h>
#include <string.h>

#if defined(_MSC_VER)
typedef __declspec(align(16)) float float4;
typedef __declspec(align(32)) float float8;
typedef __declspec(align(64)) float float16;
#else
typedef float float4 __attribute__((ext_vector_type(4)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef float float16 __attribute__((aligned(64)));
#endif

#ifdef _MSC_VER
/*
 * Microsoft Visual C/C++ Compiler doesn't support for VLA, using _alloca
 * instead.
 * https://stackoverflow.com/questions/46878963/allocate-aligned-memory-on-the-stack-like-alloca
 */
extern "C" void *__cdecl _alloca(size_t _Size);
#define __LIB_JIT_alloca_aligned(size, alignment_minus_1) \
  ((UINT_PTR)_alloca((size) + (alignment_minus_1)) + (alignment_minus_1))     \
   & ~((UINT_PTR)(alignment_minus_1))

#define LIBJIT_VLA(type, name, size)                                          \
  type *name = (type *)(__LIB_JIT_alloca_aligned(                             \
                        sizeof(type) * (size),                                \
                        __alignof(type) - 1))          

#endif /* _MSC_VER */

#ifndef LIBJIT_VLA
#define LIBJIT_VLA(type, name, size) type name[size]
#endif /* !GLOW_VLA */
#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#if defined(_MSC_VER)
#define LIBJIT_NOINLINE __declspec(noinline)
#else
#define LIBJIT_NOINLINE __attribute__((noinline))
#endif

/// Loads a simd float8 value from \p ptr.
#define LoadFloat8(PTR) *((const float8 *)(PTR))

/// Stores the simd float8 value to \p ptr.
#define StoreFloat8(PTR, VAL) *((float8 *)(PTR)) = (VAL);

/// Accumulate (+=) the simd float8 value to \p ptr.
#define AddFloat8(PTR, VAL) *((float8 *)(PTR)) += (VAL);

/// Broadcast the input value to a float8.
#define BroadcastFloat8(VAL) ((float8)(VAL))

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
inline size_t libjit_getXYZWQR(const size_t *dims, size_t x, size_t y, size_t z,
                               size_t w, size_t q, size_t r) {
  return (x * dims[1] * dims[2] * dims[3] * dims[4] * dims[5]) +
         (y * dims[2] * dims[3] * dims[4] * dims[5]) +
         (z * dims[3] * dims[4] * dims[5]) + (w * dims[4] * dims[5]) +
         (q * dims[5]) + r;
}

/// \returns the index of the element at x,y,z,w,q.
inline size_t libjit_getXYZWQ(const size_t *dims, size_t x, size_t y, size_t z,
                              size_t w, size_t q) {
  return (x * dims[1] * dims[2] * dims[3] * dims[4]) +
         (y * dims[2] * dims[3] * dims[4]) + (z * dims[3] * dims[4]) +
         (w * dims[4]) + q;
}

/// \returns the index of the element at x,y,z,w.
inline size_t libjit_getXYZW(const size_t *dims, size_t x, size_t y, size_t z,
                             size_t w) {
  return (x * dims[1] * dims[2] * dims[3]) + (y * dims[2] * dims[3]) +
         (z * dims[3]) + w;
}

/// \returns the index of the element at x,y,z.
inline size_t libjit_getXYZ(const size_t *dims, size_t x, size_t y, size_t z) {
  return (x * dims[1] * dims[2]) + (y * dims[2]) + z;
}

/// \returns the index of the element at x,y.
inline size_t libjit_getXY(const size_t *dims, size_t x, size_t y) {
  return (x * dims[1]) + y;
}

inline int8_t libjit_clip(int32_t val) {
  return (int8_t)MIN(MAX(val, -128), 127);
}

/// Scales a 32-bit integer using the integer shift-mult-shift method.
/// See QuantizationTransform32To8 for more details.
inline int32_t libjit_scale_i32i8(int32_t input, int32_t pre, int32_t post,
                                  int32_t scale, int32_t offset) {
  // The operation x >> y is rounded down to negative infinity. To get to
  // round-nearest we add (1 << (shift - 1)) to the value prior to shifting.
  int rtn = (post > 0) ? (1 << (post - 1)) : 0;

  // NOTICE: If your tests are failing because of signed integer overflow then
  // this is a bug in the test and not in the program. You should make sure that
  // the inputs to the operations do not overflow. The semantics of the
  // quantization process is such that the result for values that fall out of
  // range is undefined. The conversion procedure will only tolerate a few bits
  // of overflow and the result will be clipped.
  return ((((input >> pre) * scale) + rtn) >> post) + offset;
}

#endif // GLOW_BACKENDS_CPU_LIBJIT_DEFS_H
