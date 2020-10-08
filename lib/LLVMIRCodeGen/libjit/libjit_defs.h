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

#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "libjit_dim_t.h"

#define LIBJIT_ALWAYS_INLINE static inline __attribute__((always_inline))

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

/// Computes the function Sigmoid(x) for float \p input.
/// When the LIBJIT compile option "-ffast-math" is enabled the intermediate
/// computation expf(x) for Sigmoid operator is not handled properly for very
/// large positive values which results in NaN values for the Sigmoid output.
/// Therefore when the "-ffast-math" is enabled we compute the Sigmoid such that
/// we avoid computing large values for the "expf" function.
LIBJIT_ALWAYS_INLINE
float libjit_sigmoid_f(float input) {
#ifdef FFAST_MATH
  float sigmoidVal = 1 / (1 + expf(-std::abs(input)));
  return (float)(std::signbit(input)) + std::copysignf(sigmoidVal, input);
#else
  float e = expf(-input);
  return 1 / (e + 1);
#endif // FFAST_MATH
}

/// Computes the function Tanh(x) for float \p input.
/// When the LIBJIT compile option "-ffast-math" is enabled the intermediate
/// computation expf(x) for Tanh operator is not handled properly for very
/// large positive values which results in NaN values for the Tanh output.
/// Therefore when the "-ffast-math" is enabled we compute the Tanh such that
/// we avoid computing large values for the "expf" function.
LIBJIT_ALWAYS_INLINE
float libjit_tanh_f(float input) {
#ifdef FFAST_MATH
  float tanhVal = -1 + 2 / (expf(-2 * std::abs(input)) + 1);
  return std::copysignf(tanhVal, input);
#else
  return 1 - 2 / (expf(input * 2) + 1);
#endif // FFAST_MATH
}

/// \returns the clipped value of the input to INT8 range [-128, 127].
inline int8_t libjit_clip(int32_t val) {
  return (int8_t)MIN(MAX(val, -128), 127);
}

/// Scales a 32-bit integer using the integer shift-mult-shift method.
/// See QuantizationTransform32To8 for more details.
LIBJIT_ALWAYS_INLINE
int32_t libjit_scale_i32i8(int32_t input, int32_t pre, int32_t post,
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

/// Applies an activation function to a FLOAT input value \p input based on
/// the activation type \p actType and the activation arguments \p actArgs.
/// NOTE: The type of the activation must be in sync with the FusedActivation
/// enumeration in glow\include\glow\Graph\Nodes.h.
LIBJIT_ALWAYS_INLINE
float libjit_activation_f(float input, int32_t actType, const float *actArgs) {
  if (actType == 0) {
    // No activation.
    return input;
  } else if (actType == 1) {
    // Relu.
    return MAX(input, 0);
  } else if (actType == 2) {
    // Clip.
    return MIN(MAX(input, actArgs[0]), actArgs[1]);
  } else if (actType == 3) {
    // Tanh.
    return libjit_tanh_f(input);
  } else if (actType == 4) {
    // Sigmoid.
    return libjit_sigmoid_f(input);
  } else {
    // LeakyRelu.
    return (input >= 0) ? input : actArgs[0] * input;
  }
}

/// Applies an activation function to a QUANTIZED input value \p input based on
/// the activation type \p actType and the activation arguments \p actArgs.
/// NOTE: The type of the activation must be in sync with the FusedActivation
/// enumeration in glow\include\glow\Graph\Nodes.h.
LIBJIT_ALWAYS_INLINE
int32_t libjit_activation_i32(int32_t input, int32_t offset, int32_t actType,
                              const int32_t *actArgs) {
  if (actType == 0) {
    // No activation.
    return input;
  } else if (actType == 1) {
    // Relu.
    return MAX(input, offset);
  } else if (actType == 2) {
    // Clip.
    return MIN(MAX(input, actArgs[0]), actArgs[1]);
  } else if (actType == 3) {
    // Tanh.
    assert(false && "Fused Tanh for quantized type not supported!");
    return input;
  } else if (actType == 4) {
    // Sigmoid.
    assert(false && "Fused Sigmoid for quantized type not supported!");
    return input;
  } else {
    // LeakyRelu.
    return (input >= offset)
               ? input
               : libjit_scale_i32i8(input - offset, actArgs[0], actArgs[1],
                                    actArgs[2], offset);
  }
}

#ifdef _WIN32
#define libjit_aligned_malloc(p, a, s)                                         \
  (((*(p)) = _aligned_malloc((s), (a))), *(p) ? 0 : errno)
#define libjit_aligned_free(p) _aligned_free(p)
#else
#define libjit_aligned_malloc(p, a, s) posix_memalign(p, a, s)
#define libjit_aligned_free(p) free(p)
#endif

#endif // GLOW_LLVMIRCODEGEN_LIBJIT_LIBJIT_DEFS_H
