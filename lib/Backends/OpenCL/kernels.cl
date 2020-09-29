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

/// This type is always 32 bits.
typedef unsigned cl_uint32_t;
/// This type is always 64 bits.
typedef unsigned long cl_uint64_t;

typedef int cl_int32_t;
typedef char cl_int8_t;
typedef unsigned char cl_uint8_t;

/// The types of elements should be always matching the definitions of
/// ShapeNHWC in Type.h
typedef struct {
  dim_t n; // Number of samples
  dim_t h; // Height
  dim_t w; // Width
  dim_t c; // Number of channels
} ShapeNHWC;

typedef struct {
  dim_t n; // Number of samples
  dim_t c; // Number of channels
  dim_t h; // Height
  dim_t w; // Width
} ShapeNCHW;

typedef struct {
  dim_t height;
  dim_t width;
} ShapeHW;

/// Helper struct that contains the information for quantization.
typedef struct {
  cl_int32_t pre;
  cl_int32_t post;
  cl_int32_t scale;
  cl_int32_t offset;
} QuantizationTransform32To8;

/// The types of elements should be always matching the definitions of
/// PaddingTLBR in Type.h
typedef struct {
  dim_t top;
  dim_t left;
  dim_t bottom;
  dim_t right;
} PaddingTLBR;

#if defined(cl_khr_int32_base_atomics)
#pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable
#define ATOMICS_32_AVAILABLE
#endif

#if defined(cl_khr_global_int32_base_atomics)
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define ATOMICS_32_AVAILABLE
#endif

#ifdef ATOMICS_32_AVAILABLE
inline void atomicAdd(volatile __global float *source, const float operand) {
  union {
    uint intVal;
    float floatVal;
  } next, expected, current;
  current.floatVal = *source;
  do {
    expected.floatVal = current.floatVal;
    next.floatVal = expected.floatVal + operand;
    current.intVal = atomic_cmpxchg((volatile __global uint *)source,
                                    expected.intVal, next.intVal);
  } while (current.intVal != expected.intVal);
}
#endif

/// \returns the index of the element at n, h, w, c.
dim_t getNHWC(ShapeNHWC s, cl_uint32_t n, cl_uint32_t h, cl_uint32_t w,
              cl_uint32_t c) {
  return (n * s.c * s.w * s.h) + (h * s.c * s.w) + (w * s.c) + c;
}

/// \returns the index of the element at n, c, h, w.
dim_t getNCHW(ShapeNCHW s, cl_uint32_t n, cl_uint32_t c, cl_uint32_t h,
              cl_uint32_t w) {
  return (n * s.c * s.w * s.h) + (c * s.h * s.w) + (h * s.w) + w;
}

/// Scales a 32-bit integer using the integer shift-mult-shift method.
cl_int32_t scale_i32i8(cl_int32_t input, cl_int32_t pre, cl_int32_t post,
                       cl_int32_t scale, cl_int32_t offset) {
  // See more details in libjit_defs.h:libjit_scale_i32i8()
  int rtn = (post > 0) ? (1 << (post - 1)) : 0;
  return ((((input >> pre) * scale) + rtn) >> post) + offset;
}

/// Clips int32_t into int8_t.
cl_int8_t clip(cl_int32_t val) { return (cl_int8_t)min(max(val, -128), 127); }

/// Quantizes \p input from float to int8.
cl_int8_t quantize(float input, float scale, cl_int32_t offset) {
  float result = input / scale + offset;
  return clip((cl_int32_t)round(result));
}

/// Quantizes \p input from float to int32.
cl_int32_t quantize_i32(float input, float scale, cl_int32_t offset) {
  float result = input / scale + offset;
  return (cl_int32_t)round(result);
}

/// Dequantizes \p input from int8 to float.
float dequantize(cl_int8_t input, float scale, cl_int32_t offset) {
  return scale * (input - offset);
}

__kernel void quantize_i8K(__global cl_int8_t *dest, __global float *src,
                           float scale, cl_int32_t offset) {
  dim_t i = get_global_id(0);
  dest[i] = quantize(src[i], scale, offset);
}

__kernel void quantize_i32K(__global cl_int32_t *dest, __global float *src,
                            float scale, cl_int32_t offset) {
  dim_t i = get_global_id(0);
  dest[i] = quantize_i32(src[i], scale, offset);
}

__kernel void quantize_i8W(__global void *mem, cl_uint32_t dest,
                           cl_uint32_t src, float scale, cl_int32_t offset) {
  quantize_i8K(&mem[dest], &mem[src], scale, offset);
}

__kernel void quantize_i32W(__global void *mem, cl_uint32_t dest,
                            cl_uint32_t src, float scale, cl_int32_t offset) {
  quantize_i32K(&mem[dest], &mem[src], scale, offset);
}

__kernel void rescalequantized_i8K(__global cl_int8_t *dest,
                                   __global cl_int8_t *src,
                                   cl_int32_t destOffset, cl_int32_t srcOffset,
                                   cl_int32_t rescalePre,
                                   cl_int32_t rescalePost,
                                   cl_int32_t rescaleScale) {
  dim_t i = get_global_id(0);
  cl_int32_t s = scale_i32i8(src[i] - srcOffset, rescalePre, rescalePost,
                             rescaleScale, destOffset);
  dest[i] = clip(s);
}

__kernel void rescalequantized_i8W(__global void *mem, cl_uint32_t dest,
                                   cl_uint32_t src, cl_int32_t destOffset,
                                   cl_int32_t srcOffset,
                                   QuantizationTransform32To8 rescaleParams) {
  rescalequantized_i8K(&mem[dest], &mem[src], destOffset, srcOffset,
                       rescaleParams.pre, rescaleParams.post,
                       rescaleParams.scale);
}

__kernel void dequantizeK(__global float *dest, __global cl_int8_t *src,
                          float scale, cl_int32_t offset) {
  dim_t i = get_global_id(0);
  dest[i] = dequantize(src[i], scale, offset);
}

__kernel void dequantizeW(__global void *mem, cl_uint32_t dest, cl_uint32_t src,
                          float scale, cl_int32_t offset) {
  dequantizeK(&mem[dest], &mem[src], scale, offset);
}

/// Macro to define a kernel for data-parallel ternay operations. The body of
/// the kernel is auto-generated by the macro.
/// Defines vectorized kernels for vector sizes 1, 8 and 16.
/// \p name the name of the kernel
/// \p type the type of the tensor elements and of the return value
/// \p body the operation to be performed

/// The K16 kernel uses 8-element vector primitives on two contiguous 8-element
/// blocks of data (blocks i and i+1) to produce a 16-element vectorized
/// kernel. Consequently, the global ID needs to be multiplied by 2 so that
/// each kernel invocation will load/store blocks that are not loaded/stored by
/// other kernels. If this is not done, the kernel with global ID 0 will
/// load/store blocks 0 and 1, the kernel with global ID 1 will load/store on
/// blocks 1 and 2, etc.
#define DEFINE_OPENCL_TERNARY_DATA_PARALLEL_KERNEL(name, type, body)           \
  __kernel void name##K##16(__global type * dest, __global type * cond,        \
                            __global type * lhs, __global type * rhs) {        \
    typedef float8 vtype;                                                      \
    dim_t i = get_global_id(0);                                                \
    {                                                                          \
      vtype COND = vload8(i * 2, cond);                                        \
      vtype LHS = vload8(i * 2, lhs);                                          \
      vtype RHS = vload8(i * 2, rhs);                                          \
      vtype VAL = body;                                                        \
      vstore8(VAL, i * 2, dest);                                               \
    }                                                                          \
    {                                                                          \
      vtype COND = vload8(i * 2, cond);                                        \
      vtype LHS = vload8(i * 2 + 1, lhs);                                      \
      vtype RHS = vload8(i * 2 + 1, rhs);                                      \
      vtype VAL = body;                                                        \
      vstore8(VAL, i * 2 + 1, dest);                                           \
    }                                                                          \
  }                                                                            \
  __kernel void name##W##16(__global void *mem, cl_uint32_t dest,              \
                            cl_uint32_t cond, cl_uint32_t lhs,                 \
                            cl_uint32_t rhs) {                                 \
    name##K##16(&mem[dest], &mem[cond], &mem[lhs], &mem[rhs]);                 \
  }                                                                            \
  __kernel void name##K##8(__global type * dest, __global type * cond,         \
                           __global type * lhs, __global type * rhs) {         \
    typedef float8 vtype;                                                      \
    dim_t i = get_global_id(0);                                                \
    vtype COND = vload8(i, cond);                                              \
    vtype LHS = vload8(i, lhs);                                                \
    vtype RHS = vload8(i, rhs);                                                \
    vtype VAL = body;                                                          \
    vstore8(VAL, i, dest);                                                     \
  }                                                                            \
  __kernel void name##W##8(__global void *mem, cl_uint32_t dest,               \
                           cl_uint32_t cond, cl_uint32_t lhs,                  \
                           cl_uint32_t rhs) {                                  \
    name##K##8(&mem[dest], &mem[cond], &mem[lhs], &mem[rhs]);                  \
  }                                                                            \
  __kernel void name##K(__global type *dest, __global type *cond,              \
                        __global type *lhs, __global type *rhs) {              \
    typedef float vtype;                                                       \
    dim_t i = get_global_id(0);                                                \
    vtype COND = cond[i];                                                      \
    vtype RHS = rhs[i];                                                        \
    vtype LHS = lhs[i];                                                        \
    dest[i] = body;                                                            \
  }                                                                            \
  __kernel void name##W(__global void *mem, cl_uint32_t dest,                  \
                        cl_uint32_t cond, cl_uint32_t lhs, cl_uint32_t rhs) {  \
    name##K(&mem[dest], &mem[cond], &mem[lhs], &mem[rhs]);                     \
  }

/// Macro to define a kernel for data-parallel binary operations. The body of
/// the kernel is auto-generated by the macro.
/// Defines vectorized kernels for vector sizes 1, 8 and 16.
/// \p name the name of the kernel
/// \p type the type of the tensor elements and of the return value
/// \p body the operation to be performed

/// The K16 kernel uses 8-element vector primitives on two contiguous 8-element
/// blocks of data (blocks i and i+1) to produce a 16-element vectorized
/// kernel. Consequently, the global ID needs to be multiplied by 2 so that
/// each kernel invocation will load/store blocks that are not loaded/stored by
/// other kernels. If this is not done, the kernel with global ID 0 will
/// load/store blocks 0 and 1, the kernel with global ID 1 will load/store on
/// blocks 1 and 2, etc.
#define DEFINE_OPENCL_BINARY_DATA_PARALLEL_KERNEL(name, type, body)            \
  __kernel void name##K##16(__global type * dest, __global type * lhs,         \
                            __global type * rhs) {                             \
    typedef float8 vtype;                                                      \
    dim_t i = get_global_id(0);                                                \
    {                                                                          \
      vtype LHS = vload8(i * 2, lhs);                                          \
      vtype RHS = vload8(i * 2, rhs);                                          \
      vtype VAL = body;                                                        \
      vstore8(VAL, i * 2, dest);                                               \
    }                                                                          \
    {                                                                          \
      vtype LHS = vload8(i * 2 + 1, lhs);                                      \
      vtype RHS = vload8(i * 2 + 1, rhs);                                      \
      vtype VAL = body;                                                        \
      vstore8(VAL, i * 2 + 1, dest);                                           \
    }                                                                          \
  }                                                                            \
  __kernel void name##W##16(__global void *mem, cl_uint32_t dest,              \
                            cl_uint32_t lhs, cl_uint32_t rhs) {                \
    name##K##16(&mem[dest], &mem[lhs], &mem[rhs]);                             \
  }                                                                            \
  __kernel void name##K##8(__global type * dest, __global type * lhs,          \
                           __global type * rhs) {                              \
    typedef float8 vtype;                                                      \
    dim_t i = get_global_id(0);                                                \
    vtype LHS = vload8(i, lhs);                                                \
    vtype RHS = vload8(i, rhs);                                                \
    vtype VAL = body;                                                          \
    vstore8(VAL, i, dest);                                                     \
  }                                                                            \
  __kernel void name##W##8(__global void *mem, cl_uint32_t dest,               \
                           cl_uint32_t lhs, cl_uint32_t rhs) {                 \
    name##K##8(&mem[dest], &mem[lhs], &mem[rhs]);                              \
  }                                                                            \
  __kernel void name##K(__global type *dest, __global type *lhs,               \
                        __global type *rhs) {                                  \
    typedef float vtype;                                                       \
    dim_t i = get_global_id(0);                                                \
    vtype RHS = rhs[i];                                                        \
    vtype LHS = lhs[i];                                                        \
    dest[i] = body;                                                            \
  }                                                                            \
  __kernel void name##W(__global void *mem, cl_uint32_t dest, cl_uint32_t lhs, \
                        cl_uint32_t rhs) {                                     \
    name##K(&mem[dest], &mem[lhs], &mem[rhs]);                                 \
  }

/// Macro to define a kernel for data-parallel binary quantized operations. The
/// body of the kernel is auto-generated by the macro.
/// \p name the name of the kernel
/// \p body the operation to be performed
/// The naming follows the convention of its corresponding implementation in CPU
/// baclend.
#define DEFINE_OPENCL_BINARY_DATA_PARALLEL_KERNEL_QUANTIZED(name, body)        \
  __kernel void name##_i8K(__global cl_int8_t *dest, __global cl_int8_t *lhs,  \
                           __global cl_int8_t *rhs, cl_int32_t destOffset,     \
                           cl_int32_t lhsOffset, cl_int32_t rhsOffset,         \
                           cl_int32_t lhsPre, cl_int32_t lhsPost,              \
                           cl_int32_t lhsScale, cl_int32_t rhsPre,             \
                           cl_int32_t rhsPost, cl_int32_t rhsScale) {          \
    dim_t i = get_global_id(0);                                                \
    cl_int32_t LHS =                                                           \
        scale_i32i8(lhs[i] - lhsOffset, lhsPre, lhsPost, lhsScale, 0);         \
    cl_int32_t RHS =                                                           \
        scale_i32i8(rhs[i] - rhsOffset, rhsPre, rhsPost, rhsScale, 0);         \
    dest[i] = clip((body) + destOffset);                                       \
  }                                                                            \
  __kernel void name##_i8W(                                                    \
      __global void *mem, cl_uint32_t dest, cl_uint32_t lhs, cl_uint32_t rhs,  \
      cl_int32_t destOffset, QuantizationTransform32To8 lhsScaleParams,        \
      QuantizationTransform32To8 rhsScaleParams) {                             \
    name##_i8K(&mem[dest], &mem[lhs], &mem[rhs], destOffset,                   \
               lhsScaleParams.offset, rhsScaleParams.offset,                   \
               lhsScaleParams.pre, lhsScaleParams.post, lhsScaleParams.scale,  \
               rhsScaleParams.pre, rhsScaleParams.post, rhsScaleParams.scale); \
  }

/// Macro to define a mini-kernel for data-parallel multiplicative quantized
/// operations. The body of the kernel is auto-generated by the macro.
/// \p name the name of the kernel
/// \p body the operation to be performed
/// The naming follows the convention of its corresponding implementation in CPU
/// baclend.
#define DEFINE_OPENCL_BINARY_DATA_PARALLEL_KERNEL_QUANTIZED_M(name, body)      \
  __kernel void name##_i8K(__global cl_int8_t *dest, __global cl_int8_t *lhs,  \
                           __global cl_int8_t *rhs, cl_int32_t destOffset,     \
                           cl_int32_t lhsOffset, cl_int32_t rhsOffset,         \
                           cl_int32_t pre, cl_int32_t post,                    \
                           cl_int32_t scale) {                                 \
    dim_t i = get_global_id(0);                                                \
    cl_int32_t LHS = lhs[i] - lhsOffset;                                       \
    cl_int32_t RHS = rhs[i] - rhsOffset;                                       \
    dest[i] = clip(scale_i32i8((body), pre, post, scale, destOffset));         \
  }                                                                            \
  __kernel void name##_i8W(                                                    \
      __global void *mem, cl_uint32_t dest, cl_uint32_t lhs, cl_uint32_t rhs,  \
      cl_int32_t destOffset, QuantizationTransform32To8 lhsScaleParams,        \
      QuantizationTransform32To8 rhsScaleParams,                               \
      QuantizationTransform32To8 resultScaleParams) {                          \
    name##_i8K(&mem[dest], &mem[lhs], &mem[rhs], destOffset,                   \
               lhsScaleParams.offset, rhsScaleParams.offset,                   \
               resultScaleParams.pre, resultScaleParams.post,                  \
               resultScaleParams.scale);                                       \
  }

/// Macro to define a kernel for data-parallel unary operations. The body of
/// the kernel is auto-generated by the macro.
/// Defines vectorized kernels for vector sizes 1, 8 and 16.
/// \p name the name of the kernel
/// \p type the type of the tensor elements and of the return value
/// \p body the operation to be performed

/// The K16 kernel uses 8-element vector primitives on two contiguous 8-element
/// blocks of data (blocks i and i+1) to produce a 16-element vectorized
/// kernel. Consequently, the global ID needs to be multiplied by 2 so that
/// each kernel invocation will load/store blocks that are not loaded/stored by
/// other kernels. If this is not done, the kernel with global ID 0 will
/// load/store blocks 0 and 1, the kernel with global ID 1 will load/store on
/// blocks 1 and 2, etc.
#define DEFINE_OPENCL_UNARY_DATA_PARALLEL_KERNEL(name, type, body)             \
  __kernel void name##K##16(__global type * dest, __global type * src) {       \
    typedef float8 vtype;                                                      \
    dim_t i = get_global_id(0);                                                \
    {                                                                          \
      vtype SRC = vload8(i * 2, src);                                          \
      vtype VAL = body;                                                        \
      vstore8(VAL, i * 2, dest);                                               \
    }                                                                          \
    {                                                                          \
      vtype SRC = vload8(i * 2 + 1, src);                                      \
      vtype VAL = body;                                                        \
      vstore8(VAL, i * 2 + 1, dest);                                           \
    }                                                                          \
  }                                                                            \
  __kernel void name##W##16(__global void *mem, cl_uint32_t dest,              \
                            cl_uint32_t src) {                                 \
    name##K##16(&mem[dest], &mem[src]);                                        \
  }                                                                            \
  __kernel void name##K##8(__global type * dest, __global type * src) {        \
    typedef float8 vtype;                                                      \
    dim_t i = get_global_id(0);                                                \
    vtype SRC = vload8(i, src);                                                \
    vtype VAL = body;                                                          \
    vstore8(VAL, i, dest);                                                     \
  }                                                                            \
  __kernel void name##W##8(__global void *mem, cl_uint32_t dest,               \
                           cl_uint32_t src) {                                  \
    name##K##8(&mem[dest], &mem[src]);                                         \
  }                                                                            \
  __kernel void name##K(__global type *dest, __global type *src) {             \
    typedef float vtype;                                                       \
    dim_t i = get_global_id(0);                                                \
    vtype SRC = src[i];                                                        \
    dest[i] = body;                                                            \
  }                                                                            \
  __kernel void name##W(__global void *mem, cl_uint32_t dest,                  \
                        cl_uint32_t src) {                                     \
    name##K(&mem[dest], &mem[src]);                                            \
  }

/// Macro to define a kernel for data-parallel unary quantized operations. The
/// body of the kernel is auto-generated by the macro.
/// \p name the name of the kernel
/// \p body the operation to be performed
/// The naming follows the convention of its corresponding implementation in CPU
/// baclend.
#define DEFINE_OPENCL_UNARY_DATA_PARALLEL_KERNEL_QUANTIZED(name, body)         \
  __kernel void name##_i8K(__global cl_int8_t *dest, __global cl_int8_t *src,  \
                           cl_int32_t destOffset, cl_int32_t srcOffset,        \
                           cl_int32_t srcPre, cl_int32_t srcPost,              \
                           cl_int32_t srcScale) {                              \
    dim_t i = get_global_id(0);                                                \
    cl_int32_t SRC =                                                           \
        scale_i32i8(src[i] - srcOffset, srcPre, srcPost, srcScale, 0);         \
    dest[i] = clip((body) + destOffset);                                       \
  }                                                                            \
  __kernel void name##_i8W(__global void *mem, cl_uint32_t dest,               \
                           cl_uint32_t src, cl_int32_t destOffset,             \
                           QuantizationTransform32To8 srcScaleParams) {        \
    name##_i8K(&mem[dest], &mem[src], destOffset, srcScaleParams.offset,       \
               srcScaleParams.pre, srcScaleParams.post, srcScaleParams.scale); \
  }

/// Macro to define a kernel for data-parallel unary operations with an
/// immediate operand. The body of the kernel is auto-generated by the macro.
/// Defines vectorized kernels for vector sizes 1, 8 and 16.
/// \p name the name of the kernel
/// \p type the type of the tensor elements and of the return value
/// \p body the operation to be performed

/// The K16 kernel uses 8-element vector primitives on two contiguous 8-element
/// blocks of data (blocks i and i+1) to produce a 16-element vectorized
/// kernel. Consequently, the global ID needs to be multiplied by 2 so that
/// each kernel invocation will load/store blocks that are not loaded/stored by
/// other kernels. If this is not done, the kernel with global ID 0 will
/// load/store blocks 0 and 1, the kernel with global ID 1 will load/store on
/// blocks 1 and 2, etc.
#define DEFINE_OPENCL_UNARY_DATA_PARALLEL_KERNEL_WITH_IMM_OPERAND(name, type,  \
                                                                  body)        \
  __kernel void name##K##16(__global type * dest, type val) {                  \
    typedef type##8 vtype;                                                     \
    dim_t i = get_global_id(0);                                                \
    {                                                                          \
      vtype SRC = (vtype)val;                                                  \
      vtype VAL = body;                                                        \
      vstore8(VAL, i * 2, dest);                                               \
    }                                                                          \
    {                                                                          \
      vtype SRC = (vtype)val;                                                  \
      vtype VAL = body;                                                        \
      vstore8(VAL, i * 2 + 1, dest);                                           \
    }                                                                          \
  }                                                                            \
  __kernel void name##W##16(__global void *mem, cl_uint32_t dest, float val) { \
    name##K##16(&mem[dest], (type)val);                                        \
  }                                                                            \
  __kernel void name##K##8(__global type * dest, type val) {                   \
    typedef type##8 vtype;                                                     \
    dim_t i = get_global_id(0);                                                \
    vtype SRC = (vtype)val;                                                    \
    vtype VAL = body;                                                          \
    vstore8(VAL, i, dest);                                                     \
  }                                                                            \
  __kernel void name##W##8(__global void *mem, cl_uint32_t dest, float val) {  \
    name##K##8(&mem[dest], (type)val);                                         \
  }                                                                            \
  __kernel void name##K(__global type *dest, type val) {                       \
    typedef type vtype;                                                        \
    dim_t i = get_global_id(0);                                                \
    vtype SRC = (vtype)val;                                                    \
    dest[i] = body;                                                            \
  }                                                                            \
  __kernel void name##W(__global void *mem, cl_uint32_t dest, float val) {     \
    name##K(&mem[dest], (type)val);                                            \
  }

DEFINE_OPENCL_BINARY_DATA_PARALLEL_KERNEL(elementadd, float, LHS + RHS)
DEFINE_OPENCL_BINARY_DATA_PARALLEL_KERNEL(elementsub, float, LHS - RHS)
DEFINE_OPENCL_BINARY_DATA_PARALLEL_KERNEL(elementmul, float, LHS *RHS)
DEFINE_OPENCL_BINARY_DATA_PARALLEL_KERNEL(elementdiv, float, LHS / RHS)
DEFINE_OPENCL_BINARY_DATA_PARALLEL_KERNEL(elementmax, float, max(LHS, RHS))
DEFINE_OPENCL_BINARY_DATA_PARALLEL_KERNEL(elementmin, float, min(LHS, RHS))
DEFINE_OPENCL_BINARY_DATA_PARALLEL_KERNEL(elementpow, float, pow(LHS, RHS))

DEFINE_OPENCL_BINARY_DATA_PARALLEL_KERNEL_QUANTIZED(elementadd, LHS + RHS)
DEFINE_OPENCL_BINARY_DATA_PARALLEL_KERNEL_QUANTIZED(elementsub, LHS - RHS)
DEFINE_OPENCL_BINARY_DATA_PARALLEL_KERNEL_QUANTIZED(elementmax, max(LHS, RHS))
DEFINE_OPENCL_BINARY_DATA_PARALLEL_KERNEL_QUANTIZED(elementmin, min(LHS, RHS))
DEFINE_OPENCL_BINARY_DATA_PARALLEL_KERNEL_QUANTIZED_M(elementmul, LHS *RHS)
DEFINE_OPENCL_BINARY_DATA_PARALLEL_KERNEL_QUANTIZED_M(elementdiv, LHS / RHS)

DEFINE_OPENCL_UNARY_DATA_PARALLEL_KERNEL(tanh, float,
                                         1 - 2 / (exp(SRC * 2) + 1))
DEFINE_OPENCL_UNARY_DATA_PARALLEL_KERNEL(sigmoid, float, 1 / (1 + exp(-SRC)))
DEFINE_OPENCL_UNARY_DATA_PARALLEL_KERNEL(relu, float, max(SRC, 0.0f))

DEFINE_OPENCL_UNARY_DATA_PARALLEL_KERNEL_QUANTIZED(relu, max(SRC, 0))

DEFINE_OPENCL_UNARY_DATA_PARALLEL_KERNEL_WITH_IMM_OPERAND(splat, float, SRC)
DEFINE_OPENCL_UNARY_DATA_PARALLEL_KERNEL_WITH_IMM_OPERAND(splat_u, ulong, SRC)
DEFINE_OPENCL_UNARY_DATA_PARALLEL_KERNEL_WITH_IMM_OPERAND(splat_i8, char, SRC)

#undef DEFINE_OPENCL_UNARY_DATA_PARALLEL_KERNEL_WITH_IMM_OPERAND
#undef DEFINE_OPENCL_BINARY_DATA_PARALLEL_KERNEL
#undef DEFINE_OPENCL_BINARY_DATA_PARALLEL_KERNEL_QUANTIZED
#undef DEFINE_OPENCL_BINARY_DATA_PARALLEL_KERNEL_QUANTIZED_M
#undef DEFINE_OPENCL_UNARY_DATA_PARALLEL_KERNEL
#undef DEFINE_OPENCL_UNARY_DATA_PARALLEL_KERNEL_QUANTIZED

__kernel void elementselectK16(__global float *dest, __global cl_int8_t *cond,
                               __global float *lhs, __global float *rhs) {
  // This kernel uses 8-element vector primitives on two contiguous 8-element
  // blocks of data (blocks i and i+1) to produce a 16-element vectorized
  // kernel. Consequently, the global ID needs to be multiplied by 2 so that
  // each kernel invocation will load/store blocks that are not loaded/stored by
  // other kernels. If this is not done, the kernel with global ID 0 will
  // load/store blocks 0 and 1, the kernel with global ID 1 will load/store on
  // blocks 1 and 2, etc.
  dim_t i = 2 * get_global_id(0);
  float8 zeroVec = (float8)(0, 0, 0, 0, 0, 0, 0, 0);

  vstore8(select(vload8(i, rhs), vload8(i, lhs),
                 isnotequal(convert_float8(vload8(i, cond)), zeroVec)),
          i, dest);
  vstore8(select(vload8(i + 1, rhs), vload8(i + 1, lhs),
                 isnotequal(convert_float8(vload8(i + 1, cond)), zeroVec)),
          i + 1, dest);
}

__kernel void elementselectW16(__global void *mem, cl_uint32_t dest,
                               cl_uint32_t cond, cl_uint32_t lhs,
                               cl_uint32_t rhs) {
  elementselectK16(&mem[dest], &mem[cond], &mem[lhs], &mem[rhs]);
}

__kernel void elementselectK8(__global float *dest, __global cl_int8_t *cond,
                              __global float *lhs, __global float *rhs) {
  dim_t i = get_global_id(0);
  float8 zeroVec = (float8)(0, 0, 0, 0, 0, 0, 0, 0);

  vstore8(select(vload8(i, rhs), vload8(i, lhs),
                 isnotequal(convert_float8(vload8(i, cond)), zeroVec)),
          i, dest);
}

__kernel void elementselectW8(__global void *mem, cl_uint32_t dest,
                              cl_uint32_t cond, cl_uint32_t lhs,
                              cl_uint32_t rhs) {
  elementselectK8(&mem[dest], &mem[cond], &mem[lhs], &mem[rhs]);
}

__kernel void elementselectK(__global float *dest, __global cl_int8_t *cond,
                             __global float *lhs, __global float *rhs) {
  dim_t i = get_global_id(0);
  cl_int8_t c = cond[i];
  float RHS = rhs[i];
  float LHS = lhs[i];
  dest[i] = (c != 0) ? LHS : RHS;
}

__kernel void elementselectW(__global void *mem, cl_uint32_t dest,
                             cl_uint32_t cond, cl_uint32_t lhs,
                             cl_uint32_t rhs) {
  elementselectK(&mem[dest], &mem[cond], &mem[lhs], &mem[rhs]);
}

__kernel void elementcmplteK16(__global cl_int8_t *dest, __global float *LHS,
                               __global float *RHS) {
  // This kernel uses 8-element vector primitives on two contiguous 8-element
  // blocks of data (blocks i and i+1) to produce a 16-element vectorized
  // kernel. Consequently, the global ID needs to be multiplied by 2 so that
  // each kernel invocation will load/store blocks that are not loaded/stored by
  // other kernels. If this is not done, the kernel with global ID 0 will
  // load/store blocks 0 and 1, the kernel with global ID 1 will load/store on
  // blocks 1 and 2, etc.
  dim_t i = 2 * get_global_id(0);
  vstore8(convert_char8(islessequal(vload8(i, LHS), vload8(i, RHS))), i, dest);
  vstore8(convert_char8(islessequal(vload8(i + 1, LHS), vload8(i + 1, RHS))),
          i + 1, dest);
}

__kernel void elementcmplteW16(__global void *mem, cl_uint32_t dest,
                               cl_uint32_t LHS, cl_uint32_t RHS) {
  elementcmplteK16(&mem[dest], &mem[LHS], &mem[RHS]);
}

__kernel void elementcmplteK8(__global cl_int8_t *dest, __global float *LHS,
                              __global float *RHS) {
  dim_t i = get_global_id(0);
  vstore8(convert_char8(islessequal(vload8(i, LHS), vload8(i, RHS))), i, dest);
}

__kernel void elementcmplteW8(__global void *mem, cl_uint32_t dest,
                              cl_uint32_t LHS, cl_uint32_t RHS) {
  elementcmplteK8(&mem[dest], &mem[LHS], &mem[RHS]);
}

__kernel void elementcmplteK(__global cl_int8_t *dest, __global float *LHS,
                             __global float *RHS) {
  dim_t i = get_global_id(0);
  dest[i] = LHS[i] <= RHS[i];
}

__kernel void elementcmplteW(__global void *mem, cl_uint32_t dest,
                             cl_uint32_t LHS, cl_uint32_t RHS) {
  elementcmplteK(&mem[dest], &mem[LHS], &mem[RHS]);
}

__kernel void oclbatchedreduceaddK(__global float *dest, __global float *batch,
                                   __global cl_int32_t *destSliceSizes,
                                   __global cl_int32_t *batchSliceSizes,
                                   cl_uint32_t numSlices,
                                   cl_uint32_t axisSliceSize) {
  dim_t workDim = get_work_dim();

  // This is the component of the offset into batch that depends only on the
  // kernel's global IDs.
  dim_t batchOffset = 0;

  // This is the offset into dest. It depends only on the kernel's global IDs.
  dim_t destOffset = 0;

  // Compute batchOffset and destOffset by multiplying the kernel's global IDs
  // with the corresponding batch and dest slice sizes.
  //
  // For example, suppose the input shape is {3, 4, 5} and the reduce axis is 1.
  // Then, the output shape is {3, 5}. In this case, batchSliceSizes is {4 * 5 =
  // 20, 1} (axis 1 is missing) and destSliceSizes is {5, 1}. The global
  // workspace this kernel was launched with has dimensions {3, 5} (one for each
  // output element). A kernel with IDs {i, j} will add together elements
  // {i, 0..4, j} and store the result in element {i, j}, so (i * 20 + j * 1)
  // will be a component of every offset it uses to access batch, and (i * 5 + j
  // * 1) will be the offset it uses to access dest. This is precisely what
  // batchOffset and destOffset are. The loop below precomputes these offsets
  // before the actual reduce.
  for (dim_t i = 0; i < workDim; ++i) {
    dim_t id = get_global_id(i);
    batchOffset += id * batchSliceSizes[i];
    destOffset += id * destSliceSizes[i];
  }

  // Perform the actual reduce. Add the slice number * the slice size at the
  // axis index to batchOffset to get the elements to add together.
  dest[destOffset] = 0;
  for (dim_t n = 0; n < numSlices; n++) {
    dest[destOffset] += batch[n * axisSliceSize + batchOffset];
  }
}

__kernel void
oclbatchedreduceaddW(__global void *mem, cl_uint32_t dest, cl_uint32_t batch,
                     cl_uint32_t destSliceSizes, cl_uint32_t batchSliceSizes,
                     cl_uint32_t numSlices, cl_uint32_t axisSliceSize) {
  oclbatchedreduceaddK(&mem[dest], &mem[batch], &mem[destSliceSizes],
                       &mem[batchSliceSizes], numSlices, axisSliceSize);
}

__kernel void batchedaddK(__global float *dest, __global float *batch,
                          __global float *slice, cl_uint32_t numSlice,
                          cl_uint32_t sliceSize) {
  dim_t s = get_global_id(0);
  for (dim_t n = 0; n < numSlice; n++) {
    dest[n * sliceSize + s] = batch[n * sliceSize + s] + slice[s];
  }
}

__kernel void batchedaddW(__global void *mem, cl_uint32_t dest,
                          cl_uint32_t batch, cl_uint32_t slice,
                          cl_uint32_t numSlice, cl_uint32_t sliceSize) {
  batchedaddK(&mem[dest], &mem[batch], &mem[slice], numSlice, sliceSize);
}

__kernel void batchedadd_i8K(__global cl_int8_t *dest,
                             __global cl_int8_t *batch,
                             __global cl_int8_t *slice, cl_uint32_t numSlice,
                             cl_uint32_t sliceSize, cl_int32_t destOffset,
                             cl_int32_t batchOffset, cl_int32_t sliceOffset,
                             cl_int32_t batchPre, cl_int32_t batchPost,
                             cl_int32_t batchScale, cl_int32_t slicePre,
                             cl_int32_t slicePost, cl_int32_t sliceScale) {
  dim_t s = get_global_id(0);
  for (dim_t n = 0; n < numSlice; n++) {
    cl_int32_t batchVal = batch[n * sliceSize + s] - batchOffset;
    cl_int32_t sliceVal = slice[s] - sliceOffset;
    cl_int32_t x = scale_i32i8(batchVal, batchPre, batchPost, batchScale, 0);
    cl_int32_t y = scale_i32i8(sliceVal, slicePre, slicePost, sliceScale, 0);
    dest[n * sliceSize + s] = clip(x + y + destOffset);
  }
}

__kernel void batchedadd_i8W(__global void *mem, cl_uint32_t dest,
                             cl_uint32_t batch, cl_uint32_t slice,
                             cl_uint32_t numSlice, cl_uint32_t sliceSize,
                             cl_int32_t destOffset,
                             QuantizationTransform32To8 batchScaleParams,
                             QuantizationTransform32To8 sliceScaleParams) {
  batchedadd_i8K(&mem[dest], &mem[batch], &mem[slice], numSlice, sliceSize,
                 destOffset, batchScaleParams.offset, sliceScaleParams.offset,
                 batchScaleParams.pre, batchScaleParams.post,
                 batchScaleParams.scale, sliceScaleParams.pre,
                 sliceScaleParams.post, sliceScaleParams.scale);
}

__kernel void batchedadd_i8K_32(
    __global cl_int8_t *dest, __global cl_int8_t *batch,
    __global cl_int32_t *slice, cl_uint32_t numSlice, cl_uint32_t sliceSize,
    cl_int32_t destOffset, cl_int32_t batchOffset, cl_int32_t sliceOffset,
    cl_int32_t batchPre, cl_int32_t batchPost, cl_int32_t batchScale,
    cl_int32_t slicePre, cl_int32_t slicePost, cl_int32_t sliceScale) {
  dim_t s = get_global_id(0);
  for (dim_t n = 0; n < numSlice; n++) {
    cl_int32_t batchVal = batch[n * sliceSize + s] - batchOffset;
    cl_int32_t sliceVal = slice[s] - sliceOffset;
    cl_int32_t x = scale_i32i8(batchVal, batchPre, batchPost, batchScale, 0);
    cl_int32_t y = scale_i32i8(sliceVal, slicePre, slicePost, sliceScale, 0);
    dest[n * sliceSize + s] = clip(x + y + destOffset);
  }
}

__kernel void batchedadd_i8W_32(__global void *mem, cl_uint32_t dest,
                                cl_uint32_t batch, cl_uint32_t slice,
                                cl_uint32_t numSlice, cl_uint32_t sliceSize,
                                cl_int32_t destOffset,
                                QuantizationTransform32To8 batchScaleParams,
                                QuantizationTransform32To8 sliceScaleParams) {
  batchedadd_i8K_32(
      &mem[dest], &mem[batch], &mem[slice], numSlice, sliceSize, destOffset,
      batchScaleParams.offset, sliceScaleParams.offset, batchScaleParams.pre,
      batchScaleParams.post, batchScaleParams.scale, sliceScaleParams.pre,
      sliceScaleParams.post, sliceScaleParams.scale);
}

/// Size of the tile to be used for matrix multiplication.
/// The kernel can only be executed by the OpenCL backends that allow
/// workgroups with sizes which are at least as big as a tile.
#define TILE_SIZE 8

__kernel void matmul_tiled(__global void *mem, cl_uint32_t C_off,
                           cl_uint32_t A_off, cl_uint32_t B_off, ShapeNHWC ddim,
                           ShapeNHWC ldim, ShapeNHWC rdim) {
  __global float *C = &mem[C_off];
  __global float *A = &mem[A_off];
  __global float *B = &mem[B_off];

  int M = ldim.n;
  int N = rdim.h;
  int K = ldim.h;

  int tx = get_local_id(1);
  int ty = get_local_id(0);
  int row = get_global_id(0);
  int col = get_global_id(1);

  // Tile of LHS.
  __local float sA[TILE_SIZE][TILE_SIZE];
  // Tile of RHS.
  __local float sB[TILE_SIZE][TILE_SIZE];

  float sum = 0;
  for (int t = 0; t < (K - 1) / TILE_SIZE + 1; t += 1) {
    // Load LHS tile.
    if (row < M && t * TILE_SIZE + tx < K) {
      sA[ty][tx] = A[row * K + (t * TILE_SIZE + tx)];
    } else {
      sA[ty][tx] = 0;
    }

    // Load RHS tile and store it transposed.
    if (t * TILE_SIZE + ty < K && col < N) {
      sB[tx][ty] = B[(t * TILE_SIZE + ty) * N + col];
    } else {
      sB[tx][ty] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
      sum += sA[ty][k] * sB[tx][k];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

__kernel void matmul_tiled_i8(__global void *mem, cl_uint32_t C_off,
                              cl_uint32_t A_off, cl_uint32_t B_off,
                              ShapeNHWC ddim, ShapeNHWC ldim, ShapeNHWC rdim,
                              cl_int32_t aOffset, cl_int32_t bOffset,
                              cl_int32_t cOffset,
                              QuantizationTransform32To8 destScaleParams) {
  __global cl_int8_t *C = &mem[C_off];
  __global cl_int8_t *A = &mem[A_off];
  __global cl_int8_t *B = &mem[B_off];

  int M = ldim.n;
  int N = rdim.h;
  int K = ldim.h;

  int tx = get_local_id(1);
  int ty = get_local_id(0);
  int row = get_global_id(0);
  int col = get_global_id(1);

  // Tile of LHS.
  __local cl_int32_t sA[TILE_SIZE][TILE_SIZE];
  // Tile of RHS.
  __local cl_int32_t sB[TILE_SIZE][TILE_SIZE];

  cl_int32_t sum = 0;
  for (int t = 0; t < (K - 1) / TILE_SIZE + 1; t += 1) {
    // Load LHS tile.
    if (row < M && t * TILE_SIZE + tx < K) {
      sA[ty][tx] = A[row * K + (t * TILE_SIZE + tx)] - aOffset;
    } else {
      sA[ty][tx] = aOffset;
    }

    // Load RHS tile and store it transposed.
    if (t * TILE_SIZE + ty < K && col < N) {
      sB[tx][ty] = B[(t * TILE_SIZE + ty) * N + col] - bOffset;
    } else {
      sB[tx][ty] = bOffset;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
      sum += sA[ty][k] * sB[tx][k];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (row < M && col < N) {
    C[row * N + col] =
        clip(scale_i32i8(sum, destScaleParams.pre, destScaleParams.post,
                         destScaleParams.scale, cOffset));
  }
}
#undef TILE_SIZE

__kernel void matmulK(__global float *dest, __global float *lhs,
                      __global float *rhs, ShapeNHWC ddim, ShapeNHWC ldim,
                      ShapeNHWC rdim) {
  // For each X in the destination matrix.
  dim_t x = get_global_id(0);
  // For each Y in the destination matrix.
  dim_t y = get_global_id(1);

  // Perform DOT on the row an column.
  float sum = 0;
  for (dim_t i = 0; i < ldim.h; i++) {
    sum += lhs[getNHWC(ldim, x, i, 0, 0)] * rhs[getNHWC(rdim, i, y, 0, 0)];
  }

  dest[getNHWC(ddim, x, y, 0, 0)] = sum;
}

__kernel void matmulW(__global void *mem, cl_uint32_t dest, cl_uint32_t lhs,
                      cl_uint32_t rhs, ShapeNHWC ddim, ShapeNHWC ldim,
                      ShapeNHWC rdim) {
  matmulK(&mem[dest], &mem[lhs], &mem[rhs], ddim, ldim, rdim);
}

__kernel void matmul_i8K(__global cl_int8_t *dest, __global cl_int8_t *lhs,
                         __global cl_int8_t *rhs, ShapeNHWC ddim,
                         ShapeNHWC ldim, ShapeNHWC rdim, cl_int32_t lhsOffset,
                         cl_int32_t rhsOffset, cl_int32_t destOffset,
                         cl_int32_t destPre, cl_int32_t destPost,
                         cl_int32_t destScale) {
  // For each X in the destination matrix.
  dim_t x = get_global_id(0);
  // For each Y in the destination matrix.
  dim_t y = get_global_id(1);

  // Perform DOT on the row an column.
  cl_int32_t sum = 0;
  for (dim_t i = 0; i < ldim.h; i++) {
    sum += (lhs[getNHWC(ldim, x, i, 0, 0)] - lhsOffset) *
           (rhs[getNHWC(rdim, i, y, 0, 0)] - rhsOffset);
  }

  dest[getNHWC(ddim, x, y, 0, 0)] =
      clip(scale_i32i8(sum, destPre, destPost, destScale, destOffset));
}

__kernel void matmul_i8W(__global void *mem, cl_uint32_t dest, cl_uint32_t lhs,
                         cl_uint32_t rhs, ShapeNHWC ddim, ShapeNHWC ldim,
                         ShapeNHWC rdim, cl_int32_t lhsOffset,
                         cl_int32_t rhsOffset, cl_int32_t destOffset,
                         QuantizationTransform32To8 destScaleParams) {
  matmul_i8K(&mem[dest], &mem[lhs], &mem[rhs], ddim, ldim, rdim, lhsOffset,
             rhsOffset, destOffset, destScaleParams.pre, destScaleParams.post,
             destScaleParams.scale);
}

//__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void localresponsenormalizationW(__global void *mem, unsigned dest,
                                          unsigned src, unsigned scaleC,
                                          ShapeNHWC dim,
                                          unsigned halfWindowSize, float k,
                                          float beta, float normedAlpha) {

  global float *outW = (global float *)&mem[dest];
  global float *inW = (global float *)&mem[src];
  global float *scaleCache = (global float *)&mem[scaleC];

  unsigned h = get_global_id(0);
  unsigned w = get_global_id(1);
  unsigned c = get_global_id(2);

  // For each input in the batch:
  for (unsigned n = 0; n < dim.n; n++) {
    float squareSum = 0.0;
    for (unsigned i = (c >= halfWindowSize ? c - halfWindowSize : 0);
         i <= min(c + halfWindowSize, (unsigned)dim.c - 1); i++) {
      float val = inW[getNHWC(dim, n, h, w, i)];
      squareSum += val * val;
    }

    float scale = k + normedAlpha * squareSum;

    // This will be used to accelerate the backward pass.
    scaleCache[getNHWC(dim, n, h, w, c)] = scale;

    float normFactor = pow(scale, -beta);
    outW[getNHWC(dim, n, h, w, c)] = inW[getNHWC(dim, n, h, w, c)] * normFactor;
  }
}

__kernel void localresponsenormalizationgradW(__global void *mem, unsigned dest,
                                              unsigned src, unsigned scaleC,
                                              unsigned destGrad,
                                              unsigned srcGrad, ShapeNHWC dim,
                                              unsigned halfWindowSize, float k,
                                              float beta, float normedAlpha) {

  global float *outW = (global float *)&mem[dest];
  global float *outG = (global float *)&mem[destGrad];
  global float *inW = (global float *)&mem[src];
  global float *inG = (global float *)&mem[srcGrad];
  global float *scaleCache = (global float *)&mem[scaleC];

  unsigned n = get_global_id(0);
  unsigned h = get_global_id(1);
  unsigned w = get_global_id(2);

  float sum = 0.0;

  // Compute sum for first channel.
  for (unsigned c = 0; c <= halfWindowSize && c < dim.c; c++) {
    float outw = outW[getNHWC(dim, n, h, w, c)];
    float scale = scaleCache[getNHWC(dim, n, h, w, c)];
    float outg = outG[getNHWC(dim, n, h, w, c)];
    sum += (outg * (outw / scale));
  }

  // For every channel:
  for (unsigned c = 0; c < dim.c; c++) {
    float outg = outG[getNHWC(dim, n, h, w, c)];
    float scale = scaleCache[getNHWC(dim, n, h, w, c)];
    float inw = inW[getNHWC(dim, n, h, w, c)];

    inG[getNHWC(dim, n, h, w, c)] =
        outg * pow(scale, -beta) - 2 * normedAlpha * beta * inw * sum;

    // Modify sum for next channel.
    unsigned subIndex = c - halfWindowSize;
    unsigned addIndex = c + halfWindowSize + 1;

    if (c >= halfWindowSize) {
      float outw = outW[getNHWC(dim, n, h, w, subIndex)];
      float scale = scaleCache[getNHWC(dim, n, h, w, subIndex)];
      float outg = outG[getNHWC(dim, n, h, w, subIndex)];

      // Subtract "rear" end of this window.
      sum -= (outg * (outw / scale));
    }

    if (addIndex < dim.c) {
      float outw = outW[getNHWC(dim, n, h, w, addIndex)];
      float scale = scaleCache[getNHWC(dim, n, h, w, addIndex)];
      float outg = outG[getNHWC(dim, n, h, w, addIndex)];

      // Add "front" end of next window.
      sum += (outg * (outw / scale));
    }
  }
}

__kernel void softmaxK(__global float *dest, __global float *src,
                       __global float *e_cache, cl_uint32_t sliceSize) {
  dim_t i = get_global_id(0);
  float max_ = src[i * sliceSize];
  for (dim_t j = 0; j < sliceSize; j++) {
    max_ = max(max_, src[i * sliceSize + j]);
  }
  float sum = 0;
  for (dim_t j = 0; j < sliceSize; j++) {
    float e = exp(src[i * sliceSize + j] - max_);
    sum += e;
    dest[i * sliceSize + j] = e;
  }
  for (dim_t j = 0; j < sliceSize; j++) {
    dest[i * sliceSize + j] /= sum;
    if (e_cache)
      e_cache[i * sliceSize + j] = dest[i * sliceSize + j];
  }
}

__kernel void softmaxW(__global void *mem, cl_uint32_t dest, cl_uint32_t src,
                       cl_uint32_t sliceSize) {
  softmaxK(&mem[dest], &mem[src], (__global float *)0, sliceSize);
}

__kernel void softmaxgradK(__global float *inG, __global float *outW,
                           __global cl_uint64_t *selectedW,
                           cl_uint32_t sliceSize) {
  dim_t i = get_global_id(0);
  for (dim_t j = 0; j < sliceSize; j++) {
    float delta = (selectedW[i] == j);
    inG[i * sliceSize + j] = outW[i * sliceSize + j] - delta;
  }
}

__kernel void softmaxgradW(__global void *mem, cl_uint32_t origDest,
                           cl_uint32_t origSrc, cl_uint32_t selected,
                           cl_uint32_t srcGrad, cl_uint32_t sliceSize) {
  softmaxgradK(&mem[srcGrad], &mem[origDest], &mem[selected], sliceSize);
}

__kernel void convolutionK(__global float *dest, __global float *src,
                           __global float *filter, __global float *bias,
                           ShapeHW kernelSizes, ShapeHW strides,
                           PaddingTLBR pads, cl_uint32_t group,
                           cl_uint32_t dilation, ShapeNHWC odim, ShapeNHWC idim,
                           ShapeNHWC filterDim, const int fuseReLU) {
  dim_t ax = get_global_id(0);
  dim_t ay = get_global_id(1);
  dim_t d = get_global_id(2);
  dim_t inCperG = idim.c / group;
  dim_t outCperG = odim.c / group;
  dim_t inChannelOffset = d / outCperG * inCperG;

  // For each convolution 'jump' in the input tensor:
  sdim_t x = -(sdim_t)pads.top + ax * strides.height;
  sdim_t y = -(sdim_t)pads.left + ay * strides.width;

  // For each input in the batch:
  for (dim_t n = 0; n < idim.n; n++) {

    // For each element in the convolution-filter:
    float sum = 0;
    for (dim_t fx = 0; fx < kernelSizes.height; fx++) {
      for (dim_t fy = 0; fy < kernelSizes.width; fy++) {
        sdim_t ox = x + fx * dilation;
        sdim_t oy = y + fy * dilation;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= (sdim_t)idim.h || oy >= (sdim_t)idim.w) {
          continue;
        }

        for (dim_t fd = 0; fd < inCperG; fd++) {
          sum +=
              filter[getNHWC(filterDim, d, fx, fy, fd)] *
              src[getNHWC(idim, n, (dim_t)ox, (dim_t)oy, fd + inChannelOffset)];
        }
      }
    }

    sum += bias[d];
    if (fuseReLU) {
      sum = max(sum, 0.0f);
    }
    dest[getNHWC(odim, n, ax, ay, d)] = sum;
  } // N
}

__kernel void convolutionW(__global void *mem, cl_uint32_t dest,
                           cl_uint32_t src, cl_uint32_t filter,
                           cl_uint32_t bias, ShapeHW kernelSizes,
                           ShapeHW strides, PaddingTLBR pads, cl_uint32_t group,
                           cl_uint32_t dilation, ShapeNHWC odim, ShapeNHWC idim,
                           ShapeNHWC filterDim) {
  convolutionK(&mem[dest], &mem[src], &mem[filter], &mem[bias], kernelSizes,
               strides, pads, group, dilation, odim, idim, filterDim, 0);
}

__kernel void convolutionW_ReLU(__global void *mem, cl_uint32_t dest,
                                cl_uint32_t src, cl_uint32_t filter,
                                cl_uint32_t bias, ShapeHW kernelSizes,
                                ShapeHW strides, PaddingTLBR pads,
                                cl_uint32_t group, cl_uint32_t dilation,
                                ShapeNHWC odim, ShapeNHWC idim,
                                ShapeNHWC filterDim) {
  convolutionK(&mem[dest], &mem[src], &mem[filter], &mem[bias], kernelSizes,
               strides, pads, group, dilation, odim, idim, filterDim,
               /* fuseReLU */ 1);
}

__kernel void
convolution_i8K(__global cl_int8_t *dest, __global cl_int8_t *src,
                __global cl_int8_t *filter, __global cl_int32_t *bias,
                ShapeHW kernelSizes, ShapeHW strides, cl_int32_t destOffset,
                float destScale, cl_int32_t srcOffset, float srcScale,
                cl_int32_t filterOffset, float filterScale,
                cl_int32_t biasOffset, float biasScale, PaddingTLBR pads,
                cl_uint32_t group, cl_uint32_t dilation, ShapeNHWC odim,
                ShapeNHWC idim, ShapeNHWC filterDim, const int fuseReLU) {
  dim_t ax = get_global_id(0);
  dim_t ay = get_global_id(1);
  dim_t d = get_global_id(2);
  dim_t inCperG = idim.c / group;
  dim_t outCperG = odim.c / group;
  dim_t inChannelOffset = d / outCperG * inCperG;

  // For each convolution 'jump' in the input tensor:
  sdim_t x = -(sdim_t)pads.top + ax * strides.height;
  sdim_t y = -(sdim_t)pads.left + ay * strides.width;

  float matMulScale = srcScale * filterScale;

  // For each input in the batch:
  for (dim_t n = 0; n < idim.n; n++) {

    // For each element in the convolution-filter:
    cl_int32_t sum = 0;
    for (dim_t fx = 0; fx < kernelSizes.height; fx++) {
      for (dim_t fy = 0; fy < kernelSizes.width; fy++) {
        sdim_t ox = x + fx * dilation;
        sdim_t oy = y + fy * dilation;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= (sdim_t)idim.h || oy >= (sdim_t)idim.w) {
          continue;
        }

        for (dim_t fd = 0; fd < inCperG; fd++) {
          sum += (filter[getNHWC(filterDim, d, fx, fy, fd)] - filterOffset) *
                 (src[getNHWC(idim, n, (dim_t)ox, (dim_t)oy,
                              fd + inChannelOffset)] -
                  srcOffset);
        }
      }
    }

    sum += round((float)(bias[d] - biasOffset) * (biasScale / matMulScale));
    dest[getNHWC(odim, n, ax, ay, d)] =
        clip(round((float)(sum) * (matMulScale / destScale) + destOffset));
    if (fuseReLU) {
      dest[getNHWC(odim, n, ax, ay, d)] =
          max(dest[getNHWC(odim, n, ax, ay, d)], (cl_int8_t)0);
    }
  }
}

__kernel void
convolution_i8W(__global void *mem, cl_uint32_t dest, cl_uint32_t src,
                cl_uint32_t filter, cl_uint32_t bias, ShapeHW kernelSizes,
                ShapeHW strides, PaddingTLBR pads, cl_uint32_t group,
                cl_uint32_t dilation, ShapeNHWC odim, ShapeNHWC idim,
                ShapeNHWC filterDim, cl_int32_t destOffset, float destScale,
                cl_int32_t srcOffset, float srcScale, cl_int32_t filterOffset,
                float filterScale, cl_int32_t biasOffset, float biasScale) {
  convolution_i8K(&mem[dest], &mem[src], &mem[filter], &mem[bias], kernelSizes,
                  strides, destOffset, destScale, srcOffset, srcScale,
                  filterOffset, filterScale, biasOffset, biasScale, pads, group,
                  dilation, odim, idim, filterDim, /* fuseReLU */ 0);
}

__kernel void convolution_i8W_ReLU(
    __global void *mem, cl_uint32_t dest, cl_uint32_t src, cl_uint32_t filter,
    cl_uint32_t bias, ShapeHW kernelSizes, ShapeHW strides, PaddingTLBR pads,
    cl_uint32_t group, cl_uint32_t dilation, ShapeNHWC odim, ShapeNHWC idim,
    ShapeNHWC filterDim, cl_int32_t destOffset, float destScale,
    cl_int32_t srcOffset, float srcScale, cl_int32_t filterOffset,
    float filterScale, cl_int32_t biasOffset, float biasScale) {
  convolution_i8K(&mem[dest], &mem[src], &mem[filter], &mem[bias], kernelSizes,
                  strides, destOffset, destScale, srcOffset, srcScale,
                  filterOffset, filterScale, biasOffset, biasScale, pads, group,
                  dilation, odim, idim, filterDim, /* fuseReLU */ 1);
}

__kernel void convolutiongradK(const __global float *inW,
                               const __global float *filterW,
                               const __global float *outG, __global float *inG,
                               __global float *filterG, __global float *biasG,
                               ShapeHW kernelSizes, ShapeHW strides,
                               PaddingTLBR pads, cl_uint32_t group,
                               cl_uint32_t dilation, ShapeNHWC inWdims,
                               ShapeNHWC outGdims, ShapeNHWC filterGdims) {

  // ax and ay are coordinates in the tensor outG.
  dim_t ax = get_global_id(0);
  dim_t ay = get_global_id(1);
  dim_t d = get_global_id(2);
  dim_t inCperG = inWdims.c / group;
  dim_t outCperG = outGdims.c / group;
  dim_t inChannelOffset = d / outCperG * inCperG;

  // For each convolution 'jump' in the input tensor:
  sdim_t x = -(sdim_t)pads.top + ax * strides.height;
  sdim_t y = -(sdim_t)pads.left + ay * strides.width;

  // NHWC format is assumed

  // For each input in the batch:
  for (dim_t n = 0; n < outGdims.n; n++) {
    float grad = outG[getNHWC(outGdims, n, ax, ay, d)];

    for (dim_t fx = 0; fx < kernelSizes.height; fx++) {
      for (dim_t fy = 0; fy < kernelSizes.width; fy++) {
        sdim_t ox = x + fx * dilation;
        sdim_t oy = y + fy * dilation;

        if (ox < 0 || oy < 0 || ox >= (sdim_t)inWdims.h ||
            oy >= (sdim_t)inWdims.w) {
          continue;
        }

        for (dim_t fd = 0; fd < inCperG; fd++) {
          atomicAdd(&filterG[getNHWC(filterGdims, d, fx, fy, fd)],
                    inW[getNHWC(inWdims, n, (dim_t)ox, (dim_t)oy,
                                fd + inChannelOffset)] *
                        grad);
          atomicAdd(&inG[getNHWC(inWdims, n, (dim_t)ox, (dim_t)oy,
                                 fd + inChannelOffset)],
                    filterW[getNHWC(filterGdims, d, fx, fy, fd)] * grad);
        }
      }
    }
    atomicAdd(&biasG[d], grad);
  } // N
}

__kernel void convolutiongradW(__global void *mem, cl_uint32_t src,
                               cl_uint32_t filter, cl_uint32_t destGrad,
                               cl_uint32_t srcGrad, cl_uint32_t filterGrad,
                               cl_uint32_t biasGrad, ShapeHW kernelSizes,
                               ShapeHW strides, PaddingTLBR pads,
                               cl_uint32_t group, cl_uint32_t dilation,
                               ShapeNHWC srcDim, ShapeNHWC destGradDim,
                               ShapeNHWC filterGradDim) {
  convolutiongradK(&mem[src], &mem[filter], &mem[destGrad], &mem[srcGrad],
                   &mem[filterGrad], &mem[biasGrad], kernelSizes, strides, pads,
                   group, dilation, srcDim, destGradDim, filterGradDim);
}

__kernel void maxpoolK(__global float *dest, __global float *src,
                       cl_uint32_t kernelSize, cl_uint32_t stride,
                       PaddingTLBR pads, ShapeNHWC odim, ShapeNHWC idim) {
  dim_t ax = get_global_id(0);
  dim_t ay = get_global_id(1);
  dim_t d = get_global_id(2);

  // For each convolution 'jump' in the input tensor:
  sdim_t x = -(sdim_t)pads.top + ax * stride;
  sdim_t y = -(sdim_t)pads.left + ay * stride;

  // For each input in the batch:
  for (dim_t n = 0; n < idim.n; n++) {
    float maxVal = 0;
    bool first = true;

    // For each element in the convolution-filter:
    for (dim_t fx = 0; fx < kernelSize; fx++) {
      for (dim_t fy = 0; fy < kernelSize; fy++) {
        sdim_t ox = x + fx;
        sdim_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= (sdim_t)idim.h || oy >= (sdim_t)idim.w) {
          continue;
        }

        float val = src[getNHWC(idim, n, (dim_t)ox, (dim_t)oy, d)];

        if (first || (val >= maxVal)) {
          first = false;
          maxVal = val;
        }
      }
    }
    dest[getNHWC(odim, n, ax, ay, d)] = maxVal;
  } // N
}

__kernel void maxpoolW(__global void *mem, cl_uint32_t dest, cl_uint32_t src,
                       cl_uint32_t kernelSize, cl_uint32_t stride,
                       PaddingTLBR pads, ShapeNHWC odim, ShapeNHWC idim) {
  maxpoolK(&mem[dest], &mem[src], kernelSize, stride, pads, odim, idim);
}

/// Macro to define a kernel for oclmaxpool. The body of
/// the kernel is auto-generated by the macro.
/// \p name the name of this kernel
/// \p type the type of the tensor elements and of the return value
#define DEFINE_OPENCL_MAXPOOL_KERNEL(name, type)                               \
  __kernel void name##K(__global type *dest, __global type *src,               \
                        cl_uint32_t kernelSize, cl_uint32_t stride,            \
                        PaddingTLBR pads, ShapeNCHW odim, ShapeNCHW idim) {    \
    dim_t ax = get_global_id(0);                                               \
    dim_t ay = get_global_id(1);                                               \
    dim_t d = get_global_id(2);                                                \
    /* For each convolution 'jump' in the input tensor: */                     \
    sdim_t x = -(sdim_t)pads.top + ax * stride;                                \
    sdim_t y = -(sdim_t)pads.left + ay * stride;                               \
    /* For each input in the batch: */                                         \
    for (dim_t n = 0; n < idim.n; n++) {                                       \
      type maxVal = 0;                                                         \
      bool first = true;                                                       \
      /* For each element in the convolution-filter: */                        \
      for (dim_t fx = 0; fx < kernelSize; fx++) {                              \
        for (dim_t fy = 0; fy < kernelSize; fy++) {                            \
          sdim_t ox = x + fx;                                                  \
          sdim_t oy = y + fy;                                                  \
          /* Ignore index access below zero (this is due to padding). */       \
          if (ox < 0 || oy < 0 || ox >= (sdim_t)idim.h ||                      \
              oy >= (sdim_t)idim.w) {                                          \
            continue;                                                          \
          }                                                                    \
          type val = src[getNCHW(idim, n, d, (dim_t)ox, (dim_t)oy)];           \
          if (first || (val >= maxVal)) {                                      \
            first = false;                                                     \
            maxVal = val;                                                      \
          }                                                                    \
        }                                                                      \
      }                                                                        \
      dest[getNCHW(odim, n, d, ax, ay)] = maxVal;                              \
    }                                                                          \
  }                                                                            \
  __kernel void name##W(__global void *mem, cl_uint32_t dest, cl_uint32_t src, \
                        cl_uint32_t kernelSize, cl_uint32_t stride,            \
                        PaddingTLBR pads, ShapeNCHW odim, ShapeNCHW idim) {    \
    name##K(&mem[dest], &mem[src], kernelSize, stride, pads, odim, idim);      \
  }
DEFINE_OPENCL_MAXPOOL_KERNEL(oclmaxpool, float)
DEFINE_OPENCL_MAXPOOL_KERNEL(oclmaxpool_i8, char)
#undef DEFINE_OPENCL_BINARY_DATA_PARALLEL_KERNEL

__kernel void maxpoolwithargmaxK(__global float *dest, __global float *src,
                                 __global cl_uint64_t *argmax,
                                 cl_uint32_t kernelSize, cl_uint32_t stride,
                                 PaddingTLBR pads, ShapeNHWC odim,
                                 ShapeNHWC idim) {
  dim_t ax = get_global_id(0);
  dim_t ay = get_global_id(1);
  dim_t d = get_global_id(2);

  // For each convolution 'jump' in the input tensor:
  sdim_t x = -(sdim_t)pads.top + ax * stride;
  sdim_t y = -(sdim_t)pads.left + ay * stride;

  // For each input in the batch:
  for (dim_t n = 0; n < idim.n; n++) {
    float maxVal = 0;
    bool first = true;
    cl_uint64_t argmaxNHWC = 0;

    // For each element in the convolution-filter:
    for (dim_t fx = 0; fx < kernelSize; fx++) {
      for (dim_t fy = 0; fy < kernelSize; fy++) {
        sdim_t ox = x + fx;
        sdim_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= (sdim_t)idim.h || oy >= (sdim_t)idim.w) {
          continue;
        }

        const dim_t flatIndex = getNHWC(idim, n, (dim_t)ox, (dim_t)oy, d);
        float val = src[flatIndex];

        if (first || (val >= maxVal)) {
          first = false;
          maxVal = val;
          argmaxNHWC = flatIndex;
        }
      }
    }
    const dim_t flatIndex = getNHWC(odim, n, ax, ay, d);
    dest[flatIndex] = maxVal;
    argmax[flatIndex] = argmaxNHWC;
  } // N
}

__kernel void maxpoolwithargmaxW(__global void *mem, cl_uint32_t dest,
                                 cl_uint32_t src, cl_uint32_t argmax,
                                 cl_uint32_t kernelSize, cl_uint32_t stride,
                                 PaddingTLBR pads, ShapeNHWC odim,
                                 ShapeNHWC idim) {
  maxpoolwithargmaxK(&mem[dest], &mem[src], &mem[argmax], kernelSize, stride,
                     pads, odim, idim);
}

__kernel void maxpoolwithargmaxgradK(__global float *dest,
                                     __global cl_uint64_t *argmax,
                                     __global float *destGrad,
                                     __global float *srcGrad,
                                     cl_uint32_t kernelSize, cl_uint32_t stride,
                                     PaddingTLBR pads, ShapeNHWC srcGradDim,
                                     ShapeNHWC destGradDim) {
  dim_t n = get_global_id(0);

  // NHWC format is assumed
  for (dim_t z = 0; z < destGradDim.c; z++) {
    // Clear srcGrad
    for (dim_t x = 0; x < srcGradDim.h; x++) {
      for (dim_t y = 0; y < srcGradDim.w; y++) {
        srcGrad[getNHWC(srcGradDim, n, x, y, z)] = 0.0;
      }
    }

    for (dim_t ax = 0; ax < destGradDim.h; ax++) {
      for (dim_t ay = 0; ay < destGradDim.w; ay++) {
        // Reuse precomputed linear index of max element from argmax.
        const dim_t flatIndex = getNHWC(destGradDim, n, ax, ay, z);
        float df = destGrad[flatIndex];
        srcGrad[argmax[flatIndex]] += df;
      } // W
    }   // H
  }     // C
}

__kernel void maxpoolwithargmaxgradW(__global void *mem, cl_uint32_t dest,
                                     cl_uint32_t src, cl_uint32_t argmax,
                                     cl_uint32_t destGrad, cl_uint32_t srcGrad,
                                     cl_uint32_t kernelSize, cl_uint32_t stride,
                                     PaddingTLBR pads, ShapeNHWC srcGradDim,
                                     ShapeNHWC destDim) {
  // src operand is present on the instruction but not needed by the OpenCL
  // kernel.
  maxpoolwithargmaxgradK(&mem[dest], &mem[argmax], &mem[destGrad],
                         &mem[srcGrad], kernelSize, stride, pads, srcGradDim,
                         destDim);
}

__kernel void avgpoolK(__global float *dest, __global float *src,
                       cl_uint32_t kernelSize, cl_uint32_t stride,
                       PaddingTLBR pads, ShapeNHWC odim, ShapeNHWC idim,
                       bool countIncludePads) {
  dim_t ax = get_global_id(0);
  dim_t ay = get_global_id(1);
  dim_t d = get_global_id(2);

  // For each convolution 'jump' in the input tensor:
  sdim_t x = -(sdim_t)pads.top + ax * stride;
  sdim_t y = -(sdim_t)pads.left + ay * stride;

  float rawFilterArea = kernelSize * kernelSize;

  // For each input in the batch:
  for (dim_t n = 0; n < idim.n; n++) {
    float sumVal = 0;
    float filterArea = rawFilterArea;

    // For each element in the convolution-filter:
    for (dim_t fx = 0; fx < kernelSize; fx++) {
      for (dim_t fy = 0; fy < kernelSize; fy++) {
        sdim_t ox = x + fx;
        sdim_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= (sdim_t)idim.h || oy >= (sdim_t)idim.w) {
          if (!countIncludePads) {
            filterArea--;
          }
          continue;
        }

        sumVal += src[getNHWC(idim, n, (dim_t)ox, (dim_t)oy, d)];
      }
    }
    assert(filterArea != 0 && "FilterArea shouldn't be 0");
    dest[getNHWC(odim, n, ax, ay, d)] = sumVal / filterArea;
  } // N
}

__kernel void avgpoolW(__global void *mem, cl_uint32_t dest, cl_uint32_t src,
                       cl_uint32_t kernelSize, cl_uint32_t stride,
                       PaddingTLBR pads, ShapeNHWC odim, ShapeNHWC idim,
                       bool countIncludePads) {
  avgpoolK(&mem[dest], &mem[src], kernelSize, stride, pads, odim, idim,
           countIncludePads);
}

__kernel void oclavgpoolK(__global float *dest, __global float *src,
                          cl_uint32_t kernelSize, cl_uint32_t stride,
                          PaddingTLBR pads, ShapeNCHW odim, ShapeNCHW idim,
                          bool countIncludePads) {
  dim_t ax = get_global_id(0);
  dim_t ay = get_global_id(1);
  dim_t d = get_global_id(2);

  // For each convolution 'jump' in the input tensor:
  sdim_t x = -(sdim_t)pads.top + ax * stride;
  sdim_t y = -(sdim_t)pads.left + ay * stride;

  float rawFilterArea = kernelSize * kernelSize;

  // For each input in the batch:
  for (dim_t n = 0; n < idim.n; n++) {
    float sumVal = 0;
    float filterArea = rawFilterArea;

    // For each element in the convolution-filter:
    for (dim_t fx = 0; fx < kernelSize; fx++) {
      for (dim_t fy = 0; fy < kernelSize; fy++) {
        sdim_t ox = x + fx;
        sdim_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= (sdim_t)idim.h || oy >= (sdim_t)idim.w) {
          if (!countIncludePads) {
            filterArea--;
          }
          continue;
        }

        sumVal += src[getNCHW(idim, n, d, (dim_t)ox, (dim_t)oy)];
      }
    }
    assert(filterArea != 0 && "FilterArea shouldn't be 0");
    dest[getNCHW(odim, n, d, ax, ay)] = sumVal / filterArea;
  } // N
}

__kernel void oclavgpoolW(__global void *mem, cl_uint32_t dest, cl_uint32_t src,
                          cl_uint32_t kernelSize, cl_uint32_t stride,
                          PaddingTLBR pads, ShapeNCHW odim, ShapeNCHW idim,
                          bool countIncludePads) {
  oclavgpoolK(&mem[dest], &mem[src], kernelSize, stride, pads, odim, idim,
              countIncludePads);
}

__kernel void oclavgpool_i8K(__global cl_int8_t *dest, __global cl_int8_t *src,
                             cl_uint32_t kernelSize, cl_uint32_t stride,
                             PaddingTLBR pads, ShapeNCHW odim, ShapeNCHW idim,
                             cl_int32_t srcOffset, cl_int32_t destOffset,
                             cl_int32_t destPre, cl_int32_t destPost,
                             cl_int32_t destScale) {
  dim_t ax = get_global_id(0);
  dim_t ay = get_global_id(1);
  dim_t d = get_global_id(2);

  sdim_t x = -(sdim_t)pads.top + ax * stride;
  sdim_t y = -(sdim_t)pads.left + ay * stride;

  // For each input in the batch:
  for (dim_t n = 0; n < idim.n; n++) {
    cl_int32_t sumVal = 0;
    for (dim_t fx = 0; fx < kernelSize; fx++) {
      for (dim_t fy = 0; fy < kernelSize; fy++) {
        sdim_t ox = x + fx;
        sdim_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= (sdim_t)idim.h || oy >= (sdim_t)idim.w) {
          continue;
        }

        sumVal += src[getNCHW(idim, n, d, (dim_t)ox, (dim_t)oy)] - srcOffset;
      }
    }
    // All dest scale params are already divided by kernel*kernel(filter area),
    // we don't need to divide the sumVal by filter area here.
    dest[getNCHW(odim, n, d, ax, ay)] =
        clip(scale_i32i8(sumVal, destPre, destPost, destScale, destOffset));
  }
}

__kernel void oclavgpool_i8W(__global void *mem, cl_uint32_t dest,
                             cl_uint32_t src, cl_uint32_t kernelSize,
                             cl_uint32_t stride, PaddingTLBR pads,
                             ShapeNCHW odim, ShapeNCHW idim,
                             cl_int32_t srcOffset,
                             QuantizationTransform32To8 destScaleParams) {
  oclavgpool_i8K(&mem[dest], &mem[src], kernelSize, stride, pads, odim, idim,
                 srcOffset, destScaleParams.offset, destScaleParams.pre,
                 destScaleParams.post, destScaleParams.scale);
}

/// Macro to define a kernel for transpose operations. The body of
/// the kernel is auto-generated by the macro.
/// \p type the type of the tensor elements and of the return value
#define DEFINE_OPENCL_TRANSPOSE_KERNEL(name, type)                             \
  __kernel void name##K(__global type *dest, __global type *src,               \
                        ShapeNHWC odim, ShapeNHWC idim, ShapeNHWC shuffle) {   \
    dim_t res[4];                                                              \
    dim_t d0 = get_global_id(0);                                               \
    dim_t d1 = get_global_id(1);                                               \
    res[0] = d0;                                                               \
    res[1] = d1;                                                               \
    for (dim_t d2 = 0; d2 < idim.w; d2++) {                                    \
      res[2] = d2;                                                             \
      for (dim_t d3 = 0; d3 < idim.c; d3++) {                                  \
        res[3] = d3;                                                           \
        dim_t dstIdx = getNHWC(odim, res[shuffle.n], res[shuffle.h],           \
                               res[shuffle.w], res[shuffle.c]);                \
        dim_t srcIdx = getNHWC(idim, d0, d1, d2, d3);                          \
        dest[dstIdx] = src[srcIdx];                                            \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  __kernel void name##W(__global void *mem, cl_uint32_t dest, cl_uint32_t src, \
                        ShapeNHWC odim, ShapeNHWC idim, ShapeNHWC shuffle) {   \
    name##K(&mem[dest], &mem[src], odim, idim, shuffle);                       \
  }

DEFINE_OPENCL_TRANSPOSE_KERNEL(transpose_i8, cl_int8_t)
DEFINE_OPENCL_TRANSPOSE_KERNEL(transpose_u, cl_uint64_t)
DEFINE_OPENCL_TRANSPOSE_KERNEL(transpose, float)

#undef DEFINE_OPENCL_TRANSPOSE_KERNEL

/// Macro to define a kernel to insert tensors. The body of
/// the kernel is auto-generated by the macro.
/// \p name the name of the tensor
/// \p type the type of the tensor elements
#define DEFINE_OPENCL_INSERT_TENSOR_KERNEL(name, type)                         \
  __kernel void name##K(__global type *dest, __global type *src,               \
                        ShapeNHWC odim, ShapeNHWC idim, ShapeNHWC offset,      \
                        cl_uint32_t count, cl_uint32_t axis) {                 \
    dim_t d0 = get_global_id(0);                                               \
    dim_t d1 = get_global_id(1);                                               \
    dim_t offset_n = ((odim.n > 1) ? offset.n : 0);                            \
    dim_t offset_h = ((odim.h > 1) ? offset.h : 0);                            \
    dim_t offset_w = ((odim.w > 1) ? offset.w : 0);                            \
    dim_t offset_c = ((odim.c > 1) ? offset.c : 0);                            \
    for (dim_t c = 0; c < count; c++) {                                        \
      dim_t count_offset_n = (axis == 0) ? c * idim.n : 0;                     \
      dim_t count_offset_h = (axis == 1) ? c * idim.h : 0;                     \
      dim_t count_offset_w = (axis == 2) ? c * idim.w : 0;                     \
      dim_t count_offset_c = (axis == 3) ? c * idim.c : 0;                     \
      for (dim_t d2 = 0; d2 < idim.w; d2++) {                                  \
        for (dim_t d3 = 0; d3 < idim.c; d3++) {                                \
          dim_t r0 = d0 + offset_n + count_offset_n;                           \
          dim_t r1 = d1 + offset_h + count_offset_h;                           \
          dim_t r2 = d2 + offset_w + count_offset_w;                           \
          dim_t r3 = d3 + offset_c + count_offset_c;                           \
          dim_t srcIdx = getNHWC(idim, d0, d1, d2, d3);                        \
          dim_t destIdx = getNHWC(odim, r0, r1, r2, r3);                       \
          dest[destIdx] = src[srcIdx];                                         \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  __kernel void name##W(__global void *mem, cl_uint32_t dest, cl_uint32_t src, \
                        ShapeNHWC odim, ShapeNHWC idim, ShapeNHWC offset,      \
                        cl_uint32_t count, cl_uint32_t axis) {                 \
    name##K(&mem[dest], &mem[src], odim, idim, offset, count, axis);           \
  }
DEFINE_OPENCL_INSERT_TENSOR_KERNEL(inserttensor, float)
DEFINE_OPENCL_INSERT_TENSOR_KERNEL(inserttensor_i8, char)
#undef DEFINE_OPENCL_INSERT_TENSOR_KERNEL

/// Macro to define a kernel to extract tensors. The body of
/// the kernel is auto-generated by the macro.
/// \p name the name of the tensor
/// \p type the type of the tensor elements
#define DEFINE_OPENCL_EXTRACT_TENSOR_KERNEL(name, type)                        \
  __kernel void name##K(__global type *dest, __global type *src,               \
                        ShapeNHWC odim, ShapeNHWC idim, ShapeNHWC offset) {    \
    dim_t d0 = get_global_id(0);                                               \
    dim_t d1 = get_global_id(1);                                               \
    dim_t offset_w = ((odim.w > 1) ? offset.w : 0);                            \
    dim_t offset_c = ((odim.c > 1) ? offset.c : 0);                            \
    for (dim_t d2 = 0; d2 < odim.w; d2++) {                                    \
      for (dim_t d3 = 0; d3 < odim.c; d3++) {                                  \
        dim_t r0 = d0 + offset.n;                                              \
        dim_t r1 = d1 + offset.h;                                              \
        dim_t r2 = d2 + offset_w;                                              \
        dim_t r3 = d3 + offset_c;                                              \
        dim_t destIdx = getNHWC(odim, d0, d1, d2, d3);                         \
        dim_t srcIdx = getNHWC(idim, r0, r1, r2, r3);                          \
        dest[destIdx] = src[srcIdx];                                           \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  __kernel void name##W(__global void *mem, cl_uint32_t dest, cl_uint32_t src, \
                        ShapeNHWC odim, ShapeNHWC idim, ShapeNHWC offset) {    \
    name##K(&mem[dest], &mem[src], odim, idim, offset);                        \
  }
DEFINE_OPENCL_EXTRACT_TENSOR_KERNEL(extracttensor, float)
DEFINE_OPENCL_EXTRACT_TENSOR_KERNEL(extracttensor_i8, char)
#undef DEFINE_OPENCL_EXTRACT_TENSOR_KERNEL

void memcpy_float(__global float *dest, const __global float *src, int len) {
  for (int i = 0; i < len; i++) {
    dest[i] = src[i];
  }
}

__kernel void gatherK(__global float *dest, __global const float *src,
                      __global cl_uint64_t *indices, cl_uint32_t numIndices,
                      cl_uint32_t sliceSize, cl_uint32_t numSamples,
                      cl_uint32_t destSampleSize, cl_uint32_t srcSampleSize) {
  int idx = get_global_id(0);
  cl_uint64_t slice = indices[idx];
  // For each sample in our batch:
  for (dim_t sample = 0; sample < numSamples; sample++) {
    dim_t srcSampleStart = sample * srcSampleSize;
    dim_t destSampleStart = sample * destSampleSize;
    memcpy_float(dest + destSampleStart + idx * sliceSize,
                 src + srcSampleStart + slice * sliceSize, sliceSize);
  }
}

__kernel void gatherW(__global void *mem, cl_uint32_t dest, cl_uint32_t src,
                      cl_uint32_t indices, cl_uint32_t numIndices,
                      cl_uint32_t sliceSize, cl_uint32_t numSamples,
                      cl_uint32_t destSampleSize, cl_uint32_t srcSampleSize) {
  gatherK(&mem[dest], &mem[src], &mem[indices], numIndices, sliceSize,
          numSamples, destSampleSize, srcSampleSize);
}

__kernel void scatterdataK(__global float *data, __global cl_uint64_t *indices,
                           __global const float *slices,
                           cl_uint32_t sliceSize) {
  int idx = get_global_id(0);
  cl_uint64_t destDataIdx = indices[idx];
  memcpy_float(data + destDataIdx * sliceSize, slices + idx * sliceSize,
               sliceSize);
}

__kernel void scatterdataW(__global void *mem, cl_uint32_t data,
                           cl_uint32_t indices, cl_uint32_t slices,
                           cl_uint32_t sliceSize) {
  scatterdataK(&mem[data], &mem[indices], &mem[slices], sliceSize);
}

__kernel void sparselengthsweightedsumK(__global float *dest,
                                        __global float *data,
                                        __global float *weights,
                                        __global cl_uint64_t *indices,
                                        __global cl_int32_t *lengths,
                                        cl_uint32_t sliceSize) {
  // Get the global ID. This corresponds to the segment that this kernel will
  // compute, and therefore also the index into the output buffer at which this
  // kernel will write its output.
  cl_int32_t idx = get_global_id(0);

  // Add together the segment lengths of the segments being processed by kernels
  // with global ID 0 .. idx - 1. This is the index into the indices buffer
  // at which *this* kernel should start reading indices with which to index
  // into the data buffer and weights that the data should be multiplied by.
  cl_int32_t curIdx = 0;
  for (cl_int32_t i = 0; i < idx; ++i) {
    curIdx += lengths[i];
  }

  // For each index in the segment:
  for (cl_int32_t j = 0; j < lengths[idx]; ++j) {
    // Read the weight.
    float weight = weights[curIdx];
    // Read the data index.
    cl_uint64_t dataIdx = indices[curIdx];

    // Read an entire slice of data at index dataIdx, multiply it by the weight
    // and store it into dest at index idx (same as the segment number).
    for (cl_uint32_t k = 0; k < sliceSize; ++k) {
      dest[idx * sliceSize + k] += weight * data[dataIdx * sliceSize + k];
    }

    // Increment the index.
    curIdx++;
  }
}

__kernel void sparselengthsweightedsumW(__global void *mem, cl_uint32_t dest,
                                        cl_uint32_t data, cl_uint32_t weights,
                                        cl_uint32_t indices,
                                        cl_uint32_t lengths,
                                        cl_uint32_t dataSliceSize) {
  sparselengthsweightedsumK(&mem[dest], &mem[data], &mem[weights],
                            &mem[indices], &mem[lengths], dataSliceSize);
}

__kernel void sparselengthsweightedsumgradK(
    __global float *destGrad, __global float *dataGrad,
    __global float *weightsGrad, __global float *data, __global float *weights,
    __global cl_uint64_t *indices, __global cl_int32_t *lengths,
    cl_uint32_t segments, cl_uint32_t sliceSize) {

  // For each segment:
  for (cl_uint32_t i = 0, curIdx = 0; i < segments; ++i) {
    // For each index in the segment:
    for (cl_int32_t j = 0; j < lengths[i]; ++j, ++curIdx) {
      float weightGrad = 0.0f;
      // Get the weight for the current index.
      float weight = weights[curIdx];
      // Get the index into dataGrad corresponding to the current index.
      cl_uint64_t dataGradIdx = indices[curIdx];

      // Read an entire slice of destGrad at index i (same as the segment
      // number), multiply it by the weight (that helped produce the
      // corresponding slice of dest during the forward pass) and store it into
      // dataGrad at index dataGradIdx. In the same loop, accumulate into each
      // weight gradient the reduced sum of the elementwise product of the
      // result slice that the corresponding weight produced and the input
      // slice that the weight was multiplied with.
      for (cl_uint32_t k = 0; k < sliceSize; ++k) {
        dataGrad[dataGradIdx * sliceSize + k] +=
            weight * destGrad[i * sliceSize + k];
        weightGrad +=
            destGrad[i * sliceSize + k] * data[dataGradIdx * sliceSize + k];
      }

      weightsGrad[curIdx] = weightGrad;
    }
  }
}

__kernel void
sparselengthsweightedsumgradW(__global void *mem, cl_uint32_t data,
                              cl_uint32_t weights, cl_uint32_t indices,
                              cl_uint32_t lengths, cl_uint32_t destGrad,
                              cl_uint32_t dataGrad, cl_uint32_t weightsGrad,
                              cl_uint32_t segments, cl_uint32_t sliceSize) {
  sparselengthsweightedsumgradK(
      &mem[destGrad], &mem[dataGrad], &mem[weightsGrad], &mem[data],
      &mem[weights], &mem[indices], &mem[lengths], segments, sliceSize);
}

/// An empty kernel used as a checkpoint for TraceEvents.
__kernel void checkpoint(__global void *mem) {}
