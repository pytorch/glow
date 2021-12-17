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

#include "glow/Backends/Interpreter/Interpreter.h"

#include "glow/Base/TensorSerialization.h"
#include "glow/Base/Type.h"
#include "glow/IR/Instrs.h"
#include "glow/Quantization/Base/Base.h"
#include "glow/Quantization/Base/Profile.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cmath>
#include <numeric>
#include <random>

#ifdef WIN32
#include <corecrt_math_defines.h>
#endif

using namespace glow;

namespace IntNBitSplitEmbeddingBagsHelper {
inline int32_t unpaddedRowSizeInBytes(int32_t dim,
                                      SplitEmbeddingSparseType weight_ty) {
  if (weight_ty == SplitEmbeddingSparseType::EST_FLOAT) {
    return dim * sizeof(float);
  }
  if (weight_ty == SplitEmbeddingSparseType::EST_FLOAT16) {
    return dim * sizeof(float16_t);
  }
  if (weight_ty == SplitEmbeddingSparseType::EST_INT8) {
    return dim + 2 * sizeof(float16_t);
  }
  if (weight_ty == SplitEmbeddingSparseType::EST_INT4) {
    return dim / 2 + 2 * sizeof(float16_t);
  }
  if (weight_ty == SplitEmbeddingSparseType::EST_INT2) {
    return dim / 4 + 2 * sizeof(float16_t);
  }
  llvm_unreachable("Unsupported SparseType");
}

uint32_t roundUp(uint32_t a, uint32_t b) { return ((a + b - 1) / b) * b; }

inline int32_t paddedRowSizeInBytes(int32_t dim,
                                    SplitEmbeddingSparseType weight_ty) {
  auto r = unpaddedRowSizeInBytes(dim, weight_ty);
  return roundUp(r, 16);
}

template <typename SumTy>
SumTy add(SumTy a, const uint8_t *row, const uint8_t *data,
          SplitEmbeddingSparseType dataTy, bool isMSB, float weight) {
  float sum = a;

  if (dataTy == SplitEmbeddingSparseType::EST_FLOAT) {
    return sum + weight * static_cast<float>(
                              *(reinterpret_cast<const float *>(data)));
  }

  if (dataTy == SplitEmbeddingSparseType::EST_FLOAT16) {
    return sum + weight * static_cast<float>(
                              *(reinterpret_cast<const float16_t *>(data)));
  }

  float scale = *(reinterpret_cast<const float16_t *>(row));
  float offset = *(reinterpret_cast<const float16_t *>(row) + 1);

  if (dataTy == SplitEmbeddingSparseType::EST_INT8) {
    return sum + weight * (static_cast<float>(*data) * scale + offset);
  }

  if (dataTy == SplitEmbeddingSparseType::EST_INT4) {
    if (isMSB) {
      return sum + weight * (static_cast<float>(*data >> 4) * scale + offset);
    }
    return sum + weight * (static_cast<float>(*data & 0xF) * scale + offset);
  }

  llvm_unreachable("Unsuppored SplitEmbeddingSparseType");
}

template <typename DataTy>
void save(DataTy a, uint8_t *data, SplitEmbeddingSparseType dataTy) {
  if (dataTy == SplitEmbeddingSparseType::EST_FLOAT) {
    *(reinterpret_cast<float *>(data)) = a;
  } else if (dataTy == SplitEmbeddingSparseType::EST_FLOAT16) {
    *(reinterpret_cast<float16_t *>(data)) = a;
  }
}

} // namespace IntNBitSplitEmbeddingBagsHelper

#define dispatchImpl(functionName, elemTy, ...)                                \
  switch (elemTy) {                                                            \
  case ElemKind::FloatTy:                                                      \
    functionName<float>(__VA_ARGS__);                                          \
    break;                                                                     \
  case ElemKind::Float16Ty:                                                    \
    functionName<float16_t>(__VA_ARGS__);                                      \
    break;                                                                     \
  case ElemKind::BFloat16Ty:                                                   \
    functionName<bfloat16_t>(__VA_ARGS__);                                     \
    break;                                                                     \
  case ElemKind::Int8QTy:                                                      \
    functionName<int8_t>(__VA_ARGS__);                                         \
    break;                                                                     \
  case ElemKind::Int16QTy:                                                     \
    functionName<int16_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  case ElemKind::Int32QTy:                                                     \
    functionName<int32_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  case ElemKind::Int32ITy:                                                     \
    functionName<int32_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  case ElemKind::Int64ITy:                                                     \
    functionName<int64_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  case ElemKind::BoolTy:                                                       \
    functionName<bool>(__VA_ARGS__);                                           \
    break;                                                                     \
  default:                                                                     \
    llvm_unreachable("Type is not supported");                                 \
  }

#define dispatchFloatingPointImpl(functionName, elemTy, ...)                   \
  switch (elemTy) {                                                            \
  case ElemKind::FloatTy:                                                      \
    functionName<float>(__VA_ARGS__);                                          \
    break;                                                                     \
  case ElemKind::Float16Ty:                                                    \
    functionName<float16_t>(__VA_ARGS__);                                      \
    break;                                                                     \
  case ElemKind::BFloat16Ty:                                                   \
    functionName<bfloat16_t>(__VA_ARGS__);                                     \
    break;                                                                     \
  default:                                                                     \
    llvm_unreachable("Type is not supported");                                 \
  }

#define dispatchFloatingPointAndInt32Impl(functionName, elemTy, ...)           \
  switch (elemTy) {                                                            \
  case ElemKind::FloatTy:                                                      \
    functionName<float>(__VA_ARGS__);                                          \
    break;                                                                     \
  case ElemKind::Float16Ty:                                                    \
    functionName<float16_t>(__VA_ARGS__);                                      \
    break;                                                                     \
  case ElemKind::BFloat16Ty:                                                   \
    functionName<bfloat16_t>(__VA_ARGS__);                                     \
    break;                                                                     \
  case ElemKind::Int32ITy:                                                     \
    functionName<int>(__VA_ARGS__);                                            \
    break;                                                                     \
  default:                                                                     \
    llvm_unreachable("Type is not supported");                                 \
  }

#define dispatchFloatingPointAndIndexImpl(functionName, elemTy, elemTyIndex,   \
                                          ...)                                 \
  switch (elemTy) {                                                            \
  case ElemKind::FloatTy:                                                      \
    if (elemTyIndex == ElemKind::Int64ITy) {                                   \
      functionName<float, int64_t>(__VA_ARGS__);                               \
    } else if (elemTyIndex == ElemKind::Int32ITy) {                            \
      functionName<float, int32_t>(__VA_ARGS__);                               \
    }                                                                          \
    break;                                                                     \
  case ElemKind::Float16Ty:                                                    \
    if (elemTyIndex == ElemKind::Int64ITy) {                                   \
      functionName<float16, int64_t>(__VA_ARGS__);                             \
    } else if (elemTyIndex == ElemKind::Int32ITy) {                            \
      functionName<float16, int32_t>(__VA_ARGS__);                             \
    }                                                                          \
    break;                                                                     \
  case ElemKind::BFloat16Ty:                                                   \
    if (elemTyIndex == ElemKind::Int64ITy) {                                   \
      functionName<bfloat16, int64_t>(__VA_ARGS__);                            \
    } else if (elemTyIndex == ElemKind::Int32ITy) {                            \
      functionName<bfloat16, int32_t>(__VA_ARGS__);                            \
    }                                                                          \
    break;                                                                     \
  default:                                                                     \
    llvm_unreachable("Type is not supported");                                 \
  }

#define dispatchIndexTypeImpl(functionName, elemTy, ...)                       \
  switch (elemTy) {                                                            \
  case ElemKind::Int32ITy:                                                     \
    functionName<int32_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  case ElemKind::Int64ITy:                                                     \
    functionName<int64_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  default:                                                                     \
    llvm_unreachable("Type is not supported");                                 \
  }

#define dispatchIndexAndOutputTypeImpl(functionName, indexTy, outputTy, ...)   \
  switch (indexTy) {                                                           \
  case ElemKind::Int32ITy:                                                     \
    if (outputTy == SplitEmbeddingSparseType::EST_FLOAT) {                     \
      functionName<int32_t, float>(__VA_ARGS__);                               \
    } else if (outputTy == SplitEmbeddingSparseType::EST_FLOAT16) {            \
      functionName<int32_t, float16>(__VA_ARGS__);                             \
    }                                                                          \
    break;                                                                     \
  case ElemKind::Int64ITy:                                                     \
    if (outputTy == SplitEmbeddingSparseType::EST_FLOAT) {                     \
      functionName<int64_t, float>(__VA_ARGS__);                               \
    } else if (outputTy == SplitEmbeddingSparseType::EST_FLOAT16) {            \
      functionName<int64_t, float16>(__VA_ARGS__);                             \
    }                                                                          \
    break;                                                                     \
  default:                                                                     \
    llvm_unreachable("Type is not supported");                                 \
  }

#define dispatchArithmeticImpl(functionName, elemTy, ...)                      \
  switch (elemTy) {                                                            \
  case ElemKind::FloatTy:                                                      \
    functionName<float>(__VA_ARGS__);                                          \
    break;                                                                     \
  case ElemKind::Float16Ty:                                                    \
    functionName<float16_t>(__VA_ARGS__);                                      \
    break;                                                                     \
  case ElemKind::BFloat16Ty:                                                   \
    functionName<bfloat16_t>(__VA_ARGS__);                                     \
    break;                                                                     \
  case ElemKind::Int32ITy:                                                     \
    functionName<int32_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  case ElemKind::Int64ITy:                                                     \
    functionName<int64_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  default:                                                                     \
    llvm_unreachable("Type is not supported");                                 \
  }

#define dispatchBitwiseImpl(functionName, elemTy, ...)                         \
  switch (elemTy) {                                                            \
  case ElemKind::Int32ITy:                                                     \
    functionName<int32_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  case ElemKind::Int64ITy:                                                     \
    functionName<int64_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  case ElemKind::BoolTy:                                                       \
    functionName<bool>(__VA_ARGS__);                                           \
    break;                                                                     \
  default:                                                                     \
    llvm_unreachable("Type is not supported");                                 \
  }

#define dispatchQuantizedImpl(functionName, elemTy, ...)                       \
  switch (elemTy) {                                                            \
  case ElemKind::Int8QTy:                                                      \
    functionName<int8_t>(__VA_ARGS__);                                         \
    break;                                                                     \
  case ElemKind::Int16QTy:                                                     \
    functionName<int16_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  case ElemKind::Int32QTy:                                                     \
    functionName<int32_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  default:                                                                     \
    llvm_unreachable("Type is not supported");                                 \
  }

#define dispatchQuantizedWithAccumulationImpl(functionName, elemTy, ...)       \
  switch (elemTy) {                                                            \
  case ElemKind::Int8QTy:                                                      \
    functionName<int8_t, int32_t>(__VA_ARGS__);                                \
    break;                                                                     \
  case ElemKind::Int16QTy:                                                     \
    functionName<int16_t, int64_t>(__VA_ARGS__);                               \
    break;                                                                     \
  default:                                                                     \
    llvm_unreachable("Type is not supported");                                 \
  }

#define dispatchQuantizedWithAccumulationAndBiasImpl(functionName, elemTy,     \
                                                     biasElemType, ...)        \
  if (elemTy == ElemKind::Int8QTy && biasElemType == ElemKind::Int8QTy) {      \
    functionName<int8_t, int32_t, int8_t>(__VA_ARGS__);                        \
  } else if (elemTy == ElemKind::Int8QTy &&                                    \
             biasElemType == ElemKind::Int32QTy) {                             \
    functionName<int8_t, int32_t, int32_t>(__VA_ARGS__);                       \
  } else if (elemTy == ElemKind::Int16QTy &&                                   \
             biasElemType == ElemKind::Int16QTy) {                             \
    functionName<int16_t, int64_t, int16_t>(__VA_ARGS__);                      \
  } else if (elemTy == ElemKind::Int16QTy &&                                   \
             biasElemType == ElemKind::Int32QTy) {                             \
    functionName<int16_t, int64_t, int32_t>(__VA_ARGS__);                      \
  } else {                                                                     \
    llvm_unreachable("Type is not supported");                                 \
  }

#define staticAssertFloatingPointType(ElemTy)                                  \
  static_assert(                                                               \
      std::is_floating_point<ElemTy>::value ||                                 \
          std::is_same<float16_t,                                              \
                       typename std::remove_cv<ElemTy>::type>::value ||        \
          std::is_same<bfloat16_t,                                             \
                       typename std::remove_cv<ElemTy>::type>::value,          \
      "This implementation is for floating-point values only")

#define staticAssertArithmeticType(ElemTy)                                     \
  static_assert(                                                               \
      std::is_arithmetic<ElemTy>::value ||                                     \
          std::is_same<float16_t,                                              \
                       typename std::remove_cv<ElemTy>::type>::value ||        \
          std::is_same<bfloat16_t,                                             \
                       typename std::remove_cv<ElemTy>::type>::value,          \
      "This implementation is for arithmetic values only")

#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

//===----------------------------------------------------------------------===//
//                       Convolution
//===----------------------------------------------------------------------===//

/// This is the floating point implementation of Convolution.
template <typename ElemTy>
void BoundInterpreterFunction::fwdConvolutionInstFloatImpl(
    Value *inV, Value *outV, Value *filterV, Value *biasV,
    llvm::ArrayRef<unsigned_t> kernelSizes, llvm::ArrayRef<unsigned_t> strides,
    llvm::ArrayRef<unsigned_t> pads, size_t group,
    llvm::ArrayRef<unsigned_t> dilation) {
  staticAssertFloatingPointType(ElemTy);

  auto inW = getWeightHandle<ElemTy>(inV);
  auto outW = getWeightHandle<ElemTy>(outV);
  auto filterW = getWeightHandle<ElemTy>(filterV);
  auto biasW = getWeightHandle<ElemTy>(biasV);

  ShapeNHWC odim(outW.dims());
  ShapeNHWC idim(inW.dims());
  ShapeHW kdim(kernelSizes);
  ShapeHW sdim(strides);

  assert(idim.c % group == 0 && "Input channels must be divisible by group.");
  assert(odim.c % group == 0 && "Output channels must be divisible by group.");
  dim_t inCperG = idim.c / group;
  dim_t outCperG = odim.c / group;

  PaddingTLBR pdim(pads);

  // For each input in the batch:
  for (dim_t n = 0; n < idim.n; n++) {

    // For each group of input channels:
    for (dim_t g = 0; g < group; g++) {

      // For each output channel in the group:
      for (dim_t d = g * outCperG; d < (g + 1) * outCperG; d++) {

        // For each convolution 'jump' in the input tensor:
        ssize_t x = -ssize_t(pdim.top);
        for (dim_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
          ssize_t y = -ssize_t(pdim.left);
          for (dim_t ay = 0; ay < odim.w; y += sdim.width, ay++) {

            // For each element in the convolution-filter:
            float sum = 0;
            for (dim_t fx = 0; fx < kdim.height; fx++) {
              for (dim_t fy = 0; fy < kdim.width; fy++) {
                sdim_t ox = x + fx * dilation[0];
                sdim_t oy = y + fy * dilation[1];

                // Ignore index access below zero (this is due to padding).
                if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                    oy >= ssize_t(idim.w)) {
                  continue;
                }
                for (dim_t fd = 0; fd < inCperG; fd++) {
                  sum += float(
                      filterW.at({d, fx, fy, fd}) *
                      inW.at({n, (dim_t)ox, (dim_t)oy, g * inCperG + fd}));
                }
              }
            }

            sum += float(biasW.at({d}));
            outW.at({n, ax, ay, d}) = ElemTy(sum);
          } // W
        }   // H
      }     // C
    }       // G
  }         // N
}

/// This is the quantized implementation of Convolution.
template <typename ElemTy, typename AccumulatorTy, typename BiasElemTy>
void BoundInterpreterFunction::fwdConvolutionInstQuantizedImpl(
    Value *inV, Value *outV, Value *filterV, Value *biasV,
    llvm::ArrayRef<unsigned_t> kernelSizes, llvm::ArrayRef<unsigned_t> strides,
    llvm::ArrayRef<unsigned_t> pads, size_t group,
    llvm::ArrayRef<unsigned_t> dilation) {
  auto inW = getWeightHandle<ElemTy>(inV);
  auto outW = getWeightHandle<ElemTy>(outV);
  auto filterW = getWeightHandle<ElemTy>(filterV);
  auto biasW = getWeightHandle<BiasElemTy>(biasV);

  ShapeNHWC odim(outW.dims());
  ShapeNHWC idim(inW.dims());
  ShapeHW kdim(kernelSizes);
  ShapeHW sdim(strides);

  assert(idim.c % group == 0 && "Input channels must be divisible by group.");
  assert(odim.c % group == 0 && "Output channels must be divisible by group.");
  dim_t inCperG = idim.c / group;
  dim_t outCperG = odim.c / group;

  PaddingTLBR pdim(pads);
  auto outTy = outV->getType();
  auto inTy = inV->getType();
  auto filterTy = filterV->getType();
  auto biasTy = biasV->getType();

  int32_t outOffset = outTy->getOffset();
  int32_t inOffset = inTy->getOffset();
  int32_t filterOffset = filterTy->getOffset();
  int32_t biasOffset = biasTy->getOffset();

  float outScale = outTy->getScale();
  float inScale = inTy->getScale();
  float filterScale = filterTy->getScale();
  float biasScale = biasTy->getScale();

  // Calculate the scale of the values that come out of the matrix
  // multiplication part of the calculation.
  float matMulScale = inScale * filterScale;

  // For each input in the batch:
  for (dim_t n = 0; n < idim.n; n++) {
    // For each group of input channels:
    for (dim_t g = 0; g < group; g++) {

      // For each output channel in the group:
      for (dim_t d = g * outCperG; d < (g + 1) * outCperG; d++) {

        // For each convolution 'jump' in the input tensor:
        ssize_t x = -ssize_t(pdim.top);
        for (dim_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
          ssize_t y = -ssize_t(pdim.left);
          for (dim_t ay = 0; ay < odim.w; y += sdim.width, ay++) {

            // For each element in the convolution-filter:
            AccumulatorTy sum = 0;
            for (dim_t fx = 0; fx < kdim.height; fx++) {
              for (dim_t fy = 0; fy < kdim.width; fy++) {
                sdim_t ox = x + fx * dilation[0];
                sdim_t oy = y + fy * dilation[1];

                // Ignore index access below zero (this is due to padding).
                if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                    oy >= sdim_t(idim.w)) {
                  continue;
                }
                for (dim_t fd = 0; fd < inCperG; fd++) {

                  AccumulatorTy F = filterW.at({d, fx, fy, fd});
                  AccumulatorTy I =
                      inW.at({n, (dim_t)ox, (dim_t)oy, g * inCperG + fd});
                  // We represent the element multiplication with offset as
                  // (value - offset).
                  sum += (F - filterOffset) * (I - inOffset);
                }
              }
            }

            // Scale the bias to match the scale of the matrix multiplication.
            AccumulatorTy B = std::round(float(biasW.at({d}) - biasOffset) *
                                         (biasScale / matMulScale));

            // Add the bias.
            sum += B;

            // Scale the result back to the expected destination scale.
            outW.at({n, ax, ay, d}) = quantization::clip<AccumulatorTy, ElemTy>(
                std::round(float(sum) * (matMulScale / outScale) + outOffset));
          } // W
        }   // H
      }     // C
    }       // G
  }         // N
}

/// This is the floating point implementation of ConvTranspose.
template <typename ElemTy>
void BoundInterpreterFunction::fwdConvTransposeInstFloatImpl(
    Value *inV, Value *outV, Value *filterV, Value *biasV,
    llvm::ArrayRef<unsigned_t> kernelSizes, llvm::ArrayRef<unsigned_t> strides,
    llvm::ArrayRef<unsigned_t> pads, size_t group,
    llvm::ArrayRef<unsigned_t> dilation) {
  staticAssertFloatingPointType(ElemTy);

  auto inW = getWeightHandle<ElemTy>(inV);
  auto outW = getWeightHandle<ElemTy>(outV);
  auto filterW = getWeightHandle<ElemTy>(filterV);
  auto biasW = getWeightHandle<ElemTy>(biasV);

  ShapeNHWC odim(outW.dims());
  ShapeNHWC idim(inW.dims());
  ShapeHW kdim(kernelSizes);
  ShapeHW sdim(strides);

  assert(idim.c % group == 0 && "Input channels must be divisible by group.");
  assert(odim.c % group == 0 && "Output channels must be divisible by group.");

  dim_t inCperG = idim.c / group;
  dim_t outCperG = odim.c / group;

  PaddingTLBR pdim(pads);

  // For each input in the batch:
  for (dim_t n = 0; n < idim.n; n++) {

    // Initialize bias (TODO take out to a separate function when quant is in).
    for (dim_t ax = 0; ax < odim.h; ax++) {
      for (dim_t ay = 0; ay < odim.w; ay++) {
        for (dim_t d = 0; d < odim.c; d++) {
          outW.at({n, ax, ay, d}) = static_cast<ElemTy>(biasW.at({d}));
        }
      }
    }

    // For each group of input channels:
    for (dim_t g = 0; g < group; g++) {

      // For each input channel in the group:
      for (dim_t d = g * inCperG; d < (g + 1) * inCperG; d++) {

        // For each transposed convolution 'jump' in the input tensor:
        ssize_t x = -ssize_t(pdim.top);
        for (dim_t bx = 0; bx < idim.h; bx++, x += sdim.height) {
          ssize_t y = -ssize_t(pdim.left);
          for (dim_t by = 0; by < idim.w; by++, y += sdim.width) {

            // For each element in the each transposed convolution filter:
            ElemTy input = inW.at({n, bx, by, d});

            for (dim_t kx = 0; kx < kdim.height; kx++) {
              for (dim_t ky = 0; ky < kdim.width; ky++) {
                ssize_t ax = x + kx * dilation[0];
                ssize_t ay = y + ky * dilation[1];

                // Ignore index access below zero (this is due to padding).
                if (ax < 0 || ay < 0 || ax >= ssize_t(odim.h) ||
                    ay >= ssize_t(odim.w)) {
                  continue;
                }
                for (dim_t c = 0; c < outCperG; c++) {
                  outW.at({n, (dim_t)ax, (dim_t)ay, g * outCperG + c}) +=
                      filterW.at({c, kx, ky, d}) * input;
                }
              }
            }
          } // W
        }   // H
      }     // C
    }       // G
  }         // N
}

void BoundInterpreterFunction::fwdConvTransposeInst(
    const ConvTransposeInst *I) {
  auto kernelSizes = I->getKernels();
  auto pads = I->getPads();
  auto strides = I->getStrides();
  size_t group = I->getGroup();

  if (I->getSrc()->getType()->isQuantizedType()) {
    llvm_unreachable("Quantized ConvTranspose not supported");
    return;
  }

  dispatchFloatingPointImpl(
      fwdConvTransposeInstFloatImpl, I->getSrc()->getElementType(), I->getSrc(),
      I->getDest(), I->getFilter(), I->getBias(), kernelSizes, strides, pads,
      group, I->getDilation());
}

void BoundInterpreterFunction::fwdConvolutionInst(const ConvolutionInst *I) {
  auto kernelSizes = I->getKernels();
  auto pads = I->getPads();
  auto strides = I->getStrides();
  size_t group = I->getGroup();

  if (I->getSrc()->getType()->isQuantizedType()) {
    dispatchQuantizedWithAccumulationAndBiasImpl(
        fwdConvolutionInstQuantizedImpl, I->getSrc()->getElementType(),
        I->getBias()->getElementType(), I->getSrc(), I->getDest(),
        I->getFilter(), I->getBias(), kernelSizes, strides, pads, group,
        I->getDilation());
    return;
  }

  dispatchFloatingPointImpl(
      fwdConvolutionInstFloatImpl, I->getSrc()->getElementType(), I->getSrc(),
      I->getDest(), I->getFilter(), I->getBias(), kernelSizes, strides, pads,
      group, I->getDilation());
}

void BoundInterpreterFunction::fwdConcatInst(const ConcatInst *I) {
  (void)I;
  // TODO
  llvm_unreachable("not yet implemented");
}

void BoundInterpreterFunction::fwdConvolutionGradInst(
    const ConvolutionGradInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto inG = getWeightHandle(I->getSrcGrad());
  auto outG = getWeightHandle(I->getDestGrad());

  auto filterW = getWeightHandle(I->getFilter());
  auto filterG = getWeightHandle(I->getFilterGrad());
  auto biasG = getWeightHandle(I->getBiasGrad());

  size_t group = I->getGroup();
  auto dilation = I->getDilation();

  inG.clear();
  filterG.clear();
  biasG.clear();

  ShapeNHWC odim(outG.dims());
  ShapeNHWC idim(inW.dims());
  ShapeHW kdim(I->getKernels());
  ShapeHW sdim(I->getStrides());
  PaddingTLBR pdim(I->getPads());

  assert(idim.c % group == 0 && "Input channels must be divisible by group.");
  assert(odim.c % group == 0 && "Output channels must be divisible by group.");
  dim_t inCperG = idim.c / group;
  dim_t outCperG = odim.c / group;

  // For each input in the batch:
  for (dim_t n = 0; n < odim.n; n++) {

    // For each group of input channels:
    for (dim_t g = 0; g < group; g++) {

      // Compute the gradient. For each layer in the output tensor:
      for (dim_t d = g * outCperG; d < (g + 1) * outCperG; d++) {

        // For each convolution 'jump' in the input tensor:
        sdim_t x = -sdim_t(pdim.top);
        for (dim_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
          sdim_t y = -sdim_t(pdim.left);
          for (dim_t ay = 0; ay < odim.w; y += sdim.width, ay++) {

            float chainGrad = outG.at({n, ax, ay, d});

            // For each element in the convolution-filter:
            for (dim_t fx = 0; fx < kdim.height; fx++) {
              for (dim_t fy = 0; fy < kdim.width; fy++) {
                sdim_t ox = x + fx * dilation[0];
                sdim_t oy = y + fy * dilation[1];

                // Ignore index access below zero (this is due to padding).
                if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                    oy >= sdim_t(idim.w)) {
                  continue;
                }

                for (dim_t fd = 0; fd < inCperG; fd++) {
                  filterG.at({d, fx, fy, fd}) +=
                      inW.at({n, (dim_t)ox, (dim_t)oy, g * inCperG + fd}) *
                      chainGrad;
                  inG.at({n, (dim_t)ox, (dim_t)oy, g * inCperG + fd}) +=
                      filterW.at({d, fx, fy, fd}) * chainGrad;
                }
              }
            }

            biasG.at({d}) += chainGrad;
          } // W
        }   // H
      }     // C
    }       // G
  }         // N
}

/// This is the floating point implementation of Convolution3D.
template <typename ElemTy>
void BoundInterpreterFunction::fwdConvolution3DInstFloatImpl(
    Value *inV, Value *outV, Value *filterV, Value *biasV,
    llvm::ArrayRef<unsigned_t> kernelSizes, llvm::ArrayRef<unsigned_t> strides,
    llvm::ArrayRef<unsigned_t> pads, size_t group) {
  staticAssertFloatingPointType(ElemTy);

  auto inW = getWeightHandle<ElemTy>(inV);
  auto outW = getWeightHandle<ElemTy>(outV);
  auto filterW = getWeightHandle<ElemTy>(filterV);
  auto biasW = getWeightHandle<ElemTy>(biasV);

  ShapeNTHWC odim(outW.dims());
  ShapeNTHWC idim(inW.dims());
  ShapeTHW kdim(kernelSizes);
  ShapeTHW sdim(strides);

  assert(idim.c % group == 0 && "Input channels must be divisible by group.");
  assert(odim.c % group == 0 && "Output channels must be divisible by group.");
  dim_t inCperG = idim.c / group;
  dim_t outCperG = odim.c / group;

  PaddingNFTBLR pdim(pads);

  // For each input in the batch:
  for (dim_t n = 0; n < idim.n; n++) {

    // For each group of input channels:
    for (dim_t ig = 0; ig < group; ig++) {

      // For each output channel in the group:
      for (dim_t og = ig * outCperG; og < (ig + 1) * outCperG; og++) {

        ssize_t t = -ssize_t(pdim.near);
        for (dim_t at = 0; at < odim.t; t += sdim.temporal_frames, at++) {
          // For each convolution 'jump' in the input tensor:
          ssize_t x = -ssize_t(pdim.top);
          for (dim_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
            ssize_t y = -ssize_t(pdim.left);
            for (dim_t ay = 0; ay < odim.w; y += sdim.width, ay++) {
              // For each element in the 3D convolution-filter:
              float sum = 0;
              for (dim_t ft = 0; ft < kdim.temporal_frames; ft++) {
                for (dim_t fx = 0; fx < kdim.height; fx++) {
                  for (dim_t fy = 0; fy < kdim.width; fy++) {
                    sdim_t ot = t + ft;
                    sdim_t ox = x + fx;
                    sdim_t oy = y + fy;

                    // Ignore index access below zero (this is due to padding).
                    if (ot < 0 || ox < 0 || oy < 0 || ot >= ssize_t(idim.t) ||
                        ox >= ssize_t(idim.h) || oy >= ssize_t(idim.w)) {
                      continue;
                    }
                    for (dim_t fg = 0; fg < inCperG; fg++) {
                      sum += float(filterW.at({og, ft, fx, fy, fg}) *
                                   inW.at({n, (dim_t)ot, (dim_t)ox, (dim_t)oy,
                                           ig * inCperG + fg}));
                    }
                  }
                }
              }

              sum += float(biasW.at({og}));
              outW.at({n, at, ax, ay, og}) = ElemTy(sum);
            } // D
          }   // W
        }     // H
      }       // C
    }         // G
  }           // N
}

/// This is the quantized implementation of Convolution3D.
template <typename ElemTy, typename AccumulatorTy, typename BiasElemTy>
void BoundInterpreterFunction::fwdConvolution3DInstQuantizedImpl(
    Value *inV, Value *outV, Value *filterV, Value *biasV,
    llvm::ArrayRef<unsigned_t> kernelSizes, llvm::ArrayRef<unsigned_t> strides,
    llvm::ArrayRef<unsigned_t> pads, size_t group) {
  auto inW = getWeightHandle<ElemTy>(inV);
  auto outW = getWeightHandle<ElemTy>(outV);
  auto filterW = getWeightHandle<ElemTy>(filterV);
  auto biasW = getWeightHandle<BiasElemTy>(biasV);

  ShapeNTHWC odim(outW.dims());
  ShapeNTHWC idim(inW.dims());
  ShapeTHW kdim(kernelSizes);
  ShapeTHW sdim(strides);

  assert(idim.c % group == 0 && "Input channels must be divisible by group.");
  assert(odim.c % group == 0 && "Output channels must be divisible by group.");
  dim_t inCperG = idim.c / group;
  dim_t outCperG = odim.c / group;

  PaddingNFTBLR pdim(pads);

  auto outTy = outV->getType();
  auto inTy = inV->getType();
  auto filterTy = filterV->getType();
  auto biasTy = biasV->getType();

  int32_t outOffset = outTy->getOffset();
  int32_t inOffset = inTy->getOffset();
  int32_t filterOffset = filterTy->getOffset();
  int32_t biasOffset = biasTy->getOffset();

  float outScale = outTy->getScale();
  float inScale = inTy->getScale();
  float filterScale = filterTy->getScale();
  float biasScale = biasTy->getScale();

  // Calculate the scale of the values that come out of the matrix
  // multiplication part of the calculation.
  float matMulScale = inScale * filterScale;

  // For each input in the batch:
  for (dim_t n = 0; n < idim.n; n++) {

    // For each group of input channels:
    for (dim_t ig = 0; ig < group; ig++) {

      // For each output channel in the group:
      for (dim_t og = ig * outCperG; og < (ig + 1) * outCperG; og++) {

        // For each convolution 'jump' in the input tensor:
        ssize_t t = -ssize_t(pdim.near);
        for (dim_t at = 0; at < odim.t; t += sdim.temporal_frames, at++) {
          ssize_t x = -ssize_t(pdim.top);
          for (dim_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
            ssize_t y = -ssize_t(pdim.left);
            for (dim_t ay = 0; ay < odim.w; y += sdim.width, ay++) {

              // For each element in the convolution-filter:
              AccumulatorTy sum = 0;
              for (dim_t ft = 0; ft < kdim.temporal_frames; ft++) {
                for (dim_t fx = 0; fx < kdim.height; fx++) {
                  for (dim_t fy = 0; fy < kdim.width; fy++) {
                    ssize_t ot = t + ft;
                    ssize_t ox = x + fx;
                    ssize_t oy = y + fy;

                    // Ignore index access below zero (this is due to padding).
                    if (ot < 0 || ox < 0 || oy < 0 || ot >= ssize_t(idim.t) ||
                        ox >= ssize_t(idim.h) || oy >= ssize_t(idim.w)) {
                      continue;
                    }
                    for (dim_t fg = 0; fg < inCperG; fg++) {

                      AccumulatorTy F = filterW.at({og, ft, fx, fy, fg});
                      AccumulatorTy I = inW.at({n, (dim_t)ot, (dim_t)ox,
                                                (dim_t)oy, ig * inCperG + fg});
                      // We represent the element multiplication with offset as
                      // (value - offset).
                      sum += (F - filterOffset) * (I - inOffset);
                    }
                  }
                }
              }

              // Scale the bias to match the scale of the matrix multiplication.
              AccumulatorTy B = std::round(float(biasW.at({og}) - biasOffset) *
                                           (biasScale / matMulScale));

              // Add the bias:
              sum += B;

              // Scale the result back to the expected destination scale.
              outW.at({n, at, ax, ay, og}) =
                  quantization::clip<AccumulatorTy, ElemTy>(std::round(
                      float(sum) * (matMulScale / outScale) + outOffset));
            } // D
          }   // W
        }     // H
      }       // C
    }         // G
  }           // N
}

void BoundInterpreterFunction::fwdConvolution3DInst(
    const Convolution3DInst *I) {
  auto kernelSizes = I->getKernels();
  auto pads = I->getPads();
  auto strides = I->getStrides();
  size_t group = I->getGroup();

  if (I->getSrc()->getType()->isQuantizedType()) {
    dispatchQuantizedWithAccumulationAndBiasImpl(
        fwdConvolution3DInstQuantizedImpl, I->getSrc()->getElementType(),
        I->getBias()->getElementType(), I->getSrc(), I->getDest(),
        I->getFilter(), I->getBias(), kernelSizes, strides, pads, group);
    return;
  }

  dispatchFloatingPointImpl(fwdConvolution3DInstFloatImpl,
                            I->getSrc()->getElementType(), I->getSrc(),
                            I->getDest(), I->getFilter(), I->getBias(),
                            kernelSizes, strides, pads, group);
}

void BoundInterpreterFunction::fwdConvolution3DGradInst(
    const Convolution3DGradInst *I) {
  (void)I;
  // TODO
  llvm_unreachable("not yet implemented");
}

//===----------------------------------------------------------------------===//
//                       Channelwise quantized Convolution
//===----------------------------------------------------------------------===//
template <typename ElemTy, typename AccumulatorTy, typename BiasElemTy>
void BoundInterpreterFunction::fwdChannelwiseQuantizedConv2DInstImpl(
    const ChannelwiseQuantizedConvolutionInst *I) {
  auto inW = getWeightHandle<ElemTy>(I->getSrc());
  auto outW = getWeightHandle<ElemTy>(I->getDest());
  auto filterW = getWeightHandle<ElemTy>(I->getFilter());
  auto biasW = getWeightHandle<BiasElemTy>(I->getBias());
  auto filterScales = getWeightHandle<float>(I->getFilterScales());
  auto filterOffsets = getWeightHandle<int32_t>(I->getFilterOffsets());
  auto biasScales = getWeightHandle<float>(I->getBiasScales());
  auto biasOffsets = getWeightHandle<int32_t>(I->getBiasOffsets());

  llvm::ArrayRef<unsigned_t> kernelSizes = I->getKernels();
  llvm::ArrayRef<unsigned_t> pads = I->getPads();
  llvm::ArrayRef<unsigned_t> strides = I->getStrides();
  dim_t group = I->getGroup();
  llvm::ArrayRef<unsigned_t> dilation = I->getDilation();

  ShapeNHWC odim(outW.dims());
  ShapeNHWC idim(inW.dims());
  ShapeHW kdim(kernelSizes);
  ShapeHW sdim(strides);

  assert(idim.c % group == 0 && "Input channels must be divisible by group.");
  assert(odim.c % group == 0 && "Output channels must be divisible by group.");
  dim_t inCperG = idim.c / group;
  dim_t outCperG = odim.c / group;

  PaddingTLBR pdim(pads);

  auto &inTy = inW.getType();
  auto &outTy = outW.getType();

  float inScale = inTy.getScale();
  float outScale = outTy.getScale();

  int32_t inOffset = inTy.getOffset();
  int32_t outOffset = outTy.getOffset();

  // For each input in the batch:
  for (dim_t n = 0; n < idim.n; n++) {
    // For each group of input channels:
    for (dim_t g = 0; g < group; g++) {
      // For each output channel in the group:
      for (dim_t d = g * outCperG; d < (g + 1) * outCperG; d++) {

        // Get channel wise quantization params.
        int32_t filterOffset = filterOffsets.at(d);
        float filterScale = filterScales.at(d);
        int32_t biasOffset = biasOffsets.at(d);
        float biasScale = biasScales.at(d);
        float matMulScale = inScale * filterScale;

        // For each convolution 'jump' in the input tensor:
        sdim_t x = -sdim_t(pdim.top);
        for (dim_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
          sdim_t y = -sdim_t(pdim.left);
          for (dim_t ay = 0; ay < odim.w; y += sdim.width, ay++) {

            // For each element in the convolution-filter:
            AccumulatorTy sum = 0;
            for (dim_t fx = 0; fx < kdim.height; fx++) {
              for (dim_t fy = 0; fy < kdim.width; fy++) {
                sdim_t ox = x + fx * dilation[0];
                sdim_t oy = y + fy * dilation[1];

                // Ignore index access below zero (this is due to padding).
                if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                    oy >= sdim_t(idim.w)) {
                  continue;
                }

                // Accumulate along the filter depth.
                for (dim_t fd = 0; fd < inCperG; fd++) {
                  AccumulatorTy F = filterW.at({d, fx, fy, fd});
                  AccumulatorTy I =
                      inW.at({n, (dim_t)ox, (dim_t)oy, g * inCperG + fd});
                  // We represent the element multiplication with offset as
                  // (value - offset).
                  sum += (F - filterOffset) * (I - inOffset);
                }
              }
            }

            // Scale the bias to match the scale of the matrix multiplication.
            sum += std::round(float(biasW.at({d}) - biasOffset) *
                              (biasScale / matMulScale));

            // Scale the result back to the expected destination scale.
            outW.at({n, ax, ay, d}) = quantization::clip<AccumulatorTy, ElemTy>(
                std::round(float(sum) * (matMulScale / outScale) + outOffset));
          } // W
        }   // H
      }     // C
    }       // G
  }         // N
}

template <typename ElemTy, typename AccumulatorTy, typename BiasElemTy>
void BoundInterpreterFunction::fwdChannelwiseQuantizedConv3DInstImpl(
    const ChannelwiseQuantizedConvolutionInst *I) {
  auto inW = getWeightHandle<ElemTy>(I->getSrc());
  auto outW = getWeightHandle<ElemTy>(I->getDest());
  auto filterW = getWeightHandle<ElemTy>(I->getFilter());
  auto biasW = getWeightHandle<BiasElemTy>(I->getBias());
  auto filterScales = getWeightHandle<float>(I->getFilterScales());
  auto filterOffsets = getWeightHandle<int32_t>(I->getFilterOffsets());
  auto biasScales = getWeightHandle<float>(I->getBiasScales());
  auto biasOffsets = getWeightHandle<int32_t>(I->getBiasOffsets());

  llvm::ArrayRef<unsigned_t> kernelSizes = I->getKernels();
  llvm::ArrayRef<unsigned_t> pads = I->getPads();
  llvm::ArrayRef<unsigned_t> strides = I->getStrides();
  dim_t group = I->getGroup();

  ShapeNTHWC odim(outW.dims());
  ShapeNTHWC idim(inW.dims());
  ShapeTHW kdim(kernelSizes);
  ShapeTHW sdim(strides);

  assert(idim.c % group == 0 && "Input channels must be divisible by group.");
  assert(odim.c % group == 0 && "Output channels must be divisible by group.");
  dim_t inCperG = idim.c / group;
  dim_t outCperG = odim.c / group;

  PaddingNFTBLR pdim(pads);

  auto &inTy = inW.getType();
  auto &outTy = outW.getType();

  float inScale = inTy.getScale();
  float outScale = outTy.getScale();

  int32_t inOffset = inTy.getOffset();
  int32_t outOffset = outTy.getOffset();

  // For each input in the batch:
  for (dim_t n = 0; n < idim.n; n++) {
    // For each group of input channels:
    for (dim_t g = 0; g < group; g++) {
      // For each output channel in the group:
      for (dim_t d = g * outCperG; d < (g + 1) * outCperG; d++) {

        // Get channel wise quantization params.
        int32_t filterOffset = filterOffsets.at(d);
        float filterScale = filterScales.at(d);
        int32_t biasOffset = biasOffsets.at(d);
        float biasScale = biasScales.at(d);
        float matMulScale = inScale * filterScale;

        // For each convolution 'jump' in the input tensor:
        sdim_t t = -sdim_t(pdim.near);
        for (dim_t at = 0; at < odim.t; t += sdim.temporal_frames, at++) {
          sdim_t x = -sdim_t(pdim.top);
          for (dim_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
            sdim_t y = -sdim_t(pdim.left);
            for (dim_t ay = 0; ay < odim.w; y += sdim.width, ay++) {

              // For each element in the convolution-filter:
              AccumulatorTy sum = 0;
              for (dim_t ft = 0; ft < kdim.temporal_frames; ft++) {
                for (dim_t fx = 0; fx < kdim.height; fx++) {
                  for (dim_t fy = 0; fy < kdim.width; fy++) {
                    sdim_t ot = t + ft;
                    sdim_t ox = x + fx;
                    sdim_t oy = y + fy;

                    // Ignore index access below zero (this is due to
                    // padding).
                    if (ot < 0 || ox < 0 || oy < 0 || ot >= ssize_t(idim.t) ||
                        ox >= ssize_t(idim.h) || oy >= sdim_t(idim.w)) {
                      continue;
                    }

                    // Accumulate along the filter depth.
                    for (dim_t fd = 0; fd < inCperG; fd++) {

                      AccumulatorTy F = filterW.at({d, ft, fx, fy, fd});
                      AccumulatorTy I = inW.at({n, (dim_t)ot, (dim_t)ox,
                                                (dim_t)oy, g * inCperG + fd});
                      // We represent the element multiplication with offset
                      // as (value - offset).
                      sum += (F - filterOffset) * (I - inOffset);
                    }
                  }
                }
              }

              // Scale the bias to match the scale of the matrix multiplication.
              sum += std::round(float(biasW.at({d}) - biasOffset) *
                                (biasScale / matMulScale));

              // Scale the result back to the expected destination scale.
              outW.at({n, at, ax, ay, d}) =
                  quantization::clip<AccumulatorTy, ElemTy>(std::round(
                      float(sum) * (matMulScale / outScale) + outOffset));
            } // W
          }   // H
        }     // T
      }       // C
    }         // G
  }           // N
}

void BoundInterpreterFunction::fwdChannelwiseQuantizedConvolutionInst(
    const ChannelwiseQuantizedConvolutionInst *I) {
  bool isConv3D = (I->getSrc()->dims().size() == 5);
  if (isConv3D) {
    dispatchQuantizedWithAccumulationAndBiasImpl(
        fwdChannelwiseQuantizedConv3DInstImpl, I->getSrc()->getElementType(),
        I->getBias()->getElementType(), I);
  } else {
    dispatchQuantizedWithAccumulationAndBiasImpl(
        fwdChannelwiseQuantizedConv2DInstImpl, I->getSrc()->getElementType(),
        I->getBias()->getElementType(), I);
  }
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdBatchNormalizationFloatImpl(
    const BatchNormalizationInst *I, int numDims) {
  staticAssertFloatingPointType(ElemTy);

  // input
  auto inH = getWeightHandle<ElemTy>(I->getSrc());
  auto scaleH = getWeightHandle<ElemTy>(I->getScale());
  auto biasH = getWeightHandle<ElemTy>(I->getBias());
  auto meanH = getWeightHandle<ElemTy>(I->getMean());
  auto varH = getWeightHandle<ElemTy>(I->getVar());
  unsigned_t channelIdx = I->getChannelIdx();
  float epsilon = I->getEpsilon();

  // output
  auto outW = getWeightHandle<ElemTy>(I->getDest());

  dim_t N, C, sizeN, sizeImg;
  bool isCMinor;
  if (numDims == 3) {
    if (channelIdx == 4) {
      ShapeNTHWC idim(I->getSrc()->dims());
      N = idim.n;
      C = idim.c;
      sizeImg = idim.t * idim.h * idim.w;
      sizeN = idim.c * sizeImg;
      isCMinor = true;
    } else {
      ShapeNCTHW idim(I->getSrc()->dims());
      N = idim.n;
      C = idim.c;
      sizeImg = idim.t * idim.h * idim.w;
      sizeN = idim.c * sizeImg;
      isCMinor = false;
    }
  } else if (numDims == 2) {
    if (channelIdx == 3) {
      ShapeNHWC idim(I->getSrc()->dims());
      N = idim.n;
      C = idim.c;
      sizeImg = idim.h * idim.w;
      sizeN = idim.c * sizeImg;
      isCMinor = true;
    } else {
      ShapeNCHW idim(I->getSrc()->dims());
      N = idim.n;
      C = idim.c;
      sizeImg = idim.h * idim.w;
      sizeN = idim.c * sizeImg;
      isCMinor = false;
    }
  } else if (numDims == 1) {
    N = I->getSrc()->dims()[0];
    C = I->getSrc()->dims()[channelIdx];
    sizeImg = I->getSrc()->dims()[channelIdx == 2 ? 1 : 2];
    sizeN = C * sizeImg;
    isCMinor = (channelIdx == 2);
  } else {
    N = I->getSrc()->dims()[0];
    C = I->getSrc()->dims()[channelIdx];
    sizeImg = 1;
    sizeN = C;
    isCMinor = false;
  }

  std::vector<float> scale(C), mean(C), bias(C);
  for (dim_t c = 0; c < C; c++) {
    scale[c] = float(scaleH.at({c})) / std::sqrt(float(varH.at({c})) + epsilon);
    bias[c] = biasH.at({c});
    mean[c] = meanH.at({c});
  }

  // For each input in the batch:
  for (dim_t n = 0; n < N; n++) {
    if (isCMinor) {
      // For each H*W{*T} of the image
      for (dim_t i = 0; i < sizeImg; i++) {
        // For each channel
        for (dim_t c = 0; c < C; c++) {
          int index = n * sizeN + i * C + c;
          outW.raw(index) =
              ElemTy(scale[c] * (float(inH.raw(index)) - mean[c]) + bias[c]);
        } // C
      }   // image
    } else {
      // For each channel
      for (dim_t c = 0; c < C; c++) {
        // For each H*W{*T} of the image
        for (dim_t i = 0; i < sizeImg; i++) {
          int index = n * sizeN + c * sizeImg + i;
          outW.raw(index) =
              ElemTy(scale[c] * (float(inH.raw(index)) - mean[c]) + bias[c]);
        } // image
      }   // C
    }
  } // N
}

template <typename ParamTy>
void BoundInterpreterFunction::fwdBatchNormalizationI8Impl(
    const BatchNormalizationInst *I, int numDims) {

  // input
  auto inH = getWeightHandle<int8_t>(I->getSrc());
  auto scaleH = getWeightHandle<ParamTy>(I->getScale());
  auto biasH = getWeightHandle<ParamTy>(I->getBias());
  auto meanH = getWeightHandle<ParamTy>(I->getMean());
  auto varH = getWeightHandle<ParamTy>(I->getVar());
  unsigned_t channelIdx =
      I->getChannelIdx(); // NOTE: We only support NTHWC, NHWC, NWC and NCW
  float epsilon = I->getEpsilon();
  auto inScale = float(I->getSrc()->getType()->getScale());
  auto inZero = I->getSrc()->getType()->getOffset();

  // output
  auto outH = getWeightHandle<int8_t>(I->getDest());
  auto outScale = float(I->getDest()->getType()->getScale());
  auto outZero = I->getDest()->getType()->getOffset();

  dim_t N, C, sizeN, sizeImg;
  bool isCMinor;
  if (numDims == 3) {
    if (channelIdx == 4) {
      ShapeNTHWC idim(I->getSrc()->dims());
      N = idim.n;
      C = idim.c;
      sizeImg = idim.t * idim.h * idim.w;
      sizeN = idim.c * sizeImg;
      isCMinor = true;

    } else {
      ShapeNCTHW idim(I->getSrc()->dims());
      N = idim.n;
      C = idim.c;
      sizeImg = idim.t * idim.h * idim.w;
      sizeN = idim.c * sizeImg;
      isCMinor = false;
    }
  } else if (numDims == 2) {
    if (channelIdx == 3) {
      ShapeNHWC idim(I->getSrc()->dims());
      N = idim.n;
      C = idim.c;
      sizeImg = idim.h * idim.w;
      sizeN = idim.c * sizeImg;
      isCMinor = true;

    } else {
      ShapeNCHW idim(I->getSrc()->dims());
      N = idim.n;
      C = idim.c;
      sizeImg = idim.h * idim.w;
      sizeN = idim.c * sizeImg;
      isCMinor = false;
    }

  } else {
    // numDims == 1. This can happen due to optimization pass that sinks
    // reshape below batchnorm.
    N = I->getSrc()->dims()[0];
    C = I->getSrc()->dims()[channelIdx];
    sizeImg = I->getSrc()->dims()[channelIdx == 2 ? 1 : 2];
    sizeN = C * sizeImg;
    isCMinor = (channelIdx == 2);
  }

  // See qbatch_norm.cpp/compute_fused_params() for FBGEMM implementation
  std::vector<ParamTy> alpha(C), beta(C);
  for (dim_t c = 0; c < C; c++) {
    float invSigma = 1 / std::sqrt(float(varH.at({c})) + epsilon);
    alpha[c] = ParamTy(invSigma * float(scaleH.at({c})) * (inScale / outScale));
    beta[c] = ParamTy((float(biasH.at({c})) - float(meanH.at({c})) * invSigma *
                                                  float(scaleH.at({c}))) /
                      outScale);
  }

  // See QuantizedOpKernels.cpp/q_batch_norm_kernel() for FBGEMM implementation
  TensorQuantizationParams outputQ{1.0f, outZero};
  // For each input in the batch:
  for (dim_t n = 0; n < N; n++) {
    if (isCMinor) {
      // For each H*W{*T} of the image
      for (dim_t i = 0; i < sizeImg; i++) {
        // For each channel
        for (dim_t c = 0; c < C; c++) {
          int index = n * sizeN + i * C + c;
          ParamTy x = inH.raw(index) - inZero;
          ParamTy y = alpha[c] * x + beta[c];
          outH.raw(index) = quantization::quantize(y, outputQ);
        } // image
      }   // C
    } else {
      // For each channel
      for (dim_t c = 0; c < C; c++) {
        // For each H*W{*T} of the image
        for (dim_t i = 0; i < sizeImg; i++) {
          int index = n * sizeN + c * sizeImg + i;
          auto x = ParamTy(inH.raw(index) - inZero);
          ParamTy y = alpha[c] * x + beta[c];
          outH.raw(index) = quantization::quantize(y, outputQ);
        } // image
      }   // C
    }
  } // N
}

void BoundInterpreterFunction::fwdBatchNormalizationInst(
    const BatchNormalizationInst *I) {
  int numDims = I->getSrc()->dims().size() - 2;
  bool isQuantized = I->getSrc()->getType()->isQuantizedType();

  if (isQuantized) {
    if (I->getScale()->getType()->getElementType() == ElemKind::FloatTy) {
      fwdBatchNormalizationI8Impl<float>(I, numDims);
    } else {
      fwdBatchNormalizationI8Impl<float16_t>(I, numDims);
    }
  } else {
    dispatchFloatingPointImpl(fwdBatchNormalizationFloatImpl,
                              I->getSrc()->getElementType(), I, numDims);
  }
}

//===----------------------------------------------------------------------===//
//               LayerNormalization
//===----------------------------------------------------------------------===//

template <typename ElemTy>
void BoundInterpreterFunction::fwdLayerNormalizationInstFloatImpl(
    const glow::LayerNormalizationInst *I) {
  staticAssertFloatingPointType(ElemTy);

  // input
  auto inH = getWeightHandle<ElemTy>(I->getSrc());
  auto scaleH = getWeightHandle<ElemTy>(I->getScale());
  auto biasH = getWeightHandle<ElemTy>(I->getBias());
  float epsilon = I->getEpsilon();

  // output
  auto outW = getWeightHandle<ElemTy>(I->getDest());

  auto N = I->getSrc()->dims()[0];
  auto K = I->getSrc()->dims()[1];

  std::vector<float> val(K);
  for (dim_t n = 0; n < N; n++) {
    // 1. mean = x.mean(dim=-1, keepdim=True)
    float sum = 0.0f;
    for (dim_t k = 0; k < K; k++) {
      val[k] = inH.at({n, k});
      sum += val[k];
    }
    float mean = sum / K;

    // 2. var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    float diff_sqr_sum = 0.0f;
    for (dim_t k = 0; k < K; k++) {
      float diff = val[k] - mean;
      diff_sqr_sum += diff * diff;
    }
    float var = diff_sqr_sum / K;

    // 3. std = (var + epsilon).sqrt()
    float std = std::sqrt(var + epsilon);

    for (dim_t k = 0; k < K; k++) {
      // 4. y = ((x - mean) / std) * scale + bias
      float scale = scaleH.at({k});
      float bias = biasH.at({k});
      outW.at({n, k}) = ElemTy((((val[k] - mean) / std) * scale) + bias);
    }
  }
}

void BoundInterpreterFunction::fwdLayerNormalizationInst(
    const LayerNormalizationInst *I) {
  dispatchFloatingPointImpl(fwdLayerNormalizationInstFloatImpl,
                            I->getSrc()->getElementType(), I);
}

//===----------------------------------------------------------------------===//
//                       Pooling
//===----------------------------------------------------------------------===//
template <class T>
static void fwdMaxPool(Tensor *inW, Tensor *outW, Tensor *argmaxW,
                       llvm::ArrayRef<unsigned_t> kernelSizes,
                       llvm::ArrayRef<unsigned_t> strides,
                       llvm::ArrayRef<unsigned_t> pads) {
  ShapeNHWC odim(outW->dims());
  ShapeNHWC idim(inW->dims());
  Handle<T> inHandle = inW->getHandle<T>();
  Handle<T> outHandle = outW->getHandle<T>();
  PaddingTLBR pdim(pads);
  ShapeHW kdim(kernelSizes);
  ShapeHW sdim(strides);

  llvm::Optional<Handle<int64_t>> argmaxH;
  if (argmaxW) {
    argmaxH = argmaxW->getHandle<int64_t>();
  }
  // For each input in the batch:
  for (dim_t n = 0; n < odim.n; n++) {

    // For each layer in the output tensor:
    for (dim_t z = 0; z < idim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      sdim_t x = -sdim_t(pdim.top);
      for (dim_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
        sdim_t y = -sdim_t(pdim.left);
        for (dim_t ay = 0; ay < odim.w; y += sdim.width, ay++) {

          // When the MaxPool window includes only padding pixels then for
          // that window by convention we return 0.
          bool first = true;
          T max_value = outW->getType().isQuantizedType()
                            ? static_cast<T>(outW->getType().getOffset())
                            : static_cast<T>(0);
          dim_t argmaxNHWC = 0;

          for (dim_t fx = 0; fx < kdim.height; fx++) {
            for (dim_t fy = 0; fy < kdim.width; fy++) {
              sdim_t ox = x + fx;
              sdim_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                  oy >= ssize_t(idim.w)) {
                continue;
              }

              T val = inHandle.at({n, (dim_t)ox, (dim_t)oy, z});
              if (first || (val >= max_value)) {
                first = false;
                max_value = val;
                if (argmaxW) {
                  argmaxNHWC = &inHandle.at({n, (dim_t)ox, (dim_t)oy, z}) -
                               &inHandle.raw(0);
                }
              }
            }
          }

          outHandle.at({n, ax, ay, z}) = max_value;

          if (argmaxW) {
            (*argmaxH).at({n, ax, ay, z}) = argmaxNHWC;
          }
        } // W
      }   // H
    }     // C
  }       // N
}

void BoundInterpreterFunction::fwdMaxPoolInst(const MaxPoolInst *I) {
  auto inW = getTensor(I->getSrc());
  auto outW = getTensor(I->getDest());

  if (inW->getType().isQuantizedType()) {
    dispatchQuantizedImpl(fwdMaxPool, inW->getType().getElementType(), inW,
                          outW, nullptr, I->getKernels(), I->getStrides(),
                          I->getPads());
    return;
  }

  dispatchFloatingPointImpl(fwdMaxPool, inW->getType().getElementType(), inW,
                            outW, nullptr, I->getKernels(), I->getStrides(),
                            I->getPads());
}

void BoundInterpreterFunction::fwdMaxPoolWithArgmaxInst(
    const MaxPoolWithArgmaxInst *I) {
  auto inW = getTensor(I->getSrc());
  auto outW = getTensor(I->getDest());
  auto argmaxW = getTensor(I->getArgmax());

  if (inW->getType().isQuantizedType()) {
    dispatchQuantizedImpl(fwdMaxPool, inW->getType().getElementType(), inW,
                          outW, argmaxW, I->getKernels(), I->getStrides(),
                          I->getPads());
    return;
  }
  dispatchFloatingPointImpl(fwdMaxPool, inW->getType().getElementType(), inW,
                            outW, argmaxW, I->getKernels(), I->getStrides(),
                            I->getPads());
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdAvgPoolInstFloatImpl(const AvgPoolInst *I) {
  staticAssertFloatingPointType(ElemTy);

  ShapeNHWC odim(I->getDest()->dims());
  ShapeNHWC idim(I->getSrc()->dims());

  PaddingTLBR pdim(I->getPads());
  ShapeHW kdim(I->getKernels());
  ShapeHW sdim(I->getStrides());
  // Implement the avg pooling operation as defined here:
  // https://arxiv.org/abs/1312.4400
  float rawFilterArea = kdim.height * kdim.width;

  auto inW = getWeightHandle<ElemTy>(I->getSrc());
  auto outW = getWeightHandle<ElemTy>(I->getDest());

  // For each input in the batch:
  for (dim_t n = 0; n < odim.n; n++) {
    // For each layer in the output tensor:
    for (dim_t z = 0; z < idim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t x = -ssize_t(pdim.top);
      for (dim_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
        ssize_t y = -ssize_t(pdim.left);
        for (dim_t ay = 0; ay < odim.w; y += sdim.width, ay++) {
          float sum = 0;
          float filterArea = rawFilterArea;

          for (dim_t fx = 0; fx < kdim.height; fx++) {
            for (dim_t fy = 0; fy < kdim.width; fy++) {
              sdim_t ox = x + fx;
              sdim_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                  oy >= ssize_t(idim.w)) {
                if (!I->getCountIncludePads()) {
                  filterArea--;
                }

                continue;
              }

              sum += float(inW.at({n, (dim_t)ox, (dim_t)oy, z}));
            }
          }
          if (filterArea == 0) {
            outW.at({n, ax, ay, z}) = ElemTy(0);
          } else {
            outW.at({n, ax, ay, z}) = ElemTy(sum / filterArea);
          }
        } // W
      }   // H
    }     // C
  }       // N
}

void BoundInterpreterFunction::fwdAvgPoolInstI8Impl(const AvgPoolInst *I) {
  ShapeNHWC odim(I->getDest()->dims());
  ShapeNHWC idim(I->getSrc()->dims());

  PaddingTLBR pdim(I->getPads());
  ShapeHW kdim(I->getKernels());
  ShapeHW sdim(I->getStrides());
  // Implement the avg pooling operation as defined here:
  // https://arxiv.org/abs/1312.4400
  float rawFilterArea = kdim.height * kdim.width;

  auto inW = getWeightHandle<int8_t>(I->getSrc());
  auto outW = getWeightHandle<int8_t>(I->getDest());
  TensorQuantizationParams inQP{I->getSrc()->getType()->getScale(),
                                I->getSrc()->getType()->getOffset()};
  TensorQuantizationParams outQP{I->getDest()->getType()->getScale(),
                                 I->getDest()->getType()->getOffset()};

  // For each input in the batch:
  for (dim_t n = 0; n < odim.n; n++) {
    // For each layer in the output tensor:
    for (dim_t z = 0; z < idim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t x = -ssize_t(pdim.top);
      for (dim_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
        ssize_t y = -ssize_t(pdim.left);
        for (dim_t ay = 0; ay < odim.w; y += sdim.width, ay++) {
          int32_t sum = 0;
          float filterArea = rawFilterArea;

          for (dim_t fx = 0; fx < kdim.height; fx++) {
            for (dim_t fy = 0; fy < kdim.width; fy++) {
              sdim_t ox = x + fx;
              sdim_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                  oy >= ssize_t(idim.w)) {
                if (!I->getCountIncludePads()) {
                  filterArea--;
                }

                continue;
              }

              sum += inW.at({n, (dim_t)ox, (dim_t)oy, z}) - inQP.offset;
            }
          }
          if (filterArea == 0) {
            outW.at({n, ax, ay, z}) =
                quantization::clip<int32_t, int8_t>(outQP.offset);
          } else {
            // Instead of dividing by filterArea, just change scale.
            outW.at({n, ax, ay, z}) =
                quantization::clip<int32_t, int8_t>(std::round(
                    float(sum) * (inQP.scale / outQP.scale / filterArea) +
                    outQP.offset));
          }
        } // W
      }   // H
    }     // C
  }       // N
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdAvgPool3DInstFloatImpl(const AvgPoolInst *I) {
  staticAssertFloatingPointType(ElemTy);

  ShapeNTHWC odim(I->getDest()->dims());
  ShapeNTHWC idim(I->getSrc()->dims());

  PaddingNFTBLR pdim(I->getPads());
  ShapeTHW kdim(I->getKernels());
  ShapeTHW sdim(I->getStrides());
  // Implement the avg pooling operation as defined here:
  // https://arxiv.org/abs/1312.4400
  float rawFilterArea = kdim.temporal_frames * kdim.height * kdim.width;

  auto inW = getWeightHandle<ElemTy>(I->getSrc());
  auto outW = getWeightHandle<ElemTy>(I->getDest());

  // For each input in the batch:
  for (dim_t n = 0; n < odim.n; n++) {
    // For each layer in the output tensor:
    for (dim_t z = 0; z < idim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t t = -ssize_t(pdim.near);
      for (dim_t at = 0; at < odim.t; t += sdim.temporal_frames, at++) {
        ssize_t x = -ssize_t(pdim.top);
        for (dim_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
          ssize_t y = -ssize_t(pdim.left);
          for (dim_t ay = 0; ay < odim.w; y += sdim.width, ay++) {
            float sum = 0;
            float filterArea = rawFilterArea;

            for (dim_t ft = 0; ft < kdim.temporal_frames; ft++) {
              for (dim_t fx = 0; fx < kdim.height; fx++) {
                for (dim_t fy = 0; fy < kdim.width; fy++) {
                  sdim_t ot = t + ft;
                  sdim_t ox = x + fx;
                  sdim_t oy = y + fy;

                  // Ignore index access below zero (this is due to padding).
                  if (ot < 0 || ox < 0 || oy < 0 || ot >= ssize_t(idim.t) ||
                      ox >= ssize_t(idim.h) || oy >= ssize_t(idim.w)) {
                    if (!I->getCountIncludePads()) {
                      filterArea--;
                    }

                    continue;
                  }

                  sum += float(inW.at({n, (dim_t)ot, (dim_t)ox, (dim_t)oy, z}));
                }
              }
            }
            assert(filterArea != 0 && "filterArea can't be 0");
            outW.at({n, at, ax, ay, z}) = ElemTy(sum / filterArea);
          } // W
        }   // H
      }     // T
    }       // C
  }         // N
}

void BoundInterpreterFunction::fwdAvgPool3DInstI8Impl(const AvgPoolInst *I) {
  ShapeNTHWC odim(I->getDest()->dims());
  ShapeNTHWC idim(I->getSrc()->dims());

  PaddingNFTBLR pdim(I->getPads());
  ShapeTHW kdim(I->getKernels());
  ShapeTHW sdim(I->getStrides());
  // Implement the avg pooling operation as defined here:
  // https://arxiv.org/abs/1312.4400
  float rawFilterArea = kdim.temporal_frames * kdim.height * kdim.width;

  auto inW = getWeightHandle<int8_t>(I->getSrc());
  auto outW = getWeightHandle<int8_t>(I->getDest());
  TensorQuantizationParams inQP{I->getSrc()->getType()->getScale(),
                                I->getSrc()->getType()->getOffset()};
  TensorQuantizationParams outQP{I->getDest()->getType()->getScale(),
                                 I->getDest()->getType()->getOffset()};

  // For each input in the batch:
  for (dim_t n = 0; n < odim.n; n++) {
    // For each layer in the output tensor:
    for (dim_t z = 0; z < idim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t t = -ssize_t(pdim.near);
      for (dim_t at = 0; at < odim.t; t += sdim.temporal_frames, at++) {
        ssize_t x = -ssize_t(pdim.top);
        for (dim_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
          ssize_t y = -ssize_t(pdim.left);
          for (dim_t ay = 0; ay < odim.w; y += sdim.width, ay++) {
            int32_t sum = 0;
            float filterArea = rawFilterArea;

            for (dim_t ft = 0; ft < kdim.temporal_frames; ft++) {
              for (dim_t fx = 0; fx < kdim.height; fx++) {
                for (dim_t fy = 0; fy < kdim.width; fy++) {
                  sdim_t ot = t + ft;
                  sdim_t ox = x + fx;
                  sdim_t oy = y + fy;

                  // Ignore index access below zero (this is due to padding).
                  if (ot < 0 || ox < 0 || oy < 0 || ot >= ssize_t(idim.t) ||
                      ox >= ssize_t(idim.h) || oy >= ssize_t(idim.w)) {
                    if (!I->getCountIncludePads()) {
                      filterArea--;
                    }

                    continue;
                  }

                  sum += inW.at({n, (dim_t)ot, (dim_t)ox, (dim_t)oy, z}) -
                         inQP.offset;
                }
              }
            }
            // Instead of dividing by filterArea, just change scale.
            assert(filterArea != 0 && "filterArea can't be 0");

            float multiplier = inQP.scale / outQP.scale / filterArea;
            TensorQuantizationParams outputQ{1.0f / multiplier, outQP.offset};
            outW.at({n, at, ax, ay, z}) =
                quantization::quantize(float(sum), outputQ);
          } // W
        }   // H
      }     // T
    }       // C
  }         // N
}

void BoundInterpreterFunction::fwdAvgPoolInst(const AvgPoolInst *I) {
  bool isConv3D = is3DData(ConvolutionLayout(I->getLayout()));
  bool isQuantized = I->getSrc()->getType()->isQuantizedType();

  if (isConv3D) {
    if (isQuantized) {
      fwdAvgPool3DInstI8Impl(I);
    } else {
      dispatchFloatingPointImpl(fwdAvgPool3DInstFloatImpl,
                                I->getSrc()->getElementType(), I);
    }
  } else {
    if (isQuantized) {
      fwdAvgPoolInstI8Impl(I);
    } else {
      dispatchFloatingPointImpl(fwdAvgPoolInstFloatImpl,
                                I->getSrc()->getElementType(), I);
    }
  }
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdAdaptiveAvgPoolInstFloatImpl(
    const AdaptiveAvgPoolInst *I) {
  staticAssertFloatingPointType(ElemTy);

  ShapeNHWC odim(I->getDest()->dims());
  ShapeNHWC idim(I->getSrc()->dims());

  auto inW = getWeightHandle<ElemTy>(I->getSrc());
  auto outW = getWeightHandle<ElemTy>(I->getDest());

// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/AdaptiveAveragePooling.cpp
#define START_IND(a, b, c) (size_t) std::floor((float)((a) * (c)) / (b))
#define END_IND(a, b, c) (size_t) std::ceil((float)(((a) + 1) * (c)) / (b))

  // For each input in the batch:
  for (dim_t n = 0; n < odim.n; n++) {
    // For each layer in the output tensor:
    for (dim_t z = 0; z < idim.c; z++) {
      // For each value in the output tensor:
      for (dim_t ax = 0; ax < odim.h; ax++) {

        dim_t x = START_IND(ax, odim.h, idim.h);
        dim_t kH = END_IND(ax, odim.h, idim.h) - x;

        for (dim_t ay = 0; ay < odim.w; ay++) {

          dim_t y = START_IND(ay, odim.w, idim.w);
          dim_t kW = END_IND(ay, odim.w, idim.w) - y;

          float sum = 0;
          for (dim_t fx = 0; fx < kH; fx++) {
            for (dim_t fy = 0; fy < kW; fy++) {
              dim_t ox = x + fx;
              dim_t oy = y + fy;

              sum += float(inW.at({n, ox, oy, z}));
            }
          }
          outW.at({n, ax, ay, z}) = ElemTy(sum / kW / kH);
        } // W
      }   // H
    }     // C
  }       // N
#undef START_IND
#undef END_IND
}

void BoundInterpreterFunction::fwdAdaptiveAvgPoolInstI8Impl(
    const AdaptiveAvgPoolInst *I) {
  ShapeNHWC odim(I->getDest()->dims());
  ShapeNHWC idim(I->getSrc()->dims());

  auto inW = getWeightHandle<int8_t>(I->getSrc());
  auto outW = getWeightHandle<int8_t>(I->getDest());

  TensorQuantizationParams inQP{I->getSrc()->getType()->getScale(),
                                I->getSrc()->getType()->getOffset()};
  TensorQuantizationParams outQP{I->getDest()->getType()->getScale(),
                                 I->getDest()->getType()->getOffset()};

// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/AdaptiveAveragePooling.cpp
#define START_IND(a, b, c) (size_t) std::floor((float)((a) * (c)) / (b))
#define END_IND(a, b, c) (size_t) std::ceil((float)(((a) + 1) * (c)) / (b))

  // For each input in the batch:
  for (dim_t n = 0; n < odim.n; n++) {
    // For each layer in the output tensor:
    for (dim_t z = 0; z < idim.c; z++) {
      // For each value in the output tensor:
      for (dim_t ax = 0; ax < odim.h; ax++) {

        dim_t x = START_IND(ax, odim.h, idim.h);
        dim_t kH = END_IND(ax, odim.h, idim.h) - x;

        for (dim_t ay = 0; ay < odim.w; ay++) {

          dim_t y = START_IND(ay, odim.w, idim.w);
          dim_t kW = END_IND(ay, odim.w, idim.w) - y;

          int32_t sum = 0;
          for (dim_t fx = 0; fx < kH; fx++) {
            for (dim_t fy = 0; fy < kW; fy++) {
              dim_t ox = x + fx;
              dim_t oy = y + fy;

              sum += inW.at({n, ox, oy, z}) - inQP.offset;
            }
          }

          outW.at({n, ax, ay, z}) = quantization::clip<int32_t, int8_t>(
              std::round(float(sum) * (inQP.scale / outQP.scale / kW / kH) +
                         outQP.offset));
        } // W
      }   // H
    }     // C
  }       // N
#undef START_IND
#undef END_IND
}

void BoundInterpreterFunction::fwdAdaptiveAvgPoolInst(
    const AdaptiveAvgPoolInst *I) {
  if (I->getSrc()->getType()->isQuantizedType()) {
    fwdAdaptiveAvgPoolInstI8Impl(I);
    return;
  }

  dispatchFloatingPointImpl(fwdAdaptiveAvgPoolInstFloatImpl,
                            I->getSrc()->getElementType(), I);
}

void BoundInterpreterFunction::fwdAdaptiveAvgPoolGradInst(
    const AdaptiveAvgPoolGradInst *I) {
  auto inG = getWeightHandle(I->getSrcGrad());
  auto outW = getWeightHandle(I->getDest());
  auto outG = getWeightHandle(I->getDestGrad());

  inG.clear();

  ShapeNHWC odim(outW.dims());
  ShapeNHWC idim(inG.dims());

  const float gradCoefficient = 1. / (odim.h * odim.w);

#define START_IND(a, b, c) (size_t) std::floor((float)((a) * (c)) / (b))
#define END_IND(a, b, c) (size_t) std::ceil((float)(((a) + 1) * (c)) / (b))

  // https://software.intel.com/en-us/daal-programming-guide-2d-average-pooling-backward-layer
  // For each input in the batch:
  for (dim_t n = 0; n < odim.n; n++) {
    // For each layer in the output tensor:
    for (dim_t z = 0; z < idim.c; z++) {
      // For each value in the output tensor:
      for (dim_t ax = 0; ax < odim.h; ax++) {

        dim_t x = START_IND(ax, odim.h, idim.h);
        dim_t kH = END_IND(ax, odim.h, idim.h) - x;

        for (dim_t ay = 0; ay < odim.w; ay++) {

          dim_t y = START_IND(ay, odim.w, idim.w);
          dim_t kW = END_IND(ay, odim.w, idim.w) - y;

          const float chainGrad = outG.at({n, ax, ay, z}) * gradCoefficient;

          for (dim_t fx = 0; fx < kH; fx++) {
            for (dim_t fy = 0; fy < kW; fy++) {
              dim_t ox = x + fx;
              dim_t oy = y + fy;

              inG.at({n, ox, oy, z}) += chainGrad;
            }
          }
        } // W
      }   // H
    }     // C
  }       // N
#undef START_IND
#undef END_IND
}

void BoundInterpreterFunction::fwdMaxPoolWithArgmaxGradInst(
    const MaxPoolWithArgmaxGradInst *I) {
  auto inG = getWeightHandle(I->getSrcGrad());
  auto outW = getWeightHandle(I->getDest());
  auto outG = getWeightHandle(I->getDestGrad());

  inG.clear();

  ShapeNHWC idim(inG.dims());
  ShapeNHWC odim(outW.dims());

  auto argmax = getWeightHandle<int64_t>(I->getArgmax());

  // For each input in the batch:
  for (dim_t n = 0; n < odim.n; n++) {

    // Compute the gradient. For each layer in the output tensor:
    for (dim_t z = 0; z < odim.c; z++) {

      // For each convolution 'jump' in the input tensor:
      for (dim_t ax = 0; ax < odim.h; ax++) {
        for (dim_t ay = 0; ay < odim.w; ay++) {
          // Reuse precomputed linear index of max element from argmax.
          float chainGrad = outG.at({n, ax, ay, z});
          inG.raw(argmax.at({n, ax, ay, z})) += chainGrad;
        } // W
      }   // H
    }     // C
  }       // N
}

void BoundInterpreterFunction::fwdAvgPool2DGradInst(const AvgPoolGradInst *I) {
  auto inG = getWeightHandle(I->getSrcGrad());
  auto outW = getWeightHandle(I->getDest());
  auto outG = getWeightHandle(I->getDestGrad());

  ShapeNHWC odim(outW.dims());
  ShapeNHWC idim(inG.dims());

  PaddingTLBR pdim(I->getPads());
  ShapeHW kdim(I->getKernels());
  ShapeHW sdim(I->getStrides());

  inG.clear();

  float rawFilterArea = kdim.height * kdim.width;

  // For each input in the batch:
  for (dim_t n = 0; n < odim.n; n++) {

    // For each layer in the output tensor:
    for (dim_t z = 0; z < odim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t x = -ssize_t(pdim.top);
      for (dim_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
        ssize_t y = -ssize_t(pdim.left);
        for (dim_t ay = 0; ay < odim.w; y += sdim.width, ay++) {
          float filterArea = rawFilterArea;

          // Excludes the padding area in filterArea if the flag is false
          if (!I->getCountIncludePads()) {
            ssize_t pad_x = (-x > 0 ? -x : 0) +
                            ((x + ssize_t(kdim.height) - ssize_t(idim.h)) > 0
                                 ? (x + ssize_t(kdim.height) - ssize_t(idim.h))
                                 : 0);
            ssize_t pad_y = (-y > 0 ? -y : 0) +
                            ((y + ssize_t(kdim.width) - ssize_t(idim.w)) > 0
                                 ? (y + ssize_t(kdim.width) - ssize_t(idim.w))
                                 : 0);
            filterArea = rawFilterArea - pad_x * kdim.width -
                         pad_y * kdim.height + pad_x * pad_y;
          }
          assert(filterArea != 0 && "filterArea can't be 0");
          float dy = outG.at({n, ax, ay, z}) / filterArea;

          for (dim_t fx = 0; fx < kdim.height; fx++) {
            for (dim_t fy = 0; fy < kdim.width; fy++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                  oy >= ssize_t(idim.w)) {
                continue;
              }
              inG.at({n, (dim_t)ox, (dim_t)oy, z}) += dy;
            }
          }
        } // W
      }   // H
    }     // C
  }       // N
}

void BoundInterpreterFunction::fwdAvgPool3DGradInst(const AvgPoolGradInst *I) {
  auto inG = getWeightHandle(I->getSrcGrad());
  auto outW = getWeightHandle(I->getDest());
  auto outG = getWeightHandle(I->getDestGrad());

  ShapeNTHWC odim(outW.dims());
  ShapeNTHWC idim(inG.dims());

  PaddingNFTBLR pdim(I->getPads());
  ShapeTHW kdim(I->getKernels());
  ShapeTHW sdim(I->getStrides());

  inG.clear();

  float rawFilterArea = kdim.temporal_frames * kdim.height * kdim.width;

  // For each input in the batch:
  for (dim_t n = 0; n < odim.n; n++) {

    // For each layer in the output tensor:
    for (dim_t z = 0; z < odim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t t = -ssize_t(pdim.near);
      for (dim_t at = 0; at < odim.t; t += sdim.temporal_frames, at++) {
        ssize_t x = -ssize_t(pdim.top);
        for (dim_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
          ssize_t y = -ssize_t(pdim.left);
          for (dim_t ay = 0; ay < odim.w; y += sdim.width, ay++) {
            float filterArea = rawFilterArea;

            // Excludes the padding area in filterArea if the flag is false
            if (!I->getCountIncludePads()) {
              ssize_t pad_x =
                  (-x > 0 ? -x : 0) +
                  ((x + ssize_t(kdim.height) - ssize_t(idim.h)) > 0
                       ? (x + ssize_t(kdim.height) - ssize_t(idim.h))
                       : 0);
              ssize_t pad_y = (-y > 0 ? -y : 0) +
                              ((y + ssize_t(kdim.width) - ssize_t(idim.w)) > 0
                                   ? (y + ssize_t(kdim.width) - ssize_t(idim.w))
                                   : 0);
              ssize_t pad_z =
                  (-t > 0 ? -t : 0) +
                  ((t + ssize_t(kdim.temporal_frames) - ssize_t(idim.t)) > 0
                       ? (t + ssize_t(kdim.temporal_frames) - ssize_t(idim.t))
                       : 0);
              filterArea = rawFilterArea -
                           pad_x * kdim.width * kdim.temporal_frames -
                           pad_y * kdim.height * kdim.temporal_frames -
                           pad_z * kdim.height * kdim.width +
                           pad_x * pad_y * kdim.temporal_frames +
                           pad_x * pad_z * kdim.width +
                           pad_y * pad_z * kdim.height - pad_x * pad_y * pad_z;
            }
            assert(filterArea != 0 && "filterArea can't be 0");
            float dy = outG.at({n, at, ax, ay, z}) / filterArea;

            for (dim_t ft = 0; ft < kdim.temporal_frames; ft++) {
              for (dim_t fx = 0; fx < kdim.height; fx++) {
                for (dim_t fy = 0; fy < kdim.width; fy++) {
                  ssize_t ot = t + ft;
                  ssize_t ox = x + fx;
                  ssize_t oy = y + fy;

                  // Ignore index access below zero (this is due to padding).
                  if (ot < 0 || ox < 0 || oy < 0 || ot >= ssize_t(idim.t) ||
                      ox >= ssize_t(idim.h) || oy >= ssize_t(idim.w)) {
                    continue;
                  }
                  inG.at({n, (dim_t)ot, (dim_t)ox, (dim_t)oy, z}) += dy;
                }
              }
            }
          } // W
        }   // H
      }     // T
    }       // C
  }         // N
}

void BoundInterpreterFunction::fwdAvgPoolGradInst(const AvgPoolGradInst *I) {
  bool isConv3D = is3DData(ConvolutionLayout(I->getLayout()));

  if (isConv3D) {
    fwdAvgPool3DGradInst(I);
  } else {
    fwdAvgPool2DGradInst(I);
  }
}

//===----------------------------------------------------------------------===//
//                       Activation functions
//===----------------------------------------------------------------------===//

void BoundInterpreterFunction::fwdReluInst(const ReluInst *) {
  DCHECK(!"Found ReluInst but Relu is lowered on Interpreter");
}

void BoundInterpreterFunction::fwdClipInst(const ClipInst *) {
  DCHECK(!"Found ClipInst but Clip is lowered on Interpreter");
}

void BoundInterpreterFunction::fwdLeakyReluInst(const LeakyReluInst *) {
  DCHECK(!"Found LeakyReluInst but LeakyRelu is lowered on Interpreter");
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdSigmoidInstFloatImpl(const SigmoidInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto inW = getWeightHandle<ElemTy>(I->getSrc());
  auto outW = getWeightHandle<ElemTy>(I->getDest());

  for (dim_t i = 0, e = outW.size(); i < e; i++) {
    float val = inW.raw(i);
    outW.raw(i) = ElemTy(1 / (1 + std::exp(-val)));
  }
}

void BoundInterpreterFunction::fwdSigmoidInst(const SigmoidInst *I) {
  dispatchFloatingPointImpl(fwdSigmoidInstFloatImpl,
                            I->getSrc()->getElementType(), I);
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdTanhInstFloatImpl(const TanhInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto inW = getWeightHandle<ElemTy>(I->getSrc());
  auto outW = getWeightHandle<ElemTy>(I->getDest());

  for (dim_t i = 0, e = inW.size(); i < e; i++) {
    float val = inW.raw(i);
    outW.raw(i) = ElemTy(std::tanh(val));
  }
}

void BoundInterpreterFunction::fwdTanhInst(const TanhInst *I) {
  dispatchFloatingPointImpl(fwdTanhInstFloatImpl, I->getSrc()->getElementType(),
                            I);
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdSoftPlusInstFloatImpl(const SoftPlusInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto inW = getWeightHandle<ElemTy>(I->getSrc());
  auto outW = getWeightHandle<ElemTy>(I->getDest());

  for (dim_t i = 0, e = outW.size(); i < e; i++) {
    float val = inW.raw(i);
    outW.raw(i) = ElemTy(std::log(1 + std::exp(val)));
  }
}

void BoundInterpreterFunction::fwdSoftPlusInst(const SoftPlusInst *I) {
  dispatchFloatingPointImpl(fwdSoftPlusInstFloatImpl,
                            I->getSrc()->getElementType(), I);
}

//===----------------------------------------------------------------------===//
//                        Loss Functions (Softmax/regression/...)
//===----------------------------------------------------------------------===//

template <typename ElemTy>
void BoundInterpreterFunction::fwdSoftMaxInstImpl(const SoftMaxInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto inW = getWeightHandle<ElemTy>(I->getSrc());
  auto outW = getWeightHandle<ElemTy>(I->getDest());
  auto idim = inW.dims();

  for (dim_t n = 0; n < idim[0]; n++) {
    // Find Max.
    float max = float(inW.at({n, 0}));
    for (dim_t i = 1; i < idim[1]; i++) {
      max = std::max(max, float(inW.at({n, i})));
    }

    // Compute exp.
    float sum = 0;
    for (dim_t i = 0; i < idim[1]; i++) {
      float e = std::exp(float(inW.at({n, i})) - max);
      sum += e;
      outW.at({n, i}) = ElemTy(e);
    }

    // Normalize the output.
    for (dim_t i = 0; i < idim[1]; i++) {
      outW.at({n, i}) = ElemTy(float(outW.at({n, i})) / sum);
    }
  } // N
}

void BoundInterpreterFunction::fwdSoftMaxInst(const SoftMaxInst *I) {
  dispatchFloatingPointImpl(fwdSoftMaxInstImpl, I->getSrc()->getElementType(),
                            I);
}

void BoundInterpreterFunction::fwdSoftMaxGradInst(const SoftMaxGradInst *I) {
  auto inG = getWeightHandle(I->getSrcGrad());
  auto idim = inG.dims();
  auto outW = getWeightHandle(I->getOrigDest());
  auto selectedH = getWeightHandle<int64_t>(I->getSelected());

  inG.clear();

  // http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
  // https://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network
  for (dim_t n = 0; n < idim[0]; n++) {
    for (dim_t i = 0; i < idim[1]; i++) {
      float delta = (selectedH.at({n, 0}) == (int64_t)i);
      inG.at({n, i}) = outW.at({n, i}) - delta;
    }
  }
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdLogSoftMaxInstImpl(const LogSoftMaxInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto inW = getWeightHandle<ElemTy>(I->getSrc());
  auto outW = getWeightHandle<ElemTy>(I->getDest());
  auto idim = inW.dims();
  // using log(softmax(x)) = x - max_x - log(sum)
  // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/SoftMax.cpp#L14-L64
  for (dim_t n = 0; n < idim[0]; n++) {
    // Find Max.
    float max = float(inW.at({n, 0}));
    for (dim_t i = 1; i < idim[1]; i++) {
      max = std::max(max, float(inW.at({n, i})));
    }

    // Compute sum of exp(x-max).
    float sum = 0;
    for (dim_t i = 0; i < idim[1]; i++) {
      float e = std::exp(float(inW.at({n, i})) - max);
      sum += e;
    }

    // Output = x - max - log(sum)
    for (dim_t i = 0; i < idim[1]; i++) {
      outW.at({n, i}) = ElemTy(float(inW.at({n, i})) - max - std::log(sum));
    }
  }
}

void BoundInterpreterFunction::fwdLogSoftMaxInst(const LogSoftMaxInst *I) {
  dispatchFloatingPointImpl(fwdLogSoftMaxInstImpl,
                            I->getSrc()->getElementType(), I);
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdCrossEntropyLossInstFloatImpl(
    const CrossEntropyLossInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto P = getWeightHandle<ElemTy>(I->getP());
  auto labels = getWeightHandle<int64_t>(I->getLabels());
  auto CE = getWeightHandle<ElemTy>(I->getCE());
  auto dims = P.dims();
  CE.clear();
  for (dim_t n = 0; n < dims[0]; ++n) {
    assert(labels.raw(n) >= 0 && "Cannot use negative index.");
    dim_t y = labels.raw(n);
    float p_n = P.at({n, y});
    CE.at({0}) -= log(p_n);
  }
}

void BoundInterpreterFunction::fwdCrossEntropyLossInst(
    const CrossEntropyLossInst *I) {
  dispatchFloatingPointImpl(fwdCrossEntropyLossInstFloatImpl,
                            I->getP()->getElementType(), I);
}

void BoundInterpreterFunction::fwdCrossEntropyLossGradInst(
    const CrossEntropyLossGradInst *I) {
  auto P = getWeightHandle(I->getP());
  auto Labels = getWeightHandle<int64_t>(I->getLabels());
  auto PGrad = getWeightHandle(I->getPgrad());
  auto dims = PGrad.dims();
  PGrad.clear();
  for (dim_t n = 0; n < dims[0]; ++n) {
    assert(Labels.raw(n) >= 0 && "Cannot use negative index.");
    dim_t y = Labels.raw(n);
    PGrad.at({n, y}) = -1 / P.at({n, y}); // * CEGrad.at({0})
  }
}

//===----------------------------------------------------------------------===//
//                       Tensor shape (copy/transpose/concat/...)
//===----------------------------------------------------------------------===//

void BoundInterpreterFunction::fwdCopyInst(const CopyInst *I) {
  auto inT = getTensor(I->getSrc());
  auto outT = getTensor(I->getDest());
  outT->copyRawFrom(inT);
}

void BoundInterpreterFunction::fwdTransposeInst(const TransposeInst *I) {
  auto inT = getTensor(I->getSrc());
  (void)inT;
  auto outT = getTensor(I->getDest());

  assert(outT->size() == inT->size() && "Invalid tensor dimensions");

  if (I->getSrc()->getType()->isQuantizedType()) {
    inT->transpose(outT, I->getShuffle());
  } else {
    inT->transpose(outT, I->getShuffle());
  }
}

void BoundInterpreterFunction::fwdTensorViewInst(const TensorViewInst *I) {
  getOrCreateUnownedTensor(I, I->getSrc(), I->getOffsets());
}

void BoundInterpreterFunction::fwdSplatInst(const glow::SplatInst *I) {
  auto *T = getTensor(I->getDest());
  ElemKind k = T->getElementType();

  if (k == ElemKind::Int32ITy) {
    return T->getHandle<int32_t>().clear(I->getValue());
  }

  if (k == ElemKind::Int64ITy) {
    return T->getHandle<int64_t>().clear(I->getValue());
  }

  if (k == ElemKind::Int32ITy) {
    return T->getHandle<int32_t>().clear(I->getValue());
  }

  if (k == ElemKind::FloatTy) {
    return T->getHandle<float>().clear(I->getValue());
  }

  if (k == ElemKind::Float16Ty) {
    return T->getHandle<float16_t>().clear(I->getValue());
  }

  if (k == ElemKind::BFloat16Ty) {
    return T->getHandle<bfloat16_t>().clear(I->getValue());
  }

  if (k == ElemKind::BoolTy) {
    return T->getHandle<bool>().clear(static_cast<bool>(I->getValue()));
  }

  if (k == ElemKind::Int8QTy) {
    // Quantize the requested floating point splat value into the correct
    // integer representation.
    auto destTy = I->getDest()->getType();
    TensorQuantizationParams destQ{destTy->getScale(), destTy->getOffset()};
    float val = I->getValue();
    return T->getHandle<int8_t>().clear(quantization::quantize(val, destQ));
  }

  if (k == ElemKind::Int16QTy) {
    // Quantize the requested floating point splat value into the correct
    // integer representation.
    auto destTy = I->getDest()->getType();
    TensorQuantizationParams destQ{destTy->getScale(), destTy->getOffset()};
    float val = I->getValue();
    return T->getHandle<int16_t>().clear(quantization::quantize(val, destQ));
  }

  if (k == ElemKind::BoolTy) {
    return T->getHandle<bool>().clear(static_cast<bool>(I->getValue()));
  }

  llvm_unreachable("Unsupported tensor type");
}

void BoundInterpreterFunction::fwdTouchInst(const glow::TouchInst *) {
  // Do nothing for a TouchInst
}

void BoundInterpreterFunction::fwdInsertTensorInst(
    const glow::InsertTensorInst *I) {
  Tensor *outT = getTensor(I->getDest());
  Tensor *inT = getTensor(I->getSrc());
  ElemKind k = outT->getElementType();
#define TYPED_INSERT(TY, TYPEKIND)                                             \
  if (k == TYPEKIND) {                                                         \
    auto OH = outT->getHandle<TY>();                                           \
    auto IH = inT->getHandle<TY>();                                            \
    return OH.insertTensors(IH, I->getOffsets(), I->getCount(), I->getAxis()); \
  }

  TYPED_INSERT(int64_t, ElemKind::Int64ITy);
  TYPED_INSERT(int32_t, ElemKind::Int32ITy);
  TYPED_INSERT(float, ElemKind::FloatTy);
  TYPED_INSERT(float16_t, ElemKind::Float16Ty);
  TYPED_INSERT(bfloat16_t, ElemKind::BFloat16Ty);
  TYPED_INSERT(int8_t, ElemKind::Int8QTy);
  TYPED_INSERT(int16_t, ElemKind::Int16QTy);
  TYPED_INSERT(bool, ElemKind::BoolTy);
#undef TYPED_INSERT

  llvm_unreachable("Unsupported tensor type");
}

void BoundInterpreterFunction::fwdExtractTensorInst(
    const glow::ExtractTensorInst *I) {
  Tensor *outT = getTensor(I->getDest());
  Tensor *inT = getTensor(I->getSrc());
  ElemKind k = outT->getElementType();
#define TYPED_INSERT(TY, TYPEKIND)                                             \
  if (k == TYPEKIND) {                                                         \
    auto OH = outT->getHandle<TY>();                                           \
    auto IH = inT->getHandle<TY>();                                            \
    return IH.extractTensors(OH, I->getOffsets());                             \
  }

  TYPED_INSERT(int64_t, ElemKind::Int64ITy);
  TYPED_INSERT(float, ElemKind::FloatTy);
  TYPED_INSERT(float16_t, ElemKind::Float16Ty);
  TYPED_INSERT(bfloat16_t, ElemKind::BFloat16Ty);
  TYPED_INSERT(int8_t, ElemKind::Int8QTy);
  TYPED_INSERT(int32_t, ElemKind::Int32QTy);
  TYPED_INSERT(int32_t, ElemKind::Int32ITy);
  TYPED_INSERT(bool, ElemKind::BoolTy);
#undef TYPED_INSERT

  llvm_unreachable("Unsupported tensor type");
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdGatherInstImpl(const glow::GatherInst *I) {
  Tensor *dataT = getTensor(I->getData());
  auto &dataTy = dataT->getType();
  Tensor *indicesT = getTensor(I->getIndices());
  Tensor *outT = getTensor(I->getDest());
  unsigned_t axis = I->getBatchDims();

  size_t out_p = 0;
  dim_t elementSize = dataTy.getElementSize();
  // The size of the sample in the batch.
  dim_t dataSampleSize = dataTy.getSliceSize(axis) * elementSize;
  // The size of the slices that we gather.
  dim_t dataSliceSize = dataTy.getSliceSize(axis + 1) * elementSize;

  // Calculate the size of each sample in the batch.
  dim_t numSamples = (dataT->size() * elementSize) / dataSampleSize;

  // Calculate number of samples in the batch.
  dim_t batchSize = dataTy.dims()[axis];
  (void)batchSize;

  // For each sample in the batch:
  for (dim_t sample = 0; sample < numSamples; sample++) {
    dim_t sampleStart = sample * dataSampleSize;

    // For each slice (small fragment) that we copy from the source memory:
    for (dim_t i = 0, end = indicesT->size(); i < end; i++) {
      dim_t slice = indicesT->getHandle<ElemTy>().raw(i);
      assert(slice < batchSize && "Invalid index seen during Gather operation");
      std::copy(
          &dataT->getUnsafePtr()[sampleStart + dataSliceSize * slice],
          &dataT->getUnsafePtr()[sampleStart + dataSliceSize * (slice + 1)],
          &outT->getUnsafePtr()[out_p]);
      out_p += dataSliceSize;
    }
  }
}

void BoundInterpreterFunction::fwdGatherInst(const glow::GatherInst *I) {
  switch (I->getIndices()->getElementType()) {
  case ElemKind::Int64ITy:
    fwdGatherInstImpl<int64_t>(I);
    break;
  case ElemKind::Int32ITy:
    fwdGatherInstImpl<int32_t>(I);
    break;
  default:
    llvm_unreachable("Unsupported type for indices input of Gather.");
  }
}

//===----------------------------------------------------------------------===//
//                      Gather Elements
//===----------------------------------------------------------------------===//

template <typename IndexTy>
void BoundInterpreterFunction::fwdGatherElementsInstImpl(
    const glow::GatherElementsInst *I) {
  Tensor *outT = getTensor(I->getDest());
  Tensor *dataT = getTensor(I->getData());
  auto dataDims = dataT->getType().dims();
  Tensor *indicesT = getTensor(I->getIndices());
  auto indicesDims = indicesT->getType().dims();
  const auto dim = I->getDim();
  const auto numElems = outT->getRealNumElements();
  auto ndims = indicesDims.size();

  std::vector<dim_t> ind_dim_off(ndims);
  std::vector<dim_t> data_dim_off(ndims);

  ind_dim_off[0] = 1;
  data_dim_off[0] = 1;
  for (dim_t i = 1; i < ndims; i++) {
    ind_dim_off[i] =
        std::accumulate(indicesDims.begin() + ndims - i, indicesDims.end(), 1,
                        std::multiplies<dim_t>());
    data_dim_off[i] =
        std::accumulate(dataDims.begin() + ndims - i, dataDims.end(), 1,
                        std::multiplies<dim_t>());
  }

  const auto dimPos = ndims - 1 - dim;
  const auto elemSize = dataT->getType().getElementSize();
  // Loop over number of elements in indices
  for (size_t idx = 0; idx < numElems; idx++) {
    unsigned_t offset = 0;
    // Loop over number of dimensions to calculate offset
    for (dim_t i = 0; i < ndims; i++) {
      // Calculate axis index i.e. (i, j, k)
      const dim_t dim_idx =
          static_cast<dim_t>(std::floor(idx / ind_dim_off[i])) %
          indicesDims[ndims - (i + 1)];
      offset += (dimPos != i) * dim_idx * data_dim_off[i];
    }
    auto indIdx = indicesT->getHandle<IndexTy>().raw(idx);
    assert(indIdx < 0 ? -indIdx <= dataDims[dim]
                      : indIdx < dataDims[dim] &&
                            "[GatherElements] Got out of bounds index");
    // In case of negative indices
    if (indIdx < 0) {
      indIdx += dataDims[dim];
    }
    const auto dataRawIdx = indIdx * data_dim_off[dimPos] + offset;
    std::copy(&dataT->getUnsafePtr()[dataRawIdx * elemSize],
              &dataT->getUnsafePtr()[(dataRawIdx + 1) * elemSize],
              &outT->getUnsafePtr()[idx * elemSize]);
  }
}

void BoundInterpreterFunction::fwdGatherElementsInst(
    const glow::GatherElementsInst *I) {
  switch (I->getIndices()->getElementType()) {
  case ElemKind::Int64ITy:
    fwdGatherElementsInstImpl<int64_t>(I);
    break;
  case ElemKind::Int32ITy:
    fwdGatherElementsInstImpl<int32_t>(I);
    break;
  default:
    llvm_unreachable("[GatherElements] Unsupported type for indices input of "
                     "GatherElements.");
  }
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdGatherNDInstImpl(
    const glow::GatherNDInst *I) {

  Tensor *dataT = getTensor(I->getData());
  Tensor *indicesT = getTensor(I->getIndices());
  Tensor *outT = getTensor(I->getDest());
  auto batchDims = I->getBatchDims();

  auto dataDims = I->getData()->dims();
  auto indicesDims = I->getIndices()->dims();
  dim_t indicesDimLast = indicesDims.back();

  // Compute batch count.
  dim_t batchCount = 1;
  for (size_t idx = 0; idx < batchDims; ++idx) {
    batchCount *= dataDims[idx];
  }

  // Compute input slice count.
  dim_t inpSliceCount = 1;
  for (size_t idx = batchDims; idx < batchDims + indicesDimLast; ++idx) {
    inpSliceCount *= dataDims[idx];
  }

  // Compute output slice count.
  dim_t outSliceCount = 1;
  for (size_t idx = batchDims; idx < indicesDims.size() - 1; ++idx) {
    outSliceCount *= indicesDims[idx];
  }

  // Compute slice size (in bytes).
  dim_t sliceSize = dataT->getType().getElementSize();
  for (size_t idx = batchDims + indicesDimLast; idx < dataDims.size(); idx++) {
    sliceSize *= dataDims[idx];
  }

  // Get indices dimension products.
  std::vector<dim_t> indicesDimProd(indicesDimLast);
  indicesDimProd[indicesDimLast - 1] = 1;
  for (ssize_t idx = static_cast<ssize_t>(indicesDimLast) - 2; idx >= 0;
       idx--) {
    indicesDimProd[idx] =
        indicesDimProd[idx + 1] * dataDims[batchDims + idx + 1];
  }

  // We will view the tensors as equivalent 3D tensors with the dimensions:
  // data    - batchCount x inpSliceCount x sliceSize
  // indices - batchCount x outSliceCount x indicesDimLast
  // output  - batchCount x outSliceCount x sliceSize

  char *dataPtr = dataT->getUnsafePtr();
  ElemTy *indicesPtr = (ElemTy *)indicesT->getUnsafePtr();
  char *outPtr = outT->getUnsafePtr();

  for (size_t batchIdx = 0; batchIdx < batchCount; ++batchIdx) {
    for (size_t outSliceIdx = 0; outSliceIdx < outSliceCount; ++outSliceIdx) {

      // Compute input slice index.
      dim_t inpSliceIdx = 0;
      for (size_t idx = 0; idx < indicesDimLast; ++idx) {
        inpSliceIdx += (*indicesPtr++) * indicesDimProd[idx];
      }

      // Copy data.
      std::copy(dataPtr + (inpSliceIdx + 0) * sliceSize,
                dataPtr + (inpSliceIdx + 1) * sliceSize, outPtr);
      outPtr += sliceSize;
    }

    // Increment input pointer for next batch.
    dataPtr += inpSliceCount * sliceSize;
  }
}

void BoundInterpreterFunction::fwdGatherNDInst(const glow::GatherNDInst *I) {
  switch (I->getIndices()->getElementType()) {
  case ElemKind::Int64ITy:
    fwdGatherNDInstImpl<int64_t>(I);
    break;
  case ElemKind::Int32ITy:
    fwdGatherNDInstImpl<int32_t>(I);
    break;
  default:
    llvm_unreachable("Unsupported type for indices input of Gather.");
  }
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdGatherRangesInstImpl(
    const glow::GatherRangesInst *I) {
  Tensor *dataT = getTensor(I->getData());
  auto &dataTy = dataT->getType();
  Tensor *rangesT = getTensor(I->getRanges());
  auto &rangesTy = rangesT->getType();
  Tensor *outT = getTensor(I->getOutput());
  Tensor *lengthsT = getTensor(I->getLengths());

  // Offset into the output tensor that keeps track of where to start
  // copying data.
  size_t outP = 0;

  unsigned dataElementSize = dataTy.getElementSize();
  dim_t numExamples = rangesTy.dims()[0];
  dim_t exampleSize = rangesTy.dims()[1];

  // Keep track of the total number of elements gathered across all
  // examples for a sanity check later.
  dim_t grandTotalLen = 0;

  // For each example in ranges:
  for (dim_t example = 0; example < numExamples; ++example) {
    // Keep a running total of the lengths of all ranges in this example
    // to record into lengthsT once the entire example is processed.
    ElemTy totalLen = 0;

    // For each range in the example:
    for (dim_t range = 0; range < exampleSize; ++range) {
      // Get the start index and range length.
      ElemTy startIdx = rangesT->getHandle<ElemTy>().at({example, range, 0});
      ElemTy len = rangesT->getHandle<ElemTy>().at({example, range, 1});

      // Add the length of this current range to the example length counter.
      totalLen += len;

      // Compute the start and end offsets.
      dim_t startOffset = startIdx * dataElementSize;
      dim_t endOffset = startOffset + (len * dataElementSize);

      // Sanity checks on the offsets.
      assert(startOffset < dataT->getSizeInBytes());
      assert(endOffset <= dataT->getSizeInBytes());
      assert(endOffset >= startOffset);
      assert(outP < outT->getSizeInBytes());
      assert((outP + (len * dataElementSize)) <= outT->getSizeInBytes());

      // Copy the specified data to outT.
      std::copy(&dataT->getUnsafePtr()[startOffset],
                &dataT->getUnsafePtr()[endOffset], &outT->getUnsafePtr()[outP]);

      // Advance the offset into outT.
      outP += len * dataElementSize;
    }

    // Record the total number of elements gathered for the example in
    // lengthsT.
    lengthsT->getHandle<ElemTy>().at({example}) = totalLen;

    // Add the total length of the entire example to the grand total.
    grandTotalLen += static_cast<size_t>(totalLen);
  }

  // Make sure that number of elements written to outT is equal to the
  // total of all elements in lengthsT.
  assert(grandTotalLen == (outP / dataElementSize));
}

void BoundInterpreterFunction::fwdGatherRangesInst(
    const glow::GatherRangesInst *I) {
  switch (I->getRanges()->getElementType()) {
  case ElemKind::Int64ITy:
    fwdGatherRangesInstImpl<int64_t>(I);
    break;
  case ElemKind::Int32ITy:
    fwdGatherRangesInstImpl<int32_t>(I);
    break;
  default:
    llvm_unreachable("Unsupported type for ranges input of GatherRanges.");
  }
}

template <typename ElemTy, typename IndicesElemTy>
void BoundInterpreterFunction::fwdScatterDataInstCopyImpl(
    const glow::ScatterDataInst *I) {
  Tensor *dataT = getTensor(I->getData());
  Tensor *indicesT = getTensor(I->getIndices());
  Tensor *slicesT = getTensor(I->getSlices());

  assert(indicesT->dims().size() == 2 &&
         "Index should be stored in 2D tensor!");
  const dim_t dataSliceSize = slicesT->size() / slicesT->dims()[0] *
                              slicesT->getType().getElementSize();

  auto IH = indicesT->getHandle<IndicesElemTy>();
  // For each index, copy from the slice at that index into the location in
  // data given the offset from the indices tensor.
  for (dim_t i = 0, end = indicesT->dims()[0]; i < end; i++) {
    dim_t destDataIdx = 0;
    for (dim_t j = 0, e = indicesT->dims()[1]; j < e; j++) {
      destDataIdx *= dataT->dims()[j];
      destDataIdx += IH.at({i, j});
    }
    std::copy(&slicesT->getUnsafePtr()[i * dataSliceSize],
              &slicesT->getUnsafePtr()[(i + 1) * dataSliceSize],
              &dataT->getUnsafePtr()[dataSliceSize * destDataIdx]);
  }
}

template <typename ElemTy, typename IndicesElemTy>
void BoundInterpreterFunction::fwdScatterDataInstAddFloatImpl(
    const glow::ScatterDataInst *I) {
  Tensor *dataT = getTensor(I->getData());
  Tensor *indicesT = getTensor(I->getIndices());
  Tensor *slicesT = getTensor(I->getSlices());

  assert(!dataT->getType().isQuantizedType() && "Should be float type!");
  assert(!slicesT->getType().isQuantizedType() && "Should be float type!");

  const size_t numSlices = slicesT->size() / slicesT->dims()[0];

  auto IH = indicesT->getHandle<IndicesElemTy>();
  // For each index, copy from the slice at that index into the location in
  // data given the offset from the indices tensor.
  assert(indicesT->dims().size() == 2 &&
         "Multi-dimensional index should be stored in 2D tensor!");
  auto D = dataT->getHandle<ElemTy>(), S = slicesT->getHandle<ElemTy>();
  for (dim_t i = 0, end = indicesT->dims()[0]; i < end; i++) {
    size_t destDataIdx = 0;
    for (dim_t j = 0, e = indicesT->dims()[1]; j < e; j++) {
      destDataIdx *= dataT->dims()[j];
      destDataIdx += IH.at({i, j});
    }
    for (dim_t j = 0; j < numSlices; j++) {
      D.raw(destDataIdx * numSlices + j) += S.raw(i * numSlices + j);
    }
  }
}

template <typename ElemTy, typename IndicesElemTy>
void BoundInterpreterFunction::fwdScatterDataInstAddQuantizedImpl(
    const glow::ScatterDataInst *I) {
  Tensor *dataT = getTensor(I->getData());
  Tensor *indicesT = getTensor(I->getIndices());
  Tensor *slicesT = getTensor(I->getSlices());

  assert(dataT->getType().isQuantizedType() && "Should be quantized type!");
  assert(slicesT->getType().isQuantizedType() && "Should be quantized type!");

  const dim_t numSlices = slicesT->size() / slicesT->dims()[0];

  TensorQuantizationParams dataQ{dataT->getType().getScale(),
                                 dataT->getType().getOffset()};
  TensorQuantizationParams sliceQ{slicesT->getType().getScale(),
                                  slicesT->getType().getOffset()};

  auto IH = indicesT->getHandle<IndicesElemTy>();
  // For each index, copy from the slice at that index into the location in
  // data given the offset from the indices tensor.
  assert(indicesT->dims().size() == 2 &&
         "Multi-dimensional index should be stored in 2D tensor!");
  auto D = dataT->getHandle<ElemTy>(), S = slicesT->getHandle<ElemTy>();
  for (dim_t i = 0, end = indicesT->dims()[0]; i < end; i++) {
    dim_t destDataIdx = 0;
    for (dim_t j = 0, e = indicesT->dims()[1]; j < e; j++) {
      destDataIdx *= dataT->dims()[j];
      destDataIdx += IH.at({i, j});
    }
    for (dim_t j = 0; j < numSlices; j++) {
      float lhs =
          quantization::dequantize(D.raw(destDataIdx * numSlices + j), dataQ);
      float rhs = quantization::dequantize(S.raw(i * numSlices + j), sliceQ);
      ElemTy result = quantization::quantize(lhs + rhs, dataQ);
      D.raw(destDataIdx * numSlices + j) = result;
    }
  }
}

void BoundInterpreterFunction::fwdScatterDataInst(
    const glow::ScatterDataInst *I) {
  const auto indicesAreInt64 =
      I->getIndices()->getElementType() == ElemKind::Int64ITy;

  if (I->getCumulative()) {
    switch (I->getData()->getElementType()) {
    case ElemKind::FloatTy:
      if (indicesAreInt64) {
        fwdScatterDataInstAddFloatImpl<float, int64_t>(I);
      } else {
        fwdScatterDataInstAddFloatImpl<float, int32_t>(I);
      }

      break;
    case ElemKind::Int8QTy:
      if (indicesAreInt64) {
        fwdScatterDataInstAddQuantizedImpl<int8_t, int64_t>(I);
      } else {
        fwdScatterDataInstAddQuantizedImpl<int8_t, int32_t>(I);
      }
      break;
    default:
      llvm_unreachable("Unsupported type for ScatterData.");
    }
  } else {
    switch (I->getData()->getElementType()) {
    case ElemKind::FloatTy:
      if (indicesAreInt64) {
        fwdScatterDataInstCopyImpl<float, int64_t>(I);
      } else {
        fwdScatterDataInstCopyImpl<float, int32_t>(I);
      }
      break;
    case ElemKind::Int8QTy:
      if (indicesAreInt64) {
        fwdScatterDataInstCopyImpl<int8_t, int64_t>(I);
      } else {
        fwdScatterDataInstCopyImpl<int8_t, int32_t>(I);
      }
      break;
    default:
      llvm_unreachable("Unsupported type for ScatterData.");
    }
  }
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdBatchOneHotImpl(
    const glow::BatchOneHotInst *I) {
  auto dataH = getWeightHandle<ElemTy>(I->getData());
  auto lengthsH = getWeightHandle<int32_t>(I->getLengths());
  auto valuesH = getWeightHandle<ElemTy>(I->getValues());
  auto destH = getWeightHandle<ElemTy>(I->getDest());

  auto batchSize = dataH.dims()[0];
  auto featureCnt = dataH.dims()[1];

  for (dim_t batchId = 0; batchId < batchSize; batchId++) {
    size_t offset = 0;
    for (dim_t featureId = 0; featureId < featureCnt; featureId++) {
      auto curValue = dataH.at({batchId, featureId});
      auto curLength = lengthsH.at({featureId});
      for (dim_t i = offset, e = offset + curLength; i != e; i++) {
        destH.at({batchId, i}) = curValue == valuesH.at({i});
      }
      offset += curLength;
    }
    assert(offset == destH.dims()[1] &&
           "Sum of Lengths must be equal to size of Values");
  }
}

void BoundInterpreterFunction::fwdBatchOneHotInst(
    const glow::BatchOneHotInst *I) {
  switch (I->getData()->getElementType()) {
  case ElemKind::Int64ITy:
    fwdBatchOneHotImpl<int64_t>(I);
    break;
  case ElemKind::Int32ITy:
    fwdBatchOneHotImpl<int32_t>(I);
    break;
  case ElemKind::Int8QTy:
    fwdBatchOneHotImpl<int8_t>(I);
    break;
  default:
    dispatchFloatingPointImpl(fwdBatchOneHotImpl,
                              I->getData()->getElementType(), I);
  }
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdSpaceToDepthInstImpl(
    const glow::SpaceToDepthInst *I) {
  auto *inT = getTensor(I->getSrc());
  auto *outT = getTensor(I->getDest());

  auto inH = inT->getHandle<ElemTy>();
  auto outH = outT->getHandle<ElemTy>();

  unsigned blockSize = I->getBlockSize();

  dim_t inDepth = inT->dims()[3];

  dim_t outBatch = outT->dims()[0];
  dim_t outHeight = outT->dims()[1];
  dim_t outWidth = outT->dims()[2];
  dim_t outDepth = outT->dims()[3];

  for (dim_t ob = 0; ob < outBatch; ++ob) {
    for (dim_t oh = 0; oh < outHeight; ++oh) {
      for (dim_t ow = 0; ow < outWidth; ++ow) {
        for (dim_t oc = 0; oc < outDepth; ++oc) {
          // Gets the block layer we are on
          dim_t blockDepthLayer = oc / inDepth;
          // every multiple of block size we reset to 0 offset
          dim_t iw = ow * blockSize + blockDepthLayer % blockSize;
          // every multiple of blockSize we start height traversal + 1
          dim_t ih = oh * blockSize + blockDepthLayer / blockSize;
          // at every multiple of inDepth index in to input depths resets to 0
          dim_t ic = oc % inDepth;

          outH.at({ob, oh, ow, oc}) = inH.at({ob, ih, iw, ic});
        }
      }
    }
  }
}

void BoundInterpreterFunction::fwdSpaceToDepthInst(
    const glow::SpaceToDepthInst *I) {
  switch (I->getSrc()->getElementType()) {
  case ElemKind::FloatTy:
    fwdSpaceToDepthInstImpl<float>(I);
    break;
  case ElemKind::Int8QTy:
    fwdSpaceToDepthInstImpl<int8_t>(I);
    break;
  default:
    llvm_unreachable("Type is not supported");
    break;
  }
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdResizeNearestInstImpl(
    const ResizeNearestInst *I) {
  auto inW = getWeightHandle<ElemTy>(I->getSrc());
  auto scale = I->getScale();
  auto outW = getWeightHandle<ElemTy>(I->getDest());

  auto outputDims = outW.dims();
  auto inputDims = inW.dims();

  for (dim_t oa = 0; oa < outputDims[0]; ++oa) {
    auto ia = std::min(dim_t(oa / scale[0]), inputDims[0] - 1);
    for (dim_t ob = 0; ob < outputDims[1]; ++ob) {
      auto ib = std::min(dim_t(ob / scale[1]), inputDims[1] - 1);
      for (dim_t oc = 0; oc < outputDims[2]; ++oc) {
        auto ic = std::min(dim_t(oc / scale[2]), inputDims[2] - 1);
        if (outputDims.size() > 3) {
          for (dim_t od = 0; od < outputDims[3]; ++od) {
            auto id = std::min(dim_t(od / scale[3]), inputDims[3] - 1);
            if (outputDims.size() > 4) {
              for (dim_t oe = 0; oe < outputDims[4]; ++oe) {
                auto ie = std::min(dim_t(oe / scale[4]), inputDims[4] - 1);
                if (outputDims.size() > 5) {
                  for (dim_t of = 0; of < outputDims[4]; ++of) {
                    auto f = std::min(dim_t(of / scale[5]), inputDims[5] - 1);
                    outW.at({oa, ob, oc, od, oe, of}) =
                        inW.at({ia, ib, ic, id, ie, f});
                  }
                } else {
                  outW.at({oa, ob, oc, od, oe}) = inW.at({ia, ib, ic, id, ie});
                }
              }
            } else {
              outW.at({oa, ob, oc, od}) = inW.at({ia, ib, ic, id});
            }
          }
        } else {
          outW.at({oa, ob, oc}) = inW.at({ia, ib, ic});
        }
      }
    }
  }
}

void BoundInterpreterFunction::fwdResizeNearestInst(
    const ResizeNearestInst *I) {
  if (getTensor(I->getSrc())->getType().isQuantizedType()) {
    dispatchQuantizedImpl(fwdResizeNearestInstImpl,
                          I->getSrc()->getElementType(), I);
    return;
  }

  dispatchImpl(fwdResizeNearestInstImpl, I->getSrc()->getElementType(), I);
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdResizeBilinearInstImpl(
    const ResizeBilinearInst *I) {
  auto inW = getWeightHandle<ElemTy>(I->getSrc());
  auto scale = I->getScale();
  auto outW = getWeightHandle<ElemTy>(I->getDest());

  ShapeNHWC odim(outW.dims());
  ShapeNHWC idim(inW.dims());

  CHECK_EQ(scale[0], 1.0) << "Scaling batch not supported.";
  CHECK_EQ(scale[3], 1.0) << "Scaling channel not supported.";

  for (dim_t ob = 0; ob < odim.n; ++ob) {
    for (dim_t oh = 0; oh < odim.h; ++oh) {
      for (dim_t ow = 0; ow < odim.w; ++ow) {

        float ihf = oh / scale[1];
        float iwf = ow / scale[2];
        dim_t ih = dim_t(ihf);
        dim_t iw = dim_t(iwf);

        auto ih0 = std::min(ih, idim.h - 1);
        auto ih1 = std::min(ih + 1, idim.h - 1);
        auto iw0 = std::min(iw, idim.w - 1);
        auto iw1 = std::min(iw + 1, idim.w - 1);

        for (dim_t oc = 0; oc < odim.c; ++oc) {
          auto v00 = inW.at({ob, ih0, iw0, oc});
          auto v01 = inW.at({ob, ih0, iw1, oc});
          auto v10 = inW.at({ob, ih1, iw0, oc});
          auto v11 = inW.at({ob, ih1, iw1, oc});

          auto hd = (float)v00 + (float)(v10 - v00) * (ihf - ih);
          auto hw = (float)v01 + (float)(v11 - v01) * (ihf - ih);
          float result = hd + (hw - hd) * (iwf - iw);
          outW.at({ob, oh, ow, oc}) = result;
        }
      }
    }
  }
}

void BoundInterpreterFunction::fwdResizeBilinearInst(
    const ResizeBilinearInst *I) {
  if (getTensor(I->getSrc())->getType().isQuantizedType()) {
    dispatchQuantizedImpl(fwdResizeBilinearInstImpl,
                          I->getSrc()->getElementType(), I);
    return;
  }

  dispatchImpl(fwdResizeBilinearInstImpl, I->getSrc()->getElementType(), I);
}

//===----------------------------------------------------------------------===//
//                      Local Response Normalization
//===----------------------------------------------------------------------===//

template <typename ElemTy>
void BoundInterpreterFunction::fwdLocalResponseNormalizationInstFloatImpl(
    const glow::LocalResponseNormalizationInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto inW = getWeightHandle<ElemTy>(I->getSrc());
  auto outW = getWeightHandle<ElemTy>(I->getDest());
  auto scaleCache = getWeightHandle<ElemTy>(I->getScale());

  ShapeNHWC odim(outW.dims());
  ShapeNHWC idim(inW.dims());

  (void)odim;

  // LRN node does not change the shape of the input.
  assert(odim == idim && "Output of LRN node must be same shape as input");

  // LRN node normalizes across channels, so the input must have a minimum
  // depth of 1.
  assert(idim.c > 0 && "Input of LRN node must have a minimum depth of 1");

  auto halfWindowSize = (size_t)I->getHalfWindowSize();
  auto k = I->getK();
  auto beta = I->getBeta();
  auto windowSize = 2 * halfWindowSize + 1;
  auto normedAlpha = I->getAlpha() / windowSize;

  // For every input in the batch:
  for (dim_t n = 0; n < idim.n; n++) {

    // For every row:
    for (dim_t h = 0; h < idim.h; h++) {

      // For every column:
      for (dim_t w = 0; w < idim.w; w++) {

        // For every channel:
        for (dim_t c = 0; c < idim.c; c++) {
          float squareSum = 0.0;
          for (dim_t i = (c >= halfWindowSize ? c - halfWindowSize : 0);
               i <= std::min<dim_t>(c + halfWindowSize, (size_t)idim.c - 1);
               i++) {
            float val = inW.at({n, h, w, i});
            squareSum += val * val;
          }

          auto scale = k + normedAlpha * squareSum;

          // This will be used to accelerate the backward pass.
          scaleCache.at({n, h, w, c}) = ElemTy(scale);

          auto normFactor = std::pow(scale, -beta);
          outW.at({n, h, w, c}) =
              ElemTy(float(inW.at({n, h, w, c})) * normFactor);
        }
      }
    }
  }
}

void BoundInterpreterFunction::fwdLocalResponseNormalizationInst(
    const LocalResponseNormalizationInst *I) {
  dispatchFloatingPointImpl(fwdLocalResponseNormalizationInstFloatImpl,
                            I->getSrc()->getElementType(), I);
}

void BoundInterpreterFunction::fwdLocalResponseNormalizationGradInst(
    const glow::LocalResponseNormalizationGradInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto inG = getWeightHandle(I->getSrcGrad());
  auto outW = getWeightHandle(I->getDest());
  auto outG = getWeightHandle(I->getDestGrad());
  auto scaleCache = getWeightHandle(I->getScale());

  ShapeNHWC odim(outW.dims());

  auto halfWindowSize = I->getHalfWindowSize();
  auto beta = I->getBeta();
  auto windowSize = 2 * halfWindowSize + 1;
  auto normedAlpha = I->getAlpha() / windowSize;

  // For every input in the batch:
  for (dim_t n = 0; n < odim.n; n++) {

    // For every row:
    for (dim_t h = 0; h < odim.h; h++) {

      // For every column:
      for (dim_t w = 0; w < odim.w; w++) {

        float sum = 0.0;

        // Compute sum for first channel.
        for (dim_t c = 0; c <= halfWindowSize && c < odim.c; c++) {
          auto outw = outW.at({n, h, w, c});
          auto scale = scaleCache.at({n, h, w, c});
          auto outg = outG.at({n, h, w, c});
          sum += (outg * (outw / scale));
        }

        // For every channel:
        for (dim_t c = 0; c < odim.c; c++) {
          auto outg = outG.at({n, h, w, c});
          auto scale = scaleCache.at({n, h, w, c});
          auto inw = inW.at({n, h, w, c});

          inG.at({n, h, w, c}) = outg * std::pow(scale, -beta) -
                                 2 * normedAlpha * beta * inw * sum;

          // Modify sum for next channel.
          auto subIndex = c - halfWindowSize;
          auto addIndex = c + halfWindowSize + 1;

          if (c >= halfWindowSize) {
            auto outw = outW.at({n, h, w, subIndex});
            auto scale = scaleCache.at({n, h, w, subIndex});
            auto outg = outG.at({n, h, w, subIndex});

            // Subtract "rear" end of this window.
            sum -= (outg * (outw / scale));
          }

          if (addIndex < odim.c) {
            auto outw = outW.at({n, h, w, addIndex});
            auto scale = scaleCache.at({n, h, w, addIndex});
            auto outg = outG.at({n, h, w, addIndex});

            // Add "front" end of next window.
            sum += (outg * (outw / scale));
          }
        }
      }
    }
  }
}

//===--------------------------------------------------------------------===//
//                     Bucketing
//===--------------------------------------------------------------------===//

void BoundInterpreterFunction::fwdBucketizeInst(const BucketizeInst *I) {
  auto inputH = getTensor(I->getSrc())->getHandle<float>();
  auto outputH = getTensor(I->getDest())->getHandle<int32_t>();
  const auto boundaries = I->getBoundaries();

  const auto numItems = inputH.size();

  for (size_t i = 0; i < numItems; ++i) {
    outputH.raw(i) =
        std::lower_bound(boundaries.begin(), boundaries.end(), inputH.raw(i)) -
        boundaries.begin();
  }
}

//===----------------------------------------------------------------------===//
//                       Arithmetic operations
//===----------------------------------------------------------------------===//
void BoundInterpreterFunction::fwdElementAddInstI8Impl(
    const ElementAddInst *I) {
  assert(getTensor(I->getLHS())->getType().isQuantizedType() &&
         "Wrong function");
  auto lhsTy = I->getLHS()->getType();
  auto rhsTy = I->getRHS()->getType();
  auto destTy = I->getDest()->getType();

  float lhsScale = lhsTy->getScale();
  float rhsScale = rhsTy->getScale();
  float destScale = destTy->getScale();

  int32_t lhsOffset = lhsTy->getOffset();
  int32_t rhsOffset = rhsTy->getOffset();
  int32_t destOffset = destTy->getOffset();

  auto outW = getWeightHandle<int8_t>(I->getDest());
  auto lhsW = getWeightHandle<int8_t>(I->getLHS());
  auto rhsW = getWeightHandle<int8_t>(I->getRHS());
  for (dim_t i = 0, e = outW.size(); i < e; i++) {
    int32_t L = lhsW.raw(i);
    int32_t R = rhsW.raw(i);

    // We increase the size of the integer up to 16 bits to prevent overflow.
    const float largeScale = float(1) / (1 << 15);
    // Scale both sides from 8-bit to 16-bits.
    int32_t L32 = std::round(float(L - lhsOffset) * (lhsScale / largeScale));
    int32_t R32 = std::round(float(R - rhsOffset) * (rhsScale / largeScale));
    int32_t sum32 = L32 + R32;
    sum32 = std::round(float(sum32) * (largeScale / destScale) + destOffset);
    outW.raw(i) = quantization::clip<int32_t, int8_t>(sum32);
  }
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdElementAddInstArithmeticImpl(
    const ElementAddInst *I) {
  staticAssertArithmeticType(ElemTy);

  auto outW = getWeightHandle<ElemTy>(I->getDest());
  auto lhsW = getWeightHandle<ElemTy>(I->getLHS());
  auto rhsW = getWeightHandle<ElemTy>(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = lhsW.raw(i) + rhsW.raw(i);
  }
}

void BoundInterpreterFunction::fwdElementAddInst(const ElementAddInst *I) {
  if (getTensor(I->getLHS())->getType().isQuantizedType()) {
    fwdElementAddInstI8Impl(I);
    return;
  }

  dispatchArithmeticImpl(fwdElementAddInstArithmeticImpl,
                         I->getLHS()->getType()->getElementType(), I);
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdElementSubInstArithmeticImpl(
    const ElementSubInst *I) {
  staticAssertArithmeticType(ElemTy);

  auto outW = getWeightHandle<ElemTy>(I->getDest());
  auto lhsW = getWeightHandle<ElemTy>(I->getLHS());
  auto rhsW = getWeightHandle<ElemTy>(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = lhsW.raw(i) - rhsW.raw(i);
  }
}

void BoundInterpreterFunction::fwdElementSubInst(const ElementSubInst *I) {
  if (getTensor(I->getLHS())->getType().isQuantizedType()) {
    auto destTy = I->getDest()->getType();
    auto lhsTy = I->getLHS()->getType();
    auto rhsTy = I->getRHS()->getType();

    float destScale = destTy->getScale();
    float lhsScale = lhsTy->getScale();
    float rhsScale = rhsTy->getScale();

    int32_t destOffset = destTy->getOffset();
    int32_t lhsOffset = lhsTy->getOffset();
    int32_t rhsOffset = rhsTy->getOffset();

    auto outW = getWeightHandle<int8_t>(I->getDest());
    auto lhsW = getWeightHandle<int8_t>(I->getLHS());
    auto rhsW = getWeightHandle<int8_t>(I->getRHS());
    for (size_t i = 0, e = outW.size(); i < e; i++) {
      //    s_d * (i_d - o_d) = s_l * (i_l - o_l) - s_r * (i_r - o_r)
      // => i_d = (s_l / s_d) * (i_l - o_l) - (s_r / s_d) * (i_r - o_r) + o_d
      float l = (lhsScale / destScale) * float(lhsW.raw(i) - lhsOffset);
      float r = (rhsScale / destScale) * float(rhsW.raw(i) - rhsOffset);
      int32_t q = std::round(l - r + destOffset);
      outW.raw(i) = quantization::clip<int32_t, int8_t>(q);
    }
    return;
  }

  dispatchArithmeticImpl(fwdElementSubInstArithmeticImpl,
                         I->getDest()->getElementType(), I);
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdElementMulInstArithmeticImpl(
    const ElementMulInst *I) {
  staticAssertArithmeticType(ElemTy);

  auto outW = getWeightHandle<ElemTy>(I->getDest());
  auto lhsW = getWeightHandle<ElemTy>(I->getLHS());
  auto rhsW = getWeightHandle<ElemTy>(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = lhsW.raw(i) * rhsW.raw(i);
  }
}

void BoundInterpreterFunction::fwdElementMulInst(const ElementMulInst *I) {
  if (getTensor(I->getLHS())->getType().isQuantizedType()) {
    auto lhsTy = I->getLHS()->getType();
    auto rhsTy = I->getRHS()->getType();
    auto destTy = I->getDest()->getType();

    TensorQuantizationParams lhsQ{lhsTy->getScale(), lhsTy->getOffset()};
    TensorQuantizationParams rhsQ{rhsTy->getScale(), rhsTy->getOffset()};
    TensorQuantizationParams destQ{destTy->getScale(), destTy->getOffset()};

    auto outW = getWeightHandle<int8_t>(I->getDest());
    auto lhsW = getWeightHandle<int8_t>(I->getLHS());
    auto rhsW = getWeightHandle<int8_t>(I->getRHS());
    float scale = lhsQ.scale * rhsQ.scale / destQ.scale;
    for (size_t i = 0, e = outW.size(); i < e; i++) {
      int32_t mul = (lhsW.raw(i) - lhsQ.offset) * (rhsW.raw(i) - rhsQ.offset);
      outW.raw(i) = quantization::clip<int32_t, int8_t>(
          std::round(mul * scale) + destQ.offset);
    }
    return;
  }

  dispatchArithmeticImpl(fwdElementMulInstArithmeticImpl,
                         I->getDest()->getElementType(), I);
}

void BoundInterpreterFunction::fwdElementFmodInst(const ElementFmodInst *I) {
  if (getTensor(I->getLHS())->getType().isQuantizedType()) {
    auto destTy = I->getDest()->getType();
    auto lhsTy = I->getLHS()->getType();
    auto rhsTy = I->getRHS()->getType();

    float destScale = destTy->getScale();
    float lhsScale = lhsTy->getScale();
    float rhsScale = rhsTy->getScale();

    int32_t destOffset = destTy->getOffset();
    int32_t lhsOffset = lhsTy->getOffset();
    int32_t rhsOffset = rhsTy->getOffset();

    auto outW = getWeightHandle<int8_t>(I->getDest());
    auto lhsW = getWeightHandle<int8_t>(I->getLHS());
    auto rhsW = getWeightHandle<int8_t>(I->getRHS());
    for (size_t i = 0, e = outW.size(); i < e; i++) {
      float l = lhsScale * float(lhsW.raw(i) - lhsOffset);
      float r = rhsScale * destScale * float(rhsW.raw(i) - rhsOffset);
      int32_t q = std::round(std::fmod(l, r) + destOffset);
      outW.raw(i) = quantization::clip<int32_t, int8_t>(q);
    }
    return;
  }

#define FMOD_LOOP(TYPE_)                                                       \
  staticAssertArithmeticType(TYPE_);                                           \
  auto outW = getWeightHandle<TYPE_>(I->getDest());                            \
  auto lhsW = getWeightHandle<TYPE_>(I->getLHS());                             \
  auto rhsW = getWeightHandle<TYPE_>(I->getRHS());                             \
  for (size_t i = 0, e = outW.size(); i < e; i++) {                            \
    outW.raw(i) = std::fmod((float)lhsW.raw(i), (float)rhsW.raw(i));           \
  }

  auto *T = getTensor(I->getDest());
  switch (T->getElementType()) {
  case ElemKind::Int64ITy: {
    FMOD_LOOP(int64_t);
    return;
  }
  case ElemKind::Int32ITy: {
    FMOD_LOOP(int32_t);
    return;
  }
  case ElemKind::FloatTy: {
    FMOD_LOOP(float);
    return;
  }
  case ElemKind::Float16Ty: {
    FMOD_LOOP(float16_t);
    return;
  }
  case ElemKind::BFloat16Ty: {
    FMOD_LOOP(bfloat16_t);
    return;
  }

  default:
    llvm_unreachable("Unsupported type for Fmod.");
  }
}

void BoundInterpreterFunction::fwdElementDivInst(const ElementDivInst *I) {
  if (getTensor(I->getLHS())->getType().isQuantizedType()) {
    auto destTy = I->getDest()->getType();
    auto lhsTy = I->getLHS()->getType();
    auto rhsTy = I->getRHS()->getType();

    float destScale = destTy->getScale();
    float lhsScale = lhsTy->getScale();
    float rhsScale = rhsTy->getScale();

    int32_t destOffset = destTy->getOffset();
    int32_t lhsOffset = lhsTy->getOffset();
    int32_t rhsOffset = rhsTy->getOffset();

    auto outW = getWeightHandle<int8_t>(I->getDest());
    auto lhsW = getWeightHandle<int8_t>(I->getLHS());
    auto rhsW = getWeightHandle<int8_t>(I->getRHS());
    for (size_t i = 0, e = outW.size(); i < e; i++) {
      //    s_d * (i_d - o_d) = (s_l * (i_l - o_l)) / (s_r * (i_r - o_r))
      // => i_d = (s_l * (i_l - o_l)) / (s_d * s_r * (i_r - o_r)) + o_d
      float l = lhsScale * float(lhsW.raw(i) - lhsOffset);
      float r = rhsScale * destScale * float(rhsW.raw(i) - rhsOffset);
      int32_t q = std::round(l / r + destOffset);
      outW.raw(i) = quantization::clip<int32_t, int8_t>(q);
    }
    return;
  }

#define DIV_LOOP(TYPE_)                                                        \
  auto outW = getWeightHandle<TYPE_>(I->getDest());                            \
  auto lhsW = getWeightHandle<TYPE_>(I->getLHS());                             \
  auto rhsW = getWeightHandle<TYPE_>(I->getRHS());                             \
  for (size_t i = 0, e = outW.size(); i < e; i++) {                            \
    outW.raw(i) = lhsW.raw(i) / rhsW.raw(i);                                   \
  }

  auto *T = getTensor(I->getDest());
  switch (T->getElementType()) {
  case ElemKind::Int64ITy: {
    DIV_LOOP(int64_t);
    return;
  }
  case ElemKind::Int32ITy: {
    DIV_LOOP(int32_t);
    return;
  }
  case ElemKind::FloatTy: {
    DIV_LOOP(float);
    return;
  }
  case ElemKind::Float16Ty: {
    DIV_LOOP(float16_t);
    return;
  }
  case ElemKind::BFloat16Ty: {
    DIV_LOOP(bfloat16_t);
    return;
  }
  default:
    llvm_unreachable("Unsupported type for Div.");
  }
}

void BoundInterpreterFunction::fwdElementMaxInstI8Impl(
    const ElementMaxInst *I) {
  assert(getTensor(I->getLHS())->getType().isQuantizedType() &&
         "Wrong function");
  auto lhsTy = I->getLHS()->getType();
  auto rhsTy = I->getRHS()->getType();
  auto destTy = I->getDest()->getType();

  TensorQuantizationParams lhsQ{lhsTy->getScale(), lhsTy->getOffset()};
  TensorQuantizationParams rhsQ{rhsTy->getScale(), rhsTy->getOffset()};
  TensorQuantizationParams destQ{destTy->getScale(), destTy->getOffset()};

  auto outW = getWeightHandle<int8_t>(I->getDest());
  auto lhsW = getWeightHandle<int8_t>(I->getLHS());
  auto rhsW = getWeightHandle<int8_t>(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    // Convert both sides to the destination scale and perform a regular
    // comparison.
    int8_t L = quantization::quantize(
        quantization::dequantize(lhsW.raw(i), lhsQ), destQ);
    int8_t R = quantization::quantize(
        quantization::dequantize(rhsW.raw(i), rhsQ), destQ);
    outW.raw(i) = std::max(L, R);
  }
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdElementMaxInstArithmeticImpl(
    const ElementMaxInst *I) {
  staticAssertArithmeticType(ElemTy);

  auto outW = getWeightHandle<ElemTy>(I->getDest());
  auto lhsW = getWeightHandle<ElemTy>(I->getLHS());
  auto rhsW = getWeightHandle<ElemTy>(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = std::max(lhsW.raw(i), rhsW.raw(i));
  }
}

void BoundInterpreterFunction::fwdElementMaxInst(const ElementMaxInst *I) {
  if (getTensor(I->getLHS())->getType().isQuantizedType()) {
    fwdElementMaxInstI8Impl(I);
    return;
  }

  dispatchArithmeticImpl(fwdElementMaxInstArithmeticImpl,
                         I->getLHS()->getType()->getElementType(), I);
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdElementMinInstArithmeticImpl(
    const ElementMinInst *I) {
  staticAssertArithmeticType(ElemTy);

  auto outW = getWeightHandle<ElemTy>(I->getDest());
  auto lhsW = getWeightHandle<ElemTy>(I->getLHS());
  auto rhsW = getWeightHandle<ElemTy>(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = std::min(lhsW.raw(i), rhsW.raw(i));
  }
}

void BoundInterpreterFunction::fwdElementMinInst(const ElementMinInst *I) {
  if (getTensor(I->getLHS())->getType().isQuantizedType()) {
    auto lhsTy = I->getLHS()->getType();
    auto rhsTy = I->getRHS()->getType();
    auto destTy = I->getDest()->getType();

    TensorQuantizationParams lhsQ{lhsTy->getScale(), lhsTy->getOffset()};
    TensorQuantizationParams rhsQ{rhsTy->getScale(), rhsTy->getOffset()};
    TensorQuantizationParams destQ{destTy->getScale(), destTy->getOffset()};

    auto outW = getWeightHandle<int8_t>(I->getDest());
    auto lhsW = getWeightHandle<int8_t>(I->getLHS());
    auto rhsW = getWeightHandle<int8_t>(I->getRHS());
    for (size_t i = 0, e = outW.size(); i < e; i++) {
      // Convert both sides to the destination scale and perform a regular
      // comparison.
      int8_t L = quantization::quantize(
          quantization::dequantize(lhsW.raw(i), lhsQ), destQ);
      int8_t R = quantization::quantize(
          quantization::dequantize(rhsW.raw(i), rhsQ), destQ);
      outW.raw(i) = std::min(L, R);
    }
    return;
  }

  dispatchArithmeticImpl(fwdElementMinInstArithmeticImpl,
                         I->getDest()->getElementType(), I);
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdElementBitwiseOrInstImpl(
    const ElementBitwiseOrInst *I) {

  auto outW = getWeightHandle<ElemTy>(I->getDest());
  auto lhsW = getWeightHandle<ElemTy>(I->getLHS());
  auto rhsW = getWeightHandle<ElemTy>(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = lhsW.raw(i) | rhsW.raw(i);
  }
}

void BoundInterpreterFunction::fwdElementBitwiseOrInst(
    const ElementBitwiseOrInst *I) {

  dispatchBitwiseImpl(fwdElementBitwiseOrInstImpl,
                      I->getDest()->getElementType(), I);
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdElementBitwiseAndInstImpl(
    const ElementBitwiseAndInst *I) {

  auto outW = getWeightHandle<ElemTy>(I->getDest());
  auto lhsW = getWeightHandle<ElemTy>(I->getLHS());
  auto rhsW = getWeightHandle<ElemTy>(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = lhsW.raw(i) & rhsW.raw(i);
  }
}

void BoundInterpreterFunction::fwdElementBitwiseAndInst(
    const ElementBitwiseAndInst *I) {

  dispatchBitwiseImpl(fwdElementBitwiseAndInstImpl,
                      I->getDest()->getElementType(), I);
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdElementBitwiseXorInstImpl(
    const ElementBitwiseXorInst *I) {
  staticAssertArithmeticType(ElemTy);

  auto outW = getWeightHandle<ElemTy>(I->getDest());
  auto lhsW = getWeightHandle<ElemTy>(I->getLHS());
  auto rhsW = getWeightHandle<ElemTy>(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = lhsW.raw(i) ^ rhsW.raw(i);
  }
}

void BoundInterpreterFunction::fwdElementBitwiseXorInst(
    const ElementBitwiseXorInst *I) {

  dispatchBitwiseImpl(fwdElementBitwiseXorInstImpl,
                      I->getDest()->getElementType(), I);
}

//===----------------------------------------------------------------------===//
//                              Logical operations
//===----------------------------------------------------------------------===//
void BoundInterpreterFunction::fwdElementNotInst(const ElementNotInst *I) {
  auto inpW = getWeightHandle<bool>(I->getSrc());
  auto outW = getWeightHandle<bool>(I->getDest());
  for (size_t i = 0, e = outW.size(); i < e; ++i) {
    outW.raw(i) = (!inpW.raw(i));
  }
}

void BoundInterpreterFunction::fwdElementAndInst(const ElementAndInst *I) {
  auto lhsW = getWeightHandle<bool>(I->getLHS());
  auto rhsW = getWeightHandle<bool>(I->getRHS());
  auto outW = getWeightHandle<bool>(I->getDest());
  for (size_t i = 0, e = outW.size(); i < e; ++i) {
    outW.raw(i) = (lhsW.raw(i) && rhsW.raw(i));
  }
}

void BoundInterpreterFunction::fwdElementOrInst(const ElementOrInst *I) {
  auto lhsW = getWeightHandle<bool>(I->getLHS());
  auto rhsW = getWeightHandle<bool>(I->getRHS());
  auto outW = getWeightHandle<bool>(I->getDest());
  for (size_t i = 0, e = outW.size(); i < e; ++i) {
    outW.raw(i) = (lhsW.raw(i) || rhsW.raw(i));
  }
}

void BoundInterpreterFunction::fwdElementXorInst(const ElementXorInst *I) {
  auto lhsW = getWeightHandle<bool>(I->getLHS());
  auto rhsW = getWeightHandle<bool>(I->getRHS());
  auto outW = getWeightHandle<bool>(I->getDest());
  for (size_t i = 0, e = outW.size(); i < e; ++i) {
    outW.raw(i) = (lhsW.raw(i) ^ rhsW.raw(i));
  }
}

//===----------------------------------------------------------------------===//
//                         Unary arithmetic operations
//===----------------------------------------------------------------------===//
template <typename ElemTy, typename InstKind>
void BoundInterpreterFunction::fwdUnaryArithmeticImpl(
    const InstKind *I, std::function<float(float)> func) {
  Value *inpV = I->getSrc();
  Value *outV = I->getDest();
  auto inpTy = inpV->getType();
  auto outTy = outV->getType();
  auto inpH = getWeightHandle<ElemTy>(inpV);
  auto outH = getWeightHandle<ElemTy>(outV);

  if (inpTy->isQuantizedType()) {
    float inpScale = inpTy->getScale();
    int32_t inpOffset = inpTy->getOffset();
    float outScale = outTy->getScale();
    int32_t outOffset = outTy->getOffset();
    for (size_t i = 0, e = outH.size(); i < e; ++i) {
      float inpVal =
          quantization::dequantize<ElemTy>(inpH.raw(i), {inpScale, inpOffset});
      float outVal = func(inpVal);
      outH.raw(i) =
          quantization::quantize<ElemTy>(outVal, {outScale, outOffset});
    }
  } else {
    for (size_t i = 0, e = outH.size(); i < e; ++i) {
      float inpVal = static_cast<float>(inpH.raw(i));
      float outVal = func(inpVal);
      outH.raw(i) = static_cast<ElemTy>(outVal);
    }
  }
}

void BoundInterpreterFunction::fwdElementBitwiseNotInst(
    const ElementBitwiseNotInst *I) {
  auto func = [](int64_t i) -> int64_t { return ~i; };
  dispatchImpl(fwdUnaryArithmeticImpl, I->getSrc()->getElementType(), I, func);
}

void BoundInterpreterFunction::fwdElementAbsInst(const ElementAbsInst *I) {
  auto func = [](float x) -> float { return std::abs(x); };
  dispatchImpl(fwdUnaryArithmeticImpl, I->getSrc()->getElementType(), I, func);
}

void BoundInterpreterFunction::fwdElementNegInst(const ElementNegInst *I) {
  auto func = [](float x) -> float { return -x; };
  dispatchImpl(fwdUnaryArithmeticImpl, I->getSrc()->getElementType(), I, func);
}

void BoundInterpreterFunction::fwdElementFloorInst(const ElementFloorInst *I) {
  auto func = [](float x) -> float { return std::floor(x); };
  dispatchImpl(fwdUnaryArithmeticImpl, I->getSrc()->getElementType(), I, func);
}

void BoundInterpreterFunction::fwdElementSignInst(const ElementSignInst *I) {
  auto func = [](float x) -> float { return ((x > 0) - (x < 0)); };
  dispatchImpl(fwdUnaryArithmeticImpl, I->getSrc()->getElementType(), I, func);
}

void BoundInterpreterFunction::fwdElementCeilInst(const ElementCeilInst *I) {
  auto func = [](float x) -> float { return std::ceil(x); };
  dispatchImpl(fwdUnaryArithmeticImpl, I->getSrc()->getElementType(), I, func);
}

void BoundInterpreterFunction::fwdElementTruncateInst(
    const ElementTruncateInst *I) {
  auto func = [](float x) -> float { return std::trunc(x); };
  dispatchImpl(fwdUnaryArithmeticImpl, I->getSrc()->getElementType(), I, func);
}

void BoundInterpreterFunction::fwdElementRoundInst(const ElementRoundInst *I) {
  // Rounding mode required by ONNX, Numpy, TensorFlow is round to even which
  // rounds to nearest even integer those values with fractional part 0.5.
  auto func = [](float x) -> float { return std::nearbyintf(x); };
  dispatchImpl(fwdUnaryArithmeticImpl, I->getSrc()->getElementType(), I, func);
}

void BoundInterpreterFunction::fwdElementSqrtInst(const ElementSqrtInst *I) {
  auto func = [](float x) -> float { return std::sqrt(x); };
  dispatchImpl(fwdUnaryArithmeticImpl, I->getSrc()->getElementType(), I, func);
}

void BoundInterpreterFunction::fwdElementRsqrtInst(const ElementRsqrtInst *I) {
  auto func = [](float x) -> float { return 1 / std::sqrt(x); };
  dispatchImpl(fwdUnaryArithmeticImpl, I->getSrc()->getElementType(), I, func);
}

void BoundInterpreterFunction::fwdElementReciprocalInst(
    const ElementReciprocalInst *I) {
  auto func = [](float x) -> float { return 1 / x; };
  dispatchImpl(fwdUnaryArithmeticImpl, I->getSrc()->getElementType(), I, func);
}

void BoundInterpreterFunction::fwdElementSinInst(const ElementSinInst *I) {
  auto func = [](float x) -> float { return std::sin(x); };
  dispatchImpl(fwdUnaryArithmeticImpl, I->getSrc()->getElementType(), I, func);
}

void BoundInterpreterFunction::fwdElementCosInst(const ElementCosInst *I) {
  auto func = [](float x) -> float { return std::cos(x); };
  dispatchImpl(fwdUnaryArithmeticImpl, I->getSrc()->getElementType(), I, func);
}

void BoundInterpreterFunction::fwdElementErfInst(const ElementErfInst *I) {
  auto func = [](float x) -> float { return std::erf(x); };
  dispatchImpl(fwdUnaryArithmeticImpl, I->getSrc()->getElementType(), I, func);
}

//===----------------------------------------------------------------------===//
//                              Compare operations
//===----------------------------------------------------------------------===//
template <typename ElemTy, typename ElemOffsetTy, typename ElemScaleTy,
          typename CmpTy, typename InstCmpKind>
void BoundInterpreterFunction::fwdElementCmpHelperImpl(
    const InstCmpKind *I, std::function<bool(CmpTy LHS, CmpTy RHS)> cmpHelper) {
  Value *lhsV = I->getLHS();
  Value *rhsV = I->getRHS();
  Value *outV = I->getDest();

  auto lhsH = getWeightHandle<ElemTy>(lhsV);
  auto rhsH = getWeightHandle<ElemTy>(rhsV);
  auto oH = getWeightHandle<bool>(outV);

  ElemScaleTy lhsScale = 1.0f;
  ElemScaleTy rhsScale = 1.0f;
  ElemOffsetTy lhsOffset = 0;
  ElemOffsetTy rhsOffset = 0;

  auto lhsTy = lhsV->getType();
  auto rhsTy = rhsV->getType();

  if (lhsV->getType()->isQuantizedType()) {
    lhsScale = lhsTy->getScale();
    rhsScale = rhsTy->getScale();

    lhsOffset = lhsTy->getOffset();
    rhsOffset = rhsTy->getOffset();
  }

  // For each layer in the batch:
  for (size_t i = 0, e = oH.size(); i < e; i++) {
    oH.raw(i) = cmpHelper(lhsScale * (lhsH.raw(i) - lhsOffset),
                          rhsScale * (rhsH.raw(i) - rhsOffset));
  }
}

template <typename ElemTy, typename ElemOffsetTy, typename ElemScaleTy,
          typename CmpTy>
void BoundInterpreterFunction::fwdElementCmpLTEInstImpl(
    const ElementCmpLTEInst *I) {
  auto cmpHelper = [](CmpTy LHS, CmpTy RHS) -> bool { return LHS <= RHS; };
  fwdElementCmpHelperImpl<ElemTy, ElemOffsetTy, ElemScaleTy, CmpTy,
                          ElementCmpLTEInst>(I, cmpHelper);
}

void BoundInterpreterFunction::fwdElementCmpLTEInst(
    const ElementCmpLTEInst *I) {
  auto *T = getTensor(I->getLHS());

  if (T->getType().isQuantizedType()) {
    switch (T->getElementType()) {
    case ElemKind::Int8QTy:
      fwdElementCmpLTEInstImpl<int8_t, int32_t, float, int32_t>(I);
      break;
    case ElemKind::Int16QTy:
      fwdElementCmpLTEInstImpl<int16_t, int32_t, float, int32_t>(I);
      break;
    default:
      llvm_unreachable("Type is not supported");
    }
    return;
  }

  switch (T->getElementType()) {
  case ElemKind::FloatTy:
    fwdElementCmpLTEInstImpl<float, float, float>(I);
    break;
  case ElemKind::Float16Ty:
    fwdElementCmpLTEInstImpl<float16_t, float16_t, float16_t>(I);
    break;
  case ElemKind::BFloat16Ty:
    fwdElementCmpLTEInstImpl<bfloat16_t, bfloat16_t, bfloat16_t>(I);
    break;
  case ElemKind::Int32ITy:
    fwdElementCmpLTEInstImpl<int32_t, int32_t, float>(I);
    break;
  case ElemKind::Int64ITy:
    fwdElementCmpLTEInstImpl<int64_t, int64_t, float>(I);
    break;
  default:
    llvm_unreachable("Type is not supported");
  }
}

template <typename ElemTy, typename ElemOffsetTy, typename ElemScaleTy,
          typename CmpTy>
void BoundInterpreterFunction::fwdElementCmpEQInstImpl(
    const ElementCmpEQInst *I) {
  auto cmpHelper = [](CmpTy LHS, CmpTy RHS) -> bool { return LHS == RHS; };
  fwdElementCmpHelperImpl<ElemTy, ElemOffsetTy, ElemScaleTy, CmpTy,
                          ElementCmpEQInst>(I, cmpHelper);
}

void BoundInterpreterFunction::fwdElementCmpEQInst(const ElementCmpEQInst *I) {
  auto *T = getTensor(I->getLHS());

  if (T->getType().isQuantizedType()) {
    switch (T->getElementType()) {
    case ElemKind::Int8QTy:
      fwdElementCmpEQInstImpl<int8_t, int32_t, float, int32_t>(I);
      break;
    case ElemKind::Int16QTy:
      fwdElementCmpEQInstImpl<int16_t, int32_t, float, int32_t>(I);
      break;
    default:
      llvm_unreachable("Type is not supported");
    }
    return;
  }

  switch (T->getElementType()) {
  case ElemKind::FloatTy:
    fwdElementCmpEQInstImpl<float, float, float>(I);
    break;
  case ElemKind::Float16Ty:
    fwdElementCmpEQInstImpl<float16_t, float16_t, float16_t>(I);
    break;
  case ElemKind::BFloat16Ty:
    fwdElementCmpEQInstImpl<bfloat16_t, bfloat16_t, bfloat16_t>(I);
    break;
  case ElemKind::Int32ITy:
    fwdElementCmpEQInstImpl<int32_t, int32_t, float>(I);
    break;
  case ElemKind::Int64ITy:
    fwdElementCmpEQInstImpl<int64_t, int64_t, float>(I);
    break;
  default:
    llvm_unreachable("Type is not supported");
  }
}

template <typename ElemTy, typename ElemOffsetTy, typename ElemScaleTy,
          typename CmpTy>
void BoundInterpreterFunction::fwdElementCmpNEQInstImpl(
    const ElementCmpNEQInst *I) {
  auto cmpHelper = [](CmpTy LHS, CmpTy RHS) -> bool { return !(LHS == RHS); };
  fwdElementCmpHelperImpl<ElemTy, ElemOffsetTy, ElemScaleTy, CmpTy,
                          ElementCmpNEQInst>(I, cmpHelper);
}

void BoundInterpreterFunction::fwdElementCmpNEQInst(
    const ElementCmpNEQInst *I) {
  auto *T = getTensor(I->getLHS());

  if (T->getType().isQuantizedType()) {
    switch (T->getElementType()) {
    case ElemKind::Int8QTy:
      fwdElementCmpNEQInstImpl<int8_t, int32_t, float, int32_t>(I);
      break;
    case ElemKind::Int16QTy:
      fwdElementCmpNEQInstImpl<int16_t, int32_t, float, int32_t>(I);
      break;
    default:
      llvm_unreachable("Type is not supported");
    }
    return;
  }

  switch (T->getElementType()) {
  case ElemKind::FloatTy:
    fwdElementCmpNEQInstImpl<float, float, float>(I);
    break;
  case ElemKind::Float16Ty:
    fwdElementCmpNEQInstImpl<float16_t, float16_t, float16_t>(I);
    break;
  case ElemKind::BFloat16Ty:
    fwdElementCmpNEQInstImpl<bfloat16_t, bfloat16_t, bfloat16_t>(I);
    break;
  case ElemKind::Int32ITy:
    fwdElementCmpNEQInstImpl<int32_t, int32_t, float>(I);
    break;
  case ElemKind::Int64ITy:
    fwdElementCmpNEQInstImpl<int64_t, int64_t, float>(I);
    break;
  default:
    llvm_unreachable("Type is not supported");
  }
}

template <typename ElemTy, typename ElemOffsetTy, typename ElemScaleTy,
          typename CmpTy>
void BoundInterpreterFunction::fwdElementCmpLTInstImpl(
    const ElementCmpLTInst *I) {
  auto cmpHelper = [](CmpTy LHS, CmpTy RHS) -> bool { return LHS < RHS; };
  fwdElementCmpHelperImpl<ElemTy, ElemOffsetTy, ElemScaleTy, CmpTy,
                          ElementCmpLTInst>(I, cmpHelper);
}

void BoundInterpreterFunction::fwdElementCmpLTInst(ElementCmpLTInst const *I) {
  auto *T = getTensor(I->getLHS());
  if (T->getType().isQuantizedType()) {
    switch (T->getElementType()) {
    case ElemKind::Int8QTy:
      fwdElementCmpLTInstImpl<int8_t, int32_t, float, int32_t>(I);
      break;
    case ElemKind::Int16QTy:
      fwdElementCmpLTInstImpl<int16_t, int32_t, float, int32_t>(I);
      break;
    default:
      llvm_unreachable("Type is not supported");
    }
    return;
  }

  switch (T->getElementType()) {
  case ElemKind::FloatTy:
    fwdElementCmpLTInstImpl<float, float, float>(I);
    break;
  case ElemKind::Float16Ty:
    fwdElementCmpLTInstImpl<float16_t, float16_t, float16_t>(I);
    break;
  case ElemKind::BFloat16Ty:
    fwdElementCmpLTInstImpl<bfloat16_t, bfloat16_t, bfloat16_t>(I);
    break;
  case ElemKind::Int32ITy:
    fwdElementCmpLTInstImpl<int32_t, int32_t, float>(I);
    break;
  case ElemKind::Int64ITy:
    fwdElementCmpLTInstImpl<int64_t, int64_t, float>(I);
    break;
  default:
    llvm_unreachable("Type is not supported");
  }
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdElementPowInstFloatImpl(
    const ElementPowInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto baseW = getWeightHandle<ElemTy>(I->getLHS());
  auto expW = getWeightHandle<ElemTy>(I->getRHS());
  auto outW = getWeightHandle<ElemTy>(I->getDest());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = ElemTy(pow(float(baseW.raw(i)), float(expW.raw(i))));
  }
}

void BoundInterpreterFunction::fwdElementPowInstI8Impl(
    const ElementPowInst *I) {
  assert(getTensor(I->getLHS())->getType().isQuantizedType() &&
         "Expect quantized type");
  auto baseTy = I->getLHS()->getType();
  auto expTy = I->getRHS()->getType();
  auto destTy = I->getDest()->getType();

  float baseScale = baseTy->getScale();
  int32_t baseOffset = baseTy->getOffset();
  TensorQuantizationParams baseTQP{baseScale, baseOffset};

  float expScale = expTy->getScale();
  int32_t expOffset = expTy->getOffset();
  TensorQuantizationParams expTQP{expScale, expOffset};

  float destScale = destTy->getScale();
  int32_t destOffset = destTy->getOffset();
  TensorQuantizationParams destTQP{destScale, destOffset};

  auto outW = getWeightHandle<int8_t>(I->getDest());
  auto baseW = getWeightHandle<int8_t>(I->getLHS());
  auto expW = getWeightHandle<int8_t>(I->getRHS());

  for (dim_t i = 0, e = outW.size(); i < e; i++) {
    float base = quantization::dequantize(baseW.raw(i), baseTQP);
    float exp = quantization::dequantize(expW.raw(i), expTQP);
    outW.raw(i) = quantization::quantize(std::pow(base, exp), destTQP);
  }
}

void BoundInterpreterFunction::fwdElementPowInst(
    const glow::ElementPowInst *I) {
  auto *T = getTensor(I->getLHS());
  if (T->getType().isQuantizedType()) {
    fwdElementPowInstI8Impl(I);
    return;
  }

  dispatchFloatingPointImpl(fwdElementPowInstFloatImpl,
                            I->getLHS()->getElementType(), I);
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdElementIsNaNInstFloatImpl(
    const ElementIsNaNInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto inW = getWeightHandle<ElemTy>(I->getSrc());
  auto outW = getWeightHandle<bool>(I->getDest());
  for (size_t i = 0, e = inW.size(); i < e; i++) {
    float val = inW.raw(i);
    outW.raw(i) = std::isnan(val);
  }
}

void BoundInterpreterFunction::fwdElementIsNaNInst(
    const glow::ElementIsNaNInst *I) {
  dispatchFloatingPointImpl(fwdElementIsNaNInstFloatImpl,
                            I->getSrc()->getElementType(), I);
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdElementLogInstFloatImpl(
    const ElementLogInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto inW = getWeightHandle<ElemTy>(I->getSrc());
  auto outW = getWeightHandle<ElemTy>(I->getDest());
  for (size_t i = 0, e = inW.size(); i < e; i++) {
    float val = inW.raw(i);
    outW.raw(i) = ElemTy(log(val));
  }
}

void BoundInterpreterFunction::fwdElementLogInst(const ElementLogInst *I) {
  dispatchFloatingPointImpl(fwdElementLogInstFloatImpl,
                            I->getSrc()->getElementType(), I);
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdElementExpInstFloatImpl(
    const ElementExpInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto inW = getWeightHandle<ElemTy>(I->getSrc());
  auto outW = getWeightHandle<ElemTy>(I->getDest());
  for (size_t i = 0, e = inW.size(); i < e; i++) {
    float val = inW.raw(i);
    outW.raw(i) = ElemTy(exp(val));
  }
}

void BoundInterpreterFunction::fwdElementExpInst(const ElementExpInst *I) {
  dispatchFloatingPointImpl(fwdElementExpInstFloatImpl,
                            I->getSrc()->getElementType(), I);
}

void BoundInterpreterFunction::fwdNonZeroInst(const NonZeroInst *I) {
  auto *T = getTensor(I->getDest());
  T->zero();
  auto outW = T->getHandle<int32_t>();
  auto condW = getWeightHandle<bool>(I->getCond());
  for (size_t condIdx = 0, outIdx = 0, n = condW.size(); condIdx < n;
       condIdx++) {
    if (condW.raw(condIdx)) {
      outW.raw(outIdx) = condIdx;
      outIdx++;
    }
  }
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdElementSelectInstFloatImpl(
    const glow::ElementSelectInst *I) {
  staticAssertFloatingPointType(ElemTy);
  auto outW = getWeightHandle<ElemTy>(I->getDest());
  auto condW = getWeightHandle<bool>(I->getCond());
  auto lhsW = getWeightHandle<ElemTy>(I->getLHS());
  auto rhsW = getWeightHandle<ElemTy>(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = condW.raw(i) ? lhsW.raw(i) : rhsW.raw(i);
  }
}

void BoundInterpreterFunction::fwdElementSelectInst(
    const glow::ElementSelectInst *I) {
  if (getTensor(I->getLHS())->getType().isQuantizedType()) {
    auto destTy = I->getDest()->getType();
    auto lhsTy = I->getLHS()->getType();
    auto rhsTy = I->getRHS()->getType();

    float destScale = destTy->getScale();
    float lhsScale = lhsTy->getScale();
    float rhsScale = rhsTy->getScale();

    int32_t destOffset = destTy->getOffset();
    int32_t lhsOffset = lhsTy->getOffset();
    int32_t rhsOffset = rhsTy->getOffset();

    auto outW = getWeightHandle<int8_t>(I->getDest());
    auto condW = getWeightHandle<bool>(I->getCond());
    auto lhsW = getWeightHandle<int8_t>(I->getLHS());
    auto rhsW = getWeightHandle<int8_t>(I->getRHS());
    for (size_t i = 0, e = outW.size(); i < e; i++) {
      float val = condW.raw(i) ? lhsScale * (lhsW.raw(i) - lhsOffset)
                               : rhsScale * (rhsW.raw(i) - rhsOffset);
      int32_t q = std::round(val / destScale + destOffset);
      outW.raw(i) = quantization::clip<int32_t, int8_t>(q);
    }
    return;
  }

  dispatchFloatingPointImpl(fwdElementSelectInstFloatImpl,
                            I->getLHS()->getElementType(), I);
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdModuloInstImpl(glow::ModuloInst const *I) {
  auto srcH = getTensor(I->getSrc())->getHandle<ElemTy>();
  auto destH = getTensor(I->getDest())->getHandle<ElemTy>();

  auto divisor = I->getDivisor();
  auto signFollowDivisor = I->getSignFollowDivisor();

  for (size_t i = 0, e = srcH.size(); i < e; i++) {
    auto res = srcH.raw(i) % divisor;
    if (signFollowDivisor && res < 0) {
      res += divisor;
    }
    destH.raw(i) = res;
  }
}

void BoundInterpreterFunction::fwdModuloInst(glow::ModuloInst const *I) {
  dispatchIndexTypeImpl(fwdModuloInstImpl, I->getSrc()->getElementType(), I);
}

///=============== Trigonometric Operators===============
template <typename ElemTy, typename InstKind>
void BoundInterpreterFunction::fwdUnaryTrigonometricImpl(
    const InstKind *I, std::function<float(float)> func) {
  Value *inpV = I->getSrc();
  Value *outV = I->getDest();
  auto inpTy = inpV->getType();
  auto outTy = outV->getType();
  auto inpH = getWeightHandle<ElemTy>(inpV);
  auto outH = getWeightHandle<ElemTy>(outV);

  if (inpTy->isQuantizedType()) {
    float inpScale = inpTy->getScale();
    int32_t inpOffset = inpTy->getOffset();
    float outScale = outTy->getScale();
    int32_t outOffset = outTy->getOffset();
    for (size_t i = 0, e = outH.size(); i < e; ++i) {
      float inpVal =
          quantization::dequantize<ElemTy>(inpH.raw(i), {inpScale, inpOffset});
      float outVal = func(inpVal);
      outH.raw(i) =
          quantization::quantize<ElemTy>(outVal, {outScale, outOffset});
    }
  } else {
    for (size_t i = 0, e = outH.size(); i < e; ++i) {
      float inpVal = static_cast<float>(inpH.raw(i));
      float outVal = func(inpVal);
      outH.raw(i) = static_cast<ElemTy>(outVal);
    }
  }
}

void BoundInterpreterFunction::fwdElementAcosInst(const ElementAcosInst *I) {
  auto func = [](float x) -> float { return std::acos(x); };
  dispatchImpl(fwdUnaryTrigonometricImpl, I->getSrc()->getElementType(), I,
               func);
}

void BoundInterpreterFunction::fwdElementAsinInst(const ElementAsinInst *I) {
  auto func = [](float x) -> float { return std::asin(x); };
  dispatchImpl(fwdUnaryTrigonometricImpl, I->getSrc()->getElementType(), I,
               func);
}

void BoundInterpreterFunction::fwdElementAtanInst(const ElementAtanInst *I) {
  auto func = [](float x) -> float { return std::atan(x); };
  dispatchImpl(fwdUnaryTrigonometricImpl, I->getSrc()->getElementType(), I,
               func);
}
//===----------------------------------------------------------------------===//
//                       Mat Mul
//===----------------------------------------------------------------------===//
template <typename ElemTy, typename AccumulatorTy>
void BoundInterpreterFunction::fwdMatMulInstQuantizedImpl(
    const glow::MatMulInst *I) {
  assert(getTensor(I->getLHS())->getType().isQuantizedType());
  auto lhs = getWeightHandle<ElemTy>(I->getLHS());
  auto rhs = getWeightHandle<ElemTy>(I->getRHS());

  auto dest = getWeightHandle<ElemTy>(I->getDest());

  auto destDim = dest.dims();
  auto lhsDim = lhs.dims();

  auto destTy = I->getDest()->getType();
  auto lhsTy = I->getLHS()->getType();
  auto rhsTy = I->getRHS()->getType();

  dest.clear(0);

  // For matrix multiplication, if the offset is equal to zero the scale
  // is defined as the formula (L.scale * R.scale / D.scale).
  // In here we assume that the offset for all buffers is zero.
  float scale = lhsTy->getScale() * rhsTy->getScale() / destTy->getScale();
  int32_t lhsOffset = lhsTy->getOffset();
  int32_t rhsOffset = rhsTy->getOffset();
  int32_t destOffset = destTy->getOffset();

  // For each (x,y) in the destination matrix:
  for (dim_t x = 0; x < destDim[0]; x++) {
    for (dim_t y = 0; y < destDim[1]; y++) {

      // Perform DOT on the row an column.
      AccumulatorTy sum = 0;
      for (dim_t i = 0; i < lhsDim[1]; i++) {
        AccumulatorTy L = lhs.at({x, i});
        AccumulatorTy R = rhs.at({i, y});
        // We represent the element multiplication with offset as
        // (value - offset).
        sum += (L - lhsOffset) * (R - rhsOffset);
      }

      dest.at({x, y}) = quantization::clip<AccumulatorTy, ElemTy>(
          std::round(scale * sum + destOffset));
    }
  }
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdMatMulInstFloatImpl(const MatMulInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto lhs = getWeightHandle<ElemTy>(I->getLHS());
  auto rhs = getWeightHandle<ElemTy>(I->getRHS());
  auto dest = getWeightHandle<ElemTy>(I->getDest());

  auto destDim = dest.dims();
  auto lhsDim = lhs.dims();

  dest.clear(0);

  // For each (x,y) in the destination matrix:
  for (dim_t x = 0; x < destDim[0]; x++) {
    for (dim_t y = 0; y < destDim[1]; y++) {

      // Perform DOT on the row an column.
      float sum = 0;
      for (dim_t i = 0; i < lhsDim[1]; i++) {
        sum += float(lhs.at({x, i}) * rhs.at({i, y}));
      }
      dest.at({x, y}) = ElemTy(sum);
    }
  }
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdBatchMatMulInstFloatImpl(
    const BatchMatMulInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto lhs = getWeightHandle<ElemTy>(I->getLHS());
  auto rhs = getWeightHandle<ElemTy>(I->getRHS());
  auto dest = getWeightHandle<ElemTy>(I->getDest());

  auto destDim = dest.dims();
  auto lhsDim = lhs.dims();

  dest.clear(0);

  for (dim_t batch = 0; batch < destDim[0]; batch++) {
    // For each (x,y) in the destination matrix:
    for (dim_t x = 0; x < destDim[1]; x++) {
      for (dim_t y = 0; y < destDim[2]; y++) {
        // Perform DOT on the row an column.
        float sum = 0;
        for (dim_t i = 0; i < lhsDim[2]; i++) {
          sum += float(lhs.at({batch, x, i})) * float(rhs.at({batch, i, y}));
        }
        dest.at({batch, x, y}) = ElemTy(sum);
      }
    }
  }
}

void BoundInterpreterFunction::fwdMatMulInst(const glow::MatMulInst *I) {
  if (getTensor(I->getLHS())->getType().isQuantizedType()) {
    dispatchQuantizedWithAccumulationImpl(fwdMatMulInstQuantizedImpl,
                                          I->getLHS()->getElementType(), I);
    return;
  }

  dispatchFloatingPointImpl(fwdMatMulInstFloatImpl,
                            I->getLHS()->getElementType(), I);
}

void BoundInterpreterFunction::fwdBatchMatMulInst(
    const glow::BatchMatMulInst *I) {
  if (getTensor(I->getLHS())->getType().isQuantizedType()) {
    DCHECK(!"Quantized implementation for BatchMatmul not supported yet.");
    return;
  }

  dispatchFloatingPointImpl(fwdBatchMatMulInstFloatImpl,
                            I->getLHS()->getElementType(), I);
}

void BoundInterpreterFunction::fwdReluGradInst(const glow::ReluGradInst *I) {
  DCHECK(!"Found ReluGradInst but ReluGrad is lowered on Interpreter");
}

//===----------------------------------------------------------------------===//
//                                 FC
//===----------------------------------------------------------------------===//
template <typename ElemTy, typename AccumulatorTy, typename BiasElemTy>
void BoundInterpreterFunction::fwdFullyConnectedInstQuantizedImpl(
    const glow::FullyConnectedInst *I) {
  assert(getTensor(I->getSrc())->getType().isQuantizedType());

  auto inW = getWeightHandle<ElemTy>(I->getSrc());
  auto weightsW = getWeightHandle<ElemTy>(I->getWeights());
  auto biasW = getWeightHandle<BiasElemTy>(I->getBias());
  auto outW = getWeightHandle<ElemTy>(I->getDest());

  auto inTy = inW.getType();
  auto weightsTy = weightsW.getType();
  auto biasTy = biasW.getType();
  auto outTy = outW.getType();

  int32_t inOffset = inTy.getOffset();
  int32_t weightsOffset = weightsTy.getOffset();
  int32_t biasOffset = biasTy.getOffset();
  int32_t outOffset = outTy.getOffset();

  float outScale = outTy.getScale();
  float weightsScale = weightsTy.getScale();
  float biasScale = biasTy.getScale();
  float inScale = inTy.getScale();

  ShapeHW idim(inW.dims());
  ShapeHW odim(outW.dims());

  // Calculate the scale of the values that come out of the matrix
  // multiplication part of the calculation.
  float matMulScale = weightsScale * inScale;

  outW.clear(0);

  for (dim_t i = 0; i < idim.height; i++) {
    for (dim_t j = 0; j < odim.width; j++) {
      AccumulatorTy sum = 0;
      for (dim_t k = 0; k < idim.width; k++) {
        AccumulatorTy W = weightsW.at({k, j});
        AccumulatorTy A = inW.at({i, k});
        sum += (W - weightsOffset) * (A - inOffset);
      }

      // Scale the bias to match the scale of the matrix multiplication.
      AccumulatorTy B = std::round(float(biasW.at({j}) - biasOffset) *
                                   (biasScale / matMulScale));

      // Add the bias.
      sum += B;

      // Scale the result back to the expected destination scale.
      outW.at({i, j}) = quantization::clip<AccumulatorTy, ElemTy>(
          std::round(float(sum) * (matMulScale / outScale)) + outOffset);
    }
  }
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdFullyConnectedInstFloatImpl(
    const FullyConnectedInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto inW = getWeightHandle<ElemTy>(I->getSrc());
  auto weightsW = getWeightHandle<ElemTy>(I->getWeights());
  auto biasW = getWeightHandle<ElemTy>(I->getBias());
  auto outW = getWeightHandle<ElemTy>(I->getDest());

  ShapeHW idim(inW.dims());
  ShapeHW odim(outW.dims());

  outW.clear(0);

  for (dim_t i = 0; i < idim.height; i++) {
    for (dim_t j = 0; j < odim.width; j++) {
      float sum = 0;
      for (dim_t k = 0; k < idim.width; k++) {
        sum += float(inW.at({i, k})) * float(weightsW.at({k, j}));
      }

      outW.at({i, j}) = sum + float(biasW.at({j}));
    }
  }
}

void BoundInterpreterFunction::fwdFullyConnectedInst(
    const glow::FullyConnectedInst *I) {

  if (getTensor(I->getSrc())->getType().isQuantizedType()) {
    dispatchQuantizedWithAccumulationAndBiasImpl(
        fwdFullyConnectedInstQuantizedImpl, I->getSrc()->getElementType(),
        I->getBias()->getElementType(), I);
    return;
  } else {
    dispatchFloatingPointImpl(fwdFullyConnectedInstFloatImpl,
                              I->getSrc()->getElementType(), I);
  }
}

//===----------------------------------------------------------------------===//
//                       Dynamic quantized FC
//===----------------------------------------------------------------------===//

template <typename ElemTy, typename OutputTy, typename AccumulatorTy>
void BoundInterpreterFunction::fwdDynRowwiseQuantizedFullyConnectedInstImpl(
    Handle<ElemTy> inW, Handle<OutputTy> &outW, dim_t baseRow,
    Handle<ElemTy> weightsW, Handle<float> biasW, Handle<float> scalesW,
    Handle<int32_t> offsetsW) {
  ShapeHW idim(inW.dims());
  ShapeHW odim(outW.dims());
  auto inTy = inW.getType();
  int32_t inOffset = inTy.getOffset();
  float inScale = inTy.getScale();

  for (dim_t i = 0; i < idim.height; i++) {
    for (dim_t j = 0; j < odim.width; j++) {
      float matMulScale = inScale * static_cast<float>(scalesW.raw(j));
      AccumulatorTy sum = 0;
      for (dim_t k = 0; k < idim.width; k++) {
        AccumulatorTy W = weightsW.at({k, j});
        AccumulatorTy A = inW.at({i, k});
        sum += (W - offsetsW.raw(j)) * (A - inOffset);
      }

      float B = float(biasW.at({j}));

      // Scale the result back to the expected destination scale and add the
      // bias.
      outW.at({baseRow + i, j}) = float(sum) * matMulScale + B;
    }
  }
}

void BoundInterpreterFunction::fwdDynRowwiseQuantizedFullyConnectedInstPreimpl(
    Tensor *inputTensor, Tensor *weightsTensor, Tensor *biasTensor,
    Tensor *resultTensor, Tensor *wScaleTensor, Tensor *wOffsetTensor,
    bool isSymmetric, bool isPerBatchElement) {

  /* Check the options */
  assert(isSymmetric && "Only symmetric quantization is supported.");
  assert(isPerBatchElement && "Only quantized per batch element is supported.");

  auto weightsW = weightsTensor->getHandle<int8_t>();

  /* Dynamic Quantization */
  auto resultHandle = resultTensor->getHandle<float16_t>();
  auto offsetsW = wOffsetTensor->getHandle<int32_t>();
  dim_t N = inputTensor->dims()[0];
  dim_t L = inputTensor->dims()[1];
  if (isPerBatchElement && isSymmetric) {
    // We slice N * L input tensor to N tensors with 1 * L shape,
    // For each batch we calculate qparams, quantize, FC and dequantize
    // independently, and finally splice them together.
    for (dim_t i = 0; i < N; i++) {
      Tensor slicedInputTensor = inputTensor->getOwnedSlice({1, L}, {i, 0});
      auto slicedInputHandle = slicedInputTensor.getHandle<float16_t>();
      auto minMax = slicedInputHandle.minMaxArg();
      auto qMin = slicedInputHandle.raw(minMax.first);
      auto qMax = slicedInputHandle.raw(minMax.second);

      // TODO Currently we only support symmetric quantization.
      // We should support both symmetric/asymmetric based of isSymmetric.
      auto qParams = quantization::chooseQuantizationParams(
          {qMin, qMax}, quantization::Schema::Symmetric, ElemKind::Int8QTy);
      Tensor qInputTensor = quantization::quantizeTensor(
          slicedInputTensor, {qParams.scale, qParams.offset},
          ElemKind::Int8QTy);
      auto inW = qInputTensor.getHandle<int8_t>();

      auto biasW = biasTensor->getHandle<float>();
      auto scalesW = wScaleTensor->getHandle<float>();
      fwdDynRowwiseQuantizedFullyConnectedInstImpl<int8_t, float16_t, int32_t>(
          inW, resultHandle, i, weightsW, biasW, scalesW, offsetsW);
    }
  }
}

void BoundInterpreterFunction::fwdDynamicRowwiseQuantizedFullyConnectedInst(
    const glow::DynamicRowwiseQuantizedFullyConnectedInst *I) {
  auto *inputTensor = getTensor(I->getSrc());
  auto *weightsTensor = getTensor(I->getWeights());
  auto *biasTensor = getTensor(I->getBias());
  auto *resultTensor = getTensor(I->getDest());
  auto *scaleTensor = getTensor(I->getScales());
  auto *offsetTensor = getTensor(I->getOffsets());
  auto isSymmetric = I->getIsSymmetric();
  auto isPerBatchElement = I->getIsPerBatchElement();
  fwdDynRowwiseQuantizedFullyConnectedInstPreimpl(
      inputTensor, weightsTensor, biasTensor, resultTensor, scaleTensor,
      offsetTensor, isSymmetric, isPerBatchElement);
}

void BoundInterpreterFunction::fwdDynamicQuantizedFullyConnectedInst(
    const glow::DynamicQuantizedFullyConnectedInst *I) {

  auto *inputTensor = getTensor(I->getSrc());
  auto *weightsTensor = getTensor(I->getWeights());
  auto *biasTensor = getTensor(I->getBias());
  auto *resultTensor = getTensor(I->getDest());
  auto isSymmetric = I->getIsSymmetric();
  auto isPerBatchElement = I->getIsPerBatchElement();
  dim_t M = resultTensor->dims()[1];

  // Calc channelwise QParam
  Tensor scaleTensor = Tensor(ElemKind::FloatTy, {M});
  Tensor offsetTensor = Tensor(ElemKind::Int32ITy, {M});
  auto scalesW = scaleTensor.getHandle<float>();
  auto offsetsW = offsetTensor.getHandle<int32_t>();
  auto weightsW = weightsTensor->getHandle<int8_t>();
  for (int i = 0; i < M; i++) {
    scalesW.raw(i) = weightsW.getType().getScale();
    offsetsW.raw(i) = weightsW.getType().getOffset();
  }
  fwdDynRowwiseQuantizedFullyConnectedInstPreimpl(
      inputTensor, weightsTensor, biasTensor, resultTensor, &scaleTensor,
      &offsetTensor, isSymmetric, isPerBatchElement);
}

//===----------------------------------------------------------------------===//
//                       Row-wise quantized FC
//===----------------------------------------------------------------------===//
template <typename ElemTy, typename AccumulatorTy, typename BiasElemTy>
void BoundInterpreterFunction::fwdRowwiseQuantizedFullyConnectedInstImpl(
    Value *inV, Value *outV, Value *weightsV, Value *biasV, Value *scalesV,
    Value *offsetsV) {
  auto inW = getWeightHandle<ElemTy>(inV);
  auto outW = getWeightHandle<ElemTy>(outV);
  auto weightsW = getWeightHandle<ElemTy>(weightsV);
  auto biasW = getWeightHandle<BiasElemTy>(biasV);
  auto scalesW = getWeightHandle<float>(scalesV);
  auto offsetsW = getWeightHandle<int32_t>(offsetsV);
  ShapeHW idim(inW.dims());
  ShapeHW odim(outW.dims());
  auto inTy = inW.getType();
  auto biasTy = biasW.getType();
  auto outTy = outW.getType();
  int32_t outOffset = outTy.getOffset();
  int32_t inOffset = inTy.getOffset();
  int32_t biasOffset = biasTy.getOffset();
  float outScale = outTy.getScale();
  float inScale = inTy.getScale();
  float biasScale = biasTy.getScale();

  for (dim_t i = 0; i < idim.height; i++) {
    for (dim_t j = 0; j < odim.width; j++) {
      float matMulScale = scalesW.raw(j) * inScale;
      AccumulatorTy sum = 0;
      for (dim_t k = 0; k < idim.width; k++) {
        AccumulatorTy W = weightsW.at({j, k});
        AccumulatorTy A = inW.at({i, k});
        sum += (W - offsetsW.raw(j)) * (A - inOffset);
      }

      // Scale the bias to match the scale of the matrix multiplication.
      AccumulatorTy B = std::round(float(biasW.at({j}) - biasOffset) *
                                   (biasScale / matMulScale));

      // Add the bias.
      sum += B;

      // Scale the result back to the expected destination scale.
      outW.at({i, j}) = quantization::clip<AccumulatorTy, ElemTy>(
          std::round(float(sum) * (matMulScale / outScale) + outOffset));
    }
  }
}

void BoundInterpreterFunction::fwdRowwiseQuantizedFullyConnectedInst(
    const RowwiseQuantizedFullyConnectedInst *I) {
  dispatchQuantizedWithAccumulationAndBiasImpl(
      fwdRowwiseQuantizedFullyConnectedInstImpl, I->getSrc()->getElementType(),
      I->getBias()->getElementType(), I->getSrc(), I->getDest(),
      I->getWeights(), I->getBias(), I->getScales(), I->getOffsets());
}

//===----------------------------------------------------------------------===//
//                       Batched operations
//===----------------------------------------------------------------------===//
template <typename ElemTy, typename AccumulatorTy, typename SliceElemTy>
static void fwdBatchedAdd(Tensor *batch, Tensor *slice, Tensor *dest) {
  auto batchH = batch->getHandle<ElemTy>();
  auto sliceH = slice->getHandle<SliceElemTy>();
  auto destH = dest->getHandle<ElemTy>();

  auto batchTy = batch->getType();
  auto sliceTy = slice->getType();
  auto destTy = dest->getType();

  float sliceScale = sliceTy.getScale();
  float batchScale = batchTy.getScale();
  float destScale = destTy.getScale();

  int32_t sliceOffset = sliceTy.getOffset();
  int32_t batchOffset = batchTy.getOffset();
  int32_t destOffset = destTy.getOffset();

  auto bdim = flattenCdr(batchH.dims());
  assert(sliceH.size() == bdim.second && "Invalid slice size");
  assert(batchH.dims().drop_front() == sliceH.dims() && "Invalid batch size");

  // For each layer in the batch:
  for (dim_t n = 0; n < bdim.first; n++) {
    size_t base = batchH.getElementPtr({n});

    // For each element in the slice.
    for (dim_t i = 0; i < bdim.second; i++) {
      AccumulatorTy batchVal = batchH.raw(base + i);
      AccumulatorTy sliceVal = sliceH.raw(i);
      // We increase the size of the integer up to 16 bits for more accurate
      // arithmetic.
      const float largeScale = float(1) / (1 << 15);
      // Scale both sides from 8-bit to 16-bits.
      AccumulatorTy B =
          std::round(float(batchVal - batchOffset) * (batchScale / largeScale));
      AccumulatorTy S =
          std::round(float(sliceVal - sliceOffset) * (sliceScale / largeScale));
      AccumulatorTy R = B + S;
      destH.raw(base + i) = quantization::clip<AccumulatorTy, ElemTy>(
          std::round(float(R) * (largeScale / destScale) + destOffset));
    }
  }
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdBatchedAddInstFloatImpl(
    const glow::BatchedAddInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto batch = getWeightHandle<ElemTy>(I->getBatch());
  auto slice = getWeightHandle<ElemTy>(I->getSlice());
  auto dest = getWeightHandle<ElemTy>(I->getDest());

  auto bdim = flattenCdr(batch.dims());
  assert(slice.size() == bdim.second && "Invalid slice size");
  assert(batch.dims().drop_front() == slice.dims() && "Invalid batch size");

  // For each layer in the batch:
  for (dim_t n = 0; n < bdim.first; n++) {
    size_t base = batch.getElementPtr({n});

    // For each element in the slice.
    for (dim_t i = 0; i < bdim.second; i++) {
      dest.raw(base + i) = batch.raw(base + i) + slice.raw(i);
    }
  }
}

void BoundInterpreterFunction::fwdBatchedAddInst(
    const glow::BatchedAddInst *I) {
  if (getTensor(I->getBatch())->getType().isQuantizedType()) {
    dispatchQuantizedWithAccumulationAndBiasImpl(
        fwdBatchedAdd, I->getBatch()->getElementType(),
        I->getSlice()->getElementType(), getTensor(I->getBatch()),
        getTensor(I->getSlice()), getTensor(I->getDest()));
    return;
  }
  dispatchFloatingPointImpl(fwdBatchedAddInstFloatImpl,
                            I->getBatch()->getElementType(), I);
}

// Macro to define the ReduceAdd/Prod kernel implementation.
#define DEFINE_REDUCEADDPROD_INST_IMPL(func, init, op, inst)                   \
  template <typename ElemTy>                                                   \
  void BoundInterpreterFunction::fwdBatched##func##inst(                       \
      Value *batch, Value *dest, unsigned_t axis,                              \
      const ShapeVector &eBatchDims, const ShapeVector &eDestDims) {           \
    /*Get unowned handles of the batch and dest with these new expanded        \
     * dims.*/                                                                 \
    auto eBatch = getTensor(batch)->getUnowned(eBatchDims);                    \
    auto eDest = getTensor(dest)->getUnowned(eDestDims);                       \
    auto eBatchH = eBatch.getHandle<ElemTy>();                                 \
    auto eDestH = eDest.getHandle<ElemTy>();                                   \
    eDestH.clear(init);                                                        \
                                                                               \
    /* We can use this loop for all shapes. Use the same indices for both the  \
     * batch and dest, except for setting the axis index in the dest to 0.*/   \
    for (dim_t x = 0; x < eBatchDims[0]; x++) {                                \
      for (dim_t y = 0; y < eBatchDims[1]; y++) {                              \
        for (dim_t z = 0; z < eBatchDims[2]; z++) {                            \
          for (dim_t w = 0; w < eBatchDims[3]; w++) {                          \
            for (dim_t q = 0; q < eBatchDims[4]; q++) {                        \
              for (dim_t r = 0; r < eBatchDims[5]; r++) {                      \
                dim_t destIndices[] = {x, y, z, w, q, r};                      \
                destIndices[axis] = 0;                                         \
                eDestH.at(destIndices) =                                       \
                    eDestH.at(destIndices) op eBatchH.at({x, y, z, w, q, r});  \
              }                                                                \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

/// Define fwdBatchedReduceAddInstImpl
DEFINE_REDUCEADDPROD_INST_IMPL(ReduceAdd, 0, +, InstImpl)

/// Define fwdBatchedReduceAddInstImpl
DEFINE_REDUCEADDPROD_INST_IMPL(ReduceProd, 1, *, InstFloatImpl)

#undef DEFINE_REDUCEADDPROD_INST_IMPL

void BoundInterpreterFunction::fwdBatchedReduceAddInst(
    const glow::BatchedReduceAddInst *I) {
  static_assert(max_tensor_dimensions == 6,
                "Loops below assume max_tensor_dimensions = 6.");

  auto *batch = I->getBatch();
  auto *dest = I->getDest();
  const auto axis = I->getAxis();

  // Initialize both expanded batch and dest dims to the expanded batch
  // dims. This allows us below to iterate over the tensor regardless of its
  // shape using max_tensor_dimensions loops below.
  ShapeVector eBatchDims = expandDimsToMax(batch->dims());
  ShapeVector eDestDims = eBatchDims;

  // Set the destination axis dimension (the one we are reducing) to 1.
  eDestDims[axis] = 1;

  if (getTensor(batch)->getType().isQuantizedType()) {
    auto destTy = dest->getType();
    auto batchTy = batch->getType();

    float destScale = destTy->getScale();
    float batchScale = batchTy->getScale();

    int32_t destOffset = destTy->getOffset();
    int32_t batchOffset = batchTy->getOffset();

    // Get unowned handles of the batch and dest with these new expanded dims.
    auto eBatch = getTensor(batch)->getUnowned(eBatchDims);
    auto eDest = getTensor(dest)->getUnowned(eDestDims);
    auto eBatchH = eBatch.getHandle<int8_t>();
    auto eDestH = eDest.getHandle<int8_t>();
    eDestH.clear();

    // For quantization, we must accumulate in the inner-most loop into a
    // local float and then clip the result back into the dest tensor. Here
    // are the max_tensor_dimensions cases for this, to ensure the axis is
    // used as the inner-most loop.
    switch (axis) {
#define LOOP_AXIS_CASE(_D0, _D1, _D2, _D3, _D4, _D5_AXIS)                      \
  case _D5_AXIS:                                                               \
    for (dim_t i##_D0 = 0; i##_D0 < eBatchDims[_D0]; i##_D0++)                 \
      for (dim_t i##_D1 = 0; i##_D1 < eBatchDims[_D1]; i##_D1++)               \
        for (dim_t i##_D2 = 0; i##_D2 < eBatchDims[_D2]; i##_D2++)             \
          for (dim_t i##_D3 = 0; i##_D3 < eBatchDims[_D3]; i##_D3++)           \
            for (dim_t i##_D4 = 0; i##_D4 < eBatchDims[_D4]; i##_D4++) {       \
              float sum = 0.0;                                                 \
              for (dim_t i##_D5_AXIS = 0; i##_D5_AXIS < eBatchDims[_D5_AXIS];  \
                   i##_D5_AXIS++) {                                            \
                sum += eBatchH.at({i0, i1, i2, i3, i4, i5}) - batchOffset;     \
              }                                                                \
              dim_t i##_D5_AXIS = 0;                                           \
              int32_t res =                                                    \
                  std::round(sum * batchScale / destScale) + destOffset;       \
              eDestH.at({i0, i1, i2, i3, i4, i5}) =                            \
                  quantization::clip<int32_t, int8_t>(res);                    \
            }                                                                  \
    return;

      // Each loop order, with the inner-most dimension/index equal to the
      // axis.
      LOOP_AXIS_CASE(1, 2, 3, 4, 5, 0);
      LOOP_AXIS_CASE(0, 2, 3, 4, 5, 1);
      LOOP_AXIS_CASE(0, 1, 3, 4, 5, 2);
      LOOP_AXIS_CASE(0, 1, 2, 4, 5, 3);
      LOOP_AXIS_CASE(0, 1, 2, 3, 5, 4);
      LOOP_AXIS_CASE(0, 1, 2, 3, 4, 5);
#undef LOOP_AXIS_CASE
    default:
      llvm_unreachable("Axis should be less than max_tensor_dimensions.");
    }
  }
  dispatchFloatingPointAndInt32Impl(fwdBatchedReduceAddInstImpl,
                                    batch->getElementType(), batch, dest, axis,
                                    eBatchDims, eDestDims);
}

void BoundInterpreterFunction::fwdBatchedReduceProdInst(
    const glow::BatchedReduceProdInst *I) {
  static_assert(max_tensor_dimensions == 6,
                "Loops below assume max_tensor_dimensions = 6.");

  auto *batch = I->getBatch();
  auto *dest = I->getDest();
  const auto axis = I->getAxis();

  // Initialize both expanded batch and dest dims to the expanded batch
  // dims. This allows us below to iterate over the tensor regardless of its
  // shape using max_tensor_dimensions loops below.
  ShapeVector eBatchDims = expandDimsToMax(batch->dims());
  ShapeVector eDestDims = eBatchDims;

  // Set the destination axis dimension (the one we are reducing) to 1.
  eDestDims[axis] = 1;

  assert(!batch->getType()->isQuantizedType() &&
         "Quantized implementation for ReduceProd not supported yet.");

  dispatchArithmeticImpl(fwdBatchedReduceProdInstFloatImpl,
                         batch->getElementType(), batch, dest, axis, eBatchDims,
                         eDestDims);
}

/// Macro to define ReduceMin/Max kernel implementation.
#define DEFINE_REDUCEMINMAX_INST_IMPL(func, compare)                           \
  template <typename ElemTy>                                                   \
  void BoundInterpreterFunction::fwdBatched##func##InstImpl(                   \
      Value *batch, Value *dest, const ShapeVector &eBatchDims,                \
      const ShapeVector &eDestDims, ElemTy init) {                             \
    static_assert(max_tensor_dimensions == 6,                                  \
                  "Loops below assume max_tensor_dimensions = 6.");            \
    /* Get unowned handles of the batch and dest with these new expanded       \
     * dims.*/                                                                 \
    auto eBatch = getTensor(batch)->getUnowned(eBatchDims);                    \
    auto eDest = getTensor(dest)->getUnowned(eDestDims);                       \
    auto eBatchH = eBatch.getHandle<ElemTy>();                                 \
    auto eDestH = eDest.getHandle<ElemTy>();                                   \
    eDestH.clear(init);                                                        \
                                                                               \
    unsigned int axes[max_tensor_dimensions];                                  \
    for (dim_t i = 0; i < max_tensor_dimensions; i++) {                        \
      axes[i] = (eDestDims[i] > 1);                                            \
    }                                                                          \
                                                                               \
    /* We can use this loop for all shapes. Use the same indices for both the  \
     * batch and dest, except for setting the axis index in the dest to 0.*/   \
    for (dim_t x = 0, dx = 0; x < eBatchDims[0]; x++, dx += axes[0]) {         \
      for (dim_t y = 0, dy = 0; y < eBatchDims[1]; y++, dy += axes[1]) {       \
        for (dim_t z = 0, dz = 0; z < eBatchDims[2]; z++, dz += axes[2]) {     \
          for (dim_t w = 0, dw = 0; w < eBatchDims[3]; w++, dw += axes[3]) {   \
            for (dim_t q = 0, dq = 0; q < eBatchDims[4]; q++, dq += axes[4]) { \
              for (dim_t r = 0, dr = 0; r < eBatchDims[5];                     \
                   r++, dr += axes[5]) {                                       \
                dim_t destIndices[] = {dx, dy, dz, dw, dq, dr};                \
                dim_t srcIndices[] = {x, y, z, w, q, r};                       \
                eDestH.at(destIndices) =                                       \
                    compare(eDestH.at(destIndices), eBatchH.at(srcIndices));   \
              }                                                                \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

/// Define fwdBatchedReduceMaxInstImpl.
DEFINE_REDUCEMINMAX_INST_IMPL(ReduceMax, std::max)

/// Define fwdBatchedReduceMinInstImpl.
DEFINE_REDUCEMINMAX_INST_IMPL(ReduceMin, std::min)

#undef DEFINE_REDUCEMINMAX_INST_IMPL

/// Macro to define ReduceMin/Max instruction.
#define DEFINE_REDUCEMINMAX_INST(func, init_func)                              \
  void BoundInterpreterFunction::fwdBatched##func##Inst(                       \
      const glow::Batched##func##Inst *I) {                                    \
                                                                               \
    auto *batch = I->getBatch();                                               \
    auto *dest = I->getDest();                                                 \
    const auto axes = I->getAxes();                                            \
                                                                               \
    /* Initialize both expanded batch and dest dims to the expanded batch      \
     dims. This allows us below to iterate over the tensor regardless of its   \
     shape using max_tensor_dimensions loops below.*/                          \
    ShapeVector eBatchDims = expandDimsToMax(batch->dims());                   \
    ShapeVector eDestDims = eBatchDims;                                        \
    /* Set the destination axes dimensions (the one we are reducing) to 1.*/   \
    for (dim_t i = 0; i < axes.size(); i++) {                                  \
      eDestDims[axes[i]] = 1;                                                  \
    }                                                                          \
                                                                               \
    if (batch->getElementType() == ElemKind::Int8QTy) {                        \
      dispatchQuantizedImpl(                                                   \
          fwdBatched##func##InstImpl, batch->getElementType(), batch, dest,    \
          eBatchDims, eDestDims, std::numeric_limits<int8_t>::init_func());    \
    } else {                                                                   \
      dispatchArithmeticImpl(                                                  \
          fwdBatched##func##InstImpl, batch->getElementType(), batch, dest,    \
          eBatchDims, eDestDims, std::numeric_limits<int32_t>::init_func());   \
    }                                                                          \
  }

// Define fwdBatchedMinInst
DEFINE_REDUCEMINMAX_INST(ReduceMin, max)

// Define fwdBatchedMaxInst
DEFINE_REDUCEMINMAX_INST(ReduceMax, min)

#undef DEFINE_REDUCEMINMAX_INST

template <typename ElemTy>
void BoundInterpreterFunction::fwdCumSumInstImpl(Value *input, Value *dest,
                                                 int64_t dim, bool exclusive,
                                                 bool reverse) {
  auto eInputH = getTensor(input)->getHandle<ElemTy>();
  auto eDestH = getTensor(dest)->getHandle<ElemTy>();
  eDestH.clear();

  // deal with dim < 0
  if (dim < 0) {
    dim += eInputH.dims().size();
  }
  assert(dim < eInputH.dims().size() &&
         "Dim must be less than the number of dimensions of input tensor");

  std::vector<dim_t> accumDims(eInputH.dims());
  accumDims[dim] = 1;

  Tensor accum(eInputH.getElementType(), accumDims);
  auto accumH = accum.getHandle<ElemTy>();
  accumH.clear();

  sdim_t s = 0;
  sdim_t n = eInputH.dims()[dim];
  sdim_t dir = 1;

  if (reverse) {
    s = n - 1;
    n = -1;
    dir = -1;
  }

  for (sdim_t i = s; i != n; i += dir) {
    std::vector<dim_t> offset(eInputH.dims().size());
    offset[dim] = i;

    Tensor temp(eInputH.getElementType(), accumDims);
    auto tempH = temp.getHandle<ElemTy>();
    eInputH.extractTensors(tempH, offset);

    if (!exclusive) {
      for (auto accumIt = accumH.begin(), tempIt = tempH.begin();
           accumIt != accumH.end(); ++accumIt, ++tempIt) {
        *accumIt += *tempIt;
      }
    }

    eDestH.insertTensors(accumH, offset, 1, dim);

    if (exclusive) {
      for (auto accumIt = accumH.begin(), tempIt = tempH.begin();
           accumIt != accumH.end(); ++accumIt, ++tempIt) {
        *accumIt += *tempIt;
      }
    }
  }
}

void BoundInterpreterFunction::fwdCumSumInst(glow::CumSumInst const *I) {
  dispatchArithmeticImpl(fwdCumSumInstImpl, I->getInput()->getElementType(),
                         I->getInput(), I->getDest(), I->getDim(),
                         I->getExclusive(), I->getReverse());
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdLengthsSumInstFloatImpl(
    const LengthsSumInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto out = getTensor(I->getDest());
  auto data = getTensor(I->getData());
  auto lengths = getTensor(I->getLengths());

  out->zero();

  auto LH = lengths->getHandle<int32_t>();

  size_t segments = lengths->dims()[0];
  size_t sliceSize = data->size() / data->dims()[0];

  auto DH = data->getHandle<ElemTy>();
  auto OH = out->getHandle<ElemTy>();

  size_t offsetIn = 0;
  size_t offsetOut = 0;
  for (dim_t i = 0; i < segments; i++) {
    for (int32_t j = 0, e = LH.raw(i); j < e; j++) {
      for (dim_t k = 0; k < sliceSize; k++) {
        OH.raw(offsetOut + k) += DH.raw(offsetIn + k);
      }
      offsetIn += sliceSize;
    }
    offsetOut += sliceSize;
  }

  assert(offsetIn == data->size() && "All values in Data should be consumed");
  assert(offsetOut == out->size() && "All values in Dest should be written to");
}

void BoundInterpreterFunction::fwdLengthsSumInst(const LengthsSumInst *I) {
  dispatchFloatingPointImpl(fwdLengthsSumInstFloatImpl,
                            I->getData()->getElementType(), I)
}

template <typename TI>
void BoundInterpreterFunction::fwdSparseLengthsSumInstI8Impl(
    const SparseLengthsSumInst *I) {

  auto out = getTensor(I->getDest());
  auto data = getTensor(I->getData());
  auto indices = getTensor(I->getIndices());
  auto lengths = getTensor(I->getLengths());

  out->zero();

  auto IH = indices->getHandle<TI>();
  auto LH = lengths->getHandle<int32_t>();

  size_t segments = lengths->dims()[0];
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    totalLength += LH.raw(i);
  }
  assert(totalLength <= indices->dims()[0] &&
         "sum(Lengths) must be equal to len(Indices)");

  size_t lineSize = data->size() / data->dims()[0];

  auto DH = data->getHandle<int8_t>();
  auto OH = out->getHandle<int8_t>();

  auto TQP = [](Tensor *T) {
    return TensorQuantizationParams{T->getType().getScale(),
                                    T->getType().getOffset()};
  };

  size_t curIdx = 0;
  for (size_t i = 0; i < segments; i++) {
    std::vector<float> accum(lineSize, 0.0f);
    for (int32_t j = 0; j < LH.raw(i); j++) {
      size_t offsetIn = IH.raw(curIdx) * lineSize;
      for (size_t k = 0; k < lineSize; k++) {
        accum[k] += quantization::dequantize(DH.raw(offsetIn++), TQP(data));
      }
      curIdx++;
    }
    size_t offsetOut = i * lineSize;
    for (size_t k = 0; k < lineSize; k++) {
      OH.raw(offsetOut++) = quantization::quantize(accum[k], TQP(out));
    }
  }
}

template <typename ElemTy, typename TI>
void BoundInterpreterFunction::fwdSparseLengthsSumInstFloatImpl(
    const SparseLengthsSumInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto out = getTensor(I->getDest());
  auto data = getTensor(I->getData());
  auto indices = getTensor(I->getIndices());
  auto lengths = getTensor(I->getLengths());

  out->zero();

  auto IH = indices->getHandle<TI>();
  auto LH = lengths->getHandle<int32_t>();

  size_t segments = lengths->dims()[0];
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    totalLength += LH.raw(i);
  }
  assert(totalLength <= indices->dims()[0] &&
         "sum(Lengths) must be equal to len(Indices)");

  size_t lineSize = data->size() / data->dims()[0];

  auto DH = data->getHandle<ElemTy>();
  auto OH = out->getHandle<ElemTy>();

  size_t curIdx = 0;
  for (size_t i = 0; i < segments; i++) {
    for (size_t j = 0, e = LH.raw(i); j < e; j++) {
      size_t offsetIn = IH.raw(curIdx++) * lineSize;
      size_t offsetOut = i * lineSize;
      for (size_t k = 0; k < lineSize; k++)
        OH.raw(offsetOut++) += DH.raw(offsetIn++);
    }
  }
}

void BoundInterpreterFunction::fwdSparseLengthsSumInst(
    const SparseLengthsSumInst *I) {
  if (I->getDest()->getType()->isQuantizedType()) {
    dispatchIndexTypeImpl(fwdSparseLengthsSumInstI8Impl,
                          I->getIndices()->getElementType(), I);
    return;
  }
  dispatchFloatingPointAndIndexImpl(fwdSparseLengthsSumInstFloatImpl,
                                    I->getData()->getElementType(),
                                    I->getIndices()->getElementType(), I);
}

template <typename ElemTy, typename TI>
void BoundInterpreterFunction::fwdSparseLengthsWeightedSumInstFloatImpl(
    const SparseLengthsWeightedSumInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto out = getTensor(I->getDest());
  auto data = getTensor(I->getData());
  auto weights = getTensor(I->getWeights());
  auto indices = getTensor(I->getIndices());
  auto lengths = getTensor(I->getLengths());

  out->zero();

  auto IH = indices->getHandle<TI>();
  auto LH = lengths->getHandle<int32_t>();

  size_t segments = lengths->dims()[0];
  size_t totalLength = 0;
  for (dim_t i = 0; i < segments; i++) {
    totalLength += LH.raw(i);
  }
  assert(totalLength <= indices->dims()[0] &&
         "sum(Lengths) must be equal to len(Indices)");

  dim_t lineSize = data->size() / data->dims()[0];

  auto DH = data->getHandle<ElemTy>();
  auto WH = weights->getHandle<ElemTy>();
  auto OH = out->getHandle<ElemTy>();

  dim_t curIdx = 0;
  for (dim_t i = 0; i < segments; i++) {
    for (dim_t j = 0, e = LH.raw(i); j < e; j++) {
      ElemTy weight = WH.raw(curIdx);
      size_t offsetIn = IH.raw(curIdx++) * lineSize;
      size_t offsetOut = i * lineSize;
      for (dim_t k = 0; k < lineSize; k++)
        OH.raw(offsetOut++) += DH.raw(offsetIn++) * weight;
    }
  }
}

template <typename TI>
void BoundInterpreterFunction::fwdSparseLengthsWeightedSumInstI8Impl(
    const SparseLengthsWeightedSumInst *I) {

  auto out = getTensor(I->getDest());
  auto data = getTensor(I->getData());
  auto weights = getTensor(I->getWeights());
  auto indices = getTensor(I->getIndices());
  auto lengths = getTensor(I->getLengths());

  out->zero();

  auto IH = indices->getHandle<TI>();
  auto LH = lengths->getHandle<int32_t>();

  dim_t segments = lengths->dims()[0];
  dim_t totalLength = 0;
  for (dim_t i = 0; i < segments; i++) {
    totalLength += LH.raw(i);
  }
  assert(totalLength <= indices->dims()[0] &&
         "sum(Lengths) must be equal to len(Indices)");

  dim_t lineSize = data->size() / data->dims()[0];

  auto DH = data->getHandle<int8_t>();
  auto WH = weights->getHandle<int8_t>();
  auto OH = out->getHandle<int8_t>();

  auto TQP = [](Tensor *T) {
    return TensorQuantizationParams{T->getType().getScale(),
                                    T->getType().getOffset()};
  };
  using namespace quantization;

  dim_t curIdx = 0;
  for (dim_t i = 0; i < segments; i++) {
    std::vector<float> accum(lineSize, 0.0f);
    for (int32_t j = 0; j < LH.raw(i); j++) {
      float weight = dequantize(WH.raw(curIdx), TQP(weights));
      size_t offsetIn = IH.raw(curIdx) * lineSize;
      for (dim_t k = 0; k < lineSize; k++) {
        accum[k] += weight * dequantize(DH.raw(offsetIn++), TQP(data));
      }
      curIdx++;
    }
    dim_t offsetOut = i * lineSize;
    for (dim_t k = 0; k < lineSize; k++) {
      OH.raw(offsetOut++) = quantize(accum[k], TQP(out));
    }
  }
}

void BoundInterpreterFunction::fwdSparseLengthsSumGradInst(
    const SparseLengthsSumGradInst * /*I*/) {
  DCHECK(!"Found SparseLengthsSumGradInst but SparseLengthsSum is lowered on "
          "Interpreter");
}

void BoundInterpreterFunction::fwdSparseLengthsWeightedSumInst(
    const SparseLengthsWeightedSumInst *I) {
  if (I->getDest()->getType()->isQuantizedType()) {
    dispatchIndexTypeImpl(fwdSparseLengthsWeightedSumInstI8Impl,
                          I->getIndices()->getElementType(), I);
    return;
  }
  dispatchFloatingPointAndIndexImpl(fwdSparseLengthsWeightedSumInstFloatImpl,
                                    I->getData()->getElementType(),
                                    I->getIndices()->getElementType(), I);
}

void BoundInterpreterFunction::fwdSparseLengthsWeightedSumGradInst(
    const SparseLengthsWeightedSumGradInst *I) {
  assert(I->getDataGrad()->getType()->getElementType() == ElemKind::FloatTy &&
         "Input type must be float");

  auto destGrad = getTensor(I->getDestGrad());
  auto data = getTensor(I->getData());
  auto dataGrad = getTensor(I->getDataGrad());
  auto weightsGrad = getTensor(I->getWeightsGrad());
  auto weights = getTensor(I->getWeights());
  auto indices = getTensor(I->getIndices());
  auto lengths = getTensor(I->getLengths());

  // The data gradients not touched by this operation should
  // be 0, so set the entire buffer to 0 to start with.
  dataGrad->zero();

  auto LH = lengths->getHandle<int32_t>();
  auto IH = indices->getHandle<int64_t>();

  size_t segments = lengths->dims()[0];
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; ++i) {
    totalLength += LH.raw(i);
  }
  assert(totalLength == indices->dims()[0] &&
         "sum(Lengths) must be equal to len(Indices)");

  size_t lineSize = dataGrad->size() / dataGrad->dims()[0];

  auto IGH = destGrad->getHandle();
  auto WH = weights->getHandle();
  auto WGH = weightsGrad->getHandle();
  auto DH = data->getHandle();
  auto OGH = dataGrad->getHandle();

  // For each index in each segment:
  //    1) accumulate into the corresponding data gradient the product of the
  //    gradient of the result it was added to and the weight that it was
  //    multiplied by during the SparseLengthsWeightedSum operation.
  //
  //    2) accumulate into each weight gradient the reduced sum of the
  //    elementwise product of the result slice that the corresponding weight
  //    produced and the input slice that the weight was multiplied with.
  for (size_t i = 0, curIdx = 0; i < segments; ++i) {
    size_t destOffset = i * lineSize;
    for (size_t j = 0, e = LH.raw(i); j < e; ++j, ++curIdx) {
      float weightGrad = 0.0f;
      float weight = WH.raw(curIdx);
      size_t dataOffset = IH.raw(curIdx) * lineSize;

      for (size_t k = 0; k < lineSize; ++k) {
        OGH.raw(dataOffset + k) += IGH.raw(destOffset + k) * weight;
        weightGrad += IGH.raw(destOffset + k) * DH.raw(dataOffset + k);
      }

      WGH.raw(curIdx) = weightGrad;
    }
  }
}

template <typename ElemTy, typename IndexType>
void BoundInterpreterFunction::fwdEmbeddingBagInstFloatImpl(
    const EmbeddingBagInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto out = getTensor(I->getDest());
  auto data = getTensor(I->getData());
  auto weights = getTensor(I->getWeights());
  auto indices = getTensor(I->getIndices());
  auto offsets = getTensor(I->getOffsets());
  bool hasEndOffset = I->getHasEndOffset();

  out->zero();

  auto IH = indices->getHandle<IndexType>();
  auto OFFH = offsets->getHandle<IndexType>();

  // If an end offset is present to mark the end of the last segment then this
  // must be subtracted to get the correct number of segments
  size_t segments = hasEndOffset ? offsets->dims()[0] - 1 : offsets->dims()[0];
  size_t numIndices = indices->dims()[0];

  size_t lineSize = data->size() / data->dims()[0];

  auto DH = data->getHandle<ElemTy>();
  auto WH = weights->getHandle<ElemTy>();
  auto OH = out->getHandle<ElemTy>();

  dim_t curIdx = 0;
  for (dim_t i = 0; i < segments; i++) {
    dim_t start = OFFH.raw(i);
    dim_t end;
    if (!hasEndOffset) {
      // Note that in this case we have to use numIndices to find the end of
      // the last segment. This is an issue though because it relies on
      // knowing the total length of the indices tensor which may not be
      // possible. Future implementations of this operator should always give
      // an end offset so eventually this case should be removed.
      end = i == segments - 1 ? numIndices : OFFH.raw(i + 1);
    } else {
      end = OFFH.raw(i + 1);
    }
    if (start == end) {
      continue;
    } else if (start > end) {
      break;
    }
    for (dim_t j = start; j < end; j++) {
      ElemTy weight = WH.raw(curIdx);
      dim_t offsetIn = IH.raw(curIdx++) * lineSize;
      dim_t offsetOut = i * lineSize;
      for (dim_t k = 0; k < lineSize; k++) {
        OH.raw(offsetOut++) += DH.raw(offsetIn++) * weight;
      }
    }
  }
}

void BoundInterpreterFunction::fwdEmbeddingBagInst(const EmbeddingBagInst *I) {
  dispatchFloatingPointAndIndexImpl(fwdEmbeddingBagInstFloatImpl,
                                    I->getData()->getElementType(),
                                    I->getIndices()->getElementType(), I);
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdEmbeddingInstImpl(Tensor *wtT, Tensor *indT,
                                                    Tensor *outT,
                                                    int64_t padIdx, bool sparse,
                                                    bool scale,
                                                    dim_t embedding_dim) {

  staticAssertFloatingPointType(ElemTy);

  assert(!scale && "Currently only support scale_grad_by_freq == 'false'");
  assert(!sparse && "Currently only support sparse == 'false'");

  // Indices Tensor can be an arbitrary shape.
  // Get it flattened to 1D vector of size indLen
  // Output Tensor can be an arbitray shape.
  // Get it reshaped to a 2D tensor of size (indLen, embedding_dim)
  dim_t indLen = 1;
  for (dim_t idx = 0; idx < indT->dims().size(); ++idx) {
    indLen *= indT->dims()[idx];
  }
  auto fIndT = indT->getUnowned({indLen});
  auto fOutT = outT->getUnowned({indLen, embedding_dim});

  fOutT.zero();

  auto WH = wtT->getHandle<ElemTy>();
  auto OH = fOutT.getHandle<ElemTy>();
  auto IH = fIndT.getHandle<int32_t>();

  for (dim_t i = 0; i < indLen; i++) {
    dim_t index = IH.at(i);
    if (index != padIdx) {
      for (dim_t j = 0; j < embedding_dim; j++) {
        OH.at({i, j}) = WH.at({index, j});
      }
    }
  }
}

void BoundInterpreterFunction::fwdEmbeddingInst(const EmbeddingInst *I) {
  auto wtT = getTensor(I->getWeights());
  auto indT = getTensor(I->getIndices());
  auto outT = getTensor(I->getDest());
  auto padIdx = I->getPadIdx();
  bool sparse = I->getSparse();
  bool scale = I->getScale();
  dim_t embedding_dim = wtT->dims()[1];
  auto elemTy = wtT->getElementType();

  if (padIdx > -1) {
    assert(static_cast<dim_t>(padIdx) < wtT->dims()[0] &&
           "padIdx should be within num_embeddings");
  }

  dispatchFloatingPointImpl(fwdEmbeddingInstImpl, elemTy, wtT, indT, outT,
                            padIdx, sparse, scale, embedding_dim);
}

template <typename T, typename AccumT, typename TI>
void BoundInterpreterFunction::fwdRowwiseQuantizedSparseLengthsWeightedSumImpl(
    const RowwiseQuantizedSparseLengthsWeightedSumInst *I) {
  auto *out = getTensor(I->getDest());
  auto *data = getTensor(I->getData());
  auto *dataScales = getTensor(I->getScales());
  auto *dataOffsets = getTensor(I->getOffsets());
  auto *weights = getTensor(I->getWeights());
  auto *indices = getTensor(I->getIndices());
  auto *lengths = getTensor(I->getLengths());

  out->zero();

  auto IH = indices->getHandle<TI>();
  auto LH = lengths->getHandle<int32_t>();

  dim_t segments = lengths->dims()[0];
  dim_t totalLength = 0;
  for (dim_t i = 0; i < segments; i++) {
    totalLength += LH.raw(i);
  }
  assert(totalLength <= indices->dims()[0] &&
         "sum(Lengths) must be equal to len(Indices)");

  dim_t lineSize = data->size() / data->dims()[0];

  auto DH = data->getHandle<uint8_t>();
  auto DSH = dataScales->getHandle<T>();
  auto DOH = dataOffsets->getHandle<T>();
  auto WH = weights->getHandle<T>();
  auto OH = out->getHandle<T>();

  dim_t curIdx = 0;
  for (dim_t i = 0; i < segments; i++) {
    std::vector<AccumT> accum(lineSize, 0.0f);
    for (dim_t j = 0, e = LH.raw(i); j < e; j++) {
      const float weight = static_cast<float>(WH.raw(curIdx));
      const dim_t rowIdx = IH.raw(curIdx++);
      const float scale = static_cast<float>(DSH.at({rowIdx}));
      const float offset = static_cast<float>(DOH.at({rowIdx}));
      size_t offsetIn = rowIdx * lineSize;
      for (dim_t k = 0; k < lineSize; k++) {
        float d = quantization::dequantizeWithFloatOffset(DH.raw(offsetIn++),
                                                          scale, offset);
        accum[k] += d * weight;
      }
    }
    // Accumulation in FP32 complete, now copy back to output with cast to T.
    size_t offsetOut = i * lineSize;
    for (size_t k = 0; k < lineSize; k++) {
      OH.raw(offsetOut++) = static_cast<T>(accum[k]);
    }
  }
}

void BoundInterpreterFunction::fwdRowwiseQuantizedSparseLengthsWeightedSumInst(
    const RowwiseQuantizedSparseLengthsWeightedSumInst *I) {
  const auto ity = I->getIndices()->getElementType();
  switch (I->getDest()->getElementType()) {
  case ElemKind::FloatTy:
    if (ity == ElemKind::Int32ITy) {
      fwdRowwiseQuantizedSparseLengthsWeightedSumImpl<float, float, int32_t>(I);
    } else if (ity == ElemKind::Int64ITy) {
      fwdRowwiseQuantizedSparseLengthsWeightedSumImpl<float, float, int64_t>(I);
    } else {
      llvm_unreachable("Index type is not supported");
    }
    break;
  case ElemKind::Float16Ty:
    if (I->getUseFP16Accumulation()) {
      if (ity == ElemKind::Int32ITy) {
        fwdRowwiseQuantizedSparseLengthsWeightedSumImpl<float16_t, float16_t,
                                                        int32_t>(I);
      } else if (ity == ElemKind::Int64ITy) {
        fwdRowwiseQuantizedSparseLengthsWeightedSumImpl<float16_t, float16_t,
                                                        int64_t>(I);
      } else {
        llvm_unreachable("Index type is not supported");
      }
    } else {
      if (ity == ElemKind::Int32ITy) {
        fwdRowwiseQuantizedSparseLengthsWeightedSumImpl<float16_t, float,
                                                        int32_t>(I);
      } else if (ity == ElemKind::Int64ITy) {
        fwdRowwiseQuantizedSparseLengthsWeightedSumImpl<float16_t, float,
                                                        int64_t>(I);
      } else {
        llvm_unreachable("Index type is not supported");
      }
    }
    break;
  default:
    llvm_unreachable("Type is not supported");
  }
}

template <typename T, typename AccumT, typename TI>
void BoundInterpreterFunction::
    fwdFusedRowwiseQuantizedSparseLengthsWeightedSumImpl(
        const FusedRowwiseQuantizedSparseLengthsWeightedSumInst *I) {
  auto *out = getTensor(I->getDest());
  auto *data = getTensor(I->getData());
  auto *weights = getTensor(I->getWeights());
  auto *indices = getTensor(I->getIndices());
  auto *lengths = getTensor(I->getLengths());

  out->zero();

  auto IH = indices->getHandle<TI>();
  auto LH = lengths->getHandle<int32_t>();

  size_t segments = lengths->dims()[0];
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    totalLength += LH.raw(i);
  }
  assert(totalLength <= indices->dims()[0] &&
         "sum(Lengths) must be equal to len(Indices)");

  const bool using4BitQuantization =
      data->getType().getElementType() == ElemKind::UInt4FusedFP16QTy ||
      data->getType().getElementType() == ElemKind::UInt4FusedQTy;

  const size_t outLineSize = out->size() / out->dims()[0];

  auto DH = data->getHandle<uint8_t>();
  auto WH = weights->getHandle<T>();
  auto OH = out->getHandle<T>();

  dim_t curIdx = 0;
  for (dim_t i = 0; i < segments; i++) {
    std::vector<AccumT> accum(outLineSize, 0.0f);
    for (dim_t j = 0, e = LH.raw(i); j < e; j++) {
      const float weight = static_cast<float>(WH.raw(curIdx));
      const dim_t rowIdx = IH.raw(curIdx++);
      // Data type for the Scale and Offset for fused types need not follow
      // the type for the output Tensor passed in T.
      float scale, offset;
      switch (
          getScaleOffsetElemKindFromFused(data->getType().getElementType())) {
      case ElemKind::FloatTy:
        std::tie(scale, offset) = DH.getFusedScaleOffsetFromRow<float>(rowIdx);
        break;
      case ElemKind::Float16Ty:
        std::tie(scale, offset) =
            DH.getFusedScaleOffsetFromRow<float16_t>(rowIdx);
        break;
      default:
        llvm_unreachable("Type is not supported");
        break;
      }

      for (dim_t k = 0; k < outLineSize; k++) {
        float d = 0.0f;
        if (!using4BitQuantization) {
          d = quantization::dequantizeWithFloatOffset(
              DH.at({rowIdx, k}), static_cast<float>(scale),
              static_cast<float>(offset));
        } else {
          const bool isMSB = (k % 2 == 1);
          d = quantization::dequantize4BitWithFloatOffset(
              DH.at({rowIdx, k / 2}), static_cast<float>(scale),
              static_cast<float>(offset), isMSB);
        }
        accum[k] += d * weight;
      }
    }
    // Accumulation in FP32 complete, now copy back to output with cast to T.
    dim_t offsetOut = i * outLineSize;
    for (dim_t k = 0; k < outLineSize; k++) {
      OH.raw(offsetOut++) = static_cast<T>(accum[k]);
    }
  }
}

void BoundInterpreterFunction::
    fwdFusedRowwiseQuantizedSparseLengthsWeightedSumInst(
        const FusedRowwiseQuantizedSparseLengthsWeightedSumInst *I) {
  const auto ity = I->getIndices()->getElementType();
  const bool fp32FusedScaleOffset =
      (I->getData()->getElementType() == ElemKind::UInt4FusedQTy) ||
      (I->getData()->getElementType() == ElemKind::UInt8FusedQTy);

  switch (I->getDest()->getElementType()) {
  case ElemKind::FloatTy:
    if (ity == ElemKind::Int32ITy) {
      fwdFusedRowwiseQuantizedSparseLengthsWeightedSumImpl<float, float,
                                                           int32_t>(I);
    } else if (ity == ElemKind::Int64ITy) {
      fwdFusedRowwiseQuantizedSparseLengthsWeightedSumImpl<float, float,
                                                           int64_t>(I);
    } else {
      llvm_unreachable("Index type is not supported");
    }
    break;
  case ElemKind::Float16Ty:
    if (I->getUseFP16Accumulation() && !fp32FusedScaleOffset) {
      if (ity == ElemKind::Int32ITy) {
        fwdFusedRowwiseQuantizedSparseLengthsWeightedSumImpl<
            float16_t, float16_t, int32_t>(I);
      } else if (ity == ElemKind::Int64ITy) {
        fwdFusedRowwiseQuantizedSparseLengthsWeightedSumImpl<
            float16_t, float16_t, int64_t>(I);
      } else {
        llvm_unreachable("Index type is not supported");
      }
    } else {
      if (ity == ElemKind::Int32ITy) {
        fwdFusedRowwiseQuantizedSparseLengthsWeightedSumImpl<float16_t, float,
                                                             int32_t>(I);
      } else if (ity == ElemKind::Int64ITy) {
        fwdFusedRowwiseQuantizedSparseLengthsWeightedSumImpl<float16_t, float,
                                                             int64_t>(I);
      } else {
        llvm_unreachable("Index type is not supported");
      }
    }
    break;
  default:
    llvm_unreachable("Type is not supported");
  }
}

void BoundInterpreterFunction::fwdFusedRowwiseQuantizedSparseLengthsSumInst(
    const FusedRowwiseQuantizedSparseLengthsSumInst *I) {
  llvm_unreachable("Not supported");
}

template <typename T, typename AccumT, typename IndexT>
void BoundInterpreterFunction::fwdEmbeddingBagByteRowwiseOffsetsImpl(
    const EmbeddingBagByteRowwiseOffsetsInst *I) {
  auto *out = getTensor(I->getDest());
  auto *data = getTensor(I->getData());
  auto *weights = getTensor(I->getWeights());
  auto *indices = getTensor(I->getIndices());
  auto *offsets = getTensor(I->getOffsets());
  bool hasEndOffset = I->getHasEndOffset();

  out->zero();

  auto IH = indices->getHandle<IndexT>();
  auto OFFH = offsets->getHandle<IndexT>();

  // If an end offset is present to mark the end of the last segment then this
  // must be subtracted to get the correct number of segments
  size_t segments = hasEndOffset ? offsets->dims()[0] - 1 : offsets->dims()[0];
  dim_t numIndices = indices->dims()[0];

  const bool using4BitQuantization =
      data->getType().getElementType() == ElemKind::UInt4FusedFP16QTy;

  const size_t outLineSize = out->size() / out->dims()[0];

  auto DH = data->getHandle<uint8_t>();
  auto WH = weights->getHandle<T>();
  auto OH = out->getHandle<T>();

  for (dim_t i = 0; i < segments; i++) {
    std::vector<AccumT> accum(outLineSize, 0.0f);
    size_t start = OFFH.raw(i);
    dim_t end;
    if (!hasEndOffset) {
      // Note that in this case we have to use numIndices to find the end of
      // the last segment. This is an issue though because it relies on
      // knowing the total length of the indices tensor which may not be
      // possible. Future implementations of this operator should always give
      // an end offset so eventually this case should be removed.
      end = i == segments - 1 ? numIndices : OFFH.raw(i + 1);
    } else {
      end = OFFH.raw(i + 1);
    }
    if (start == end) {
      continue;
    } else if (start > end) {
      break;
    }

    for (dim_t j = start; j < end; j++) {
      const float weight = static_cast<float>(WH.raw(j));
      const dim_t rowIdx = IH.raw(j);
      T scale, offset;
      std::tie(scale, offset) = DH.getFusedScaleOffsetFromRow<T>(rowIdx);
      for (dim_t k = 0; k < outLineSize; k++) {
        float d = 0.0f;
        if (!using4BitQuantization) {
          d = quantization::dequantizeWithFloatOffset(
              DH.at({rowIdx, k}), static_cast<float>(scale),
              static_cast<float>(offset));
        } else {
          const bool isMSB = (k % 2 == 1);
          d = quantization::dequantize4BitWithFloatOffset(
              DH.at({rowIdx, k / 2}), static_cast<float>(scale),
              static_cast<float>(offset), isMSB);
        }
        accum[k] += d * weight;
      }
    }
    // Accumulation in FP32 complete, now copy back to output with cast to T.
    dim_t offsetOut = i * outLineSize;
    for (dim_t k = 0; k < outLineSize; k++) {
      OH.raw(offsetOut++) = static_cast<T>(accum[k]);
    }
  }
}

void BoundInterpreterFunction::fwdEmbeddingBagByteRowwiseOffsetsInst(
    const EmbeddingBagByteRowwiseOffsetsInst *I) {
  const auto ity = I->getIndices()->getElementType();
  const bool fp32FusedScaleOffset =
      (I->getData()->getElementType() == ElemKind::UInt4FusedQTy) ||
      (I->getData()->getElementType() == ElemKind::UInt8FusedQTy);

  switch (I->getDest()->getElementType()) {
  case ElemKind::FloatTy:
    if (ity == ElemKind::Int32ITy) {
      fwdEmbeddingBagByteRowwiseOffsetsImpl<float, float, int32_t>(I);
    } else if (ity == ElemKind::Int64ITy) {
      fwdEmbeddingBagByteRowwiseOffsetsImpl<float, float, int64_t>(I);
    } else {
      llvm_unreachable("Index type is not supported");
    }
    break;
  case ElemKind::Float16Ty:
    if (I->getUseFP16Accumulation() && !fp32FusedScaleOffset) {
      if (ity == ElemKind::Int32ITy) {
        fwdEmbeddingBagByteRowwiseOffsetsImpl<float16_t, float16_t, int32_t>(I);
      } else if (ity == ElemKind::Int64ITy) {
        fwdEmbeddingBagByteRowwiseOffsetsImpl<float16_t, float16_t, int64_t>(I);
      } else {
        llvm_unreachable("Index type is not supported");
      }
    } else {
      if (ity == ElemKind::Int32ITy) {
        fwdEmbeddingBagByteRowwiseOffsetsImpl<float16_t, float, int32_t>(I);
      } else if (ity == ElemKind::Int64ITy) {
        fwdEmbeddingBagByteRowwiseOffsetsImpl<float16_t, float, int64_t>(I);
      } else {
        llvm_unreachable("Index type is not supported");
      }
    }
    break;
  default:
    llvm_unreachable("Type is not supported");
  }
}

void BoundInterpreterFunction::fwdLengthsToRangesInst(
    const LengthsToRangesInst *I) {
  auto ranges = getTensor(I->getDest())->getHandle<int32_t>();
  auto lengths = getTensor(I->getLengths())->getHandle<int32_t>();
  int32_t offset = 0;
  for (dim_t i = 0; i < lengths.dims()[0]; i++) {
    auto length = lengths.at({i});
    ranges.at({i, 0}) = offset;
    ranges.at({i, 1}) = length;
    offset += length;
  }
}

void BoundInterpreterFunction::fwdLengthsRangeFillInst(
    const LengthsRangeFillInst *I) {
  auto lengthsH = getTensor(I->getLengths())->getHandle<int32_t>();
  auto resultH = getTensor(I->getDest())->getHandle<int32_t>();
  dim_t curIdx = 0;
  for (dim_t i = 0, e = lengthsH.dims()[0]; i < e; i++) {
    for (int32_t j = 0, f = lengthsH.at({i}); j < f; j++) {
      resultH.at({curIdx++}) = j;
    }
  }
}

void BoundInterpreterFunction::fwdGaussianFillInst(const GaussianFillInst *I) {
  std::mt19937 rnd(I->getSeed());
  std::normal_distribution<float> dist(I->getMean(), I->getScale());
  auto outT = getTensor(I->getDest());
  auto outH = outT->getHandle<float16_t>();
  for (auto &elem : outH) {
    elem = dist(rnd);
  }
}

template <typename ElemTy, typename LengthsTy, typename IndicesTy>
void BoundInterpreterFunction::fwdBatchSparseToDenseInstImpl2(
    const BatchSparseToDenseInst *I) {
  auto outH = getWeightHandle<ElemTy>(I->getDest());
  auto lengthsH = getWeightHandle<LengthsTy>(I->getLengths());
  auto valuesH = getWeightHandle<ElemTy>(I->getValues());
  auto indicesH = getWeightHandle<IndicesTy>(I->getIndices());
  auto denseLastDim = I->getDenseLastDim();
  auto defaultValue = I->getDefaultValue();
  outH.clear(defaultValue);

  // Verifying input sizes.
  size_t lengthsSum = 0;
  auto batchSize = lengthsH.size();
  for (dim_t i = 0; i < batchSize; ++i) {
    lengthsSum += lengthsH.at(i);
  }
  CHECK_EQ(lengthsSum, indicesH.size());

  dim_t k = 0;
  for (dim_t i = 0; i < batchSize; ++i) {
    for (dim_t j = 0; j < lengthsH.at(i); ++j) {
      CHECK_LT(indicesH.at(i), denseLastDim);
      outH.at({static_cast<dim_t>(i), static_cast<dim_t>(indicesH.at(k))}) =
          valuesH.at(k);
      k++;
    }
  }
}

template <typename ElemTy, typename LengthsTy>
void BoundInterpreterFunction::fwdBatchSparseToDenseInstImpl1(
    const BatchSparseToDenseInst *I) {
  switch (I->getLengths()->getElementType()) {
  case ElemKind::Int32ITy:
    fwdBatchSparseToDenseInstImpl2<ElemTy, LengthsTy, int32_t>(I);
    break;
  case ElemKind::Int64ITy:
    fwdBatchSparseToDenseInstImpl2<ElemTy, LengthsTy, int64_t>(I);
    break;
  default:
    llvm_unreachable("Index type is not supported");
  }
}

void BoundInterpreterFunction::fwdBatchSparseToDenseInst(
    const BatchSparseToDenseInst *I) {
  dispatchFloatingPointAndIndexImpl(fwdBatchSparseToDenseInstImpl1,
                                    I->getDest()->getElementType(),
                                    I->getLengths()->getElementType(), I);
}

template <typename ElemTy, typename IndicatorTy>
void BoundInterpreterFunction::fwdFillExamplesWithIndicatorInstImpl2(
    const FillExamplesWithIndicatorInst *I) {
  auto outT = getTensor(I->getDest());
  auto dataT = getTensor(I->getData());
  auto elemSize = dataT->getType().getElementSize();
  auto indicatorH = getWeightHandle<IndicatorTy>(I->getIndicator());

  size_t numBatches = indicatorH.dims()[0];
  outT->zero();

  size_t nonzero = 0;
  for (size_t i = 0; i < numBatches; ++i) {
    if (static_cast<bool>(indicatorH.at(i))) {
      nonzero++;
    }
  }
  CHECK_EQ(dataT->dims()[0], nonzero);

  // Calculate size of last n-1 data dims
  size_t blockSize = 1;
  for (size_t i = 1; i < dataT->dims().size(); i++) {
    blockSize *= dataT->dims()[i];
  }
  size_t blockByteSize = blockSize * elemSize;
  size_t dataP = 0;
  for (size_t i = 0; i < numBatches; i++) {
    if (static_cast<bool>(indicatorH.at(i))) {
      std::copy(&dataT->getUnsafePtr()[dataP],
                &dataT->getUnsafePtr()[dataP + blockByteSize],
                &outT->getUnsafePtr()[i * blockByteSize]);
      dataP += blockByteSize;
    }
  }
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdFillExamplesWithIndicatorInstImpl1(
    const FillExamplesWithIndicatorInst *I) {
  switch (I->getIndicator()->getElementType()) {
  case ElemKind::Int32ITy:
    fwdFillExamplesWithIndicatorInstImpl2<ElemTy, int32_t>(I);
    break;
  case ElemKind::Int64ITy:
    fwdFillExamplesWithIndicatorInstImpl2<ElemTy, int64_t>(I);
    break;
  case ElemKind::BoolTy:
    fwdFillExamplesWithIndicatorInstImpl2<ElemTy, bool>(I);
    break;
  default:
    llvm_unreachable("Indicator type is not supported");
  }
}

void BoundInterpreterFunction::fwdFillExamplesWithIndicatorInst(
    const FillExamplesWithIndicatorInst *I) {
  dispatchArithmeticImpl(fwdFillExamplesWithIndicatorInstImpl1,
                         I->getDest()->getElementType(), I);
}
void BoundInterpreterFunction::fwdSparseToDenseMaskInst(
    const SparseToDenseMaskInst *I) {
  auto out = getTensor(I->getDest());
  auto values = getTensor(I->getValues());
  auto defaultValue = getTensor(I->getDefaultValue());

  auto indicesH = getTensor(I->getIndices())->getHandle<int64_t>();
  auto lengthsH = getTensor(I->getLengths())->getHandle<int32_t>();

  const std::vector<dim_t> &mask = I->getMask();
  size_t maskSize = mask.size();
  // Create a reverse map from ID to its position in the mask.
  std::unordered_map<int64_t, size_t> reverseMap;
  for (size_t i = 0; i < maskSize; i++) {
    assert(reverseMap.find(mask[i]) == reverseMap.end() &&
           "duplicate IDs in the mask");
    reverseMap[mask[i]] = i;
  }

  auto valueSize = defaultValue->getSizeInBytes();

  // First un-processed index-value pair.
  size_t posIn = 0;
  // Beginning of output block for first unprocessed batch.
  size_t byteOffsetOut = 0;
  // Lengths can be scalar, which means that all pairs belong to one batch.
  size_t numBatches = lengthsH.dims().empty() ? 1 : lengthsH.dims()[0];
  for (size_t batch = 0; batch < numBatches; batch++) {
    // Fill everything with maskSize copies of defaultValue.
    for (size_t i = 0; i < maskSize; i++) {
      std::copy(defaultValue->getUnsafePtr(),
                &defaultValue->getUnsafePtr()[valueSize],
                &out->getUnsafePtr()[byteOffsetOut + valueSize * i]);
    }
    // Go through input pairs and find matches.
    for (size_t i = 0, batchLen = lengthsH.raw(batch); i < batchLen;
         i++, posIn++) {
      int64_t idx = indicesH.raw(posIn);
      auto it = reverseMap.find(idx);
      // Skip if ID is not present in the mask.
      if (it == reverseMap.end())
        continue;
      size_t to = it->second;

      std::copy(&values->getUnsafePtr()[posIn * valueSize],
                &values->getUnsafePtr()[(posIn + 1) * valueSize],
                &out->getUnsafePtr()[byteOffsetOut + valueSize * to]);
    }

    byteOffsetOut += maskSize * valueSize;
  }

  assert(posIn == indicesH.dims()[0] &&
         "Sum of Lengths must be equal to size of indices.");
}

void BoundInterpreterFunction::fwdSparseLabelSplitInst(
    const SparseLabelSplitInst *I) {
  auto lengthsH = getTensor(I->getLengths())->getHandle<int32_t>();
  auto indicesH = getTensor(I->getIndices())->getHandle<int64_t>();
  auto valuesH = getTensor(I->getValues())->getHandle();

  const auto numLabels = I->getNumLabels();

  auto labelValuesH = getTensor(I->getLabelValues())->getHandle();
  auto exampleIdsH = getTensor(I->getExampleIds())->getHandle<int32_t>();
  auto gradientOffsetMapH =
      getTensor(I->getGradientOffsetMap())->getHandle<int32_t>();

  // Verifying input sizes.
  size_t lengthsSum = 0;
  for (size_t i = 0; i < lengthsH.size(); ++i) {
    lengthsSum += lengthsH.at(i);
  }
  CHECK_EQ(lengthsSum, indicesH.size());
  CHECK_EQ(indicesH.size(), valuesH.size());

  // Verifying that outputs have same sizes.
  const auto numValuesPerRow = indicesH.size() / numLabels;
  std::vector<size_t> numExamplesPerTask(numLabels, 0);
  for (size_t i = 0; i < indicesH.size(); ++i) {
    numExamplesPerTask[indicesH.at(i)] += 1;
  }
  for (size_t i = 0; i < numLabels; ++i) {
    CHECK_EQ(numValuesPerRow, numExamplesPerTask[i])
        << "Unexpected number of values at " << i;
  }

  // Populating outputs
  size_t pos = 0;
  std::fill(numExamplesPerTask.begin(), numExamplesPerTask.end(), 0);
  for (size_t i = 0; i < lengthsH.size(); ++i) {
    for (size_t l = 0; l < lengthsH.at(i); ++l) {
      auto ind = indicesH.at(pos);
      auto val = valuesH.at(pos);

      auto posOutput = numExamplesPerTask[ind]++;
      gradientOffsetMapH.at(pos) = posOutput;

      labelValuesH.at(
          {static_cast<dim_t>(ind), static_cast<dim_t>(posOutput)}) = val;
      exampleIdsH.at({static_cast<dim_t>(ind), static_cast<dim_t>(posOutput)}) =
          i;
      pos++;
    }
  }
}

//===----------------------------------------------------------------------===//
//                Instructions used by RNN
//===----------------------------------------------------------------------===//
template <typename T, typename TI>
static void fwdTopK(Tensor *outW, Tensor *indW, Tensor *inW, size_t k) {
  auto values = outW->getHandle<T>();
  auto indices = indW->getHandle<TI>();
  auto in = inW->getHandle<T>();
  size_t n = in.dims().back();

  size_t in_p = 0, out_p = 0;
  size_t tensor_end = in.size();
  using pairType = std::pair<float, size_t>;
  std::vector<pairType> buf(n);

  while (in_p < tensor_end) {
    for (size_t i = 0; i < n; i++) {
      buf[i].first = in.raw(in_p++);
      buf[i].second = i;
    }
    // NOTE: it's possible to do N + KlogK, while this version is NlogN
    std::sort(buf.begin(), buf.end(), [](const pairType &a, const pairType &b) {
      if (a.first != b.first)
        return a.first > b.first;
      return a.second < b.second;
    });
    for (size_t i = 0; i < k; i++) {
      values.raw(out_p) = buf[i].first;
      indices.raw(out_p) = buf[i].second;
      out_p++;
    }
  }
}

template <typename inpType, typename outType>
static void fwdArgMax(Tensor *inpT, Tensor *outT, size_t axis) {

  // Get input/output handles with dimensions expanded to maximum.
  ShapeVector inpDims = expandDimsToMax(inpT->dims());
  ShapeVector outDims = inpDims;
  outDims[axis] = 1;
  auto eInpT = inpT->getUnowned(inpDims);
  auto eOutT = outT->getUnowned(outDims);
  auto inpH = eInpT.getHandle<inpType>();
  auto outH = eOutT.getHandle<outType>();

  static_assert(max_tensor_dimensions == 6,
                "Loops below assume max_tensor_dimensions = 6.");

  for (dim_t idx0 = 0; idx0 < outDims[0]; idx0++) {
    for (dim_t idx1 = 0; idx1 < outDims[1]; idx1++) {
      for (dim_t idx2 = 0; idx2 < outDims[2]; idx2++) {
        for (dim_t idx3 = 0; idx3 < outDims[3]; idx3++) {
          for (dim_t idx4 = 0; idx4 < outDims[4]; idx4++) {
            for (dim_t idx5 = 0; idx5 < outDims[5]; idx5++) {

              // Initialize maximum value/index.
              inpType maxVal = std::numeric_limits<inpType>::lowest();
              outType maxIdx = 0;

              // Iterate input axis dimension.
              for (dim_t axisIdx = 0; axisIdx < inpDims[axis]; axisIdx++) {
                std::vector<dim_t> inpIdx = {idx0, idx1, idx2,
                                             idx3, idx4, idx5};
                inpIdx[axis] = axisIdx;
                inpType inpVal = inpH.at(inpIdx);
                if (inpVal > maxVal) {
                  maxVal = inpVal;
                  maxIdx = axisIdx;
                }
              }

              // Store maximum index.
              outH.at({idx0, idx1, idx2, idx3, idx4, idx5}) = maxIdx;
            }
          }
        }
      }
    }
  }
}

template <typename inpType, typename outType>
static void fwdArgMin(Tensor *inpT, Tensor *outT, size_t axis) {

  // Get input/output handles with dimensions expanded to maximum.
  ShapeVector inpDims = expandDimsToMax(inpT->dims());
  ShapeVector outDims = inpDims;
  outDims[axis] = 1;
  auto eInpT = inpT->getUnowned(inpDims);
  auto eOutT = outT->getUnowned(outDims);
  auto inpH = eInpT.getHandle<inpType>();
  auto outH = eOutT.getHandle<outType>();

  static_assert(max_tensor_dimensions == 6,
                "Loops below assume max_tensor_dimensions = 6.");

  for (dim_t idx0 = 0; idx0 < outDims[0]; idx0++) {
    for (dim_t idx1 = 0; idx1 < outDims[1]; idx1++) {
      for (dim_t idx2 = 0; idx2 < outDims[2]; idx2++) {
        for (dim_t idx3 = 0; idx3 < outDims[3]; idx3++) {
          for (dim_t idx4 = 0; idx4 < outDims[4]; idx4++) {
            for (dim_t idx5 = 0; idx5 < outDims[5]; idx5++) {

              // Initialize minimum value/index.
              inpType minVal = std::numeric_limits<inpType>::max();
              outType minIdx = 0;

              // Iterate input axis dimension.
              for (dim_t axisIdx = 0; axisIdx < inpDims[axis]; axisIdx++) {
                std::vector<dim_t> inpIdx = {idx0, idx1, idx2,
                                             idx3, idx4, idx5};
                inpIdx[axis] = axisIdx;
                inpType inpVal = inpH.at(inpIdx);
                if (inpVal < minVal) {
                  minVal = inpVal;
                  minIdx = axisIdx;
                }
              }

              // Store minimum index.
              outH.at({idx0, idx1, idx2, idx3, idx4, idx5}) = minIdx;
            }
          }
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
//                       Sorting operators
//===----------------------------------------------------------------------===//
template <typename ElemTy>
static void
CollectRpnProposalsHelper(const std::vector<std::vector<ElemTy>> &roisIn,
                          const std::vector<ElemTy> &scores,
                          std::vector<std::vector<ElemTy>> &rois,
                          dim_t rpnPostNmsTopN) {
  // N + KlogK implementation, K is rpnPostNmsTopN
  dim_t roisSecondDim = roisIn[0].size();

  std::vector<dim_t> idx(scores.size());
  std::iota(idx.begin(), idx.end(), 0);

  auto comp = [&](dim_t leftIdx, dim_t rightIdx) -> bool {
    if (scores[leftIdx] > scores[rightIdx]) {
      return true;
    }
    if (scores[leftIdx] < scores[rightIdx]) {
      return false;
    }
    // Sort on indices if two scores are equal
    return leftIdx < rightIdx;
  };

  dim_t k = rpnPostNmsTopN;
  // Getting the indices in order according to scores
  // Arrange Kth element at its proper location of sorted array, O(N)
  if (k < roisIn.size()) {
    std::nth_element(idx.begin(), idx.begin() + k, idx.end(), comp);
  } else {
    k = roisIn.size();
  }

  rois.resize(k, std::vector<ElemTy>(roisSecondDim));

  // KlogK, K is rpnPostNmsTopN
  std::sort(idx.begin(), idx.begin() + k, comp);

  // Output rois according to new order of indices
  for (dim_t i = 0; i < k; i++) {
    rois[i] = roisIn[idx[i]];
  }
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdCollectRpnProposalsInstImpl(
    const glow::CollectRpnProposalsInst *I) {

  // Get params
  dim_t rpnPostNmsTopN = I->getRpnPostNmsTopN();
  int64_t rpnMaxLevel = I->getRpnMaxLevel();
  int64_t rpnMinLevel = I->getRpnMinLevel();
  int64_t rpnLevels = rpnMaxLevel - rpnMinLevel + 1;

  std::vector<std::vector<ElemTy>> roisIn;
  std::vector<ElemTy> scores;
  std::vector<std::vector<ElemTy>> roisOut;

  auto getRoisAndScores = [&](Tensor *input, bool isroi) {
    auto inputHndl = input->getHandle<ElemTy>();

    if (isroi) {
      for (dim_t i = 0; i < input->dims()[0]; i++) {
        std::vector<ElemTy> roi;

        for (dim_t j = 0; j < input->dims()[1]; j++) {
          roi.push_back(inputHndl.at({i, j}));
        }

        roisIn.push_back(roi);
      }
    } else {
      for (dim_t i = 0; i < input->dims()[0]; i++) {
        scores.push_back(inputHndl.at({i}));
      }
    }
  };

  // Input starts from index 1
  for (dim_t idx = 1; idx <= rpnLevels; idx++) {
    getRoisAndScores(getTensor(I->getOperand(idx).first), true);
    getRoisAndScores(getTensor(I->getOperand(idx + rpnLevels).first), false);
  }

  // Sorting the roisIn according to scores limited by rpnPostNmsTopN
  CollectRpnProposalsHelper<ElemTy>(roisIn, scores, roisOut, rpnPostNmsTopN);

  // Storing roisOut in result tensor
  dim_t roisSecondDim = roisIn[0].size();
  Tensor *result = getTensor(I->getResult());

  for (dim_t i = 0; i < rpnPostNmsTopN; i++) {
    for (dim_t j = 0; j < roisSecondDim; j++) {
      result->getHandle<ElemTy>().at({i, j}) = roisOut[i][j];
      // Invalid rois are set to 0
      if (i > roisOut.size()) {
        result->getHandle<ElemTy>().at({i, j}) = ElemTy(0);
      }
    }
  }
}

void BoundInterpreterFunction::fwdCollectRpnProposalsInst(
    const glow::CollectRpnProposalsInst *I) {
  dispatchFloatingPointImpl(fwdCollectRpnProposalsInstImpl,
                            I->getOperand(1).first->getElementType(), I);
}

void BoundInterpreterFunction::fwdTopKInst(const TopKInst *I) {
  auto outW = getTensor(I->getValues());
  auto indW = getTensor(I->getIndices());
  auto inW = getTensor(I->getInput());
  size_t k = I->getK();

  if (inW->getType().isQuantizedType()) {
    if (indW->getElementType() == ElemKind::Int64ITy) {

      fwdTopK<int8_t, int64_t>(outW, indW, inW, k);
    } else if (indW->getElementType() == ElemKind::Int32ITy) {
      fwdTopK<int8_t, int32_t>(outW, indW, inW, k);
    }
    return;
  }

  dispatchFloatingPointAndIndexImpl(fwdTopK, inW->getElementType(),
                                    indW->getElementType(), outW, indW, inW, k);
}

void BoundInterpreterFunction::fwdBatchedUnaryEmbeddingsBagsInst(
    const BatchedUnaryEmbeddingsBagsInst *I) {
  dispatchFloatingPointAndIndexImpl(fwdBatchedUnaryEmbeddingsBagsInstImpl,
                                    I->getWeights()->getElementType(),
                                    I->getIndices()->getElementType(), I);
}

template <typename ElemTy, typename IndexType>
void BoundInterpreterFunction::fwdBatchedUnaryEmbeddingsBagsInstImpl(
    const BatchedUnaryEmbeddingsBagsInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto out = getTensor(I->getDest());
  auto weights = getTensor(I->getWeights());
  auto tableOffsets = getTensor(I->getTableOffsets());
  auto indices = getTensor(I->getIndices());
  auto offsets = getTensor(I->getOffsets());

  out->zero();

  auto indicesH = indices->getHandle<IndexType>();
  auto offsetsH = offsets->getHandle<IndexType>();
  auto weightsH = weights->getHandle<ElemTy>();
  auto tableOffsetsH = tableOffsets->getHandle<IndexType>();
  auto outH = out->getHandle<ElemTy>();

  size_t numTasks = weightsH.dims()[0];
  size_t numTables = tableOffsets->size() - 1;
  size_t numBatches = (offsets->size() - 1) / numTables;

  IndexType sumTable = tableOffsetsH.raw(numTables);

  for (size_t n = 0; n < numTasks; n++) {
    for (size_t b = 0; b < numBatches; b++) {
      for (size_t t = 0; t < numTables; t++) {
        IndexType indicesStart = offsetsH.raw(t * numBatches + b);
        IndexType indicesEnd = offsetsH.raw(t * numBatches + b + 1);
        ElemTy sum = 0;
        for (IndexType i = indicesStart; i < indicesEnd; i++) {
          IndexType idx = n * sumTable + tableOffsetsH.raw(t) + indicesH.raw(i);
          assert(idx < weightsH.size() &&
                 "Index shall be within weights boundary.");
          sum += weightsH.raw(idx);
        }
        outH.raw((n * numBatches + b) * numTables + t) = sum;
      }
    }
  }
}

void BoundInterpreterFunction::fwdIntNBitSplitEmbeddingBagsInst(
    const IntNBitSplitEmbeddingBagsInst *I) {
  dispatchIndexAndOutputTypeImpl(fwdIntNBitSplitEmbeddingBagsInstImpl,
                                 I->getIndices()->getElementType(),
                                 I->getOutputDType(), I);
}

void BoundInterpreterFunction::fwdIntNBitSplitEmbeddingWeightedBagsInst(
    const IntNBitSplitEmbeddingWeightedBagsInst *I) {
  dispatchIndexAndOutputTypeImpl(fwdIntNBitSplitEmbeddingWeightedBagsInstImpl,
                                 I->getIndices()->getElementType(),
                                 I->getOutputDType(), I);
}

template <typename IndexTy, typename OutputTy>
void BoundInterpreterFunction::fwdIntNBitSplitEmbeddingBagsInstImpl(
    const IntNBitSplitEmbeddingBagsInst *I) {
  auto out = getTensor(I->getDest());
  auto devWeights = getTensor(I->getDevWeights());
  auto uvmWeights = getTensor(I->getUvmWeights());
  auto weightsPlacements = getTensor(I->getWeightsPlacements());
  auto weightsTys = getTensor(I->getWeightsTys());
  auto dimOffsets = getTensor(I->getDimOffsets());
  auto indices = getTensor(I->getIndices());
  auto offsets = getTensor(I->getOffsets());
  auto weightsOffsets = getTensor(I->getWeightsOffsets());
  auto poolingMode = I->getPoolingMode();
  auto totalDims = I->getTotalDims();
  auto outputDType = I->getOutputDType();

  fwdIntNBitSplitEmbeddingWeightedBagsImpl<IndexTy, OutputTy>(
      out, devWeights, uvmWeights, weightsPlacements, weightsTys, dimOffsets,
      indices, offsets, weightsOffsets, poolingMode,
      /* indiceWeights */ nullptr, totalDims, outputDType);
}

template <typename IndexTy, typename OutputTy>
void BoundInterpreterFunction::fwdIntNBitSplitEmbeddingWeightedBagsInstImpl(
    const IntNBitSplitEmbeddingWeightedBagsInst *I) {
  auto out = getTensor(I->getDest());
  auto devWeights = getTensor(I->getDevWeights());
  auto uvmWeights = getTensor(I->getUvmWeights());
  auto weightsPlacements = getTensor(I->getWeightsPlacements());
  auto weightsTys = getTensor(I->getWeightsTys());
  auto dimOffsets = getTensor(I->getDimOffsets());
  auto indices = getTensor(I->getIndices());
  auto offsets = getTensor(I->getOffsets());
  auto weightsOffsets = getTensor(I->getWeightsOffsets());
  auto poolingMode = I->getPoolingMode();
  auto totalDims = I->getTotalDims();
  auto outputDType = I->getOutputDType();
  auto indiceWeights = getTensor(I->getIndiceWeight());

  fwdIntNBitSplitEmbeddingWeightedBagsImpl<IndexTy, OutputTy>(
      out, devWeights, uvmWeights, weightsPlacements, weightsTys, dimOffsets,
      indices, offsets, weightsOffsets, poolingMode, indiceWeights, totalDims,
      outputDType);
}

template <typename IndexTy, typename OutputTy>
void BoundInterpreterFunction::fwdIntNBitSplitEmbeddingWeightedBagsImpl(
    Tensor *out, Tensor *devWeights, Tensor *uvmWeights,
    Tensor *weightsPlacements, Tensor *weightsTys, Tensor *dimOffsets,
    Tensor *indices, Tensor *offsets, Tensor *weightsOffsets,
    int64_t poolingMode, Tensor *indiceWeights, int64_t totalDims,
    int64_t outputDType) {
  out->zero();

  auto indicesH = indices->getHandle<IndexTy>();
  auto offsetsH = offsets->getHandle<IndexTy>();
  auto devWeightsH = devWeights->getHandle<uint8_t>();
  auto uvmWeightsH = uvmWeights->getHandle<uint8_t>();
  auto weightsPlacementH = weightsPlacements->getHandle<int32_t>();
  auto weightsTysH = weightsTys->getHandle<uint8_t>();
  auto dimOffsetsH = dimOffsets->getHandle<int32_t>();
  auto weightsOffsetsH = weightsOffsets->getHandle<int32_t>();
  llvm::Optional<Handle<float>> indiceWeightsH;
  if (indiceWeights) {
    indiceWeightsH = indiceWeights->getHandle<float>();
  }
  auto outH = out->getHandle<uint8_t>();

  size_t numTables = dimOffsets->size() - 1;
  size_t numBatches = (offsets->size() - 1) / numTables;
  auto outputSparseType = static_cast<SplitEmbeddingSparseType>(outputDType);
  auto numTotalBytes = IntNBitSplitEmbeddingBagsHelper::unpaddedRowSizeInBytes(
      totalDims, outputSparseType);
  int64_t maxDevWeightsBytes = devWeights->getSizeInBytes();
  int64_t maxUvmWeightsBytes = uvmWeights->getSizeInBytes();

  for (int32_t t = 0; t < numTables; t++) {
    const int32_t dimStart = dimOffsetsH.raw(t);
    const int32_t numDims = dimOffsetsH.raw(t + 1) - dimOffsetsH.raw(t);
    const auto placement = weightsPlacementH.raw(t);
    assert(placement != WeightsPlacement::DEVICE);
    auto weightsH = uvmWeightsH;
    auto maxWeightsBytes = maxUvmWeightsBytes;
    if (placement == WeightsPlacement::HOST) {
      weightsH = devWeightsH;
      maxWeightsBytes = maxDevWeightsBytes;
    }
    auto weightTy = static_cast<SplitEmbeddingSparseType>(weightsTysH.raw(t));
    assert(weightTy != SplitEmbeddingSparseType::EST_INT2 &&
           "Int2 sparse type isn't supported yet.");
    auto numDimBytes = IntNBitSplitEmbeddingBagsHelper::paddedRowSizeInBytes(
        numDims, weightTy);

    for (int32_t b = 0; b < numBatches; b++) {
      IndexTy indicesStart = offsetsH.raw(t * numBatches + b);
      IndexTy indicesEnd = offsetsH.raw(t * numBatches + b + 1);
      for (int32_t d = 0; d < numDims; d++) {
        OutputTy sum = 0;
        for (IndexTy i = indicesStart; i < indicesEnd; i++) {
          int64_t idxRow =
              weightsOffsetsH.raw(t) + indicesH.raw(i) * numDimBytes;
          int64_t idxData =
              idxRow + IntNBitSplitEmbeddingBagsHelper::unpaddedRowSizeInBytes(
                           d, weightTy);
          assert(idxData < maxWeightsBytes &&
                 "Index shall be within weights boundary.");
          sum = IntNBitSplitEmbeddingBagsHelper::add(
              sum, &weightsH.raw(idxRow), &weightsH.raw(idxData), weightTy,
              d % 2, indiceWeights ? indiceWeightsH->raw(i) : 1.0);
        }
        int64_t idxOut =
            b * numTotalBytes +
            IntNBitSplitEmbeddingBagsHelper::unpaddedRowSizeInBytes(
                dimStart + d, outputSparseType);
        if (poolingMode == SplitEmbeddingPoolingMode::EP_MEAN &&
            indicesEnd > indicesStart) {
          OutputTy scale = (indicesEnd - indicesStart);
          IntNBitSplitEmbeddingBagsHelper::save(sum / scale, &outH.raw(idxOut),
                                                outputSparseType);
        } else {
          IntNBitSplitEmbeddingBagsHelper::save(sum, &outH.raw(idxOut),
                                                outputSparseType);
        }
      }
    }
  }
}

#define DISPATCH_ARG_MIN_MAX(functionName, elemTy, elemTyIndex, ...)           \
  switch (elemTy) {                                                            \
  case ElemKind::FloatTy:                                                      \
    if (elemTyIndex == ElemKind::Int64ITy) {                                   \
      functionName<float, int64_t>(__VA_ARGS__);                               \
    } else if (elemTyIndex == ElemKind::Int32ITy) {                            \
      functionName<float, int32_t>(__VA_ARGS__);                               \
    }                                                                          \
    break;                                                                     \
  case ElemKind::Float16Ty:                                                    \
    if (elemTyIndex == ElemKind::Int64ITy) {                                   \
      functionName<float16_t, int64_t>(__VA_ARGS__);                           \
    } else if (elemTyIndex == ElemKind::Int32ITy) {                            \
      functionName<float16_t, int32_t>(__VA_ARGS__);                           \
    }                                                                          \
    break;                                                                     \
  case ElemKind::Int8QTy:                                                      \
    if (elemTyIndex == ElemKind::Int64ITy) {                                   \
      functionName<int8_t, int64_t>(__VA_ARGS__);                              \
    } else if (elemTyIndex == ElemKind::Int32ITy) {                            \
      functionName<int8_t, int32_t>(__VA_ARGS__);                              \
    }                                                                          \
    break;                                                                     \
  default:                                                                     \
    llvm_unreachable("Type is not supported");                                 \
  }

void BoundInterpreterFunction::fwdArgMaxInst(const ArgMaxInst *I) {
  auto inpT = getTensor(I->getSrc());
  auto outT = getTensor(I->getDest());
  size_t axis = I->getAxis();
  auto inpElemType = inpT->getElementType();
  auto outElemType = outT->getElementType();
  DISPATCH_ARG_MIN_MAX(fwdArgMax, inpElemType, outElemType, inpT, outT, axis);
}

void BoundInterpreterFunction::fwdArgMinInst(const ArgMinInst *I) {
  auto inpT = getTensor(I->getSrc());
  auto outT = getTensor(I->getDest());
  size_t axis = I->getAxis();
  auto inpElemType = inpT->getElementType();
  auto outElemType = outT->getElementType();
  DISPATCH_ARG_MIN_MAX(fwdArgMin, inpElemType, outElemType, inpT, outT, axis);
}
#undef DISPATCH_ARG_MIN_MAX

//===----------------------------------------------------------------------===//
//                  Tensor allocation operations
//===----------------------------------------------------------------------===//

void BoundInterpreterFunction::fwdAllocActivationInst(
    const AllocActivationInst *I) {
  getOrCreateTensor(I);
}

void BoundInterpreterFunction::fwdDeallocActivationInst(
    const DeallocActivationInst *I) {
  deleteTensor(I->getSrc());
}

//===----------------------------------------------------------------------===//
//                       Debug instructions
//===----------------------------------------------------------------------===//
/// Prints a value of the instruction's operand.
/// In most cases it will be the name of the variable and the value of the
/// tensor.
void BoundInterpreterFunction::fwdDebugPrintInst(const DebugPrintInst *I) {
  auto *V = I->getSrc();
  auto *T = getTensor(V);
  std::string format = I->getFormat();
  std::string filename = I->getFileName();

  if (format == "console") {
    // Dump tensor in console.
    llvm::outs() << I->getName() << ": ";
    V->dump();
    llvm::outs() << "\n";
    dumpImpl(T);
    llvm::outs() << "\n";
  } else if (format == "bin") {
    TensorSerializationOptions opts;
    opts.withType = true;
    glow::dumpTensorToBinaryFile(*T, filename, opts);
  } else if (format == "txt") {
    TensorSerializationOptions opts;
    opts.withType = true;
    glow::dumpTensorToTextFile(*T, filename, opts);
  } else if (format == "rawbin") {
    TensorSerializationOptions opts;
    opts.withType = false;
    glow::dumpTensorToBinaryFile(*T, filename, opts);
  } else if (format == "rawtxt") {
    TensorSerializationOptions opts;
    opts.withType = false;
    glow::dumpTensorToTextFile(*T, filename, opts);
  } else {
    llvm_unreachable("DebugPrint format not supported!");
  }
}

void BoundInterpreterFunction::fwdTraceEventInst(const TraceEventInst *I) {
  auto T = getTensor(I->getData());
  auto IH = T->getHandle<int64_t>();
  size_t index = I->getIndex();
  IH.raw(index) = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count();
}

void BoundInterpreterFunction::fwdInstrumentInst(const InstrumentInst *I) {
  // The instrument instruction is not implemented on the Interpreter backend.
  // We cannot throw error though because the Interpreter can be potentially
  // used when constant folding parts of the graph while compiling for the
  // CPU backend with IR instrumentation.
}

//===----------------------------------------------------------------------===//
//                Instructions used by Quantization
//===----------------------------------------------------------------------===//
void BoundInterpreterFunction::fwdQuantizationProfileInst(
    const glow::QuantizationProfileInst *I) {
  auto inputTensor = getWeightHandle(I->getInputTensor());
  auto currentHistogram = getWeightHandle(I->getHistogram());
  auto computationInfo = getWeightHandle(I->getComputationInfo());

  float &min = computationInfo.raw(0);
  float &max = computationInfo.raw(1);

  // Update current histogram, min and max based on the inputTensor data.
  quantization::generateTensorHistogram(inputTensor, currentHistogram, min,
                                        max);
}

/// Quantize floating point tensor. Scale and Offset are based on return type
/// of the instruction \p I.
void BoundInterpreterFunction::fwdQuantizeInst(const glow::QuantizeInst *I) {
  auto *srcTensor = getTensor(I->getSrc());
  auto *destTensor = getTensor(I->getDest());
  auto destTy = destTensor->getType();
  Tensor qTensor = quantization::quantizeTensor(
      *srcTensor, {destTy.getScale(), destTy.getOffset()},
      destTy.getElementType());
  destTensor->assign(&qTensor);
}

/// Dequantize integer tensor. Scale and Offset are based
/// on the source tensor type.
void BoundInterpreterFunction::fwdDequantizeInst(
    const glow::DequantizeInst *I) {
  auto *srcTensor = getTensor(I->getSrc());
  auto *destTensor = getTensor(I->getDest());
  auto destTy = destTensor->getType();
  Tensor fTensor =
      quantization::dequantizeTensor(*srcTensor, destTy.getElementType());
  destTensor->assign(&fTensor);
}

template <class eTy>
void BoundInterpreterFunction::fwdRescaleQuantizedInstImpl(
    Value *src, Value *dest, TensorQuantizationParams &srcQ,
    TensorQuantizationParams &destQ) {

  auto srcH = getWeightHandle<eTy>(src);
  auto destH = getWeightHandle<eTy>(dest);

  for (size_t i = 0, e = destH.size(); i < e; ++i) {
    float val = quantization::dequantize(srcH.raw(i), srcQ);
    destH.raw(i) = quantization::quantize(val, destQ);
  }
}

void BoundInterpreterFunction::fwdRescaleQuantizedInst(
    const glow::RescaleQuantizedInst *I) {
  auto src = I->getSrc();
  auto dest = I->getDest();
  auto srcTy = src->getType();
  auto destTy = dest->getType();

  TensorQuantizationParams srcQ{srcTy->getScale(), srcTy->getOffset()};
  TensorQuantizationParams destQ{destTy->getScale(), destTy->getOffset()};

  dispatchQuantizedImpl(fwdRescaleQuantizedInstImpl, destTy->getElementType(),
                        src, dest, srcQ, destQ);
}

void BoundInterpreterFunction::fwdIntLookupTableInst(
    const IntLookupTableInst *I) {
  if (I->getSrc()->getElementType() == ElemKind::Int8QTy) {
    auto srcH = getWeightHandle<int8_t>(I->getSrc());
    auto destH = getWeightHandle<int8_t>(I->getDest());
    auto mappingH = getWeightHandle<int8_t>(I->getMapping());
    for (size_t i = 0, e = destH.size(); i < e; i++) {
      destH.raw(i) = mappingH.raw((int)srcH.raw(i) + 128);
    }
  } else if (I->getSrc()->getElementType() == ElemKind::Int16QTy) {
    auto srcH = getWeightHandle<int16_t>(I->getSrc());
    auto destH = getWeightHandle<int16_t>(I->getDest());
    auto mappingH = getWeightHandle<int16_t>(I->getMapping());
    for (size_t i = 0, e = destH.size(); i < e; i++) {
      destH.raw(i) = mappingH.raw((int)srcH.raw(i) + 32768);
    }
  } else {
    llvm_unreachable("Type not supported for IntLookupTable!");
  }
}

void BoundInterpreterFunction::fwdLookupTableInst(const LookupTableInst *I) {
  llvm_unreachable("LookupTable instruction is not supported yet");
}

void BoundInterpreterFunction::fwdConvertToInst(const glow::ConvertToInst *I) {
  Tensor *source = getTensor(I->getInput());
  Tensor *dest = getTensor(I->getResult());
  auto srcElType = source->getType().getElementType();
  auto destElType = dest->getType().getElementType();
  if (srcElType == destElType) {
    // This is a noop conversion.
    dest->copyRawFrom(source);
    return;
  }

#define CONVERT(T_FROM, T_TO, DTY_FROM, DTY_TO)                                \
  if (srcElType == DTY_FROM && destElType == DTY_TO) {                         \
    dest->copyWithCast<T_TO, T_FROM>(source);                                  \
    return;                                                                    \
  }
  CONVERT(float, float16_t, ElemKind::FloatTy, ElemKind::Float16Ty)
  CONVERT(float, bfloat16_t, ElemKind::FloatTy, ElemKind::BFloat16Ty)
  CONVERT(float, bool, ElemKind::FloatTy, ElemKind::BoolTy)
  CONVERT(float, int32_t, ElemKind::FloatTy, ElemKind::Int32ITy)
  CONVERT(float, int64_t, ElemKind::FloatTy, ElemKind::Int64ITy)
  CONVERT(float16_t, float, ElemKind::Float16Ty, ElemKind::FloatTy)
  CONVERT(float16_t, bfloat16_t, ElemKind::Float16Ty, ElemKind::BFloat16Ty)
  CONVERT(float16_t, int32_t, ElemKind::Float16Ty, ElemKind::Int32ITy)
  CONVERT(float16_t, int64_t, ElemKind::Float16Ty, ElemKind::Int64ITy)
  CONVERT(bfloat16_t, float, ElemKind::BFloat16Ty, ElemKind::FloatTy)
  CONVERT(bfloat16_t, float16_t, ElemKind::BFloat16Ty, ElemKind::Float16Ty)
  CONVERT(bfloat16_t, int32_t, ElemKind::BFloat16Ty, ElemKind::Int32ITy)
  CONVERT(bfloat16_t, int64_t, ElemKind::BFloat16Ty, ElemKind::Int64ITy)
  CONVERT(bool, float, ElemKind::BoolTy, ElemKind::FloatTy)
  CONVERT(bool, bfloat16_t, ElemKind::BoolTy, ElemKind::BFloat16Ty)
  CONVERT(int32_t, float, ElemKind::Int32ITy, ElemKind::FloatTy)
  CONVERT(int32_t, float16_t, ElemKind::Int32ITy, ElemKind::Float16Ty)
  CONVERT(int32_t, bfloat16_t, ElemKind::Int32ITy, ElemKind::BFloat16Ty)
  CONVERT(int32_t, int64_t, ElemKind::Int32ITy, ElemKind::Int64ITy)
  CONVERT(int64_t, float, ElemKind::Int64ITy, ElemKind::FloatTy)
  CONVERT(int64_t, float16_t, ElemKind::Int64ITy, ElemKind::Float16Ty)
  CONVERT(int64_t, bfloat16_t, ElemKind::Int64ITy, ElemKind::BFloat16Ty)
  CONVERT(int64_t, int32_t, ElemKind::Int64ITy, ElemKind::Int32ITy)
  CONVERT(bool, int32_t, ElemKind::BoolTy, ElemKind::Int32ITy)
#undef CONVERT

  if (srcElType == ElemKind::UInt8FusedQTy &&
      destElType == ElemKind::UInt8FusedFP16QTy) {
    Tensor result = source->getCopyConvertedToType(ElemKind::UInt8FusedFP16QTy);
    dest->assign(&result);
    return;
  }

  if ((srcElType == ElemKind::UInt8FusedFP16QTy ||
       srcElType == ElemKind::UInt4FusedFP16QTy) &&
      destElType == ElemKind::UInt8FusedQTy) {
    Tensor result = source->getCopyConvertedToType(ElemKind::UInt8FusedQTy);
    dest->assign(&result);
    return;
  }

  if (srcElType == ElemKind::UInt4FusedFP16QTy &&
      destElType == ElemKind::UInt4FusedQTy) {
    Tensor result = source->getCopyConvertedToType(ElemKind::UInt4FusedQTy);
    dest->assign(&result);
    return;
  }

  llvm_unreachable("Type not supported");
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdBatchedPairwiseDotProductInstImpl(
    const BatchedPairwiseDotProductInst *I) {
  auto destT = getTensor(I->getDest());
  auto destH = destT->getHandle<ElemTy>();

  dim_t batchCount = destT->getType().dims()[0];

  // Gather all batched vector operands into an array so that they can be
  // indexed easily.
  std::vector<Value *> srcs;
  for (unsigned i = 1, e = I->getNumOperands(); i < e; ++i) {
    auto op = I->getOperand(i);
    srcs.emplace_back(op.first);
  }

  // pairIdx is the total number of pairs (i, j) that have been processed.
  unsigned pairIdx = 0;

  // For each src operand:
  for (unsigned i = 1, e = I->getNumInputs(); i < e; ++i) {
    auto vAH = getTensor(srcs[i])->getHandle<ElemTy>();
    dim_t vectorSize = getTensor(srcs[i])->getType().dims()[1];

    // Compute the dot product of src[i] with every other vector with a
    // smaller index.
    for (unsigned j = 0; j < i; ++j) {
      auto vBH = getTensor(srcs[j])->getHandle<ElemTy>();

      // Process all batches for a given pair (i, j).
      for (dim_t b = 0; b < batchCount; ++b) {
        ElemTy accum = 0;

        for (dim_t k = 0; k < vectorSize; ++k) {
          accum += vAH.at({b, k}) * vBH.at({b, k});
        }

        destH.at({b, pairIdx}) = accum;
      }

      ++pairIdx;
    }
  }
}

void BoundInterpreterFunction::fwdBatchedPairwiseDotProductInst(
    const BatchedPairwiseDotProductInst *I) {
  dispatchImpl(fwdBatchedPairwiseDotProductInstImpl,
               I->getDest()->getElementType(), I);
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdBatchedPairwiseDotProductGradInstImpl(
    const BatchedPairwiseDotProductGradInst *I) {
  auto destGradT = getTensor(I->getDestGrad());
  auto destGradH = destGradT->getHandle<ElemTy>();

  dim_t batchCount = destGradT->getType().dims()[0];

  // Gather all batched vector operands into arrays so that they can be
  // indexed easily. Operands 1 -> numInputs are gradients of inputs, and
  // operands numInputs + 1 -> numOperands - 1 are the corresponding original
  // inputs.
  std::vector<Value *> srcs, srcGrads;
  for (unsigned i = 0, e = I->getNumInputs(); i < e; ++i) {
    auto gradOp = I->getOperand(i + 1);
    auto inputOp = I->getOperand(i + 1 + e);

    srcGrads.emplace_back(gradOp.first);
    srcs.emplace_back(inputOp.first);
  }

  // Zero initialize all srcGrad tensors.
  for (auto &s : srcGrads) {
    getTensor(s)->zero();
  }

  // pairIdx is the total number of pairs (i, j) that have been processed.
  unsigned pairIdx = 0;

  // For each srcGrad operand:
  for (unsigned i = 0, e = I->getNumInputs(); i < e; ++i) {
    auto dvAH = getTensor(srcGrads[i])->getHandle<ElemTy>();
    dim_t vectorSize = getTensor(srcs[i])->getType().dims()[1];

    // Accmulate into it the product of the gradient of all dot products that
    // src[i] contributed to and the corresponding vectors that src[i] was
    // dotted with.
    for (unsigned j = i + 1; j < e; ++j) {
      auto vBH = getTensor(srcs[j])->getHandle<ElemTy>();

      // Process all batches for a given pair (i, j).
      for (dim_t b = 0; b < batchCount; ++b) {
        ElemTy grad = destGradH.at({b, pairIdx});

        for (dim_t k = 0; k < vectorSize; ++k) {
          dvAH.at({b, k}) += grad * vBH.at({b, k});
        }
      }

      ++pairIdx;
    }
  }
}

void BoundInterpreterFunction::fwdBatchedPairwiseDotProductGradInst(
    const BatchedPairwiseDotProductGradInst *I) {
  dispatchImpl(fwdBatchedPairwiseDotProductGradInstImpl,
               I->getDestGrad()->getElementType(), I);
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdFlipInstImpl(const FlipInst *I) {

  static_assert(max_tensor_dimensions == 6,
                "Loops below assume max_tensor_dimensions = 6.");

  auto *src = I->getSrc();
  auto *dest = I->getDest();

  // Get unowned handles of src and dest with dims expanded to maximum.
  ShapeVector eDims = expandDimsToMax(src->dims());
  auto eSrc = getTensor(src)->getUnowned(eDims);
  auto eDest = getTensor(dest)->getUnowned(eDims);
  auto srcH = eSrc.getHandle<ElemTy>();
  auto destH = eDest.getHandle<ElemTy>();

#define LOOP_AXIS_CASE(_D0, _D1, _D2, _D3, _D4, _D5)                           \
  for (dim_t idx0 = 0; idx0 < eDims[0]; idx0++)                                \
    for (dim_t idx1 = 0; idx1 < eDims[1]; idx1++)                              \
      for (dim_t idx2 = 0; idx2 < eDims[2]; idx2++)                            \
        for (dim_t idx3 = 0; idx3 < eDims[3]; idx3++)                          \
          for (dim_t idx4 = 0; idx4 < eDims[4]; idx4++)                        \
            for (dim_t idx5 = 0; idx5 < eDims[5]; idx5++) {                    \
              destH.at({_D0, _D1, _D2, _D3, _D4, _D5}) =                       \
                  srcH.at({idx0, idx1, idx2, idx3, idx4, idx5});               \
            }                                                                  \
  return;

  switch (I->getAxis()) {
  case 0:
    LOOP_AXIS_CASE(eDims[0] - 1 - idx0, idx1, idx2, idx3, idx4, idx5);
  case 1:
    LOOP_AXIS_CASE(idx0, eDims[1] - 1 - idx1, idx2, idx3, idx4, idx5);
  case 2:
    LOOP_AXIS_CASE(idx0, idx1, eDims[2] - 1 - idx2, idx3, idx4, idx5);
  case 3:
    LOOP_AXIS_CASE(idx0, idx1, idx2, eDims[3] - 1 - idx3, idx4, idx5);
  case 4:
    LOOP_AXIS_CASE(idx0, idx1, idx2, idx3, eDims[4] - 1 - idx4, idx5);
  case 5:
    LOOP_AXIS_CASE(idx0, idx1, idx2, idx3, idx4, eDims[5] - 1 - idx5);
  default:
    llvm_unreachable("Axis should be less than max_tensor_dimensions.");
  }
}

void BoundInterpreterFunction::fwdFlipInst(const FlipInst *I) {
  dispatchImpl(fwdFlipInstImpl, I->getSrc()->getElementType(), I);
}

//===----------------------------------------------------------------------===//
//                Instructions used by ObjectDetection
//===----------------------------------------------------------------------===//
static void maxMin(float lhs, float rhs, float &min, float &max) {
  if (lhs >= rhs) {
    min = rhs;
    max = lhs;
  } else {
    min = lhs;
    max = rhs;
  }
}

using ClassBox = std::pair<float, dim_t>;

struct Box {
  float classValue{0.0f};
  dim_t batchIndex{0};
  dim_t classIndex{0};
  dim_t boxIndex{0};
};

template <typename ElemTy>
static bool doIOU(Handle<ElemTy> &boxes, dim_t batchIndex,
                  dim_t selectedBoxIndex, dim_t candidateBoxIndex,
                  int centerPointBox, float iouThreshold, bool isV4) {
  float sx[] = {0.0f, 0.0f, 0.0f, 0.0f};
  float cx[] = {0.0f, 0.0f, 0.0f, 0.0f};

  if (isV4) {
    for (dim_t i = 0; i < 4; ++i) {
      sx[i] = boxes.at({selectedBoxIndex, i});
      cx[i] = boxes.at({candidateBoxIndex, i});
    }
  } else {
    for (dim_t i = 0; i < 4; ++i) {
      sx[i] = boxes.at({batchIndex, selectedBoxIndex, i});
      cx[i] = boxes.at({batchIndex, candidateBoxIndex, i});
    }
  }

  float xSMin = 0.0f;
  float ySMin = 0.0f;
  float xSMax = 0.0f;
  float ySMax = 0.0f;

  float xCMin = 0.0f;
  float yCMin = 0.0f;
  float xCMax = 0.0f;
  float yCMax = 0.0f;

  // Standardizing coordinates so that (xmin, ymin) is upper left corner of a
  // box and (xmax, ymax) is lower right corner of the box.
  if (!centerPointBox) {
    // 0 means coordinates for diagonal ends of a box.
    // Coordinates can either be absolute or normalized.
    maxMin(sx[0], sx[2], xSMin, xSMax);
    maxMin(sx[1], sx[3], ySMin, ySMax);

    maxMin(cx[0], cx[2], xCMin, xCMax);
    maxMin(cx[1], cx[3], yCMin, yCMax);
  } else {
    float halfWidthS = sx[2] / 2.0f;
    float halfHeightS = sx[3] / 2.0f;
    float halfWidthC = cx[2] / 2.0f;
    float halfHeightC = cx[3] / 2.0f;

    xSMin = sx[0] - halfWidthS;
    ySMin = sx[1] - halfHeightS;
    xSMax = sx[0] + halfWidthS;
    ySMax = sx[1] + halfHeightS;

    xCMin = cx[0] - halfWidthC;
    yCMin = cx[1] - halfHeightC;
    xCMax = cx[0] + halfWidthC;
    yCMax = cx[1] + halfHeightC;
  }

  // finding upper left and lower right corner of a box formed by
  // intersection.
  float xMin = std::max(xSMin, xCMin);
  float yMin = std::max(ySMin, yCMin);
  float xMax = std::min(xSMax, xCMax);
  float yMax = std::min(ySMax, yCMax);

  float intersectionArea =
      std::max(0.0f, xMax - xMin) * std::max(0.0f, yMax - yMin);

  if (intersectionArea == 0.0f) {
    return false;
  }

  float sArea = (xSMax - xSMin) * (ySMax - ySMin);
  float cArea = (xCMax - xCMin) * (yCMax - yCMin);
  float unionArea = sArea + cArea - intersectionArea;

  return intersectionArea > iouThreshold * unionArea;
}

template <typename T>
void BoundInterpreterFunction::fwdNonMaxSuppressionInstImpl(
    glow::NonMaxSuppressionInst const *I) {

  auto boxes = I->getBoxes();
  auto scores = I->getScores();
  auto indices = I->getIndices();
  auto numDetected = I->getNumberOfSelectedIndices();
  float iouThreshold = I->getIouThreshold();
  dim_t maxBoxesPerClass = I->getMaxOutputBoxesPerClass();
  float scoreThreshold = I->getScoreThreshold();
  unsigned centerPointBox = I->getCenterPointBox();
  bool isV4 = I->getIsTFVersion4();

  auto boxesH = getTensor(boxes)->getHandle<float>();
  auto scoresH = getTensor(scores)->getHandle<float>();
  auto indicesH = getTensor(indices)->getHandle<T>();
  auto numDetectedH = getTensor(numDetected)->getHandle<T>();

  int boxesBoxDim = boxes->dims().size() - 2;

  dim_t numBatches = 1;
  dim_t numClasses = 1;
  dim_t numBoxes = boxes->dims()[boxesBoxDim];

  size_t maxOutputPerBatch = 0;

  if (!isV4) {
    int boxesBatchDim = boxes->dims().size() - 3;

    int scoresBatchDim = scores->dims().size() - 3;
    int scoresBoxDim = scores->dims().size() - 1;
    int scoresClassDim = scores->dims().size() - 2;
    assert(scores->dims()[scoresBoxDim] == boxes->dims()[boxesBoxDim] &&
           "Mismatch between number of scores and number of boxes.");
    assert(scores->dims()[scoresBatchDim] == boxes->dims()[boxesBatchDim] &&
           "Mismatch in batch dimension.");
    (void)boxesBatchDim;
    (void)scoresBoxDim;
    numBatches = scores->dims()[scoresBatchDim];
    numClasses = scores->dims()[scoresClassDim];
    numBoxes = boxes->dims()[boxesBoxDim];
    maxOutputPerBatch =
        indices->dims()[indices->dims().size() - 2] / numBatches;
  } else {
    maxOutputPerBatch =
        indices->dims()[indices->dims().size() - 1] / numBatches;
  }

  auto cmpFunc = [](const ClassBox &a, const ClassBox &b) {
    return a.first < b.first;
  };

  std::vector<ClassBox> selectedIndices(numBoxes);
  dim_t outPutBoxIndex = 0;

  for (dim_t batchIndex = 0; batchIndex < numBatches; ++batchIndex) {
    Box minBox{scoresH.raw(batchIndex * numClasses * numBoxes), batchIndex, 0,
               0};
    int32_t detectedPerBatch = 0;
    for (dim_t classIndex = 0; classIndex < numClasses; ++classIndex) {
      selectedIndices.clear();
      size_t detectedPerClass = 0;
      std::priority_queue<ClassBox, std::vector<ClassBox>, decltype(cmpFunc)>
          queue(cmpFunc);

      for (size_t boxIndex = 0; boxIndex < numBoxes; ++boxIndex) {
        float classValue = scoresH.raw(
            (batchIndex * numClasses + classIndex) * numBoxes + boxIndex);
        if (classValue > scoreThreshold) {
          queue.emplace(classValue, boxIndex);
        }
      }

      float tScore = minBox.classValue;
      while (!queue.empty()) {
        auto priorBox = queue.top();
        queue.pop();

        bool selected = true;
        for (auto &sBox : selectedIndices) {
          if (doIOU(boxesH, batchIndex, sBox.second, priorBox.second,
                    centerPointBox, iouThreshold, isV4)) {
            selected = false;
            break;
          }
        }

        if (selected) {
          selectedIndices.emplace_back(priorBox);
          if (isV4) {
            indicesH.at({outPutBoxIndex}) = priorBox.second;
            tScore = scoresH.at({priorBox.second});
          } else {
            indicesH.at({outPutBoxIndex, 0}) = batchIndex;
            indicesH.at({outPutBoxIndex, 1}) = classIndex;
            indicesH.at({outPutBoxIndex, 2}) = priorBox.second;
            tScore = scoresH.at({batchIndex, classIndex, priorBox.second});
          }

          ++outPutBoxIndex;
          ++detectedPerClass;
          ++detectedPerBatch;
        }
        if (maxBoxesPerClass == detectedPerClass) {
          break;
        }
      }

      if (tScore < minBox.classValue) {
        minBox.classValue = tScore;
        minBox.classIndex = classIndex;
        if (isV4) {
          minBox.boxIndex = indicesH.at({outPutBoxIndex - 1});
        } else {
          minBox.boxIndex = indicesH.at({outPutBoxIndex - 1, 2});
        }
      }
    }

    for (size_t i = detectedPerBatch; i < maxOutputPerBatch; ++i) {
      if (isV4) {
        indicesH.at({outPutBoxIndex}) = minBox.boxIndex;
      } else {
        indicesH.at({outPutBoxIndex, 0}) = minBox.batchIndex;
        indicesH.at({outPutBoxIndex, 1}) = minBox.classIndex;
        indicesH.at({outPutBoxIndex, 2}) = minBox.boxIndex;
      }

      ++outPutBoxIndex;
    }
    // For ONNX NMS it's not used, for TF Batch Dimension is 1.
    for (dim_t i = 0; i < maxBoxesPerClass; ++i) {
      numDetectedH.at({batchIndex * maxBoxesPerClass + i}) = detectedPerBatch;
    }
  }
}

void BoundInterpreterFunction::fwdNonMaxSuppressionInst(
    glow::NonMaxSuppressionInst const *I) {
  switch (I->getBoxes()->getElementType()) {
  case ElemKind::FloatTy:
    if (I->getIndices()->getElementType() == ElemKind::Int32ITy) {
      fwdNonMaxSuppressionInstImpl<int32_t>(I);
    } else if (I->getIndices()->getElementType() == ElemKind::Int64ITy) {
      fwdNonMaxSuppressionInstImpl<int64_t>(I);
    } else {
      llvm_unreachable("Output type is not supported.");
    }
    break;
  default:
    llvm_unreachable("Type is not supported.");
    break;
  }
}

//===----------------------------------------------------------------------===//
//                       TensorFlowLite NonMaxSuppression
//===----------------------------------------------------------------------===//
static int32_t partition(int32_t *arr, int32_t low, int32_t high,
                         float *values) {
  float pivot = values[high];
  int32_t i = (low - 1);
  float swap_float;
  int32_t swap_int;

  for (int32_t j = low; j <= high - 1; j++) {
    if (values[j] > pivot) {
      i++;

      swap_float = values[i];
      values[i] = values[j];
      values[j] = swap_float;

      swap_int = arr[i];
      arr[i] = arr[j];
      arr[j] = swap_int;
    }
  }

  swap_float = values[i + 1];
  values[i + 1] = values[high];
  values[high] = swap_float;

  swap_int = arr[i + 1];
  arr[i + 1] = arr[high];
  arr[high] = swap_int;

  return (i + 1);
}

static void partial_sort(int32_t *arr, int32_t i, int32_t j, int32_t k,
                         float *values) {
  int32_t p;
  if (i < j) {
    p = partition(arr, i, j, values);

    partial_sort(arr, i, p - 1, k, values);

    if (p < k - 1)
      partial_sort(arr, p + 1, j, k, values);
  }
}

static void iota(int32_t *first, int32_t *last, int32_t value) {
  while (first != last) {
    *first++ = value;
    value++;
  }
}

static void decreasing_partial_arg_sort(float *values, int32_t num_values,
                                        int32_t num_to_sort, int32_t *indices,
                                        float *aux_values) {
  iota(indices, indices + num_values, 0);

  memcpy(aux_values, values, sizeof(float) * num_values);

  partial_sort(indices, 0, num_values - 1, num_to_sort, aux_values);
}

static void select_detection_above_score_threshold(
    float *scores, int32_t num_scores, float threshold, float *keep_values,
    int32_t *keep_indices, int32_t *num_indices) {
  int32_t idx = 0;
  for (int32_t i = 0; i < num_scores; i++) {
    if (scores[i] >= threshold) {
      keep_indices[idx] = i;
      keep_values[idx] = scores[i];
      idx++;
    }
  }
  *num_indices = idx;
}

/// Compute the IOU (Intersection Over Union) metric between two boxes. Each
/// of box1 and box2 is a vector with 4 floating-point values with the box
/// coordinates in the following format: [ymin, xmin, ymax, xmax].
static float tflite_compute_iou(float *box1, float *box2) {

  // Compute the areas of the two boxes.
  float box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
  float box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1]);

  // If box coordinates are invalid we return 0.
  if (box1Area <= 0 || box2Area <= 0) {
    return 0.0f;
  }

  // Determine the coordinates of the intersection rectangle.
  float iYmin = MAX(box1[0], box2[0]);
  float iXmin = MAX(box1[1], box2[1]);
  float iYmax = MIN(box1[2], box2[2]);
  float iXmax = MIN(box1[3], box2[3]);

  // Compute the area of the intersection rectangle.
  float iArea = MAX(0.0f, iXmax - iXmin) * MAX(0.0f, iYmax - iYmin);

  // Compute the area of the union (reunion) rectangle.
  float uArea = box1Area + box2Area - iArea;

  // Compute the Intersection Over Union metric.
  return iArea / uArea;
}

static void tflite_helper(float *boxesPtr, int32_t num_boxes,
                          float nms_score_threshold, float nms_iou_treshold,
                          float *class_scores, int32_t num_scores,
                          int32_t *selected, int32_t *num_selected,
                          int32_t max_detections, int32_t *keep_indices,
                          float *keep_scores, int32_t *sorted_indices_helper) {

  *num_selected = 0;

  int32_t num_scores_kept;
  select_detection_above_score_threshold(class_scores, num_boxes,
                                         nms_score_threshold, keep_scores,
                                         keep_indices, &num_scores_kept);

  decreasing_partial_arg_sort(keep_scores, num_scores_kept, num_scores_kept,
                              sorted_indices_helper, (float *)selected);

  int32_t num_boxes_kept = num_scores_kept;
  int32_t output_size = MIN(num_boxes_kept, max_detections);

  int32_t num_active_candidate = num_boxes_kept;

  uint8_t *active_box_candidate = (uint8_t *)keep_scores;

  for (int32_t row = 0; row < num_boxes_kept; row++) {
    active_box_candidate[row] = 1;
  }

  for (int32_t i = 0; i < num_boxes_kept; i++) {
    if (num_active_candidate == 0 || *num_selected >= output_size)
      break;
    if (active_box_candidate[i] == 1) {
      selected[*num_selected] = keep_indices[sorted_indices_helper[i]];
      (*num_selected)++;
      active_box_candidate[i] = 0;
      num_active_candidate--;
    } else {
      continue;
    }

    for (int32_t j = i + 1; j < num_boxes_kept; ++j) {
      if (active_box_candidate[j] == 1) {

        float *box1 = boxesPtr + 4 * keep_indices[sorted_indices_helper[i]];
        float *box2 = boxesPtr + 4 * keep_indices[sorted_indices_helper[j]];
        float iou = tflite_compute_iou(box1, box2);

        if (iou > nms_iou_treshold) {
          active_box_candidate[j] = 0;
          num_active_candidate--;
        }
      }
    }
  }
}

static void tflite_detection_post_process_f(
    float *boxes, float *scores, float *anchors, float *detectionBoxes,
    int32_t *detectionClasses, float *detectionScores, int32_t *numDetections,
    int8_t *scratch, int32_t numBoxes, int32_t numTotalClasses,
    int32_t numClasses, int32_t maxDetections, int32_t maxClassesPerDetection,
    int32_t maxDetectionsPerClass, float iouThreshold, float scoreThreshold,
    float xScaleInv, float yScaleInv, float hScaleInv, float wScaleInv,
    bool regularNMS) {

  // Decode the box coordinates in-place using the anchors.
  for (int32_t i = 0; i < numBoxes; i++) {

    float *box = &boxes[i * 4];
    float *anchor = &anchors[i * 4];

    float ycenter = box[0] * yScaleInv * anchor[2] + anchor[0];
    float xcenter = box[1] * xScaleInv * anchor[3] + anchor[1];

    float half_h = 0.5f * expf(box[2] * hScaleInv) * anchor[2];
    float half_w = 0.5f * expf(box[3] * wScaleInv) * anchor[3];

    box[0] = ycenter - half_h;
    box[1] = xcenter - half_w;
    box[2] = ycenter + half_h;
    box[3] = xcenter + half_w;
  }

  int32_t max_categories_per_anchor = maxClassesPerDetection;
  int32_t num_categories_per_anchor =
      MIN(max_categories_per_anchor, numClasses);
  int32_t label_offset = numTotalClasses - numClasses;

  if (regularNMS) {
    int32_t num_detections_per_class = maxDetectionsPerClass;

    float *class_scores = (float *)(scratch);
    scratch += numBoxes * sizeof(float);

    int32_t *box_indices_after_regular_nms = (int32_t *)(scratch);
    scratch += (numBoxes + maxDetections) * sizeof(int32_t);

    float *scores_after_regular_nms = (float *)(scratch);
    scratch += (numBoxes + maxDetections) * sizeof(float);

    int32_t size_of_sorted_indices = 0;

    int32_t *sorted_indices = (int32_t *)(scratch);
    scratch += (numBoxes + maxDetections) * sizeof(int32_t);

    float *sorted_values = (float *)(scratch);
    scratch += MIN(numBoxes, maxDetectionsPerClass) * sizeof(float);

    int32_t *selected = (int32_t *)scratch;
    scratch += numBoxes * sizeof(int32_t);

    int32_t *keep_indices = (int32_t *)(scratch);
    scratch += numBoxes * sizeof(int32_t);

    float *keep_scores = (float *)(scratch);
    scratch += numBoxes * sizeof(float);

    int32_t *sorted_indices_helper = (int32_t *)scratch;
    scratch += numBoxes * sizeof(int32_t);

    for (int32_t col = 0; col < numClasses; col++) {
      for (int32_t row = 0; row < numBoxes; row++) {
        class_scores[row] =
            *(scores + row * numTotalClasses + col + label_offset);
      }

      int32_t num_selected;
      tflite_helper(boxes, numBoxes, scoreThreshold, iouThreshold, class_scores,
                    numBoxes, selected, &num_selected, num_detections_per_class,
                    keep_indices, keep_scores, sorted_indices_helper);

      int32_t output_index = size_of_sorted_indices;
      for (int32_t i = 0; i < num_selected; i++) {
        int32_t selected_index = selected[i];
        box_indices_after_regular_nms[output_index] =
            (selected_index * numTotalClasses + col + label_offset);
        scores_after_regular_nms[output_index] = class_scores[selected_index];
        output_index++;
      }

      int32_t num_indices_to_sort = MIN(output_index, maxDetections);

      decreasing_partial_arg_sort(scores_after_regular_nms, output_index,
                                  num_indices_to_sort, sorted_indices,
                                  keep_scores);

      for (int32_t row = 0; row < num_indices_to_sort; row++) {
        int32_t temp = sorted_indices[row];
        sorted_indices[row] = box_indices_after_regular_nms[temp];
        sorted_values[row] = scores_after_regular_nms[temp];
      }

      for (int32_t row = 0; row < num_indices_to_sort; row++) {
        box_indices_after_regular_nms[row] = sorted_indices[row];
        scores_after_regular_nms[row] = sorted_values[row];
      }

      size_of_sorted_indices = num_indices_to_sort;
    }

    for (int32_t output_box_index = 0;
         output_box_index < size_of_sorted_indices; output_box_index++) {

      int32_t anchor_index =
          box_indices_after_regular_nms[output_box_index] / numTotalClasses;
      int32_t class_index = box_indices_after_regular_nms[output_box_index] -
                            anchor_index * numTotalClasses - label_offset;
      float selected_score = scores_after_regular_nms[output_box_index];
      float *box = boxes + anchor_index * 4;

      *detectionBoxes++ = *box++;
      *detectionBoxes++ = *box++;
      *detectionBoxes++ = *box++;
      *detectionBoxes++ = *box++;
      *detectionClasses++ = class_index;
      *detectionScores++ = selected_score;
    }

    *numDetections = size_of_sorted_indices;
  } else {
    float *max_scores = (float *)scratch;
    scratch += numBoxes * sizeof(float);

    int32_t *sorted_classes_indices = (int32_t *)scratch;
    scratch += numBoxes * MIN(maxDetections, numClasses) * sizeof(int32_t);

    int32_t *selected = (int32_t *)scratch;
    scratch += numBoxes * sizeof(int32_t);

    int32_t *keep_indices = (int32_t *)(scratch);
    scratch += numBoxes * sizeof(int32_t);

    float *keep_scores = (float *)(scratch);
    scratch += numBoxes * sizeof(float);

    int32_t *sorted_indices_helper = (int32_t *)scratch;
    scratch += numBoxes * sizeof(int32_t);

    for (int32_t row = 0; row < numBoxes; row++) {
      float *box_scores = scores + row * numTotalClasses + label_offset;
      int32_t *class_indices =
          sorted_classes_indices + row * num_categories_per_anchor;

      decreasing_partial_arg_sort(box_scores, numClasses,
                                  num_categories_per_anchor, keep_indices,
                                  keep_scores);

      for (int32_t i = 0; i < num_categories_per_anchor; i++) {
        class_indices[i] = keep_indices[i];
      }

      max_scores[row] = box_scores[class_indices[0]];
    }

    int32_t selected_size = 0;
    tflite_helper(boxes, numBoxes, scoreThreshold, iouThreshold, max_scores,
                  numBoxes, selected, &selected_size, maxDetections,
                  keep_indices, keep_scores, sorted_indices_helper);

    int32_t num_detections = 0;
    for (int32_t i = 0; i < selected_size; i++) {

      int32_t selected_index = selected[i];
      float *box = boxes + selected_index * 4;
      float *box_scores =
          scores + selected_index * numTotalClasses + label_offset;
      int32_t *class_indices =
          sorted_classes_indices + selected_index * num_categories_per_anchor;

      for (int32_t col = 0; (col < num_categories_per_anchor) &&
                            (num_detections <= selected_size);
           ++col) {
        *detectionBoxes++ = box[0];
        *detectionBoxes++ = box[1];
        *detectionBoxes++ = box[2];
        *detectionBoxes++ = box[3];
        *detectionClasses++ = class_indices[col];
        *detectionScores++ = box_scores[class_indices[col]];
        num_detections++;
      }
    }

    *numDetections = selected_size;
  }
}

void BoundInterpreterFunction::fwdTFLiteDetectionPostProcessInst(
    glow::TFLiteDetectionPostProcessInst const *I) {
  auto boxes = I->getBoxes();
  auto scores = I->getScores();
  auto anchors = I->getAnchors();
  auto detectionBoxes = I->getDetectionBoxes();
  auto detectionClasses = I->getDetectionClasses();
  auto detectionScores = I->getDetectionScores();
  auto numDetections = I->getNumDetections();
  auto scratch = I->getScratch();

  // Get raw pointers.
  float *boxesPtr = (float *)getTensor(boxes)->getUnsafePtr();
  float *scoresPtr = (float *)getTensor(scores)->getUnsafePtr();
  float *anchorsPtr = (float *)getTensor(anchors)->getUnsafePtr();
  float *detectionBoxesPtr = (float *)getTensor(detectionBoxes)->getUnsafePtr();
  int32_t *detectionClassesPtr =
      (int32_t *)getTensor(detectionClasses)->getUnsafePtr();
  float *detectionScoresPtr =
      (float *)getTensor(detectionScores)->getUnsafePtr();
  int32_t *numDetectionsPtr =
      (int32_t *)getTensor(numDetections)->getUnsafePtr();
  int8_t *scratchPtr = (int8_t *)getTensor(scratch)->getUnsafePtr();

  // Get parameters.
  int32_t numBoxes = boxes->dims()[1];
  int32_t numTotalClasses = scores->dims()[2];
  int32_t numClasses = I->getNumClasses();
  int32_t maxDetections = I->getMaxDetections();
  int32_t maxClassesPerDetection = I->getMaxClassesPerDetection();
  int32_t maxDetectionsPerClass = I->getMaxDetectionsPerClass();
  float iouThreshold = I->getIouThreshold();
  float scoreThreshold = I->getScoreThreshold();
  float xScaleInv = 1.0f / I->getXScale();
  float yScaleInv = 1.0f / I->getYScale();
  float hScaleInv = 1.0f / I->getHScale();
  float wScaleInv = 1.0f / I->getWScale();
  bool regularNMS = I->getRegularNMS();

  // Compute TFLite NMS.
  tflite_detection_post_process_f(
      boxesPtr, scoresPtr, anchorsPtr, detectionBoxesPtr, detectionClassesPtr,
      detectionScoresPtr, numDetectionsPtr, scratchPtr, numBoxes,
      numTotalClasses, numClasses, maxDetections, maxClassesPerDetection,
      maxDetectionsPerClass, iouThreshold, scoreThreshold, xScaleInv, yScaleInv,
      hScaleInv, wScaleInv, regularNMS);
}

void BoundInterpreterFunction::fwdAudioSpectrogramInstFloatImpl(
    glow::AudioSpectrogramInst const *I) {

  auto spectrogram = I->getSpectrogram();
  auto input = I->getInput();
  auto window = I->getWindow();
  int64_t windowSize = I->getWindowSize();
  int64_t windowStride = I->getWindowStride();

  auto spectrogramH = getTensor(spectrogram)->getHandle<float>();
  auto inputH = getTensor(input)->getHandle<float>();
  auto windowH = getTensor(window)->getHandle<float>();

  // Compute window count.
  int64_t inputLength = input->size();
  int64_t windowCount =
      std::floor((inputLength - windowSize) / windowStride) + 1;

  // Compute FFT length (next power of 2) and spectrogram length.
  dim_t fftLen = 1 << (dim_t)std::ceil(std::log2((double)windowSize));
  dim_t specLen = fftLen / 2 + 1;

  // Allocate temporary buffers.
  auto winOut = std::make_unique<float[]>(windowSize);
  auto fftRealOut = std::make_unique<float[]>(specLen);
  auto fftImagOut = std::make_unique<float[]>(specLen);

  // Compute the spectrogram.
  for (dim_t winIdx = 0; int64_t(winIdx) < windowCount; winIdx++) {

    // Windowing.
    for (int64_t n = 0; n < windowSize; n++) {
      winOut[n] = inputH.raw(winIdx * windowStride + n) * windowH.raw(n);
    }

    // Compute spectrum (perform FFT).
    for (dim_t k = 0; k < specLen; k++) {
      fftRealOut[k] = 0;
      fftImagOut[k] = 0;
      for (int n = 0; n < windowSize; n++) {
        fftRealOut[k] +=
            winOut[n] * cos(2.0 * M_PI * (double)(n * k) / (double)(fftLen));
        fftImagOut[k] -=
            winOut[n] * sin(2.0 * M_PI * (double)(n * k) / (double)(fftLen));
      }
    }

    // Compute spectrum magnitude/power.
    if (I->getMagnitudeSquared()) {
      for (dim_t k = 0; k < specLen; k++) {
        spectrogramH.at({winIdx, k}) =
            fftRealOut[k] * fftRealOut[k] + fftImagOut[k] * fftImagOut[k];
      }
    } else {
      for (dim_t k = 0; k < specLen; k++) {
        spectrogramH.at({winIdx, k}) =
            sqrt(fftRealOut[k] * fftRealOut[k] + fftImagOut[k] * fftImagOut[k]);
      }
    }
  }
}

void BoundInterpreterFunction::fwdAudioSpectrogramInst(
    glow::AudioSpectrogramInst const *I) {
  auto inputTy = I->getInput()->getElementType();
  auto spectrogramTy = I->getSpectrogram()->getElementType();
  if ((inputTy == ElemKind::FloatTy) && (spectrogramTy == ElemKind::FloatTy)) {
    fwdAudioSpectrogramInstFloatImpl(I);
  } else {
    llvm_unreachable("Type is not supported.");
  }
}

void BoundInterpreterFunction::fwdMFCCInstFloatImpl(glow::MFCCInst const *I) {

  auto coefficients = I->getCoefficients();
  auto spectrogram = I->getSpectrogram();
  auto melWeights = I->getMelWeights();
  auto melRanges = I->getMelRanges();
  auto dctMat = I->getDctMat();
  int64_t filterBankCount = I->getFilterBankCount();
  int64_t numCoefficients = I->getNumCoefficients();

  auto coefficientsH = getTensor(coefficients)->getHandle<float>();
  auto spectrogramH = getTensor(spectrogram)->getHandle<float>();
  auto melWeightsH = getTensor(melWeights)->getHandle<float>();
  auto melRangesH = getTensor(melRanges)->getHandle<int32_t>();
  auto dctMatH = getTensor(dctMat)->getHandle<float>();

  // Perform MFCC for all the windows.
  auto winNum = spectrogram->dims()[0];
  auto melBuff = std::make_unique<float[]>(filterBankCount);
  for (dim_t winIdx = 0; winIdx < winNum; winIdx++) {

    // Apply Mel filter bank mapping. We use sqrt for the spectrogram since we
    // assume the spectrogram is a power value and not a magnitude.
    dim_t melBinCoeffIdx = 0;
    for (int64_t melIdx = 0; melIdx < filterBankCount; melIdx++) {
      int32_t freqIdxStart = melRangesH.raw(2 * melIdx + 0);
      int32_t freqIdxStop = melRangesH.raw(2 * melIdx + 1);
      float melPwr = 0.0f;
      for (dim_t freqIdx = freqIdxStart; int32_t(freqIdx) <= freqIdxStop;
           freqIdx++) {
        melPwr += std::sqrt(spectrogramH.at({winIdx, freqIdx})) *
                  melWeightsH.raw(melBinCoeffIdx++);
      }
      melBuff[melIdx] = melPwr;
    }

    // Take logarithm in-place (avoid log(0)).
    for (int64_t melIdx = 0; melIdx < filterBankCount; melIdx++) {
      float melPwr = melBuff[melIdx];
      melBuff[melIdx] = (melPwr == 0.0)
                            ? logf(std::numeric_limits<float>::min())
                            : logf(melPwr);
    }

    // Compute DCT transform.
    for (dim_t k = 0; int64_t(k) < numCoefficients; k++) {
      float dctOut = 0.0f;
      for (dim_t n = 0; int64_t(n) < filterBankCount; n++) {
        dctOut += dctMatH.at({k, n}) * melBuff[n];
      }
      coefficientsH.at({winIdx, k}) = dctOut;
    }
  }
}

void BoundInterpreterFunction::fwdMFCCInst(glow::MFCCInst const *I) {
  auto spectrogramTy = I->getSpectrogram()->getElementType();
  auto coefficientsTy = I->getCoefficients()->getElementType();
  if ((spectrogramTy == ElemKind::FloatTy) &&
      (coefficientsTy == ElemKind::FloatTy)) {
    fwdMFCCInstFloatImpl(I);
  } else {
    llvm_unreachable("Type is not supported.");
  }
}

namespace {
/// Positions of the input values to be used for bilinear interpolation for
/// each sample point and the weights to use for each.
template <typename T> struct BinGrid {
  dim_t left;
  dim_t top;
  dim_t right;
  dim_t bottom;
  T leftW;
  T topW;
  T rightW;
  T bottomW;
};
} // namespace

/// Function to calculate the xy coordinates of the resized image (grid)
/// Ref: https://arxiv.org/pdf/1703.06870.pdf and OnnxRuntime implementation
/// \p featureMapHeight and \p featureMapWidth are the dimensions of the
/// input feature map to the operator, \p outputHeight and \p outputWidth are
/// the dimensions of the operator output tensor, \p samplingRatioH and \p
/// samplingRatioW are the number of sampling points to use in each bin in the
/// height and width directions respectively (total sample points is
/// samplingRatioH * samplingRatioW), \p boxHeight and \p boxWidth are the
/// height and width of the RoI box, \p yRef and \p xRef are the adjustment to
/// be made for each sampling point, this is either the top left corer of the
/// box for RoiAlign or a vector to be added to center point after rotation
/// for RoiAlignRotated, \p rotated is true if the op is RoiAlignRotated, \p
/// theta is the rotation angle in the case of RoiAlignRotated and is unused
/// in RoiAlign, \p boxCenterH and \p boxCenterW are the center of the box
/// used for rotation in the case of RoiAlignRotated and unused in the case of
/// RoiAlign. \returns a vector of BinGrids, each one to be used to compute a
/// single sample point value.
template <typename T>
static std::vector<BinGrid<T>> getROIAlignInterpolationCoordinates(
    dim_t featureMapHeight, dim_t featureMapWidth, dim_t outputHeight,
    dim_t outputWidth, dim_t samplingRatioH, dim_t samplingRatioW, T boxHeight,
    T boxWidth, T yRef, T xRef, bool rotated, T theta, T boxCenterH,
    T boxCenterW) {

  T sinTheta = T(0.0f);
  T cosTheta = T(0.0f);
  if (rotated) {
    sinTheta = T(std::sin(float(theta)));
    cosTheta = T(std::cos(float(theta)));
  }

  std::vector<BinGrid<T>> binGrids;

  // height and width of the each bin in the final output
  const T binH = boxHeight / T(outputHeight);
  const T binW = boxWidth / T(outputWidth);
  const T roiBinSizeH = binH / T(samplingRatioH);
  const T roiBinSizeW = binW / T(samplingRatioW);
  for (dim_t oh = 0; oh < outputHeight; oh++) {
    for (dim_t ow = 0; ow < outputWidth; ow++) {
      for (dim_t gh = 0; gh < samplingRatioH; gh++) {
        for (dim_t gw = 0; gw < samplingRatioW; gw++) {
          // x,y coordinates or vector w.r.t input dimensions
          T inY = yRef + (T(oh) * binH) + ((T(gh) + T(0.5f)) * roiBinSizeH);
          T inX = xRef + (T(ow) * binW) + ((T(gw) + T(0.5f)) * roiBinSizeW);

          // If ROI is rotated, rotate by theta around the box center then
          // translate
          if (rotated) {
            T inYY = inY;
            T inXX = inX;
            inY = inYY * cosTheta - inXX * sinTheta + boxCenterH;
            inX = inXX * cosTheta + inYY * sinTheta + boxCenterW;
          }

          // zero pad mal-formed boxes
          if (inY < T(-1) || inY > T(featureMapHeight)) {
            BinGrid<T> bg = BinGrid<T>{0, 0, 0, 0, 0, 0, 0, 0};
            binGrids.push_back(bg);
            continue;
          }
          if (inX < T(-1) || inX > T(featureMapWidth)) {
            BinGrid<T> bg = BinGrid<T>{0, 0, 0, 0, 0, 0, 0, 0};
            binGrids.push_back(bg);
            continue;
          }

          // clip to input dimensions
          T y = std::min(std::max(inY, T(0)), T(featureMapHeight - 1));
          T x = std::min(std::max(inX, T(0)), T(featureMapWidth - 1));

          // calc interpolation parameters
          const dim_t yl = dim_t(std::floor(float(y)));
          const dim_t xl = dim_t(std::floor(float(x)));
          const dim_t yh = std::min(yl + 1, featureMapHeight - 1);
          const dim_t xh = std::min(xl + 1, featureMapWidth - 1);

          BinGrid<T> bg;
          bg.left = xl;
          bg.top = yl;
          bg.right = xh;
          bg.bottom = yh;

          bg.rightW = x - T(xl);
          bg.bottomW = y - T(yl);
          bg.leftW = T(1.0) - bg.rightW;
          bg.topW = T(1.0) - bg.bottomW;

          binGrids.push_back(bg);
        } // end of w
      }   // end of h
    }     // end of W
  }       // end of H

  return binGrids;
}

// Implementation of ROIAlign as described in
// https://arxiv.org/pdf/1703.06870.pdf ROIAlign is similar to crop_and_resize
// + pooling with minor modifications in the crop_and_resize.
template <typename T>
void BoundInterpreterFunction::fwdROIAlignInstFloatImpl(
    glow::ROIAlignInst const *I) {
  auto featureMap = I->getFeatureMap();
  auto boxes = I->getBoxes();
  auto batchIndices = I->getBatchIndices();
  auto result = I->getResult();

  auto boxesH = getTensor(boxes)->getHandle<T>();
  auto featureMapH = getTensor(featureMap)->getHandle<T>();
  auto resultH = getTensor(result)->getHandle<T>();

  const bool rotated = I->getRotated();
  const PoolingMode mode = PoolingMode(I->getMode());
  const bool aligned = I->getAligned();
  const dim_t samplingRatio = I->getSamplingRatio();
  const T spatialScale = I->getSpatialScale();

  const dim_t featureMapHeight = featureMapH.dims()[1];
  const dim_t featureMapWidth = featureMapH.dims()[2];
  const dim_t numBoxes = resultH.dims()[0];
  const dim_t outputHeight = resultH.dims()[1];
  const dim_t outputWidth = resultH.dims()[2];
  const dim_t depth = resultH.dims()[3];

  const T offset = aligned ? T(0.5) : T(0);

  bool useSeparateBatchIndexVector = true;
  dim_t boxesStartCol = 0;
  if (rotated || boxes->dims()[1] == 5) {
    boxesStartCol = 1;
    useSeparateBatchIndexVector = false;
  }

  // Extract batch indices from batchIndices tensor if that is used (only used
  // by ONNX which may provide Int64ITy tensors.)
  std::vector<dim_t> batchIndicesExtracted;
  if (useSeparateBatchIndexVector) {
    Tensor *batchIndicesTensor = getTensor(batchIndices);
    auto batchIndicesElemKind = batchIndicesTensor->getElementType();
    for (dim_t b = 0; b < numBoxes; b++) {
      if (batchIndicesElemKind == ElemKind::Int32ITy) {
        batchIndicesExtracted.push_back(
            batchIndicesTensor->getHandle<int32_t>().at({b}));
      } else {
        batchIndicesExtracted.push_back(
            batchIndicesTensor->getHandle<int64_t>().at({b}));
      }
    }
  }

  for (dim_t b = 0; b < numBoxes; b++) {
    dim_t batchIndex;
    if (useSeparateBatchIndexVector) {
      batchIndex = batchIndicesExtracted[b];
    } else {
      batchIndex = dim_t(float(boxesH.at({b, 0})));
    }

    // Values used to determine sampling points during bilinear interpolation.
    // yRef and xRef have different interpreterations for rotated vs unrotated
    // cases (vector vs coordinates) but are used very similarly.
    T yRef;
    T xRef;
    T boxHeight;
    T boxWidth;

    // Values only used in rotated case.
    T theta = T(0.0);
    T boxCenterH = T(0.0);
    T boxCenterW = T(0.0);

    if (rotated) {
      // Do not round
      boxCenterW = boxesH.at({b, boxesStartCol + 0}) * spatialScale - offset;
      boxCenterH = boxesH.at({b, boxesStartCol + 1}) * spatialScale - offset;
      boxWidth = boxesH.at({b, boxesStartCol + 2}) * spatialScale;
      boxHeight = boxesH.at({b, boxesStartCol + 3}) * spatialScale;
      theta = boxesH.at({b, boxesStartCol + 4}) * T(M_PI) / T(180.0);

      if (aligned) {
        assert(boxWidth >= T(0.0) && boxHeight >= T(0.0) &&
               "ROIs in ROIAlign must not have non-negative size!");
      } else { // backward compatibility
        // Force malformed ROIs to be 1x1
        boxHeight = std::max(boxHeight, T(1.0));
        boxWidth = std::max(boxWidth, T(1.0));
      }

      // These are computed wrt the center of RoI (x, y).
      // Appropriate translation needs to be applied after.
      yRef = (T(-1.0) * boxHeight) / T(2.0);
      xRef = (T(-1.0) * boxWidth) / T(2.0);
    } else {
      llvm::SmallVector<T, 4> box = {
          boxesH.at({b, boxesStartCol + 0}) * spatialScale - offset,
          boxesH.at({b, boxesStartCol + 1}) * spatialScale - offset,
          boxesH.at({b, boxesStartCol + 2}) * spatialScale - offset,
          boxesH.at({b, boxesStartCol + 3}) * spatialScale - offset};

      if (aligned) {
        CHECK_GE(box[3] - box[1], T(0.0)) << "Roi height cannot be negative.";
        CHECK_GE(box[2] - box[0], T(0.0)) << "Roi width cannot be negative.";
      } else {
        // Caffe2 backwards compatibility for mal-formed ROIs:
        // Force ROI size to be at least 1x1.
        box[2] = std::max(box[2], box[0] + T(1.0));
        box[3] = std::max(box[3], box[1] + T(1.0));
      }

      yRef = box[1];
      xRef = box[0];
      boxHeight = (box[3] - box[1]);
      boxWidth = (box[2] - box[0]);
    }

    const dim_t samplingRatioH =
        (samplingRatio > 0) ? samplingRatio
                            : std::ceil(float(boxHeight) / outputHeight);
    const dim_t samplingRatioW = (samplingRatio > 0)
                                     ? samplingRatio
                                     : std::ceil(float(boxWidth) / outputWidth);

    // get the xy coordinates in the resized image(grid)
    std::vector<BinGrid<T>> binGrids = getROIAlignInterpolationCoordinates<T>(
        featureMapHeight, featureMapWidth, outputHeight, outputWidth,
        samplingRatioH, samplingRatioW, boxHeight, boxWidth, yRef, xRef,
        rotated, theta, boxCenterH, boxCenterW);

    uint64_t binCount = 0;
    for (dim_t oh = 0; oh < outputHeight; ++oh) {
      for (dim_t ow = 0; ow < outputWidth; ++ow) {
        for (dim_t d = 0; d < depth; ++d) {
          std::vector<T> values;
          for (dim_t gh = 0; gh < samplingRatioH; ++gh) {
            for (dim_t gw = 0; gw < samplingRatioW; ++gw) {
              BinGrid<T> bg = binGrids[binCount++];
              // The four values of  the i/p image surrounding the point of
              // interest (POI) in the resized image
              const T topLeft =
                  featureMapH.at({batchIndex, bg.top, bg.left, d});
              const T topRight =
                  featureMapH.at({batchIndex, bg.top, bg.right, d});
              const T bottomLeft =
                  featureMapH.at({batchIndex, bg.bottom, bg.left, d});
              const T bottomRight =
                  featureMapH.at({batchIndex, bg.bottom, bg.right, d});

              // bilinear interpolation
              const T value = (topLeft * (bg.topW * bg.leftW)) +
                              (topRight * (bg.topW * bg.rightW)) +
                              (bottomLeft * (bg.bottomW * bg.leftW)) +
                              (bottomRight * (bg.bottomW * bg.rightW));
              // interpolation along vertical line
              values.push_back(value);
            } // end of w
          }   // end of h
              // {Average or Max} pooling
          resultH.at({b, oh, ow, d}) =
              (mode == PoolingMode::AVG)
                  ? std::accumulate(values.begin(), values.end(), T(0.0)) /
                        T(values.size())
                  : *std::max_element(values.begin(), values.end());

          binCount = binCount - (samplingRatioH * samplingRatioW);
        } // end of d
        binCount = binCount + (samplingRatioH * samplingRatioW);
      } // end of W
    }   // end of H
  }     // end of b
}

void BoundInterpreterFunction::fwdROIAlignInst(glow::ROIAlignInst const *I) {
  dispatchFloatingPointImpl(fwdROIAlignInstFloatImpl,
                            I->getFeatureMap()->getElementType(), I);
}

// Forward transform that maps proposal boxes to ground-truth boxes using
//     bounding-box regression deltas.
// boxes: pixel coordinates of the bounding boxes
//     size (M, 4), format [x1; y1; x2; y2], x2 >= x1, y2 >= y1
// deltas: bounding box translations and scales
//     size (M, 4), format [dx; dy; dw; dh]
//     dx, dy: scale-invariant translation of the center of the bounding box
//     dw, dh: log-space scaling of the width and height of the bounding box
// weights: weights [wx, wy, ww, wh] for the deltas
// bboxXformClip: minimum bounding box width and height in log-space after
//     transofmration
// correct_transform_coords: Correct bounding box transform coordates. Set to
//     true to match the detectron code, set to false for backward
//     compatibility
// return: pixel coordinates of the bounding boxes
//     size (M, 4), format [x1; y1; x2; y2]
// see "Rich feature hierarchies for accurate object detection and semantic
//     segmentation" Appendix C for more details
// reference: detectron/lib/utils/boxes.py bbox_transform()
template <typename T>
static void bbox_transform_upright(
    Handle<T> &boxesOut, const Handle<T> &boxes, const Handle<T> &deltas,
    dim_t startRowBoxesOut, dim_t startColBoxesOut, dim_t startRowBoxes,
    dim_t startColBoxes, dim_t startRowDeltas, dim_t startColDeltas, dim_t rows,
    llvm::ArrayRef<float> weights, const T &bboxXformClip, T scaleBeforeInv,
    const bool legacyPlusOne = false) {

  if (boxes.dims()[0] == 0) {
    return;
  }

  std::vector<T> widths(rows), heights(rows), ctrX(rows), ctrY(rows);
  for (dim_t i = 0; i < rows; i++) {
    widths[i] = boxes.at({startRowBoxes + i, startColBoxes + dim_t(2)}) *
                    scaleBeforeInv -
                boxes.at({startRowBoxes + i, startColBoxes}) * scaleBeforeInv +
                T(((legacyPlusOne) ? 1 : 0));
    heights[i] = boxes.at({startRowBoxes + i, startColBoxes + dim_t(3)}) *
                     scaleBeforeInv -
                 boxes.at({startRowBoxes + i, startColBoxes + dim_t(1)}) *
                     scaleBeforeInv +
                 T(((legacyPlusOne) ? 1 : 0));

    ctrX[i] = boxes.at({startRowBoxes + i, startColBoxes}) * scaleBeforeInv +
              T(0.5) * widths[i];
    ctrY[i] = boxes.at({startRowBoxes + i, startColBoxes + dim_t(1)}) *
                  scaleBeforeInv +
              T(0.5) * heights[i];
  }

  std::vector<T> dx(rows), dy(rows), dw(rows), dh(rows);
  for (dim_t i = 0; i < rows; i++) {
    dx[i] = deltas.at({startRowDeltas + i, startColDeltas}) / T(weights[0]);
    dy[i] = deltas.at({startRowDeltas + i, startColDeltas + dim_t(1)}) /
            T(weights[1]);
    dw[i] =
        std::min(deltas.at({startRowDeltas + i, startColDeltas + dim_t(2)}) /
                     T(weights[2]),
                 bboxXformClip);
    dh[i] =
        std::min(deltas.at({startRowDeltas + i, startColDeltas + dim_t(3)}) /
                     T(weights[3]),
                 bboxXformClip);
  }

  std::vector<T> predCtrX(rows), predCtrY(rows), predW(rows), predH(rows);
  for (dim_t i = 0; i < rows; i++) {
    predCtrX[i] = dx[i] * widths[i] + ctrX[i];
    predCtrY[i] = dy[i] * heights[i] + ctrY[i];
    predW[i] = T(std::exp(float(dw[i]))) * widths[i];
    predH[i] = T(std::exp(float(dh[i]))) * heights[i];
  }

  for (dim_t i = 0; i < rows; i++) {
    // x1
    boxesOut.at({startRowBoxesOut + i, startColBoxesOut}) =
        predCtrX[i] - T(0.5) * predW[i];
    // x2
    boxesOut.at({startRowBoxesOut + i, startColBoxesOut + dim_t(1)}) =
        predCtrY[i] - T(0.5) * predH[i];
    // y1
    boxesOut.at({startRowBoxesOut + i, startColBoxesOut + dim_t(2)}) =
        predCtrX[i] + T(0.5) * predW[i] - T(((legacyPlusOne) ? 1 : 0));
    // y2
    boxesOut.at({startRowBoxesOut + i, startColBoxesOut + dim_t(3)}) =
        predCtrY[i] + T(0.5) * predH[i] - T(((legacyPlusOne) ? 1 : 0));
  }
}

// Like bbox_transform_upright, but works on rotated boxes.
// boxes: pixel coordinates of the bounding boxes
//     size (M, 5), format [ctr_x; ctr_y; width; height; angle (in degrees)]
// deltas: bounding box translations and scales
//     size (M, 5), format [dx; dy; dw; dh; da]
//     dx, dy: scale-invariant translation of the center of the bounding box
//     dw, dh: log-space scaling of the width and height of the bounding box
//     da: delta for angle in radians
// return: pixel coordinates of the bounding boxes
//     size (M, 5), format [ctr_x; ctr_y; width; height; angle (in degrees)]
template <typename T>
static void bbox_transform_rotated(
    Handle<T> &boxesOut, const Handle<T> &boxes, const Handle<T> &deltas,
    dim_t startRowBoxesOut, dim_t startColBoxesOut, dim_t startRowBoxes,
    dim_t startColBoxes, dim_t startRowDeltas, dim_t startColDeltas, dim_t rows,
    llvm::ArrayRef<float> weights, const T &bboxXformClip, T scaleBeforeInv,
    const bool angleBoundOn, ssize_t angleBoundLo, ssize_t angleBoundHi) {

  if (boxes.dims()[0] == 0) {
    return;
  }

  const T PI = 3.1415926535897931;

  std::vector<T> dx(rows), dy(rows), dw(rows), dh(rows), da(rows);
  for (dim_t i = 0; i < rows; i++) {
    dx[i] = deltas.at({startRowDeltas + i, startColDeltas}) / T(weights[0]);
    dy[i] = deltas.at({startRowDeltas + i, startColDeltas + dim_t(1)}) /
            T(weights[1]);
    dw[i] =
        std::min(deltas.at({startRowDeltas + i, startColDeltas + dim_t(2)}) /
                     T(weights[2]),
                 bboxXformClip);
    dh[i] =
        std::min(deltas.at({startRowDeltas + i, startColDeltas + dim_t(3)}) /
                     T(weights[3]),
                 bboxXformClip);
    // Convert back to degrees
    da[i] = deltas.at({startRowDeltas + i, startColDeltas + dim_t(4)}) *
            T(180.0) / PI;
  }

  for (dim_t i = 0; i < rows; i++) {
    // new ctr_x
    boxesOut.at({startRowBoxesOut + i, startColBoxesOut}) =
        dx[i] * boxes.at({startRowBoxes + i, startColBoxes + dim_t(2)}) *
            scaleBeforeInv +
        boxes.at({startRowBoxes + i, startColBoxes}) * scaleBeforeInv;
    // new ctr_y
    boxesOut.at({startRowBoxesOut + i, startColBoxesOut + dim_t(1)}) =
        dy[i] * boxes.at({startRowBoxes + i, startColBoxes + dim_t(3)}) *
            scaleBeforeInv +
        boxes.at({startRowBoxes + i, startColBoxes + dim_t(1)}) *
            scaleBeforeInv;
    // new width
    boxesOut.at({startRowBoxesOut + i, startColBoxesOut + dim_t(2)}) =
        T(std::exp(float(dw[i]))) *
        boxes.at({startRowBoxes + i, startColBoxes + dim_t(2)}) *
        scaleBeforeInv;
    // new height
    boxesOut.at({startRowBoxesOut + i, startColBoxesOut + dim_t(3)}) =
        T(std::exp(float(dh[i]))) *
        boxes.at({startRowBoxes + i, startColBoxes + dim_t(3)}) *
        scaleBeforeInv;
    // new angle
    boxesOut.at({startRowBoxesOut + i, startColBoxesOut + dim_t(4)}) =
        da[i] + boxes.at({startRowBoxes + i, startColBoxes + dim_t(4)});
  }

  if (angleBoundOn) {
    const ssize_t period = angleBoundHi - angleBoundLo;

    for (dim_t i = 0; i < rows; i++) {
      if (ssize_t(boxesOut.at({startRowBoxesOut + i,
                               startColBoxesOut + dim_t(4)})) < angleBoundLo) {
        boxesOut.at({startRowBoxesOut + i, startColBoxesOut + dim_t(4)}) +=
            T(period);
      } else if (ssize_t(boxesOut.at(
                     {startRowBoxesOut + i, startColBoxesOut + dim_t(4)})) >
                 angleBoundHi) {
        boxesOut.at({startRowBoxesOut + i, startColBoxesOut + dim_t(4)}) -=
            T(period);
      }
    }
  }
}

// Clip boxes to image boundaries
// boxes: pixel coordinates of bounding box, size (M * 4)
template <typename T>
void clip_boxes_upright(Handle<T> &boxes, dim_t startRowBoxes,
                        dim_t startColBoxes, dim_t rows, int height, int width,
                        T scaleAfter, bool legacyPlusOne = false,
                        std::vector<dim_t> uprightRows = {}) {
  for (dim_t i = 0; i < rows; i++) {
    if (uprightRows.size() == rows && !uprightRows[i]) {
      continue;
    }
    // x1 >= 0 && x1 < width
    boxes.at({startRowBoxes + i, startColBoxes}) =
        scaleAfter *
        std::max(std::min(boxes.at({startRowBoxes + i, startColBoxes}),
                          T(width - int(((legacyPlusOne) ? 1 : 0)))),
                 T(0));
    // y1 >= 0 && y1 < height
    boxes.at({startRowBoxes + i, startColBoxes + dim_t(1)}) =
        scaleAfter * std::max(std::min(boxes.at({startRowBoxes + i,
                                                 startColBoxes + dim_t(1)}),
                                       T(height - ((legacyPlusOne) ? 1 : 0))),
                              T(0));

    // x2 >= 0 && x2 < width
    boxes.at({startRowBoxes + i, startColBoxes + 2}) =
        scaleAfter * std::max(std::min(boxes.at({startRowBoxes + i,
                                                 startColBoxes + dim_t(2)}),
                                       T(width - ((legacyPlusOne) ? 1 : 0))),
                              T(0));
    // y2 >= 0 && y2 < height
    boxes.at({startRowBoxes + i, startColBoxes + 3}) =
        scaleAfter * std::max(std::min(boxes.at({startRowBoxes + i,
                                                 startColBoxes + dim_t(3)}),
                                       T(height - ((legacyPlusOne) ? 1 : 0))),
                              T(0));
  }
}

// Similar to clip_boxes_upright but handles rotated boxes with angle info.
// boxes: size (M, 5), format [ctr_x; ctr_y; width; height; angle (in
// degrees)]
//
// Clipping is only performed for boxes that are almost upright
// (within a given `angle_thresh` tolerance) to maintain backward
// compatibility for non-rotated boxes.
//
// We don't clip rotated boxes due to a couple of reasons:
// (1) There are potentially multiple ways to clip a rotated box to make it
//     fit within the image.
// (2) It's tricky to make the entire rectangular box fit within the image and
//     still be able to not leave out pixels of interest.
// Therefore, we rely on upstream ops like RoIAlignRotated safely handling
// this.
template <typename T>
void clip_boxes_rotated(Handle<T> &boxes, dim_t startRowBoxes,
                        dim_t startColBoxes, dim_t rows, int imH, int imW,
                        T scaleAfter, float angleThresh = 1.0,
                        bool legacyPlusOne = false) {
  std::vector<dim_t> uprightRows(rows, 0);
  for (dim_t i = 0; i < rows; i++) {
    if (std::abs(float(boxes.at(
            {startRowBoxes + i, startColBoxes + dim_t(4)}))) <= angleThresh) {
      const T ctrX = boxes.at({startRowBoxes + i, startColBoxes});
      const T ctrY = boxes.at({startRowBoxes + i, startColBoxes + dim_t(1)});
      const T width = boxes.at({startRowBoxes + i, startColBoxes + dim_t(2)});
      const T height = boxes.at({startRowBoxes + i, startColBoxes + dim_t(3)});
      boxes.at({startRowBoxes + i, startColBoxes}) =
          ctrX - (width - T(((legacyPlusOne) ? 1 : 0))) / T(2.0);
      boxes.at({startRowBoxes + i, startColBoxes + dim_t(1)}) =
          ctrY - (height - T(((legacyPlusOne) ? 1 : 0))) / T(2.0);
      boxes.at({startRowBoxes + i, startColBoxes + dim_t(2)}) =
          ctrX + (width - T(((legacyPlusOne) ? 1 : 0))) / T(2.0);
      boxes.at({startRowBoxes + i, startColBoxes + dim_t(3)}) =
          ctrY + (height - T(((legacyPlusOne) ? 1 : 0))) / T(2.0);
      uprightRows[i] = 1;
    }
  }
  clip_boxes_upright(boxes, startRowBoxes, startColBoxes, rows, imH, imW,
                     /* scaleAfter */ T(1.0), legacyPlusOne, uprightRows);

  for (dim_t i = 0; i < rows; i++) {
    if (uprightRows[i] == 1) {
      const T x1 = boxes.at({startRowBoxes + i, startColBoxes});
      const T y1 = boxes.at({startRowBoxes + i, startColBoxes + dim_t(1)});
      const T x2 = boxes.at({startRowBoxes + i, startColBoxes + dim_t(2)});
      const T y2 = boxes.at({startRowBoxes + i, startColBoxes + dim_t(3)});
      boxes.at({startRowBoxes + i, startColBoxes}) = (x1 + x2) / T(2.0);
      boxes.at({startRowBoxes + i, startColBoxes + dim_t(1)}) =
          (y1 + y2) / T(2.0);
      boxes.at({startRowBoxes + i, startColBoxes + dim_t(2)}) =
          x2 - x1 + T(((legacyPlusOne) ? 1 : 0));
      boxes.at({startRowBoxes + i, startColBoxes + dim_t(3)}) =
          y2 - y1 + T(((legacyPlusOne) ? 1 : 0));
    }

    for (dim_t j = 0; j < 4; j++) {
      boxes.at({startRowBoxes + i, startColBoxes + j}) *= scaleAfter;
    }
  }
}

template <typename T>
void BoundInterpreterFunction::fwdBBoxTransformInstFloatImpl(
    glow::BBoxTransformInst const *I) {
  auto roiIn = I->getRois();
  auto deltaIn = I->getDeltas();
  auto imInfoIn = I->getImInfo();

  auto boxOut = I->getBoxOut();
  auto roiBatchSplits = I->getRoiBatchSplits();

  auto weights = I->getWeights();
  const dim_t boxDim = I->getRotated() ? 5 : 4;
  auto applyScale = I->getApplyScale();
  auto rotated = I->getRotated();
  auto angleBoundOn = I->getAngleBoundOn();
  auto angleBoundLo = I->getAngleBoundLo();
  auto angleBoundHi = I->getAngleBoundHi();
  auto angleThresh = I->getClipAngleThresh();
  auto legacyPlusOne = I->getLegacyPlusOne();
  const dim_t N = roiIn->dims()[0];
  const dim_t numClasses = deltaIn->dims()[1] / boxDim;
  const dim_t batchSize = imInfoIn->dims()[0];

  auto roisH = getTensor(roiIn)->getHandle<T>();
  auto deltasH = getTensor(deltaIn)->getHandle<T>();
  auto roiBatchSplitsH = getTensor(roiBatchSplits)->getHandle<T>();

  // Count the number of RoIs per batch
  std::vector<int> numRoisPerBatch(batchSize, 0);
  if (roiIn->dims()[1] == boxDim) {
    numRoisPerBatch[0] = N;
  } else {
    for (dim_t i = 0; i < N; ++i) {
      const int roiBatchId = roisH.at({i, 0});
      numRoisPerBatch[roiBatchId]++;
    }
  }

  auto imInfoH = getTensor(imInfoIn)->getHandle<T>();
  auto boxOutH = getTensor(boxOut)->getHandle<T>();
  getTensor(boxOut)->zero();

  // Default value for minimum bounding box width and height after bounding
  // box transformation (bbox_transform()) in log-space
  const T bboxXformClip = std::log(1000.0 / 16.0);

  // We assume roiIn and deltaIn over multiple batches are grouped
  // together in increasing order as generated by GenerateProposalsOp
  dim_t offset = 0;
  for (dim_t i = 0; i < batchSize; ++i) {
    const dim_t numRois = numRoisPerBatch[i];
    const T scaleBefore = imInfoH.at({i, 2});
    const T scaleAfter = applyScale ? scaleBefore : T(1.0);
    dim_t imgH = dim_t(float(imInfoH.at({i, 0}) / scaleBefore) + 0.5);
    dim_t imgW = dim_t(float(imInfoH.at({i, 1}) / scaleBefore) + 0.5);

    // Apply for the rectangle starting at (startRowRoi, startColRoi)
    // with height (Rows) of num_rois, and width (Cols) of boxDim.
    dim_t startRowRoi = offset;
    dim_t startColRoi = roiIn->dims()[1] != boxDim ? 1 : 0;
    dim_t rows = numRois;
    T scaleBeforeInv = T(1) / scaleBefore;

    // scale before and after on the fly.
    // Do not apply scale for angle in rotated boxes
    for (dim_t k = 0; k < numClasses; k++) {
      dim_t startRowDelta = offset;
      dim_t startColDelta = k * boxDim;
      if (rotated) {
        bbox_transform_rotated<T>(boxOutH, roisH, deltasH, startRowDelta,
                                  startColDelta, startRowRoi, startColRoi,
                                  startRowDelta, startColDelta, rows, weights,
                                  bboxXformClip, scaleBeforeInv, angleBoundOn,
                                  angleBoundLo, angleBoundHi);
        clip_boxes_rotated<T>(boxOutH, startRowDelta, startColDelta, rows, imgH,
                              imgW, scaleAfter, angleThresh, legacyPlusOne);
      } else {
        bbox_transform_upright<T>(boxOutH, roisH, deltasH, startRowDelta,
                                  startColDelta, startRowRoi, startColRoi,
                                  startRowDelta, startColDelta, rows, weights,
                                  bboxXformClip, scaleBeforeInv, legacyPlusOne);
        clip_boxes_upright<T>(boxOutH, startRowDelta, startColDelta, rows, imgH,
                              imgW, scaleAfter, legacyPlusOne);
      }
    }

    offset += rows;
  }

  for (dim_t i = 0; i < batchSize; i++) {
    roiBatchSplitsH.at({i}) = numRoisPerBatch[i];
  }
}

void BoundInterpreterFunction::fwdBBoxTransformInst(
    glow::BBoxTransformInst const *I) {
  dispatchFloatingPointImpl(fwdBBoxTransformInstFloatImpl,
                            I->getRois()->getElementType(), I);
}

void BoundInterpreterFunction::fwdExternalFunctionCallInst(
    glow::ExternalFunctionCallInst const *) {
  LOG(FATAL) << "ExternalFunctionCallInst is not supported yet";
}
