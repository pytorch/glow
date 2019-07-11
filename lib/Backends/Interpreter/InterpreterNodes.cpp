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

#include "Interpreter.h"

#include "glow/IR/Instrs.h"
#include "glow/Quantization/Base/Base.h"
#include "glow/Quantization/Base/Profile.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>

using namespace glow;

#define dispatchFloatingPointImpl(functionName, elemTy, ...)                   \
  switch (elemTy) {                                                            \
  case ElemKind::FloatTy:                                                      \
    functionName<float>(__VA_ARGS__);                                          \
    break;                                                                     \
  case ElemKind::Float16Ty:                                                    \
    functionName<float16_t>(__VA_ARGS__);                                      \
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

#define dispatchArithmeticImpl(functionName, elemTy, ...)                      \
  switch (elemTy) {                                                            \
  case ElemKind::FloatTy:                                                      \
    functionName<float>(__VA_ARGS__);                                          \
    break;                                                                     \
  case ElemKind::Float16Ty:                                                    \
    functionName<float16_t>(__VA_ARGS__);                                      \
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

#define staticAssertFloatingPointType(ElemTy)                                  \
  static_assert(                                                               \
      std::is_floating_point<ElemTy>::value ||                                 \
          std::is_same<float16_t,                                              \
                       typename std::remove_cv<ElemTy>::type>::value,          \
      "This implementation is for floating-point values only")

#define staticAssertArithmeticType(ElemTy)                                     \
  static_assert(                                                               \
      std::is_arithmetic<ElemTy>::value ||                                     \
          std::is_same<float16_t,                                              \
                       typename std::remove_cv<ElemTy>::type>::value,          \
      "This implementation is for arithmetic values only")

//===----------------------------------------------------------------------===//
//                       Convolution
//===----------------------------------------------------------------------===//

/// This is the floating point implementation of Convolution.
template <typename ElemTy>
void BoundInterpreterFunction::fwdConvolutionInstFloatImpl(
    Value *inV, Value *outV, Value *filterV, Value *biasV,
    llvm::ArrayRef<unsigned_t> kernelSizes, llvm::ArrayRef<unsigned_t> strides,
    llvm::ArrayRef<unsigned_t> pads, size_t group, size_t dilation) {
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
  size_t inCperG = idim.c / group;
  size_t outCperG = odim.c / group;

  PaddingTLBR pdim(pads);

  // For each input in the batch:
  for (size_t n = 0; n < idim.n; n++) {

    // For each group of input channels:
    for (size_t g = 0; g < group; g++) {

      // For each output channel in the group:
      for (size_t d = g * outCperG; d < (g + 1) * outCperG; d++) {

        // For each convolution 'jump' in the input tensor:
        ssize_t x = -ssize_t(pdim.top);
        for (size_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
          ssize_t y = -ssize_t(pdim.left);
          for (size_t ay = 0; ay < odim.w; y += sdim.width, ay++) {

            // For each element in the convolution-filter:
            float sum = 0;
            for (size_t fx = 0; fx < kdim.height; fx++) {
              for (size_t fy = 0; fy < kdim.width; fy++) {
                ssize_t ox = x + fx * dilation;
                ssize_t oy = y + fy * dilation;

                // Ignore index access below zero (this is due to padding).
                if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                    oy >= ssize_t(idim.w)) {
                  continue;
                }
                for (size_t fd = 0; fd < inCperG; fd++) {
                  sum += float(
                      filterW.at({d, fx, fy, fd}) *
                      inW.at({n, (size_t)ox, (size_t)oy, g * inCperG + fd}));
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
/// For bias, we support int32 quantization.
template <typename ElemTy, typename AccumulatorTy>
void BoundInterpreterFunction::fwdConvolutionInstQuantizedImpl(
    Value *inV, Value *outV, Value *filterV, Value *biasV,
    llvm::ArrayRef<unsigned_t> kernelSizes, llvm::ArrayRef<unsigned_t> strides,
    llvm::ArrayRef<unsigned_t> pads, size_t group, size_t dilation) {
  auto inW = getWeightHandle<ElemTy>(inV);
  auto outW = getWeightHandle<ElemTy>(outV);
  auto filterW = getWeightHandle<ElemTy>(filterV);
  auto biasW = getWeightHandle<int32_t>(biasV);

  ShapeNHWC odim(outW.dims());
  ShapeNHWC idim(inW.dims());
  ShapeHW kdim(kernelSizes);
  ShapeHW sdim(strides);

  assert(idim.c % group == 0 && "Input channels must be divisible by group.");
  assert(odim.c % group == 0 && "Output channels must be divisible by group.");
  size_t inCperG = idim.c / group;
  size_t outCperG = odim.c / group;

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
  for (size_t n = 0; n < idim.n; n++) {
    // For each group of input channels:
    for (size_t g = 0; g < group; g++) {

      // For each output channel in the group:
      for (size_t d = g * outCperG; d < (g + 1) * outCperG; d++) {

        // For each convolution 'jump' in the input tensor:
        ssize_t x = -ssize_t(pdim.top);
        for (size_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
          ssize_t y = -ssize_t(pdim.left);
          for (size_t ay = 0; ay < odim.w; y += sdim.width, ay++) {

            // For each element in the convolution-filter:
            AccumulatorTy sum = 0;
            for (size_t fx = 0; fx < kdim.height; fx++) {
              for (size_t fy = 0; fy < kdim.width; fy++) {
                ssize_t ox = x + fx * dilation;
                ssize_t oy = y + fy * dilation;

                // Ignore index access below zero (this is due to padding).
                if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                    oy >= ssize_t(idim.w)) {
                  continue;
                }
                for (size_t fd = 0; fd < inCperG; fd++) {

                  AccumulatorTy F = filterW.at({d, fx, fy, fd});
                  AccumulatorTy I =
                      inW.at({n, (size_t)ox, (size_t)oy, g * inCperG + fd});
                  // We represent the element multiplication with offset as
                  // (value - offset).
                  sum += (F - filterOffset) * (I - inOffset);
                }
              }
            }

            // Scale the bias to match the scale of the matrix multiplication.
            AccumulatorTy B = std::round(float(biasW.at({d}) - biasOffset) *
                                         (biasScale / matMulScale));

            // Add the bias:
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

void BoundInterpreterFunction::fwdConvolutionInst(const ConvolutionInst *I) {
  auto kernelSizes = I->getKernels();
  auto pads = I->getPads();
  auto strides = I->getStrides();
  size_t group = I->getGroup();

  if (I->getSrc()->getType()->isQuantizedType()) {
    dispatchQuantizedWithAccumulationImpl(
        fwdConvolutionInstQuantizedImpl, I->getSrc()->getElementType(),
        I->getSrc(), I->getDest(), I->getFilter(), I->getBias(), kernelSizes,
        strides, pads, group, I->getDilation());
    return;
  }

  dispatchFloatingPointImpl(
      fwdConvolutionInstFloatImpl, I->getSrc()->getElementType(), I->getSrc(),
      I->getDest(), I->getFilter(), I->getBias(), kernelSizes, strides, pads,
      group, I->getDilation());
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
  size_t dilation = I->getDilation();

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
  size_t inCperG = idim.c / group;
  size_t outCperG = odim.c / group;

  // For each input in the batch:
  for (size_t n = 0; n < odim.n; n++) {

    // For each group of input channels:
    for (size_t g = 0; g < group; g++) {

      // Compute the gradient. For each layer in the output tensor:
      for (size_t d = g * outCperG; d < (g + 1) * outCperG; d++) {

        // For each convolution 'jump' in the input tensor:
        ssize_t x = -ssize_t(pdim.top);
        for (size_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
          ssize_t y = -ssize_t(pdim.left);
          for (size_t ay = 0; ay < odim.w; y += sdim.width, ay++) {

            float chainGrad = outG.at({n, ax, ay, d});

            // For each element in the convolution-filter:
            for (size_t fx = 0; fx < kdim.height; fx++) {
              for (size_t fy = 0; fy < kdim.width; fy++) {
                ssize_t ox = x + fx * dilation;
                ssize_t oy = y + fy * dilation;

                // Ignore index access below zero (this is due to padding).
                if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                    oy >= ssize_t(idim.w)) {
                  continue;
                }

                for (size_t fd = 0; fd < inCperG; fd++) {
                  filterG.at({d, fx, fy, fd}) +=
                      inW.at({n, (size_t)ox, (size_t)oy, g * inCperG + fd}) *
                      chainGrad;
                  inG.at({n, (size_t)ox, (size_t)oy, g * inCperG + fd}) +=
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

  ShapeNHWDC odim(outW.dims());
  ShapeNHWDC idim(inW.dims());
  ShapeHWD kdim(kernelSizes);
  ShapeHWD sdim(strides);

  assert(idim.c % group == 0 && "Input channels must be divisible by group.");
  assert(odim.c % group == 0 && "Output channels must be divisible by group.");
  size_t inCperG = idim.c / group;
  size_t outCperG = odim.c / group;

  PaddingTLNBRF pdim(pads);

  // For each input in the batch:
  for (size_t n = 0; n < idim.n; n++) {

    // For each group of input channels:
    for (size_t ig = 0; ig < group; ig++) {

      // For each output channel in the group:
      for (size_t og = ig * outCperG; og < (ig + 1) * outCperG; og++) {

        // For each convolution 'jump' in the input tensor:
        ssize_t x = -ssize_t(pdim.top);
        for (size_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
          ssize_t y = -ssize_t(pdim.left);
          for (size_t ay = 0; ay < odim.w; y += sdim.width, ay++) {
            ssize_t z = -ssize_t(pdim.near);
            for (size_t az = 0; az < odim.d; z += sdim.depth, az++) {

              // For each element in the 3D convolution-filter:
              float sum = 0;
              for (size_t fx = 0; fx < kdim.height; fx++) {
                for (size_t fy = 0; fy < kdim.width; fy++) {
                  for (size_t fz = 0; fz < kdim.depth; fz++) {
                    ssize_t ox = x + fx;
                    ssize_t oy = y + fy;
                    ssize_t oz = z + fz;

                    // Ignore index access below zero (this is due to padding).
                    if (ox < 0 || oy < 0 || oz < 0 || ox >= ssize_t(idim.h) ||
                        oy >= ssize_t(idim.w) || oz >= ssize_t(idim.d)) {
                      continue;
                    }
                    for (size_t fg = 0; fg < inCperG; fg++) {
                      sum += float(filterW.at({og, fx, fy, fz, fg}) *
                                   inW.at({n, (size_t)ox, (size_t)oy,
                                           (size_t)oz, ig * inCperG + fg}));
                    }
                  }
                }
              }

              sum += float(biasW.at({og}));
              outW.at({n, ax, ay, az, og}) = ElemTy(sum);
            } // D
          }   // W
        }     // H
      }       // C
    }         // G
  }           // N
}

/// This is the quantized implementation of Convolution3D.
/// For bias, we support int32 quantization.
template <typename ElemTy, typename AccumulatorTy>
void BoundInterpreterFunction::fwdConvolution3DInstQuantizedImpl(
    Value *inV, Value *outV, Value *filterV, Value *biasV,
    llvm::ArrayRef<unsigned_t> kernelSizes, llvm::ArrayRef<unsigned_t> strides,
    llvm::ArrayRef<unsigned_t> pads, size_t group) {
  auto inW = getWeightHandle<ElemTy>(inV);
  auto outW = getWeightHandle<ElemTy>(outV);
  auto filterW = getWeightHandle<ElemTy>(filterV);
  auto biasW = getWeightHandle<int32_t>(biasV);

  ShapeNHWDC odim(outW.dims());
  ShapeNHWDC idim(inW.dims());
  ShapeHWD kdim(kernelSizes);
  ShapeHWD sdim(strides);

  assert(idim.c % group == 0 && "Input channels must be divisible by group.");
  assert(odim.c % group == 0 && "Output channels must be divisible by group.");
  size_t inCperG = idim.c / group;
  size_t outCperG = odim.c / group;

  PaddingTLNBRF pdim(pads);

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
  for (size_t n = 0; n < idim.n; n++) {

    // For each group of input channels:
    for (size_t ig = 0; ig < group; ig++) {

      // For each output channel in the group:
      for (size_t og = ig * outCperG; og < (ig + 1) * outCperG; og++) {

        // For each convolution 'jump' in the input tensor:
        ssize_t x = -ssize_t(pdim.top);
        for (size_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
          ssize_t y = -ssize_t(pdim.left);
          for (size_t ay = 0; ay < odim.w; y += sdim.width, ay++) {
            ssize_t z = -ssize_t(pdim.near);
            for (size_t az = 0; az < odim.d; z += sdim.depth, az++) {

              // For each element in the convolution-filter:
              AccumulatorTy sum = 0;
              for (size_t fx = 0; fx < kdim.height; fx++) {
                for (size_t fy = 0; fy < kdim.width; fy++) {
                  for (size_t fz = 0; fz < kdim.depth; fz++) {
                    ssize_t ox = x + fx;
                    ssize_t oy = y + fy;
                    ssize_t oz = z + fz;

                    // Ignore index access below zero (this is due to padding).
                    if (ox < 0 || oy < 0 || oz < 0 || ox >= ssize_t(idim.h) ||
                        oy >= ssize_t(idim.w) || oz >= ssize_t(idim.d)) {
                      continue;
                    }
                    for (size_t fg = 0; fg < inCperG; fg++) {

                      AccumulatorTy F = filterW.at({og, fx, fy, fz, fg});
                      AccumulatorTy I = inW.at({n, (size_t)ox, (size_t)oy,
                                                (size_t)oz, ig * inCperG + fg});
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
              outW.at({n, ax, ay, az, og}) =
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
    dispatchQuantizedWithAccumulationImpl(
        fwdConvolution3DInstQuantizedImpl, I->getSrc()->getElementType(),
        I->getSrc(), I->getDest(), I->getFilter(), I->getBias(), kernelSizes,
        strides, pads, group);
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

void BoundInterpreterFunction::fwdChannelwiseQuantizedConvolutionInst(
    const ChannelwiseQuantizedConvolutionInst *I) {
  assert(I->getGroupwise() && "Non-groupwise not supported");

  using AccumulatorTy = int32_t;

  auto inW = getWeightHandle<int8_t>(I->getSrc());
  auto outW = getWeightHandle<int8_t>(I->getDest());
  auto filterW = getWeightHandle<int8_t>(I->getFilter());
  auto biasW = getWeightHandle<float>(I->getBias());
  auto scalesW = getWeightHandle<float>(I->getScales());
  auto offsetsW = getWeightHandle<int32_t>(I->getOffsets());

  llvm::ArrayRef<unsigned_t> kernelSizes = I->getKernels();
  llvm::ArrayRef<unsigned_t> pads = I->getPads();
  llvm::ArrayRef<unsigned_t> strides = I->getStrides();
  size_t group = I->getGroup();

  ShapeNHWC odim(outW.dims());
  ShapeNHWC idim(inW.dims());
  ShapeHW kdim(kernelSizes);
  ShapeHW sdim(strides);

  assert(idim.c % group == 0 && "Input channels must be divisible by group.");
  assert(odim.c % group == 0 && "Output channels must be divisible by group.");
  size_t inCperG = idim.c / group;
  size_t outCperG = odim.c / group;

  PaddingTLBR pdim(pads);

  auto &inTy = inW.getType();
  auto &outTy = outW.getType();

  float inScale = inTy.getScale();
  float outScale = outTy.getScale();

  int32_t inOffset = inTy.getOffset();
  int32_t outOffset = outTy.getOffset();

  // For each input in the batch:
  for (size_t n = 0; n < idim.n; n++) {
    // For each group of input channels:
    for (size_t g = 0; g < group; g++) {

      // get groupwise qparams params
      int32_t filterOffset = offsetsW.at(g);
      float filterScale = scalesW.at(g);
      float matMulScale = inScale * filterScale;

      // For each output channel in the group:
      for (size_t d = g * outCperG; d < (g + 1) * outCperG; d++) {

        // For each convolution 'jump' in the input tensor:
        ssize_t x = -ssize_t(pdim.top);
        for (size_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
          ssize_t y = -ssize_t(pdim.left);
          for (size_t ay = 0; ay < odim.w; y += sdim.width, ay++) {

            // For each element in the convolution-filter:
            AccumulatorTy sum = 0;
            for (size_t fx = 0; fx < kdim.height; fx++) {
              for (size_t fy = 0; fy < kdim.width; fy++) {
                ssize_t ox = x + fx;
                ssize_t oy = y + fy;

                // Ignore index access below zero (this is due to padding).
                if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                    oy >= ssize_t(idim.w)) {
                  continue;
                }
                for (size_t fd = 0; fd < inCperG; fd++) {

                  AccumulatorTy F = filterW.at({d, fx, fy, fd});
                  AccumulatorTy I =
                      inW.at({n, (size_t)ox, (size_t)oy, g * inCperG + fd});
                  // We represent the element multiplication with offset as
                  // (value - offset).
                  sum += (F - filterOffset) * (I - inOffset);
                }
              }
            }

            // Scale the bias to match the scale of the matrix multiplication.
            AccumulatorTy B = std::round(biasW.at({d}) / matMulScale);

            // Add the bias:
            sum += B;

            // Scale the result back to the expected destination scale.
            outW.at({n, ax, ay, d}) = quantization::clip<AccumulatorTy, int8_t>(
                std::round(float(sum) * (matMulScale / outScale) + outOffset));
          } // W
        }   // H
      }     // C
    }       // G
  }         // N
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
  for (size_t n = 0; n < odim.n; n++) {

    // For each layer in the output tensor:
    for (size_t z = 0; z < idim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t x = -ssize_t(pdim.top);
      for (size_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
        ssize_t y = -ssize_t(pdim.left);
        for (size_t ay = 0; ay < odim.w; y += sdim.width, ay++) {

          bool first = true;
          T max_value = 0;
          int64_t argmaxNHWC = 0;

          for (size_t fx = 0; fx < kdim.height; fx++) {
            for (size_t fy = 0; fy < kdim.width; fy++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                  oy >= ssize_t(idim.w)) {
                continue;
              }

              T val = inHandle.at({n, (size_t)ox, (size_t)oy, z});
              if (first || (val >= max_value)) {
                first = false;
                max_value = val;
                if (argmaxW) {
                  argmaxNHWC = &inHandle.at({n, (size_t)ox, (size_t)oy, z}) -
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
  float filterArea = kdim.height * kdim.width;

  auto inW = getWeightHandle<ElemTy>(I->getSrc());
  auto outW = getWeightHandle<ElemTy>(I->getDest());

  // For each input in the batch:
  for (size_t n = 0; n < odim.n; n++) {
    // For each layer in the output tensor:
    for (size_t z = 0; z < idim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t x = -ssize_t(pdim.top);
      for (size_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
        ssize_t y = -ssize_t(pdim.left);
        for (size_t ay = 0; ay < odim.w; y += sdim.width, ay++) {
          float sum = 0;

          for (size_t fx = 0; fx < kdim.height; fx++) {
            for (size_t fy = 0; fy < kdim.width; fy++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                  oy >= ssize_t(idim.w)) {
                continue;
              }

              sum += float(inW.at({n, (size_t)ox, (size_t)oy, z}));
            }
          }
          outW.at({n, ax, ay, z}) = ElemTy(sum / filterArea);
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
  float filterArea = kdim.height * kdim.width;

  auto inW = getWeightHandle<int8_t>(I->getSrc());
  auto outW = getWeightHandle<int8_t>(I->getDest());
  TensorQuantizationParams inQP{I->getSrc()->getType()->getScale(),
                                I->getSrc()->getType()->getOffset()};
  TensorQuantizationParams outQP{I->getDest()->getType()->getScale(),
                                 I->getDest()->getType()->getOffset()};

  // For each input in the batch:
  for (size_t n = 0; n < odim.n; n++) {
    // For each layer in the output tensor:
    for (size_t z = 0; z < idim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t x = -ssize_t(pdim.top);
      for (size_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
        ssize_t y = -ssize_t(pdim.left);
        for (size_t ay = 0; ay < odim.w; y += sdim.width, ay++) {
          int32_t sum = 0;

          for (size_t fx = 0; fx < kdim.height; fx++) {
            for (size_t fy = 0; fy < kdim.width; fy++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                  oy >= ssize_t(idim.w)) {
                continue;
              }

              sum += inW.at({n, (size_t)ox, (size_t)oy, z}) - inQP.offset;
            }
          }
          // Instead of dividing by filterArea, just change scale.
          outW.at({n, ax, ay, z}) = quantization::clip<int32_t, int8_t>(
              std::round(float(sum) * (inQP.scale / outQP.scale / filterArea) +
                         outQP.offset));
        } // W
      }   // H
    }     // C
  }       // N
}

void BoundInterpreterFunction::fwdAvgPoolInst(const AvgPoolInst *I) {
  if (I->getSrc()->getType()->isQuantizedType()) {
    fwdAvgPoolInstI8Impl(I);
    return;
  }

  dispatchFloatingPointImpl(fwdAvgPoolInstFloatImpl,
                            I->getSrc()->getElementType(), I);
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
  for (size_t n = 0; n < odim.n; n++) {

    // Compute the gradient. For each layer in the output tensor:
    for (size_t z = 0; z < odim.c; z++) {

      // For each convolution 'jump' in the input tensor:
      for (size_t ax = 0; ax < odim.h; ax++) {
        for (size_t ay = 0; ay < odim.w; ay++) {
          // Reuse precomputed linear index of max element from argmax.
          float chainGrad = outG.at({n, ax, ay, z});
          inG.raw(argmax.at({n, ax, ay, z})) += chainGrad;
        } // W
      }   // H
    }     // C
  }       // N
}

void BoundInterpreterFunction::fwdAvgPoolGradInst(const AvgPoolGradInst *I) {
  auto inG = getWeightHandle(I->getSrcGrad());
  auto outW = getWeightHandle(I->getDest());
  auto outG = getWeightHandle(I->getDestGrad());

  ShapeNHWC odim(outW.dims());
  ShapeNHWC idim(inG.dims());

  PaddingTLBR pdim(I->getPads());
  ShapeHW kdim(I->getKernels());
  ShapeHW sdim(I->getStrides());

  inG.clear();

  float filterArea = kdim.height * kdim.width;

  // For each input in the batch:
  for (size_t n = 0; n < odim.n; n++) {

    // For each layer in the output tensor:
    for (size_t z = 0; z < odim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t x = -ssize_t(pdim.top);
      for (size_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
        ssize_t y = -ssize_t(pdim.left);
        for (size_t ay = 0; ay < odim.w; y += sdim.width, ay++) {

          float dy = outG.at({n, ax, ay, z}) / filterArea;

          for (size_t fx = 0; fx < kdim.height; fx++) {
            for (size_t fy = 0; fy < kdim.width; fy++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                  oy >= ssize_t(idim.w)) {
                continue;
              }
              inG.at({n, (size_t)ox, (size_t)oy, z}) += dy;
            }
          }
        } // W
      }   // H
    }     // C
  }       // N
}

//===----------------------------------------------------------------------===//
//                       Activation functions
//===----------------------------------------------------------------------===//
template <typename ElemTy>
void BoundInterpreterFunction::fwdSigmoidInstFloatImpl(const SigmoidInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto inW = getWeightHandle<ElemTy>(I->getSrc());
  auto outW = getWeightHandle<ElemTy>(I->getDest());

  for (size_t i = 0, e = outW.size(); i < e; i++) {
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

  for (size_t i = 0, e = inW.size(); i < e; i++) {
    float val = inW.raw(i);
    outW.raw(i) = ElemTy(std::tanh(val));
  }
}

void BoundInterpreterFunction::fwdTanhInst(const TanhInst *I) {
  dispatchFloatingPointImpl(fwdTanhInstFloatImpl, I->getSrc()->getElementType(),
                            I);
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

  for (size_t n = 0; n < idim[0]; n++) {
    // Find Max.
    float max = float(inW.at({n, 0}));
    for (size_t i = 1; i < idim[1]; i++) {
      max = std::max(max, float(inW.at({n, i})));
    }

    // Compute exp.
    float sum = 0;
    for (size_t i = 0; i < idim[1]; i++) {
      float e = std::exp(float(inW.at({n, i})) - max);
      sum += e;
      outW.at({n, i}) = ElemTy(e);
    }

    // Normalize the output.
    for (size_t i = 0; i < idim[1]; i++) {
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
  for (size_t n = 0; n < idim[0]; n++) {
    for (size_t i = 0; i < idim[1]; i++) {
      float delta = (selectedH.at({n, 0}) == (int64_t)i);
      inG.at({n, i}) = outW.at({n, i}) - delta;
    }
  }
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
  for (size_t n = 0; n < dims[0]; ++n) {
    assert(labels.raw(n) >= 0 && "Cannot use negative index.");
    size_t y = labels.raw(n);
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
  for (size_t n = 0; n < dims[0]; ++n) {
    assert(Labels.raw(n) >= 0 && "Cannot use negative index.");
    size_t y = Labels.raw(n);
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

  if (k == ElemKind::Int8QTy) {
    // Quantize the requested floating point splat value into the correct
    // integer representation.
    auto destTy = I->getDest()->getType();
    TensorQuantizationParams destQ{destTy->getScale(), destTy->getOffset()};
    float val = I->getValue();
    return T->getHandle<int8_t>().clear(quantization::quantize(val, destQ));
  }

  if (k == ElemKind::BoolTy) {
    return T->getHandle<bool>().clear(static_cast<bool>(I->getValue()));
  }

  llvm_unreachable("Unsupported tensor type");
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
  TYPED_INSERT(int8_t, ElemKind::Int8QTy);
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
  TYPED_INSERT(int8_t, ElemKind::Int8QTy)
#undef TYPED_INSERT

  llvm_unreachable("Unsupported tensor type");
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdGatherInstImpl(const glow::GatherInst *I) {
  Tensor *dataT = getTensor(I->getData());
  auto &dataTy = dataT->getType();
  Tensor *indicesT = getTensor(I->getIndices());
  Tensor *outT = getTensor(I->getDest());
  unsigned_t batchDims = I->getBatchDims();

  size_t out_p = 0;
  unsigned elementSize = dataTy.getElementSize();
  // The size of the sample in the batch.
  size_t dataSampleSize = dataTy.getSliceSize(batchDims) * elementSize;
  // The size of the slices that we gather.
  size_t dataSliceSize = dataTy.getSliceSize(batchDims + 1) * elementSize;

  // Calculate the size of each sample in the batch.
  size_t numSamples = (dataT->size() * elementSize) / dataSampleSize;

  // Calculate number of samples in the batch.
  size_t batchSize = dataTy.dims()[batchDims];
  (void)batchSize;

  // For each sample in the batch:
  for (size_t sample = 0; sample < numSamples; sample++) {
    size_t sampleStart = sample * dataSampleSize;

    // For each slice (small fragment) that we copy from the source memory:
    for (size_t i = 0, end = indicesT->size(); i < end; i++) {
      size_t slice = indicesT->getHandle<ElemTy>().raw(i);
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
  size_t numExamples = rangesTy.dims()[0];
  size_t exampleSize = rangesTy.dims()[1];

  // Keep track of the total number of elements gathered across all
  // examples for a sanity check later.
  size_t grandTotalLen = 0;

  // For each example in ranges:
  for (size_t example = 0; example < numExamples; ++example) {
    // Keep a running total of the lengths of all ranges in this example
    // to record into lengthsT once the entire example is processed.
    ElemTy totalLen = 0;

    // For each range in the example:
    for (size_t range = 0; range < exampleSize; ++range) {
      // Get the start index and range length.
      ElemTy startIdx = rangesT->getHandle<ElemTy>().at({example, range, 0});
      ElemTy len = rangesT->getHandle<ElemTy>().at({example, range, 1});

      // Add the length of this current range to the example length counter.
      totalLen += len;

      // Compute the start and end offsets.
      size_t startOffset = startIdx * dataElementSize;
      size_t endOffset = startOffset + (len * dataElementSize);

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

void BoundInterpreterFunction::fwdScatterAssignInst(
    const glow::ScatterAssignInst *I) {
  Tensor *dataT = getTensor(I->getData());
  Tensor *indicesT = getTensor(I->getIndices());
  Tensor *slicesT = getTensor(I->getSlices());

  size_t dataSliceSize =
      dataT->size() / dataT->dims()[0] * dataT->getType().getElementSize();

  // For each index, copy from the slice at that index into the location in data
  // given the offset from the indices tensor.
  for (size_t i = 0, end = indicesT->size(); i < end; i++) {
    size_t destDataIdx = indicesT->getHandle<int64_t>().raw(i);
    std::copy(&slicesT->getUnsafePtr()[i * dataSliceSize],
              &slicesT->getUnsafePtr()[(i + 1) * dataSliceSize],
              &dataT->getUnsafePtr()[dataSliceSize * destDataIdx]);
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

  for (size_t batchId = 0; batchId < batchSize; batchId++) {
    size_t offset = 0;
    for (size_t featureId = 0; featureId < featureCnt; featureId++) {
      auto curValue = dataH.at({batchId, featureId});
      auto curLength = lengthsH.at({featureId});
      for (size_t i = offset, e = offset + curLength; i != e; i++) {
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

  size_t inDepth = inT->dims()[3];

  size_t outBatch = outT->dims()[0];
  size_t outHeight = outT->dims()[1];
  size_t outWidth = outT->dims()[2];
  size_t outDepth = outT->dims()[3];

  for (size_t ob = 0; ob < outBatch; ++ob) {
    for (size_t oh = 0; oh < outHeight; ++oh) {
      for (size_t ow = 0; ow < outWidth; ++ow) {
        for (size_t oc = 0; oc < outDepth; ++oc) {
          // Gets the block layer we are on
          size_t blockDepthLayer = oc / inDepth;
          // every multiple of block size we reset to 0 offset
          size_t iw = ow * blockSize + blockDepthLayer % blockSize;
          // every multiple of blockSize we start height traversal + 1
          size_t ih = oh * blockSize + blockDepthLayer / blockSize;
          // at every multiple of inDepth index in to input depths resets to 0
          size_t ic = oc % inDepth;

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
  for (size_t n = 0; n < idim.n; n++) {

    // For every row:
    for (size_t h = 0; h < idim.h; h++) {

      // For every column:
      for (size_t w = 0; w < idim.w; w++) {

        // For every channel:
        for (size_t c = 0; c < idim.c; c++) {
          float squareSum = 0.0;
          for (size_t i = (c >= halfWindowSize ? c - halfWindowSize : 0);
               i <= std::min(c + halfWindowSize, idim.c - 1); i++) {
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
  for (size_t n = 0; n < odim.n; n++) {

    // For every row:
    for (size_t h = 0; h < odim.h; h++) {

      // For every column:
      for (size_t w = 0; w < odim.w; w++) {

        float sum = 0.0;

        // Compute sum for first channel.
        for (size_t c = 0; c <= halfWindowSize && c < odim.c; c++) {
          auto outw = outW.at({n, h, w, c});
          auto scale = scaleCache.at({n, h, w, c});
          auto outg = outG.at({n, h, w, c});
          sum += (outg * (outw / scale));
        }

        // For every channel:
        for (size_t c = 0; c < odim.c; c++) {
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
  for (size_t i = 0, e = outW.size(); i < e; i++) {
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
  case ElemKind::FloatTy: {
    DIV_LOOP(float);
    return;
  }
  case ElemKind::Float16Ty: {
    DIV_LOOP(float16_t);
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
void BoundInterpreterFunction::fwdElementCmpLTEInstFloatImpl(
    const ElementCmpLTEInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto outW = getWeightHandle<bool>(I->getDest());
  auto lhsW = getWeightHandle<ElemTy>(I->getLHS());
  auto rhsW = getWeightHandle<ElemTy>(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = lhsW.raw(i) <= rhsW.raw(i);
  }
}

void BoundInterpreterFunction::fwdElementCmpLTEInst(
    const ElementCmpLTEInst *I) {
  if (getTensor(I->getLHS())->getType().isQuantizedType()) {
    auto lhsTy = I->getLHS()->getType();
    auto rhsTy = I->getRHS()->getType();

    float lhsScale = lhsTy->getScale();
    float rhsScale = rhsTy->getScale();

    int32_t lhsOffset = lhsTy->getOffset();
    int32_t rhsOffset = rhsTy->getOffset();

    auto outW = getWeightHandle<bool>(I->getDest());
    auto lhsW = getWeightHandle<int8_t>(I->getLHS());
    auto rhsW = getWeightHandle<int8_t>(I->getRHS());
    for (size_t i = 0, e = outW.size(); i < e; i++) {
      outW.raw(i) = lhsScale * (lhsW.raw(i) - lhsOffset) <=
                    rhsScale * (rhsW.raw(i) - rhsOffset);
    }
    return;
  }

  dispatchFloatingPointImpl(fwdElementCmpLTEInstFloatImpl,
                            I->getLHS()->getElementType(), I);
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdElementCmpEQInstImpl(
    const ElementCmpEQInst *I) {
  auto outW = getWeightHandle<bool>(I->getDest());
  auto lhsW = getWeightHandle<ElemTy>(I->getLHS());
  auto rhsW = getWeightHandle<ElemTy>(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = lhsW.raw(i) == rhsW.raw(i);
  }
}

void BoundInterpreterFunction::fwdElementCmpEQInst(const ElementCmpEQInst *I) {
  auto *T = getTensor(I->getLHS());

  switch (T->getElementType()) {
  case ElemKind::Int64ITy: {
    fwdElementCmpEQInstImpl<int64_t>(I);
    break;
  }
  default:
    dispatchFloatingPointImpl(fwdElementCmpEQInstImpl, T->getElementType(), I);
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

void BoundInterpreterFunction::fwdElementPowInst(
    const glow::ElementPowInst *I) {
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
  for (size_t x = 0; x < destDim[0]; x++) {
    for (size_t y = 0; y < destDim[1]; y++) {

      // Perform DOT on the row an column.
      AccumulatorTy sum = 0;
      for (size_t i = 0; i < lhsDim[1]; i++) {
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
  for (size_t x = 0; x < destDim[0]; x++) {
    for (size_t y = 0; y < destDim[1]; y++) {

      // Perform DOT on the row an column.
      float sum = 0;
      for (size_t i = 0; i < lhsDim[1]; i++) {
        sum += float(lhs.at({x, i}) * rhs.at({i, y}));
      }
      dest.at({x, y}) = ElemTy(sum);
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

//===----------------------------------------------------------------------===//
//                       Row-wise quantized FC
//===----------------------------------------------------------------------===//
void BoundInterpreterFunction::fwdRowwiseQuantizedFullyConnectedInst(
    const RowwiseQuantizedFullyConnectedInst *I) {
  auto inW = getWeightHandle<int8_t>(I->getSrc());
  auto outW = getWeightHandle<int8_t>(I->getDest());
  auto weightsW = getWeightHandle<int8_t>(I->getWeights());
  auto biasW = getWeightHandle<int32_t>(I->getBias());
  auto scalesW = getWeightHandle<float>(I->getScales());
  auto offsetsW = getWeightHandle<int32_t>(I->getOffsets());
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

  for (size_t i = 0; i < idim.height; i++) {
    for (size_t j = 0; j < odim.width; j++) {
      float matMulScale = scalesW.raw(j) * inScale;
      int32_t sum = 0;
      for (size_t k = 0; k < idim.width; k++) {
        int32_t W = weightsW.at({j, k});
        int32_t A = inW.at({i, k});
        sum += (W - offsetsW.raw(j)) * (A - inOffset);
      }
      int32_t B = std::round(float(biasW.at({j}) - biasOffset) *
                             (biasScale / matMulScale));
      sum += B;
      // Scale the result back to the expected destination scale.
      outW.at({i, j}) = quantization::clip<int32_t, int8_t>(
          std::round(float(sum) * (matMulScale / outScale) + outOffset));
    }
  }
}

//===----------------------------------------------------------------------===//
//                       Batched operations
//===----------------------------------------------------------------------===//
template <class T>
static void fwdBatchedAdd(Tensor *batch, Tensor *slice, Tensor *dest) {
  auto batchH = batch->getHandle<int8_t>();
  auto sliceH = slice->getHandle<T>();
  auto destH = dest->getHandle<int8_t>();

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
  for (size_t n = 0; n < bdim.first; n++) {
    size_t base = batchH.getElementPtr({n});

    // For each element in the slice.
    for (size_t i = 0; i < bdim.second; i++) {
      int32_t batchVal = batchH.raw(base + i);
      int32_t sliceVal = sliceH.raw(i);
      // We increase the size of the integer up to 16 bits for more accurate
      // arithmetic.
      const float largeScale = float(1) / (1 << 15);
      // Scale both sides from 8-bit to 16-bits.
      int32_t B =
          std::round(float(batchVal - batchOffset) * (batchScale / largeScale));
      int32_t S =
          std::round(float(sliceVal - sliceOffset) * (sliceScale / largeScale));
      int32_t R = B + S;
      destH.raw(base + i) = quantization::clip<int32_t, int8_t>(
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
  for (size_t n = 0; n < bdim.first; n++) {
    size_t base = batch.getElementPtr({n});

    // For each element in the slice.
    for (size_t i = 0; i < bdim.second; i++) {
      dest.raw(base + i) = batch.raw(base + i) + slice.raw(i);
    }
  }
}

void BoundInterpreterFunction::fwdBatchedAddInst(
    const glow::BatchedAddInst *I) {
  if (getTensor(I->getBatch())->getType().isQuantizedType()) {
    dispatchQuantizedImpl(fwdBatchedAdd, I->getSlice()->getElementType(),
                          getTensor(I->getBatch()), getTensor(I->getSlice()),
                          getTensor(I->getDest()));
    return;
  }
  dispatchFloatingPointImpl(fwdBatchedAddInstFloatImpl,
                            I->getBatch()->getElementType(), I);
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdBatchedReduceAddInstFloatImpl(
    Value *batch, Value *dest, unsigned_t axis, const ShapeVector &eBatchDims,
    const ShapeVector &eDestDims) {
  staticAssertFloatingPointType(ElemTy);

  // Get unowned handles of the batch and dest with these new expanded dims.
  auto eBatch = getTensor(batch)->getUnowned(eBatchDims);
  auto eDest = getTensor(dest)->getUnowned(eDestDims);
  auto eBatchH = eBatch.getHandle<ElemTy>();
  auto eDestH = eDest.getHandle<ElemTy>();
  eDestH.clear();

  // We can use this loop for all shapes. Use the same indices for both the
  // batch and dest, except for setting the axis index in the dest to 0.
  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              size_t destIndices[] = {x, y, z, w, q, r};
              destIndices[axis] = 0;
              eDestH.at(destIndices) =
                  eDestH.at(destIndices) + eBatchH.at({x, y, z, w, q, r});
            }
          }
        }
      }
    }
  }
}

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

    // For quantization, we must accumulate in the inner-most loop into a local
    // float and then clip the result back into the dest tensor. Here are the
    // max_tensor_dimensions cases for this, to ensure the axis is used as the
    // inner-most loop.
    switch (axis) {
#define LOOP_AXIS_CASE(_D0, _D1, _D2, _D3, _D4, _D5_AXIS)                      \
  case _D5_AXIS:                                                               \
    for (size_t i##_D0 = 0; i##_D0 < eBatchDims[_D0]; i##_D0++)                \
      for (size_t i##_D1 = 0; i##_D1 < eBatchDims[_D1]; i##_D1++)              \
        for (size_t i##_D2 = 0; i##_D2 < eBatchDims[_D2]; i##_D2++)            \
          for (size_t i##_D3 = 0; i##_D3 < eBatchDims[_D3]; i##_D3++)          \
            for (size_t i##_D4 = 0; i##_D4 < eBatchDims[_D4]; i##_D4++) {      \
              float sum = 0.0;                                                 \
              for (size_t i##_D5_AXIS = 0; i##_D5_AXIS < eBatchDims[_D5_AXIS]; \
                   i##_D5_AXIS++) {                                            \
                sum += eBatchH.at({i0, i1, i2, i3, i4, i5}) - batchOffset;     \
              }                                                                \
              size_t i##_D5_AXIS = 0;                                          \
              int32_t res =                                                    \
                  std::round(sum * batchScale / destScale) + destOffset;       \
              eDestH.at({i0, i1, i2, i3, i4, i5}) =                            \
                  quantization::clip<int32_t, int8_t>(res);                    \
            }                                                                  \
    return;

      // Each loop order, with the inner-most dimension/index equal to the axis.
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
  dispatchFloatingPointImpl(fwdBatchedReduceAddInstFloatImpl,
                            batch->getElementType(), batch, dest, axis,
                            eBatchDims, eDestDims);
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
  for (size_t i = 0; i < segments; i++) {
    for (int32_t j = 0, e = LH.raw(i); j < e; j++) {
      for (size_t k = 0; k < sliceSize; k++) {
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

void BoundInterpreterFunction::fwdSparseLengthsSumInstI8Impl(
    const SparseLengthsSumInst *I) {

  auto out = getTensor(I->getDest());
  auto data = getTensor(I->getData());
  auto indices = getTensor(I->getIndices());
  auto lengths = getTensor(I->getLengths());

  out->zero();

  auto IH = indices->getHandle<int64_t>();
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

template <typename ElemTy>
void BoundInterpreterFunction::fwdSparseLengthsSumInstFloatImpl(
    const SparseLengthsSumInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto out = getTensor(I->getDest());
  auto data = getTensor(I->getData());
  auto indices = getTensor(I->getIndices());
  auto lengths = getTensor(I->getLengths());

  out->zero();

  auto IH = indices->getHandle<int64_t>();
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
    return fwdSparseLengthsSumInstI8Impl(I);
  }
  dispatchFloatingPointImpl(fwdSparseLengthsSumInstFloatImpl,
                            I->getData()->getElementType(), I);
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdSparseLengthsWeightedSumInstFloatImpl(
    const SparseLengthsWeightedSumInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto out = getTensor(I->getDest());
  auto data = getTensor(I->getData());
  auto weights = getTensor(I->getWeights());
  auto indices = getTensor(I->getIndices());
  auto lengths = getTensor(I->getLengths());

  out->zero();

  auto IH = indices->getHandle<int64_t>();
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
  auto WH = weights->getHandle<ElemTy>();
  auto OH = out->getHandle<ElemTy>();

  size_t curIdx = 0;
  for (size_t i = 0; i < segments; i++) {
    for (size_t j = 0, e = LH.raw(i); j < e; j++) {
      ElemTy weight = WH.raw(curIdx);
      size_t offsetIn = IH.raw(curIdx++) * lineSize;
      size_t offsetOut = i * lineSize;
      for (size_t k = 0; k < lineSize; k++)
        OH.raw(offsetOut++) += DH.raw(offsetIn++) * weight;
    }
  }
}

void BoundInterpreterFunction::fwdSparseLengthsWeightedSumInstI8Impl(
    const SparseLengthsWeightedSumInst *I) {

  auto out = getTensor(I->getDest());
  auto data = getTensor(I->getData());
  auto weights = getTensor(I->getWeights());
  auto indices = getTensor(I->getIndices());
  auto lengths = getTensor(I->getLengths());

  out->zero();

  auto IH = indices->getHandle<int64_t>();
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
  auto WH = weights->getHandle<int8_t>();
  auto OH = out->getHandle<int8_t>();

  auto TQP = [](Tensor *T) {
    return TensorQuantizationParams{T->getType().getScale(),
                                    T->getType().getOffset()};
  };
  using namespace quantization;

  size_t curIdx = 0;
  for (size_t i = 0; i < segments; i++) {
    std::vector<float> accum(lineSize, 0.0f);
    for (int32_t j = 0; j < LH.raw(i); j++) {
      float weight = dequantize(WH.raw(curIdx), TQP(weights));
      size_t offsetIn = IH.raw(curIdx) * lineSize;
      for (size_t k = 0; k < lineSize; k++) {
        accum[k] += weight * dequantize(DH.raw(offsetIn++), TQP(data));
      }
      curIdx++;
    }
    size_t offsetOut = i * lineSize;
    for (size_t k = 0; k < lineSize; k++) {
      OH.raw(offsetOut++) = quantize(accum[k], TQP(out));
    }
  }
}

void BoundInterpreterFunction::fwdSparseLengthsWeightedSumInst(
    const SparseLengthsWeightedSumInst *I) {
  if (I->getDest()->getType()->isQuantizedType()) {
    return fwdSparseLengthsWeightedSumInstI8Impl(I);
  }
  dispatchFloatingPointImpl(fwdSparseLengthsWeightedSumInstFloatImpl,
                            I->getData()->getElementType(), I);
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

void BoundInterpreterFunction::fwdRowwiseQuantizedSparseLengthsWeightedSumInst(
    const RowwiseQuantizedSparseLengthsWeightedSumInst *I) {
  auto *out = getTensor(I->getDest());
  auto *data = getTensor(I->getData());
  auto *dataScales = getTensor(I->getScales());
  auto *dataOffsets = getTensor(I->getOffsets());
  auto *weights = getTensor(I->getWeights());
  auto *indices = getTensor(I->getIndices());
  auto *lengths = getTensor(I->getLengths());

  out->zero();

  auto IH = indices->getHandle<int64_t>();
  auto LH = lengths->getHandle<int32_t>();

  size_t segments = lengths->dims()[0];
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    totalLength += LH.raw(i);
  }
  assert(totalLength <= indices->dims()[0] &&
         "sum(Lengths) must be equal to len(Indices)");

  size_t lineSize = data->size() / data->dims()[0];

  auto DH = data->getHandle<uint8_t>();
  auto DSH = dataScales->getHandle<float>();
  auto DOH = dataOffsets->getHandle<float>();
  auto WH = weights->getHandle<float>();
  auto OH = out->getHandle<float>();

  size_t curIdx = 0;
  for (size_t i = 0; i < segments; i++) {
    for (size_t j = 0, e = LH.raw(i); j < e; j++) {
      const float weight = WH.raw(curIdx);
      const size_t rowIdx = IH.raw(curIdx++);
      const float scale = DSH.at({rowIdx});
      const float offset = DOH.at({rowIdx});
      size_t offsetIn = rowIdx * lineSize;
      size_t offsetOut = i * lineSize;
      for (size_t k = 0; k < lineSize; k++) {
        float d = quantization::dequantizeWithFloatOffset(DH.raw(offsetIn++),
                                                          scale, offset);
        OH.raw(offsetOut++) += d * weight;
      }
    }
  }
}

void BoundInterpreterFunction::
    fwdFusedRowwiseQuantizedSparseLengthsWeightedSumInst(
        const FusedRowwiseQuantizedSparseLengthsWeightedSumInst *I) {
  auto *out = getTensor(I->getDest());
  auto *data = getTensor(I->getData());
  auto *weights = getTensor(I->getWeights());
  auto *indices = getTensor(I->getIndices());
  auto *lengths = getTensor(I->getLengths());

  out->zero();

  auto IH = indices->getHandle<int64_t>();
  auto LH = lengths->getHandle<int32_t>();

  size_t segments = lengths->dims()[0];
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    totalLength += LH.raw(i);
  }
  assert(totalLength <= indices->dims()[0] &&
         "sum(Lengths) must be equal to len(Indices)");

  const size_t inLineSize = data->size() / data->dims()[0];
  const size_t outLineSize = out->size() / out->dims()[0];

  auto DH = data->getHandle<uint8_t>();
  auto WH = weights->getHandle<float>();
  auto OH = out->getHandle<float>();

  size_t curIdx = 0;
  for (size_t i = 0; i < segments; i++) {
    for (size_t j = 0, e = LH.raw(i); j < e; j++) {
      const float weight = WH.raw(curIdx);
      const size_t rowIdx = IH.raw(curIdx++);
      size_t offsetIn = rowIdx * inLineSize;
      size_t offsetOut = i * outLineSize;
      // Get the scale and offset from the row; go to the current row and offset
      // into it up until the last 8 bytes. Use memcpy to get the values out to
      // avoid alignment issues of accessing 4-byte values.
      const char *currRowScaleOffsetPtr =
          data->getUnsafePtr() + offsetIn + inLineSize - 8;
      float scale;
      float offset;
      memcpy(&scale, currRowScaleOffsetPtr, sizeof(float));
      memcpy(&offset, currRowScaleOffsetPtr + sizeof(float), sizeof(float));
      for (size_t k = 0; k < outLineSize; k++) {
        float d = quantization::dequantizeWithFloatOffset(DH.raw(offsetIn++),
                                                          scale, offset);
        OH.raw(offsetOut++) += d * weight;
      }
    }
  }
}

void BoundInterpreterFunction::fwdLengthsToRangesInst(
    const LengthsToRangesInst *I) {
  auto ranges = getTensor(I->getDest())->getHandle<int32_t>();
  auto lengths = getTensor(I->getLengths())->getHandle<int32_t>();
  int32_t offset = 0;
  for (size_t i = 0; i < lengths.dims()[0]; i++) {
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
  size_t curIdx = 0;
  for (size_t i = 0, e = lengthsH.dims()[0]; i < e; i++) {
    for (int32_t j = 0, f = lengthsH.at({i}); j < f; j++) {
      resultH.at({curIdx++}) = j;
    }
  }
}

template <typename ElemTy>
void BoundInterpreterFunction::fwdSparseToDenseInstFloatImpl(
    const SparseToDenseInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto out = getTensor(I->getDest());
  auto indices = getTensor(I->getIndices());
  auto values = getTensor(I->getValues());

  out->zero();

  auto IH = indices->getHandle<int64_t>();

  size_t numIndices = indices->dims()[0];
  size_t numOutDims = out->dims().size();

  // Convert sparse representation to dense representation by taking
  // slices of output and values and accumulating the value slice into
  // the output slice.

  // Dimensions and offsets for the output and values slices. sliceDims
  // will always be {1, [rest of output dimensions]} since the first dimension
  // is the index in this operation. sliceOffsets will be {indices[j], 0, ...}
  // for the output slice and {j, 0, ...} for the values slice so that the
  // slice at index j gets mapped to index indices[j] in the dense
  // representation.
  ShapeVector sliceDims(out->dims().begin(), out->dims().end());
  ShapeVector sliceOffsets(numOutDims, 0);
  sliceDims[0] = 1;

  for (size_t j = 0; j < numIndices; ++j) {
    // Create values slice with offsets {j, 0, ...}.
    sliceOffsets[0] = j;
    auto VS = values->getUnowned(sliceDims, sliceOffsets);
    auto VSH = VS.getHandle<ElemTy>();

    // Create output slice with offsets {indices[j], 0, ...}.
    sliceOffsets[0] = IH.at({j});
    auto OS = out->getUnowned(sliceDims, sliceOffsets);
    auto OSH = OS.getHandle<ElemTy>();

    // Accumulate values slice into output slice.
    size_t outputSliceSize = OS.size();
    for (size_t k = 0; k < outputSliceSize; ++k) {
      OSH.raw(k) += VSH.raw(k);
    }
  }
}

void BoundInterpreterFunction::fwdSparseToDenseInst(
    const SparseToDenseInst *I) {
  dispatchFloatingPointImpl(fwdSparseToDenseInstFloatImpl,
                            I->getDest()->getElementType(), I);
}

void BoundInterpreterFunction::fwdSparseToDenseMaskInst(
    const SparseToDenseMaskInst *I) {
  auto out = getTensor(I->getDest());
  auto values = getTensor(I->getValues());
  auto defaultValue = getTensor(I->getDefaultValue());

  auto indicesH = getTensor(I->getIndices())->getHandle<int64_t>();
  auto lengthsH = getTensor(I->getLengths())->getHandle<int32_t>();

  const std::vector<int64_t> &mask = I->getMask();
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

//===----------------------------------------------------------------------===//
//                Instructions used by RNN
//===----------------------------------------------------------------------===//
template <typename T>
static void fwdTopK(Tensor *outW, Tensor *indW, Tensor *inW, size_t k) {
  auto values = outW->getHandle<T>();
  auto indices = indW->getHandle<int64_t>();
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

//===----------------------------------------------------------------------===//
//                       Sorting operators
//===----------------------------------------------------------------------===//

void BoundInterpreterFunction::fwdTopKInst(const TopKInst *I) {
  auto outW = getTensor(I->getValues());
  auto indW = getTensor(I->getIndices());
  auto inW = getTensor(I->getInput());
  size_t k = I->getK();

  if (inW->getType().isQuantizedType()) {
    fwdTopK<int8_t>(outW, indW, inW, k);
    return;
  }

  dispatchFloatingPointImpl(fwdTopK, inW->getElementType(), outW, indW, inW, k);
}

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
  llvm::outs() << I->getName() << ": ";
  // Dump the content of a value.
  V->dump();
  llvm::outs() << "\n";
  dumpImpl(getTensor(V));
  llvm::outs() << "\n";
}

void BoundInterpreterFunction::fwdTraceEventInst(const TraceEventInst *I) {
  auto T = getTensor(I->getData());
  auto IH = T->getHandle<int64_t>();
  size_t index = I->getIndex();
  IH.raw(index) = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count();
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
  auto srcH = getWeightHandle<int8_t>(I->getSrc());
  auto destH = getWeightHandle<int8_t>(I->getDest());
  auto mappingH = getWeightHandle<int8_t>(I->getMapping());

  for (size_t i = 0, e = destH.size(); i < e; i++) {
    destH.raw(i) = mappingH.raw((int)srcH.raw(i) + 128);
  }
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
  CONVERT(float, int32_t, ElemKind::FloatTy, ElemKind::Int32ITy)
  CONVERT(float, int64_t, ElemKind::FloatTy, ElemKind::Int64ITy)
  CONVERT(float16_t, float, ElemKind::Float16Ty, ElemKind::FloatTy)
  CONVERT(float16_t, int32_t, ElemKind::Float16Ty, ElemKind::Int32ITy)
  CONVERT(float16_t, int64_t, ElemKind::Float16Ty, ElemKind::Int64ITy)
  CONVERT(int32_t, float, ElemKind::Int32ITy, ElemKind::FloatTy)
  CONVERT(int32_t, float16_t, ElemKind::Int32ITy, ElemKind::Float16Ty)
  CONVERT(int32_t, int64_t, ElemKind::Int32ITy, ElemKind::Int64ITy)
  CONVERT(int64_t, float, ElemKind::Int64ITy, ElemKind::FloatTy)
  CONVERT(int64_t, float16_t, ElemKind::Int64ITy, ElemKind::Float16Ty)
  CONVERT(int64_t, int32_t, ElemKind::Int64ITy, ElemKind::Int32ITy)
#undef CONVERT
  llvm_unreachable("Type not supported");
}
