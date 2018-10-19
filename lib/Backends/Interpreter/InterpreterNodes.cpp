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

using namespace glow;

//===----------------------------------------------------------------------===//
//                       Convolution
//===----------------------------------------------------------------------===//

// This is the floating point implementation of Convolution.
template <typename ElemTy>
void InterpreterFunction::fwdConvolutionInst_FloatImpl(
    Value *inV, Value *outV, Value *filterV, Value *biasV,
    llvm::ArrayRef<unsigned_t> kernelSizes, llvm::ArrayRef<unsigned_t> strides,
    llvm::ArrayRef<unsigned_t> pads, size_t group) {

  static_assert(
      std::is_floating_point<ElemTy>::value ||
          std::is_same<float16_t, typename std::remove_cv<ElemTy>::type>::value,
      "This implementation is for floating-point values only");
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
                ssize_t ox = x + fx;
                ssize_t oy = y + fy;

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

// This is the quantized i8 implementation of Convolution.
void InterpreterFunction::fwdConvolutionInst_I8Impl(
    Value *inV, Value *outV, Value *filterV, Value *biasV,
    llvm::ArrayRef<unsigned_t> kernelSizes, llvm::ArrayRef<unsigned_t> strides,
    llvm::ArrayRef<unsigned_t> pads, size_t group) {
  auto inW = getWeightHandle<int8_t>(inV);
  auto outW = getWeightHandle<int8_t>(outV);
  auto filterW = getWeightHandle<int8_t>(filterV);
  auto biasW = getWeightHandle<int8_t>(biasV);

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
                for (size_t fd = 0; fd < inCperG; fd++) {

                  int32_t F = filterW.at({d, fx, fy, fd});
                  int32_t I =
                      inW.at({n, (size_t)ox, (size_t)oy, g * inCperG + fd});
                  // We represent the element multiplication with offset as
                  // (value - offset).
                  sum += (F - filterOffset) * (I - inOffset);
                }
              }
            }

            // Scale the bias to match the scale of the matrix multiplication.
            int32_t B = std::round(float(biasW.at({d}) - biasOffset) *
                                   (biasScale / matMulScale));

            // Add the bias:
            sum += B;

            // Scale the result back to the expected destination scale.
            outW.at({n, ax, ay, d}) = quantization::clip<int32_t, int8_t>(
                std::round(float(sum) * (matMulScale / outScale) + outOffset));
          } // W
        }   // H
      }     // C
    }       // G
  }         // N
}

void InterpreterFunction::fwdConvolutionInst(const ConvolutionInst *I) {
  auto kernelSizes = I->getKernels();
  auto pads = I->getPads();
  auto strides = I->getStrides();
  size_t group = I->getGroup();

  if (I->getSrc()->getType()->isQuantizedType()) {
    fwdConvolutionInst_I8Impl(I->getSrc(), I->getDest(), I->getFilter(),
                              I->getBias(), kernelSizes, strides, pads, group);
    return;
  }

  if (I->getSrc()->getType()->isType<float16_t>()) {
    fwdConvolutionInst_FloatImpl<float16_t>(I->getSrc(), I->getDest(),
                                            I->getFilter(), I->getBias(),
                                            kernelSizes, strides, pads, group);
    return;
  }
  fwdConvolutionInst_FloatImpl<float>(I->getSrc(), I->getDest(), I->getFilter(),
                                      I->getBias(), kernelSizes, strides, pads,
                                      group);
}

void InterpreterFunction::fwdConvolutionGradInst(const ConvolutionGradInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto inG = getWeightHandle(I->getSrcGrad());
  auto outG = getWeightHandle(I->getDestGrad());

  auto filterW = getWeightHandle(I->getFilter());
  auto filterG = getWeightHandle(I->getFilterGrad());
  auto biasG = getWeightHandle(I->getBiasGrad());

  size_t group = I->getGroup();

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
                ssize_t ox = x + fx;
                ssize_t oy = y + fy;

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

//===----------------------------------------------------------------------===//
//                       Pooling
//===----------------------------------------------------------------------===//
template <class T>
static void fwdMaxPool(Tensor *inW, Tensor *outW, Handle<int64_t> *SXY,
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

  // For each input in the batch:
  for (size_t n = 0; n < odim.n; n++) {

    // For each layer in the output tensor:
    for (size_t z = 0; z < idim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t x = -ssize_t(pdim.top);
      for (size_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
        ssize_t y = -ssize_t(pdim.left);
        for (size_t ay = 0; ay < odim.w; y += sdim.width, ay++) {
          size_t maxX = x;
          size_t maxY = y;

          bool first = true;
          T max_value = 0;

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
                maxX = ox;
                maxY = oy;
              }
            }
          }

          outHandle.at({n, ax, ay, z}) = max_value;

          if (SXY) {
            SXY->at({n, ax, ay, z, 0}) = maxX;
            SXY->at({n, ax, ay, z, 1}) = maxY;
          }
        } // W
      }   // H
    }     // C
  }       // N
}

void InterpreterFunction::fwdMaxPoolInst(const MaxPoolInst *I) {
  auto inW = getTensor(I->getSrc());
  auto outW = getTensor(I->getDest());

  if (inW->getType().isQuantizedType()) {
    fwdMaxPool<int8_t>(inW, outW, nullptr, I->getKernels(), I->getStrides(),
                       I->getPads());
  } else if (inW->getType().getElementType() == ElemKind::FloatTy) {
    fwdMaxPool<float>(inW, outW, nullptr, I->getKernels(), I->getStrides(),
                      I->getPads());
  } else if (inW->getType().getElementType() == ElemKind::Float16Ty) {
    fwdMaxPool<float16_t>(inW, outW, nullptr, I->getKernels(), I->getStrides(),
                          I->getPads());
  } else {
    llvm_unreachable("Type not supported");
  }
}

void InterpreterFunction::fwdMaxPoolWithXYInst(const MaxPoolWithXYInst *I) {
  auto inW = getTensor(I->getSrc());
  auto outW = getTensor(I->getDest());
  auto SXY = getWeightHandle<int64_t>(I->getSrcXY());

  if (inW->getType().isQuantizedType()) {
    fwdMaxPool<int8_t>(inW, outW, &SXY, I->getKernels(), I->getStrides(),
                       I->getPads());
  } else if (inW->getType().getElementType() == ElemKind::FloatTy) {
    fwdMaxPool<float>(inW, outW, &SXY, I->getKernels(), I->getStrides(),
                      I->getPads());
  } else if (inW->getType().getElementType() == ElemKind::Float16Ty) {
    fwdMaxPool<float16_t>(inW, outW, &SXY, I->getKernels(), I->getStrides(),
                          I->getPads());
  } else {
    llvm_unreachable("Type not supported");
  }
}

template <typename ElemTy>
void InterpreterFunction::fwdAvgPoolInst_FloatImpl(const AvgPoolInst *I) {
  static_assert(
      std::is_floating_point<ElemTy>::value ||
          std::is_same<float16_t, typename std::remove_cv<ElemTy>::type>::value,
      "This implementation is for floating-point values only");

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

void InterpreterFunction::fwdAvgPoolInst_I8Impl(const AvgPoolInst *I) {
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

void InterpreterFunction::fwdAvgPoolInst(const AvgPoolInst *I) {
  if (I->getSrc()->getType()->isQuantizedType()) {
    fwdAvgPoolInst_I8Impl(I);
  } else if (I->getSrc()->getType()->isType<float16_t>()) {
    fwdAvgPoolInst_FloatImpl<float16_t>(I);
  } else if (I->getSrc()->getType()->isType<float>()) {
    fwdAvgPoolInst_FloatImpl<float>(I);
  } else {
    llvm_unreachable("Type not supported");
  }
}

void InterpreterFunction::fwdMaxPoolWithXYGradInst(
    const MaxPoolWithXYGradInst *I) {
  auto inG = getWeightHandle(I->getSrcGrad());
  auto outW = getWeightHandle(I->getDest());
  auto outG = getWeightHandle(I->getDestGrad());

  inG.clear();

  ShapeNHWC odim(outW.dims());

  auto SXY = getWeightHandle<int64_t>(I->getSrcXY());

  // For each input in the batch:
  for (size_t n = 0; n < odim.n; n++) {

    // Compute the gradient. For each layer in the output tensor:
    for (size_t z = 0; z < odim.c; z++) {

      // For each convolution 'jump' in the input tensor:
      for (size_t ax = 0; ax < odim.h; ax++) {
        for (size_t ay = 0; ay < odim.w; ay++) {

          float chainGrad = outG.at({n, ax, ay, z});

          size_t maxX = SXY.at({n, ax, ay, z, 0});
          size_t maxY = SXY.at({n, ax, ay, z, 1});

          inG.at({n, maxX, maxY, z}) += chainGrad;
        } // W
      }   // H
    }     // C
  }       // N
}

void InterpreterFunction::fwdAvgPoolGradInst(const AvgPoolGradInst *I) {
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

void InterpreterFunction::fwdSigmoidInst(const SigmoidInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto outW = getWeightHandle(I->getDest());

  for (size_t i = 0, e = outW.size(); i < e; i++) {
    float val = inW.raw(i);
    outW.raw(i) = 1 / (1 + std::exp(-val));
  }
}

void InterpreterFunction::fwdTanhInst(const TanhInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto outW = getWeightHandle(I->getDest());

  for (size_t i = 0, e = inW.size(); i < e; i++) {
    float val = inW.raw(i);
    outW.raw(i) = std::tanh(val);
  }
}

//===----------------------------------------------------------------------===//
//                        Loss Functions (Softmax/regression/...)
//===----------------------------------------------------------------------===//

template <typename ElemTy>
void InterpreterFunction::fwdSoftMaxInst_Impl(const SoftMaxInst *I) {
  static_assert(
      std::is_floating_point<ElemTy>::value ||
          std::is_same<float16_t, typename std::remove_cv<ElemTy>::type>::value,
      "This implementation is for floating-point values only");

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

void InterpreterFunction::fwdSoftMaxInst(const SoftMaxInst *I) {
  switch (I->getSrc()->getElementType()) {
  case ElemKind::FloatTy:
    fwdSoftMaxInst_Impl<float>(I);
    break;
  case ElemKind::Float16Ty:
    fwdSoftMaxInst_Impl<float16_t>(I);
    break;
  default:
    llvm_unreachable("Type is not supported");
  }
}

void InterpreterFunction::fwdSoftMaxGradInst(const SoftMaxGradInst *I) {
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

void InterpreterFunction::fwdCrossEntropyLossInst(
    const CrossEntropyLossInst *I) {
  auto P = getWeightHandle(I->getP());
  auto labels = getWeightHandle<int64_t>(I->getLabels());
  auto CE = getWeightHandle(I->getCE());
  auto dims = P.dims();
  for (size_t n = 0; n < dims[0]; ++n) {
    assert(labels.raw(n) >= 0 && "Cannot use negative index.");
    size_t y = labels.raw(n);
    auto p_n = P.at({n, y});
    CE.at({0}) -= log(p_n);
  }
}

void InterpreterFunction::fwdCrossEntropyLossGradInst(
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

void InterpreterFunction::fwdCopyInst(const CopyInst *I) {
  auto inT = getTensor(I->getSrc());
  auto outT = getTensor(I->getDest());
  outT->copyRawFrom(inT);
}

void InterpreterFunction::fwdTransposeInst(const TransposeInst *I) {
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

void InterpreterFunction::fwdTensorViewInst(const TensorViewInst *I) {
  getOrCreateUnownedTensor(I, I->getSrc(), I->getOffsets());
}

void InterpreterFunction::fwdSplatInst(const glow::SplatInst *I) {
  auto *T = getTensor(I->getDest());
  ElemKind k = T->getElementType();

  if (k == ElemKind::Int64ITy) {
    return T->getHandle<int64_t>().clear(I->getValue());
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

  llvm_unreachable("Unsupported tensor type");
}

void InterpreterFunction::fwdInsertTensorInst(const glow::InsertTensorInst *I) {
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
  TYPED_INSERT(float, ElemKind::FloatTy);
  TYPED_INSERT(int8_t, ElemKind::Int8QTy);
#undef TYPED_INSERT

  llvm_unreachable("Unsupported tensor type");
}

void InterpreterFunction::fwdExtractTensorInst(
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
  TYPED_INSERT(int8_t, ElemKind::Int8QTy)
#undef TYPED_INSERT

  llvm_unreachable("Unsupported tensor type");
}

void InterpreterFunction::fwdGatherInst(const glow::GatherInst *I) {
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
      size_t slice = indicesT->getHandle<int64_t>().raw(i);
      assert(slice < batchSize && "Invalid index seen during Gather operation");
      std::copy(
          &dataT->getUnsafePtr()[sampleStart + dataSliceSize * slice],
          &dataT->getUnsafePtr()[sampleStart + dataSliceSize * (slice + 1)],
          &outT->getUnsafePtr()[out_p]);
      out_p += dataSliceSize;
    }
  }
}

void InterpreterFunction::fwdScatterAssignInst(
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

//===----------------------------------------------------------------------===//
//                      Local Response Normalization
//===----------------------------------------------------------------------===//

void InterpreterFunction::fwdLocalResponseNormalizationInst(
    const glow::LocalResponseNormalizationInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto outW = getWeightHandle(I->getDest());
  auto scaleCache = getWeightHandle(I->getScale());

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
            auto val = inW.at({n, h, w, i});
            squareSum += val * val;
          }

          auto scale = k + normedAlpha * squareSum;

          // This will be used to accelerate the backward pass.
          scaleCache.at({n, h, w, c}) = scale;

          auto normFactor = std::pow(scale, -beta);
          outW.at({n, h, w, c}) = inW.at({n, h, w, c}) * normFactor;
        }
      }
    }
  }
}

void InterpreterFunction::fwdLocalResponseNormalizationGradInst(
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
void InterpreterFunction::fwdElementAddInst_I8Impl(const ElementAddInst *I) {
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
void InterpreterFunction::fwdElementAddInst_FloatImpl(const ElementAddInst *I) {
  static_assert(
      std::is_floating_point<ElemTy>::value ||
          std::is_same<float16_t, typename std::remove_cv<ElemTy>::type>::value,
      "This implementation is for floating-point values only");

  auto outW = getWeightHandle<ElemTy>(I->getDest());
  auto lhsW = getWeightHandle<ElemTy>(I->getLHS());
  auto rhsW = getWeightHandle<ElemTy>(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = lhsW.raw(i) + rhsW.raw(i);
  }
}

void InterpreterFunction::fwdElementAddInst(const ElementAddInst *I) {
  if (getTensor(I->getLHS())->getType().isQuantizedType()) {
    fwdElementAddInst_I8Impl(I);
  } else if (I->getLHS()->getType()->getElementType() == ElemKind::FloatTy) {
    fwdElementAddInst_FloatImpl<float>(I);
  } else if (I->getLHS()->getType()->getElementType() == ElemKind::Float16Ty) {
    fwdElementAddInst_FloatImpl<float16_t>(I);
  } else {
    llvm_unreachable("Type is not supported");
  }
}

void InterpreterFunction::fwdElementSubInst(const ElementSubInst *I) {
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

  auto outW = getWeightHandle(I->getDest());
  auto lhsW = getWeightHandle(I->getLHS());
  auto rhsW = getWeightHandle(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = lhsW.raw(i) - rhsW.raw(i);
  }
}

void InterpreterFunction::fwdElementMulInst(const ElementMulInst *I) {
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

  auto outW = getWeightHandle(I->getDest());
  auto lhsW = getWeightHandle(I->getLHS());
  auto rhsW = getWeightHandle(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = lhsW.raw(i) * rhsW.raw(i);
  }
}

void InterpreterFunction::fwdElementDivInst(const ElementDivInst *I) {
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
  default:
    llvm_unreachable("Unsupported type for Div.");
  }
}

void InterpreterFunction::fwdElementMaxInst_I8Impl(const ElementMaxInst *I) {
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
void InterpreterFunction::fwdElementMaxInst_FloatImpl(const ElementMaxInst *I) {
  static_assert(
      std::is_floating_point<ElemTy>::value ||
          std::is_same<float16_t, typename std::remove_cv<ElemTy>::type>::value,
      "This implementation is for floating-point values only");

  auto outW = getWeightHandle<ElemTy>(I->getDest());
  auto lhsW = getWeightHandle<ElemTy>(I->getLHS());
  auto rhsW = getWeightHandle<ElemTy>(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = std::max(lhsW.raw(i), rhsW.raw(i));
  }
}

void InterpreterFunction::fwdElementMaxInst(const ElementMaxInst *I) {
  if (getTensor(I->getLHS())->getType().isQuantizedType()) {
    fwdElementMaxInst_I8Impl(I);
  } else if (I->getLHS()->getType()->getElementType() == ElemKind::FloatTy) {
    fwdElementMaxInst_FloatImpl<float>(I);
  } else if (I->getLHS()->getType()->getElementType() == ElemKind::Float16Ty) {
    fwdElementMaxInst_FloatImpl<float16_t>(I);
  } else {
    llvm_unreachable("Type is not supported");
  }
}

void InterpreterFunction::fwdElementMinInst(const ElementMinInst *I) {
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

  auto outW = getWeightHandle(I->getDest());
  auto lhsW = getWeightHandle(I->getLHS());
  auto rhsW = getWeightHandle(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = std::min(lhsW.raw(i), rhsW.raw(i));
  }
}

// For both quantized and non-quantized CmpLTE, we set the result to 1.0/0.0.
// In the quantized case, we assume that the scale params are (1.0, 0).
void InterpreterFunction::fwdElementCmpLTEInst(const ElementCmpLTEInst *I) {
  if (getTensor(I->getLHS())->getType().isQuantizedType()) {
    auto lhsTy = I->getLHS()->getType();
    auto rhsTy = I->getRHS()->getType();

    float lhsScale = lhsTy->getScale();
    float rhsScale = rhsTy->getScale();

    int32_t lhsOffset = lhsTy->getOffset();
    int32_t rhsOffset = rhsTy->getOffset();

    auto outW = getWeightHandle<int8_t>(I->getDest());
    auto lhsW = getWeightHandle<int8_t>(I->getLHS());
    auto rhsW = getWeightHandle<int8_t>(I->getRHS());
    for (size_t i = 0, e = outW.size(); i < e; i++) {
      outW.raw(i) = lhsScale * (lhsW.raw(i) - lhsOffset) <=
                            rhsScale * (rhsW.raw(i) - rhsOffset)
                        ? 1.0
                        : 0.0;
    }
    return;
  }

  auto outW = getWeightHandle(I->getDest());
  auto lhsW = getWeightHandle(I->getLHS());
  auto rhsW = getWeightHandle(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = lhsW.raw(i) <= rhsW.raw(i) ? 1.0 : 0.0;
  }
}

template <typename ElemTy>
void InterpreterFunction::fwdElementCmpEQInstImpl(const ElementCmpEQInst *I) {
  auto outW = getWeightHandle<ElemTy>(I->getDest());
  auto lhsW = getWeightHandle<ElemTy>(I->getLHS());
  auto rhsW = getWeightHandle<ElemTy>(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = lhsW.raw(i) == rhsW.raw(i) ? 1 : 0;
  }
}

void InterpreterFunction::fwdElementCmpEQInst(const ElementCmpEQInst *I) {
  auto *T = getTensor(I->getDest());

  switch (T->getElementType()) {
  case ElemKind::FloatTy: {
    fwdElementCmpEQInstImpl<float>(I);
    break;
  }
  case ElemKind::Int64ITy: {
    fwdElementCmpEQInstImpl<int64_t>(I);
    break;
  }
  default:
    llvm_unreachable("fwdElementCmpEQInst not implemented for this type!");
  }
}

void InterpreterFunction::fwdElementPowInst(const glow::ElementPowInst *I) {
  auto baseW = getWeightHandle(I->getLHS());
  auto expW = getWeightHandle(I->getRHS());
  auto outW = getWeightHandle(I->getDest());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = pow(baseW.raw(i), expW.raw(i));
  }
}

void InterpreterFunction::fwdElementIsNaNInst(const glow::ElementIsNaNInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto outW = getWeightHandle(I->getDest());
  for (size_t i = 0, e = inW.size(); i < e; i++) {
    float val = inW.raw(i);
    outW.raw(i) = std::isnan(val);
  }
}

void InterpreterFunction::fwdElementLogInst(const ElementLogInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto outW = getWeightHandle(I->getDest());
  for (size_t i = 0, e = inW.size(); i < e; i++) {
    float val = inW.raw(i);
    outW.raw(i) = log(val);
  }
}

void InterpreterFunction::fwdElementSelectInst(
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
    auto condW = getWeightHandle<int8_t>(I->getCond());
    auto lhsW = getWeightHandle<int8_t>(I->getLHS());
    auto rhsW = getWeightHandle<int8_t>(I->getRHS());
    for (size_t i = 0, e = outW.size(); i < e; i++) {
      float val = (condW.raw(i) != 0) ? lhsScale * (lhsW.raw(i) - lhsOffset)
                                      : rhsScale * (rhsW.raw(i) - rhsOffset);
      int32_t q = std::round(val / destScale + destOffset);
      outW.raw(i) = quantization::clip<int32_t, int8_t>(q);
    }
    return;
  }

  auto outW = getWeightHandle(I->getDest());
  auto condW = getWeightHandle(I->getCond());
  auto lhsW = getWeightHandle(I->getLHS());
  auto rhsW = getWeightHandle(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = (condW.raw(i) != 0.0) ? lhsW.raw(i) : rhsW.raw(i);
  }
}

//===----------------------------------------------------------------------===//
//                       Mat Mul
//===----------------------------------------------------------------------===//

void InterpreterFunction::fwdMatMulInst_I8Impl(const glow::MatMulInst *I) {
  assert(getTensor(I->getLHS())->getType().isQuantizedType());
  auto lhs = getWeightHandle<int8_t>(I->getLHS());
  auto rhs = getWeightHandle<int8_t>(I->getRHS());

  auto dest = getWeightHandle<int8_t>(I->getDest());

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
      int32_t sum = 0;
      for (size_t i = 0; i < lhsDim[1]; i++) {
        int32_t L = lhs.at({x, i});
        int32_t R = rhs.at({i, y});
        // We represent the element multiplication with offset as
        // (value - offset).
        sum += (L - lhsOffset) * (R - rhsOffset);
      }

      dest.at({x, y}) = quantization::clip<int32_t, int8_t>(
          std::round(scale * sum + destOffset));
    }
  }
}

template <typename ElemTy>
void InterpreterFunction::fwdMatMulInst_FloatImpl(const MatMulInst *I) {
  static_assert(
      std::is_floating_point<ElemTy>::value ||
          std::is_same<float16_t, typename std::remove_cv<ElemTy>::type>::value,
      "This implementation is for floating-point values only");

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

void InterpreterFunction::fwdMatMulInst(const glow::MatMulInst *I) {
  if (getTensor(I->getLHS())->getType().isQuantizedType()) {
    fwdMatMulInst_I8Impl(I);
  } else if (I->getLHS()->getType()->getElementType() == ElemKind::FloatTy) {
    fwdMatMulInst_FloatImpl<float>(I);
  } else if (I->getLHS()->getType()->getElementType() == ElemKind::Float16Ty) {
    fwdMatMulInst_FloatImpl<float16_t>(I);
  } else {
    llvm_unreachable("Type is not supported");
  }
}

//===----------------------------------------------------------------------===//
//                       Row-wise quantized FC
//===----------------------------------------------------------------------===//
void InterpreterFunction::fwdRowwiseQuantizedFullyConnectedInst(
    const RowwiseQuantizedFullyConnectedInst *I) {
  auto inW = getWeightHandle<int8_t>(I->getSrc());
  auto outW = getWeightHandle<int8_t>(I->getDest());
  auto weightsW = getWeightHandle<int8_t>(I->getWeights());
  auto biasW = getWeightHandle<int8_t>(I->getBias());
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

void InterpreterFunction::fwdBatchedAddInst_I8Impl(
    const glow::BatchedAddInst *I) {
  assert(getTensor(I->getBatch())->getType().isQuantizedType() &&
         "Wrong function");
  auto batch = getWeightHandle<int8_t>(I->getBatch());
  auto slice = getWeightHandle<int8_t>(I->getSlice());
  auto dest = getWeightHandle<int8_t>(I->getDest());

  auto batchTy = I->getBatch()->getType();
  auto sliceTy = I->getSlice()->getType();
  auto destTy = I->getDest()->getType();

  float sliceScale = sliceTy->getScale();
  float batchScale = batchTy->getScale();
  float destScale = destTy->getScale();

  int32_t sliceOffset = sliceTy->getOffset();
  int32_t batchOffset = batchTy->getOffset();
  int32_t destOffset = destTy->getOffset();

  auto bdim = flattenCdr(batch.dims());
  assert(slice.size() == bdim.second && "Invalid slice size");
  assert(batch.dims().drop_front() == slice.dims() && "Invalid batch size");

  // For each layer in the batch:
  for (size_t n = 0; n < bdim.first; n++) {
    size_t base = batch.getElementPtr({n});

    // For each element in the slice.
    for (size_t i = 0; i < bdim.second; i++) {
      int32_t batchVal = batch.raw(base + i);
      int32_t sliceVal = slice.raw(i);
      // We increase the size of the integer up to 16 bits for more accurate
      // arithmetic.
      const float largeScale = float(1) / (1 << 15);
      // Scale both sides from 8-bit to 16-bits.
      int32_t B =
          std::round(float(batchVal - batchOffset) * (batchScale / largeScale));
      int32_t S =
          std::round(float(sliceVal - sliceOffset) * (sliceScale / largeScale));
      int32_t R = B + S;
      dest.raw(base + i) = quantization::clip<int32_t, int8_t>(
          std::round(float(R) * (largeScale / destScale) + destOffset));
    }
  }
}

template <typename ElemTy>
void InterpreterFunction::fwdBatchedAddInst_FloatImpl(
    const glow::BatchedAddInst *I) {
  static_assert(
      std::is_floating_point<ElemTy>::value ||
          std::is_same<float16_t, typename std::remove_cv<ElemTy>::type>::value,
      "This implementation is for floating-point values only");

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

void InterpreterFunction::fwdBatchedAddInst(const glow::BatchedAddInst *I) {
  if (getTensor(I->getBatch())->getType().isQuantizedType()) {
    fwdBatchedAddInst_I8Impl(I);
  } else if (I->getBatch()->getType()->getElementType() == ElemKind::FloatTy) {
    fwdBatchedAddInst_FloatImpl<float>(I);
  } else if (I->getBatch()->getType()->getElementType() ==
             ElemKind::Float16Ty) {
    fwdBatchedAddInst_FloatImpl<float16_t>(I);
  } else {
    llvm_unreachable("Type is not supported");
  }
}
void InterpreterFunction::fwdBatchedReduceAddInst(
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

  // Get unowned handles of the batch and dest with these new expanded dims.
  auto eBatch = getTensor(batch)->getUnowned(eBatchDims);
  auto eDest = getTensor(dest)->getUnowned(eDestDims);
  auto eBatchH = eBatch.getHandle();
  auto eDestH = eDest.getHandle();
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
              eDestH.at(destIndices) += eBatchH.at({x, y, z, w, q, r});
            }
          }
        }
      }
    }
  }
}

void InterpreterFunction::fwdSparseLengthsWeightedSumInst(
    const SparseLengthsWeightedSumInst *I) {
  auto out = getTensor(I->getDest());
  auto data = getTensor(I->getData());
  auto weights = getTensor(I->getWeights());
  auto indices = getTensor(I->getIndices());
  auto lengths = getTensor(I->getLengths());

  out->zero();

  auto IH = indices->getHandle<int64_t>();
  auto LH = lengths->getHandle<int64_t>();

  size_t segments = lengths->dims()[0];
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    totalLength += LH.raw(i);
  }
  assert(totalLength == indices->dims()[0] &&
         "sum(Lengths) must be equal to len(Indices)");

  size_t lineSize = data->size() / data->dims()[0];

  assert(!data->getType().isQuantizedType() &&
         "Quantization is not yet supported for SparseLengthsWeightedSum.");

  auto DH = data->getHandle<float>();
  auto WH = weights->getHandle<float>();
  auto OH = out->getHandle<float>();

  size_t curIdx = 0;
  for (size_t i = 0; i < segments; i++) {
    for (size_t j = 0, e = LH.raw(i); j < e; j++) {
      float weight = WH.raw(curIdx);
      size_t offsetIn = IH.raw(curIdx++) * lineSize;
      size_t offsetOut = i * lineSize;
      for (size_t k = 0; k < lineSize; k++)
        OH.raw(offsetOut++) += DH.raw(offsetIn++) * weight;
    }
  }
}

void InterpreterFunction::fwdLengthsToRangesInst(const LengthsToRangesInst *I) {
  auto ranges = getTensor(I->getDest())->getHandle<int64_t>();
  auto lengths = getTensor(I->getLengths())->getHandle<int64_t>();
  int64_t offset = 0;
  for (size_t i = 0; i < lengths.dims()[0]; i++) {
    auto length = lengths.at({i});
    ranges.at({i, 0}) = offset;
    ranges.at({i, 1}) = length;
    offset += length;
  }
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

void InterpreterFunction::fwdTopKInst(const TopKInst *I) {
  auto outW = getTensor(I->getValues());
  auto indW = getTensor(I->getIndices());
  auto inW = getTensor(I->getInput());
  size_t k = I->getK();

  if (inW->getType().isQuantizedType()) {
    fwdTopK<int8_t>(outW, indW, inW, k);
  } else {
    fwdTopK<float>(outW, indW, inW, k);
  }
}

//===----------------------------------------------------------------------===//
//                  Tensor allocation operations
//===----------------------------------------------------------------------===//

void InterpreterFunction::fwdAllocActivationInst(const AllocActivationInst *I) {
  getOrCreateTensor(I);
}

void InterpreterFunction::fwdDeallocActivationInst(
    const DeallocActivationInst *I) {
  deleteTensor(I->getSrc());
}

//===----------------------------------------------------------------------===//
//                       Debug instructions
//===----------------------------------------------------------------------===//

/// Prints a value of the instruction's operand.
/// In most cases it will be the name of the variable and the value of the
/// tensor.
void InterpreterFunction::fwdDebugPrintInst(const DebugPrintInst *I) {
  auto *V = I->getSrc();
  llvm::outs() << I->getName() << ": ";
  // Dump the content of a value.
  V->dump();
  llvm::outs() << "\n";
  dumpImpl(getTensor(V));
  llvm::outs() << "\n";
}

//===----------------------------------------------------------------------===//
//                Instructions used by Quantization
//===----------------------------------------------------------------------===//

void InterpreterFunction::fwdQuantizationProfileInst(
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
void InterpreterFunction::fwdQuantizeInst(const glow::QuantizeInst *I) {
  auto srcHandle = getWeightHandle(I->getSrc());
  auto *destTensor = getTensor(I->getDest());

  TensorQuantizationParams params{destTensor->getType().getScale(),
                                  destTensor->getType().getOffset()};

  auto destHandle = destTensor->getHandle<int8_t>();
  for (size_t i = 0, e = destHandle.size(); i < e; ++i) {
    destHandle.raw(i) = quantization::quantize(srcHandle.raw(i), params);
  }
}
/// Dequantize integer tensor. Scale and Offset are based
/// on the source tensor type.
void InterpreterFunction::fwdDequantizeInst(const glow::DequantizeInst *I) {
  auto *srcTensor = getTensor(I->getSrc());
  auto destHandle = getWeightHandle(I->getDest());

  TensorQuantizationParams params{srcTensor->getType().getScale(),
                                  srcTensor->getType().getOffset()};

  auto srcHandle = srcTensor->getHandle<int8_t>();
  for (size_t i = 0, e = destHandle.size(); i < e; ++i) {
    destHandle.raw(i) = quantization::dequantize(srcHandle.raw(i), params);
  }
}

void InterpreterFunction::fwdRescaleQuantizedInst(
    const glow::RescaleQuantizedInst *I) {
  auto src = I->getSrc();
  auto dest = I->getDest();
  auto srcTy = src->getType();
  auto destTy = dest->getType();

  TensorQuantizationParams srcQ{srcTy->getScale(), srcTy->getOffset()};
  TensorQuantizationParams destQ{destTy->getScale(), destTy->getOffset()};

  auto srcH = getWeightHandle<int8_t>(src);
  auto destH = getWeightHandle<int8_t>(dest);

  for (size_t i = 0, e = destH.size(); i < e; ++i) {
    float val = quantization::dequantize(srcH.raw(i), srcQ);
    destH.raw(i) = quantization::quantize(val, destQ);
  }
}

void InterpreterFunction::fwdIntLookupTableInst(const IntLookupTableInst *I) {
  auto srcH = getWeightHandle<int8_t>(I->getSrc());
  auto destH = getWeightHandle<int8_t>(I->getDest());
  auto mappingH = getWeightHandle<int8_t>(I->getMapping());

  for (size_t i = 0, e = destH.size(); i < e; i++) {
    destH.raw(i) = mappingH.raw((int)srcH.raw(i) + 128);
  }
}

void InterpreterFunction::fwdConvertToInst(const glow::ConvertToInst *I) {
  Tensor *source = getTensor(I->getInput());
  Tensor *dest = getTensor(I->getResult());
  switch (source->getElementType()) {
  case ElemKind::FloatTy:
    assert(dest->getElementType() == ElemKind::Float16Ty &&
           "Conversion not supported");
    dest->copyWithCast<float16_t, float>(source);
    break;
  case ElemKind::Float16Ty:
    assert(dest->getElementType() == ElemKind::FloatTy &&
           "Conversion not supported");
    dest->copyWithCast<float, float16_t>(source);
    break;
  default:
    llvm_unreachable("Type not supported");
  }
}
