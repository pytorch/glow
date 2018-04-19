// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "Interpreter.h"

#include "glow/IR/Instrs.h"
#include "glow/Quantization/Profile.h"
#include "glow/Quantization/Quantization.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace glow;

//===----------------------------------------------------------------------===//
//                       Convolution
//===----------------------------------------------------------------------===//

void Interpreter::fwdCopyInst(const CopyInst *I) {
  auto inT = getTensor(I->getSrc());
  auto outT = getTensor(I->getDest());
  outT->copyRawFrom(inT);
}

// This is the floating point implementation of Convolution.
void Interpreter::fwdConvolutionInst_FloatImpl(Value *inV, Value *outV,
                                               Value *filterV, Value *biasV,
                                               size_t filterSize, size_t stride,
                                               size_t pad) {

  auto inW = getWeightHandle(inV);
  auto outW = getWeightHandle(outV);
  auto filterW = getWeightHandle(filterV);
  auto biasW = getWeightHandle(biasV);

  ShapeNHWC odim(outW.dims());
  ShapeNHWC idim(inW.dims());

  size_t inChannels = idim.c;

  // For each input in the batch:
  for (size_t n = 0; n < idim.n; n++) {

    // For each layer in the output tensor:
    for (size_t d = 0; d < odim.c; d++) {

      // For each convolution 'jump' in the input tensor:
      ssize_t x = -ssize_t(pad);
      for (size_t ax = 0; ax < odim.h; x += stride, ax++) {
        ssize_t y = -ssize_t(pad);
        for (size_t ay = 0; ay < odim.w; y += stride, ay++) {

          // For each element in the convolution-filter:
          float sum = 0;
          for (size_t fx = 0; fx < filterSize; fx++) {
            for (size_t fy = 0; fy < filterSize; fy++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                  oy >= ssize_t(idim.w)) {
                continue;
              }
              for (size_t fd = 0; fd < inChannels; fd++) {
                sum += filterW.at({d, fx, fy, fd}) *
                       inW.at({n, (size_t)ox, (size_t)oy, fd});
              }
            }
          }

          sum += biasW.at({d});
          outW.at({n, ax, ay, d}) = sum;
        } // W
      }   // H
    }     // C
  }       // N
}

// This is the quantized i8 implementation of Convolution.
void Interpreter::fwdConvolutionInst_I8Impl(Value *inV, Value *outV,
                                            Value *filterV, Value *biasV,
                                            size_t filterSize, size_t stride,
                                            size_t pad) {
  auto inW = getWeightHandle<int8_t>(inV);
  auto outW = getWeightHandle<int8_t>(outV);
  auto filterW = getWeightHandle<int8_t>(filterV);
  auto biasW = getWeightHandle<int8_t>(biasV);

  ShapeNHWC odim(outW.dims());
  ShapeNHWC idim(inW.dims());

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

  size_t inChannels = idim.c;

  // For each input in the batch:
  for (size_t n = 0; n < idim.n; n++) {

    // For each layer in the output tensor:
    for (size_t d = 0; d < odim.c; d++) {

      // For each convolution 'jump' in the input tensor:
      ssize_t x = -ssize_t(pad);
      for (size_t ax = 0; ax < odim.h; x += stride, ax++) {
        ssize_t y = -ssize_t(pad);
        for (size_t ay = 0; ay < odim.w; y += stride, ay++) {

          // For each element in the convolution-filter:
          int32_t sum = 0;
          for (size_t fx = 0; fx < filterSize; fx++) {
            for (size_t fy = 0; fy < filterSize; fy++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                  oy >= ssize_t(idim.w)) {
                continue;
              }
              for (size_t fd = 0; fd < inChannels; fd++) {

                int32_t F = filterW.at({d, fx, fy, fd});
                int32_t I = inW.at({n, (size_t)ox, (size_t)oy, fd});
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
  }       // N
}

void Interpreter::fwdConvolutionInst(const ConvolutionInst *I) {
  size_t filterSize = I->getKernel();
  size_t pad = I->getPad();
  size_t stride = I->getStride();

  if (I->getSrc()->getType()->isQuantizedType()) {
    fwdConvolutionInst_I8Impl(I->getSrc(), I->getDest(), I->getFilter(),
                              I->getBias(), filterSize, stride, pad);
    return;
  }

  fwdConvolutionInst_FloatImpl(I->getSrc(), I->getDest(), I->getFilter(),
                               I->getBias(), filterSize, stride, pad);
}

void Interpreter::fwdConvolutionGradInst(const ConvolutionGradInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto inG = getWeightHandle(I->getSrcGrad());
  auto outG = getWeightHandle(I->getDestGrad());

  auto filterW = getWeightHandle(I->getFilter());
  auto filterG = getWeightHandle(I->getFilterGrad());
  auto biasG = getWeightHandle(I->getBiasGrad());

  size_t filterSize = I->getKernel();
  size_t pad = I->getPad();
  size_t stride = I->getStride();

  inG.clear();
  filterG.clear();
  biasG.clear();

  ShapeNHWC odim(outG.dims());
  ShapeNHWC idim(inW.dims());

  // For each input in the batch:
  for (size_t n = 0; n < odim.n; n++) {

    // Compute the gradient. For each layer in the output tensor:
    for (size_t d = 0; d < odim.c; d++) {

      // For each convolution 'jump' in the input tensor:
      ssize_t x = -ssize_t(pad);
      for (size_t ax = 0; ax < odim.h; x += stride, ax++) {
        ssize_t y = -ssize_t(pad);
        for (size_t ay = 0; ay < odim.w; y += stride, ay++) {

          float chainGrad = outG.at({n, ax, ay, d});

          // For each element in the convolution-filter:
          for (size_t fx = 0; fx < filterSize; fx++) {
            for (size_t fy = 0; fy < filterSize; fy++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                  oy >= ssize_t(idim.w)) {
                continue;
              }

              for (size_t fd = 0; fd < idim.c; fd++) {
                filterG.at({d, fx, fy, fd}) +=
                    inW.at({n, (size_t)ox, (size_t)oy, fd}) * chainGrad;
                inG.at({n, (size_t)ox, (size_t)oy, fd}) +=
                    filterW.at({d, fx, fy, fd}) * chainGrad;
              }
            }
          }

          biasG.at({d}) += chainGrad;
        } // W
      }   // H
    }     // C
  }       // N
}

//===----------------------------------------------------------------------===//
//                       Pooling
//===----------------------------------------------------------------------===//
template <class T>
static void fwdPoolMax(Tensor *inW, Tensor *outW, Handle<size_t> *SXY,
                       size_t filterSize, size_t stride, size_t pad) {
  ShapeNHWC odim(outW->dims());
  ShapeNHWC idim(inW->dims());
  Handle<T> inHandle = inW->getHandle<T>();
  Handle<T> outHandle = outW->getHandle<T>();

  // For each input in the batch:
  for (size_t n = 0; n < odim.n; n++) {

    // For each layer in the output tensor:
    for (size_t z = 0; z < idim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t x = -ssize_t(pad);
      for (size_t ax = 0; ax < odim.h; x += stride, ax++) {
        ssize_t y = -ssize_t(pad);
        for (size_t ay = 0; ay < odim.w; y += stride, ay++) {
          size_t maxX = x;
          size_t maxY = y;

          bool first = true;
          T max_value = 0;

          for (size_t fx = 0; fx < filterSize; fx++) {
            for (size_t fy = 0; fy < filterSize; fy++) {
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

          assert(!first && "Max value is uninitialized");
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

void Interpreter::fwdPoolMaxInst(const PoolMaxInst *I) {
  auto inW = getTensor(I->getSrc());
  auto outW = getTensor(I->getDest());

  if (inW->getType().isQuantizedType()) {
    fwdPoolMax<int8_t>(inW, outW, nullptr, I->getKernel(), I->getStride(),
                       I->getPad());
  } else {
    fwdPoolMax<float>(inW, outW, nullptr, I->getKernel(), I->getStride(),
                      I->getPad());
  }
}

void Interpreter::fwdPoolMaxWithXYInst(const PoolMaxWithXYInst *I) {
  auto inW = getTensor(I->getSrc());
  auto outW = getTensor(I->getDest());
  auto SXY = getTensor(I->getSrcXY())->getHandle<size_t>();

  if (inW->getType().isQuantizedType()) {
    fwdPoolMax<int8_t>(inW, outW, &SXY, I->getKernel(), I->getStride(),
                       I->getPad());
  } else {
    fwdPoolMax<float>(inW, outW, &SXY, I->getKernel(), I->getStride(),
                      I->getPad());
  }
}

void Interpreter::fwdPoolAvgInst(const PoolAvgInst *I) {
  ShapeNHWC odim(I->getDest()->dims());
  ShapeNHWC idim(I->getSrc()->dims());

  auto pad = I->getPad();
  auto filterSize = I->getKernel();
  auto stride = I->getStride();
  // Implement the avg pooling operation as defined here:
  // https://arxiv.org/abs/1312.4400
  float filterArea = filterSize * filterSize;

  if (I->getSrc()->getType()->isQuantizedType()) {
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
        ssize_t x = -ssize_t(pad);
        for (size_t ax = 0; ax < odim.h; x += stride, ax++) {
          ssize_t y = -ssize_t(pad);
          for (size_t ay = 0; ay < odim.w; y += stride, ay++) {
            int32_t sum = 0;

            for (size_t fx = 0; fx < filterSize; fx++) {
              for (size_t fy = 0; fy < filterSize; fy++) {
                ssize_t ox = x + fx;
                ssize_t oy = y + fy;

                // Ignore index access below zero (this is due to padding).
                if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                    oy >= ssize_t(idim.w)) {
                  continue;
                }

                sum += inW.at({n, (size_t)ox, (size_t)oy, z}) - inQP.offset_;
              }
            }
            // Instead of dividing by filterArea, just change scale.
            outW.at({n, ax, ay, z}) =
                quantization::clip<int32_t, int8_t>(std::round(
                    float(sum) * (inQP.scale_ / outQP.scale_ / filterArea) +
                    outQP.offset_));
          } // W
        }   // H
      }     // C
    }       // N

    return;
  }

  auto inW = getWeightHandle(I->getSrc());
  auto outW = getWeightHandle(I->getDest());

  // For each input in the batch:
  for (size_t n = 0; n < odim.n; n++) {
    // For each layer in the output tensor:
    for (size_t z = 0; z < idim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t x = -ssize_t(pad);
      for (size_t ax = 0; ax < odim.h; x += stride, ax++) {
        ssize_t y = -ssize_t(pad);
        for (size_t ay = 0; ay < odim.w; y += stride, ay++) {
          float sum = 0;

          for (size_t fx = 0; fx < filterSize; fx++) {
            for (size_t fy = 0; fy < filterSize; fy++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                  oy >= ssize_t(idim.w)) {
                continue;
              }

              sum += inW.at({n, (size_t)ox, (size_t)oy, z});
            }
          }
          outW.at({n, ax, ay, z}) = sum / filterArea;
        } // W
      }   // H
    }     // C
  }       // N
}

void Interpreter::fwdPoolMaxWithXYGradInst(const PoolMaxWithXYGradInst *I) {
  auto inG = getWeightHandle(I->getSrcGrad());
  auto outW = getWeightHandle(I->getDest());
  auto outG = getWeightHandle(I->getDestGrad());

  inG.clear();

  ShapeNHWC odim(outW.dims());

  auto SXY = getTensor(I->getSrcXY())->getHandle<size_t>();

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

void Interpreter::fwdPoolAvgGradInst(const PoolAvgGradInst *I) {
  auto inG = getWeightHandle(I->getSrcGrad());
  auto outW = getWeightHandle(I->getDest());
  auto outG = getWeightHandle(I->getDestGrad());

  ShapeNHWC odim(outW.dims());
  ShapeNHWC idim(inG.dims());

  auto pad = I->getPad();
  auto filterSize = I->getKernel();
  auto stride = I->getStride();

  inG.clear();

  float filterArea = filterSize * filterSize;

  // For each input in the batch:
  for (size_t n = 0; n < odim.n; n++) {

    // For each layer in the output tensor:
    for (size_t z = 0; z < odim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t x = -ssize_t(pad);
      for (size_t ax = 0; ax < odim.h; x += stride, ax++) {
        ssize_t y = -ssize_t(pad);
        for (size_t ay = 0; ay < odim.w; y += stride, ay++) {

          float dy = outG.at({n, ax, ay, z}) / filterArea;

          for (size_t fx = 0; fx < filterSize; fx++) {
            for (size_t fy = 0; fy < filterSize; fy++) {
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

void Interpreter::fwdSigmoidInst(const SigmoidInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto outW = getWeightHandle(I->getDest());

  for (size_t i = 0, e = outW.size(); i < e; i++) {
    float val = inW.raw(i);
    outW.raw(i) = 1 / (1 + std::exp(-val));
  }
}

void Interpreter::fwdTanhInst(const TanhInst *I) {
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

void Interpreter::fwdSoftMaxInst(const SoftMaxInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto outW = getWeightHandle(I->getDest());
  auto idim = inW.dims();

  for (size_t n = 0; n < idim[0]; n++) {
    // Find Max.
    float max = inW.at({n, 0});
    for (size_t i = 1; i < idim[1]; i++) {
      max = std::max(max, inW.at({n, i}));
    }

    // Compute exp.
    float sum = 0;
    for (size_t i = 0; i < idim[1]; i++) {
      float e = std::exp(inW.at({n, i}) - max);
      sum += e;
      outW.at({n, i}) = e;
    }

    // Normalize the output.
    for (size_t i = 0; i < idim[1]; i++) {
      outW.at({n, i}) = outW.at({n, i}) / sum;
    }
  } // N
}

void Interpreter::fwdSoftMaxGradInst(const SoftMaxGradInst *I) {
  auto inG = getWeightHandle(I->getSrcGrad());
  auto idim = inG.dims();
  auto outW = getWeightHandle(I->getOrigDest());
  auto selectedH = getTensor(I->getSelected())->getHandle<size_t>();

  inG.clear();

  // http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
  // https://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network
  for (size_t n = 0; n < idim[0]; n++) {
    for (size_t i = 0; i < idim[1]; i++) {
      float delta = (selectedH.at({n, 0}) == i);
      inG.at({n, i}) = outW.at({n, i}) - delta;
    }
  }
}

void Interpreter::fwdCrossEntropyLossInst(const CrossEntropyLossInst *I) {
  auto P = getWeightHandle(I->getP());
  auto labels = getTensor(I->getLabels())->getHandle<size_t>();
  auto CE = getWeightHandle(I->getCE());
  auto dims = P.dims();
  for (size_t n = 0; n < dims[0]; ++n) {
    auto y = labels.raw(n);
    auto p_n = P.at({n, y});
    CE.at({0}) -= log(p_n);
  }
}

void Interpreter::fwdCrossEntropyLossGradInst(
    const CrossEntropyLossGradInst *I) {
  auto P = getWeightHandle(I->getP());
  auto Labels = getTensor(I->getLabels())->getHandle<size_t>();
  auto PGrad = getWeightHandle(I->getPgrad());
  auto dims = PGrad.dims();
  PGrad.clear();
  for (size_t n = 0; n < dims[0]; ++n) {
    auto y = Labels.raw(n);
    PGrad.at({n, y}) = -1 / P.at({n, y}); // * CEGrad.at({0})
  }
}

//===----------------------------------------------------------------------===//
//                       Tensor shape (transpose/concat/...)
//===----------------------------------------------------------------------===//
void Interpreter::fwdTransposeInst(const TransposeInst *I) {
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

void Interpreter::fwdBroadcastInst(const BroadcastInst *I) {
  auto inT = getTensor(I->getSrc());
  auto outT = getTensor(I->getDest());
  auto shape = I->getShape();
  auto axis = I->getAxis();

  inT->broadcastToNewShape(outT, shape, axis);
}

void Interpreter::fwdTensorViewInst(const TensorViewInst *I) {
  getOrCreateUnownedTensor(I, I->getSrc());
}

void Interpreter::fwdSplatInst(const glow::SplatInst *I) {
  auto *T = getTensor(I->getDest());
  ElemKind k = T->getElementType();

  if (k == ElemKind::IndexTy) {
    return T->getHandle<size_t>().clear(I->getValue());
  }

  if (k == ElemKind::FloatTy) {
    return T->getHandle<float>().clear(I->getValue());
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

void Interpreter::fwdInsertTensorInst(const glow::InsertTensorInst *I) {
  Tensor *outT = getTensor(I->getDest());
  Tensor *inT = getTensor(I->getSrc());
  ElemKind k = outT->getElementType();
#define TYPED_INSERT(TY, TYPEKIND)                                             \
  if (k == TYPEKIND) {                                                         \
    auto OH = outT->getHandle<TY>();                                           \
    auto IH = inT->getHandle<TY>();                                            \
    return OH.insertTensors(IH, I->getOffsets());                              \
  }

  TYPED_INSERT(size_t, ElemKind::IndexTy);
  TYPED_INSERT(float, ElemKind::FloatTy);
  TYPED_INSERT(int8_t, ElemKind::Int8QTy);
#undef TYPED_INSERT

  llvm_unreachable("Unsupported tensor type");
}

void Interpreter::fwdExtractTensorInst(const glow::ExtractTensorInst *I) {
  Tensor *outT = getTensor(I->getDest());
  Tensor *inT = getTensor(I->getSrc());
  ElemKind k = outT->getElementType();
#define TYPED_INSERT(TY, TYPEKIND)                                             \
  if (k == TYPEKIND) {                                                         \
    auto OH = outT->getHandle<TY>();                                           \
    auto IH = inT->getHandle<TY>();                                            \
    return IH.extractTensors(OH, I->getOffsets());                             \
  }

  TYPED_INSERT(size_t, ElemKind::IndexTy);
  TYPED_INSERT(float, ElemKind::FloatTy);
  TYPED_INSERT(int8_t, ElemKind::Int8QTy)
#undef TYPED_INSERT

  llvm_unreachable("Unsupported tensor type");
}

void Interpreter::fwdGatherInst(const glow::GatherInst *I) {
  Tensor *dataT = getTensor(I->getData());
  Tensor *indicesT = getTensor(I->getIndices());
  Tensor *outT = getTensor(I->getDest());

  size_t out_p = 0;
  size_t dataSliceSize =
      dataT->size() / dataT->dims()[0] * dataT->getType().getElementSize();
  for (size_t i = 0, end = indicesT->size(); i < end; i++) {
    size_t slice = indicesT->getHandle<size_t>().raw(i);
    std::copy(&dataT->getUnsafePtr()[dataSliceSize * slice],
              &dataT->getUnsafePtr()[dataSliceSize * (slice + 1)],
              &outT->getUnsafePtr()[out_p]);
    out_p += dataSliceSize;
  }
}

//===----------------------------------------------------------------------===//
//                      Local Response Normalization
//===----------------------------------------------------------------------===//

void Interpreter::fwdLocalResponseNormalizationInst(
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

  auto halfWindowSize = I->getHalfWindowSize();
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

void Interpreter::fwdLocalResponseNormalizationGradInst(
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

void Interpreter::fwdElementAddInst(const ElementAddInst *I) {
  if (getTensor(I->getLHS())->getType().isQuantizedType()) {
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
    return;
  }

  auto outW = getWeightHandle(I->getDest());
  auto lhsW = getWeightHandle(I->getLHS());
  auto rhsW = getWeightHandle(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = lhsW.raw(i) + rhsW.raw(i);
  }
}

void Interpreter::fwdElementSubInst(const ElementSubInst *I) {
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

void Interpreter::fwdElementMulInst(const ElementMulInst *I) {
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
    float scale = lhsQ.scale_ * rhsQ.scale_ / destQ.scale_;
    for (size_t i = 0, e = outW.size(); i < e; i++) {
      int32_t mul = (lhsW.raw(i) - lhsQ.offset_) * (rhsW.raw(i) - rhsQ.offset_);
      outW.raw(i) = quantization::clip<int32_t, int8_t>(
          std::round(mul * scale) + destQ.offset_);
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

void Interpreter::fwdElementDivInst(const ElementDivInst *I) {
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

  auto outW = getWeightHandle(I->getDest());
  auto lhsW = getWeightHandle(I->getLHS());
  auto rhsW = getWeightHandle(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = lhsW.raw(i) / rhsW.raw(i);
  }
}

void Interpreter::fwdElementMaxInst(const ElementMaxInst *I) {
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
      outW.raw(i) = std::max(L, R);
    }
    return;
  }

  auto *lhs = getTensor(I->getLHS());
  auto *rhs = getTensor(I->getRHS());
  auto *out = getTensor(I->getDest());
  auto outW = out->getHandle();
  auto lhsW = lhs->getHandle();
  auto rhsW = rhs->getHandle();
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = std::max(lhsW.raw(i), rhsW.raw(i));
  }
}

void Interpreter::fwdElementMinInst(const ElementMinInst *I) {
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
void Interpreter::fwdElementCmpLTEInst(const ElementCmpLTEInst *I) {
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

void Interpreter::fwdElementPowInst(const glow::ElementPowInst *I) {
  auto baseW = getWeightHandle(I->getBase());
  float exp = I->getExp();
  auto outW = getWeightHandle(I->getDest());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = pow(baseW.raw(i), exp);
  }
}

void Interpreter::fwdElementSelectInst(const glow::ElementSelectInst *I) {
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

void Interpreter::fwdMatMulInst(const glow::MatMulInst *I) {
  if (getTensor(I->getLHS())->getType().isQuantizedType()) {
    auto lhs = getTensor(I->getLHS())->getHandle<int8_t>();
    auto rhs = getTensor(I->getRHS())->getHandle<int8_t>();

    auto dest = getTensor(I->getDest())->getHandle<int8_t>();

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
    return;
  }

  auto lhs = getWeightHandle(I->getLHS());
  auto rhs = getWeightHandle(I->getRHS());
  auto dest = getWeightHandle(I->getDest());

  auto destDim = dest.dims();
  auto lhsDim = lhs.dims();

  dest.clear(0);

  // For each (x,y) in the destination matrix:
  for (size_t x = 0; x < destDim[0]; x++) {
    for (size_t y = 0; y < destDim[1]; y++) {

      // Perform DOT on the row an column.
      float sum = 0;
      for (size_t i = 0; i < lhsDim[1]; i++) {
        sum += lhs.at({x, i}) * rhs.at({i, y});
      }
      dest.at({x, y}) = sum;
    }
  }
}

void Interpreter::fwdBatchedAddInst(const glow::BatchedAddInst *I) {
  if (getTensor(I->getBatch())->getType().isQuantizedType()) {
    auto batch = getTensor(I->getBatch())->getHandle<int8_t>();
    auto slice = getTensor(I->getSlice())->getHandle<int8_t>();
    auto dest = getTensor(I->getDest())->getHandle<int8_t>();

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
        int32_t B = std::round(float(batchVal - batchOffset) *
                               (batchScale / largeScale));
        int32_t S = std::round(float(sliceVal - sliceOffset) *
                               (sliceScale / largeScale));
        int32_t R = B + S;
        dest.raw(base + i) = quantization::clip<int32_t, int8_t>(
            std::round(float(R) * (largeScale / destScale) + destOffset));
      }
    }
    return;
  }

  auto batch = getWeightHandle(I->getBatch());
  auto slice = getWeightHandle(I->getSlice());
  auto dest = getWeightHandle(I->getDest());

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

void Interpreter::fwdBatchedReduceAddInst(const glow::BatchedReduceAddInst *I) {
  if (getTensor(I->getBatch())->getType().isQuantizedType()) {
    auto dest = getWeightHandle<int8_t>(I->getDest());
    auto batch = getWeightHandle<int8_t>(I->getBatch());

    auto destTy = I->getDest()->getType();
    auto batchTy = I->getBatch()->getType();

    float destScale = destTy->getScale();
    float batchScale = batchTy->getScale();

    int32_t destOffset = destTy->getOffset();
    int32_t batchOffset = batchTy->getOffset();

    auto bdim = flattenCdr(batch.dims());

    // The following loop order is inefficient but easy to implement correctly;
    // as this is the Interpreter, we prioritize simplicity and correctness
    // above all else.
    // For each element in the slice:
    for (size_t i = 0; i < bdim.second; i++) {
      float sum = 0.0;

      // For each layer in the batch:
      for (size_t n = 0; n < bdim.first; n++) {
        size_t base = batch.getElementPtr({n});
        sum += batch.raw(base + i) - batchOffset;
      }

      int32_t q = std::round(sum * batchScale / destScale) + destOffset;
      dest.raw(i) = quantization::clip<int32_t, int8_t>(q);
    }
    return;
  }

  auto batch = getWeightHandle(I->getBatch());
  auto dest = getWeightHandle(I->getDest());

  auto bdim = flattenCdr(batch.dims());

  dest.clear();

  // For each layer in the batch:
  for (size_t n = 0; n < bdim.first; n++) {
    size_t base = batch.getElementPtr({n});

    // For each element in the slice:
    for (size_t i = 0; i < bdim.second; i++) {
      dest.raw(i) += batch.raw(base + i);
    }
  }
}

//===----------------------------------------------------------------------===//
//                Instructions used by RNN
//===----------------------------------------------------------------------===//
template <typename T>
static void fwdTopK(Tensor *outW, Tensor *indW, Tensor *inW, size_t k) {
  auto values = outW->getHandle<T>();
  auto indices = indW->getHandle<size_t>();
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

void Interpreter::fwdTopKInst(const TopKInst *I) {
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

void Interpreter::fwdAllocActivationInst(const AllocActivationInst *I) {
  getOrCreateTensor(I);
}

void Interpreter::fwdDeallocActivationInst(const DeallocActivationInst *I) {
  deleteTensor(I->getSrc());
}

/// Prints a value of the instruction's operand.
/// In most cases it will be the name of the variable and the value of the
/// tensor.
void Interpreter::fwdDebugPrintInst(const DebugPrintInst *I) {
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

void Interpreter::fwdQuantizationProfileInst(
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
void Interpreter::fwdQuantizeInst(const glow::QuantizeInst *I) {
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
void Interpreter::fwdDequantizeInst(const glow::DequantizeInst *I) {
  auto *srcTensor = getTensor(I->getSrc());
  auto destHandle = getWeightHandle(I->getDest());

  TensorQuantizationParams params{srcTensor->getType().getScale(),
                                  srcTensor->getType().getOffset()};

  auto srcHandle = srcTensor->getHandle<int8_t>();
  for (size_t i = 0, e = destHandle.size(); i < e; ++i) {
    destHandle.raw(i) = quantization::dequantize(srcHandle.raw(i), params);
  }
}

void Interpreter::fwdRescaleQuantizedInst(const glow::RescaleQuantizedInst *I) {
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
