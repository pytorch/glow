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

void Interpreter::fwdCopyInst(bool isTrain, const CopyInst *I) {
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
          outW.at({n, ax, ay, d}) = QuantizationTransform32To8::clip(
              std::round(float(sum) * (matMulScale / outScale) + outOffset));
        } // W
      }   // H
    }     // C
  }       // N
}

void Interpreter::fwdConvolutionInst(bool isTrain, const ConvolutionInst *I) {
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

void Interpreter::fwdConvolutionGradInst(bool isTrain,
                                         const ConvolutionGradInst *I) {
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
                    inW.at({0u, (size_t)ox, (size_t)oy, fd}) * chainGrad;
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

static void fwdPoolMax(Handle<float> inW, Handle<float> outW,
                       Handle<size_t> *SXY, size_t filterSize, size_t stride,
                       size_t pad) {
  ShapeNHWC odim(outW.dims());
  ShapeNHWC idim(inW.dims());

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
          float max = 0;

          for (size_t fx = 0; fx < filterSize; fx++) {
            for (size_t fy = 0; fy < filterSize; fy++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                  oy >= ssize_t(idim.w)) {
                continue;
              }

              float val = inW.at({n, (size_t)ox, (size_t)oy, z});

              if (first || (val >= max)) {
                first = false;
                max = val;
                maxX = ox;
                maxY = oy;
              }
            }
          }

          assert(!first && "Max value is uninitialized");
          outW.at({n, ax, ay, z}) = max;
          if (SXY) {
            SXY->at({n, ax, ay, z, 0}) = maxX;
            SXY->at({n, ax, ay, z, 1}) = maxY;
          }
        } // W
      }   // H
    }     // C
  }       // N
}

void Interpreter::fwdPoolMaxInst(bool isTrain, const PoolMaxInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto outW = getWeightHandle(I->getDest());
  fwdPoolMax(inW, outW, nullptr, I->getKernel(), I->getStride(), I->getPad());
}

void Interpreter::fwdPoolMaxWithXYInst(bool isTrain,
                                       const PoolMaxWithXYInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto outW = getWeightHandle(I->getDest());
  auto SXY = getTensor(I->getSrcXY())->getHandle<size_t>();
  fwdPoolMax(inW, outW, &SXY, I->getKernel(), I->getStride(), I->getPad());
}

void Interpreter::fwdPoolAvgInst(bool isTrain, const PoolAvgInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto outW = getWeightHandle(I->getDest());

  ShapeNHWC odim(outW.dims());
  ShapeNHWC idim(inW.dims());

  auto pad = I->getPad();
  auto filterSize = I->getKernel();
  auto stride = I->getStride();

  // Implement the avg pooling operation as defined here:
  // https://arxiv.org/abs/1312.4400

  float filterArea = filterSize * filterSize;

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

void Interpreter::fwdPoolMaxWithXYGradInst(bool isTrain,
                                           const PoolMaxWithXYGradInst *I) {
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

void Interpreter::fwdPoolAvgGradInst(bool isTrain, const PoolAvgGradInst *I) {
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

void Interpreter::fwdSigmoidInst(bool isTrain, const SigmoidInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto outW = getWeightHandle(I->getDest());

  for (size_t i = 0, e = outW.size(); i < e; i++) {
    float val = inW.raw(i);
    outW.raw(i) = 1 / (1 + std::exp(-val));
  }
}

void Interpreter::fwdTanhInst(bool isTrain, const TanhInst *I) {
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

void Interpreter::fwdSoftMaxInst(bool isTrain, const SoftMaxInst *I) {
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

void Interpreter::fwdSoftMaxGradInst(bool isTrain, const SoftMaxGradInst *I) {
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

//===----------------------------------------------------------------------===//
//                       Tensor shape (transpose/reshape/concat/...)
//===----------------------------------------------------------------------===//

void Interpreter::fwdTransposeInst(bool isTrain, const TransposeInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto outW = getTensor(I->getDest());

  assert(outW->size() == inW.size() && "Invalid tensor dimensions");
  inW.transpose(outW, I->getShuffle());
}

void Interpreter::fwdBroadcastInst(bool isTrain, const BroadcastInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto outW = getTensor(I->getDest());
  auto shape = I->getShape();
  auto axis = I->getAxis();

  inW.broadcastToNewShape(outW, shape, axis);
}

void Interpreter::fwdReshapeInst(bool isTrain, const ReshapeInst *I) {
  auto inT = getTensor(I->getSrc());
  auto outT = getTensor(I->getDest());
  outT->copyRawFrom(inT);
}

void Interpreter::fwdTensorViewInst(bool isTrain, const TensorViewInst *I) {
  getOrCreateUnownedTensor(I, I->getSrc());
}

void Interpreter::fwdSplatInst(bool isTrain, const glow::SplatInst *I) {
  auto *T = getTensor(I->getDest());
  ElemKind k = T->getElementType();

#define TYPED_SPLAT(TY, TYPEKIND)                                              \
  if (k == TYPEKIND) {                                                         \
    return T->getHandle<TY>().clear(I->getValue());                            \
  }

  TYPED_SPLAT(size_t, ElemKind::IndexTy);
  TYPED_SPLAT(float, ElemKind::FloatTy);
#undef TYPED_SPLAT

  llvm_unreachable("Unsupported tensor type");
}

void Interpreter::fwdInsertTensorInst(bool isTrain,
                                      const glow::InsertTensorInst *I) {
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
#undef TYPED_INSERT
}

void Interpreter::fwdExtractTensorInst(bool isTrain,
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

  TYPED_INSERT(size_t, ElemKind::IndexTy);
  TYPED_INSERT(float, ElemKind::FloatTy);
#undef TYPED_INSERT
}

void Interpreter::fwdGatherInst(bool isTrain, const glow::GatherInst *I) {
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
//                      Batch Normalization
//===----------------------------------------------------------------------===//

void Interpreter::fwdBatchNormalizationInst(bool isTrain,
                                            const BatchNormalizationInst *I) {
  if (isTrain) {
    return fwdBatchNormalizationInst_train(I);
  }

  return fwdBatchNormalizationInst_infer(I);
}

void Interpreter::fwdBatchNormalizationInst_infer(
    const BatchNormalizationInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto outW = getWeightHandle(I->getDest());

  auto betaWH = getWeightHandle(I->getBias());
  auto gammaWH = getWeightHandle(I->getScale());
  auto varH = getWeightHandle(I->getVar());
  auto meanH = getWeightHandle(I->getMean());

  // http://cthorey.github.io./backpropagation/
  //
  // mu = 1/N*np.sum(h,axis =0)
  // sigma2 = 1/N*np.sum((h-mu)**2)
  // hath = (h-mu)*(sigma2+epsilon)**(-1./2.)
  // y = gamma*hath+beta

  // In inference mode just apply the transformation:
  // y[i] = (x - mu) * gamma / stdvar + beta;

  auto channelIdx = I->getChannelIdx();
  auto epsilon = I->getEpsilon();

  // Fast path:
  // This loop is specialized for the case where the shape is NHWC and we are
  // normalizing on the channel dimension.
  if (inW.dims().size() == 4 && channelIdx == 3) {
    ShapeNHWC odim(outW.dims());
    ShapeNHWC idim(inW.dims());

    for (size_t n = 0; n < idim.n; n++) {
      for (size_t x = 0; x < odim.h; x++) {
        for (size_t y = 0; y < odim.w; y++) {
          for (size_t c = 0; c < odim.c; c++) {
            float inp = inW.at({n, x, y, c});
            float mu = meanH.at(c);
            float var = varH.at(c);
            float stdvar = 1.0f / std::sqrt(var + epsilon);
            float gamma = gammaWH.at(c);
            float beta = betaWH.at(c);
            outW.at({n, x, y, c}) = (inp - mu) * gamma * stdvar + beta;
          } // C
        }   // W
      }     // H
    }       // N
    return;
  }

  // Slow path:
  // This is the general batch normalization implementation for
  // n-dimensional tensors.
  for (size_t i = 0, e = inW.size(); i < e; i++) {
    size_t channelId = inW.getDimForPtr(channelIdx, i);
    float x = inW.raw(i);

    float mu = meanH.at({channelId});
    float var = varH.at({channelId});

    float stdvar = 1.0f / std::sqrt(var + epsilon);

    float gamma = gammaWH.at({channelId});
    float beta = betaWH.at({channelId});

    outW.raw(i) = (x - mu) * gamma * stdvar + beta;
  }
}

void Interpreter::fwdBatchNormalizationInst_train(
    const BatchNormalizationInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto varH = getWeightHandle(I->getVar());
  auto meanH = getWeightHandle(I->getMean());

  auto channelIdx = I->getChannelIdx();
  auto momentum = I->getMomentum();

  Tensor localMean(ElemKind::FloatTy, meanH.dims());
  Tensor localVar(ElemKind::FloatTy, varH.dims());
  auto localMeanH = localMean.getHandle<>();
  auto localVarH = localVar.getHandle<>();

  // The number of different channels.
  const size_t numChannels = inW.dims()[channelIdx];
  // THe number of elements that each channel holds.
  const size_t samplesPerChannel = inW.size() / numChannels;

  // Calculate Mean:

  // sum(in[i])
  for (size_t i = 0, e = inW.size(); i < e; i++) {
    size_t channelId = inW.getDimForPtr(channelIdx, i);
    float v = inW.raw(i);
    localMeanH.raw(channelId) += v;
  }
  // Mean = sum(in[i]) / N
  for (size_t i = 0, e = localMeanH.size(); i < e; i++) {
    localMeanH.at({i}) /= samplesPerChannel;
  }

  // Calculate Variance:

  // sum((x - mu) ^ 2)
  for (size_t i = 0, e = inW.size(); i < e; i++) {
    size_t channelId = inW.getDimForPtr(channelIdx, i);
    float v = inW.raw(i) - localMeanH.at({channelId});
    localVarH.raw(channelId) += v * v;
  }
  // Var = sum((x - mu) ^ 2) / N
  for (size_t i = 0, e = localMeanH.size(); i < e; i++) {
    localVarH.at({i}) /= samplesPerChannel;
  }

  // Update the global variance and mean:
  for (size_t i = 0, e = localMeanH.size(); i < e; i++) {
    auto P = momentum;
    meanH.at({i}) = P * localMeanH.at({i}) + (1 - P) * meanH.at({i});
    varH.at({i}) = P * localVarH.at({i}) + (1 - P) * varH.at({i});
  }

  // TODO: should we be using the running mean or the local mean?
  fwdBatchNormalizationInst_infer(I);
}

void Interpreter::fwdBatchNormalizationGradInst(
    bool isTrain, const BatchNormalizationGradInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto inG = getWeightHandle(I->getSrcGrad());
  auto outG = getWeightHandle(I->getDestGrad());

  auto gammaWH = getWeightHandle(I->getScale());
  auto betaGH = getWeightHandle(I->getBiasGrad());
  auto gammaGH = getWeightHandle(I->getScaleGrad());

  inG.clear();
  betaGH.clear();
  gammaGH.clear();

  auto varH = getWeightHandle(I->getVar());
  auto meanH = getWeightHandle(I->getMean());

  auto channelIdx = I->getChannelIdx();
  auto epsilon = I->getEpsilon();

  // Update the gradient of the incoming buffer:
  Tensor dyhmu(ElemKind::FloatTy, meanH.dims());
  Tensor sumDy(ElemKind::FloatTy, meanH.dims());
  auto dyhmuH = dyhmu.getHandle<>();
  auto sumDyH = sumDy.getHandle<>();

  // The number of different channels.
  const size_t numChannels = inW.dims()[channelIdx];
  // THe number of elements that each channel holds.
  const size_t samplesPerChannel = inW.size() / numChannels;

  // Calculate: sum(dy * (h - mu))
  for (size_t i = 0, e = inW.size(); i < e; i++) {
    size_t channelId = inW.getDimForPtr(channelIdx, i);
    // x - mean.
    float cx = inW.raw(i) - meanH.at({channelId});
    // dy * (h - mu)
    dyhmuH.at({channelId}) += outG.raw(i) * cx;
  }

  // Calculate: sum(dy)
  for (size_t i = 0, e = inW.size(); i < e; i++) {
    size_t channelId = inW.getDimForPtr(channelIdx, i);
    sumDyH.at({channelId}) += outG.raw(i);
  }

  // http://cthorey.github.io./backpropagation/
  //
  // mu = 1./N*np.sum(h)
  // var = 1./N*np.sum((h-mu)**2)
  // dbeta = np.sum(dy)
  // dgamma = np.sum((h - mu) * (var + eps)**(-1. / 2.) * dy)
  // dh = (1. / N) * gamma * (var + eps)**(-1. / 2.) *
  //     (N * dy - np.sum(dy) - (h - mu) * 1/(var + eps) *
  //     np.sum(dy * (h - mu)))
  //
  for (size_t i = 0, e = inW.size(); i < e; i++) {
    size_t channelId = inW.getDimForPtr(channelIdx, i);

    float invN = 1.0f / samplesPerChannel;
    float gamma = gammaWH.at({channelId});
    float var = varH.at({channelId});
    float mu = meanH.at({channelId});
    float invVarSqrt = 1.0f / std::sqrt(var + epsilon);
    float invVar = 1.0f / (var + epsilon);

    float dy = outG.raw(i);
    float hmu = inW.raw(i) - mu;
    float sdy = sumDyH.at(channelId);
    float sdyhmu = dyhmuH.at(channelId);
    inG.raw(i) += invN * gamma * invVarSqrt *
                  (samplesPerChannel * dy - sdy - hmu * invVar * sdyhmu);
  }

  // Update the gradient of beta and gamma.
  for (size_t i = 0, e = inW.size(); i < e; i++) {
    size_t channelId = inW.getDimForPtr(channelIdx, i);

    float mu = meanH.at({channelId});
    float var = varH.at({channelId});
    float invVarSqrt = 1.0f / std::sqrt(var + epsilon);

    betaGH.at({channelId}) += outG.raw(i);
    gammaGH.at({channelId}) += (inW.raw(i) - mu) * invVarSqrt * outG.raw(i);
  }
}

void Interpreter::fwdLocalResponseNormalizationInst(
    bool isTrain, const glow::LocalResponseNormalizationInst *I) {
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
    bool isTrain, const glow::LocalResponseNormalizationGradInst *I) {
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

void Interpreter::fwdElementAddInst(bool isTrain, const ElementAddInst *I) {
  auto outW = getWeightHandle(I->getDest());
  auto lhsW = getWeightHandle(I->getLHS());
  auto rhsW = getWeightHandle(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = lhsW.raw(i) + rhsW.raw(i);
  }
}

void Interpreter::fwdElementSubInst(bool isTrain, const ElementSubInst *I) {
  auto outW = getWeightHandle(I->getDest());
  auto lhsW = getWeightHandle(I->getLHS());
  auto rhsW = getWeightHandle(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = lhsW.raw(i) - rhsW.raw(i);
  }
}

void Interpreter::fwdElementMulInst(bool isTrain, const ElementMulInst *I) {
  auto outW = getWeightHandle(I->getDest());
  auto lhsW = getWeightHandle(I->getLHS());
  auto rhsW = getWeightHandle(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = lhsW.raw(i) * rhsW.raw(i);
  }
}

void Interpreter::fwdElementDivInst(bool isTrain, const ElementDivInst *I) {
  auto outW = getWeightHandle(I->getDest());
  auto lhsW = getWeightHandle(I->getLHS());
  auto rhsW = getWeightHandle(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = lhsW.raw(i) / rhsW.raw(i);
  }
}

void Interpreter::fwdElementMaxInst(bool isTrain, const ElementMaxInst *I) {
  auto outW = getWeightHandle(I->getDest());
  auto lhsW = getWeightHandle(I->getLHS());
  auto rhsW = getWeightHandle(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = std::max(lhsW.raw(i), rhsW.raw(i));
  }
}

void Interpreter::fwdElementMinInst(bool isTrain, const ElementMinInst *I) {
  auto outW = getWeightHandle(I->getDest());
  auto lhsW = getWeightHandle(I->getLHS());
  auto rhsW = getWeightHandle(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = std::min(lhsW.raw(i), rhsW.raw(i));
  }
}

void Interpreter::fwdElementCmpLTEInst(bool isTrain,
                                       const ElementCmpLTEInst *I) {
  auto outW = getWeightHandle(I->getDest());
  auto lhsW = getWeightHandle(I->getLHS());
  auto rhsW = getWeightHandle(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = lhsW.raw(i) <= rhsW.raw(i) ? 1.0 : 0.0;
  }
}

void Interpreter::fwdElementSelectInst(bool isTrain,
                                       const glow::ElementSelectInst *I) {
  auto outW = getWeightHandle(I->getDest());
  auto condW = getWeightHandle(I->getCond());
  auto lhsW = getWeightHandle(I->getLHS());
  auto rhsW = getWeightHandle(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = (condW.raw(i) != 0.0) ? lhsW.raw(i) : rhsW.raw(i);
  }
}

void Interpreter::fwdBatchedMatMulInst(bool isTrain,
                                       const glow::BatchedMatMulInst *I) {
  if (getTensor(I->getLHS())->getType().isQuantizedType()) {
    auto lhs = getTensor(I->getLHS())->getHandle<int8_t>();
    auto rhs = getTensor(I->getRHS())->getHandle<int8_t>();

    auto dest = getTensor(I->getDest())->getHandle<int8_t>();

    auto destDim = dest.dims();
    auto lhsDim = lhs.dims();
    auto rhsDim = rhs.dims();

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

    // For each layer in the batch:
    for (size_t n = 0; n < destDim[0]; n++) {
      // Broadcast tensors with a batch size of 1 by selecting the right slice.
      size_t ln = (lhsDim[0] == 1 ? 0 : n);
      size_t rn = (rhsDim[0] == 1 ? 0 : n);

      // For each (x,y) in the destination matrix:
      for (size_t x = 0; x < destDim[1]; x++) {
        for (size_t y = 0; y < destDim[2]; y++) {

          // Perform DOT on the row an column.
          int32_t sum = 0;
          for (size_t i = 0; i < lhsDim[2]; i++) {
            int32_t L = lhs.at({ln, x, i});
            int32_t R = rhs.at({rn, i, y});
            // We represent the element multiplication with offset as
            // (value - offset).
            sum += (L - lhsOffset) * (R - rhsOffset);
          }

          dest.at({n, x, y}) = std::round(scale * sum + destOffset);
        }
      }
    } // N
    return;
  }

  auto lhs = getWeightHandle(I->getLHS());
  auto rhs = getWeightHandle(I->getRHS());
  auto dest = getWeightHandle(I->getDest());

  auto destDim = dest.dims();
  auto lhsDim = lhs.dims();
  auto rhsDim = rhs.dims();

  dest.clear(0);

  // For each layer in the batch:
  for (size_t n = 0; n < destDim[0]; n++) {
    // Broadcast tensors with a batch size of 1 by selecting the right slice.
    size_t ln = (lhsDim[0] == 1 ? 0 : n);
    size_t rn = (rhsDim[0] == 1 ? 0 : n);

    // For each (x,y) in the destination matrix:
    for (size_t x = 0; x < destDim[1]; x++) {
      for (size_t y = 0; y < destDim[2]; y++) {

        // Perform DOT on the row an column.
        float sum = 0;
        for (size_t i = 0; i < lhsDim[2]; i++) {
          sum += lhs.at({ln, x, i}) * rhs.at({rn, i, y});
        }
        dest.at({n, x, y}) = sum;
      }
    }
  } // N
}

void Interpreter::fwdBatchedAddInst(bool isTrain,
                                    const glow::BatchedAddInst *I) {
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
        dest.raw(base + i) = QuantizationTransform32To8::clip(
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
      dest.raw(base + i) =
          QuantizationTransform32To8::clip(batch.raw(base + i) + slice.raw(i));
    }
  }
}

void Interpreter::fwdBatchedReduceAddInst(bool isTrain,
                                          const glow::BatchedReduceAddInst *I) {
  auto batch = getWeightHandle(I->getBatch());
  auto dest = getWeightHandle(I->getDest());

  auto bdim = flattenCdr(batch.dims());

  dest.clear();

  // For each layer in the batch:
  for (size_t n = 0; n < bdim.first; n++) {
    size_t base = batch.getElementPtr({n});

    // For each element in the slice.
    for (size_t i = 0; i < bdim.second; i++) {
      dest.raw(i) += batch.raw(base + i);
    }
  }
}

//===----------------------------------------------------------------------===//
//                       Training Instructions
//===----------------------------------------------------------------------===//

void Interpreter::fwdSGDInst(bool isTrain, const glow::SGDInst *I) {
  auto W = getWeightHandle(I->getWeight());
  auto G = getWeightHandle(I->getGradient());
  auto Gsum = Handle<float>::createInvalidHandle();

  /// Described in the paper: Alex Krizhevsky [2014]
  // "One weird trick for parallelizing convolutional neural networks"

  if (I->getMomentum() > 0.0) {
    Gsum = getWeightHandle(I->getGsum());
  }

  assert(W.dims() == G.dims() && "Invalid tensor sizes");

  float L1Decay = I->getL1Decay();
  float L2Decay = I->getL2Decay();
  float learningRate = I->getLearningRate();
  float momentum = I->getMomentum();
  float batchSize = I->getBatchSize();
  auto sz = W.size();

  // For each weight/gradient pair:
  for (size_t x = 0; x < sz; x++) {
    // Do a simple SGD update:
    float L1Grad = L1Decay * (W.raw(x) > 0 ? 1 : -1);
    float L2Grad = L2Decay * (W.raw(x));
    float gij = (L2Grad + L1Grad + G.raw(x)) / batchSize;

    // Use the momentum to improve the gradient descent:
    // http://ufldl.stanford.edu/tutorial/supervised/
    // OptimizationStochasticGradientDescent/
    if (momentum > 0.0) {
      // Momentum update:
      float dx = momentum * Gsum.raw(x) - learningRate * gij;
      // Save this value for the next iteration:
      Gsum.raw(x) = dx;
      // Apply the gradient.
      W.raw(x) += dx;
    } else {
      // Use regular SGD:
      W.raw(x) -= learningRate * gij;
    }
  }
}

//===----------------------------------------------------------------------===//
//                Instructions used by RNN
//===----------------------------------------------------------------------===//

void Interpreter::fwdTopKInst(bool isTrain, const TopKInst *I) {
  auto in = getTensor(I->getInput())->getHandle();
  size_t k = I->getK();
  size_t n = in.dims().back();
  auto values = getTensor(I->getValues())->getHandle();
  auto indices = getTensor(I->getIndices())->getHandle<size_t>();

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
//                  Tensor allocation operations
//===----------------------------------------------------------------------===//

void Interpreter::fwdAllocActivationInst(bool isTrain,
                                         const AllocActivationInst *I) {
  getOrCreateTensor(I);
}

void Interpreter::fwdDeallocActivationInst(bool isTrain,
                                           const DeallocActivationInst *I) {
  deleteTensor(I->getOperand(0).first);
}

/// Prints a value of the instruction's operand.
/// In most cases it will be the name of the variable and the value of the
/// tensor.
void Interpreter::fwdDebugPrintInst(bool isTrain, const DebugPrintInst *I) {
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
    bool isTrain, const glow::QuantizationProfileInst *I) {
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
void Interpreter::fwdQuantizeInst(bool isTrain, const glow::QuantizeInst *I) {
  auto srcHandle = getWeightHandle(I->getSrc());
  auto *destTensor = getTensor(I->getDest());

  TensorQuantizationParams params{destTensor->getType().getScale(),
                                  destTensor->getType().getOffset()};

  auto destHandle = destTensor->getHandle<int8_t>();
  for (size_t i = 0, e = destHandle.size(); i < e; ++i) {
    destHandle.raw(i) = quantize(srcHandle.raw(i), params);
  }
}
/// Dequantize integer tensor. Scale and Offset are based
/// on the source tensor type.
void Interpreter::fwdDequantizeInst(bool isTrain,
                                    const glow::DequantizeInst *I) {
  auto *srcTensor = getTensor(I->getSrc());
  auto destHandle = getWeightHandle(I->getDest());

  TensorQuantizationParams params{srcTensor->getType().getScale(),
                                  srcTensor->getType().getOffset()};

  auto srcHandle = srcTensor->getHandle<int8_t>();
  for (size_t i = 0, e = destHandle.size(); i < e; ++i) {
    destHandle.raw(i) = dequantize(srcHandle.raw(i), params);
  }
}

void Interpreter::fwdRescaleQuantizedInst(bool isTrain,
                                          const glow::RescaleQuantizedInst *I) {
  llvm_unreachable("Not implemented");
}

void Interpreter::fwdIntrinsicInst(bool isTrain, const glow::IntrinsicInst *I) {
  llvm_unreachable("The interpreter should not handle intrinsic instructions");
}
