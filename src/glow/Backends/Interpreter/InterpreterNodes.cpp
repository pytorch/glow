// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "Interpreter.h"

#include "glow/IR/Instrs.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

using namespace glow;

//===----------------------------------------------------------------------===//
//                       Convolution
//===----------------------------------------------------------------------===//

void Interpreter::fwdCopyInst(bool isTrain, const CopyInst *I) {
  auto inT = getTensor(I->getSrc());
  auto outT = getTensor(I->getDest());
  outT->copyRawFrom(inT);
}

template <bool SpecializeFPS, size_t filterX, size_t padX, size_t strideX,
          bool SpecializeChannel, size_t channelX>
[[gnu::noinline]] void
fwdConvolutionInst_Impl(Handle<float> inW, Handle<float> outW,
                        Handle<float> filterW, Handle<float> biasW,
                        size_t filterSize, size_t pad, size_t stride) {
  ShapeNHWC odim(outW.dims());
  ShapeNHWC idim(inW.dims());

  // If the method is specialized then we can override the parameters with the
  // specialized constant values.
  if (SpecializeFPS) {
    filterSize = filterX;
    pad = padX;
    stride = strideX;
  }

  size_t inChannels = idim.c;
  if (SpecializeChannel) {
    inChannels = channelX;
  }

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
              if (ox < 0 || oy < 0 || ox >= ssize_t(odim.h) ||
                  oy >= ssize_t(odim.w)) {
                continue;
              }
#pragma clang loop interleave_count(8)
              for (size_t fd = 0; fd < inChannels; fd++) {
                sum += filterW.at({d, fx, fy, fd}) *
                       inW.at({n, (size_t)ox, (size_t)oy, fd});
              }
            }
          }

          sum += biasW.at({d});
          outW.at({n, ax, ay, d}) = sum;
        } // H
      }   // W
    }     // C
  }       // N
}

void Interpreter::fwdConvolutionInst(bool isTrain, const ConvolutionInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto outW = getWeightHandle(I->getDest());
  auto filterW = getWeightHandle(I->getFilter());
  auto biasW = getWeightHandle(I->getBias());

  size_t filterSize = I->getKernel();
  size_t pad = I->getPad();
  size_t stride = I->getStride();

  ShapeNHWC idim(inW.dims());

#define SPECIALIZE_CONV_FPS(F, P, S)                                           \
  if (filterSize == F && pad == P && stride == S)                              \
    return fwdConvolutionInst_Impl<true, F, P, S, false, 0>(                   \
        inW, outW, filterW, biasW, 0, 0, 0);

#define SPECIALIZE_CONV_FPSC(F, P, S, C)                                       \
  if (filterSize == F && pad == P && stride == S && idim.c == C)               \
    return fwdConvolutionInst_Impl<true, F, P, S, true, C>(inW, outW, filterW, \
                                                           biasW, 0, 0, 0);

  // Specialize the convolution on popular Conv kernels:
  SPECIALIZE_CONV_FPSC(7, 3, 2, 3)
  SPECIALIZE_CONV_FPS(7, 3, 2)
  SPECIALIZE_CONV_FPS(7, 4, 2)
  SPECIALIZE_CONV_FPS(3, 1, 2)
  SPECIALIZE_CONV_FPS(1, 0, 1)
  SPECIALIZE_CONV_FPS(1, 0, 2)
  SPECIALIZE_CONV_FPS(3, 1, 1)

  fwdConvolutionInst_Impl<false, 0, 0, 0, false, 0>(inW, outW, filterW, biasW,
                                                    filterSize, pad, stride);
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
              if (ox < 0 || oy < 0 || ox >= ssize_t(odim.h) ||
                  oy >= ssize_t(odim.w)) {
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
        } // H
      }   // W
    }     // C
  }       // N
}

//===----------------------------------------------------------------------===//
//                       Pooling
//===----------------------------------------------------------------------===//

void Interpreter::fwdPoolMaxInst(bool isTrain, const PoolMaxInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto outW = getWeightHandle(I->getDest());

  ShapeNHWC odim(outW.dims());
  ShapeNHWC idim(inW.dims());

  auto pad = I->getPad();
  auto filterSize = I->getKernel();
  auto stride = I->getStride();

  auto SXY = getTensor(I->getSrcXY())->getHandle<size_t>();

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
          SXY.at({n, ax, ay, z, 0}) = maxX;
          SXY.at({n, ax, ay, z, 1}) = maxY;
          outW.at({n, ax, ay, z}) = max;
        } // H
      }   // W
    }     // C
  }       // N
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
        } // H
      }   // W
    }     // C
  }       // N
}

void Interpreter::fwdPoolMaxGradInst(bool isTrain, const PoolMaxGradInst *I) {
  auto inG = getWeightHandle(I->getSrcGrad());
  auto outW = getWeightHandle(I->getDest());
  auto outG = getWeightHandle(I->getDestGrad());

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
        } // H
      }   // W
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
        } // H
      }   // W
    }     // C
  }       // N
}

//===----------------------------------------------------------------------===//
//                       Fully Connected
//===----------------------------------------------------------------------===//

void Interpreter::fwdFullyConnectedInst(bool isTrain,
                                        const FullyConnectedInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto outW = getWeightHandle(I->getDest());

  auto odim = flattenCdr(outW.dims());
  auto idim = flattenCdr(inW.dims());
  assert(odim.first == idim.first && "Mismatch batch size");

  auto filterW = getWeightHandle(I->getFilter());
  auto biasW = getWeightHandle(I->getBias());

  size_t inputSize = idim.second;

  for (size_t n = 0; n < odim.first; n++) {
    size_t base = inW.getElementPtr({n});

    for (size_t i = 0; i < odim.second; i++) {

      float sum = 0;
      for (size_t j = 0; j < inputSize; j++) {
        sum += inW.raw(base + j) * filterW.at({i, j});
      }

      sum += biasW.at({i});
      outW.at({n, i}) = sum;
    }
  } // N
}

void Interpreter::fwdFullyConnectedGradInst(bool isTrain,
                                            const FullyConnectedGradInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto inG = getWeightHandle(I->getSrcGrad());
  auto outG = getWeightHandle(I->getDestGrad());

  auto odim = flattenCdr(outG.dims());
  auto idim = flattenCdr(inW.dims());

  auto filterW = getWeightHandle(I->getFilter());
  auto filterG = getWeightHandle(I->getFilterGrad());
  auto biasG = getWeightHandle(I->getBiasGrad());

  size_t inSize = idim.second;

  for (size_t n = 0; n < odim.first; n++) {
    size_t base = inW.getElementPtr({n});

    // Compute the gradient:
    for (size_t i = 0; i < odim.second; i++) {
      float chainGrad = outG.at({n, i});

      for (size_t j = 0, e = inSize; j < e; j++) {
        // Input gradient:
        inG.raw(base + j) += filterW.at({i, j}) * chainGrad;
        // Param gradient:
        filterG.at({i, j}) += inW.raw(base + j) * chainGrad;
      }

      biasG.at({i}) += chainGrad;
    }
  } // N
}

//===----------------------------------------------------------------------===//
//                       Activation functions
//===----------------------------------------------------------------------===//

void Interpreter::fwdReluInst(bool isTrain, const ReluInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto outW = getWeightHandle(I->getDest());

  for (size_t i = 0, e = inW.size(); i < e; i++) {
    float val = inW.raw(i);
    outW.raw(i) = val < 0 ? 0 : val;
  }
}

void Interpreter::fwdReluGradInst(bool isTrain, const ReluGradInst *I) {
  auto inG = getWeightHandle(I->getSrcGrad());
  auto outW = getWeightHandle(I->getDest());
  auto outG = getWeightHandle(I->getDestGrad());

  for (size_t i = 0, e = outW.size(); i < e; i++) {
    float val = outW.raw(i);
    inG.raw(i) += (val <= 0 ? 0 : outG.raw(i));
  }
}

void Interpreter::fwdSigmoidInst(bool isTrain, const SigmoidInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto outW = getWeightHandle(I->getDest());

  for (size_t i = 0, e = outW.size(); i < e; i++) {
    float val = inW.raw(i);
    outW.raw(i) = 1 / (1 + std::exp(-val));
  }
}
void Interpreter::fwdSigmoidGradInst(bool isTrain, const SigmoidGradInst *I) {
  auto inG = getWeightHandle(I->getSrcGrad());
  auto outW = getWeightHandle(I->getDest());
  auto outG = getWeightHandle(I->getDestGrad());

  for (size_t i = 0, e = outW.size(); i < e; i++) {
    float val = outW.raw(i);
    inG.raw(i) += val * (1 - val) * outG.raw(i);
  }
}

void Interpreter::fwdTanhInst(bool isTrain, const TanhInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto outW = getWeightHandle(I->getDest());

  for (size_t i = 0, e = inW.size(); i < e; i++) {
    float val = inW.raw(i);
    float exp_val = std::exp(val);
    float exp_neg_val = std::exp(-val);
    outW.raw(i) = (exp_val - exp_neg_val) / (exp_val + exp_neg_val);
  }
}
void Interpreter::fwdTanhGradInst(bool isTrain, const TanhGradInst *I) {
  auto inG = getWeightHandle(I->getSrcGrad());
  auto outW = getWeightHandle(I->getDest());
  auto outG = getWeightHandle(I->getDestGrad());

  for (size_t i = 0, e = outW.size(); i < e; i++) {
    float val = outW.raw(i);
    inG.raw(i) += (1 - val * val) * outG.raw(i);
  }
}

//===----------------------------------------------------------------------===//
//                        Loss Functions (Softmax/regression/...)
//===----------------------------------------------------------------------===//

void Interpreter::fwdSoftMaxInst(bool isTrain, const SoftMaxInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto outW = getWeightHandle(I->getDest());
  auto idim = inW.dims();

  auto EH = getWeightHandle(I->getE());

  for (size_t n = 0; n < idim[0]; n++) {
    float max = inW.at({n, 0});

    // Find Max.
    for (size_t i = 0; i < idim[1]; i++) {
      max = std::max(max, inW.at({n, i}));
    }

    float sum = 0;

    // Compute exp.
    for (size_t i = 0; i < idim[1]; i++) {
      float e = std::exp(inW.at({n, i}) - max);
      sum += e;
      EH.at({n, i}) = e;
    }

    // Normalize the output.
    for (size_t i = 0; i < idim[1]; i++) {
      EH.at({n, i}) /= sum;
      outW.at({n, i}) = EH.at({n, i});
    }
  } // N
}

void Interpreter::fwdSoftMaxGradInst(bool isTrain, const SoftMaxGradInst *I) {
  auto inG = getWeightHandle(I->getSrcGrad());

  auto idim = inG.dims();
  auto EH = getTensor(I->getE())->getHandle<>();
  auto selectedH = getTensor(I->getSelected())->getHandle<size_t>();

  // http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
  // https://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network
  for (size_t n = 0; n < idim[0]; n++) {
    for (size_t i = 0; i < idim[1]; i++) {
      float delta = (selectedH.at({n, 0}) == i);
      float sigma = (EH.at({n, i}) - delta);
      inG.at({n, i}) += sigma;
    }
  }
}

void Interpreter::fwdRegressionInst(bool isTrain, const RegressionInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto outW = getWeightHandle(I->getDest());

  for (size_t i = 0, e = inW.size(); i < e; i++) {
    outW.raw(i) = inW.raw(i);
  }
}

void Interpreter::fwdRegressionGradInst(bool isTrain,
                                        const RegressionGradInst *I) {
  auto inW = getWeightHandle(I->getSrc());
  auto inG = getWeightHandle(I->getSrcGrad());
  auto expected = getTensor(I->getExpected());

  auto idim = inW.dims();
  assert(idim.size() == 2 && "Input is expected to be a vector per input");

  auto e = expected->getHandle<>();

  // For each input in the batch:
  for (size_t n = 0; n < idim[0]; n++) {

    for (size_t i = 0; i < idim[1]; i++) {
      float dy = inW.at({n, i}) - e.at({n, i});
      inG.at({n, i}) += dy;
    }
  } // N
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

void Interpreter::fwdTransposeGradInst(bool isTrain,
                                       const TransposeGradInst *I) {
  auto inG = getTensor(I->getSrcGrad());
  auto outG = getWeightHandle(I->getDestGrad());

  assert(outG.size() == inG->size() && "Invalid tensor dimensions");

  // Generate the reverse shuffle.
  auto shuffle = I->getShuffle();
  llvm::SmallVector<unsigned, 6> reverseShuffle(shuffle.begin(), shuffle.end());
  for (unsigned int i = 0; i < shuffle.size(); i++) {
    reverseShuffle[shuffle[i]] = i;
  }

  // Perform the reverse transpsose.
  // TODO: this wipes out the gradients and may cause a bug for operators with
  // multiple users.
  outG.transpose(inG, reverseShuffle);
}

void Interpreter::fwdReshapeInst(bool isTrain, const ReshapeInst *I) {
  auto inT = getTensor(I->getSrc());
  auto outT = getTensor(I->getDest());
  outT->copyRawFrom(inT);
}
void Interpreter::fwdReshapeGradInst(bool isTrain, const ReshapeGradInst *I) {
  auto inT = getTensor(I->getSrcGrad());
  auto outT = getTensor(I->getDestGrad());
  inT->copyRawFrom(outT);
}

void Interpreter::fwdZeroInst(bool isTrain, const glow::ZeroInst *I) {
  auto *T = getTensor(I->getDest());
  T->zero();
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

        float squareSum = 0.0;

        // Compute squareSum for first channel.
        for (size_t c = 1; c <= halfWindowSize && c < idim.c; c++) {
          auto val = inW.at({n, h, w, c});
          squareSum += (val * val);
        }

        // For every channel:
        for (size_t c = 0; c < idim.c; c++) {
          auto scale = k + normedAlpha * squareSum;

          // This will be used to accelerate the backward pass.
          scaleCache.at({n, h, w, c}) = scale;

          auto normFactor = std::pow(scale, -beta);
          outW.at({n, h, w, c}) = inW.at({n, h, w, c}) * normFactor;

          // Modify squareSum for next channel.
          auto subIndex = c - halfWindowSize;
          auto addIndex = c + halfWindowSize + 1;
          auto sub = (c >= halfWindowSize) ? inW.at({n, h, w, subIndex}) : 0;
          auto add = (addIndex < idim.c) ? inW.at({n, h, w, addIndex}) : 0;

          // Subtract out "rear" end of this window, add "front" end of next.
          squareSum = squareSum - (sub * sub) + (add * add);
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
        for (size_t c = 1; c <= halfWindowSize && c < odim.c; c++) {
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
  auto LHSW = getWeightHandle(I->getLHS());
  auto RHSW = getWeightHandle(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = LHSW.raw(i) + RHSW.raw(i);
  }
}

void Interpreter::fwdElementMulInst(bool isTrain, const ElementMulInst *I) {
  auto outW = getWeightHandle(I->getDest());
  auto LHSW = getWeightHandle(I->getLHS());
  auto RHSW = getWeightHandle(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = LHSW.raw(i) * RHSW.raw(i);
  }
}

void Interpreter::fwdElementAddGradInst(bool isTrain,
                                        const ElementAddGradInst *I) {
  auto outG = getWeightHandle(I->getDestGrad());
  auto LHSG = getWeightHandle(I->getLHSGrad());
  auto RHSG = getWeightHandle(I->getRHSGrad());
  for (size_t i = 0, e = outG.size(); i < e; i++) {
    LHSG.raw(i) = outG.raw(i);
    RHSG.raw(i) = outG.raw(i);
  }
}
void Interpreter::fwdElementMulGradInst(bool isTrain,
                                        const ElementMulGradInst *I) {
  auto LHSW = getWeightHandle(I->getLHS());
  auto RHSW = getWeightHandle(I->getRHS());
  auto outG = getWeightHandle(I->getDestGrad());
  auto LHSG = getWeightHandle(I->getLHSGrad());
  auto RHSG = getWeightHandle(I->getRHSGrad());
  for (size_t i = 0, e = outG.size(); i < e; i++) {
    LHSG.raw(i) = RHSW.raw(i) * outG.raw(i);
    RHSG.raw(i) = LHSW.raw(i) * outG.raw(i);
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
