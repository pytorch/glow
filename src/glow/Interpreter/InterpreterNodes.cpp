#include "glow/IR/Instrs.h"
#include "glow/Interpreter/Interpreter.h"
#include "glow/Support/Casting.h"

using namespace glow;

#define DBG std::cout << __FUNCTION__ << "\n";

//===----------------------------------------------------------------------===//
//                       Convolution
//===----------------------------------------------------------------------===//

void Interpreter::fwdCopyInst(Context *ctx, bool isTrain, CopyInst *I) {
  auto S = getWeightHandle(ctx, I->getSrc());
  auto D = getWeightHandle(ctx, I->getDest());

  for (size_t i = 0, e = S.size(); i < e; i++) {
    D.raw(i) = S.raw(i);
  }
}

void Interpreter::bwdCopyInst(Context *ctx, CopyInst *I) {
  auto inG = getGradHandle(ctx, I->getSrc());
  auto outG = getGradHandle(ctx, I->getDest());

  for (size_t i = 0, e = outG.size(); i < e; i++) {
    inG.raw(i) += outG.raw(i);
  }
}

void Interpreter::fwdConvolutionInst(Context *ctx, bool isTrain,
                                     ConvolutionInst *I) {
  auto inW = getWeightHandle(ctx, I->getSrc());
  auto outW = getWeightHandle(ctx, I->getDest());
  auto filterW = getWeightHandle(ctx, I->getFilter());
  auto biasW = getWeightHandle(ctx, I->getBias());

  size_t filterSize = I->getKernel();
  size_t pad = I->getPad();
  size_t stride = I->getStride();

  ShapeNHWC odim = outW.dims();
  ShapeNHWC idim = inW.dims();

  // For each input in the batch:
  for (size_t n = 0; n < idim.n; n++) {

    // For each layer in the output tensor:
    for (size_t d = 0; d < odim.c; d++) {

      // For each convolution 'jump' in the input tensor:
      ssize_t y = -ssize_t(pad);
      for (size_t ay = 0; ay < odim.w; y += stride, ay++) {
        ssize_t x = -ssize_t(pad);
        for (size_t ax = 0; ax < odim.h; x += stride, ax++) {

          // For each element in the convolution-filter:
          FloatTy sum = 0;
          for (size_t fy = 0; fy < filterSize; fy++) {
            for (size_t fx = 0; fx < filterSize; fx++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(odim.h) ||
                  oy >= ssize_t(odim.w)) {
                continue;
              }

              for (size_t fd = 0; fd < idim.c; fd++) {
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

void Interpreter::bwdConvolutionInst(Context *ctx, ConvolutionInst *I) {
  auto inW = getWeightHandle(ctx, I->getSrc());
  auto inG = getGradHandle(ctx, I->getSrc());
  auto outW = getWeightHandle(ctx, I->getDest());
  auto outG = getGradHandle(ctx, I->getDest());

  auto filterW = getWeightHandle(ctx, I->getFilter());
  auto filterG = getGradHandle(ctx, I->getFilter());
  auto biasG = getGradHandle(ctx, I->getBias());

  size_t filterSize = I->getKernel();
  size_t pad = I->getPad();
  size_t stride = I->getStride();

  ShapeNHWC odim = outW.dims();
  ShapeNHWC idim = inW.dims();

  // For each input in the batch:
  for (size_t n = 0; n < odim.n; n++) {

    // Compute the gradient. For each layer in the output tensor:
    for (size_t d = 0; d < odim.c; d++) {

      // For each convolution 'jump' in the input tensor:
      ssize_t y = -ssize_t(pad);
      for (size_t ay = 0; ay < odim.w; y += stride, ay++) {
        ssize_t x = -ssize_t(pad);
        for (size_t ax = 0; ax < odim.h; x += stride, ax++) {
          FloatTy chainGrad = outG.at({n, ax, ay, d});

          // For each element in the convolution-filter:
          for (size_t fy = 0; fy < filterSize; fy++) {
            for (size_t fx = 0; fx < filterSize; fx++) {
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

void Interpreter::fwdPoolInst(Context *ctx, bool isTrain, PoolInst *I) { DBG; }
void Interpreter::bwdPoolInst(Context *ctx, PoolInst *I) { DBG; }

//===----------------------------------------------------------------------===//
//                       Fully Connected
//===----------------------------------------------------------------------===//

void Interpreter::fwdFullyConnectedInst(Context *ctx, bool isTrain,
                                        FullyConnectedInst *I) {
  auto inW = getWeightHandle(ctx, I->getSrc());
  auto outW = getWeightHandle(ctx, I->getDest());

  auto odim = flattenCdr(outW.dims());
  auto idim = flattenCdr(inW.dims());

  auto filterW = getWeightHandle(ctx, I->getFilter());
  auto biasW = getWeightHandle(ctx, I->getBias());

  size_t inputSize = idim.second;

  for (size_t n = 0; n < odim.first; n++) {
    size_t base = inW.getElementPtr({n});

    for (size_t i = 0; i < odim.second; i++) {

      FloatTy sum = 0;
      for (size_t j = 0; j < inputSize; j++) {
        sum += inW.raw(base + j) * filterW.at({i, j});
      }

      sum += biasW.at({i});
      outW.at({n, i}) = sum;
    }
  } // N
}

void Interpreter::bwdFullyConnectedInst(Context *ctx, FullyConnectedInst *I) {
  auto inW = getWeightHandle(ctx, I->getSrc());
  auto inG = getGradHandle(ctx, I->getSrc());
  auto outW = getWeightHandle(ctx, I->getDest());
  auto outG = getGradHandle(ctx, I->getDest());

  auto odim = flattenCdr(outW.dims());
  auto idim = flattenCdr(inW.dims());

  auto filterW = getWeightHandle(ctx, I->getFilter());
  auto filterG = getGradHandle(ctx, I->getFilter());
  auto biasG = getGradHandle(ctx, I->getBias());

  size_t inSize = idim.second;

  for (size_t n = 0; n < odim.first; n++) {
    size_t base = inW.getElementPtr({n});

    // Compute the gradient:
    for (size_t i = 0; i < odim.second; i++) {
      FloatTy chainGrad = outG.at({n, i});

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

void Interpreter::fwdReluInst(Context *ctx, bool isTrain, ReluInst *I) {
  auto inW = getWeightHandle(ctx, I->getSrc());
  auto outW = getWeightHandle(ctx, I->getDest());

  for (size_t i = 0, e = inW.size(); i < e; i++) {
    FloatTy val = inW.raw(i);
    outW.raw(i) = val < 0 ? 0 : val;
  }
}

void Interpreter::bwdReluInst(Context *ctx, ReluInst *I) {
  auto inG = getGradHandle(ctx, I->getSrc());
  auto outW = getWeightHandle(ctx, I->getDest());
  auto outG = getGradHandle(ctx, I->getDest());

  for (size_t i = 0, e = outW.size(); i < e; i++) {
    FloatTy val = outW.raw(i);
    inG.raw(i) += (val <= 0 ? 0 : outG.raw(i));
  }
}

void Interpreter::fwdSigmoidInst(Context *ctx, bool isTrain, SigmoidInst *I) {
  auto inW = getWeightHandle(ctx, I->getSrc());
  auto outW = getWeightHandle(ctx, I->getDest());

  for (size_t i = 0, e = outW.size(); i < e; i++) {
    FloatTy val = inW.raw(i);
    outW.raw(i) = 1 / (1 + std::exp(-val));
  }
}
void Interpreter::bwdSigmoidInst(Context *ctx, SigmoidInst *I) {
  auto inG = getGradHandle(ctx, I->getSrc());
  auto outW = getWeightHandle(ctx, I->getDest());
  auto outG = getGradHandle(ctx, I->getDest());

  for (size_t i = 0, e = outW.size(); i < e; i++) {
    FloatTy val = outW.raw(i);
    inG.raw(i) += val * (1 - val) * outG.raw(i);
  }
}

void Interpreter::fwdTanhInst(Context *ctx, bool isTrain, TanhInst *I) { DBG; }
void Interpreter::bwdTanhInst(Context *ctx, TanhInst *I) { DBG; }

//===----------------------------------------------------------------------===//
//                        Loss Functions (Softmax/regression/...)
//===----------------------------------------------------------------------===//

void Interpreter::fwdSoftMaxInst(Context *ctx, bool isTrain, SoftMaxInst *I) {
  DBG;
}
void Interpreter::bwdSoftMaxInst(Context *ctx, SoftMaxInst *I) { DBG; }

void Interpreter::fwdRegressionInst(Context *ctx, bool isTrain,
                                    RegressionInst *I) {
  auto inW = getWeightHandle(ctx, I->getSrc());
  auto outW = getWeightHandle(ctx, I->getDest());

  for (size_t i = 0, e = inW.size(); i < e; i++) {
    outW.raw(i) = inW.raw(i);
  }
}

void Interpreter::bwdRegressionInst(Context *ctx, RegressionInst *I) {
  auto inW = getWeightHandle(ctx, I->getSrc());
  auto inG = getGradHandle(ctx, I->getSrc());
  auto expected = getTensorForValue(I->getExpected());

  auto idim = inW.dims();
  assert(idim.size() == 2 && "Input is expected to be a vector per input");

  auto e = expected->getHandle<FloatTy>();

  // For each input in the batch:
  for (size_t n = 0; n < idim[0]; n++) {

    for (size_t i = 0; i < idim[1]; i++) {
      FloatTy dy = inW.at({n, i}) - e.at({n, i});
      inG.at({n, i}) += dy;
    }
  } // N
}

//===----------------------------------------------------------------------===//
//                       Tensor shape (transpose/reshape/concat/...)
//===----------------------------------------------------------------------===//

void Interpreter::fwdTransposeInst(Context *ctx, bool isTrain,
                                   TransposeInst *I) {
  DBG;
}

void Interpreter::bwdTransposeInst(Context *ctx, TransposeInst *I) { DBG; }

void Interpreter::fwdReshapeInst(Context *ctx, bool isTrain, ReshapeInst *I) {
  DBG;
}
void Interpreter::bwdReshapeInst(Context *ctx, ReshapeInst *I) { DBG; }
void Interpreter::fwdConcatInst(Context *ctx, bool isTrain, ConcatInst *I) {
  DBG;
}
void Interpreter::bwdConcatInst(Context *ctx, ConcatInst *I) { DBG; }

//===----------------------------------------------------------------------===//
//                      Batch Normalization
//===----------------------------------------------------------------------===//

void Interpreter::fwdBatchNormalizationInst(Context *ctx, bool isTrain,
                                            BatchNormalizationInst *I) {
  DBG;
}
void Interpreter::bwdBatchNormalizationInst(Context *ctx,
                                            BatchNormalizationInst *I) {
  DBG;
}

//===----------------------------------------------------------------------===//
//                       Arithmetic operations
//===----------------------------------------------------------------------===//

void Interpreter::fwdArithmeticInst(Context *ctx, bool isTrain,
                                    ArithmeticInst *I) {
  DBG;
}

void Interpreter::bwdArithmeticInst(Context *ctx, ArithmeticInst *I) { DBG; }

#undef DBG
