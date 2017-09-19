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

void Interpreter::fwdPoolInst(Context *ctx, bool isTrain, PoolInst *I) {
  if (I->getKind() == PoolInst::OpKind::kMax) {
    return fwdPoolMax_impl(ctx, I);
  }

  return fwdPoolAvg_impl(ctx, I);
}
void Interpreter::fwdPoolMax_impl(Context *ctx, PoolInst *I) {
  auto inW = getWeightHandle(ctx, I->getSrc());
  auto outW = getWeightHandle(ctx, I->getDest());

  ShapeNHWC odim = outW.dims();
  ShapeNHWC idim = inW.dims();

  auto pad = I->getPad();
  auto filterSize = I->getKernel();
  auto stride = I->getStride();

  auto SXY = getTensorForValue(I->srcXY())->getHandle<size_t>();

  // For each input in the batch:
  for (size_t n = 0; n < odim.n; n++) {

    // For each layer in the output tensor:
    for (size_t z = 0; z < idim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t y = -ssize_t(pad);
      for (size_t ay = 0; ay < odim.w; y += stride, ay++) {
        ssize_t x = -ssize_t(pad);
        for (size_t ax = 0; ax < odim.h; x += stride, ax++) {
          size_t maxX = x;
          size_t maxY = y;

          bool first = true;
          FloatTy max = 0;

          for (size_t fy = 0; fy < filterSize; fy++) {
            for (size_t fx = 0; fx < filterSize; fx++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                  oy >= ssize_t(idim.w)) {
                continue;
              }

              FloatTy val = inW.at({n, (size_t)ox, (size_t)oy, z});

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

void Interpreter::fwdPoolAvg_impl(Context *ctx, PoolInst *I) {
  auto inW = getWeightHandle(ctx, I->getSrc());
  auto outW = getWeightHandle(ctx, I->getDest());

  ShapeNHWC odim = outW.dims();
  ShapeNHWC idim = inW.dims();

  auto pad = I->getPad();
  auto filterSize = I->getKernel();
  auto stride = I->getStride();

  // Implement the avg pooling operation as defined here:
  // https://arxiv.org/abs/1312.4400

  FloatTy filterArea = filterSize * filterSize;

  // For each input in the batch:
  for (size_t n = 0; n < odim.n; n++) {

    // For each layer in the output tensor:
    for (size_t z = 0; z < idim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t y = -ssize_t(pad);
      for (size_t ay = 0; ay < odim.w; y += stride, ay++) {
        ssize_t x = -ssize_t(pad);
        for (size_t ax = 0; ax < odim.h; x += stride, ax++) {
          FloatTy sum = 0;

          for (size_t fy = 0; fy < filterSize; fy++) {
            for (size_t fx = 0; fx < filterSize; fx++) {
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

void Interpreter::bwdPoolInst(Context *ctx, PoolInst *I) {
  if (I->getKind() == PoolInst::OpKind::kMax) {
    return bwdPoolMax_impl(ctx, I);
  }

  return bwdPoolAvg_impl(ctx, I);
}

void Interpreter::bwdPoolMax_impl(Context *ctx, PoolInst *I) {
  auto inG = getGradHandle(ctx, I->getSrc());
  auto outW = getWeightHandle(ctx, I->getDest());
  auto outG = getGradHandle(ctx, I->getDest());

  ShapeNHWC odim = outW.dims();

  auto SXY = getTensorForValue(I->srcXY())->getHandle<size_t>();

  // For each input in the batch:
  for (size_t n = 0; n < odim.n; n++) {

    // Compute the gradient. For each layer in the output tensor:
    for (size_t z = 0; z < odim.c; z++) {

      // For each convolution 'jump' in the input tensor:
      for (size_t ay = 0; ay < odim.w; ay++) {
        for (size_t ax = 0; ax < odim.h; ax++) {

          FloatTy chainGrad = outG.at({n, (size_t)ax, (size_t)ay, z});

          size_t maxX = SXY.at({n, (size_t)ax, (size_t)ay, z, 0});
          size_t maxY = SXY.at({n, (size_t)ax, (size_t)ay, z, 1});

          inG.at({n, maxX, maxY, z}) += chainGrad;
        } // H
      }   // W
    }     // C
  }       // N
}

void Interpreter::bwdPoolAvg_impl(Context *ctx, PoolInst *I) {
  auto inW = getWeightHandle(ctx, I->getSrc());
  auto inG = getGradHandle(ctx, I->getSrc());
  auto outW = getWeightHandle(ctx, I->getDest());
  auto outG = getGradHandle(ctx, I->getDest());

  ShapeNHWC odim = outW.dims();
  ShapeNHWC idim = inW.dims();

  auto pad = I->getPad();
  auto filterSize = I->getKernel();
  auto stride = I->getStride();

  FloatTy filterArea = filterSize * filterSize;

  // For each input in the batch:
  for (size_t n = 0; n < odim.n; n++) {

    // For each layer in the output tensor:
    for (size_t z = 0; z < odim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t y = -ssize_t(pad);
      for (size_t ay = 0; ay < odim.w; y += stride, ay++) {
        ssize_t x = -ssize_t(pad);
        for (size_t ax = 0; ax < odim.h; x += stride, ax++) {
          FloatTy dy = outG.at({n, ax, ay, z}) / filterArea;

          for (size_t fy = 0; fy < filterSize; fy++) {
            for (size_t fx = 0; fx < filterSize; fx++) {
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

void Interpreter::fwdTanhInst(Context *ctx, bool isTrain, TanhInst *I) {
  auto inW = getWeightHandle(ctx, I->getSrc());
  auto outW = getWeightHandle(ctx, I->getDest());

  for (size_t i = 0, e = inW.size(); i < e; i++) {
    FloatTy val = inW.raw(i);
    FloatTy exp_val = std::exp(val);
    FloatTy exp_neg_val = std::exp(-val);
    outW.raw(i) = (exp_val - exp_neg_val) / (exp_val + exp_neg_val);
  }
}
void Interpreter::bwdTanhInst(Context *ctx, TanhInst *I) {
  auto inG = getGradHandle(ctx, I->getSrc());
  auto outW = getWeightHandle(ctx, I->getDest());
  auto outG = getGradHandle(ctx, I->getDest());

  for (size_t i = 0, e = outW.size(); i < e; i++) {
    FloatTy val = outW.raw(i);
    inG.raw(i) += (1 - val * val) * outG.raw(i);
  }
}

//===----------------------------------------------------------------------===//
//                        Loss Functions (Softmax/regression/...)
//===----------------------------------------------------------------------===//

void Interpreter::fwdSoftMaxInst(Context *ctx, bool isTrain, SoftMaxInst *I) {
  auto inW = getWeightHandle(ctx, I->getSrc());
  auto outW = getWeightHandle(ctx, I->getDest());
  auto idim = inW.dims();

  auto EH = getWeightHandle(ctx, I->getE());

  for (size_t n = 0; n < idim[0]; n++) {
    FloatTy max = inW.at({n, 0});

    // Find Max.
    for (size_t i = 0; i < idim[1]; i++) {
      max = std::max(max, inW.at({n, i}));
    }

    FloatTy sum = 0;

    // Compute exp.
    for (size_t i = 0; i < idim[1]; i++) {
      FloatTy e = std::exp(inW.at({n, i}) - max);
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
void Interpreter::bwdSoftMaxInst(Context *ctx, SoftMaxInst *I) {
  auto inG = getGradHandle(ctx, I->getSrc());

  auto idim = inG.dims();
  auto EH = getTensorForValue(I->getE())->getHandle<FloatTy>();
  auto selectedH = getTensorForValue(I->getSelected())->getHandle<size_t>();

  // http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
  // https://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network
  for (size_t n = 0; n < idim[0]; n++) {
    for (size_t i = 0; i < idim[1]; i++) {
      FloatTy delta = (selectedH.at({n, 0}) == i);
      FloatTy sigma = (EH.at({n, i}) - delta);
      inG.at({n, i}) += sigma;
    }
  }
}

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
  auto inW = getWeightHandle(ctx, I->getSrc());
  auto outW = getTensorForValue(I->getDest());

  assert(outW->size() == inW.size() && "Invalid tensor dimensions");
  inW.transpose(outW, I->getShuffle());
}

void Interpreter::bwdTransposeInst(Context *ctx, TransposeInst *I) {
  auto inG = getOrCreateGradTensor(I->getSrc());
  auto outG = getGradHandle(ctx, I->getDest());

  assert(outG.size() == inG->size() && "Invalid tensor dimensions");

  // Generate the reverse shuffle.
  auto shuffle = I->getShuffle();
  std::vector<unsigned> reverseShuffle = shuffle.vec();
  for (unsigned int i = 0; i < shuffle.size(); i++) {
    reverseShuffle[shuffle[i]] = i;
  }

  // Perform the reverse transpsose.
  // TODO: this wipes out the gradients and may cause a bug for operators with
  // multiple users.
  outG.transpose(inG, reverseShuffle);
}

void Interpreter::fwdReshapeInst(Context *ctx, bool isTrain, ReshapeInst *I) {
  auto inW = getWeightHandle(ctx, I->getSrc());
  auto outW = getWeightHandle(ctx, I->getDest());
  for (size_t i = 0, e = inW.size(); i < e; i++) {
    outW.raw(i) = inW.raw(i);
  }
}
void Interpreter::bwdReshapeInst(Context *ctx, ReshapeInst *I) {
  auto inG = getGradHandle(ctx, I->getSrc());
  auto outW = getWeightHandle(ctx, I->getDest());
  auto outG = getGradHandle(ctx, I->getDest());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    inG.raw(i) += outG.raw(i);
  }
}

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
