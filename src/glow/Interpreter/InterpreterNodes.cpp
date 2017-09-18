#include "glow/IR/Instrs.h"
#include "glow/Interpreter/Interpreter.h"
#include "glow/Support/Casting.h"

using namespace glow;

#define DBG std::cout << __FUNCTION__ << "\n";
void Interpreter::fwdCopyInst(Context *ctx, bool isTrain, CopyInst *I) {
  auto S = getTensorHandle(I->getSrc());
  auto D = getTensorHandle(I->getDest());

  for (size_t i = 0, e = S.size(); i < e; i++) {
    D.raw(i) = S.raw(i);
  }
}
void Interpreter::fwdConvolutionInst(Context *ctx, bool isTrain,
                                     ConvolutionInst *I) {
  auto inW = getTensorHandle(I->getSrc());
  auto outW = getTensorHandle(I->getDest());
  auto filterW = getTensorHandle(I->getFilter());
  auto biasW = getTensorHandle(I->getBias());

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

void Interpreter::fwdPoolInst(Context *ctx, bool isTrain, PoolInst *I) { DBG; }
void Interpreter::fwdFullyConnectedInst(Context *ctx, bool isTrain,
                                        FullyConnectedInst *I) {
  DBG;
}
void Interpreter::fwdReluInst(Context *ctx, bool isTrain, ReluInst *I) { DBG; }
void Interpreter::fwdSigmoidInst(Context *ctx, bool isTrain, SigmoidInst *I) {
  DBG;
}
void Interpreter::fwdTanhInst(Context *ctx, bool isTrain, TanhInst *I) { DBG; }
void Interpreter::fwdSoftMaxInst(Context *ctx, bool isTrain, SoftMaxInst *I) {
  DBG;
}
void Interpreter::fwdRegressionInst(Context *ctx, bool isTrain,
                                    RegressionInst *I) {
  DBG;
}
void Interpreter::fwdTransposeInst(Context *ctx, bool isTrain,
                                   TransposeInst *I) {
  DBG;
}
void Interpreter::fwdReshapeInst(Context *ctx, bool isTrain, ReshapeInst *I) {
  DBG;
}
void Interpreter::fwdConcatInst(Context *ctx, bool isTrain, ConcatInst *I) {
  DBG;
}
void Interpreter::fwdBatchNormalizationInst(Context *ctx, bool isTrain,
                                            BatchNormalizationInst *I) {
  DBG;
}
void Interpreter::fwdArithmeticInst(Context *ctx, bool isTrain,
                                    ArithmeticInst *I) {
  DBG;
}
