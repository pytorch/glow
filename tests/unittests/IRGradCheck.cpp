#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "glow/Interpreter/Interpreter.h"
#include "glow/Network/Tensor.h"

#include "gtest/gtest.h"

#include <iostream>

using namespace glow;

/// Compute the regression loss for the tensor \p X with regard to Y.
FloatTy computeL2Loss(Tensor *X, Tensor *Y) {
  assert(X->dims() == Y->dims() && "Invalid input dims");
  auto xH = X->getHandle<FloatTy>();
  auto yH = Y->getHandle<FloatTy>();
  FloatTy loss = 0;

  for (size_t i = 0, e = X->size(); i < e; i++) {
    FloatTy dy = (xH.raw(i) - yH.raw(i));
    loss += 0.5 * dy * dy;
  }

  return loss;
}

/// \returns the error rate when comparing two grads.
FloatTy gradDiff(FloatTy G1, FloatTy G2) {
  return std::abs(G1 - G2) / std::abs(G1 + G2 + 1);
}

void performGradCheck(Interpreter &IP, Value *result, Value *inputVar,
                      Value *expVar, Tensor *inputs, Tensor *outputs,
                      float delta, float allowedError) {
  auto inputsH = inputs->getHandle<FloatTy>();

  // Train the network.
  IP.train(300, {inputVar, expVar}, {inputs, outputs});

  // Clear the gradients of the first layer.
  IP.getGradHandle(nullptr, inputVar).clear();

  // Train the network just once to calculate the grads.
  IP.train(1, {inputVar, expVar}, {inputs, outputs});

  auto analyticalGradsH = IP.getGradHandle(nullptr, inputVar);

  for (size_t i = 0; i < analyticalGradsH.size(); i++) {
    auto old = inputsH.raw(i);

    // Calculate f(x+e):
    inputsH.raw(i) = old + delta;
    IP.infer({inputVar}, {inputs});
    Tensor *res = IP.getTensorForValue(result);
    auto plusLoss = computeL2Loss(outputs, res);

    // Calculate f(x-e):
    inputsH.raw(i) = old - delta;
    IP.infer({inputVar}, {inputs});
    res = IP.getTensorForValue(result);
    auto minusLoss = computeL2Loss(outputs, res);
    inputsH.raw(i) = old;

    auto numericGrad = (plusLoss - minusLoss) / (2 * delta);
    auto analyticalGrad = analyticalGradsH.raw(i);

    auto err = gradDiff(analyticalGrad, numericGrad);

    // Make sure that the analytical and numerical gradients agree.
    EXPECT_LE(err, allowedError);
  }
}

TEST(Network, gradientCheck_FC_Concat_RELU) {
  Interpreter IP;
  IP.getConfig().maxNumThreads = 1;

  size_t numInputElem = 20;
  size_t numOutputElem = 10;

  Value *result;
  Value *A;
  Value *Exp;

  {
    IRBuilder bb(IP.getModule());

    A = bb.createWeightVar(ElemKind::FloatTy, {1, numInputElem});
    Exp = bb.createWeightVar(ElemKind::FloatTy, {1, numOutputElem});

    Instruction *FA = bb.createFullyConnectedOp(A, numOutputElem / 2);
    FA = bb.createRELUOp(*FA);

    auto *B = bb.createWeightVar(ElemKind::FloatTy, {1, numInputElem});
    Instruction *FB = bb.createFullyConnectedOp(B, numOutputElem / 2);
    FB = bb.createRELUOp(*FB);

    Value *v0 = *FA;
    Value *v1 = *FB;
    Instruction *O = bb.createConcatOp({v0, v1}, 1);
    O = bb.createRegressionOp(*O, Exp);
    result = bb.createReturnOp(*O);
  }

  IP.getModule().verify();
  IP.initVars();

  Tensor inputs(ElemKind::FloatTy, {{1, numInputElem}});
  Tensor outputs(ElemKind::FloatTy, {{1, numOutputElem}});

  auto inputsH = inputs.getHandle<FloatTy>();
  auto outputsH = outputs.getHandle<FloatTy>();

  inputsH.randomize(100);
  outputsH.randomize(100);

  performGradCheck(IP, result, A, Exp, &inputs, &outputs, 0.001, 0.001);
}

TEST(Network, gradientCheck_Conv) {
  Interpreter IP;
  IP.getConfig().maxNumThreads = 1;

  size_t numDim = 10;
  size_t numOutputElem = 10;

  Value *A;
  Value *Ex;
  Value *result;
  {
    IRBuilder bb(IP.getModule());

    A = bb.createWeightVar(ElemKind::FloatTy, {1, numDim, numDim, 1});
    Ex = bb.createWeightVar(ElemKind::FloatTy, {1, numOutputElem});

    Instruction *O = bb.createConvOp(A, 16, 5, 1, 2);
    O = bb.createPoolOp(*O, PoolInst::OpKind::kMax, 3, 3, 0);
    O = bb.createFullyConnectedOp(*O, numOutputElem);
    O = bb.createRELUOp(*O);
    O = bb.createRegressionOp(*O, Ex);
    result = bb.createReturnOp(*O);
  }

  IP.getModule().verify();
  IP.initVars();

  Tensor inputs(ElemKind::FloatTy, {1, numDim, numDim, 1});
  Tensor outputs(ElemKind::FloatTy, {1, numOutputElem});

  auto inputsH = inputs.getHandle<FloatTy>();
  auto outputsH = outputs.getHandle<FloatTy>();

  inputsH.randomize(1);
  outputsH.randomize(1);

  performGradCheck(IP, result, A, Ex, &inputs, &outputs, 0.001, 0.004);
}

TEST(Network, gradientCheck_AvgPool) {
  Interpreter IP;
  IP.getConfig().maxNumThreads = 1;

  size_t numDim = 10;
  size_t numOutputElem = 10;

  Value *A;
  Value *Exp;
  Value *result;
  {
    IRBuilder bb(IP.getModule());

    A = bb.createWeightVar(ElemKind::FloatTy, {1, numDim, numDim, 1});
    Exp = bb.createWeightVar(ElemKind::FloatTy, {1, numOutputElem});

    Instruction *O = bb.createPoolOp(A, PoolInst::OpKind::kAvg, 3, 3, 0);
    O = bb.createFullyConnectedOp(*O, numOutputElem);
    O = bb.createRegressionOp(*O, Exp);
    result = bb.createReturnOp(*O);
  }

  IP.getModule().verify();
  IP.getModule().dump();
  IP.initVars();

  Tensor inputs(ElemKind::FloatTy, {1, numDim, numDim, 1});
  Tensor outputs(ElemKind::FloatTy, {1, numOutputElem});

  auto inputsH = inputs.getHandle<FloatTy>();
  auto outputsH = outputs.getHandle<FloatTy>();

  inputsH.randomize(1);
  outputsH.randomize(1);

  performGradCheck(IP, result, A, Exp, &inputs, &outputs, 0.001, 0.004);
}

TEST(Network, gradientCheck_batchNorm) {
  Interpreter IP;
  IP.getConfig().maxNumThreads = 1;

  size_t numDim = 5;
  size_t numOutputElem = numDim * numDim * 3;

  Value *A;
  Value *Ex;
  Value *result;
  {
    IRBuilder bb(IP.getModule());

    A = bb.createWeightVar(ElemKind::FloatTy, {1, numDim, numDim, 3});
    Ex = bb.createWeightVar(ElemKind::FloatTy, {1, numOutputElem});

    Instruction *O = bb.createBatchNormalizationOp(A, 3, 0.0001, 0.9);
    O = bb.createReshapeOp(*O, {1, numDim * numDim * 3});
    O = bb.createRegressionOp(*O, Ex);
    result = bb.createReturnOp(*O);
  }

  IP.getModule().verify();
  IP.initVars();

  Tensor inputs(ElemKind::FloatTy, {1, numDim, numDim, 3});
  Tensor outputs(ElemKind::FloatTy, {1, numOutputElem});

  auto inputsH = inputs.getHandle<FloatTy>();
  auto outputsH = outputs.getHandle<FloatTy>();

  inputsH.randomize(1);
  outputsH.randomize(1);

  for (int i = 0, e = inputsH.size(); i < e; i++) {
    inputsH.raw(i) *= 6;
    inputsH.raw(i) += 4;
  }

  performGradCheck(IP, result, A, Ex, &inputs, &outputs, 0.001, 0.004);
}

TEST(Network, gradientCheck_Arithmetic) {
  Interpreter IP;
  IP.getConfig().maxNumThreads = 1;

  size_t numDim = 5;

  Value *result;
  Value *A;
  Value *B;
  Value *C;
  Value *Exp;

  {
    IRBuilder bb(IP.getModule());

    A = bb.createWeightVar(ElemKind::FloatTy, {1, numDim});
    B = bb.createWeightVar(ElemKind::FloatTy, {1, numDim});
    C = bb.createWeightVar(ElemKind::FloatTy, {1, numDim});
    Exp = bb.createWeightVar(ElemKind::FloatTy, {1, numDim});

    Instruction *O = bb.createArithmeticOp(A, B, ArithmeticInst::OpKind::kMul);
    O = bb.createArithmeticOp(*O, C, ArithmeticInst::OpKind::kAdd);
    O = bb.createRegressionOp(*O, Exp);
    result = bb.createReturnOp(*O);
  }

  IP.getModule().verify();
  IP.initVars();

  Tensor iA(ElemKind::FloatTy, {1, numDim});
  Tensor iB(ElemKind::FloatTy, {1, numDim});
  Tensor iC(ElemKind::FloatTy, {1, numDim});
  Tensor outputs(ElemKind::FloatTy, {1, numDim});

  auto iAH = iA.getHandle<FloatTy>();
  auto iBH = iB.getHandle<FloatTy>();
  auto iCH = iC.getHandle<FloatTy>();
  auto outputsH = outputs.getHandle<FloatTy>();

  iAH.randomize(1);
  iBH.randomize(1);
  iCH.randomize(1);
  outputsH.randomize(1);

  // Train the network just once to calculate the grads.
  IP.train(30, {A, B, C, Exp}, {&iA, &iB, &iC, &outputs});

  // Clear the gradients of the last layer.
  IP.getGradHandle(nullptr, A).clear();
  IP.getGradHandle(nullptr, B).clear();
  IP.getGradHandle(nullptr, C).clear();

  IP.train(1, {A, B, C, Exp}, {&iA, &iB, &iC, &outputs});

  auto check = [&](Value *var, Tensor *t) {
    auto iH = t->getHandle<FloatTy>();

    auto analyticalGradsH = IP.getGradHandle(nullptr, var);

    float delta = 0.001;
    for (size_t i = 0; i < numDim; i++) {
      auto old = iH.at({0, i});

      // Calculate f(x+e):
      iH.at({0, i}) = old + delta;
      IP.infer({A, B, C, Exp}, {&iA, &iB, &iC, &outputs});
      Tensor *res = IP.getTensorForValue(result);

      auto plusLoss = computeL2Loss(&outputs, res);

      // Calculate f(x-e):
      iH.at({0, i}) = old - delta;
      IP.infer({A, B, C, Exp}, {&iA, &iB, &iC, &outputs});
      res = IP.getTensorForValue(result);
      auto minusLoss = computeL2Loss(&outputs, res);
      iH.at({0, i}) = old;

      auto numericGrad = (plusLoss - minusLoss) / (2 * delta);
      auto analyticalGrad = analyticalGradsH.at({0, i});

      auto err = gradDiff(analyticalGrad, numericGrad);

      // Make sure that the analytical and numerical gradients agree.
      EXPECT_LE(err, 0.04);
    }
  };

  check(A, &iA);
  check(B, &iB);
  check(C, &iC);
}

TEST(Network, gradientCheck_FC_Concat_Tanh) {
  // Using the same gradient check test setup as gradientCheck_FC_Concat_RELU
  Interpreter IP;
  IP.getConfig().maxNumThreads = 1;

  size_t numInputElem = 20;
  size_t numOutputElem = 10;

  Value *A;
  Value *Exp;
  Value *result;
  {
    IRBuilder bb(IP.getModule());
    A = bb.createWeightVar(ElemKind::FloatTy, {1, numInputElem});
    Exp = bb.createWeightVar(ElemKind::FloatTy, {1, numOutputElem});

    Instruction *FA = bb.createFullyConnectedOp(A, numOutputElem);
    FA = bb.createTanhOp(*FA);
    FA = bb.createRegressionOp(*FA, Exp);
    result = bb.createReturnOp(*FA);
  }

  IP.getModule().verify();
  IP.initVars();

  Tensor inputs(ElemKind::FloatTy, {{1, numInputElem}});
  Tensor outputs(ElemKind::FloatTy, {{1, numOutputElem}});

  auto inputsH = inputs.getHandle<FloatTy>();
  auto outputsH = outputs.getHandle<FloatTy>();

  inputsH.randomize(100);
  outputsH.randomize(100);

  performGradCheck(IP, result, A, Exp, &inputs, &outputs, 0.0001, 0.001);
}

TEST(Network, gradientCheck_Transpose) {
  // Using the same gradient check test setup as gradientCheck_FC_Concat_RELU
  Interpreter IP;
  IP.getConfig().maxNumThreads = 1;
  size_t numOutputElem = 10;

  Value *A;
  Value *Exp;
  Value *result;
  {
    IRBuilder bb(IP.getModule());

    A = bb.createWeightVar(ElemKind::FloatTy, {1, 5, 10, 15});
    Exp = bb.createWeightVar(ElemKind::FloatTy, {1, numOutputElem});
    Instruction *TA = bb.createTransposeOp(A, {0, 3, 1, 2});
    TA = bb.createFullyConnectedOp(*TA, numOutputElem);
    TA = bb.createRegressionOp(*TA, Exp);
    result = bb.createReturnOp(*TA);
  }

  IP.getModule().verify();
  IP.initVars();

  Tensor inputs(ElemKind::FloatTy, {1, 5, 10, 15});
  Tensor outputs(ElemKind::FloatTy, {1, numOutputElem});

  auto inputsH = inputs.getHandle<FloatTy>();
  auto outputsH = outputs.getHandle<FloatTy>();

  inputsH.randomize(100);
  outputsH.randomize(100);

  performGradCheck(IP, result, A, Exp, &inputs, &outputs, 0.0001, 0.001);
}
