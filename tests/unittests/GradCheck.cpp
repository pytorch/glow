#include "noether/Image.h"
#include "noether/Network.h"
#include "noether/Nodes.h"
#include "noether/Tensor.h"

#include "gtest/gtest.h"

#include <iostream>

using namespace noether;

/// Compute the regression loss for the tensor \p X with regard to Y.
FloatTy computeL2Loss(Tensor *X, Tensor *Y) {
  assert(X->dims().size() == 1 && "Invalid input dims");
  assert(X->dims() == Y->dims() && "Invalid input dims");
  auto xH = X->getHandle<FloatTy>();
  auto yH = Y->getHandle<FloatTy>();
  FloatTy loss = 0;

  for (size_t i = 0, e = X->size(); i < e; i++) {
    FloatTy dy = (xH.at({i}) - yH.at({i}));
    loss += 0.5 * dy * dy;
  }

  return loss;
}

/// \returns the error rate when comparing two grads.
FloatTy gradDiff(FloatTy G1, FloatTy G2) {
  return std::abs(G1 - G2) / std::abs(G1 + G2 + 1);
}

TEST(Network, gradientCheck_FC_Concat_RELU) {
  Network N;
  N.getConfig().batchSize = 10;

  size_t numInputElem = 20;
  size_t numOutputElem = 10;

  auto *A = N.createArrayNode(numInputElem);
  auto *B = N.createArrayNode(numInputElem);
  NodeBase *FA = N.createFullyConnectedNode(A, numOutputElem / 2);
  FA = N.createRELUNode(FA);

  NodeBase *FB = N.createFullyConnectedNode(B, numOutputElem / 2);
  FB = N.createRELUNode(FB);

  NodeBase *O = N.createConcatNode({FA, FB}, 0);
  auto *RN = N.createRegressionNode(O);

  Tensor inputs(ElemKind::FloatTy, {numInputElem});
  Tensor outputs(ElemKind::FloatTy, {numOutputElem});

  auto inputsH = inputs.getHandle<FloatTy>();
  auto outputsH = outputs.getHandle<FloatTy>();

  inputsH.randomize(100);
  outputsH.randomize(100);

  // Train the network.
  for (int i = 0; i < 10; i++) {
    N.train(RN, {A, RN}, {&inputs, &outputs});
  }

  // Clear the gradients of the first layer.
  A->getGradHandle(N.getMainContext()).clear();

  // Train the network just once to calculate the grads.
  N.train(RN, {A, RN}, {&inputs, &outputs});

  float delta = 0.001;

  auto analyticalGrads = A->getGradHandle(N.getMainContext()).clone();
  auto analyticalGradsH = analyticalGrads.getHandle<FloatTy>();

  for (size_t i = 0; i < analyticalGrads.size(); i++) {
    auto old = inputsH.at({i});

    // Calculate f(x+e):
    inputsH.at({i}) = old + delta;
    Tensor *res = N.infer(RN, {A}, {&inputs});
    auto plusLoss = computeL2Loss(&outputs, res);

    // Calculate f(x-e):
    inputsH.at({i}) = old - delta;
    res = N.infer(RN, {A}, {&inputs});
    auto minusLoss = computeL2Loss(&outputs, res);
    inputsH.at({i}) = old;

    auto numericGrad = (plusLoss - minusLoss) / (2 * delta);
    auto analyticalGrad = analyticalGradsH.at({i});

    auto err = gradDiff(analyticalGrad, numericGrad);

    // Make sure that the analytical and numerical gradients agree.
    EXPECT_LE(err, 0.01);
  }
}

TEST(Network, gradientCheck_Conv) {
  Network N;
  N.getConfig().batchSize = 10;

  size_t numDim = 10;
  size_t numOutputElem = 10;

  auto *A = N.createArrayNode({numDim, numDim, 1});
  NodeBase *O = N.createConvNode(A, 16, 5, 1, 2);
  O = N.createMaxPoolNode(O, MaxPoolNode::OpKind::kMax, 3, 3, 0);
  O = N.createFullyConnectedNode(O, numOutputElem);
  O = N.createRELUNode(O);
  auto *RN = N.createRegressionNode(O);

  Tensor inputs(ElemKind::FloatTy, {numDim, numDim, 1});
  Tensor outputs(ElemKind::FloatTy, {numOutputElem});

  auto inputsH = inputs.getHandle<FloatTy>();
  auto outputsH = outputs.getHandle<FloatTy>();

  inputsH.randomize(1);
  outputsH.randomize(1);

  // Train the network.
  for (int i = 0; i < 10; i++) {
    N.train(RN, {A, RN}, {&inputs, &outputs});
  }

  // Clear the gradients of the first layer.
  A->getGradHandle(N.getMainContext()).clear();

  // Train the network just once to calculate the grads.
  N.train(RN, {A, RN}, {&inputs, &outputs});

  float delta = 0.001;

  auto analyticalGrads = A->getGradHandle(N.getMainContext()).clone();
  auto analyticalGradsH = analyticalGrads.getHandle<FloatTy>();

  for (size_t i = 0; i < numDim; i++) {
    for (size_t j = 0; j < numDim; j++) {
      auto old = inputsH.at({i, j, 0});

      // Calculate f(x+e):
      inputsH.at({i, j, 0}) = old + delta;
      Tensor *res = N.infer(RN, {A}, {&inputs});
      auto plusLoss = computeL2Loss(&outputs, res);

      // Calculate f(x-e):
      inputsH.at({i, j, 0}) = old - delta;
      res = N.infer(RN, {A}, {&inputs});
      auto minusLoss = computeL2Loss(&outputs, res);
      inputsH.at({i, j, 0}) = old;

      auto numericGrad = (plusLoss - minusLoss) / (2 * delta);
      auto analyticalGrad = analyticalGradsH.at({i, j, 0});

      auto err = gradDiff(analyticalGrad, numericGrad);

      // Make sure that the analytical and numerical gradients agree.
      EXPECT_LE(err, 0.04);
    }
  }
}

TEST(Network, gradientCheck_AvgPool) {
  Network N;
  N.getConfig().batchSize = 10;

  size_t numDim = 10;
  size_t numOutputElem = 10;

  auto *A = N.createArrayNode({numDim, numDim, 1});
  NodeBase *O = N.createMaxPoolNode(A, MaxPoolNode::OpKind::kAvg, 3, 3, 0);
  O = N.createFullyConnectedNode(O, numOutputElem);
  auto *RN = N.createRegressionNode(O);

  Tensor inputs(ElemKind::FloatTy, {numDim, numDim, 1});
  Tensor outputs(ElemKind::FloatTy, {numOutputElem});

  auto inputsH = inputs.getHandle<FloatTy>();
  auto outputsH = outputs.getHandle<FloatTy>();

  inputsH.randomize(1);
  outputsH.randomize(1);

  // Train the network.
  for (int i = 0; i < 10; i++) {
    N.train(RN, {A, RN}, {&inputs, &outputs});
  }

  // Clear the gradients of the first layer.
  A->getGradHandle(N.getMainContext()).clear();

  // Train the network just once to calculate the grads.
  N.train(RN, {A, RN}, {&inputs, &outputs});

  float delta = 0.001;

  auto analyticalGrads = A->getGradHandle(N.getMainContext()).clone();
  auto analyticalGradsH = analyticalGrads.getHandle<FloatTy>();

  for (size_t i = 0; i < numDim; i++) {
    for (size_t j = 0; j < numDim; j++) {
      auto old = inputsH.at({i, j, 0});

      // Calculate f(x+e):
      inputsH.at({i, j, 0}) = old + delta;
      Tensor *res = N.infer(RN, {A}, {&inputs});
      auto plusLoss = computeL2Loss(&outputs, res);

      // Calculate f(x-e):
      inputsH.at({i, j, 0}) = old - delta;
      res = N.infer(RN, {A}, {&inputs});
      auto minusLoss = computeL2Loss(&outputs, res);
      inputsH.at({i, j, 0}) = old;

      auto numericGrad = (plusLoss - minusLoss) / (2 * delta);
      auto analyticalGrad = analyticalGradsH.at({i, j, 0});

      auto err = gradDiff(analyticalGrad, numericGrad);

      // Make sure that the analytical and numerical gradients agree.
      EXPECT_LE(err, 0.04);
    }
  }
}

TEST(Network, gradientCheck_batchNorm) {
  Network N;
  N.getConfig().batchSize = 2;

  size_t numDim = 5;
  size_t numOutputElem = numDim * numDim * 3;

  auto *A = N.createArrayNode({numDim, numDim, 3});

  NodeBase *O = N.createBatchNormalizationNode(A, 2, 0.0001, 0.9);
  O = N.createReshapeNode(O, {numDim * numDim * 3});
  auto *RN = N.createRegressionNode(O);

  Tensor inputs(ElemKind::FloatTy, {numDim, numDim, 3});
  Tensor outputs(ElemKind::FloatTy, {numOutputElem});

  auto inputsH = inputs.getHandle<FloatTy>();
  auto outputsH = outputs.getHandle<FloatTy>();

  inputsH.randomize(1);
  outputsH.randomize(1);

  for (int i = 0, e = inputsH.size(); i < e; i++) {
    inputsH.raw(i) *= 6;
    inputsH.raw(i) += 4;
  }

  // Train the network just once to calculate the grads.
  for (int i = 0; i < 30; i++) {
    N.train(RN, {A, RN}, {&inputs, &outputs});
  }
  // Clear the gradients of the last layer.
  A->getGradHandle(N.getMainContext()).clear();

  N.train(RN, {A, RN}, {&inputs, &outputs});

  auto analyticalGrads = A->getGradHandle(N.getMainContext()).clone();
  auto analyticalGradsH = analyticalGrads.getHandle<FloatTy>();

  float delta = 0.001;

  for (size_t i = 0; i < numDim; i++) {
    for (size_t j = 0; j < numDim; j++) {
      auto old = inputsH.at({i, j, 0});

      // Calculate f(x+e):
      inputsH.at({i, j, 0}) = old + delta;
      Tensor *res = N.infer(RN, {A}, {&inputs});
      auto plusLoss = computeL2Loss(&outputs, res);

      // Calculate f(x-e):
      inputsH.at({i, j, 0}) = old - delta;
      res = N.infer(RN, {A}, {&inputs});
      auto minusLoss = computeL2Loss(&outputs, res);
      inputsH.at({i, j, 0}) = old;

      auto numericGrad = (plusLoss - minusLoss) / (2 * delta);
      auto analyticalGrad = analyticalGradsH.at({i, j, 0});

      auto err = gradDiff(analyticalGrad, numericGrad);

      // Make sure that the analytical and numerical gradients agree.
      EXPECT_LE(err, 0.04);
    }
  }
}

TEST(Network, gradientCheck_Arithmetic) {
  Network N;
  N.getConfig().batchSize = 2;

  size_t numDim = 5;

  auto *A = N.createArrayNode({numDim});
  auto *B = N.createArrayNode({numDim});
  auto *C = N.createArrayNode({numDim});

  NodeBase *O = N.createArithmeticNode(A, B, ArithmeticNode::OpKind::kMul);
  O = N.createArithmeticNode(O, C, ArithmeticNode::OpKind::kAdd);
  auto *RN = N.createRegressionNode(O);

  Tensor iA(ElemKind::FloatTy, {numDim});
  Tensor iB(ElemKind::FloatTy, {numDim});
  Tensor iC(ElemKind::FloatTy, {numDim});
  Tensor outputs(ElemKind::FloatTy, {numDim});

  auto iAH = iA.getHandle<FloatTy>();
  auto iBH = iB.getHandle<FloatTy>();
  auto iCH = iC.getHandle<FloatTy>();
  auto outputsH = outputs.getHandle<FloatTy>();

  iAH.randomize(1);
  iBH.randomize(1);
  iCH.randomize(1);
  outputsH.randomize(1);

  // Train the network just once to calculate the grads.
  for (int i = 0; i < 30; i++) {
    N.train(RN, {A, B, C, RN}, {&iA, &iB, &iC, &outputs});
  }
  // Clear the gradients of the last layer.
  A->getGradHandle(N.getMainContext()).clear();
  B->getGradHandle(N.getMainContext()).clear();
  C->getGradHandle(N.getMainContext()).clear();

  N.train(RN, {A, B, C, RN}, {&iA, &iB, &iC, &outputs});

  auto check = [&](NodeBase *node, Tensor *t) {
    auto iH = t->getHandle<FloatTy>();

    auto analyticalGrads = node->getGradHandle(N.getMainContext()).clone();
    auto analyticalGradsH = analyticalGrads.getHandle<FloatTy>();

    float delta = 0.001;
    for (size_t i = 0; i < numDim; i++) {
      auto old = iH.at({i});

      // Calculate f(x+e):
      iH.at({i}) = old + delta;
      Tensor *res = N.infer(RN, {A, B, C, RN}, {&iA, &iB, &iC, &outputs});
      auto plusLoss = computeL2Loss(&outputs, res);

      // Calculate f(x-e):
      iH.at({i}) = old - delta;
      res = N.infer(RN, {A, B, C, RN}, {&iA, &iB, &iC, &outputs});
      auto minusLoss = computeL2Loss(&outputs, res);
      iH.at({i}) = old;

      auto numericGrad = (plusLoss - minusLoss) / (2 * delta);
      auto analyticalGrad = analyticalGradsH.at({i});

      auto err = gradDiff(analyticalGrad, numericGrad);

      // Make sure that the analytical and numerical gradients agree.
      EXPECT_LE(err, 0.04);
    }
  };

  check(A, &iA);
  check(B, &iB);
  check(C, &iC);
}
