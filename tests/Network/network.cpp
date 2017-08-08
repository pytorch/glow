#include "noether/Image.h"
#include "noether/Network.h"
#include "noether/Nodes.h"
#include "noether/Tensor.h"

#include "gtest/gtest.h"

#include <iostream>
#include <random>

using namespace noether;

TEST(Network, learnSingleValue) {
  // Learning a single input vector.
  Network N;
  N.getConfig().learningRate = 0.05;

  auto *A = N.createArrayNode(4);
  auto *FCL0 = N.createFullyConnectedNode(A, 10);
  auto *RL0 = N.createRELUNode(FCL0);
  auto *FCL1 = N.createFullyConnectedNode(RL0, 4);
  auto *RL1 = N.createRELUNode(FCL1);
  auto *RN = N.createRegressionNode(RL1);

  Tensor inputs(ElemKind::FloatTy, {4});
  Tensor expected(ElemKind::FloatTy, {4});

  inputs.getHandle<FloatTy>() = {0.15, 0.15, 0.15, 0.15};
  expected.getHandle<FloatTy>() = {0.9, 0.9, 0.9, 0.9};

  // Train the network:
  for (int iter = 0; iter < 1000; iter++) {
    N.train(RN, {A, RN}, {&inputs, &expected});
  }

  // Testing the output vector.

  auto res = N.infer(RN, {A}, {&inputs});
  auto RNWH = res->getHandle<FloatTy>();
  (void)RNWH;

  // Test the output:
  EXPECT_NEAR(RNWH.at({0}), 0.9, 0.1);
}

TEST(Network, learnXor) {
  // Learning the Xor function.

  Network N;
  N.getConfig().learningRate = 0.1;

  auto *A = N.createArrayNode(2);
  auto *FCL0 = N.createFullyConnectedNode(A, 6);
  auto *RL0 = N.createRELUNode(FCL0);
  auto *FCL1 = N.createFullyConnectedNode(RL0, 1);
  auto *RL1 = N.createRELUNode(FCL1);
  auto *RN = N.createRegressionNode(RL1);

  Tensor inputs(ElemKind::FloatTy, {4, 2});
  Tensor expected(ElemKind::FloatTy, {4, 1});

  auto I = inputs.getHandle<FloatTy>();
  auto E = expected.getHandle<FloatTy>();

  /// The XOR lookup table:
  I.at({0, 0}) = 0;
  I.at({0, 1}) = 0;
  I.at({1, 0}) = 0;
  I.at({1, 1}) = 1;
  I.at({2, 0}) = 1;
  I.at({2, 1}) = 0;
  I.at({3, 0}) = 1;
  I.at({3, 1}) = 1;
  // Xor result:
  E.at({0, 0}) = 0;
  E.at({1, 0}) = 1;
  E.at({2, 0}) = 1;
  E.at({3, 0}) = 0;

  // Train the network:
  N.train(RN, 400, {A, RN}, {&inputs, &expected});

  // Testing the output vector.
  for (size_t i = 0; i < 4; i++) {
    Tensor in = I.extractSlice(i);
    Tensor exp = E.extractSlice(i);
    auto expH = exp.getHandle<FloatTy>();
    (void)expH;

    auto res = N.infer(RN, {A}, {&in});
    auto resH = res->getHandle<FloatTy>();
    (void)resH;

    // Test the output:
    EXPECT_NEAR(expH.at({0}), resH.at({0}), 0.1);
  }
}


TEST(Network, regression) {
  // Testing the regression layer.
  /// This test takes the first element from the input vector, adds one to it
  /// and places the result in the second element of the output vector.
  constexpr int numInputs = 4;

  Network N;
  auto *A = N.createArrayNode({numInputs});
  auto *FCL0 = N.createFullyConnectedNode(A, 4);
  auto *RL0 = N.createRELUNode(FCL0);
  auto *RN = N.createRegressionNode(RL0);

  Tensor inputs(ElemKind::FloatTy, {numInputs});
  Tensor expected(ElemKind::FloatTy, {numInputs});
  auto I = inputs.getHandle<FloatTy>();
  auto E = expected.getHandle<FloatTy>();

  // Train the network:
  for (int iter = 0; iter < 9000; iter++) {
    float target = float(iter % 9);
    I = {target, 0., 0., 0.};
    E = {0., target + 1, 0., 0.};
    N.train(RN, {A, RN}, {&inputs, &expected});
  }
    
  // Verify the result of the regression layer.

  // Test the output:
  for (int iter = 0; iter < 5; iter++) {
    float target = iter % 9 + 1;
    I = {target, 0., 0., 0.};
    auto *res = N.infer(RN, {A}, {&inputs});

    auto resH = res->getHandle<FloatTy>();
    (void)resH;

    EXPECT_NEAR(I.at({0}) + 1, resH.at({1}), 0.1);
  }
 }

unsigned numSamples = 100;

/// Generate data in two classes. The circle of dots that's close to the axis is
/// L0, and the rest of the dots, away from the axis are L1.
void generateCircleData(Tensor &coordinates, Tensor &labels) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> r_radius(0, 0.4);
  std::uniform_real_distribution<> r_angle(0, 3.14159 * 2);

  auto C = coordinates.getHandle<FloatTy>();
  auto L = labels.getHandle<size_t>();

  for (size_t i = 0; i < numSamples / 2; i++) {
    float r = r_radius(gen);
    float a = r_angle(gen);
    float y = r * sin(a);
    float x = r * cos(a);

    C.at({i * 2, 0u}) = x;
    C.at({i * 2, 1u}) = y;
    L.at({i * 2}) = 1;

    r = r_radius(gen) + 0.8;
    a = r_angle(gen);
    y = r * sin(a);
    x = r * cos(a);

    C.at({i * 2 + 1, 0u}) = x;
    C.at({i * 2 + 1, 1u}) = y;
    L.at({i * 2 + 1}) = 0;
  }
}

/// Test the fully connected layer and the softmax function.
/// Example from:
/// http://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html
TEST(Network, circle) {
  // Testing the softmax layer.

  // Construct the network:
  Network N;
  N.getConfig().momentum = 0.0;
  N.getConfig().learningRate = 0.1;
  N.getConfig().batchSize = 10;

  auto *A = N.createArrayNode({2});
  auto *FCL0 = N.createFullyConnectedNode(A, 6);
  auto *RL0 = N.createRELUNode(FCL0);
  auto *FCL1 = N.createFullyConnectedNode(RL0, 2);
  auto *RL1 = N.createRELUNode(FCL1);
  auto *SM = N.createSoftMaxNode(RL1);

  Tensor coordinates(ElemKind::FloatTy, {numSamples, 2});
  Tensor labels(ElemKind::IndexTy, {numSamples});
  generateCircleData(coordinates, labels);

  // Training:
  for (int iter = 0; iter < 2000; iter++) {
    N.train(SM, 1, {A, SM}, {&coordinates, &labels});
  }

  // Print a diagram that depicts the network decision on a grid.

    for (int x = -10; x < 10; x++) {
      for (int y = -10; y < 10; y++) {
        // Load the inputs:
        Tensor sample(ElemKind::FloatTy, {2});
        sample.getHandle<FloatTy>() = {float(x) / 10, float(y) / 10};

        auto res = N.infer(SM, {A}, {&sample});

        auto SMH = res->getHandle<FloatTy>();

        auto A = SMH.at({0});
        auto B = SMH.at({1});

        char ch = '=';
        if (A > (B + 0.2)) {
          ch = '+';
        } else if (B > (A + 0.2)) {
          ch = '-';
        }

        std::cout << ch;
      }
      std::cout << "\n";
    }
    std::cout << "\n";


    {
    // The dot in the middle must be zero.
    Tensor sample(ElemKind::FloatTy, {2});
    sample.getHandle<FloatTy>() = {0., 0.};
    auto res = N.infer(SM, {A}, {&sample});
    auto SMH = res->getHandle<FloatTy>();
    auto A = SMH.at({0});
    auto B = SMH.at({1});
    EXPECT_LE(A, 0.1);
    EXPECT_GE(B, 0.9);
    }

    {
    // Far away dot must be one. 
    Tensor sample(ElemKind::FloatTy, {2});
    sample.getHandle<FloatTy>() = {1., 1.};
    auto res = N.infer(SM, {A}, {&sample});
    auto SMH = res->getHandle<FloatTy>();
    auto A = SMH.at({0});
    auto B = SMH.at({1});
    EXPECT_GE(A, 0.9);
    EXPECT_LE(B, 0.1);
    }
}

TEST(Network, learnSingleValueConcat) {
  // Learning inputs in two concatenated vectors.
  Network N;
  N.getConfig().learningRate = 0.05;

  // Left side of the network:
  NodeBase *A = N.createArrayNode(4);
  A = N.createFullyConnectedNode(A, 4);
  A = N.createRELUNode(A);

  // Right side of the network:
  NodeBase *B = N.createArrayNode(4);
  B = N.createFullyConnectedNode(B, 4);
  B = N.createRELUNode(B);

  // Concat:
  auto *C = N.createConcatNode({A, B}, 0);
  auto *RN = N.createRegressionNode(C);

  Tensor inputs(ElemKind::FloatTy, {4});
  Tensor expected(ElemKind::FloatTy, {8});
  inputs.getHandle<FloatTy>() = {0.15, 0.15, 0.15, 0.15};
  expected.getHandle<FloatTy>() = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

  // Train the network:
  for (int iter = 0; iter < 1000; iter++) {
    N.train(RN, {A, B, RN}, {&inputs, & inputs, &expected});
  }

  // Testing the output vector.
  auto res = N.infer(RN, {A}, {&inputs});
  auto RNWH = res->getHandle<FloatTy>();
  (void)RNWH;

  // Test the output:
  EXPECT_NEAR(RNWH.at({0}), 0.9, 0.1);
}

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

TEST(Network, gradientCheck_FC_Concat_RELU) {
  // Learning a single input vector.
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
  auto outputsH = inputs.getHandle<FloatTy>();

  inputsH.randomize(100);
  outputsH.randomize(100);

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

    auto numericGrad = (plusLoss - minusLoss)/(2 * delta);
    auto analyticalGrad = analyticalGradsH.at({i});

    auto err = std::abs(analyticalGrad - numericGrad) /
               std::abs(analyticalGrad + numericGrad);

    // Make sure that the analytical and numerical gradients agree.
    EXPECT_LE(err, 0.01);
  }
}
