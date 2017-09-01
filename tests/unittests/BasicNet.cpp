#include "glow/Network/Image.h"
#include "glow/Network/Network.h"
#include "glow/Network/Nodes.h"
#include "glow/Network/Tensor.h"
#include "glow/Support/Random.h"

#include "gtest/gtest.h"

#include <iostream>

using namespace glow;

TEST(Network, learnSingleValue) {
  // Learning a single input vector.
  Network N;
  N.getConfig().maxNumThreads = 1;
  N.getConfig().learningRate = 0.05;

  // Create a variable with 1 input, which is a vector of 4 elements.
  auto *A = N.createVariable({1, 4}, ElemKind::FloatTy);
  auto *E = N.createVariable({1, 4}, ElemKind::FloatTy);

  NodeBase *O = N.createFullyConnectedNode(A, 10);
  O = N.createSigmoidNode(O);
  O = N.createFullyConnectedNode(O, 4);
  O = N.createSigmoidNode(O);
  auto *RN = N.createRegressionNode(O, E);

  // Values for the input and output variables.
  Tensor inputs(ElemKind::FloatTy, {1, 4});
  Tensor expected(ElemKind::FloatTy, {1, 4});
  inputs.getHandle<FloatTy>() = {0.15, 0.15, 0.15, 0.15};
  expected.getHandle<FloatTy>() = {0.9, 0.9, 0.9, 0.9};

  // Train the network. Learn 1000 batches.
  N.train(RN, 1000, {A, E}, {&inputs, &expected});

  // Testing the output vector.

  auto res = N.infer(RN, {A}, {&inputs});
  auto RNWH = res->getHandle<FloatTy>();
  (void)RNWH;

  // Test the output:
  EXPECT_NEAR(RNWH.at({0, 0}), 0.9, 0.05);
}

TEST(Network, learnXor) {
  unsigned numInputs = 10;
  unsigned numTests = 10;

  // Learning the Xor function.
  Network N;
  N.getConfig().maxNumThreads = 1;
  N.getConfig().learningRate = 0.05;

  auto *A = N.createVariable({numInputs, 2}, ElemKind::FloatTy);
  auto *Ex = N.createVariable({numInputs, 1}, ElemKind::FloatTy);

  NodeBase *O = N.createFullyConnectedNode(A, 6);
  O = N.createRELUNode(O);
  O = N.createFullyConnectedNode(O, 1);
  O = N.createRELUNode(O);
  auto *RN = N.createRegressionNode(O, Ex);

  /// Prepare the training set and the testing set.
  Tensor trainingSet(ElemKind::FloatTy, {numInputs, 2});
  Tensor testingSet(ElemKind::FloatTy, {numInputs, 2});
  Tensor trainingLabels(ElemKind::FloatTy, {numInputs, 1});

  auto TS = trainingSet.getHandle<FloatTy>();
  auto TL = trainingLabels.getHandle<FloatTy>();
  auto TT = testingSet.getHandle<FloatTy>();

  // Prepare the training data:
  for (unsigned i = 0; i < numInputs; i++) {
    int a = i % 2;
    int b = (i >> 1) % 2;
    TS.at({i, 0}) = a;
    TS.at({i, 1}) = b;
    TL.at({i, 0}) = a ^ b;
  }

  // Train the network:
  N.train(RN, 2500, {A, Ex}, {&trainingSet, &trainingLabels});

  // Prepare the training data:
  for (unsigned i = 0; i < numTests; i++) {
    TT.at({i, 0}) = i % 2;
    TT.at({i, 1}) = (i >> 1) % 2;
  }

  auto res = N.infer(RN, {A}, {&trainingSet});
  auto resH = res->getHandle<FloatTy>();

  // Test the output:
  for (size_t i = 0; i < numTests; i++) {
    int a = TS.at({i, 0});
    int b = TS.at({i, 1});
    std::cout << "a = " << a << " b = " << b << " => " << resH.at({i, 0})
              << "\n";
    EXPECT_NEAR(resH.at({i, 0}), (a ^ b), 0.1);
  }
}

TEST(Network, regression) {
  // Testing the regression layer. This test takes the first element from the
  // input vector, adds one to it and places the result in the second element of
  // the output vector.
  const int numInputs = 4;

  Network N;
  N.getConfig().maxNumThreads = 1;
  auto *A = N.createVariable({1, numInputs}, ElemKind::FloatTy);
  auto *Ex = N.createVariable({1, numInputs}, ElemKind::FloatTy);

  NodeBase *O = N.createFullyConnectedNode(A, 4);
  O = N.createRELUNode(O);
  auto *RN = N.createRegressionNode(O, Ex);

  Tensor inputs(ElemKind::FloatTy, {1, numInputs});
  Tensor expected(ElemKind::FloatTy, {1, numInputs});
  auto I = inputs.getHandle<FloatTy>();
  auto E = expected.getHandle<FloatTy>();

  // Train the network:
  for (int iter = 0; iter < 1000; iter++) {
    float target = float(iter % 9);
    I = {target, 0., 0., 0.};
    E = {0., target + 1, 0., 0.};
    N.train(RN, 1, {A, Ex}, {&inputs, &expected});
  }

  // Verify the result of the regression layer.

  // Test the output:
  for (int iter = 0; iter < 5; iter++) {
    float target = iter % 9 + 1;
    I = {target, 0., 0., 0.};
    auto *res = N.infer(RN, {A}, {&inputs});

    auto resH = res->getHandle<FloatTy>();
    (void)resH;

    EXPECT_NEAR(I.at({0, 0}) + 1, resH.at({0, 1}), 0.1);
  }
}

unsigned numSamples = 100;

/// Generate data in two classes. The circle of dots that's close to the axis is
/// L0, and the rest of the dots, away from the axis are L1.
void generateCircleData(Tensor &coordinates, Tensor &labels) {
  auto C = coordinates.getHandle<FloatTy>();
  auto L = labels.getHandle<size_t>();

  for (size_t i = 0; i < numSamples / 2; i++) {
    float r = nextRand() * 0.4;
    float a = nextRand() * 3.141592 * 2;
    float y = r * sin(a);
    float x = r * cos(a);

    C.at({i * 2, 0u}) = x;
    C.at({i * 2, 1u}) = y;
    L.at({i * 2, 0}) = 1;

    r = nextRand() * 0.4 + 0.8;
    a = nextRand() * 3.141592 * 2;
    y = r * sin(a);
    x = r * cos(a);

    C.at({i * 2 + 1, 0u}) = x;
    C.at({i * 2 + 1, 1u}) = y;
    L.at({i * 2 + 1, 0}) = 0;
  }
}

/// Test the fully connected layer and the softmax function.
/// Example from:
/// http://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html
TEST(Network, circle) {
  // Testing the softmax layer.

  // Construct the network:
  Network N;
  N.getConfig().maxNumThreads = 1;
  N.getConfig().momentum = 0.9;
  N.getConfig().learningRate = 0.01;

  auto *A = N.createVariable({1, 2}, ElemKind::FloatTy);
  auto *S = N.createVariable({1, 1}, ElemKind::IndexTy);

  auto *FCL0 = N.createFullyConnectedNode(A, 6);
  auto *RL0 = N.createRELUNode(FCL0);
  auto *FCL1 = N.createFullyConnectedNode(RL0, 2);
  auto *RL1 = N.createRELUNode(FCL1);
  auto *SM = N.createSoftMaxNode(RL1, S);

  Tensor coordinates(ElemKind::FloatTy, {numSamples, 2});
  Tensor labels(ElemKind::IndexTy, {numSamples, 1});
  generateCircleData(coordinates, labels);

  // Training:
  N.train(SM, 4000, {A, S}, {&coordinates, &labels});

  // Print a diagram that depicts the network decision on a grid.

  for (int x = -10; x < 10; x++) {
    for (int y = -10; y < 10; y++) {
      // Load the inputs:
      Tensor sample(ElemKind::FloatTy, {1, 2});
      sample.getHandle<FloatTy>() = {float(x) / 10, float(y) / 10};

      auto res = N.infer(SM, {A}, {&sample});

      auto SMH = res->getHandle<FloatTy>();

      auto A = SMH.at({0, 0});
      auto B = SMH.at({0, 1});

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
    Tensor sample(ElemKind::FloatTy, {1, 2});
    sample.getHandle<FloatTy>() = {0., 0.};
    auto res = N.infer(SM, {A}, {&sample});
    auto SMH = res->getHandle<FloatTy>();
    auto A = SMH.at({0, 0});
    auto B = SMH.at({0, 1});
    EXPECT_LE(A, 0.1);
    EXPECT_GE(B, 0.9);
  }

  {
    // Far away dot must be one.
    Tensor sample(ElemKind::FloatTy, {1, 2});
    sample.getHandle<FloatTy>() = {1., 1.};
    auto res = N.infer(SM, {A}, {&sample});
    auto SMH = res->getHandle<FloatTy>();
    auto A = SMH.at({0, 0});
    auto B = SMH.at({0, 1});
    EXPECT_GE(A, 0.9);
    EXPECT_LE(B, 0.1);
  }
}

TEST(Network, learnSingleValueConcat) {
  // Learning inputs in two concatenated vectors.
  Network N;
  N.getConfig().maxNumThreads = 1;
  N.getConfig().learningRate = 0.05;

  // Left side of the network:
  Variable *A = N.createVariable({1, 4}, ElemKind::FloatTy);
  Variable *Ex = N.createVariable({1, 8}, ElemKind::FloatTy);

  NodeBase *L = N.createFullyConnectedNode(A, 4);
  L = N.createRELUNode(L);

  // Right side of the network:
  Variable *B = N.createVariable({1, 4}, ElemKind::FloatTy);
  NodeBase *R = N.createFullyConnectedNode(B, 4);
  R = N.createRELUNode(R);

  // Concat:
  auto *C = N.createConcatNode({L, R}, 1);
  auto *RN = N.createRegressionNode(C, Ex);

  Tensor inputs(ElemKind::FloatTy, {1, 4});
  Tensor expected(ElemKind::FloatTy, {1, 8});
  inputs.getHandle<FloatTy>() = {0.15, 0.15, 0.15, 0.15};
  expected.getHandle<FloatTy>() = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

  // Train the network:
  N.train(RN, 1000, {A, B, Ex}, {&inputs, &inputs, &expected});

  // Testing the output vector.
  auto res = N.infer(RN, {A}, {&inputs});
  auto RNWH = res->getHandle<FloatTy>();
  (void)RNWH;

  // Test the output:
  EXPECT_NEAR(RNWH.at({0, 0}), 0.9, 0.1);
}
