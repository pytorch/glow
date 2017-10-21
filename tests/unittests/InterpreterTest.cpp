// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"

#include "gtest/gtest.h"

#include <cassert>
#include <string>

using namespace glow;

TEST(Interpreter, interpret) {
  ExecutionEngine EE;

  Tensor inputs(ElemKind::FloatTy, {1, 32, 32, 3});

  auto &G = EE.getGraph();
  auto *input = G.createVariable(ElemKind::FloatTy, {1, 32, 32, 3}, "input");

  auto *ex = G.createVariable(ElemKind::IndexTy, {1, 1}, "exp");

  auto *CV0 = G.createConv("conv", input, 16, 5, 1, 2);
  auto *RL0 = G.createRELU("relu", CV0);
  auto *MP0 = G.createPool("pool", RL0, PoolNode::OpKind::Max, 2, 2, 0);

  auto *CV1 = G.createConv("conv", MP0, 20, 5, 1, 2);
  auto *RL1 = G.createRELU("relu", CV1);
  auto *MP1 = G.createPool("pool", RL1, PoolNode::OpKind::Max, 2, 2, 0);

  auto *CV2 = G.createConv("conv", MP1, 20, 5, 1, 2);
  auto *RL2 = G.createRELU("relu", CV2);
  auto *MP2 = G.createPool("pool", RL2, PoolNode::OpKind::Max, 2, 2, 0);

  auto *FCL1 = G.createFullyConnected("fc", MP2, 10);
  auto *RL3 = G.createRELU("relu", FCL1);
  auto *SM = G.createSoftMax("sm", RL3, ex);
  G.createSave("ret", SM);

  EE.compile(OptimizationMode::Infer);
  EE.infer({input}, {&inputs});
}

TEST(Interpreter, trainASimpleNetwork) {
  ExecutionEngine EE;
  // Learning a single input vector.
  EE.getConfig().maxNumThreads = 1;
  EE.getConfig().learningRate = 0.05;

  auto &G = EE.getGraph();

  // Create a variable with 1 input, which is a vector of 4 elements.
  auto *A = G.createVariable(ElemKind::FloatTy, {1, 4}, "A");
  auto *E = G.createVariable(ElemKind::FloatTy, {1, 4}, "E");

  Node *O = G.createFullyConnected("fc", A, 10);
  O = G.createSigmoid("sig", O);
  O = G.createFullyConnected("fc", O, 4);
  O = G.createSigmoid("sig", O);
  O = G.createRegression("reg", O, E);
  auto *result = G.createSave("return", O);

  // Values for the input and output variables.
  Tensor inputs(ElemKind::FloatTy, {1, 4});
  Tensor expected(ElemKind::FloatTy, {1, 4});
  inputs.getHandle<>() = {0.15, 0.15, 0.15, 0.15};
  expected.getHandle<>() = {0.9, 0.9, 0.9, 0.9};

  EE.compile(OptimizationMode::Train);

  // Train the network. Learn 1000 batches.
  EE.train(1000, {A, E}, {&inputs, &expected});

  // Testing the output vector.

  EE.optimize(OptimizationMode::Infer);
  EE.infer({A}, {&inputs});
  auto RNWH = result->getOutput()->getPayload().getHandle<>();
  (void)RNWH;

  // Test the output:
  EXPECT_NEAR(RNWH.at({0, 0}), 0.9, 0.05);
}

TEST(Interpreter, simpleRegression) {
  // Testing the regression layer. This test takes the first element from the
  // input vector, adds one to it and places the result in the second element of
  // the output vector.
  const int numInputs = 4;

  // Learning the Xor function.
  ExecutionEngine EE;

  // Learning a single input vector.
  EE.getConfig().maxNumThreads = 1;
  EE.getConfig().learningRate = 0.05;

  Tensor inputs(ElemKind::FloatTy, {1, numInputs});
  Tensor expected(ElemKind::FloatTy, {1, numInputs});

  auto &G = EE.getGraph();
  auto *A = G.createVariable(ElemKind::FloatTy, {1, numInputs}, "A");
  auto *Ex = G.createVariable(ElemKind::FloatTy, {1, numInputs}, "E");
  Node *O = G.createFullyConnected("fc", A, 4);
  O = G.createRELU("relu", O);
  O = G.createRegression("reg", O, Ex);
  auto *result = G.createSave("result", O);

  auto I = inputs.getHandle<>();
  auto E = expected.getHandle<>();

  EE.compile(OptimizationMode::Train);

  // Train the network:
  for (int iter = 0; iter < 1000; iter++) {
    float target = float(iter % 9);
    I = {target, 0., 0., 0.};
    E = {0., target + 1, 0., 0.};
    EE.train(1, {A, Ex}, {&inputs, &expected});
  }

  // Verify the result of the regression layer.

  // Test the output:
  for (int iter = 0; iter < 5; iter++) {
    float target = iter % 9 + 1;
    I = {target, 0., 0., 0.};
    EE.infer({A}, {&inputs});

    auto resH = result->getOutput()->getPayload().getHandle<>();
    (void)resH;

    EXPECT_NEAR(I.at({0, 0}) + 1, resH.at({0, 1}), 0.1);
  }
}

TEST(Interpreter, learnXor) {
  unsigned numInputs = 10;
  unsigned numTests = 10;

  // Learning the Xor function.
  ExecutionEngine EE;

  // Learning a single input vector.
  EE.getConfig().maxNumThreads = 1;
  EE.getConfig().learningRate = 0.05;

  auto &G = EE.getGraph();

  auto *A = G.createVariable(ElemKind::FloatTy, {numInputs, 2}, "A");
  auto *Ex = G.createVariable(ElemKind::FloatTy, {numInputs, 1}, "Ex");

  Node *O = G.createFullyConnected("fc", A, 6);
  O = G.createRELU("relu", O);
  O = G.createFullyConnected("fc", O, 1);
  O = G.createRELU("relu", O);
  O = G.createRegression("reg", O, Ex);
  auto *result = G.createSave("ret", O);

  /// Prepare the training set and the testing set.
  Tensor trainingSet(ElemKind::FloatTy, {numInputs, 2});
  Tensor testingSet(ElemKind::FloatTy, {numInputs, 2});
  Tensor trainingLabels(ElemKind::FloatTy, {numInputs, 1});

  auto TS = trainingSet.getHandle<>();
  auto TL = trainingLabels.getHandle<>();
  auto TT = testingSet.getHandle<>();

  // Prepare the training data:
  for (unsigned i = 0; i < numInputs; i++) {
    int a = i % 2;
    int b = (i >> 1) % 2;
    TS.at({i, 0}) = a;
    TS.at({i, 1}) = b;
    TL.at({i, 0}) = a ^ b;
  }

  EE.compile(OptimizationMode::Train);

  // Train the network:
  EE.train(2500, {A, Ex}, {&trainingSet, &trainingLabels});

  // Prepare the testing tensor:
  for (unsigned i = 0; i < numTests; i++) {
    TT.at({i, 0}) = i % 2;
    TT.at({i, 1}) = (i >> 1) % 2;
  }

  EE.infer({A}, {&trainingSet});
  auto resH = result->getOutput()->getPayload().getHandle<>();

  // Test the output:
  for (size_t i = 0; i < numTests; i++) {
    int a = TS.at({i, 0});
    int b = TS.at({i, 1});
    std::cout << "a = " << a << " b = " << b << " => " << resH.at({i, 0})
              << "\n";
    EXPECT_NEAR(resH.at({i, 0}), (a ^ b), 0.1);
  }
}

unsigned numSamples = 100;

/// Generate data in two classes. The circle of dots that's close to the axis is
/// L0, and the rest of the dots, away from the axis are L1.
void generateCircleData(Tensor &coordinates, Tensor &labels) {
  auto C = coordinates.getHandle<>();
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
  ExecutionEngine EE;

  // Learning a single input vector.
  EE.getConfig().maxNumThreads = 1;
  EE.getConfig().momentum = 0.9;
  EE.getConfig().learningRate = 0.01;

  auto &G = EE.getGraph();
  auto *A = G.createVariable(ElemKind::FloatTy, {1, 2}, "A");
  auto *S = G.createVariable(ElemKind::IndexTy, {1, 1}, "S",
                             Variable::InitKind::Extern);

  auto *FCL0 = G.createFullyConnected("fc", A, 8);
  auto *RL0 = G.createRELU("relu", FCL0);
  auto *FCL1 = G.createFullyConnected("fc", RL0, 2);
  auto *RL1 = G.createRELU("relu", FCL1);
  auto *SM = G.createSoftMax("soft", RL1, S);
  auto *result = G.createSave("ret", SM);

  EE.compile(OptimizationMode::Train);

  Tensor coordinates(ElemKind::FloatTy, {numSamples, 2});
  Tensor labels(ElemKind::IndexTy, {numSamples, 1});
  generateCircleData(coordinates, labels);

  // Training:
  EE.train(4000, {A, S}, {&coordinates, &labels});

  // Print a diagram that depicts the network decision on a grid.
  for (int x = -10; x < 10; x++) {
    for (int y = -10; y < 10; y++) {
      // Load the inputs:
      Tensor sample(ElemKind::FloatTy, {1, 2});
      sample.getHandle<>() = {float(x) / 10, float(y) / 10};

      EE.infer({A}, {&sample});

      auto SMH = result->getOutput()->getPayload().getHandle<>();
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
    sample.getHandle<>() = {0., 0.};
    EE.infer({A}, {&sample});
    auto SMH = result->getOutput()->getPayload().getHandle<>();
    auto A = SMH.at({0, 0});
    auto B = SMH.at({0, 1});
    EXPECT_LE(A, 0.1);
    EXPECT_GE(B, 0.9);
  }

  {
    // Far away dot must be one.
    Tensor sample(ElemKind::FloatTy, {1, 2});
    sample.getHandle<>() = {1., 1.};
    EE.infer({A}, {&sample});
    auto SMH = result->getOutput()->getPayload().getHandle<>();
    auto A = SMH.at({0, 0});
    auto B = SMH.at({0, 1});
    EXPECT_GE(A, 0.9);
    EXPECT_LE(B, 0.1);
  }
}

TEST(Network, learnSingleValueConcat) {
  ExecutionEngine EE;
  unsigned width = 6;

  // Learning a single input vector.
  EE.getConfig().maxNumThreads = 1;
  EE.getConfig().momentum = 0.9;
  EE.getConfig().learningRate = 0.01;

  auto &G = EE.getGraph();

  auto *Ex = G.createVariable(ElemKind::FloatTy, {1, width * 2}, "Ex");

  // Left side of the network:
  auto *A = G.createVariable(ElemKind::FloatTy, {1, width}, "A");
  Node *L = G.createFullyConnected("fc", A, width);
  L = G.createSigmoid("", L);

  // Right side of the network:
  auto *B = G.createVariable(ElemKind::FloatTy, {1, width}, "B");
  Node *R = G.createFullyConnected("fc", B, width);
  R = G.createSigmoid("sig", R);

  // Concat:
  auto *C = G.createConcat("con", {L, R}, 1);
  auto *RN = G.createRegression("reg", C, Ex);
  auto *result = G.createSave("ret", RN);

  Tensor inputs(ElemKind::FloatTy, {1, width});
  Tensor expected(ElemKind::FloatTy, {1, width * 2});
  inputs.getHandle<>().clear(0.15);
  expected.getHandle<>().clear(0.9);

  EE.compile(OptimizationMode::Train);

  // Train the network:
  EE.train(1000, {A, B, Ex}, {&inputs, &inputs, &expected});

  EE.optimize(OptimizationMode::Infer);

  // Testing the output vector.
  EE.infer({A}, {&inputs});
  auto RNWH = result->getOutput()->getPayload().getHandle<>();
  (void)RNWH;

  // Test the output:
  EXPECT_NEAR(RNWH.at({0, 0}), 0.9, 0.1);
}
