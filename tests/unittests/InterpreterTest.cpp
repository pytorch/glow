// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"

#include "gtest/gtest.h"

#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <string>

using namespace glow;

TEST(Interpreter, interpret) {
  ExecutionEngine EE;

  Tensor inputs(ElemKind::FloatTy, {1, 32, 32, 3});

  auto &G = EE.getGraph();
  G.setName("interpret");
  auto *input = G.createVariable(ElemKind::FloatTy, {1, 32, 32, 3}, "input");

  auto *ex = G.createVariable(ElemKind::IndexTy, {1, 1}, "exp");

  auto *CV0 = G.createConv("conv1", input, 16, 5, 1, 2);
  auto *RL0 = G.createRELU("relu1", CV0);
  auto *MP0 = G.createPool("pool1", RL0, PoolNode::Mode::Max, 2, 2, 0);

  auto *CV1 = G.createConv("conv2", MP0, 20, 5, 1, 2);
  auto *RL1 = G.createRELU("relu2", CV1);
  auto *MP1 = G.createPool("pool2", RL1, PoolNode::Mode::Max, 2, 2, 0);

  auto *CV2 = G.createConv("conv3", MP1, 20, 5, 1, 2);
  auto *RL2 = G.createRELU("relu3", CV2);
  auto *MP2 = G.createPool("pool3", RL2, PoolNode::Mode::Max, 2, 2, 0);

  auto *FCL1 = G.createFullyConnected("fc", MP2, 10);
  auto *RL3 = G.createRELU("relu4", FCL1);
  auto *SM = G.createSoftMax("sm", RL3, ex);
  G.createSave("ret", SM);

  EE.compile(CompilationMode::Infer);

  /// Add a debug_action instruction to check that it can be
  /// processed by the interpreter.
  auto &M = EE.getModule();
  IRBuilder builder(&M);
  builder.createDebugPrintInst("print1", *M.getWeights().begin());

  EE.run({input}, {&inputs});
}

TEST(Interpreter, trainASimpleNetwork) {
  ExecutionEngine EE;
  // Learning a single input vector.
  EE.getConfig().maxNumThreads = 1;
  EE.getConfig().learningRate = 0.05;

  auto &G = EE.getGraph();
  G.setName("trainASimpleNetwork");

  // Create a variable with 1 input, which is a vector of 4 elements.
  auto *A = G.createVariable(ElemKind::FloatTy, {1, 4}, "A");
  auto *E = G.createVariable(ElemKind::FloatTy, {1, 4}, "E");

  Node *O = G.createFullyConnected("fc1", A, 10);
  O = G.createSigmoid("sig1", O);
  O = G.createFullyConnected("fc2", O, 4);
  O = G.createSigmoid("sig2", O);
  O = G.createRegression("reg", O, E);
  auto *result = G.createSave("return", O);

  // Values for the input and output variables.
  Tensor inputs(ElemKind::FloatTy, {1, 4});
  Tensor expected(ElemKind::FloatTy, {1, 4});
  inputs.getHandle<>() = {0.15, 0.15, 0.15, 0.15};
  expected.getHandle<>() = {0.9, 0.9, 0.9, 0.9};

  EE.compile(CompilationMode::Train);

  // Train the network. Learn 1000 batches.
  EE.runBatch(1000, {A, E}, {&inputs, &expected});

  // Testing the output vector.

  EE.compile(CompilationMode::Infer);
  EE.run({A}, {&inputs});
  auto RNWH = result->getVariable()->getPayload().getHandle<>();
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
  G.setName("simpleRegression");
  auto *A = G.createVariable(ElemKind::FloatTy, {1, numInputs}, "A");
  auto *Ex = G.createVariable(ElemKind::FloatTy, {1, numInputs}, "E");
  Node *O = G.createFullyConnected("fc", A, 4);
  O = G.createRELU("relu", O);
  O = G.createRegression("reg", O, Ex);
  auto *result = G.createSave("result", O);

  auto I = inputs.getHandle<>();
  auto E = expected.getHandle<>();

  EE.compile(CompilationMode::Train);

  // Train the network:
  for (int iter = 0; iter < 1000; iter++) {
    float target = float(iter % 9);
    I = {target, 0., 0., 0.};
    E = {0., target + 1, 0., 0.};
    EE.runBatch(1, {A, Ex}, {&inputs, &expected});
  }

  // Verify the result of the regression layer.
  EE.compile(CompilationMode::Infer);

  // Test the output:
  for (int iter = 0; iter < 5; iter++) {
    float target = iter % 9 + 1;
    I = {target, 0., 0., 0.};
    EE.run({A}, {&inputs});

    auto resH = result->getVariable()->getPayload().getHandle<>();
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
  G.setName("learnXor");

  auto *A = G.createVariable(ElemKind::FloatTy, {numInputs, 2}, "A");
  auto *Ex = G.createVariable(ElemKind::FloatTy, {numInputs, 1}, "Ex");

  Node *O = G.createFullyConnected("fc1", A, 6);
  O = G.createRELU("relu1", O);
  O = G.createFullyConnected("fc2", O, 1);
  O = G.createRELU("relu2", O);
  O = G.createRegression("reg", O, Ex);
  auto *result = G.createSave("ret", O);

  // Prepare the training set and the testing set.
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

  EE.compile(CompilationMode::Train);

  // Train the network:
  EE.runBatch(2500, {A, Ex}, {&trainingSet, &trainingLabels});

  EE.compile(CompilationMode::Infer);

  // Prepare the testing tensor:
  for (unsigned i = 0; i < numTests; i++) {
    TT.at({i, 0}) = i % 2;
    TT.at({i, 1}) = (i >> 1) % 2;
  }

  EE.run({A}, {&trainingSet});
  auto resH = result->getVariable()->getPayload().getHandle<>();

  // Test the output:
  for (size_t i = 0; i < numTests; i++) {
    int a = TS.at({i, 0});
    int b = TS.at({i, 1});
    llvm::outs() << "a = " << a << " b = " << b << " => " << resH.at({i, 0})
                 << "\n";
    EXPECT_NEAR(resH.at({i, 0}), (a ^ b), 0.1);
  }
}

unsigned numSamples = 230;

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

  unsigned minibatchSize = 11;

  auto &G = EE.getGraph();
  G.setName("circle");
  auto *A = G.createVariable(ElemKind::FloatTy, {minibatchSize, 2}, "A");
  auto *S = G.createVariable(ElemKind::IndexTy, {minibatchSize, 1}, "S",
                             Variable::InitKind::Extern);

  auto *FCL0 = G.createFullyConnected("fc1", A, 8);
  auto *T0 = G.createTanh("tanh1", FCL0);
  auto *FCL1 = G.createFullyConnected("fc2", T0, 2);
  auto *T1 = G.createTanh("tanh2", FCL1);
  auto *SM = G.createSoftMax("soft", T1, S);
  auto *result = G.createSave("ret", SM);

  EE.compile(CompilationMode::Train);

  Tensor coordinates(ElemKind::FloatTy, {numSamples, 2});
  Tensor labels(ElemKind::IndexTy, {numSamples, 1});
  generateCircleData(coordinates, labels);

  // Training:
  EE.runBatch(4000, {A, S}, {&coordinates, &labels});

  EE.compile(CompilationMode::Infer);

  // Print a diagram that depicts the network decision on a grid.
  Tensor sample(ElemKind::FloatTy, {minibatchSize, 2});
  sample.zero();
  for (int x = -10; x < 10; x++) {
    for (int y = -10; y < 10; y++) {
      // Load the inputs:
      sample.getHandle<>().at({0, 0}) = float(x) / 10;
      sample.getHandle<>().at({0, 1}) = float(y) / 10;

      EE.run({A}, {&sample});

      auto SMH = result->getVariable()->getPayload().getHandle<>();
      auto A = SMH.at({0, 0});
      auto B = SMH.at({0, 1});

      char ch = '=';
      if (A > (B + 0.2)) {
        ch = '+';
      } else if (B > (A + 0.2)) {
        ch = '-';
      }

      llvm::outs() << ch;
    }
    llvm::outs() << "\n";
  }
  llvm::outs() << "\n";

  {
    // The dot in the middle must be one.
    sample.getHandle<>().at({0, 0}) = 0;
    sample.getHandle<>().at({0, 1}) = 0;
    EE.run({A}, {&sample});
    auto SMH = result->getVariable()->getPayload().getHandle<>();
    auto A = SMH.at({0, 0});
    auto B = SMH.at({0, 1});
    EXPECT_TRUE(B > (A + 0.2));
  }

  {
    // Far away dot must be zero.
    sample.getHandle<>().at({0, 0}) = 1;
    sample.getHandle<>().at({0, 1}) = 1;
    EE.run({A}, {&sample});
    auto SMH = result->getVariable()->getPayload().getHandle<>();
    auto A = SMH.at({0, 0});
    auto B = SMH.at({0, 1});
    EXPECT_TRUE(A > (B + 0.2));
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
  G.setName("learnSingleValueConcat");

  auto *Ex = G.createVariable(ElemKind::FloatTy, {1, width * 2}, "Ex");

  // Left side of the network:
  auto *A = G.createVariable(ElemKind::FloatTy, {1, width}, "A");
  Node *L = G.createFullyConnected("fc1", A, width);
  L = G.createSigmoid("", L);

  // Right side of the network:
  auto *B = G.createVariable(ElemKind::FloatTy, {1, width}, "B");
  Node *R = G.createFullyConnected("fc2", B, width);
  R = G.createSigmoid("sig", R);

  // Concat:
  auto *C = G.createConcat("con", {L, R}, 1);
  auto *RN = G.createRegression("reg", C, Ex);
  auto *result = G.createSave("ret", RN);

  Tensor inputs(ElemKind::FloatTy, {1, width});
  Tensor expected(ElemKind::FloatTy, {1, width * 2});
  inputs.getHandle<>().clear(0.15);
  expected.getHandle<>().clear(0.9);

  EE.compile(CompilationMode::Train);

  // Train the network:
  EE.runBatch(1000, {A, B, Ex}, {&inputs, &inputs, &expected});

  EE.compile(CompilationMode::Infer);

  // Testing the output vector.
  EE.run({A}, {&inputs});
  auto RNWH = result->getVariable()->getPayload().getHandle<>();
  (void)RNWH;

  // Test the output:
  EXPECT_NEAR(RNWH.at({0, 0}), 0.9, 0.1);
}

TEST(Network, concatVectors) {
  ExecutionEngine EE;

  auto &G = EE.getGraph();
  G.setName("concatVectors");

  auto *V1 = G.createVariable(ElemKind::IndexTy, {10}, "V1");
  auto *V2 = G.createVariable(ElemKind::IndexTy, {20}, "V2");
  auto *V3 = G.createVariable(ElemKind::IndexTy, {30}, "V3");

  Node *L = G.createConcat("concat", {V1, V2, V3}, 0);
  auto *result = G.createSave("ret", L);

  Tensor I1(ElemKind::IndexTy, {10});
  Tensor I2(ElemKind::IndexTy, {20});
  Tensor I3(ElemKind::IndexTy, {30});

  for (size_t i = 0; i < 10; i++) {
    I1.getHandle<size_t>().at({i}) = i;

    I2.getHandle<size_t>().at({i}) = i + 10;
    I2.getHandle<size_t>().at({i + 10}) = i + 20;
    I3.getHandle<size_t>().at({i}) = i + 30;
    I3.getHandle<size_t>().at({i + 10}) = i + 40;
    I3.getHandle<size_t>().at({i + 20}) = i + 40;
  }

  EE.compile(CompilationMode::Infer);

  // Testing the output vector.
  EE.run({V1, V2, V3}, {&I1, &I2, &I3});
  auto RNWH = result->getVariable()->getPayload().getHandle<size_t>();
  (void)RNWH;

  for (size_t i = 0; i < 50; i++) {
    EXPECT_NEAR(RNWH.at({i}), i, 0.001);
  }
}

TEST(Network, sliceVectors) {
  ExecutionEngine EE;

  auto &G = EE.getGraph();
  G.setName("sliceVectors");

  auto *V = G.createVariable(ElemKind::IndexTy, {3, 30}, "V");

  Node *S1 = G.createSlice("slice1", V, {0, 10}, {3, 13});
  Node *S2 = G.createSlice("slice2", V, {1, 10}, {2, 30});
  Node *S3 = G.createSlice("slice3", V, {2, 10}, {3, 12});

  auto *result1 = G.createSave("ret1", S1);
  auto *result2 = G.createSave("ret2", S2);
  auto *result3 = G.createSave("ret3", S3);

  Tensor I(ElemKind::IndexTy, {3, 30});

  for (size_t j = 0; j < 30; j++) {
    I.getHandle<size_t>().at({0, j}) = j;
    I.getHandle<size_t>().at({1, j}) = j + 30;
    I.getHandle<size_t>().at({2, j}) = j + 60;
  }

  EE.compile(CompilationMode::Infer);

  // Testing the output slices.
  EE.run({V}, {&I});
  auto RNWH1 = result1->getVariable()->getPayload().getHandle<size_t>();
  (void)RNWH1;
  auto RNWH2 = result2->getVariable()->getPayload().getHandle<size_t>();
  (void)RNWH2;
  auto RNWH3 = result3->getVariable()->getPayload().getHandle<size_t>();
  (void)RNWH3;

  EXPECT_EQ(3, RNWH1.dims()[0]);
  EXPECT_EQ(3, RNWH1.dims()[1]);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 10; j < 13; j++) {
      EXPECT_NEAR(RNWH1.at({i, j - 10}), j + i * 30, 0.001);
    }
  }
  EXPECT_EQ(1, RNWH2.dims()[0]);
  EXPECT_EQ(20, RNWH2.dims()[1]);
  for (size_t j = 10; j < 30; j++) {
    EXPECT_NEAR(RNWH2.at({0, j - 10}), j + 30, 0.001);
  }
  EXPECT_EQ(1, RNWH3.dims()[0]);
  EXPECT_EQ(2, RNWH3.dims()[1]);
  for (size_t j = 10; j < 12; j++) {
    EXPECT_NEAR(RNWH3.at({0, j - 10}), j + 60, 0.001);
  }
}

TEST(Network, trainASimpleRNN) {
  ExecutionEngine EE;
  // Learning a single input vector.
  EE.getConfig().maxNumThreads = 1;
  EE.getConfig().learningRate = 0.05;

  auto &G = EE.getGraph();
  G.setName("trainASimpleRNN");

  // Create a variable with 1 input, which is 3 consecutive vectors
  // of 4 elements each.
  auto *X = G.createVariable(ElemKind::FloatTy, {1, 3, 4}, "X",
                             Variable::InitKind::Extern);
  auto *Y = G.createVariable(ElemKind::FloatTy, {1, 3}, "Y",
                             Variable::InitKind::Extern);

  // Initialize the state to zero.
  auto *HInit = G.createVariable(ElemKind::FloatTy, {1, 3}, "initial_state",
                                 Variable::InitKind::Broadcast, 0);

  // Extract a slice for each input.
  auto *X1 = G.createSlice("X1", X, {0, 0, 0}, {1, 1, 4});
  auto *X2 = G.createSlice("X2", X, {0, 1, 0}, {1, 2, 4});
  auto *X3 = G.createSlice("X3", X, {0, 2, 0}, {1, 3, 4});

  // Extract a slice for each output.
  auto *Y1 = G.createSlice("Y1", Y, {0, 0}, {1, 1});
  auto *Y2 = G.createSlice("Y2", Y, {0, 1}, {1, 2});
  auto *Y3 = G.createSlice("Y3", Y, {0, 2}, {1, 3});

  // Create the first block in the RNN
  auto *FC11 = G.createFullyConnected("fc11", HInit, 5);
  auto *FC12 = G.createFullyConnected("fc12", X1, 5);
  auto *A1 = G.createArithmetic("fc1", FC11, FC12, ArithmeticNode::Mode::Add);
  auto *H1 = G.createTanh("tan1", A1);
  auto *O1 = G.createFullyConnected("O1", H1, 1);
  auto *R1 = G.createRegression("reg1", O1, Y1);

  // Create the second block in the RNN
  auto *FC21 = G.createFullyConnected("fc21", H1, 5);
  auto *FC22 = G.createFullyConnected("fc22", X2, 5);
  auto *A2 = G.createArithmetic("fc2", FC21, FC22, ArithmeticNode::Mode::Add);
  auto *H2 = G.createTanh("tan2", A2);
  auto *O2 = G.createFullyConnected("O2", H2, 1);
  auto *R2 = G.createRegression("reg2", O2, Y2);

  // Create the third block in the RNN
  auto *FC31 = G.createFullyConnected("fc31", H2, 5);
  auto *FC32 = G.createFullyConnected("fc32", X3, 5);
  auto *A3 = G.createArithmetic("fc3", FC31, FC32, ArithmeticNode::Mode::Add);
  auto *H3 = G.createTanh("tan3", A3);
  auto *O3 = G.createFullyConnected("O3", H3, 1);
  auto *R3 = G.createRegression("reg3", O3, Y3);

  auto *R = G.createConcat("O", {R1, R2, R3}, 1);
  auto *result = G.createSave("result", R);

  EE.compile(CompilationMode::Train);

  // Values for the input and output variables.
  Tensor inputs(ElemKind::FloatTy, {1, 3, 4});
  Tensor expected(ElemKind::FloatTy, {1, 3});
  for (size_t i = 0; i < 3; i++) {
    inputs.getHandle<float_t>().at({0, i, 1}) = i;
    expected.getHandle<float_t>().at({0, i}) = i;
  }

  // Train the network. Learn 1000 batches.
  EE.runBatch(1000, {X, Y}, {&inputs, &expected});

  // Testing the output vector.
  EE.compile(CompilationMode::Infer);

  EE.run({X}, {&inputs});
  auto RNWH = result->getVariable()->getPayload().getHandle<>();
  (void)RNWH;

  // Test the output:
  EXPECT_NEAR(RNWH.at({0, 0}), 0, 0.05);
  EXPECT_NEAR(RNWH.at({0, 1}), 1, 0.05);
  EXPECT_NEAR(RNWH.at({0, 2}), 2, 0.05);
}

TEST(Optimizer, copyPropagation) {
  ExecutionEngine EE;

  auto &G = EE.getGraph();
  G.setName("CopyPropagation");

  Node *K = G.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "input");
  Node *S = G.createVariable(ElemKind::IndexTy, {4, 1}, "select");

  K = G.createConv("Conv1", K, 16, 3, 2, 3);
  K = G.createRELU("Relu", K);
  K = G.createSoftMax("SoftMax", K, S);
  K = G.createSave("result", K);
  EE.compile(CompilationMode::Infer);

  // Check that all copy instructions are eliminated.
  auto &instrs = EE.getModule().getInstrs();
  EXPECT_TRUE(std::none_of(
      instrs.begin(), instrs.end(), [](const Instruction *I) -> bool {
        return I->getKind() == Instruction::Kind::CopyInstKind;
      }));
}

/// Learn the square root of two.
TEST(Interpreter, learnSqrt2) {
  ExecutionEngine EE;

  EE.getConfig().learningRate = 0.03;

  auto &G = EE.getGraph();
  G.setName("Square root of 2");

  auto *A = G.createVariable(ElemKind::FloatTy, {1}, "A",
                             Variable::InitKind::Broadcast, 1);
  auto *Ex = G.createVariable(ElemKind::FloatTy, {1}, "Ex",
                              Variable::InitKind::Broadcast, 2);

  Node *O = G.createArithmetic("Mult", A, A, ArithmeticNode::Mode::Mul);
  O = G.createRegression("reg", O, Ex);
  G.createSave("ret", O);

  EE.compile(CompilationMode::Train);

  // Train the network:
  for (int i = 0; i < 50; i++) {
    EE.run({}, {});
  }

  float res = A->getPayload().getHandle().at({0});
  EXPECT_NEAR(res, 1.4142, 0.01);
}

TEST(LinearRegression, trainSimpleLinearRegression) {
  // Given 1-D vectors x and y, find real numbers m and b such that
  // m * x + b is approximately equal to y.
  ExecutionEngine EE;
  EE.getConfig().maxNumThreads = 1;
  EE.getConfig().learningRate = 0.1;

  auto &G = EE.getGraph();
  G.setName("Gradient descent solution for simple linear regression");

  // These m and b are only used to generate training data.
  float referenceM = 3.0;
  float referenceB = 6.0;

  unsigned numSamples = 500;
  Tensor tensorX(ElemKind::FloatTy, {numSamples, 1});
  Tensor tensorY(ElemKind::FloatTy, {numSamples, 1});
  for (unsigned i = 0; i < numSamples; i++) {
    float x_i = -2.0 + 4.0 * i / numSamples;
    float y_i = referenceM * x_i + referenceB + nextRand() / 10.0;
    tensorX.getHandle<>().at({i, 0}) = x_i;
    tensorY.getHandle<>().at({i, 0}) = y_i;
  }

  // Create a variable with 1 input, which is a real number.
  auto *inputX = G.createVariable(ElemKind::FloatTy, {numSamples, 1}, "input",
                                  Variable::InitKind::Extern);
  auto *expectedY = G.createVariable(ElemKind::FloatTy, {numSamples, 1},
                                     "expected", Variable::InitKind::Extern);

  FullyConnectedNode *FC = G.createFullyConnected("fc", inputX, 1);
  Node *coef =
      G.createSplat("coef", FC->getType(), 1.0 / sqrt((double)numSamples));
  Node *normX =
      G.createArithmetic("normX", FC, coef, ArithmeticNode::Mode::Mul);
  Node *normY =
      G.createArithmetic("normY", expectedY, coef, ArithmeticNode::Mode::Mul);
  Node *R = G.createRegression("reg", normX, normY);
  G.createSave("return", R);

  Variable *M = llvm::cast<Variable>(FC->getFilter());
  Variable *B = llvm::cast<Variable>(FC->getBias());

  EE.compile(CompilationMode::Train);

  // Train the network doing 100 steps. Learn on 500 samples.
  EE.runBatch(100, {inputX, expectedY}, {&tensorX, &tensorY});

  // Testing trained m and b:
  EXPECT_NEAR(M->getPayload().getHandle<>().at({0, 0}), referenceM, 0.01);
  EXPECT_NEAR(B->getPayload().getHandle<>().at({0}), referenceB, 0.01);
}
