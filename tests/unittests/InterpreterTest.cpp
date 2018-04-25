// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"

#include "gtest/gtest.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <string>

using namespace glow;
using llvm::isa;

TEST(Interpreter, interpret) {
  ExecutionEngine EE;

  Tensor inputs(ElemKind::FloatTy, {1, 32, 32, 3});

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  F->setName("interpret");
  auto *input = mod.createVariable(ElemKind::FloatTy, {1, 32, 32, 3}, "input",
                                   Variable::VisibilityKind::Public);

  auto *ex = mod.createVariable(ElemKind::IndexTy, {1, 1}, "exp");

  auto *CV0 = F->createConv("conv1", input, 16, 5, 1, 2, 1);
  auto *RL0 = F->createRELU("relu1", CV0);
  auto *MP0 = F->createPoolMax("pool1", RL0, 2, 2, 0);

  auto *CV1 = F->createConv("conv2", MP0, 20, 5, 1, 2, 1);
  auto *RL1 = F->createRELU("relu2", CV1);
  auto *MP1 = F->createPoolMax("pool2", RL1, 2, 2, 0);

  auto *CV2 = F->createConv("conv3", MP1, 20, 5, 1, 2, 1);
  auto *RL2 = F->createRELU("relu3", CV2);
  auto *MP2 = F->createPoolMax("pool3", RL2, 2, 2, 0);

  auto *FCL1 = F->createFullyConnected("fc", MP2, 10);
  auto *SM = F->createSoftMax("sm", FCL1, ex);
  F->createSave("ret", SM);

  EE.compile(CompilationMode::Infer, F);

  /// Add a debug_action instruction to check that it can be
  /// processed by the interpreter.
  auto &M = EE.getIR();
  IRBuilder builder(&M);
  builder.createDebugPrintInst("print1", *M.getWeights().begin());

  EE.run({input}, {&inputs});
}

TEST(Interpreter, profileQuantizationForANetwork) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  Tensor inputs(ElemKind::FloatTy, {1, 4});
  inputs.getHandle() = {1, 1.2, 0.5, 1.3};

  auto *A = mod.createVariable(ElemKind::FloatTy, {1, 4}, "A",
                               Variable::VisibilityKind::Public);
  auto *Ex = mod.createVariable(ElemKind::FloatTy, {1, 4}, "E",
                                Variable::VisibilityKind::Public);
  Node *O = F->createFullyConnected("fc", A, 4);
  O = F->createRELU("relu", O);
  O = F->createRegression("reg", O, Ex);

  ::glow::profileQuantization(F);

  EE.compile(CompilationMode::Infer, F);

  // TODO: Verify histogram itself, for now just verify min and max.
  // Run inference first time and capture tensor stats.
  EE.run({A}, {&inputs});

  QuantizationProfileNode *profile{nullptr};
  // Find QPN for node A.
  for (auto node : F->getNodes()) {
    if (QuantizationProfileNode *QPN =
            llvm::dyn_cast<QuantizationProfileNode>(node)) {
      Node *observedNode = node->getNthInput(0).getNode();
      if (observedNode == A) {
        profile = QPN;
        break;
      }
    }
  }

  EXPECT_TRUE(profile != nullptr);

  auto CI = profile->getComputationInfoVar()->getHandle<float>();
  float min = CI.raw(0);
  float max = CI.raw(1);
  EXPECT_NEAR(0.5, min, 0.00001);
  EXPECT_NEAR(1.3, max, 0.00001);

  // Run inference for the second time with new min and max.
  inputs.getHandle() = {0.2, 1.6, 0.5, 1.3};
  EE.run({A}, {&inputs});
  min = CI.raw(0);
  max = CI.raw(1);
  EXPECT_NEAR(0.2, min, 0.00001);
  EXPECT_NEAR(1.6, max, 0.00001);
}

TEST(Interpreter, trainASimpleNetwork) {
  ExecutionEngine EE;
  // Learning a single input vector.
  EE.getConfig().learningRate = 0.05;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  F->setName("trainASimpleNetwork");

  // Create a variable with 1 input, which is a vector of 4 elements.
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, 4}, "A",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  auto *E = mod.createVariable(ElemKind::FloatTy, {1, 4}, "E",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);

  Node *O = F->createFullyConnected("fc1", A, 10);
  O = F->createSigmoid("sig1", O);
  O = F->createFullyConnected("fc2", O, 4);
  O = F->createSigmoid("sig2", O);
  O = F->createRegression("reg", O, E);
  auto *result = F->createSave("return", O);

  // Values for the input and output variables.
  Tensor inputs(ElemKind::FloatTy, {1, 4});
  Tensor expected(ElemKind::FloatTy, {1, 4});
  inputs.getHandle<>() = {0.15, 0.15, 0.15, 0.15};
  expected.getHandle<>() = {0.9, 0.9, 0.9, 0.9};

  Function *TF = glow::differentiate(F, EE.getConfig());
  EE.compile(CompilationMode::Train, TF);

  // Train the network. Learn 1000 batches.
  EE.runBatch(1000, {A, E}, {&inputs, &expected});

  // Testing the output vector.

  EE.compile(CompilationMode::Infer, F);
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
  EE.getConfig().learningRate = 0.05;

  Tensor inputs(ElemKind::FloatTy, {1, numInputs});
  Tensor expected(ElemKind::FloatTy, {1, numInputs});

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  F->setName("simpleRegression");
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, numInputs}, "A",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  auto *Ex = mod.createVariable(ElemKind::FloatTy, {1, numInputs}, "E",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);
  Node *O = F->createFullyConnected("fc", A, 4);
  O = F->createRELU("relu", O);
  O = F->createRegression("reg", O, Ex);
  auto *result = F->createSave("result", O);

  auto I = inputs.getHandle<>();
  auto E = expected.getHandle<>();

  Function *TF = glow::differentiate(F, EE.getConfig());
  EE.compile(CompilationMode::Train, TF);

  // Train the network:
  for (int iter = 0; iter < 1000; iter++) {
    float target = float(iter % 9);
    I = {target, 0., 0., 0.};
    E = {0., target + 1, 0., 0.};
    EE.runBatch(1, {A, Ex}, {&inputs, &expected});
  }

  // Verify the result of the regression layer.
  EE.compile(CompilationMode::Infer, F);

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

  // Learning the Xor function.
  ExecutionEngine EE;

  // Learning a single input vector.
  EE.getConfig().learningRate = 0.05;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  F->setName("learnXor");

  auto *A = mod.createVariable(ElemKind::FloatTy, {numInputs, 2}, "A",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  auto *Ex = mod.createVariable(ElemKind::FloatTy, {numInputs, 1}, "Ex",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);

  Node *O = F->createFullyConnected("fc1", A, 6);
  O = F->createTanh("tanh1", O);
  O = F->createFullyConnected("fc2", O, 1);
  O = F->createTanh("tanh2", O);
  O = F->createRegression("reg", O, Ex);
  auto *result = F->createSave("ret", O);

  // Prepare the training set and the testing set.
  Tensor trainingSet(ElemKind::FloatTy, {numInputs, 2});
  Tensor trainingLabels(ElemKind::FloatTy, {numInputs, 1});

  auto TS = trainingSet.getHandle<>();
  auto TL = trainingLabels.getHandle<>();

  // Prepare the training data:
  for (unsigned i = 0; i < numInputs; i++) {
    int a = i % 2;
    int b = (i >> 1) % 2;
    TS.at({i, 0}) = a;
    TS.at({i, 1}) = b;
    TL.at({i, 0}) = a ^ b;
  }

  Function *TF = glow::differentiate(F, EE.getConfig());
  EE.compile(CompilationMode::Train, TF);

  // Train the network:
  EE.runBatch(2500, {A, Ex}, {&trainingSet, &trainingLabels});

  EE.compile(CompilationMode::Infer, F);

  // Prepare the testing tensor:
  for (unsigned i = 0; i < numInputs; i++) {
    int a = (numInputs - i) % 2;
    int b = ((numInputs - i) >> 1) % 2;
    TS.at({i, 0}) = a;
    TS.at({i, 1}) = b;
  }

  EE.run({A}, {&trainingSet});
  auto resH = result->getVariable()->getPayload().getHandle<>();

  // Test the output:
  for (size_t i = 0; i < numInputs; i++) {
    int a = TS.at({i, 0});
    int b = TS.at({i, 1});
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
TEST(Interpreter, circle) {
  // Testing the softmax layer.
  ExecutionEngine EE;

  // Learning a single input vector.
  EE.getConfig().momentum = 0.9;
  EE.getConfig().learningRate = 0.01;

  unsigned minibatchSize = 11;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  F->setName("circle");
  auto *A = mod.createVariable(ElemKind::FloatTy, {minibatchSize, 2}, "A",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  auto *S = mod.createVariable(ElemKind::IndexTy, {minibatchSize, 1}, "S",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);

  auto *FCL0 = F->createFullyConnected("fc1", A, 8);
  auto *T0 = F->createTanh("tanh1", FCL0);
  auto *FCL1 = F->createFullyConnected("fc2", T0, 2);
  auto *SM = F->createSoftMax("soft", FCL1, S);
  auto *result = F->createSave("ret", SM);

  Function *TF = glow::differentiate(F, EE.getConfig());
  EE.compile(CompilationMode::Train, TF);

  Tensor coordinates(ElemKind::FloatTy, {numSamples, 2});
  Tensor labels(ElemKind::IndexTy, {numSamples, 1});
  generateCircleData(coordinates, labels);

  // Training:
  EE.runBatch(4000, {A, S}, {&coordinates, &labels});

  EE.compile(CompilationMode::Infer, F);

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

TEST(Interpreter, learnSingleValueConcat) {
  ExecutionEngine EE;
  unsigned width = 6;

  // Learning a single input vector.
  EE.getConfig().momentum = 0.9;
  EE.getConfig().learningRate = 0.01;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  F->setName("learnSingleValueConcat");

  auto *Ex = mod.createVariable(ElemKind::FloatTy, {1, width * 2}, "Ex",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);

  // Left side of the network:
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, width}, "A",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  Node *L = F->createFullyConnected("fc1", A, width);
  L = F->createSigmoid("", L);

  // Right side of the network:
  auto *B = mod.createVariable(ElemKind::FloatTy, {1, width}, "B",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  Node *R = F->createFullyConnected("fc2", B, width);
  R = F->createSigmoid("sig", R);

  // Concat:
  auto *C = F->createConcat("con", {L, R}, 1);
  auto *RN = F->createRegression("reg", C, Ex);
  auto *result = F->createSave("ret", RN);

  Tensor inputs(ElemKind::FloatTy, {1, width});
  Tensor expected(ElemKind::FloatTy, {1, width * 2});
  inputs.getHandle<>().clear(0.15);
  expected.getHandle<>().clear(0.9);

  Function *TF = glow::differentiate(F, EE.getConfig());
  EE.compile(CompilationMode::Train, TF);

  // Train the network:
  EE.runBatch(1000, {A, B, Ex}, {&inputs, &inputs, &expected});

  EE.compile(CompilationMode::Infer, F);

  // Testing the output vector.
  EE.run({A}, {&inputs});
  auto RNWH = result->getVariable()->getPayload().getHandle<>();
  (void)RNWH;

  // Test the output:
  EXPECT_NEAR(RNWH.at({0, 0}), 0.9, 0.1);
}

void buildGRU(Function *F, const std::vector<Node *> &slicesX,
              unsigned hiddenSize, unsigned outputSize,
              std::vector<Node *> &outputs) {
  return F->createGRU("GRU", slicesX, 1, hiddenSize, outputSize, outputs);
};

void buildRNN(Function *F, const std::vector<Node *> &slicesX,
              unsigned hiddenSize, unsigned outputSize,
              std::vector<Node *> &outputs) {
  return F->createSimpleRNN("SimpleRNN", slicesX, 1, hiddenSize, outputSize,
                            outputs);
};

void buildLSTM(Function *F, const std::vector<Node *> &slicesX,
               unsigned hiddenSize, unsigned outputSize,
               std::vector<Node *> &outputs) {
  return F->createLSTM("LSTM", slicesX, 1, hiddenSize, outputSize, outputs);
};

using TCellGenerator = void (*)(Function *, const std::vector<Node *> &,
                                unsigned, unsigned, std::vector<Node *> &);

void testRNNCell(TCellGenerator cell) {
  ExecutionEngine EE;
  // Learning a single input vector.
  EE.getConfig().learningRate = 0.05;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  F->setName("testRNNCell");

  const unsigned NumVectors = 3;
  const unsigned NumElements = 4;
  // Create a variable with 1 input, which is 3 consecutive vectors
  // of 4 elements each.
  auto *X = mod.createVariable(ElemKind::FloatTy, {1, NumVectors, NumElements},
                               "X", Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  auto *Y = mod.createVariable(ElemKind::FloatTy, {1, NumVectors}, "Y",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);

  // Extract a slice for each input.
  std::vector<Node *> XSliced;

  for (unsigned i = 0; i < NumVectors; ++i) {
    std::string Name{"X"};
    Name.append(std::to_string(i + 1));
    XSliced.push_back(F->createSlice(Name, X, {0, i, 0}, {1, i + 1, 4}));
  }

  // Extract a slice for each output.
  std::vector<Node *> YSliced;

  for (unsigned i = 0; i < NumVectors; ++i) {
    std::string Name{"Y"};
    Name.append(std::to_string(i + 1));
    YSliced.push_back(F->createSlice(Name, Y, {0, i}, {1, i + 1}));
  }

  const unsigned hiddenSize = 5;
  const unsigned outputSize = 1;

  std::vector<Node *> outputNodes;
  cell(F, XSliced, hiddenSize, outputSize, outputNodes);

  std::vector<Node *> regressionNodes;
  for (unsigned t = 0; t < NumVectors; t++) {
    regressionNodes.push_back(
        F->createRegression("", outputNodes[t], YSliced[t]));
  };

  auto *R = F->createConcat("O", regressionNodes, 1);
  auto *result = F->createSave("result", R);

  Function *TF = glow::differentiate(F, EE.getConfig());
  EE.compile(CompilationMode::Train, TF);

  // Values for the input and output variables.
  Tensor inputs(ElemKind::FloatTy, {1, NumVectors, NumElements});
  Tensor expected(ElemKind::FloatTy, {1, NumVectors});
  inputs.zero();
  expected.zero();
  for (size_t i = 0; i < NumVectors; i++) {
    inputs.getHandle<float_t>().at({0, i, 1}) = i;
    expected.getHandle<float_t>().at({0, i}) = i;
  }

  // Train the network. Learn 1000 batches.
  EE.runBatch(1000, {X, Y}, {&inputs, &expected});

  // Testing the output vector.
  EE.compile(CompilationMode::Infer, F);

  EE.run({X}, {&inputs});
  auto RNWH = result->getVariable()->getPayload().getHandle<>();
  (void)RNWH;

  // Test the output:
  for (size_t t = 0; t < NumVectors; ++t) {
    EXPECT_NEAR(RNWH.at({0, t}), t, 0.05);
  }
};

TEST(Interpreter, trainASimpleRNN) { testRNNCell(buildRNN); };

TEST(Interpreter, trainGRU) { testRNNCell(buildGRU); };

TEST(Interpreter, trainLSTM) { testRNNCell(buildLSTM); };

/// Learn the square root of two.
TEST(Interpreter, learnSqrt2) {
  ExecutionEngine EE;

  EE.getConfig().learningRate = 0.03;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  F->setName("Square root of 2");

  auto *A = mod.createVariable(ElemKind::FloatTy, {1}, "A",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::Broadcast, 1);
  auto *Ex = mod.createVariable(ElemKind::FloatTy, {1}, "Ex",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);
  Ex->getPayload().getHandle() = {2};

  Node *O = F->createMul("Mult", A, A);
  O = F->createRegression("reg", O, Ex);
  F->createSave("ret", O);

  Function *TF = glow::differentiate(F, EE.getConfig());
  EE.compile(CompilationMode::Train, TF);

  // Train the network:
  for (int i = 0; i < 50; i++) {
    EE.run({}, {});
  }

  float res = A->getPayload().getHandle().at({0});
  EXPECT_NEAR(res, 1.4142, 0.01);
}

TEST(Interpreter, trainSimpleLinearRegression) {
  // Given 1-D vectors x and y, find real numbers m and b such that
  // m * x + b is approximately equal to y.
  unsigned numSamples = 500;

  ExecutionEngine EE;
  EE.getConfig().learningRate = 0.1;
  EE.getConfig().batchSize = numSamples;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  F->setName("Gradient descent solution for simple linear regression");

  // These m and b are only used to generate training data.
  float referenceM = 3.0;
  float referenceB = 6.0;

  Tensor tensorX(ElemKind::FloatTy, {numSamples, 1});
  Tensor tensorY(ElemKind::FloatTy, {numSamples, 1});
  for (unsigned i = 0; i < numSamples; i++) {
    float x_i = -2.0 + 4.0 * i / numSamples;
    float y_i = referenceM * x_i + referenceB + nextRand() / 10.0;
    tensorX.getHandle<>().at({i, 0}) = x_i;
    tensorY.getHandle<>().at({i, 0}) = y_i;
  }

  // Create a variable with 1 input, which is a real number.
  auto *inputX = mod.createVariable(ElemKind::FloatTy, {numSamples, 1}, "input",
                                    Variable::VisibilityKind::Public,
                                    Variable::TrainKind::None);
  auto *expectedY = mod.createVariable(
      ElemKind::FloatTy, {numSamples, 1}, "expected",
      Variable::VisibilityKind::Public, Variable::TrainKind::None);

  FullyConnectedNode *FC = F->createFullyConnected("fc", inputX, 1);
  Node *R = F->createRegression("reg", FC, expectedY);
  F->createSave("return", R);

  Variable *M = llvm::cast<Variable>(FC->getWeights());
  Variable *B = llvm::cast<Variable>(FC->getBias());

  Function *TF = glow::differentiate(F, EE.getConfig());
  EE.compile(CompilationMode::Train, TF);

  // Train the network doing 100 steps. Learn on 500 samples.
  EE.runBatch(100, {inputX, expectedY}, {&tensorX, &tensorY});

  // Testing trained m and b:
  EXPECT_NEAR(M->getPayload().getHandle<>().at({0, 0}), referenceM, 0.01);
  EXPECT_NEAR(B->getPayload().getHandle<>().at({0}), referenceB, 0.01);
}

enum class Sport : size_t { BASKETBALL = 0, SOCCER = 1 };

void generatePlayerData(Tensor &players, Tensor &labels,
                        unsigned numTrainPlayers) {
  auto P = players.getHandle<>();
  auto L = labels.getHandle<size_t>();

  // Auto generate height/weights for basketball players.
  for (size_t i = 0; i < numTrainPlayers / 2; i++) {
    auto heightInches = nextRandInt(70, 88);
    auto weightLbs = 4 * heightInches + nextRandInt(-85, -55); // [195, 297]
    P.at({i, 0}) = heightInches;
    P.at({i, 1}) = weightLbs;
    L.at({i, 0}) = static_cast<size_t>(Sport::BASKETBALL);
  }

  // Auto generate height/weights for soccer players.
  for (size_t i = numTrainPlayers / 2; i < numTrainPlayers; i++) {
    auto heightInches = nextRandInt(60, 76);
    auto weightLbs = static_cast<unsigned>(2 * heightInches) +
                     nextRandInt(20, 50); // [140, 202]
    P.at({i, 0}) = heightInches;
    P.at({i, 1}) = weightLbs;
    L.at({i, 0}) = static_cast<size_t>(Sport::SOCCER);
  }
}

// Given a player's height and weight, classify them as a basketball or
// soccer player.
TEST(Interpreter, classifyPlayerSport) {
  ExecutionEngine EE;
  EE.getConfig().learningRate = 0.05;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  F->setName("classifyPlayers");

  const unsigned numTrainPlayers = 200;
  const size_t numFeatures = 2;
  const size_t numClasses = 2;

  auto *A = mod.createVariable(
      ElemKind::FloatTy, {numTrainPlayers, numFeatures}, "A",
      Variable::VisibilityKind::Public, Variable::TrainKind::None);
  auto *S = mod.createVariable(ElemKind::IndexTy, {numTrainPlayers, 1}, "S",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);

  auto *FC = F->createFullyConnected("fc", A, numClasses);
  auto *SM = F->createSoftMax("softmax", FC, S);
  auto *result = F->createSave("result", SM);

  Function *TF = glow::differentiate(F, EE.getConfig());
  EE.compile(CompilationMode::Train, TF);

  Tensor players(ElemKind::FloatTy, {numTrainPlayers, numFeatures});
  Tensor labels(ElemKind::IndexTy, {numTrainPlayers, 1});
  generatePlayerData(players, labels, numTrainPlayers);

  // Training:
  EE.runBatch(2000, {A, S}, {&players, &labels});

  EE.compile(CompilationMode::Infer, F);

  std::vector<std::tuple<unsigned, unsigned, Sport>> testPlayers;
  testPlayers.emplace_back(82, 240, Sport::BASKETBALL);
  testPlayers.emplace_back(86, 260, Sport::BASKETBALL);
  testPlayers.emplace_back(90, 270, Sport::BASKETBALL);
  testPlayers.emplace_back(60, 160, Sport::SOCCER);
  testPlayers.emplace_back(63, 155, Sport::SOCCER);
  testPlayers.emplace_back(66, 170, Sport::SOCCER);

  Tensor testPlayersTensor(ElemKind::FloatTy, {numTrainPlayers, numFeatures});
  for (size_t i = 0; i < testPlayers.size(); i++) {
    testPlayersTensor.getHandle<>().at({i, 0}) = std::get<0>(testPlayers[i]);
    testPlayersTensor.getHandle<>().at({i, 1}) = std::get<1>(testPlayers[i]);
  }

  EE.run({A}, {&testPlayersTensor});

  for (size_t i = 0; i < testPlayers.size(); i++) {
    auto SMH = result->getVariable()->getPayload().getHandle<>();
    const size_t sport = static_cast<size_t>(std::get<2>(testPlayers[i]));
    EXPECT_NEAR(SMH.at({i, sport}), 1.0, 0.1);
  }
}

TEST(Interpreter, learnSinus) {
  // Try to learn the sin(x) function.
  float epsilon = 0.1;
  unsigned numSamples = 50;

  ExecutionEngine EE;
  EE.getConfig().learningRate = 0.2;
  EE.getConfig().batchSize = numSamples;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  F->setName("Gradient descent solution for sin(x)");

  // Function that should be learned by the NN
  auto FF = [](float x) -> float {
    // Return a shifted sin(x) value, so that it is in the range [0, 1].
    return (sin(x) + 1) / 2;
  };

  Tensor tensorX(ElemKind::FloatTy, {numSamples, 1});
  Tensor tensorY(ElemKind::FloatTy, {numSamples, 1});

  for (unsigned i = 0; i < numSamples; i++) {
    // Scale x to cover the range [0, 4] as this leads to a good convergence
    // during training.
    float x = i / (numSamples / 4.0);
    float y = FF(x);
    tensorX.getHandle<>().at({i, 0}) = x;
    tensorY.getHandle<>().at({i, 0}) = y;
  }

  auto *inputX = mod.createVariable(ElemKind::FloatTy, {numSamples, 1}, "input",
                                    Variable::VisibilityKind::Public,
                                    Variable::TrainKind::None);

  auto *expectedY = mod.createVariable(
      ElemKind::FloatTy, {numSamples, 1}, "expected",
      Variable::VisibilityKind::Public, Variable::TrainKind::None);

  FullyConnectedNode *FC1 = F->createFullyConnected("fc1", inputX, 10);
  Node *O = F->createSigmoid("sigmoid1", FC1);
  FullyConnectedNode *FC2 = F->createFullyConnected("fc2", O, 1);
  Node *R = F->createRegression("reg", FC2, expectedY);
  auto *result = F->createSave("return", R);

  Function *TF = glow::differentiate(F, EE.getConfig());
  EE.compile(CompilationMode::Train, TF);

  // Learn on numSamples samples.
  EE.runBatch(2700, {inputX, expectedY}, {&tensorX, &tensorY});

  // Create a test set, which is similar, but different from the training set.
  for (unsigned i = 0; i < numSamples; i++) {
    // Scale x to cover the range [0, 4.2] as this leads to a good convergence
    // during training.
    float x = i / (numSamples / 4.2) + 0.123456;
    float y = FF(x);
    tensorX.getHandle<>().at({i, 0}) = x;
    tensorY.getHandle<>().at({i, 0}) = y;
  }

  EE.compile(CompilationMode::Infer, F);
  EE.run({inputX}, {&tensorX});
  auto resH = result->getVariable()->getPayload().getHandle<>();

  for (size_t i = 0; i < numSamples; i++) {
    float x = tensorX.getHandle().at({i, 0});
    EXPECT_NEAR(resH.at({i, 0}), FF(x), epsilon);
  }
}

TEST(Interpreter, nonLinearClassifier) {
  // Test non-linear classification on a set of 2d points. Generate x and y in
  // (-1, 1) and classify according to XOR of the sign bit.
  unsigned batchSize = 46;
  unsigned numSamples = 230;

  ExecutionEngine EE;
  EE.getConfig().learningRate = 0.01;
  EE.getConfig().batchSize = batchSize;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  F->setName("nonLinearClassifier");

  auto *A = mod.createVariable(ElemKind::FloatTy, {batchSize, 2}, "A",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  auto *S = mod.createVariable(ElemKind::IndexTy, {batchSize, 1}, "S",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);

  auto *FCL0 = F->createFullyConnected("fc1", A, 8);
  auto *T0 = F->createTanh("tanh1", FCL0);
  auto *FCL1 = F->createFullyConnected("fc2", T0, 8);
  auto *T1 = F->createTanh("tanh2", FCL1);
  auto *FCL2 = F->createFullyConnected("fc2", T1, 2);
  auto *SM = F->createSoftMax("soft", FCL2, S);
  auto *result = F->createSave("ret", SM);

  Function *TF = glow::differentiate(F, EE.getConfig());
  EE.compile(CompilationMode::Train, TF);

  Tensor samples(ElemKind::FloatTy, {numSamples, 2});
  Tensor labels(ElemKind::IndexTy, {numSamples, 1});

  for (size_t i = 0; i < numSamples; i++) {
    float x = nextRand();
    float y = nextRand();
    size_t label = (x < 0.0) ^ (y < 0.0);
    samples.getHandle<>().at({i, 0}) = x;
    samples.getHandle<>().at({i, 1}) = y;
    labels.getHandle<size_t>().at({i, 0}) = label;
  }

  EE.runBatch(4000, {A, S}, {&samples, &labels});

  EE.compile(CompilationMode::Infer, F);

  std::vector<std::tuple<float, float, size_t>> tests;
  tests.emplace_back(-0.8, -0.8, 0);
  tests.emplace_back(0.8, -0.8, 1);
  tests.emplace_back(-0.8, 0.8, 1);
  tests.emplace_back(0.8, 0.8, 0);
  for (size_t i = 0; i < tests.size(); i++) {
    Tensor T(ElemKind::FloatTy, {batchSize, 2});
    T.getHandle<>().at({0, 0}) = std::get<0>(tests[i]);
    T.getHandle<>().at({0, 1}) = std::get<1>(tests[i]);
    EE.run({A}, {&T});
    auto RH = result->getVariable()->getPayload().getHandle<>();
    EXPECT_NEAR(RH.at({0, std::get<2>(tests[i])}), 1.0, 0.2);
  }
}
