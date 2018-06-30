/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Quantization/Quantization.h"

#include "gtest/gtest.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace glow;
using llvm::isa;

class TestRunnerBase : public ::testing::TestWithParam<BackendKind> {
public:
  ExecutionEngine EE_{GetParam()};
};

class InterpreterAndCPU : public TestRunnerBase {};
class MLTest : public TestRunnerBase {};

TEST_P(MLTest, trainASimpleNetwork) {
  // Learning a single input vector.
  EE_.getConfig().learningRate = 0.05;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("trainASimpleNetwork");

  // Create a variable with 1 input, which is a vector of 4 elements.
  auto *A =
      mod.createVariable(ElemKind::FloatTy, {1, 4}, "A", VisibilityKind::Public,
                         Variable::TrainKind::None);
  auto *E =
      mod.createVariable(ElemKind::FloatTy, {1, 4}, "E", VisibilityKind::Public,
                         Variable::TrainKind::None);

  Node *O = F->createFullyConnected("fc1", A, 10);
  O = F->createSigmoid("sig1", O);
  O = F->createFullyConnected("fc2", O, 4);
  O = F->createRegression("reg", O, E);
  auto *result = F->createSave("return", O);

  // Values for the input and output variables.
  Tensor inputs(ElemKind::FloatTy, {1, 4});
  Tensor expected(ElemKind::FloatTy, {1, 4});
  inputs.getHandle<>() = {0.15f, 0.15f, 0.15f, 0.15f};
  expected.getHandle<>() = {0.9f, 0.9f, 0.9f, 0.9f};

  Function *TF = glow::differentiate(F, EE_.getConfig());
  EE_.compile(CompilationMode::Train, TF);

  // Train the network. Learn 1000 batches.
  EE_.runBatch(1000, {A, E}, {&inputs, &expected});

  // Testing the output vector.

  EE_.compile(CompilationMode::Infer, F);
  EE_.run({A}, {&inputs});
  auto RNWH = result->getVariable()->getPayload().getHandle<>();
  (void)RNWH;

  // Test the output:
  EXPECT_NEAR(RNWH.at({0, 0}), 0.9, 0.05);
}

TEST_P(MLTest, simpleRegression) {
  // Testing the regression layer. This test takes the first element from the
  // input vector, adds one to it and places the result in the second element of
  // the output vector.
  const size_t numInputs = 4;

  // Learning a single input vector.
  EE_.getConfig().learningRate = 0.05;

  Tensor inputs(ElemKind::FloatTy, {1, numInputs});
  Tensor expected(ElemKind::FloatTy, {1, numInputs});

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("simpleRegression");
  auto *A =
      mod.createVariable(ElemKind::FloatTy, {1, numInputs}, "A",
                         VisibilityKind::Public, Variable::TrainKind::None);
  auto *Ex =
      mod.createVariable(ElemKind::FloatTy, {1, numInputs}, "E",
                         VisibilityKind::Public, Variable::TrainKind::None);
  Node *O = F->createFullyConnected("fc", A, 4);
  O = F->createRELU("relu", O);
  O = F->createRegression("reg", O, Ex);
  auto *result = F->createSave("result", O);

  auto I = inputs.getHandle<>();
  auto E = expected.getHandle<>();

  Function *TF = glow::differentiate(F, EE_.getConfig());
  EE_.compile(CompilationMode::Train, TF);

  // Train the network:
  for (int iter = 0; iter < 1000; iter++) {
    float target = float(iter % 9);
    I = {target, 0., 0., 0.};
    E = {0., target + 1, 0., 0.};
    EE_.runBatch(1, {A, Ex}, {&inputs, &expected});
  }

  // Verify the result of the regression layer.
  EE_.compile(CompilationMode::Infer, F);

  // Test the output:
  for (int iter = 0; iter < 5; iter++) {
    float target = iter % 9 + 1;
    I = {target, 0., 0., 0.};
    EE_.run({A}, {&inputs});

    auto resH = result->getVariable()->getPayload().getHandle<>();
    (void)resH;

    EXPECT_NEAR(I.at({0, 0}) + 1, resH.at({0, 1}), 0.1);
  }
}

TEST_P(MLTest, learnXor) {
  unsigned numInputs = 10;

  // Learning a single input vector.
  EE_.getConfig().learningRate = 0.05;
  EE_.getConfig().batchSize = numInputs;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("learnXor");

  auto *A =
      mod.createVariable(ElemKind::FloatTy, {numInputs, 2}, "A",
                         VisibilityKind::Public, Variable::TrainKind::None);
  auto *Ex =
      mod.createVariable(ElemKind::FloatTy, {numInputs, 1}, "Ex",
                         VisibilityKind::Public, Variable::TrainKind::None);

  Node *O = F->createFullyConnected("fc1", A, 6);
  O = F->createTanh("tanh1", O);
  O = F->createFullyConnected("fc2", O, 1);
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

  Function *TF = glow::differentiate(F, EE_.getConfig());
  EE_.compile(CompilationMode::Train, TF);

  // Train the network:
  EE_.runBatch(2500, {A, Ex}, {&trainingSet, &trainingLabels});

  EE_.compile(CompilationMode::Infer, F);

  // Prepare the testing tensor:
  for (unsigned i = 0; i < numInputs; i++) {
    int a = (numInputs - i) % 2;
    int b = ((numInputs - i) >> 1) % 2;
    TS.at({i, 0}) = a;
    TS.at({i, 1}) = b;
  }

  EE_.run({A}, {&trainingSet});
  auto resH = result->getVariable()->getPayload().getHandle<>();

  // Test the output:
  for (size_t i = 0; i < numInputs; i++) {
    int a = TS.at({i, 0});
    int b = TS.at({i, 1});
    EXPECT_NEAR(resH.at({i, 0}), (a ^ b), 0.1);
  }
}

/// Learn the logarithmic function.
TEST_P(MLTest, learnLog) {
  unsigned numInputs = 50;
  unsigned batchSize = 5;
  EE_.getConfig().learningRate = 0.07;
  EE_.getConfig().batchSize = batchSize;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("learnLog");

  auto *A =
      mod.createVariable(ElemKind::FloatTy, {batchSize, 1}, "A",
                         VisibilityKind::Public, Variable::TrainKind::None);
  auto *Ex =
      mod.createVariable(ElemKind::FloatTy, {batchSize, 1}, "Ex",
                         VisibilityKind::Public, Variable::TrainKind::None);

  Node *O = F->createFullyConnected("fc1", A, 4);
  O = F->createTanh("tanh1", O);
  O = F->createFullyConnected("fc2", O, 1);
  O = F->createRegression("reg", O, Ex);
  auto *result = F->createSave("ret", O);

  // Set the training data.
  Tensor trainingSet(ElemKind::FloatTy, {numInputs, 1});
  Tensor trainingLabels(ElemKind::FloatTy, {numInputs, 1});

  auto TS = trainingSet.getHandle<>();
  auto TL = trainingLabels.getHandle<>();

  // Set the training data as floating number from range [0.75, 1.5).
  const float LO = 0.75; // Lower bound of training data.
  const float HI = 1.5;  // Upper bound of training data.
  for (size_t i = 0; i < numInputs; i++) {
    // Generate a floating number in the range of [LO,HI).
    float a = LO + i * (HI - LO) / numInputs;
    TS.at({i, 0}) = a;
    TL.at({i, 0}) = std::log(a);
  }

  Function *TF = glow::differentiate(F, EE_.getConfig());
  EE_.compile(CompilationMode::Train, TF);

  // Train the network:
  EE_.runBatch(1000, {A, Ex}, {&trainingSet, &trainingLabels});

  EE_.compile(CompilationMode::Infer, F);

  // Set the testing data.
  Tensor testSet(ElemKind::FloatTy, {batchSize, 1});

  auto TES = testSet.getHandle<>();

  const float LO_T = 0.85; // Lower bound of testing data.
  const float HI_T = 1.45; // Upper bound of testing data.

  for (size_t i = 0; i < batchSize; i++) {
    // Generate a floating number in the range of [LO_T,HI_T).
    float a = mod.getPRNG().nextRandReal(LO_T, HI_T);
    TES.at({i, 0}) = a;
  }

  EE_.run({A}, {&testSet});
  auto resH = result->getVariable()->getPayload().getHandle<>();

  // Test the output:
  for (size_t i = 0; i < batchSize; i++) {
    float a = TES.at({i, 0});
    EXPECT_NEAR(resH.at({i, 0}), (std::log(a)), 0.02);
  }
}

unsigned numSamples = 230;

/// Generate data in two classes. The circle of dots that's close to the axis is
/// L0, and the rest of the dots, away from the axis are L1.
void generateCircleData(Tensor &coordinates, Tensor &labels, PseudoRNG &PRNG) {
  auto C = coordinates.getHandle<>();
  auto L = labels.getHandle<size_t>();

  for (size_t i = 0; i < numSamples / 2; i++) {
    float r = PRNG.nextRand() * 0.4;
    float a = PRNG.nextRand() * 3.141592 * 2;
    float y = r * sin(a);
    float x = r * cos(a);

    C.at({i * 2, 0u}) = x;
    C.at({i * 2, 1u}) = y;
    L.at({i * 2, 0}) = 1;

    r = PRNG.nextRand() * 0.4 + 0.8;
    a = PRNG.nextRand() * 3.141592 * 2;
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
TEST_P(MLTest, circle) {
  unsigned minibatchSize = 11;

  // Testing the softmax layer.
  // Learning a single input vector.
  EE_.getConfig().momentum = 0.9;
  EE_.getConfig().learningRate = 0.01;
  EE_.getConfig().batchSize = minibatchSize;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("circle");
  auto *A =
      mod.createVariable(ElemKind::FloatTy, {minibatchSize, 2}, "A",
                         VisibilityKind::Public, Variable::TrainKind::None);
  auto *S =
      mod.createVariable(ElemKind::IndexTy, {minibatchSize, 1}, "S",
                         VisibilityKind::Public, Variable::TrainKind::None);

  auto *FCL0 = F->createFullyConnected("fc1", A, 8);
  auto *T0 = F->createTanh("tanh1", FCL0);
  auto *FCL1 = F->createFullyConnected("fc2", T0, 2);
  auto *SM = F->createSoftMax("soft", FCL1, S);
  auto *result = F->createSave("ret", SM);

  Function *TF = glow::differentiate(F, EE_.getConfig());
  EE_.compile(CompilationMode::Train, TF);

  Tensor coordinates(ElemKind::FloatTy, {numSamples, 2});
  Tensor labels(ElemKind::IndexTy, {numSamples, 1});
  generateCircleData(coordinates, labels, mod.getPRNG());

  // Training:
  EE_.runBatch(4000, {A, S}, {&coordinates, &labels});

  EE_.compile(CompilationMode::Infer, F);

  // Print a diagram that depicts the network decision on a grid.
  Tensor sample(ElemKind::FloatTy, {minibatchSize, 2});
  sample.zero();
  for (int x = -10; x < 10; x++) {
    for (int y = -10; y < 10; y++) {
      // Load the inputs:
      sample.getHandle<>().at({0, 0}) = float(x) / 10;
      sample.getHandle<>().at({0, 1}) = float(y) / 10;

      EE_.run({A}, {&sample});

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
    EE_.run({A}, {&sample});
    auto SMH = result->getVariable()->getPayload().getHandle<>();
    auto A = SMH.at({0, 0});
    auto B = SMH.at({0, 1});
    EXPECT_TRUE(B > (A + 0.2));
  }

  {
    // Far away dot must be zero.
    sample.getHandle<>().at({0, 0}) = 1;
    sample.getHandle<>().at({0, 1}) = 1;
    EE_.run({A}, {&sample});
    auto SMH = result->getVariable()->getPayload().getHandle<>();
    auto A = SMH.at({0, 0});
    auto B = SMH.at({0, 1});
    EXPECT_TRUE(A > (B + 0.2));
  }
}

TEST_P(MLTest, learnSingleValueConcat) {
  unsigned width = 6;

  // Learning a single input vector.
  EE_.getConfig().momentum = 0.9;
  EE_.getConfig().learningRate = 0.01;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("learnSingleValueConcat");

  auto *Ex =
      mod.createVariable(ElemKind::FloatTy, {1, width * 2}, "Ex",
                         VisibilityKind::Public, Variable::TrainKind::None);

  // Left side of the network:
  auto *A =
      mod.createVariable(ElemKind::FloatTy, {1, width}, "A",
                         VisibilityKind::Public, Variable::TrainKind::None);
  Node *L = F->createFullyConnected("fc1", A, width);
  L = F->createSigmoid("", L);

  // Right side of the network:
  auto *B =
      mod.createVariable(ElemKind::FloatTy, {1, width}, "B",
                         VisibilityKind::Public, Variable::TrainKind::None);
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

  Function *TF = glow::differentiate(F, EE_.getConfig());
  EE_.compile(CompilationMode::Train, TF);

  // Train the network:
  EE_.runBatch(1000, {A, B, Ex}, {&inputs, &inputs, &expected});

  EE_.compile(CompilationMode::Infer, F);

  // Testing the output vector.
  EE_.run({A}, {&inputs});
  auto RNWH = result->getVariable()->getPayload().getHandle<>();
  (void)RNWH;

  // Test the output:
  EXPECT_NEAR(RNWH.at({0, 0}), 0.9, 0.1);
}

void buildGRU(Function *F, const std::vector<Node *> &slicesX,
              unsigned hiddenSize, unsigned outputSize,
              std::vector<NodeValue> &outputs) {
  return F->createGRU("GRU", slicesX, 1, hiddenSize, outputSize, outputs);
};

void buildRNN(Function *F, const std::vector<Node *> &slicesX,
              unsigned hiddenSize, unsigned outputSize,
              std::vector<NodeValue> &outputs) {
  return F->createSimpleRNN("SimpleRNN", slicesX, 1, hiddenSize, outputSize,
                            outputs);
};

void buildLSTM(Function *F, const std::vector<Node *> &slicesX,
               unsigned hiddenSize, unsigned outputSize,
               std::vector<NodeValue> &outputs) {
  return F->createLSTM("LSTM", slicesX, 1, hiddenSize, outputSize, outputs);
};

using TCellGenerator = void (*)(Function *, const std::vector<Node *> &,
                                unsigned, unsigned, std::vector<NodeValue> &);

void testRNNCell(TCellGenerator cell) {
  ExecutionEngine EE;
  // Learning a single input vector.
  EE.getConfig().learningRate = 0.05;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("testRNNCell");

  const unsigned NumVectors = 3;
  const unsigned NumElements = 4;
  // Create a variable with 1 input, which is 3 consecutive vectors
  // of 4 elements each.
  auto *X =
      mod.createVariable(ElemKind::FloatTy, {1, NumVectors, NumElements}, "X",
                         VisibilityKind::Public, Variable::TrainKind::None);
  auto *Y =
      mod.createVariable(ElemKind::FloatTy, {1, NumVectors}, "Y",
                         VisibilityKind::Public, Variable::TrainKind::None);

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

  std::vector<NodeValue> outputNodes;
  cell(F, XSliced, hiddenSize, outputSize, outputNodes);

  std::vector<NodeValue> regressionNodes;
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

TEST_P(MLTest, trainASimpleRNN) { testRNNCell(buildRNN); };

TEST_P(MLTest, trainGRU) { testRNNCell(buildGRU); };

TEST_P(MLTest, trainLSTM) { testRNNCell(buildLSTM); };

/// Learn the square root of two.
TEST_P(MLTest, learnSqrt2) {
  EE_.getConfig().learningRate = 0.03;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("Square root of 2");

  auto *A =
      mod.createVariable(ElemKind::FloatTy, {1}, "A", VisibilityKind::Public,
                         Variable::TrainKind::Broadcast, 1);
  auto *Ex =
      mod.createVariable(ElemKind::FloatTy, {1}, "Ex", VisibilityKind::Public,
                         Variable::TrainKind::None);
  Ex->getPayload().getHandle() = {2};

  Node *O = F->createMul("Mult", A, A);
  O = F->createRegression("reg", O, Ex);
  F->createSave("ret", O);

  Function *TF = glow::differentiate(F, EE_.getConfig());
  EE_.compile(CompilationMode::Train, TF);

  // Train the network:
  for (int i = 0; i < 50; i++) {
    EE_.run({}, {});
  }

  float res = A->getPayload().getHandle().at({0});
  EXPECT_NEAR(res, 1.4142, 0.01);
}

TEST_P(MLTest, trainSimpleLinearRegression) {
  // Given 1-D vectors x and y, find real numbers m and b such that
  // m * x + b is approximately equal to y.
  unsigned numSamples = 500;

  EE_.getConfig().learningRate = 0.1;
  EE_.getConfig().batchSize = numSamples;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("Gradient descent solution for simple linear regression");

  // These m and b are only used to generate training data.
  float referenceM = 3.0;
  float referenceB = 6.0;

  Tensor tensorX(ElemKind::FloatTy, {numSamples, 1});
  Tensor tensorY(ElemKind::FloatTy, {numSamples, 1});
  for (unsigned i = 0; i < numSamples; i++) {
    float x_i = -2.0 + 4.0 * i / numSamples;
    float y_i = referenceM * x_i + referenceB + mod.getPRNG().nextRand() / 10.0;
    tensorX.getHandle<>().at({i, 0}) = x_i;
    tensorY.getHandle<>().at({i, 0}) = y_i;
  }

  // Create a variable with 1 input, which is a real number.
  auto *inputX =
      mod.createVariable(ElemKind::FloatTy, {numSamples, 1}, "input",
                         VisibilityKind::Public, Variable::TrainKind::None);
  auto *expectedY =
      mod.createVariable(ElemKind::FloatTy, {numSamples, 1}, "expected",
                         VisibilityKind::Public, Variable::TrainKind::None);

  FullyConnectedNode *FC = F->createFullyConnected("fc", inputX, 1);
  Node *R = F->createRegression("reg", FC, expectedY);
  F->createSave("return", R);

  Variable *M = llvm::cast<Variable>(FC->getWeights());
  Variable *B = llvm::cast<Variable>(FC->getBias());

  Function *TF = glow::differentiate(F, EE_.getConfig());
  EE_.compile(CompilationMode::Train, TF);

  // Train the network doing 100 steps. Learn on 500 samples.
  EE_.runBatch(100, {inputX, expectedY}, {&tensorX, &tensorY});

  // Testing trained m and b:
  EXPECT_NEAR(M->getPayload().getHandle<>().at({0, 0}), referenceM, 0.01);
  EXPECT_NEAR(B->getPayload().getHandle<>().at({0}), referenceB, 0.01);
}

enum class Sport : size_t { BASKETBALL = 0, SOCCER = 1 };

void generatePlayerData(Tensor &players, Tensor &labels,
                        unsigned numTrainPlayers, PseudoRNG &PRNG) {
  auto P = players.getHandle<>();
  auto L = labels.getHandle<size_t>();

  // Auto generate height/weights for basketball players.
  for (size_t i = 0; i < numTrainPlayers / 2; i++) {
    auto heightInches = PRNG.nextRandInt(70, 88);
    auto weightLbs =
        4 * heightInches + PRNG.nextRandInt(-85, -55); // [195, 297]
    P.at({i, 0}) = heightInches;
    P.at({i, 1}) = weightLbs;
    L.at({i, 0}) = static_cast<size_t>(Sport::BASKETBALL);
  }

  // Auto generate height/weights for soccer players.
  for (size_t i = numTrainPlayers / 2; i < numTrainPlayers; i++) {
    auto heightInches = PRNG.nextRandInt(60, 76);
    auto weightLbs = static_cast<unsigned>(2 * heightInches) +
                     PRNG.nextRandInt(20, 50); // [140, 202]
    P.at({i, 0}) = heightInches;
    P.at({i, 1}) = weightLbs;
    L.at({i, 0}) = static_cast<size_t>(Sport::SOCCER);
  }
}

// Given a player's height and weight, classify them as a basketball or
// soccer player.
TEST_P(MLTest, classifyPlayerSport) {
  const unsigned numTrainPlayers = 200;
  const size_t numFeatures = 2;
  const size_t numClasses = 2;

  EE_.getConfig().learningRate = 0.05;
  EE_.getConfig().batchSize = numTrainPlayers;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("classifyPlayers");

  auto *A =
      mod.createVariable(ElemKind::FloatTy, {numTrainPlayers, numFeatures}, "A",
                         VisibilityKind::Public, Variable::TrainKind::None);
  auto *S =
      mod.createVariable(ElemKind::IndexTy, {numTrainPlayers, 1}, "S",
                         VisibilityKind::Public, Variable::TrainKind::None);

  auto *FC = F->createFullyConnected("fc", A, numClasses);
  auto *SM = F->createSoftMax("softmax", FC, S);
  auto *result = F->createSave("result", SM);

  Function *TF = glow::differentiate(F, EE_.getConfig());
  EE_.compile(CompilationMode::Train, TF);

  Tensor players(ElemKind::FloatTy, {numTrainPlayers, numFeatures});
  Tensor labels(ElemKind::IndexTy, {numTrainPlayers, 1});
  generatePlayerData(players, labels, numTrainPlayers, mod.getPRNG());

  // Training:
  EE_.runBatch(2000, {A, S}, {&players, &labels});

  EE_.compile(CompilationMode::Infer, F);

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

  EE_.run({A}, {&testPlayersTensor});

  for (size_t i = 0; i < testPlayers.size(); i++) {
    auto SMH = result->getVariable()->getPayload().getHandle<>();
    const size_t sport = static_cast<size_t>(std::get<2>(testPlayers[i]));
    EXPECT_NEAR(SMH.at({i, sport}), 1.0, 0.1);
  }
}

TEST_P(MLTest, learnSinus) {
  // Try to learn the sin(x) function.
  float epsilon = 0.1;
  unsigned numSamples = 50;

  EE_.getConfig().learningRate = 0.2;
  EE_.getConfig().batchSize = numSamples;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("Gradient descent solution for sin(x)");

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

  auto *inputX =
      mod.createVariable(ElemKind::FloatTy, {numSamples, 1}, "input",
                         VisibilityKind::Public, Variable::TrainKind::None);

  auto *expectedY =
      mod.createVariable(ElemKind::FloatTy, {numSamples, 1}, "expected",
                         VisibilityKind::Public, Variable::TrainKind::None);

  FullyConnectedNode *FC1 = F->createFullyConnected("fc1", inputX, 10);
  Node *O = F->createSigmoid("sigmoid1", FC1);
  FullyConnectedNode *FC2 = F->createFullyConnected("fc2", O, 1);
  Node *R = F->createRegression("reg", FC2, expectedY);
  auto *result = F->createSave("return", R);

  Function *TF = glow::differentiate(F, EE_.getConfig());
  EE_.compile(CompilationMode::Train, TF);

  // Learn on numSamples samples.
  EE_.runBatch(2700, {inputX, expectedY}, {&tensorX, &tensorY});

  // Create a test set, which is similar, but different from the training set.
  for (unsigned i = 0; i < numSamples; i++) {
    // Scale x to cover the range [0, 4.2] as this leads to a good convergence
    // during training.
    float x = i / (numSamples / 4.2) + 0.123456;
    float y = FF(x);
    tensorX.getHandle<>().at({i, 0}) = x;
    tensorY.getHandle<>().at({i, 0}) = y;
  }

  EE_.compile(CompilationMode::Infer, F);
  EE_.run({inputX}, {&tensorX});
  auto resH = result->getVariable()->getPayload().getHandle<>();

  for (size_t i = 0; i < numSamples; i++) {
    float x = tensorX.getHandle().at({i, 0});
    EXPECT_NEAR(resH.at({i, 0}), FF(x), epsilon);
  }
}

TEST_P(MLTest, nonLinearClassifier) {
  // Test non-linear classification on a set of 2d points. Generate x and y in
  // (-1, 1) and classify according to XOR of the sign bit.
  unsigned batchSize = 46;
  unsigned numSamples = 230;

  EE_.getConfig().learningRate = 0.01;
  EE_.getConfig().momentum = 0.9;
  EE_.getConfig().batchSize = batchSize;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("nonLinearClassifier");

  auto *A =
      mod.createVariable(ElemKind::FloatTy, {batchSize, 2}, "A",
                         VisibilityKind::Public, Variable::TrainKind::None);
  auto *S =
      mod.createVariable(ElemKind::IndexTy, {batchSize, 1}, "S",
                         VisibilityKind::Public, Variable::TrainKind::None);

  auto *FCL0 = F->createFullyConnected("fc1", A, 8);
  auto *T0 = F->createTanh("tanh1", FCL0);
  auto *FCL1 = F->createFullyConnected("fc2", T0, 8);
  auto *T1 = F->createTanh("tanh2", FCL1);
  auto *FCL2 = F->createFullyConnected("fc2", T1, 2);
  auto *SM = F->createSoftMax("soft", FCL2, S);
  auto *result = F->createSave("ret", SM);

  Function *TF = glow::differentiate(F, EE_.getConfig());
  EE_.compile(CompilationMode::Train, TF);

  Tensor samples(ElemKind::FloatTy, {numSamples, 2});
  Tensor labels(ElemKind::IndexTy, {numSamples, 1});

  for (size_t i = 0; i < numSamples; i++) {
    float x = mod.getPRNG().nextRand();
    float y = mod.getPRNG().nextRand();
    size_t label = (x < 0.0) ^ (y < 0.0);
    samples.getHandle<>().at({i, 0}) = x;
    samples.getHandle<>().at({i, 1}) = y;
    labels.getHandle<size_t>().at({i, 0}) = label;
  }

  EE_.runBatch(500, {A, S}, {&samples, &labels});

  EE_.compile(CompilationMode::Infer, F);

  std::vector<std::tuple<float, float, size_t>> tests;
  tests.emplace_back(-0.8, -0.8, 0);
  tests.emplace_back(0.8, -0.8, 1);
  tests.emplace_back(-0.8, 0.8, 1);
  tests.emplace_back(0.8, 0.8, 0);
  for (size_t i = 0; i < tests.size(); i++) {
    Tensor T(ElemKind::FloatTy, {batchSize, 2});
    T.getHandle<>().at({0, 0}) = std::get<0>(tests[i]);
    T.getHandle<>().at({0, 1}) = std::get<1>(tests[i]);
    EE_.run({A}, {&T});
    auto RH = result->getVariable()->getPayload().getHandle<>();
    EXPECT_NEAR(RH.at({0, std::get<2>(tests[i])}), 1.0, 0.2);
  }
}

/// Generate images in two classes.
/// A "line" is labeled as 0 and a "cross" is labeled as 1.
static void generateImageData(Tensor &images, Tensor &labels, PseudoRNG &PRNG) {
  auto L = labels.getHandle<size_t>();
  auto image = images.getHandle<>();
  unsigned numSamples = images.dims()[0];
  images.zero();

  for (size_t i = 0; i < numSamples; i++) {
    bool isLine = i % 2 == 0;
    L.at({i, 0}) = isLine ? 0 : 1;
    size_t target = PRNG.nextRandInt(1, 6);
    if (isLine) {
      for (size_t y = 0; y < 8; y++)
        image.at({i, target, y, 0u}) = 1;
    } else {
      for (size_t pos = 0; pos < 8; pos++) {
        image.at({i, pos, target, 0u}) = 1;
        image.at({i, target, pos, 0u}) = 1;
      }
    }
  }
}

/// Test the convolutional layer.
/// Use a simple convnet to learn two classes of images: Line and Cross.
/// This test checks the results of the quantized network.
TEST_P(InterpreterAndCPU, convNetForImageRecognition) {
  ExecutionEngine EE{BackendKind::Interpreter};
  const unsigned numSamples = 500;
  const unsigned batchSize = 7;

  EE.getConfig().learningRate = 0.01;
  EE.getConfig().batchSize = batchSize;
  EE.getConfig().momentum = 0.9;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("convNetForImageRecognition");

  auto *input =
      mod.createVariable(ElemKind::FloatTy, {batchSize, 8, 8, 1}, "input",
                         VisibilityKind::Public, Variable::TrainKind::None);

  auto *ex =
      mod.createVariable(ElemKind::IndexTy, {batchSize, 1}, "exp",
                         VisibilityKind::Public, Variable::TrainKind::None);

  auto *CV = F->createConv("conv", input, 1, 3, 1, 0, 1);
  auto *TANH = F->createTanh("tanh", CV);
  auto *FCL = F->createFullyConnected("fc", TANH, 2);
  auto *SM = F->createSoftMax("sm", FCL, ex);
  auto *result = F->createSave("ret", SM);
  Function *TF = glow::differentiate(F, EE.getConfig());
  EE.compile(CompilationMode::Train, TF);

  Tensor images(ElemKind::FloatTy, {numSamples, 8, 8, 1});
  Tensor labels(ElemKind::IndexTy, {numSamples, 1});
  generateImageData(images, labels, mod.getPRNG());

  // Training:
  EE.runBatch(500, {input, ex}, {&images, &labels});

  // Profiling:
  Function *PF = glow::profileQuantization(F);
  EE.compile(CompilationMode::Infer, PF);
  EE.runBatch(100, {input}, {&images});

  // Get the quantization info and build the new quantized graph.
  std::vector<NodeQuantizationInfo> QI =
      quantization::generateNodeQuantizationInfos(PF);
  Function *QP = quantization::quantizeFunction(EE, QI, F);

  // Evaluate on the quantized function:
  // Set the execution backend to the backend that we test.
  EE.setBackend(GetParam());
  EE.compile(CompilationMode::Infer, QP);
  // Generate the images used for testing.
  Tensor testImages(ElemKind::FloatTy, {batchSize, 8, 8, 1});
  Tensor testLabels(ElemKind::IndexTy, {batchSize, 1});
  generateImageData(testImages, testLabels, mod.getPRNG());
  EE.run({input}, {&testImages});
  auto SMH = result->getVariable()->getHandle<>();
  for (size_t i = 0; i < batchSize; i++) {
    bool isLine = testLabels.getHandle<size_t>().at({i, 0}) == 0;
    auto lineWeight = SMH.at({i, 0});
    auto crossWeight = SMH.at({i, 1});
    EXPECT_TRUE((isLine && lineWeight > crossWeight) ||
                (!isLine && crossWeight > lineWeight));
  }
}

/// Generate data for the regression test. Put a '1' in a random location in a
/// clear tensor and report the coordinates of that pixel.
static void generateRegressionTestData(Tensor &images, Tensor &labels,
                                       PseudoRNG &PRNG) {
  auto L = labels.getHandle<>();
  auto image = images.getHandle<>();
  unsigned numSamples = images.dims()[0];
  image.clear(0);

  for (size_t i = 0; i < numSamples; i++) {
    // Generate the X,Y coordinates to place our object.
    size_t x = PRNG.nextRandInt(0, 9);
    size_t y = PRNG.nextRandInt(0, 9);
    L.at({i, 0}) = x;
    L.at({i, 1}) = y;
    image.at({i, x, y, 0u}) = 1;
  }
}

/// This is the "Where's Waldo" test. We place a pixel in a tensor and the
/// network reports the coordinate of the pixel.
TEST_P(InterpreterAndCPU, testFindPixelRegression) {
  ExecutionEngine EE{BackendKind::Interpreter};

  const unsigned numSamples = 1000;
  const unsigned batchSize = 10;

  EE.getConfig().learningRate = 0.01;
  EE.getConfig().batchSize = batchSize;
  EE.getConfig().momentum = 0.9;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *input =
      mod.createVariable(ElemKind::FloatTy, {batchSize, 10, 10, 1}, "input",
                         VisibilityKind::Public, Variable::TrainKind::None);
  auto *ex =
      mod.createVariable(ElemKind::FloatTy, {batchSize, 2}, "coordinates",
                         VisibilityKind::Public, Variable::TrainKind::None);

  // A simple single-layer FC network could solve this program but we use a two
  // layer FC network to give the compiler something slightly more complex to
  // work with so we are adding another FC layer.
  auto *FC0 = F->createFullyConnected("fc0", input, 6);
  auto *RL0 = F->createRELU("relu0", FC0);
  auto *FC1 = F->createFullyConnected("fc1", RL0, 2);
  auto *R = F->createRegression("regression", FC1, ex);
  auto *result = F->createSave("ret", R);
  Function *TF = glow::differentiate(F, EE.getConfig());
  EE.compile(CompilationMode::Train, TF);

  // --  STEP1 - train the network. --
  Tensor images(ElemKind::FloatTy, {numSamples, 10, 10, 1});
  Tensor labels(ElemKind::FloatTy, {numSamples, 2});
  generateRegressionTestData(images, labels, mod.getPRNG());

  // Training:
  EE.runBatch(400, {input, ex}, {&images, &labels});

  // -- STEP2 - Profile and quantize the network. --

  // Profiled 'F'.
  Function *PF = glow::profileQuantization(F);
  EE.compile(CompilationMode::Infer, PF);

  // Run the graph to capture the profile.
  EE.runBatch(100, {input}, {&images});

  // Get quantization infos and build new quantized graph.
  std::vector<NodeQuantizationInfo> QI =
      quantization::generateNodeQuantizationInfos(PF);

  Function *QP = quantization::quantizeFunction(EE, QI, F);

  // -- STEP3 - evaluate the quantized function. --

  // Set the execution backend to the backend that we test.
  EE.setBackend(GetParam());

  EE.compile(CompilationMode::Infer, QP);

  // Generate the images used for testing.
  Tensor testImages(ElemKind::FloatTy, {batchSize, 10, 10, 1});
  Tensor testLabels(ElemKind::FloatTy, {batchSize, 2});
  generateRegressionTestData(testImages, testLabels, mod.getPRNG());

  // Run the inference:
  EE.run({input}, {&testImages});

  // A handle to the projected result.
  auto RH = result->getVariable()->getHandle<>();
  // A handle to the true label.
  auto LH = testLabels.getHandle<>();

  // Check how many of the coordinates that were reported are close to the real
  // values.
  unsigned correct = 0;

  for (size_t i = 0; i < batchSize; i++) {
    // Calculate the distance between the predicted value and correct value.
    auto dx = LH.at({i, 0}) - RH.at({i, 0});
    auto dy = LH.at({i, 1}) - RH.at({i, 1});
    auto distance = std::sqrt(std::pow(dx, 2) + std::pow(dy, 2));

    // A correct answer is one where the projected distance is somewhat close.
    correct += distance < 3;
  }

  // Expect 90% of the results to be correct.
  EXPECT_GE(correct, batchSize * 0.90);
}

// Generate tests for a toy neural network that can recognize if a matrix 3x3
// is a rotation of another matrix 3x3.
// This is *not* about rotation matrices used for computer graphics, but a much
// simpler concept.
// Informally, let M1 and M2 be two 3x3 matrices, M2 is a rotation of M1 if it
// exists a way to rotate the cells of M1 to get M2. The rotations are all
// centered in the middle of the matrices.
// E.g.,
// Rotate clockwise 1 cell centered in 'e':
//              --+
//      | a b c | |
// M1 = | d e f | V
//      | g h i |
// =>
//      | d a b |
// M2 = | g e c |
//      | h i f |
static void generateMatrixRotationRecognitionData(Tensor &matricesA,
                                                  Tensor &matricesB,
                                                  Tensor &expected,
                                                  PseudoRNG &PRNG) {

  using CellIdx = std::pair<uint8_t, uint8_t>;
  // List the indices in a clockwise ordering starting from the top left
  // corner.
  // Note: This does not include the cell in the middle given it is
  // never rotated.
  static const CellIdx clockwiseOrder[] = {{0, 0}, {0, 1}, {0, 2}, {1, 2},
                                           {2, 2}, {2, 1}, {2, 0}, {1, 0}};
  static const uint8_t possibleTargetCells =
      sizeof(clockwiseOrder) / sizeof(clockwiseOrder[0]);
  const unsigned numSamples = matricesA.dims()[0];
  assert(expected.dims()[0] == numSamples &&
         matricesB.dims()[0] == numSamples &&
         "Size of the tensors is incompatible");
  auto handleMatricesA = matricesA.getHandle<float>();
  auto handleMatricesB = matricesB.getHandle<float>();
  auto handleExpected = expected.getHandle<size_t>();

  handleMatricesA.randomize<int>(0, 1, PRNG);
  handleMatricesB.randomize<int>(0, 1, PRNG);
  for (unsigned idx = 0; idx < numSamples; ++idx) {
    // Toss a coin and create a rotation relationship or not.
    bool shouldHaveRotation = PRNG.nextRandInt(0, 1);
    handleExpected.at({idx, 0}) = shouldHaveRotation;
    if (shouldHaveRotation) {
      // On a 3x3 matrix we have 8 different possbile clockwise steps.
      // Pick one.
      size_t clockwiseSteps = PRNG.nextRandInt(0, possibleTargetCells - 1);
      // Generate the rotation matrix from A.
      // The center never changes.
      handleMatricesB.at({idx, 1, 1}) = handleMatricesA.at({idx, 1, 1});
      // Fetch the cell registered in the clockwiseOrder at the desired step.
      for (size_t i = 0; i != possibleTargetCells; ++i) {
        const CellIdx &sourceCellIdx = clockwiseOrder[i];
        const CellIdx &targetCellIdx =
            clockwiseOrder[(i + clockwiseSteps) % possibleTargetCells];
        handleMatricesB.at({idx, targetCellIdx.first, targetCellIdx.second}) =
            handleMatricesA.at(
                {idx, sourceCellIdx.first, sourceCellIdx.second});
      }
    }
    // Else:
    // There is a high probability that A and B don't have a rotation
    // relationship and thus, there is nothing to do.
    // Worse case, we mislabeled a relationship.

    // Alternatively we could always alter one of the matrix such that it is
    // impossible to have a rotation between them (e.g., make sure the center
    // is different), but that would bias the kind of differences that could
    // occur.
  }
}

TEST_P(MLTest, matrixRotationRecognition) {
  TrainingConfig &config = EE_.getConfig();
  config.learningRate = 0.15;
  config.batchSize = 17;

  Module &mod = EE_.getModule();
  PseudoRNG &PRNG = mod.getPRNG();
  Function *F = mod.createFunction("MatrixRotationRecognition");
  Variable *varMatricesA =
      mod.createVariable(ElemKind::FloatTy, {config.batchSize, 3, 3}, "matrixA",
                         VisibilityKind::Public, Variable::TrainKind::None);
  Variable *varMatricesB =
      mod.createVariable(ElemKind::FloatTy, {config.batchSize, 3, 3}, "matrixB",
                         VisibilityKind::Public, Variable::TrainKind::None);
  Variable *varExpected =
      mod.createVariable(ElemKind::IndexTy, {config.batchSize, 1}, "expected",
                         VisibilityKind::Public, Variable::TrainKind::None);

  // Simply concatenating the matrices first would probability be as effective
  // but we want to build something more complex than a straight chain of
  // operators to stress the scheduler.
  auto *FCA = F->createFullyConnected("hidden_matrixA_fc", varMatricesA, 10);
  auto *FCB = F->createFullyConnected("hidden_matrixB_fc", varMatricesB, 10);
  auto *ReLUA = F->createRELU("hidden_matrixA_ReLU", FCA);
  auto *ReLUB = F->createRELU("hidden_matrixB_ReLU", FCB);
  auto *concat = F->createConcat("hidden_concat_A_B", {ReLUA, ReLUB}, 1);
  auto *hiddenFC = F->createFullyConnected("hidden_fc", concat, 30);
  auto *finalReLU = F->createRELU("hidden_concat_ReLU", hiddenFC);
  auto *finalFC = F->createFullyConnected("output_fc", finalReLU, 2);
  auto *softMax = F->createSoftMax("output", finalFC, varExpected);
  SaveNode *forwardInferenceResult = F->createSave("result", softMax);
  Function *trainingGradientFunction = glow::differentiate(F, config);

  // Train the network.
  const unsigned numSamples = 50;
  Tensor matricesA(ElemKind::FloatTy, {numSamples, 3, 3});
  Tensor matricesB(ElemKind::FloatTy, {numSamples, 3, 3});
  Tensor expected(ElemKind::IndexTy, {numSamples, 1});
  generateMatrixRotationRecognitionData(matricesA, matricesB, expected, PRNG);

  EE_.compile(CompilationMode::Train, trainingGradientFunction);
  // Training:
  EE_.runBatch(200, {varMatricesA, varMatricesB, varExpected},
              {&matricesA, &matricesB, &expected});

  // Switch to inference mode.
  EE_.compile(CompilationMode::Infer, F);

  // At this point we should have overfitted the data.
  // Take a random batch and check that the values match what we expect.
  auto RHtrain = forwardInferenceResult->getVariable()->getHandle<>();
  auto batchSize = config.batchSize;
  unsigned numBatches = numSamples / batchSize;
  unsigned batchStartIdx = PRNG.nextRandInt(0, numBatches - 1) * batchSize;

  auto batchMatricesA =
      matricesA.getUnowned({batchSize, 3, 3}, {batchStartIdx, 0, 0});
  auto batchMatricesB =
      matricesB.getUnowned({batchSize, 3, 3}, {batchStartIdx, 0, 0});
  EE_.run({varMatricesA, varMatricesB}, {&batchMatricesA, &batchMatricesB});

  unsigned errors = 0;
  // Check each output in the batch.
  for (size_t i = 0; i != batchSize; i++) {
    // Note that the two softmax outputs always sum to 1, so we only look at
    // one. Index one is true if there is a rotation.
    float value = RHtrain.at({i, 1});
    bool hasRotation = expected.getHandle<size_t>().at({batchStartIdx + i, 0});
    if ((value > 0.5) != hasRotation) {
      ++errors;
    }
  }

  EXPECT_LE(errors, 1);
}

INSTANTIATE_TEST_CASE_P(Interpreter, MLTest,
                        ::testing::Values(BackendKind::Interpreter));
#ifdef GLOW_WITH_CPU
INSTANTIATE_TEST_CASE_P(JIT, MLTest, ::testing::Values(BackendKind::CPU));
#endif
#ifdef GLOW_WITH_OPENCL
INSTANTIATE_TEST_CASE_P(OpenCL, MLTest, ::testing::Values(BackendKind::OpenCL));
#endif

INSTANTIATE_TEST_CASE_P(Interpreter, InterpreterAndCPU,
                        ::testing::Values(BackendKind::Interpreter));
#ifdef GLOW_WITH_CPU
INSTANTIATE_TEST_CASE_P(JIT, InterpreterAndCPU,
                        ::testing::Values(BackendKind::CPU));
#endif
