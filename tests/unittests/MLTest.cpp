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

class TestRunnerBase : public ::testing::TestWithParam<std::string> {
public:
  ExecutionEngine EE_{GetParam()};
};

class InterpreterAndCPU : public TestRunnerBase {};
class MLTest : public TestRunnerBase {};

/// Use placeholders (and not variables) to learn the square root of two.
TEST_P(MLTest, learnSqrt2Placeholder) {
  TrainingConfig TC;
  PlaceholderBindings bindings;

  TC.learningRate = 0.03;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("Square root of 2");

  auto *A = mod.createPlaceholder(ElemKind::FloatTy, {1}, "A", true);
  auto *inputTensor = bindings.allocate(A);
  inputTensor->init(Tensor::InitKind::Broadcast, 1, mod.getPRNG());

  auto *E = mod.createPlaceholder(ElemKind::FloatTy, {1}, "Ex", false);
  bindings.allocate(E)->getHandle() = {2};

  auto *O = mod.createPlaceholder(ElemKind::FloatTy, {1}, "output", false);
  bindings.allocate(O);

  Node *M = F->createMul("Mult", A, A);
  M = F->createRegression("reg", M, E);
  SaveNode *SN = F->createSave("ret", M);

  bindings.allocate(SN->getPlaceholder());

  Function *TF = glow::differentiate(F, TC);
  EE_.compile(CompilationMode::Train, TF);

  // Train the network:
  for (int i = 0; i < 100; i++) {
    EE_.run(bindings);
  }

  float res = inputTensor->getHandle().at({0});
  EXPECT_NEAR(res, 1.4142, 0.01);
}

TEST_P(MLTest, trainASimpleNetwork) {
  TrainingConfig TC;
  PlaceholderBindings bindings;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  // Learning a single input vector.
  TC.learningRate = 0.05;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("trainASimpleNetwork");

  // Create a variable with 1 input, which is a vector of 4 elements.
  Placeholder *A = mod.createPlaceholder(ElemKind::FloatTy, {1, 4}, "A", false);
  Placeholder *E = mod.createPlaceholder(ElemKind::FloatTy, {1, 4}, "E", false);
  Node *O = F->createFullyConnected(bindings, "fc1", A, 10);
  O = F->createSigmoid("sig1", O);
  O = F->createFullyConnected(bindings, "fc2", O, 4);
  O = F->createRegression("reg", O, E);
  SaveNode *result = F->createSave("return", O);

  bindings.allocate(A);
  bindings.allocate(E);
  auto *res = bindings.allocate(result->getPlaceholder());

  // Values for the input and output variables.
  Tensor inputs(ElemKind::FloatTy, {1, 4});
  Tensor expected(ElemKind::FloatTy, {1, 4});
  inputs.getHandle<>() = {0.15f, 0.15f, 0.15f, 0.15f};
  expected.getHandle<>() = {0.9f, 0.9f, 0.9f, 0.9f};

  Function *TF = glow::differentiate(F, TC);
  EE_.compile(CompilationMode::Train, TF);

  // Train the network. Learn 1000 batches.
  runBatch(EE_, bindings, 1000, sampleCounter, {A, E}, {&inputs, &expected});

  // Testing the output vector.

  EE_.compile(CompilationMode::Infer, F);
  updateInputPlaceholders(bindings, {A}, {&inputs});
  EE_.run(bindings);

  auto RNWH = res->getHandle<>();
  (void)RNWH;

  // Test the output:
  EXPECT_NEAR(RNWH.at({0, 0}), 0.9, 0.05);
}

TEST_P(MLTest, simpleRegression) {
  TrainingConfig TC;
  PlaceholderBindings bindings;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  // Testing the regression layer. This test takes the first element from the
  // input vector, adds one to it and places the result in the second element of
  // the output vector.
  const size_t numInputs = 4;

  // Learning a single input vector.
  TC.learningRate = 0.05;

  Tensor inputs(ElemKind::FloatTy, {1, numInputs});
  Tensor expected(ElemKind::FloatTy, {1, numInputs});

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("simpleRegression");
  Placeholder *A =
      mod.createPlaceholder(ElemKind::FloatTy, {1, numInputs}, "A", false);
  Placeholder *Ex =
      mod.createPlaceholder(ElemKind::FloatTy, {1, numInputs}, "E", false);
  Node *O = F->createFullyConnected(bindings, "fc", A, 4);
  O = F->createRELU("relu", O);
  O = F->createRegression("reg", O, Ex);
  auto *result = F->createSave("result", O);

  bindings.allocate(A);
  bindings.allocate(Ex);
  auto *res = bindings.allocate(result->getPlaceholder());

  auto I = inputs.getHandle<>();
  auto E = expected.getHandle<>();

  Function *TF = glow::differentiate(F, TC);
  EE_.compile(CompilationMode::Train, TF);

  // Train the network:
  for (int iter = 0; iter < 1000; iter++) {
    float target = float(iter % 9);
    I = {target, 0., 0., 0.};
    E = {0., target + 1, 0., 0.};
    runBatch(EE_, bindings, 1, sampleCounter, {A, Ex}, {&inputs, &expected});
  }

  // Verify the result of the regression layer.
  EE_.compile(CompilationMode::Infer, F);

  // Test the output:
  for (int iter = 0; iter < 5; iter++) {
    float target = iter % 9 + 1;
    I = {target, 0., 0., 0.};
    updateInputPlaceholders(bindings, {A}, {&inputs});
    EE_.run(bindings);

    auto resH = res->getHandle<>();
    (void)resH;

    EXPECT_NEAR(I.at({0, 0}) + 1, resH.at({0, 1}), 0.1);
  }
}

TEST_P(MLTest, learnXor) {
  TrainingConfig TC;
  PlaceholderBindings bindings;

  unsigned numInputs = 10;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  // Learning a single input vector.
  TC.learningRate = 0.05;
  TC.batchSize = numInputs;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("learnXor");

  Placeholder *A =
      mod.createPlaceholder(ElemKind::FloatTy, {numInputs, 2}, "A", false);
  Placeholder *Ex =
      mod.createPlaceholder(ElemKind::FloatTy, {numInputs, 1}, "Ex", false);

  Node *O = F->createFullyConnected(bindings, "fc1", A, 6);
  O = F->createTanh("tanh1", O);
  O = F->createFullyConnected(bindings, "fc2", O, 1);
  O = F->createRegression("reg", O, Ex);
  auto *result = F->createSave("ret", O);

  bindings.allocate(A);
  bindings.allocate(Ex);
  auto *res = bindings.allocate(result->getPlaceholder());

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

  Function *TF = glow::differentiate(F, TC);
  EE_.compile(CompilationMode::Train, TF);

  // Train the network:
  runBatch(EE_, bindings, 2500, sampleCounter, {A, Ex},
           {&trainingSet, &trainingLabels});

  EE_.compile(CompilationMode::Infer, F);

  // Prepare the testing tensor:
  for (unsigned i = 0; i < numInputs; i++) {
    int a = (numInputs - i) % 2;
    int b = ((numInputs - i) >> 1) % 2;
    TS.at({i, 0}) = a;
    TS.at({i, 1}) = b;
  }

  updateInputPlaceholders(bindings, {A}, {&trainingSet});
  EE_.run(bindings);

  auto resH = res->getHandle<>();

  // Test the output:
  for (size_t i = 0; i < numInputs; i++) {
    int a = TS.at({i, 0});
    int b = TS.at({i, 1});
    EXPECT_NEAR(resH.at({i, 0}), (a ^ b), 0.1);
  }
}

/// Learn the logarithmic function.
TEST_P(MLTest, learnLog) {
  TrainingConfig TC;
  PlaceholderBindings bindings;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  unsigned numInputs = 50;
  unsigned batchSize = 5;
  TC.learningRate = 0.07;
  TC.batchSize = batchSize;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("learnLog");

  Placeholder *A =
      mod.createPlaceholder(ElemKind::FloatTy, {batchSize, 1}, "A", false);
  Placeholder *Ex =
      mod.createPlaceholder(ElemKind::FloatTy, {batchSize, 1}, "Ex", false);

  Node *O = F->createFullyConnected(bindings, "fc1", A, 4);
  O = F->createTanh("tanh1", O);
  O = F->createFullyConnected(bindings, "fc2", O, 1);
  O = F->createRegression("reg", O, Ex);
  auto *result = F->createSave("ret", O);

  bindings.allocate(A);
  bindings.allocate(Ex);
  auto *res = bindings.allocate(result->getPlaceholder());

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

  Function *TF = glow::differentiate(F, TC);
  EE_.compile(CompilationMode::Train, TF);

  // Train the network:
  runBatch(EE_, bindings, 1000, sampleCounter, {A, Ex},
           {&trainingSet, &trainingLabels});

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

  updateInputPlaceholders(bindings, {A}, {&testSet});
  EE_.run(bindings);

  auto resH = res->getHandle<>();

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
  auto L = labels.getHandle<int64_t>();

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
  TrainingConfig TC;
  PlaceholderBindings bindings;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  unsigned minibatchSize = 11;

  // Testing the softmax layer.
  // Learning a single input vector.
  TC.momentum = 0.9;
  TC.learningRate = 0.01;
  TC.batchSize = minibatchSize;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("circle");
  Placeholder *A =
      mod.createPlaceholder(ElemKind::FloatTy, {minibatchSize, 2}, "A", false);
  Placeholder *S =
      mod.createPlaceholder(ElemKind::Int64ITy, {minibatchSize, 1}, "S", false);

  auto *FCL0 = F->createFullyConnected(bindings, "fc1", A, 8);
  auto *T0 = F->createTanh("tanh1", FCL0);
  auto *FCL1 = F->createFullyConnected(bindings, "fc2", T0, 2);
  auto *SM = F->createSoftMax("soft", FCL1, S);
  auto *result = F->createSave("ret", SM);

  bindings.allocate(A);
  bindings.allocate(S);
  auto *res = bindings.allocate(result->getPlaceholder());

  Function *TF = glow::differentiate(F, TC);
  EE_.compile(CompilationMode::Train, TF);

  Tensor coordinates(ElemKind::FloatTy, {numSamples, 2});
  Tensor labels(ElemKind::Int64ITy, {numSamples, 1});
  generateCircleData(coordinates, labels, mod.getPRNG());

  // Training:
  runBatch(EE_, bindings, 4000, sampleCounter, {A, S}, {&coordinates, &labels});

  EE_.compile(CompilationMode::Infer, F);

  // Print a diagram that depicts the network decision on a grid.
  Tensor sample(ElemKind::FloatTy, {minibatchSize, 2});
  sample.zero();
  for (int x = -10; x < 10; x++) {
    for (int y = -10; y < 10; y++) {
      // Load the inputs:
      sample.getHandle<>().at({0, 0}) = float(x) / 10;
      sample.getHandle<>().at({0, 1}) = float(y) / 10;

      updateInputPlaceholders(bindings, {A}, {&sample});
      EE_.run(bindings);

      auto SMH = res->getHandle<>();
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
    updateInputPlaceholders(bindings, {A}, {&sample});
    EE_.run(bindings);

    auto SMH = res->getHandle<>();
    auto A = SMH.at({0, 0});
    auto B = SMH.at({0, 1});
    EXPECT_TRUE(B > (A + 0.2));
  }

  {
    // Far away dot must be zero.
    sample.getHandle<>().at({0, 0}) = 1;
    sample.getHandle<>().at({0, 1}) = 1;
    updateInputPlaceholders(bindings, {A}, {&sample});
    EE_.run(bindings);
    auto SMH = res->getHandle<>();
    auto A = SMH.at({0, 0});
    auto B = SMH.at({0, 1});
    EXPECT_TRUE(A > (B + 0.2));
  }
}

TEST_P(MLTest, learnSingleValueConcat) {
  unsigned width = 6;
  PlaceholderBindings bindings;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  // Learning a single input vector.
  TrainingConfig TC;
  TC.momentum = 0.9;
  TC.learningRate = 0.01;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("learnSingleValueConcat");

  Placeholder *Ex =
      mod.createPlaceholder(ElemKind::FloatTy, {1, width * 2}, "Ex", false);

  // Left side of the network:
  Placeholder *A =
      mod.createPlaceholder(ElemKind::FloatTy, {1, width}, "A", false);
  Node *L = F->createFullyConnected(bindings, "fc1", A, width);
  L = F->createSigmoid("", L);

  // Right side of the network:
  auto *B = mod.createPlaceholder(ElemKind::FloatTy, {1, width}, "B", false);
  Node *R = F->createFullyConnected(bindings, "fc2", B, width);
  R = F->createSigmoid("sig", R);

  // Concat:
  auto *C = F->createConcat("con", {L, R}, 1);
  auto *RN = F->createRegression("reg", C, Ex);
  auto *result = F->createSave("ret", RN);

  bindings.allocate(A);
  bindings.allocate(B);
  bindings.allocate(Ex);
  auto *res = bindings.allocate(result->getPlaceholder());

  Tensor inputs(ElemKind::FloatTy, {1, width});
  Tensor expected(ElemKind::FloatTy, {1, width * 2});
  inputs.getHandle<>().clear(0.15);
  expected.getHandle<>().clear(0.9);

  Function *TF = glow::differentiate(F, TC);
  EE_.compile(CompilationMode::Train, TF);

  // Train the network:
  runBatch(EE_, bindings, 1000, sampleCounter, {A, B, Ex},
           {&inputs, &inputs, &expected});

  EE_.compile(CompilationMode::Infer, F);

  // Testing the output vector.
  updateInputPlaceholders(bindings, {A}, {&inputs});
  EE_.run(bindings);
  auto RNWH = res->getHandle<>();
  (void)RNWH;

  // Test the output:
  EXPECT_NEAR(RNWH.at({0, 0}), 0.9, 0.1);
}

void buildGRU(PlaceholderBindings &bindings, Function *F,
              const std::vector<NodeValue> &slicesX, unsigned hiddenSize,
              unsigned outputSize, std::vector<NodeValue> &outputs) {
  return F->createGRU(bindings, "GRU", slicesX, 1, hiddenSize, outputSize,
                      outputs);
};

void buildRNN(PlaceholderBindings &bindings, Function *F,
              const std::vector<NodeValue> &slicesX, unsigned hiddenSize,
              unsigned outputSize, std::vector<NodeValue> &outputs) {
  return F->createSimpleRNN(bindings, "SimpleRNN", slicesX, 1, hiddenSize,
                            outputSize, outputs);
};

void buildLSTM(PlaceholderBindings &bindings, Function *F,
               const std::vector<NodeValue> &slicesX, unsigned hiddenSize,
               unsigned outputSize, std::vector<NodeValue> &outputs) {
  return F->createLSTM(bindings, "LSTM", slicesX, 1, hiddenSize, outputSize,
                       outputs);
};

using TCellGenerator = void (*)(PlaceholderBindings &, Function *,
                                const std::vector<NodeValue> &, unsigned,
                                unsigned, std::vector<NodeValue> &);

void testRNNCell(TCellGenerator cell) {
  TrainingConfig TC;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  PlaceholderBindings bindings;
  ExecutionEngine EE;
  // Learning a single input vector.
  TC.learningRate = 0.05;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("testRNNCell");

  const unsigned NumVectors = 3;
  const unsigned NumElements = 4;
  // Create a variable with 1 input, which is 3 consecutive vectors
  // of 4 elements each.
  Placeholder *X = mod.createPlaceholder(
      ElemKind::FloatTy, {1, NumVectors, NumElements}, "X", false);
  Placeholder *Y =
      mod.createPlaceholder(ElemKind::FloatTy, {1, NumVectors}, "Y", false);
  bindings.allocate(X);
  bindings.allocate(Y);

  // Extract a slice for each input.
  std::vector<NodeValue> XSliced;

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
  cell(bindings, F, XSliced, hiddenSize, outputSize, outputNodes);

  std::vector<NodeValue> regressionNodes;
  for (unsigned t = 0; t < NumVectors; t++) {
    regressionNodes.push_back(
        F->createRegression("", outputNodes[t], YSliced[t]));
  };

  auto *R = F->createConcat("O", regressionNodes, 1);
  SaveNode *result = F->createSave("result", R);

  Tensor *res = bindings.allocate(result->getPlaceholder());

  Function *TF = glow::differentiate(F, TC);
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
  runBatch(EE, bindings, 1000, sampleCounter, {X, Y}, {&inputs, &expected});

  // Testing the output vector.
  EE.compile(CompilationMode::Infer, F);

  updateInputPlaceholders(bindings, {X}, {&inputs});
  EE.run(bindings);

  auto RNWH = res->getHandle<>();
  (void)RNWH;

  // Test the output:
  for (size_t t = 0; t < NumVectors; ++t) {
    EXPECT_NEAR(RNWH.at({0, t}), t, 0.05);
  }
};

TEST_P(MLTest, trainASimpleRNN) { testRNNCell(buildRNN); };

TEST_P(MLTest, trainGRU) { testRNNCell(buildGRU); };

TEST_P(MLTest, trainLSTM) { testRNNCell(buildLSTM); };

TEST_P(MLTest, trainSimpleLinearRegression) {
  TrainingConfig TC;
  PlaceholderBindings bindings;

  // Given 1-D vectors x and y, find real numbers m and b such that
  // m * x + b is approximately equal to y.
  unsigned numSamples = 500;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  TC.learningRate = 0.1;
  TC.batchSize = numSamples;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction(
      "Gradient descent solution for simple linear regression");

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
  Placeholder *inputX =
      mod.createPlaceholder(ElemKind::FloatTy, {numSamples, 1}, "input", false);
  Placeholder *expectedY = mod.createPlaceholder(
      ElemKind::FloatTy, {numSamples, 1}, "expected", false);

  FullyConnectedNode *FC = F->createFullyConnected(bindings, "fc", inputX, 1);
  Node *R = F->createRegression("reg", FC, expectedY);
  SaveNode *SN = F->createSave("return", R);

  bindings.allocate(inputX);
  bindings.allocate(expectedY);
  bindings.allocate(SN->getPlaceholder());

  Placeholder *M = llvm::cast<Placeholder>(FC->getWeights());
  Placeholder *B = llvm::cast<Placeholder>(FC->getBias());

  Function *TF = glow::differentiate(F, TC);
  EE_.compile(CompilationMode::Train, TF);

  // Train the network doing 100 steps. Learn on 500 samples.
  runBatch(EE_, bindings, 100, sampleCounter, {inputX, expectedY},
           {&tensorX, &tensorY});

  // Testing trained m and b:
  EXPECT_NEAR(bindings.get(M)->getHandle<>().at({0, 0}), referenceM, 0.01);
  EXPECT_NEAR(bindings.get(B)->getHandle<>().at({0}), referenceB, 0.01);
}

enum class Sport : size_t { BASKETBALL = 0, SOCCER = 1 };

void generatePlayerData(Tensor &players, Tensor &labels,
                        unsigned numTrainPlayers, PseudoRNG &PRNG) {
  auto P = players.getHandle<>();
  auto L = labels.getHandle<int64_t>();

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

  TrainingConfig TC;
  PlaceholderBindings bindings;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  TC.learningRate = 0.05;
  TC.batchSize = numTrainPlayers;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("classifyPlayers");

  Placeholder *A = mod.createPlaceholder(
      ElemKind::FloatTy, {numTrainPlayers, numFeatures}, "A", false);
  Placeholder *S = mod.createPlaceholder(ElemKind::Int64ITy,
                                         {numTrainPlayers, 1}, "S", false);

  auto *FC = F->createFullyConnected(bindings, "fc", A, numClasses);
  auto *SM = F->createSoftMax("softmax", FC, S);
  SaveNode *result = F->createSave("result", SM);

  bindings.allocate(A);
  bindings.allocate(S);
  bindings.allocate(result->getPlaceholder());

  Function *TF = glow::differentiate(F, TC);
  EE_.compile(CompilationMode::Train, TF);

  Tensor players(ElemKind::FloatTy, {numTrainPlayers, numFeatures});
  Tensor labels(ElemKind::Int64ITy, {numTrainPlayers, 1});
  generatePlayerData(players, labels, numTrainPlayers, mod.getPRNG());

  // Training:
  runBatch(EE_, bindings, 2000, sampleCounter, {A, S}, {&players, &labels});

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

  updateInputPlaceholders(bindings, {A}, {&testPlayersTensor});
  EE_.run(bindings);

  auto SMH = bindings.get(result->getPlaceholder())->getHandle<>();
  for (size_t i = 0; i < testPlayers.size(); i++) {
    const size_t sport = static_cast<size_t>(std::get<2>(testPlayers[i]));
    EXPECT_NEAR(SMH.at({i, sport}), 1.0, 0.1);
  }
}

TEST_P(MLTest, learnSinus) {
  TrainingConfig TC;
  PlaceholderBindings bindings;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  // Try to learn the sin(x) function.
  float epsilon = 0.1;
  unsigned numSamples = 50;

  TC.learningRate = 0.2;
  TC.batchSize = numSamples;

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

  Placeholder *inputX =
      mod.createPlaceholder(ElemKind::FloatTy, {numSamples, 1}, "input", false);

  Placeholder *expectedY = mod.createPlaceholder(
      ElemKind::FloatTy, {numSamples, 1}, "expected", false);

  FullyConnectedNode *FC1 =
      F->createFullyConnected(bindings, "fc1", inputX, 10);
  Node *O = F->createSigmoid("sigmoid1", FC1);
  FullyConnectedNode *FC2 = F->createFullyConnected(bindings, "fc2", O, 1);
  Node *R = F->createRegression("reg", FC2, expectedY);
  SaveNode *result = F->createSave("return", R);

  bindings.allocate(inputX);
  bindings.allocate(expectedY);
  Tensor *res = bindings.allocate(result->getPlaceholder());

  Function *TF = glow::differentiate(F, TC);
  EE_.compile(CompilationMode::Train, TF);

  // Learn on numSamples samples.
  runBatch(EE_, bindings, 2700, sampleCounter, {inputX, expectedY},
           {&tensorX, &tensorY});

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
  updateInputPlaceholders(bindings, {inputX}, {&tensorX});
  EE_.run(bindings);
  auto resH = res->getHandle<>();

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

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  PlaceholderBindings bindings;
  TrainingConfig TC;
  TC.learningRate = 0.01;
  TC.momentum = 0.9;
  TC.batchSize = batchSize;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("nonLinearClassifier");

  Placeholder *A =
      mod.createPlaceholder(ElemKind::FloatTy, {batchSize, 2}, "A", false);
  Placeholder *S =
      mod.createPlaceholder(ElemKind::Int64ITy, {batchSize, 1}, "S", false);

  auto *FCL0 = F->createFullyConnected(bindings, "fc1", A, 8);
  auto *T0 = F->createTanh("tanh1", FCL0);
  auto *FCL1 = F->createFullyConnected(bindings, "fc2", T0, 8);
  auto *T1 = F->createTanh("tanh2", FCL1);
  auto *FCL2 = F->createFullyConnected(bindings, "fc2", T1, 2);
  auto *SM = F->createSoftMax("soft", FCL2, S);
  SaveNode *result = F->createSave("ret", SM);

  bindings.allocate(A);
  bindings.allocate(S);
  Tensor *res = bindings.allocate(result->getPlaceholder());

  Function *TF = glow::differentiate(F, TC);
  EE_.compile(CompilationMode::Train, TF);

  Tensor samples(ElemKind::FloatTy, {numSamples, 2});
  Tensor labels(ElemKind::Int64ITy, {numSamples, 1});

  for (size_t i = 0; i < numSamples; i++) {
    float x = mod.getPRNG().nextRand();
    float y = mod.getPRNG().nextRand();
    size_t label = (x < 0.0) ^ (y < 0.0);
    samples.getHandle<>().at({i, 0}) = x;
    samples.getHandle<>().at({i, 1}) = y;
    labels.getHandle<int64_t>().at({i, 0}) = label;
  }

  runBatch(EE_, bindings, 500, sampleCounter, {A, S}, {&samples, &labels});

  EE_.compile(CompilationMode::Infer, F);

  std::vector<std::tuple<float, float, size_t>> tests;
  tests.emplace_back(-0.8, -0.8, 0);
  tests.emplace_back(0.8, -0.8, 1);
  tests.emplace_back(-0.8, 0.8, 1);
  tests.emplace_back(0.8, 0.8, 0);
  auto RH = res->getHandle<>();
  for (size_t i = 0; i < tests.size(); i++) {
    Tensor T(ElemKind::FloatTy, {batchSize, 2});
    T.getHandle<>().at({0, 0}) = std::get<0>(tests[i]);
    T.getHandle<>().at({0, 1}) = std::get<1>(tests[i]);
    updateInputPlaceholders(bindings, {A}, {&T});
    EE_.run(bindings);
    EXPECT_NEAR(RH.at({0, std::get<2>(tests[i])}), 1.0, 0.2);
  }
}

/// Generate images in two classes.
/// A "line" is labeled as 0 and a "cross" is labeled as 1.
static void generateImageData(Tensor &images, Tensor &labels, PseudoRNG &PRNG) {
  auto L = labels.getHandle<int64_t>();
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
  ExecutionEngine EE{"Interpreter"};
  const unsigned numSamples = 500;
  const unsigned batchSize = 7;
  PlaceholderBindings bindings;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  TrainingConfig TC;
  TC.learningRate = 0.01;
  TC.batchSize = batchSize;
  TC.momentum = 0.9;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("convNetForImageRecognition");

  Placeholder *input = mod.createPlaceholder(
      ElemKind::FloatTy, {batchSize, 8, 8, 1}, "input", false);

  Placeholder *ex =
      mod.createPlaceholder(ElemKind::Int64ITy, {batchSize, 1}, "exp", false);

  auto *CV = F->createConv(bindings, "conv", input, 1, 3, 1, 0, 1);
  auto *TANH = F->createTanh("tanh", CV);
  auto *FCL = F->createFullyConnected(bindings, "fc", TANH, 2);
  auto *SM = F->createSoftMax("sm", FCL, ex);
  SaveNode *result = F->createSave("ret", SM);

  bindings.allocate(input);
  bindings.allocate(ex);
  Tensor *res = bindings.allocate(result->getPlaceholder());

  Function *TF = glow::differentiate(F, TC);
  EE.compile(CompilationMode::Train, TF);

  Tensor images(ElemKind::FloatTy, {numSamples, 8, 8, 1});
  Tensor labels(ElemKind::Int64ITy, {numSamples, 1});
  generateImageData(images, labels, mod.getPRNG());

  // Training:
  runBatch(EE, bindings, 500, sampleCounter, {input, ex}, {&images, &labels});

  LoweredInfoMap loweredMapForProf;
  CompilationContext cctxProf{&bindings, &loweredMapForProf};
  cctxProf.precisionConfig.quantMode = QuantizationMode::Profile;

  Function *PF = F->clone("profile");
  EE.compile(PF, cctxProf);
  runBatch(EE, bindings, 100, sampleCounter, {input}, {&images});

  // Evaluate on the quantized function:
  // Set the execution backend to the backend that we test.
  EE.setBackend(GetParam());

  LoweredInfoMap loweredMapForQuant;
  CompilationContext cctxQuant{&bindings, &loweredMapForQuant};
  PrecisionConfiguration &precConfig = cctxQuant.precisionConfig;
  cctxQuant.precisionConfig.quantMode = QuantizationMode::Quantize;
  precConfig.quantConfig.infos = quantization::generateNodeQuantizationInfos(
      bindings, PF, loweredMapForProf);
  precConfig.quantConfig.assertAllNodesQuantized = true;

  // Softmax is not supported in Int8QTy, so signal the quantizer it's OK to
  // keep it unquantized.
  precConfig.precisionModeKindSet.insert(Kinded::Kind::SoftMaxNodeKind);

  EE.compile(F, cctxQuant);

  // Generate the images used for testing.
  Tensor testImages(ElemKind::FloatTy, {batchSize, 8, 8, 1});
  Tensor testLabels(ElemKind::Int64ITy, {batchSize, 1});
  generateImageData(testImages, testLabels, mod.getPRNG());
  updateInputPlaceholders(bindings, {input}, {&testImages});
  EE.run(bindings);
  auto SMH = res->getHandle<>();
  for (size_t i = 0; i < batchSize; i++) {
    bool isLine = testLabels.getHandle<int64_t>().at({i, 0}) == 0;
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
  ExecutionEngine EE{"Interpreter"};
  PlaceholderBindings bindings;

  const unsigned numSamples = 1000;
  const unsigned batchSize = 10;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  TrainingConfig TC;
  TC.learningRate = 0.01;
  TC.batchSize = batchSize;
  TC.momentum = 0.9;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  Placeholder *input = mod.createPlaceholder(
      ElemKind::FloatTy, {batchSize, 10, 10, 1}, "input", false);
  Placeholder *ex = mod.createPlaceholder(ElemKind::FloatTy, {batchSize, 2},
                                          "coordinates", false);

  // A simple single-layer FC network could solve this program but we use a two
  // layer FC network to give the compiler something slightly more complex to
  // work with so we are adding another FC layer.
  auto *FC0 = F->createFullyConnected(bindings, "fc0", input, 6);
  auto *RL0 = F->createRELU("relu0", FC0);
  auto *FC1 = F->createFullyConnected(bindings, "fc1", RL0, 2);
  auto *R = F->createRegression("regression", FC1, ex);
  SaveNode *result = F->createSave("ret", R);

  bindings.allocate(input);
  bindings.allocate(ex);
  Tensor *res = bindings.allocate(result->getPlaceholder());

  Function *TF = glow::differentiate(F, TC);
  EE.compile(CompilationMode::Train, TF);

  // --  STEP1 - train the network. --
  Tensor images(ElemKind::FloatTy, {numSamples, 10, 10, 1});
  Tensor labels(ElemKind::FloatTy, {numSamples, 2});
  generateRegressionTestData(images, labels, mod.getPRNG());

  // Training:
  runBatch(EE, bindings, 400, sampleCounter, {input, ex}, {&images, &labels});

  // -- STEP2 - Profile and quantize the network. --

  LoweredInfoMap loweredMapForProf;
  CompilationContext cctxProf{&bindings, &loweredMapForProf};
  cctxProf.precisionConfig.quantMode = QuantizationMode::Profile;

  Function *PF = F->clone("profile");
  EE.compile(PF, cctxProf);

  // Run the graph to capture the profile.
  runBatch(EE, bindings, 100, sampleCounter, {input}, {&images});

  // -- STEP3 - evaluate the quantized function. --

  // Set the execution backend to the backend that we test.
  EE.setBackend(GetParam());

  LoweredInfoMap loweredMapForQuant;
  CompilationContext cctxQuant{&bindings, &loweredMapForQuant};
  cctxQuant.precisionConfig.quantMode = QuantizationMode::Quantize;
  cctxQuant.loweredInfoMap = &loweredMapForQuant;
  cctxQuant.bindings = &bindings;
  cctxQuant.precisionConfig.quantConfig.infos =
      quantization::generateNodeQuantizationInfos(bindings, PF,
                                                  loweredMapForProf);
  cctxQuant.precisionConfig.quantConfig.assertAllNodesQuantized = true;

  EE.compile(F, cctxQuant);

  // Generate the images used for testing.
  Tensor testImages(ElemKind::FloatTy, {batchSize, 10, 10, 1});
  Tensor testLabels(ElemKind::FloatTy, {batchSize, 2});
  generateRegressionTestData(testImages, testLabels, mod.getPRNG());

  // Run the inference:
  updateInputPlaceholders(bindings, {input}, {&testImages});
  EE.run(bindings);

  // A handle to the projected result.
  auto RH = res->getHandle<>();
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
  auto handleExpected = expected.getHandle<int64_t>();

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
  TrainingConfig TC;
  TC.learningRate = 0.15;
  TC.batchSize = 17;
  PlaceholderBindings bindings;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  Module &mod = EE_.getModule();
  PseudoRNG &PRNG = mod.getPRNG();
  Function *F = mod.createFunction("MatrixRotationRecognition");
  Placeholder *varMatricesA = mod.createPlaceholder(
      ElemKind::FloatTy, {TC.batchSize, 3, 3}, "matrixA", false);
  Placeholder *varMatricesB = mod.createPlaceholder(
      ElemKind::FloatTy, {TC.batchSize, 3, 3}, "matrixB", false);
  Placeholder *varExpected = mod.createPlaceholder(
      ElemKind::Int64ITy, {TC.batchSize, 1}, "expected", false);

  // Simply concatenating the matrices first would probability be as effective
  // but we want to build something more complex than a straight chain of
  // operators to stress the scheduler.
  auto *FCA =
      F->createFullyConnected(bindings, "hidden_matrixA_fc", varMatricesA, 10);
  auto *FCB =
      F->createFullyConnected(bindings, "hidden_matrixB_fc", varMatricesB, 10);
  auto *ReLUA = F->createRELU("hidden_matrixA_ReLU", FCA);
  auto *ReLUB = F->createRELU("hidden_matrixB_ReLU", FCB);
  auto *concat = F->createConcat("hidden_concat_A_B", {ReLUA, ReLUB}, 1);
  auto *hiddenFC = F->createFullyConnected(bindings, "hidden_fc", concat, 30);
  auto *finalReLU = F->createRELU("hidden_concat_ReLU", hiddenFC);
  auto *finalFC = F->createFullyConnected(bindings, "output_fc", finalReLU, 2);
  auto *softMax = F->createSoftMax("output", finalFC, varExpected);
  SaveNode *result = F->createSave("result", softMax);

  bindings.allocate(varMatricesA);
  bindings.allocate(varMatricesB);
  bindings.allocate(varExpected);
  Tensor *res = bindings.allocate(result->getPlaceholder());

  Function *trainingGradientFunction = glow::differentiate(F, TC);

  // Train the network.
  const unsigned numSamples = 50;
  Tensor matricesA(ElemKind::FloatTy, {numSamples, 3, 3});
  Tensor matricesB(ElemKind::FloatTy, {numSamples, 3, 3});
  Tensor expected(ElemKind::Int64ITy, {numSamples, 1});
  generateMatrixRotationRecognitionData(matricesA, matricesB, expected, PRNG);

  EE_.compile(CompilationMode::Train, trainingGradientFunction);
  // Training:
  runBatch(EE_, bindings, 200, sampleCounter,
           {varMatricesA, varMatricesB, varExpected},
           {&matricesA, &matricesB, &expected});

  // Switch to inference mode.
  EE_.compile(CompilationMode::Infer, F);

  // At this point we should have overfitted the data.
  // Take a random batch and check that the values match what we expect.
  auto RHtrain = res->getHandle<>();
  auto batchSize = TC.batchSize;
  unsigned numBatches = numSamples / batchSize;
  unsigned batchStartIdx = PRNG.nextRandInt(0, numBatches - 1) * batchSize;

  auto batchMatricesA =
      matricesA.getUnowned({batchSize, 3, 3}, {batchStartIdx, 0, 0});
  auto batchMatricesB =
      matricesB.getUnowned({batchSize, 3, 3}, {batchStartIdx, 0, 0});
  updateInputPlaceholders(bindings, {varMatricesA, varMatricesB},
                          {&batchMatricesA, &batchMatricesB});
  EE_.run(bindings);

  unsigned errors = 0;
  // Check each output in the batch.
  for (size_t i = 0; i != batchSize; i++) {
    // Note that the two softmax outputs always sum to 1, so we only look at
    // one. Index one is true if there is a rotation.
    float value = RHtrain.at({i, 1});
    bool hasRotation = expected.getHandle<int64_t>().at({batchStartIdx + i, 0});
    if ((value > 0.5) != hasRotation) {
      ++errors;
    }
  }

  EXPECT_LE(errors, 1);
}

/// Simple test case that learns the embedding table for a
/// SparseLengthsWeightedSum operator.
TEST_P(MLTest, learnSparseLengthsWeightedSumEmbeddings) {
  TrainingConfig TC;
  TC.learningRate = 0.3;
  TC.batchSize = 1;

  PlaceholderBindings bindings;

  Module &mod = EE_.getModule();
  PseudoRNG &PRNG = mod.getPRNG();

  // Create a model consisting of one SparseLengthsWeightedSum operator followed
  // by a Regression node to get some non-zero gradients.
  Function *F = mod.createFunction("SparseLengthsWeightedSum");
  Placeholder *dataP = mod.createPlaceholder(ElemKind::FloatTy, {10}, "dataP",
                                             /*isTrainable=*/true);
  Placeholder *indicesP = mod.createPlaceholder(
      ElemKind::Int64ITy, {10}, "indicesP", /*isTrainable=*/false);
  Placeholder *weightsP = mod.createPlaceholder(
      ElemKind::FloatTy, {10}, "weightsP", /*isTrainable=*/false);
  Placeholder *lengthsP = mod.createPlaceholder(
      ElemKind::Int32ITy, {5}, "lengthsP", /*isTrainable=*/false);
  Placeholder *expectedP = mod.createPlaceholder(
      ElemKind::FloatTy, {5}, "expectedP", /*isTrainable=*/false);

  auto *SLWS = F->createSparseLengthsWeightedSum("SLWS", dataP, weightsP,
                                                 indicesP, lengthsP);
  auto *reg = F->createRegression("reg", SLWS, expectedP);
  auto *result = F->createSave("save", reg);

  // Allocate and randomly initialize embeddings.
  auto DH = bindings.allocate(dataP)->getHandle();
  DH.randomize(-5.0, 5.0, PRNG);

  // Allocate and set indices such that input embeddings are reversed.
  bindings.allocate(indicesP)->getHandle<int64_t>() = {9, 8, 7, 6, 5,
                                                       4, 3, 2, 1, 0};
  // Allocate and set weights.
  bindings.allocate(weightsP)->getHandle() = {0.75, 0.25, 0.75, 0.25, 0.75,
                                              0.25, 0.75, 0.25, 0.75, 0.25};

  // Allocate and set lengths.
  bindings.allocate(lengthsP)->getHandle<int32_t>() = {2, 2, 2, 2, 2};

  // Allocate and set expected outputs. The embedding table will be adjusted
  // during training so that the final result is this.
  auto EH = bindings.allocate(expectedP)->getHandle();
  EH = {1, 2, 3, 4, 5};

  // Allocate and store a handle to the result for testing later.
  auto RH = bindings.allocate(result->getPlaceholder())->getHandle();

  // Train the network.
  Function *trainingGradientFunction = glow::differentiate(F, TC);
  EE_.compile(CompilationMode::Train, trainingGradientFunction);

  const size_t numIterations = 1000;

  for (size_t i = 0; i < numIterations; ++i) {
    EE_.run(bindings);
  }

  // Switch to inference mode and run the network.
  EE_.compile(CompilationMode::Infer, F);
  EE_.run(bindings);

  // Make sure that the network output matches expectations after training.
  for (size_t j = 0; j < EH.size(); ++j) {
    EXPECT_NEAR(RH.raw(j), EH.raw(j), 0.02);
  }
}

/// Simple test case that learns the weights for a
/// SparseLengthsWeightedSum operator.
TEST_P(MLTest, learnSparseLengthsWeightedSumWeights) {
  TrainingConfig TC;
  TC.learningRate = 0.001;
  TC.batchSize = 1;

  PlaceholderBindings bindings;

  Module &mod = EE_.getModule();
  PseudoRNG &PRNG = mod.getPRNG();

  // Create a model consisting of one SparseLengthsWeightedSum operator followed
  // by a Regression node to get some non-zero gradients.
  Function *F = mod.createFunction("SparseLengthsWeightedSum");
  Placeholder *dataP = mod.createPlaceholder(ElemKind::FloatTy, {10}, "dataP",
                                             /*isTrainable=*/false);
  Placeholder *indicesP = mod.createPlaceholder(
      ElemKind::Int64ITy, {10}, "indicesP", /*isTrainable=*/false);
  Placeholder *weightsP = mod.createPlaceholder(
      ElemKind::FloatTy, {10}, "weightsP", /*isTrainable=*/true);
  Placeholder *lengthsP = mod.createPlaceholder(
      ElemKind::Int32ITy, {5}, "lengthsP", /*isTrainable=*/false);
  Placeholder *expectedP = mod.createPlaceholder(
      ElemKind::FloatTy, {5}, "expectedP", /*isTrainable=*/false);

  auto *SLWS = F->createSparseLengthsWeightedSum("SLWS", dataP, weightsP,
                                                 indicesP, lengthsP);
  auto *reg = F->createRegression("reg", SLWS, expectedP);
  auto *result = F->createSave("save", reg);

  // Allocate and set embeddings.
  bindings.allocate(dataP)->getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  // Allocate and set indices such that input embeddings are reversed.
  bindings.allocate(indicesP)->getHandle<int64_t>() = {9, 8, 7, 6, 5,
                                                       4, 3, 2, 1, 0};
  // Allocate and randomly initialize weights.
  auto WH = bindings.allocate(weightsP)->getHandle();
  WH.randomize(-1.0, 1.0, PRNG);

  // Allocate and set lengths.
  bindings.allocate(lengthsP)->getHandle<int32_t>() = {2, 2, 2, 2, 2};

  // Allocate and set expected outputs. The weighs will be adjusted
  // during training so that the final result is this.
  auto EH = bindings.allocate(expectedP)->getHandle();
  EH = {10, 7, 6, 3, 2};

  // Allocate and store a handle to the result for testing later.
  auto RH = bindings.allocate(result->getPlaceholder())->getHandle();

  // Train the network.
  Function *trainingGradientFunction = glow::differentiate(F, TC);
  EE_.compile(CompilationMode::Train, trainingGradientFunction);

  const size_t numIterations = 1000;

  for (size_t i = 0; i < numIterations; ++i) {
    EE_.run(bindings);
  }

  // Switch to inference mode and run the network.
  EE_.compile(CompilationMode::Infer, F);
  EE_.run(bindings);

  // Make sure that the network output matches expectations after training.
  for (size_t j = 0; j < EH.size(); ++j) {
    EXPECT_NEAR(RH.raw(j), EH.raw(j), 0.02);
  }
}

INSTANTIATE_TEST_CASE_P(Interpreter, MLTest, ::testing::Values("Interpreter"));
#ifdef GLOW_WITH_CPU
INSTANTIATE_TEST_CASE_P(JIT, MLTest, ::testing::Values("CPU"));
#endif // GLOW_WITH_CPU

#ifdef GLOW_WITH_OPENCL
INSTANTIATE_TEST_CASE_P(OpenCL, MLTest, ::testing::Values("OpenCL"));
#endif // GLOW_WITH_OPENCL

INSTANTIATE_TEST_CASE_P(Interpreter, InterpreterAndCPU,
                        ::testing::Values("Interpreter"));
#ifdef GLOW_WITH_CPU
INSTANTIATE_TEST_CASE_P(JIT, InterpreterAndCPU, ::testing::Values("CPU"));
#endif // GLOW_WITH_CPU
