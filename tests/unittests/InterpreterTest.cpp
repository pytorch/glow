#include "glow/Interpreter/Interpreter.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"

#include "gtest/gtest.h"

#include <cassert>
#include <string>

using namespace glow;

TEST(Interpreter, interpret) {
  Interpreter IP;

  Value *input;
  Tensor inputs(ElemKind::FloatTy, {1, 32, 32, 3});

  {
    IRBuilder builder(IP.getModule());
    input = builder.createWeightVar(ElemKind::FloatTy, {1, 32, 32, 3});

    auto *ex = builder.createWeightVar(ElemKind::IndexTy, {1, 1});

    auto *CV0 = builder.createConvOp(input, 16, 5, 1, 2);
    auto *RL0 = builder.createRELUOp(*CV0);
    auto *MP0 = builder.createPoolOp(*RL0, PoolInst::OpKind::Max, 2, 2, 0);

    auto *CV1 = builder.createConvOp(*MP0, 20, 5, 1, 2);
    auto *RL1 = builder.createRELUOp(*CV1);
    auto *MP1 = builder.createPoolOp(*RL1, PoolInst::OpKind::Max, 2, 2, 0);

    auto *CV2 = builder.createConvOp(*MP1, 20, 5, 1, 2);
    auto *RL2 = builder.createRELUOp(*CV2);
    auto *MP2 = builder.createPoolOp(*RL2, PoolInst::OpKind::Max, 2, 2, 0);

    auto *FCL1 = builder.createFullyConnectedOp(*MP2, 10);
    auto *RL3 = builder.createRELUOp(*FCL1);
    auto *SM = builder.createSoftMaxOp(*RL3, ex);
    builder.createReturnOp(*SM);
  }

  IP.optimize(OptimizationMode::Infer);
  IP.initVars();
  IP.infer({input}, {&inputs});
}

TEST(Interpreter, trainASimpleNetwork) {
  Interpreter IP;
  // Learning a single input vector.
  IP.getConfig().maxNumThreads = 1;
  IP.getConfig().learningRate = 0.05;

  Value *A;
  Value *E;
  Value *result;
  {
    IRBuilder builder(IP.getModule());

    // Create a variable with 1 input, which is a vector of 4 elements.
    A = builder.createWeightVar(ElemKind::FloatTy, {1, 4});
    E = builder.createWeightVar(ElemKind::FloatTy, {1, 4});

    Instruction *O = builder.createFullyConnectedOp(A, 10);
    O = builder.createSigmoidOp(*O);
    O = builder.createFullyConnectedOp(*O, 4);
    O = builder.createSigmoidOp(*O);
    O = builder.createRegressionOp(*O, E);
    result = builder.createReturnOp(*O);
  }

  // Values for the input and output variables.
  Tensor inputs(ElemKind::FloatTy, {1, 4});
  Tensor expected(ElemKind::FloatTy, {1, 4});
  inputs.getHandle<FloatTy>() = {0.15, 0.15, 0.15, 0.15};
  expected.getHandle<FloatTy>() = {0.9, 0.9, 0.9, 0.9};

  IP.optimize(OptimizationMode::Train);
  IP.initVars();

  // Train the network. Learn 1000 batches.
  IP.train(1000, {A, E}, {&inputs, &expected});

  // Testing the output vector.

  IP.optimize(OptimizationMode::Infer);
  IP.infer({A}, {&inputs});
  auto RNWH = IP.getTensorForValue(result)->getHandle<FloatTy>();
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
  Interpreter IP;

  // Learning a single input vector.
  IP.getConfig().maxNumThreads = 1;
  IP.getConfig().learningRate = 0.05;

  Tensor inputs(ElemKind::FloatTy, {1, numInputs});
  Tensor expected(ElemKind::FloatTy, {1, numInputs});

  Value *A;
  Value *Ex;
  Value *result;
  {
    IRBuilder bb(IP.getModule());
    A = bb.createWeightVar(ElemKind::FloatTy, {1, numInputs});
    Ex = bb.createWeightVar(ElemKind::FloatTy, {1, numInputs});
    Instruction *O = bb.createFullyConnectedOp(A, 4);
    O = bb.createRELUOp(*O);
    O = bb.createRegressionOp(*O, Ex);
    result = bb.createReturnOp(*O);
  }
  auto I = inputs.getHandle<FloatTy>();
  auto E = expected.getHandle<FloatTy>();

  IP.optimize(OptimizationMode::Train);
  IP.initVars();

  // Train the network:
  for (int iter = 0; iter < 1000; iter++) {
    float target = float(iter % 9);
    I = {target, 0., 0., 0.};
    E = {0., target + 1, 0., 0.};
    IP.train(1, {A, Ex}, {&inputs, &expected});
  }

  // Verify the result of the regression layer.

  // Test the output:
  for (int iter = 0; iter < 5; iter++) {
    float target = iter % 9 + 1;
    I = {target, 0., 0., 0.};
    IP.infer({A}, {&inputs});

    auto resH = IP.getTensorForValue(result)->getHandle<FloatTy>();
    (void)resH;

    EXPECT_NEAR(I.at({0, 0}) + 1, resH.at({0, 1}), 0.1);
  }
}

TEST(Interpreter, learnXor) {
  unsigned numInputs = 10;
  unsigned numTests = 10;

  // Learning the Xor function.
  Interpreter IP;

  // Learning a single input vector.
  IP.getConfig().maxNumThreads = 1;
  IP.getConfig().learningRate = 0.05;

  Value *result;
  Value *A;
  Value *Ex;
  {
    IRBuilder bb(IP.getModule());

    A = bb.createWeightVar(ElemKind::FloatTy, {numInputs, 2});
    Ex = bb.createWeightVar(ElemKind::FloatTy, {numInputs, 1});

    Instruction *O = bb.createFullyConnectedOp(A, 6);
    O = bb.createRELUOp(*O);
    O = bb.createFullyConnectedOp(*O, 1);
    O = bb.createRELUOp(*O);
    O = bb.createRegressionOp(*O, Ex);
    result = bb.createReturnOp(*O);
  }

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

  IP.optimize(OptimizationMode::Train);
  IP.initVars();

  // Train the network:
  IP.train(2500, {A, Ex}, {&trainingSet, &trainingLabels});

  // Prepare the testing tensor:
  for (unsigned i = 0; i < numTests; i++) {
    TT.at({i, 0}) = i % 2;
    TT.at({i, 1}) = (i >> 1) % 2;
  }

  IP.infer({A}, {&trainingSet});
  auto resH = IP.getTensorForValue(result)->getHandle<FloatTy>();

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
  Interpreter IP;

  // Learning a single input vector.
  IP.getConfig().maxNumThreads = 1;
  IP.getConfig().momentum = 0.9;
  IP.getConfig().learningRate = 0.01;

  Value *A;
  Value *S;
  Value *result;
  {
    IRBuilder bb(IP.getModule());

    A = bb.createWeightVar(ElemKind::FloatTy, {1, 2});
    S = bb.createWeightVar(ElemKind::IndexTy, {1, 1});

    auto *FCL0 = bb.createFullyConnectedOp(A, 8);
    auto *RL0 = bb.createRELUOp(*FCL0);
    auto *FCL1 = bb.createFullyConnectedOp(*RL0, 2);
    auto *RL1 = bb.createRELUOp(*FCL1);
    auto *SM = bb.createSoftMaxOp(*RL1, S);
    result = bb.createReturnOp(*SM);
  }

  IP.optimize(OptimizationMode::Train);
  IP.initVars();

  Tensor coordinates(ElemKind::FloatTy, {numSamples, 2});
  Tensor labels(ElemKind::IndexTy, {numSamples, 1});
  generateCircleData(coordinates, labels);

  // Training:
  IP.train(4000, {A, S}, {&coordinates, &labels});

  // Print a diagram that depicts the network decision on a grid.
  for (int x = -10; x < 10; x++) {
    for (int y = -10; y < 10; y++) {
      // Load the inputs:
      Tensor sample(ElemKind::FloatTy, {1, 2});
      sample.getHandle<FloatTy>() = {float(x) / 10, float(y) / 10};

      IP.infer({A}, {&sample});

      auto SMH = IP.getTensorForValue(result)->getHandle<FloatTy>();
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
    IP.infer({A}, {&sample});
    auto SMH = IP.getTensorForValue(result)->getHandle<FloatTy>();
    auto A = SMH.at({0, 0});
    auto B = SMH.at({0, 1});
    EXPECT_LE(A, 0.1);
    EXPECT_GE(B, 0.9);
  }

  {
    // Far away dot must be one.
    Tensor sample(ElemKind::FloatTy, {1, 2});
    sample.getHandle<FloatTy>() = {1., 1.};
    IP.infer({A}, {&sample});
    auto SMH = IP.getTensorForValue(result)->getHandle<FloatTy>();
    auto A = SMH.at({0, 0});
    auto B = SMH.at({0, 1});
    EXPECT_GE(A, 0.9);
    EXPECT_LE(B, 0.1);
  }
}

TEST(Network, learnSingleValueConcat) {
  Interpreter IP;
  unsigned width = 6;

  // Learning a single input vector.
  IP.getConfig().maxNumThreads = 1;
  IP.getConfig().momentum = 0.9;
  IP.getConfig().learningRate = 0.01;

  Value *Ex;
  Value *A;
  Value *B;
  Value *result;
  {
    IRBuilder bb(IP.getModule());

    Ex = bb.createWeightVar(ElemKind::FloatTy, {1, width * 2});

    // Left side of the network:
    A = bb.createWeightVar(ElemKind::FloatTy, {1, width});
    Instruction *L = bb.createFullyConnectedOp(A, width);
    L = bb.createSigmoidOp(*L);

    // Right side of the network:
    B = bb.createWeightVar(ElemKind::FloatTy, {1, width});
    Instruction *R = bb.createFullyConnectedOp(B, width);
    R = bb.createSigmoidOp(*R);

    // Concat:
    auto *C = bb.createConcatOp({*L, *R}, 1);
    auto *RN = bb.createRegressionOp(*C, Ex);
    result = bb.createReturnOp(*RN);
  }

  Tensor inputs(ElemKind::FloatTy, {1, width});
  Tensor expected(ElemKind::FloatTy, {1, width * 2});
  inputs.getHandle<FloatTy>().clear(0.15);
  expected.getHandle<FloatTy>().clear(0.9);

  IP.optimize(OptimizationMode::Train);
  IP.initVars();

  // Train the network:
  IP.train(1000, {A, B, Ex}, {&inputs, &inputs, &expected});

  IP.optimize(OptimizationMode::Infer);

  // Testing the output vector.
  IP.infer({A}, {&inputs});
  auto RNWH = IP.getTensorForValue(result)->getHandle<FloatTy>();
  (void)RNWH;

  // Test the output:
  EXPECT_NEAR(RNWH.at({0, 0}), 0.9, 0.1);
}
