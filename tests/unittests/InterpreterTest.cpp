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

  auto &builder = IP.getBuilder();
  auto *input = builder.createStaticVariable(ElemKind::FloatTy, {1, 32, 32, 3});
  Tensor inputs(ElemKind::FloatTy, {1, 32, 32, 3});

  auto *ex = builder.createStaticVariable(ElemKind::IndexTy, {1, 1});

  auto *CV0 = builder.createConvOp(input, 16, 5, 1, 2);
  auto *RL0 = builder.createRELUOp(*CV0);
  auto *MP0 = builder.createPoolOp(*RL0, PoolInst::OpKind::kMax, 2, 2, 0);

  auto *CV1 = builder.createConvOp(*MP0, 20, 5, 1, 2);
  auto *RL1 = builder.createRELUOp(*CV1);
  auto *MP1 = builder.createPoolOp(*RL1, PoolInst::OpKind::kMax, 2, 2, 0);

  auto *CV2 = builder.createConvOp(*MP1, 20, 5, 1, 2);
  auto *RL2 = builder.createRELUOp(*CV2);
  auto *MP2 = builder.createPoolOp(*RL2, PoolInst::OpKind::kMax, 2, 2, 0);

  auto *FCL1 = builder.createFullyConnectedOp(*MP2, 10);
  auto *RL3 = builder.createRELUOp(*FCL1);
  auto *SM = builder.createSoftMaxOp(*RL3, ex);
  (void)SM;

  IP.getModule().dump();
  IP.getModule().verify();

  IP.initVars();
  IP.infer({input}, {&inputs});
}

TEST(Interpreter, trainASimpleNetwork) {
  Interpreter IP;
  auto &builder = IP.getBuilder();

  // Learning a single input vector.
  IP.getConfig().maxNumThreads = 1;
  IP.getConfig().learningRate = 0.05;

  // Create a variable with 1 input, which is a vector of 4 elements.
  auto *A = builder.createStaticVariable(ElemKind::FloatTy, {1, 4});
  auto *E = builder.createStaticVariable(ElemKind::FloatTy, {1, 4});

  Instruction *O = builder.createFullyConnectedOp(A, 10);
  O = builder.createSigmoidOp(*O);
  O = builder.createFullyConnectedOp(*O, 4);
  O = builder.createSigmoidOp(*O);
  auto *RN = builder.createRegressionOp(*O, E);

  // Values for the input and output variables.
  Tensor inputs(ElemKind::FloatTy, {1, 4});
  Tensor expected(ElemKind::FloatTy, {1, 4});
  inputs.getHandle<FloatTy>() = {0.15, 0.15, 0.15, 0.15};
  expected.getHandle<FloatTy>() = {0.9, 0.9, 0.9, 0.9};

  IP.initVars();

  // Train the network. Learn 1000 batches.
  IP.train(1000, {A, E}, {&inputs, &expected});

  // Testing the output vector.

  IP.infer({A}, {&inputs});
  auto RNWH = IP.getTensorForValue(RN->getDest())->getHandle<FloatTy>();
  (void)RNWH;

  // Test the output:
  EXPECT_NEAR(RNWH.at({0, 0}), 0.9, 0.05);
}
