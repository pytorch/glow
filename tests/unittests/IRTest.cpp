#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

using namespace glow;

TEST(IR, uniqueTypes) {
  Module M;
  Type T1(ElemKind::FloatTy, {320, 200});
  Type T2(ElemKind::FloatTy, {320, 200});
  Type T3(ElemKind::FloatTy, {1, 2});

  auto *u1 = M.uniqueType(T1);
  auto *u2 = M.uniqueType(T2);
  auto *u3 = M.uniqueType(T3);

  EXPECT_EQ(u1, u2);
  EXPECT_NE(u1, u3);

  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(u1, M.uniqueType(T1));
  }
}

TEST(IR, basicUseList) {
  Module M;
  IRBuilder builder(M);

  auto *V1 = builder.createStaticVariable(ElemKind::FloatTy, {320, 200});
  auto *V2 = builder.createStaticVariable(ElemKind::FloatTy, {320, 200});

  // Check that we can construct a new instruction.
  auto *CC = builder.createCopyInst(V1, V2);
  CC->verifyUseList();

  // Check the getOperand and setOperand functions.
  EXPECT_EQ(CC->getOperand(0).first, V1);
  CC->setOperand(0, V2);
  EXPECT_EQ(CC->getOperand(0).first, V2);
  CC->verifyUseList();

  // Check that we can destroy the operands.
  // ...
}

TEST(IR, allInstrs) {
  using InitKind = StaticVariable::InitKind;

  Module M;
  auto T1 = M.uniqueType(ElemKind::FloatTy, {1, 24, 24, 3});
  auto T2 = M.uniqueType(ElemKind::FloatTy, {64});
  auto T4 = M.uniqueType(ElemKind::IndexTy, {1, 1});
  auto T5 = M.uniqueType(ElemKind::FloatTy, {3});

  IRBuilder builder(M);

  auto *I0 = builder.createStaticVariable(T1, InitKind::kExtern, 0);
  auto *I1 = builder.createStaticVariable(T1, InitKind::kExtern, 0);

  auto *I3 = builder.createStaticVariable(ElemKind::FloatTy, {1, 12, 12, 64});
  auto *I4 = builder.createStaticVariable(ElemKind::FloatTy, {1, 12, 12, 3});

  auto *XY = builder.createStaticVariable(ElemKind::IndexTy, {1, 12, 12, 3, 2});
  auto *B0 = builder.createStaticVariable(T2, InitKind::kBroadcast, 0.1);
  auto *F0 = builder.createStaticVariable(ElemKind::FloatTy, {64, 7, 7, 3});
  auto *E0 = builder.createStaticVariable(T4, InitKind::kExtern, 0);
  auto *S0 = builder.createStaticVariable(T5, InitKind::kExtern, 0);

  B0->setName("bias");
  F0->setName("filter");
  E0->setName("expected");
  XY->setName("srcXY");

  builder.createCopyInst(I1, I0);
  builder.createConvolutionInst(I3, I1, F0, B0, 7, 2, 3, 64);
  builder.createPoolInst(I4, I0, XY, PoolInst::OpKind::kMax, 7, 2, 3);
  builder.createFullyConnectedInst(I1, I0, F0, B0, 32);
  builder.createReluInst(I1, I0);
  builder.createSigmoidInst(I1, I0);
  builder.createTanhInst(I1, I0);
  builder.createSoftMaxInst(I1, I0, E0);
  builder.createRegressionInst(I1, I0, E0);
  builder.createTransposeInst(I1, I0, {0, 2, 3, 1});
  builder.createConcatInst(I1, {I0, I0}, 2);
  builder.createBatchNormalizationInst(I1, I0, S0, S0, S0, S0, 3, 0.01, 0.9);
  builder.createArithmeticInst(I1, I0, I0, ArithmeticInst::OpKind::kMul);
  M.verify();
  M.dump();
}

TEST(IR, highLevelBuilder) {
  Module M;
  IRBuilder bb(M);

  auto *input = bb.createStaticVariable(ElemKind::FloatTy, {1, 224, 224, 3});
  auto *conv = bb.createConvOp(input, 16, 7, 2, 3);
  auto *pool = bb.createMaxPoolOp(conv->getOperand(0).first,
                                  PoolInst::OpKind::kMax, 7, 2, 3);
  (void)pool;
  M.verify();
  M.dump();
}
