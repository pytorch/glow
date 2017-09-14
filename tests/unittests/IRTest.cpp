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
  Type T1(ElemKind::FloatTy, {320, 200});

  Value V1;
  Value V2;

  // Check that we can construct a new instruction.
  Instruction I({{&V1, OperandKind::kIn}, {&V2, OperandKind::kIn}});
  I.verifyUseList();

  // Check the getOperand and setOperand functions.
  EXPECT_EQ(I.getOperand(0).first, &V1);
  I.setOperand(0, &V2);
  EXPECT_EQ(I.getOperand(0).first, &V2);
  I.verifyUseList();

  // Check that we can destroy the operands.
  // ...
}

TEST(IR, basisInstrs) {
  using InitKind = StaticVariable::InitKind;

  Module M;
  auto T1 = M.uniqueType(ElemKind::FloatTy, {1, 24, 24, 3});
  auto T2 = M.uniqueType(ElemKind::FloatTy, {64});
  auto T3 = M.uniqueType(ElemKind::FloatTy, {1, 320, 200, 3});

  IRBuilder builder(M);

  auto *I0 = builder.createStaticVariable(T1, InitKind::kBroadcast, 1.1);
  auto *I1 = builder.createStaticVariable(T1, InitKind::kBroadcast, 1.1);
  auto *I2 = builder.createStaticVariable(T1, InitKind::kBroadcast, 1.1);

  auto *B0 = builder.createStaticVariable(T2, InitKind::kExtern, 0);
  auto *F0 = builder.createStaticVariable(T3, InitKind::kExtern, 0);

  builder.createTransposeInst(I1, I0, {0, 2, 3, 1});

  builder.createConvolutionInst(I2, I1, F0, B0, 7, 2, 3, 64);
  builder.createReluInst(I2, I2);
  builder.createCopyInst(I0, I2);
  M.dump();
}
