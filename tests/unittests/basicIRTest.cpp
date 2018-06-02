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

#include "glow/Graph/Graph.h"

#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"

#include "llvm/Support/Casting.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

using namespace glow;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

TEST(IR, uniqueTypes) {
  Module mod;
  Type T1(ElemKind::FloatTy, {320, 200});
  Type T2(ElemKind::FloatTy, {320, 200});
  Type T3(ElemKind::FloatTy, {1, 2});

  auto *u1 = mod.uniqueType(T1);
  auto *u2 = mod.uniqueType(T2);
  auto *u3 = mod.uniqueType(T3);

  EXPECT_EQ(u1, u2);
  EXPECT_NE(u1, u3);

  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(u1, mod.uniqueType(T1));
  }

  // Check the uniqueing of quantized tensors.
  Type T4(ElemKind::Int8QTy, {1, 2}, 0.4, 2);
  auto *t4 = mod.uniqueType(T4);
  auto *u4 = mod.uniqueTypeWithNewShape(&T4, {2, 1});
  auto *q4 = mod.uniqueTypeWithNewShape(u4, {1, 2});

  EXPECT_NE(t4, u4);
  EXPECT_EQ(t4, q4);
}

TEST(IR, basicUseList) {
  Module mod;
  Function *F = mod.createFunction("main");
  IRFunction M(F);
  {
    IRBuilder builder(&M);

    auto *V1 = builder.createWeightVar(ElemKind::FloatTy, {320, 200});
    auto *V2 = builder.createWeightVar(ElemKind::FloatTy, {320, 200});

    // Check that we can construct a new instruction.
    auto *CC = builder.createCopyInst("C", V1, V2);
    InstructionNumbering IN(M);
    CC->verifyUseList(IN);

    // Check the getOperand and setOperand functions.
    EXPECT_EQ(CC->getDest(), V1);
    CC->setOperand(0, V2);
    EXPECT_EQ(CC->getDest(), V2);
    CC->verifyUseList(IN);
  }

  // Check that we can destroy the operands.
  // ...
}

TEST(IR, allInstrs) {
  using MK = WeightVar::MutabilityKind;

  Module mod;
  Function *F = mod.createFunction("main");
  IRFunction M(F);
  auto T1 = mod.uniqueType(ElemKind::FloatTy, {1, 24, 24, 3});
  auto T2 = mod.uniqueType(ElemKind::FloatTy, {64});
  auto T4 = mod.uniqueType(ElemKind::IndexTy, {1, 1});

  {
    IRBuilder builder(&M);

    auto *I0 = builder.createWeightVar(T1, "I0");
    auto *I1 = builder.createWeightVar(T1, "I1");
    auto *I2 = builder.createWeightVar(ElemKind::FloatTy, {1, 3, 24, 24}, "I2",
                                       MK::Constant);

    auto *I3 = builder.createWeightVar(ElemKind::FloatTy, {1, 12, 12, 64});
    auto *I4 = builder.createWeightVar(ElemKind::FloatTy, {1, 12, 12, 3});
    auto *I6 = builder.createWeightVar(ElemKind::FloatTy, {2, 12, 12, 64});
    auto *I8 = builder.createWeightVar(ElemKind::FloatTy, {1, 24, 3, 24}, "I8");
    auto *ComputationInfo =
        builder.createWeightVar(ElemKind::FloatTy, {2}, "ComputationInfo");

    auto *XY = builder.createWeightVar(ElemKind::IndexTy, {1, 12, 12, 3, 2});
    auto *B0 = builder.createWeightVar(T2, "B0");
    auto *B1 =
        builder.createWeightVar(ElemKind::FloatTy, {32}, "B1", MK::Mutable);
    auto *F0 = builder.createWeightVar(ElemKind::FloatTy, {64, 7, 7, 3});
    auto *F1 = builder.createWeightVar(ElemKind::FloatTy, {32, 1728});
    auto *E0 = builder.createWeightVar(T4, "E0");

    B0->setName("bias");
    B1->setName("FC_bias");
    F0->setName("filter");
    F1->setName("FC_filter");
    E0->setName("expected");
    XY->setName("srcXY");

    builder.createCopyInst("", I1, I0);
    builder.createConvolutionInst("", I3, I1, F0, B0, 7, 2, 3, 1);
    builder.createPoolMaxInst("", I4, I0, 7, 2, 3);
    builder.createSigmoidInst("", I1, I0);
    builder.createTanhInst("", I1, I0);
    builder.createSoftMaxInst("", I1, I0);
    builder.createTransposeInst("", I8, I2, {0, 3, 1, 2});
    builder.createTensorView(ElemKind::FloatTy, {1, 24, 3, 24}, I2, "I2_view");
    builder.createInsertTensorInst("", I6, I3, {0, 0, 0, 0}, 1, 0);
    builder.createElementMulInst("", I1, I0, I0);
    builder.createDebugPrintInst("", I0);
    builder.createQuantizationProfileInst("", I0, B0, ComputationInfo);
  }
  M.verify();
}

TEST(IR, casting) {
  Module mod;
  Function *F = mod.createFunction("main");
  IRFunction M(F);
  {
    IRBuilder bb(&M);

    auto *input = bb.createWeightVar(ElemKind::FloatTy, {1, 224, 224, 3});
    auto *res = bb.createAllocActivationInst("sigmoid.res", input->getType());
    auto *sig = bb.createSigmoidInst("sigmoid", res, input);
    auto *pool = bb.createPoolAvgOp(sig->getDest(), 7, 2, 3);

    EXPECT_EQ(isa<PoolAvgInst>(pool), true);
    EXPECT_EQ(isa<PoolAvgInst>(input), false);
    EXPECT_EQ(isa<SigmoidInst>(sig), true);
    EXPECT_EQ(isa<SigmoidInst>(pool), false);

    EXPECT_NE(dyn_cast<PoolAvgInst>(pool), nullptr);
    EXPECT_EQ(dyn_cast<PoolAvgInst>(pool), pool);

    EXPECT_NE(dyn_cast<WeightVar>(input), nullptr);
    EXPECT_EQ(dyn_cast<WeightVar>(input), input);
  }
}

TEST(IR, predicateIR) {
  Module mod;
  Function *F = mod.createFunction("predicated");
  IRFunction M(F);
  {
    IRBuilder builder(&M);
    auto *V1 = builder.createWeightVar(ElemKind::FloatTy, {320, 200});
    auto *V2 = builder.createWeightVar(ElemKind::FloatTy, {320, 200});
    auto *P = builder.createWeightVar(ElemKind::IndexTy, {320}, "p1");

    // Check that we can construct a new instruction.
    auto *CC = builder.createCopyInst("C", V1, V2);
    // Set the predicate.
    CC->setPredicate(P);
    InstructionNumbering IN(M);
    CC->verifyUseList(IN);
    M.verify();
  }
}
