/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

#include "llvm/ADT/StringSet.h"
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
  Type T4(ElemKind::Int8QTy, {1, 2}, 0.4f, 2);
  auto *t4 = mod.uniqueType(T4);
  auto *u4 = mod.uniqueTypeWithNewShape(&T4, {2, 1});
  auto *q4 = mod.uniqueTypeWithNewShape(u4, {1, 2});

  EXPECT_NE(t4, u4);
  EXPECT_EQ(t4, q4);
}

#define TEST_QUANT_TYPE(kind, type, type_name, scale, offset)                  \
  {                                                                            \
    Type T(ElemKind::kind, {2, 3}, (scale), (offset));                         \
    EXPECT_EQ(T.getElementType(), ElemKind::kind);                             \
    EXPECT_EQ(T.getScale(), (scale));                                          \
    EXPECT_EQ(T.getOffset(), (offset));                                        \
    EXPECT_TRUE(T.isQuantizedType());                                          \
    EXPECT_EQ(T.getElementSize(), sizeof(type));                               \
    EXPECT_TRUE(T.isType<type>());                                             \
    auto range = T.getQuantizedValueRange();                                   \
    EXPECT_EQ(range.first, ((int64_t)type_name##_MIN - (offset)) * (scale));   \
    EXPECT_EQ(range.second, ((int64_t)type_name##_MAX - (offset)) * (scale));  \
  }

TEST(IR, basicQuantizedTypes) {
  // Quantized types
  TEST_QUANT_TYPE(Int8QTy, int8_t, INT8, 0.3f, -45);
  TEST_QUANT_TYPE(UInt8QTy, uint8_t, UINT8, 0.3f, -45);
  TEST_QUANT_TYPE(Int16QTy, int16_t, INT16, 0.3f, -45);
  TEST_QUANT_TYPE(Int32QTy, int32_t, INT32, 0.3f, -45);

  // Sanity check for non quantized types
  Type TF(ElemKind::FloatTy, {2, 3});
  EXPECT_FALSE(TF.isQuantizedType());
  Type T64I(ElemKind::Int64ITy, {2, 3});
  EXPECT_FALSE(T64I.isQuantizedType());
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

static IRFunction *createTestIRFunction(Module &mod) {
  using MK = WeightVar::MutabilityKind;

  Function *F = mod.createFunction("main");
  IRFunction *M = new IRFunction(F);
  auto T1 = mod.uniqueType(ElemKind::FloatTy, {1, 24, 24, 3});
  auto T2 = mod.uniqueType(ElemKind::FloatTy, {64});
  auto T4 = mod.uniqueType(ElemKind::Int64ITy, {1, 1});

  {
    IRBuilder builder(M);

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

    auto *argmax = builder.createWeightVar(ElemKind::Int64ITy, {1, 12, 12, 3});
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
    argmax->setName("argmax");

    builder.createCopyInst("", I1, I0);
    builder.createConvolutionInst("", I3, I1, F0, B0, {7, 7}, {2, 2},
                                  {3, 3, 3, 3}, 1, 1, NHWC,
                                  FusedActivation::NONE, {});
    builder.createMaxPoolInst("", I4, I0, {7, 7}, {2, 2}, {3, 3, 3, 3}, NHWC);
    builder.createSigmoidInst("", I1, I0);
    builder.createTanhInst("", I1, I0);
    builder.createSoftMaxInst("", I1, I0);
    builder.createTransposeInst("", I8, I2, NHWC2NCHW);
    builder.createTensorView(ElemKind::FloatTy, {1, 24, 3, 24}, I2, "I2_view");
    builder.createInsertTensorInst("", I6, I3, {0, 0, 0, 0}, 1, 0);
    builder.createElementMulInst("", I1, I0, I0);
    builder.createDebugPrintInst("", I0, "console", "");
    builder.createQuantizationProfileInst("", I0, B0, ComputationInfo);
  }
  return M;
}

TEST(IR, allInstrs) {
  Module mod;
  std::unique_ptr<IRFunction> M(createTestIRFunction(mod));
  M->verify();
}

/// Check the IR Functions cloning functionality.
TEST(IR, cloning) {
  Module mod;
  std::unique_ptr<IRFunction> M(createTestIRFunction(mod));
  std::unique_ptr<IRFunction> clonedM(M->clone(M->getName()));
  auto dumpedM = M->toString();
  auto dumpedClonedM = clonedM->toString();
  EXPECT_EQ(dumpedM, dumpedClonedM);
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
    auto *pool =
        bb.createAvgPoolOp(sig->getDest(), {7, 7}, {2, 2}, {3, 3, 3, 3}, NHWC,
                           /* countIncludePads */ true);

    EXPECT_EQ(isa<AvgPoolInst>(pool), true);
    EXPECT_EQ(isa<AvgPoolInst>(input), false);
    EXPECT_EQ(isa<SigmoidInst>(sig), true);
    EXPECT_EQ(isa<SigmoidInst>(pool), false);

    EXPECT_NE(dyn_cast<AvgPoolInst>(pool), nullptr);
    EXPECT_EQ(dyn_cast<AvgPoolInst>(pool), pool);

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
    auto *P = builder.createWeightVar(ElemKind::Int64ITy, {320}, "p1");

    // Check that we can construct a new instruction.
    auto *CC = builder.createCopyInst("C", V1, V2);
    // Set the predicate.
    CC->setPredicate(P);
    InstructionNumbering IN(M);
    CC->verifyUseList(IN);
    M.verify();
  }
}

/// Note that IRFunction validation uses asserts, so these tests only die when
/// asserts are turned on.
#ifndef NDEBUG

/// Check that the verify call dies when verifying an IRFunction with a
/// non-memory/view Instruction with another non-memory/view Instruction as
/// an input operand.
TEST(IR, VerifyDiesOnInvalidInputOperand) {
  Module mod;
  Function *F = mod.createFunction("InvalidOperands");
  IRFunction M(F);
  {
    IRBuilder builder(&M);
    auto *LHS = builder.createWeightVar(ElemKind::FloatTy, {10, 10});
    auto *RHS = builder.createWeightVar(ElemKind::FloatTy, {10, 10});
    auto *O = builder.createWeightVar(ElemKind::FloatTy, {10, 10});
    auto *EAI = builder.createElementAddInst("Add", O, LHS, RHS);

    // Invalid to use a non-memory/view Instruction as input operand.
    builder.createElementAddInst("Add", O, EAI, RHS);

    EXPECT_DEATH(M.verify(), "");
  }
}

/// Check that the verify call dies when verifying an IRFunction with a
/// non-memory/view Instruction with another non-memory/view Instruction as
/// an output operand.
TEST(IR, VerifyDiesOnInvalidOutputOperand) {
  Module mod;
  Function *F = mod.createFunction("InvalidOperands");
  IRFunction M(F);
  {
    IRBuilder builder(&M);
    auto *LHS = builder.createWeightVar(ElemKind::FloatTy, {10, 10});
    auto *RHS = builder.createWeightVar(ElemKind::FloatTy, {10, 10});
    auto *O = builder.createWeightVar(ElemKind::FloatTy, {10, 10});
    auto *EAI = builder.createElementAddInst("Add", O, LHS, RHS);

    // Invalid to use a non-memory/view Instruction as output operand.
    builder.createElementAddInst("Add", EAI, LHS, RHS);

    EXPECT_DEATH(M.verify(), "");
  }
}

#endif /* NDEBUG */

/// Verify that names of Instructions and WeightVars are uniqued when given the
/// same name.
TEST(IR, InstUniqueNames) {
  Module mod;
  Function *F = mod.createFunction("main");
  IRFunction M(F);
  {
    const std::string name = "name";

    // Add all of the names of the Instructions/WeightVars created to this set
    // to verify they are unique.
    llvm::StringSet<> nameSet;

    IRBuilder builder(&M);
    WeightVar *V1 =
        builder.createWeightVar(ElemKind::FloatTy, {1, 4, 4, 1}, name);
    auto it = nameSet.insert(V1->getName());
    EXPECT_TRUE(it.second);

    WeightVar *V2 = builder.createWeightVar(ElemKind::FloatTy, {4}, name);
    it = nameSet.insert(V2->getName());
    EXPECT_TRUE(it.second);

    MaxPoolWithArgmaxInst *MP1 = builder.createMaxPoolWithArgmaxOp(
        name, V1, {2, 2}, {1, 1}, {0, 2, 1, 3}, NHWC, ElemKind::Int64ITy);
    it = nameSet.insert(MP1->getName());
    EXPECT_TRUE(it.second);

    // IRBuilder::createMaxPoolWithArgmaxOp() creates alloc activation insts
    // internally, so we dealloc them here to keep the instruction list
    // well-formed.
    DeallocActivationInst *DAI1 =
        builder.createDeallocActivationInst(name, MP1->getArgmax());
    it = nameSet.insert(DAI1->getName());
    EXPECT_TRUE(it.second);

    DeallocActivationInst *DAI2 =
        builder.createDeallocActivationInst(name, MP1->getDest());
    it = nameSet.insert(DAI2->getName());
    EXPECT_TRUE(it.second);

    // IRBuilder::createTopKOp() creates alloc activation insts internally, so
    // we dealloc them here to keep the instruction list well-formed.
    TopKInst *TK = builder.createTopKOp(name, V2, 2, ElemKind::Int64ITy);
    it = nameSet.insert(TK->getName());
    EXPECT_TRUE(it.second);

    DeallocActivationInst *DAI3 =
        builder.createDeallocActivationInst(name, TK->getScratch());
    it = nameSet.insert(DAI3->getName());
    EXPECT_TRUE(it.second);

    DeallocActivationInst *DAI4 =
        builder.createDeallocActivationInst(name, TK->getValues());
    it = nameSet.insert(DAI4->getName());
    EXPECT_TRUE(it.second);

    DeallocActivationInst *DAI5 =
        builder.createDeallocActivationInst(name, TK->getIndices());
    it = nameSet.insert(DAI5->getName());
    EXPECT_TRUE(it.second);

    M.verify();
  }
}

TEST(IR, getOperandName) {
  Module mod;
  Function *F = mod.createFunction("main");
  IRFunction M(F);
  {
    IRBuilder bb(&M);

    auto *input = bb.createWeightVar(ElemKind::FloatTy, {1, 224, 224, 3});
    auto *res = bb.createAllocActivationInst("sigmoid.res", input->getType());
    auto *sig = bb.createSigmoidInst("sigmoid", res, input);
    auto *pool =
        bb.createAvgPoolOp(sig->getDest(), {7, 7}, {2, 2}, {3, 3, 3, 3}, NHWC,
                           /* countIncludePads */ true);

    EXPECT_EQ(pool->getNumOperands(), 2);
    EXPECT_EQ(pool->getOperandName(0), "Dest");
    EXPECT_EQ(pool->getOperandName(1), "Src");
  }
}

/// Check that Scratch is allocated properly for instructions.
TEST(IR, scratchAllocation) {
  Module mod;
  Function *F = mod.createFunction("main");
  IRFunction M(F);
  {
    IRBuilder bb(&M);
    auto *input = bb.createWeightVar(ElemKind::FloatTy, {10});
    TopKInst *topk = bb.createTopKOp("topk", input, 3, ElemKind::Int64ITy);
    // Verify scratch is allocated and has correct size.
    auto *scratch = topk->getScratch();
    EXPECT_TRUE(isa<AllocActivationInst>(scratch));
    EXPECT_EQ(scratch->getType()->size(), topk->getScratchSize());
  }
}
