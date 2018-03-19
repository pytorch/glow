// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"

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

/// Basic test of DSE (Dead Store Elimination)
TEST(Optimizer, dseBasic) {
  Module mod;
  Function *F = mod.createFunction("DeadStoreElimination");
  IRFunction M(F);
  IRBuilder bb(&M);

  auto *input1 = bb.createWeightVar(glow::ElemKind::FloatTy, {1}, "input1",
                                    WeightVar::MutabilityKind::Constant);
  auto *input2 = bb.createWeightVar(glow::ElemKind::FloatTy, {1}, "input2",
                                    WeightVar::MutabilityKind::Constant);
  auto *output = bb.createWeightVar(glow::ElemKind::FloatTy, {1}, "ouput",
                                    WeightVar::MutabilityKind::Mutable);

  bb.createElementAddInst("elem_add1", output, input1, input1);
  bb.createElementSelectInst("select", output, input1, output, input2);
  bb.createElementAddInst("elem_add2", output, input2, input2);

  optimize(M, CompilationMode::Infer);

  // Check that the first relu instruction  and select are eliminated, because
  // their outputs are never read.
  EXPECT_EQ(M.getInstrs().size(), 1);
}

/// Check that DSE does not remove the last write into a WeightVar.
TEST(Optimizer, dseDoNotRemloveLastWriteIntoWeightVar) {
  Module mod;
  Function *F = mod.createFunction("DeadStoreElimination");
  IRFunction M(F);
  IRBuilder bb(&M);

  auto *input1 = bb.createWeightVar(glow::ElemKind::FloatTy, {1}, "input1",
                                    WeightVar::MutabilityKind::Constant);
  auto *input2 = bb.createWeightVar(glow::ElemKind::FloatTy, {1}, "input2",
                                    WeightVar::MutabilityKind::Constant);
  auto *output = bb.createWeightVar(glow::ElemKind::FloatTy, {1}, "ouput",
                                    WeightVar::MutabilityKind::Mutable);

  // Last write into a WeightVar should not be removed even if there is
  // no instruction that reads it, because it is an observable side-effect.
  bb.createElementAddInst("elem_add", output, input1, input2);
  bb.createTensorViewInst(
      "cast", output, mod.uniqueType(Type(glow::ElemKind::FloatTy, {1, 1, 1})));

  optimize(M, CompilationMode::Infer);

  // Check that the first relu instruction  and select are eliminated, because
  // their outputs are never read.
  EXPECT_EQ(M.getInstrs().size(), 1);
}

TEST(Optimizer, shareBuffers) {
  Module mod;
  Function *F = mod.createFunction("ShareBuffers");
  IRFunction M(F);
  IRBuilder bb(&M);

  auto *input = bb.createWeightVar(glow::ElemKind::FloatTy, {1}, "input",
                                   WeightVar::MutabilityKind::Constant);
  auto *output = bb.createWeightVar(glow::ElemKind::FloatTy, {1}, "ouput",
                                    WeightVar::MutabilityKind::Mutable);

  auto *alloc1 =
      bb.createAllocActivationInst("alloc1", glow::ElemKind::FloatTy, 1);
  auto *alloc2 =
      bb.createAllocActivationInst("alloc2", glow::ElemKind::FloatTy, 1);
  auto *alloc3 =
      bb.createAllocActivationInst("alloc3", glow::ElemKind::FloatTy, 1);
  bb.createSplatInst("splat1", alloc1, 0.0);
  bb.createSplatInst("splat2", alloc2, 1.0);
  bb.createElementAddInst("elem_add1", alloc3, alloc1, input);
  bb.createElementAddInst("elem_add2", alloc2, input, input);
  // alloc1 and alloc2 are not live after this instruction.
  bb.createElementAddInst("elem_add3", alloc1, alloc2, input);
  bb.createCopyInst("copy", output, alloc3);
  bb.createDeallocActivationInst("dealloc3", alloc3);
  bb.createDeallocActivationInst("dealloc2", alloc2);
  bb.createDeallocActivationInst("dealloc1", alloc1);

  optimize(M, CompilationMode::Infer);

  // Check that the first relu instruction  and select are eliminated, because
  // their outputs are never read.
  EXPECT_EQ(M.getInstrs().size(), 2);
}

TEST(Optimizer, copyPropagation) {
  Module mod;
  Function *F = mod.createFunction("ShareBuffers");
  IRFunction M(F);
  IRBuilder bb(&M);

  auto *input = bb.createWeightVar(glow::ElemKind::FloatTy, {1}, "input",
                                   WeightVar::MutabilityKind::Constant);
  auto *output = bb.createWeightVar(glow::ElemKind::FloatTy, {1}, "ouput",
                                    WeightVar::MutabilityKind::Mutable);

  auto *alloc1 =
      bb.createAllocActivationInst("alloc1", glow::ElemKind::FloatTy, 1);
  auto *alloc2 =
      bb.createAllocActivationInst("alloc2", glow::ElemKind::FloatTy, 1);
  auto *alloc3 =
      bb.createAllocActivationInst("alloc3", glow::ElemKind::FloatTy, 1);
  bb.createSplatInst("splat1", alloc1, 1.0);
  bb.createCopyInst("copy1", alloc2, alloc1);
  bb.createElementAddInst("elem_add1", output, alloc2, input);
  bb.createSplatInst("splat2", alloc1, 0.0);
  bb.createElementAddInst("elem_add2", output, alloc2, alloc1);
  bb.createDeallocActivationInst("dealloc3", alloc3);
  bb.createDeallocActivationInst("dealloc2", alloc2);
  bb.createDeallocActivationInst("dealloc1", alloc1);

  optimize(M, CompilationMode::Infer);

  EXPECT_EQ(M.getInstrs().size(), 5);

  auto &instrs = M.getInstrs();
  EXPECT_TRUE(std::none_of(
      instrs.begin(), instrs.end(), [](const Instruction *I) -> bool {
        return I->getKind() == Instruction::Kind::CopyInstKind;
      }));
}

TEST(Optimizer, copyPropagationSimple) {
  Module mod;
  auto *F = mod.createFunction("ShareBuffers");
  IRFunction M(F);
  IRBuilder bb(&M);

  auto *input = bb.createWeightVar(glow::ElemKind::FloatTy, {1}, "input",
                                   WeightVar::MutabilityKind::Constant);
  auto *output = bb.createWeightVar(glow::ElemKind::FloatTy, {1}, "ouput",
                                    WeightVar::MutabilityKind::Mutable);

  auto *alloc1 =
      bb.createAllocActivationInst("alloc1", glow::ElemKind::FloatTy, 1);
  auto *alloc2 =
      bb.createAllocActivationInst("alloc2", glow::ElemKind::FloatTy, 1);
  bb.createSplatInst("splat1", alloc1, 1.0);
  bb.createCopyInst("copy1", alloc2, alloc1);
  bb.createElementAddInst("elem_add1", output, alloc2, input);
  bb.createDeallocActivationInst("dealloc2", alloc2);
  bb.createDeallocActivationInst("dealloc1", alloc1);

  optimize(M, CompilationMode::Infer);

  EXPECT_EQ(M.getInstrs().size(), 2);

  auto &instrs = M.getInstrs();
  EXPECT_TRUE(std::none_of(
      instrs.begin(), instrs.end(), [](const Instruction *I) -> bool {
        return isa<AllocActivationInst>(I) || isa<DeallocActivationInst>(I) ||
               isa<CopyInst>(I);
      }));
}

TEST(Optimizer, copyPropagationTranspose) {
  Module mod;
  Function *F = mod.createFunction("ShareBuffers");
  IRFunction M(F);
  IRBuilder bb(&M);

  auto *output1 =
      bb.createWeightVar(glow::ElemKind::FloatTy, {3, 1, 1}, "ouput1",
                         WeightVar::MutabilityKind::Mutable);
  auto *output2 =
      bb.createWeightVar(glow::ElemKind::FloatTy, {1, 1, 3}, "ouput2",
                         WeightVar::MutabilityKind::Mutable);

  auto *alloc1 = bb.createAllocActivationInst("alloc1", glow::ElemKind::FloatTy,
                                              {1, 1, 3});
  auto *alloc2 = bb.createAllocActivationInst("alloc2", glow::ElemKind::FloatTy,
                                              {3, 1, 1});
  bb.createSplatInst("splat1", alloc1, 1.0);
  bb.createTransposeInst("transpose", alloc2, alloc1, {2, 0, 1});
  bb.createElementAddInst("elem_add2", output1, alloc2, alloc2);
  bb.createElementAddInst("elem_add2", output2, alloc1, alloc1);
  bb.createDeallocActivationInst("dealloc2", alloc2);
  bb.createDeallocActivationInst("dealloc1", alloc1);

  optimize(M, CompilationMode::Infer);

  EXPECT_EQ(M.getInstrs().size(), 5);

  auto &instrs = M.getInstrs();
  EXPECT_TRUE(std::none_of(
      instrs.begin(), instrs.end(), [](const Instruction *I) -> bool {
        return isa<TransposeInst>(I) || isa<AllocActivationInst>(I) ||
               isa<DeallocActivationInst>(I);
      }));
}
