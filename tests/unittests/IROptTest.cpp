// Copyright 2017 Facebook Inc.  All Rights Reserved.

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

/// Basic test of DSE (Dead Store Elimination)
TEST(Optimizer, dseBasic) {
  Graph G("DeadStoreElimination");
  Module M(&G);
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

/// Basic test of DSE (Dead Store Elimination)
TEST(Optimizer, dseDoNotRemloveLastWriteIntoWeightVar) {
  Graph G("DeadStoreElimination");
  Module M(&G);
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
      "cast", output, G.uniqueType(Type(glow::ElemKind::FloatTy, {1, 1, 1})));

  optimize(M, CompilationMode::Infer);

  // Check that the first relu instruction  and select are eliminated, because
  // their outputs are never read.
  EXPECT_EQ(M.getInstrs().size(), 1);
}