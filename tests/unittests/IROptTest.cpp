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

TEST(Optimizer, DeadStoreElimination) {
  Graph G("DeadStoreElimination");
  Module M(&G);
  IRBuilder bb(&M);

  auto *input1 = bb.createWeightVar(glow::ElemKind::FloatTy, {1}, "input1",
                                    WeightVar::MutabilityKind::Constant);
  auto *input2 = bb.createWeightVar(glow::ElemKind::FloatTy, {1}, "input2",
                                    WeightVar::MutabilityKind::Constant);
  auto *output = bb.createWeightVar(glow::ElemKind::FloatTy, {1}, "ouput",
                                    WeightVar::MutabilityKind::Mutable);

  bb.createReluInst("relu1", output, input1);
  bb.createReluInst("relu2", output, input2);

  optimize(M, CompilationMode::Infer);

  // Check that the first relu instruction is eliminated, because its result is
  // never read.
  EXPECT_EQ(M.getInstrs().size(), 1);
}
