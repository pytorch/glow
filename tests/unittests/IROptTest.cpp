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

#include "BackendTestUtils.h"
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

  optimize(M, CompilationMode::Infer, MockBackend());

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
      "cast", output, mod.uniqueType(Type(glow::ElemKind::FloatTy, {1, 1, 1})),
      {0, 0, 0});

  optimize(M, CompilationMode::Infer, MockBackend());

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

  optimize(M, CompilationMode::Infer, MockBackend());

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

  optimize(M, CompilationMode::Infer, MockBackend());

  EXPECT_EQ(M.getInstrs().size(), 5);

  auto &instrs = M.getInstrs();
  EXPECT_TRUE(std::none_of(
      instrs.begin(), instrs.end(), [](const Instruction &I) -> bool {
        return I.getKind() == Instruction::Kind::CopyInstKind;
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

  optimize(M, CompilationMode::Infer, MockBackend());

  EXPECT_EQ(M.getInstrs().size(), 2);

  auto &instrs = M.getInstrs();
  EXPECT_TRUE(std::none_of(
      instrs.begin(), instrs.end(), [](const Instruction &I) -> bool {
        return isa<AllocActivationInst>(&I) || isa<DeallocActivationInst>(&I) ||
               isa<CopyInst>(&I);
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

  optimize(M, CompilationMode::Infer, MockBackend());

  EXPECT_EQ(M.getInstrs().size(), 5);

  auto &instrs = M.getInstrs();
  EXPECT_TRUE(std::none_of(
      instrs.begin(), instrs.end(), [](const Instruction &I) -> bool {
        return isa<TransposeInst>(&I) || isa<AllocActivationInst>(&I) ||
               isa<DeallocActivationInst>(&I);
      }));
}

/// Simple test where a single insert is replaced by a tensor view with offsets.
TEST(Optimizer, insertOptimizer) {
  Module mod;
  Function *F = mod.createFunction("InsertOptimizer");
  IRFunction M(F);
  IRBuilder bb(&M);

  auto *output =
      bb.createWeightVar(glow::ElemKind::FloatTy, {4, 4, 5}, "output",
                         WeightVar::MutabilityKind::Mutable);

  auto *allocSrc = bb.createAllocActivationInst(
      "allocSrc", glow::ElemKind::FloatTy, {3, 4, 5});

  bb.createSplatInst("splatSrc", allocSrc, 1.0);
  bb.createSplatInst("splatDest", output, 2.0);

  bb.createInsertTensorInst("insert", output, allocSrc, {1, 0, 0}, 1, 0);

  bb.createDeallocActivationInst("deallocSrc", allocSrc);

  optimize(M, CompilationMode::Infer, MockBackend());

  // After optimization, should be left with two splats and a tensorview; the
  // insert, alloc, and dealloc should be gone.
  auto &instrs = M.getInstrs();
  EXPECT_EQ(instrs.size(), 3);
  EXPECT_TRUE(std::all_of(
      instrs.begin(), instrs.end(), [](const Instruction &I) -> bool {
        return isa<SplatInst>(&I) || isa<TensorViewInst>(&I);
      }));
}

/// This is representative of what a ConcatNode is IRGen'd into: src1 and src2
/// represent the two tensors that are being concatenated, and dest represents
/// the resulting concatenated tensor.
TEST(Optimizer, twoInsertsWithBuffersOptimizer) {
  Module mod;
  Function *F = mod.createFunction("InsertWithBufferOptimizer");
  IRFunction M(F);
  IRBuilder bb(&M);

  auto *output =
      bb.createWeightVar(glow::ElemKind::FloatTy, {4, 4, 5}, "output",
                         WeightVar::MutabilityKind::Mutable);

  auto *allocSrc1 = bb.createAllocActivationInst(
      "allocSrc1", glow::ElemKind::FloatTy, {2, 4, 5});
  auto *allocSrc2 = bb.createAllocActivationInst(
      "allocSrc2", glow::ElemKind::FloatTy, {2, 4, 5});
  auto *allocDest = bb.createAllocActivationInst(
      "allocDest", glow::ElemKind::FloatTy, {4, 4, 5});

  bb.createSplatInst("splatSrc1", allocSrc1, 1.0);
  bb.createSplatInst("splatSrc2", allocSrc2, 2.0);
  bb.createSplatInst("splatDest", allocDest, 3.0);

  bb.createInsertTensorInst("insert1", allocDest, allocSrc1, {0, 0, 0}, 1, 0);
  bb.createInsertTensorInst("insert2", allocDest, allocSrc2, {2, 0, 0}, 1, 0);

  bb.createCopyInst("copy", output, allocDest);

  bb.createDeallocActivationInst("deallocDest", allocDest);
  bb.createDeallocActivationInst("deallocSrc2", allocSrc2);
  bb.createDeallocActivationInst("deallocSrc1", allocSrc1);

  optimize(M, CompilationMode::Infer, MockBackend());

  // After optimization, should be left with three splats and two tensorviews;
  // the inserts, allocs, and deallocs should be gone.
  auto &instrs = M.getInstrs();
  EXPECT_EQ(instrs.size(), 5);
  EXPECT_TRUE(std::all_of(
      instrs.begin(), instrs.end(), [](const Instruction &I) -> bool {
        return isa<SplatInst>(&I) || isa<TensorViewInst>(&I);
      }));
}

/// This is representative of what a SliceNode is IRGen'd into: src is the
/// original source tensor, and then two slices are created into dest1 and
/// dest2.
TEST(Optimizer, twoExtractsWithBuffersOptimizer) {
  Module mod;
  Function *F = mod.createFunction("ExtractWithBufferOptimizer");
  IRFunction M(F);
  IRBuilder bb(&M);

  auto *output1 =
      bb.createWeightVar(glow::ElemKind::FloatTy, {2, 4, 5}, "output1",
                         WeightVar::MutabilityKind::Mutable);
  auto *output2 =
      bb.createWeightVar(glow::ElemKind::FloatTy, {2, 4, 5}, "output2",
                         WeightVar::MutabilityKind::Mutable);

  auto *allocSrc = bb.createAllocActivationInst(
      "allocSrc", glow::ElemKind::FloatTy, {4, 4, 5});
  auto *allocDest1 = bb.createAllocActivationInst(
      "allocDest1", glow::ElemKind::FloatTy, {2, 4, 5});
  auto *allocDest2 = bb.createAllocActivationInst(
      "allocDest2", glow::ElemKind::FloatTy, {2, 4, 5});

  bb.createSplatInst("splatSrc", allocSrc, 3.0);

  bb.createExtractTensorInst("extract1", allocDest1, allocSrc, {0, 0, 0});
  bb.createExtractTensorInst("extract2", allocDest2, allocSrc, {2, 0, 0});

  bb.createCopyInst("copy", output1, allocDest1);
  bb.createCopyInst("copy", output2, allocDest2);

  bb.createDeallocActivationInst("deallocSrc", allocSrc);
  bb.createDeallocActivationInst("deallocDest2", allocDest2);
  bb.createDeallocActivationInst("deallocDest1", allocDest1);

  optimize(M, CompilationMode::Infer, MockBackend());

  // After optimization, the extracts should be gone, as well as both allocDests
  // and their deallocs. Should be left with splatSrc, allocSrc, deallocSrc, two
  // tensorviews, and two copies from the tensorviews into the outputs.
  auto &instrs = M.getInstrs();
  EXPECT_EQ(instrs.size(), 7);
  EXPECT_TRUE(std::none_of(
      instrs.begin(), instrs.end(),
      [](const Instruction &I) -> bool { return isa<ExtractTensorInst>(&I); }));
}
