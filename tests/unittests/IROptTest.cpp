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
#include "glow/IR/IRUtils.h"
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
  auto *output = bb.createWeightVar(glow::ElemKind::FloatTy, {1}, "output",
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
  auto *output = bb.createWeightVar(glow::ElemKind::FloatTy, {1}, "output",
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
  auto *output = bb.createWeightVar(glow::ElemKind::FloatTy, {1}, "output",
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

TEST(Optimizer, deleteDeadViews) {
  Module mod;
  Function *F = mod.createFunction("DeleteDeadViews");
  IRFunction M(F);
  IRBuilder bb(&M);

  auto *input = bb.createWeightVar(glow::ElemKind::FloatTy, {1}, "input",
                                   WeightVar::MutabilityKind::Constant);
  auto *output = bb.createWeightVar(glow::ElemKind::FloatTy, {1}, "output",
                                    WeightVar::MutabilityKind::Mutable);

  auto *tensorView1 = bb.createTensorViewInst(
      "tensor_view1", input,
      mod.uniqueType(Type{glow::ElemKind::FloatTy, {1, 1}}), {0, 0});

  bb.createTensorViewInst("tensor_view2", tensorView1,
                          mod.uniqueType(Type{glow::ElemKind::FloatTy, {1}}),
                          {0});
  bb.createCopyInst("copy", output, input);

  optimize(M, CompilationMode::Infer, MockBackend());

  // Check that all tensor_view instructions are eliminated, because they are
  // never used.
  EXPECT_EQ(M.getInstrs().size(), 1);
}

TEST(Optimizer, copyPropagation) {
  Module mod;
  Function *F = mod.createFunction("ShareBuffers");
  IRFunction M(F);
  IRBuilder bb(&M);

  auto *input = bb.createWeightVar(glow::ElemKind::FloatTy, {1}, "input",
                                   WeightVar::MutabilityKind::Constant);
  auto *output = bb.createWeightVar(glow::ElemKind::FloatTy, {1}, "output",
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
  auto *output = bb.createWeightVar(glow::ElemKind::FloatTy, {1}, "output",
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
      bb.createWeightVar(glow::ElemKind::FloatTy, {3, 1, 1}, "output1",
                         WeightVar::MutabilityKind::Mutable);
  auto *output2 =
      bb.createWeightVar(glow::ElemKind::FloatTy, {1, 1, 3}, "output2",
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

/// Check that we are able to coalesce a copy forward from the input.
/// This test consists in copy from the input variable.
/// Its may characteristic is that this copy cannot be coalesced with
/// the output (otherwise it would be a backward chain of
/// copies from output).
/// The shareBuffers optimization works backward, so as long as
/// it manages to coalesce things with output one by one, we
/// won't see if the forward copies are properly handled.
TEST(Optimizer, forwardCopy) {
  Module mod;
  Function *F = mod.createFunction("forwardCopy");
  IRFunction M(F);
  IRBuilder bb(&M);

  auto *input = bb.createWeightVar(glow::ElemKind::FloatTy, {64}, "input",
                                   WeightVar::MutabilityKind::Mutable);
  auto *output = bb.createWeightVar(glow::ElemKind::FloatTy, {2, 64}, "output",
                                    WeightVar::MutabilityKind::Mutable);
  auto *tmp1 =
      bb.createAllocActivationInst("tmp1", glow::ElemKind::FloatTy, {64});
  bb.createCopyInst("copy1", tmp1, input);

  auto *view = bb.createTensorViewInst(
      "view", tmp1, mod.uniqueType(Type(glow::ElemKind::FloatTy, {1, 64})),
      {0});
  bb.createInsertTensorInst("copyOutput", output, view, {0, 0}, 1, 0);

  bb.createDeallocActivationInst("dealloc1", tmp1);

  auto &instrs = M.getInstrs();
  auto nbInstrsBeforeOpt = instrs.size();
  optimize(M, CompilationMode::Infer, MockBackend());

  // After optimization, the copy should have been coalesced with input.
  // nbIntrsBeforeOpt - 1 copy - 1 dealloc - 1 alloc
  EXPECT_EQ(instrs.size(),
            nbInstrsBeforeOpt /*copy*/ - 1 /*alloca*/ - 1 /*dealloc*/ - 1);
  EXPECT_TRUE(std::none_of(instrs.begin(), instrs.end(),
                           [](const Instruction &I) -> bool {
                             return isa<AllocActivationInst>(&I);
                           }));
  EXPECT_TRUE(std::none_of(
      instrs.begin(), instrs.end(),
      [](const Instruction &I) -> bool { return isa<CopyInst>(&I); }));
}

/// Check that we are able to coalesce chain of copies
/// forward from the input.
/// This test is similar to forwardCopy, expect it uses a chain of copies (more
/// than one) instead of just on copy from input.
TEST(Optimizer, chainOfTwoForwardCopies) {
  Module mod;
  Function *F = mod.createFunction("chainOfTwoForwardCopies");
  IRFunction M(F);
  IRBuilder bb(&M);

  auto *input = bb.createWeightVar(glow::ElemKind::FloatTy, {64}, "input",
                                   WeightVar::MutabilityKind::Mutable);
  auto *output = bb.createWeightVar(glow::ElemKind::FloatTy, {2, 64}, "output",
                                    WeightVar::MutabilityKind::Mutable);
  auto *tmp1 =
      bb.createAllocActivationInst("tmp1", glow::ElemKind::FloatTy, {64});
  bb.createCopyInst("copy1", tmp1, input);

  auto *tmp2 =
      bb.createAllocActivationInst("tmp2", glow::ElemKind::FloatTy, {64});
  bb.createCopyInst("copy2", tmp2, tmp1);
  auto *view = bb.createTensorViewInst(
      "view", tmp2, mod.uniqueType(Type(glow::ElemKind::FloatTy, {1, 64})),
      {0});
  bb.createInsertTensorInst("copyOutput", output, view, {0, 0}, 1, 0);

  bb.createDeallocActivationInst("dealloc1", tmp1);
  bb.createDeallocActivationInst("dealloc2", tmp2);

  auto &instrs = M.getInstrs();
  auto nbInstrsBeforeOpt = instrs.size();
  optimize(M, CompilationMode::Infer, MockBackend());

  // After optimization, the copies should have been coalesced with
  // input.
  // Ideally, we should get rid of 2 copies, the related 2 allocactivations and
  // deallocation.
  // Therefore expected instructions should be
  // nbIntrsBeforeOpt - 2 copies - 2 dealloc - 2 alloc
  EXPECT_EQ(instrs.size(),
            nbInstrsBeforeOpt /*copy*/ - 2 /*alloca*/ - 2 /*dealloc*/ - 2);
  EXPECT_TRUE(std::none_of(instrs.begin(), instrs.end(),
                           [](const Instruction &I) -> bool {
                             return isa<AllocActivationInst>(&I);
                           }));
  EXPECT_TRUE(std::none_of(
      instrs.begin(), instrs.end(),
      [](const Instruction &I) -> bool { return isa<CopyInst>(&I); }));
}

/// The idea of this test is to have live intervals looking like this:
/// A          B
/// |   <-copy |
/// inout      |
/// |          |
/// Because of the inout on A, A and B interfere.
/// Make sure we don't coalesce such buffers.
TEST(Optimizer, inoutCopy) {
  Module mod;
  Function *F = mod.createFunction("inoutCopy");
  IRFunction M(F);
  IRBuilder bb(&M);

  auto *input = bb.createWeightVar(glow::ElemKind::FloatTy, {2, 64}, "input",
                                   WeightVar::MutabilityKind::Mutable);
  auto *output = bb.createWeightVar(glow::ElemKind::FloatTy, {3, 64}, "output",
                                    WeightVar::MutabilityKind::Mutable);
  auto *output2 =
      bb.createWeightVar(glow::ElemKind::FloatTy, {2, 64}, "output2",
                         WeightVar::MutabilityKind::Mutable);
  // This copy cannot be eliminated because input must not be changed.
  // Indeed, this is an observable variable plus it is used as a source
  // for a copy to output2.
  auto *tmp1 =
      bb.createAllocActivationInst("tmp1", glow::ElemKind::FloatTy, {2, 64});
  bb.createCopyInst("copy1", tmp1, input);

  auto *tmp2 =
      bb.createAllocActivationInst("tmp2", glow::ElemKind::FloatTy, {64});
  bb.createSplatInst("splat", tmp2, 3.0);
  auto *view =
      bb.createTensorView(ElemKind::FloatTy, {1, 64}, tmp2, "view", {0});
  bb.createInsertTensorInst("insertTmp1", tmp1, view, {0, 0}, 1, 0);
  bb.createInsertTensorInst("insertOutput", output, tmp1, {1, 0}, 1, 0);
  bb.createCopyInst("copyOutput2", output2, input);

  bb.createDeallocActivationInst("dealloc1", tmp1);
  bb.createDeallocActivationInst("dealloc2", tmp2);

  optimize(M, CompilationMode::Infer, MockBackend());

  // After optimization, the copies shouldn't have been touched.
  // tmp1 = copy input cannot be coalesced because tmp1 is inout.
  // output2 = copy input cannot be coalesced because they are both
  // externally visible.
  EXPECT_EQ(input->getNumUsers(), 2);
  EXPECT_TRUE(
      std::all_of(input->getUsers().begin(), input->getUsers().end(),
                  [](const Use &I) -> bool { return isa<CopyInst>(I.get()); }));
  const Value *expectedDest[] = {tmp1, output2};
  unsigned idx = 0;
  for (const Use &use : input->getUsers()) {
    if (idx == sizeof(expectedDest) / sizeof(expectedDest[0])) {
      // If we end up here that means that input has too many users.
      EXPECT_FALSE(true);
      break;
    }
    EXPECT_EQ(use.get()->getOperand(0).first, expectedDest[idx++]);
  }
}

/// Check that we properly define a buffer when we extend its live-range
/// on a segment of the source that does not have any definition.
/// A source live-range without any definition can happen when this
/// is the first use of a WeightVar.
/// At the high level, this test looks like this:
/// WeightVar    Buffer
///    | useA
///    | useB      | def
///    | redef     | save to output
/// - UseA is the first use of WeightVar and we want it to be replaced by
///   a use of Buffer. I.e., Buffer live-range is extended toward the top.
/// - UseB involves both WeightVar and Buffer. It exposes the buffer sharing
///   opportunity between these two variables. It must happen after useA
///   to expose the case of extending the live-range of a buffer toward
///   the top where no definition exists.
/// - redef redefines WeightVar. It is necessary otherwise both useA and useB
///   could all share the same buffer and thus, we would extend the live-range
///   of the buffer in useA downward (or the use of Buffer up to the
///   definition of the buffer in useA), which is not what we want to test.
/// - save to output is required to keep the def of Buffer alive. Moreover,
///   the save must be done in such a way that the output buffer and Buffer
///   must not be able to share the same buffer. Otherwise, the live-range
///   of output buffer will be extended upward to useB and given output and
///   WeightVar are both externally observable, the output buffer cannot be
///   merged with WeightVar.
///   Therefore, we won't expose an extension of output up to useA
///   and won't test the case where the replaced buffer doesn't have any
///   definition.
///
/// The expected result at a high level looks like this:
/// WeightVar    Buffer
///    |   copy    | <- Buffer gets WeightVar
///      useA      | <- Buffer is used instead of WeightVar
///      useB      | def <- ditto
///    | redef     | save to output
TEST(Optimizer, bufferReuseWithoutDefs) {
  Module mod;
  Function *F = mod.createFunction("bufferReuseWithoutDefs");
  IRFunction M(F);
  IRBuilder bb(&M);

  auto *input = bb.createWeightVar(glow::ElemKind::FloatTy, {64}, "input",
                                   WeightVar::MutabilityKind::Mutable);
  auto *output = bb.createWeightVar(glow::ElemKind::FloatTy, {2, 64}, "output",
                                    WeightVar::MutabilityKind::Mutable);
  auto *tmp1 =
      bb.createAllocActivationInst("tmp1", glow::ElemKind::FloatTy, {64});

  auto *tmp2 =
      bb.createAllocActivationInst("tmp2", glow::ElemKind::FloatTy, {64});
  auto *tmp3 =
      bb.createAllocActivationInst("tmp3", glow::ElemKind::FloatTy, {64});

  bb.createSplatInst("tmp2init", tmp2, 1.0);
  // use input for some stuff.
  auto *useA = bb.createElementAddInst("useA", tmp3, tmp2, input);
  // Make the first user of input a dependency of the definition
  // of tmp1 that way the scheduler cannot mess with the layout
  // we want for the instructions ordering.
  bb.createElementAddInst("useB", tmp1, input, tmp3);
  bb.createCopyInst("redef", input, tmp3);
  auto *view = bb.createTensorViewInst(
      "view", tmp1, mod.uniqueType(Type(glow::ElemKind::FloatTy, {1, 64})),
      {0});
  bb.createInsertTensorInst("save", output, view, {0, 0}, 1, 0);

  bb.createDeallocActivationInst("dealloc1", tmp1);
  bb.createDeallocActivationInst("dealloc2", tmp2);
  bb.createDeallocActivationInst("dealloc2", tmp3);

  optimize(M, CompilationMode::Infer, MockBackend());

  // Check that we manage to expose the problematic case we wanted:
  // tmp1 is extended upward and replace the use of input.
  EXPECT_EQ(useA->getRHS(), tmp1);
  // Check that tmp1 is properly defined before useA.
  Instruction *instBeforeUseA = &*std::prev(useA->getIterator());

  EXPECT_TRUE(isa<CopyInst>(instBeforeUseA));
  // The somewhat complicated check is to make sure we don't crash the test
  // when instBeforeUseA is not a copy.
  // I.e., this test was failing (instead of crashing) when the
  // bug was present.
  EXPECT_EQ(instBeforeUseA->getNumOperands() > 0
                ? instBeforeUseA->getOperand(0).first
                : nullptr,
            tmp1);
  EXPECT_EQ(instBeforeUseA->getNumOperands() > 1
                ? instBeforeUseA->getOperand(1).first
                : nullptr,
            input);
}

/// Same as bufferReuseWithoutDefs but with casts in the middle.
/// This makes sure that we properly set the types for whatever fixup
/// code we will insert.
/// The high level view of the test is:
/// WeightVar    Buffer
///    | useA
///    | useB(cast)| def
///    | redef     | save to output
///
/// The expected result at a high level looks like this:
/// WeightVar    Buffer
///    | copy(cast)| <- Buffer gets WeightVar
///      useA(cast)| <- Buffer is used instead of WeightVar
///      useB      | def <- ditto
///    | redef     | save to output
TEST(Optimizer, bufferReuseWithoutDefsPlusCasts) {
  Module mod;
  Function *F = mod.createFunction("bufferReuseWithoutDefsPlusCasts");
  IRFunction M(F);
  IRBuilder bb(&M);

  auto *input = bb.createWeightVar(glow::ElemKind::FloatTy, {1, 64}, "input",
                                   WeightVar::MutabilityKind::Mutable);
  auto *output = bb.createWeightVar(glow::ElemKind::FloatTy, {2, 64}, "output",
                                    WeightVar::MutabilityKind::Mutable);
  auto *tmp1 =
      bb.createAllocActivationInst("tmp1", glow::ElemKind::FloatTy, {64});

  auto *tmp2 =
      bb.createAllocActivationInst("tmp2", glow::ElemKind::FloatTy, {1, 64});
  auto *tmp3 =
      bb.createAllocActivationInst("tmp3", glow::ElemKind::FloatTy, {1, 64});

  bb.createSplatInst("tmp2init", tmp2, 1.0);
  auto *useA = bb.createElementAddInst("useA", tmp3, tmp2, input);
  auto *inputView = bb.createTensorViewInst(
      "inputView", input, mod.uniqueType(Type(glow::ElemKind::FloatTy, {64})),
      {0});
  auto *tmp3View = bb.createTensorViewInst(
      "tmp3View", tmp3, mod.uniqueType(Type(glow::ElemKind::FloatTy, {64})),
      {0});

  bb.createElementAddInst("useB", tmp1, inputView, tmp3View);
  bb.createCopyInst("redef", input, tmp3);
  auto *view = bb.createTensorViewInst(
      "view", tmp1, mod.uniqueType(Type(glow::ElemKind::FloatTy, {1, 64})),
      {0});
  bb.createInsertTensorInst("save", output, view, {0, 0}, 1, 0);

  bb.createDeallocActivationInst("dealloc1", tmp1);
  bb.createDeallocActivationInst("dealloc2", tmp2);
  bb.createDeallocActivationInst("dealloc2", tmp3);

  optimize(M, CompilationMode::Infer, MockBackend());

  // Check that we manage to expose the problematic case we wanted:
  // tmp1 is extended upward and replace the use of input.
  Value *useARHS = useA->getRHS();
  EXPECT_EQ(getOrigin(useARHS), tmp1);
  Instruction *tmp1TensorView = dyn_cast<TensorViewInst>(useARHS);
  EXPECT_TRUE(tmp1TensorView && tmp1TensorView->getOperand(0).first == tmp1);
  // Check that tmp1 is properly defined before useA.
  Instruction *tmp1Fixup =
      tmp1TensorView ? &*std::prev(tmp1TensorView->getIterator()) : nullptr;
  EXPECT_TRUE(tmp1Fixup && isa<CopyInst>(tmp1Fixup));
  // The somewhat complicated check is to make sure we don't crash the test
  // when instBeforeUseA is not a copy.
  EXPECT_EQ((tmp1Fixup && tmp1Fixup->getNumOperands() > 0)
                ? getOrigin(tmp1Fixup->getOperand(0).first)
                : nullptr,
            tmp1);
  // Now check that input feeds tmp1Fixup and was properly casted.
  Instruction *inputCast =
      (tmp1Fixup && tmp1Fixup->getNumOperands() > 1)
          ? dyn_cast<TensorViewInst>(tmp1Fixup->getOperand(1).first)
          : nullptr;
  EXPECT_EQ(inputCast ? getOrigin(inputCast) : nullptr, input);
  EXPECT_EQ(inputCast ? inputCast->getOperand(0).first : nullptr, input);
}
