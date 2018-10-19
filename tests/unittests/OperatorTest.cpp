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

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "glow/Quantization/Quantization.h"

#include "gtest/gtest.h"

#include "llvm/Support/raw_ostream.h"

using namespace glow;

class Operator : public ::testing::TestWithParam<BackendKind> {
public:
  Operator() : mod_(EE_.getModule()) { F_ = mod_.createFunction("main"); }

  ~Operator() { mod_.clear(); }

protected:
  ExecutionEngine EE_{GetParam()};
  Module &mod_;
  Function *F_;
  Context ctx_;
};

class InterpAndCPU : public Operator {};

class InterpOnly : public Operator {};

TEST_P(Operator, pow) {
  auto *X = mod_.createPlaceholder(ElemKind::FloatTy, {1, 1, 3}, "X", false);
  auto *Y = mod_.createPlaceholder(ElemKind::FloatTy, {2}, "Y", false);
  auto *Exp = mod_.createPlaceholder(ElemKind::FloatTy, {2}, "Exp", false);

  ctx_.allocate(X)->getHandle() = {5, 0.1f, -3};
  ctx_.allocate(Y)->getHandle() = {2, 100};
  ctx_.allocate(Exp)->getHandle() = {2, -1};

  auto *Pow1 = F_->createPow("Pow1", X, 2.0);
  auto *Pow2 = F_->createPow("Pow2", Y, 0.5);
  auto *Pow3 = F_->createPow("Pow3", Y, Exp);

  auto *save1 = F_->createSave("save", Pow1);
  auto *savePlaceholder1 = save1->getPlaceholder();

  auto *save2 = F_->createSave("save", Pow2);
  auto *savePlaceholder2 = save2->getPlaceholder();

  auto *save3 = F_->createSave("save", Pow3);
  auto *savePlaceholder3 = save3->getPlaceholder();

  ctx_.allocate(savePlaceholder1);
  ctx_.allocate(savePlaceholder2);
  ctx_.allocate(savePlaceholder3);

  EE_.compile(CompilationMode::Infer, F_, ctx_);

  EE_.run();

  auto HX = ctx_.get(savePlaceholder1)->getHandle();
  EXPECT_NEAR(HX.at({0, 0, 0}), 25, 1E-5);
  EXPECT_NEAR(HX.at({0, 0, 1}), 0.01, 1E-5);
  EXPECT_NEAR(HX.at({0, 0, 2}), 9, 1E-5);

  auto HY = ctx_.get(savePlaceholder2)->getHandle();
  EXPECT_NEAR(HY.at({0}), sqrt(2.0), 1E-5);
  EXPECT_NEAR(HY.at({1}), 10, 1E-5);

  auto HZ = ctx_.get(savePlaceholder3)->getHandle();
  EXPECT_NEAR(HZ.at({0}), 4, 1E-5);
  EXPECT_NEAR(HZ.at({1}), 0.01, 1E-5);
}

TEST_P(InterpAndCPU, replaceNaN) {
  auto value = 1.0f;
  auto *X = mod_.createPlaceholder(ElemKind::FloatTy, {6}, "X", false);
  auto XH = ctx_.allocate(X)->getHandle();
  XH = {1, NAN, 2, NAN, 3, NAN};

  auto *RNN = F_->createReplaceNaN("replaceNaN", X, value);

  auto *save = F_->createSave("save", RNN);
  auto *saveTensor = ctx_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);

  EE_.run();

  auto saveH = saveTensor->getHandle();

  for (size_t i = 0; i < 6; i++) {
    if (std::isnan(XH.raw(i))) {
      EXPECT_EQ(saveH.raw(i), value);
    } else {
      EXPECT_EQ(XH.raw(i), saveH.raw(i));
    }
  }
}

TEST_P(InterpAndCPU, log) {
  auto *X = mod_.createPlaceholder(ElemKind::FloatTy, {6}, "X", false);
  auto XH = ctx_.allocate(X)->getHandle();
  XH = {210030, 600, 4, 0.7f, .005f, 0.000829f};

  auto *LN = F_->createLog("log", X);

  auto *save = F_->createSave("save", LN);
  auto *saveTensor = ctx_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);

  EE_.run();

  auto saveH = saveTensor->getHandle();

  for (size_t i = 0; i < 6; i++) {
    EXPECT_NEAR(saveH.at({i}), log(XH.at({i})), 1E-5);
  }
}

TEST_P(InterpAndCPU, CmpEQ) {
  auto *X = mod_.createPlaceholder(ElemKind::Int64ITy, {2, 7}, "X", false);
  ctx_.allocate(X)->getHandle<int64_t>() = {
      0, 1, 17, 876, 1000, 44444, 9999999, 0, 1, 17, 876, 1000, 44444, 9999999};
  auto *Y = mod_.createPlaceholder(ElemKind::Int64ITy, {2, 7}, "Y", false);
  ctx_.allocate(Y)->getHandle<int64_t>() = {
      1, 2, 16, 900, 1111, 44544, 1999999, 0, 1, 17, 876, 1000, 44444, 9999999};

  auto *cmpEQ = F_->createCmpEQ("cmpEQ", X, Y);
  auto *save = F_->createSave("save", cmpEQ);
  auto *saveTensor = ctx_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);

  EE_.run();

  auto saveH = saveTensor->getHandle<int64_t>();
  for (size_t i = 0; i < 7; ++i) {
    EXPECT_FALSE(saveH.at({0, i}));
  }
  for (size_t i = 0; i < 7; ++i) {
    EXPECT_TRUE(saveH.at({1, i}));
  }
}

/// Check that the add operator works properly.
TEST_P(Operator, add) {
  PseudoRNG PRNG;

  auto *inputA =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1}, "A", false);
  ctx_.allocate(inputA)->getHandle<float>().randomize(-3.0, 3.0, PRNG);
  auto *inputB =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1}, "B", false);
  ctx_.allocate(inputB)->getHandle<float>().randomize(-3.0, 3.0, PRNG);
  auto *Pool = F_->createAdd("pool", inputA, inputB);
  auto *S = F_->createSave("save", Pool);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto result = ctx_.get(S->getPlaceholder())->getHandle();
  auto handleA = ctx_.get(inputA)->getHandle();
  auto handleB = ctx_.get(inputB)->getHandle();
  ASSERT_EQ(result.size(), handleA.size());
  for (size_t idx = 0, end = result.size(); idx != end; ++idx) {
    EXPECT_EQ(result.raw(idx), handleA.raw(idx) + handleB.raw(idx));
  }
}

/// Check that the add operator works properly with FP16.
TEST_P(InterpOnly, FP16Add) {
  PseudoRNG PRNG;

  auto *inputA =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 3, 3, 1}, "A", false);
  ctx_.allocate(inputA)->getHandle<float16_t>().randomize(-3.0, 3.0, PRNG);
  auto *inputB =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 3, 3, 1}, "B", false);
  ctx_.allocate(inputB)->getHandle<float16_t>().randomize(-3.0, 3.0, PRNG);
  auto *Pool = F_->createAdd("pool", inputA, inputB);
  auto *S = F_->createSave("save", Pool);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto result = ctx_.get(S->getPlaceholder())->getHandle<float16_t>();
  auto handleA = ctx_.get(inputA)->getHandle<float16_t>();
  auto handleB = ctx_.get(inputB)->getHandle<float16_t>();
  ASSERT_EQ(result.size(), handleA.size());
  for (size_t idx = 0, end = result.size(); idx != end; ++idx) {
    EXPECT_EQ(result.raw(idx), handleA.raw(idx) + handleB.raw(idx));
  }
}

TEST_P(Operator, matmul) {
  auto *lhs = mod_.createPlaceholder(ElemKind::FloatTy, {3, 2}, "lhs", false);
  auto *rhs = mod_.createPlaceholder(ElemKind::FloatTy, {2, 1}, "rhs", false);
  ctx_.allocate(lhs)->getHandle() = {1, 2, 3, 4, 5, 6};
  ctx_.allocate(rhs)->getHandle() = {7, 10};

  auto R = F_->createMatMul("MM", lhs, rhs);

  auto *save = F_->createSave("save", R);
  auto *saveTensor = ctx_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto H = saveTensor->getHandle();
  EXPECT_NEAR(H.at({0, 0}), 27, 0.001);
  EXPECT_NEAR(H.at({1, 0}), 61, 0.001);
  EXPECT_NEAR(H.at({2, 0}), 95, 0.001);
}

/// Check that the matmul operator behaves correctly with FP16.
TEST_P(InterpOnly, FP16Matmul) {
  auto *lhs = mod_.createPlaceholder(ElemKind::Float16Ty, {3, 2}, "lhs", false);
  auto *rhs = mod_.createPlaceholder(ElemKind::Float16Ty, {2, 1}, "rhs", false);
  ctx_.allocate(lhs)->getHandle<float16_t>() = {1, 2, 3, 4, 5, 6};
  ctx_.allocate(rhs)->getHandle<float16_t>() = {7, 10};

  auto R = F_->createMatMul("MM", lhs, rhs);

  auto *save = F_->createSave("save", R);
  auto *saveTensor = ctx_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto H = saveTensor->getHandle<float16_t>();
  EXPECT_NEAR(H.at({0, 0}), 27, 0.001);
  EXPECT_NEAR(H.at({1, 0}), 61, 0.001);
  EXPECT_NEAR(H.at({2, 0}), 95, 0.001);
}

/// Test that the broadcasted batch mat mul operator works as expected.
TEST_P(Operator, BroadcastedBatchMatMul) {
  auto *lhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 2}, "lhs", false);
  auto *rhs = mod_.createPlaceholder(ElemKind::FloatTy, {2, 1}, "rhs", false);
  ctx_.allocate(lhs)->getHandle() = {1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6};
  ctx_.allocate(rhs)->getHandle() = {7, 10};

  auto *R = F_->createBroadcastedBatchMatMul("BMM", lhs, rhs);

  auto *save = F_->createSave("save", R);
  auto *result = ctx_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto H = result->getHandle();
  EXPECT_NEAR(H.at({0, 0, 0}), 27, 0.001);
  EXPECT_NEAR(H.at({0, 1, 0}), 61, 0.001);
  EXPECT_NEAR(H.at({0, 2, 0}), 95, 0.001);
  EXPECT_NEAR(H.at({1, 0, 0}), -27, 0.001);
  EXPECT_NEAR(H.at({1, 1, 0}), -61, 0.001);
  EXPECT_NEAR(H.at({1, 2, 0}), -95, 0.001);
}

TEST_P(Operator, ParallelBatchMatMul) {
  auto *lhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 2}, "lhs", false);
  auto *rhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 1}, "rhs", false);
  ctx_.allocate(lhs)->getHandle() = {1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6};
  ctx_.allocate(rhs)->getHandle() = {7, 10, 12, -1};

  auto *R = F_->createParallelBatchMatMul("BMM", lhs, rhs);

  auto *save = F_->createSave("save", R);
  auto *result = ctx_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto H = result->getHandle();
  EXPECT_NEAR(H.at({0, 0, 0}), 27, 0.001);
  EXPECT_NEAR(H.at({0, 1, 0}), 61, 0.001);
  EXPECT_NEAR(H.at({0, 2, 0}), 95, 0.001);
  EXPECT_NEAR(H.at({1, 0, 0}), -10, 0.001);
  EXPECT_NEAR(H.at({1, 1, 0}), -32, 0.001);
  EXPECT_NEAR(H.at({1, 2, 0}), -54, 0.001);
}

TEST_P(Operator, batchedReduceAdd) {
  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 4}, "batch", false);
  ctx_.allocate(batch)->getHandle() = {10, 20, 30, 40, 1, 2, 3, 4};

  auto R = F_->createBatchedReduceAdd("reduce.add", batch, /* axis */ 0);

  auto *save = F_->createSave("save", R);
  auto *result = ctx_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto H = result->getHandle();
  EXPECT_NEAR(H.at({0}), 11, 0.001);
  EXPECT_NEAR(H.at({1}), 22, 0.001);
  EXPECT_NEAR(H.at({2}), 33, 0.001);
  EXPECT_NEAR(H.at({3}), 44, 0.001);
}

TEST_P(InterpAndCPU, batchedReduceAddWithAxis) {
  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 2}, "batch", false);
  ctx_.allocate(batch)->getHandle() = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  auto R = F_->createBatchedReduceAdd("reduce.add", batch, /* axis */ 1);

  auto *save = F_->createSave("save", R);
  auto *result = ctx_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto H = result->getHandle();
  EXPECT_NEAR(H.at({0, 0}), 6, 0.001);
  EXPECT_NEAR(H.at({0, 1}), 9, 0.001);
  EXPECT_NEAR(H.at({1, 0}), 24, 0.001);
  EXPECT_NEAR(H.at({1, 1}), 27, 0.001);
}

TEST_P(InterpAndCPU, batchedReduceAddQuantized) {
  auto BT = mod_.uniqueType(ElemKind::Int8QTy, {3, 8}, 0.5, 3);
  auto OT = mod_.uniqueType(ElemKind::Int8QTy, {8}, 2.0, -1);

  auto *batch =
      mod_.createPlaceholder(ElemKind::Int8QTy, {3, 8}, BT->getScale(),
                             BT->getOffset(), "batch", false);

  ctx_.allocate(batch)->getHandle<int8_t>() = {
      27, -31, 16,  7,  20, 34, -2, 8,   -10, 83, 29,  -17,
      19, 13,  -11, -9, 50, 58, 0,  -20, -72, 43, -25, -1};

  auto BH = ctx_.get(batch)->getHandle<int8_t>();

  auto *R =
      F_->createBatchedReduceAdd("batched.reduce.add", OT, batch, /* axis */ 0);

  auto *save = F_->createSave("save", R);
  auto OH = ctx_.allocate(save->getPlaceholder())->getHandle<int8_t>();

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  for (size_t i = 0; i < 8; i++) {
    std::array<int32_t, 3> b{{BH.at({0, i}), BH.at({1, i}), BH.at({2, i})}};
    float s = BT->getScale() / OT->getScale();
    int32_t o = BT->getOffset();
    float result = (b[0] - o) + (b[1] - o) + (b[2] - o);
    result = s * result + OT->getOffset();

    EXPECT_NEAR(std::round(result), OH.at({i}), 1.0);
  }
}

TEST_P(InterpAndCPU, batchedReduceAddQuantizedWithAxis) {
  auto BT = mod_.uniqueType(ElemKind::Int8QTy, {2, 3, 4}, 0.5, 3);
  auto OT = mod_.uniqueType(ElemKind::Int8QTy, {2, 4}, 2.0, -1);

  auto *batch =
      mod_.createPlaceholder(ElemKind::Int8QTy, {2, 3, 4}, BT->getScale(),
                             BT->getOffset(), "batch", false);

  ctx_.allocate(batch)->getHandle<int8_t>() = {
      27, -31, 16,  7,  20, 34, -2, 8,   -10, 83, 29,  -17,
      19, 13,  -11, -9, 50, 58, 0,  -20, -72, 43, -25, -1};

  auto BH = ctx_.get(batch)->getHandle<int8_t>();

  auto *R =
      F_->createBatchedReduceAdd("batched.reduce.add", OT, batch, /* axis */ 1);
  auto *save = F_->createSave("save", R);
  auto OH = ctx_.allocate(save->getPlaceholder())->getHandle<int8_t>();

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 4; j++) {
      std::array<int32_t, 3> b{
          {BH.at({i, 0, j}), BH.at({i, 1, j}), BH.at({i, 2, j})}};
      float s = BT->getScale() / OT->getScale();
      int32_t o = BT->getOffset();
      float result = (b[0] - o) + (b[1] - o) + (b[2] - o);
      result = s * result + OT->getOffset();

      EXPECT_NEAR(std::round(result), OH.at({i, j}), 1.0);
    }
  }
}

TEST_P(Operator, batchedReduceMean) {
  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 4}, "batch", false);
  ctx_.allocate(batch)->getHandle() = {10, 20, 30, 40, 1, 2, 3, 4};

  auto R = F_->createBatchedReduceMean("reduce.add", batch, /* axis */ 0);

  auto *save = F_->createSave("save", R);
  auto *result = ctx_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto H = result->getHandle();
  EXPECT_NEAR(H.at({0}), 5.5, 0.001);
  EXPECT_NEAR(H.at({1}), 11.0, 0.001);
  EXPECT_NEAR(H.at({2}), 16.5, 0.001);
  EXPECT_NEAR(H.at({3}), 22.0, 0.001);
}

TEST_P(InterpAndCPU, batchedReduceMeanWithAxis) {
  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 2}, "batch", false);
  ctx_.allocate(batch)->getHandle() = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  auto R = F_->createBatchedReduceMean("reduce.add", batch, /* axis */ 1);

  auto *save = F_->createSave("save", R);
  auto *result = ctx_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto H = result->getHandle();
  EXPECT_NEAR(H.at({0, 0}), 2.0, 0.001);
  EXPECT_NEAR(H.at({0, 1}), 3.0, 0.001);
  EXPECT_NEAR(H.at({1, 0}), 8.0, 0.001);
  EXPECT_NEAR(H.at({1, 1}), 9.0, 0.001);
}

TEST_P(InterpAndCPU, batchedReduceMeanQuantized) {
  auto BT = mod_.uniqueType(ElemKind::Int8QTy, {3, 8}, 0.5, 3);
  auto OT = mod_.uniqueType(ElemKind::Int8QTy, {8}, 2.0, -1);

  auto *batch =
      mod_.createPlaceholder(ElemKind::Int8QTy, {3, 8}, BT->getScale(),
                             BT->getOffset(), "batch", false);

  ctx_.allocate(batch)->getHandle<int8_t>() = {
      27, -31, 16,  7,  20, 34, -2, 8,   -10, 83, 29,  -17,
      19, 13,  -11, -9, 50, 58, 0,  -20, -72, 43, -25, -1};

  auto BH = ctx_.get(batch)->getHandle<int8_t>();

  auto *R = F_->createBatchedReduceMean("batched.reduce.add", OT, batch,
                                        /* axis */ 0);

  auto *save = F_->createSave("save", R);
  auto OH = ctx_.allocate(save->getPlaceholder())->getHandle<int8_t>();

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  for (size_t i = 0; i < 8; i++) {
    std::array<int32_t, 3> b{{BH.at({0, i}), BH.at({1, i}), BH.at({2, i})}};
    float s = BT->getScale() / OT->getScale();
    int32_t o = BT->getOffset();
    float result = ((b[0] - o) + (b[1] - o) + (b[2] - o)) / 3;
    result = s * result + OT->getOffset();

    EXPECT_NEAR(std::round(result), OH.at({i}), 1.0);
  }
}

TEST_P(InterpAndCPU, batchedReduceMeanQuantizedWithAxis) {
  auto BT = mod_.uniqueType(ElemKind::Int8QTy, {2, 3, 4}, 0.5, 3);
  auto OT = mod_.uniqueType(ElemKind::Int8QTy, {2, 4}, 2.0, -1);

  auto *batch =
      mod_.createPlaceholder(ElemKind::Int8QTy, {2, 3, 4}, BT->getScale(),
                             BT->getOffset(), "batch", false);

  ctx_.allocate(batch)->getHandle<int8_t>() = {
      27, -31, 16,  7,  20, 34, -2, 8,   -10, 83, 29,  -17,
      19, 13,  -11, -9, 50, 58, 0,  -20, -72, 43, -25, -1};

  auto BH = ctx_.get(batch)->getHandle<int8_t>();

  auto *R = F_->createBatchedReduceMean("batched.reduce.add", OT, batch,
                                        /* axis */ 1);
  auto *save = F_->createSave("save", R);
  auto OH = ctx_.allocate(save->getPlaceholder())->getHandle<int8_t>();

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 4; j++) {
      std::array<int32_t, 3> b{
          {BH.at({i, 0, j}), BH.at({i, 1, j}), BH.at({i, 2, j})}};
      float s = BT->getScale() / OT->getScale();
      int32_t o = BT->getOffset();
      float result = ((b[0] - o) + (b[1] - o) + (b[2] - o)) / 3;
      result = s * result + OT->getOffset();

      EXPECT_NEAR(std::round(result), OH.at({i, j}), 1.0);
    }
  }
}

/// Test that the BatchedAdd operator works.
TEST_P(Operator, BatchedAdd) {
  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 3}, "batch", false);
  auto *added =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 3}, "added", false);

  ctx_.allocate(batch)->getHandle() = {9, 8, 7, 6, 5,  4,  3,  4,  5,
                                       6, 7, 8, 9, 10, 11, 12, 13, 14};
  ctx_.allocate(added)->getHandle().clear(1.0);

  auto R = F_->createBatchedAdd("batch.add", batch, added);
  auto *save = F_->createSave("save", R);
  auto *result = ctx_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto BH = ctx_.get(batch)->getHandle();
  auto RH = result->getHandle();
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 3; j++) {
      for (size_t k = 0; k < 3; k++) {
        EXPECT_NEAR(RH.at({i, j, k}), BH.at({i, j, k}) + 1.0, 0.001);
      }
    }
  }
}

/// Broadcast Tensor of shape (2,1,1) to (2,4,2) with axis 0.
TEST_P(InterpAndCPU, broadcastSimple) {
  const size_t numDims_A = 3;
  const size_t dimY_A = 2;
  const size_t dimZ_A = 4;
  const size_t dimW_A = 2;
  const size_t dims_A[numDims_A] = {dimY_A, dimZ_A, dimW_A};

  const size_t numDims_B = 3;
  const size_t dimY_B = 2;
  const size_t dimZ_B = 1;
  const size_t dimW_B = 1;
  const size_t dims_B[numDims_B] = {dimY_B, dimZ_B, dimW_B};

  auto *B = mod_.createPlaceholder(ElemKind::FloatTy, dims_B, "B", false);
  auto *QB =
      mod_.createPlaceholder(ElemKind::Int8QTy, dims_B, 1.1, -2, "QB", false);
  auto H_B = ctx_.allocate(B)->getHandle();
  auto H_QB = ctx_.allocate(QB)->getHandle<int8_t>();
  H_B = {20, 10};
  H_QB = {35, -18};

  const unsigned axis = 0;

  auto R = F_->createBroadcast("broadcasted", B, dims_A, axis);
  auto QR = F_->createBroadcast("broadcastedQ", QB, dims_A, axis);

  auto *save = F_->createSave("save", R);
  auto *broadcasted = ctx_.allocate(save->getPlaceholder());

  auto *saveQ = F_->createSave("saveQ", QR);
  auto *broadcastedQ = ctx_.allocate(saveQ->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto broadcastedBHandle = broadcasted->getHandle();
  auto broadcastedQBHandle = broadcastedQ->getHandle<int8_t>();
  // Verify broadcasted B has same shape.
  EXPECT_EQ(broadcastedBHandle.dims().size(), numDims_A);
  EXPECT_EQ(broadcastedQBHandle.dims().size(), numDims_A);
  for (size_t i = 0; i < broadcastedBHandle.dims().size(); i++) {
    EXPECT_EQ(broadcastedBHandle.dims()[i], dims_A[i]);
    EXPECT_EQ(broadcastedQBHandle.dims()[i], dims_A[i]);
  }

  // Look at the two values in X_B and verify in the three dimensions it was
  // broadcasted that the values were correctly broadcasted.
  const size_t k_B = 0;
  const size_t l_B = 0;
  for (size_t j_B = 0; j_B < dimY_B; ++j_B) {
    const float origVal = H_B.at({j_B, k_B, l_B});
    const int8_t origValQ = H_QB.at({j_B, k_B, l_B});
    const size_t j_A = j_B; // This dim was not broadcasted (dims were equal).
    for (size_t k_A = 0; k_A < dimZ_A; k_A++) {
      for (size_t l_A = 0; l_A < dimW_A; l_A++) {
        EXPECT_EQ(broadcastedBHandle.at({j_A, k_A, l_A}), origVal);
        EXPECT_EQ(broadcastedQBHandle.at({j_A, k_A, l_A}), origValQ);
      }
    }
  }
}

/// Broadcast a Tensor of shape (2,1) to (3,2,4,2) with axis 1.
TEST_P(InterpAndCPU, broadcast) {
  const size_t numDims_A = 4;
  const size_t dimX_A = 3;
  const size_t dimY_A = 2;
  const size_t dimZ_A = 4;
  const size_t dimW_A = 2;
  const size_t dims_A[numDims_A] = {dimX_A, dimY_A, dimZ_A, dimW_A};

  const size_t numDims_B = 2;
  const size_t dimY_B = 2;
  const size_t dimZ_B = 1;
  const size_t dims_B[numDims_B] = {dimY_B, dimZ_B};

  auto *B = mod_.createPlaceholder(ElemKind::FloatTy, dims_B, "B", false);
  auto *QB =
      mod_.createPlaceholder(ElemKind::Int8QTy, dims_B, 0.8, 3, "QB", false);

  auto H_B = ctx_.allocate(B)->getHandle();
  auto H_QB = ctx_.allocate(QB)->getHandle<int8_t>();
  H_B = {20, 10};
  H_QB = {-8, 41};

  const unsigned axis = 1;

  auto R = F_->createBroadcast("broadcasted", B, dims_A, axis);
  auto QR = F_->createBroadcast("broadcastedQ", QB, dims_A, axis);

  auto *save = F_->createSave("save", R);
  auto *broadcasted = ctx_.allocate(save->getPlaceholder());

  auto *saveQ = F_->createSave("saveQ", QR);
  auto *broadcastedQ = ctx_.allocate(saveQ->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto broadcastedBHandle = broadcasted->getHandle();
  auto broadcastedQBHandle = broadcastedQ->getHandle<int8_t>();
  // Verify broadcasted B has same shape.
  EXPECT_EQ(broadcastedBHandle.dims().size(), numDims_A);
  EXPECT_EQ(broadcastedQBHandle.dims().size(), numDims_A);
  for (size_t i = 0; i < broadcastedBHandle.dims().size(); i++) {
    EXPECT_EQ(broadcastedBHandle.dims()[i], dims_A[i]);
    EXPECT_EQ(broadcastedQBHandle.dims()[i], dims_A[i]);
  }

  // Look at the two values in X_B and verify in the three dimensions it was
  // broadcasted that the values were correctly broadcasted.
  const size_t k_B = 0;
  for (size_t j_B = 0; j_B < dimY_B; ++j_B) {
    const float origVal = H_B.at({j_B, k_B});
    const int8_t origValQ = H_QB.at({j_B, k_B});
    const size_t j_A = j_B; // This dim was not broadcasted (dims were equal).
    for (size_t i_A = 0; i_A < dimX_A; i_A++) {
      for (size_t k_A = 0; k_A < dimZ_A; k_A++) {
        for (size_t l_A = 0; l_A < dimW_A; l_A++) {
          EXPECT_EQ(broadcastedBHandle.at({i_A, j_A, k_A, l_A}), origVal);
          EXPECT_EQ(broadcastedQBHandle.at({i_A, j_A, k_A, l_A}), origValQ);
        }
      }
    }
  }
}

/// Perform a simple weighted sum.
TEST_P(Operator, weightedSum) {
  // Create the data.
  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "A", false);
  ctx_.allocate(A)->getHandle() = {1.0, 2.0, 3.0, 4.0};

  auto *B = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "B", false);
  ctx_.allocate(B)->getHandle() = {5.0, 6.0, 7.0, 8.0};

  // Create the weights.
  auto *AW = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "AW", false);
  ctx_.allocate(AW)->getHandle() = {0.1f};

  auto *BW = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "BW", false);
  ctx_.allocate(BW)->getHandle() = {10.0f};

  // Create the weighted sum with the data and weights, and save it.
  auto *WS = F_->createWeightedSum("ws", {A, B}, {AW, BW});
  auto *save = F_->createSave("save", WS);
  auto *saveTensor = ctx_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  // Verify the weighted sum was correctly calculated.
  auto resultH = saveTensor->getHandle();
  EXPECT_NEAR(resultH.at({0, 0}), 50.1, 1E-5);
  EXPECT_NEAR(resultH.at({0, 1}), 60.2, 1E-5);
  EXPECT_NEAR(resultH.at({1, 0}), 70.3, 1E-5);
  EXPECT_NEAR(resultH.at({1, 1}), 80.4, 1E-5);
}

TEST_P(Operator, minElem) {
  PseudoRNG PRNG;
  unsigned len = 5;

  auto *LHS = mod_.createPlaceholder(ElemKind::FloatTy, {len}, "lhs", false);
  auto *RHS = mod_.createPlaceholder(ElemKind::FloatTy, {len}, "rhs", false);
  auto *min = F_->createMin("min", LHS, RHS);
  auto *save = F_->createSave("min", min);
  auto *result = ctx_.allocate(save->getPlaceholder());

  ctx_.allocate(LHS)->getHandle().randomize(-10, 10, PRNG);
  ctx_.allocate(RHS)->getHandle().randomize(-10, 10, PRNG);

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto resultH = result->getHandle();
  auto LHSH = ctx_.get(LHS)->getHandle();
  auto RHSH = ctx_.get(RHS)->getHandle();

  for (size_t i = 0; i < len; i++) {
    EXPECT_EQ(resultH.raw(i), std::min(LHSH.raw(i), RHSH.raw(i)));
  }
}

/// Verify that the Max operator works correctly.
TEST_P(Operator, maxElem) {
  PseudoRNG PRNG;
  unsigned len = 5;

  auto *LHS = mod_.createPlaceholder(ElemKind::FloatTy, {len}, "lhs", false);
  auto *RHS = mod_.createPlaceholder(ElemKind::FloatTy, {len}, "rhs", false);
  auto *max = F_->createMax("max", LHS, RHS);
  auto *save = F_->createSave("max", max);
  auto *result = ctx_.allocate(save->getPlaceholder());

  ctx_.allocate(LHS)->getHandle().randomize(-10, 10, PRNG);
  ctx_.allocate(RHS)->getHandle().randomize(-10, 10, PRNG);

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto resultH = result->getHandle();
  auto LHSH = ctx_.get(LHS)->getHandle();
  auto RHSH = ctx_.get(RHS)->getHandle();

  for (size_t i = 0; i < len; i++) {
    EXPECT_EQ(resultH.raw(i), std::max(LHSH.raw(i), RHSH.raw(i)));
  }
}

/// Verify that the RELU operator works correctly.
TEST_P(Operator, ReluSimple) {
  auto *in = mod_.createPlaceholder(ElemKind::FloatTy, {7}, "in", false);
  auto *relu = F_->createRELU("relu", in);
  auto *save = F_->createSave("relu", relu);
  auto *result = ctx_.allocate(save->getPlaceholder());

  ctx_.allocate(in)->getHandle() = {0, -1, -2, -3, 4, 5, 6};

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto resultH = result->getHandle();

  for (size_t i = 0; i < 7; i++) {
    if (i < 4) {
      EXPECT_EQ(resultH.raw(i), 0);
    } else {
      EXPECT_EQ(resultH.raw(i), i);
    }
  }
}

TEST_P(Operator, TopK) {
  auto *inp =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 1, 5}, "input", false);
  auto *values =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 1, 3}, "values", false);
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {3, 1, 3}, "indices", false);

  ctx_.allocate(inp)->getHandle() = {
      28, 4, 411, 19, 42, 0.4f, 0.4f, 0.4f, -0.4f, 0.45f, 7, 5, 9, 8, 100,
  };
  ctx_.allocate(values);
  ctx_.allocate(indices);

  auto R = F_->createTopK("TopK", inp, 3);

  F_->createSave("save.values", {R, 0}, values);
  F_->createSave("save.indices", {R, 1}, indices);

  EE_.compile(CompilationMode::Infer, F_, ctx_);

  EE_.run();

  auto V = ctx_.get(values)->getHandle();
  auto I = ctx_.get(indices)->getHandle<int64_t>();

  EXPECT_FLOAT_EQ(V.at({0, 0, 0}), 411);
  EXPECT_EQ(I.at({0, 0, 0}), 2);
  EXPECT_FLOAT_EQ(V.at({0, 0, 1}), 42);
  EXPECT_EQ(I.at({0, 0, 1}), 4);
  EXPECT_FLOAT_EQ(V.at({0, 0, 2}), 28);
  EXPECT_EQ(I.at({0, 0, 2}), 0);

  EXPECT_FLOAT_EQ(V.at({1, 0, 0}), 0.45);
  EXPECT_EQ(I.at({1, 0, 0}), 4);
  EXPECT_FLOAT_EQ(V.at({1, 0, 1}), 0.4);
  EXPECT_EQ(I.at({1, 0, 1}), 0);
  EXPECT_FLOAT_EQ(V.at({1, 0, 2}), 0.4);
  EXPECT_EQ(I.at({1, 0, 2}), 1);

  EXPECT_FLOAT_EQ(V.at({2, 0, 0}), 100);
  EXPECT_EQ(I.at({2, 0, 0}), 4);
  EXPECT_FLOAT_EQ(V.at({2, 0, 1}), 9);
  EXPECT_EQ(I.at({2, 0, 1}), 2);
  EXPECT_FLOAT_EQ(V.at({2, 0, 2}), 8);
  EXPECT_EQ(I.at({2, 0, 2}), 3);
}

// Check that concatenating Nodes with multiple outputs works correctly.
TEST_P(InterpAndCPU, ConcatTopK) {
  auto *inp1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 1, 3}, "input", false);
  auto *inp2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 1, 3}, "input", false);
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {4, 1, 2}, "indices", false);

  ctx_.allocate(inp1)->getHandle() = {1, 2, 3, 17.4f, -0.1f, -10.1f};
  ctx_.allocate(inp2)->getHandle() = {1, 2, -3, -17.4f, -0.1f, -10.1f};

  auto *R1 = F_->createTopK("TopK1", inp1, 2);
  auto *R2 = F_->createTopK("TopK2", inp2, 2);

  // Concat the values and indices separately, both on the 0th dimension,
  // matching the shapes of the values and indices variables above.
  auto *CV =
      F_->createConcat("Concat.Values", {R1->getValues(), R2->getValues()}, 0);
  auto *CI = F_->createConcat("Concat.Indices",
                              {R1->getIndices(), R2->getIndices()}, 0);

  auto *saveValues = F_->createSave("Save.Values", CV);
  auto *saveValuesTensor = ctx_.allocate(saveValues->getPlaceholder());

  auto *saveIndices = F_->createSave("Save.Indices", CI, indices);
  auto *saveIndicesTensor = ctx_.allocate(saveIndices->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);

  EE_.run();

  auto V = saveValuesTensor->getHandle();
  auto I = saveIndicesTensor->getHandle<int64_t>();

  EXPECT_FLOAT_EQ(V.at({0, 0, 0}), 3);
  EXPECT_FLOAT_EQ(I.at({0, 0, 0}), 2);
  EXPECT_FLOAT_EQ(V.at({0, 0, 1}), 2);
  EXPECT_FLOAT_EQ(I.at({0, 0, 1}), 1);

  EXPECT_FLOAT_EQ(V.at({1, 0, 0}), 17.4f);
  EXPECT_FLOAT_EQ(I.at({1, 0, 0}), 0);
  EXPECT_FLOAT_EQ(V.at({1, 0, 1}), -0.1f);
  EXPECT_FLOAT_EQ(I.at({1, 0, 1}), 1);

  EXPECT_FLOAT_EQ(V.at({2, 0, 0}), 2);
  EXPECT_FLOAT_EQ(I.at({2, 0, 0}), 1);
  EXPECT_FLOAT_EQ(V.at({2, 0, 1}), 1);
  EXPECT_FLOAT_EQ(I.at({2, 0, 1}), 0);

  EXPECT_FLOAT_EQ(V.at({3, 0, 0}), -0.1f);
  EXPECT_FLOAT_EQ(I.at({3, 0, 0}), 1);
  EXPECT_FLOAT_EQ(V.at({3, 0, 1}), -10.1f);
  EXPECT_FLOAT_EQ(I.at({3, 0, 1}), 2);
}

// Check that matrix multiplication works well on some predefined values.
TEST_P(Operator, matMul) {
  auto *inp0 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2}, "input0", false);
  auto *inp1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2}, "input1", false);
  auto *inp2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2}, "input2", false);
  auto *rot = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "rot", false);

  float deg = 45.0 / 180.0 * 3.1415926;
  // Use the rotation matrix to manipulate some values.
  // https://en.wikipedia.org/wiki/Rotation_matrix
  ctx_.allocate(rot)->getHandle() = {
      cosf(deg),
      -sinf(deg),
      sinf(deg),
      cosf(deg),
  };

  // Some test vectors.
  ctx_.allocate(inp0)->getHandle() = {1, 4};
  ctx_.allocate(inp1)->getHandle() = {14, 2};
  ctx_.allocate(inp2)->getHandle() = {5, 2};

  auto *A0 = F_->createMatMul("m0", inp0, rot);
  auto *A1 = F_->createMatMul("m1", inp1, rot);
  auto *A2 = F_->createMatMul("m2", inp2, rot);

  auto *res0 = F_->createSave("save.values", A0);
  ctx_.allocate(res0->getPlaceholder());
  auto *res1 = F_->createSave("save.values", A1);
  ctx_.allocate(res1->getPlaceholder());
  auto *res2 = F_->createSave("save.values", A2);
  ctx_.allocate(res2->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);

  EE_.run();

  auto R0 = ctx_.get(res0->getPlaceholder())->getHandle();
  auto R1 = ctx_.get(res1->getPlaceholder())->getHandle();
  auto R2 = ctx_.get(res2->getPlaceholder())->getHandle();

  EXPECT_FLOAT_EQ(R0.at({0, 0}), 3.5355339);
  EXPECT_FLOAT_EQ(R0.at({0, 1}), 2.1213205);
  EXPECT_FLOAT_EQ(R1.at({0, 0}), 11.313709);
  EXPECT_FLOAT_EQ(R1.at({0, 1}), -8.485281);
  EXPECT_FLOAT_EQ(R2.at({0, 0}), 4.9497476);
  EXPECT_FLOAT_EQ(R2.at({0, 1}), -2.1213202);
}

// Check the TopK operator for the special case of K=1.
TEST_P(InterpAndCPU, TopK1) {
  auto *inp =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 1, 5}, "input", false);

  ctx_.allocate(inp)->getHandle() = {
      0, 18, 7, 16, 5, 14, 33, 2, 41, 0, 1, -23, 34, 4, -5,
  };

  auto R = F_->createTopK("TopK", inp, 1);

  auto *values = F_->createSave("save.values", {R, 0});
  ctx_.allocate(values->getPlaceholder());

  auto *indices = F_->createSave("save.indices", {R, 1});
  ctx_.allocate(indices->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto V = ctx_.get(values->getPlaceholder())->getHandle();
  auto I = ctx_.get(indices->getPlaceholder())->getHandle<int64_t>();

  EXPECT_FLOAT_EQ(V.at({0, 0, 0}), 18);
  EXPECT_EQ(I.at({0, 0, 0}), 1);
  EXPECT_FLOAT_EQ(V.at({1, 0, 0}), 41);
  EXPECT_EQ(I.at({1, 0, 0}), 3);
  EXPECT_FLOAT_EQ(V.at({2, 0, 0}), 34);
  EXPECT_EQ(I.at({2, 0, 0}), 2);
}

TEST_P(InterpAndCPU, QuantizedTopK) {
  auto *INV = mod_.createPlaceholder(ElemKind::Int8QTy, {3, 1, 5}, 1.2, 5,
                                     "input", false);
  ctx_.allocate(INV)->getHandle<int8_t>() = {
      -12, -28, -7, 8, -93, 0, 10, 3, -1, 10, -2, 3, -2, 3, 3,
  };

  auto TK = F_->createTopK("TopK", INV, 3);

  auto *values = F_->createSave("save.values", TK->getValues());
  ctx_.allocate(values->getPlaceholder());
  auto *indices = F_->createSave("save.indices", TK->getIndices());
  ctx_.allocate(indices->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto VH = ctx_.get(values->getPlaceholder())->getHandle<int8_t>();
  auto IH = ctx_.get(indices->getPlaceholder())->getHandle<int64_t>();

  EXPECT_EQ(VH.at({0, 0, 0}), 8);
  EXPECT_EQ(IH.at({0, 0, 0}), 3);
  EXPECT_EQ(VH.at({0, 0, 1}), -7);
  EXPECT_EQ(IH.at({0, 0, 1}), 2);
  EXPECT_EQ(VH.at({0, 0, 2}), -12);
  EXPECT_EQ(IH.at({0, 0, 2}), 0);

  EXPECT_EQ(VH.at({1, 0, 0}), 10);
  EXPECT_EQ(IH.at({1, 0, 0}), 1);
  EXPECT_EQ(VH.at({1, 0, 1}), 10);
  EXPECT_EQ(IH.at({1, 0, 1}), 4);
  EXPECT_EQ(VH.at({1, 0, 2}), 3);
  EXPECT_EQ(IH.at({1, 0, 2}), 2);

  EXPECT_EQ(VH.at({2, 0, 0}), 3);
  EXPECT_EQ(IH.at({2, 0, 0}), 1);
  EXPECT_EQ(VH.at({2, 0, 1}), 3);
  EXPECT_EQ(IH.at({2, 0, 1}), 3);
  EXPECT_EQ(VH.at({2, 0, 2}), 3);
  EXPECT_EQ(IH.at({2, 0, 2}), 4);
}

TEST_P(Operator, Gather) {
  /*
    DATA  = [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]
    INDICES = [
        [0, 1, 0, 1],
        [1, 2, 2, 0],
    ]
    OUTPUT = [
        [
            [1.0, 1.2],
            [2.3, 3.4],
            [1.0, 1.2],
            [2.3, 3.4],
        ],
        [
            [2.3, 3.4],
            [4.5, 5.7],
            [4.5, 5.7],
            [1.0, 1.2],
        ],
    ]
  */
  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {3, 2}, "data", false);
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {2, 4}, "indices", false);

  ctx_.allocate(data)->getHandle() = {
      1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f,
  };
  ctx_.allocate(indices)->getHandle<int64_t>() = {
      0, 1, 0, 1, 1, 2, 2, 0,
  };

  auto R = F_->createGather("gather", data, indices);

  auto *result = F_->createSave("save", R);
  ctx_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto H = ctx_.get(result->getPlaceholder())->getHandle();

  EXPECT_FLOAT_EQ(H.at({0, 0, 0}), 1.0);
  EXPECT_FLOAT_EQ(H.at({0, 0, 1}), 1.2);
  EXPECT_FLOAT_EQ(H.at({0, 1, 0}), 2.3);
  EXPECT_FLOAT_EQ(H.at({0, 1, 1}), 3.4);
  EXPECT_FLOAT_EQ(H.at({0, 2, 0}), 1.0);
  EXPECT_FLOAT_EQ(H.at({0, 2, 1}), 1.2);
  EXPECT_FLOAT_EQ(H.at({0, 3, 0}), 2.3);
  EXPECT_FLOAT_EQ(H.at({0, 3, 1}), 3.4);

  EXPECT_FLOAT_EQ(H.at({1, 0, 0}), 2.3);
  EXPECT_FLOAT_EQ(H.at({1, 0, 1}), 3.4);
  EXPECT_FLOAT_EQ(H.at({1, 1, 0}), 4.5);
  EXPECT_FLOAT_EQ(H.at({1, 1, 1}), 5.7);
  EXPECT_FLOAT_EQ(H.at({1, 2, 0}), 4.5);
  EXPECT_FLOAT_EQ(H.at({1, 2, 1}), 5.7);
  EXPECT_FLOAT_EQ(H.at({1, 3, 0}), 1.0);
  EXPECT_FLOAT_EQ(H.at({1, 3, 1}), 1.2);
}

/// Check if the code generation of transposes
/// is correct for tensors with 2 dimensions.
/// Note: This assumes that Tensor::transpose is correct.
TEST_P(Operator, Transpose2Dims) {
  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {20, 13}, "A", false);
  ctx_.allocate(A)->getHandle().randomize(-3.0, 3.0, mod_.getPRNG());

  auto *tr = F_->createTranspose("tr", A, {1, 0});
  auto *result = F_->createSave("saveTranspose", tr);
  ctx_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  Tensor dest(ElemKind::FloatTy, {13, 20});
  ctx_.get(A)->transpose(&dest, {1, 0});
  EXPECT_TRUE(ctx_.get(result->getPlaceholder())->isEqual(dest));
}

/// Check that transpose is supported for FP16.
TEST_P(InterpOnly, FP16Transpose2Dims) {
  auto *A = mod_.createPlaceholder(ElemKind::Float16Ty, {20, 13}, "A", false);
  ctx_.allocate(A)->getHandle<float16_t>().randomize(-3.0, 3.0, mod_.getPRNG());

  auto *tr = F_->createTranspose("tr", A, {1, 0});
  auto *result = F_->createSave("saveTranspose", tr);
  ctx_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  Tensor dest(ElemKind::Float16Ty, {13, 20});
  ctx_.get(A)->transpose(&dest, {1, 0});
  EXPECT_TRUE(ctx_.get(result->getPlaceholder())->isEqual(dest));
}

/// Check if the code generation of transposes
/// is correct for tensors with 3 dimensions.
/// Note: This assumes that Tensor::transpose is correct.
TEST_P(Operator, Transpose3Dims) {
  constexpr size_t dims[] = {20, 13, 7};
  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, dims, "A", false);
  ctx_.allocate(A)->getHandle().randomize(-3.0, 3.0, mod_.getPRNG());

  int nbOfShuffle = 0;
  SaveNode *savedTransposes[6];
  unsigned_t shuffles[6][3];

  for (unsigned_t i = 0; i < 3; ++i) {
    for (unsigned_t j = 0; j < 3; ++j) {
      if (j == i) {
        continue;
      }
      for (unsigned_t k = 0; k < 3; ++k) {
        if (k == j || k == i) {
          continue;
        }
        shuffles[nbOfShuffle][0] = i;
        shuffles[nbOfShuffle][1] = j;
        shuffles[nbOfShuffle][2] = k;
        auto *tr = F_->createTranspose("tr", A, shuffles[nbOfShuffle]);
        savedTransposes[nbOfShuffle] = F_->createSave("saveTranspose", tr);
        ctx_.allocate(savedTransposes[nbOfShuffle]->getPlaceholder());
        ++nbOfShuffle;
      }
    }
  }

  // We should have exactly 6 possible permutations for 3 dimensions.
  EXPECT_EQ(6, nbOfShuffle);

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  for (int i = 0; i < 6; ++i) {
    Tensor dest(ElemKind::FloatTy, {dims[shuffles[i][0]], dims[shuffles[i][1]],
                                    dims[shuffles[i][2]]});
    ctx_.get(A)->transpose(&dest, shuffles[i]);
    EXPECT_TRUE(ctx_.get(savedTransposes[i]->getPlaceholder())->isEqual(dest));
  }
}

/// Check that gather on Int64ITy/size_t works.
TEST_P(InterpAndCPU, GatherSizeT) {
  /*
    DATA  = [
        [1, 2],
        [3, 4],
        [5, 6],
    ]
    INDICES = [
        [0, 1, 0, 1],
        [1, 2, 2, 0],
    ]
    OUTPUT = [
        [
            [1, 2],
            [3, 4],
            [1, 2],
            [3, 4],
        ],
        [
            [3, 4],
            [5, 6],
            [5, 6],
            [1, 2],
        ],
    ]
  */
  auto *data =
      mod_.createPlaceholder(ElemKind::Int64ITy, {3, 2}, "data", false);
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {2, 4}, "indices", false);

  ctx_.allocate(data)->getHandle<int64_t>() = {
      1, 2, 3, 4, 5, 6,
  };
  ctx_.allocate(indices)->getHandle<int64_t>() = {
      0, 1, 0, 1, 1, 2, 2, 0,
  };

  auto R = F_->createGather("gather", data, indices);

  auto *result = F_->createSave("save", R);
  ctx_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto H = ctx_.get(result->getPlaceholder())->getHandle<int64_t>();

  EXPECT_EQ(H.at({0, 0, 0}), 1);
  EXPECT_EQ(H.at({0, 0, 1}), 2);
  EXPECT_EQ(H.at({0, 1, 0}), 3);
  EXPECT_EQ(H.at({0, 1, 1}), 4);
  EXPECT_EQ(H.at({0, 2, 0}), 1);
  EXPECT_EQ(H.at({0, 2, 1}), 2);
  EXPECT_EQ(H.at({0, 3, 0}), 3);
  EXPECT_EQ(H.at({0, 3, 1}), 4);

  EXPECT_EQ(H.at({1, 0, 0}), 3);
  EXPECT_EQ(H.at({1, 0, 1}), 4);
  EXPECT_EQ(H.at({1, 1, 0}), 5);
  EXPECT_EQ(H.at({1, 1, 1}), 6);
  EXPECT_EQ(H.at({1, 2, 0}), 5);
  EXPECT_EQ(H.at({1, 2, 1}), 6);
  EXPECT_EQ(H.at({1, 3, 0}), 1);
  EXPECT_EQ(H.at({1, 3, 1}), 2);
}

TEST_P(Operator, BatchedGather) {
  /*
   DATA  = [
    [1.0, 1.2, 2.4, 4.5],
    [2.3, 3.4, 3.6, 2.3],
    [4.5, 5.7, 1.2, 4.5],
   ]

   INDICES = [0, 2],

   OUTPUT = [
    [1.0, 2.4],
    [2.3, 3.6],
    [4.5, 1.2],
   ]
   */
  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {3, 4}, "data", false);
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {2}, "indices", false);

  ctx_.allocate(data)->getHandle() = {
      1.0f, 1.2f, 2.4f, 4.5f, 2.3f, 3.4f, 3.6f, 2.3f, 4.5f, 5.7f, 1.2f, 4.5f,
  };
  ctx_.allocate(indices)->getHandle<int64_t>() = {
      0,
      2,
  };

  // Create a batched gather (a single batch dimension).
  auto R = F_->createGather("gather", data, indices, 1);

  auto *result = F_->createSave("save", R);
  ctx_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto H = ctx_.get(result->getPlaceholder())->getHandle();
  EXPECT_FLOAT_EQ(H.at({0, 0}), 1.0);
  EXPECT_FLOAT_EQ(H.at({0, 1}), 2.4);
  EXPECT_FLOAT_EQ(H.at({1, 0}), 2.3);
  EXPECT_FLOAT_EQ(H.at({1, 1}), 3.6);
  EXPECT_FLOAT_EQ(H.at({2, 0}), 4.5);
  EXPECT_FLOAT_EQ(H.at({2, 1}), 1.2);
}

TEST_P(Operator, ScatterAssign) {
  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {5, 2}, "data", false);
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {2}, "indices", false);
  auto *slices =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "slices", false);

  ctx_.allocate(data)->getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  ctx_.allocate(indices)->getHandle<int64_t>() = {1, 3};
  ctx_.allocate(slices)->getHandle() = {-3, -4, -7, -8};

  auto R = F_->createScatterAssign("scatterassign", data, indices, slices);

  auto *result = F_->createSave("save", R);
  ctx_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto H = ctx_.get(result->getPlaceholder())->getHandle();

  EXPECT_FLOAT_EQ(H.at({0, 0}), 1.0);
  EXPECT_FLOAT_EQ(H.at({0, 1}), 2.0);
  EXPECT_FLOAT_EQ(H.at({1, 0}), -3.0);
  EXPECT_FLOAT_EQ(H.at({1, 1}), -4.0);
  EXPECT_FLOAT_EQ(H.at({2, 0}), 5.0);
  EXPECT_FLOAT_EQ(H.at({2, 1}), 6.0);
  EXPECT_FLOAT_EQ(H.at({3, 0}), -7.0);
  EXPECT_FLOAT_EQ(H.at({3, 1}), -8.0);
  EXPECT_FLOAT_EQ(H.at({4, 0}), 9.0);
  EXPECT_FLOAT_EQ(H.at({4, 1}), 10.0);
}

TEST_P(InterpAndCPU, ScatterAssignQuantized) {
  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {5, 2}, "data", false);
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {2}, "indices", false);
  auto *slices =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "slices", false);

  ctx_.allocate(data)->getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  ctx_.allocate(indices)->getHandle<int64_t>() = {1, 3};
  ctx_.allocate(slices)->getHandle() = {-3, -4, -7, -8};

  auto qParams = glow::quantization::chooseQuantizationParams(-11, 11);
  auto dataTy =
      mod_.uniqueType(ElemKind::Int8QTy, {5, 2}, qParams.scale, qParams.offset);
  auto slicesTy =
      mod_.uniqueType(ElemKind::Int8QTy, {2, 2}, qParams.scale, qParams.offset);

  auto *dataQ = F_->createQuantize("quantizeQ", data, dataTy);
  auto *slicesQ = F_->createQuantize("quantizeS", slices, slicesTy);
  auto *SA = F_->createScatterAssign("scatterassign", dataQ, indices, slicesQ);
  auto *DQ = F_->createDequantize("dequantize", SA);

  auto *result = F_->createSave("save", DQ);
  ctx_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto H = ctx_.get(result->getPlaceholder())->getHandle();

  EXPECT_NEAR(H.at({0, 0}), 1.0, 0.05);
  EXPECT_NEAR(H.at({0, 1}), 2.0, 0.05);
  EXPECT_NEAR(H.at({1, 0}), -3.0, 0.05);
  EXPECT_NEAR(H.at({1, 1}), -4.0, 0.05);
  EXPECT_NEAR(H.at({2, 0}), 5.0, 0.05);
  EXPECT_NEAR(H.at({2, 1}), 6.0, 0.05);
  EXPECT_NEAR(H.at({3, 0}), -7.0, 0.05);
  EXPECT_NEAR(H.at({3, 1}), -8.0, 0.05);
  EXPECT_NEAR(H.at({4, 0}), 9.0, 0.05);
  EXPECT_NEAR(H.at({4, 1}), 10.0, 0.05);
}

TEST_P(Operator, QuantizeAndDequantize) {
  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {1, 4}, "A", false);
  auto *B = mod_.createPlaceholder(ElemKind::FloatTy, {1, 4}, "B", false);
  ctx_.allocate(A)->getHandle() = {1, 1.2f, 0.5f, 1.3f};
  ctx_.allocate(B)->getHandle() = {1.8f, 5.2f, 3.5f, 11.3f};

  auto qType = mod_.uniqueType(ElemKind::Int8QTy, {1, 4}, 0.05, -138);
  auto *quantizeA = F_->createQuantize("quantize", A, qType);
  auto *quantizeB = F_->createQuantize("quantize", B, qType);
  auto *add = F_->createAdd("add", quantizeA, quantizeB);
  auto *dequantize = F_->createDequantize("dequantize", add);
  auto *result = F_->createSave("save", dequantize);
  ctx_.allocate(result->getPlaceholder());

  auto *fpAdd = F_->createAdd("fpAdd", A, B);
  auto *fpResult = F_->createSave("fpSave", fpAdd);
  ctx_.allocate(fpResult->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  EXPECT_TRUE(ctx_.get(result->getPlaceholder())
                  ->isEqual(*ctx_.get(fpResult->getPlaceholder())));
}

TEST_P(Operator, IntMatMul) {
  // The scaling factor 1.4x was carefully selected to make sure we don't
  // overflow or underflow the calculation.
  TypeRef resTy = mod_.uniqueType(ElemKind::Int8QTy, {3, 3}, 0.60, 4);
  TypeRef lhsTy = mod_.uniqueType(ElemKind::Int8QTy, {3, 3}, 0.075, -2);
  TypeRef rhsTy = mod_.uniqueType(ElemKind::Int8QTy, {3, 3}, 0.075, 2);

  auto *lhs = mod_.createPlaceholder(ElemKind::FloatTy, {3, 3}, "lhs", false);
  auto *rhs = mod_.createPlaceholder(ElemKind::FloatTy, {3, 3}, "rhs", false);

  ctx_.allocate(lhs)->getHandle() = {
      1.0, 2.0, 3.0, 4.0, 5.0, -5.0, -4.0, -3.0, 9.0,
  };

  ctx_.allocate(rhs)->getHandle() = {
      0.1f, -0.2f, 0.3f, 9.0f, -8.0f, 7.0f, 6.0f, 5.0f, 9.0f,
  };

  auto *lhsq = F_->createQuantize("lhs.q", lhs, lhsTy);
  auto *rhsq = F_->createQuantize("rhs.q", rhs, rhsTy);

  auto *matmulq = F_->createMatMul("matmul.q", resTy, lhsq, rhsq);

  auto *rq = F_->createDequantize("dequant", matmulq);

  auto *result = F_->createSave("save", rq);
  ctx_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  /*
   Test the following matrix multiplication:
   A = [[1.0, 2.0, 3.0], [4.0, 5.0, -5.0], [-4.0, -3.0, 9.0]]
   B = [[0.1, -0.2, 0.3], [9.0, -8.0, 7.0], [6.0, 5.0, 9.0]]
   A x B = [36.1,  -1.2,  41.3], [15.4, -65.8, -8.8], [26.6, 69.8,  58.8]]
   */

  auto H = ctx_.get(result->getPlaceholder())->getHandle();
  EXPECT_NEAR(H.at({0, 0}), 36.1, 1.0);
  EXPECT_NEAR(H.at({0, 1}), -1.2, 1.0);
  EXPECT_NEAR(H.at({0, 2}), 41.3, 1.0);
  EXPECT_NEAR(H.at({1, 0}), 15.4, 1.0);
  EXPECT_NEAR(H.at({1, 1}), -65.8, 1.0);
  EXPECT_NEAR(H.at({1, 2}), -8.8, 1.0);
  EXPECT_NEAR(H.at({2, 0}), 26.6, 1.0);
  EXPECT_NEAR(H.at({2, 1}), 69.8, 1.0);
  EXPECT_NEAR(H.at({2, 2}), 58.8, 1.0);
}

TEST_P(InterpAndCPU, IntBatchedArith) {
  TypeRef resTy = mod_.uniqueType(ElemKind::Int8QTy, {1, 3, 3}, 0.10, 1.0);
  TypeRef lhsTy = mod_.uniqueType(ElemKind::Int8QTy, {1, 3, 3}, 0.11, 4.0);
  TypeRef rhsTy = mod_.uniqueType(ElemKind::Int8QTy, {3, 3}, 0.14, -2.0);

  auto *lhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3}, "lhs", false);
  ctx_.allocate(lhs);
  auto *rhs = mod_.createPlaceholder(ElemKind::FloatTy, {3, 3}, "rhs", false);
  ctx_.allocate(rhs);

  ctx_.get(lhs)->getHandle() = {
      8.7f, 6.5f, 4.3f, 2.1f, 1.0f, -5.1f, -4.0f, -12.0f, 0.2f,
  };

  ctx_.get(rhs)->getHandle() = {
      -9.1f, -0.4f, 1.3f, 2.2f, -8.1f, 7.6f, -6.4f, 10.0f, 9.1f,
  };

  auto *lhsq = F_->createQuantize("lhs.q", lhs, lhsTy);
  auto *rhsq = F_->createQuantize("rhs.q", rhs, rhsTy);

  auto *matmulq = F_->createBatchedAdd("add", resTy, lhsq, rhsq);

  auto *rq = F_->createDequantize("dequant", matmulq);

  auto *result = F_->createSave("save", rq);
  ctx_.allocate(result->getPlaceholder());
  EE_.compile(CompilationMode::Infer, F_, ctx_);

  EE_.run();

  // A = [8.7, 6.5, 4.3, 2.1, 1.0, -5.1, -4.0, -12.0, 0.2]
  // B = [-9.1, -0.4, 1.3, 2.2, -8.1, 7.6, -6.4, 10.0, 9.1]
  // A + B = [-0.4, 6.1, 5.6, 4.3, -7.1, 2.5, -10.4, -2. , 9.3]
  auto H = ctx_.get(result->getPlaceholder())->getHandle();
  EXPECT_NEAR(H.at({0, 0, 0}), -0.4, 0.1);
  EXPECT_NEAR(H.at({0, 0, 1}), 6.1, 0.1);
  EXPECT_NEAR(H.at({0, 0, 2}), 5.6, 0.1);
  EXPECT_NEAR(H.at({0, 1, 0}), 4.3, 0.1);
  EXPECT_NEAR(H.at({0, 1, 1}), -7.1, 0.1);
  EXPECT_NEAR(H.at({0, 1, 2}), 2.5, 0.1);
  EXPECT_NEAR(H.at({0, 2, 0}), -10.4, 0.1);
  // TODO: verify slight deviation for this test case.
  EXPECT_NEAR(H.at({0, 2, 1}), -2, 0.11);
  EXPECT_NEAR(H.at({0, 2, 2}), 9.3, 0.1);
}

void checkFloat16Convolution(ExecutionEngine &EE, Function *F,
                             unsigned convDepth) {
  // In this test we generate a single precision floating-point based
  // convolution and an half precision one. We pass the same values
  // and we check that the results are below some
  // known delta.

  auto &mod = EE.getModule();
  Context ctx;

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 10, 10, 3}, "in", false);
  ctx.allocate(input);
  auto *inputFloat16 = mod.createPlaceholder(
      ElemKind::Float16Ty, {1, 10, 10, 3}, "in_float16", false);
  ctx.allocate(inputFloat16);
  auto *conv = F->createConv(ctx, "conv", input, convDepth, 5, 1, 0, 1);
  auto *convFloat16 =
      F->createConv(ctx, "convFloat16", inputFloat16, convDepth, 5, 1, 0, 1);

  // Make sure the inputs are the same for both convolutions.
  Tensor *inputFloat16Tensor = ctx.get(inputFloat16);
  auto inputFloat16Handle = inputFloat16Tensor->getHandle<float16_t>();
  inputFloat16Handle.randomize(-1.0, 1.0, mod.getPRNG());

  ctx.get(input)->copyWithCast<float, float16_t>(inputFloat16Tensor);

  auto *filterFloat16Var =
      llvm::cast<Placeholder>(convFloat16->getFilter().getNode());
  auto *filterVar = llvm::cast<Placeholder>(conv->getFilter().getNode());
  ctx.get(filterVar)->copyWithCast<float, float16_t>(ctx.get(filterFloat16Var));

  auto *biasFloat16Var =
      llvm::cast<Placeholder>(convFloat16->getBias().getNode());
  auto *biasVar = llvm::cast<Placeholder>(conv->getBias().getNode());
  ctx.get(biasVar)->copyWithCast<float, float16_t>(ctx.get(biasFloat16Var));

  SaveNode *save = F->createSave("save", conv);
  SaveNode *saveFloat16 = F->createSave("saveFloat16", convFloat16);

  auto floatOut =
      ctx.allocate(llvm::cast<Placeholder>(save->getOutput()))->getHandle();
  auto float16Out =
      ctx.allocate(llvm::cast<Placeholder>(saveFloat16->getOutput()))
          ->getHandle<float16_t>();

  EE.compile(CompilationMode::Infer, F, ctx);

  EE.run();

  // Check that the difference in the results is less than 0.1.
  for (int i = 0, e = floatOut.size(); i < e; i++) {
    EXPECT_NEAR(floatOut.raw(i), float(float16Out.raw(i)), 0.1);
  }
}

TEST_P(InterpOnly, FP16ConvolutionDepth10) {
  checkFloat16Convolution(EE_, F_, 10);
}

TEST_P(InterpOnly, FP16ConvolutionDepth8) {
  checkFloat16Convolution(EE_, F_, 8);
}

void checkIntConvolution(ExecutionEngine &EE, Function *F, unsigned convDepth,
                         Context &ctx) {
  // In this test we generate a Floating-point based convolution and an integer
  // convolution. We pass the same values and then subtract the results. We
  // check that the results are below some known delta.

  // In this test the output of the convolution is in the range [-256 ... 256].
  // The inputs (specified below) are in the range [-1 .. 1],
  auto &mod = EE.getModule();

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 10, 10, 3}, "in", false);
  auto *conv = F->createConv(ctx, "conv", input, convDepth, 5, 1, 0, 1);
  auto *filter = llvm::cast<Placeholder>(conv->getFilter().getNode());
  auto *bias = llvm::cast<Placeholder>(conv->getBias().getNode());

  ctx.allocate(input)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());
  ctx.get(bias)->getHandle().randomize(-2.0, 2.0, mod.getPRNG());

  TypeRef resTy =
      mod.uniqueType(ElemKind::Int8QTy, conv->getResult().dims(), 0.08, 0.0);
  TypeRef inputTy = mod.uniqueType(ElemKind::Int8QTy, input->dims(), 0.01, 0.0);
  TypeRef filterTy =
      mod.uniqueType(ElemKind::Int8QTy, filter->dims(), 0.01, 0.0);
  TypeRef biasTy = mod.uniqueType(ElemKind::Int8QTy, bias->dims(), 0.04, 0.0);

  auto *inputq = F->createQuantize("input.q", input, inputTy);
  auto *filterq = F->createQuantize("filter.q", filter, filterTy);
  auto *biasq = F->createQuantize("bias.q", bias, biasTy);

  auto *convq =
      F->createConv("convq", inputq, filterq, biasq, resTy, conv->getKernels(),
                    conv->getStrides(), conv->getPads(), 1);
  auto *dequantRes = F->createDequantize("dequant", convq);

  // Subtract the results of the convolution from the quantized convolution.
  auto *sub = F->createSub("compare", dequantRes, conv);

  auto *res = F->createSave("save", sub);
  ctx.allocate(res->getPlaceholder());
  EE.compile(CompilationMode::Infer, F, ctx);

  EE.run();
  auto H = ctx.get(res->getPlaceholder())->getHandle();

  // Check that the difference in the results is less than 0.1.
  for (int i = 0, e = H.size(); i < e; i++) {
    EXPECT_NEAR(H.raw(i), 0, 0.1);
  }
}

TEST_P(Operator, IntConvolutionDepth10) {
  checkIntConvolution(EE_, F_, 10, ctx_);
}

TEST_P(Operator, IntConvolutionDepth8) {
  checkIntConvolution(EE_, F_, 8, ctx_);
}

TEST_P(InterpAndCPU, IntConcat) {
  auto A = mod_.createPlaceholder(ElemKind::FloatTy, {3, 3}, "A", false);
  auto B = mod_.createPlaceholder(ElemKind::FloatTy, {2, 3}, "B", false);
  ctx_.allocate(A)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  ctx_.allocate(B)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());

  auto ATy = mod_.uniqueType(ElemKind::Int8QTy, A->dims(), 0.01, 0);
  auto BTy = mod_.uniqueType(ElemKind::Int8QTy, B->dims(), 0.01, 0);
  auto outTy = mod_.uniqueType(ElemKind::Int8QTy, {5, 3}, 0.01, 0);

  auto QA = F_->createQuantize("QA", A, ATy);
  auto QB = F_->createQuantize("QB", B, BTy);

  auto C = F_->createConcat("concat", {A, B}, 0);
  auto CQ = F_->createConcat("concatQ", {QA, QB}, 0, outTy);
  auto DCQ = F_->createDequantize("DQ", CQ);

  // Subtract the results of the Concat from the quantized Concat.
  auto sub = F_->createSub("compare", C, DCQ);

  auto *res = F_->createSave("save", sub);
  ctx_.allocate(res->getPlaceholder());
  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto R = ctx_.get(res->getPlaceholder())->getHandle();
  // Check that the difference in the results is less than 0.2.
  for (int i = 0, e = R.size(); i < e; i++) {
    EXPECT_NEAR(R.raw(i), 0, 0.2);
  }
}

TEST_P(InterpAndCPU, IntFC) {
  // In this test we subtract the outputs of a quantized FC and a floating-point
  // FC and ensure that the error is below some low value.
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 10, 3}, "in", false);
  auto *fc = F_->createFullyConnected(ctx_, "FC", input, 30);

  auto *weights = llvm::cast<Placeholder>(fc->getWeights());
  auto *bias = llvm::cast<Placeholder>(fc->getBias());

  ctx_.allocate(input)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  ctx_.get(bias)->getHandle().randomize(0, 0.00001, mod_.getPRNG());
  ctx_.get(weights)->getHandle().randomize(-1.1, 1.1, mod_.getPRNG());

  TypeRef resTy =
      mod_.uniqueType(ElemKind::Int8QTy, fc->getResult().dims(), 0.15, 4);
  TypeRef inputTy = mod_.uniqueType(ElemKind::Int8QTy, input->dims(), 0.01, 0);
  TypeRef weightsTy =
      mod_.uniqueType(ElemKind::Int8QTy, weights->dims(), 0.01, 2);
  TypeRef biasTy = mod_.uniqueType(ElemKind::Int8QTy, bias->dims(), 0.02, 1);

  auto *inputq = F_->createQuantize("input.q", input, inputTy);
  auto *weightsq = F_->createQuantize("filter.q", weights, weightsTy);
  auto *biasq = F_->createQuantize("bias.q", bias, biasTy);

  auto *fcq = F_->createFullyConnected("fcq", inputq, weightsq, biasq, resTy);
  auto *dequantRes = F_->createDequantize("dequant", fcq);

  // Subtract the results of the convolution from the quantized fc.
  auto *sub = F_->createSub("compare", dequantRes, fc);

  auto *res = F_->createSave("save", sub);
  ctx_.allocate(res->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto H = ctx_.get(res->getPlaceholder())->getHandle();
  // Check that there aren't too many elements with a difference in the results
  // of greater than 0.2.
  int count = 0;
  for (int i = 0, e = H.size(); i < e; i++) {
    if (std::abs(H.raw(i)) > 0.2) {
      count++;
    }
  }
  EXPECT_LT(count, 2);
}

TEST_P(InterpAndCPU, EntropyLossTest) {
  auto *P = mod_.createPlaceholder(ElemKind::FloatTy, {2, 3}, "P", false);
  auto *Y = mod_.createPlaceholder(ElemKind::Int64ITy, {2}, "Y", false);

  ctx_.allocate(P)->getHandle() = {0.2f, 0.5f, 0.3f, 0.4f, 0.3f, 0.3f};
  ctx_.allocate(Y)->getHandle<int64_t>() = {1, 2};
  auto *ceLoss = F_->createCrossEntropyLoss("CELoss", P, Y);
  auto *L = F_->createSave("save", ceLoss);
  ctx_.allocate(L->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto R = ctx_.get(L->getPlaceholder())->getHandle();
  EXPECT_NEAR(R.at({0}), -log(0.5) - log(0.3), 0.1);
}

/// Check that the max operator works properly.
TEST_P(Operator, Max) {
  PseudoRNG PRNG;

  auto *inputA =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1}, "A", false);
  ctx_.allocate(inputA)->getHandle<float>().randomize(-3.0, 3.0, PRNG);
  auto *inputB =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1}, "B", false);
  ctx_.allocate(inputB)->getHandle<float>().randomize(-3.0, 3.0, PRNG);
  auto *Max = F_->createMax("max", inputA, inputB);
  auto *S = F_->createSave("save", Max);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto result = ctx_.get(S->getPlaceholder())->getHandle<float>();
  auto handleA = ctx_.get(inputA)->getHandle<float>();
  auto handleB = ctx_.get(inputB)->getHandle<float>();
  ASSERT_EQ(result.size(), handleA.size());
  for (size_t idx = 0, end = result.size(); idx != end; ++idx) {
    EXPECT_EQ(result.raw(idx), std::max(handleA.raw(idx), handleB.raw(idx)));
  }
}

/// Check that the max operator works properly with FP16.
TEST_P(InterpOnly, FP16Max) {
  PseudoRNG PRNG;

  auto *inputA =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 3, 3, 1}, "A", false);
  ctx_.allocate(inputA)->getHandle<float16_t>().randomize(-3.0, 3.0, PRNG);
  auto *inputB =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 3, 3, 1}, "B", false);
  ctx_.allocate(inputB)->getHandle<float16_t>().randomize(-3.0, 3.0, PRNG);
  auto *Max = F_->createMax("max", inputA, inputB);
  auto *S = F_->createSave("save", Max);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto result = ctx_.get(S->getPlaceholder())->getHandle<float16_t>();
  auto handleA = ctx_.get(inputA)->getHandle<float16_t>();
  auto handleB = ctx_.get(inputB)->getHandle<float16_t>();
  ASSERT_EQ(result.size(), handleA.size());
  for (size_t idx = 0, end = result.size(); idx != end; ++idx) {
    EXPECT_EQ(result.raw(idx), std::max(handleA.raw(idx), handleB.raw(idx)));
  }
}

TEST_P(Operator, RescaleNode) {
  // Check the outputs of the RescaleQuantized operation.
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {4, 10}, 0.4, -3,
                                       "input", false);
  ctx_.allocate(input)->init(Tensor::InitKind::Broadcast, 40, mod_.getPRNG());

  auto T1 = mod_.uniqueType(ElemKind::Int8QTy, {4, 10}, 0.7, 5);
  auto T2 = mod_.uniqueType(ElemKind::Int8QTy, {4, 10}, 0.3, -4);
  auto resTy = mod_.uniqueType(ElemKind::Int8QTy, {4, 10}, 0.4, -4);

  // Test a sequence of rescale operations that the optimizer may try to
  // optimize at some point.
  auto *X = F_->createRescaleQuantized("R1", input, T1);
  auto *Y = F_->createRescaleQuantized("R2", X, T2);
  auto *Z = F_->createRescaleQuantized("R3", Y, resTy);

  auto *output = F_->createSave("save", Z);
  ctx_.allocate(output->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto RI = ctx_.get(input)->getHandle<int8_t>();
  auto RO = ctx_.get(output->getPlaceholder())->getHandle<int8_t>();

  EXPECT_EQ(RI.raw(0), 40);
  EXPECT_NEAR(RO.raw(0), 40, 1);
}

TEST_P(InterpAndCPU, QuantizedArithmeticRescaled) {
  const size_t len = 100;

  // In this test we check the correctness of the quantized Max, Min, Add, Sub,
  // Mul, and Div nodes as well as how they interact with the rescaling node.
  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {len}, "A", false);
  auto *B = mod_.createPlaceholder(ElemKind::FloatTy, {len}, "B", false);
  auto *C = mod_.createPlaceholder(ElemKind::FloatTy, {len}, "C", false);

  auto AH = ctx_.allocate(A)->getHandle();
  auto BH = ctx_.allocate(B)->getHandle();
  auto CH = ctx_.allocate(C)->getHandle();

  AH.randomize(-10, 10, mod_.getPRNG());
  BH.randomize(-10, 10, mod_.getPRNG());
  // Below, randomize between 1 and 10 to avoid division by 0 later.
  CH.randomize(1, 10, mod_.getPRNG());

  auto TA = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.2, 0);
  auto TB = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.1, 0);
  auto TC = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.3, 0);

  auto TI1 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.1, 0);
  auto TI2 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.8, 0);
  auto TI3 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.9, 0);
  auto TI4 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.0, 0);
  auto TI5 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.2, 0);
  auto TI6 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.7, 0);

  auto TO1 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.0, 0);
  auto TO2 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.9, 0);
  auto TO3 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.1, 0);
  auto TO4 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.2, 0);
  auto TO5 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.0, 0);
  auto TO6 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.1, 0);

  // Quantize input vars and apply max/min/add/sub/mul/div quantized.
  auto *QA = F_->createQuantize("QA", A, TA);
  auto *QB = F_->createQuantize("QB", B, TB);
  auto *QC = F_->createQuantize("QC", C, TC);

  Node *max = F_->createMax("max", TI1, QA, QB);
  Node *min = F_->createMin("min", TI2, QA, QB);
  Node *add = F_->createAdd("add", TI3, QA, QB);
  Node *sub = F_->createSub("sub", TI4, QA, QB);
  Node *mul = F_->createMul("mul", TI5, QA, QB);
  Node *div = F_->createDiv("div", TI6, QB, QC);

  // Rescale quantized results.
  max = F_->createRescaleQuantized("rescaleMax", max, TO1);
  min = F_->createRescaleQuantized("rescaleMin", min, TO2);
  add = F_->createRescaleQuantized("rescaleAdd", add, TO3);
  sub = F_->createRescaleQuantized("rescaleSub", sub, TO4);
  mul = F_->createRescaleQuantized("rescaleMul", mul, TO5);
  div = F_->createRescaleQuantized("rescaleDiv", div, TO6);

  // Dequantize results back to floating-point.
  max = F_->createDequantize("maxDQ", max);
  min = F_->createDequantize("minDQ", min);
  add = F_->createDequantize("addDQ", add);
  sub = F_->createDequantize("subDQ", sub);
  mul = F_->createDequantize("mulDQ", mul);
  div = F_->createDequantize("divDQ", div);

  // Save results of the operations.
  auto *O1 = F_->createSave("saveMax", max);
  auto *O2 = F_->createSave("saveMin", min);
  auto *O3 = F_->createSave("saveAdd", add);
  auto *O4 = F_->createSave("saveSub", sub);
  auto *O5 = F_->createSave("saveMul", mul);
  auto *O6 = F_->createSave("saveDiv", div);

  ctx_.allocate(O1->getPlaceholder());
  ctx_.allocate(O2->getPlaceholder());
  ctx_.allocate(O3->getPlaceholder());
  ctx_.allocate(O4->getPlaceholder());
  ctx_.allocate(O5->getPlaceholder());
  ctx_.allocate(O6->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  for (size_t i = 0; i < len; i++) {
    auto max = std::max(AH.at({i}), BH.at({i}));
    auto min = std::min(AH.at({i}), BH.at({i}));
    auto add = AH.at({i}) + BH.at({i});
    auto sub = AH.at({i}) - BH.at({i});
    auto mul = AH.at({i}) * BH.at({i});
    auto div = BH.at({i}) / CH.at({i});

    // We generate numbers up to 110, so a difference of 2 (~2%) is reasonable.
    EXPECT_NEAR(max, ctx_.get(O1->getPlaceholder())->getHandle().at({i}), 2.0);
    EXPECT_NEAR(min, ctx_.get(O2->getPlaceholder())->getHandle().at({i}), 2.0);
    EXPECT_NEAR(add, ctx_.get(O3->getPlaceholder())->getHandle().at({i}), 2.0);
    EXPECT_NEAR(sub, ctx_.get(O4->getPlaceholder())->getHandle().at({i}), 2.0);
    EXPECT_NEAR(mul, ctx_.get(O5->getPlaceholder())->getHandle().at({i}), 2.0);
    EXPECT_NEAR(div, ctx_.get(O6->getPlaceholder())->getHandle().at({i}), 2.0);
  }
}

TEST_P(Operator, QuantizedTranspose) {
  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {2, 3}, "A", false);
  auto *B = mod_.createPlaceholder(ElemKind::FloatTy, {3, 2}, "B", false);

  ctx_.allocate(A)->getHandle() = {1, 1.2f, 0.5f, 1.3f, 2.7f, 5.8f};
  ctx_.allocate(B);
  ctx_.get(A)->transpose(ctx_.get(B), {1, 0});
  auto qType = mod_.uniqueType(ElemKind::Int8QTy, {2, 3}, 0.05, -138);
  auto *quantizeA = F_->createQuantize("quantize", A, qType);
  auto *tr = F_->createTranspose("tr", quantizeA, {1, 0});
  auto *dequantize = F_->createDequantize("dequantize", tr);

  auto *result = F_->createSave("ret", dequantize);
  ctx_.allocate(result->getPlaceholder());
  auto *fpTr = F_->createTranspose("fpTr", A, {1, 0});

  auto *fpResult = F_->createSave("fpRet", fpTr);
  ctx_.allocate(fpResult->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  EXPECT_TRUE(ctx_.get(result->getPlaceholder())->isEqual(*ctx_.get(B)));
  EXPECT_TRUE(ctx_.get(fpResult->getPlaceholder())->isEqual(*ctx_.get(B)));
}

TEST_P(Operator, QuantizedArithmeticUnrescaled) {
  const size_t len = 1000;

  // In this test we check the correctness of the quantized Max, Min, Add, Sub,
  // Mul, and Div operations.
  auto TQA = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.1, -1);
  auto TQB = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.9, 2);
  // For TQC, set offset to -11 to avoid division by 0 later.
  auto TQC = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.2, -11);
  auto TO1 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.4, 3);
  auto TO2 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.8, 2);
  auto TO3 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.7, 5);
  auto TO4 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.3, -7);
  auto TO5 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.2, 3);
  auto TO6 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.0, -2);

  auto *QA = mod_.createPlaceholder(ElemKind::Int8QTy, {len}, TQA->getScale(),
                                    TQA->getOffset(), "QA", false);
  auto *QB = mod_.createPlaceholder(ElemKind::Int8QTy, {len}, TQB->getScale(),
                                    TQB->getOffset(), "QB", false);
  auto *QC = mod_.createPlaceholder(ElemKind::Int8QTy, {len}, TQC->getScale(),
                                    TQC->getOffset(), "QC", false);

  ctx_.allocate(QA)->getHandle<int8_t>().randomize(-10, 10, mod_.getPRNG());
  ctx_.allocate(QB)->getHandle<int8_t>().randomize(-10, 10, mod_.getPRNG());
  ctx_.allocate(QC)->getHandle<int8_t>().randomize(-10, 10, mod_.getPRNG());

  // Apply max/min/add/sub/mul/div quantized.
  Node *max = F_->createMax("max", TO1, QA, QB);
  Node *min = F_->createMin("min", TO2, QA, QB);
  Node *add = F_->createAdd("add", TO3, QA, QB);
  Node *sub = F_->createSub("sub", TO4, QA, QB);
  Node *mul = F_->createMul("mul", TO5, QA, QB);
  Node *div = F_->createDiv("div", TO6, QB, QC);

  // Save results of the operations.
  auto *O1 = F_->createSave("saveMax", max);
  auto *O2 = F_->createSave("saveMin", min);
  auto *O3 = F_->createSave("saveAdd", add);
  auto *O4 = F_->createSave("saveSub", sub);
  auto *O5 = F_->createSave("saveMul", mul);
  auto *O6 = F_->createSave("saveDiv", div);

  ctx_.allocate(O1->getPlaceholder());
  ctx_.allocate(O2->getPlaceholder());
  ctx_.allocate(O3->getPlaceholder());
  ctx_.allocate(O4->getPlaceholder());
  ctx_.allocate(O5->getPlaceholder());
  ctx_.allocate(O6->getPlaceholder());

  auto QAH = ctx_.get(QA)->getHandle<int8_t>();
  auto QBH = ctx_.get(QB)->getHandle<int8_t>();
  auto QCH = ctx_.get(QC)->getHandle<int8_t>();
  auto O1H = ctx_.get(O1->getPlaceholder())->getHandle<int8_t>();
  auto O2H = ctx_.get(O2->getPlaceholder())->getHandle<int8_t>();
  auto O3H = ctx_.get(O3->getPlaceholder())->getHandle<int8_t>();
  auto O4H = ctx_.get(O4->getPlaceholder())->getHandle<int8_t>();
  auto O5H = ctx_.get(O5->getPlaceholder())->getHandle<int8_t>();
  auto O6H = ctx_.get(O6->getPlaceholder())->getHandle<int8_t>();

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  for (size_t i = 0; i < len; i++) {
    float a = TQA->getScale() * (QAH.at({i}) - TQA->getOffset());
    float b = TQB->getScale() * (QBH.at({i}) - TQB->getOffset());
    float c = TQC->getScale() * (QCH.at({i}) - TQC->getOffset());
    float max = std::max(a, b) / TO1->getScale() + TO1->getOffset();
    float min = std::min(a, b) / TO2->getScale() + TO2->getOffset();
    float add = (a + b) / TO3->getScale() + TO3->getOffset();
    float sub = (a - b) / TO4->getScale() + TO4->getOffset();
    float mul = (a * b) / TO5->getScale() + TO5->getOffset();
    float div = (b / c) / TO6->getScale() + TO6->getOffset();

    EXPECT_NEAR(std::round(max), O1H.at({i}), 1.0);
    EXPECT_NEAR(std::round(min), O2H.at({i}), 1.0);
    EXPECT_NEAR(std::round(add), O3H.at({i}), 1.0);
    EXPECT_NEAR(std::round(sub), O4H.at({i}), 1.0);
    EXPECT_NEAR(std::round(mul), O5H.at({i}), 1.0);
    EXPECT_NEAR(std::round(div), O6H.at({i}), 1.0);
  }
}

TEST_P(InterpAndCPU, QuantizedCmpLTEAndSelect) {
  // In this test we check the correctness of the quantized
  // less-than-or-equal-to comparison operator.
  const size_t len = 1000;
  auto TQA = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.1, -3);
  auto TQB = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.9, 5);
  auto TQC = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.8, 3);
  auto TQD = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.2, -4);
  auto OT = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.5, -2);

  auto *QA = mod_.createPlaceholder(ElemKind::Int8QTy, {len}, TQA->getScale(),
                                    TQA->getOffset(), "QA", false);
  auto *QB = mod_.createPlaceholder(ElemKind::Int8QTy, {len}, TQB->getScale(),
                                    TQB->getOffset(), "QB", false);
  auto *QC = mod_.createPlaceholder(ElemKind::Int8QTy, {len}, TQC->getScale(),
                                    TQC->getOffset(), "QC", false);
  auto *QD = mod_.createPlaceholder(ElemKind::Int8QTy, {len}, TQD->getScale(),
                                    TQD->getOffset(), "QD", false);

  auto QAH = ctx_.allocate(QA)->getHandle<int8_t>();
  auto QBH = ctx_.allocate(QB)->getHandle<int8_t>();
  auto QCH = ctx_.allocate(QC)->getHandle<int8_t>();
  auto QDH = ctx_.allocate(QD)->getHandle<int8_t>();

  QAH.randomize(-128, 127, mod_.getPRNG());
  QBH.randomize(-128, 127, mod_.getPRNG());
  QCH.randomize(-128, 127, mod_.getPRNG());
  QDH.randomize(-128, 127, mod_.getPRNG());

  // Apply comparison and selection quantized.
  Node *cmpLTE = F_->createCmpLTE("cmpLTE", QA, QB);
  Node *select = F_->createSelect("select", OT, cmpLTE, QC, QD);

  // Save result of the operation.
  auto *out = F_->createSave("save", select);
  auto OH = ctx_.allocate(out->getPlaceholder())->getHandle<int8_t>();

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  int count_strict = 0;
  int count = 0;
  for (size_t i = 0; i < len; i++) {
    float a = TQA->getScale() * (QAH.at({i}) - TQA->getOffset());
    float b = TQB->getScale() * (QBH.at({i}) - TQB->getOffset());
    float c = TQC->getScale() * (QCH.at({i}) - TQC->getOffset());
    float d = TQD->getScale() * (QDH.at({i}) - TQD->getOffset());
    float tmp = (a <= b) ? c : d;
    int32_t q = std::round(tmp / 1.5 - 2);
    int8_t select = quantization::clip<int32_t, int8_t>(q);

    if (OH.at({i}) != select) {
      count_strict++;
      if (std::abs(OH.at({i}) - select) > 1) {
        count++;
      }
    }
  }
  // Require that the number of off-by-1 errors be at most 0.6%.
  EXPECT_LE(count_strict, 6);
  EXPECT_LE(count, 4);
}

TEST_P(Operator, TestQuantizedRescaleSequence) {
  const size_t len = 100;

  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {len}, "A", false);

  auto AH = ctx_.allocate(A)->getHandle();

  // Notice that the range below is the an approximation of the scale factors in
  // T3 and T4. If we increase the size of the range we may start losing some
  // values.
  AH.randomize(-12, 12, mod_.getPRNG());

  auto T1 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.0, 0);
  auto T2 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.9, 2);
  auto T3 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.1, -3);
  auto T4 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.1, 7);
  auto T5 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.3, -3);

  Node *R = F_->createQuantize("R", A, T1);
  // Check that a sequence of type conversions does not change the result.
  R = F_->createRescaleQuantized("R", R, T1);
  R = F_->createRescaleQuantized("R", R, T2);
  R = F_->createRescaleQuantized("R", R, T3);
  // Check that adding the quantized zero does not change the result.
  auto *G = F_->createSplat("splatZero", T3, 0.0);
  R = F_->createAdd("addZero", G, R);
  R = F_->createRescaleQuantized("R", R, T4);
  R = F_->createRescaleQuantized("R", R, T5);
  R = F_->createRescaleQuantized("R", R, T1);
  auto *DQ = F_->createDequantize("DQ", R);

  // Test a sequence of rescale operations t
  auto *result = F_->createSave("save", DQ);
  auto OH = ctx_.allocate(result->getPlaceholder())->getHandle();
  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  for (size_t i = 0; i < len; i++) {
    EXPECT_NEAR(AH.at({i}), OH.at({i}), 1.0);
  }
}

TEST_P(Operator, FCGradientCheck) {
  // Create net representing A*X+Y=B, where X and Y are trainable, while
  // A and B are fixed. Record gradients for X and Y after 3 steps and compare
  // with reference values.
  TrainingConfig TC;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {2, 1}, "A", false);
  auto *B = mod_.createPlaceholder(ElemKind::FloatTy, {2, 1}, "B", false);
  auto *X = mod_.createPlaceholder(ElemKind::FloatTy, {1, 1}, "X", true);
  auto *Y = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "Y", true);

  ctx_.allocate(A);
  ctx_.allocate(B);
  ctx_.allocate(X)->init(Tensor::InitKind::Broadcast, -1.26274, mod_.getPRNG());
  ctx_.allocate(Y)->init(Tensor::InitKind::Broadcast, 0.1, mod_.getPRNG());

  auto *FC = F_->createFullyConnected("fc", A, X, Y);
  auto *S = F_->createRegression("reg", FC, B);
  auto *save = F_->createSave("ret", S);
  ctx_.allocate(save->getPlaceholder());

  Tensor initA(ElemKind::FloatTy, {2, 1});
  Tensor initB(ElemKind::FloatTy, {2, 1});
  initA.getHandle() = {4.2f, 9.875f};
  initB.getHandle() = {-13.1f, 3.14f};

  Function *DF = glow::differentiate(F_, TC, "d_main");
  EE_.compile(CompilationMode::Train, DF, ctx_);
  runBatch(EE_, ctx_, 3, sampleCounter, {A, B}, {&initA, &initB});

  EXPECT_NEAR(ctx_.get(X)->getHandle().raw(0), -0.21294, 1E-5);
  EXPECT_NEAR(ctx_.get(Y)->getHandle().raw(0), 0.01656, 1E-5);
}

TEST_P(InterpAndCPU, concatVectors) {
  F_->setName("concatVectors");

  auto *V1 = mod_.createPlaceholder(ElemKind::Int64ITy, {10}, "V1", false);
  auto *V2 = mod_.createPlaceholder(ElemKind::Int64ITy, {20}, "V2", false);
  auto *V3 = mod_.createPlaceholder(ElemKind::Int64ITy, {30}, "V3", false);

  ctx_.allocate(V1);
  ctx_.allocate(V2);
  ctx_.allocate(V3);

  Node *L = F_->createConcat("concat", {V1, V2, V3}, 0);
  auto *result = F_->createSave("ret", L);
  ctx_.allocate(result->getPlaceholder());

  Tensor I1(ElemKind::Int64ITy, {10});
  Tensor I2(ElemKind::Int64ITy, {20});
  Tensor I3(ElemKind::Int64ITy, {30});

  for (size_t i = 0; i < 10; i++) {
    I1.getHandle<int64_t>().at({i}) = i;

    I2.getHandle<int64_t>().at({i}) = i + 10;
    I2.getHandle<int64_t>().at({i + 10}) = i + 20;
    I3.getHandle<int64_t>().at({i}) = i + 30;
    I3.getHandle<int64_t>().at({i + 10}) = i + 40;
    I3.getHandle<int64_t>().at({i + 20}) = i + 50;
  }

  EE_.compile(CompilationMode::Infer, F_, ctx_);

  // Testing the output vector.
  updateInputPlaceholders(ctx_, {V1, V2, V3}, {&I1, &I2, &I3});
  EE_.run();

  auto RNWH = ctx_.get(result->getPlaceholder())->getHandle<int64_t>();
  (void)RNWH;

  for (size_t i = 0; i < 60; i++) {
    EXPECT_NEAR(RNWH.at({i}), i, 0.001);
  }
}

/// Check that concatenating two tensors repeatedly is correct. This is intended
/// to verify that IRGen to InsertTensor instructions with axis/count works
/// correctly.
TEST_P(InterpAndCPU, concatVectorsRepeated) {
  F_->setName("concatVectors");

  auto *V1 = mod_.createPlaceholder(ElemKind::Int64ITy, {10}, "V1", false);
  auto *V2 = mod_.createPlaceholder(ElemKind::Int64ITy, {20}, "V2", false);
  ctx_.allocate(V1);
  ctx_.allocate(V2);

  // Alternate adding sequences of V1 and V2, so that the IRGen'd InsertTensors
  // have different counts.
  Node *L = F_->createConcat("concat", {V2, V1, V1, V1, V2, V2, V1, V1, V2}, 0);
  auto *result = F_->createSave("ret", L);
  ctx_.allocate(result->getPlaceholder());

  Tensor I1(ElemKind::Int64ITy, {10});
  Tensor I2(ElemKind::Int64ITy, {20});

  for (size_t i = 0; i < 10; i++) {
    I1.getHandle<int64_t>().at({i}) = 1;

    I2.getHandle<int64_t>().at({i}) = 2;
    I2.getHandle<int64_t>().at({i + 10}) = 2;
  }

  EE_.compile(CompilationMode::Infer, F_, ctx_);

  // Testing the output vector.
  updateInputPlaceholders(ctx_, {V1, V2}, {&I1, &I2});
  EE_.run();

  auto outH = ctx_.get(result->getPlaceholder())->getHandle<int64_t>();

  // Simply verify here that the values are in their correct places, based on
  // the number of times/order V1 and V2 are concatenated and their sizes.
  for (size_t i = 0; i < 130; i++) {
    if ((i < 20) || (i >= 50 && i < 90) || (i >= 110)) {
      EXPECT_EQ(outH.at({i}), 2);
    } else {
      EXPECT_EQ(outH.at({i}), 1);
    }
  }
}

TEST_P(InterpAndCPU, sliceVectors) {
  F_->setName("sliceVectors");

  auto *V = mod_.createPlaceholder(ElemKind::Int64ITy, {3, 30}, "V", false);
  ctx_.allocate(V);

  Node *S1 = F_->createSlice("slice1", V, {0, 10}, {3, 13});
  Node *S2 = F_->createSlice("slice2", V, {1, 0}, {2, 30});
  Node *S3 = F_->createSlice("slice3", V, {2, 10}, {3, 12});

  auto *result1 = F_->createSave("ret1", S1);
  auto *result2 = F_->createSave("ret2", S2);
  auto *result3 = F_->createSave("ret3", S3);

  ctx_.allocate(result1->getPlaceholder());
  ctx_.allocate(result2->getPlaceholder());
  ctx_.allocate(result3->getPlaceholder());

  Tensor I(ElemKind::Int64ITy, {3, 30});

  for (size_t j = 0; j < 30; j++) {
    I.getHandle<int64_t>().at({0, j}) = j;
    I.getHandle<int64_t>().at({1, j}) = j + 30;
    I.getHandle<int64_t>().at({2, j}) = j + 60;
  }

  EE_.compile(CompilationMode::Infer, F_, ctx_);

  // Testing the output slices.
  updateInputPlaceholders(ctx_, {V}, {&I});
  EE_.run();

  auto RNWH1 = ctx_.get(result1->getPlaceholder())->getHandle<int64_t>();
  auto RNWH2 = ctx_.get(result2->getPlaceholder())->getHandle<int64_t>();
  auto RNWH3 = ctx_.get(result3->getPlaceholder())->getHandle<int64_t>();

  EXPECT_EQ(3, RNWH1.dims()[0]);
  EXPECT_EQ(3, RNWH1.dims()[1]);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 10; j < 13; j++) {
      EXPECT_NEAR(RNWH1.at({i, j - 10}), j + i * 30, 0.001);
    }
  }
  EXPECT_EQ(1, RNWH2.dims()[0]);
  EXPECT_EQ(30, RNWH2.dims()[1]);
  for (size_t j = 0; j < 30; j++) {
    EXPECT_NEAR(RNWH2.at({0, j}), j + 30, 0.001);
  }
  EXPECT_EQ(1, RNWH3.dims()[0]);
  EXPECT_EQ(2, RNWH3.dims()[1]);
  for (size_t j = 10; j < 12; j++) {
    EXPECT_NEAR(RNWH3.at({0, j - 10}), j + 60, 0.001);
  }
}

TEST_P(InterpAndCPU, sliceConcatVectors) {
  F_->setName("sliceConcatVectors");

  auto *V = mod_.createPlaceholder(ElemKind::Int64ITy, {5, 4}, "V", false);
  ctx_.allocate(V);

  Tensor I(ElemKind::Int64ITy, {5, 4});
  for (size_t i = 0; i < 5; i++) {
    for (size_t j = 0; j < 4; j++) {
      I.getHandle<int64_t>().at({i, j}) = i * 100 + j;
    }
  }

  Node *S0 = F_->createSlice("slice0", V, {1, 0}, {5, 4});
  Node *S1 = F_->createSlice("slice1", S0, {0, 0}, {2, 4});
  Node *S2 = F_->createSlice("slice2", S0, {2, 0}, {4, 4});
  Node *S3 = F_->createSlice("slice3", S0, {0, 0}, {2, 2});
  Node *S4 = F_->createSlice("slice4", S0, {2, 2}, {4, 4});
  Node *S5 = F_->createSlice("slice5", V, {0, 0}, {1, 4});

  Node *C0 = F_->createConcat("concat0", {S5, S1}, 0);
  Node *C1 = F_->createConcat("concat1", {S3, S4}, 1);
  Node *C2 = F_->createConcat("concat2", {S2, C1, C0}, 0);

  auto *result = F_->createSave("ret", C2);
  ctx_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);

  updateInputPlaceholders(ctx_, {V}, {&I});
  EE_.run();

  const size_t expected[7][4] = {{300, 301, 302, 303}, {400, 401, 402, 403},
                                 {100, 101, 302, 303}, {200, 201, 402, 403},
                                 {0, 1, 2, 3},         {100, 101, 102, 103},
                                 {200, 201, 202, 203}};

  auto resultH = ctx_.get(result->getPlaceholder())->getHandle<int64_t>();
  EXPECT_EQ(7, resultH.dims()[0]);
  EXPECT_EQ(4, resultH.dims()[1]);
  for (size_t i = 0; i < 7; i++) {
    for (size_t j = 0; j < 4; j++) {
      EXPECT_EQ(resultH.at({i, j}), expected[i][j]);
    }
  }
}

TEST_P(InterpAndCPU, Tile) {
  F_->setName("concatVectors");

  auto *V = mod_.createPlaceholder(ElemKind::FloatTy, {4, 5}, "V", false);
  ctx_.allocate(V);

  Node *T0 = F_->createTile("tile0", V, /* tiles */ 3, /* axis */ 0);
  auto *result0 = F_->createSave("res0", T0);
  ctx_.allocate(result0->getPlaceholder());

  Node *T1 = F_->createTile("tile1", V, /* tiles */ 3, /* axis */ 1);
  auto *result1 = F_->createSave("res1", T1);
  ctx_.allocate(result1->getPlaceholder());

  Tensor VT(ElemKind::FloatTy, {4, 5});

  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 5; j++) {
      VT.getHandle<float>().at({i, j}) = i * 5 + j;
    }
  }

  EE_.compile(CompilationMode::Infer, F_, ctx_);

  updateInputPlaceholders(ctx_, {V}, {&VT});
  EE_.run();

  // Testing the output vector with axis 0.
  auto res0 = ctx_.get(result0->getPlaceholder())->getHandle<float>();
  for (size_t i = 0; i < res0.dims()[0]; i++) {
    for (size_t j = 0; j < res0.dims()[1]; j++) {
      EXPECT_EQ(res0.at({i, j}), (i % 4) * 5 + j);
    }
  }

  // Testing the output vector with axis 1.
  auto res1 = ctx_.get(result1->getPlaceholder())->getHandle<float>();
  for (size_t i = 0; i < res1.dims()[0]; i++) {
    for (size_t j = 0; j < res1.dims()[1]; j++) {
      EXPECT_EQ(res1.at({i, j}), i * 5 + (j % 5));
    }
  }
}

TEST_P(InterpAndCPU, QuantizedTile) {
  F_->setName("concatVectors");

  auto *V = mod_.createPlaceholder(ElemKind::FloatTy, {4, 5}, "V", false);
  ctx_.allocate(V);

  auto quantizationParams = glow::quantization::chooseQuantizationParams(0, 20);
  auto quantizeTy =
      mod_.uniqueType(ElemKind::Int8QTy, {4, 5}, quantizationParams.scale,
                      quantizationParams.offset);
  auto *Q = F_->createQuantize("quantize", V, quantizeTy);

  Node *T0 = F_->createTile("tile0", Q, /* tiles */ 3, /* axis */ 0);
  auto *DQ0 = F_->createDequantize("dequantize0", T0);
  auto *result0 = F_->createSave("res0", DQ0);
  ctx_.allocate(result0->getPlaceholder());

  Node *T1 = F_->createTile("tile1", Q, /* tiles */ 3, /* axis */ 1);
  auto *DQ1 = F_->createDequantize("dequantize1", T1);
  auto *result1 = F_->createSave("res1", DQ1);
  ctx_.allocate(result1->getPlaceholder());

  Tensor VT(ElemKind::FloatTy, {4, 5});

  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 5; j++) {
      VT.getHandle<float>().at({i, j}) = i * 5 + j;
    }
  }

  EE_.compile(CompilationMode::Infer, F_, ctx_);

  updateInputPlaceholders(ctx_, {V}, {&VT});
  EE_.run();

  // Testing the output vector with axis 0.
  auto res0 = ctx_.get(result0->getPlaceholder())->getHandle<float>();
  for (size_t i = 0; i < res0.dims()[0]; i++) {
    for (size_t j = 0; j < res0.dims()[1]; j++) {
      EXPECT_NEAR(res0.at({i, j}), (i % 4) * 5 + j, 0.05);
    }
  }

  // Testing the output vector with axis 1.
  auto res1 = ctx_.get(result1->getPlaceholder())->getHandle<float>();
  (void)res1;
  for (size_t i = 0; i < res1.dims()[0]; i++) {
    for (size_t j = 0; j < res1.dims()[1]; j++) {
      EXPECT_NEAR(res1.at({i, j}), i * 5 + (j % 5), 0.05);
    }
  }
}

TEST_P(Operator, simpleCmpSelectPredication) {
  // A simple test that checks predication of some values using the
  // compare-select pair of instructions. Keep doubling some values
  // until some condition is met.

  auto *inputs =
      mod_.createPlaceholder(ElemKind::FloatTy, {10}, "inputs", false);
  auto *counters =
      mod_.createPlaceholder(ElemKind::FloatTy, {10}, "counters", false);

  ctx_.allocate(counters)->getHandle() = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  ctx_.allocate(inputs)->getHandle().clear(1);

  Node *cnt = counters;
  Node *data = inputs;
  Node *const1 = F_->createSplat("const1", counters->getType(), 1.0);
  Node *const0 = F_->createSplat("const0", counters->getType(), 0.0);

  for (int i = 0; i < 10; i++) {
    cnt = F_->createSub("sub1", cnt, const1);
    Node *pred = F_->createCmpLTE("cmp", const0, cnt);

    Node *const2 = F_->createSplat("const2", data->getType(0), 2.0);
    Node *newData = F_->createMul("mul2x", data, const2);

    data = F_->createSelect("select", pred, newData, data);
  }

  auto *SN = F_->createSave("ret", data);
  ctx_.allocate(SN->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto H = ctx_.get(SN->getPlaceholder())->getHandle();
  ASSERT_NEAR(H.at(0), 1, 0.001);
  ASSERT_NEAR(H.at(1), 2, 0.001);
  ASSERT_NEAR(H.at(2), 4, 0.001);
  ASSERT_NEAR(H.at(3), 8, 0.001);
  ASSERT_NEAR(H.at(4), 16, 0.001);
  ASSERT_NEAR(H.at(5), 32, 0.001);
  ASSERT_NEAR(H.at(6), 64, 0.001);
  ASSERT_NEAR(H.at(7), 128, 0.001);
  ASSERT_NEAR(H.at(8), 256, 0.001);
  ASSERT_NEAR(H.at(9), 512, 0.001);
}

TEST_P(Operator, simplePredication) {
  auto *inputs =
      mod_.createPlaceholder(ElemKind::FloatTy, {10, 10, 10}, "inputs", false);
  auto *counters =
      mod_.createPlaceholder(ElemKind::FloatTy, {10}, "counters", false);

  ctx_.allocate(counters)->getHandle() = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  ctx_.allocate(inputs)->getHandle().randomize(-10, 10, mod_.getPRNG());

  Node *C5 = F_->createSplat("C5", counters->getType(), 5.0);
  Node *pred = F_->createCmpLTE("cmp", C5, counters);

  auto *FC0 = F_->createFullyConnected(ctx_, "FC0", inputs, 128);
  auto *RL0 = F_->createRELU("RL0", FC0);
  auto *FC1 = F_->createFullyConnected(ctx_, "FC1", RL0, 64);
  auto *RL1 = F_->createRELU("RL1", FC1);
  auto *FC2 = F_->createFullyConnected(ctx_, "FC2", RL1, 32);
  auto *RL2 = F_->createRELU("RL2", FC2);

  auto *save = F_->createSave("ret", RL2);
  ctx_.allocate(save->getPlaceholder());

  FC0->setPredicate(pred);
  FC1->setPredicate(pred);
  FC2->setPredicate(pred);

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();
}

TEST_P(InterpAndCPU, ChannelShuffle) {
  auto *inputs =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 12, 1, 1}, "inputs", false);
  ctx_.allocate(inputs)->getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  Node *CS = F_->createChannelShuffle("CS", inputs, 3, 1);
  SaveNode *S = F_->createSave("save", CS);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto results = ctx_.get(S->getPlaceholder())->getHandle();

  EXPECT_EQ(results.size(), 12);
  std::vector<float> expected = {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
  for (size_t i = 0; i < expected.size(); i++)
    EXPECT_FLOAT_EQ(results.at({0, i, 0, 0}), expected[i]);
}

TEST_P(Operator, Squeeze) {
  auto *inputs =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 1, 5}, "inputs", false);
  ctx_.allocate(inputs)->getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::vector<float> expectedValues = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  // Test 1:
  {
    std::vector<size_t> axes = {0};
    Node *SQZ = F_->createSqueeze("SQZ", inputs, axes);
    SaveNode *S = F_->createSave("save", SQZ);
    ctx_.allocate(S->getPlaceholder());

    EE_.compile(CompilationMode::Infer, F_, ctx_);
    EE_.run();

    auto results = ctx_.get(S->getPlaceholder())->getHandle();
    std::vector<size_t> expectedDims = {2, 1, 5};
    EXPECT_TRUE(results.dims().vec() == expectedDims);
    for (size_t i = 0; i < 10; i++)
      EXPECT_FLOAT_EQ(results.raw(i), expectedValues[i]);
  }

  // Test 2:
  {
    std::vector<size_t> axes = {0, 2, 2};
    Node *SQZ = F_->createSqueeze("SQZ", inputs, axes);
    SaveNode *S = F_->createSave("save", SQZ);
    ctx_.allocate(S->getPlaceholder());

    EE_.compile(CompilationMode::Infer, F_, ctx_);
    EE_.run();

    auto results = ctx_.get(S->getPlaceholder())->getHandle();
    std::vector<size_t> expectedDims = {2, 5};
    EXPECT_TRUE(results.dims().vec() == expectedDims);
    for (size_t i = 0; i < 10; i++)
      EXPECT_FLOAT_EQ(results.raw(i), expectedValues[i]);
  }

  // Test 3: 0-dimensional Tensor
  {
    auto *emptyInput =
        mod_.createPlaceholder(ElemKind::FloatTy, {1}, "emptyInput", false);
    ctx_.allocate(emptyInput)->getHandle() = {42.0};

    std::vector<size_t> axes = {0};
    Node *SQZ = F_->createSqueeze("SQZ", emptyInput, axes);
    SaveNode *S1 = F_->createSave("save", SQZ);
    Node *UnSQZ = F_->createExpandDims("UnSQZ", SQZ, axes);
    SaveNode *S2 = F_->createSave("save", UnSQZ);

    ctx_.allocate(S1->getPlaceholder());
    ctx_.allocate(S2->getPlaceholder());

    EE_.compile(CompilationMode::Infer, F_, ctx_);
    EE_.run();

    auto res1 = ctx_.get(S1->getPlaceholder())->getHandle();
    EXPECT_TRUE(res1.dims().vec() == std::vector<size_t>());
    EXPECT_FLOAT_EQ(res1.raw(0), 42.0);
    auto res2 = ctx_.get(S2->getPlaceholder())->getHandle();
    EXPECT_TRUE(res2.dims().vec() == std::vector<size_t>(1, 1));
    EXPECT_FLOAT_EQ(res2.raw(0), 42.0);
  }
}

/// Check that the expand dims operator works, which is implemented with a
/// reshape.
TEST_P(Operator, ExpandDims) {
  auto *inputs =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "inputs", false);
  auto IH = ctx_.allocate(inputs)->getHandle();
  IH = {1, 2, 3, 4};

  // This should be uniqued and sorted, so should become {0, 1, 3, 5}.
  std::vector<size_t> axes = {3, 0, 5, 1, 3};
  Node *EDN = F_->createExpandDims("expand", inputs, axes);
  SaveNode *S = F_->createSave("save", EDN);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  // Expected dims based on the axes above; inserted new dimensions of 1 in
  // every unique axes location, based on the output tensor shape.
  std::vector<size_t> expectedDims = {1, 1, 2, 1, 2, 1};
  auto results = ctx_.get(S->getPlaceholder())->getHandle();
  EXPECT_TRUE(results.dims().vec() == expectedDims);

  // The data should be the same, as this was just a reshape.
  for (size_t i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(results.raw(i), IH.raw(i));
  }
}

TEST_P(InterpAndCPU, Split) {
  auto *inputs =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 6}, "inputs", false);
  ctx_.allocate(inputs)->getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  std::vector<Node *> outputs1;
  F_->createSplit("Split1", inputs, /*outputNum = */ 2, /*axis = */ 2,
                  /*split = */ {}, outputs1);
  std::vector<Node *> outputs2;
  F_->createSplit("Split2", inputs, /*outputNum = */ 2, /*axis = */ 2,
                  /*split = */ {2, 4}, outputs2);
  auto S1 = F_->createSave("save1", outputs1[0]);
  auto S2 = F_->createSave("save2", outputs1[1]);
  auto S3 = F_->createSave("save3", outputs2[0]);
  auto S4 = F_->createSave("save4", outputs2[1]);

  ctx_.allocate(S1->getPlaceholder());
  ctx_.allocate(S2->getPlaceholder());
  ctx_.allocate(S3->getPlaceholder());
  ctx_.allocate(S4->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto result = ctx_.get(S1->getPlaceholder())->getHandle();
  EXPECT_EQ(result.dims().vec(), std::vector<size_t>({1, 2, 3}));
  EXPECT_FLOAT_EQ(result.at({0, 0, 0}), 1);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1}), 2);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2}), 3);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0}), 7);
  EXPECT_FLOAT_EQ(result.at({0, 1, 1}), 8);
  EXPECT_FLOAT_EQ(result.at({0, 1, 2}), 9);

  result = ctx_.get(S2->getPlaceholder())->getHandle();
  EXPECT_EQ(result.dims().vec(), std::vector<size_t>({1, 2, 3}));
  EXPECT_FLOAT_EQ(result.at({0, 0, 0}), 4);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1}), 5);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2}), 6);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0}), 10);
  EXPECT_FLOAT_EQ(result.at({0, 1, 1}), 11);
  EXPECT_FLOAT_EQ(result.at({0, 1, 2}), 12);

  result = ctx_.get(S3->getPlaceholder())->getHandle();
  EXPECT_EQ(result.dims().vec(), std::vector<size_t>({1, 2, 2}));
  EXPECT_FLOAT_EQ(result.at({0, 0, 0}), 1);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1}), 2);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0}), 7);
  EXPECT_FLOAT_EQ(result.at({0, 1, 1}), 8);

  result = ctx_.get(S4->getPlaceholder())->getHandle();
  EXPECT_EQ(result.dims().vec(), std::vector<size_t>({1, 2, 4}));
  EXPECT_FLOAT_EQ(result.at({0, 0, 0}), 3);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1}), 4);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2}), 5);
  EXPECT_FLOAT_EQ(result.at({0, 0, 3}), 6);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0}), 9);
  EXPECT_FLOAT_EQ(result.at({0, 1, 1}), 10);
  EXPECT_FLOAT_EQ(result.at({0, 1, 2}), 11);
  EXPECT_FLOAT_EQ(result.at({0, 1, 3}), 12);
}

TEST_P(Operator, IntRelu) {
  const float splatValue = 10;
  const float scale = 1.0;
  const float rescaleScale = 2.0;
  const int32_t reluOffset = -128;
  const int32_t offset = 5;
  const size_t size = 5;

  auto splatTy = mod_.uniqueType(ElemKind::Int8QTy, {size}, scale, offset);
  auto rescaleTy =
      mod_.uniqueType(ElemKind::Int8QTy, {size}, rescaleScale, offset);

  auto *splat = F_->createSplat("splat", splatTy, splatValue);
  auto *rescale = F_->createRescaleQuantized("rescale", splat, rescaleTy);
  auto *reluOutTy =
      mod_.uniqueType(ElemKind::Int8QTy, {size}, rescaleScale, reluOffset);
  auto *relu = F_->createRELU("relu", rescale, reluOutTy);
  auto *dequantize = F_->createDequantize("dequantize", relu);

  auto *save = F_->createSave("save", dequantize);
  ctx_.allocate(mod_.getPlaceholders());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto result = ctx_.get(save->getPlaceholder())->getHandle();
  float expectedValue = std::max(0.0f, splatValue);
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_EQ(expectedValue, result.raw(i));
  }
}

TEST_P(Operator, IntSplat) {
  const float splatValue = 10;
  const float scale = 1.0;
  const int32_t offset = 5;
  const size_t size = 3;

  auto splatTy = mod_.uniqueType(ElemKind::Int8QTy, {size}, scale, offset);
  auto *splat = F_->createSplat("splat", splatTy, splatValue);
  auto *dequantize = F_->createDequantize("dequantize", splat);

  auto *save = F_->createSave("save", dequantize);
  ctx_.allocate(mod_.getPlaceholders());
  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto result = ctx_.get(save->getPlaceholder())->getHandle();
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_EQ(splatValue, result.raw(i));
  }
}

TEST_P(InterpOnly, Fp16Splat) {
  const float splatValue = 10;
  const size_t size = 3;

  auto splatTy = mod_.uniqueType(ElemKind::Float16Ty, {size});
  auto *splat = F_->createSplat("splat", splatTy, splatValue);

  auto *save = F_->createSave("save", splat);
  ctx_.allocate(mod_.getPlaceholders());
  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto result = ctx_.get(save->getPlaceholder())->getHandle<float16_t>();
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_EQ(float16_t(splatValue), result.raw(i));
  }
}

TEST_P(Operator, GroupConvolution) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 1, 8}, "input", false);
  auto IH = ctx_.allocate(input)->getHandle();
  for (size_t i = 0; i < 2 * 8; i++) {
    IH.raw(i) = i + 1;
  }

  auto filter =
      mod_.createPlaceholder(ElemKind::FloatTy, {6, 1, 1, 4}, "filter", false);
  auto FH = ctx_.allocate(filter)->getHandle();
  for (size_t i = 0; i < 6; i++)
    for (size_t j = 0; j < 4; j++) {
      FH.at({i, 0, 0, j}) = pow(10.0, i);
    }

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {6}, "bias", false);
  ctx_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 2, 1, 6});

  ConvolutionNode *CN =
      F_->createConv("Conv", input, filter, zeroBias, outTy, 1, 1, 0, 2);
  SaveNode *S = F_->createSave("save", CN);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto result = ctx_.get(S->getPlaceholder())->getHandle();

  std::vector<size_t> expectedDims = {1, 2, 1, 6};
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0}), 1 + 2 + 3 + 4);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 1}), (1 + 2 + 3 + 4) * 10);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 2}), (1 + 2 + 3 + 4) * 100);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 3}), (5 + 6 + 7 + 8) * 1000);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 4}), (5 + 6 + 7 + 8) * 10000);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 5}), (5 + 6 + 7 + 8) * 100000);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 0}), 9 + 10 + 11 + 12);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 1}), (9 + 10 + 11 + 12) * 10);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 2}), (9 + 10 + 11 + 12) * 100);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 3}), (13 + 14 + 15 + 16) * 1000);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 4}), (13 + 14 + 15 + 16) * 10000);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 5}), (13 + 14 + 15 + 16) * 100000);
}

/// Check non-square padding for convolution. The first conv has non-square
/// padding, while the second one has zero padding. The second conv's input is
/// the same as the first one's after-padding input. All other parameters of the
/// two convs are the same.
TEST_P(Operator, NonSquarePaddingConvolution) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = ctx_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }

  auto filter =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 2, 1}, "filter", false);
  auto FH = ctx_.allocate(filter)->getHandle();
  for (size_t i = 0; i < 2 * 2 * 2; i++) {
    FH.raw(i) = pow(2.0, i);
  }
  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {2}, "bias", false);
  ctx_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 4, 8, 2});

  ConvolutionNode *CN = F_->createConv("Conv", input, filter, zeroBias, outTy,
                                       {2, 2}, {1, 1}, {0, 2, 1, 3}, 1);
  SaveNode *S = F_->createSave("save", CN);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();
  Tensor &result = *ctx_.get(S->getPlaceholder());

  // Create the reference conv operator whose input is the same as the
  // after-padding-input above.
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 9, 1}, "input1", false);
  ctx_.allocate(input1)->zero();
  auto IH1 = ctx_.get(input1)->getHandle();
  for (size_t i = 0; i < 4; i++)
    for (size_t j = 2; j < 6; j++) {
      IH1.at({0, i, j, 0}) = i * 4 + j - 2 + 1;
    }

  Function *refF = mod_.createFunction("mainRef");
  CN = refF->createConv("Conv1", input1, filter, zeroBias, outTy, {2, 2},
                        {1, 1}, {0, 0, 0, 0}, 1);
  S = refF->createSave("save1", CN);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, refF, ctx_);
  EE_.run();
  Tensor &result1 = *ctx_.get(S->getPlaceholder());

  EXPECT_TRUE(result.isEqual(result1));
}

/// Check non-square padding for AveragePool. The first pool op has non-square
/// padding, while the second one has zero padding. The second pool op's input
/// is the same as the first one's after-padding input. All other parameters of
/// the two convs are the same.
TEST_P(Operator, NonSquarePaddingAveragePool) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = ctx_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }
  auto *Pool = F_->createAvgPool("pool", input, {2, 2}, {1, 1}, {0, 2, 1, 3});
  auto *S = F_->createSave("save", Pool);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();
  Tensor &result = *ctx_.get(S->getPlaceholder());

  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 9, 1}, "input1", false);
  ctx_.allocate(input1)->zero();
  auto IH1 = ctx_.get(input1)->getHandle();
  for (size_t i = 0; i < 4; i++)
    for (size_t j = 2; j < 6; j++) {
      IH1.at({0, i, j, 0}) = i * 4 + j - 2 + 1;
    }

  Function *refF = mod_.createFunction("mainRef");
  Pool = refF->createAvgPool("pool1", input1, 2, 1, 0);
  S = refF->createSave("save1", Pool);
  ctx_.allocate(S->getPlaceholder());
  EE_.compile(CompilationMode::Infer, refF, ctx_);
  EE_.run();
  Tensor &result1 = *ctx_.get(S->getPlaceholder());

  EXPECT_TRUE(result.isEqual(result1));
}

/// Check non-square padding for MaxPool. The first pool op has non-square
/// padding, while the second one has zero padding. The second pool-op's input
/// is the same as the first one's after-padding input. All other parameters
/// of the two convs are the same.
TEST_P(Operator, NonSquarePaddingMaxPool) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = ctx_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }
  auto *Pool = F_->createMaxPool("pool", input, {2, 2}, {1, 1}, {0, 2, 1, 3});
  auto *S = F_->createSave("save", Pool);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  Tensor &result = *ctx_.get(S->getPlaceholder());

  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 9, 1}, "input1", false);
  ctx_.allocate(input1)->zero();
  auto IH1 = ctx_.get(input1)->getHandle();
  for (size_t i = 0; i < 4; i++)
    for (size_t j = 2; j < 6; j++) {
      IH1.at({0, i, j, 0}) = i * 4 + j - 2 + 1;
    }

  Function *refF = mod_.createFunction("mainRef");
  Pool = refF->createMaxPool("pool1", input1, 2, 1, 0);
  S = refF->createSave("save1", Pool);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, refF, ctx_);
  EE_.run();

  Tensor &result1 = *ctx_.get(S->getPlaceholder());

  EXPECT_TRUE(result.isEqual(result1));
}

TEST_P(InterpOnly, FP16AvgPool) {
  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 3, 3, 1}, "input", false);
  ctx_.allocate(input)->getHandle<float16_t>() = {0., 1., 2., 3., 4.,
                                                  5., 6., 7., 8.};
  auto *Pool = F_->createAvgPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", Pool);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto *result = ctx_.get(S->getPlaceholder());
  Tensor out(ElemKind::Float16Ty, {1, 2, 2, 1});
  out.getHandle<float16_t>() = {2., 3., 5., 6.};
  EXPECT_TRUE(out.isEqual(*result));
}

/// Verify that the AvgPool operator works correctly.
TEST_P(Operator, AvgPool) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1}, "input", false);
  ctx_.allocate(input)->getHandle() = {0., 1., 2., 3., 4., 5., 6., 7., 8.};
  auto *Pool = F_->createAvgPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", Pool);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto *result = ctx_.get(S->getPlaceholder());
  Tensor out(ElemKind::FloatTy, {1, 2, 2, 1});
  out.getHandle() = {2., 3., 5., 6.};
  EXPECT_TRUE(out.isEqual(*result));
}

TEST_P(Operator, Int8AvgPool) {
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {1, 3, 3, 1}, 1, 0,
                                       "input", false);
  ctx_.allocate(input)->getHandle<int8_t>() = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  auto *Pool = F_->createAvgPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", Pool);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto result = ctx_.get(S->getPlaceholder())->getHandle<int8_t>();
  Tensor out(ElemKind::Int8QTy, {2, 2}, 1, 0);
  out.getHandle<int8_t>() = {2, 3, 5, 6};
  for (size_t i = 0; i < 2 * 2; i++) {
    EXPECT_EQ(result.raw(i), out.getHandle<int8_t>().raw(i));
  }
}

TEST_P(Operator, MaxPool) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1}, "input", false);
  ctx_.allocate(input)->getHandle() = {0., 1., 2., 3., 4., 5., 6., 7., 8.};
  auto *pool = F_->createMaxPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", pool);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto result = ctx_.get(S->getPlaceholder());
  Tensor out(ElemKind::FloatTy, {1, 2, 2, 1});
  out.getHandle() = {4., 5., 7., 8.};
  EXPECT_TRUE(out.isEqual(*result));
}

TEST_P(InterpOnly, FP16MaxPool) {
  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 3, 3, 1}, "input", false);
  ctx_.allocate(input)->getHandle<float16_t>() = {0., 1., 2., 3., 4.,
                                                  5., 6., 7., 8.};
  auto *pool = F_->createMaxPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", pool);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto result = ctx_.get(S->getPlaceholder());
  Tensor out(ElemKind::Float16Ty, {1, 2, 2, 1});
  out.getHandle<float16_t>() = {4., 5., 7., 8.};
  EXPECT_TRUE(out.isEqual(*result));
}

TEST_P(Operator, Int8MaxPool) {
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {1, 3, 3, 1}, 1, 0,
                                       "input", false);
  ctx_.allocate(input)->getHandle<int8_t>() = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  auto *Pool = F_->createMaxPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", Pool);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto result = ctx_.get(S->getPlaceholder())->getHandle<int8_t>();
  Tensor out(ElemKind::Int8QTy, {2, 2}, 1, 0);
  out.getHandle<int8_t>() = {4, 5, 7, 8};
  for (size_t i = 0; i < 2 * 2; i++) {
    EXPECT_EQ(result.raw(i), out.getHandle<int8_t>().raw(i));
  }
}

TEST_P(InterpAndCPU, Int8Tanh) {
  constexpr size_t size = 10;
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {size}, "input", false);
  ctx_.allocate(input)->getHandle().randomize(-10.0, 10.0, mod_.getPRNG());

  auto *fpTanh = F_->createTanh("fpTanh", input);
  auto *saveFp = F_->createSave("fpSave", fpTanh);
  ctx_.allocate(saveFp->getPlaceholder());

  auto quantizationParams =
      glow::quantization::chooseQuantizationParams(-3.0, 3.0);
  auto quantizeTy =
      mod_.uniqueType(ElemKind::Int8QTy, {size}, quantizationParams.scale,
                      quantizationParams.offset);
  auto *quantize = F_->createQuantize("quantize", input, quantizeTy);

  quantizationParams = glow::quantization::chooseQuantizationParams(-1.0, 1.0);
  auto tanhTy =
      mod_.uniqueType(ElemKind::Int8QTy, {size}, quantizationParams.scale,
                      quantizationParams.offset);

  auto *intTanh = F_->createIntTanh("int8Tanh", quantize, tanhTy);
  auto *dequantize = F_->createDequantize("dequantize", intTanh);
  auto *saveInt = F_->createSave("int8Save", dequantize);
  ctx_.allocate(saveInt->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto fpResult = ctx_.get(saveFp->getPlaceholder())->getHandle();
  auto intResult = ctx_.get(saveInt->getPlaceholder())->getHandle();

  for (size_t i = 0; i < size; i++) {
    EXPECT_NEAR(fpResult.raw(i), intResult.raw(i), 0.05);
  }
}

/// Verify that the Tanh operator works correctly.
TEST_P(Operator, Tanh) {
  constexpr size_t size = 10;
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {size}, "input", false);
  ctx_.allocate(input)->getHandle().randomize(-10.0, 10.0, mod_.getPRNG());

  auto *tanh = F_->createTanh("Tanh", input);
  auto *save = F_->createSave("Save", tanh);
  ctx_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto resultH = ctx_.get(save->getPlaceholder())->getHandle();
  auto inputH = ctx_.get(input)->getHandle();

  for (size_t i = 0; i < size; i++) {
    EXPECT_NEAR(resultH.at({i}), std::tanh(inputH.at({i})), 0.001);
  }
}

/// Verify that a quantized Log works correctly.
TEST_P(InterpAndCPU, Int8Log) {
  constexpr size_t size = 1000;
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {size}, "input", false);

  const float min = 1.0;
  const float max = 100.0;
  ctx_.allocate(input)->getHandle().randomize(min, max, mod_.getPRNG());

  // Input some random data into an fp log.
  auto *fpLog = F_->createLog("fpLog", input);
  auto *saveFp = F_->createSave("fpSave", fpLog);
  ctx_.allocate(saveFp->getPlaceholder());

  // Quantize the input that was also used for the fpLog, and pass it to the
  // quantized Log.
  auto quantizationParams =
      glow::quantization::chooseQuantizationParams(min, max);
  auto quantizeTy =
      mod_.uniqueType(ElemKind::Int8QTy, {size}, quantizationParams.scale,
                      quantizationParams.offset);
  auto *quantize = F_->createQuantize("quantize", input, quantizeTy);

  // Use log of min/max to calculate quantization output params.
  quantizationParams =
      glow::quantization::chooseQuantizationParams(log(min), log(max));
  auto logTy =
      mod_.uniqueType(ElemKind::Int8QTy, {size}, quantizationParams.scale,
                      quantizationParams.offset);

  // Create a quantized log with the quantized version of the input.
  auto *intLog = F_->createLog("int8Log", quantize, logTy);
  auto *dequantize = F_->createDequantize("dequantize", intLog);
  auto *saveInt = F_->createSave("int8Save", dequantize);
  ctx_.allocate(saveInt->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  // Compare the results of the fp and quantized log.
  auto &fpResult = *ctx_.get(saveFp->getPlaceholder());
  auto &intResult = *ctx_.get(saveInt->getPlaceholder());
  EXPECT_TRUE(fpResult.isEqual(intResult, 0.1));
}

/// Check Non-square kernel for conv.
TEST_P(Operator, NonSquareKernelConvolution) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = ctx_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }

  auto filter =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 3, 1}, "filter", false);
  auto FH = ctx_.allocate(filter)->getHandle();
  for (size_t i = 0; i < 1 * 2 * 3; i++) {
    FH.raw(i) = i + 1;
  }

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {1}, "bias", false);
  ctx_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 3, 2, 1});
  ConvolutionNode *CN = F_->createConv("Conv", input, filter, zeroBias, outTy,
                                       {2, 3}, {1, 1}, {0, 0, 0, 0}, 1);
  SaveNode *S = F_->createSave("save", CN);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();
  Tensor &result = *ctx_.get(S->getPlaceholder());

  static const float ref[] = {106, 127, 190, 211, 274, 295};
  for (size_t i = 0; i < 6; i++)
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
}

/// Check Non-square kernel for AveragePool.
TEST_P(InterpAndCPU, NonSquareKernelAveragePool) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = ctx_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }
  auto *Pool = F_->createAvgPool("pool", input, {2, 3}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", Pool);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();
  Tensor &result = *ctx_.get(S->getPlaceholder());

  static const float ref[] = {4, 5, 8, 9, 12, 13};
  for (size_t i = 0; i < 6; i++)
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
}

/// Check Non-square kernel for MaxPool.
TEST_P(InterpAndCPU, NonSquareKernelMaxPool) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = ctx_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }
  auto *Pool = F_->createMaxPool("pool", input, {2, 3}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", Pool);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();
  Tensor &result = *ctx_.get(S->getPlaceholder());

  static const float ref[] = {7, 8, 11, 12, 15, 16};
  for (size_t i = 0; i < 6; i++)
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
}

/// Check Non-square stride for conv.
TEST_P(Operator, NonSquareStrideConvolution) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = ctx_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }

  auto filter =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 2, 1}, "filter", false);
  auto FH = ctx_.allocate(filter)->getHandle();
  for (size_t i = 0; i < 1 * 2 * 2; i++) {
    FH.raw(i) = i + 1;
  }

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {1}, "bias", false);
  ctx_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 2, 2, 1});
  ConvolutionNode *CN = F_->createConv("Conv", input, filter, zeroBias, outTy,
                                       {2, 2}, {3, 2}, {0, 0, 1, 1}, 1);
  SaveNode *S = F_->createSave("save", CN);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();
  Tensor &result = *ctx_.get(S->getPlaceholder());

  static const float ref[] = {44, 64, 41, 47};
  for (size_t i = 0; i < 4; i++)
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
}

/// Check Non-square stride for AveragePool.
TEST_P(InterpAndCPU, NonSquareStrideAveragePool) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = ctx_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }
  auto *Pool = F_->createAvgPool("pool", input, {2, 2}, {3, 2}, {0, 0, 1, 1});
  auto *S = F_->createSave("save", Pool);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();
  Tensor &result = *ctx_.get(S->getPlaceholder());

  static const float ref[] = {3.5, 5.5, 6.75, 7.75};
  for (size_t i = 0; i < 4; i++)
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
}

/// Check Non-square stride for MaxPool.
TEST_P(InterpAndCPU, NonSquareStrideMaxPool) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = ctx_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }
  auto *Pool = F_->createMaxPool("pool", input, {2, 2}, {3, 2}, {0, 0, 1, 1});
  auto *S = F_->createSave("save", Pool);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();
  Tensor &result = *ctx_.get(S->getPlaceholder());

  static const float ref[] = {6, 8, 14, 16};
  for (size_t i = 0; i < 4; i++)
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
}

TEST_P(InterpAndCPU, Int8Sigmoid) {
  constexpr size_t size = 10;
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {size}, "input", false);
  ctx_.allocate(input)->getHandle().randomize(-10.0, 10.0, mod_.getPRNG());

  auto *fpSigmoid = F_->createSigmoid("fpSigmoid", input);
  auto *saveFp = F_->createSave("fpSave", fpSigmoid);
  ctx_.allocate(saveFp->getPlaceholder());

  auto quantizationParams =
      glow::quantization::chooseQuantizationParams(-6.0, 6.0);
  auto quantizeTy =
      mod_.uniqueType(ElemKind::Int8QTy, {size}, quantizationParams.scale,
                      quantizationParams.offset);
  auto *quantize = F_->createQuantize("quantize", input, quantizeTy);

  quantizationParams = glow::quantization::chooseQuantizationParams(0, 1.0);
  auto sigmoidTy =
      mod_.uniqueType(ElemKind::Int8QTy, {size}, quantizationParams.scale,
                      quantizationParams.offset);
  auto *intSigmoid = F_->createIntSigmoid("int8Sigmoid", quantize, sigmoidTy);
  auto *dequantize = F_->createDequantize("dequantize", intSigmoid);
  auto *saveInt = F_->createSave("int8Save", dequantize);
  ctx_.allocate(saveInt->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto fpResult = ctx_.get(saveFp->getPlaceholder())->getHandle();
  auto intResult = ctx_.get(saveInt->getPlaceholder())->getHandle();

  for (size_t i = 0; i < size; i++) {
    EXPECT_NEAR(fpResult.raw(i), intResult.raw(i), 0.05);
  }
}

/// Check that the batch add operator works properly.
TEST_P(Operator, BatchAdd) {
  PseudoRNG PRNG;

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {13, 3, 3}, "A", false);
  ctx_.allocate(input)->getHandle<float>().randomize(-3.0, 3.0, PRNG);
  auto *slice =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 3}, "slice", false);
  ctx_.allocate(slice)->getHandle<float>().randomize(-3.0, 3.0, PRNG);
  auto *batchAdd = F_->createBatchedAdd("batchAdd", input, slice);
  auto *S = F_->createSave("save", batchAdd);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto result = ctx_.get(S->getPlaceholder())->getHandle<float>();
  auto handleInput = ctx_.get(input)->getHandle<float>();
  auto handleSlice = ctx_.get(slice)->getHandle<float>();
  ASSERT_EQ(result.size(), handleInput.size());
  for (size_t idx = 0, end = result.size(); idx != end; ++idx) {
    EXPECT_EQ(result.raw(idx),
              handleInput.raw(idx) + handleSlice.raw(idx % handleSlice.size()));
  }
}

/// Check that the batch add operator works properly for FP16.
TEST_P(InterpOnly, FP16BatchAdd) {
  PseudoRNG PRNG;

  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {13, 3, 3}, "A", false);
  ctx_.allocate(input)->getHandle<float16_t>().randomize(-3.0, 3.0, PRNG);
  auto *slice =
      mod_.createPlaceholder(ElemKind::Float16Ty, {3, 3}, "slice", false);
  ctx_.allocate(slice)->getHandle<float16_t>().randomize(-3.0, 3.0, PRNG);
  auto *batchAdd = F_->createBatchedAdd("batchAdd", input, slice);
  auto *S = F_->createSave("save", batchAdd);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto result = ctx_.get(S->getPlaceholder())->getHandle<float16_t>();
  auto handleInput = ctx_.get(input)->getHandle<float16_t>();
  auto handleSlice = ctx_.get(slice)->getHandle<float16_t>();
  ASSERT_EQ(result.size(), handleInput.size());
  for (size_t idx = 0, end = result.size(); idx != end; ++idx) {
    EXPECT_EQ(result.raw(idx),
              handleInput.raw(idx) + handleSlice.raw(idx % handleSlice.size()));
  }
}

/// Verify that the Sigmoid operator works correctly.
TEST_P(Operator, Sigmoid) {
  constexpr size_t size = 10;
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {size}, "input", false);
  ctx_.allocate(input)->getHandle().randomize(-10.0, 10.0, mod_.getPRNG());

  auto *sigmoid = F_->createSigmoid("sigmoid", input);
  auto *save = F_->createSave("Save", sigmoid);
  ctx_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto RH = ctx_.get(save->getPlaceholder())->getHandle();
  auto inH = ctx_.get(input)->getHandle();

  for (size_t i = 0; i < size; i++) {
    float val = 1 / (1 + std::exp(-inH.at({i})));
    EXPECT_NEAR(RH.at({i}), val, 0.001);
  }
}

TEST_P(InterpAndCPU, IntLookupTable) {
  constexpr size_t size = 6;
  auto *input =
      mod_.createPlaceholder(ElemKind::Int8QTy, {size}, 1, 0, "input", false);
  ctx_.allocate(input)->getHandle<int8_t>() = {0, 1, 2, 3, 4, 5};

  auto outTy = mod_.uniqueType(ElemKind::Int8QTy, {size}, 3, 3);

  // Mapping i -> i.
  std::vector<int8_t> initValues(256);
  for (size_t i = 0; i < 256; ++i) {
    initValues[i] = i - 128;
  }

  auto lookupTable =
      F_->createIntLookupTable("lookupTable", input, initValues, outTy);
  auto save = F_->createSave("save", lookupTable);
  ctx_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto result = ctx_.get(save->getPlaceholder())->getHandle<int8_t>();
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(result.raw(i), i);
  }
}

/// Check that the sequence of extract-batchedadd-concat works.
TEST_P(Operator, testBatchAdd) {
  unsigned numSlices = 10;
  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {numSlices, 10, 10},
                                       "input", false);
  auto *slice =
      mod_.createPlaceholder(ElemKind::FloatTy, {10, 10}, "slice", false);

  ctx_.allocate(input)->getHandle().randomize(-10.0, 10.0, mod_.getPRNG());
  ctx_.allocate(slice)->getHandle().randomize(-10.0, 10.0, mod_.getPRNG());

  std::vector<NodeValue> adds;
  for (size_t i = 0; i < numSlices; i++) {
    auto *ex = F_->createSlice("slice", input, {i, 0, 0}, {i + 1, 10, 10});
    auto *ba = F_->createBatchedAdd("add", ex, slice);
    adds.push_back(ba);
  }

  auto *cc = F_->createConcat("concat", adds, 0);

  // Remove the reference to the graph nodes to allow DCE to remove them.
  adds.clear();

  auto *result = F_->createSave("save", cc);
  ctx_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto RH = ctx_.get(result->getPlaceholder())->getHandle();
  auto IH = ctx_.get(input)->getHandle();
  auto SH = ctx_.get(slice)->getHandle();

  // Check that batched add works as expected.
  for (size_t i = 0; i < numSlices; i++) {
    for (size_t j = 0; j < 10; j++) {
      for (size_t k = 0; k < 10; k++) {
        EXPECT_NEAR(IH.at({i, j, k}) + SH.at({j, k}), RH.at({i, j, k}),
                    0.00001);
      }
    }
  }
}

/// Tests quantized batched-add arithmetic.
TEST_P(Operator, testQuantizedBatchAdd) {
  unsigned numSlices = 10;
  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {numSlices, 10, 10},
                                       "input", false);
  auto *slice =
      mod_.createPlaceholder(ElemKind::FloatTy, {10, 10}, "slice", false);

  ctx_.allocate(input)->getHandle().randomize(-5.0, 5.0, mod_.getPRNG());
  ctx_.allocate(slice)->getHandle().randomize(-5.0, 5.0, mod_.getPRNG());

  // Scale the numbers in the range (-5. .. 5.) to (-50 .. 50).
  auto qInType = mod_.uniqueType(ElemKind::Int8QTy, {numSlices, 10, 10}, .1, 0);
  auto qSliceType2 = mod_.uniqueType(ElemKind::Int8QTy, {10, 10}, .1, 0);
  auto qSliceType3 = mod_.uniqueType(ElemKind::Int8QTy, {1, 10, 10}, .1, 0);

  auto *intInput = F_->createQuantize("qinput", input, qInType);
  auto *intSlice = F_->createQuantize("qslice", slice, qSliceType2);

  std::vector<NodeValue> adds;
  for (size_t i = 0; i < numSlices; i++) {
    auto *ex = F_->createSlice("slice", intInput, {i, 0, 0}, qSliceType3);
    auto *ba = F_->createBatchedAdd("add", ex, intSlice);
    adds.push_back(ba);
  }

  Node *cc = F_->createConcat("concat", adds, 0, qInType);
  cc = F_->createDequantize("dq", cc);
  auto *result = F_->createSave("save", cc);
  ctx_.allocate(result->getPlaceholder());

  // Remove the reference to the graph nodes to allow DCE to remove them.
  adds.clear();

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto RH = ctx_.get(result->getPlaceholder())->getHandle();
  auto IH = ctx_.get(input)->getHandle();
  auto SH = ctx_.get(slice)->getHandle();

  // Check that batched add works as expected.
  for (size_t i = 0; i < numSlices; i++) {
    for (size_t j = 0; j < 10; j++) {
      for (size_t k = 0; k < 10; k++) {
        EXPECT_NEAR(IH.at({i, j, k}) + SH.at({j, k}), RH.at({i, j, k}), 0.1);
      }
    }
  }
}

TEST_P(InterpOnly, SparseLengthsSum) {
  /*
    DATA  = [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]
    INDICES = [2, 0, 1, 2, 0, 0, 0, 0]
    LENGTHS = [2, 0, 2, 1, 3]
    OUTPUT = [
        [5.5, 6.9],
        [0.0, 0.0],
        [6.8, 9.1],
        [1.0, 1.2],
        [3.0, 3.6],
    ]
  */
  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {3, 2}, "data", false);
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {8}, "indices", false);
  auto *lengths =
      mod_.createPlaceholder(ElemKind::Int64ITy, {5}, "lengths", false);

  ctx_.allocate(data)->getHandle() = {
      1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f,
  };
  ctx_.allocate(indices)->getHandle<int64_t>() = {
      2, 0, 1, 2, 0, 0, 0, 0,
  };
  ctx_.allocate(lengths)->getHandle<int64_t>() = {
      2, 0, 2, 1, 3,
  };

  auto R = F_->createSparseLengthsSum("SLS", data, indices, lengths);
  auto *S = F_->createSave("save", R);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  Tensor &result = *ctx_.get(S->getPlaceholder());
  Tensor expected(ElemKind::FloatTy, {5, 2});
  expected.getHandle() = {
      5.5f, 6.9f, 0.0f, 0.0f, 6.8f, 9.1f, 1.0f, 1.2f, 3.0f, 3.6f,
  };

  EXPECT_TRUE(expected.isEqual(result));
}

TEST_P(InterpOnly, SparseLengthsWeightedSum) {
  /*
    DATA  =   [2.0, -0.5, 13]
    WEIGHTS = [3, 1, 0, 0, 0, 0, 2, -0.5]
    INDICES = [1, 0, 2, 0, 1, 2, 2, 0]
    LENGTHS = [3, 0, 3, 2]
    OUTPUT =  [0.5, 0, 0, 25]
  */
  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {3}, "data", false);
  auto *weights =
      mod_.createPlaceholder(ElemKind::FloatTy, {8}, "weights", false);
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {8}, "indices", false);
  auto *lengths =
      mod_.createPlaceholder(ElemKind::Int64ITy, {4}, "lengths", false);

  ctx_.allocate(data)->getHandle() = {
      2.0,
      -0.5,
      13,
  };
  ctx_.allocate(weights)->getHandle() = {
      3, 1, 0, 0, 0, 0, 2, -0.5,
  };
  ctx_.allocate(indices)->getHandle<int64_t>() = {
      1, 0, 2, 0, 1, 2, 2, 0,
  };
  ctx_.allocate(lengths)->getHandle<int64_t>() = {
      3,
      0,
      3,
      2,
  };

  auto R = F_->createSparseLengthsWeightedSum("SLWS", data, weights, indices,
                                              lengths);
  auto *S = F_->createSave("save", R);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  Tensor &result = *ctx_.get(S->getPlaceholder());
  Tensor expected(ElemKind::FloatTy, {4});
  expected.getHandle() = {
      0.5,
      0,
      0,
      25,
  };

  EXPECT_TRUE(expected.isEqual(result));
}

TEST_P(InterpOnly, FP16Reshape) {
  auto *A = mod_.createPlaceholder(ElemKind::Float16Ty, {20, 13}, "A", false);
  auto inputHandle = ctx_.allocate(A)->getHandle<float16_t>();
  inputHandle.randomize(-3.0, 3.0, mod_.getPRNG());

  auto *tr = F_->createReshape("tr", A, {13, 20, 1});
  auto *result = F_->createSave("saveTranspose", tr);
  ctx_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto outputHandle =
      ctx_.get(result->getPlaceholder())->getHandle<float16_t>();
  ASSERT_EQ(outputHandle.size(), inputHandle.size());
  for (size_t idx = 0, end = inputHandle.size(); idx != end; ++idx) {
    EXPECT_EQ(inputHandle.raw(idx), outputHandle.raw(idx));
  }
}

/// Verify that the Reshape operator works correctly.
TEST_P(Operator, Reshape) {
  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {5, 7}, "A", false);
  auto inputHandle = ctx_.allocate(A)->getHandle();
  inputHandle.randomize(-3.0, 3.0, mod_.getPRNG());

  auto *RN = F_->createReshape("reshape", A, {7, 5, 1});
  auto *result = F_->createSave("saveReshape", RN);
  ctx_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto outputHandle = ctx_.get(result->getPlaceholder())->getHandle();
  ASSERT_EQ(outputHandle.size(), inputHandle.size());
  ASSERT_EQ(outputHandle.dims().size(), 3);
  EXPECT_EQ(outputHandle.dims()[0], 7);
  EXPECT_EQ(outputHandle.dims()[1], 5);
  EXPECT_EQ(outputHandle.dims()[2], 1);

  // Check values are still in the same order.
  for (size_t idx = 0, end = inputHandle.size(); idx != end; ++idx) {
    EXPECT_EQ(inputHandle.raw(idx), outputHandle.raw(idx));
  }
}

/// Verify that the Reshape operator works correctly with Int64ITy..
TEST_P(Operator, ReshapeInt) {
  auto *A = mod_.createPlaceholder(ElemKind::Int64ITy, {5, 7}, "A", false);
  auto inputHandle = ctx_.allocate(A)->getHandle<int64_t>();
  inputHandle.randomize<int64_t>(0, 100, mod_.getPRNG());

  auto *RN = F_->createReshape("reshape", A, {7, 5, 1});
  auto *result = F_->createSave("saveReshape", RN);
  ctx_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto outputHandle = ctx_.get(result->getPlaceholder())->getHandle<int64_t>();
  ASSERT_EQ(outputHandle.size(), inputHandle.size());
  ASSERT_EQ(outputHandle.dims().size(), 3);
  EXPECT_EQ(outputHandle.dims()[0], 7);
  EXPECT_EQ(outputHandle.dims()[1], 5);
  EXPECT_EQ(outputHandle.dims()[2], 1);

  // Check values are still in the same order.
  for (size_t idx = 0, end = inputHandle.size(); idx != end; ++idx) {
    EXPECT_EQ(inputHandle.raw(idx), outputHandle.raw(idx));
  }
}

/// Verify that the Select operator works correctly.
TEST_P(Operator, Select) {
  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {5}, "A", false);
  ctx_.allocate(A)->getHandle() = {0.0, 1.0, 1.0, 0.0, 0.0};

  auto SNTy = mod_.uniqueType(ElemKind::FloatTy, {5});
  SplatNode *SN10 = F_->createSplat("zero", SNTy, 10.0);
  SplatNode *SN20 = F_->createSplat("zero", SNTy, 20.0);

  auto *SN = F_->createSelect("select", A, SN10, SN20);
  auto *result = F_->createSave("saveSelect", SN);
  ctx_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto resH = ctx_.get(result->getPlaceholder())->getHandle();
  EXPECT_EQ(resH.at({0}), 20.0);
  EXPECT_EQ(resH.at({1}), 10.0);
  EXPECT_EQ(resH.at({2}), 10.0);
  EXPECT_EQ(resH.at({3}), 20.0);
  EXPECT_EQ(resH.at({4}), 20.0);
}

/// Stack many slices/reshapes together. Some of these may be turned into tensor
/// views stacked onto each other.
TEST_P(Operator, sliceReshape) {
  auto *X = mod_.createPlaceholder(ElemKind::FloatTy, {3, 3}, "X", false);

  auto XH = ctx_.allocate(X)->getHandle();
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      XH.at({i, j}) = i * 3 + j;
    }
  }

  // Do an assortment of slices/reshapes stacked on top of each other.
  auto *SX = F_->createSlice("sliceX", X, {2, 0}, {3, 3});
  auto *RSX = F_->createReshape("reshapeSX", SX, {3});
  auto *SSX = F_->createSlice("sliceSliceX", SX, {0, 2}, {1, 3});
  auto *RSSX = F_->createReshape("reshapeSliceSliceX", SSX, {1});

  auto *resultSX = F_->createSave("saveSX", SX);
  auto *resultRSX = F_->createSave("saveRSX", RSX);
  auto *resultSSX = F_->createSave("saveSSX", SSX);
  auto *resultRSSX = F_->createSave("saveRSSX", RSSX);

  ctx_.allocate(resultSX->getPlaceholder());
  ctx_.allocate(resultRSX->getPlaceholder());
  ctx_.allocate(resultSSX->getPlaceholder());
  ctx_.allocate(resultRSSX->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);

  EE_.run();

  // Verify the slice has the same data as the original X.
  auto SXH = ctx_.get(resultSX->getPlaceholder())->getHandle();
  for (size_t i = 0; i < 3; i++) {
    EXPECT_NEAR(SXH.at({0, i}), XH.at({2, i}), 1E-5);
  }

  // Verify the reshaped slice has the same data as the slice.
  auto RSXH = ctx_.get(resultRSX->getPlaceholder())->getHandle();
  for (size_t i = 0; i < 3; i++) {
    EXPECT_NEAR(SXH.at({0, i}), RSXH.at({i}), 1E-5);
  }

  // Verify the slice of the slice has the same data as the slice.
  auto SSXH = ctx_.get(resultSSX->getPlaceholder())->getHandle();
  EXPECT_NEAR(SXH.at({0, 2}), SSXH.at({0, 0}), 1E-5);

  // Verify the reshape of the slice of the slice has the same data as the slice
  // of the slice.
  auto RSSXH = ctx_.get(resultRSSX->getPlaceholder())->getHandle();
  EXPECT_NEAR(RSSXH.at({0}), SSXH.at({0, 0}), 1E-5);
}

/// Check that the flatten operator produces 2D tensors of the right dimensions.
TEST_P(Operator, Flatten) {
  auto *tensor4D =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 2, 4, 3}, "4D", false);
  ctx_.allocate(tensor4D)->init(Tensor::InitKind::Xavier, 1.0, mod_.getPRNG());

  auto *reshape4Dto2DAxis1 = F_->createFlatten("flat4Dto2Da1", tensor4D, 1);
  EXPECT_EQ(reshape4Dto2DAxis1->dims(0).size(), 2);
  EXPECT_EQ(reshape4Dto2DAxis1->dims(0)[0], 3);
  EXPECT_EQ(reshape4Dto2DAxis1->dims(0)[1], 24);

  auto *reshape4Dto2DAxis2 = F_->createFlatten("flat4Dto2Da2", tensor4D, 2);
  EXPECT_EQ(reshape4Dto2DAxis2->dims(0).size(), 2);
  EXPECT_EQ(reshape4Dto2DAxis2->dims(0)[0], 6);
  EXPECT_EQ(reshape4Dto2DAxis2->dims(0)[1], 12);

  auto *reshape4Dto2DAxis3 = F_->createFlatten("flat4Dto2Da3", tensor4D, 3);
  EXPECT_EQ(reshape4Dto2DAxis3->dims(0).size(), 2);
  EXPECT_EQ(reshape4Dto2DAxis3->dims(0)[0], 24);
  EXPECT_EQ(reshape4Dto2DAxis3->dims(0)[1], 3);

  // Now, let us do the fifth (4) axis.
  // This comes straight from caffe2 because flattening is
  // supported for every axis up and including the rank of a tensor.
  // The rank of this tensor is 4, so axis 4 is fine.
  auto *reshape4Dto2DAxis4 = F_->createFlatten("flat4Dto2Da4", tensor4D, 4);
  EXPECT_EQ(reshape4Dto2DAxis4->dims(0).size(), 2);
  EXPECT_EQ(reshape4Dto2DAxis4->dims(0)[0], 72);
  EXPECT_EQ(reshape4Dto2DAxis4->dims(0)[1], 1);

  // This one is weird because we flatten something that is already flat, but
  // again because flattening is supported for every axis up and including the
  // rank of a tensor, 1D vector means we can flatten it on axis 1.
  auto *tensor1D = mod_.createPlaceholder(ElemKind::FloatTy, {15}, "1D", false);
  ctx_.allocate(tensor1D)->init(Tensor::InitKind::Xavier, 1.0, mod_.getPRNG());

  auto *reshape1Dto2DAxis1 = F_->createFlatten("flat1Dto2D", tensor1D, 1);
  EXPECT_EQ(reshape1Dto2DAxis1->dims(0).size(), 2);
  EXPECT_EQ(reshape1Dto2DAxis1->dims(0)[0], 15);
  EXPECT_EQ(reshape1Dto2DAxis1->dims(0)[1], 1);

  // Save all the reshapes so that the optimizations won't kill the network.
  auto *save1Dto2D = F_->createSave("save1Dto2D", reshape1Dto2DAxis1);
  auto *save4Dto2Da1 = F_->createSave("save4Dto2Da1", reshape4Dto2DAxis1);
  auto *save4Dto2Da2 = F_->createSave("save4Dto2Da2", reshape4Dto2DAxis2);
  auto *save4Dto2Da3 = F_->createSave("save4Dto2Da3", reshape4Dto2DAxis3);
  auto *save4Dto2Da4 = F_->createSave("save4Dto2Da4", reshape4Dto2DAxis4);

  ctx_.allocate(save1Dto2D->getPlaceholder());
  ctx_.allocate(save4Dto2Da1->getPlaceholder());
  ctx_.allocate(save4Dto2Da2->getPlaceholder());
  ctx_.allocate(save4Dto2Da3->getPlaceholder());
  ctx_.allocate(save4Dto2Da4->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);

  EE_.run();

  // Verify the reshapes have the same data as the original value.
  auto tensor4DH = ctx_.get(tensor4D)->getHandle();
  auto save4Dto2Da1H = ctx_.get(save4Dto2Da1->getPlaceholder())->getHandle();
  for (size_t i = 0; i < 72; i++) {
    EXPECT_NEAR(tensor4DH.raw(i), save4Dto2Da1H.raw(i), 1E-5);
  }

  auto save4Dto2Da2H = ctx_.get(save4Dto2Da2->getPlaceholder())->getHandle();
  for (size_t i = 0; i < 72; i++) {
    EXPECT_NEAR(tensor4DH.raw(i), save4Dto2Da2H.raw(i), 1E-5);
  }

  auto save4Dto2Da3H = ctx_.get(save4Dto2Da3->getPlaceholder())->getHandle();
  for (size_t i = 0; i < 72; i++) {
    EXPECT_NEAR(tensor4DH.raw(i), save4Dto2Da3H.raw(i), 1E-5);
  }

  auto save4Dto2Da4H = ctx_.get(save4Dto2Da4->getPlaceholder())->getHandle();
  for (size_t i = 0; i < 72; i++) {
    EXPECT_NEAR(tensor4DH.raw(i), save4Dto2Da4H.raw(i), 1E-5);
  }

  auto tensor1DH = ctx_.get(tensor1D)->getHandle();
  auto save1Dto2DH = ctx_.get(save1Dto2D->getPlaceholder())->getHandle();
  for (size_t i = 0; i < 15; i++) {
    EXPECT_NEAR(tensor1DH.raw(i), save1Dto2DH.raw(i), 1E-5);
  }
}

/// Check that div on Int64ITy/size_t works.
TEST_P(InterpAndCPU, DivSizeT) {
  auto *LHS = mod_.createPlaceholder(ElemKind::Int64ITy, {3, 2}, "LHS", false);
  auto *RHS = mod_.createPlaceholder(ElemKind::Int64ITy, {3, 2}, "RHS", false);
  auto LHSH = ctx_.allocate(LHS)->getHandle<int64_t>();
  auto RHSH = ctx_.allocate(RHS)->getHandle<int64_t>();

  LHSH = {10, 20, 30, 40, 50, 60};
  RHSH = {2, 20, 100, 41, 3, 59};

  auto R = F_->createDiv("div", LHS, RHS);

  auto *result = F_->createSave("save", R);
  ctx_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto H = ctx_.get(result->getPlaceholder())->getHandle<int64_t>();

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 2; j++) {
      EXPECT_EQ(LHSH.at({i, j}) / RHSH.at({i, j}), H.at({i, j}));
    }
  }
}

TEST_P(InterpAndCPU, SigmoidCrossEntropyWithLogits) {
  /*
    LOGITS  = [
      [
        [1.0, 1.2, -0.5],
        [0.1, 0.6, 0.5],
      ],
      [
        [-0.1, -2., 0.3],
        [1, 2, 3],
      ],
    ]
    TARGETS = [
      [
        [0.7, 0.7, 0.7],
        [-0.7, -0.99, 1.0],
      ],
      [
        [0, 0, 0],
        [1, 2, 3],
      ],
    ]
    OUTPUT = [
      [ 0.68687367,  0.97332054],
      [ 0.5418933,  -2.50374103],
    ]
  */
  auto *logits =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 3}, "logits", false);
  auto *targets =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 3}, "targets", false);

  ctx_.allocate(logits)->getHandle() = {1.0f,  1.2f, -0.5f, 0.1f, 0.6f, 0.5f,
                                        -0.1f, -2.f, 0.3f,  1.f,  2.f,  3.f};
  ctx_.allocate(targets)->getHandle() = {0.7f, 0.7f, 0.7f, -0.7f, -0.99f, 1.0f,
                                         0.f,  0.f,  0.f,  1.f,   2.f,    3.f};

  auto *R = F_->createSigmoidCrossEntropyWithLogits("SCEL", logits, targets);

  auto *result = F_->createSave("save", R);
  ctx_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  Tensor expected(ElemKind::FloatTy, {2, 2});
  expected.getHandle() = {
      0.68687367f,
      0.97332054f,
      0.5418933f,
      -2.50374103f,
  };

  EXPECT_TRUE(expected.isEqual(*ctx_.get(result->getPlaceholder())));
}

/// Test the InsertTensor node works correctly.
TEST_P(InterpAndCPU, insertTensorTest) {
  auto SN0Ty = mod_.uniqueType(ElemKind::Int64ITy, {4, 6});
  auto SN1Ty = mod_.uniqueType(ElemKind::Int64ITy, {2, 2});

  // 0 0 0 0 0 0
  // 0 0 0 0 0 0
  // 0 0 0 0 0 0
  // 0 0 0 0 0 0
  Node *SN0 = F_->createSplat("zero", SN0Ty, 0.);

  // 1 1
  // 1 1
  Node *SN1 = F_->createSplat("one", SN1Ty, 1.);

  // 0 0 0 0 0 0
  // 0 1 1 1 1 0
  // 0 1 1 1 1 0
  // 0 0 0 0 0 0
  Node *IN = F_->createInsertTensor("insert", SN0, SN1, /* start */ {1, 1},
                                    /* count */ 2, /* axis */ 1);
  SaveNode *result = F_->createSave("result", IN);
  ctx_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);

  EE_.run();

  // Verify the output looks as expected (pictured above).
  auto resultH = ctx_.get(result->getPlaceholder())->getHandle<int64_t>();
  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 6; j++) {
      int64_t expected = 1;
      if (i == 0 || i == 3 || j == 0 || j == 5)
        expected = 0;
      EXPECT_EQ(resultH.at({i, j}), expected);
    }
  }
}

/// Test RowwiseQuantizedFullyConnected Node.
TEST_P(InterpAndCPU, rowwiseQuantizedFCTest) {
  // In this test we subtract the outputs of a row-wise quantized FC and a
  // floating-point FC and ensure that the error is below some low value.
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 100}, "in", false);
  auto *fc = F_->createFullyConnected(ctx_, "FC", input, 5);

  auto *weights = llvm::cast<Placeholder>(fc->getWeights());
  auto *bias = llvm::cast<Placeholder>(fc->getBias());

  ctx_.allocate(input)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  ctx_.get(bias)->getHandle().randomize(0, 0.1, mod_.getPRNG());
  ctx_.get(weights)->getHandle().randomize(-1.1, 1.1, mod_.getPRNG());

  // Create rowwise quantized FC.
  // The FC fomula is I * W + B, while the RWQFC is I * transpose(W) + B.
  // So get the tranpose of weights from the above FC.
  auto *newWeights = mod_.createConstant(
      ElemKind::FloatTy, {weights->dims()[1], weights->dims()[0]}, "newW");
  ctx_.get(weights)->transpose(&newWeights->getPayload(), {1, 0});

  TypeRef inputTy =
      mod_.uniqueType(ElemKind::Int8QTy, input->dims(), 0.0086, -1);
  TypeRef resTy =
      mod_.uniqueType(ElemKind::Int8QTy, fc->getResult().dims(), 0.05, 49);
  TypeRef biasTy =
      mod_.uniqueType(ElemKind::Int8QTy, bias->dims(), 0.00036, -128);

  auto *inputq = F_->createQuantize("input.q", input, inputTy);
  auto *biasq = F_->createQuantize("bias.q", bias, biasTy);

  auto *fcq = F_->createRowwiseQuantizedFullyConnected(
      "fcq", inputq, newWeights, biasq, resTy);
  auto *dequantRes = F_->createDequantize("dequant", fcq);

  // Subtract the results of the convolution from the rowwise quantized fc.
  auto *sub = F_->createSub("compare", dequantRes, fc);

  auto *save = F_->createSave("save", sub);

  ctx_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto H = ctx_.get(save->getPlaceholder())->getHandle();

  // The difference in the results should be less than 0.05.
  for (int i = 0, e = H.size(); i < e; i++) {
    EXPECT_LE(std::abs(H.raw(i)), 0.05);
  }
}

/// Check the correctness of the SoftMax operator.
/// The semantic of SoftMax is
/// res_i = exp(input_i) / (exp(input_0) + ... + exp(input_N)).
TEST_P(Operator, SoftMax) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 6}, "input", false);
  ctx_.allocate(input)->getHandle<float>() = {1., 3., 2.5, 5., 4., 2.};
  auto *selected =
      mod_.createPlaceholder(ElemKind::Int64ITy, {1, 1}, "expected", false);
  auto *Pool = F_->createSoftMax("pool", input, selected);
  auto *S = F_->createSave("save", Pool);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto result = ctx_.get(S->getPlaceholder());
  Tensor out(ElemKind::FloatTy, {1, 6});
  // Expected results are:
  // sum = exp(input_0) + ... + exp(input_N) = ~245.387
  // res_0 = exp(1) / sum = ~0.011
  // res_1 = exp(3) / sum = ~0.082
  // And so on.
  out.getHandle<float>() = {0.011f, 0.082f, 0.05f, 0.605f, 0.222f, 0.03f};
  EXPECT_TRUE(out.isEqual(*result, 0.001));
}

/// Check that the softmax operator works properly with FP16.
/// See the test that check the SoftMax operator for more details.
TEST_P(InterpOnly, FP16SoftMax) {
  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 6}, "input", false);
  ctx_.allocate(input)->getHandle<float16_t>() = {1., 3., 2.5, 5., 4., 2.};
  auto *selected =
      mod_.createPlaceholder(ElemKind::Int64ITy, {1, 1}, "expected", false);
  auto *Pool = F_->createSoftMax("pool", input, selected);
  auto *S = F_->createSave("save", Pool);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto result = ctx_.get(S->getPlaceholder());
  Tensor out(ElemKind::Float16Ty, {1, 6});
  out.getHandle<float16_t>() = {0.011f, 0.082f, 0.05f, 0.605f, 0.222f, 0.03f};
  EXPECT_TRUE(out.isEqual(*result, 0.001));
}

/// Verify that Quantize, Rescale, Dequantize work correctly together.
TEST_P(Operator, QuantizeSimple) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 1}, "input", true);
  ctx_.allocate(input)->init(Tensor::InitKind::Broadcast, 21, mod_.getPRNG());

  auto *Q = F_->createQuantize(
      "quant", input, mod_.uniqueType(ElemKind::Int8QTy, {1, 1}, 0.25, 4));
  auto *RS = F_->createRescaleQuantized(
      "rescale", Q, mod_.uniqueType(ElemKind::Int8QTy, {1, 1}, 0.5, 11));
  auto *D = F_->createDequantize("dequantize", RS);
  auto *save = F_->createSave("ret", D);
  auto *result = ctx_.allocate(save->getPlaceholder());

  EXPECT_EQ(F_->getNodes().size(), 4);
  EE_.compile(CompilationMode::Infer, F_, ctx_);

  EE_.run();
  EXPECT_EQ(F_->getNodes().size(), 1);

  auto RH = result->getHandle();
  EXPECT_NEAR(RH.at({0, 0}), 21.0, 0.001);
}

/// Check that convertTo node works properly from float16_t to float.
TEST_P(InterpOnly, ConvertFromFloat16ToFloat) {
  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {20, 13}, "A", false);
  auto inputHandle = ctx_.allocate(A)->getHandle<float>();
  inputHandle.randomize(-3.0, 3.0, mod_.getPRNG());

  TypeRef outTy = mod_.uniqueType(ElemKind::Float16Ty, A->dims());

  auto *convertTo = F_->createConvertTo("convertTo", A, outTy);
  auto *result = F_->createSave("save", convertTo);
  ctx_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto *outputTensor = ctx_.get(result->getPlaceholder());
  Tensor convertedInput = ctx_.get(A)->clone();
  convertedInput.convertToType(ElemKind::Float16Ty);
  ASSERT_EQ(outputTensor->size(), inputHandle.size());
  EXPECT_TRUE(convertedInput.isEqual(*outputTensor));
}

/// Check that convertTo node works properly from float to float16_t.
TEST_P(InterpOnly, ConvertFromFloatToFloat16) {
  auto *A = mod_.createPlaceholder(ElemKind::Float16Ty, {20, 13}, "A", false);
  auto inputHandle = ctx_.allocate(A)->getHandle<float16_t>();
  inputHandle.randomize(-3.0, 3.0, mod_.getPRNG());

  TypeRef outTy = mod_.uniqueType(ElemKind::FloatTy, A->dims());

  auto *convertTo = F_->createConvertTo("convertTo", A, outTy);
  auto *result = F_->createSave("save", convertTo);
  ctx_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  auto *outputTensor = ctx_.get(result->getPlaceholder());
  Tensor convertedInput = ctx_.get(A)->clone();
  convertedInput.convertToType(ElemKind::FloatTy);
  ASSERT_EQ(outputTensor->size(), inputHandle.size());
  EXPECT_TRUE(convertedInput.isEqual(*outputTensor));
}

TEST_P(InterpOnly, LengthsToRanges) {
  /*
    LENGTHS = [1, 3, 0, 2]
    OUTPUT =  [[0, 1], [1, 3], [4, 0], [4, 2]]
  */
  auto *lengths =
      mod_.createPlaceholder(ElemKind::Int64ITy, {4}, "lengths", false);

  ctx_.allocate(lengths)->getHandle<int64_t>() = {1, 3, 0, 2};

  auto R = F_->createLengthsToRanges("LTR", lengths);
  auto *S = F_->createSave("save", R);
  ctx_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F_, ctx_);
  EE_.run();

  Tensor &result = *ctx_.get(S->getPlaceholder());
  Tensor expected(ElemKind::Int64ITy, {4, 2});
  expected.getHandle<int64_t>() = {
      0, 1, 1, 3, 4, 0, 4, 2,
  };

  EXPECT_TRUE(expected.isEqual(result));
}

INSTANTIATE_TEST_CASE_P(Interpreter, InterpOnly,
                        ::testing::Values(BackendKind::Interpreter));

INSTANTIATE_TEST_CASE_P(Interpreter, InterpAndCPU,
                        ::testing::Values(BackendKind::Interpreter));

INSTANTIATE_TEST_CASE_P(Interpreter, Operator,
                        ::testing::Values(BackendKind::Interpreter));

#ifdef GLOW_WITH_CPU
INSTANTIATE_TEST_CASE_P(CPU, Operator, ::testing::Values(BackendKind::CPU));
INSTANTIATE_TEST_CASE_P(CPU, InterpAndCPU, ::testing::Values(BackendKind::CPU));
#endif // GLOW_WITH_CPU

#ifdef GLOW_WITH_OPENCL
INSTANTIATE_TEST_CASE_P(OpenCL, Operator,
                        ::testing::Values(BackendKind::OpenCL));
#endif // GLOW_WITH_OPENCL
