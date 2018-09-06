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
  Operator() {
    mod_ = EE_.getModule();
    F_ = mod_.createFunction("main");
  }

protected:
  ExecutionEngine EE_{GetParam()};
  Module mod_;
  Function *F_;
};

class InterpAndCPU : public Operator {};

class InterpOnly : public Operator {};

TEST_P(Operator, pow) {
  auto *X = mod_.createVariable(ElemKind::FloatTy, {1, 1, 3}, "X");
  auto *Y = mod_.createVariable(ElemKind::FloatTy, {2}, "Y");
  auto *Exp = mod_.createVariable(ElemKind::FloatTy, {2}, "Exp");
  X->getPayload().getHandle() = {5, 0.1f, -3};
  Y->getPayload().getHandle() = {2, 100};
  Exp->getPayload().getHandle() = {2, -1};

  auto *Pow1 = F_->createPow("Pow1", X, 2.0);
  auto *Pow2 = F_->createPow("Pow2", Y, 0.5);
  auto *Pow3 = F_->createPow("Pow3", Y, Exp);

  auto *Save1 = F_->createSave("save", Pow1);
  auto *Save2 = F_->createSave("save", Pow2);
  auto *Save3 = F_->createSave("save", Pow3);

  EE_.compile(CompilationMode::Infer, F_);

  EE_.run({}, {});

  auto HX = llvm::cast<Variable>(Save1->getOutput())->getPayload().getHandle();
  EXPECT_NEAR(HX.at({0, 0, 0}), 25, 1E-5);
  EXPECT_NEAR(HX.at({0, 0, 1}), 0.01, 1E-5);
  EXPECT_NEAR(HX.at({0, 0, 2}), 9, 1E-5);

  auto HY = llvm::cast<Variable>(Save2->getOutput())->getPayload().getHandle();
  EXPECT_NEAR(HY.at({0}), sqrt(2.0), 1E-5);
  EXPECT_NEAR(HY.at({1}), 10, 1E-5);

  auto HZ = llvm::cast<Variable>(Save3->getOutput())->getPayload().getHandle();
  EXPECT_NEAR(HZ.at({0}), 4, 1E-5);
  EXPECT_NEAR(HZ.at({1}), 0.01, 1E-5);
}

TEST_P(InterpAndCPU, log) {
  auto *X = mod_.createVariable(ElemKind::FloatTy, {6}, "X");
  auto XH = X->getPayload().getHandle();
  XH = {210030, 600, 4, 0.7f, .005f, 0.000829f};

  auto *LN = F_->createLog("log", X);

  auto *save = F_->createSave("save", LN);

  EE_.compile(CompilationMode::Infer, F_);

  EE_.run({}, {});

  auto saveH = save->getVariable()->getHandle();

  for (size_t i = 0; i < 6; i++) {
    EXPECT_NEAR(saveH.at({i}), log(XH.at({i})), 1E-5);
  }
}

TEST_P(InterpAndCPU, CmpEQ) {
  auto *X = mod_.createVariable(ElemKind::Int64ITy, {2, 7}, "X");
  X->getPayload().getHandle<int64_t>() = {0, 1, 17, 876, 1000, 44444, 9999999,
                                          0, 1, 17, 876, 1000, 44444, 9999999};
  auto *Y = mod_.createVariable(ElemKind::Int64ITy, {2, 7}, "Y");
  Y->getPayload().getHandle<int64_t>() = {1, 2, 16, 900, 1111, 44544, 1999999,
                                          0, 1, 17, 876, 1000, 44444, 9999999};

  auto *cmpEQ = F_->createCmpEQ("cmpEQ", X, Y);
  auto *save = F_->createSave("save", cmpEQ);

  EE_.compile(CompilationMode::Infer, F_);

  EE_.run({}, {});

  auto saveH = save->getVariable()->getHandle<int64_t>();
  for (size_t i = 0; i < 7; ++i) {
    EXPECT_FALSE(saveH.at({0, i}));
  }
  for (size_t i = 0; i < 7; ++i) {
    EXPECT_TRUE(saveH.at({1, i}));
  }
}

TEST_P(Operator, matmul) {
  auto *lhs = mod_.createVariable(ElemKind::FloatTy, {3, 2}, "lhs");
  auto *rhs = mod_.createVariable(ElemKind::FloatTy, {2, 1}, "rhs");
  auto *result = mod_.createVariable(ElemKind::FloatTy, {3, 1}, "result");
  lhs->getPayload().getHandle() = {1, 2, 3, 4, 5, 6};
  rhs->getPayload().getHandle() = {7, 10};

  auto R = F_->createMatMul("MM", lhs, rhs);

  F_->createSave("save", R, result);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto H = result->getPayload().getHandle();
  EXPECT_NEAR(H.at({0, 0}), 27, 0.001);
  EXPECT_NEAR(H.at({1, 0}), 61, 0.001);
  EXPECT_NEAR(H.at({2, 0}), 95, 0.001);
}

/// Test that the broadcasted batch mat mul operator works as expected.
TEST_P(Operator, BroadcastedBatchMatMul) {
  auto *lhs = mod_.createVariable(ElemKind::FloatTy, {2, 3, 2}, "lhs");
  auto *rhs = mod_.createVariable(ElemKind::FloatTy, {2, 1}, "rhs");
  auto *result = mod_.createVariable(ElemKind::FloatTy, {2, 3, 1}, "result");
  lhs->getPayload().getHandle() = {1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6};
  rhs->getPayload().getHandle() = {7, 10};

  auto *R = F_->createBroadcastedBatchMatMul("BMM", lhs, rhs);

  F_->createSave("save", R, result);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto H = result->getPayload().getHandle();
  EXPECT_NEAR(H.at({0, 0, 0}), 27, 0.001);
  EXPECT_NEAR(H.at({0, 1, 0}), 61, 0.001);
  EXPECT_NEAR(H.at({0, 2, 0}), 95, 0.001);
  EXPECT_NEAR(H.at({1, 0, 0}), -27, 0.001);
  EXPECT_NEAR(H.at({1, 1, 0}), -61, 0.001);
  EXPECT_NEAR(H.at({1, 2, 0}), -95, 0.001);
}

TEST_P(Operator, ParallelBatchMatMul) {
  auto *lhs = mod_.createVariable(ElemKind::FloatTy, {2, 3, 2}, "lhs");
  auto *rhs = mod_.createVariable(ElemKind::FloatTy, {2, 2, 1}, "rhs");
  auto *result = mod_.createVariable(ElemKind::FloatTy, {2, 3, 1}, "result");
  lhs->getPayload().getHandle() = {1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6};
  rhs->getPayload().getHandle() = {7, 10, 12, -1};

  auto *R = F_->createParallelBatchMatMul("BMM", lhs, rhs);

  F_->createSave("save", R, result);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto H = result->getPayload().getHandle();
  EXPECT_NEAR(H.at({0, 0, 0}), 27, 0.001);
  EXPECT_NEAR(H.at({0, 1, 0}), 61, 0.001);
  EXPECT_NEAR(H.at({0, 2, 0}), 95, 0.001);
  EXPECT_NEAR(H.at({1, 0, 0}), -10, 0.001);
  EXPECT_NEAR(H.at({1, 1, 0}), -32, 0.001);
  EXPECT_NEAR(H.at({1, 2, 0}), -54, 0.001);
}

TEST_P(Operator, batchedReduceAdd) {
  auto *batch = mod_.createVariable(ElemKind::FloatTy, {2, 4}, "batch");
  auto *result = mod_.createVariable(ElemKind::FloatTy, {4}, "result");
  batch->getPayload().getHandle() = {10, 20, 30, 40, 1, 2, 3, 4};

  auto R = F_->createBatchedReduceAdd("reduce.add", batch, /* axis */ 0);

  F_->createSave("save", R, result);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto H = result->getPayload().getHandle();
  EXPECT_NEAR(H.at({0}), 11, 0.001);
  EXPECT_NEAR(H.at({1}), 22, 0.001);
  EXPECT_NEAR(H.at({2}), 33, 0.001);
  EXPECT_NEAR(H.at({3}), 44, 0.001);
}

TEST_P(InterpAndCPU, batchedReduceAddWithAxis) {
  auto *batch = mod_.createVariable(ElemKind::FloatTy, {2, 3, 2}, "batch");
  auto *result = mod_.createVariable(ElemKind::FloatTy, {2, 2}, "result");
  batch->getPayload().getHandle() = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  auto R = F_->createBatchedReduceAdd("reduce.add", batch, /* axis */ 1);

  F_->createSave("save", R, result);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto H = result->getPayload().getHandle();
  EXPECT_NEAR(H.at({0, 0}), 6, 0.001);
  EXPECT_NEAR(H.at({0, 1}), 9, 0.001);
  EXPECT_NEAR(H.at({1, 0}), 24, 0.001);
  EXPECT_NEAR(H.at({1, 1}), 27, 0.001);
}

TEST_P(InterpAndCPU, batchedReduceAddQuantized) {
  auto BT = mod_.uniqueType(ElemKind::Int8QTy, {3, 8}, 0.5, 3);
  auto OT = mod_.uniqueType(ElemKind::Int8QTy, {8}, 2.0, -1);

  auto *batch = mod_.createVariable(ElemKind::Int8QTy, {3, 8}, BT->getScale(),
                                    BT->getOffset(), "batch");
  auto *result = mod_.createVariable(ElemKind::Int8QTy, {8}, OT->getScale(),
                                     OT->getOffset(), "result");

  batch->getPayload().getHandle<int8_t>() = {
      27, -31, 16,  7,  20, 34, -2, 8,   -10, 83, 29,  -17,
      19, 13,  -11, -9, 50, 58, 0,  -20, -72, 43, -25, -1};

  auto BH = batch->getHandle<int8_t>();
  auto OH = result->getHandle<int8_t>();

  auto *R =
      F_->createBatchedReduceAdd("batched.reduce.add", OT, batch, /* axis */ 0);

  F_->createSave("save", R, result);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

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

  auto *batch = mod_.createVariable(ElemKind::Int8QTy, {2, 3, 4},
                                    BT->getScale(), BT->getOffset(), "batch");
  auto *result = mod_.createVariable(ElemKind::Int8QTy, {2, 4}, OT->getScale(),
                                     OT->getOffset(), "result");

  batch->getPayload().getHandle<int8_t>() = {
      27, -31, 16,  7,  20, 34, -2, 8,   -10, 83, 29,  -17,
      19, 13,  -11, -9, 50, 58, 0,  -20, -72, 43, -25, -1};

  auto BH = batch->getHandle<int8_t>();
  auto OH = result->getHandle<int8_t>();

  auto *R =
      F_->createBatchedReduceAdd("batched.reduce.add", OT, batch, /* axis */ 1);
  F_->createSave("save", R, result);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

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
  auto *batch = mod_.createVariable(ElemKind::FloatTy, {2, 4}, "batch");
  auto *result = mod_.createVariable(ElemKind::FloatTy, {4}, "result");
  batch->getPayload().getHandle() = {10, 20, 30, 40, 1, 2, 3, 4};

  auto R = F_->createBatchedReduceMean("reduce.add", batch, /* axis */ 0);

  F_->createSave("save", R, result);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto H = result->getPayload().getHandle();
  EXPECT_NEAR(H.at({0}), 5.5, 0.001);
  EXPECT_NEAR(H.at({1}), 11.0, 0.001);
  EXPECT_NEAR(H.at({2}), 16.5, 0.001);
  EXPECT_NEAR(H.at({3}), 22.0, 0.001);
}

TEST_P(InterpAndCPU, batchedReduceMeanWithAxis) {
  auto *batch = mod_.createVariable(ElemKind::FloatTy, {2, 3, 2}, "batch");
  auto *result = mod_.createVariable(ElemKind::FloatTy, {2, 2}, "result");
  batch->getPayload().getHandle() = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  auto R = F_->createBatchedReduceMean("reduce.add", batch, /* axis */ 1);

  F_->createSave("save", R, result);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto H = result->getPayload().getHandle();
  EXPECT_NEAR(H.at({0, 0}), 2.0, 0.001);
  EXPECT_NEAR(H.at({0, 1}), 3.0, 0.001);
  EXPECT_NEAR(H.at({1, 0}), 8.0, 0.001);
  EXPECT_NEAR(H.at({1, 1}), 9.0, 0.001);
}

TEST_P(InterpAndCPU, batchedReduceMeanQuantized) {
  auto BT = mod_.uniqueType(ElemKind::Int8QTy, {3, 8}, 0.5, 3);
  auto OT = mod_.uniqueType(ElemKind::Int8QTy, {8}, 2.0, -1);

  auto *batch = mod_.createVariable(ElemKind::Int8QTy, {3, 8}, BT->getScale(),
                                    BT->getOffset(), "batch");
  auto *result = mod_.createVariable(ElemKind::Int8QTy, {8}, OT->getScale(),
                                     OT->getOffset(), "result");

  batch->getPayload().getHandle<int8_t>() = {
      27, -31, 16,  7,  20, 34, -2, 8,   -10, 83, 29,  -17,
      19, 13,  -11, -9, 50, 58, 0,  -20, -72, 43, -25, -1};

  auto BH = batch->getHandle<int8_t>();
  auto OH = result->getHandle<int8_t>();

  auto *R = F_->createBatchedReduceMean("batched.reduce.add", OT, batch,
                                        /* axis */ 0);

  F_->createSave("save", R, result);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

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

  auto *batch = mod_.createVariable(ElemKind::Int8QTy, {2, 3, 4},
                                    BT->getScale(), BT->getOffset(), "batch");
  auto *result = mod_.createVariable(ElemKind::Int8QTy, {2, 4}, OT->getScale(),
                                     OT->getOffset(), "result");

  batch->getPayload().getHandle<int8_t>() = {
      27, -31, 16,  7,  20, 34, -2, 8,   -10, 83, 29,  -17,
      19, 13,  -11, -9, 50, 58, 0,  -20, -72, 43, -25, -1};

  auto BH = batch->getHandle<int8_t>();
  auto OH = result->getHandle<int8_t>();

  auto *R = F_->createBatchedReduceMean("batched.reduce.add", OT, batch,
                                        /* axis */ 1);
  F_->createSave("save", R, result);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

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

TEST_P(Operator, batchedBatchedAdd) {
  auto *batch = mod_.createVariable(ElemKind::FloatTy, {2, 3, 3}, "batch");
  auto *added = mod_.createVariable(ElemKind::FloatTy, {3, 3}, "added");
  auto *result = mod_.createVariable(ElemKind::FloatTy, {2, 3, 3}, "result");

  batch->getPayload().getHandle() = {9, 8, 7, 6, 5,  4,  3,  4,  5,
                                     6, 7, 8, 9, 10, 11, 12, 13, 14};
  added->getPayload().getHandle().clear(1.0);

  auto R = F_->createBatchedAdd("batch.add", batch, added);
  F_->createSave("save", R, result);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto H = result->getPayload().getHandle();
  EXPECT_NEAR(H.at({0, 0, 0}), 10, 0.001);
  EXPECT_NEAR(H.at({0, 0, 1}), 9, 0.001);
  EXPECT_NEAR(H.at({0, 0, 2}), 8, 0.001);
  EXPECT_NEAR(H.at({0, 1, 0}), 7, 0.001);
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

  auto *B = mod_.createVariable(ElemKind::FloatTy, dims_B, "B");
  auto *QB = mod_.createVariable(ElemKind::Int8QTy, dims_B, 1.1, -2, "QB");
  auto H_B = B->getPayload().getHandle();
  auto H_QB = QB->getPayload().getHandle<int8_t>();
  H_B = {20, 10};
  H_QB = {35, -18};

  const unsigned axis = 0;

  auto R = F_->createBroadcast("broadcasted", B, dims_A, axis);
  auto QR = F_->createBroadcast("broadcastedQ", QB, dims_A, axis);
  auto *broadcasted = mod_.createVariable(ElemKind::FloatTy, dims_A, "A");
  auto *broadcastedQ =
      mod_.createVariable(ElemKind::Int8QTy, dims_A, 1.1, -2, "QA");
  F_->createSave("save", R, broadcasted);
  F_->createSave("saveQ", QR, broadcastedQ);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto broadcastedBHandle = broadcasted->getPayload().getHandle();
  auto broadcastedQBHandle = broadcastedQ->getPayload().getHandle<int8_t>();
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

  auto *B = mod_.createVariable(ElemKind::FloatTy, dims_B, "B");
  auto *QB = mod_.createVariable(ElemKind::Int8QTy, dims_B, 0.8, 3, "QB");
  auto H_B = B->getPayload().getHandle();
  auto H_QB = QB->getPayload().getHandle<int8_t>();
  H_B = {20, 10};
  H_QB = {-8, 41};

  const unsigned axis = 1;

  auto R = F_->createBroadcast("broadcasted", B, dims_A, axis);
  auto QR = F_->createBroadcast("broadcastedQ", QB, dims_A, axis);
  auto *broadcasted = mod_.createVariable(ElemKind::FloatTy, dims_A, "A");
  auto *broadcastedQ =
      mod_.createVariable(ElemKind::Int8QTy, dims_A, 0.8, 3, "QB");
  F_->createSave("save", R, broadcasted);
  F_->createSave("saveQ", QR, broadcastedQ);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto broadcastedBHandle = broadcasted->getPayload().getHandle();
  auto broadcastedQBHandle = broadcastedQ->getPayload().getHandle<int8_t>();
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
  auto *A = mod_.createVariable(ElemKind::FloatTy, {2, 2}, "A");
  A->getPayload().getHandle() = {1.0, 2.0, 3.0, 4.0};
  auto *B = mod_.createVariable(ElemKind::FloatTy, {2, 2}, "B");
  B->getPayload().getHandle() = {5.0, 6.0, 7.0, 8.0};

  // Create the weights.
  auto *AW = mod_.createVariable(ElemKind::FloatTy, {1}, "AW");
  AW->getPayload().getHandle() = {0.1f};
  auto *BW = mod_.createVariable(ElemKind::FloatTy, {1}, "BW");
  BW->getPayload().getHandle() = {10.0};

  // Create the weighted sum with the data and weights, and save it.
  auto *WS = F_->createWeightedSum("ws", {A, B}, {AW, BW});
  auto *save = F_->createSave("save", WS);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  // Verify the weighted sum was correctly calculated.
  auto resultH = save->getVariable()->getHandle();
  EXPECT_NEAR(resultH.at({0, 0}), 50.1, 1E-5);
  EXPECT_NEAR(resultH.at({0, 1}), 60.2, 1E-5);
  EXPECT_NEAR(resultH.at({1, 0}), 70.3, 1E-5);
  EXPECT_NEAR(resultH.at({1, 1}), 80.4, 1E-5);
}

TEST_P(Operator, minElem) {
  PseudoRNG PRNG;
  unsigned len = 5;

  auto *LHS = mod_.createVariable(ElemKind::FloatTy, {len}, "lhs");
  auto *RHS = mod_.createVariable(ElemKind::FloatTy, {len}, "rhs");
  auto *min = F_->createMin("min", LHS, RHS);
  auto *save = F_->createSave("min", min);

  LHS->getHandle().randomize(-10, 10, PRNG);
  RHS->getHandle().randomize(-10, 10, PRNG);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto resultH = save->getVariable()->getHandle();
  auto LHSH = LHS->getHandle();
  auto RHSH = RHS->getHandle();

  for (size_t i = 0; i < len; i++) {
    EXPECT_EQ(resultH.raw(i), std::min(LHSH.raw(i), RHSH.raw(i)));
  }
}

TEST_P(Operator, TopK) {
  auto *inp = mod_.createVariable(ElemKind::FloatTy, {3, 1, 5}, "input");
  auto *values = mod_.createVariable(ElemKind::FloatTy, {3, 1, 3}, "values");
  auto *indices = mod_.createVariable(ElemKind::Int64ITy, {3, 1, 3}, "indices");

  inp->getPayload().getHandle() = {
      28, 4, 411, 19, 42, 0.4f, 0.4f, 0.4f, -0.4f, 0.45f, 7, 5, 9, 8, 100,
  };

  auto R = F_->createTopK("TopK", inp, 3);

  F_->createSave("save.values", {R, 0}, values);
  F_->createSave("save.indices", {R, 1}, indices);

  EE_.compile(CompilationMode::Infer, F_);

  EE_.run({}, {});

  auto V = values->getPayload().getHandle();
  auto I = indices->getPayload().getHandle<int64_t>();

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
  auto *inp1 = mod_.createVariable(ElemKind::FloatTy, {2, 1, 3}, "input");
  auto *inp2 = mod_.createVariable(ElemKind::FloatTy, {2, 1, 3}, "input");
  auto *values = mod_.createVariable(ElemKind::FloatTy, {4, 1, 2}, "values");
  auto *indices = mod_.createVariable(ElemKind::Int64ITy, {4, 1, 2}, "indices");

  inp1->getPayload().getHandle() = {1, 2, 3, 17.4f, -0.1f, -10.1f};
  inp2->getPayload().getHandle() = {1, 2, -3, -17.4f, -0.1f, -10.1f};

  auto *R1 = F_->createTopK("TopK1", inp1, 2);
  auto *R2 = F_->createTopK("TopK2", inp2, 2);

  // Concat the values and indices separately, both on the 0th dimension,
  // matching the shapes of the values and indices variables above.
  auto *CV =
      F_->createConcat("Concat.Values", {R1->getValues(), R2->getValues()}, 0);
  auto *CI = F_->createConcat("Concat.Indices",
                              {R1->getIndices(), R2->getIndices()}, 0);

  F_->createSave("Save.Values", CV, values);
  F_->createSave("Save.Indices", CI, indices);

  EE_.compile(CompilationMode::Infer, F_);

  EE_.run({}, {});

  auto V = values->getPayload().getHandle();
  auto I = indices->getPayload().getHandle<int64_t>();

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
  auto *inp0 = mod_.createVariable(ElemKind::FloatTy, {1, 2}, "input0");
  auto *inp1 = mod_.createVariable(ElemKind::FloatTy, {1, 2}, "input1");
  auto *inp2 = mod_.createVariable(ElemKind::FloatTy, {1, 2}, "input1");
  auto *res0 = mod_.createVariable(ElemKind::FloatTy, {1, 2}, "res0");
  auto *res1 = mod_.createVariable(ElemKind::FloatTy, {1, 2}, "res1");
  auto *res2 = mod_.createVariable(ElemKind::FloatTy, {1, 2}, "res1");
  auto *rot = mod_.createVariable(ElemKind::FloatTy, {2, 2}, "rot");

  float deg = 45.0 / 180.0 * 3.1415926;
  // Use the rotation matrix to manipulate some values.
  // https://en.wikipedia.org/wiki/Rotation_matrix
  rot->getPayload().getHandle() = {
      cosf(deg),
      -sinf(deg),
      sinf(deg),
      cosf(deg),
  };

  // Some test vectors.
  inp0->getPayload().getHandle() = {1, 4};
  inp1->getPayload().getHandle() = {14, 2};
  inp2->getPayload().getHandle() = {5, 2};

  auto *A0 = F_->createMatMul("m0", inp0, rot);
  auto *A1 = F_->createMatMul("m1", inp1, rot);
  auto *A2 = F_->createMatMul("m2", inp2, rot);

  F_->createSave("save.values", A0, res0);
  F_->createSave("save.values", A1, res1);
  F_->createSave("save.values", A2, res2);

  EE_.compile(CompilationMode::Infer, F_);

  EE_.run({}, {});

  auto R0 = res0->getPayload().getHandle();
  auto R1 = res1->getPayload().getHandle();
  auto R2 = res2->getPayload().getHandle();

  EXPECT_FLOAT_EQ(R0.at({0, 0}), 3.5355339);
  EXPECT_FLOAT_EQ(R0.at({0, 1}), 2.1213205);
  EXPECT_FLOAT_EQ(R1.at({0, 0}), 11.313709);
  EXPECT_FLOAT_EQ(R1.at({0, 1}), -8.485281);
  EXPECT_FLOAT_EQ(R2.at({0, 0}), 4.9497476);
  EXPECT_FLOAT_EQ(R2.at({0, 1}), -2.1213202);
}

// Check the TopK operator for the special case of K=1.
TEST_P(InterpAndCPU, TopK1) {
  auto *inp = mod_.createVariable(ElemKind::FloatTy, {3, 1, 5}, "input");
  auto *values = mod_.createVariable(ElemKind::FloatTy, {3, 1, 1}, "values");
  auto *indices = mod_.createVariable(ElemKind::Int64ITy, {3, 1, 1}, "indices");

  inp->getPayload().getHandle() = {
      0, 18, 7, 16, 5, 14, 33, 2, 41, 0, 1, -23, 34, 4, -5,
  };

  auto R = F_->createTopK("TopK", inp, 1);

  F_->createSave("save.values", {R, 0}, values);
  F_->createSave("save.indices", {R, 1}, indices);

  EE_.compile(CompilationMode::Infer, F_);

  EE_.run({}, {});

  auto V = values->getPayload().getHandle();
  auto I = indices->getPayload().getHandle<int64_t>();

  EXPECT_FLOAT_EQ(V.at({0, 0, 0}), 18);
  EXPECT_EQ(I.at({0, 0, 0}), 1);
  EXPECT_FLOAT_EQ(V.at({1, 0, 0}), 41);
  EXPECT_EQ(I.at({1, 0, 0}), 3);
  EXPECT_FLOAT_EQ(V.at({2, 0, 0}), 34);
  EXPECT_EQ(I.at({2, 0, 0}), 2);
}

TEST_P(InterpAndCPU, QuantizedTopK) {
  auto *INV =
      mod_.createVariable(ElemKind::Int8QTy, {3, 1, 5}, 1.2, 5, "input");
  auto *OV =
      mod_.createVariable(ElemKind::Int8QTy, {3, 1, 3}, 1.2, 5, "values");
  auto *IV = mod_.createVariable(ElemKind::Int64ITy, {3, 1, 3}, "indices");

  INV->getPayload().getHandle<int8_t>() = {
      -12, -28, -7, 8, -93, 0, 10, 3, -1, 10, -2, 3, -2, 3, 3,
  };

  auto TK = F_->createTopK("TopK", INV, 3);

  F_->createSave("save.values", TK->getValues(), OV);
  F_->createSave("save.indices", TK->getIndices(), IV);

  EE_.compile(CompilationMode::Infer, F_);

  EE_.run({}, {});

  auto VH = OV->getPayload().getHandle<int8_t>();
  auto IH = IV->getPayload().getHandle<int64_t>();

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
  auto *data = mod_.createVariable(ElemKind::FloatTy, {3, 2}, "data");
  auto *indices = mod_.createVariable(ElemKind::Int64ITy, {2, 4}, "indices");
  auto *result = mod_.createVariable(ElemKind::FloatTy, {2, 4, 2}, "result");

  data->getPayload().getHandle() = {
      1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f,
  };
  indices->getPayload().getHandle<int64_t>() = {
      0, 1, 0, 1, 1, 2, 2, 0,
  };

  auto R = F_->createGather("gather", data, indices);

  F_->createSave("save", R, result);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto H = result->getPayload().getHandle();

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
  auto *A = mod_.createVariable(ElemKind::FloatTy, {20, 13}, "A",
                                VisibilityKind::Public);
  A->getHandle().randomize(-3.0, 3.0, mod_.getPRNG());

  auto *tr = F_->createTranspose("tr", A, {1, 0});
  auto *result = F_->createSave("saveTranspose", tr);
  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});
  Tensor dest(ElemKind::FloatTy, {13, 20});
  A->getPayload().transpose(&dest, {1, 0});
  EXPECT_TRUE(result->getVariable()->getPayload().isEqual(dest));
}

/// Check if the code generation of transposes
/// is correct for tensors with 3 dimensions.
/// Note: This assumes that Tensor::transpose is correct.
TEST_P(Operator, Transpose3Dims) {
  constexpr size_t dims[] = {20, 13, 7};
  auto *A =
      mod_.createVariable(ElemKind::FloatTy, dims, "A", VisibilityKind::Public);
  A->getHandle().randomize(-3.0, 3.0, mod_.getPRNG());

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
        ++nbOfShuffle;
      }
    }
  }

  // We should have exactly 6 possible permutations for 3 dimensions.
  EXPECT_EQ(6, nbOfShuffle);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  for (int i = 0; i < 6; ++i) {
    Tensor dest(ElemKind::FloatTy, {dims[shuffles[i][0]], dims[shuffles[i][1]],
                                    dims[shuffles[i][2]]});
    A->getPayload().transpose(&dest, shuffles[i]);
    EXPECT_TRUE(savedTransposes[i]->getVariable()->getPayload().isEqual(dest));
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
  auto *data = mod_.createVariable(ElemKind::Int64ITy, {3, 2}, "data");
  auto *indices = mod_.createVariable(ElemKind::Int64ITy, {2, 4}, "indices");
  auto *result = mod_.createVariable(ElemKind::Int64ITy, {2, 4, 2}, "result");

  data->getPayload().getHandle<int64_t>() = {
      1, 2, 3, 4, 5, 6,
  };
  indices->getPayload().getHandle<int64_t>() = {
      0, 1, 0, 1, 1, 2, 2, 0,
  };

  auto R = F_->createGather("gather", data, indices);

  F_->createSave("save", R, result);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto H = result->getPayload().getHandle<int64_t>();

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
  auto *data = mod_.createVariable(ElemKind::FloatTy, {3, 4}, "data");
  auto *indices = mod_.createVariable(ElemKind::Int64ITy, {2}, "indices");
  auto *result = mod_.createVariable(ElemKind::FloatTy, {3, 2}, "result");

  data->getPayload().getHandle() = {
      1.0f, 1.2f, 2.4f, 4.5f, 2.3f, 3.4f, 3.6f, 2.3f, 4.5f, 5.7f, 1.2f, 4.5f,
  };
  indices->getPayload().getHandle<int64_t>() = {
      0,
      2,
  };

  // Create a batched gather (a single batch dimension).
  auto R = F_->createGather("gather", data, indices, 1);

  F_->createSave("save", R, result);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto H = result->getPayload().getHandle();
  EXPECT_FLOAT_EQ(H.at({0, 0}), 1.0);
  EXPECT_FLOAT_EQ(H.at({0, 1}), 2.4);
  EXPECT_FLOAT_EQ(H.at({1, 0}), 2.3);
  EXPECT_FLOAT_EQ(H.at({1, 1}), 3.6);
  EXPECT_FLOAT_EQ(H.at({2, 0}), 4.5);
  EXPECT_FLOAT_EQ(H.at({2, 1}), 1.2);
}

TEST_P(Operator, ScatterAssign) {
  auto *data = mod_.createVariable(ElemKind::FloatTy, {5, 2}, "data");
  auto *indices = mod_.createVariable(ElemKind::Int64ITy, {2}, "indices");
  auto *slices = mod_.createVariable(ElemKind::FloatTy, {2, 2}, "slices");
  auto *result = mod_.createVariable(ElemKind::FloatTy, {5, 2}, "result");

  data->getPayload().getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  indices->getPayload().getHandle<int64_t>() = {1, 3};
  slices->getPayload().getHandle() = {-3, -4, -7, -8};

  auto R = F_->createScatterAssign("scatterassign", data, indices, slices);

  F_->createSave("save", R, result);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto H = result->getPayload().getHandle();

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
  auto *data = mod_.createVariable(ElemKind::FloatTy, {5, 2}, "data");
  auto *indices = mod_.createVariable(ElemKind::Int64ITy, {2}, "indices");
  auto *slices = mod_.createVariable(ElemKind::FloatTy, {2, 2}, "slices");
  auto *result = mod_.createVariable(ElemKind::FloatTy, {5, 2}, "result");

  data->getPayload().getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  indices->getPayload().getHandle<int64_t>() = {1, 3};
  slices->getPayload().getHandle() = {-3, -4, -7, -8};

  auto qParams = glow::quantization::chooseQuantizationParams(-11, 11);
  auto dataTy =
      mod_.uniqueType(ElemKind::Int8QTy, {5, 2}, qParams.scale, qParams.offset);
  auto slicesTy =
      mod_.uniqueType(ElemKind::Int8QTy, {2, 2}, qParams.scale, qParams.offset);

  auto *dataQ = F_->createQuantize("quantizeQ", data, dataTy);
  auto *slicesQ = F_->createQuantize("quantizeS", slices, slicesTy);
  auto *SA = F_->createScatterAssign("scatterassign", dataQ, indices, slicesQ);
  auto *DQ = F_->createDequantize("dequantize", SA);

  F_->createSave("save", DQ, result);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto H = result->getPayload().getHandle();

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
  auto *A = mod_.createVariable(ElemKind::FloatTy, {1, 4}, "A",
                                VisibilityKind::Public);
  auto *B = mod_.createVariable(ElemKind::FloatTy, {1, 4}, "B",
                                VisibilityKind::Public);
  A->getPayload().getHandle() = {1, 1.2f, 0.5f, 1.3f};
  B->getPayload().getHandle() = {1.8f, 5.2f, 3.5f, 11.3f};

  auto qType = mod_.uniqueType(ElemKind::Int8QTy, {1, 4}, 0.05, -138);
  auto *quantizeA = F_->createQuantize("quantize", A, qType);
  auto *quantizeB = F_->createQuantize("quantize", B, qType);
  auto *add = F_->createAdd("add", quantizeA, quantizeB);
  auto *dequantize = F_->createDequantize("dequantize", add);
  auto *result = F_->createSave("save", dequantize);
  auto *fpAdd = F_->createAdd("fpAdd", A, B);
  auto *fpResult = F_->createSave("fpSave", fpAdd);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  EXPECT_TRUE(result->getVariable()->getPayload().isEqual(
      fpResult->getVariable()->getPayload()));
}

TEST_P(Operator, IntMatMul) {
  // The scaling factor 1.4x was carefully selected to make sure we don't
  // overflow or underflow the calculation.
  TypeRef resTy = mod_.uniqueType(ElemKind::Int8QTy, {3, 3}, 0.60, 4);
  TypeRef lhsTy = mod_.uniqueType(ElemKind::Int8QTy, {3, 3}, 0.075, -2);
  TypeRef rhsTy = mod_.uniqueType(ElemKind::Int8QTy, {3, 3}, 0.075, 2);

  auto *res = mod_.createVariable(ElemKind::FloatTy, {3, 3}, "res");
  auto *lhs = mod_.createVariable(ElemKind::FloatTy, {3, 3}, "lhs");
  auto *rhs = mod_.createVariable(ElemKind::FloatTy, {3, 3}, "rhs");

  lhs->getPayload().getHandle() = {
      1.0, 2.0, 3.0, 4.0, 5.0, -5.0, -4.0, -3.0, 9.0,
  };

  rhs->getPayload().getHandle() = {
      0.1f, -0.2f, 0.3f, 9.0f, -8.0f, 7.0f, 6.0f, 5.0f, 9.0f,
  };

  auto *lhsq = F_->createQuantize("lhs.q", lhs, lhsTy);
  auto *rhsq = F_->createQuantize("rhs.q", rhs, rhsTy);

  auto *matmulq = F_->createMatMul("matmul.q", resTy, lhsq, rhsq);

  auto *rq = F_->createDequantize("dequant", matmulq);

  F_->createSave("save", rq, res);
  EE_.compile(CompilationMode::Infer, F_);

  EE_.run({}, {});

  /*
   Test the following matrix multiplication:
   A = [[1.0, 2.0, 3.0], [4.0, 5.0, -5.0], [-4.0, -3.0, 9.0]]
   B = [[0.1, -0.2, 0.3], [9.0, -8.0, 7.0], [6.0, 5.0, 9.0]]
   A x B = [36.1,  -1.2,  41.3], [15.4, -65.8, -8.8], [26.6, 69.8,  58.8]]
   */

  auto H = res->getPayload().getHandle();
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

  auto *res = mod_.createVariable(ElemKind::FloatTy, {1, 3, 3}, "res");
  auto *lhs = mod_.createVariable(ElemKind::FloatTy, {1, 3, 3}, "lhs");
  auto *rhs = mod_.createVariable(ElemKind::FloatTy, {3, 3}, "rhs");

  lhs->getPayload().getHandle() = {
      8.7f, 6.5f, 4.3f, 2.1f, 1.0f, -5.1f, -4.0f, -12.0f, 0.2f,
  };

  rhs->getPayload().getHandle() = {
      -9.1f, -0.4f, 1.3f, 2.2f, -8.1f, 7.6f, -6.4f, 10.0f, 9.1f,
  };

  auto *lhsq = F_->createQuantize("lhs.q", lhs, lhsTy);
  auto *rhsq = F_->createQuantize("rhs.q", rhs, rhsTy);

  auto *matmulq = F_->createBatchedAdd("add", resTy, lhsq, rhsq);

  auto *rq = F_->createDequantize("dequant", matmulq);

  F_->createSave("save", rq, res);
  EE_.compile(CompilationMode::Infer, F_);

  EE_.run({}, {});

  // A = [8.7, 6.5, 4.3, 2.1, 1.0, -5.1, -4.0, -12.0, 0.2]
  // B = [-9.1, -0.4, 1.3, 2.2, -8.1, 7.6, -6.4, 10.0, 9.1]
  // A + B = [-0.4, 6.1, 5.6, 4.3, -7.1, 2.5, -10.4, -2. , 9.3]
  auto H = res->getPayload().getHandle();
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

void checkIntConvolution(ExecutionEngine &EE, unsigned convDepth) {
  // In this test we generate a Floating-point based convolution and an integer
  // convolution. We pass the same values and then subtract the results. We
  // check that the results are below some known delta.

  // In this test the output of the convolution is in the range [-256 ... 256].
  // The inputs (specified below) are in the range [-1 .. 1],
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *input = mod.createVariable(ElemKind::FloatTy, {1, 10, 10, 3}, "in");
  auto *conv = F->createConv("conv", input, convDepth, 5, 1, 0, 1);
  auto *res =
      mod.createVariable(ElemKind::FloatTy, conv->getResult().dims(), "res");

  auto filter = conv->getFilter();
  auto bias = conv->getBias();

  input->getPayload().getHandle().randomize(-1.0, 1.0, mod.getPRNG());
  llvm::cast<Variable>(bias)->getPayload().getHandle().randomize(-2.0, 2.0,
                                                                 mod.getPRNG());

  TypeRef resTy = mod.uniqueType(ElemKind::Int8QTy, res->dims(), 0.08, 0.0);
  TypeRef inputTy = mod.uniqueType(ElemKind::Int8QTy, input->dims(), 0.01, 0.0);
  TypeRef filterTy =
      mod.uniqueType(ElemKind::Int8QTy, filter.dims(), 0.01, 0.0);
  TypeRef biasTy = mod.uniqueType(ElemKind::Int8QTy, bias.dims(), 0.04, 0.0);

  auto *inputq = F->createQuantize("input.q", input, inputTy);
  auto *filterq = F->createQuantize("filter.q", filter, filterTy);
  auto *biasq = F->createQuantize("bias.q", bias, biasTy);

  auto *convq =
      F->createConv("convq", inputq, filterq, biasq, resTy, conv->getKernels(),
                    conv->getStrides(), conv->getPads(), 1);
  auto *dequantRes = F->createDequantize("dequant", convq);

  // Subtract the results of the convolution from the quantized convolution.
  auto *sub = F->createSub("compare", dequantRes, conv);

  F->createSave("save", sub, res);
  EE.compile(CompilationMode::Infer, F);
  EE.run({}, {});
  auto H = res->getPayload().getHandle();

  // Check that the difference in the results is less than 0.1.
  for (int i = 0, e = H.size(); i < e; i++) {
    EXPECT_NEAR(H.raw(i), 0, 0.1);
  }
}

TEST_P(Operator, IntConvolutionDepth10) { checkIntConvolution(EE_, 10); }

TEST_P(Operator, IntConvolutionDepth8) { checkIntConvolution(EE_, 8); }

TEST_P(InterpAndCPU, IntConcat) {
  auto A = mod_.createVariable(ElemKind::FloatTy, {3, 3}, "A");
  auto B = mod_.createVariable(ElemKind::FloatTy, {2, 3}, "B");
  A->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  B->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());

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

  auto res = F_->createSave("save", sub);
  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto R = res->getVariable()->getHandle();
  // Check that the difference in the results is less than 0.2.
  for (int i = 0, e = R.size(); i < e; i++) {
    EXPECT_NEAR(R.raw(i), 0, 0.2);
  }
}

TEST_P(InterpAndCPU, IntFC) {
  // In this test we subtract the outputs of a quantized FC and a floating-point
  // FC and ensure that the error is below some low value.
  auto *input = mod_.createVariable(ElemKind::FloatTy, {1, 10, 10, 3}, "in");
  auto *fc = F_->createFullyConnected("FC", input, 30);
  auto *res =
      mod_.createVariable(ElemKind::FloatTy, fc->getResult().dims(), "res");

  auto weights = fc->getWeights();
  auto bias = fc->getBias();

  input->getPayload().getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  llvm::cast<Variable>(bias)->getPayload().getHandle().randomize(
      0, 0.00001, mod_.getPRNG());
  llvm::cast<Variable>(weights)->getPayload().getHandle().randomize(
      -1.1, 1.1, mod_.getPRNG());

  TypeRef resTy = mod_.uniqueType(ElemKind::Int8QTy, res->dims(), 0.15, 4);
  TypeRef inputTy = mod_.uniqueType(ElemKind::Int8QTy, input->dims(), 0.01, 0);
  TypeRef weightsTy =
      mod_.uniqueType(ElemKind::Int8QTy, weights.dims(), 0.01, 2);
  TypeRef biasTy = mod_.uniqueType(ElemKind::Int8QTy, bias.dims(), 0.02, 1);

  auto *inputq = F_->createQuantize("input.q", input, inputTy);
  auto *weightsq = F_->createQuantize("filter.q", weights, weightsTy);
  auto *biasq = F_->createQuantize("bias.q", bias, biasTy);

  auto *fcq = F_->createFullyConnected("fcq", inputq, weightsq, biasq, resTy);
  auto *dequantRes = F_->createDequantize("dequant", fcq);

  // Subtract the results of the convolution from the quantized fc.
  auto *sub = F_->createSub("compare", dequantRes, fc);

  F_->createSave("save", sub, res);
  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto H = res->getPayload().getHandle();
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
  auto *P = mod_.createVariable(ElemKind::FloatTy, {2, 3}, "P");
  auto *Y = mod_.createVariable(ElemKind::Int64ITy, {2}, "Y");
  auto *L = mod_.createVariable(ElemKind::FloatTy, {1}, "L");

  P->getPayload().getHandle() = {0.2f, 0.5f, 0.3f, 0.4f, 0.3f, 0.3f};
  Y->getPayload().getHandle<int64_t>() = {1, 2};
  auto *ceLoss = F_->createCrossEntropyLoss("CELoss", P, Y);
  F_->createSave("save", ceLoss, L);
  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});
  auto R = L->getPayload().getHandle();
  EXPECT_NEAR(R.at({0}), -log(0.5) - log(0.3), 0.1);
}

TEST_P(Operator, RescaleNode) {
  // Check the outputs of the RescaleQuantized operation.
  auto *input = mod_.createVariable(ElemKind::Int8QTy, {4, 10}, 0.4, -3,
                                    "input", VisibilityKind::Public, true);
  auto *output = mod_.createVariable(ElemKind::Int8QTy, {4, 10}, 0.4, -4,
                                     "output", VisibilityKind::Public, false);

  input->getPayload().init(Tensor::InitKind::Broadcast, 40, mod_.getPRNG());

  auto T1 = mod_.uniqueType(ElemKind::Int8QTy, {4, 10}, 0.7, 5);
  auto T2 = mod_.uniqueType(ElemKind::Int8QTy, {4, 10}, 0.3, -4);

  // Test a sequence of rescale operations that the optimizer may try to
  // optimize at some point.
  auto *X = F_->createRescaleQuantized("R1", input, T1);
  auto *Y = F_->createRescaleQuantized("R2", X, T2);
  auto *Z = F_->createRescaleQuantized("R3", Y, output->getType());

  F_->createSave("save", Z, output);
  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto RI = input->getPayload().getHandle<int8_t>();
  auto RO = output->getPayload().getHandle<int8_t>();

  EXPECT_EQ(RI.raw(0), 40);
  EXPECT_NEAR(RO.raw(0), 40, 1);
}

TEST_P(InterpAndCPU, QuantizedArithmeticRescaled) {
  const size_t len = 100;

  // In this test we check the correctness of the quantized Max, Min, Add, Sub,
  // Mul, and Div nodes as well as how they interact with the rescaling node.
  auto *A = mod_.createVariable(ElemKind::FloatTy, {len}, "A",
                                VisibilityKind::Public);
  auto *B = mod_.createVariable(ElemKind::FloatTy, {len}, "B",
                                VisibilityKind::Public);
  auto *C = mod_.createVariable(ElemKind::FloatTy, {len}, "C",
                                VisibilityKind::Public);
  auto *O1 = mod_.createVariable(ElemKind::FloatTy, {len}, "Max",
                                 VisibilityKind::Public, false);
  auto *O2 = mod_.createVariable(ElemKind::FloatTy, {len}, "Min",
                                 VisibilityKind::Public, false);
  auto *O3 = mod_.createVariable(ElemKind::FloatTy, {len}, "Add",
                                 VisibilityKind::Public, false);
  auto *O4 = mod_.createVariable(ElemKind::FloatTy, {len}, "Sub",
                                 VisibilityKind::Public, false);
  auto *O5 = mod_.createVariable(ElemKind::FloatTy, {len}, "Mul",
                                 VisibilityKind::Public, false);
  auto *O6 = mod_.createVariable(ElemKind::FloatTy, {len}, "Div",
                                 VisibilityKind::Public, false);

  auto AH = A->getHandle();
  auto BH = B->getHandle();
  auto CH = C->getHandle();
  auto O1H = O1->getHandle();
  auto O2H = O2->getHandle();
  auto O3H = O3->getHandle();
  auto O4H = O4->getHandle();
  auto O5H = O5->getHandle();
  auto O6H = O6->getHandle();

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
  F_->createSave("saveMax", max, O1);
  F_->createSave("saveMin", min, O2);
  F_->createSave("saveAdd", add, O3);
  F_->createSave("saveSub", sub, O4);
  F_->createSave("saveMul", mul, O5);
  F_->createSave("saveDiv", div, O6);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  for (size_t i = 0; i < len; i++) {
    auto max = std::max(AH.at({i}), BH.at({i}));
    auto min = std::min(AH.at({i}), BH.at({i}));
    auto add = AH.at({i}) + BH.at({i});
    auto sub = AH.at({i}) - BH.at({i});
    auto mul = AH.at({i}) * BH.at({i});
    auto div = BH.at({i}) / CH.at({i});

    // We generate numbers up to 110, so a difference of 2 (~2%) is reasonable.
    EXPECT_NEAR(max, O1H.at({i}), 2.0);
    EXPECT_NEAR(min, O2H.at({i}), 2.0);
    EXPECT_NEAR(add, O3H.at({i}), 2.0);
    EXPECT_NEAR(sub, O4H.at({i}), 2.0);
    EXPECT_NEAR(mul, O5H.at({i}), 2.0);
    EXPECT_NEAR(div, O6H.at({i}), 2.0);
  }
}

TEST_P(Operator, QuantizedTranspose) {
  auto *A = mod_.createVariable(ElemKind::FloatTy, {2, 3}, "A",
                                VisibilityKind::Public);
  auto *B = mod_.createVariable(ElemKind::FloatTy, {3, 2}, "B",
                                VisibilityKind::Public);
  A->getPayload().getHandle() = {1, 1.2f, 0.5f, 1.3f, 2.7f, 5.8f};
  A->getPayload().transpose(&B->getPayload(), {1, 0});
  auto qType = mod_.uniqueType(ElemKind::Int8QTy, {2, 3}, 0.05, -138);
  auto *quantizeA = F_->createQuantize("quantize", A, qType);
  auto *tr = F_->createTranspose("tr", quantizeA, {1, 0});
  auto *dequantize = F_->createDequantize("dequantize", tr);
  auto *result = F_->createSave("ret", dequantize);
  auto *fpTr = F_->createTranspose("fpTr", A, {1, 0});
  auto *fpResult = F_->createSave("fpRet", fpTr);
  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});
  EXPECT_TRUE(result->getVariable()->getPayload().isEqual(B->getPayload()));
  EXPECT_TRUE(fpResult->getVariable()->getPayload().isEqual(B->getPayload()));
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

  auto *QA =
      mod_.createVariable(ElemKind::Int8QTy, {len}, TQA->getScale(),
                          TQA->getOffset(), "QA", VisibilityKind::Public);
  auto *QB =
      mod_.createVariable(ElemKind::Int8QTy, {len}, TQB->getScale(),
                          TQB->getOffset(), "QB", VisibilityKind::Public);
  auto *QC =
      mod_.createVariable(ElemKind::Int8QTy, {len}, TQC->getScale(),
                          TQC->getOffset(), "QC", VisibilityKind::Public);
  auto *O1 = mod_.createVariable(ElemKind::Int8QTy, {len}, TO1->getScale(),
                                 TO1->getOffset(), "Max",
                                 VisibilityKind::Public, false);
  auto *O2 = mod_.createVariable(ElemKind::Int8QTy, {len}, TO2->getScale(),
                                 TO2->getOffset(), "Min",
                                 VisibilityKind::Public, false);
  auto *O3 = mod_.createVariable(ElemKind::Int8QTy, {len}, TO3->getScale(),
                                 TO3->getOffset(), "Add",
                                 VisibilityKind::Public, false);
  auto *O4 = mod_.createVariable(ElemKind::Int8QTy, {len}, TO4->getScale(),
                                 TO4->getOffset(), "Sub",
                                 VisibilityKind::Public, false);
  auto *O5 = mod_.createVariable(ElemKind::Int8QTy, {len}, TO5->getScale(),
                                 TO5->getOffset(), "Mul",
                                 VisibilityKind::Public, false);
  auto *O6 = mod_.createVariable(ElemKind::Int8QTy, {len}, TO6->getScale(),
                                 TO6->getOffset(), "Div",
                                 VisibilityKind::Public, false);

  auto QAH = QA->getHandle<int8_t>();
  auto QBH = QB->getHandle<int8_t>();
  auto QCH = QC->getHandle<int8_t>();
  auto O1H = O1->getHandle<int8_t>();
  auto O2H = O2->getHandle<int8_t>();
  auto O3H = O3->getHandle<int8_t>();
  auto O4H = O4->getHandle<int8_t>();
  auto O5H = O5->getHandle<int8_t>();
  auto O6H = O6->getHandle<int8_t>();

  QAH.randomize(-10, 10, mod_.getPRNG());
  QBH.randomize(-10, 10, mod_.getPRNG());
  QCH.randomize(-10, 10, mod_.getPRNG());

  // Apply max/min/add/sub/mul/div quantized.
  Node *max = F_->createMax("max", TO1, QA, QB);
  Node *min = F_->createMin("min", TO2, QA, QB);
  Node *add = F_->createAdd("add", TO3, QA, QB);
  Node *sub = F_->createSub("sub", TO4, QA, QB);
  Node *mul = F_->createMul("mul", TO5, QA, QB);
  Node *div = F_->createDiv("div", TO6, QB, QC);

  // Save results of the operations.
  F_->createSave("saveMax", max, O1);
  F_->createSave("saveMin", min, O2);
  F_->createSave("saveAdd", add, O3);
  F_->createSave("saveSub", sub, O4);
  F_->createSave("saveMul", mul, O5);
  F_->createSave("saveDiv", div, O6);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

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

  auto *QA =
      mod_.createVariable(ElemKind::Int8QTy, {len}, TQA->getScale(),
                          TQA->getOffset(), "QA", VisibilityKind::Public);
  auto *QB =
      mod_.createVariable(ElemKind::Int8QTy, {len}, TQB->getScale(),
                          TQB->getOffset(), "QB", VisibilityKind::Public);
  auto *QC =
      mod_.createVariable(ElemKind::Int8QTy, {len}, TQC->getScale(),
                          TQC->getOffset(), "QC", VisibilityKind::Public);
  auto *QD =
      mod_.createVariable(ElemKind::Int8QTy, {len}, TQD->getScale(),
                          TQD->getOffset(), "QD", VisibilityKind::Public);
  auto *Out = mod_.createVariable(ElemKind::Int8QTy, {len}, 1.5, -2, "out",
                                  VisibilityKind::Public, false);

  auto QAH = QA->getHandle<int8_t>();
  auto QBH = QB->getHandle<int8_t>();
  auto QCH = QC->getHandle<int8_t>();
  auto QDH = QD->getHandle<int8_t>();
  auto OH = Out->getHandle<int8_t>();

  QAH.randomize(-128, 127, mod_.getPRNG());
  QBH.randomize(-128, 127, mod_.getPRNG());
  QCH.randomize(-128, 127, mod_.getPRNG());
  QDH.randomize(-128, 127, mod_.getPRNG());

  // Apply comparison and selection quantized.
  Node *cmpLTE = F_->createCmpLTE("cmpLTE", QA, QB);
  Node *select = F_->createSelect("select", OT, cmpLTE, QC, QD);

  // Save result of the operation.
  F_->createSave("save", select, Out);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

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

  auto *A = mod_.createVariable(ElemKind::FloatTy, {len}, "A",
                                VisibilityKind::Public);
  auto *O = mod_.createVariable(ElemKind::FloatTy, {len}, "Out",
                                VisibilityKind::Public, false);

  auto AH = A->getPayload().getHandle();
  auto OH = O->getPayload().getHandle();

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
  F_->createSave("save", DQ, O);
  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  for (size_t i = 0; i < len; i++) {
    EXPECT_NEAR(AH.at({i}), OH.at({i}), 1.0);
  }
}

TEST_P(Operator, FCGradientCheck) {
  // Create net representing A*X+Y=B, where X and Y are trainable, while
  // A and B are fixed. Record gradients for X and Y after 3 steps and compare
  // with reference values.
  TrainingConfig TC;

  auto *A = mod_.createVariable(ElemKind::FloatTy, {2, 1}, "A",
                                VisibilityKind::Public, false);
  auto *B = mod_.createVariable(ElemKind::FloatTy, {2, 1}, "B",
                                VisibilityKind::Public, false);
  auto *X = mod_.createVariable(ElemKind::FloatTy, {1, 1}, "X",
                                VisibilityKind::Public, true);
  auto *Y = mod_.createVariable(ElemKind::FloatTy, {1}, "Y",
                                VisibilityKind::Public, true);

  X->getPayload().init(Tensor::InitKind::Broadcast, -1.26274, mod_.getPRNG());
  Y->getPayload().init(Tensor::InitKind::Broadcast, 0.1, mod_.getPRNG());

  auto *FC = F_->createFullyConnected("fc", A, X, Y);
  auto *S = F_->createRegression("reg", FC, B);
  F_->createSave("ret", S);

  Tensor initA(ElemKind::FloatTy, {2, 1});
  Tensor initB(ElemKind::FloatTy, {2, 1});
  initA.getHandle() = {4.2f, 9.875f};
  initB.getHandle() = {-13.1f, 3.14f};

  Function *DF = glow::differentiate(F_, TC, "d_main");
  EE_.compile(CompilationMode::Train, DF);
  EE_.runBatch(3, {A, B}, {&initA, &initB});

  EXPECT_NEAR(X->getPayload().getHandle().raw(0), -0.21294, 1E-5);
  EXPECT_NEAR(Y->getPayload().getHandle().raw(0), 0.01656, 1E-5);
}

TEST_P(InterpAndCPU, concatVectors) {
  F_->setName("concatVectors");

  auto *V1 = mod_.createVariable(ElemKind::Int64ITy, {10}, "V1",
                                 VisibilityKind::Public);
  auto *V2 = mod_.createVariable(ElemKind::Int64ITy, {20}, "V2",
                                 VisibilityKind::Public);
  auto *V3 = mod_.createVariable(ElemKind::Int64ITy, {30}, "V3",
                                 VisibilityKind::Public);

  Node *L = F_->createConcat("concat", {V1, V2, V3}, 0);
  auto *result = F_->createSave("ret", L);

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

  EE_.compile(CompilationMode::Infer, F_);

  // Testing the output vector.
  EE_.run({V1, V2, V3}, {&I1, &I2, &I3});
  auto RNWH = result->getVariable()->getPayload().getHandle<int64_t>();
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

  auto *V1 = mod_.createVariable(ElemKind::Int64ITy, {10}, "V1",
                                 VisibilityKind::Public);
  auto *V2 = mod_.createVariable(ElemKind::Int64ITy, {20}, "V2",
                                 VisibilityKind::Public);

  // Alternate adding sequences of V1 and V2, so that the IRGen'd InsertTensors
  // have different counts.
  Node *L = F_->createConcat("concat", {V2, V1, V1, V1, V2, V2, V1, V1, V2}, 0);
  auto *result = F_->createSave("ret", L);

  Tensor I1(ElemKind::Int64ITy, {10});
  Tensor I2(ElemKind::Int64ITy, {20});

  for (size_t i = 0; i < 10; i++) {
    I1.getHandle<int64_t>().at({i}) = 1;

    I2.getHandle<int64_t>().at({i}) = 2;
    I2.getHandle<int64_t>().at({i + 10}) = 2;
  }

  EE_.compile(CompilationMode::Infer, F_);

  // Testing the output vector.
  EE_.run({V1, V2}, {&I1, &I2});
  auto outH = result->getVariable()->getPayload().getHandle<int64_t>();
  (void)outH;

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

  auto *V = mod_.createVariable(ElemKind::Int64ITy, {3, 30}, "V",
                                VisibilityKind::Public);

  Node *S1 = F_->createSlice("slice1", V, {0, 10}, {3, 13});
  Node *S2 = F_->createSlice("slice2", V, {1, 0}, {2, 30});
  Node *S3 = F_->createSlice("slice3", V, {2, 10}, {3, 12});

  auto *result1 = F_->createSave("ret1", S1);
  auto *result2 = F_->createSave("ret2", S2);
  auto *result3 = F_->createSave("ret3", S3);

  Tensor I(ElemKind::Int64ITy, {3, 30});

  for (size_t j = 0; j < 30; j++) {
    I.getHandle<int64_t>().at({0, j}) = j;
    I.getHandle<int64_t>().at({1, j}) = j + 30;
    I.getHandle<int64_t>().at({2, j}) = j + 60;
  }

  EE_.compile(CompilationMode::Infer, F_);

  // Testing the output slices.
  EE_.run({V}, {&I});
  auto RNWH1 = result1->getVariable()->getPayload().getHandle<int64_t>();
  (void)RNWH1;
  auto RNWH2 = result2->getVariable()->getPayload().getHandle<int64_t>();
  (void)RNWH2;
  auto RNWH3 = result3->getVariable()->getPayload().getHandle<int64_t>();
  (void)RNWH3;

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

  auto *V = mod_.createVariable(ElemKind::Int64ITy, {5, 4}, "V",
                                VisibilityKind::Public);

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

  EE_.compile(CompilationMode::Infer, F_);

  EE_.run({V}, {&I});

  const size_t expected[7][4] = {{300, 301, 302, 303}, {400, 401, 402, 403},
                                 {100, 101, 302, 303}, {200, 201, 402, 403},
                                 {0, 1, 2, 3},         {100, 101, 102, 103},
                                 {200, 201, 202, 203}};

  auto resultH = result->getVariable()->getPayload().getHandle<int64_t>();
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

  auto *V = mod_.createVariable(ElemKind::FloatTy, {4, 5}, "V",
                                VisibilityKind::Public);

  Node *T0 = F_->createTile("tile0", V, /* tiles */ 3, /* axis */ 0);
  auto *result0 = F_->createSave("res0", T0);

  Node *T1 = F_->createTile("tile1", V, /* tiles */ 3, /* axis */ 1);
  auto *result1 = F_->createSave("res1", T1);

  Tensor VT(ElemKind::FloatTy, {4, 5});

  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 5; j++) {
      VT.getHandle<float>().at({i, j}) = i * 5 + j;
    }
  }

  EE_.compile(CompilationMode::Infer, F_);

  EE_.run({V}, {&VT});

  // Testing the output vector with axis 0.
  auto res0 = result0->getVariable()->getHandle<float>();
  for (size_t i = 0; i < res0.dims()[0]; i++) {
    for (size_t j = 0; j < res0.dims()[1]; j++) {
      EXPECT_EQ(res0.at({i, j}), (i % 4) * 5 + j);
    }
  }

  // Testing the output vector with axis 1.
  auto res1 = result1->getVariable()->getHandle<float>();
  for (size_t i = 0; i < res1.dims()[0]; i++) {
    for (size_t j = 0; j < res1.dims()[1]; j++) {
      EXPECT_EQ(res1.at({i, j}), i * 5 + (j % 5));
    }
  }
}

TEST_P(InterpAndCPU, QuantizedTile) {
  F_->setName("concatVectors");

  auto *V = mod_.createVariable(ElemKind::FloatTy, {4, 5}, "V",
                                VisibilityKind::Public);
  auto quantizationParams = glow::quantization::chooseQuantizationParams(0, 20);
  auto quantizeTy =
      mod_.uniqueType(ElemKind::Int8QTy, {4, 5}, quantizationParams.scale,
                      quantizationParams.offset);
  auto *Q = F_->createQuantize("quantize", V, quantizeTy);

  Node *T0 = F_->createTile("tile0", Q, /* tiles */ 3, /* axis */ 0);
  auto *DQ0 = F_->createDequantize("dequantize0", T0);
  auto *result0 = F_->createSave("res0", DQ0);

  Node *T1 = F_->createTile("tile1", Q, /* tiles */ 3, /* axis */ 1);
  auto *DQ1 = F_->createDequantize("dequantize1", T1);
  auto *result1 = F_->createSave("res1", DQ1);

  Tensor VT(ElemKind::FloatTy, {4, 5});

  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 5; j++) {
      VT.getHandle<float>().at({i, j}) = i * 5 + j;
    }
  }

  EE_.compile(CompilationMode::Infer, F_);

  EE_.run({V}, {&VT});

  // Testing the output vector with axis 0.
  auto res0 = result0->getVariable()->getHandle<float>();
  for (size_t i = 0; i < res0.dims()[0]; i++) {
    for (size_t j = 0; j < res0.dims()[1]; j++) {
      EXPECT_NEAR(res0.at({i, j}), (i % 4) * 5 + j, 0.05);
    }
  }

  // Testing the output vector with axis 1.
  auto res1 = result1->getVariable()->getHandle<float>();
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

  auto *inputs = mod_.createVariable(ElemKind::FloatTy, {10}, "inputs");
  auto *counters = mod_.createVariable(ElemKind::FloatTy, {10}, "counters");

  counters->getPayload().getHandle() = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  inputs->getPayload().getHandle().clear(1);

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

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto H = SN->getVariable()->getHandle();
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
  auto *inputs = mod_.createVariable(ElemKind::FloatTy, {10, 10, 10}, "inputs");
  auto *counters = mod_.createVariable(ElemKind::FloatTy, {10}, "counters");

  counters->getHandle() = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  inputs->getHandle().randomize(-10, 10, mod_.getPRNG());

  Node *C5 = F_->createSplat("C5", counters->getType(), 5.0);
  Node *pred = F_->createCmpLTE("cmp", C5, counters);

  auto *FC0 = F_->createFullyConnected("FC0", inputs, 128);
  auto *RL0 = F_->createRELU("RL0", FC0);
  auto *FC1 = F_->createFullyConnected("FC1", RL0, 64);
  auto *RL1 = F_->createRELU("RL1", FC1);
  auto *FC2 = F_->createFullyConnected("FC2", RL1, 32);
  auto *RL2 = F_->createRELU("RL2", FC2);

  F_->createSave("ret", RL2);

  FC0->setPredicate(pred);
  FC1->setPredicate(pred);
  FC2->setPredicate(pred);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});
}

TEST_P(InterpAndCPU, ChannelShuffle) {
  auto *inputs =
      mod_.createVariable(ElemKind::FloatTy, {1, 12, 1, 1}, "inputs");

  inputs->getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  Node *CS = F_->createChannelShuffle("CS", inputs, 3, 1);
  SaveNode *S = F_->createSave("save", CS);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto results = llvm::cast<Variable>(S->getOutput())->getPayload().getHandle();

  EXPECT_EQ(results.size(), 12);
  std::vector<float> expected = {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
  for (size_t i = 0; i < expected.size(); i++)
    EXPECT_FLOAT_EQ(results.at({0, i, 0, 0}), expected[i]);
}

TEST_P(Operator, Squeeze) {
  auto *inputs = mod_.createVariable(ElemKind::FloatTy, {1, 2, 1, 5}, "inputs");
  inputs->getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::vector<float> expectedValues = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  // Test 1:
  {
    std::vector<size_t> axes = {0};
    Node *SQZ = F_->createSqueeze("SQZ", inputs, axes);
    SaveNode *S = F_->createSave("save", SQZ);

    EE_.compile(CompilationMode::Infer, F_);
    EE_.run({}, {});

    auto results = S->getVariable()->getHandle();
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

    EE_.compile(CompilationMode::Infer, F_);
    EE_.run({}, {});

    auto results = S->getVariable()->getHandle();
    std::vector<size_t> expectedDims = {2, 5};
    EXPECT_TRUE(results.dims().vec() == expectedDims);
    for (size_t i = 0; i < 10; i++)
      EXPECT_FLOAT_EQ(results.raw(i), expectedValues[i]);
  }

  // Test 3: 0-dimensional Tensor
  {
    auto *emptyInput =
        mod_.createVariable(ElemKind::FloatTy, {1}, "emptyInput");
    emptyInput->getHandle() = {42.0};

    std::vector<size_t> axes = {0};
    Node *SQZ = F_->createSqueeze("SQZ", emptyInput, axes);
    SaveNode *S1 = F_->createSave("save", SQZ);
    Node *UnSQZ = F_->createExpandDims("UnSQZ", SQZ, axes);
    SaveNode *S2 = F_->createSave("save", UnSQZ);

    EE_.compile(CompilationMode::Infer, F_);
    EE_.run({}, {});

    auto res1 = S1->getVariable()->getHandle();
    EXPECT_TRUE(res1.dims().vec() == std::vector<size_t>());
    EXPECT_FLOAT_EQ(res1.raw(0), 42.0);
    auto res2 = S2->getVariable()->getHandle();
    EXPECT_TRUE(res2.dims().vec() == std::vector<size_t>(1, 1));
    EXPECT_FLOAT_EQ(res2.raw(0), 42.0);
  }
}

/// Check that the expand dims operator works, which is implemented with a
/// reshape.
TEST_P(Operator, ExpandDims) {
  auto *inputs = mod_.createVariable(ElemKind::FloatTy, {2, 2}, "inputs");
  auto IH = inputs->getHandle();
  IH = {1, 2, 3, 4};

  // This should be uniqued and sorted, so should become {0, 1, 3, 5}.
  std::vector<size_t> axes = {3, 0, 5, 1, 3};
  Node *EDN = F_->createExpandDims("expand", inputs, axes);
  SaveNode *S = F_->createSave("save", EDN);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  // Expected dims based on the axes above; inserted new dimensions of 1 in
  // every unique axes location, based on the output tensor shape.
  std::vector<size_t> expectedDims = {1, 1, 2, 1, 2, 1};
  auto results = S->getVariable()->getHandle();
  EXPECT_TRUE(results.dims().vec() == expectedDims);

  // The data should be the same, as this was just a reshape.
  for (size_t i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(results.raw(i), IH.raw(i));
  }
}

TEST_P(InterpAndCPU, Split) {
  auto *inputs = mod_.createVariable(ElemKind::FloatTy, {1, 2, 6}, "inputs");
  inputs->getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

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

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto result = S1->getVariable()->getHandle();
  EXPECT_EQ(result.dims().vec(), std::vector<size_t>({1, 2, 3}));
  EXPECT_FLOAT_EQ(result.at({0, 0, 0}), 1);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1}), 2);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2}), 3);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0}), 7);
  EXPECT_FLOAT_EQ(result.at({0, 1, 1}), 8);
  EXPECT_FLOAT_EQ(result.at({0, 1, 2}), 9);

  result = S2->getVariable()->getHandle();
  EXPECT_EQ(result.dims().vec(), std::vector<size_t>({1, 2, 3}));
  EXPECT_FLOAT_EQ(result.at({0, 0, 0}), 4);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1}), 5);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2}), 6);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0}), 10);
  EXPECT_FLOAT_EQ(result.at({0, 1, 1}), 11);
  EXPECT_FLOAT_EQ(result.at({0, 1, 2}), 12);

  result = S3->getVariable()->getHandle();
  EXPECT_EQ(result.dims().vec(), std::vector<size_t>({1, 2, 2}));
  EXPECT_FLOAT_EQ(result.at({0, 0, 0}), 1);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1}), 2);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0}), 7);
  EXPECT_FLOAT_EQ(result.at({0, 1, 1}), 8);

  result = S4->getVariable()->getHandle();
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
  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto result = save->getVariable()->getHandle();
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
  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto result = save->getVariable()->getHandle();
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_EQ(splatValue, result.raw(i));
  }
}

TEST_P(Operator, GroupConvolution) {
  auto *input = mod_.createVariable(ElemKind::FloatTy, {1, 2, 1, 8}, "input");
  auto IH = input->getHandle();
  for (size_t i = 0; i < 2 * 8; i++) {
    IH.raw(i) = i + 1;
  }

  auto filter = mod_.createVariable(ElemKind::FloatTy, {6, 1, 1, 4}, "filter");
  auto FH = filter->getHandle();
  for (size_t i = 0; i < 6; i++)
    for (size_t j = 0; j < 4; j++) {
      FH.at({i, 0, 0, j}) = pow(10.0, i);
    }

  auto *zeroBias = mod_.createVariable(ElemKind::FloatTy, {6}, "bias");
  zeroBias->getPayload().zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 2, 1, 6});

  ConvolutionNode *CN =
      F_->createConv("Conv", input, filter, zeroBias, outTy, 1, 1, 0, 2);
  SaveNode *S = F_->createSave("save", CN);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto result = llvm::cast<Variable>(S->getOutput())->getPayload().getHandle();

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
  auto *input = mod_.createVariable(ElemKind::FloatTy, {1, 4, 4, 1}, "input");
  auto IH = input->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }

  auto filter = mod_.createVariable(ElemKind::FloatTy, {2, 2, 2, 1}, "filter",
                                    VisibilityKind::Public, false);
  auto FH = filter->getHandle();
  for (size_t i = 0; i < 2 * 2 * 2; i++) {
    FH.raw(i) = pow(2.0, i);
  }
  auto *zeroBias = mod_.createVariable(ElemKind::FloatTy, {2}, "bias",
                                       VisibilityKind::Public, false);
  zeroBias->getPayload().zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 4, 8, 2});

  ConvolutionNode *CN = F_->createConv("Conv", input, filter, zeroBias, outTy,
                                       {2, 2}, {1, 1}, {0, 2, 1, 3}, 1);
  SaveNode *S = F_->createSave("save", CN);
  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});
  Tensor &result = S->getVariable()->getPayload();

  // Create the reference conv operator whose input is the same as the
  // after-padding-input above.
  auto *input1 = mod_.createVariable(ElemKind::FloatTy, {1, 5, 9, 1}, "input1");
  input1->getPayload().zero();
  auto IH1 = input1->getHandle();
  for (size_t i = 0; i < 4; i++)
    for (size_t j = 2; j < 6; j++) {
      IH1.at({0, i, j, 0}) = i * 4 + j - 2 + 1;
    }

  Function *refF = mod_.createFunction("mainRef");
  CN = refF->createConv("Conv1", input1, filter, zeroBias, outTy, {2, 2},
                        {1, 1}, {0, 0, 0, 0}, 1);
  S = refF->createSave("save1", CN);
  EE_.compile(CompilationMode::Infer, refF);
  EE_.run({}, {});
  Tensor &result1 = S->getVariable()->getPayload();

  EXPECT_TRUE(result.isEqual(result1));
}

/// Check non-square padding for AveragePool. The first pool op has non-square
/// padding, while the second one has zero padding. The second pool op's input
/// is the same as the first one's after-padding input. All other parameters of
/// the two convs are the same.
TEST_P(Operator, NonSquarePaddingAveragePool) {
  auto *input = mod_.createVariable(ElemKind::FloatTy, {1, 4, 4, 1}, "input");
  auto IH = input->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }
  auto *Pool = F_->createAvgPool("pool", input, {2, 2}, {1, 1}, {0, 2, 1, 3});
  auto *S = F_->createSave("save", Pool);
  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});
  Tensor &result = S->getVariable()->getPayload();

  auto *input1 = mod_.createVariable(ElemKind::FloatTy, {1, 5, 9, 1}, "input1");
  input1->getPayload().zero();
  auto IH1 = input1->getHandle();
  for (size_t i = 0; i < 4; i++)
    for (size_t j = 2; j < 6; j++) {
      IH1.at({0, i, j, 0}) = i * 4 + j - 2 + 1;
    }

  Function *refF = mod_.createFunction("mainRef");
  Pool = refF->createAvgPool("pool1", input1, 2, 1, 0);
  S = refF->createSave("save1", Pool);
  EE_.compile(CompilationMode::Infer, refF);
  EE_.run({}, {});
  Tensor &result1 = S->getVariable()->getPayload();

  EXPECT_TRUE(result.isEqual(result1));
}

/// Check non-square padding for MaxPool. The first pool op has non-square
/// padding, while the second one has zero padding. The second pool-op's input
/// is the same as the first one's after-padding input. All other parameters
/// of the two convs are the same.
TEST_P(Operator, NonSquarePaddingMaxPool) {
  auto *input = mod_.createVariable(ElemKind::FloatTy, {1, 4, 4, 1}, "input");
  auto IH = input->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }
  auto *Pool = F_->createMaxPool("pool", input, {2, 2}, {1, 1}, {0, 2, 1, 3});
  auto *S = F_->createSave("save", Pool);
  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});
  Tensor &result = S->getVariable()->getPayload();

  auto *input1 = mod_.createVariable(ElemKind::FloatTy, {1, 5, 9, 1}, "input1");
  input1->getPayload().zero();
  auto IH1 = input1->getHandle();
  for (size_t i = 0; i < 4; i++)
    for (size_t j = 2; j < 6; j++) {
      IH1.at({0, i, j, 0}) = i * 4 + j - 2 + 1;
    }

  Function *refF = mod_.createFunction("mainRef");
  Pool = refF->createMaxPool("pool1", input1, 2, 1, 0);
  S = refF->createSave("save1", Pool);
  EE_.compile(CompilationMode::Infer, refF);
  EE_.run({}, {});
  Tensor &result1 = S->getVariable()->getPayload();

  EXPECT_TRUE(result.isEqual(result1));
}

TEST_P(Operator, Int8AvgPool) {
  auto *input =
      mod_.createVariable(ElemKind::Int8QTy, {1, 3, 3, 1}, 1, 0, "input");
  input->getHandle<int8_t>() = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  auto *Pool = F_->createAvgPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", Pool);
  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});
  auto result = S->getVariable()->getHandle<int8_t>();
  Tensor out(ElemKind::Int8QTy, {2, 2}, 1, 0);
  out.getHandle<int8_t>() = {2, 3, 5, 6};
  for (size_t i = 0; i < 2 * 2; i++) {
    EXPECT_EQ(result.raw(i), out.getHandle<int8_t>().raw(i));
  }
}

TEST_P(Operator, Int8MaxPool) {
  auto *input =
      mod_.createVariable(ElemKind::Int8QTy, {1, 3, 3, 1}, 1, 0, "input");
  input->getHandle<int8_t>() = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  auto *Pool = F_->createMaxPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", Pool);
  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});
  auto result = S->getVariable()->getHandle<int8_t>();
  Tensor out(ElemKind::Int8QTy, {2, 2}, 1, 0);
  out.getHandle<int8_t>() = {4, 5, 7, 8};
  for (size_t i = 0; i < 2 * 2; i++) {
    EXPECT_EQ(result.raw(i), out.getHandle<int8_t>().raw(i));
  }
}

TEST_P(InterpAndCPU, Int8Tanh) {
  constexpr size_t size = 10;
  auto *input = mod_.createVariable(ElemKind::FloatTy, {size}, "input");
  input->getHandle().randomize(-10.0, 10.0, mod_.getPRNG());

  auto *fpTanh = F_->createTanh("fpTanh", input);
  auto *saveFp = F_->createSave("fpSave", fpTanh);

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

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto fpResult = saveFp->getVariable()->getHandle();
  auto intResult = saveInt->getVariable()->getHandle();

  for (size_t i = 0; i < size; i++) {
    EXPECT_NEAR(fpResult.raw(i), intResult.raw(i), 0.05);
  }
}

/// Check Non-square kernel for conv.
TEST_P(Operator, NonSquareKernelConvolution) {
  auto *input = mod_.createVariable(ElemKind::FloatTy, {1, 4, 4, 1}, "input");
  auto IH = input->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }

  auto filter = mod_.createVariable(ElemKind::FloatTy, {1, 2, 3, 1}, "filter",
                                    VisibilityKind::Public, false);
  auto FH = filter->getHandle();
  for (size_t i = 0; i < 1 * 2 * 3; i++) {
    FH.raw(i) = i + 1;
  }

  auto *zeroBias = mod_.createVariable(ElemKind::FloatTy, {1}, "bias",
                                       VisibilityKind::Public, false);
  zeroBias->getPayload().zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 3, 2, 1});
  ConvolutionNode *CN = F_->createConv("Conv", input, filter, zeroBias, outTy,
                                       {2, 3}, {1, 1}, {0, 0, 0, 0}, 1);
  SaveNode *S = F_->createSave("save", CN);
  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});
  Tensor &result = S->getVariable()->getPayload();

  static const float ref[] = {106, 127, 190, 211, 274, 295};
  for (size_t i = 0; i < 6; i++)
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
}

/// Check Non-square kernel for AveragePool.
TEST_P(InterpAndCPU, NonSquareKernelAveragePool) {
  auto *input = mod_.createVariable(ElemKind::FloatTy, {1, 4, 4, 1}, "input");
  auto IH = input->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }
  auto *Pool = F_->createAvgPool("pool", input, {2, 3}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", Pool);
  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});
  Tensor &result = S->getVariable()->getPayload();

  static const float ref[] = {4, 5, 8, 9, 12, 13};
  for (size_t i = 0; i < 6; i++)
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
}

/// Check Non-square kernel for MaxPool.
TEST_P(InterpAndCPU, NonSquareKernelMaxPool) {
  auto *input = mod_.createVariable(ElemKind::FloatTy, {1, 4, 4, 1}, "input");
  auto IH = input->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }
  auto *Pool = F_->createMaxPool("pool", input, {2, 3}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", Pool);
  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});
  Tensor &result = S->getVariable()->getPayload();

  static const float ref[] = {7, 8, 11, 12, 15, 16};
  for (size_t i = 0; i < 6; i++)
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
}

/// Check Non-square stride for conv.
TEST_P(Operator, NonSquareStrideConvolution) {
  auto *input = mod_.createVariable(ElemKind::FloatTy, {1, 4, 4, 1}, "input");
  auto IH = input->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }

  auto filter = mod_.createVariable(ElemKind::FloatTy, {1, 2, 2, 1}, "filter",
                                    VisibilityKind::Public, false);
  auto FH = filter->getHandle();
  for (size_t i = 0; i < 1 * 2 * 2; i++) {
    FH.raw(i) = i + 1;
  }

  auto *zeroBias = mod_.createVariable(ElemKind::FloatTy, {1}, "bias",
                                       VisibilityKind::Public, false);
  zeroBias->getPayload().zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 2, 2, 1});
  ConvolutionNode *CN = F_->createConv("Conv", input, filter, zeroBias, outTy,
                                       {2, 2}, {3, 2}, {0, 0, 1, 1}, 1);
  SaveNode *S = F_->createSave("save", CN);
  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});
  Tensor &result = S->getVariable()->getPayload();

  static const float ref[] = {44, 64, 41, 47};
  for (size_t i = 0; i < 4; i++)
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
}

/// Check Non-square stride for AveragePool.
TEST_P(InterpAndCPU, NonSquareStrideAveragePool) {
  auto *input = mod_.createVariable(ElemKind::FloatTy, {1, 4, 4, 1}, "input");
  auto IH = input->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }
  auto *Pool = F_->createAvgPool("pool", input, {2, 2}, {3, 2}, {0, 0, 1, 1});
  auto *S = F_->createSave("save", Pool);
  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});
  Tensor &result = S->getVariable()->getPayload();

  static const float ref[] = {3.5, 5.5, 6.75, 7.75};
  for (size_t i = 0; i < 4; i++)
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
}

/// Check Non-square stride for MaxPool.
TEST_P(InterpAndCPU, NonSquareStrideMaxPool) {
  auto *input = mod_.createVariable(ElemKind::FloatTy, {1, 4, 4, 1}, "input");
  auto IH = input->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }
  auto *Pool = F_->createMaxPool("pool", input, {2, 2}, {3, 2}, {0, 0, 1, 1});
  auto *S = F_->createSave("save", Pool);
  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});
  Tensor &result = S->getVariable()->getPayload();

  static const float ref[] = {6, 8, 14, 16};
  for (size_t i = 0; i < 4; i++)
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
}

TEST_P(InterpAndCPU, Int8Sigmoid) {
  constexpr size_t size = 10;
  auto *input = mod_.createVariable(ElemKind::FloatTy, {size}, "input");
  input->getHandle().randomize(-10.0, 10.0, mod_.getPRNG());

  auto *fpSigmoid = F_->createSigmoid("fpSigmoid", input);
  auto *saveFp = F_->createSave("fpSave", fpSigmoid);

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

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto fpResult = saveFp->getVariable()->getHandle();
  auto intResult = saveInt->getVariable()->getHandle();

  for (size_t i = 0; i < size; i++) {
    EXPECT_NEAR(fpResult.raw(i), intResult.raw(i), 0.05);
  }
}

TEST_P(InterpAndCPU, IntLookupTable) {
  constexpr size_t size = 6;
  auto *input = mod_.createVariable(ElemKind::Int8QTy, {size}, 1, 0, "input");
  input->getHandle<int8_t>() = {0, 1, 2, 3, 4, 5};

  auto outTy = mod_.uniqueType(ElemKind::Int8QTy, {size}, 3, 3);

  // Mapping i -> i.
  std::vector<int8_t> initValues(256);
  for (size_t i = 0; i < 256; ++i) {
    initValues[i] = i - 128;
  }

  auto lookupTable =
      F_->createIntLookupTable("lookupTable", input, initValues, outTy);
  auto save = F_->createSave("save", lookupTable);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto result = save->getVariable()->getHandle<int8_t>();
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(result.raw(i), i);
  }
}

/// Check that the sequence of extract-batchedadd-concat works.
TEST_P(Operator, testBatchAdd) {
  unsigned numSlices = 10;
  auto *input =
      mod_.createVariable(ElemKind::FloatTy, {numSlices, 10, 10}, "input");
  auto *slice = mod_.createVariable(ElemKind::FloatTy, {10, 10}, "slice");
  auto *result =
      mod_.createVariable(ElemKind::FloatTy, {numSlices, 10, 10}, "result");

  input->getHandle().randomize(-10.0, 10.0, mod_.getPRNG());
  slice->getHandle().randomize(-10.0, 10.0, mod_.getPRNG());

  std::vector<NodeValue> adds;
  for (size_t i = 0; i < numSlices; i++) {
    auto *ex = F_->createSlice("slice", input, {i, 0, 0}, {i + 1, 10, 10});
    auto *ba = F_->createBatchedAdd("add", ex, slice);
    adds.push_back(ba);
  }

  auto *cc = F_->createConcat("concat", adds, 0);

  // Remove the reference to the graph nodes to allow DCE to remove them.
  adds.clear();

  F_->createSave("save", cc, result);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto RH = result->getPayload().getHandle();
  auto IH = input->getPayload().getHandle();
  auto SH = slice->getPayload().getHandle();

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
  auto *input = mod_.createVariable(ElemKind::FloatTy, {numSlices, 10, 10},
                                    "input", VisibilityKind::Public);
  auto *slice = mod_.createVariable(ElemKind::FloatTy, {10, 10}, "slice",
                                    VisibilityKind::Public);
  auto *result = mod_.createVariable(ElemKind::FloatTy, {numSlices, 10, 10},
                                     "result", VisibilityKind::Public);

  input->getHandle().randomize(-5.0, 5.0, mod_.getPRNG());
  slice->getHandle().randomize(-5.0, 5.0, mod_.getPRNG());

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
  F_->createSave("save", cc, result);

  // Remove the reference to the graph nodes to allow DCE to remove them.
  adds.clear();

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto RH = result->getPayload().getHandle();
  auto IH = input->getPayload().getHandle();
  auto SH = slice->getPayload().getHandle();

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
  auto *data = mod_.createVariable(ElemKind::FloatTy, {3, 2}, "data");
  auto *indices = mod_.createVariable(ElemKind::Int64ITy, {8}, "indices");
  auto *lengths = mod_.createVariable(ElemKind::Int64ITy, {5}, "lengths");

  data->getPayload().getHandle() = {
      1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f,
  };
  indices->getPayload().getHandle<int64_t>() = {
      2, 0, 1, 2, 0, 0, 0, 0,
  };
  lengths->getPayload().getHandle<int64_t>() = {
      2, 0, 2, 1, 3,
  };

  auto R = F_->createSparseLengthsSum("SLS", data, indices, lengths);
  auto S = F_->createSave("save", R);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  Tensor &result = llvm::cast<Variable>(S->getOutput())->getPayload();
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
  auto *data = mod_.createVariable(ElemKind::FloatTy, {3}, "data");
  auto *weights = mod_.createVariable(ElemKind::FloatTy, {8}, "weights");
  auto *indices = mod_.createVariable(ElemKind::Int64ITy, {8}, "indices");
  auto *lengths = mod_.createVariable(ElemKind::Int64ITy, {4}, "lengths");

  data->getPayload().getHandle() = {
      2.0,
      -0.5,
      13,
  };
  weights->getPayload().getHandle() = {
      3, 1, 0, 0, 0, 0, 2, -0.5,
  };
  indices->getPayload().getHandle<int64_t>() = {
      1, 0, 2, 0, 1, 2, 2, 0,
  };
  lengths->getPayload().getHandle<int64_t>() = {
      3,
      0,
      3,
      2,
  };

  auto R = F_->createSparseLengthsWeightedSum("SLWS", data, weights, indices,
                                              lengths);
  auto S = F_->createSave("save", R);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  Tensor &result = llvm::cast<Variable>(S->getOutput())->getPayload();
  Tensor expected(ElemKind::FloatTy, {4});
  expected.getHandle() = {
      0.5,
      0,
      0,
      25,
  };

  EXPECT_TRUE(expected.isEqual(result));
}

/// Stack many slices/reshapes together. Some of these may be turned into tensor
/// views stacked onto each other.
TEST_P(Operator, sliceReshape) {
  auto *X = mod_.createVariable(ElemKind::FloatTy, {3, 3}, "X");
  auto *resultSX = mod_.createVariable(ElemKind::FloatTy, {1, 3}, "resultSX",
                                       VisibilityKind::Public);
  auto *resultSSX = mod_.createVariable(ElemKind::FloatTy, {1, 1}, "resultSSX",
                                        VisibilityKind::Public);
  auto *resultRSX = mod_.createVariable(ElemKind::FloatTy, {3}, "resultRSX",
                                        VisibilityKind::Public);
  auto *resultRSSX = mod_.createVariable(ElemKind::FloatTy, {1}, "resultRSSX",
                                         VisibilityKind::Public);

  auto XH = X->getPayload().getHandle();
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

  F_->createSave("saveSX", SX, resultSX);
  F_->createSave("saveRSX", RSX, resultRSX);
  F_->createSave("saveSSX", SSX, resultSSX);
  F_->createSave("saveRSSX", RSSX, resultRSSX);

  EE_.compile(CompilationMode::Infer, F_);

  EE_.run({}, {});

  // Verify the slice has the same data as the original X.
  auto SXH = resultSX->getPayload().getHandle();
  for (size_t i = 0; i < 3; i++) {
    EXPECT_NEAR(SXH.at({0, i}), XH.at({2, i}), 1E-5);
  }

  // Verify the reshaped slice has the same data as the slice.
  auto RSXH = resultRSX->getPayload().getHandle();
  for (size_t i = 0; i < 3; i++) {
    EXPECT_NEAR(SXH.at({0, i}), RSXH.at({i}), 1E-5);
  }

  // Verify the slice of the slice has the same data as the slice.
  auto SSXH = resultSSX->getPayload().getHandle();
  EXPECT_NEAR(SXH.at({0, 2}), SSXH.at({0, 0}), 1E-5);

  // Verify the reshape of the slice of the slice has the same data as the slice
  // of the slice.
  auto RSSXH = resultRSSX->getPayload().getHandle();
  EXPECT_NEAR(RSSXH.at({0}), SSXH.at({0, 0}), 1E-5);
}

/// Check that the flatten operator produces 2D tensors of the right dimensions.
TEST_P(Operator, Flatten) {
  auto *tensor4D = mod_.createVariable(ElemKind::FloatTy, {3, 2, 4, 3}, "4D");
  tensor4D->getPayload().init(Tensor::InitKind::Xavier, 1.0, mod_.getPRNG());

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
  auto *tensor1D = mod_.createVariable(ElemKind::FloatTy, {15}, "1D");
  tensor1D->getPayload().init(Tensor::InitKind::Xavier, 1.0, mod_.getPRNG());

  auto *reshape1Dto2DAxis1 = F_->createFlatten("flat1Dto2D", tensor1D, 1);
  EXPECT_EQ(reshape1Dto2DAxis1->dims(0).size(), 2);
  EXPECT_EQ(reshape1Dto2DAxis1->dims(0)[0], 15);
  EXPECT_EQ(reshape1Dto2DAxis1->dims(0)[1], 1);

  // Save all the reshapes so that the optimizations won't kill the network.
  auto *save1Dto2D = mod_.createVariable(ElemKind::FloatTy, {15, 1}, "1Dto2D");
  F_->createSave("save1Dto2D", reshape1Dto2DAxis1, save1Dto2D);

  auto *save4Dto2Da1 =
      mod_.createVariable(ElemKind::FloatTy, {3, 24}, "4Dto2Da1");
  F_->createSave("save4Dto2Da1", reshape4Dto2DAxis1, save4Dto2Da1);

  auto *save4Dto2Da2 =
      mod_.createVariable(ElemKind::FloatTy, {6, 12}, "4Dto2Da2");
  F_->createSave("save4Dto2Da2", reshape4Dto2DAxis2, save4Dto2Da2);

  auto *save4Dto2Da3 =
      mod_.createVariable(ElemKind::FloatTy, {24, 3}, "4Dto2Da3");
  F_->createSave("save4Dto2Da3", reshape4Dto2DAxis3, save4Dto2Da3);

  auto *save4Dto2Da4 =
      mod_.createVariable(ElemKind::FloatTy, {72, 1}, "4Dto2Da4");
  F_->createSave("save4Dto2Da4", reshape4Dto2DAxis4, save4Dto2Da4);

  EE_.compile(CompilationMode::Infer, F_);

  EE_.run({}, {});

  // Verify the reshapes have the same data as the original value.
  auto tensor4DH = tensor4D->getPayload().getHandle();
  auto save4Dto2Da1H = save4Dto2Da1->getPayload().getHandle();
  for (size_t i = 0; i < 72; i++) {
    EXPECT_NEAR(tensor4DH.raw(i), save4Dto2Da1H.raw(i), 1E-5);
  }

  auto save4Dto2Da2H = save4Dto2Da2->getPayload().getHandle();
  for (size_t i = 0; i < 72; i++) {
    EXPECT_NEAR(tensor4DH.raw(i), save4Dto2Da2H.raw(i), 1E-5);
  }

  auto save4Dto2Da3H = save4Dto2Da3->getPayload().getHandle();
  for (size_t i = 0; i < 72; i++) {
    EXPECT_NEAR(tensor4DH.raw(i), save4Dto2Da3H.raw(i), 1E-5);
  }

  auto save4Dto2Da4H = save4Dto2Da4->getPayload().getHandle();
  for (size_t i = 0; i < 72; i++) {
    EXPECT_NEAR(tensor4DH.raw(i), save4Dto2Da4H.raw(i), 1E-5);
  }

  auto tensor1DH = tensor1D->getPayload().getHandle();
  auto save1Dto2DH = save1Dto2D->getPayload().getHandle();
  for (size_t i = 0; i < 15; i++) {
    EXPECT_NEAR(tensor1DH.raw(i), save1Dto2DH.raw(i), 1E-5);
  }
}

/// Check that div on Int64ITy/size_t works.
TEST_P(InterpAndCPU, DivSizeT) {
  auto *LHS = mod_.createVariable(ElemKind::Int64ITy, {3, 2}, "LHS");
  auto *RHS = mod_.createVariable(ElemKind::Int64ITy, {3, 2}, "RHS");
  auto *result = mod_.createVariable(ElemKind::Int64ITy, {3, 2}, "result");
  auto LHSH = LHS->getPayload().getHandle<int64_t>();
  auto RHSH = RHS->getPayload().getHandle<int64_t>();

  LHSH = {10, 20, 30, 40, 50, 60};
  RHSH = {2, 20, 100, 41, 3, 59};

  auto R = F_->createDiv("div", LHS, RHS);

  F_->createSave("save", R, result);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  auto H = result->getPayload().getHandle<int64_t>();

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
  auto *logits = mod_.createVariable(ElemKind::FloatTy, {2, 2, 3}, "logits");
  auto *targets = mod_.createVariable(ElemKind::FloatTy, {2, 2, 3}, "targets");
  auto *result = mod_.createVariable(ElemKind::FloatTy, {2, 2}, "result");

  logits->getPayload().getHandle() = {1.0f,  1.2f, -0.5f, 0.1f, 0.6f, 0.5f,
                                      -0.1f, -2.f, 0.3f,  1.f,  2.f,  3.f};
  targets->getPayload().getHandle() = {0.7f, 0.7f, 0.7f, -0.7f, -0.99f, 1.0f,
                                       0.f,  0.f,  0.f,  1.f,   2.f,    3.f};

  auto *R = F_->createSigmoidCrossEntropyWithLogits("SCEL", logits, targets);

  F_->createSave("save", R, result);

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run({}, {});

  Tensor expected(ElemKind::FloatTy, {2, 2});
  expected.getHandle() = {
      0.68687367f,
      0.97332054f,
      0.5418933f,
      -2.50374103f,
  };

  EXPECT_TRUE(expected.isEqual(result->getPayload()));
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
