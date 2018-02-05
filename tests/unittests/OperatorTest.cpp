// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"

#include "gtest/gtest.h"

#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <string>

using namespace glow;

TEST(Operator, matmul) {
  ExecutionEngine EE;

  auto &G = EE.getGraph();

  auto *lhs = G.createVariable(ElemKind::FloatTy, {1, 3, 2}, "lhs");
  auto *rhs = G.createVariable(ElemKind::FloatTy, {1, 2, 1}, "rhs");
  auto *result = G.createVariable(ElemKind::FloatTy, {1, 3, 1}, "result");
  lhs->getPayload().getHandle() = {1, 2, 3, 4, 5, 6};
  rhs->getPayload().getHandle() = {7, 10};

  auto R = G.createBatchedMatMul("MM", lhs, rhs);

  G.createSave("save", R, result);

  EE.compile(CompilationMode::Infer);

  EE.run({}, {});

  auto H = result->getPayload().getHandle();
  EXPECT_NEAR(H.at({0, 0, 0}), 27, 0.001);
  EXPECT_NEAR(H.at({0, 1, 0}), 61, 0.001);
  EXPECT_NEAR(H.at({0, 2, 0}), 95, 0.001);
}

TEST(Operator, batchedReduceAdd) {
  ExecutionEngine EE;

  auto &G = EE.getGraph();

  auto *batch = G.createVariable(ElemKind::FloatTy, {2, 4}, "batch");
  auto *result = G.createVariable(ElemKind::FloatTy, {4}, "result");
  batch->getPayload().getHandle() = {10, 20, 30, 40, 1, 2, 3, 4};

  auto R =
      G.createBatchedReduce("reduce.add", BatchedReduceNode::Mode::Add, batch);

  G.createSave("save", R, result);

  EE.compile(CompilationMode::Infer);

  EE.run({}, {});

  auto H = result->getPayload().getHandle();
  EXPECT_NEAR(H.at({0}), 11, 0.001);
  EXPECT_NEAR(H.at({1}), 22, 0.001);
  EXPECT_NEAR(H.at({2}), 33, 0.001);
  EXPECT_NEAR(H.at({3}), 44, 0.001);
}

TEST(Operator, batchedBatchedAdd) {
  ExecutionEngine EE;

  auto &G = EE.getGraph();

  auto *batch = G.createVariable(ElemKind::FloatTy, {2, 3, 3}, "batch");
  auto *added = G.createVariable(ElemKind::FloatTy, {3, 3}, "added");
  auto *result = G.createVariable(ElemKind::FloatTy, {2, 3, 3}, "result");

  batch->getPayload().getHandle() = {9, 8, 7, 6, 5,  4,  3,  4,  5,
                                     6, 7, 8, 9, 10, 11, 12, 13, 14};
  added->getPayload().getHandle().clear(1.0);

  auto R = G.createBatchedArithmetic(
      "batch.add", BatchedArithmeticNode::Mode::Add, batch, added);
  G.createSave("save", R, result);

  EE.compile(CompilationMode::Infer);

  EE.run({}, {});

  auto H = result->getPayload().getHandle();
  EXPECT_NEAR(H.at({0, 0, 0}), 10, 0.001);
  EXPECT_NEAR(H.at({0, 0, 1}), 9, 0.001);
  EXPECT_NEAR(H.at({0, 0, 2}), 8, 0.001);
  EXPECT_NEAR(H.at({0, 1, 0}), 7, 0.001);
}

/// Broadcast a Tensor of shape (2,1) to (3,2,4,2) with axis 1.
TEST(Operator, broadcast) {
  ExecutionEngine EE;

  auto &G = EE.getGraph();

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

  auto *B = G.createVariable(ElemKind::FloatTy, dims_B, "B");
  auto H_B = B->getPayload().getHandle();
  H_B = {20, 10};

  const unsigned axis = 1;

  auto R = G.createBroadcast("broadcasted", B, dims_A, axis);
  auto *broadcasted = G.createVariable(ElemKind::FloatTy, dims_A, "A");
  G.createSave("save", R, broadcasted);

  EE.compile(CompilationMode::Infer);

  EE.run({}, {});

  auto broadcastedBHandle = broadcasted->getPayload().getHandle();
  // Verify broadcasted B has same shape.
  EXPECT_EQ(numDims_A, broadcastedBHandle.dims().size());
  for (int i = 0; i < broadcastedBHandle.dims().size(); i++) {
    EXPECT_EQ(dims_A[i], broadcastedBHandle.dims()[i]);
  }

  // Look at the two values in X_B and verify in the three dimensions it was
  // broadcasted that the values were correctly broadcasted.
  const size_t k_B = 0;
  for (size_t j_B = 0; j_B < dimY_B; ++j_B) {
    const float origVal = H_B.at({j_B, k_B});
    const size_t j_A = j_B; // This dim was not broadcasted (dims were equal).
    for (size_t i_A = 0; i_A < dimX_A; ++i_A) {
      for (size_t k_A = 0; k_A < dimZ_A; ++k_A) {
        for (size_t l_A = 0; l_A < dimW_A; ++l_A) {
          EXPECT_EQ(origVal, broadcastedBHandle.at({i_A, j_A, k_A, l_A}));
        }
      }
    }
  }
}

TEST(Operator, TopK) {
  ExecutionEngine EE;

  auto &G = EE.getGraph();

  auto *inp = G.createVariable(ElemKind::FloatTy, {3, 1, 5}, "input");
  auto *values = G.createVariable(ElemKind::FloatTy, {3, 1, 3}, "values");
  auto *indices = G.createVariable(ElemKind::IndexTy, {3, 1, 3}, "indices");

  inp->getPayload().getHandle() = {
      28, 4, 411, 19, 42, 0.4, 0.4, 0.4, -0.4, 0.45, 7, 5, 9, 8, 100,
  };

  auto R = G.createTopK("TopK", inp, 3);

  G.createSave("save.values", {R, 0}, values);
  G.createSave("save.indices", {R, 1}, indices);

  EE.compile(CompilationMode::Infer);

  EE.run({}, {});

  auto V = values->getPayload().getHandle();
  auto I = indices->getPayload().getHandle<size_t>();

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

TEST(Operator, Gather) {
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
  ExecutionEngine EE;

  auto &G = EE.getGraph();

  auto *data = G.createVariable(ElemKind::FloatTy, {3, 2}, "data");
  auto *indices = G.createVariable(ElemKind::IndexTy, {2, 4}, "indices");
  auto *result = G.createVariable(ElemKind::FloatTy, {2, 4, 2}, "result");

  data->getPayload().getHandle() = {
      1.0, 1.2, 2.3, 3.4, 4.5, 5.7,
  };
  indices->getPayload().getHandle<size_t>() = {
      0, 1, 0, 1, 1, 2, 2, 0,
  };

  auto R = G.createGather("gather", data, indices);

  G.createSave("save", R, result);

  EE.compile(CompilationMode::Infer);

  EE.run({}, {});

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

