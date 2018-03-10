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

class Operator : public ::testing::TestWithParam<BackendKind> {
protected:
  ExecutionEngine EE{GetParam()};
};

TEST_P(Operator, pow) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *X = mod.createVariable(ElemKind::FloatTy, {1, 1, 3}, "X");
  auto *Y = mod.createVariable(ElemKind::FloatTy, {2}, "Y");
  X->getPayload().getHandle() = {5, 0.1, -3};
  Y->getPayload().getHandle() = {2, 100};

  auto *Pow1 = F->createPow("Pow1", X, 2.0);
  auto *Pow2 = F->createPow("Pow2", Y, 0.5);

  auto *Save1 = F->createSave("save", Pow1);
  auto *Save2 = F->createSave("save", Pow2);

  EE.compile(CompilationMode::Infer, F);

  EE.run({}, {});

  auto HX = llvm::cast<Variable>(Save1->getOutput())->getPayload().getHandle();
  EXPECT_NEAR(HX.at({0, 0, 0}), 25, 1E-5);
  EXPECT_NEAR(HX.at({0, 0, 1}), 0.01, 1E-5);
  EXPECT_NEAR(HX.at({0, 0, 2}), 9, 1E-5);

  auto HY = llvm::cast<Variable>(Save2->getOutput())->getPayload().getHandle();
  EXPECT_NEAR(HY.at({0}), sqrt(2.0), 1E-5);
  EXPECT_NEAR(HY.at({1}), 10, 1E-5);
}

TEST_P(Operator, matmul) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *lhs = mod.createVariable(ElemKind::FloatTy, {3, 2}, "lhs");
  auto *rhs = mod.createVariable(ElemKind::FloatTy, {2, 1}, "rhs");
  auto *result = mod.createVariable(ElemKind::FloatTy, {3, 1}, "result");
  lhs->getPayload().getHandle() = {1, 2, 3, 4, 5, 6};
  rhs->getPayload().getHandle() = {7, 10};

  auto R = F->createMatMul("MM", lhs, rhs);

  F->createSave("save", R, result);

  EE.compile(CompilationMode::Infer, F);

  EE.run({}, {});

  auto H = result->getPayload().getHandle();
  EXPECT_NEAR(H.at({0, 0}), 27, 0.001);
  EXPECT_NEAR(H.at({1, 0}), 61, 0.001);
  EXPECT_NEAR(H.at({2, 0}), 95, 0.001);
}

TEST_P(Operator, batchedReduceAdd) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *batch = mod.createVariable(ElemKind::FloatTy, {2, 4}, "batch");
  auto *result = mod.createVariable(ElemKind::FloatTy, {4}, "result");
  batch->getPayload().getHandle() = {10, 20, 30, 40, 1, 2, 3, 4};

  auto R = F->createBatchedReduceAdd("reduce.add", batch);

  F->createSave("save", R, result);

  EE.compile(CompilationMode::Infer, F);

  EE.run({}, {});

  auto H = result->getPayload().getHandle();
  EXPECT_NEAR(H.at({0}), 11, 0.001);
  EXPECT_NEAR(H.at({1}), 22, 0.001);
  EXPECT_NEAR(H.at({2}), 33, 0.001);
  EXPECT_NEAR(H.at({3}), 44, 0.001);
}

TEST_P(Operator, batchedBatchedAdd) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *batch = mod.createVariable(ElemKind::FloatTy, {2, 3, 3}, "batch");
  auto *added = mod.createVariable(ElemKind::FloatTy, {3, 3}, "added");
  auto *result = mod.createVariable(ElemKind::FloatTy, {2, 3, 3}, "result");

  batch->getPayload().getHandle() = {9, 8, 7, 6, 5,  4,  3,  4,  5,
                                     6, 7, 8, 9, 10, 11, 12, 13, 14};
  added->getPayload().getHandle().clear(1.0);

  auto R = F->createBatchedAdd("batch.add", batch, added);
  F->createSave("save", R, result);

  EE.compile(CompilationMode::Infer, F);

  EE.run({}, {});

  auto H = result->getPayload().getHandle();
  EXPECT_NEAR(H.at({0, 0, 0}), 10, 0.001);
  EXPECT_NEAR(H.at({0, 0, 1}), 9, 0.001);
  EXPECT_NEAR(H.at({0, 0, 2}), 8, 0.001);
  EXPECT_NEAR(H.at({0, 1, 0}), 7, 0.001);
}

/// Broadcast Tensor of shape (2,1,1) to (2,4,2) with axis 0.
TEST_P(Operator, broadcastSimple) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

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

  auto *B = mod.createVariable(ElemKind::FloatTy, dims_B, "B");
  auto H_B = B->getPayload().getHandle();
  H_B = {20, 10};

  const unsigned axis = 0;

  auto R = F->createBroadcast("broadcasted", B, dims_A, axis);
  auto *broadcasted = mod.createVariable(ElemKind::FloatTy, dims_A, "A");
  F->createSave("save", R, broadcasted);

  EE.compile(CompilationMode::Infer, F);

  EE.run({}, {});

  auto broadcastedBHandle = broadcasted->getPayload().getHandle();
  // Verify broadcasted B has same shape.
  EXPECT_EQ(numDims_A, broadcastedBHandle.dims().size());
  for (size_t i = 0; i < broadcastedBHandle.dims().size(); i++) {
    EXPECT_EQ(dims_A[i], broadcastedBHandle.dims()[i]);
  }

  // Look at the two values in X_B and verify in the three dimensions it was
  // broadcasted that the values were correctly broadcasted.
  const size_t k_B = 0;
  const size_t l_B = 0;
  for (size_t j_B = 0; j_B < dimY_B; ++j_B) {
    const float origVal = H_B.at({j_B, k_B, l_B});
    const size_t j_A = j_B; // This dim was not broadcasted (dims were equal).
    for (size_t k_A = 0; k_A < dimZ_A; ++k_A) {
      for (size_t l_A = 0; l_A < dimW_A; ++l_A) {
        EXPECT_EQ(origVal, broadcastedBHandle.at({j_A, k_A, l_A}));
      }
    }
  }
}

/// Broadcast a Tensor of shape (2,1) to (3,2,4,2) with axis 1.
TEST_P(Operator, broadcast) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

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

  auto *B = mod.createVariable(ElemKind::FloatTy, dims_B, "B");
  auto H_B = B->getPayload().getHandle();
  H_B = {20, 10};

  const unsigned axis = 1;

  auto R = F->createBroadcast("broadcasted", B, dims_A, axis);
  auto *broadcasted = mod.createVariable(ElemKind::FloatTy, dims_A, "A");
  F->createSave("save", R, broadcasted);

  EE.compile(CompilationMode::Infer, F);

  EE.run({}, {});

  auto broadcastedBHandle = broadcasted->getPayload().getHandle();
  // Verify broadcasted B has same shape.
  EXPECT_EQ(numDims_A, broadcastedBHandle.dims().size());
  for (size_t i = 0; i < broadcastedBHandle.dims().size(); i++) {
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

TEST_P(Operator, minElem) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  unsigned len = 5;

  auto *LHS = mod.createVariable(ElemKind::FloatTy, {len}, "lhs");
  auto *RHS = mod.createVariable(ElemKind::FloatTy, {len}, "rhs");
  auto *min = F->createMin("min", LHS, RHS);
  auto *save = F->createSave("min", min);

  LHS->getHandle().randomize(-10, 10);
  RHS->getHandle().randomize(-10, 10);

  EE.compile(CompilationMode::Infer, F);
  EE.run({}, {});

  auto resultH = save->getVariable()->getHandle();
  auto LHSH = LHS->getHandle();
  auto RHSH = RHS->getHandle();

  for (size_t i = 0; i < len; i++) {
    EXPECT_EQ(resultH.raw(i), std::min(LHSH.raw(i), RHSH.raw(i)));
  }
}

TEST_P(Operator, TopK) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *inp = mod.createVariable(ElemKind::FloatTy, {3, 1, 5}, "input");
  auto *values = mod.createVariable(ElemKind::FloatTy, {3, 1, 3}, "values");
  auto *indices = mod.createVariable(ElemKind::IndexTy, {3, 1, 3}, "indices");

  inp->getPayload().getHandle() = {
      28, 4, 411, 19, 42, 0.4, 0.4, 0.4, -0.4, 0.45, 7, 5, 9, 8, 100,
  };

  auto R = F->createTopK("TopK", inp, 3);

  F->createSave("save.values", {R, 0}, values);
  F->createSave("save.indices", {R, 1}, indices);

  EE.compile(CompilationMode::Infer, F);

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
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *data = mod.createVariable(ElemKind::FloatTy, {3, 2}, "data");
  auto *indices = mod.createVariable(ElemKind::IndexTy, {2, 4}, "indices");
  auto *result = mod.createVariable(ElemKind::FloatTy, {2, 4, 2}, "result");

  data->getPayload().getHandle() = {
      1.0, 1.2, 2.3, 3.4, 4.5, 5.7,
  };
  indices->getPayload().getHandle<size_t>() = {
      0, 1, 0, 1, 1, 2, 2, 0,
  };

  auto R = F->createGather("gather", data, indices);

  F->createSave("save", R, result);

  EE.compile(CompilationMode::Infer, F);

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

TEST(Interpreter, QuantizeAndDequantize) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  Tensor inputs(ElemKind::FloatTy, {1, 4});
  inputs.getHandle() = {1, 1.2, 0.5, 1.3};

  auto *A = mod.createVariable(ElemKind::FloatTy, {1, 4}, "A",
                               Variable::VisibilityKind::Public);

  auto qType = mod.uniqueType(ElemKind::Int8QTy, {1, 4}, 0.05, -138);
  auto *quantize = F->createQuantize("quantize", A, qType);
  auto *dequantize = F->createDequantize("dequantize", quantize);
  auto *result = F->createSave("save", dequantize);

  EE.compile(CompilationMode::Infer, F);
  EE.run({A}, {&inputs});

  auto resultHandle = result->getVariable()->getHandle();
  auto expectedHandle = inputs.getHandle();
  EXPECT_TRUE(expectedHandle.isEqual(resultHandle));
}

TEST(OperatorInterpOnly, IntMatMul) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  // The scaling factor 1.4x was carefully selected to make sure we don't
  // overflow or underflow the calculation.
  TypeRef resTy = mod.uniqueType(ElemKind::Int8QTy, {3, 3}, 0.60, 4);
  TypeRef lhsTy = mod.uniqueType(ElemKind::Int8QTy, {3, 3}, 0.075, -2);
  TypeRef rhsTy = mod.uniqueType(ElemKind::Int8QTy, {3, 3}, 0.075, 2);

  auto *res = mod.createVariable(ElemKind::FloatTy, {3, 3}, "res");
  auto *lhs = mod.createVariable(ElemKind::FloatTy, {3, 3}, "lhs");
  auto *rhs = mod.createVariable(ElemKind::FloatTy, {3, 3}, "rhs");

  lhs->getPayload().getHandle() = {
      1.0, 2.0, 3.0, 4.0, 5.0, -5.0, -4.0, -3.0, 9.0,
  };

  rhs->getPayload().getHandle() = {
      0.1, -0.2, 0.3, 9.0, -8.0, 7.0, 6.0, 5.0, 9.0,
  };

  auto *lhsq = F->createQuantize("lhs.q", lhs, lhsTy);
  auto *rhsq = F->createQuantize("rhs.q", rhs, rhsTy);

  auto *matmulq = F->createMatMul("matmul.q", resTy, lhsq, rhsq);

  auto *rq = F->createDequantize("dequant", matmulq);

  F->createSave("save", rq, res);
  EE.compile(CompilationMode::Infer, F);

  EE.run({}, {});

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

TEST(OperatorInterpOnly, IntBatchedArith) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  TypeRef resTy = mod.uniqueType(ElemKind::Int8QTy, {1, 3, 3}, 0.10, 1.0);
  TypeRef lhsTy = mod.uniqueType(ElemKind::Int8QTy, {1, 3, 3}, 0.11, 4.0);
  TypeRef rhsTy = mod.uniqueType(ElemKind::Int8QTy, {3, 3}, 0.14, -2.0);

  auto *res = mod.createVariable(ElemKind::FloatTy, {1, 3, 3}, "res");
  auto *lhs = mod.createVariable(ElemKind::FloatTy, {1, 3, 3}, "lhs");
  auto *rhs = mod.createVariable(ElemKind::FloatTy, {3, 3}, "rhs");

  lhs->getPayload().getHandle() = {
      8.7, 6.5, 4.3, 2.1, 1.0, -5.1, -4.0, -12.0, 0.2,
  };

  rhs->getPayload().getHandle() = {
      -9.1, -0.4, 1.3, 2.2, -8.1, 7.6, -6.4, 10.0, 9.1,
  };

  auto *lhsq = F->createQuantize("lhs.q", lhs, lhsTy);
  auto *rhsq = F->createQuantize("rhs.q", rhs, rhsTy);

  auto *matmulq = F->createBatchedAdd("add", resTy, lhsq, rhsq);

  auto *rq = F->createDequantize("dequant", matmulq);

  F->createSave("save", rq, res);
  EE.compile(CompilationMode::Infer, F);

  EE.run({}, {});

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
  EXPECT_NEAR(H.at({0, 2, 1}), -2, 0.1);
  EXPECT_NEAR(H.at({0, 2, 2}), 9.3, 0.1);
}

TEST_P(Operator, IntConvolution) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  // In this test we generate a Floating-point based convolution and an integer
  // convolution. We pass the same values and then subtract the results. We
  // check that the results are below some known delta.

  // In this test the output of the convolution is in the range [-256 ... 256].
  // The inputs (specified below) are in the range [-1 .. 1],

  auto *input = mod.createVariable(ElemKind::FloatTy, {1, 10, 10, 3}, "in");
  auto *conv = F->createConv("conv", input, 10, 5, 1, 0);
  auto *res = mod.createVariable(ElemKind::FloatTy, conv->dims(), "res");

  auto filter = conv->getFilter();
  auto bias = conv->getBias();

  input->getPayload().getHandle().randomize(-1.0, 1.0);
  llvm::cast<Variable>(bias)->getPayload().getHandle().randomize(-2.0, 2.0);

  TypeRef resTy = mod.uniqueType(ElemKind::Int8QTy, res->dims(), 0.08, 0.0);
  TypeRef inputTy = mod.uniqueType(ElemKind::Int8QTy, input->dims(), 0.01, 0.0);
  TypeRef filterTy =
      mod.uniqueType(ElemKind::Int8QTy, filter.dims(), 0.01, 0.0);
  TypeRef biasTy = mod.uniqueType(ElemKind::Int8QTy, bias.dims(), 0.04, 0.0);

  auto *inputq = F->createQuantize("input.q", input, inputTy);
  auto *filterq = F->createQuantize("filter.q", filter, filterTy);
  auto *biasq = F->createQuantize("bias.q", bias, biasTy);

  auto *convq =
      F->createConv("convq", inputq, filterq, biasq, resTy, conv->getDepth(),
                    conv->getKernel(), conv->getStride(), conv->getPad());
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

TEST(OperatorInterpOnly, IntConcat) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto A = mod.createVariable(ElemKind::FloatTy, {3, 3}, "A");
  auto B = mod.createVariable(ElemKind::FloatTy, {2, 3}, "B");
  A->getHandle().randomize(-1.0, 1.0);
  B->getHandle().randomize(-1.0, 1.0);

  auto ATy = mod.uniqueType(ElemKind::Int8QTy, A->dims(), 0.01, 0);
  auto BTy = mod.uniqueType(ElemKind::Int8QTy, B->dims(), 0.01, 0);
  auto outTy = mod.uniqueType(ElemKind::Int8QTy, {5, 3}, 0.01, 0);

  auto QA = F->createQuantize("QA", A, ATy);
  auto QB = F->createQuantize("QB", B, BTy);

  auto C = F->createConcat("concat", {A, B}, 0);
  auto CQ = F->createConcat("concatQ", {QA, QB}, 0, outTy);
  auto DCQ = F->createDequantize("DQ", CQ);

  // Subtract the results of the Concat from the quantized Concat.
  auto sub = F->createSub("compare", C, DCQ);

  auto res = F->createSave("save", sub);
  EE.compile(CompilationMode::Infer, F);
  EE.run({}, {});

  auto R = res->getVariable()->getHandle();
  // Check that the difference in the results is less than 0.2.
  for (int i = 0, e = R.size(); i < e; i++) {
    EXPECT_NEAR(R.raw(i), 0, 0.2);
  }
}

TEST(Operator, IntFC) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  // In this test we subtract the outputs of a quantized FC and a floating-point
  // FC and ensure that the error is below some low value.

  auto *input = mod.createVariable(ElemKind::FloatTy, {1, 10, 10, 3}, "in");
  auto *fc = F->createFullyConnected("FC", input, 30);
  auto *res = mod.createVariable(ElemKind::FloatTy, fc->dims(), "res");

  auto weights = fc->getWeights();
  auto bias = fc->getBias();

  input->getPayload().getHandle().randomize(-1.0, 1.0);
  llvm::cast<Variable>(bias)->getPayload().getHandle().randomize(0, 0.00001);
  llvm::cast<Variable>(weights)->getPayload().getHandle().randomize(-1.1, 1.1);

  TypeRef resTy = mod.uniqueType(ElemKind::Int8QTy, res->dims(), 0.15, 4);
  TypeRef inputTy = mod.uniqueType(ElemKind::Int8QTy, input->dims(), 0.01, 0);
  TypeRef weightsTy =
      mod.uniqueType(ElemKind::Int8QTy, weights.dims(), 0.01, 2);
  TypeRef biasTy = mod.uniqueType(ElemKind::Int8QTy, bias.dims(), 0.02, 1);

  auto *inputq = F->createQuantize("input.q", input, inputTy);
  auto *weightsq = F->createQuantize("filter.q", weights, weightsTy);
  auto *biasq = F->createQuantize("bias.q", bias, biasTy);

  auto *fcq = F->createFullyConnected("fcq", inputq, weightsq, biasq, resTy);
  auto *dequantRes = F->createDequantize("dequant", fcq);

  // Subtract the results of the convolution from the quantized fc.
  auto *sub = F->createSub("compare", dequantRes, fc);

  F->createSave("save", sub, res);
  EE.compile(CompilationMode::Infer, F);
  EE.run({}, {});

  auto H = res->getPayload().getHandle();
  // Check that the difference in the results is less than 0.2.
  for (int i = 0, e = H.size(); i < e; i++) {
    EXPECT_NEAR(H.raw(i), 0, 0.2);
  }
}

TEST(Operator, CrossEntropyLossTest) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *P = mod.createVariable(ElemKind::FloatTy, {2, 3}, "P");
  auto *Y = mod.createVariable(ElemKind::IndexTy, {2}, "Y");
  auto *L = mod.createVariable(ElemKind::FloatTy, {1}, "L");

  P->getPayload().getHandle() = {0.2, 0.5, 0.3, 0.4, 0.3, 0.3};
  Y->getPayload().getHandle<size_t>() = {1, 2};
  auto *ceLoss = F->createCrossEntropyLoss("CELoss", P, Y);
  F->createSave("save", ceLoss, L);
  EE.compile(CompilationMode::Infer, F);
  EE.run({}, {});
  auto R = L->getPayload().getHandle();
  EXPECT_NEAR(R.at({0}), -log(0.5) - log(0.3), 0.1);
}

TEST(OperatorInterpOnly, RescaleNode) {
  // Check the outputs of the RescaleQuantized operation.
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *input = mod.createVariable(ElemKind::Int8QTy, {4, 10}, 0.4, -3, "input",
                                   Variable::VisibilityKind::Public,
                                   Variable::TrainKind::Broadcast, 40);

  auto *output = mod.createVariable(ElemKind::Int8QTy, {4, 10}, 0.4, -3,
                                    "output", Variable::VisibilityKind::Public,
                                    Variable::TrainKind::None);

  auto T1 = mod.uniqueType(ElemKind::Int8QTy, {4, 10}, 0.7, 5);
  auto T2 = mod.uniqueType(ElemKind::Int8QTy, {4, 10}, 0.3, -4);

  // Test a sequence of rescale operations that the optimizer may try to
  // optimize at some point.
  auto *X = F->createRescaleQuantized("R1", input, T1);
  auto *Y = F->createRescaleQuantized("R2", X, T2);
  auto *Z = F->createRescaleQuantized("R3", Y, output->getType());

  F->createSave("save", Z, output);
  EE.compile(CompilationMode::Infer, F);
  EE.run({}, {});

  auto RI = input->getPayload().getHandle<int8_t>();
  auto RO = output->getPayload().getHandle<int8_t>();

  EXPECT_EQ(RI.raw(0), 40);
  EXPECT_NEAR(RO.raw(0), 40, 1);
}

TEST(OperatorInterpOnly, QuantizedArithmeticNode) {
  const int len = 100;

  // In this test we check the correctness of the quantized MAX, ADD,
  // MIN nodes as well as how they interact with the rescaling node.
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *A = mod.createVariable(ElemKind::FloatTy, {len}, "A",
                               Variable::VisibilityKind::Public);
  auto *B = mod.createVariable(ElemKind::FloatTy, {len}, "B",
                               Variable::VisibilityKind::Public);
  auto *O1 = mod.createVariable(ElemKind::FloatTy, {len}, "OutSum",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);
  auto *O2 = mod.createVariable(ElemKind::FloatTy, {len}, "OutMax",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);
  auto *O3 = mod.createVariable(ElemKind::FloatTy, {len}, "OutMin",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);
  auto *O4 = mod.createVariable(ElemKind::FloatTy, {len}, "OutMul",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);

  auto AH = A->getHandle();
  auto BH = B->getHandle();
  auto O1H = O1->getHandle();
  auto O2H = O2->getHandle();
  auto O3H = O3->getHandle();
  auto O4H = O4->getHandle();

  AH.randomize(-10, 10);
  BH.randomize(-10, 10);

  auto TA = mod.uniqueType(ElemKind::Int8QTy, {len}, 0.2, 0);
  auto TB = mod.uniqueType(ElemKind::Int8QTy, {len}, 0.1, 0);
  auto TO1 = mod.uniqueType(ElemKind::Int8QTy, {len}, 1.0, 0);
  auto TO2 = mod.uniqueType(ElemKind::Int8QTy, {len}, 0.9, 0);
  auto TO3 = mod.uniqueType(ElemKind::Int8QTy, {len}, 1.1, 0);
  auto TO4 = mod.uniqueType(ElemKind::Int8QTy, {len}, 1.2, 0);

  // Quantize input vars and apply max/add/min/mul quantized.
  auto *QA = F->createQuantize("QA", A, TA);
  auto *QB = F->createQuantize("QB", B, TB);
  Node *max = F->createMax("max", TO1, QA, QB);
  Node *add = F->createAdd("add", TO2, QA, QB);
  Node *min = F->createMin("min", TO1, QA, QB);
  Node *mul = F->createMul("mul", TO1, QA, QB);

  // Rescale quantized results.
  max = F->createRescaleQuantized("rescaleMax", max, TO3);
  add = F->createRescaleQuantized("rescaleAdd", add, TO4);
  min = F->createRescaleQuantized("rescaleMin", min, TO3);

  // Dequantize results back to fp.
  add = F->createDequantize("addDq", add);
  max = F->createDequantize("maxDq", max);
  min = F->createDequantize("minDq", min);
  mul = F->createDequantize("mulDq", mul);

  // Save results of the operations.
  F->createSave("saveAdd", add, O1);
  F->createSave("saveMax", max, O2);
  F->createSave("saveMin", min, O3);
  F->createSave("saveMul", mul, O4);

  EE.compile(CompilationMode::Infer, F);
  EE.run({}, {});

  for (size_t i = 0; i < len; i++) {
    auto max = std::max(AH.at({i}), BH.at({i}));
    auto add = AH.at({i}) + BH.at({i});
    auto min = std::min(AH.at({i}), BH.at({i}));
    auto mul = AH.at({i}) * BH.at({i});

    // We generate numbers up to 110, so a difference of 2 (~2%) is reasonable.
    EXPECT_NEAR(add, O1H.at({i}), 2.0);
    EXPECT_NEAR(max, O2H.at({i}), 2.0);
    EXPECT_NEAR(min, O3H.at({i}), 2.0);
    EXPECT_NEAR(mul, O4H.at({i}), 2.0);
  }
}

TEST(OperatorInterpOnly, TestQuantizedRescaleSequence) {
  const int len = 100;

  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *A = mod.createVariable(ElemKind::FloatTy, {len}, "A",
                               Variable::VisibilityKind::Public);
  auto *O = mod.createVariable(ElemKind::FloatTy, {len}, "Out",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);

  auto AH = A->getPayload().getHandle();
  auto OH = O->getPayload().getHandle();

  // Notice that the range below is the an approximation of the scale factors in
  // T3 and T4. If we increase the size of the range we may start losing some
  // values.
  AH.randomize(-12, 12);

  auto T1 = mod.uniqueType(ElemKind::Int8QTy, {len}, 1.0, 0);
  auto T2 = mod.uniqueType(ElemKind::Int8QTy, {len}, 0.9, 2);
  auto T3 = mod.uniqueType(ElemKind::Int8QTy, {len}, 0.1, -3);
  auto T4 = mod.uniqueType(ElemKind::Int8QTy, {len}, 0.1, 7);
  auto T5 = mod.uniqueType(ElemKind::Int8QTy, {len}, 0.3, -3);

  Node *R = F->createQuantize("R", A, T1);
  // Check that a sequence of type conversions does not change the result.
  R = F->createRescaleQuantized("R", R, T1);
  R = F->createRescaleQuantized("R", R, T2);
  R = F->createRescaleQuantized("R", R, T3);
  // Check that adding the quantized zero does not change the result.
  auto *G = F->createSplat("splatZero", T3, 0.0);
  R = F->createAdd("addZero", G, R);
  R = F->createRescaleQuantized("R", R, T4);
  R = F->createRescaleQuantized("R", R, T5);
  R = F->createRescaleQuantized("R", R, T1);
  auto *DQ = F->createDequantize("DQ", R);

  // Test a sequence of rescale operations t
  F->createSave("save", DQ, O);
  EE.compile(CompilationMode::Infer, F);
  EE.run({}, {});

  for (size_t i = 0; i < len; i++) {
    EXPECT_NEAR(AH.at({i}), OH.at({i}), 1.0);
  }
}

TEST_P(Operator, FCGradientCheck) {
  // Create net representing A*X+Y=B, where X and Y are trainable, while
  // A and B are fixed. Record gradients for X and Y after 3 steps and compare
  // with reference values.
  auto &mod = EE.getModule();
  auto *A = mod.createVariable(ElemKind::FloatTy, {2, 1}, "A",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  auto *B = mod.createVariable(ElemKind::FloatTy, {2, 1}, "B",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  auto *X = mod.createVariable(ElemKind::FloatTy, {1, 1}, "X",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::Broadcast, -1.26274);
  auto *Y = mod.createVariable(ElemKind::FloatTy, {1}, "Y",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::Broadcast, 0.10000);
  Function *F = mod.createFunction("main");
  auto *FC = F->createFullyConnected("fc", A, X, Y);
  auto *S = F->createRegression("reg", FC, B);
  F->createSave("ret", S);

  Tensor initA(ElemKind::FloatTy, {2, 1});
  Tensor initB(ElemKind::FloatTy, {2, 1});
  initA.getHandle() = {4.2, 9.875};
  initB.getHandle() = {-13.1, 3.14};

  Function *DF = glow::differentiate(F, EE.getConfig(), "d_main");
  EE.compile(CompilationMode::Train, DF);

  EE.runBatch(3, {A, B}, {&initA, &initB});

  EXPECT_NEAR(X->getPayload().getHandle().raw(0), -0.21294, 1E-5);
  EXPECT_NEAR(Y->getPayload().getHandle().raw(0), 0.01656, 1E-5);
}

TEST_P(Operator, concatVectors) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  F->setName("concatVectors");

  auto *V1 = mod.createVariable(ElemKind::IndexTy, {10}, "V1",
                                Variable::VisibilityKind::Public);
  auto *V2 = mod.createVariable(ElemKind::IndexTy, {20}, "V2",
                                Variable::VisibilityKind::Public);
  auto *V3 = mod.createVariable(ElemKind::IndexTy, {30}, "V3",
                                Variable::VisibilityKind::Public);

  Node *L = F->createConcat("concat", {V1, V2, V3}, 0);
  auto *result = F->createSave("ret", L);

  Tensor I1(ElemKind::IndexTy, {10});
  Tensor I2(ElemKind::IndexTy, {20});
  Tensor I3(ElemKind::IndexTy, {30});

  for (size_t i = 0; i < 10; i++) {
    I1.getHandle<size_t>().at({i}) = i;

    I2.getHandle<size_t>().at({i}) = i + 10;
    I2.getHandle<size_t>().at({i + 10}) = i + 20;
    I3.getHandle<size_t>().at({i}) = i + 30;
    I3.getHandle<size_t>().at({i + 10}) = i + 40;
    I3.getHandle<size_t>().at({i + 20}) = i + 50;
  }

  EE.compile(CompilationMode::Infer, F);

  // Testing the output vector.
  EE.run({V1, V2, V3}, {&I1, &I2, &I3});
  auto RNWH = result->getVariable()->getPayload().getHandle<size_t>();
  (void)RNWH;

  for (size_t i = 0; i < 60; i++) {
    EXPECT_NEAR(RNWH.at({i}), i, 0.001);
  }
}

TEST_P(Operator, sliceVectors) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  F->setName("sliceVectors");

  auto *V = mod.createVariable(ElemKind::IndexTy, {3, 30}, "V",
                               Variable::VisibilityKind::Public);

  Node *S1 = F->createSlice("slice1", V, {0, 10}, {3, 13});
  Node *S2 = F->createSlice("slice2", V, {1, 10}, {2, 30});
  Node *S3 = F->createSlice("slice3", V, {2, 10}, {3, 12});

  auto *result1 = F->createSave("ret1", S1);
  auto *result2 = F->createSave("ret2", S2);
  auto *result3 = F->createSave("ret3", S3);

  Tensor I(ElemKind::IndexTy, {3, 30});

  for (size_t j = 0; j < 30; j++) {
    I.getHandle<size_t>().at({0, j}) = j;
    I.getHandle<size_t>().at({1, j}) = j + 30;
    I.getHandle<size_t>().at({2, j}) = j + 60;
  }

  EE.compile(CompilationMode::Infer, F);

  // Testing the output slices.
  EE.run({V}, {&I});
  auto RNWH1 = result1->getVariable()->getPayload().getHandle<size_t>();
  (void)RNWH1;
  auto RNWH2 = result2->getVariable()->getPayload().getHandle<size_t>();
  (void)RNWH2;
  auto RNWH3 = result3->getVariable()->getPayload().getHandle<size_t>();
  (void)RNWH3;

  EXPECT_EQ(3, RNWH1.dims()[0]);
  EXPECT_EQ(3, RNWH1.dims()[1]);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 10; j < 13; j++) {
      EXPECT_NEAR(RNWH1.at({i, j - 10}), j + i * 30, 0.001);
    }
  }
  EXPECT_EQ(1, RNWH2.dims()[0]);
  EXPECT_EQ(20, RNWH2.dims()[1]);
  for (size_t j = 10; j < 30; j++) {
    EXPECT_NEAR(RNWH2.at({0, j - 10}), j + 30, 0.001);
  }
  EXPECT_EQ(1, RNWH3.dims()[0]);
  EXPECT_EQ(2, RNWH3.dims()[1]);
  for (size_t j = 10; j < 12; j++) {
    EXPECT_NEAR(RNWH3.at({0, j - 10}), j + 60, 0.001);
  }
}

TEST_P(Operator, simpleCmpSelectPredication) {
  // A simple test that checks predication of some values using the
  // compare-select pair of instructions. Keep doubling some values
  // until some condition is met.

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *inputs = mod.createVariable(ElemKind::FloatTy, {10}, "inputs");
  auto *counters = mod.createVariable(ElemKind::FloatTy, {10}, "counters");

  counters->getPayload().getHandle() = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  inputs->getPayload().getHandle().clear(1);

  Node *cnt = counters;
  Node *data = inputs;
  Node *const1 = F->createSplat("const1", counters->getType(), 1.0);
  Node *const0 = F->createSplat("const0", counters->getType(), 0.0);

  for (int i = 0; i < 10; i++) {
    cnt = F->createSub("sub1", cnt, const1);
    Node *pred = F->createCmpLTE("cmp", const0, cnt);

    Node *const2 = F->createSplat("const2", data->getType(), 2.0);
    Node *newData = F->createMul("mul2x", data, const2);

    data = F->createSelect("select", pred, newData, data);
  }

  auto *SN = F->createSave("ret", data);

  EE.compile(CompilationMode::Infer, F);
  EE.run({}, {});

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

INSTANTIATE_TEST_CASE_P(Interpreter, Operator,
                        ::testing::Values(BackendKind::Interpreter));

#ifdef GLOW_WITH_CPU
INSTANTIATE_TEST_CASE_P(JIT, Operator, ::testing::Values(BackendKind::JIT));
#endif // GLOW_WITH_CPU
