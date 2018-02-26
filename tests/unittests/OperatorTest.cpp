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
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *lhs = mod.createVariable(ElemKind::FloatTy, {1, 3, 2}, "lhs");
  auto *rhs = mod.createVariable(ElemKind::FloatTy, {1, 2, 1}, "rhs");
  auto *result = mod.createVariable(ElemKind::FloatTy, {1, 3, 1}, "result");
  lhs->getPayload().getHandle() = {1, 2, 3, 4, 5, 6};
  rhs->getPayload().getHandle() = {7, 10};

  auto R = F->createBatchedMatMul("MM", lhs, rhs);

  F->createSave("save", R, result);

  EE.compile(CompilationMode::Infer, F);

  EE.run({}, {});

  auto H = result->getPayload().getHandle();
  EXPECT_NEAR(H.at({0, 0, 0}), 27, 0.001);
  EXPECT_NEAR(H.at({0, 1, 0}), 61, 0.001);
  EXPECT_NEAR(H.at({0, 2, 0}), 95, 0.001);
}

TEST(Operator, batchedReduceAdd) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *batch = mod.createVariable(ElemKind::FloatTy, {2, 4}, "batch");
  auto *result = mod.createVariable(ElemKind::FloatTy, {4}, "result");
  batch->getPayload().getHandle() = {10, 20, 30, 40, 1, 2, 3, 4};

  auto R =
      F->createBatchedReduce("reduce.add", BatchedReduceNode::Mode::Add, batch);

  F->createSave("save", R, result);

  EE.compile(CompilationMode::Infer, F);

  EE.run({}, {});

  auto H = result->getPayload().getHandle();
  EXPECT_NEAR(H.at({0}), 11, 0.001);
  EXPECT_NEAR(H.at({1}), 22, 0.001);
  EXPECT_NEAR(H.at({2}), 33, 0.001);
  EXPECT_NEAR(H.at({3}), 44, 0.001);
}

TEST(Operator, batchedBatchedAdd) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *batch = mod.createVariable(ElemKind::FloatTy, {2, 3, 3}, "batch");
  auto *added = mod.createVariable(ElemKind::FloatTy, {3, 3}, "added");
  auto *result = mod.createVariable(ElemKind::FloatTy, {2, 3, 3}, "result");

  batch->getPayload().getHandle() = {9, 8, 7, 6, 5,  4,  3,  4,  5,
                                     6, 7, 8, 9, 10, 11, 12, 13, 14};
  added->getPayload().getHandle().clear(1.0);

  auto R = F->createBatchedArithmetic(
      "batch.add", BatchedArithmeticNode::Mode::Add, batch, added);
  F->createSave("save", R, result);

  EE.compile(CompilationMode::Infer, F);

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

TEST(Operator, TopK) {
  ExecutionEngine EE;
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

TEST(Operator, IntMatMul) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  // The scaling factor 1.4x was carefully selected to make sure we don't
  // overflow or underflow the calculation.
  TypeRef resTy = mod.uniqueType(ElemKind::Int8QTy, {1, 3, 3}, 0.60, 4);
  TypeRef lhsTy = mod.uniqueType(ElemKind::Int8QTy, {1, 3, 3}, 0.075, -2);
  TypeRef rhsTy = mod.uniqueType(ElemKind::Int8QTy, {1, 3, 3}, 0.075, 2);

  auto *res = mod.createVariable(ElemKind::FloatTy, {1, 3, 3}, "res");
  auto *lhs = mod.createVariable(ElemKind::FloatTy, {1, 3, 3}, "lhs");
  auto *rhs = mod.createVariable(ElemKind::FloatTy, {1, 3, 3}, "rhs");

  lhs->getPayload().getHandle() = {
      1.0, 2.0, 3.0, 4.0, 5.0, -5.0, -4.0, -3.0, 9.0,
  };

  rhs->getPayload().getHandle() = {
      0.1, -0.2, 0.3, 9.0, -8.0, 7.0, 6.0, 5.0, 9.0,
  };

  auto *lhsq = F->createQuantize("lhs.q", lhs, lhsTy);
  auto *rhsq = F->createQuantize("rhs.q", rhs, rhsTy);

  auto *matmulq = F->createBatchedMatMul("matmul.q", resTy, lhsq, rhsq);

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
  EXPECT_NEAR(H.at({0, 0, 0}), 36.1, 1.0);
  EXPECT_NEAR(H.at({0, 0, 1}), -1.2, 1.0);
  EXPECT_NEAR(H.at({0, 0, 2}), 41.3, 1.0);
  EXPECT_NEAR(H.at({0, 1, 0}), 15.4, 1.0);
  EXPECT_NEAR(H.at({0, 1, 1}), -65.8, 1.0);
  EXPECT_NEAR(H.at({0, 1, 2}), -8.8, 1.0);
  EXPECT_NEAR(H.at({0, 2, 0}), 26.6, 1.0);
  EXPECT_NEAR(H.at({0, 2, 1}), 69.8, 1.0);
  EXPECT_NEAR(H.at({0, 2, 2}), 58.8, 1.0);
}

TEST(Operator, IntBatchedArith) {
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

  auto *matmulq = F->createBatchedArithmetic(
      "add", resTy, BatchedArithmeticNode::Mode::Add, lhsq, rhsq);

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

TEST(Operator, IntConvolution) {
  ExecutionEngine EE;
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
  TypeRef filterTy = mod.uniqueType(ElemKind::Int8QTy, filter.dims(), 0.01, 0.);
  TypeRef biasTy = mod.uniqueType(ElemKind::Int8QTy, bias.dims(), 0.04, 0.0);

  auto *inputq = F->createQuantize("input.q", input, inputTy);
  auto *filterq = F->createQuantize("filter.q", filter, filterTy);
  auto *biasq = F->createQuantize("bias.q", bias, biasTy);

  auto *convq =
      F->createConv("convq", inputq, filterq, biasq, resTy, conv->getDepth(),
                    conv->getKernel(), conv->getStride(), conv->getPad());
  auto *dequantRes = F->createDequantize("dequant", convq);

  // Subtract the results of the convolution from the quantized convolution.
  auto *sub = F->createArithmetic("compare", dequantRes, conv,
                                  ArithmeticNode::Mode::Sub);

  F->createSave("save", sub, res);
  EE.compile(CompilationMode::Infer, F);

  EE.run({}, {});
  auto H = res->getPayload().getHandle();

  // Check that the difference in the results is less than 0.1.
  for (int i = 0, e = H.size(); i < e; i++) {
    EXPECT_NEAR(H.raw(i), 0, 0.1);
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

  auto filter = fc->getWeights();
  auto bias = fc->getBias();

  input->getPayload().getHandle().randomize(-1.2, 1.2);
  llvm::cast<Variable>(bias)->getPayload().getHandle().randomize(0, 0.00001);
  llvm::cast<Variable>(filter)->getPayload().getHandle().randomize(-1.2, 1.2);

  TypeRef resTy = mod.uniqueType(ElemKind::Int8QTy, res->dims(), 0.15, 4);
  TypeRef inputTy = mod.uniqueType(ElemKind::Int8QTy, input->dims(), 0.01, 0);
  TypeRef filterTy = mod.uniqueType(ElemKind::Int8QTy, filter.dims(), 0.01, 2);
  TypeRef biasTy = mod.uniqueType(ElemKind::Int8QTy, bias.dims(), 0.02, 1);

  auto *inputq = F->createQuantize("input.q", input, inputTy);
  auto *filterq = F->createQuantize("filter.q", filter, filterTy);
  auto *biasq = F->createQuantize("bias.q", bias, biasTy);

  auto *fcq = F->createFullyConnected("fcq", inputq, filterq, biasq, resTy);
  auto *dequantRes = F->createDequantize("dequant", fcq);

  // Subtract the results of the convolution from the quantized fc.
  auto *sub =
      F->createArithmetic("compare", dequantRes, fc, ArithmeticNode::Mode::Sub);

  F->createSave("save", sub, res);
  EE.compile(CompilationMode::Infer, F);

  EE.run({}, {});

  auto H = res->getPayload().getHandle();
  // Check that the difference in the results is less than 0.1.
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

TEST(Operator, RescaleNode) {
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

TEST(Operator, QuantizedMaxNode) {
  const int len = 100;

  // In this test we check the correctness of the quantized MAX and ADD
  // nodes as well as how they interact with the rescaling node.

  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *A = mod.createVariable(ElemKind::FloatTy, {len}, "A",
                               Variable::VisibilityKind::Public);
  auto *B = mod.createVariable(ElemKind::FloatTy, {len}, "B",
                               Variable::VisibilityKind::Public);
  auto *O1 = mod.createVariable(ElemKind::FloatTy, {len}, "Out1",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);
  auto *O2 = mod.createVariable(ElemKind::FloatTy, {len}, "Out2",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);

  auto AH = A->getPayload().getHandle();
  auto BH = B->getPayload().getHandle();
  auto O1H = O1->getPayload().getHandle();
  auto O2H = O2->getPayload().getHandle();

  AH.randomize(-100, 100);
  BH.randomize(-10, 10);

  auto TA = mod.uniqueType(ElemKind::Int8QTy, {len}, 1.0, 0);
  auto TB = mod.uniqueType(ElemKind::Int8QTy, {len}, 0.1, 0);
  auto TO1 = mod.uniqueType(ElemKind::Int8QTy, {len}, 1.0, 0);
  auto TO2 = mod.uniqueType(ElemKind::Int8QTy, {len}, 0.9, 0);
  auto TO3 = mod.uniqueType(ElemKind::Int8QTy, {len}, 1.1, 0);
  auto TO4 = mod.uniqueType(ElemKind::Int8QTy, {len}, 1.2, 0);

  auto *QA = F->createQuantize("QA", A, TA);
  auto *QB = F->createQuantize("QB", B, TB);

  Node *max = F->createArithmetic("MX", TO1, QA, QB, ArithmeticNode::Mode::Max);
  Node *add = F->createArithmetic("MX", TO2, QA, QB, ArithmeticNode::Mode::Add);

  max = F->createRescaleQuantized("Rescale", max, TO3);
  add = F->createRescaleQuantized("Rescale", add, TO4);

  auto *addQ = F->createDequantize("addQ", add);
  auto *maxQ = F->createDequantize("maxW", max);

  // Test a sequence of rescale operations t
  F->createSave("save_add", addQ, O1);
  F->createSave("save_max", maxQ, O2);

  EE.compile(CompilationMode::Infer, F);
  EE.run({}, {});

  for (size_t i = 0; i < len; i++) {
    auto max = std::max(AH.at({i}), BH.at({i}));
    auto add = AH.at({i}) + BH.at({i});

    EXPECT_NEAR(add, O1H.at({i}), 1.0);
    EXPECT_NEAR(max, O2H.at({i}), 1.0);
  }
}

TEST(Operator, TestQuantizedRescaleSequence) {
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
  R = F->createArithmetic("addZero", G, R, ArithmeticNode::Mode::Add);
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
