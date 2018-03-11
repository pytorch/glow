// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "BackendTestUtils.h"

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Random.h"

#include "gtest/gtest.h"

#include <cassert>
#include <string>

using namespace glow;
using llvm::cast;

TEST(JITCorrectnessTest, batchedAddTest) {
  Tensor inputs1(ElemKind::FloatTy, {8, 3, 3, 6});
  Tensor inputs2(ElemKind::FloatTy, {3, 3, 6});
  inputs1.getHandle().initXavier(1);
  inputs2.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferBatchedAddNet(&inputs1, &inputs2, &out1, BackendKind::JIT);
  inferBatchedAddNet(&inputs1, &inputs2, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, batchedReduceAddTest) {
  Tensor inputs(ElemKind::FloatTy, {7, 5, 9, 2});
  inputs.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferBatchedReduceAddNet(&inputs, &out1, BackendKind::JIT);
  inferBatchedReduceAddNet(&inputs, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, convTest) {
  Tensor inputs(ElemKind::FloatTy, {20, 41, 32, 6});
  Tensor kernel(ElemKind::FloatTy, {10, 5, 5, 6});
  Tensor bias(ElemKind::FloatTy, {10});
  inputs.getHandle().initXavier(1);
  kernel.getHandle().randomize(-3.0, 3.0);
  bias.getHandle().randomize(-0.5, 0.5);
  std::array<size_t, 4> S{{20, 15, 12, 10}};
  llvm::ArrayRef<size_t> shape(S);
  Tensor out1(ElemKind::FloatTy, shape);
  Tensor out2(ElemKind::FloatTy, shape);

  inferConvNet(&inputs, &kernel, &bias, &out1, BackendKind::JIT);
  inferConvNet(&inputs, &kernel, &bias, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, quantizedConvTest) {
  Tensor inputs(ElemKind::Int8QTy, {20, 41, 32, 6}, 0.025, -7);
  Tensor kernel(ElemKind::Int8QTy, {10, 5, 5, 6}, 0.003, 3);
  Tensor bias(ElemKind::Int8QTy, {10}, 0.5, -4);
  inputs.getHandle<int8_t>().randomize(-129, 128);
  kernel.getHandle<int8_t>().randomize(-129, 128);
  bias.getHandle<int8_t>().randomize(-11, 8);
  std::array<size_t, 4> S{{20, 15, 12, 10}};
  llvm::ArrayRef<size_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 0.05, -17);
  Tensor out2(ElemKind::Int8QTy, shape, 0.05, -17);

  inferConvNet(&inputs, &kernel, &bias, &out1, BackendKind::JIT);
  inferConvNet(&inputs, &kernel, &bias, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle<int8_t>();
  auto H2 = out2.getHandle<int8_t>();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, convGradTest) {
  Tensor inputs(ElemKind::FloatTy, {9, 8, 9, 4});
  Tensor kernel1(ElemKind::FloatTy, {3, 3, 3, 4});
  Tensor bias1(ElemKind::FloatTy, {3});
  Tensor kernel2(ElemKind::FloatTy, {2, 2, 2, 1});
  Tensor bias2(ElemKind::FloatTy, {2});
  Tensor selected(ElemKind::IndexTy, {9, 1});
  inputs.getHandle().initXavier(1);
  kernel1.getHandle().randomize(-1.0, 1.4);
  bias1.getHandle().randomize(-0.2, 0.5);
  kernel2.getHandle().randomize(-1.8, 2.3);
  bias2.getHandle().randomize(-0.5, 1.0);
  auto selectedH = selected.getHandle<size_t>();
  for (size_t i = 0; i < 9; i++) {
    selectedH.raw(i) = nextRandInt(0, 29);
  }
  std::array<size_t, 4> S1{{9, 6, 10, 1}};
  llvm::ArrayRef<size_t> shape1(S1);
  std::array<size_t, 2> S2{{9, 30}};
  llvm::ArrayRef<size_t> shape2(S2);
  Tensor out1(ElemKind::FloatTy, shape2);
  Tensor out2(ElemKind::FloatTy, shape2);

  trainConvNet(&inputs, &kernel1, &bias1, &kernel2, &bias2, &selected, shape1,
               shape2, &out1, BackendKind::JIT);
  trainConvNet(&inputs, &kernel1, &bias1, &kernel2, &bias2, &selected, shape1,
               shape2, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, gatherTest) {
  constexpr size_t nSlices = 16;
  constexpr size_t nGathered = 8;

  Tensor data(ElemKind::FloatTy, {nSlices, 16, 3, 2});
  data.getHandle().initXavier(1);

  Tensor indices(ElemKind::IndexTy, {nGathered});
  auto indicesH = indices.getHandle<size_t>();
  for (size_t i = 0; i < nGathered; i++) {
    indicesH.raw(i) = nextRandInt(0, nSlices - 1);
  }

  Tensor out1(ElemKind::FloatTy, {nGathered, 16, 3, 2});
  Tensor out2(ElemKind::FloatTy, {nGathered, 16, 3, 2});

  inferGatherNet(&data, &indices, &out1, BackendKind::JIT);
  inferGatherNet(&data, &indices, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, localResponseNormalizationTest) {
  Tensor inputs(ElemKind::FloatTy, {8, 15, 13, 30});
  inputs.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferLocalResponseNormalizationNet(&inputs, &out1, BackendKind::JIT);
  inferLocalResponseNormalizationNet(&inputs, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, localResponseNormalizationGradTest) {
  Tensor inputs(ElemKind::FloatTy, {5, 4, 7, 3});
  Tensor weights(ElemKind::FloatTy, {84, 180});
  Tensor bias(ElemKind::FloatTy, {180});
  Tensor selected(ElemKind::IndexTy, {5, 1});
  inputs.getHandle().initXavier(1);
  weights.getHandle().randomize(-2.0, 3.0);
  bias.getHandle().randomize(-1.0, 1.3);
  auto selectedH = selected.getHandle<size_t>();
  for (size_t i = 0; i < 5; i++) {
    selectedH.raw(i) = nextRandInt(0, 179);
  }
  std::array<size_t, 4> S1{{5, 2, 2, 45}};
  llvm::ArrayRef<size_t> shape1(S1);
  std::array<size_t, 2> S2{{5, 180}};
  llvm::ArrayRef<size_t> shape2(S2);
  Tensor out1(ElemKind::FloatTy, shape2);
  Tensor out2(ElemKind::FloatTy, shape1);

  trainLocalResponseNormalizationNet(&inputs, &weights, &bias, &selected,
                                     shape1, shape2, &out1, BackendKind::JIT);
  trainLocalResponseNormalizationNet(&inputs, &weights, &bias, &selected,
                                     shape1, shape2, &out2,
                                     BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, maxTest) {
  std::array<size_t, 1> S{{1941}};
  llvm::ArrayRef<size_t> shape(S);
  Tensor inputs1(ElemKind::FloatTy, shape);
  Tensor inputs2(ElemKind::FloatTy, shape);
  inputs1.getHandle().initXavier(1);
  inputs2.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferMaxNet(&inputs1, &inputs2, &out1, BackendKind::JIT);
  inferMaxNet(&inputs1, &inputs2, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, minTest) {
  std::array<size_t, 1> S{{1123}};
  llvm::ArrayRef<size_t> shape(S);
  Tensor inputs1(ElemKind::FloatTy, shape);
  Tensor inputs2(ElemKind::FloatTy, shape);
  inputs1.getHandle().initXavier(1);
  inputs2.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferMinNet(&inputs1, &inputs2, &out1, BackendKind::JIT);
  inferMinNet(&inputs1, &inputs2, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, poolAvgTest) {
  Tensor inputs(ElemKind::FloatTy, {14, 12, 19, 7});
  inputs.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferPoolAvgNet(&inputs, &out1, BackendKind::JIT);
  inferPoolAvgNet(&inputs, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, poolAvgGradTest) {
  Tensor inputs(ElemKind::FloatTy, {5, 7, 6, 3});
  Tensor weights(ElemKind::FloatTy, {126, 72});
  Tensor bias(ElemKind::FloatTy, {72});
  Tensor selected(ElemKind::IndexTy, {5, 1});
  inputs.getHandle().initXavier(1);
  weights.getHandle().randomize(-0.3, 0.6);
  bias.getHandle().randomize(-0.2, 0.1);
  auto selectedH = selected.getHandle<size_t>();
  for (size_t i = 0; i < 5; i++) {
    selectedH.raw(i) = nextRandInt(0, 17);
  }
  std::array<size_t, 4> S1{{5, 6, 4, 3}};
  llvm::ArrayRef<size_t> shape1(S1);
  std::array<size_t, 2> S2{{5, 18}};
  llvm::ArrayRef<size_t> shape2(S2);
  Tensor out1(ElemKind::FloatTy, shape2);
  Tensor out2(ElemKind::FloatTy, shape2);

  trainPoolAvgNet(&inputs, &weights, &bias, &selected, shape1, shape2, &out1,
                  BackendKind::JIT);
  trainPoolAvgNet(&inputs, &weights, &bias, &selected, shape1, shape2, &out2,
                  BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, poolMaxTest) {
  Tensor inputs(ElemKind::FloatTy, {5, 53, 71, 14});
  inputs.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferPoolMaxNet(&inputs, &out1, BackendKind::JIT);
  inferPoolMaxNet(&inputs, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, poolMaxGradTest) {
  Tensor inputs(ElemKind::FloatTy, {4, 8, 7, 2});
  Tensor weights(ElemKind::FloatTy, {112, 84});
  Tensor bias(ElemKind::FloatTy, {84});
  Tensor selected(ElemKind::IndexTy, {4, 1});
  inputs.getHandle().initXavier(1);
  weights.getHandle().randomize(-0.1, 0.7);
  bias.getHandle().randomize(-0.3, 0.1);
  auto selectedH = selected.getHandle<size_t>();
  for (size_t i = 0; i < 4; i++) {
    selectedH.raw(i) = nextRandInt(0, 31);
  }
  std::array<size_t, 4> S1{{4, 6, 7, 2}};
  llvm::ArrayRef<size_t> shape1(S1);
  std::array<size_t, 2> S2{{4, 32}};
  llvm::ArrayRef<size_t> shape2(S2);
  Tensor out1(ElemKind::FloatTy, shape2);
  Tensor out2(ElemKind::FloatTy, shape2);

  trainPoolMaxNet(&inputs, &weights, &bias, &selected, shape1, shape2, &out1,
                  BackendKind::JIT);
  trainPoolMaxNet(&inputs, &weights, &bias, &selected, shape1, shape2, &out2,
                  BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, quantizeTest) {
  std::array<size_t, 4> S{{26, 51, 29, 32}};
  llvm::ArrayRef<size_t> shape(S);
  Tensor inputs(ElemKind::FloatTy, shape);
  inputs.getHandle().randomize(-10000.0, 5000.0);
  float scale{4500.0 / 128};
  int32_t offset{-2500};
  Tensor out1(ElemKind::FloatTy, shape);
  Tensor out2(ElemKind::FloatTy, shape);

  inferQuantizeNet(&inputs, scale, offset, &out1, BackendKind::JIT);
  inferQuantizeNet(&inputs, scale, offset, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, reluTest) {
  Tensor inputs(ElemKind::FloatTy, {2, 16});
  inputs.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferReluNet(&inputs, &out1, BackendKind::JIT);
  inferReluNet(&inputs, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, reshapeTest) {
  Tensor inputs(ElemKind::FloatTy, {12, 6, 8, 12});
  inputs.getHandle().initXavier(1);
  std::array<size_t, 4> S{{18, 4, 24, 4}};
  llvm::ArrayRef<size_t> shape(S);
  Tensor out1;
  Tensor out2;

  inferReshapeNet(&inputs, shape, &out1, BackendKind::JIT);
  inferReshapeNet(&inputs, shape, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, reshapeIndexTest) {
  Tensor inputs(ElemKind::IndexTy, {12, 6, 8, 12});
  auto H = inputs.getHandle<size_t>();
  for (size_t i = 0; i < H.size(); i++) {
    H.raw(i) = i;
  }
  std::array<size_t, 4> S{{18, 4, 24, 4}};
  llvm::ArrayRef<size_t> shape(S);
  Tensor out1;
  Tensor out2;

  inferReshapeNet(&inputs, shape, &out1, BackendKind::JIT);
  inferReshapeNet(&inputs, shape, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle<size_t>();
  auto H2 = out2.getHandle<size_t>();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, selectTest) {
  std::array<size_t, 4> S{{5, 3, 9, 2}};
  llvm::ArrayRef<size_t> shape(S);
  Tensor cond(ElemKind::FloatTy, shape);
  Tensor inputs1(ElemKind::FloatTy, shape);
  Tensor inputs2(ElemKind::FloatTy, shape);
  auto condH = cond.getHandle();
  for (size_t i = 0; i < 270; i++) {
    condH.raw(i) = nextRandInt(0, 1);
  }
  inputs1.getHandle().initXavier(1);
  inputs2.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferSelectNet(&cond, &inputs1, &inputs2, &out1, BackendKind::JIT);
  inferSelectNet(&cond, &inputs1, &inputs2, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, sigmoidTest) {
  Tensor inputs(ElemKind::FloatTy, {11, 4, 5, 2});
  inputs.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferSigmoidNet(&inputs, &out1, BackendKind::JIT);
  inferSigmoidNet(&inputs, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, softmaxTest) {
  Tensor inputs(ElemKind::FloatTy, {14, 19});
  Tensor selected(ElemKind::IndexTy, {14, 1});
  inputs.getHandle().initXavier(1);
  auto selectedH = selected.getHandle<size_t>();
  for (size_t i = 0; i < 14; i++) {
    selectedH.raw(i) = nextRandInt(0, 18);
  }
  Tensor out1;
  Tensor out2;

  inferSoftMaxNet(&inputs, &selected, &out1, BackendKind::JIT);
  inferSoftMaxNet(&inputs, &selected, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, softmaxGradTest) {
  std::array<size_t, 2> S{{8, 23}};
  llvm::ArrayRef<size_t> shape(S);
  Tensor inputs(ElemKind::FloatTy, shape);
  Tensor weights(ElemKind::FloatTy, {23, 23});
  Tensor bias(ElemKind::FloatTy, {23});
  Tensor selected(ElemKind::IndexTy, {8, 1});
  inputs.getHandle().initXavier(1);
  weights.getHandle().randomize(0.0, 0.5);
  bias.getHandle().randomize(-0.2, 0.0);
  auto selectedH = selected.getHandle<size_t>();
  for (size_t i = 0; i < 8; i++) {
    selectedH.raw(i) = nextRandInt(0, 22);
  }
  Tensor out1(ElemKind::FloatTy, shape);
  Tensor out2(ElemKind::FloatTy, shape);

  trainSoftMaxNet(&inputs, &weights, &bias, &selected, &out1, BackendKind::JIT);
  trainSoftMaxNet(&inputs, &weights, &bias, &selected, &out2,
                  BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, tanhTest) {
  Tensor inputs(ElemKind::FloatTy, {14151});
  inputs.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferTanhNet(&inputs, &out1, BackendKind::JIT);
  inferTanhNet(&inputs, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, convOps) {
  Tensor inputs(ElemKind::FloatTy, {2, 3, 16, 16});
  inputs.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferBasicConvNet(&inputs, &out1, BackendKind::JIT);
  inferBasicConvNet(&inputs, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, basicFCNet) {
  Tensor inputs(ElemKind::FloatTy, {2, 3, 16, 16});
  inputs.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferBasicFCNet(&inputs, &out1, BackendKind::JIT);
  inferBasicFCNet(&inputs, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, complexNet1) {
  std::array<size_t, 4> S{{8, 7, 14, 11}};
  llvm::ArrayRef<size_t> shape(S);
  Tensor inputs1(ElemKind::FloatTy, shape);
  Tensor inputs2(ElemKind::FloatTy, {8, 4, 7, 9});
  Tensor inputs3(ElemKind::FloatTy, shape);
  Tensor inputs4(ElemKind::FloatTy, {8, 8, 7, 4});
  inputs1.getHandle().initXavier(1);
  inputs2.getHandle().initXavier(1);
  inputs3.getHandle().initXavier(1);
  inputs4.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferComplexNet1(&inputs1, &inputs2, &inputs3, &inputs4, &out1,
                   BackendKind::JIT);
  inferComplexNet1(&inputs1, &inputs2, &inputs3, &inputs4, &out2,
                   BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}
