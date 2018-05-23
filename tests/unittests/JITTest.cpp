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

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Random.h"

#include "gtest/gtest.h"

using namespace glow;
using llvm::cast;

class BackendCorrectnessTest : public ::testing::TestWithParam<BackendKind> {
protected:
  BackendKind backendKind_{GetParam()};
};

class CPUOnly : public BackendCorrectnessTest {};

TEST_P(BackendCorrectnessTest, batchedAddTest) {
  Tensor batch(ElemKind::FloatTy, {8, 3, 3, 6});
  Tensor slice(ElemKind::FloatTy, {3, 3, 6});
  batch.getHandle().initXavier(1);
  slice.getHandle().initXavier(1);
  Tensor out1(ElemKind::FloatTy, {8, 3, 3, 6});
  Tensor out2(ElemKind::FloatTy, {8, 3, 3, 6});

  inferBatchedAddNet(&batch, &slice, &out1, backendKind_);
  inferBatchedAddNet(&batch, &slice, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, DISABLED_quantizedBatchedAddTest) {
  std::array<size_t, 4> S{{10, 1, 1, 2}};
  llvm::ArrayRef<size_t> shape(S);
  Tensor batch(ElemKind::Int8QTy, shape, 0.875, -1);
  Tensor slice(ElemKind::Int8QTy, {1, 1, 2}, 1.4, 5);
  batch.getHandle<int8_t>().randomize(-129, 128);
  slice.getHandle<int8_t>().randomize(-129, 128);
  Tensor out1(ElemKind::Int8QTy, shape, 0.375, -10);
  Tensor out2(ElemKind::Int8QTy, shape, 0.375, -10);

  inferBatchedAddNet(&batch, &slice, &out1, backendKind_);
  inferBatchedAddNet(&batch, &slice, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, batchedReduceAddTest) {
  Tensor inputs(ElemKind::FloatTy, {7, 5, 9, 2});
  inputs.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferBatchedReduceAddNet(&inputs, &out1, backendKind_);
  inferBatchedReduceAddNet(&inputs, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, convTest) {
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

  inferConvNet(&inputs, &kernel, &bias, &out1, backendKind_);
  inferConvNet(&inputs, &kernel, &bias, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(CPUOnly, extract3Dtest) {
  Tensor inputs(ElemKind::FloatTy, {5, 100, 100});
  inputs.getHandle().initXavier(1);
  std::array<size_t, 4> S{{1, 95, 100}};
  llvm::ArrayRef<size_t> shape(S);
  Tensor out1(ElemKind::FloatTy, shape);
  Tensor out2(ElemKind::FloatTy, shape);

  inferExtract3D(&inputs, &out1, BackendKind::CPU);
  inferExtract3D(&inputs, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(CPUOnly, quantizedConvTest) {
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

  inferConvNet(&inputs, &kernel, &bias, &out1, backendKind_);
  inferConvNet(&inputs, &kernel, &bias, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2, 1.0));
}

TEST_P(BackendCorrectnessTest, convGradTest) {
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
               shape2, &out1, backendKind_);
  trainConvNet(&inputs, &kernel1, &bias1, &kernel2, &bias2, &selected, shape1,
               shape2, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, gatherTest) {
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

  inferGatherNet(&data, &indices, &out1, backendKind_);
  inferGatherNet(&data, &indices, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(CPUOnly, localResponseNormalizationTest) {
  Tensor inputs(ElemKind::FloatTy, {8, 15, 13, 30});
  inputs.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferLocalResponseNormalizationNet(&inputs, &out1, backendKind_);
  inferLocalResponseNormalizationNet(&inputs, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(CPUOnly, localResponseNormalizationGradTest) {
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
                                     shape1, shape2, &out1, backendKind_);
  trainLocalResponseNormalizationNet(&inputs, &weights, &bias, &selected,
                                     shape1, shape2, &out2,
                                     BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, matMulTest) {
  Tensor lhs(ElemKind::FloatTy, {10, 9});
  Tensor rhs(ElemKind::FloatTy, {9, 8});
  lhs.getHandle().randomize(-7.2, 8.3);
  rhs.getHandle().randomize(-6.3, 10.1);
  std::array<size_t, 2> S{{10, 8}};
  llvm::ArrayRef<size_t> shape(S);
  Tensor out1(ElemKind::FloatTy, shape);
  Tensor out2(ElemKind::FloatTy, shape);

  inferMatMulNet(&lhs, &rhs, &out1, backendKind_);
  inferMatMulNet(&lhs, &rhs, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2, 0.001));
}

TEST_P(CPUOnly, quantizedMatMulTest) {
  Tensor lhs(ElemKind::Int8QTy, {10, 9}, 2.7, 31);
  Tensor rhs(ElemKind::Int8QTy, {9, 8}, 3.2, -12);
  lhs.getHandle<int8_t>().randomize(-129, 128);
  rhs.getHandle<int8_t>().randomize(-129, 128);
  std::array<size_t, 2> S{{10, 8}};
  llvm::ArrayRef<size_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 8.1, 7);
  Tensor out2(ElemKind::Int8QTy, shape, 8.1, 7);

  inferMatMulNet(&lhs, &rhs, &out1, backendKind_);
  inferMatMulNet(&lhs, &rhs, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, maxTest) {
  std::array<size_t, 1> S{{1941}};
  llvm::ArrayRef<size_t> shape(S);
  Tensor inputs1(ElemKind::FloatTy, shape);
  Tensor inputs2(ElemKind::FloatTy, shape);
  inputs1.getHandle().initXavier(1);
  inputs2.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferMaxNet(&inputs1, &inputs2, &out1, backendKind_);
  inferMaxNet(&inputs1, &inputs2, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, minTest) {
  std::array<size_t, 1> S{{1123}};
  llvm::ArrayRef<size_t> shape(S);
  Tensor inputs1(ElemKind::FloatTy, shape);
  Tensor inputs2(ElemKind::FloatTy, shape);
  inputs1.getHandle().initXavier(1);
  inputs2.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferMinNet(&inputs1, &inputs2, &out1, backendKind_);
  inferMinNet(&inputs1, &inputs2, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, poolAvgTest) {
  Tensor inputs(ElemKind::FloatTy, {14, 12, 19, 7});
  inputs.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferPoolAvgNet(&inputs, &out1, backendKind_);
  inferPoolAvgNet(&inputs, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(CPUOnly, poolAvgGradTest) {
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
                  backendKind_);
  trainPoolAvgNet(&inputs, &weights, &bias, &selected, shape1, shape2, &out2,
                  BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, poolMaxTest) {
  Tensor inputs(ElemKind::FloatTy, {5, 53, 71, 14});
  inputs.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferPoolMaxNet(&inputs, &out1, backendKind_);
  inferPoolMaxNet(&inputs, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, poolMaxGradTest) {
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
                  backendKind_);
  trainPoolMaxNet(&inputs, &weights, &bias, &selected, shape1, shape2, &out2,
                  BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(CPUOnly, intLookupTable) {
  constexpr size_t inputSize = 100;
  Tensor inputs(ElemKind::Int8QTy, {inputSize}, 0.8, 4);
  inputs.getHandle<int8_t>().randomize(-128, 127);
  Tensor out1, out2;

  // Mapping i -> i.
  std::vector<int8_t> initValues(256);
  for (size_t i = 0; i < 256; ++i) {
    initValues[i] = i - 128;
  }

  inferIntLookupTableNet(&inputs, &out1, initValues, backendKind_);
  inferIntLookupTableNet(&inputs, &out2, initValues, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, quantizeTest) {
  std::array<size_t, 4> S{{26, 51, 29, 32}};
  llvm::ArrayRef<size_t> shape(S);
  Tensor inputs(ElemKind::FloatTy, shape);
  inputs.getHandle().randomize(-10000.0, 5000.0);
  float scale{4500.0 / 128};
  int32_t offset{-2500};
  Tensor out1(ElemKind::FloatTy, shape);
  Tensor out2(ElemKind::FloatTy, shape);

  inferQuantizeNet(&inputs, scale, offset, &out1, backendKind_);
  inferQuantizeNet(&inputs, scale, offset, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, reluTest) {
  Tensor inputs(ElemKind::FloatTy, {2, 16});
  inputs.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferReluNet(&inputs, &out1, backendKind_);
  inferReluNet(&inputs, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, reshapeTest) {
  Tensor inputs(ElemKind::FloatTy, {12, 6, 8, 12});
  inputs.getHandle().initXavier(1);
  std::array<size_t, 4> S{{18, 4, 24, 4}};
  llvm::ArrayRef<size_t> shape(S);
  Tensor out1;
  Tensor out2;

  inferReshapeNet(&inputs, shape, &out1, backendKind_);
  inferReshapeNet(&inputs, shape, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, reshapeIndexTest) {
  Tensor inputs(ElemKind::IndexTy, {12, 6, 8, 12});
  auto H = inputs.getHandle<size_t>();
  for (size_t i = 0; i < H.size(); i++) {
    H.raw(i) = i;
  }
  std::array<size_t, 4> S{{18, 4, 24, 4}};
  llvm::ArrayRef<size_t> shape(S);
  Tensor out1;
  Tensor out2;

  inferReshapeNet(&inputs, shape, &out1, backendKind_);
  inferReshapeNet(&inputs, shape, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, selectTest) {
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

  inferSelectNet(&cond, &inputs1, &inputs2, &out1, backendKind_);
  inferSelectNet(&cond, &inputs1, &inputs2, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, sigmoidTest) {
  Tensor inputs(ElemKind::FloatTy, {11, 4, 5, 2});
  inputs.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferSigmoidNet(&inputs, &out1, backendKind_);
  inferSigmoidNet(&inputs, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, smallConv) {
  Tensor input(ElemKind::FloatTy, {1, 3, 3, 32});
  input.getHandle().clear(0.2);
  Tensor out1;
  Tensor out2;

  inferSmallConv(&input, &out1, backendKind_);
  inferSmallConv(&input, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, softmaxTest) {
  Tensor inputs(ElemKind::FloatTy, {14, 19});
  Tensor selected(ElemKind::IndexTy, {14, 1});
  inputs.getHandle().initXavier(1);
  auto selectedH = selected.getHandle<size_t>();
  for (size_t i = 0; i < 14; i++) {
    selectedH.raw(i) = nextRandInt(0, 18);
  }
  Tensor out1;
  Tensor out2;

  inferSoftMaxNet(&inputs, &selected, &out1, backendKind_);
  inferSoftMaxNet(&inputs, &selected, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, softmaxGradTest) {
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

  trainSoftMaxNet(&inputs, &weights, &bias, &selected, &out1, backendKind_);
  trainSoftMaxNet(&inputs, &weights, &bias, &selected, &out2,
                  BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, tanhTest) {
  Tensor inputs(ElemKind::FloatTy, {14151});
  inputs.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferTanhNet(&inputs, &out1, backendKind_);
  inferTanhNet(&inputs, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, transposeTest) {
  Tensor inputs(ElemKind::FloatTy, {32, 32});
  inputs.getHandle().randomize(-1.0, 1.0);
  Tensor out1;
  Tensor out2;

  inferTransposeNet(&inputs, &out1, backendKind_);
  inferTransposeNet(&inputs, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, convOps) {
  // Construct networks with a different convolution depth.
  for (auto depth : {4, 12, 128}) {
    Tensor inputs(ElemKind::FloatTy, {2, 3, 16, 16});
    inputs.getHandle().initXavier(1);
    Tensor out1;
    Tensor out2;

    inferBasicConvNet(&inputs, &out1, backendKind_, depth);
    inferBasicConvNet(&inputs, &out2, BackendKind::Interpreter, depth);

    EXPECT_TRUE(out1.isEqual(out2));
  }
}

TEST_P(BackendCorrectnessTest, basicFCNet) {
  Tensor inputs(ElemKind::FloatTy, {2, 3, 16, 16});
  inputs.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferBasicFCNet(&inputs, &out1, backendKind_);
  inferBasicFCNet(&inputs, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(CPUOnly, complexNet1) {
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

  inferComplexNet1(&inputs1, &inputs2, &inputs3, &inputs4, &out1, backendKind_);
  inferComplexNet1(&inputs1, &inputs2, &inputs3, &inputs4, &out2,
                   BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, tinyResnet) {
  Tensor input(ElemKind::FloatTy, {1, 7, 7, 64});
  input.getHandle().randomize(0, 1.0);

  std::vector<Tensor> weights;
  using Dims = llvm::ArrayRef<size_t>;
  weights.emplace_back(ElemKind::FloatTy, Dims{256, 1, 1, 64});
  weights.emplace_back(ElemKind::FloatTy, Dims{256});
  weights.emplace_back(ElemKind::FloatTy, Dims{64, 1, 1, 64});
  weights.emplace_back(ElemKind::FloatTy, Dims{64});
  weights.emplace_back(ElemKind::FloatTy, Dims{64, 3, 3, 64});
  weights.emplace_back(ElemKind::FloatTy, Dims{64});
  weights.emplace_back(ElemKind::FloatTy, Dims{256, 1, 1, 64});
  weights.emplace_back(ElemKind::FloatTy, Dims{256});
  for (auto &T : weights) {
    T.getHandle().initXavier(1.0);
  }

  Tensor out1;
  Tensor out2;
  inferTinyResnet(&input, &out1, weights, BackendKind::Interpreter);
  inferTinyResnet(&input, &out2, weights, backendKind_);

  EXPECT_TRUE(out1.isEqual(out2, 0.001));
}

INSTANTIATE_TEST_CASE_P(CPU, BackendCorrectnessTest,
                        ::testing::Values(BackendKind::CPU));
INSTANTIATE_TEST_CASE_P(CPU, CPUOnly, ::testing::Values(BackendKind::CPU));

#ifdef GLOW_WITH_OPENCL
INSTANTIATE_TEST_CASE_P(OpenCL, BackendCorrectnessTest,
                        ::testing::Values(BackendKind::OpenCL));
#endif // GLOW_WITH_OPENCL
