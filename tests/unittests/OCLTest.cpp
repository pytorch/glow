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

#include "gtest/gtest.h"

using namespace glow;
using llvm::cast;

TEST(OpenCLCorrectnessTest, reluTest) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {2, 16});
  inputs.getHandle().initXavier(1, PRNG);
  Tensor out1;
  Tensor out2;

  inferReluNet(&inputs, &out1, BackendKind::OpenCL);
  inferReluNet(&inputs, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST(OpenCLCorrectnessTest, convOps) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {2, 3, 16, 16});
  inputs.getHandle().initXavier(1, PRNG);
  Tensor out1;
  Tensor out2;

  inferBasicConvNet(&inputs, &out1, BackendKind::OpenCL, 4);
  inferBasicConvNet(&inputs, &out2, BackendKind::Interpreter, 4);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST(OpenCLCorrectnessTest, basicFCNet) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {2, 3, 16, 16});
  inputs.getHandle().initXavier(1, PRNG);
  Tensor out1;
  Tensor out2;

  inferBasicFCNet(&inputs, &out1, BackendKind::OpenCL);
  inferBasicFCNet(&inputs, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST(OpenCLCorrectnessTest, inferMixedNet) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {2, 3, 16, 16});
  inputs.getHandle().initXavier(1, PRNG);
  Tensor out1;
  Tensor out2;

  inferMixedNet(&inputs, &out1, BackendKind::OpenCL);
  inferMixedNet(&inputs, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST(OpenCLCorrectnessTest, softmaxGradTest) {
  PseudoRNG PRNG;
  std::array<uint64_t, 2> S{{8, 23}};
  llvm::ArrayRef<uint64_t> shape(S);
  Tensor inputs(ElemKind::FloatTy, shape);
  Tensor weights(ElemKind::FloatTy, {23, 23});
  Tensor bias(ElemKind::FloatTy, {23});
  Tensor selected(ElemKind::IndexTy, {8, 1});
  inputs.getHandle().initXavier(1, PRNG);
  weights.getHandle().randomize(0.0, 0.5, PRNG);
  bias.getHandle().randomize(-0.2, 0.0, PRNG);
  auto selectedH = selected.getHandle<uint64_t>();
  for (size_t i = 0; i < 8; i++) {
    selectedH.raw(i) = PRNG.nextRandInt(0, 22);
  }
  Tensor out1(ElemKind::FloatTy, shape);
  Tensor out2(ElemKind::FloatTy, shape);

  trainSoftMaxNet(&inputs, &weights, &bias, &selected, &out1,
                  BackendKind::OpenCL);
  trainSoftMaxNet(&inputs, &weights, &bias, &selected, &out2,
                  BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST(OpenCLCorrectnessTest, convGradTest) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {9, 8, 9, 4});
  Tensor kernel1(ElemKind::FloatTy, {3, 3, 3, 4});
  Tensor bias1(ElemKind::FloatTy, {3});
  Tensor kernel2(ElemKind::FloatTy, {2, 2, 2, 1});
  Tensor bias2(ElemKind::FloatTy, {2});
  Tensor selected(ElemKind::IndexTy, {9, 1});
  inputs.getHandle().initXavier(1, PRNG);
  kernel1.getHandle().randomize(-1.0, 1.4, PRNG);
  bias1.getHandle().randomize(-0.2, 0.5, PRNG);
  kernel2.getHandle().randomize(-1.8, 2.3, PRNG);
  bias2.getHandle().randomize(-0.5, 1.0, PRNG);
  auto selectedH = selected.getHandle<uint64_t>();
  for (size_t i = 0; i < 9; i++) {
    selectedH.raw(i) = PRNG.nextRandInt(0, 29);
  }
  std::array<uint64_t, 4> S1{{9, 6, 10, 1}};
  llvm::ArrayRef<uint64_t> shape1(S1);
  std::array<uint64_t, 2> S2{{9, 30}};
  llvm::ArrayRef<uint64_t> shape2(S2);
  Tensor out1(ElemKind::FloatTy, shape2);
  Tensor out2(ElemKind::FloatTy, shape2);

  trainConvNet(&inputs, &kernel1, &bias1, &kernel2, &bias2, &selected, shape1,
               shape2, &out1, BackendKind::OpenCL);
  trainConvNet(&inputs, &kernel1, &bias1, &kernel2, &bias2, &selected, shape1,
               shape2, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST(OpenCLCorrectnessTest, gatherTest) {
  PseudoRNG PRNG;
  constexpr size_t nSlices = 16;
  constexpr size_t nGathered = 8;

  Tensor data(ElemKind::FloatTy, {nSlices, 16, 3, 2});
  data.getHandle().initXavier(1, PRNG);

  Tensor indices(ElemKind::IndexTy, {nGathered});
  auto indicesH = indices.getHandle<uint64_t>();
  for (size_t i = 0; i < nGathered; i++) {
    indicesH.raw(i) = PRNG.nextRandInt(0, nSlices - 1);
  }

  Tensor out1(ElemKind::FloatTy, {nGathered, 16, 3, 2});
  Tensor out2(ElemKind::FloatTy, {nGathered, 16, 3, 2});

  inferGatherNet(&data, &indices, &out1, BackendKind::OpenCL);
  inferGatherNet(&data, &indices, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST(OpenCLCorrectnessTest, tanhConcatTest) {
  Tensor I1(ElemKind::FloatTy, {10, 5});
  Tensor I2(ElemKind::FloatTy, {20, 5});
  Tensor I3(ElemKind::FloatTy, {30, 5});

  for (size_t i = 0; i < 10; i++) {
    for (size_t j = 0; j < 5; j++) {
      I1.getHandle<float>().at({i, j}) = 0.05 * (i + j * 10 + 1);

      I2.getHandle<float>().at({i, j}) = 0.10 * (i + j * 10 + 1);
      I2.getHandle<float>().at({i + 10, j}) = 0.15 * (i + j * 10 + 1);

      I3.getHandle<float>().at({i, j}) = 0.20 * (i + j * 10 + 1);
      I3.getHandle<float>().at({i + 10, j}) = 0.25 * (i + j * 10 + 1);
      I3.getHandle<float>().at({i + 20, j}) = 0.30 * (i + j * 10 + 1);
    }
  }

  Tensor out1(ElemKind::FloatTy, {100, 5});
  Tensor out2(ElemKind::FloatTy, {100, 5});

  inferTanhConcatNet(&I1, &I2, &I3, &out1, BackendKind::OpenCL);
  inferTanhConcatNet(&I1, &I2, &I3, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}
