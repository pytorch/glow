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

#include "../../lib/Backends/OpenCL/OpenCLDeviceManager.h"
#include "glow/Backends/DeviceManager.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "gtest/gtest.h"

using namespace glow;
using llvm::cast;

TEST(OpenCLCorrectnessTest, convOps) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {2, 3, 16, 16});
  inputs.getHandle().initXavier(1, PRNG);
  Tensor out1;
  Tensor out2;

  inferBasicConvNet(&inputs, &out1, BackendKind::OpenCL, 8);
  inferBasicConvNet(&inputs, &out2, BackendKind::Interpreter, 8);

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
  std::array<size_t, 2> S{{8, 23}};
  llvm::ArrayRef<size_t> shape(S);
  Tensor inputs(ElemKind::FloatTy, shape);
  Tensor weights(ElemKind::FloatTy, {23, 23});
  Tensor bias(ElemKind::FloatTy, {23});
  Tensor selected(ElemKind::Int64ITy, {8, 1});
  inputs.getHandle().initXavier(1, PRNG);
  weights.getHandle().randomize(0.0, 0.5, PRNG);
  bias.getHandle().randomize(-0.2, 0.0, PRNG);
  auto selectedH = selected.getHandle<int64_t>();
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

TEST(OpenCLCorrectnessTest, SetDeviceMemory) {
  using namespace glow;
  using namespace runtime;
  // Test OpenCL BackendKind::OpenCL.
  auto openCLConfigEmpty = DeviceConfig(BackendKind::OpenCL);
  auto openCLConfigFull = DeviceConfig(BackendKind::OpenCL);
  openCLConfigFull.setDeviceMemory(32768);
  // Default device memory size is from OpenCL device info.
  // This memory size can be limited by deviceConfig.
  // No setting at all, default memory size from OpenCL device info.
  OpenCLDeviceManager openCLDeviceDefault(openCLConfigEmpty);
  llvm::Error err1 = openCLDeviceDefault.init();
  uint64_t memSize = openCLDeviceDefault.getMaximumMemory();
  // If limited by deviceConfig.
  OpenCLDeviceManager openCLDeviceSetByDeviceConfig(openCLConfigFull);
  llvm::Error err2 = openCLDeviceSetByDeviceConfig.init();
  EXPECT_EQ(openCLDeviceSetByDeviceConfig.getMaximumMemory(), 32768);
  // If devicConfig defines larger memory size than the OpenCL device info,
  // then fall back to default.
  auto openCLConfigLarger = DeviceConfig(BackendKind::OpenCL);
  openCLConfigLarger.setDeviceMemory(memSize + 10000);
  OpenCLDeviceManager openCLDeviceLarger(openCLConfigLarger);
  llvm::Error err3 = openCLDeviceLarger.init();
  EXPECT_EQ(openCLDeviceLarger.getMaximumMemory(), memSize);
}
