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

// Silence Apple's warning about the deprecation of OpenCL.
#define CL_SILENCE_DEPRECATION

// Silence warnings about using deprecated OpenCL 1.2 functions.
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "../../lib/Backends/OpenCL/OpenCLDeviceManager.h"
#include "glow/Backends/DeviceManager.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "gtest/gtest.h"

/// Takes an llvm::Expected<T> \p rhsOrErrV, asserts that it is not an error,
/// and takes the value from rhsOrErrV and assigns it to \p lhs.
#define ASSERT_AND_ASSIGN_VALUE(lhs, rhsOrErrV)                                \
  do {                                                                         \
    if (rhsOrErrV) {                                                           \
      lhs = std::move(rhsOrErrV.get());                                        \
    } else {                                                                   \
      ASSERT_TRUE(false);                                                      \
    }                                                                          \
  } while (0)

using namespace glow;
using llvm::cast;

TEST(OpenCLCorrectnessTest, convOps) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {2, 3, 16, 16});
  inputs.getHandle().initXavier(1, PRNG);
  Tensor out1;
  Tensor out2;

  inferBasicConvNet(&inputs, &out1, "OpenCL", 8);
  inferBasicConvNet(&inputs, &out2, "Interpreter", 8);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST(OpenCLCorrectnessTest, inferMixedNet) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {2, 3, 16, 16});
  inputs.getHandle().initXavier(1, PRNG);
  Tensor out1;
  Tensor out2;

  inferMixedNet(&inputs, &out1, "OpenCL");
  inferMixedNet(&inputs, &out2, "Interpreter");

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

  trainSoftMaxNet(&inputs, &weights, &bias, &selected, &out1, "OpenCL");
  trainSoftMaxNet(&inputs, &weights, &bias, &selected, &out2, "Interpreter");

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

  inferTanhConcatNet(&I1, &I2, &I3, &out1, "OpenCL");
  inferTanhConcatNet(&I1, &I2, &I3, &out2, "Interpreter");

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST(OpenCLCorrectnessTest, SetDeviceMemory) {
  using namespace glow;
  using namespace runtime;
  auto openCLConfigEmpty = DeviceConfig("OpenCL");
  auto openCLConfigFull = DeviceConfig("OpenCL");
  openCLConfigFull.setDeviceMemory(32768);
  // Default device memory size is from OpenCL device info.
  // This memory size can be limited by deviceConfig.
  // No setting at all, default memory size from OpenCL device info.
  OpenCLDeviceManager openCLDeviceDefault(openCLConfigEmpty);
  llvm::Error err1 = openCLDeviceDefault.init();
  ASSERT_FALSE(errToBool(std::move(err1)));
  uint64_t memSize = openCLDeviceDefault.getMaximumMemory();
  // If limited by deviceConfig.
  OpenCLDeviceManager openCLDeviceSetByDeviceConfig(openCLConfigFull);
  llvm::Error err2 = openCLDeviceSetByDeviceConfig.init();
  ASSERT_FALSE(errToBool(std::move(err2)));
  EXPECT_EQ(openCLDeviceSetByDeviceConfig.getMaximumMemory(), 32768);
  // If devicConfig defines larger memory size than the OpenCL device info,
  // then fall back to default.
  auto openCLConfigLarger = DeviceConfig("OpenCL");
  openCLConfigLarger.setDeviceMemory(memSize + 10000);
  OpenCLDeviceManager openCLDeviceLarger(openCLConfigLarger);
  llvm::Error err3 = openCLDeviceLarger.init();
  ASSERT_FALSE(errToBool(std::move(err3)));
  EXPECT_EQ(openCLDeviceLarger.getMaximumMemory(), memSize);
}

class OpenCLCommandQueuePoolTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Get an OpenCL platform ID.
    std::vector<cl_platform_id> platforms(1);
    cl_int err = clGetPlatformIDs(1, platforms.data(), NULL);
    ASSERT_EQ(err, CL_SUCCESS) << "clGetPlatformIDs failed.";

    // Get an OpenCL device ID.
    cl_platform_id platform_id_used = platforms[0];
    std::vector<cl_device_id> devices(1);
    err = clGetDeviceIDs(platform_id_used, CL_DEVICE_TYPE_ALL, 1,
                         devices.data(), NULL);
    ASSERT_EQ(err, CL_SUCCESS) << "clGetDeviceIDs failed";

    // Create an OpenCL context.
    device_ = devices[0];
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, nullptr);
    ASSERT_TRUE(context_) << "clCreateContext failed";

    // Set the context and device on the pool to prepare for the tests.
    pool_.setContext(context_);
    pool_.setDevice(device_);
  }

  void TearDown() override {
    // Release the context.
    cl_int err = clReleaseContext(context_);
    ASSERT_EQ(err, CL_SUCCESS) << "clReleaseContext failed";
  }

  cl_context context_{nullptr};
  cl_device_id device_{0};
  runtime::OpenCLCommandQueuePool pool_;
};

/// Tests that the command queue pool returns an error when a queue is requested
/// but the pool is not correctly initialized.
TEST_F(OpenCLCommandQueuePoolTest, ErrorWhenNotInitialized) {
  // Set the context and device to nonsensical values.
  pool_.setContext(nullptr);
  pool_.setDevice(0);

  // A request for a command queue should return an llvm::Error.
  ASSERT_FALSE(pool_.requestCommandQueue());
}

/// Tests that the pool reuses queues.
TEST_F(OpenCLCommandQueuePoolTest, QueueReuse) {
  cl_command_queue backingQueue1, backingQueue2;
  runtime::OpenCLCommandQueue queue;

  // Request a queue.
  llvm::Expected<runtime::OpenCLCommandQueue> queueOrError =
      pool_.requestCommandQueue(0);
  ASSERT_AND_ASSIGN_VALUE(queue, queueOrError);
  backingQueue1 = queue.backingQueue;

  // Put it back and request another.
  pool_.returnCommandQueue(queue);
  queueOrError = pool_.requestCommandQueue(0);
  ASSERT_AND_ASSIGN_VALUE(queue, queueOrError);
  backingQueue2 = queue.backingQueue;

  // The retuned queues should have been the same and only one should have been
  // allocated.
  EXPECT_EQ(backingQueue1, backingQueue2);
  EXPECT_EQ(pool_.getNumAllocatedQueues(), 1);
  EXPECT_EQ(pool_.getNumAllocatedQueuesForProperties(0), 1);

  pool_.returnCommandQueue(queue);
}

/// Tests that the pool does not reuse queues if the requested properties are
/// different.
TEST_F(OpenCLCommandQueuePoolTest, NoQueueReuseWithDifferentProps) {
  cl_command_queue backingQueue1, backingQueue2;
  runtime::OpenCLCommandQueue queue;

  // Request a queue.
  llvm::Expected<runtime::OpenCLCommandQueue> queueOrError =
      pool_.requestCommandQueue(0);
  ASSERT_AND_ASSIGN_VALUE(queue, queueOrError);
  backingQueue1 = queue.backingQueue;

  // Put it back and request another with profiling enabled.
  pool_.returnCommandQueue(queue);
  queueOrError = pool_.requestCommandQueue(CL_QUEUE_PROFILING_ENABLE);
  ASSERT_AND_ASSIGN_VALUE(queue, queueOrError);
  backingQueue2 = queue.backingQueue;

  // The retuned queues should not have been the same and two should have been
  // allocated; one with profiling enabled and one without.
  EXPECT_NE(backingQueue1, backingQueue2);
  EXPECT_EQ(pool_.getNumAllocatedQueues(), 2);
  EXPECT_EQ(pool_.getNumAllocatedQueuesForProperties(0), 1);
  EXPECT_EQ(pool_.getNumAllocatedQueuesForProperties(CL_QUEUE_PROFILING_ENABLE),
            1);

  pool_.returnCommandQueue(queue);
}
