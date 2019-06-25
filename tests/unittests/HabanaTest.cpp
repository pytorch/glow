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

#include "glow/Support/Compiler.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "gtest/gtest.h"

#include <glog/logging.h>

#include "synapse.h"

#include "perf_lib_layer_params.h"

#include <future>
#include <mutex>
#include <thread>
#include <vector>

#define chk(X) CHECK_EQ((X), synSuccess) << "Expected synStatus be synSuccess";

class Habana : public ::testing::Test {
protected:
  Habana() { llvm::sys::fs::createTemporaryFile("Habana", "recipe", recipe_); }

  ~Habana() { llvm::sys::fs::remove(recipe_); }

  const char *recipe() { return recipe_.c_str(); }

private:
  llvm::SmallString<64> recipe_;
};

/// Init device.
static uint32_t initDevice() {
  uint32_t deviceId;
  chk(synInitialize());
  chk(synAcquireDevice(&deviceId, nullptr));
  return deviceId;
}

/// Destroy device.
static void destroyDevice() { chk(synDestroy()); }

// Test device initialization.
TEST_F(Habana, Init) {
  initDevice();
  destroyDevice();
}

// Test multithreaded device initialization.
TEST_F(Habana, MultithreadedAcquireReleaseDevice) {
  constexpr unsigned numThreads = 6;
  constexpr unsigned numIterations = 50;

  chk(synInitialize());

  std::mutex acquireReleaseMtx;
  std::promise<void> readyPromise;
  std::shared_future<void> readyFuture(readyPromise.get_future());
  std::vector<std::thread> threads;

  for (unsigned i = 0; i < numThreads; ++i) {
    threads.emplace_back([&acquireReleaseMtx, numIterations = numIterations,
                          readyFuture = readyFuture]() {
      readyFuture.wait();

      uint32_t deviceId;
      std::unique_lock<std::mutex> lk(acquireReleaseMtx, std::defer_lock);
      for (unsigned j = 0; j < numIterations; ++j) {
        // synAcquireDevice and synReleaseDevice are in two separate critical
        // sections below to allow for more interleaving between threads.
        lk.lock();
        synStatus acquireStatus = synAcquireDevice(&deviceId, nullptr);
        lk.unlock();
        if (acquireStatus == synSuccess) {
          lk.lock();
          synReleaseDevice(deviceId);
          lk.unlock();
        }
      }
    });
  }

  readyPromise.set_value();

  for (auto &thread : threads) {
    thread.join();
  }

  chk(synDestroy());
}

/// Test device memory allocation.
TEST_F(Habana, Allocate) {
  uint32_t deviceId = initDevice();

  // Allocate memory.
  void *data = nullptr;
  chk(synMalloc(deviceId, 4 * 1024, synMemFlags::synMemHost, &data, 0));

  // Free memory.
  chk(synFree(deviceId, data));

  destroyDevice();
}

/// Test tensor creation.
TEST_F(Habana, CreateTensor) {
  uint32_t deviceId = initDevice();

  // Allocate memory.
  void *data = nullptr;
  chk(synMalloc(deviceId, 4 * 1024, synMemFlags::synMemHost, &data, 0));

  // Create tensor.
  unsigned sizes[SYN_MAX_TENSOR_DIM] = {4, 1024};
  synTensorDescriptor desc(syn_type_fixed, 2u, sizes, data, synMemoryHost,
                           false, nullptr, 1);
  synTensor tensor;
  chk(synCreateTensor(&desc, &tensor, false));

  // Destroy tensor.
  chk(synDestroyTensor(tensor));

  // Free memory.
  chk(synFree(deviceId, data));

  destroyDevice();
}

/// Helper to compile a single-layer FC network.
static void compileFC(const char *recipe, unsigned batchSize, unsigned inputF,
                      unsigned outputF) {
  uint32_t deviceId = initDevice();

  // Allocate input tensor.
  void *data = nullptr;
  chk(synMalloc(deviceId, batchSize * inputF, synMemFlags::synMemHost, &data,
                0));
  unsigned sizes[SYN_MAX_TENSOR_DIM] = {inputF, batchSize};
  synTensorDescriptor desc(syn_type_fixed, 2u, sizes, data, synMemoryHost,
                           false, nullptr, 1);
  synTensor tensor;
  chk(synCreateTensor(&desc, &tensor, false));

  // Allocate weight tensor.
  void *weightData = nullptr;
  chk(synMalloc(deviceId, inputF * outputF, synMemFlags::synMemHost,
                &weightData, 0));
  memset(weightData, 1, inputF * outputF);
#ifdef SYNAPSE_0_1_5
  unsigned weightSize[SYN_MAX_TENSOR_DIM] = {outputF, inputF};
#else
  unsigned weightSize[SYN_MAX_TENSOR_DIM] = {inputF, outputF};
#endif
  synTensorDescriptor weightDesc(syn_type_fixed, 2u, weightSize, weightData,
                                 synMemoryHost, false, nullptr, 1);
  synTensor weightTensor;
  chk(synCreateTensor(&weightDesc, &weightTensor, false));

  // Allocate bias memory.
  void *biasData = nullptr;
  chk(synMalloc(deviceId, sizeof(int32_t) * outputF, synMemFlags::synMemHost,
                &biasData, 0));
  memset(biasData, 0, sizeof(int32_t) * outputF);
  unsigned biasSize[SYN_MAX_TENSOR_DIM] = {outputF};
  synTensorDescriptor biasDesc(syn_type_int32, 1u, biasSize, biasData,
                               synMemoryHost, false, nullptr, 0);
  synTensor biasTensor;
  chk(synCreateTensor(&biasDesc, &biasTensor, false));

  // Allocate output tensor.
  void *outputData = nullptr;
  chk(synMalloc(deviceId, batchSize * outputF, synMemFlags::synMemHost,
                &outputData, 0));
  unsigned outputSize[SYN_MAX_TENSOR_DIM] = {outputF, batchSize};
  synTensorDescriptor outputDesc(syn_type_fixed, 2u, outputSize, outputData,
                                 synMemoryHost, false, nullptr, 1);
  synTensor outputTensor;
  chk(synCreateTensor(&outputDesc, &outputTensor, false));

  // Create fully connected node.
  synFCParams fcParams;
  fcParams.activation.reluEnable = false;
  chk(synFullyConnected(tensor, weightTensor, biasTensor, outputTensor,
                        fcParams, ""));

  // Compile graph.
  CompilationAttribute compileParams[1];
  compileParams[0].type = VISUALIZATION;
  compileParams[0].u32 = 1;
  chk(synCompileGraph(compileParams, 1, recipe));

  // Destroy tensors.
  chk(synDestroyTensor(outputTensor));
  chk(synDestroyTensor(biasTensor));
  chk(synDestroyTensor(weightTensor));
  chk(synDestroyTensor(tensor));

  // Free memory.
  chk(synFree(deviceId, outputData));
  chk(synFree(deviceId, biasData));
  chk(synFree(deviceId, weightData));
  chk(synFree(deviceId, data));

  destroyDevice();
}

/// Test compilation (recipe creation).
TEST_F(Habana, CompileFC) { compileFC(recipe(), 4, 1024, 512); }

/// Test compilation and running on device.
TEST_F(Habana, RunFC) {
  constexpr unsigned batchSize = 2;
  constexpr unsigned inputF = 32;
  constexpr unsigned outputF = 32;

  // First, compile a recipe.
  compileFC(recipe(), batchSize, inputF, outputF);

  uint32_t deviceId = initDevice();

  // Allocate input tensor.
  void *data = nullptr;
  size_t dataSize = batchSize * inputF;
  chk(synMalloc(deviceId, dataSize, synMemFlags::synMemHost, &data, 0));
  memset(data, 1, dataSize);
  unsigned sizes[SYN_MAX_TENSOR_DIM] = {inputF, batchSize};
  synTensorDescriptor desc(syn_type_fixed, 2u, sizes, data, synMemoryHost,
                           false, nullptr, 1);
  synTensor tensor;
  chk(synCreateTensor(&desc, &tensor, false));

  // Allocate output tensor.
  void *outputData = nullptr;
  size_t outputSize = batchSize * outputF;
  chk(synMalloc(deviceId, outputSize, synMemFlags::synMemHost, &outputData, 0));
  unsigned outputSizes[SYN_MAX_TENSOR_DIM] = {outputF, batchSize};
  synTensorDescriptor outputDesc(syn_type_fixed, 2u, outputSizes, outputData,
                                 synMemoryHost, false, nullptr, 1);
  synTensor outputTensor;
  chk(synCreateTensor(&outputDesc, &outputTensor, false));

  // Load the recipe.
  uint64_t topologyId = 0;
  chk(synLoadRecipe(deviceId, recipe(), &topologyId));
  chk(synActivateTopology(deviceId, topologyId));

  // Run the recipe.
  synWaitHandle handle;
  chk(synEnqueue(deviceId, data, dataSize, outputData, outputSize, &handle));
  synWaitForEvent(deviceId, handle);
  synDestroyHandle(handle);

  // Test the results.
  for (size_t i = 0; i < outputSize; i++) {
    EXPECT_EQ(static_cast<int8_t *>(outputData)[i], 32);
  }

  // Destroy tensors.
  chk(synDestroyTensor(outputTensor));
  chk(synDestroyTensor(tensor));

  // Free memory.
  chk(synFree(deviceId, outputData));
  chk(synFree(deviceId, data));

  destroyDevice();
}

TEST_F(Habana, Relu) {
  constexpr unsigned size = 32;

  uint32_t deviceId = initDevice();

  void *inputData = nullptr;
  uint64_t inputSize = sizeof(int8_t) * size;
  chk(synMalloc(deviceId, inputSize, synMemFlags::synMemHost, &inputData));
  unsigned inputDims[SYN_MAX_TENSOR_DIM] = {size};
  synTensorDescriptor inputDesc(syn_type_fixed, 1u, inputDims, inputData,
                                synMemoryHost);
  synTensor inputTensor;
  chk(synCreateTensor(&inputDesc, &inputTensor));

  void *outputData = nullptr;
  uint64_t outputSize = sizeof(int8_t) * size;
  chk(synMalloc(deviceId, outputSize, synMemFlags::synMemHost, &outputData));
  unsigned outputDims[SYN_MAX_TENSOR_DIM] = {size};
  synTensorDescriptor outputDesc(syn_type_fixed, 1u, outputDims, outputData,
                                 synMemoryHost);
  synTensor outputTensor;
  chk(synCreateTensor(&outputDesc, &outputTensor));

  chk(synCreateGenericNode(&inputTensor, &outputTensor, 1, 1, nullptr,
                           "relu_i8", "test_relu", nullptr, nullptr));

  // Compile graph.
  CompilationAttribute compileParams[1];
  compileParams[0].type = VISUALIZATION;
  compileParams[0].u32 = 1;
  chk(synCompileGraph(compileParams, 1, recipe()));

  // Load the recipe.
  uint64_t topologyId = 0;
  chk(synLoadRecipe(deviceId, recipe(), &topologyId));
  chk(synActivateTopology(deviceId, topologyId));

  // Set inputs.
  for (unsigned i = 0; i < size; i++) {
    static_cast<int8_t *>(inputData)[i] = i - size / 2;
  }

  // Run the recipe.
  synWaitHandle handle;
  chk(synEnqueue(deviceId, inputData, inputSize, outputData, outputSize,
                 &handle));
  synWaitForEvent(deviceId, handle);
  synDestroyHandle(handle);

  // Check outputs.
  for (unsigned i = 0; i < size; i++) {
    int8_t out = static_cast<int8_t *>(outputData)[i];
    ASSERT_EQ(out, std::max<int8_t>(0, i - size / 2));
  }

  chk(synDestroyTensor(outputTensor));
  chk(synDestroyTensor(inputTensor));
  chk(synFree(deviceId, outputData));
  chk(synFree(deviceId, inputData));

  destroyDevice();
}

TEST_F(Habana, MatmulFp32) {
  // For simplicity assuming two 2*2 matrices.
  constexpr unsigned N = 2;

  uint32_t deviceId = initDevice();

  // Allocate LHS.
  void *lhs = nullptr;
  uint64_t lhsSize = sizeof(float) * N * N;
  chk(synMalloc(deviceId, lhsSize, synMemFlags::synMemHost, &lhs));
  unsigned lhsDims[SYN_MAX_TENSOR_DIM] = {N, N};
  synTensorDescriptor lhsDesc(syn_type_single, 2u, lhsDims, lhs, synMemoryHost,
                              false, "lhs");
  synTensor lhsTensor;
  chk(synCreateTensor(&lhsDesc, &lhsTensor));

  // Allocate RHS.
  void *rhs = nullptr;
  uint64_t rhsSize = sizeof(float) * N * N;
  chk(synMalloc(deviceId, rhsSize, synMemFlags::synMemHost, &rhs));
  unsigned rhsDims[SYN_MAX_TENSOR_DIM] = {N, N};
  synTensorDescriptor rhsDesc(syn_type_single, 2u, rhsDims, rhs, synMemoryHost,
                              false, "rhs");
  synTensor rhsTensor;
  chk(synCreateTensor(&rhsDesc, &rhsTensor, false, false, true));

  // Allocate output.
  void *outputData = nullptr;
  uint64_t outputSize = sizeof(float) * N * N;
  chk(synMalloc(deviceId, outputSize, synMemFlags::synMemHost, &outputData));
  unsigned outputDims[SYN_MAX_TENSOR_DIM] = {N, N};
  synTensorDescriptor outputDesc(syn_type_single, 2u, outputDims, outputData,
                                 synMemoryHost, false, "output");
  synTensor outputTensor;
  chk(synCreateTensor(&outputDesc, &outputTensor));

  synTensor inputs[] = {lhsTensor, rhsTensor};
  chk(synCreateGenericNode(inputs, &outputTensor, 2, 1, nullptr,
                           "matrix_multiply_f32", "test_mm_fp32", nullptr,
                           nullptr));

  // Set weights.
  for (unsigned i = 0; i < rhsSize; i++) {
    static_cast<float *>(rhs)[i] = i;
  }

  // Compile graph.
  CompilationAttribute compileParams[1];
  compileParams[0].type = VISUALIZATION;
  compileParams[0].u32 = 1;
  chk(synCompileGraph(compileParams, 1, recipe()));

  // Load the recipe.
  uint64_t topologyId = 0;
  chk(synLoadRecipe(deviceId, recipe(), &topologyId));
  chk(synActivateTopology(deviceId, topologyId));

  // Set inputs.
  for (unsigned i = 0; i < lhsSize; i++) {
    static_cast<float *>(lhs)[i] = i;
  }

  // Run the recipe.
  EnqueueTensorInfo inputTensors[2];
  inputTensors[0].tensorName = "lhs";
  inputTensors[0].pTensorData = static_cast<char *>(lhs);
  inputTensors[0].tensorSize = lhsSize;
  inputTensors[1].tensorName = "rhs";
  inputTensors[1].pTensorData = static_cast<char *>(rhs);
  inputTensors[1].tensorSize = rhsSize;

  EnqueueTensorInfo outputs;
  outputs.tensorName = "output";
  outputs.pTensorData = static_cast<char *>(outputData);
  outputs.tensorSize = outputSize;

  synWaitHandle handle;
  chk(synEnqueueByName(deviceId, inputTensors, 1, &outputs, 1, &handle));
  synWaitForEvent(deviceId, handle);
  synDestroyHandle(handle);

  // Check outputs.
  float expectedOutput[] = {2.0, 3.0, 6.0, 11.0};
  for (unsigned i = 0; i < N * N; i++) {
    float out = static_cast<float *>(outputData)[i];
    EXPECT_EQ(out, expectedOutput[i]);
  }

  chk(synDestroyTensor(outputTensor));
  chk(synDestroyTensor(rhsTensor));
  chk(synDestroyTensor(lhsTensor));
  chk(synFree(deviceId, outputData));
  chk(synFree(deviceId, rhs));
  chk(synFree(deviceId, lhs));

  destroyDevice();
}

// Data type for specific pooling parameters.
// Note: DO NOT CHANGE THE ORDER OF MEMBERS!
struct synPoolParams {
  // Padding
  int pWbegin;
  int pWend;
  int pHbegin;
  int pHend;
  // Kernel
  int kW;
  int kH;
  // Stride
  int sW;
  int sH;
  // Dilation
  int dilW;
  int dilH;
  int poolingConvention;

  synPoolParams()
      : pWbegin(0), pWend(0), pHbegin(0), pHend(0), kW(1), kH(1), sW(1), sH(1),
        dilW(1), dilH(1), poolingConvention(0) {}
};

TEST_F(Habana, MaxPoolRelu) {
  uint32_t deviceId = initDevice();

  // Allocate input tensor.
  void *inputData = nullptr;
  uint64_t inputSize = sizeof(int8_t) * 112 * 112 * 64;
  chk(synMalloc(deviceId, inputSize, synMemFlags::synMemHost, &inputData));
  unsigned inputDims[4] = {64, 112, 112, 1};
  synTensorDescriptor inputDesc(syn_type_fixed, 4u, inputDims, inputData,
                                synMemoryHost);
  synTensor inputTensor;
  chk(synCreateTensor(&inputDesc, &inputTensor));

  // Allocate pool tensor.
  void *poolData = nullptr;
  uint64_t poolSize = sizeof(int8_t) * 56 * 56 * 64;
  chk(synMalloc(deviceId, poolSize, synMemFlags::synMemHost, &poolData));
  unsigned outputDims[4] = {64, 56, 56, 1};
  synTensorDescriptor poolDesc(syn_type_fixed, 4u, outputDims, poolData,
                               synMemoryHost);
  synTensor poolTensor;
  chk(synCreateTensor(&poolDesc, &poolTensor));

  // Allocate relu tensor.
  void *reluData = nullptr;
  chk(synMalloc(deviceId, poolSize, synMemFlags::synMemHost, &reluData));
  synTensorDescriptor reluDesc(syn_type_fixed, 4u, outputDims, reluData,
                               synMemoryHost);
  synTensor reluTensor;
  chk(synCreateTensor(&reluDesc, &reluTensor));

  // Create pooling parameters.
  synPoolParams params;
  params.pWbegin = params.pWend = params.pHbegin = params.pHend = 1;
  params.kW = params.kH = 3;
  params.sW = params.sH = 2;
  params.dilW = params.dilH = 1;

  // Create pool and relu nodes.
  chk(synCreateGenericNode(&inputTensor, &poolTensor, 1, 1, (void *)(&params),
                           "maxpool_2d_i8", "test_maxpool", nullptr, nullptr));
  chk(synCreateGenericNode(&poolTensor, &reluTensor, 1, 1, nullptr, "relu_i8",
                           "test_relu", nullptr, nullptr));

  // Compile graph.
  CompilationAttribute compileParams[1];
  compileParams[0].type = VISUALIZATION;
  compileParams[0].u32 = 1;
  chk(synCompileGraph(compileParams, 1, recipe()));

  // Load the recipe.
  uint64_t topologyId = 0;
  chk(synLoadRecipe(deviceId, recipe(), &topologyId));
  chk(synActivateTopology(deviceId, topologyId));

  // Run the recipe.
  synWaitHandle handle;
  chk(synEnqueue(deviceId, inputData, inputSize, reluData, poolSize, &handle));
  synWaitForEvent(deviceId, handle);
  synDestroyHandle(handle);

  chk(synDestroyTensor(reluTensor));
  chk(synDestroyTensor(poolTensor));
  chk(synDestroyTensor(inputTensor));
  chk(synFree(deviceId, reluData));
  chk(synFree(deviceId, poolData));
  chk(synFree(deviceId, inputData));
  destroyDevice();
}

TEST_F(Habana, Concat) {
  constexpr unsigned size = 4;

  uint32_t deviceId = initDevice();

  void *i1Data = nullptr;
  uint64_t i1Size = sizeof(int8_t) * size;
  chk(synMalloc(deviceId, i1Size, synMemFlags::synMemHost, &i1Data));
  unsigned i1Dims[SYN_MAX_TENSOR_DIM] = {size};
  synTensorDescriptor i1Desc(syn_type_fixed, 1u, i1Dims, i1Data, synMemoryHost,
                             false, "input1");
  synTensor i1Tensor;
  chk(synCreateTensor(&i1Desc, &i1Tensor));

  void *i2Data = nullptr;
  uint64_t i2Size = sizeof(int8_t) * size;
  chk(synMalloc(deviceId, i2Size, synMemFlags::synMemHost, &i2Data));
  unsigned i2Dims[SYN_MAX_TENSOR_DIM] = {size};
  synTensorDescriptor i2Desc(syn_type_fixed, 1u, i2Dims, i2Data, synMemoryHost,
                             false, "input2");
  synTensor i2Tensor;
  chk(synCreateTensor(&i2Desc, &i2Tensor));

  void *i3Data = nullptr;
  uint64_t i3Size = sizeof(int8_t) * size;
  chk(synMalloc(deviceId, i3Size, synMemFlags::synMemHost, &i3Data));
  unsigned i3Dims[SYN_MAX_TENSOR_DIM] = {size};
  synTensorDescriptor i3Desc(syn_type_fixed, 1u, i3Dims, i3Data, synMemoryHost,
                             false, "input3");
  synTensor i3Tensor;
  chk(synCreateTensor(&i3Desc, &i3Tensor));

  void *outputData = nullptr;
  uint64_t outputSize = sizeof(int8_t) * size * 3;
  chk(synMalloc(deviceId, outputSize, synMemFlags::synMemHost, &outputData));
  unsigned outputDims[SYN_MAX_TENSOR_DIM] = {size * 3};
  synTensorDescriptor outputDesc(syn_type_fixed, 1u, outputDims, outputData,
                                 synMemoryHost, false, "output");
  synTensor outputTensor;
  chk(synCreateTensor(&outputDesc, &outputTensor));

  synTensor inputs[3] = {i1Tensor, i2Tensor, i3Tensor};
  chk(synCreateGenericNode(inputs, &outputTensor, 3, 1, nullptr, "Concat",
                           "test_concat", nullptr, nullptr));

  // Compile graph.
  CompilationAttribute compileParams[1];
  compileParams[0].type = VISUALIZATION;
  compileParams[0].u32 = 1;
  chk(synCompileGraph(compileParams, 1, recipe()));

  // Load the recipe.
  uint64_t topologyId = 0;
  chk(synLoadRecipe(deviceId, recipe(), &topologyId));
  chk(synActivateTopology(deviceId, topologyId));

  // Set inputs.
  for (unsigned i = 0; i < size; i++) {
    static_cast<int8_t *>(i1Data)[i] = i;
  }
  for (unsigned i = 0; i < size; i++) {
    static_cast<int8_t *>(i2Data)[i] = size - i;
  }
  for (unsigned i = 0; i < size; i++) {
    static_cast<int8_t *>(i3Data)[i] = i * 3;
  }

  // Run the recipe.
  EnqueueTensorInfo etii[3];
  etii[0].tensorName = "input1";
  etii[0].pTensorData = static_cast<char *>(i1Data);
  etii[0].tensorSize = i1Size;
  etii[1].tensorName = "input2";
  etii[1].pTensorData = static_cast<char *>(i2Data);
  etii[1].tensorSize = i2Size;
  etii[2].tensorName = "input3";
  etii[2].pTensorData = static_cast<char *>(i3Data);
  etii[2].tensorSize = i3Size;
  EnqueueTensorInfo etio;
  etio.tensorName = "output";
  etio.pTensorData = static_cast<char *>(outputData);
  etio.tensorSize = outputSize;

  synWaitHandle handle;
  chk(synEnqueueByName(deviceId, etii, 3, &etio, 1, &handle));
  synWaitForEvent(deviceId, handle);
  synDestroyHandle(handle);

  // Check outputs.
  for (unsigned i = 0; i < size; i++) {
    auto *outp = static_cast<int8_t *>(outputData);
    EXPECT_EQ(outp[i], i);
    EXPECT_EQ(outp[i + size], size - i);
    EXPECT_EQ(outp[i + size * 2], 3 * i);
  }

  chk(synDestroyTensor(outputTensor));
  chk(synDestroyTensor(i3Tensor));
  chk(synDestroyTensor(i2Tensor));
  chk(synDestroyTensor(i1Tensor));
  chk(synFree(deviceId, outputData));
  chk(synFree(deviceId, i3Data));
  chk(synFree(deviceId, i2Data));
  chk(synFree(deviceId, i1Data));
  destroyDevice();
}

TEST_F(Habana, NoInputs) {
  constexpr unsigned size = 32;

  {
    uint32_t deviceId = initDevice();

    // Create constant tensor c1.
    std::array<float, size> c1;
    std::fill(c1.begin(), c1.end(), 1.0f);
    unsigned c1Dims[SYN_MAX_TENSOR_DIM] = {size};
    synTensorDescriptor c1Desc(syn_type_fixed, 1u, c1Dims, c1.data(),
                               synMemoryHost, false, "c1");
    synTensor c1Tensor;
    chk(synCreateTensor(&c1Desc, &c1Tensor, /*isOutput*/ false,
                        /*isInput*/ false, /*isStaticParam*/ true));

    // Create constant tensor c2.
    std::array<float, size> c2;
    std::fill(c2.begin(), c2.end(), 2.0f);
    unsigned c2Dims[SYN_MAX_TENSOR_DIM] = {size};
    synTensorDescriptor c2Desc(syn_type_fixed, 1u, c2Dims, c2.data(),
                               synMemoryHost, false, "c2");
    synTensor c2Tensor;
    chk(synCreateTensor(&c2Desc, &c2Tensor, /*isOutput*/ false,
                        /*isInput*/ false, /*isStaticParam*/ true));

    // Create output tensor p1.
    void *p1Data;
    uint64_t p1Size = sizeof(uint8_t) * size;
    chk(synMalloc(deviceId, p1Size, synMemFlags::synMemHost, &p1Data));
    unsigned p1Dims[SYN_MAX_TENSOR_DIM] = {size};
    synTensorDescriptor p1Desc(syn_type_fixed, 1u, p1Dims, p1Data,
                               synMemoryHost, false, "p1");
    synTensor p1Tensor;
    chk(synCreateTensor(&p1Desc, &p1Tensor));

    // Create output tensor p2.
    void *p2Data;
    uint64_t p2Size = sizeof(uint8_t) * size;
    chk(synMalloc(deviceId, p2Size, synMemFlags::synMemHost, &p2Data));
    unsigned p2Dims[SYN_MAX_TENSOR_DIM] = {size};
    synTensorDescriptor p2Desc(syn_type_fixed, 1u, p2Dims, p2Data,
                               synMemoryHost, false, "p2");
    synTensor p2Tensor;
    chk(synCreateTensor(&p2Desc, &p2Tensor));

    chk(synCreateGenericNode(&c1Tensor, &p1Tensor, 1, 1, nullptr, "memcpy",
                             "test_memcpy1", nullptr, nullptr));
    chk(synCreateGenericNode(&c2Tensor, &p2Tensor, 1, 1, nullptr, "memcpy",
                             "test_memcpy2", nullptr, nullptr));

    // Compile graph.
    CompilationAttribute compileParams[1];
    compileParams[0].type = VISUALIZATION;
    compileParams[0].u32 = 1;
    chk(synCompileGraph(compileParams, 1, recipe()));

    chk(synDestroyTensor(p2Tensor));
    chk(synDestroyTensor(c2Tensor));
    chk(synDestroyTensor(p1Tensor));
    chk(synDestroyTensor(c1Tensor));

    chk(synFree(deviceId, p2Data));
    chk(synFree(deviceId, p1Data));

    destroyDevice();
  }

  {
    uint32_t deviceId = initDevice();

    void *p1Data;
    uint64_t p1Size = sizeof(uint8_t) * size;
    chk(synMalloc(deviceId, p1Size, synMemFlags::synMemHost, &p1Data));
    unsigned p1Dims[SYN_MAX_TENSOR_DIM] = {size};
    synTensorDescriptor p1Desc(syn_type_fixed, 1u, p1Dims, p1Data,
                               synMemoryHost, false, "p1");
    synTensor p1Tensor;
    chk(synCreateTensor(&p1Desc, &p1Tensor));

    void *p2Data;
    uint64_t p2Size = sizeof(uint8_t) * size;
    chk(synMalloc(deviceId, p2Size, synMemFlags::synMemHost, &p2Data));
    unsigned p2Dims[SYN_MAX_TENSOR_DIM] = {size};
    synTensorDescriptor p2Desc(syn_type_fixed, 1u, p2Dims, p2Data,
                               synMemoryHost, false, "p2");
    synTensor p2Tensor;
    chk(synCreateTensor(&p2Desc, &p2Tensor));

    // Load the recipe.
    uint64_t topologyId = 0;
    chk(synLoadRecipe(deviceId, recipe(), &topologyId));
    chk(synActivateTopology(deviceId, topologyId));

    EnqueueTensorInfo etii = {"unused", (char *)nullptr, 0};
    EnqueueTensorInfo etio[2] = {
        {"p1", (char *)p1Data, (uint32_t)p1Size},
        {"p2", (char *)p2Data, (uint32_t)p2Size},
    };
    synWaitHandle handle;
    chk(synEnqueueByName(deviceId, &etii, 0, etio, 2, &handle));
    synWaitForEvent(deviceId, handle);
    synDestroyHandle(handle);

    for (unsigned i = 0; i < size; i++) {
      EXPECT_EQ(static_cast<uint8_t *>(p1Data)[i], 1);
      EXPECT_EQ(static_cast<uint8_t *>(p2Data)[i], 2);
    }

    chk(synDestroyTensor(p1Tensor));
    chk(synDestroyTensor(p2Tensor));

    chk(synFree(deviceId, p2Data));
    chk(synFree(deviceId, p1Data));

    destroyDevice();
  }
}

///
/// Questions to answer:
///   Do you need synMalloc'ed memory to createTensor? (no)
///   Do you need the original memory buffers around to run a graph (no)?
///   Do you need to provide a memory buffer for an activation synTensor
///     (yes... maybe?  It seems like if you provide `nullptr` things are OK as
///     long as you dont't destroy the tensor? But that seems like a bug.)
///   Do you need to synDestroy after each compilation? (I think not!  You can
///     synDestroyGraph instead.
///
TEST_F(Habana, CompileInferenceInterleave) {
  constexpr unsigned batchSize = 2;
  constexpr unsigned inputF = 16;
  constexpr unsigned outputF = 16;

  uint32_t deviceId;

  auto compile = [&] {
    chk(synCreateGraph(synDeviceGoya));

    std::vector<int8_t> input(batchSize * inputF);
    std::fill(input.begin(), input.end(), 1);
    unsigned inputDims[SYN_MAX_TENSOR_DIM] = {inputF, batchSize};
    synTensorDescriptor inputDesc(syn_type_fixed, 2u, inputDims, input.data(),
                                  synMemoryHost, false);
    synTensor inputT;
    chk(synCreateTensor(&inputDesc, &inputT, false, true, false));

    std::vector<int8_t> weights(inputF * outputF);
    std::fill(weights.begin(), weights.end(), 1);
    unsigned weightsDims[SYN_MAX_TENSOR_DIM] = {inputF, outputF};
    synTensorDescriptor weightsDesc(syn_type_fixed, 2u, weightsDims,
                                    weights.data(), synMemoryHost);
    synTensor weightsT;
    chk(synCreateTensor(&weightsDesc, &weightsT));

    std::vector<int32_t> bias(outputF);
    std::fill(bias.begin(), bias.end(), 1);
    unsigned biasDims[SYN_MAX_TENSOR_DIM] = {outputF};
    synTensorDescriptor biasDesc(syn_type_int32, 1u, biasDims, bias.data(),
                                 synMemoryHost);
    synTensor biasT;
    chk(synCreateTensor(&biasDesc, &biasT));

    std::vector<int8_t> output(batchSize * outputF);
    std::fill(output.begin(), output.end(), 1);
    unsigned outputDims[SYN_MAX_TENSOR_DIM] = {outputF, batchSize};
    synTensorDescriptor outputDesc(syn_type_fixed, 2u, outputDims,
                                   output.data(), synMemoryHost);
    synTensor outputT;
    chk(synCreateTensor(&outputDesc, &outputT));

    synFCParams fcParams;
    fcParams.activation.reluEnable = false;
    chk(synFullyConnected(inputT, weightsT, biasT, outputT, fcParams, ""));

    // Compile graph.
    CompilationAttribute compileParams[1];
    compileParams[0].type = VISUALIZATION;
    compileParams[0].u32 = 1;
    chk(synCompileGraph(compileParams, 1, recipe()));

    chk(synDestroyTensor(outputT));
    chk(synDestroyTensor(biasT));
    chk(synDestroyTensor(weightsT));
    chk(synDestroyTensor(inputT));

    chk(synDestroyGraph());
  };

  auto execute = [&] {
    uint64_t topologyId = 0;
    chk(synLoadRecipe(deviceId, recipe(), &topologyId));
    chk(synActivateTopology(deviceId, topologyId));

    std::vector<int8_t> input(batchSize * inputF);
    std::fill(input.begin(), input.end(), 2);
    std::vector<int8_t> output(batchSize * outputF);
    std::fill(output.begin(), output.end(), 42);

    uint64_t pTopo = 42;
    EXPECT_EQ(synGetActiveTopologyID(deviceId, &pTopo), synSuccess);
    EXPECT_EQ(pTopo, topologyId);
    if (pTopo == 1) {
      EXPECT_EQ(synUnloadTopology(deviceId, 0), synSuccess);
    }

    chk(synMap(deviceId, input.size() * sizeof(input[0]), input.data()));
    chk(synMap(deviceId, output.size() * sizeof(output[0]), output.data()));

    synWaitHandle handle;
    chk(synEnqueue(deviceId, input.data(), input.size(), output.data(),
                   output.size(), &handle));
    chk(synWaitForEvent(deviceId, handle));
    synDestroyHandle(handle);

    chk(synUnmap(deviceId, output.data()));
    chk(synUnmap(deviceId, input.data()));

    for (auto i : input) {
      EXPECT_EQ(i, 2);
    }
    for (auto i : output) {
      EXPECT_EQ(i, 33);
    }
  };

  chk(synInitialize());

  compile();
  chk(synAcquireDevice(&deviceId, nullptr));
  execute();
  compile();
  execute();
  chk(synReleaseDevice(deviceId));
  chk(synDestroy());
}

// Test multithreaded inference.
TEST_F(Habana, MultithreadedInference) {
  constexpr unsigned size = 32;
  constexpr unsigned numThreads = 6;
  constexpr unsigned numIterations = 10;
  constexpr unsigned numEnqueues = 10;

  chk(synInitialize());

  uint32_t deviceId;
  chk(synAcquireDevice(&deviceId, nullptr));

  chk(synCreateGraph(synDeviceGoya));

  void *inputData = nullptr;
  uint64_t inputSize = sizeof(int8_t) * size;
  chk(synMalloc(deviceId, inputSize, synMemFlags::synMemHost, &inputData));
  unsigned inputDims[SYN_MAX_TENSOR_DIM] = {size};
  synTensorDescriptor inputDesc(syn_type_fixed, 1u, inputDims, inputData,
                                synMemoryHost);
  synTensor inputTensor;
  chk(synCreateTensor(&inputDesc, &inputTensor));

  void *outputData = nullptr;
  uint64_t outputSize = sizeof(int8_t) * size;
  chk(synMalloc(deviceId, outputSize, synMemFlags::synMemHost, &outputData));
  unsigned outputDims[SYN_MAX_TENSOR_DIM] = {size};
  synTensorDescriptor outputDesc(syn_type_fixed, 1u, outputDims, outputData,
                                 synMemoryHost);
  synTensor outputTensor;
  chk(synCreateTensor(&outputDesc, &outputTensor));

  chk(synCreateGenericNode(&inputTensor, &outputTensor, 1, 1, nullptr,
                           "relu_i8", "test_relu", nullptr, nullptr));

  CompilationAttribute compileParams[1];
  compileParams[0].type = VISUALIZATION;
  compileParams[0].u32 = 1;
  chk(synCompileGraph(compileParams, 1, recipe()));

  chk(synDestroyGraph());

  chk(synDestroyTensor(inputTensor));
  chk(synDestroyTensor(outputTensor));

  chk(synFree(deviceId, inputData));
  chk(synFree(deviceId, outputData));

  chk(synReleaseDevice(deviceId));

  std::mutex synapseMtx;
  std::promise<void> readyPromise;
  std::shared_future<void> readyFuture(readyPromise.get_future());
  std::vector<std::thread> threads;

  for (unsigned i = 0; i < numThreads; ++i) {
    threads.emplace_back([&synapseMtx, numIterations = numIterations,
                          numEnqueues = numEnqueues, size = size,
                          recipe = recipe()]() {
      for (unsigned j = 0; j < numIterations; ++j) {
        synStatus status = synFail;
        uint32_t deviceId;
        {
          std::lock_guard<std::mutex> lk(synapseMtx);
          status = synAcquireDevice(&deviceId, nullptr);
        }

        if (status == synSuccess) {
          void *inputData = nullptr;
          uint64_t inputSize = sizeof(int8_t) * size;
          chk(synMalloc(deviceId, size, synMemFlags::synMemHost, &inputData));

          void *outputData = nullptr;
          uint64_t outputSize = sizeof(int8_t) * size;
          chk(synMalloc(deviceId, size, synMemFlags::synMemHost, &outputData));

          uint64_t topologyId = 0;
          {
            std::lock_guard<std::mutex> lk(synapseMtx);
            chk(synLoadRecipe(deviceId, recipe, &topologyId));
          }
          chk(synActivateTopology(deviceId, topologyId));

          for (unsigned k = 0; k < size; k++) {
            static_cast<int8_t *>(inputData)[k] = k - size / 2;
          }

          for (unsigned k = 0; k < numEnqueues; ++k) {
            memset(outputData, 0, outputSize);
            synWaitHandle handle;
            chk(synEnqueue(deviceId, inputData, inputSize, outputData,
                           outputSize, &handle));
            synWaitForEvent(deviceId, handle);
            synDestroyHandle(handle);

            for (unsigned l = 0; l < size; l++) {
              int8_t out = static_cast<int8_t *>(outputData)[l];
              ASSERT_EQ(out, std::max<int8_t>(0, l - size / 2));
            }
          }
          chk(synFree(deviceId, outputData));
          chk(synFree(deviceId, inputData));

          {
            std::lock_guard<std::mutex> lk(synapseMtx);
            chk(synReleaseDevice(deviceId));
          }
        }
      }
    });
  }

  readyPromise.set_value();

  for (auto &thread : threads) {
    thread.join();
  }

  chk(synDestroy());
}

TEST_F(Habana, IntermediateReshapeMLP) {
  chk(synInitialize());

  std::vector<float> dense(1000 * 128);
  unsigned denseDims[SYN_MAX_TENSOR_DIM] = {128, 1000};
  synTensorDescriptor denseDesc(syn_type_single, 2u, denseDims, dense.data(),
                                synMemoryHost, false, "dense");
  synTensor denseT;
  chk(synCreateTensor(&denseDesc, &denseT, false, true, false));

  std::vector<float> weights1(128 * 584);
  unsigned weights1Dims[SYN_MAX_TENSOR_DIM] = {584, 128};
  synTensorDescriptor weights1Desc(syn_type_single, 2u, weights1Dims,
                                   weights1.data(), synMemoryHost, false);
  synTensor weights1T;
  chk(synCreateTensor(&weights1Desc, &weights1T, false, false, true));

  std::vector<float> fc2(1000 * 584);
  unsigned fc2Dims[SYN_MAX_TENSOR_DIM] = {584, 1000};
  synTensorDescriptor fc2Desc(syn_type_single, 2u, fc2Dims, fc2.data(),
                              synMemoryHost, false);
  synTensor fc2T;
  chk(synCreateTensor(&fc2Desc, &fc2T, false, false, false));

  std::vector<float> reshape(1000 * 73 * 8);
  unsigned reshapeDims[SYN_MAX_TENSOR_DIM] = {8, 73, 1000};
  synTensorDescriptor reshapeDesc(syn_type_single, 3u, reshapeDims,
                                  reshape.data(), synMemoryHost, false,
                                  "reshape");
  synTensor reshapeT;
  chk(synCreateTensor(&reshapeDesc, &reshapeT, true, false, false));

  std::vector<float> weights2(584 * 128);
  unsigned weights2Dims[SYN_MAX_TENSOR_DIM] = {128, 584};
  synTensorDescriptor weights2Desc(syn_type_single, 2u, weights2Dims,
                                   weights2.data(), synMemoryHost, false);
  synTensor weights2T;
  chk(synCreateTensor(&weights2Desc, &weights2T, false, false, true));

  std::vector<float> fc3(1000 * 128);
  unsigned fc3Dims[SYN_MAX_TENSOR_DIM] = {128, 1000};
  synTensorDescriptor fc3Desc(syn_type_single, 2u, fc3Dims, fc3.data(),
                              synMemoryHost, false, "fc3");
  synTensor fc3T;
  chk(synCreateTensor(&fc3Desc, &fc3T, true, false, false));

  synTensor fc2In[] = {denseT, weights1T};
  chk(synCreateGenericNode(fc2In, &fc2T, 2, 1, nullptr, "matrix_multiply_f32",
                           "fc2", nullptr, nullptr));
  synTensor fc3In[] = {fc2T, weights2T};
  chk(synCreateGenericNode(fc3In, &fc3T, 2, 1, nullptr, "matrix_multiply_f32",
                           "fc3", nullptr, nullptr));
  chk(synCreateGenericNode(&fc2T, &reshapeT, 1, 1, nullptr, "reshape",
                           "reshape", nullptr, nullptr));

  // Compile graph.
  CompilationAttribute compileParams[1];
  compileParams[0].type = VISUALIZATION;
  compileParams[0].u32 = 1;
  chk(synCompileGraph(compileParams, 1, recipe()));

  chk(synDestroyTensor(fc3T));
  chk(synDestroyTensor(weights2T));
  chk(synDestroyTensor(reshapeT));
  chk(synDestroyTensor(fc2T));
  chk(synDestroyTensor(weights1T));
  chk(synDestroyTensor(denseT));

  uint32_t deviceId;
  uint64_t topologyId;
  synWaitHandle handle;
  EnqueueTensorInfo eti[3] = {
      {"dense", (char *)dense.data(),
       (uint32_t)(dense.size() * sizeof(dense[0]))},
      {"reshape", (char *)reshape.data(),
       (uint32_t)(reshape.size() * sizeof(reshape[0]))},
      {"fc3", (char *)fc3.data(), (uint32_t)(fc3.size() * sizeof(fc3[0]))},
  };

  chk(synAcquireDevice(&deviceId, nullptr));

  synMap(deviceId, dense.size() * sizeof(float), dense.data());
  synMap(deviceId, reshape.size() * sizeof(float), reshape.data());
  synMap(deviceId, fc3.size() * sizeof(float), fc3.data());

  chk(synLoadRecipe(deviceId, recipe(), &topologyId));
  chk(synActivateTopology(deviceId, topologyId));
  chk(synEnqueueByName(deviceId, &eti[0], 1, &eti[1], 2, &handle));
  chk(synWaitForEvent(deviceId, handle));
  synDestroyHandle(handle);

  synUnmap(deviceId, dense.data());
  synUnmap(deviceId, reshape.data());
  synUnmap(deviceId, fc3.data());

  chk(synReleaseDevice(deviceId));
  chk(synDestroy());
}

template <typename T> void fill(T &t) { std::fill(t.begin(), t.end(), 1.0); }

TEST_F(Habana, SparseLengthsSum) {
  chk(synInitialize());

  // MLP //////////////////////////////////////////////////////////////////////

  std::vector<float> x(281000);
  unsigned xDims[SYN_MAX_TENSOR_DIM] = {
      281,
      1000,
  };
  synTensorDescriptor xDesc((synDataType)4, 2, xDims, x.data(), synMemoryHost,
                            false, "x");
  synTensor xT;
  synCreateTensor(&xDesc, &xT, false, true, false);

  std::vector<float> w(44960);
  fill(w);
  unsigned wDims[SYN_MAX_TENSOR_DIM] = {
      160,
      281,
  };
  synTensorDescriptor wDesc((synDataType)4, 2, wDims, w.data(), synMemoryHost,
                            false, "w");
  synTensor wT;
  synCreateTensor(&wDesc, &wT, false, false, true);

  std::vector<float> b(160);
  fill(b);
  unsigned bDims[SYN_MAX_TENSOR_DIM] = {
      160,
  };
  synTensorDescriptor bDesc((synDataType)4, 1, bDims, b.data(), synMemoryHost,
                            false, "b");
  synTensor bT;
  synCreateTensor(&bDesc, &bT, false, false, true);

  std::vector<float> fc_dot(160000);
  unsigned fc_dotDims[SYN_MAX_TENSOR_DIM] = {
      160,
      1000,
  };
  synTensorDescriptor fc_dotDesc((synDataType)4, 2, fc_dotDims, fc_dot.data(),
                                 synMemoryHost, false, "fc_dot");
  synTensor fc_dotT;
  synCreateTensor(&fc_dotDesc, &fc_dotT, false, false, false);

  std::vector<float> fc_add_bias_bcast(160000);
  unsigned fc_add_bias_bcastDims[SYN_MAX_TENSOR_DIM] = {
      160,
      1000,
  };
  synTensorDescriptor fc_add_bias_bcastDesc(
      (synDataType)4, 2, fc_add_bias_bcastDims, fc_add_bias_bcast.data(),
      synMemoryHost, false, "fc_add_bias_bcast");
  synTensor fc_add_bias_bcastT;
  synCreateTensor(&fc_add_bias_bcastDesc, &fc_add_bias_bcastT, false, false,
                  false);

  std::vector<float> save2(160000);
  unsigned save2Dims[SYN_MAX_TENSOR_DIM] = {
      160,
      1000,
  };
  synTensorDescriptor save2Desc((synDataType)4, 2, save2Dims, save2.data(),
                                synMemoryHost, false, "save2");
  synTensor save2T;
  synCreateTensor(&save2Desc, &save2T, true, false, false);

  synTensor fc_dotInputs[] = {xT, wT};
  synCreateGenericNode(fc_dotInputs, &fc_dotT, 2, 1, nullptr,
                       "matrix_multiply_f32", "fc_dot", nullptr, nullptr);

  synCreateGenericNode(&bT, &fc_add_bias_bcastT, 1, 1, nullptr, "broadcast",
                       "fc_add_bias", nullptr, nullptr);

  synTensor fc_add_biasInputs[] = {fc_add_bias_bcastT, fc_dotT};
  synCreateGenericNode(fc_add_biasInputs, &save2T, 2, 1, nullptr, "add_f32",
                       "fc_add_bias", nullptr, nullptr);

  // SLWS //////////////////////////////////////////////////////////////////////

  std::vector<int32_t> i1(4000);
  unsigned i1Dims[SYN_MAX_TENSOR_DIM] = {
      4000,
  };
  synTensorDescriptor i1Desc((synDataType)16, 1, i1Dims, i1.data(),
                             synMemoryHost, false, "i1");
  synTensor i1T;
  synCreateTensor(&i1Desc, &i1T, false, true, false);

  std::vector<int32_t> i11(1000);
  unsigned i11Dims[SYN_MAX_TENSOR_DIM] = {
      1000,
  };
  synTensorDescriptor i11Desc((synDataType)16, 1, i11Dims, i11.data(),
                              synMemoryHost, false, "i11");
  synTensor i11T;
  synCreateTensor(&i11Desc, &i11T, false, true, false);

  std::vector<float> save1(40000);
  unsigned save1Dims[SYN_MAX_TENSOR_DIM] = {
      40,
      1000,
  };
  synTensorDescriptor save1Desc((synDataType)4, 2, save1Dims, save1.data(),
                                synMemoryHost, false, "save1");
  synTensor save1T;
  synCreateTensor(&save1Desc, &save1T, true, false, false);

  std::vector<float> data(80000000);
  fill(data);
  unsigned dataDims[SYN_MAX_TENSOR_DIM] = {
      40,
      2000000,
  };
  synTensorDescriptor dataDesc(syn_type_fixed, 2, dataDims, data.data(),
                               synMemoryHost, false, "data");
  synTensor dataT;
  synCreateTensor(&dataDesc, &dataT, false, false, true);

  std::vector<float> sb(4000000);
  unsigned sbDims[SYN_MAX_TENSOR_DIM] = {
      2,
      2000000,
  };
  synTensorDescriptor sbDesc(syn_type_single, 2, sbDims, sb.data(),
                             synMemoryHost, false, "sb");
  synTensor sbT;
  synCreateTensor(&sbDesc, &sbT, false, false, true);

  synTensor sls1Inputs[] = {dataT, i1T, i11T, sbT};
  ns_SparseLengthsSum::Params sls1Params;
  sls1Params.mode = SEPARATE_SC_ZP;
  synCreateGenericNode(sls1Inputs, &save1T, 4, 1, (void *)&sls1Params,
                       "sparse_lengths_sum_f32", "sls1", nullptr, nullptr);

  // Compile graph.
  CompilationAttribute compileParams[1];
  compileParams[0].type = VISUALIZATION;
  compileParams[0].u32 = 1;
  chk(synCompileGraph(compileParams, 1, recipe()));
}
