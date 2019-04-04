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

#include "SynapseShim.h"

#include <assert.h>
#include <dlfcn.h>
#include <type_traits>

namespace {
class SynapseLibrary {
#define APIS                                                                   \
  API(Initialize)                                                              \
  API(LoadRecipe)                                                              \
  API(GetMemInfo)                                                              \
  API(ActivateTopology)                                                        \
  API(Enqueue)                                                                 \
  API(EnqueueByName)                                                           \
  API(DestroyHandle)                                                           \
  API(AcquireDevice)                                                           \
  API(ReleaseDevice)                                                           \
  API(Destroy)                                                                 \
  API(CreateGenericNode)                                                       \
  API(FullyConnected)                                                          \
  API(SpatialConvolution)                                                      \
  API(CompileGraph)                                                            \
  API(Malloc)                                                                  \
  API(Free)                                                                    \
  API(CreateTensor)                                                            \
  API(DestroyTensor)                                                           \
  API(GetTensorsName)                                                          \
  API(GetIOTensorsAmount)                                                      \
  API(RetrieveTensorInfoByName)                                                \
  API(CreateGraph)                                                             \
  API(DestroyGraph)                                                            \
  API(UnloadTopology)                                                          \
  API(GetActiveTopologyID)                                                     \
  API(WaitForEvent)                                                            \
  API(Map)                                                                     \
  API(Unmap)

private:
  void *handle_;

public:
#define API(fn) using fn##Ptr = std::add_pointer<decltype(syn##fn)>::type;
  APIS
#undef API

  SynapseLibrary() {
    handle_ = dlopen("/usr/lib/habanalabs/libSynapse.so", RTLD_LAZY);
    if (!handle_) {
      fprintf(stderr, "dlopen failed: %s\n", dlerror());
      fprintf(stderr, "Did you remember to set "
                      "LD_LIBRARY_PATH=/usr/lib/habanalabs?\n");
      abort();
    }
#define API(fn)                                                                \
  fn = (fn##Ptr)dlsym(handle_, "syn" #fn);                                     \
  assert(fn);
    APIS
#undef API
  }

  ~SynapseLibrary() { dlclose(handle_); }

#define API(fn) fn##Ptr fn;
  APIS
#undef API
};
} // namespace

static SynapseLibrary &syn() {
  static SynapseLibrary libSynapse;
  return libSynapse;
}

synStatus SYN_API_CALL synInitialize() { return syn().Initialize(); }

synStatus SYN_API_CALL synDestroy() { return syn().Destroy(); }

synStatus SYN_API_CALL synAcquireDevice(uint32_t *pDeviceId,
                                        const char *pciBus) {
  return syn().AcquireDevice(pDeviceId, pciBus);
}

synStatus SYN_API_CALL synReleaseDevice(uint32_t deviceId) {
  return syn().ReleaseDevice(deviceId);
}

synStatus SYN_API_CALL synGetMemInfo(uint32_t deviceId, uint64_t *freeMemory,
                                     uint64_t *totalMemory) {
  return syn().GetMemInfo(deviceId, freeMemory, totalMemory);
}

synStatus SYN_API_CALL synUnloadTopology(uint32_t deviceId,
                                         uint64_t topologyId) {
  return syn().UnloadTopology(deviceId, topologyId);
}

synStatus SYN_API_CALL synMalloc(int32_t deviceId, uint64_t size,
                                 uint32_t flags, void **buffer,
                                 uint64_t reqVAAddress) {
  return syn().Malloc(deviceId, size, flags, buffer, reqVAAddress);
}

synStatus SYN_API_CALL synFree(int32_t deviceId, void *buffer, uint32_t flags) {
  return syn().Free(deviceId, buffer, flags);
}

synStatus SYN_API_CALL synCreateTensor(synTensorDescriptor *pDescriptor,
                                       synTensor *tensor, bool isOutput,
                                       bool isInput, bool isStaticParam) {
  return syn().CreateTensor(pDescriptor, tensor, isOutput, isInput,
                            isStaticParam);
}

synStatus SYN_API_CALL synDestroyTensor(synTensor tensor) {
  return syn().DestroyTensor(tensor);
}

synStatus SYN_API_CALL synCreateGenericNode(
    synTensor *inputs, synTensor *outputs, uint32_t sizeInputs,
    uint32_t sizeOutputs, void *userParams, const char *guid, const char *name,
    const char **inputLayouts, const char **outputLayouts) {
  return syn().CreateGenericNode(inputs, outputs, sizeInputs, sizeOutputs,
                                 userParams, guid, name, inputLayouts,
                                 outputLayouts);
}

synStatus SYN_API_CALL synSpatialConvolution(synTensor IFM, synTensor weights,
                                             synTensor bias, synTensor OFM,
                                             synTensor cin,
                                             const synConvolutionParams &params,
                                             const char *name) {
  return syn().SpatialConvolution(IFM, weights, bias, OFM, cin, params, name);
}

synStatus SYN_API_CALL synFullyConnected(synTensor IFM, synTensor weights,
                                         synTensor bias, synTensor OFM,
                                         const synFCParams &params,
                                         const char *name) {
  return syn().FullyConnected(IFM, weights, bias, OFM, params, name);
}

synStatus SYN_API_CALL synCompileGraph(CompilationAttribute *compileParams,
                                       uint32_t sizeCompileParams,
                                       const char *outputFile,
                                       const char *buildLog) {
  return syn().CompileGraph(compileParams, sizeCompileParams, outputFile,
                            buildLog);
}

synStatus SYN_API_CALL synCreateGraph(synDeviceType deviceType) {
  return syn().CreateGraph(deviceType);
}

synStatus SYN_API_CALL synDestroyGraph() { return syn().DestroyGraph(); }

synStatus SYN_API_CALL synLoadRecipe(const uint32_t deviceId,
                                     const char *fileName,
                                     uint64_t *pTopologyId) {
  return syn().LoadRecipe(deviceId, fileName, pTopologyId);
}

synStatus SYN_API_CALL synActivateTopology(const uint32_t deviceId,
                                           const uint64_t topologyId) {
  return syn().ActivateTopology(deviceId, topologyId);
}

synStatus SYN_API_CALL synEnqueue(uint32_t deviceId, void *inputPtr,
                                  uint32_t inputBufferSize, void *outputPtr,
                                  uint32_t outputBufferSize,
                                  synWaitHandle *handle) {
  return syn().Enqueue(deviceId, inputPtr, inputBufferSize, outputPtr,
                       outputBufferSize, handle);
}

synStatus SYN_API_CALL synEnqueueByName(
    uint32_t deviceId, const EnqueueTensorInfo *enqueueInputTensorsInfo,
    const uint32_t numOfInputs,
    const EnqueueTensorInfo *enqueueOutputTensorsInfo,
    const uint32_t numOfOutputs, synWaitHandle *handle) {
  return syn().EnqueueByName(deviceId, enqueueInputTensorsInfo, numOfInputs,
                             enqueueOutputTensorsInfo, numOfOutputs, handle);
}

synStatus synWaitForEvent(uint32_t deviceId, synWaitHandle handle) {
  return syn().WaitForEvent(deviceId, handle);
}

void synDestroyHandle(synWaitHandle handle) {
  return syn().DestroyHandle(handle);
}

synStatus SYN_API_CALL synGetIOTensorsAmount(const uint32_t deviceId,
                                             const uint64_t topologyId,
                                             uint32_t &numOfInputs,
                                             uint32_t &numOfOutputs,
                                             uint32_t &numOfIntermediates) {
  return syn().GetIOTensorsAmount(deviceId, topologyId, numOfInputs,
                                  numOfOutputs, numOfIntermediates);
}

synStatus SYN_API_CALL
synGetTensorsName(uint32_t deviceId, const uint64_t topologyId,
                  char inputTensorsName[][ENQUEUE_TENSOR_NAME_MAX_SIZE],
                  const uint32_t numOfInputs,
                  char outputTensorsName[][ENQUEUE_TENSOR_NAME_MAX_SIZE],
                  const uint32_t numOfOutputs,
                  char intermediatesTensorsName[][ENQUEUE_TENSOR_NAME_MAX_SIZE],
                  const uint32_t numOfIntermediates) {
  return syn().GetTensorsName(deviceId, topologyId, inputTensorsName,
                              numOfInputs, outputTensorsName, numOfOutputs,
                              intermediatesTensorsName, numOfIntermediates);
}

synStatus SYN_API_CALL synRetrieveTensorInfoByName(
    const uint32_t deviceId, const uint64_t topologyId,
    const uint32_t numOfTensors, TensorMetadataInfo *tensorsMetadataInfo) {
  return syn().RetrieveTensorInfoByName(deviceId, topologyId, numOfTensors,
                                        tensorsMetadataInfo);
}

synStatus SYN_API_CALL synGetActiveTopologyID(const uint32_t deviceId,
                                              uint64_t *pTopologyId) {
  return syn().GetActiveTopologyID(deviceId, pTopologyId);
}

synStatus SYN_API_CALL synMap(int32_t deviceId, uint64_t size, void *buffer,
                              uint64_t reqVAAddress) {
  return syn().Map(deviceId, size, buffer, reqVAAddress);
}

synStatus SYN_API_CALL synUnmap(int32_t deviceId, void *buffer) {
  return syn().Unmap(deviceId, buffer);
}
