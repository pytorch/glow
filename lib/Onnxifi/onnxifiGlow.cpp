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

#include "Base.h"
#include "GlowOnnxifiManager.h"

#include "glow/Importer/ONNXIFIModelLoader.h"

/// Allow defining names for onnxifi implementation.
#ifndef GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER
#define GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(name) name
#endif

#define EXTERNC extern "C"

/**
 * This file contains implementation of the onnxifi interface.
 * Documentation on the functions implementing onnxifi interface in
 * this file is very shallow.
 * Please see more documentation on functions that need to be
 * implemented: https://github.com/onnx/onnx/blob/master/onnx/onnxifi.h.
 */

/// Return stable IDs of available backends on the system.
/// \param backendIDs output parameter and represents pointer to the memory
///                   where the backend IDs will be returned. If it's NULL,
///                   numBackends will be populated with the number of backends
///                   supported.
/// \param numBackends input/output parameter.
///                    As an input, it specifies the capacity allocated in the
///                    backendIDs. As an output, it specifies the number of
///                    actual available backends.
EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxGetBackendIDs)(
    onnxBackendID *backendIDs, size_t *numBackends) {
  if (!numBackends) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  auto &manager = glow::onnxifi::GlowOnnxifiManager::get();

  const size_t numBackendsCapacity = *numBackends;

#ifdef GLOW_WITH_CPU
  *numBackends = 2;

  // In case backendIDs is nullptr or does not have enough capacity just return
  // the total number of supported backends.
  if (numBackendsCapacity < *numBackends || !backendIDs) {
    return ONNXIFI_STATUS_FALLBACK;
  }

  // TODO: change concurrency level to std::thread::hardware_concurrency()
  // when Glow CPU backend can handle concurrent execution.
  // For now, limit concurrent execution to a single worker thread..
  auto *cpuBackend =
      new glow::onnxifi::BackendId(glow::BackendKind::CPU, /*id*/ 1,
                                   /*concurrency*/ 1);
  auto *interpreterBackend =
      new glow::onnxifi::BackendId(glow::BackendKind::Interpreter,
                                   /*id*/ 2, /*concurrency*/ 1);
  manager.addBackendId(cpuBackend);
  manager.addBackendId(interpreterBackend);

  backendIDs[0] = cpuBackend;
  backendIDs[1] = interpreterBackend;
#else
  *numBackends = 1;

  // In case backendIDs is nullptr or does not have enough capacity just return
  // the total number of supported backends.
  if (numBackendsCapacity < *numBackends || !backendIDs) {
    return ONNXIFI_STATUS_FALLBACK;
  }

  auto *interpreterBackend =
      new glow::onnxifi::BackendId(glow::BackendKind::Interpreter,
                                   /*id*/ 1, /*concurrency*/ 1);

  manager.addBackendId(interpreterBackend);

  backendIDs[0] = interpreterBackend;
#endif

  return ONNXIFI_STATUS_SUCCESS;
}

/// Deinitialize ONNXIFI backend ID and release associated resources.
/// Caller is responsible to release objects associated with the backend ID
/// (onnxBackend, onnxGraph, onnxEvent) before calling this function.
EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxReleaseBackendID)(
    onnxBackendID backendID) {
  auto &manager = glow::onnxifi::GlowOnnxifiManager::get();

  auto *glowBackendId = static_cast<glow::onnxifi::BackendIdPtr>(backendID);
  if (!manager.isValid(glowBackendId)) {
    return ONNXIFI_STATUS_INVALID_ID;
  }

  manager.release(glowBackendId);

  return ONNXIFI_STATUS_SUCCESS;
}

static onnxStatus setBackendInfoString(void *infoValue, size_t *infoValueSize,
                                       const char *str) {
  size_t len = strlen(str) + 1;
  if (!infoValue || *infoValueSize < len) {
    *infoValueSize = len;
    return ONNXIFI_STATUS_FALLBACK;
  }

  strncpy((char *)infoValue, str, len);
  *infoValueSize = len;
  return ONNXIFI_STATUS_SUCCESS;
}

static onnxStatus setBackendInfoUInt64(void *infoValue, size_t *infoValueSize,
                                       uint64_t value) {
  if (!infoValue || *infoValueSize < sizeof(uint64_t)) {
    *infoValueSize = sizeof(uint64_t);
    return ONNXIFI_STATUS_FALLBACK;
  }

  *(uint64_t *)(infoValue) = value;
  *infoValueSize = sizeof(uint64_t);
  return ONNXIFI_STATUS_SUCCESS;
}

/// Query high-level information about the backend and its target device.
EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxGetBackendInfo)(
    onnxBackendID backendID, onnxBackendInfo infoType, void *infoValue,
    size_t *infoValueSize) {
  if (!infoValueSize) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  if (!infoValue) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  auto &manager = glow::onnxifi::GlowOnnxifiManager::get();

  auto *glowBackendId = static_cast<glow::onnxifi::BackendIdPtr>(backendID);
  if (!manager.isValid(glowBackendId)) {
    return ONNXIFI_STATUS_INVALID_ID;
  }

  // TODO: support more info type values. Here is the minimal required
  // subset of info types.
  switch (infoType) {
  case ONNXIFI_BACKEND_NAME:
    return setBackendInfoString(infoValue, infoValueSize, "Glow");
  case ONNXIFI_BACKEND_VENDOR:
    return setBackendInfoString(infoValue, infoValueSize, "PyTorch");
  case ONNXIFI_BACKEND_VERSION:
    return setBackendInfoString(infoValue, infoValueSize, "1.0.0");
  case ONNXIFI_BACKEND_DEVICE:
    return setBackendInfoString(infoValue, infoValueSize, "Glow");
  case ONNXIFI_BACKEND_MEMORY_TYPES:
    return setBackendInfoUInt64(infoValue, infoValueSize,
                                ONNXIFI_MEMORY_TYPE_CPU);
  case ONNXIFI_BACKEND_SYNCHRONIZATION_TYPES:
    return setBackendInfoUInt64(infoValue, infoValueSize,
                                ONNXIFI_SYNCHRONIZATION_EVENT);
  default:
    return ONNXIFI_STATUS_UNSUPPORTED_PROPERTY;
  }
}

/// Query if an ONNX model graph is compatible with the backend.
EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxGetBackendCompatibility)(
    onnxBackendID backendID, size_t onnxModelSize, const void *onnxModel) {
  if (!onnxModel) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  if (!onnxModelSize) {
    return ONNXIFI_STATUS_INVALID_SIZE;
  }

  auto &manager = glow::onnxifi::GlowOnnxifiManager::get();

  auto *glowBackendId = static_cast<glow::onnxifi::BackendIdPtr>(backendID);
  if (!manager.isValid(glowBackendId)) {
    return ONNXIFI_STATUS_INVALID_ID;
  }

  return glowBackendId->checkGraphCompatibility(onnxModel, onnxModelSize);
}

/// Initialize an ONNXIFI backend.
EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxInitBackend)(
    onnxBackendID backendID, const uint64_t *auxpropertiesList,
    onnxBackend *backend) {
  if (!backend) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  auto &manager = glow::onnxifi::GlowOnnxifiManager::get();

  auto *glowBackendId = static_cast<glow::onnxifi::BackendIdPtr>(backendID);
  if (!manager.isValid(glowBackendId)) {
    return ONNXIFI_STATUS_INVALID_ID;
  }

  *backend = manager.createBackend(glowBackendId);

  return ONNXIFI_STATUS_SUCCESS;
}

/// Deinitialize an ONNXIFI backend and release associated resources.
EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxReleaseBackend)(onnxBackend backend) {
  auto &manager = glow::onnxifi::GlowOnnxifiManager::get();

  auto *glowBackend = static_cast<glow::onnxifi::BackendPtr>(backend);
  if (!manager.isValid(glowBackend)) {
    return ONNXIFI_STATUS_INVALID_BACKEND;
  }

  manager.release(glowBackend);

  return ONNXIFI_STATUS_SUCCESS;
}

/// Initialize a single-shot ONNXIFI event.
EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxInitEvent)(onnxBackend backend,
                                                     onnxEvent *event) {
  if (!event) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  auto &manager = glow::onnxifi::GlowOnnxifiManager::get();

  auto *glowBackend = static_cast<glow::onnxifi::BackendPtr>(backend);
  if (!manager.isValid(glowBackend)) {
    return ONNXIFI_STATUS_INVALID_BACKEND;
  }

  *event = manager.createEvent();

  return ONNXIFI_STATUS_SUCCESS;
}

/// Change the state of the ONNXIFI event \p event to signalled.
EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxSignalEvent)(onnxEvent event) {
  auto &manager = glow::onnxifi::GlowOnnxifiManager::get();

  auto *glowEvent = static_cast<glow::onnxifi::EventPtr>(event);
  if (!manager.isValid(glowEvent)) {
    return ONNXIFI_STATUS_INVALID_EVENT;
  }

  if (!glowEvent->signal()) {
    return ONNXIFI_STATUS_INVALID_STATE;
  }

  return ONNXIFI_STATUS_SUCCESS;
}

/// Wait until an ONNXIFI event is signalled.
EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxWaitEvent)(onnxEvent event) {
  auto &manager = glow::onnxifi::GlowOnnxifiManager::get();

  auto *glowEvent = static_cast<glow::onnxifi::EventPtr>(event);
  if (!manager.isValid(glowEvent)) {
    return ONNXIFI_STATUS_INVALID_EVENT;
  }

  glowEvent->wait();

  return ONNXIFI_STATUS_SUCCESS;
}

/// Query ONNXIFI event state without blocking.
EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxGetEventState)(
    onnxEvent event, onnxEventState *state) {
  if (!state) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  *state = ONNXIFI_EVENT_STATE_INVALID;

  auto &manager = glow::onnxifi::GlowOnnxifiManager::get();

  auto *glowEvent = static_cast<glow::onnxifi::EventPtr>(event);
  if (!manager.isValid(glowEvent)) {
    return ONNXIFI_STATUS_INVALID_EVENT;
  }

  *state = glowEvent->isSignalled() ? ONNXIFI_EVENT_STATE_SIGNALLED
                                    : ONNXIFI_EVENT_STATE_NONSIGNALLED;
  return ONNXIFI_STATUS_SUCCESS;
}

/// Deinitialize an ONNXIFI event and release associated resources.
EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxReleaseEvent)(onnxEvent event) {
  auto &manager = glow::onnxifi::GlowOnnxifiManager::get();

  auto *glowEvent = static_cast<glow::onnxifi::EventPtr>(event);
  if (!manager.isValid(glowEvent)) {
    return ONNXIFI_STATUS_INVALID_EVENT;
  }

  manager.release(glowEvent);

  return ONNXIFI_STATUS_SUCCESS;
}

/// Parse an ONNXIFI graph and convert it for a particular backend.
EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxInitGraph)(
    onnxBackend backend, const uint64_t *auxPropertiesList,
    size_t onnxModelSize, const void *onnxModel, uint32_t weightsCount,
    const onnxTensorDescriptorV1 *weightDescriptors, onnxGraph *graph) {
  if (!onnxModel || (!weightDescriptors && weightsCount) || !graph) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  if (!onnxModelSize) {
    return ONNXIFI_STATUS_INVALID_SIZE;
  }

  auto &manager = glow::onnxifi::GlowOnnxifiManager::get();

  auto *glowBackend = static_cast<glow::onnxifi::BackendPtr>(backend);
  if (!manager.isValid(glowBackend)) {
    return ONNXIFI_STATUS_INVALID_BACKEND;
  }

  auto *glowGraph = manager.createGraph(glowBackend);
  auto ret = glowGraph->initGraph(onnxModel, onnxModelSize, weightsCount,
                                  weightDescriptors);
  if (ret != ONNXIFI_STATUS_SUCCESS) {
    manager.release(glowGraph);
    return ret;
  }

  *graph = glowGraph;
  return ONNXIFI_STATUS_SUCCESS;
}

static bool verifyDescriptors(uint32_t count,
                              const onnxTensorDescriptorV1 *descriptors) {
  for (unsigned i = 0; i < count; i++) {
    const auto &descriptor = descriptors[i];
    if (descriptor.tag != ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1) {
      return ONNXIFI_STATUS_UNSUPPORTED_TAG;
    }

    if (descriptor.memoryType != ONNXIFI_MEMORY_TYPE_CPU) {
      return ONNXIFI_STATUS_INVALID_MEMORY_TYPE;
    }

    if (!descriptor.buffer) {
      return ONNXIFI_STATUS_INVALID_MEMORY_LOCATION;
    }
  }

  return ONNXIFI_STATUS_SUCCESS;
}

/// Binds inputs and outputs of an ONNXIFI graph to specific addresses.
EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxSetGraphIO)(
    onnxGraph graph, uint32_t inputsCount,
    const onnxTensorDescriptorV1 *inputDescriptors, uint32_t outputsCount,
    const onnxTensorDescriptorV1 *outputDescriptors) {
  if ((!inputDescriptors && inputsCount) || !outputDescriptors) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  auto &manager = glow::onnxifi::GlowOnnxifiManager::get();

  auto *glowGraph = static_cast<glow::onnxifi::GraphPtr>(graph);
  if (!manager.isValid(glowGraph)) {
    return ONNXIFI_STATUS_INVALID_GRAPH;
  }

  auto inputStatus = verifyDescriptors(inputsCount, inputDescriptors);
  if (inputStatus != ONNXIFI_STATUS_SUCCESS) {
    return inputStatus;
  }

  auto outputStatus = verifyDescriptors(outputsCount, outputDescriptors);
  if (outputStatus != ONNXIFI_STATUS_SUCCESS) {
    return outputStatus;
  }

  return glowGraph->setIO(inputsCount, inputDescriptors, outputsCount,
                          outputDescriptors);
}

/// Asynchronously execute operations in an ONNXIFI graph using pre-specified
/// locations for inputs and outputs.
EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxRunGraph)(
    onnxGraph graph, const onnxMemoryFenceV1 *inputFence,
    onnxMemoryFenceV1 *outputFence) {
  if (!inputFence || !outputFence) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  auto &manager = glow::onnxifi::GlowOnnxifiManager::get();

  auto *glowGraph = static_cast<glow::onnxifi::GraphPtr>(graph);
  if (!manager.isValid(glowGraph)) {
    return ONNXIFI_STATUS_INVALID_GRAPH;
  }

  if (inputFence->type != ONNXIFI_SYNCHRONIZATION_EVENT ||
      inputFence->tag != ONNXIFI_TAG_MEMORY_FENCE_V1 ||
      outputFence->type != ONNXIFI_SYNCHRONIZATION_EVENT ||
      outputFence->tag != ONNXIFI_TAG_MEMORY_FENCE_V1) {
    return ONNXIFI_STATUS_UNSUPPORTED_TAG;
  }

  auto initStatus = GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxInitEvent)(
      glowGraph->backend(), &outputFence->event);
  if (initStatus != ONNXIFI_STATUS_SUCCESS) {
    return initStatus;
  }

  auto *inputEvent = static_cast<glow::onnxifi::EventPtr>(inputFence->event);
  auto *outputEvent = static_cast<glow::onnxifi::EventPtr>(outputFence->event);

  glowGraph->runAsync(inputEvent, outputEvent);
  return ONNXIFI_STATUS_SUCCESS;
}

/// Deinitialize an ONNXIFI graph and release associated resources.
/// It blocks until all in-flight inference operations complete.
EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxReleaseGraph)(onnxGraph graph) {
  auto &manager = glow::onnxifi::GlowOnnxifiManager::get();

  auto *glowGraph = static_cast<glow::onnxifi::GraphPtr>(graph);
  if (!manager.isValid(glowGraph)) {
    return ONNXIFI_STATUS_INVALID_GRAPH;
  }

  manager.release(glowGraph);

  return ONNXIFI_STATUS_SUCCESS;
}
