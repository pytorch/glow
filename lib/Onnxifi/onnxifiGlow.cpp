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

#include "glow/Importer/ONNXIFILoader.h"

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
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxGetBackendIDs(onnxBackendID *backendIDs, size_t *numBackends) {
  if (!numBackends) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  // In case backendIDs are not set, just return total number of supported
  // backends.
  if (!backendIDs) {
    *numBackends = 1;
    return ONNXIFI_STATUS_FALLBACK;
  }

  // Glow represents a single backend.
  *numBackends = 1;
#ifdef GLOW_WITH_CPU
  backendIDs[0] = new glow::onnxifi::BackendId(glow::BackendKind::CPU, 1);
#else
  backendIDs[0] =
      new glow::onnxifi::BackendId(glow::BackendKind::Interpreter, 1);
#endif

  return ONNXIFI_STATUS_SUCCESS;
}

/// Deinitialize ONNXIFI backend ID and release associated resources.
/// Caller is responsible to release objects associated with the backend ID
/// (onnxBackend, onnxGraph, onnxEvent) before calling this function.
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxReleaseBackendID(onnxBackendID backendID) {
  auto *backendId = static_cast<glow::onnxifi::BackendIdPtr>(backendID);
  if (!backendID) {
    return ONNXIFI_STATUS_INVALID_ID;
  }

  delete backendId;
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
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxGetBackendInfo(onnxBackendID backendID, onnxBackendInfo infoType,
                   void *infoValue, size_t *infoValueSize) {
  if (!infoValueSize) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  auto *glowBackendId = static_cast<glow::onnxifi::BackendIdPtr>(backendID);
  if (!glowBackendId) {
    return ONNXIFI_STATUS_INVALID_POINTER;
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
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxGetBackendCompatibility(onnxBackendID backendID, size_t onnxModelSize,
                            const void *onnxModel) {
  if (!onnxModel) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  if (!onnxModelSize) {
    return ONNXIFI_STATUS_INVALID_SIZE;
  }

  auto *glowBackendId = static_cast<glow::onnxifi::BackendIdPtr>(backendID);
  if (!glowBackendId) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  std::vector<std::pair<glow::Kinded::Kind, glow::ElemKind>> operations =
      glow::onnxifi::ModelLoader::parseOperator(onnxModel, onnxModelSize);

  // TODO: Make better error reporting.
  if (operations.empty()) {
    return ONNXIFI_STATUS_UNSUPPORTED_OPERATOR;
  }

  // Make sure that the backend itself is capable of executing
  // the operation.
  for(const auto &op : operations) {
    if (!glowBackendId->isOpSupported(op.first, op.second)) {
      return ONNXIFI_STATUS_UNSUPPORTED_OPERATOR;
    }
  }

  return ONNXIFI_STATUS_SUCCESS;
}

/// Initialize an ONNXIFI backend.
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxInitBackend(onnxBackendID backendID, const uint64_t *auxpropertiesList,
                onnxBackend *backend) {
  if (!backend) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  auto *backendId = static_cast<glow::onnxifi::BackendIdPtr>(backendID);
  if (!backendId) {
    return ONNXIFI_STATUS_INVALID_ID;
  }

  auto *glowBackend = new glow::onnxifi::Backend(backendId);
  *backend = glowBackend;

  return ONNXIFI_STATUS_SUCCESS;
}

/// Deinitialize an ONNXIFI backend and release associated resources.
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxReleaseBackend(onnxBackend backend) {
  auto *glowBackend = static_cast<glow::onnxifi::BackendPtr>(backend);
  if (!glowBackend) {
    return ONNXIFI_STATUS_INVALID_BACKEND;
  }

  delete glowBackend;

  return ONNXIFI_STATUS_SUCCESS;
}

/// Initialize a single-shot ONNXIFI event.
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxInitEvent(onnxBackend backend, onnxEvent *event) {
  if (!event) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  auto *glowBackend = static_cast<glow::onnxifi::BackendPtr>(backend);
  if (!glowBackend) {
    return ONNXIFI_STATUS_INVALID_BACKEND;
  }

  *event = new glow::onnxifi::Event();
  return ONNXIFI_STATUS_SUCCESS;
}

/// Change the state of the ONNXIFI event \p event to signalled.
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxSignalEvent(onnxEvent event) {
  auto *glowEvent = static_cast<glow::onnxifi::EventPtr>(event);
  if (!event) {
    return ONNXIFI_STATUS_INVALID_EVENT;
  }

  if (!glowEvent->signal()) {
    return ONNXIFI_STATUS_INVALID_STATE;
  }

  return ONNXIFI_STATUS_SUCCESS;
}

/// Wait until an ONNXIFI event is signalled.
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxWaitEvent(onnxEvent event) {
  auto *glowEvent = static_cast<glow::onnxifi::EventPtr>(event);
  if (!glowEvent) {
    return ONNXIFI_STATUS_INVALID_EVENT;
  }

  glowEvent->wait();

  return ONNXIFI_STATUS_SUCCESS;
}

/// Query ONNXIFI event state without blocking.
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxGetEventState(onnxEvent event, onnxEventState *state) {
  if (!state) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  *state = ONNXIFI_EVENT_STATE_INVALID;

  auto *glowEvent = static_cast<glow::onnxifi::EventPtr>(event);
  if (!glowEvent) {
    return ONNXIFI_STATUS_INVALID_EVENT;
  }

  *state = glowEvent->isSignalled() ? ONNXIFI_EVENT_STATE_SIGNALLED
                                    : ONNXIFI_EVENT_STATE_NONSIGNALLED;
  return ONNXIFI_STATUS_SUCCESS;
}

/// Deinitialize an ONNXIFI event and release associated resources.
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxReleaseEvent(onnxEvent event) {
  auto *glowEvent = static_cast<glow::onnxifi::EventPtr>(event);
  if (!glowEvent) {
    return ONNXIFI_STATUS_INVALID_EVENT;
  }

  delete glowEvent;

  return ONNXIFI_STATUS_SUCCESS;
}

/// Parse an ONNXIFI graph and convert it for a particular backend.
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI onnxInitGraph(
    onnxBackend backend, const uint64_t *auxPropertiesList,
    size_t onnxModelSize, const void *onnxModel, uint32_t weightsCount,
    const onnxTensorDescriptorV1 *weightDescriptors, onnxGraph *graph) {
  if (!onnxModel || (!weightDescriptors && weightsCount) || !graph) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  if (!onnxModelSize) {
    return ONNXIFI_STATUS_INVALID_SIZE;
  }

  auto *glowBackend = static_cast<glow::onnxifi::BackendPtr>(backend);
  if (!glowBackend) {
    return ONNXIFI_STATUS_INVALID_BACKEND;
  }

  auto *glowGraph = new glow::onnxifi::Graph(glowBackend);
  auto ret = glowGraph->initGraph(onnxModel, onnxModelSize, weightsCount,
                                  weightDescriptors);
  if (ret != ONNXIFI_STATUS_SUCCESS) {
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
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI onnxSetGraphIO(
    onnxGraph graph, uint32_t inputsCount,
    const onnxTensorDescriptorV1 *inputDescriptors, uint32_t outputsCount,
    const onnxTensorDescriptorV1 *outputDescriptors) {
  if ((!inputDescriptors && inputsCount) || !outputDescriptors) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  auto *glowGraph = static_cast<glow::onnxifi::GraphPtr>(graph);
  if (!glowGraph) {
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
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxRunGraph(onnxGraph graph, const onnxMemoryFenceV1 *inputFence,
             onnxMemoryFenceV1 *outputFence) {
  if (!inputFence || !outputFence) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  auto *glowGraph = static_cast<glow::onnxifi::GraphPtr>(graph);
  if (!glowGraph) {
    return ONNXIFI_STATUS_INVALID_GRAPH;
  }

  if (inputFence->type != ONNXIFI_SYNCHRONIZATION_EVENT ||
      inputFence->tag != ONNXIFI_TAG_MEMORY_FENCE_V1 ||
      outputFence->type != ONNXIFI_SYNCHRONIZATION_EVENT ||
      outputFence->tag != ONNXIFI_TAG_MEMORY_FENCE_V1) {
    return ONNXIFI_STATUS_UNSUPPORTED_TAG;
  }

  auto initStatus = onnxInitEvent(glowGraph->backend(), &outputFence->event);
  if (initStatus != ONNXIFI_STATUS_SUCCESS) {
    return initStatus;
  }

  // Wait for all inputs to be ready.
  auto waitStatus = onnxWaitEvent(inputFence->event);
  if (waitStatus != ONNXIFI_STATUS_SUCCESS) {
    return waitStatus;
  }

  // TODO: Implement async graph run procedure. For now run that sync.
  glowGraph->run();

  auto signalStatus = onnxSignalEvent(outputFence->event);
  if (signalStatus != ONNXIFI_STATUS_SUCCESS) {
    return signalStatus;
  }

  return ONNXIFI_STATUS_SUCCESS;
}

/// Deinitialize an ONNXIFI graph and release associated resources.
/// It blocks until all in-flight inference operations complete.
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxReleaseGraph(onnxGraph graph) {
  auto *glowGraph = static_cast<glow::onnxifi::GraphPtr>(graph);
  if (!glowGraph) {
    return ONNXIFI_STATUS_INVALID_GRAPH;
  }

  delete glowGraph;

  return ONNXIFI_STATUS_SUCCESS;
}
