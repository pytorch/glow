/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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
#include "folly/String.h"
#include "llvm/Support/CommandLine.h"

#include "glow/Flags/Flags.h"
#include "glow/Importer/ONNXIFIModelLoader.h"
#include "glow/Runtime/RequestData.h"

#include <cstring>
#include <glog/logging.h>

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
 * implemented: https://github.com/houseroad/foxi/blob/master/foxi/onnxifi.h.
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

  using namespace glow::runtime;
  using namespace glow::onnxifi;
  const bool withCPU = DeviceManager::numDevices("CPU") > 0;
  const bool withHabana = DeviceManager::numDevices("Habana") > 0;
  const bool withNNPI = DeviceManager::numDevices("NNPI") > 0;
#ifdef GLOW_EXTRABACKEND
#define V(name) with##name
#define makeVar(name) V(name)

#define Q(name) #name
#define makeQuote(name) Q(name)
  const bool makeVar(GLOW_EXTRABACKEND) =
      DeviceManager::numDevices(makeQuote(GLOW_EXTRABACKEND)) > 0;
#endif

  // Only return quantization backend if GLOW_DUMP_PROFILE.
  if (getenv("GLOW_DUMP_PROFILE")) {
    *numBackends = 1;

    // In case backendIDs is nullptr or does not have enough capacity just
    // return the total number of supported backends.
    if (numBackendsCapacity < *numBackends || !backendIDs) {
      return ONNXIFI_STATUS_FALLBACK;
    }

    auto backendName = glow::onnxifi::flags::BackendName.empty()
                           ? "Interpreter"
                           : glow::onnxifi::flags::BackendName;
    LOG(INFO) << "ONNXIFI: Executing on " << backendName << " Glow backend";
    auto *quantizationBackendC2 =
        manager.createBackend(backendName,
                              /*useOnnx*/ false, /*forQuantization*/ true);
    backendIDs[0] = quantizationBackendC2;
  } else {
    *numBackends = 2;

    auto backendName = glow::onnxifi::flags::BackendName;

    if (backendName.empty()) {
      if (withNNPI) {
        backendName = "NNPI";
      } else if (withHabana) {
        backendName = "Habana";
#ifdef GLOW_EXTRABACKEND
      } else if (makeVar(GLOW_EXTRABACKEND)) {
        backendName = makeQuote(GLOW_EXTRABACKEND);
#undef V
#undef makeVar
#undef Q
#undef makeQuote
#endif
      } else if (withCPU) {
        backendName = "CPU";
      } else {
        backendName = "Interpreter";
      }
    }

    // In case backendIDs is nullptr or does not have enough capacity just
    // return the total number of supported backends.
    if (numBackendsCapacity < *numBackends || !backendIDs) {
      return ONNXIFI_STATUS_FALLBACK;
    }

    LOG(INFO) << "ONNXIFI: Executing on " << backendName << " Glow backend";

    backendIDs[0] = manager.createBackend(backendName,
                                          /*useOnnx*/ false);
    backendIDs[1] = manager.createBackend(backendName,
                                          /*useOnnx*/ true);
  }

  return ONNXIFI_STATUS_SUCCESS;
}

/// Deinitialize ONNXIFI backend ID and release associated resources.
/// Caller is responsible to release objects associated with the backend ID
/// (onnxBackend, onnxGraph, onnxEvent) before calling this function.
EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxReleaseBackendID)(
    onnxBackendID backendID) {
  auto &manager = glow::onnxifi::GlowOnnxifiManager::get();

  auto *glowBackend = static_cast<glow::onnxifi::BackendPtr>(backendID);
  if (!manager.isValid(glowBackend)) {
    return ONNXIFI_STATUS_INVALID_ID;
  }

  manager.release(glowBackend);

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

  auto *glowBackend = static_cast<glow::onnxifi::BackendPtr>(backendID);
  if (!manager.isValid(glowBackend)) {
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
    return setBackendInfoString(infoValue, infoValueSize,
                                glowBackend->getUseOnnx() ? "Glow Onnx"
                                                          : "Glow Caffe2");
  case ONNXIFI_BACKEND_MEMORY_TYPES:
    return setBackendInfoUInt64(infoValue, infoValueSize,
                                ONNXIFI_MEMORY_TYPE_CPU);
  case ONNXIFI_BACKEND_SYNCHRONIZATION_TYPES:
    return setBackendInfoUInt64(infoValue, infoValueSize,
                                ONNXIFI_SYNCHRONIZATION_EVENT);
  case ONNXIFI_BACKEND_EXTENSIONS:
    return setBackendInfoString(
        infoValue, infoValueSize,
        "onnxSetIOAndRunGraphFunction onnxWaitEventForFunction "
        "onnxReleaseTraceEventsFunction onnxGetCurrentBatchSizeFunction");
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

  auto *glowBackend = static_cast<glow::onnxifi::BackendPtr>(backendID);
  if (!manager.isValid(glowBackend)) {
    return ONNXIFI_STATUS_INVALID_ID;
  }

  return glowBackend->checkGraphCompatibility(onnxModel, onnxModelSize);
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

  auto *glowBackend = static_cast<glow::onnxifi::BackendPtr>(backendID);
  if (!manager.isValid(glowBackend)) {
    return ONNXIFI_STATUS_INVALID_ID;
  }

  *backend = glowBackend;

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

  if (!glowEvent->signal(ONNXIFI_STATUS_SUCCESS)) {
    return ONNXIFI_STATUS_INVALID_STATE;
  }

  return ONNXIFI_STATUS_SUCCESS;
}

/// Wait until an ONNXIFI \p event is signalled.
EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxWaitEvent)(onnxEvent event) {
  auto &manager = glow::onnxifi::GlowOnnxifiManager::get();

  auto *glowEvent = static_cast<glow::onnxifi::EventPtr>(event);
  if (!manager.isValid(glowEvent)) {
    return ONNXIFI_STATUS_INVALID_EVENT;
  }

  return glowEvent->wait();
}

/// Wait until an ONNXIFI \p event is signalled or until \p timeoutMs
/// milliseconds have elapsed. If \p timeoutMs is 0 then wait fallback to
/// waiting indefinitely for the event to be signalled.
EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxWaitEventFor)(
    onnxEvent event, uint32_t timeoutMs, onnxEventState *eventState,
    onnxStatus *eventStatus, char *outMessage, size_t *outMessageLength) {
  size_t maxMessageLength = *outMessageLength - 1;
  *outMessageLength = 0;

  auto &manager = glow::onnxifi::GlowOnnxifiManager::get();

  if (!eventState || !eventStatus) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  auto *glowEvent = static_cast<glow::onnxifi::EventPtr>(event);
  if (!manager.isValid(glowEvent)) {
    return ONNXIFI_STATUS_INVALID_EVENT;
  }

  if (timeoutMs == 0) {
    auto res = glowEvent->wait();
    *eventState = ONNXIFI_EVENT_STATE_SIGNALLED;
    *eventStatus = res;
  } else {
    auto resPair = glowEvent->waitFor(timeoutMs);
    if (resPair.first) {
      *eventState = ONNXIFI_EVENT_STATE_SIGNALLED;
      *eventStatus = resPair.second;
    } else {
      *eventState = ONNXIFI_EVENT_STATE_NONSIGNALLED;
    }
  }

  const auto &message = glowEvent->getMessage();
  if (message.size() > 0) {
    std::strncpy(outMessage, message.c_str(), maxMessageLength);
    // make sure the message is null termininated. strncpy will pad 0 iff
    // message size is smaller than maxMessageLength.
    outMessage[maxMessageLength] = '\0';
    *outMessageLength = std::min(message.size(), maxMessageLength) + 1;
  }

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
    const onnxTensorDescriptorV1 *weightDescriptors, onnxGraph *graph,
    uint32_t maxSeqLength, void *deferredBlobReader) {
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

  glow::QuantizationMode quantizationMode;
  if (getenv("GLOW_DUMP_PROFILE")) {
    quantizationMode = glow::QuantizationMode::Profile;
  } else if (getenv("GLOW_LOAD_PROFILE")) {
    quantizationMode = glow::QuantizationMode::Quantize;
  } else {
    quantizationMode = glow::QuantizationMode::None;
  }

  bool loadingGlowAOT = false;
  if (auxPropertiesList) {
    for (; *auxPropertiesList != ONNXIFI_GRAPH_PROPERTY_NONE;
         auxPropertiesList++) {
      if (*auxPropertiesList == ONNXIFI_OPTIMIZATION_AOT) {
        loadingGlowAOT = true;
      } else {
        return ONNXIFI_STATUS_UNSUPPORTED_PROPERTY;
      }
    }
  }

  auto *glowGraph = manager.createGraph(glowBackend, quantizationMode);
  auto ret = glowGraph->initGraph(onnxModel, onnxModelSize, weightsCount,
                                  weightDescriptors, maxSeqLength,
                                  deferredBlobReader, loadingGlowAOT);
  if (ret != ONNXIFI_STATUS_SUCCESS) {
    manager.release(glowGraph);
    return ret;
  }

  *graph = glowGraph;
  return ONNXIFI_STATUS_SUCCESS;
}

/// Sanity check for tensor descriptors
static onnxStatus verifyDescriptors(uint32_t count,
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
      bool hasZeroDims = false;
      for (int i = 0; i < descriptor.dimensions; ++i) {
        if (descriptor.shape[i] == 0) {
          hasZeroDims = true;
          break;
        }
      }

      if (!hasZeroDims) {
        LOG(ERROR) << "Bad memory on input " << descriptor.name << " (" << i
                   << " out of " << count
                   << "). It has no memory buffer, but has "
                   << descriptor.dimensions << " dimensions: ["
                   << folly::join(
                          ",", llvm::ArrayRef<uint64_t>(descriptor.shape,
                                                        descriptor.dimensions))
                   << "]";
        return ONNXIFI_STATUS_INVALID_MEMORY_LOCATION;
      }
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
  LOG(ERROR) << "Use onnxSetIOAndRunGraph instead of onnxSetGraphIO";
  return ONNXIFI_STATUS_INTERNAL_ERROR;
}

/// Asynchronously execute operations in an ONNXIFI graph using pre-specified
/// locations for inputs and outputs.
EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxRunGraph)(
    onnxGraph graph, const onnxMemoryFenceV1 *inputFence,
    onnxMemoryFenceV1 *outputFence) {
  LOG(ERROR) << "Use onnxSetIOAndRunGraph instead of onnxRunGraph";
  return ONNXIFI_STATUS_INTERNAL_ERROR;
}

/// Binds inputs and outputs of an ONNXIFI graph to specific addresses then
/// asynchronously execute operations in the graph using the provided
/// addresses.
EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxSetIOAndRunGraph)(
    onnxGraph graph, uint32_t inputsCount,
    const onnxTensorDescriptorV1 *inputDescriptors, uint32_t outputsCount,
    const onnxTensorDescriptorV1 *outputDescriptors,
    onnxMemoryFenceV1 *outputFence, onnxTraceEventList *traceEvents) {
  auto &manager = glow::onnxifi::GlowOnnxifiManager::get();

  if ((inputsCount && !inputDescriptors) ||
      (outputsCount && !outputDescriptors) || !outputFence) {
    LOG(ERROR) << "inputsCount " << inputsCount << ", outputsCount "
               << outputsCount << ", inputDescriptors: " << inputDescriptors
               << ", outputDescriptors: " << outputDescriptors
               << ", outputFence: " << outputFence;
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  // Check output fence is correct type and tag.
  if (outputFence->type != ONNXIFI_SYNCHRONIZATION_EVENT ||
      outputFence->tag != ONNXIFI_TAG_MEMORY_FENCE_V1) {
    return ONNXIFI_STATUS_UNSUPPORTED_TAG;
  }

  // Check glowGraph is valid.
  auto *glowGraph = static_cast<glow::onnxifi::GraphPtr>(graph);
  if (!manager.isValid(glowGraph)) {
    return ONNXIFI_STATUS_INVALID_GRAPH;
  }

  // Initialize outputFence's event.
  auto outputEventInitStatus = GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(
      onnxInitEvent)(glowGraph->backend(), &outputFence->event);
  if (outputEventInitStatus != ONNXIFI_STATUS_SUCCESS) {
    return outputEventInitStatus;
  }

  // Verify inputs.
  auto inputStatus = verifyDescriptors(inputsCount, inputDescriptors);
  if (inputStatus != ONNXIFI_STATUS_SUCCESS) {
    return inputStatus;
  }

  // Verify outputs.
  auto outputStatus = verifyDescriptors(outputsCount, outputDescriptors);
  if (outputStatus != ONNXIFI_STATUS_SUCCESS) {
    return outputStatus;
  }

  auto *outputEvent = static_cast<glow::onnxifi::EventPtr>(outputFence->event);

  // Set graph IO and run async
  return glowGraph->setIOAndRun(inputsCount, inputDescriptors, outputsCount,
                                outputDescriptors, outputEvent, traceEvents);
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

/// Release onnxTraceEvents in \p traceEvents.
EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxReleaseTraceEvents)(
    onnxTraceEventList *traceEvents) {
  if (!traceEvents) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  glow::onnxifi::Graph::releaseTraceEvents(traceEvents);
  return ONNXIFI_STATUS_SUCCESS;
}

EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxGetCurrentBatchSize)(
    int64_t *currentBatchSize) {
  if (!currentBatchSize) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  auto requestData = glow::runtime::RequestData::get();
  if (!requestData) {
    return ONNXIFI_STATUS_INVALID_STATE;
  }
  *currentBatchSize = requestData->currentBatchSize;
  return ONNXIFI_STATUS_SUCCESS;
}

/// Set Onnxifi option
EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxSetOption)(const char *optionName,
                                                     const char *optionValue) {
  if (!optionName || !optionValue) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  onnxStatus ret = ONNXIFI_STATUS_SUCCESS;
  int d = 0;
  if (!strcmp(optionName, "glow_num_devices")) {
    if (sscanf(optionValue, "%d", &d) == 1) {
      glow::flags::NumDevices = d;
    } else {
      ret = ONNXIFI_STATUS_UNSUPPORTED_ATTRIBUTE;
    }
  } else {
    ret = ONNXIFI_STATUS_INVALID_NAME;
  }
  return ret;
}

/// Get Onnxifi option
EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxGetOption)(
    const char *optionName, char *optionValue, size_t *optionValueLength) {
  if (!optionName || !optionValue || !optionValueLength) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  onnxStatus ret = ONNXIFI_STATUS_SUCCESS;
  if (!strcmp(optionName, "glow_num_devices")) {
    int n = snprintf(optionValue, *optionValueLength, "%d",
                     glow::flags::NumDevices);
    if (n < 0) {
      ret = ONNXIFI_STATUS_UNSUPPORTED_ATTRIBUTE;
    } else if (n < *optionValueLength) {
      *optionValueLength = n;
    }
  } else {
    ret = ONNXIFI_STATUS_INVALID_NAME;
  }
  return ret;
}
/// Get pointer to onnxifi extension function with \p name.
EXTERNC ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxGetExtensionFunctionAddress)(
    onnxBackendID backendID, const char *name,
    onnxExtensionFunctionPointer *function) {
  if (!name || !function) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  // We don't check backend id for set/get option functions as the options
  // global to Glow.
  static const std::unordered_set<std::string> bypass{"onnxSetOptionFunction",
                                                      "onnxGetOptionFunction"};
  if (bypass.find(name) == bypass.end()) {
    auto &manager = glow::onnxifi::GlowOnnxifiManager::get();
    auto *glowBackend = static_cast<glow::onnxifi::BackendPtr>(backendID);
    if (!manager.isValid(glowBackend)) {
      return ONNXIFI_STATUS_INVALID_ID;
    }
  }

  // Map of name to onnxExtensionFunctionPointer, one entry for each implemented
  // onnxifi extension.
  // NOTE: when updating this map, also update the response from
  // onnxGetBackendInfo for the ONNXIFI_BACKEND_EXTENSIONS query.
  static const std::unordered_map<std::string, onnxExtensionFunctionPointer>
      extensionMap = {
          {"onnxSetIOAndRunGraphFunction",
           reinterpret_cast<onnxExtensionFunctionPointer>(
               GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxSetIOAndRunGraph))},
          {"onnxWaitEventForFunction",
           reinterpret_cast<onnxExtensionFunctionPointer>(
               GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxWaitEventFor))},
          {"onnxReleaseTraceEventsFunction",
           reinterpret_cast<onnxExtensionFunctionPointer>(
               GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxReleaseTraceEvents))},
          {"onnxGetCurrentBatchSizeFunction",
           reinterpret_cast<onnxExtensionFunctionPointer>(
               GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxGetCurrentBatchSize))},
          {"onnxSetOptionFunction",
           reinterpret_cast<onnxExtensionFunctionPointer>(
               GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxSetOption))},
          {"onnxGetOptionFunction",
           reinterpret_cast<onnxExtensionFunctionPointer>(
               GLOW_ONNXIFI_LIBRARY_FUNCTION_WRAPPER(onnxGetOption))}};

  auto extensionIt = extensionMap.find(name);

  if (extensionIt == extensionMap.end()) {
    // No function found for the given name.
    return ONNXIFI_STATUS_UNIDENTIFIED_NAME;
  }

  *function = extensionIt->second;
  return ONNXIFI_STATUS_SUCCESS;
}
