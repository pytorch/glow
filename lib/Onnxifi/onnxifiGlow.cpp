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
#include "onnx/onnxifi.h"

/**
 * This file contains implementation of the onnxifi interface.
 * Documentation on the functions implementing onnxifi interface in
 * this file is very shallow.
 * Please see more documentation on functions that need to be
 * implemented: https://github.com/onnx/onnx/blob/master/onnx/onnxifi.h.
 */

/// Return stable IDs of available backends on the system.
/// \param backendIDs output parameter and represents pointer to the memory
///                   where the backend IDs will be returned. If it's NULL, numBackends
///                   will be populated with the number of backends supported.
/// \param numBackends input/output parameter.
///                    As an input, it specifies the capacity allocated in the backendIDs.
///                    As an output, it specifies the number of actual available backends. 
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxGetBackendIDs(onnxBackendID* backendIDs, size_t* numBackends) {
  if (!numBackends) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  if (!backendIDs) {
    *numBackends = 1;
    return ONNXIFI_STATUS_FALLBACK;
  }

  *numBackends = 1;
  *backendIDs = 0;
  return ONNXIFI_STATUS_SUCCESS;
}

/// Deinitialize ONNXIFI backend ID and release associated resources.
/// Caller is responsible to release objects associated with the backend ID
/// (onnxBackend, onnxGraph, onnxEvent) before calling this function.
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxReleaseBackendID(onnxBackendID backendID) {
  return ONNXIFI_STATUS_SUCCESS;
}

/// Query high-level information about the backend and its target device.
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI onnxGetBackendInfo(
    onnxBackendID backendID,
    onnxBackendInfo infoType,
    void* infoValue,
    size_t* infoValueSize) {
  if (!infoValueSize) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  return ONNXIFI_STATUS_SUCCESS;
}

/// Query if an ONNX model graph is compatible with the backend.
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxGetBackendCompatibility(
    onnxBackendID backendID,
    size_t onnxModelSize,
    const void* onnxModel) {
  if (!onnxModel) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  if (!onnxModelSize) {
    return ONNXIFI_STATUS_INVALID_SIZE;
  }

  return ONNXIFI_STATUS_SUCCESS;
}

/// Initialize an ONNXIFI backend.
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI onnxInitBackend(
    onnxBackendID backendID,
    const uint64_t* auxpropertiesList,
    onnxBackend* backend) {
  if (!backend) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  return ONNXIFI_STATUS_SUCCESS;
}

/// Deinitialize an ONNXIFI backend and release associated resources.
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxReleaseBackend(onnxBackend backend) {
  return ONNXIFI_STATUS_SUCCESS;
}

/// Initialize a single-shot ONNXIFI event.
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxInitEvent(onnxBackend backend, onnxEvent* event) {
  if (!event) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  return ONNXIFI_STATUS_SUCCESS;
}

/// Change the state of the ONNXIFI event \p event to signalled.
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxSignalEvent(onnxEvent event) {
  return ONNXIFI_STATUS_SUCCESS;
}

/// Wait until an ONNXIFI event is signalled.
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxWaitEvent(onnxEvent event) {
  return ONNXIFI_STATUS_SUCCESS;
}

/// Deinitialize an ONNXIFI event and release associated resources.
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxReleaseEvent(onnxEvent event) {
  return ONNXIFI_STATUS_SUCCESS;
}

/// Parse an ONNXIFI graph and convert it for a particular backend.
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI onnxInitGraph(
    onnxBackend backend,
    size_t onnxModelSize,
    const void* onnxModel,
    uint32_t weightCount,
    const onnxTensorDescriptor* weightDescriptors,
    onnxGraph* graph) {
  if (!onnxModel || !weightDescriptors || !graph) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  if (!onnxModelSize) {
    return ONNXIFI_STATUS_INVALID_SIZE;
  }
  
  return ONNXIFI_STATUS_SUCCESS;
}

/// Binds inputs and outputs of an ONNXIFI graph to specific addresses.
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI onnxSetGraphIO(
    onnxGraph graph,
    uint32_t inputsCount,
    const onnxTensorDescriptor* inputDescriptors,
    uint32_t outputsCount,
    const onnxTensorDescriptor* outputDescriptors) {
  if (!inputDescriptors || !outputDescriptors) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  return ONNXIFI_STATUS_SUCCESS;
}

/// Asynchronously execute operations in an ONNXIFI graph using pre-specified
/// locations for inputs and outputs.
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI onnxRunGraph(
    onnxGraph graph,
    const onnxMemoryFence* inputFence,
    onnxMemoryFence* outputFence) {
  if (!inputFence || !outputFence) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  return ONNXIFI_STATUS_SUCCESS;
}

/// Deinitialize an ONNXIFI graph and release associated resources.
/// It blocks until all in-flight inference operations complete.
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxReleaseGraph(onnxGraph graph) {
  return ONNXIFI_STATUS_SUCCESS;
}
