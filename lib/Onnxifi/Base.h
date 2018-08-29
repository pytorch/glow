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
#ifndef GLOW_ONNXIFI_BASE_H
#define GLOW_ONNXIFI_BASE_H

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Importer/ONNXIFILoader.h"

#include "onnx/onnxifi.h"

#include <atomic>
#include <condition_variable>
#include <mutex>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

namespace glow {
namespace onnxifi {

/// BackendId associated with the Glow backend.
class BackendId {
public:
  /// Create Glow ONNXIFI backend identifier with the
  /// given Glow backend \p kind and \p id.
  explicit BackendId(glow::BackendKind kind, int id)
      : id_(id), executionEngine_(kind) {}

  /// Verify that given operation kind is supported by the backend.
  bool isOpSupported(Kinded::Kind opKind, ElemKind elementTy);

  /// \returns Execution Engine associated with the Backend.
  glow::ExecutionEngine &getEE() { return executionEngine_; }

private:
  int id_;
  glow::ExecutionEngine executionEngine_;
};

typedef BackendId *BackendIdPtr;

class Backend {
public:
  explicit Backend(BackendIdPtr backendId) : backendIdPtr_(backendId) {}

  /// \returns Execution Engine associated with the Backend.
  glow::ExecutionEngine &getEE() { return backendIdPtr_->getEE(); }

private:
  BackendIdPtr backendIdPtr_;
};

typedef Backend *BackendPtr;

class Event {
public:
  Event() : fired_{false} {}
  /// Signal the event.
  bool signal();

  /// Wait until the event is signalled.
  void wait();

  /// Check if event was signalled.
  bool isSignalled() { return fired_; }

private:
  std::atomic<bool> fired_;
  std::mutex mutex_;
  std::condition_variable cond_;
};

typedef Event *EventPtr;

class Graph {
public:
  explicit Graph(BackendPtr backendPtr) : backendPtr_(backendPtr) {}

  BackendPtr backend() { return backendPtr_; }

  /// Init Glow graph based on the ONNX model \p onnxModel and
  /// static trained weights \p weightDescriptors.
  onnxStatus initGraph(const void *onnxModel, size_t onnxModelSize,
                       uint32_t weightCount,
                       const onnxTensorDescriptorV1 *weightDescriptors);

  /// Setup Glow graph in preparation for the inference.
  /// Set input memory addresses for inputs based on the \p inputDescriptors.
  /// Set output memory addresses for outputs based on
  /// the \p outputDescriptors.
  onnxStatus setIO(uint32_t inputsCount,
                   const onnxTensorDescriptorV1 *inputDescriptors,
                   uint32_t outputsCount,
                   const onnxTensorDescriptorV1 *outputDescriptors);

  /// Run inference.
  onnxStatus run();

private:
  BackendPtr backendPtr_;
  Function *function_;

  /// Mapping between ONNX name for the input variable and Glow variable.
  llvm::StringMap<Variable *> onnxNameToInputVar_;

  /// Mapping between ONNX name for the output variable and Glow output
  /// node.
  llvm::StringMap<Variable *> onnxNameToOutputNode_;

  /// Mapping between input var and the actual memory address.
  /// Inputs will be read from these addresses.
  llvm::DenseMap<Variable *, onnxPointer> inputVarToBuffer_;

  /// Mapping between output var and the actual memory address.
  /// Results must be written to these addresses.
  llvm::DenseMap<Variable *, onnxPointer> outputNodeToBuffer_;
};

typedef Graph *GraphPtr;

} // namespace onnxifi
} // namespace glow

#endif // GLOW_ONNXIFI_BASE_H
