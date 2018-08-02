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

#include "onnx/onnxifi.h"

#include <condition_variable>
#include <mutex>

namespace glow {
namespace onnxifi {

/// BackendId associated with the Glow backend.
class BackendId {
public:
  explicit BackendId(int id) : id_(id) {}

  /// Verify that given operation is supported by the backend.
  bool isOpSupported(const glow::Node &node);

  /// \returns Execution Engine associated with the Backend.
  glow::ExecutionEngine &getEE() { return executionEngine_; }

private:
  int id_;
  // By default use the Interpreter backend.
  glow::ExecutionEngine executionEngine_{glow::BackendKind::Interpreter};
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
  /// Signal.
  bool signal();
  /// Wait.
  void wait();

private:
  bool fired_;
  std::mutex mutex_;
  std::condition_variable cond_;
};

typedef Event *EventPtr;

class Graph {
public:
  explicit Graph(BackendPtr backendPtr) : backendPtr_(backendPtr) {}

  BackendPtr backend() { return backendPtr_; }

  /// InitGraph.
  onnxStatus initGraph(const void *onnxModel, size_t onnxModelSize,
                       uint32_t weightCount,
                       const onnxTensorDescriptorV1 *weightDescriptors);
  /// Set IO.
  onnxStatus setIO(uint32_t inputsCount,
                   const onnxTensorDescriptorV1 *inputDescriptors,
                   uint32_t outputsCount,
                   const onnxTensorDescriptorV1 *outputDescriptors);
  /// Run graph.
  onnxStatus run();

private:
  BackendPtr backendPtr_;
  Function *function_;
};

typedef Graph *GraphPtr;

} // namespace onnxifi
} // namespace glow

#endif // GLOW_ONNXIFI_BASE_H
