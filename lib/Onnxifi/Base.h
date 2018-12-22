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
#include "glow/Importer/ONNXIFIModelLoader.h"
#include "glow/Support/ThreadPool.h"

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
  /// given Glow backend \p kind, \p id, \p concurrency and whether to use onnx
  /// or caffe2 for models (\p use_onnx).
  explicit BackendId(glow::BackendKind kind, int id, int concurrency,
                     bool use_onnx)
      : id_(id), use_onnx_(use_onnx), concurrency_(concurrency),
        executionEngine_(kind) {}

  bool isOpSupported(Kinded::Kind opKind, ElemKind elementTy);

  /// Verify that a given onnx graph is supported by the backend by importing
  /// the onnx graph to a glow function, lowering this function, and checking
  /// that all of the glow nodes that are contained in the lowered graph are
  /// compatible with the glow backend.
  onnxStatus checkGraphCompatibility(const void *onnxModel,
                                     size_t onnxModelSize);

  /// \returns Execution Engine associated with the Backend.
  glow::ExecutionEngine &getEE() { return executionEngine_; }

  /// \returns the whether use onnx or not
  bool getUseOnnx() const { return use_onnx_; }

  /// \returns the backend id.
  int getID() const { return id_; }

  /// \returns concurrency for the backend.
  int getConcurrency() const { return concurrency_; }

private:
  int id_;
  bool use_onnx_;
  int concurrency_;
  glow::ExecutionEngine executionEngine_;
};

typedef BackendId *BackendIdPtr;

class Backend {
public:
  explicit Backend(BackendIdPtr backendId)
      : backendIdPtr_(backendId), threadPool_(backendIdPtr_->getConcurrency()) {
  }

  /// \returns Execution Engine associated with the Backend.
  glow::ExecutionEngine &getEE() { return backendIdPtr_->getEE(); }

  /// Run inference async using backend thread pool.
  void runAsync(const std::function<void(void)> &fn);

private:
  BackendIdPtr backendIdPtr_;
  // ThreadPool instance for the backend.
  ThreadPool threadPool_;
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

  /// Run inference synchronously.
  /// \params inputPlaceholderToBuffer contains mapping between input
  ///         placeholders and memory addresses input can be read from.
  /// \params outputPlaceholderToBuffer contains mapping between output
  ///         placeholders and memory addresses output needs to be written to.
  void run(const llvm::DenseMap<Placeholder *, onnxPointer>
               &inputPlaceholderToBuffer,
           const llvm::DenseMap<Placeholder *, onnxPointer>
               &outputPlaceholderToBuffer);

  /// Run inference asynchronously.
  /// Inputs are ready when \p inputEvent is signalled.
  /// \p outputEvent needs to be signalled when outputs are computed.
  void runAsync(EventPtr inputEvent, EventPtr outputEvent);

private:
  BackendPtr backendPtr_;
  Function *function_;

  /// This is the compilation context that represents a single thread.
  /// TODO: Once we finish the migration to placeholders we'll need to manage
  /// the state properly.
  Context ctx_;

  /// Mapping between ONNX name for the input variable and Glow
  /// placeholder for input.
  llvm::StringMap<Placeholder *> onnxInputToPlaceholder_;

  /// Mapping between ONNX name for the output variable and Glow
  /// placeholder for output.
  llvm::StringMap<Placeholder *> onnxOutputToPlaceholder_;

  /// Mapping between input placeholder and the actual memory address.
  /// Inputs will be read from these addresses.
  llvm::DenseMap<Placeholder *, onnxPointer> inputPlaceholderToBuffer_;

  /// Mapping between output placeholder and the actual memory address.
  /// Results must be written to these addresses.
  llvm::DenseMap<Placeholder *, onnxPointer> outputPlaceholderToBuffer_;

  /// Guard setIO and run. Make sequence of setIO and run
  /// to be atomic.
  std::mutex inputRunMutex_;
};

typedef Graph *GraphPtr;

} // namespace onnxifi
} // namespace glow

#endif // GLOW_ONNXIFI_BASE_H
