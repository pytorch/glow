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

#include "glow/Backends/Backend.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Importer/ONNXIFIModelLoader.h"
#include "glow/Support/ThreadPool.h"

#include "onnx/onnxifi.h"
#include "onnx/onnxifi_ext.h"

#include <atomic>
#include <condition_variable>
#include <mutex>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

namespace glow {
namespace onnxifi {

// TODO get rid of this once HostManager is landed.
struct HostManager {
  bool addNetwork(Module *M) {
    llvm_unreachable("HostManager is not yet implemented.");
  }
};

// TODO use the actual type here once available.
using ResultCBTy =
    std::function<void(int, int, std::unique_ptr<glow::Context>)>;

/// BackendId associated with the Glow backend.
class BackendId {
public:
  /// Create Glow ONNXIFI backend identifier with the
  /// given Glow backend \p kind, \p id, \p concurrency, whether to use onnx
  /// or caffe2 for models (\p useInnx), and whether to use HostManager instead
  /// of ExecutionEngine for running graphs (useHostManager).
  /// NOTE: useHostManager is not yet supported as HostManager is yet to be
  /// fully implemented.
  explicit BackendId(glow::BackendKind kind, int id, int concurrency,
                     bool useOnnx, bool useHostManager)
      : id_(id), useOnnx_(useOnnx), concurrency_(concurrency),
        glowBackend_(createBackend(kind)), useHostManager_(useHostManager) {}

  /// Verify that a given onnx graph is supported by the backend by importing
  /// the onnx graph to a glow function, lowering this function, and checking
  /// that all of the glow nodes that are contained in the lowered graph are
  /// compatible with the glow backend.
  onnxStatus checkGraphCompatibility(const void *onnxModel,
                                     size_t onnxModelSize);

  /// \returns the whether use onnx or not.
  bool getUseOnnx() const { return useOnnx_; }

  /// \returns the whether use HostManager for inference or not.
  bool getUseHostManager() const { return useHostManager_; }

  /// \returns HostManager associated with the BackendId.
  HostManager &getHostManager() { return hostManager_; }

  /// \returns the backend id.
  int getID() const { return id_; }

  /// \returns concurrency for the backend.
  int getConcurrency() const { return concurrency_; }

  /// \returns the glow Backend of this BackendId.
  glow::Backend *getGlowBackend() { return glowBackend_.get(); }

  /// Run the network named by \p networkName using HostManager with context \p
  /// ctx afterwhich the result callback \p cb will be called.
  void runOnHostManager(llvm::StringRef networkName,
                        std::unique_ptr<Context> ctx, ResultCBTy cb) {
    // TODO enable once HostManager is landed.
    // hostManager_->runNetwork(networkName, std::move(ctx), std::move(cb));
  }

private:
  int id_;
  bool useOnnx_;
  int concurrency_;
  std::unique_ptr<glow::Backend> glowBackend_;
  bool useHostManager_;
  HostManager hostManager_; // TODO use real HostManager once landed.
};

typedef BackendId *BackendIdPtr;

class Backend {
public:
  explicit Backend(BackendIdPtr backendId)
      : backendIdPtr_(backendId), threadPool_(backendIdPtr_->getConcurrency()) {
  }

  /// Whether this backend uses ONNX proto or Caffe2 proto.
  bool getUseOnnx() const { return backendIdPtr_->getUseOnnx(); }

  /// \returns the whether use HostManager for inference or not.
  bool getUseHostManager() const { return backendIdPtr_->getUseHostManager(); }

  /// \returns HostManager for the associated BackendId.
  HostManager &getHostManager() { return backendIdPtr_->getHostManager(); }

  /// Run inference async using backend thread pool.
  void runAsync(std::function<void(void)> &&fn);

  /// \returns the glow Backend of the associated BackendId.
  glow::Backend *getGlowBackend() { return backendIdPtr_->getGlowBackend(); }

  // Call BackendId::runOnHostManager
  void runOnHostManager(llvm::StringRef networkName,
                        std::unique_ptr<Context> ctx, ResultCBTy cb) {
    backendIdPtr_->runOnHostManager(networkName, std::move(ctx), std::move(cb));
  }

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
  explicit Graph(BackendPtr backendPtr) : backendPtr_(backendPtr) {
    executionEngine_.setBackend(backendPtr->getGlowBackend(),
                                /*ownsBackend*/ false);
  }

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
  /// Execution engine to use to run this graph.
  glow::ExecutionEngine executionEngine_;
  BackendPtr backendPtr_;
  Function *function_;

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
