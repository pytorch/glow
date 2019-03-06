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
#include "glow/Importer/ONNXIFIModelLoader.h"
#include "glow/Runtime/HostManager/HostManager.h"

#include "foxi/onnxifi.h"
#include "foxi/onnxifi_ext.h"

#include <atomic>
#include <condition_variable>
#include <mutex>

#include "llvm/ADT/StringMap.h"

namespace glow {
namespace onnxifi {

/// BackendId associated with the Glow backend.
class BackendId {
public:
  /// Create Glow ONNXIFI backend identifier with the
  /// given Glow backend \p kind, \p concurrency, whether to use onnx
  /// or caffe2 for models (\p useOnnx), and whether to use HostManager instead
  /// of ExecutionEngine for running graphs (useHostManager).
  /// NOTE: useHostManager is not yet supported as HostManager is yet to be
  /// fully implemented.
  explicit BackendId(runtime::HostManager *hostManager, glow::BackendKind kind,
                     bool useOnnx)
      : hostManager_(hostManager), useOnnx_(useOnnx),
        glowBackend_(createBackend(kind)) {}

  /// Verify that a given onnx graph is supported by the backend by importing
  /// the onnx graph to a glow function, lowering this function, and checking
  /// that all of the glow nodes that are contained in the lowered graph are
  /// compatible with the glow backend.
  onnxStatus checkGraphCompatibility(const void *onnxModel,
                                     size_t onnxModelSize);

  /// \returns the whether use onnx or not.
  bool getUseOnnx() const { return useOnnx_; }

  /// \returns HostManager associated with the BackendId.
  runtime::HostManager &getHostManager() { return *hostManager_; }

<<<<<<< HEAD
  /// \returns the backend id.
  int getID() const { return id_; }

  /// \returns concurrency for the backend.
  int getConcurrency() const { return concurrency_; }

  /// \returns the glow Backend of this BackendId.
  glow::Backend *getGlowBackend() { return glowBackend_.get(); }

  /// Run the network named by \p networkName using HostManager with bindings \p
  /// bindings afterwhich the result callback \p cb will be called.
  void runOnHostManager(llvm::StringRef networkName,
                        std::unique_ptr<PlaceholderBindings> bindings,
                        ResultCBTy cb) {
    // TODO enable once HostManager is landed.
    // hostManager_->runNetwork(networkName, std::move(bindings),
    // std::move(cb));
  }
=======
  // \returns a unique_ptr to a new HostManager for the given BackendKind \p
  // kind.
  static std::unique_ptr<runtime::HostManager>
  createHostManager(glow::BackendKind kind);
>>>>>>> Use HostManager for onnxifi

private:
  runtime::HostManager *hostManager_;
  bool useOnnx_;
  // TODO: glowBackend_ is need Backend for checkGraphCompatibility because
  // isOpSupported, shouldLower, etc aren't exposed by HostManager. These
  // methods should be made static however and then glowBackend_ can be removed.
  std::unique_ptr<glow::Backend> glowBackend_;
};

typedef BackendId *BackendIdPtr;

class Backend {
public:
  explicit Backend(BackendIdPtr backendId) : backendIdPtr_(backendId) {}

  /// Whether this backend uses ONNX proto or Caffe2 proto.
  bool getUseOnnx() const { return backendIdPtr_->getUseOnnx(); }

  /// \returns HostManager for the associated BackendId.
  runtime::HostManager &getHostManager() {
    return backendIdPtr_->getHostManager();
  }

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
  explicit Graph(BackendPtr backendPtr);
  ~Graph();

  BackendPtr backend() { return backendPtr_; }

  /// Init Glow graph based on the ONNX model \p onnxModel and
  /// static trained weights \p weightDescriptors.
  onnxStatus initGraph(const void *onnxModel, size_t onnxModelSize,
                       uint32_t weightCount,
                       const onnxTensorDescriptorV1 *weightDescriptors);

  /// \returns HostManager for the associated BackendId.
  runtime::HostManager &getHostManager() {
    return backendPtr_->getHostManager();
  }

  /// Setup Glow graph in preparation for the inference and run.
  /// Set input memory addresses for inputs based on the \p inputDescriptors.
  /// Set output memory addresses for outputs based on the \p outputDescriptors.
  /// Will async signal the \p outputEvent when run is complete.
  onnxStatus setIOAndRunAsync(uint32_t inputsCount,
                              const onnxTensorDescriptorV1 *inputDescriptors,
                              uint32_t outputsCount,
                              const onnxTensorDescriptorV1 *outputDescriptors,
                              EventPtr outputEvent);

private:
  /// \returns a globally unique graph id.
  static size_t makeUniqueGraphId();

  BackendPtr backendPtr_;
  Module m_;
  std::string netName_;

  /// Mapping between ONNX name for the input variable and Glow
  /// placeholder for input.
  llvm::StringMap<Placeholder *> onnxInputToPlaceholder_;

  /// Mapping between ONNX name for the output variable and Glow
  /// placeholder for output.
  llvm::StringMap<Placeholder *> onnxOutputToPlaceholder_;
};

typedef Graph *GraphPtr;

} // namespace onnxifi
} // namespace glow

#endif // GLOW_ONNXIFI_BASE_H
