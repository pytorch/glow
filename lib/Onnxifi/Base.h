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
#ifndef GLOW_ONNXIFI_BASE_H
#define GLOW_ONNXIFI_BASE_H

#include "glow/Backend/Backend.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Importer/ONNXIFIModelLoader.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "glow/Support/TensorPool.h"

#include "foxi/onnxifi.h"
#include "foxi/onnxifi_ext.h"

#include <atomic>
#include <condition_variable>
#include <mutex>

#include "llvm/ADT/StringMap.h"

namespace glow {
namespace onnxifi {

class Graph;

/// Backend associated with the Glow backend.
class Backend {
public:
  /// Create Glow ONNXIFI backend identifier with the
  /// given Glow backend \p backendName, whether to use onnx or caffe2 for
  /// models
  /// (\p useOnnx)
  Backend(llvm::StringRef backendName, bool useOnnx)
      : useOnnx_(useOnnx), glowBackend_(createBackend(backendName)) {}

  virtual ~Backend() = default;

  /// Verify that a given onnx graph is supported by the backend by importing
  /// the onnx graph to a glow function, lowering this function, and checking
  /// that all of the glow nodes that are contained in the lowered graph are
  /// compatible with the glow backend.
  onnxStatus checkGraphCompatibility(const void *onnxModel,
                                     size_t onnxModelSize);

  /// \returns the whether use onnx or not.
  bool getUseOnnx() const { return useOnnx_; }

  /// \returns a reference to the backend.
  const glow::Backend &getBackend() const { return *glowBackend_; }

  virtual void runNetwork(const Graph *graph,
                          std::unique_ptr<ExecutionContext> context,
                          runtime::ResultCBTy callback, uint64_t priority = 0) {
  }

  virtual onnxStatus removeNetwork(const Graph *graph) {
    return ONNXIFI_STATUS_SUCCESS;
  }

protected:
  bool useOnnx_;
  std::unique_ptr<glow::Backend> glowBackend_;
};

typedef Backend *BackendPtr;

class Event {
public:
  Event() : fired_{false} {}
  /// Signal the event.
  bool signal(onnxStatus status);

  /// Wait until the event is signalled.
  onnxStatus wait();

  /// Wait until the event is signalled or until at least \p timeoutMs
  /// milliseconds have elapsed. \returns a pair with the first value being a
  /// boolean that is true if the event was signalled (no timeout occurred) and
  /// the second is the value of the event's status.
  std::pair<bool, onnxStatus> waitFor(size_t timeoutMs);

  /// Check if event was signalled.
  bool isSignalled() { return fired_; }

  const std::string &getMessage() const { return message_; }

  void setMessage(const std::string &message) { message_ = message; }

private:
  std::atomic<bool> fired_;
  std::mutex mutex_;
  std::condition_variable cond_;
  std::string message_;
  /// Used to hold an onnxStatus that will be passed for the signaller of the
  /// event to a waiter. Should only be accessed while holding mutex_.
  onnxStatus status_ = ONNXIFI_STATUS_SUCCESS;
};

typedef Event *EventPtr;

class Graph {
public:
  explicit Graph(BackendPtr backendPtr);
  virtual ~Graph() = default;

  BackendPtr backend() { return backendPtr_; }

  /// Setup Glow graph in preparation for the inference and run.
  /// Set input memory addresses for inputs based on the \p inputDescriptors.
  /// Set output memory addresses for outputs based on the \p
  /// outputDescriptors. Will async signal the \p outputEvent when run is
  /// complete. \p traceEvents is a pointer to onnxTraceEventList, if it is not
  /// null then it is expected that this will be populated with trace events
  /// from the run before signalling the outputEvent.
  onnxStatus setIOAndRun(uint32_t inputsCount,
                         const onnxTensorDescriptorV1 *inputDescriptors,
                         uint32_t outputsCount,
                         const onnxTensorDescriptorV1 *outputDescriptors,
                         EventPtr outputEvent, onnxTraceEventList *traceEvents);

  /// Init Glow graph based on the ONNX model \p onnxModel and
  /// static trained weights \p weightDescriptors. Weights can be read in later
  /// by a \p deferedBlobReader.
  virtual onnxStatus initGraph(const void *onnxModel, size_t onnxModelSize,
                               uint32_t weightCount,
                               const onnxTensorDescriptorV1 *weightDescriptors,
                               uint32_t maxSeqLength, void *deferedBlobReader,
                               bool loadingGlowAOT) = 0;

  virtual onnxStatus run(std::unique_ptr<ExecutionContext> ctx,
                         EventPtr outputEvent,
                         onnxTraceEventList *traceEvents) = 0;

  /// Copy any trace events \p traceContext into \p traceEvents. If
  /// \p traceEvents is null then do nothing.
  static void setTraceEvents(onnxTraceEventList *traceEvents,
                             TraceContext *traceContext);

  /// Free all memory that was allocated by setTraceEvents when creating \p
  /// traceEvents.
  static void releaseTraceEvents(onnxTraceEventList *traceEvents);

protected:
  BackendPtr backendPtr_;

  /// Mapping between ONNX name for the input variable and Glow
  /// placeholder for input.
  llvm::StringMap<Placeholder *> onnxInputToPlaceholder_;

  /// Mapping between ONNX name for the output variable and Glow
  /// placeholder for output.
  llvm::StringMap<Placeholder *> onnxOutputToPlaceholder_;

  /// A list of input names ordered by their position in ONNXIFI input
  /// descriptor array.
  std::vector<std::string> onnxInputNames_;

  /// A list of input placeholders ordered by their position in ONNXIFI input
  /// descriptor array.
  std::vector<Placeholder *> onnxInputPlaceholders_;

  /// A list of output names ordered by their position in ONNXIFI output
  /// descriptor array.
  std::vector<std::string> onnxOutputNames_;

  /// A list of output placeholders ordered by their position in ONNXIFI output
  /// descriptor array.
  std::vector<Placeholder *> onnxOutputPlaceholders_;

  /// An object pool for tensors, to share allocations.
  TensorPool tensorPool_;

  /// An anchor tensor specialized for zero length indices
  Tensor zeroLengthSequence_;

  /// Set the zero length tensor
  void setZeroLengthSequence(dim_t maxSeqLength);

  /// Bind input/output placeholders
  bool bindPlaceholders(const ONNXIFIModelLoader &loader,
                        LoadedPlaceholderNameMap *loadedPHNames = nullptr);

private:
  /// inference dump counter
  std::atomic<size_t> ioDumpCounter_{0};

  /// Setup input mapping and adjust inputs if necessary
  onnxStatus adjustInputs(uint32_t inputsCount,
                          const onnxTensorDescriptorV1 *inputDescriptors,
                          ExecutionContext *ctx);
};

typedef Graph *GraphPtr;

/// Save the Function from ONNXIFI to a file
void saveOnnxifiModel(Function *F);

} // namespace onnxifi
} // namespace glow

#endif // GLOW_ONNXIFI_BASE_H
