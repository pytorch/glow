/*
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

#include "glow/Importer/ONNXIFIModelLoader.h"

#include "llvm/Support/Format.h"
#include <glog/logging.h>

namespace glow {
namespace onnxifi {
extern bool GlowDumpDebugTraces;

namespace {
const char *compatibilityFunctionName = "check";
} // namespace

onnxStatus BackendId::checkGraphCompatibility(const void *onnxModel,
                                              size_t onnxModelSize) {
  Module module;

  auto function = module.createFunction(compatibilityFunctionName);

  std::unique_ptr<ONNXIFIModelLoader> loader;
  auto loaderOrErr = ONNXIFIModelLoader::parse(
      onnxModel, onnxModelSize, 0 /*weightCount*/,
      nullptr /*weightDescriptors*/, *function,
      false /*loadInputsAsPlaceholders*/, getUseOnnx());
  if (loaderOrErr) {
    loader = std::move(*loaderOrErr);
  } else {
    // TODO: Use a more specific ONNXIFI error code here to denote what about
    // this operator is not supported (shape, type, etc).
    LOG(ERROR) << "Error when loading protobuf: "
               << llvm::toString(loaderOrErr.takeError());
    return ONNXIFI_STATUS_UNSUPPORTED_OPERATOR;
  }

  if (!glowBackend_) {
    return ONNXIFI_STATUS_INTERNAL_ERROR;
  }

  glow::lower(function, /* loweredMap */ nullptr, glowBackend_.get());

  // Call the backend's transformPostLowering to match the normal compilation
  // pipeline then DCE any nodes that are no longer needed.
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  if (glowBackend_->transformPostLowering(function, cctx)) {
    glow::DCE(function);
  }

  if (!function->verify()) {
    LOG(ERROR) << "ONNXIFI: Function verification failed.";
    return ONNXIFI_STATUS_UNSUPPORTED_OPERATOR;
  }

  const auto &nodes = function->getNodes();
  for (const auto &node : nodes) {
    if (!glowBackend_->isOpSupported(node)) {
      LOG(ERROR) << "ONNXIFI: Not supported op: " << node.getDebugDesc();
      // TODO: Use a more specific ONNXIFI error code here to denote what about
      // this operator is not supported (shape, type, etc).
      return ONNXIFI_STATUS_UNSUPPORTED_OPERATOR;
    }
  }

  return ONNXIFI_STATUS_SUCCESS;
}

bool Event::signal() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (fired_) {
      return false;
    }
    fired_ = true;
  }
  cond_.notify_all();
  return true;
}

void Event::wait() {
  std::unique_lock<std::mutex> guard(mutex_);
  cond_.wait(guard, [this] { return fired_ == true; });
}

onnxStatus Graph::setIOAndRun(uint32_t inputsCount,
                              const onnxTensorDescriptorV1 *inputDescriptors,
                              uint32_t outputsCount,
                              const onnxTensorDescriptorV1 *outputDescriptors,
                              EventPtr outputEvent,
                              onnxTraceEventList *traceEvents) {
  auto ctx = llvm::make_unique<ExecutionContext>();

  TraceContext *traceContext = nullptr;
  if (traceEvents || GlowDumpDebugTraces) {
    ctx->setTraceContext(llvm::make_unique<TraceContext>(TraceLevel::STANDARD));
    traceContext = ctx->getTraceContext();
  }
  TRACE_EVENT_SCOPE(traceContext, "Onnxifi::setIOAndRun");

  TRACE_EVENT_BEGIN(traceContext, "adjustInputs");
  size_t totalInputOnnxTensorSize = 0;
  size_t totalInputGlowTensorSize = 0;
  // Create tensors for input placeholders
  for (unsigned i = 0; i < inputsCount; ++i) {
    const auto &inOnnxTensor = inputDescriptors[i];
    auto *inOnnxBuffer = reinterpret_cast<void *>(inOnnxTensor.buffer);

    auto inPhIt = onnxInputToPlaceholder_.find(inOnnxTensor.name);
    if (inPhIt == onnxInputToPlaceholder_.end()) {
      return ONNXIFI_STATUS_UNIDENTIFIED_NAME;
    }

    auto &inPhPtr = inPhIt->getValue();

    std::vector<size_t> inOnnxTensorDims(inOnnxTensor.dimensions);
    size_t inOnnxTensorSize = 1;
    for (unsigned j = 0; j < inOnnxTensor.dimensions; ++j) {
      inOnnxTensorDims[j] = inOnnxTensor.shape[j];
      inOnnxTensorSize *= inOnnxTensorDims[j];
    }

    if (inOnnxTensorSize > inPhPtr->getType()->size()) {
      LOG(ERROR) << "Input tensor is too large: " << inOnnxTensorSize << " vs "
                 << inPhPtr->getType()->size() << ": " << inOnnxTensor.name;
      return ONNXIFI_STATUS_INVALID_SHAPE;
    }

    // Only re-allocate a tensor in case padding is required.
    // Otherwise just back the tensor by memory provided by the caller.
    if (inPhPtr->dims().equals(inOnnxTensorDims)) {
      ctx->getPlaceholderBindings()->insert(
          inPhPtr, new Tensor(inOnnxBuffer, inPhPtr->getType()));
    } else {
      Tensor *inputTensor = tensorPool_.get(inPhPtr->getType());
      assert(inputTensor);
      // If input onnxTensorDescriptor has a NULL buffer pointer, which is a
      // valid case for empty tensor, skip copying
      if (inOnnxBuffer) {
        unsigned elementSize = inPhPtr->getType()->getElementSize();
        char *onnxBuffer = static_cast<char *>(inOnnxBuffer);
        std::copy(onnxBuffer, onnxBuffer + inOnnxTensorSize * elementSize,
                  inputTensor->getUnsafePtr());
      }

      ctx->getPlaceholderBindings()->insert(inPhPtr, inputTensor);
    }
    unsigned elementSize = inPhPtr->getType()->getElementSize();
    totalInputOnnxTensorSize += inOnnxTensorSize * elementSize;
    totalInputGlowTensorSize += inPhPtr->getType()->size() * elementSize;
  }
  VLOG_EVERY_N(1, 100) << "Tensor size blow up from static sizing : "
                       << totalInputOnnxTensorSize << " bytes to "
                       << totalInputGlowTensorSize << " bytes ("
                       << float(totalInputGlowTensorSize) /
                              float(totalInputOnnxTensorSize)
                       << " X)";

  TRACE_EVENT_END(traceContext, "adjustInputs");
  TRACE_EVENT_BEGIN(traceContext, "setOnnxifiOutputs");

  // Create tensors for output placeholders
  for (unsigned i = 0; i < outputsCount; ++i) {
    const auto &outOnnxTensor = outputDescriptors[i];
    auto *outOnnxBuffer = reinterpret_cast<void *>(outOnnxTensor.buffer);

    auto outPhIt = onnxOutputToPlaceholder_.find(outOnnxTensor.name);
    if (outPhIt == onnxOutputToPlaceholder_.end()) {
      return ONNXIFI_STATUS_UNIDENTIFIED_NAME;
    }

    auto &outPhPtr = outPhIt->getValue();

    // Compute the total size of the onnxifi tensor.
    std::vector<size_t> outOnnxTensorDims(outOnnxTensor.dimensions);
    size_t outOnnxTensorSize = 1;
    for (unsigned j = 0; j < outOnnxTensor.dimensions; ++j) {
      outOnnxTensorDims[j] = outOnnxTensor.shape[j];
      outOnnxTensorSize *= outOnnxTensorDims[j];
    }

    // Check that tensor provided by onnxifi is the correct size.
    if (!outPhPtr->dims().equals(outOnnxTensorDims)) {
      LOG(ERROR) << "Output tensor is the wrong shape: " << outOnnxTensorSize
                 << " total dims vs " << outPhPtr->getType()->size() << ": "
                 << outOnnxTensor.name;
      return ONNXIFI_STATUS_INVALID_SHAPE;
    }

    // Create a Glow tensor backed by the memory from the provided onnxifi
    // tensor and bind it to the appropriate placeholder for the graph output.
    Tensor outputTensor(outOnnxBuffer, outPhPtr->getType());
    ctx->getPlaceholderBindings()->insert(outPhPtr, std::move(outputTensor));
  }

  TRACE_EVENT_END(traceContext, "setOnnxifiOutputs");
  return run(std::move(ctx), outputEvent, traceEvents);
}

void Graph::setTraceEvents(onnxTraceEventList *traceEvents,
                           TraceContext *traceContext) {
  if (!traceEvents || !traceContext) {
    return;
  }

  TRACE_EVENT_SCOPE(traceContext, "Onnxifi::setTraceEvents");

  std::vector<onnxTraceEvent *> traceEventsVec;
  for (const auto &glowTraceEvent : traceContext->getTraceEvents()) {
    auto *traceEvent = new onnxTraceEvent();
    assert(
        glowTraceEvent.type.size() == 1 &&
        "Events with types longer than a single char not supported by onnxifi");
    traceEvent->eventType = glowTraceEvent.type[0];
    traceEvent->timestamp = glowTraceEvent.timestamp;
    traceEvent->tid = glowTraceEvent.tid;
    traceEvent->duration = 0;
    size_t nameSize = std::min(glowTraceEvent.name.size(),
                               (size_t)ONNXIFI_TRACE_EVENT_NAME_SIZE);
    strncpy(traceEvent->eventName, glowTraceEvent.name.c_str(), nameSize);
    traceEvent->eventName[nameSize] = '\0';
    traceEventsVec.push_back(traceEvent);
  }

  traceEvents->numEvents = traceEventsVec.size();
  traceEvents->traceEvents = new onnxTraceEvent *[traceEventsVec.size()];
  assert(traceEvents->traceEvents);
  std::copy(traceEventsVec.begin(), traceEventsVec.end(),
            traceEvents->traceEvents);
}

void Graph::releaseTraceEvents(onnxTraceEventList *traceEvents) {
  assert(traceEvents);
  for (uint64_t i = 0; i < traceEvents->numEvents; ++i) {
    onnxTraceEvent *traceEvent = traceEvents->traceEvents[i];
    delete traceEvent;
  }

  delete[] traceEvents->traceEvents;
}

Graph::Graph(BackendPtr backendPtr) : backendPtr_(backendPtr) {}

} // namespace onnxifi
} // namespace glow
