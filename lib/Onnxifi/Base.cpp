/*
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

#include "glow/Exporter/ONNXModelWriter.h"
#include "glow/Importer/ONNXIFIModelLoader.h"
#include "glow/Optimizer/GraphOptimizer/FunctionPasses.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

#include "llvm/Support/Format.h"
#include <glog/logging.h>

namespace glow {
namespace onnxifi {
bool GlowSaveOnnxifiModel = false;
bool GlowSaveOnnxifiIO = false;
bool GlowEnablePartialTensors = true;

extern bool GlowDumpDebugTraces;

namespace {
const char *compatibilityFunctionName = "check";
} // namespace

void saveOnnxifiModel(Function *F) {
  std::string fname = F->getName().str() + ".zip";
  LOG(INFO) << "Saving model to " << fname;
  Error err = Error::empty();
  constexpr size_t kIrVer = 7, kOpsetVer = 9;
  { ONNXModelWriter onnxWR(fname, *F, kIrVer, kOpsetVer, &err, false, true); }
  if (ERR_TO_BOOL(std::move(err))) {
    LOG(ERROR) << "ONNXModelWriter failed to write model: " << fname;
  }
}

onnxStatus Backend::checkGraphCompatibility(const void *onnxModel,
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
               << ERR_TO_STRING(loaderOrErr.takeError());
    return ONNXIFI_STATUS_UNSUPPORTED_OPERATOR;
  }

  if (!glowBackend_) {
    return ONNXIFI_STATUS_INTERNAL_ERROR;
  }

  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  glow::lower(function, cctx, glowBackend_.get());

  // Call the backend's transformPostLowering to match the normal compilation
  // pipeline then DCE any nodes that are no longer needed.
  if (glowBackend_->transformPostLowering(function, cctx)) {
    runDCEPass(function, cctx);
  }

  if (!function->verify()) {
    LOG(ERROR) << "ONNXIFI: Function verification failed.";
    return ONNXIFI_STATUS_UNSUPPORTED_OPERATOR;
  }

  const auto &nodes = function->getNodes();
  for (const auto &node : nodes) {
    if (!glowBackend_->isOpSupported(node)) {
      LOG(ERROR) << "ONNXIFI: Not supported op: " << node.getDebugDesc();
      // TODO: Use a more specific ONNXIFI error code here to denote what
      // about this operator is not supported (shape, type, etc).
      return ONNXIFI_STATUS_UNSUPPORTED_OPERATOR;
    }
  }

  return ONNXIFI_STATUS_SUCCESS;
}

bool Event::signal(onnxStatus status) {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (fired_) {
      return false;
    }
    status_ = status;
    fired_ = true;
  }
  cond_.notify_all();
  return true;
}

onnxStatus Event::wait() {
  std::unique_lock<std::mutex> guard(mutex_);
  cond_.wait(guard, [this] { return fired_ == true; });
  return status_;
}

std::pair<bool, onnxStatus> Event::waitFor(size_t timeoutMs) {
  DCHECK_GT(timeoutMs, 0)
      << "0 timeoutMs should instead use Event::wait to wait indefinitely";

  auto endTime =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);

  std::unique_lock<std::mutex> guard(mutex_);
  while (!fired_) {
    if (std::cv_status::timeout == cond_.wait_until(guard, endTime)) {
      return {/*signalled*/ false, status_};
    }
  }

  return {/*signalled*/ true, status_};
}

void Graph::setZeroLengthSequence(dim_t maxSeqLength) {
  Type ty(ElemKind::Int64ITy, {maxSeqLength});
  zeroLengthSequence_.reset(ty);
  zeroLengthSequence_.zero();
}

onnxStatus Graph::setIOAndRun(uint32_t inputsCount,
                              const onnxTensorDescriptorV1 *inputDescriptors,
                              uint32_t outputsCount,
                              const onnxTensorDescriptorV1 *outputDescriptors,
                              EventPtr outputEvent,
                              onnxTraceEventList *traceEvents) {
  auto ctx = glow::make_unique<ExecutionContext>();

  TraceContext *traceContext = nullptr;
  if (traceEvents || GlowDumpDebugTraces) {
    ctx->setTraceContext(glow::make_unique<TraceContext>(TraceLevel::STANDARD));
    traceContext = ctx->getTraceContext();
    traceContext->setThreadName("Onnxifi");
  }
  TRACE_EVENT_SCOPE(traceContext, TraceLevel::RUNTIME, "Onnxifi::setIOAndRun");
  TRACE_EVENT_SCOPE_NAMED(traceContext, TraceLevel::RUNTIME, "adjustInputs",
                          aiEvent);

  // Create tensors for input placeholders
  for (unsigned i = 0; i < inputsCount; ++i) {
    const auto &inOnnxTensor = inputDescriptors[i];
    auto *inOnnxBuffer = reinterpret_cast<void *>(inOnnxTensor.buffer);

    auto inPhIt = onnxInputToPlaceholder_.find(inOnnxTensor.name);
    if (inPhIt == onnxInputToPlaceholder_.end()) {
      return ONNXIFI_STATUS_UNIDENTIFIED_NAME;
    }

    auto &inPhPtr = inPhIt->getValue();

    std::vector<dim_t> inOnnxTensorDims(inOnnxTensor.dimensions);
    size_t inOnnxTensorSize = 1;
    for (unsigned j = 0; j < inOnnxTensor.dimensions; ++j) {
      inOnnxTensorDims[j] = inOnnxTensor.shape[j];
      inOnnxTensorSize *= inOnnxTensorDims[j];
    }

    if (inOnnxTensorSize > inPhPtr->getType()->size()) {
      std::stringstream ss;
      for (const auto j : inOnnxTensorDims) {
        ss << j << ", ";
      }
      ss << " vs ";
      auto sizes = inPhPtr->getType()->dims();
      for (const auto j : sizes) {
        ss << j << ", ";
      }
      LOG(ERROR) << "Input tensor is too large: " << inOnnxTensorSize << " vs "
                 << inPhPtr->getType()->size() << ": " << inOnnxTensor.name
                 << ", shape: " << ss.str();
      return ONNXIFI_STATUS_INVALID_SHAPE;
    }

    // Only allocate a tensor if insufficient backing storage is provided.
    unsigned elementSize = inPhPtr->getType()->getElementSize();
    size_t onnxBytes = inOnnxTensorSize * elementSize;
    if (inPhPtr->dims().equals(inOnnxTensorDims)) {
      ctx->getPlaceholderBindings()->insert(
          inPhPtr, Tensor(inOnnxBuffer, inPhPtr->getType()));
    } else if (GlowEnablePartialTensors &&
               backendPtr_->getBackend().supportsPartialTensors() &&
               inOnnxBuffer && inOnnxTensorSize > 0) {
      // We have a partial input buffer.  Create a padded unowned tensor that
      // remembers the actual size of the input.
      ctx->getPlaceholderBindings()->insert(
          inPhPtr, Tensor(inOnnxBuffer, inPhPtr->getType(), onnxBytes));
    } else if (!inOnnxBuffer && inPhPtr->getType()->size() <=
                                    zeroLengthSequence_.getType().size()) {
      ctx->getPlaceholderBindings()->insert(
          inPhPtr, Tensor((void *)(zeroLengthSequence_.getUnsafePtr()),
                          inPhPtr->getType()));
    } else {
      Tensor *inputTensor = tensorPool_.get(inPhPtr->getType());
      if (!inputTensor) {
        DLOG(FATAL) << "Tensorpool tensor not found for input "
                    << inOnnxTensor.name;
        return ONNXIFI_STATUS_INTERNAL_ERROR;
      }
      // We want fresh DeviceResidencyInfo for this fresh Tensor.
      inputTensor->resetDeviceInfo();
      // Copy the input from onnxTensorDescriptor unless it has a NULL buffer
      // pointer (which is a valid case if the tensor is empty).
      if (inOnnxBuffer) {
        memcpy(inputTensor->getUnsafePtr(), inOnnxBuffer, onnxBytes);
        // Pad remaining space with zeroes.
        memset(inputTensor->getUnsafePtr() + onnxBytes, 0,
               inputTensor->getSizeInBytes() - onnxBytes);
      } else {
        inputTensor->zero();
      }
      ctx->getPlaceholderBindings()->insert(inPhPtr, inputTensor);
    }
  }

  size_t seq = 0;
  if (GlowSaveOnnxifiIO) {
    seq = ioDumpCounter_++;
    std::stringstream ss;
    ss << "input_" << seq << ".onnx";
    std::ofstream of(ss.str(), std::ios::binary);
    if (!of) {
      LOG(ERROR) << "Cannot create input file " << ss.str();
    } else {
      ONNX_NAMESPACE::GraphProto inputG;
      for (const auto &p : ctx->getPlaceholderBindings()->pairs()) {
        auto *t = inputG.add_initializer();
        const auto &inputTensor = *p.second;
        size_t unpaddedSize = inputTensor.getUnpaddedSizeInBytes();
        size_t tensorSize = inputTensor.getSizeInBytes();
        if (unpaddedSize == tensorSize) {
          ONNXModelWriter::writeTensor(inputTensor, t);
        } else {
          // If the input is a partial tensor, then save only the part that has
          // data.
          auto ty = inputTensor.getType();
          auto dims = ty.dims().vec();
          dims[0] = dims[0] * unpaddedSize / tensorSize;
          const auto &resized = inputTensor.getUnowned(dims);
          ONNXModelWriter::writeTensor(resized, t);
          VLOG(1) << "Writing partial tensor " << p.first->getName().str()
                  << " full size=" << inputTensor.getType().toString()
                  << " partial size=" << inputTensor.getUnpaddedSizeInBytes()
                  << " resized size=" << resized.getType().toString();
        }
        t->set_name(p.first->getName());
      }
      std::string buffer;
      inputG.SerializeToString(&buffer);
      of << buffer;
    }
  }

  TRACE_EVENT_SCOPE_END_NAMED(aiEvent);
  TRACE_EVENT_SCOPE_NAMED(traceContext, TraceLevel::RUNTIME,
                          "setOnnxifiOutputs", soEvent);

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
    std::vector<dim_t> outOnnxTensorDims(outOnnxTensor.dimensions);
    dim_t outOnnxTensorSize = 1;
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
  TRACE_EVENT_SCOPE_END_NAMED(soEvent);

  auto ret = run(std::move(ctx), outputEvent, traceEvents);
  if (GlowSaveOnnxifiIO) {
    // We need to wait for the execution to finish in order to extract output
    // values.
    outputEvent->wait();
    std::stringstream ss;
    ss << "output_" << seq << ".onnx";
    std::ofstream of(ss.str(), std::ios::binary);
    if (!of) {
      LOG(ERROR) << "Cannot create output file " << ss.str();
    } else {
      ONNX_NAMESPACE::GraphProto inputG;
      for (unsigned i = 0; i < outputsCount; ++i) {
        const auto &outOnnxTensor = outputDescriptors[i];
        auto *outOnnxBuffer = reinterpret_cast<void *>(outOnnxTensor.buffer);
        auto outPhIt = onnxOutputToPlaceholder_.find(outOnnxTensor.name);
        CHECK(outPhIt != onnxOutputToPlaceholder_.end());
        auto &outPhPtr = outPhIt->getValue();
        Tensor outputTensor(outOnnxBuffer, outPhPtr->getType());
        auto *t = inputG.add_initializer();
        ONNXModelWriter::writeTensor(outputTensor, t);
        t->set_name(outPhPtr->getName());
      }
      std::string buffer;
      inputG.SerializeToString(&buffer);
      of << buffer;
    }
  }

  return ret;
}

void Graph::setTraceEvents(onnxTraceEventList *traceEvents,
                           TraceContext *traceContext) {
  if (!traceEvents || !traceContext) {
    return;
  }

  /// Internally we use steady_clock, but our interface is system_clock
  /// timestamps. Do a simple conversion.
  auto steadyTS = TraceEvent::now();
  auto systemTS = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();

  // Timestamps are uint64_t so branch rather than use abs(), we want to make
  // sure we always subtract the smaller from the larger value to avoid
  // underflowing the uint64_t. Then if the timestamp should be moved backwards
  // negate the result.
  int64_t offset =
      steadyTS > systemTS ? -(steadyTS - systemTS) : (systemTS - steadyTS);
  TRACE_EVENT_SCOPE(traceContext, TraceLevel::RUNTIME,
                    "Onnxifi::setTraceEvents");

  std::vector<onnxTraceEvent *> traceEventsVec;
  for (const auto &glowTraceEvent : traceContext->getTraceEvents()) {
    auto *traceEvent = new onnxTraceEvent();
    traceEvent->eventType = glowTraceEvent.type;
    traceEvent->timestamp = glowTraceEvent.timestamp + offset;
    traceEvent->tid = glowTraceEvent.tid;
    traceEvent->duration = glowTraceEvent.duration;
    size_t nameSize = std::min(glowTraceEvent.name.size(),
                               (size_t)ONNXIFI_TRACE_EVENT_NAME_SIZE);
    strncpy(traceEvent->eventName, glowTraceEvent.name.c_str(), nameSize);
    traceEvent->eventName[nameSize] = '\0';
    traceEventsVec.push_back(traceEvent);
  }

  traceEvents->numEvents = traceEventsVec.size();
  traceEvents->traceEvents = new onnxTraceEvent *[traceEventsVec.size()];
  DCHECK(traceEvents->traceEvents);
  std::copy(traceEventsVec.begin(), traceEventsVec.end(),
            traceEvents->traceEvents);
}

void Graph::releaseTraceEvents(onnxTraceEventList *traceEvents) {
  DCHECK(traceEvents);
  for (uint64_t i = 0; i < traceEvents->numEvents; ++i) {
    onnxTraceEvent *traceEvent = traceEvents->traceEvents[i];
    delete traceEvent;
  }

  delete[] traceEvents->traceEvents;
}

Graph::Graph(BackendPtr backendPtr) : backendPtr_(backendPtr) {}

} // namespace onnxifi
} // namespace glow
