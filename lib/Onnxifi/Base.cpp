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

namespace glow {
namespace onnxifi {
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
    llvm::errs() << "Error when loading protobuf: "
                 << llvm::toString(loaderOrErr.takeError()) << "\n";
    return ONNXIFI_STATUS_UNSUPPORTED_OPERATOR;
  }

  if (!glowBackend_) {
    return ONNXIFI_STATUS_INTERNAL_ERROR;
  }

  glow::lower(function, /* loweredMap */ nullptr, glowBackend_.get());

  // Call the backend's transformPostLowering to match the normal compilation
  // pipeline then DCE any nodes that are no longer needed.
  CompilationOptions opts;
  opts.mode = CompilationMode::Infer;
  if (glowBackend_->transformPostLowering(function, opts)) {
    glow::DCE(function);
  }

  const auto &nodes = function->getNodes();

  for (const auto &node : nodes) {
    if (!glowBackend_->isOpSupported(node)) {
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
                              EventPtr outputEvent) {
  auto ctx = llvm::make_unique<ExecutionContext>();

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
      return ONNXIFI_STATUS_INVALID_SHAPE;
    }

    // Only re-allocate a tensor in case padding is required.
    // Otherwise just back the tensor by memory provided by the caller.
    Tensor inputTensor;
    if (inPhPtr->dims().equals(inOnnxTensorDims)) {
      inputTensor = Tensor(inOnnxBuffer, inPhPtr->getType());
    } else {
      inputTensor = Tensor(inPhPtr->getType());
      unsigned elementSize = inPhPtr->getType()->getElementSize();
      char *onnxBuffer = static_cast<char *>(inOnnxBuffer);
      std::copy(onnxBuffer, onnxBuffer + inOnnxTensorSize * elementSize,
                inputTensor.getUnsafePtr());
    }

    ctx->getPlaceholderBindings()->insert(inPhPtr, std::move(inputTensor));
  }

  std::unordered_map<Placeholder *, onnxTensorDescriptorV1>
      phNameToOnnxTensorOutputs;

  // Create tensors for output placeholders
  for (unsigned i = 0; i < outputsCount; ++i) {
    const auto &outOnnxTensor = outputDescriptors[i];

    auto outPhIt = onnxOutputToPlaceholder_.find(outOnnxTensor.name);
    if (outPhIt == onnxOutputToPlaceholder_.end()) {
      return ONNXIFI_STATUS_UNIDENTIFIED_NAME;
    }

    auto &outPhPtr = outPhIt->getValue();

    phNameToOnnxTensorOutputs[outPhPtr] = outOnnxTensor;

    Tensor t(outPhPtr->getType());

    ctx->getPlaceholderBindings()->insert(outPhPtr, std::move(t));
  }

  return run(std::move(ctx), outputEvent, std::move(phNameToOnnxTensorOutputs));
}

Graph::Graph(BackendPtr backendPtr) : backendPtr_(backendPtr) {}

} // namespace onnxifi
} // namespace glow
