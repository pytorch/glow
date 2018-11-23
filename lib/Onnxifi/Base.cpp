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

#include "llvm/ADT/SmallVector.h"

namespace glow {
namespace onnxifi {

bool BackendId::isOpSupported(Kinded::Kind opKind, ElemKind elementTy) {
  return executionEngine_.isOpSupported(opKind, elementTy);
}

void Backend::runAsync(const std::function<void(void)> &fn) {
  threadPool_.submit(fn);
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

onnxStatus Graph::initGraph(const void *onnxModel, size_t onnxModelSize,
                            uint32_t weightCount,
                            const onnxTensorDescriptorV1 *weightDescriptors) {
  // TODO: support multiple functions here.
  function_ = backendPtr_->getEE().getModule().createFunction("inference");

  // TODO: make better error reporting.
  ONNXIFIModelLoader loader = TEMP_UNWRAP(ONNXIFIModelLoader::parse(
      onnxModel, onnxModelSize, weightCount, weightDescriptors, *function_));

  onnxInputToPlaceholder_ = loader.getInputVarsMapping();
  onnxOutputToPlaceholder_ = loader.getOutputVarsMapping();

  // Emit IR for the graph and compile it.
  backendPtr_->getEE().compile(CompilationMode::Infer, function_);

  return ONNXIFI_STATUS_SUCCESS;
}

void Graph::runAsync(EventPtr inputEvent, EventPtr outputEvent) {
  llvm::DenseMap<Placeholder *, onnxPointer> inputPlaceholderToBuffer =
      inputPlaceholderToBuffer_;
  llvm::DenseMap<Placeholder *, onnxPointer> outputPlaceholderToBuffer =
      outputPlaceholderToBuffer_;

  // Once inputs are copied we can allow processing concurrent requests.
  inputRunMutex_.unlock();

  // Submit graph for asynchronous execution.
  // Concurrency for executing 'run' method is limited by number of threads in
  // the backend specific thread pool.
  backend()->runAsync([inputEvent, outputEvent, inputPlaceholderToBuffer,
                       outputPlaceholderToBuffer, this]() {
    // Wait for all inputs to be ready.
    inputEvent->wait();
    // Run inference.
    this->run(inputPlaceholderToBuffer, outputPlaceholderToBuffer);
    // Signal that the outputs are ready.
    outputEvent->signal();
  });
}

void Graph::run(
    const llvm::DenseMap<Placeholder *, onnxPointer> &inputPlaceholderToBuffer,
    const llvm::DenseMap<Placeholder *, onnxPointer>
        &outputPlaceholderToBuffer) {
  // Copy tensors from the input addresses to the Glow tensors.
  llvm::SmallVector<Tensor *, 4> tensors;
  llvm::SmallVector<Placeholder *, 4> phs;
  for (auto inputVar : inputPlaceholderToBuffer) {
    auto *var = inputVar.first;
    auto *type = var->getType();
    void *inputBuffer = reinterpret_cast<void *>(inputVar.second);
    tensors.push_back(new Tensor(inputBuffer, type));
    phs.push_back(var);
  }

  // Run inference.
  auto &EE = backendPtr_->getEE();
  auto &mod = EE.getModule();
  ctx_.allocate(mod.getPlaceholders());
  updateInputPlaceholders(ctx_, phs, tensors);
  EE.run(ctx_);

  // Tensors do not own underlying memory for input buffer,
  // just delete memory allocated for the tensor object itself.
  for (size_t i = 0; i < tensors.size(); ++i) {
    delete tensors[i];
  }

  // Copy outputs to the addresses specified in the outputPlaceholderToBuffer.
  for (auto outputVar : outputPlaceholderToBuffer) {
    void *outputAddress = reinterpret_cast<void *>(outputVar.second);
    const Tensor *res = ctx_.get(outputVar.first);

    memcpy(outputAddress, res->getUnsafePtr(),
           res->size() * res->getType().getElementSize());
  }
}

onnxStatus Graph::setIO(uint32_t inputsCount,
                        const onnxTensorDescriptorV1 *inputDescriptors,
                        uint32_t outputsCount,
                        const onnxTensorDescriptorV1 *outputDescriptors) {
  // Avoid race on setting inputs/outputs and graph run.
  inputRunMutex_.lock();

  // Build name to buffer mapping for inputs and outputs from scratch.
  inputPlaceholderToBuffer_.clear();
  outputPlaceholderToBuffer_.clear();

  // Process inputs.
  for (unsigned i = 0; i < inputsCount; ++i) {
    const auto &in = inputDescriptors[i];
    // TODO: Fix this.
    // This check is to avoid issues when weight is passed in input descriptors.
    // The issue needs to be fixed on the caller side first. Once it is fixed
    // we'd need to handle missing variable accordingly here, e.g., return
    // ONNXIFI_STATUS_UNIDENTIFIED_NAME.
    if (!onnxInputToPlaceholder_.count(in.name)) {
      continue;
    }

    auto *input = onnxInputToPlaceholder_[in.name];
    inputPlaceholderToBuffer_.insert({input, in.buffer});
  }

  // Process outputs.
  for (unsigned i = 0; i < outputsCount; ++i) {
    const auto &out = outputDescriptors[i];

    if (!onnxOutputToPlaceholder_.count(out.name)) {
      return ONNXIFI_STATUS_UNIDENTIFIED_NAME;
    }

    auto *output = onnxOutputToPlaceholder_[out.name];
    outputPlaceholderToBuffer_.insert({output, out.buffer});
  }

  return ONNXIFI_STATUS_SUCCESS;
}

} // namespace onnxifi
} // namespace glow
