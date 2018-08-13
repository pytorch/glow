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

#include "glow/Importer/ONNXIFILoader.h"

#include "llvm/ADT/SmallVector.h"

namespace glow {
namespace onnxifi {

bool BackendId::isOpSupported(const Node &node) {
  // TODO: add support for node with multiple outputs.
  return executionEngine_.isOpSupported(node.getKind(), node.getElementType(0));
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

  std::unique_ptr<ModelLoader> loader = ModelLoader::parse(
      onnxModel, onnxModelSize, weightCount, weightDescriptors, *function_);
  // TODO: make better error reporting.
  if (!loader) {
    return ONNXIFI_STATUS_INTERNAL_ERROR;
  }

  onnxNameToInputVar_ = loader->getInputVarsMapping();
  onnxNameToOutputNode_ = loader->getOutputVarsMapping();

  // Emit IR for the graph and compile it.
  backendPtr_->getEE().compile(CompilationMode::Infer, function_);

  return ONNXIFI_STATUS_SUCCESS;
}

onnxStatus Graph::run() {
  // Copy tensors from the input addresses to the Glow tensors.
  llvm::SmallVector<Tensor *, 4> tensors;
  llvm::SmallVector<Variable *, 4> vars;
  for (auto inputVar : inputVarToBuffer_) {
    auto *var = inputVar.first;
    auto *type = var->getType();
    void *inputBuffer = reinterpret_cast<void *>(inputVar.second);
    tensors.push_back(new Tensor(inputBuffer, type));
    vars.push_back(var);
  }

  // Run inference.
  backendPtr_->getEE().run(vars, tensors);

  // Copy outputs to the addresses specified in the outputNodeToBuffer_.
  for (auto outputVar : outputNodeToBuffer_) {
    void *outputAddress = reinterpret_cast<void *>(outputVar.second);
    const Tensor &res = outputVar.first->getVariable()->getPayload();

    memcpy(outputAddress, res.getUnsafePtr(),
           res.size() * res.getType().getElementSize());
  }

  return ONNXIFI_STATUS_SUCCESS;
}

onnxStatus Graph::setIO(uint32_t inputsCount,
                        const onnxTensorDescriptorV1 *inputDescriptors,
                        uint32_t outputsCount,
                        const onnxTensorDescriptorV1 *outputDescriptors) {
  // Process inputs.
  for (unsigned i = 0; i < inputsCount; ++i) {
    const auto &in = inputDescriptors[i];
    // TODO: Fix this.
    // This check is to avoid issues when weight is passed in input descriptors.
    // The issue needs to be fixed on the caller side first. Once it is fixed
    // we'd need to handle missing variable accordingly here, e.g., return
    // ONNXIFI_STATUS_UNIDENTIFIED_NAME.
    if (!onnxNameToInputVar_.count(in.name)) {
      continue;
    }

    auto *input = onnxNameToInputVar_[in.name];
    inputVarToBuffer_.insert({input, in.buffer});
  }

  // Process outputs.
  for (unsigned i = 0; i < outputsCount; ++i) {
    const auto &out = outputDescriptors[i];

    if (!onnxNameToOutputNode_.count(out.name)) {
      return ONNXIFI_STATUS_UNIDENTIFIED_NAME;
    }

    auto *output = onnxNameToOutputNode_[out.name];
    outputNodeToBuffer_.insert({output, out.buffer});
  }

  return ONNXIFI_STATUS_SUCCESS;
}

} // namespace onnxifi
} // namespace glow
