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

#include "HostManagerOnnxifi.h"

namespace glow {
namespace onnxifi {

std::unique_ptr<runtime::HostManager>
HostManagerBackendId::createHostManager(glow::BackendKind kind) {
  std::vector<runtime::DeviceManagerConfig> configs;
  runtime::DeviceManagerConfig config;
  config.deviceConfig = nullptr;
  config.backendKind = kind;
  configs.push_back(std::move(config));
  return llvm::make_unique<runtime::HostManager>(configs);
}

void HostManagerBackendId::runNetwork(const Graph *graph,
                                      std::unique_ptr<ExecutionContext> context,
                                      runtime::ResultCBTy callback) {
  auto hostManagerGraph = static_cast<const HostManagerGraph *>(graph);
  hostManager_->runNetwork(hostManagerGraph->getName(), std::move(context),
                           std::move(callback));
}

onnxStatus HostManagerBackendId::addNetwork(Module *module) {
  auto err = hostManager_->addNetwork(module);

  if (errToBool(std::move(err))) {
    return ONNXIFI_STATUS_INTERNAL_ERROR;
  }

  return ONNXIFI_STATUS_SUCCESS;
}

void HostManagerBackendId::removeNetwork(const Graph *graph) {
  auto hostManagerGraph = static_cast<const HostManagerGraph *>(graph);
  hostManager_->removeNetwork(hostManagerGraph->getName());
}

onnxStatus
HostManagerGraph::initGraph(const void *onnxModel, size_t onnxModelSize,
                            uint32_t weightCount,
                            const onnxTensorDescriptorV1 *weightDescriptors) {

  netName_ = strFormat("onnxifi_function_%lu", makeUniqueGraphId());

  Function *function = m_.createFunction(netName_);

  // TODO: make better error reporting.
  std::unique_ptr<ONNXIFIModelLoader> loader =
      TEMP_EXIT_ON_ERR(ONNXIFIModelLoader::parse(
          onnxModel, onnxModelSize, weightCount, weightDescriptors, *function,
          true /*loadInputsAsPlaceholders*/, backendPtr_->getUseOnnx()));

  onnxInputToPlaceholder_ = loader->getInputVarsMapping();
  onnxOutputToPlaceholder_ = loader->getOutputVarsMapping();

  return backendPtr_->getBackendId()->addNetwork(&m_);
}

onnxStatus
HostManagerGraph::run(std::unique_ptr<ExecutionContext> ctx,
                      EventPtr outputEvent,
                      std::unordered_map<Placeholder *, onnxTensorDescriptorV1>
                          phNameToOnnxTensorOutputs) {
  backendPtr_->getBackendId()->runNetwork(
      this, std::move(ctx),
      [phNameToOnnxTensorOutputs = std::move(phNameToOnnxTensorOutputs),
       outputEvent](runtime::RunIdentifierTy runId, llvm::Error err,
                    std::unique_ptr<ExecutionContext> ctx) {
        // If an Error occurred then log it in errToBool and signal the output
        // event.
        if (errToBool(std::move(err))) {
          outputEvent->signal();
          return;
        }

        for (auto &ph : ctx->getPlaceholderBindings()->pairs()) {
          if (phNameToOnnxTensorOutputs.count(ph.first) == 0) {
            continue;
          }

          auto &outOnnxTensor = phNameToOnnxTensorOutputs.at(ph.first);

          void *outputAddress = reinterpret_cast<void *>(outOnnxTensor.buffer);
          Tensor *res = ph.second;
          memcpy(outputAddress, res->getUnsafePtr(),
                 res->size() * res->getType().getElementSize());
        }

        outputEvent->signal();
      });

  return ONNXIFI_STATUS_SUCCESS;
}

HostManagerGraph::~HostManagerGraph() {
  // Remove network from the BackendId
  backendPtr_->getBackendId()->removeNetwork(this);
}

size_t HostManagerGraph::makeUniqueGraphId() {
  static std::atomic<size_t> nextId{0};
  return nextId++;
}

} // namespace onnxifi
} // namespace glow