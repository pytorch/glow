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

#include "CachingGraphRunner.h"

#include "glow/Support/Support.h"

#include <mutex>
#include <torch/csrc/jit/argument_spec.h>
#include <torch/csrc/utils/hash.h>

namespace glow {
// TODO: this should also return the list of TensorTypes used to compute the
// hash to check for equality. Will make a nicer wrapper for this in the future.
size_t CachingGraphRunner::computeGraphHash(
    const c10::ArrayRef<c10::IValue> inputs) const {
  // Start off hash with pointer to this CachingGraphRunner to avoid collisions
  // with Glow functions created by other CachingGraphRunners.
  size_t hash = reinterpret_cast<size_t>(this);

  for (auto &input : inputs) {
    if (!input.isTensor()) {
      continue;
    }

    const auto ptTensorType = c10::TensorType::create(input.toTensor());
    size_t tensorHash = std::hash<c10::TensorType>()(*ptTensorType);
    hash = torch::hash_combine(hash, tensorHash);
  }
  return hash;
}

Expected<CachingGraphRunner::PerGlowGraphInfo *>
CachingGraphRunner::loadImpl(torch::jit::Stack &stack) {
  const auto inputs = torch::jit::last(stack, graph_->inputs().size());

  size_t hash = computeGraphHash(stack);

  // If we already have a Glow function compiled for this graph with and the
  // given inputs then use that.
  auto it = perGlowGraphInfoMap_.find(hash);
  if (it != perGlowGraphInfoMap_.end()) {
    return it->second.get();
  }

  auto info = std::make_shared<PerGlowGraphInfo>();
  info->functionName = strFormat("pt_function_%lu", hash);

  std::unique_ptr<Module> module = glow::make_unique<Module>();
  Function *f = module->createFunction(info->functionName);

  RETURN_IF_ERR(PyTorchModelLoader::loadJITGraph(
      *f, *graph_, info->inputPlaceholders, info->outputPlaceholders,
      getPyTorchLoaderSettings(), inputs, {}));

  glow::CompilationContext cctx;

  RETURN_IF_ERR(hostManager_->addNetwork(std::move(module), cctx));

  perGlowGraphInfoMap_[hash] = std::move(info);

  return perGlowGraphInfoMap_[hash].get();
}

Error CachingGraphRunner::runImpl(const PerGlowGraphInfo &info,
                                  torch::jit::Stack &stack) const {
  size_t numInputs = graph_->inputs().size();
  const auto inputs = torch::jit::last(stack, numInputs);

  std::unique_ptr<ExecutionContext> ctx = glow::make_unique<ExecutionContext>();
  auto *bindings = ctx->getPlaceholderBindings();

  // We only hold placeholders for tensor inputs so indexing them is different
  // than indexing all inputs.
  size_t placeholderI = 0;
  for (const auto &input : inputs) {
    if (input.isTensor()) {
      glow::Placeholder *ph = info.inputPlaceholders[placeholderI++];
      glow::TypeRef ty = ph->getType();

      auto pttensor = input.toTensor();
      glow::Tensor t;
      if (!pttensor.is_contiguous()) {
        pttensor = pttensor.contiguous();
        t = glow::Tensor(pttensor.data_ptr(), ty).clone();
      } else {
        t = glow::Tensor(pttensor.data_ptr(), ty);
      }

      bindings->insert(ph, std::move(t));
    } else if (input.isObject()) {
      // Objects are only used for loading attributes at compile time.
      continue;
    } else {
      return MAKE_ERR("Only Tensor and Object IValue inputs are accepted");
    }
  }

  std::vector<at::IValue> outputs;
  for (auto *ph : info.outputPlaceholders) {
    std::vector<int64_t> sizes;
    for (auto size : ph->dims()) {
      sizes.push_back(static_cast<int64_t>(size));
    }

    auto ptT = glowTypeToEmptyPTTensor(*ph->getType());

    glow::Tensor t(ptT.data_ptr(), ph->getType());

    outputs.push_back(std::move(ptT));
    bindings->insert(ph, std::move(t));
  }

  auto err =
      hostManager_->runNetworkBlocking(info.functionName, std::move(ctx));

  torch::jit::drop(stack, numInputs);

  for (auto &output : outputs) {
    auto var = torch::autograd::make_variable(output.toTensor());
    stack.push_back(at::IValue(var));
  }

  return err;
}

Error CachingGraphRunner::run(torch::jit::Stack &stack) {
  PerGlowGraphInfo *info;
  ASSIGN_VALUE_OR_RETURN_ERR(info, loadImpl(stack));
  return runImpl(*DCHECK_NOTNULL(info), stack);
}

Error CachingGraphRunner::runOnly(torch::jit::Stack &stack) {
  if (perGlowGraphInfoMap_.size() != 1) {
    return MAKE_ERR(strFormat(
        "There should be one and only one compiled graph, but got %lu",
        perGlowGraphInfoMap_.size()));
  }
  return runImpl(*DCHECK_NOTNULL(perGlowGraphInfoMap_[0].get()), stack);
}

Error CachingGraphRunner::warmCache(const std::vector<InputMeta> &inputMeta) {
  if (!hostManager_) {
    return MAKE_ERR("Host manager is null!");
  }

  if (!graph_) {
    return MAKE_ERR("No graph found!");
  }

  if (perGlowGraphInfoMap_.size() != 0) {
    return MAKE_ERR(strFormat("There is already a compiled graph!"));
  }

  auto info = std::make_shared<PerGlowGraphInfo>();
  info->functionName = strFormat("PTFunction_precompiled");

  std::unique_ptr<Module> glowModule = llvm::make_unique<Module>();
  Function *f = glowModule->createFunction(info->functionName);

  RETURN_IF_ERR(PyTorchModelLoader::loadJITGraph(
      *f, *graph_, info->inputPlaceholders, info->outputPlaceholders,
      getPyTorchLoaderSettings(), {}, inputMeta));

  glow::CompilationContext cctx;

  RETURN_IF_ERR(hostManager_->addNetwork(std::move(glowModule), cctx));
  // Randomly picked one key. There should be only one element in the map
  // when model is precompiled.
  perGlowGraphInfoMap_[0] = std::move(info);
  return Error::success();
}

CachingGraphRunner::CachingGraphRunner(
    std::shared_ptr<torch::jit::Graph> graph,
    std::shared_ptr<runtime::HostManager> hostManager)
    : graph_(graph), hostManager_(hostManager) {}

CachingGraphRunner::~CachingGraphRunner() {
  // Remove Glow functions saved in HostManager when being destroyed.
  for (auto &kv : perGlowGraphInfoMap_) {
    ERR_TO_BOOL(hostManager_->removeNetwork(kv.second->functionName));
  }
}

} // namespace glow
