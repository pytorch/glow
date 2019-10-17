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

namespace {
static std::mutex graphCacheMutex;
}

Expected<CachingGraphRunner::PerGlowGraphInfo *>
CachingGraphRunner::loadImpl(torch::jit::Stack &stack) {
  const auto inputs = torch::jit::last(stack, graph_->inputs().size());

  size_t hash = computeGraphHash(stack);

  // If we already have a Glow function compiled for this graph with and the
  // given inputs then use that.
  std::lock_guard<std::mutex> guard(graphCacheMutex);
  auto it = perGlowGraphInfoMap_.find(hash);
  if (it != perGlowGraphInfoMap_.end()) {
    return it->second.get();
  }

  auto info = std::make_shared<PerGlowGraphInfo>();
  info->functionName = strFormat("PTFunction%lu", hash);

  std::unique_ptr<Module> module = llvm::make_unique<Module>();
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
  size_t numInputs = info.inputPlaceholders.size();
  const auto inputs = torch::jit::last(stack, numInputs);

  std::unique_ptr<ExecutionContext> ctx = llvm::make_unique<ExecutionContext>();
  auto *bindings = ctx->getPlaceholderBindings();

  for (size_t i = 0; i < numInputs; ++i) {
    glow::Placeholder *ph = info.inputPlaceholders[i];
    glow::TypeRef ty = ph->getType();
    glow::Tensor t(inputs[i].toTensor().data_ptr(), ty);
    bindings->insert(ph, std::move(t));
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

Error CachingGraphRunner::run(const std::string &key,
                              torch::jit::Stack &stack) {
  std::shared_ptr<PerGlowGraphInfo> info;
  {
    std::lock_guard<std::mutex> guard(graphCacheMutex);
    auto it = glowGraphInfoMap.find(key);
    if (it == glowGraphInfoMap.end()) {
      return MAKE_ERR(
          strFormat("Key: %s not found in glowGraphInfoMap!", key.c_str()));
    }
    info = it->second;
  }
  return runImpl(*DCHECK_NOTNULL(info.get()), stack);
}

Error CachingGraphRunner::CompileModule(const torch::jit::script::Module &m,
                                        const std::vector<InputMeta> &inputMeta,
                                        const std::string &opname) {
  if (hostManager_ == nullptr) {
    return MAKE_ERR("Host manager is null!");
  }
  const std::string name = "glow::" + m.name().qualifiedName();

  std::lock_guard<std::mutex> guard(graphCacheMutex);
  // Currently only support one method per module
  auto it = glowGraphInfoMap.find(name);
  if (it != glowGraphInfoMap.end()) {
    // Already compiled
    return Error::success();
  }

  auto info = std::make_shared<PerGlowGraphInfo>();
  info->functionName = strFormat("PTFunction%s", name.c_str());

  auto methods = m.get_methods();
  if (methods.size() != 1) {
    return MAKE_ERR("Currently only support one method each module!");
  }

  std::shared_ptr<torch::jit::Graph> g = nullptr;
  // XXX: Here we assume the fusion node is a top level node. This constraint
  // can be relaxed if needed.
  for (auto node : methods[0].function().graph()->nodes()) {
    if (node->kind().toQualString() == opname) {
      if (!node->hasAttribute(torch::jit::attr::Subgraph)) {
        return MAKE_ERR("Fusion node should have a subgraph!");
      }
      g = node->g(torch::jit::attr::Subgraph);
    }
  }
  if (!g) {
    return MAKE_ERR("No fusion node found");
  }
  std::unique_ptr<Module> glowModule = llvm::make_unique<Module>();
  Function *f = glowModule->createFunction(info->functionName);

  RETURN_IF_ERR(PyTorchModelLoader::loadJITGraph(
      *f, *g, info->inputPlaceholders, info->outputPlaceholders,
      getPyTorchLoaderSettings(), {}, inputMeta));

  glow::CompilationContext cctx;

  RETURN_IF_ERR(hostManager_->addNetwork(std::move(glowModule), cctx));
  glowGraphInfoMap[name] = std::move(info);
  return Error::success();
}

CachingGraphRunner *CachingGraphRunner::getCachingGraphRunner() {
  static CachingGraphRunner runner = CachingGraphRunner();
  return &runner;
}

CachingGraphRunner::CachingGraphRunner()
    : hostManager_(glow::getHostManager()) {}

CachingGraphRunner::CachingGraphRunner(torch::jit::Graph *graph,
                                       runtime::HostManager *hostManager)
    : graph_(graph), hostManager_(hostManager) {}

CachingGraphRunner::~CachingGraphRunner() {
  // Remove Glow functions saved in HostManager when being destroyed.
  for (auto &kv : perGlowGraphInfoMap_) {
    ERR_TO_BOOL(hostManager_->removeNetwork(kv.second->functionName));
  }
  for (auto &kv : glowGraphInfoMap) {
    ERR_TO_BOOL(hostManager_->removeNetwork(kv.second->functionName));
  }
}

} // namespace glow
