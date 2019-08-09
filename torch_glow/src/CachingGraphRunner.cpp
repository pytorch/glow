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

#include "CachingGraphRunner.h"

#include "PyTorchModelLoader.h"

#include "glow/Support/Support.h"

#include <torch/csrc/jit/argument_spec.h>
#include <torch/csrc/utils/hash.h>

namespace glow {

namespace {
/// Returns a hash that represents a given PT subgraph represented by the PT
/// node \p node and a set of tensor shape inputs to that subgraph from the \p
/// stack.
size_t getGraphHash(const torch::jit::Node *node,
                    const torch::jit::Stack &stack) {
  const std::shared_ptr<torch::jit::Graph> graph = node->g(at::attr::Subgraph);
  const auto &graphInputs = graph->inputs();
  const auto inputs = torch::jit::last(stack, graphInputs.size());
  torch::jit::CompleteArgumentSpec spec(/*with_grad*/ false, inputs);

  return torch::hash_combine(reinterpret_cast<size_t>(node), spec.hashCode());
}

} // namespace

llvm::Expected<CachingGraphRunner::PerGlowGraphInfo *>
CachingGraphRunner::loadGraphImpl(const torch::jit::Node *node,
                                  torch::jit::Stack &stack) {
  const std::shared_ptr<torch::jit::Graph> graph = node->g(at::attr::Subgraph);
  RETURN_ERR_IF_NOT(graph, "A subgraph couldn't be found for this node");

  const auto &graphInputs = graph->inputs();
  auto inputs = torch::jit::last(stack, graphInputs.size());

  size_t hash = getGraphHash(node, stack);

  // If we already have a Glow function compiled for this graph with and the
  // given inputs then use that.
  auto it = perGlowGraphInfoMap.find(hash);
  if (it != perGlowGraphInfoMap.end()) {
    return it->second.get();
  }

  auto info = llvm::make_unique<PerGlowGraphInfo>();
  info->functionName = strFormat("PTFunction%lu", hash);
  info->hash = hash;
  info->node = node;

  std::unique_ptr<Module> module = llvm::make_unique<Module>();
  Function *f = module->createFunction(info->functionName);

  RETURN_IF_ERR(PyTorchModelLoader::loadJITGraph(
      *f, *graph, inputs, info->inputPlaceholders, info->outputPlaceholders,
      getPyTorchLoaderSettings()));

  glow::CompilationContext cctx;

  RETURN_IF_ERR(hostManager_->addNetwork(std::move(module), cctx));

  perGlowGraphInfoMap.insert({hash, std::move(info)});

  return perGlowGraphInfoMap[hash].get();
}

llvm::Error CachingGraphRunner::runGraphImpl(const PerGlowGraphInfo &info,
                                             torch::jit::Stack &stack) {
  DCHECK_EQ(getGraphHash(info.node, stack), info.hash)
      << "Tried to run a graph with incompatible input shapes.";

  size_t numInputs = info.inputPlaceholders.size();

  auto inputs = torch::jit::last(stack, numInputs);

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
    auto ptT = at::empty(
        sizes, at::TensorOptions().dtype(
                   PyTorchModelLoader::convertGlowType(ph->getType())));
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

llvm::Error CachingGraphRunner::runGraph(const torch::jit::Node *node,
                                         torch::jit::Stack &stack) {
  PerGlowGraphInfo *info;
  ASSIGN_VALUE_OR_RETURN_ERR(info, loadGraphImpl(node, stack));
  RETURN_IF_ERR(runGraphImpl(*DCHECK_NOTNULL(info), stack));
  return llvm::Error::success();
}

CachingGraphRunner::CachingGraphRunner() {
  constexpr size_t numGlowDevices = 1;
  constexpr const char *glowBackendName = "Interpreter";

  std::vector<std::unique_ptr<runtime::DeviceConfig>> deviceConfigs;
  for (int i = 0; i < numGlowDevices; i++) {
    deviceConfigs.push_back(
        llvm::make_unique<runtime::DeviceConfig>(glowBackendName));
  }

  hostManager_ =
      llvm::make_unique<runtime::HostManager>(std::move(deviceConfigs));
}
} // namespace glow
