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

llvm::Error CachingGraphRunner::runGraph(const torch::jit::Node *node,
                                         torch::jit::Stack &stack) {
  // If this is the first time this subgraph has been run then create a new
  // GraphInfo object to store information about it.

  const std::shared_ptr<torch::jit::Graph> graph = node->g(at::attr::Subgraph);
  const auto &graphInputs = graph->inputs();
  const auto numInputs = graphInputs.size();
  auto inputs = torch::jit::last(stack, numInputs);
  const char *const functionName = "PyTorchFunction";

  glow::Function *f = nullptr;
  executionEngine_.setBackendName(executionEngine_.getBackendName());
  auto &mod = executionEngine_.getModule();
  f = mod.createFunction(functionName);
  std::vector<glow::Placeholder *> inputPlaceholders;
  std::vector<glow::Placeholder *> outputPlaceholders;

  RETURN_IF_ERR(PyTorchModelLoader::loadJITGraph(
      *f, *graph, inputs, inputPlaceholders, outputPlaceholders));

  glow::CompilationContext cctx;
  executionEngine_.compile(cctx);

  glow::PlaceholderBindings bindings;
  for (size_t i = 0; i < inputs.size(); ++i) {
    glow::Placeholder *ph = inputPlaceholders[i];
    glow::TypeRef ty = ph->getType();
    glow::Tensor t(inputs[i].toTensor().data_ptr(), ty);
    bindings.insert(ph, std::move(t));
  }

  std::vector<at::IValue> outputs;
  for (auto *ph : outputPlaceholders) {
    std::vector<int64_t> sizes;
    for (auto size : ph->dims()) {
      sizes.push_back(static_cast<int64_t>(size));
    }
    auto ptT = at::empty(sizes);
    glow::Tensor t(ptT.data_ptr(), ph->getType());
    outputs.push_back(std::move(ptT));
    bindings.insert(ph, std::move(t));
  }

  executionEngine_.run(bindings);

  torch::jit::drop(stack, numInputs);
  for (auto &output : outputs) {
    auto var = torch::autograd::make_variable(output.toTensor());
    stack.push_back(at::IValue(var));
  }

  return llvm::Error::success();
}
