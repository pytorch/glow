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

GraphInfo::GraphInfo(const torch::jit::Node *node)
    : subgraph(node->g(at::attr::Subgraph)) {}

void CachingGraphRunner::addGraph(const torch::jit::Node *node) {
  // TODO: just use node* as key
  GraphInfo graphInfo(node);
  jitNodeToInfoMap_.insert({node, std::move(graphInfo)});
}

void CachingGraphRunner::runGraph(const torch::jit::Node *node,
                                  torch::jit::Stack &stack) {
  // Make sure this is a valid id
  assert(jitNodeToInfoMap_.count(node) > 0);
  auto &graphInfo = jitNodeToInfoMap_.at(node);

  const at::ArrayRef<torch::jit::Value *> &graphInputs =
      graphInfo.subgraph->inputs();
  const auto numInputs = graphInputs.size();

  at::ArrayRef<torch::jit::IValue> inputs = torch::jit::last(stack, numInputs);

  // TODO: cache loaderResult so we don't have to recompile every run.
  PyTorchModelLoader loader(executionEngine_.getModule(),
                            graphInfo.subgraph.get(), inputs);
  loader.load();

  glow::Function *f = loader.getFunction();
  std::vector<glow::Placeholder *> inputPlaceholders =
      loader.getInputPlaceholders();
  std::vector<glow::Placeholder *> outputPlaceholders =
      loader.getOutputPlaceholders();

  glow::CompilationContext cctx;
  executionEngine_.compile(f, cctx);

  assert(inputs.size() == inputPlaceholders.size());

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
}
