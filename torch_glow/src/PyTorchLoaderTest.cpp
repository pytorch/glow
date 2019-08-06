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

#include "PyTorchFileLoader.h"
#include "PyTorchModelLoader.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "gtest/gtest.h"
#include <torch/csrc/jit/pass_manager.h>
#include <torch/csrc/jit/passes/graph_fuser.h>

#ifndef GLOW_DATA_PATH
#define GLOW_DATA_PATH
#endif

namespace {
static thread_local glow::Function *localFunction = nullptr;

// Find all Save nodes and their placeholders.
void findInputAndOutputPlaceholders(
    glow::Function &f, std::vector<glow::Placeholder *> &inputPlaceholders,
    std::vector<glow::Placeholder *> &outputPlaceholders) {
  std::unordered_set<glow::Placeholder *> out;
  for (const auto &node : f.getNodes()) {
    if (const glow::SaveNode *SN = llvm::dyn_cast<glow::SaveNode>(&node)) {
      out.insert(llvm::cast<glow::Placeholder>(SN->getOutput().getNode()));
    }
  }

  for (auto *ph : f.findPlaceholders()) {
    if (out.count(ph)) {
      outputPlaceholders.push_back(ph);
    } else {
      inputPlaceholders.push_back(ph);
    }
  }
}

// Convert JIT Graph into Glow Function.
void convertJitGraphToGlowFunction(torch::jit::Stack &stack,
                                   torch::jit::Graph &graph,
                                   glow::Function &f) {
  const auto &graphInputs = graph.inputs();
  const auto numInputs = graphInputs.size();
  auto inputs = torch::jit::last(stack, numInputs);

  // Load JIT Graph into Glow Function.
  std::vector<glow::Placeholder *> inputPlaceholders;
  std::vector<glow::Placeholder *> outputPlaceholders;
  llvm::Error errPtr = glow::PyTorchModelLoader::loadJITGraph(
      f, graph, inputs, inputPlaceholders, outputPlaceholders);

  EXPECT_FALSE(errPtr);

  // Remove from stack input parameters.
  torch::jit::drop(stack, numInputs);
  // Lookup placeholders for the output shapes.
  for (auto *ph : outputPlaceholders) {
    std::vector<int64_t> sizes;
    for (auto size : ph->dims()) {
      sizes.push_back(static_cast<int64_t>(size));
    }
    // Create an empty tensor with the correct shape.
    auto ptT = at::empty(sizes);
    auto var = torch::autograd::make_variable(ptT);
    // Add to stack output parameter.
    stack.push_back(at::IValue(var));
  }
}
} // namespace

TEST(PyTorchModelLoader, Model) {
  const std::string fileName{GLOW_DATA_PATH
                             "tests/models/pytorchModels/resnet18.pt"};
  std::shared_ptr<torch::jit::script::Module> module;
  llvm::Error errPtr = llvm::Error::success();
  glow::PyTorchFileLoader fileLoader(fileName, module, &errPtr);
  EXPECT_FALSE(errPtr);

  std::vector<torch::jit::IValue> vec;
  auto emptyTensor = at::empty({1, 3, 224, 224});
  vec.push_back(torch::autograd::make_variable(emptyTensor));
  module->forward(vec);

  std::shared_ptr<torch::jit::Graph> subgraph =
      torch::jit::lastExecutedOptimizedGraph();

  glow::Module mod;
  auto *f = mod.createFunction("PyTorchFunction");
  std::vector<glow::Placeholder *> inputPlaceholders;
  std::vector<glow::Placeholder *> outputPlaceholders;

  // Add self parameter.
  vec.insert(vec.begin(), module->module_object());

  at::ArrayRef<torch::jit::IValue> inputs(vec);
  errPtr = glow::PyTorchModelLoader::loadJITGraph(
      *f, *subgraph, inputs, inputPlaceholders, outputPlaceholders);
  EXPECT_FALSE(!errPtr); // TODO
}

TEST(PyTorchModelLoader, Fuser) {
  const std::string fileName{GLOW_DATA_PATH
                             "tests/models/pytorchModels/resnet18.pt"};
  std::shared_ptr<torch::jit::script::Module> module;
  llvm::Error errPtr = llvm::Error::success();
  glow::PyTorchFileLoader fileLoader(fileName, module, &errPtr);
  EXPECT_FALSE(errPtr);
  std::shared_ptr<torch::jit::Graph> graph;
  static auto testSymbol = at::Symbol::fromQualString("glow::LoaderTest");

  glow::Module mod;
  localFunction = mod.createFunction("GlowFunction");

  LOG(INFO) << "Register operator and pass";
  {
    auto options = c10::OperatorOptions();
    options.setAliasAnalysis(at::AliasAnalysisKind::PURE);
    torch::jit::RegisterOperators op({torch::jit::Operator(
        testSymbol,
        [](const torch::jit::Node *node) {
          return [node](torch::jit::Stack &stack) {
            // Get JIT Graph.
            auto graph = node->g(at::attr::Subgraph);
            convertJitGraphToGlowFunction(stack, *graph, *localFunction);
            return 0;
          };
        },
        options)});

    torch::jit::RegisterPass pass([&](std::shared_ptr<torch::jit::Graph> &g) {
      torch::jit::CustomFuseGraph(g, glow::PyTorchModelLoader::isNodeSupported,
                                  testSymbol);
    });
  }

  LOG(INFO) << "Running forward method...";
  std::vector<torch::jit::IValue> vec;
  auto emptyTensor = at::empty({1, 3, 224, 224});
  vec.push_back(torch::autograd::make_variable(emptyTensor));

  module->forward(vec);

  // Find all Save nodes and their placeholders.
  std::vector<glow::Placeholder *> inputPlaceholders;
  std::vector<glow::Placeholder *> outputPlaceholders;
  findInputAndOutputPlaceholders(*localFunction, inputPlaceholders,
                                 outputPlaceholders);

  EXPECT_EQ(inputPlaceholders.size(), 1);
  localFunction = nullptr;
}
