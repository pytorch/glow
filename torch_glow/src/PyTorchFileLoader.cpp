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

#include "PyTorchFileLoader.h"

#include "FuseKnownPatterns.h"
#include "GlowFuser.h"
#include "PyTorchModelLoader.h"

#include "glow/Support/Error.h"
#include "glow/Support/Support.h"

#include <ATen/core/grad_mode.h>
#include <torch/csrc/jit/passes/lower_graph.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator_options.h>

namespace glow {

namespace {
/// struct keeps pointers to Glow Function and Placeholders as a local thread
/// variable.
struct LocalFusionFunction {
  glow::Function *function{nullptr};
  std::vector<glow::Placeholder *> *inputPlaceholders{nullptr};
  std::vector<glow::Placeholder *> *outputPlaceholders{nullptr};
  const PyTorchLoaderSettings *settings{nullptr};

  void set(glow::Function *f, std::vector<glow::Placeholder *> *in,
           std::vector<glow::Placeholder *> *out,
           const PyTorchLoaderSettings *s) {
    function = f;
    inputPlaceholders = in;
    outputPlaceholders = out;
    settings = s;
  }

  void reset() { set(nullptr, nullptr, nullptr, nullptr); }
};

/// Meyer's singleton for static fusion symbol.
static at::Symbol getFusionSymbol() {
  /// Fusion pass unique symbol.
  static const auto fusionSymbol =
      at::Symbol::fromQualString("glow::LoaderFusionPass");
  return fusionSymbol;
}

/// Local thread variable, used in PyTorch Custom Fusion pass.
static thread_local LocalFusionFunction localFusionInfo;

/// Loads JIT Graph into Glow Function.
Error loadJitGraphToGlowFunction(
    torch::jit::Stack &stack, torch::jit::Graph &graph, glow::Function &f,
    std::vector<glow::Placeholder *> &inputPlaceholders,
    std::vector<glow::Placeholder *> &outputPlaceholders,
    const PyTorchLoaderSettings &settings) {
  const auto &graphInputs = graph.inputs();
  const auto numInputs = graphInputs.size();
  auto inputs = torch::jit::last(stack, numInputs);

  // FileLoader not yet support quantized inputs/outputs.
  // These is just dummy type vectors for API.
  std::vector<c10::ScalarType> dummyOutputType;

  // Load JIT Graph into Glow Function.
  RETURN_IF_ERR(PyTorchModelLoader::loadJITGraph(
      f, graph, inputPlaceholders, outputPlaceholders, dummyOutputType,
      settings, inputs, {}));

  // Remove from stack input parameters.
  torch::jit::drop(stack, numInputs);

  // Lookup placeholders for the output shapes.
  for (auto *ph : outputPlaceholders) {
    // Create an empty tensor with the correct shape.
    auto ptT = glowTypeToEmptyPTTensor(*ph->getType());
    auto var = torch::autograd::make_variable(ptT);
    // Add to stack output parameter.
    stack.push_back(at::IValue(var));
  }

  return Error::success();
}

/// Runs Module forward pass, triggers custom fusion pass if local thread
/// Glow function is set.
Error evaluateModuleGraph(std::shared_ptr<torch::jit::Module> &module,
                          const std::vector<torch::jit::IValue> &inputs) {
  try {
    module->forward(inputs);
  } catch (const std::exception &x) {
    RETURN_ERR(x.what());
  }
  return Error::success();
}

/// Helper struct, which on constructor registers custom fusion pass
/// and operator.
struct RegisterCustomFusionPass {
  RegisterCustomFusionPass() {
    torch::jit::RegisterOperators op({torch::jit::Operator(
        getFusionSymbol(),
        [](const torch::jit::Node *node) -> torch::jit::Operation {
          return [node](torch::jit::Stack &stack) {
            // Get JIT Graph.
            auto graph = node->g(at::attr::Subgraph);
            auto err = loadJitGraphToGlowFunction(
                stack, *graph, *localFusionInfo.function,
                *localFusionInfo.inputPlaceholders,
                *localFusionInfo.outputPlaceholders, *localFusionInfo.settings);
            if (static_cast<bool>(err)) {
              throw std::invalid_argument(ERR_TO_STRING(std::move(err)));
            }
            return 0;
          };
        },
        at::AliasAnalysisKind::PURE_FUNCTION)});

    torch::jit::RegisterPass pass([&](std::shared_ptr<torch::jit::Graph> &g) {
      // Trigger custom fusion pass only if local thread Glow Function is set.
      if (localFusionInfo.settings &&
          localFusionInfo.settings->fusionPassEnabled) {
        glowCustomFuse(g, getFusionSymbol());
      }
    });
  }
};
} // namespace

/*static*/
Error PyTorchFileLoader::loadPyTorchModel(
    const std::string &fileName, std::shared_ptr<torch::jit::Module> &module) {
  try {
    module = std::make_shared<torch::jit::Module>(torch::jit::load(fileName));
  } catch (const std::exception &x) {
    RETURN_ERR(strFormat("Cannot load model from file: %s, , reason: %s",
                         fileName.c_str(), x.what()));
  }
  return Error::success();
}

/*static*/
Error PyTorchFileLoader::loadPyTorchGraph(
    const std::string &fileName, const std::vector<torch::jit::IValue> &inputs,
    glow::Function &F, std::vector<glow::Placeholder *> &inputPlaceholders,
    std::vector<glow::Placeholder *> &outputPlaceholders, bool sanityCheck) {
  // Register custom pass once.
  static RegisterCustomFusionPass fuser;

  PyTorchLoaderSettings settings;
  settings.fusionPassEnabled = true;
  settings.weightFreezingEnabled = false;

  // Convert PyTorch model into JIT Module.
  std::shared_ptr<torch::jit::Module> module;
  RETURN_IF_ERR(PyTorchFileLoader::loadPyTorchModel(fileName, module));

  // Disable gradient nodes generation.
  at::NoGradGuard guard;

  // Set thread local pointer, not null values activate custom fusion pass.
  localFusionInfo.set(&F, &inputPlaceholders, &outputPlaceholders, &settings);
  auto err = evaluateModuleGraph(module, inputs);
  localFusionInfo.reset();

  RETURN_IF_ERR(err);

  return sanityCheck ? performSanityCheck() : Error::success();
}

/*static*/
Error PyTorchFileLoader::parsePyTorchGraphForOnnxTraining(
    const std::string &fileName, const std::vector<torch::jit::IValue> &inputs,
    glow::Function &F, std::vector<glow::Placeholder *> &inputPlaceholders,
    std::vector<glow::Placeholder *> &outputPlaceholders) {

  // Convert PyTorch model into JIT Module.
  std::shared_ptr<torch::jit::Module> module;
  RETURN_IF_ERR(PyTorchFileLoader::loadPyTorchModel(fileName, module));

  // Disable gradient nodes generation.
  at::NoGradGuard guard;

  auto method = module->get_method("forward");
  auto graphAndTensors =
      torch::jit::LowerGraph(*method.graph(), module->_ivalue());

  fuseKnownPatterns(graphAndTensors.first);

  // Parse JIT Graph and load into Glow Function.
  return PyTorchModelLoader::loadJITGraphForOnnxTraining(
      F, *graphAndTensors.first, inputs, graphAndTensors.second,
      inputPlaceholders, outputPlaceholders);
}

// Sanity check, after "fusionSymbol" node, not other nodes should exist.
/*static*/
Error PyTorchFileLoader::performSanityCheck() {
  std::shared_ptr<torch::jit::Graph> subgraph =
      torch::jit::lastExecutedOptimizedGraph();
  size_t fusedNodes = 0, missedNodes = 0;
  const auto symbol = getFusionSymbol();
  for (const auto &node : subgraph->nodes()) {
    if (node->kind() == symbol) {
      ++fusedNodes;
    } else if (fusedNodes) {
      // fusionSymbol node has been found, output missing node information.
      LOG(ERROR) << "Missing node: " << *node;
      ++missedNodes;
    }
  }

  RETURN_ERR_IF_NOT(
      fusedNodes == 1 && missedNodes == 0,
      glow::strFormat("Fused optimized nodes: %lu, missing nodes: %lu",
                      fusedNodes, missedNodes));
  return Error::success();
}

} // namespace glow
