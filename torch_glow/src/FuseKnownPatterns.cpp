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

#include "FuseKnownPatterns.h"

#include <glog/logging.h>

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>

namespace glow {
namespace {
/// Registers an operator with symbol \p opName but with no implementation.
/// Dummy operators can be used by glow-specific fusion passes prior to loading
/// a glow graph in order to eliminate intermediate values that are unnecessary
/// to Glow such as those created by quantization packing nodes.
void registerDummyOperator(const char *opName) {
  torch::jit::RegisterOperators op({torch::jit::Operator(
      at::Symbol::fromQualString(opName),
      [](const torch::jit::Node *node) -> torch::jit::Operation {
        LOG(FATAL) << "Operator \"" << (*node)
                   << "\" has no implementation and is meant only as a "
                      "placeholder while fusing ops to run with Glow";
      },
      at::AliasAnalysisKind::PURE_FUNCTION)});
}

void removeExceptionsImpl(torch::jit::Block *block) {
  auto nodes = block->nodes().reverse();
  for (auto it = nodes.begin(); it != nodes.end(); it++) {
    if (it->kind() == torch::jit::prim::RaiseException) {
      it.destroyCurrent();
      continue;
    }
    for (auto *subblock : it->blocks()) {
      removeExceptionsImpl(subblock);
    }
  }
}
} // namespace

namespace detail {
/// This pass fuse the quantized::conv2d_prepack + quantized::conv2d generated
/// by JIT back to quantized::unpacked_conv2d since we dont have
/// quantized::conv2d_prepack in glow. However regular packed conv's
/// implementation in glow is still needed.
void fuseConvPrepack(std::shared_ptr<torch::jit::Graph> &graph) {
  std::string convPrepackPattern = R"IR(
graph(%input, %w, %b, %4, %5, %6, %7, %8, %9, %10, %11, %12):
  %prepacked_weight = quantized::conv2d_prepack(%w, %b, %4, %5, %6, %7)
  %res = quantized::conv2d(%input, %prepacked_weight, %8, %9, %10, %7, %11, %12)
  return (%res))IR";

  std::string convFused = R"IR(
graph(%input, %w, %b, %4, %5, %6, %7, %8, %9, %10, %11, %12):
  %res = glow::unpacked_quantized_conv2d(%input, %w, %b, %8, %9, %10, %7, %11, %12)
  return (%res))IR";

  // Replace conv_prepack + conv2d to unpacked_quantized_conv2d
  torch::jit::SubgraphRewriter convToUnpackedConv;
  convToUnpackedConv.RegisterRewritePattern(convPrepackPattern, convFused);
  convToUnpackedConv.runOnGraph(graph);

  std::string conv3DPrepackPattern = R"IR(
graph(%input, %w, %b, %4, %5, %6, %7, %8, %9, %10, %11, %12):
  %prepacked_weight = quantized::conv3d_prepack(%w, %b, %4, %5, %6, %7)
  %res = quantized::conv3d(%input, %prepacked_weight, %8, %9, %10, %7, %11, %12)
  return (%res))IR";

  std::string conv3DFused = R"IR(
graph(%input, %w, %b, %4, %5, %6, %7, %8, %9, %10, %11, %12):
  %res = glow::unpacked_quantized_conv3d(%input, %w, %b, %8, %9, %10, %7, %11, %12)
  return (%res))IR";

  // Replace conv_prepack + conv3d to unpacked_quantized_conv3d
  torch::jit::SubgraphRewriter conv3DToUnpackedConv3D;
  conv3DToUnpackedConv3D.RegisterRewritePattern(conv3DPrepackPattern,
                                                conv3DFused);
  conv3DToUnpackedConv3D.runOnGraph(graph);
}

void fuseLinearPrepack(std::shared_ptr<torch::jit::Graph> &graph) {
  std::string beforePattern = R"IR(
graph(%input, %weights, %bias, %scale, %zero_point):
  %packed_params = quantized::linear_prepack(%weights, %bias)
  %res = quantized::linear(%input, %packed_params, %scale, %zero_point)
  return (%res))IR";

  std::string afterPattern = R"IR(
graph(%input, %weights, %bias, %scale, %zero_point):
  %res = glow::unpacked_quantized_linear(%input, %weights, %bias, %scale, %zero_point)
  return (%res))IR";

  // Replace linear_prepack + quantized::linear to
  // glow::unpacked_quantized_linear
  torch::jit::SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(beforePattern, afterPattern);
  rewriter.runOnGraph(graph);
}

void fuseNumToTensorToNum(std::shared_ptr<torch::jit::Graph> &graph) {
  std::string originalPat = R"IR(
graph(%input):
  %res1 = prim::NumToTensor(%input)
  %res2 = aten::Int(%res1)
  return (%res2))IR";

  std::string replacementPat = R"IR(
graph(%input):
  return (%input))IR";

  torch::jit::SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(originalPat, replacementPat);
  rewriter.runOnGraph(graph);
}

void fuseConcat(std::shared_ptr<torch::jit::Graph> &graph) {
  auto block = graph->block();
  for (auto it = block->nodes().rbegin(); it != block->nodes().rend(); it++) {
    auto *node = *it;
    const auto kind = node->kind();

    if (kind != at::aten::cat && kind != at::aten::stack) {
      continue;
    }

    // Can only fuse nodes with statically known dim input.
    if (!node->is_constant(torch::jit::attr::dim)) {
      continue;
    }

    auto dim = node->get<int64_t>(torch::jit::attr::dim).value();

    // Can only fuse nodes with inputs from prim::ListConstruct.
    auto *inputNode = node->namedInput(torch::jit::attr::tensors)->node();
    if (inputNode->kind() != at::prim::ListConstruct) {
      continue;
    }

    auto symbolS = node->kind() == at::aten::cat ? "prim::FusedConcat"
                                                 : "glow::fused_stack";

    auto *fusedNode = graph->create(torch::jit::Symbol::fromQualString(symbolS),
                                    inputNode->inputs(), /*num_outputs*/ 1);

    fusedNode->i_(torch::jit::attr::dim, dim);

    fusedNode->insertBefore(inputNode);
    fusedNode->output()->copyMetadata(node->output());
    node->output()->replaceAllUsesWith(fusedNode->output());
  }
}

void removeExceptions(std::shared_ptr<torch::jit::Graph> &graph) {
  return removeExceptionsImpl(graph->block());
}

void fuseBranchedLinearPattern(std::shared_ptr<torch::jit::Graph> &graph) {
  // before:
  // graph(%input, %weight, %bias, %c %d):
  //   %1 = aten::dim(%input)
  //   %2 = aten::eq(%1, %c)
  //   %3 = prim::If(%2)
  //     block0():
  //       %4 = aten::t(%weight)
  //       %5 = prim::Constant[value=1]()
  //       %6 = aten::mm(%input, %4)
  //       %7 = aten::add(%bias, %6, %5)
  //       -> (%7)
  //     block1():
  //       %8 = aten::t(%weight)
  //       %9 = aten::matmul(%input, %8)
  //       %10 : Tensor = aten::add_(%9, %bias, %d)
  //       -> (%10)
  //   return (%3)";
  //
  // after:
  // graph(%input, %weight, %bias, %c %d):
  //   %1 = glow::fused_linear(%input, %weight, %bias, %c %d)
  //   return (%1)";

  auto nodes = graph->block()->nodes().reverse();
  for (auto it = nodes.begin(); it != nodes.end(); it++) {
    auto *ifNode = *it;
    if (ifNode->kind() != torch::jit::prim::If) {
      continue;
    }

    // Define all Values we need to find.
    torch::jit::Value *inputValue = nullptr;
    torch::jit::Value *cValue = nullptr;
    torch::jit::Value *weightValue = nullptr;
    torch::jit::Value *biasValue = nullptr;
    torch::jit::Value *dValue = nullptr;

    // step 1: walk upwards from if to get values from aten::eq and aten::dim
    {
      // find aten::eq node input to prim::If node
      auto *eqNode = ifNode->input()->node();
      if (eqNode->kind() != torch::jit::aten::eq) {
        continue;
      }

      // find aten::dim node input to aten::eq node
      torch::jit::Node *dimNode = nullptr;
      auto eqInputs = eqNode->inputs();
      if (eqInputs[0]->node()->kind() == torch::jit::aten::dim) {
        dimNode = eqInputs[0]->node();
        cValue = eqInputs[1];
      } else if (eqInputs[1]->node()->kind() == torch::jit::aten::dim) {
        dimNode = eqInputs[1]->node();
        cValue = eqInputs[0];
      } else {
        continue;
      }

      inputValue = dimNode->input();
    }

    // step 2: walk if-block collecting values and verifying structure
    {
      torch::jit::Value *tOutputValue = nullptr;
      torch::jit::Value *mmOutputValue = nullptr;
      torch::jit::Value *constantOutputValue = nullptr;
      torch::jit::Value *addOutputValue = nullptr;
      size_t numNodes = 0;
      for (auto *node : ifNode->blocks()[0]->nodes()) {
        numNodes++;
        if (node->kind() == torch::jit::aten::t) {
          weightValue = node->input();
          tOutputValue = node->output();
        } else if (node->kind() == torch::jit::prim::Constant) {
          // Make sure the constant value is 1
          if (node->output()->type()->kind() != torch::jit::TypeKind::IntType) {
            continue;
          }
          if (node->i(torch::jit::attr::value) != 1) {
            continue;
          }
          constantOutputValue = node->output();
        } else if (node->kind() == torch::jit::aten::mm) {
          // Get inputValue and check that second input is output of the aten::t
          inputValue = node->inputs()[0];
          if (node->inputs()[1] != tOutputValue) {
            continue;
          }
          mmOutputValue = node->output();
        } else if (node->kind() == torch::jit::aten::add) {
          // Get biasValue and check that the second input is the output of mm
          biasValue = node->inputs()[0];
          if (node->inputs()[1] != mmOutputValue) {
            continue;
          }
          addOutputValue = node->output();
        } else {
          continue;
        }
      }
      if (!(tOutputValue && mmOutputValue && constantOutputValue &&
            addOutputValue && numNodes == 4)) {
        continue;
      }
    }

    // step 3: walk else-block collecting values and verifying structure
    {
      torch::jit::Value *tOutputValue = nullptr;
      torch::jit::Value *matmulOutputValue = nullptr;
      torch::jit::Value *addOutputValue = nullptr;
      size_t numNodes = 0;
      for (auto *node : ifNode->blocks()[1]->nodes()) {
        numNodes++;
        if (node->kind() == torch::jit::aten::t) {
          if (node->input() != weightValue) {
            continue;
          }
          tOutputValue = node->output();
        } else if (node->kind() == torch::jit::aten::matmul) {
          if (node->inputs()[0] != inputValue) {
            continue;
          }
          if (node->inputs()[1] != tOutputValue) {
            continue;
          }
          matmulOutputValue = node->output();
        } else if (node->kind() == torch::jit::aten::add_) {
          if (node->inputs()[0] != matmulOutputValue) {
            continue;
          }
          if (node->inputs()[1] != biasValue) {
            continue;
          }
          dValue = node->inputs()[2];
          addOutputValue = node->output();
        }
      }
      if (!(tOutputValue && matmulOutputValue && addOutputValue &&
            numNodes == 3)) {
        continue;
      }
    }

    // step 4: create a glow::fused_linear
    assert(inputValue && weightValue && biasValue && cValue && dValue);

    std::vector<torch::jit::Value *> fusedLinearInputs = {
        inputValue, weightValue, biasValue, cValue, dValue};

    auto *fusedNode =
        graph->create(torch::jit::Symbol::fromQualString("glow::fused_linear"),
                      fusedLinearInputs, /*num_outputs*/ 1);

    fusedNode->insertAfter(ifNode);
    fusedNode->output()->copyMetadata(ifNode->output());
    ifNode->replaceAllUsesWith(fusedNode);
  }
}
} // namespace detail

void fuseKnownPatterns(std::shared_ptr<torch::jit::Graph> &graph) {
  // Register dummy nodes used by custom fusers.
  static std::once_flag onceFlag;
  std::call_once(onceFlag, []() {
    registerDummyOperator("glow::unpacked_quantized_linear");
    registerDummyOperator("glow::unpacked_quantized_conv2d");
    registerDummyOperator("glow::unpacked_quantized_conv3d");
    registerDummyOperator("glow::fused_stack");
    registerDummyOperator("glow::fused_linear");
  });

  detail::removeExceptions(graph);
  EliminateDeadCode(graph);

  detail::fuseBranchedLinearPattern(graph);
  EliminateDeadCode(graph);

  detail::fuseConcat(graph);
  detail::fuseConvPrepack(graph);
  detail::fuseLinearPrepack(graph);
  detail::fuseNumToTensorToNum(graph);

  EliminateCommonSubexpression(graph);
  EliminateDeadCode(graph);
}
} // namespace glow
