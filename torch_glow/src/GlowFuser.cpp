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

#include "GlowFuser.h"

#include "FuseKnownPatterns.h"
#include "PyTorchCommon.h"
#include "PyTorchModelLoader.h"
#include "Registration.h"

#include <glog/logging.h>

#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

namespace glow {
namespace {
using isSupportFunc = std::function<bool(torch::jit::Node *)>;

torch::jit::value_list
sortReverseTopological(at::ArrayRef<torch::jit::Value *> inputs,
                       torch::jit::Block *block) {
  torch::jit::value_list result;
  for (auto i : inputs) {
    if (i->node()->owningBlock() == block) {
      result.push_back(i);
    }
  }

  std::sort(result.begin(), result.end(),
            [&](torch::jit::Value *a, torch::jit::Value *b) {
              return a->node()->isAfter(b->node());
            });
  return result;
}

bool canMerge(torch::jit::Node *node, isSupportFunc fn) {
  return node->kind() == torch::jit::prim::Constant || fn(node);
}

#define REQ(cond, ignore, log_info)                                            \
  if (!(cond)) {                                                               \
    if (!(ignore)) {                                                           \
      DLOG(ERROR) << (log_info);                                               \
    }                                                                          \
    return c10::nullopt;                                                       \
  }

// Check if a node is a known not-lowered operator, etc, a tensor generator.
static bool isLogIgnoredOp(const std::string &op_name) {
  static const std::unordered_set<std::string> white_list{"prim::GetAttr",
                                                          "prim::Param"};
  return white_list.count(op_name) > 0;
}

c10::optional<torch::jit::Node *> tryMerge(torch::jit::Node *consumer,
                                           torch::jit::Node *producer,
                                           torch::jit::AliasDb &aliasDb,
                                           isSupportFunc fn, at::Symbol kind) {

  std::string symbol_name_producer = producer->kind().toQualString();
  std::string symbol_name_consumer = consumer->kind().toQualString();
  REQ(canMerge(producer, fn), isLogIgnoredOp(symbol_name_producer),
      "Detected unknown node: " + symbol_name_producer + ".\n");
  REQ(consumer->kind() == kind || canMerge(consumer, fn),
      isLogIgnoredOp(symbol_name_consumer),
      "Detected unknown node: " + symbol_name_consumer + ".\n");

  // Alias checks
  // Requirement:
  // - moveAfterTopologicallyValid(consumer, producer)
  // - One of:
  //   1) Both are in-place ops
  //   2) Consumer is in-place, producer !hasInputWriters
  //   3) Producer is in-place, consumer !hasOutputWriters
  REQ(aliasDb.moveAfterTopologicallyValid(consumer, producer), false,
      "Unable to move after topologically valid.");

  // 1)
  if (!(aliasDb.isMutable(consumer) && aliasDb.isMutable(producer))) {
    // 2)
    if (aliasDb.isMutable(consumer)) {
      REQ(!aliasDb.hasInputWriters(producer), false,
          "Producer does not have input writer when merging.");
      // 3)
    } else if (aliasDb.isMutable(producer)) {
      REQ(!aliasDb.hasOutputWriters(consumer), false,
          "Consumer does not have output writer when merging.");
    }
  }

  if (!consumer->hasAttribute(torch::jit::attr::Subgraph) &&
      consumer->kind() != kind) {
    consumer =
        torch::jit::SubgraphUtils::createSingletonSubgraph(consumer, kind);
  }
  if (producer->kind() == torch::jit::prim::Constant) {
    auto &subgraph = consumer->g(torch::jit::attr::Subgraph);
    torch::jit::Node *in_const = subgraph->createClone(
        producer, [](torch::jit::Value *) -> torch::jit::Value * {
          throw std::runtime_error("unexpected input");
        });
    subgraph->insertNode(in_const);
  } else {
    torch::jit::SubgraphUtils::mergeNodeIntoSubgraph(producer, consumer);
  }
  return consumer;
}
#undef REQ

std::pair<torch::jit::graph_node_list::iterator, bool>
getNewNode(torch::jit::Node *node, torch::jit::AliasDb &aliasDb,
           torch::jit::Block *block, isSupportFunc fn, at::Symbol kind) {
  auto node_inputs = sortReverseTopological(node->inputs(), block);
  for (auto input : node_inputs) {
    if (auto group = tryMerge(node, input->node(), aliasDb, fn, kind)) {
      return {group.value()->reverseIterator(), true};
    }
  }
  return {++node->reverseIterator(), false};
}

void fuseJITNodesToGlow(std::shared_ptr<torch::jit::Graph> graph,
                        isSupportFunc fn, at::Symbol kind) {
  torch::jit::AliasDb aliasDb(graph);
  auto block = graph->block();

  bool is_changed;
  do {
    is_changed = false;
    for (auto it = block->nodes().rbegin(); it != block->nodes().rend();) {
      bool is_changed_thisnode;
      std::tie(it, is_changed_thisnode) =
          getNewNode(*it, aliasDb, block, fn, kind);
      is_changed |= is_changed_thisnode;
    }
  } while (is_changed);
  EliminateCommonSubexpression(graph);
  EliminateDeadCode(graph);
}

} // namespace

void glowCustomFuse(std::shared_ptr<torch::jit::Graph> graph) {
  auto symbol = getGlowSymbol();

  static std::once_flag onceFlag;
  std::call_once(onceFlag, [&symbol]() { registerGlowOp(symbol); });

  glowCustomFuse(graph, symbol);
}

void glowCustomFuse(std::shared_ptr<torch::jit::Graph> graph, at::Symbol kind) {
  // Prepare the graph by fusing known patterns for the model loader.
  // TODO: this should be done only on Glow subgraphs to avoid modifying parts
  // of the graph that Glow will not be running.
  fuseKnownPatterns(graph);

  fuseJITNodesToGlow(graph, PyTorchModelLoader::isNodeSupported, kind);
}

} // namespace glow
