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
#include "PyTorchModelLoader.h"
#include "Registration.h"

#include <glog/logging.h>

#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

namespace glow {
namespace {
using IsSupportFunc = std::function<bool(const torch::jit::Node *)>;

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

bool canMerge(torch::jit::Node *node, IsSupportFunc fn,
              torch::jit::Node *consumer) {
  if (node->kind() == torch::jit::prim::Param) {
    return false;
  }

  // Check that the node is supported
  if (!(fn(node) || node->kind() == torch::jit::prim::Constant ||
        node->kind() == torch::jit::prim::GetAttr)) {
    return false;
  }

  // If the node is a producer (has a consumer), check that all non-tensor
  // outputs are only consumed by the consumer.
  for (torch::jit::Value *output : node->outputs()) {
    if (output->type()->isSubtypeOf(torch::jit::TensorType::get())) {
      continue;
    }

    // Producers can have non-tensor outputs as long as they are only consumed
    // by consumer. Consumers cannot have non-tensor outputs.
    if (consumer) {
      for (auto use : output->uses()) {
        if (use.user != consumer) {
          return false;
        }
      }
    } else {
      return false;
    }
  }

  return true;
}

/// Alias checks, takes a \p consumer node and a \p producer node and using \p
/// aliasDB, checks to see if it is valid to fuse the producer into the
/// consumer.
/// Requirement:
/// - moveAfterTopologicallyValid(consumer, producer)
/// - One of:
///   1) Both are in-place ops
///   2) Consumer is in-place, producer !hasInputWriters
///   3) Producer is in-place, consumer !hasOutputWriters
bool aliasChecks(torch::jit::Node *consumer, torch::jit::Node *producer,
                 torch::jit::AliasDb &aliasDb) {
  if (!aliasDb.moveAfterTopologicallyValid(consumer, producer)) {
    return false;
  }

  if (aliasDb.isMutable(consumer) && aliasDb.isMutable(producer)) {
    return true;
  }
  if (aliasDb.isMutable(consumer) && aliasDb.hasInputWriters(producer)) {
    return false;
  }
  if (aliasDb.isMutable(producer) && aliasDb.hasOutputWriters(consumer)) {
    return false;
  }

  return true;
}

// Try to merge producer and consumer into a single fused node.
torch::jit::Node *tryMerge(torch::jit::Node *consumer,
                           torch::jit::Node *producer,
                           torch::jit::AliasDb &aliasDb, IsSupportFunc fn,
                           at::Symbol kind) {
  // Check that producer can be merged
  if (!canMerge(producer, fn, consumer)) {
    return nullptr;
  }

  // Check that consumer can be merged
  if (!(consumer->kind() == kind ||
        canMerge(consumer, fn, /*consumer*/ nullptr))) {
    return nullptr;
  }

  // Check for aliases
  if (!aliasChecks(consumer, producer, aliasDb)) {
    return nullptr;
  }

  // Wrap consumer as a subgraph
  if (!consumer->hasAttribute(torch::jit::attr::Subgraph) &&
      consumer->kind() != kind) {
    consumer =
        torch::jit::SubgraphUtils::createSingletonSubgraph(consumer, kind);
  }

  // Move (or for constants, copy) node into subgraph
  if (producer->kind() == torch::jit::prim::Constant) {
    auto &subgraph = consumer->g(torch::jit::attr::Subgraph);
    torch::jit::Node *inConst = subgraph->createClone(
        producer, [](torch::jit::Value *) -> torch::jit::Value * {
          throw std::runtime_error("unexpected input to Constant node");
        });
    subgraph->insertNode(inConst);
  } else {
    torch::jit::SubgraphUtils::mergeNodeIntoSubgraph(producer, consumer);
  }

  return consumer;
}

std::pair<torch::jit::graph_node_list::iterator, bool>
getNewNode(torch::jit::Node *node, torch::jit::AliasDb &aliasDb,
           torch::jit::Block *block, IsSupportFunc fn, at::Symbol kind) {
  auto nodeInputs = sortReverseTopological(node->inputs(), block);
  for (auto input : nodeInputs) {
    if (auto *group = tryMerge(node, input->node(), aliasDb, fn, kind)) {
      return {group->reverseIterator(), true};
    }
  }
  return {++node->reverseIterator(), false};
}

size_t graphSize(const std::shared_ptr<torch::jit::Graph> graph) {
  size_t size = 0;
  for (auto it = graph->nodes().begin(); it != graph->nodes().end(); ++it) {
    ++size;
  }
  return size;
}

std::shared_ptr<torch::jit::Graph> getSubgraph(torch::jit::Node *n) {
  return n->g(torch::jit::attr::Subgraph);
}

const std::shared_ptr<torch::jit::Graph>
getSubgraph(const torch::jit::Node *n) {
  return n->g(torch::jit::attr::Subgraph);
}

void fuseJITNodesToGlow(std::shared_ptr<torch::jit::Graph> graph,
                        IsSupportFunc fn, at::Symbol kind) {
  torch::jit::AliasDb aliasDb(graph);
  auto block = graph->block();

  bool changed;
  do {
    changed = false;
    for (auto it = block->nodes().rbegin(); it != block->nodes().rend();) {
      bool nodeChanged;
      std::tie(it, nodeChanged) = getNewNode(*it, aliasDb, block, fn, kind);
      changed |= nodeChanged;
    }
  } while (changed);
}

void unmergeSubgraph(torch::jit::Node *subgraphNode) {
  // Inline the graph, replace uses of node outputs and destroy the node
  auto outerGraph = subgraphNode->owningGraph();
  torch::jit::WithInsertPoint guard(subgraphNode);
  const auto subgraphOutputs = insertGraph(
      *outerGraph, *getSubgraph(subgraphNode), subgraphNode->inputs());
  assert(subgraphOutputs.size() >= subgraphNode->outputs().size());
  for (size_t i = 0; i < subgraphNode->outputs().size(); ++i) {
    subgraphNode->outputs()[i]->replaceAllUsesWith(subgraphOutputs[i]);
  }
  subgraphNode->destroy();
}

void unfuseSmallGraphs(std::shared_ptr<torch::jit::Graph> &graph,
                       size_t minFusionGroupSize, at::Symbol kind) {
  bool changed;
  do {
    changed = false;
    for (auto *n : graph->nodes()) {
      if (n->kind() == kind) {
        if (graphSize(getSubgraph(n)) < minFusionGroupSize) {
          changed = true;
          unmergeSubgraph(n);
          break; // start over
        }
      }
    }
  } while (changed);
}

void verifyFusions(const std::shared_ptr<torch::jit::Graph> graph,
                   at::Symbol kind) {
  for (const auto *n : graph->nodes()) {
    if (n->kind() != kind) {
      continue;
    }

    auto g = getSubgraph(n);

    // Verify that all outputs are tensors.
    for (auto output : g->outputs()) {
      if (!output->type()->isSubtypeOf(torch::jit::TensorType::get())) {
        throw std::runtime_error(
            "Glow fusion group should only have Tensor outputs");
      }
    }
  }
}

void glowCustomFuseImpl(std::shared_ptr<torch::jit::Graph> graph,
                        at::Symbol kind, const PyTorchLoaderSettings &settings,
                        IsSupportFunc fn) {
  // Reason for node being blacklisted.
  enum class NodeBlacklistReason {
    Kind,
    Index,
  };

  std::unordered_map<const torch::jit::Node *, NodeBlacklistReason>
      blacklistedNodes;

  size_t i = 0;
  for (const torch::jit::Node *node : graph->nodes()) {
    if (settings.fusionStartIndex >= 0 && i < settings.fusionStartIndex) {
      blacklistedNodes[node] = NodeBlacklistReason::Index;
    }

    if (settings.fusionEndIndex >= 0 && i >= settings.fusionEndIndex) {
      blacklistedNodes[node] = NodeBlacklistReason::Index;
    }

    if (settings.opBlacklist.count(node->kind())) {
      blacklistedNodes[node] = NodeBlacklistReason::Kind;
    }
    i++;
  }

  const auto minFusionGroupSize = settings.minFusionGroupSize;

  // Wrap fn in function that first checks the blacklist.
  IsSupportFunc nodeSupportedFn = [blacklist = std::move(blacklistedNodes),
                                   fn](const torch::jit::Node *ptNode) {
    if (blacklist.count(ptNode)) {
      switch (blacklist.at(ptNode)) {
      case NodeBlacklistReason::Kind:
        LOG(INFO) << "Skipping " << ptNode->kind().toQualString()
                  << " op because its kind is blacklisted";
        break;
      case NodeBlacklistReason::Index:
        LOG(INFO) << "Skipping " << ptNode->kind().toQualString()
                  << " op because it's outside of the fusion range";
        break;
      }
      return false;
    }

    return fn(ptNode);
  };

  Inline(*graph);

  // Prepare the graph by fusing known patterns for the model loader.
  // TODO: this should be done only on Glow subgraphs to avoid modifying parts
  // of the graph that Glow will not be running.
  fuseKnownPatterns(graph);

  fuseJITNodesToGlow(graph, nodeSupportedFn, kind);

  if (minFusionGroupSize > 0) {
    unfuseSmallGraphs(graph, minFusionGroupSize, kind);
  }

  EliminateCommonSubexpression(graph);

  EliminateDeadCode(graph);

  verifyFusions(graph, kind);
}

void registDefaultGlowFusionSymbolOnce() {
  static std::once_flag onceFlag;
  std::call_once(onceFlag, []() { registerGlowOp(getGlowSymbol()); });
}

} // namespace

void glowCustomFuse(std::shared_ptr<torch::jit::Graph> graph) {
  registDefaultGlowFusionSymbolOnce();
  auto symbol = getGlowSymbol();
  const auto &settings = getPyTorchLoaderSettings();
  return glowCustomFuseImpl(graph, symbol, settings,
                            PyTorchModelLoader::isNodeSupported);
}

void glowCustomFuse(std::shared_ptr<torch::jit::Graph> graph,
                    const PyTorchLoaderSettings &settings) {
  registDefaultGlowFusionSymbolOnce();
  auto symbol = getGlowSymbol();
  return glowCustomFuseImpl(graph, symbol, settings,
                            PyTorchModelLoader::isNodeSupported);
}

void glowCustomFuse(std::shared_ptr<torch::jit::Graph> graph, at::Symbol kind) {
  const auto &settings = getPyTorchLoaderSettings();
  return glowCustomFuseImpl(graph, kind, settings,
                            PyTorchModelLoader::isNodeSupported);
}

void glowCustomFuseDebug(std::shared_ptr<torch::jit::Graph> graph,
                         std::vector<std::string> acceptableKinds) {
  registDefaultGlowFusionSymbolOnce();
  auto symbol = getGlowSymbol();

  std::unordered_set<at::Symbol> kindSet;

  for (const auto &kind : acceptableKinds) {
    kindSet.insert(at::Symbol::fromQualString(kind));
  }

  auto fn = [kindSet = std::move(kindSet)](const torch::jit::Node *node) {
    return kindSet.count(node->kind());
  };

  const auto &settings = getPyTorchLoaderSettings();

  return glowCustomFuseImpl(graph, symbol, settings, fn);
}

} // namespace glow
