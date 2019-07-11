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

#ifndef GLOW_IMPORTER_PYTORCH_CACHINGGRAPHRUNNER_H
#define GLOW_IMPORTER_PYTORCH_CACHINGGRAPHRUNNER_H

#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/ir.h>

#include "glow/ExecutionEngine/ExecutionEngine.h"

/// All information per graph.
struct GraphInfo {
  std::shared_ptr<torch::jit::Graph> subgraph;
  GraphInfo(const torch::jit::Node *node);
};

/// Responsible for maintaining a mapping from PyTorch subgraphs and their
/// unique input types to compiled Glow Functions.
class CachingGraphRunner {
  /// Map of from PyTorch JIT Node containing contracted JIT subgraph to
  /// to GraphInfo containing information relevent to Glow about that subgraph.
  std::unordered_map<const torch::jit::Node *, GraphInfo> jitNodeToInfoMap_;
  glow::ExecutionEngine executionEngine_;

public:
  CachingGraphRunner() = default;

  /// Given a PyTorch glow::CompilationGroup Node \p node that contains a
  /// PyTorch subgraph and corresponding PyTorch Stack \p stack of inputs, run
  /// that subgraph on those inputs. If this is the first time this node has
  /// been seen then this first loads it as a Glow Function and compiles.
  void runGraph(const torch::jit::Node *node, torch::jit::Stack &stack);
};

#endif // GLOW_IMPORTER_PYTORCH_CACHINGGRAPHRUNNER_H
