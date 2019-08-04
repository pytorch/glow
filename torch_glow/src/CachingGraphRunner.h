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

#ifndef GLOW_TORCH_GLOW_SRC_CACHINGGRAPHRUNNER_H
#define GLOW_TORCH_GLOW_SRC_CACHINGGRAPHRUNNER_H

#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/ir.h>

#include "glow/ExecutionEngine/ExecutionEngine2.h"

namespace glow {

/// Responsible for maintaining a mapping from PyTorch subgraphs and their
/// unique input types to compiled Glow Functions.
class CachingGraphRunner {
  /// Glow ExecutionEngine.
  glow::ExecutionEngine2 executionEngine_;

public:
  CachingGraphRunner() = default;

  /// Given a PyTorch glow::FusionGroup Node \p node that contains a
  /// PyTorch subgraph and corresponding PyTorch Stack \p stack of inputs, run
  /// that subgraph on those inputs. If this is the first time this node has
  /// been seen then this first loads it as a Glow Function and compiles.
  /// \returns error of failure.
  llvm::Error runGraph(const torch::jit::Node *node, torch::jit::Stack &stack);
};
} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_CACHINGGRAPHRUNNER_H
