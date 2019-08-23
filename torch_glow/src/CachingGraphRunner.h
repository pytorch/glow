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

#include "glow/Runtime/HostManager/HostManager.h"

#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/ir.h>

namespace glow {

/// Responsible for maintaining a mapping from PyTorch subgraphs and their
/// unique input types to a compiled Glow Function.
class CachingGraphRunner {
  struct PerGlowGraphInfo {
    std::vector<glow::Placeholder *> inputPlaceholders;
    std::vector<glow::Placeholder *> outputPlaceholders;

    /// Name of the Glow function maintained by HostManager for this subgraph.
    std::string functionName;

    /// The PyTorch node containing the subgraph that this PerGlowGraphInfo
    /// represents.
    const torch::jit::Node *node;
  };

  std::unique_ptr<runtime::HostManager> hostManager_;
  std::unordered_map<size_t, std::unique_ptr<PerGlowGraphInfo>>
      perGlowGraphInfoMap;

  /// Given a PyTorch node \p node representing a fused subgraph of PyTorch
  /// nodes and an input stack \p stack, this hashes the node and shape of the
  /// inputs and checks to see if a matching function was loaded previously. If
  /// a matching function was loaded previously then its cached
  /// PerGlowGraphInfo is returned immediately. Otherwise this loads the
  /// subgraph into the owned HostManager, creates a PerGlowGraphInfo which is
  /// cached for the given node and the shapes of the inputs, and then \returns
  /// this PerGlowGraphInfo.
  llvm::Expected<PerGlowGraphInfo *> loadGraphImpl(const torch::jit::Node *node,
                                                   torch::jit::Stack &stack);

  /// Given a PerGlowGraphInfo \p info for a subgraph that was previously
  /// loaded, this runs the Glow function that corresponds to that
  /// PerGlowGraphInfo in the shape of the inputs with the given \p stack.
  llvm::Error runGraphImpl(const PerGlowGraphInfo &info,
                           torch::jit::Stack &stack);

  CachingGraphRunner();

public:
  /// Given a PyTorch glow::FusionGroup Node \p node that contains a
  /// PyTorch subgraph and corresponding PyTorch Stack \p stack of inputs, run
  /// that subgraph on those inputs. If this is the first time this node has
  /// been seen then this first loads it as a Glow Function and compiles.
  /// \returns error of failure.
  llvm::Error runGraph(const torch::jit::Node *node, torch::jit::Stack &stack);

  /// Manages a CachingGraphRunner singleton.
  static CachingGraphRunner *getGraphRunner();
};
} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_CACHINGGRAPHRUNNER_H
