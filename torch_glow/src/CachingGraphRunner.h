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

/// For a given PyTorch JIT graph, this class is responsible for maintaining a
/// mapping from PyTorch input information to Glow Function used to run that
/// graph in Glow.
class CachingGraphRunner {
  /// Information that is stored per-Glow graph for running it using
  /// HostManager.
  struct PerGlowGraphInfo {
    /// Input and output placeholders to the Glow function.
    std::vector<glow::Placeholder *> inputPlaceholders;
    std::vector<glow::Placeholder *> outputPlaceholders;

    /// Name of the Glow function maintained by HostManager for this subgraph.
    std::string functionName;
  };

  /// The PyTorch JIT Graph that this CachingGraphRunner caches Glow functions
  /// for.
  torch::jit::Graph *graph_ = nullptr;

  /// The HostManager used to store and run Glow graphs.
  runtime::HostManager *hostManager_ = nullptr;

  /// Mapping from hash of PyTorch inputs to PerGlowGraphInfo for the Glow
  /// function that will run inputs matching that hash.
  std::unordered_map<size_t, std::unique_ptr<PerGlowGraphInfo>>
      perGlowGraphInfoMap_;

  /// Given a PyTorch input stack \p stack, this generates a hash from the
  /// values on the stack and checks to see if a matching function was loaded
  /// previously. If a matching function was loaded previously then its cached
  /// info is returned immediately. Otherwise this loads the
  /// subgraph into the owned HostManager, creates a PerGlowGraphInfo which is
  /// cached for the given inputs, and then \returns this PerGlowGraphInfo.
  llvm::Expected<PerGlowGraphInfo *> loadImpl(torch::jit::Stack &stack);

  /// Given a PerGlowGraphInfo \p info for a subgraph that was previously
  /// loaded, this runs the Glow function that corresponds to that
  /// PerGlowGraphInfo in the shape of the inputs with the given \p stack.
  llvm::Error runImpl(const PerGlowGraphInfo &info,
                      torch::jit::Stack &stack) const;

  /// Given a \p stack of inputs, computes the hash for the inputs on the stack.
  size_t computeGraphHash(const c10::ArrayRef<c10::IValue> inputs) const;

public:
  CachingGraphRunner(torch::jit::Graph *graph,
                     runtime::HostManager *hostManager);

  ~CachingGraphRunner();

  /// Given a PyTorch Stack \p stack of inputs, run he stored PyTorch graph on
  /// those inputs. If this is the first time this PyTorch graph has been run
  /// with inputs matching the hash of those on the stack then this first loads
  /// it as a Glow Function and compiles. \returns error of failure.
  llvm::Error run(torch::jit::Stack &stack);
};
} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_CACHINGGRAPHRUNNER_H
