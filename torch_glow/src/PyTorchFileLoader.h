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

#ifndef GLOW_TORCH_GLOW_SRC_PYTORCHFILELOADER_H
#define GLOW_TORCH_GLOW_SRC_PYTORCHFILELOADER_H

#include "PyTorchCommon.h"
#include "glow/Graph/Graph.h"
#include "glow/Support/Error.h"
#include <torch/csrc/jit/serialization/import.h>

namespace glow {

/// Loads PyTorch model from file to JIT IR subgraphs.
class PyTorchFileLoader {
  /// Performs sanity check making sure custom fuse pass succeeded as expected,
  /// \returns error otherwise.
  static Error performSanityCheck();

public:
  /// Takes a model file \p fileName, loads model into torch Module \p module,
  /// \returns error if any.
  static Error loadPyTorchModel(const std::string &fileName,
                                std::shared_ptr<torch::jit::Module> &module);

  /// Takes a model file \p fileName, loads optimized graph and a
  /// stack of \p inputs into Glow Function \p F and fills out input \p
  /// inputPlaceholders, output \p outputPlaceholders placeholders, \returns
  /// error if any. Optionally method performs a sanity check if \p sanityCheck
  /// is set.
  /// Method is thread safe, internally it uses local thread structures for
  /// executing custom fusion pass, registered globally. No other passes or
  /// other treads calling this method will be affected.
  static Error
  loadPyTorchGraph(const std::string &fileName,
                   const std::vector<torch::jit::IValue> &inputs,
                   glow::Function &F,
                   std::vector<glow::Placeholder *> &inputPlaceholders,
                   std::vector<glow::Placeholder *> &outputPlaceholders,
                   bool sanityCheck = true);

  /// Takes a model file \p fileName, loads optimized graph and a
  /// stack of \p inputs into Glow Function \p F and fills out input \p
  /// inputPlaceholders, output \p outputPlaceholders placeholders, \returns
  /// error if any. Method is thread safe.
  static Error parsePyTorchGraphForOnnxTraining(
      const std::string &fileName,
      const std::vector<torch::jit::IValue> &inputs, glow::Function &F,
      std::vector<glow::Placeholder *> &inputPlaceholders,
      std::vector<glow::Placeholder *> &outputPlaceholders);
};

} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_PYTORCHFILELOADER_H
