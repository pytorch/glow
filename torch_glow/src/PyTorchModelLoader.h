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

#ifndef GLOW_IMPORTER_PYTORCH_PYTORCHMODELLOADER_H
#define GLOW_IMPORTER_PYTORCH_PYTORCHMODELLOADER_H

#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/ir.h>

#include "glow/Graph/Graph.h"

class PyTorchModelLoader {
  glow::Module &mod_;

  torch::jit::Graph *subgraph_;
  at::ArrayRef<torch::jit::IValue> &inputs_;

  glow::Function *f_ = nullptr;
  std::vector<glow::Placeholder *> inputPlaceholders_;
  std::vector<glow::Placeholder *> outputPlaceholders_;

  std::unordered_map<const torch::jit::Value *, glow::NodeValue> valueMap_;

public:
  /// Takes a glow::Module \p mod, a jit::Graph \p subgraph to load, and a stack
  /// of \p inputs for the subgraph to be loaded and retains references to those
  /// things to be used during loading.
  PyTorchModelLoader(glow::Module &mod, torch::jit::Graph *subgraph,
                     at::ArrayRef<torch::jit::IValue> &inputs);

  /// Creates a glow::Function that represents the loader's subgraph imported
  /// for the shapes seen in the given inputs stack.
  // TODO: return and error upon failure.
  void load();

  /// Returns whether or not a PyTorch node is supported.
  static bool isNodeSupported(const torch::jit::Node *node);

  glow::Function *getFunction() { return f_; }

  const std::vector<glow::Placeholder *> &getInputPlaceholders() {
    return inputPlaceholders_;
  }

  const std::vector<glow::Placeholder *> &getOutputPlaceholders() {
    return outputPlaceholders_;
  }

private:
  glow::Placeholder *loadValue(const torch::jit::Value *val);
  void loadNode(const torch::jit::Node *ptNode);
};

#endif // GLOW_IMPORTER_PYTORCH_PYTORCHMODELLOADER_H
