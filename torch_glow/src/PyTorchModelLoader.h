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

/// Loads PyTorch JIT IR subgraphs as a Glow Function.
class PyTorchModelLoader {
  /// Glow module in which the loaded Glow Function will be created.
  glow::Module *mod_ = nullptr;

  /// The PyTorch subgraph being loaded.
  torch::jit::Graph *subgraph_ = nullptr;

  /// The reference PyTorch inputs used for loading. This is required for shape
  /// information.
  at::ArrayRef<torch::jit::IValue> *inputs_ = nullptr;

  /// The Glow function that will created.
  glow::Function *f_ = nullptr;

  /// Glow Placeholders corresponding to stack inputs from PyTorch.
  std::vector<glow::Placeholder *> inputPlaceholders_;

  /// Glow Placeholders corresponding to stack outputs to back PyTorch.
  std::vector<glow::Placeholder *> outputPlaceholders_;

  /// Mapping from PyTorch Values to Glow NodeValues created during loading.
  std::unordered_map<const torch::jit::Value *, glow::NodeValue> valueMap_;

  /// A mapping from jit Symbols representing jit operators to the
  /// PyTorchModelLoader method that is used to load that operator.
  std::unordered_map<torch::jit::Symbol,
                     std::function<void(const torch::jit::Node *)>>
      nodeLoaderMapping_;

public:
  PyTorchModelLoader();

  /// Takes a glow::Module \p mod, a jit::Graph \p subgraph to load, and a stack
  /// of \p inputs for the subgraph to be loaded and retains references to those
  /// things to be used during loading.
  PyTorchModelLoader(glow::Module *mod, torch::jit::Graph *subgraph,
                     at::ArrayRef<torch::jit::IValue> *inputs);

  /// Creates a glow::Function that represents the loader's subgraph imported
  /// for the shapes seen in the given inputs stack.
  // TODO: return and error upon failure.
  void load();

  /// Returns whether or not a PyTorch node is supported.
  /// NOTE: For now this is just an enumeration of all type of PyTorch nodes
  /// that the loader knows about but doesn't really guarantee that loading will
  /// will succeed because determining this requires more informations such as
  /// shape info that isn't yet available when this is run.
  static bool isNodeSupported(const torch::jit::Node *node);

  /// Get the Glow function that this loader has created.
  glow::Function *getFunction() { return f_; }

  /// \returns the Glow input placeholders for the loaded function in the order
  /// that is expected by the stack of PyTorch inputs when running the function.
  const std::vector<glow::Placeholder *> &getInputPlaceholders() {
    return inputPlaceholders_;
  }

  /// \returns the Glow output placeholders for the loaded function in the order
  /// that is expected by the stack of PyTorch inputs when running the function.
  const std::vector<glow::Placeholder *> &getOutputPlaceholders() {
    return outputPlaceholders_;
  }

private:
  /// Populates nodeLoaderMapping_ with a mapping from jit Symbols representing
  /// jit operators to the PyTorchModelLoader method that is used to load that
  /// operator.
  /// NOTE: This must be called by all PyTorchModelLoader constructors so that
  /// this mapping is available when loading begins.
  void populateNodeLoaderMapping();

  /// Find the Glow NodeValue that maps to a given PyTorch value \p value.
  glow::NodeValue getGlowNodeValue(const torch::jit::Value *value) const;

  /// Returns true if a Glow NodeValue has been created for a given PyTorch
  /// Value \p value.
  bool hasGlowNodeValue(const torch::jit::Value *value) const;

  /// Add a new mapping from the PyTorch Value \p value to the Glow NodeValue
  /// \nodeValue.
  void addGlowNodeValue(const torch::jit::Value *value,
                        glow::NodeValue nodeValue);

  /// Given a PyTorch Value \p value, returns a handle to the tensor backning
  /// the glow Constant that is mapped to this PyTorch Value. This requires that
  /// the a mapping from the given Value to a Glow NodeValue has already been
  /// created and that the NodeValue is the output of a Glow ConstantNode.
  template <typename T>
  glow::Handle<T> getGlowConstantHandle(const torch::jit::Value *value) const;

  /// Creates and \returns a new Glow Placeholder corresponding to the given
  /// PyTorch Value \p value.
  glow::Placeholder *loadValue(const torch::jit::Value *value);

  /// Load a given PyTorch Node \p ptNode.
  void loadNode(const torch::jit::Node *ptNode);

  /// Load a PyTorch Constant node as a Glow Constant.
  void loadConstant(const torch::jit::Node *ptNode);

  /// Load a PyTorch mul node.
  void loadMul(const torch::jit::Node *ptNode);

  /// Load a PyTorch div node.
  void loadDiv(const torch::jit::Node *ptNode);

  /// Load a PyTorch add node.
  void loadAdd(const torch::jit::Node *ptNode);

  /// Load a PyTorch sub node.
  void loadSub(const torch::jit::Node *ptNode);

  /// Load a PyTorch relu node.
  void loadRelu(const torch::jit::Node *ptNode);

  /// Load a PyTorch _convolution node.
  void loadConvolution(const torch::jit::Node *ptNode);

  /// Load a PyTorch batch_norm node.
  void loadBatchNorm(const torch::jit::Node *ptNode);

  /// Load a PyTorch max_pool2d node.
  void loadMaxPool2d(const torch::jit::Node *ptNode);

  /// Load a PyTorch adaptive_avg_pool2d node.
  void loadAdaptiveAvgPool2d(const torch::jit::Node *ptNode);

  /// Load a PyTorch t (transpose) node.
  void loadTranspose(const torch::jit::Node *ptNode);
};

#endif // GLOW_IMPORTER_PYTORCH_PYTORCHMODELLOADER_H
