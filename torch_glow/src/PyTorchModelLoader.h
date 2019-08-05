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

#ifndef GLOW_TORCH_GLOW_SRC_PYTORCHMODELLOADER_H
#define GLOW_TORCH_GLOW_SRC_PYTORCHMODELLOADER_H

#include <llvm/Support/Error.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/ir.h>

#include "glow/Graph/Graph.h"

namespace glow {

/// Loads PyTorch JIT IR subgraphs as a Glow Function.
class PyTorchModelLoader {
  /// Glow Function created outside this class.
  glow::Function &F_;

  /// Map from input placeholders to their location on the input stack.
  std::unordered_map<glow::Placeholder *, size_t>
      inputPlaceholdersReverseIndex_;

  /// The reference PyTorch inputs used for loading. This is required for shape
  /// information.
  at::ArrayRef<torch::jit::IValue> &inputs_;

  /// Mapping from PyTorch Values to Glow NodeValues created during loading.
  using ValueMap =
      std::unordered_map<const torch::jit::Value *, glow::NodeValue>;
  ValueMap valueMap_;

  /// Values in the MappingOfMemberFunctions map. These values contain the
  /// information necessary to load PyTorch nodes such as which
  /// PyTorchModelLoader method to use and which inputs should be considered as
  /// constants.
  struct MappingOfMemberFunctionsValue {
    /// The type of functions used to load PyTorch nodes in PyTorchModelLoader.
    using LoadFn =
        llvm::Error (PyTorchModelLoader::*)(const torch::jit::Node *);

    /// Symbols (as strings) that this mapping value is applicable to.
    std::vector<const char *> symbols;

    /// The PyTorchModelLoader method that should be used to load the given
    /// PyTorch node.
    LoadFn loadFn;

    /// The set of inputs that should be loaded as Glow Constants instead of
    /// as placeholders for inference because they should be expected to not
    /// change between inferences. An example would be the weights for
    /// convolutions.
    std::unordered_set<size_t> constInputs;

    MappingOfMemberFunctionsValue(std::vector<const char *> symbolsP,
                                  LoadFn loadFnP,
                                  std::unordered_set<size_t> constInputsP)
        : symbols(symbolsP), loadFn(loadFnP), constInputs(constInputsP) {}
  };

  /// Defines type for mapping Symbols to PyTorchModelLoader member functions
  /// for loading torch::jit::Node objects.
  class MappingOfMemberFunctions
      : public std::unordered_map<torch::jit::Symbol,
                                  MappingOfMemberFunctionsValue> {
  public:
    /// Construct a MappingOfMemberFunctions from a list of
    /// MappingOfMemberFunctionsValues \p initList.
    MappingOfMemberFunctions(
        std::initializer_list<MappingOfMemberFunctionsValue> initList) {
      for (const auto &val : initList) {
        for (const char *symbolStr : val.symbols) {
          auto res = this->insert({at::Symbol::fromQualString(symbolStr), val});
          DCHECK(res.second) << "Duplicate symbol mapping for " << symbolStr;
        }
      }
    }
  };

public:
  /// Returns whether or not a PyTorch node is supported.
  /// NOTE: For now this is just an enumeration of all type of PyTorch nodes
  /// that the loader knows about but doesn't really guarantee that loading
  /// will succeed because determining this requires more informations such as
  /// shape info that isn't yet available when this is run.
  static bool isNodeSupported(const torch::jit::Node *node);

  /// Takes a glow::Function \p F, a jit::Graph \p subgraph to load, and a
  /// stack of \p inputs for the subgraph to be loaded. Output parameters
  /// \p inputPlaceholders and \p outputPlaceholders are filled out.
  /// \returns error on failure
  static llvm::Error
  loadJITGraph(glow::Function &F, torch::jit::Graph &subgraph,
               at::ArrayRef<torch::jit::IValue> &inputs,
               std::vector<glow::Placeholder *> &inputPlaceholders,
               std::vector<glow::Placeholder *> &outputPlaceholders);

private:
  /// Takes a glow::Function \p F, a jit::Graph \p subgraph to load, and a
  /// stack of \p inputs for the subgraph to be loaded. Output parameters
  /// \p inputPlaceholders and \p outputPlaceholders are filled out.
  PyTorchModelLoader(glow::Function &F, torch::jit::Graph &subgraph,
                     at::ArrayRef<torch::jit::IValue> &inputs,
                     std::vector<glow::Placeholder *> &inputPlaceholders,
                     std::vector<glow::Placeholder *> &outputPlaceholders,
                     llvm::Error &error);

  /// Save access to the mapping.
  static const MappingOfMemberFunctions &getSymbolsMapping();

  /// Find the Glow NodeValue that maps to a given PyTorch value \p value.
  llvm::Expected<glow::NodeValue>
  getGlowNodeValue(const torch::jit::Value *value) const;

  /// Returns true if a Glow NodeValue has been created for a given PyTorch
  /// Value \p value.
  bool hasGlowNodeValue(const torch::jit::Value *value) const;

  /// Add a new mapping from the PyTorch Value \p value to the Glow NodeValue
  /// \nodeValue.
  /// \returns error on failure.
  llvm::Error addGlowNodeValue(const torch::jit::Value *value,
                               glow::NodeValue nodeValue);

  /// Given a PyTorch Value \p value, returns a handle to the tensor backning
  /// the glow Constant that is mapped to this PyTorch Value. This requires that
  /// the a mapping from the given Value to a Glow NodeValue has already been
  /// created and that the NodeValue is the output of a Glow ConstantNode.
  template <typename T>
  llvm::Expected<glow::Handle<T>>
  getGlowConstantHandle(const torch::jit::Value *value) const;

  /// For each Placeholder input to \p ptNode, if this input has been marked
  /// as being an input that should be converted to a glow Constant in \p
  /// constInputs, create a glow Constant for that placeholder with the iValue
  /// from the stack of inputs for this loader. \returns a ValueMap containing
  /// just these new Constants.
  llvm::Expected<ValueMap>
  createConstantPhReplacements(const torch::jit::Node *ptNode,
                               const std::unordered_set<size_t> &constInputs);

  /// \returns a glow Constant from a given PyTorch IValue \p iVal.
  llvm::Expected<glow::Constant *>
  createGlowConstantFromIValue(const torch::jit::IValue &iVal);

  /// Creates and \returns a new Glow Placeholder corresponding to the given
  /// PyTorch Value \p value.
  llvm::Expected<glow::Placeholder *> loadValue(const torch::jit::Value *value);

  /// Load a given PyTorch Node \p ptNode. If \p weightFreezingEnabled then
  /// load inputs that have been marked as constInputs in
  /// MappingOfMemberFunctions as Constants instead of as Placeholders. \returns
  /// error on failure.
  llvm::Error loadNode(const torch::jit::Node *ptNode,
                       bool weightFreezingEnabled);

  /// Load a PyTorch Constant node as a Glow Constant.
  /// \returns error on failure.
  llvm::Error loadConstant(const torch::jit::Node *ptNode);

  /// Load a PyTorch mul node.
  /// \returns error on failure.
  llvm::Error loadMul(const torch::jit::Node *ptNode);

  /// Load a PyTorch div node.
  /// \returns error on failure.
  llvm::Error loadDiv(const torch::jit::Node *ptNode);

  /// Load a PyTorch add node.
  /// \returns error on failure.
  llvm::Error loadAdd(const torch::jit::Node *ptNode);

  /// Load a PyTorch sub node.
  /// \returns error on failure.
  llvm::Error loadSub(const torch::jit::Node *ptNode);

  /// Load a PyTorch max node.
  /// \returns error on failure.
  llvm::Error loadMax(const torch::jit::Node *ptNode);

  /// Load a PyTorch relu node.
  /// \returns error on failure.
  llvm::Error loadRelu(const torch::jit::Node *ptNode);

  /// Load a PyTorch exp node.
  /// \returns error on failure.
  llvm::Error loadExp(const torch::jit::Node *ptNode);

  /// Load a PyTorch sqrt node.
  /// \returns error on failure.
  llvm::Error loadSqrt(const torch::jit::Node *ptNode);

  /// Load a PyTorch reciprocal node.
  llvm::Error loadReciprocal(const torch::jit::Node *ptNode);

  /// Load a PyTorch _convolution node.
  /// \returns error on failure.
  llvm::Error loadConvolution(const torch::jit::Node *ptNode);

  /// Load a PyTorch batch_norm node.
  /// \returns error on failure.
  llvm::Error loadBatchNorm(const torch::jit::Node *ptNode);

  /// Load a PyTorch max_pool2d node.
  /// \returns error on failure.
  llvm::Error loadMaxPool2d(const torch::jit::Node *ptNode);

  /// Load a PyTorch avg_pool2d node.
  /// \returns error on failure.
  llvm::Error loadAvgPool2d(const torch::jit::Node *ptNode);

  /// Load a PyTorch adaptive_avg_pool2d node.
  /// \returns error on failure.
  llvm::Error loadAdaptiveAvgPool2d(const torch::jit::Node *ptNode);

  /// Load a PyTorch t (transpose) node.
  /// \returns error on failure.
  llvm::Error loadTranspose(const torch::jit::Node *ptNode);

  /// Load a PyTorch aten::linear node.
  /// \returns error on failure.
  llvm::Error loadLinear(const torch::jit::Node *ptNode);

  /// Load a PyTorch min node.
  /// \returns error on failure.
  llvm::Error loadMin(const torch::jit::Node *ptNode);
};
} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_PYTORCHMODELLOADER_H
