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

#ifndef GLOW_TORCH_GLOW_SRC_PYTORCHMODELLOADER_H
#define GLOW_TORCH_GLOW_SRC_PYTORCHMODELLOADER_H

#include "PyTorchCommon.h"
#include <torch/csrc/jit/custom_operator.h>

#include "GlowIValue.h"

#include "glow/Graph/Graph.h"

namespace glow {
/// Tag that indicates the type of mapping for ValueMapping.
enum class ValueMappingType {
  IValue, // Tag representing GlowIValues: These are non-tensor things for which
          // we can inspect their values at load time. Any GlowIValue tensors
          // will be converted to glow::Constants so will not be present as
          // GlowIValues.

  NodeValue, // Tag representing NodeValues: These can be glow::Constants,
             // glow::Placeholders, or the outputs of intermediary nodes in the
             // graph. It is safe to reason about the dimensions and data types
             // of these at load time but not about their values.

  FrozenNodeValue // Tag representing NodeValues that are the outputs of
                  // glow::Constant that were produced during weight freezing at
                  // load time. Weight freezing is not guaranteed to happen and
                  // thus it is also not safe to reason about the values of
                  // these Constants at load time.
};

/// The class is effectively a union of GlowIValues and NodeValue and is used to
/// represent things that can be mapped from PyTorch Values during model
/// loading, see ValueMappingType which provides tags for this class for
/// descriptions of how each of these things can be used during model loading.
class ValueMapping {
  /// Tag of which member is valid. Only one member is valid at a time.
  ValueMappingType mappingType_;

  /// Members that store either a NodeValue or a pointer to a GlowIValue
  /// depending on what the PyTorch Value being mapped is.
  NodeValue nodeValue_;
  std::unique_ptr<GlowIValue> glowIValue_;

public:
  /// \returns the ValueMappingType representing the type that is mapped.
  ValueMappingType getMappingType() const;

  /// Create a ValueMapping from a NodeValue \p nodeValue. \p wasFrozen should
  /// be set to true if this NodeValue comes from weight freezing.
  ValueMapping(NodeValue nodeValue, bool wasFrozen);

  /// Create a ValueMapping from a GlowIValue \p noglowIValuedeValue.
  ValueMapping(GlowIValue glowIValue);

  /// \returns the mapped NodeValue if one is mapped otherwise return an error.
  Expected<NodeValue> getMappedNodeValue();

  /// \returns the mapped GlowIValue if one is mapped otherwise return an error.
  Expected<GlowIValue *> getMappedGlowIValue();

  /// \returns the mapped GlowIValue if one is mapped otherwise return an error.
  Expected<const GlowIValue *> getMappedGlowIValue() const;
};

// Input's shape and type
struct InputMeta {
  c10::ScalarType type;
  std::vector<size_t> dims;

  InputMeta(c10::ScalarType type_, std::vector<size_t> &&dims_) {
    type = type_;
    dims = dims_;
  }
};

/// Loads PyTorch JIT IR graphs as a Glow Function.
class PyTorchModelLoader {
  /// Glow Function created outside this class.
  glow::Function &F_;

  /// Map from input placeholders to their location on the input stack.
  std::unordered_map<glow::Placeholder *, size_t>
      inputPlaceholdersReverseIndex_;

  /// The reference PyTorch inputs used for loading. This is required for shape
  /// information.
  const at::ArrayRef<torch::jit::IValue> inputs_;

  /// Mapping from PyTorch Values to GlowIValues and Glow NodeValues created
  /// during loading.
  std::unordered_map<const torch::jit::Value *, ValueMapping> valueMap_;

  /// Indices of stack inputs that were frozen during loading. This set is
  /// optionally provided by the user of PyTorchModelLoader and will be returned
  /// to them after loading is complete.
  std::set<size_t> *frozenInputIndices_ = nullptr;

  /// Flags if the memory held by aten::Constants of Tensor type should be
  /// copied.
  const bool copyTensorMemory_;

  /// Values in the MappingOfMemberFunctions map. These values contain the
  /// information necessary to load PyTorch nodes such as which
  /// PyTorchModelLoader method to use and which inputs should be considered as
  /// constants.
  struct MappingOfMemberFunctionsValue {
    /// The type of functions used to load PyTorch nodes in PyTorchModelLoader.
    using LoadFn = Error (PyTorchModelLoader::*)(const torch::jit::Node *);

    /// Symbols (as strings) that this mapping value is applicable to.
    const std::vector<const char *> symbols;

    /// The PyTorchModelLoader method that should be used to load the given
    /// PyTorch node.
    LoadFn loadFn;

    /// The set of inputs that should be loaded as Glow Constants instead of
    /// as placeholders for inference because they should be expected to not
    /// change between inferences. An example would be the weights for
    /// convolutions.
    const std::unordered_set<size_t> inputsToFreeze;

    MappingOfMemberFunctionsValue(std::vector<const char *> symbolsP,
                                  LoadFn loadFnP,
                                  std::unordered_set<size_t> inputsToFreezeP)
        : symbols(symbolsP), loadFn(loadFnP), inputsToFreeze(inputsToFreezeP) {}
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
  /// will succeed because determining this requires more information such as
  /// shape info that isn't yet available when this is run.
  static bool isNodeSupported(const torch::jit::Node *ptNode);

  /// Takes a glow::Function \p F, a jit::Graph \p subgraph to load, and a
  /// stack of \p inputs for the subgraph to be loaded. Parameter \p
  /// settings control the fusion details. Output parameters \p
  /// inputPlaceholders and \p outputPlaceholders are filled out. \returns
  /// error on failure.
  static Error
  loadJITGraph(glow::Function &F, const torch::jit::Graph &graph,
               std::vector<glow::Placeholder *> &inputPlaceholders,
               std::vector<glow::Placeholder *> &outputPlaceholders,
               const PyTorchLoaderSettings &settings,
               const at::ArrayRef<torch::jit::IValue> inputs,
               const std::vector<InputMeta> &inputMeta);

  /// Takes a glow::Function \p F, a jit::Graph \p subgraph to load, \p inputs
  /// as graph external inputs, and \parameters as known tensors. Output
  /// parameters \p inputPlaceholders and \p outputPlaceholders are filled out.
  /// \returns error on failure.
  static Error loadJITGraphForOnnxTraining(
      glow::Function &F, const torch::jit::Graph &graph,
      const at::ArrayRef<torch::jit::IValue> inputs,
      const std::vector<at::Tensor> &parameters,
      std::vector<glow::Placeholder *> &inputPlaceholders,
      std::vector<glow::Placeholder *> &outputPlaceholders);

private:
  /// Takes a glow::Function \p F, a jit::Graph \p graph to load, and a
  /// stack of \p inputs for the graph to be loaded. Parameter \p settings
  /// control the fusion details. Output parameters \p inputPlaceholders and
  /// \p outputPlaceholders are filled out. \p frozenInputIndices is an optional
  /// parameter that, if provided, will be filled with the set of stack indices
  /// that were frozen during loading.
  PyTorchModelLoader(glow::Function &F, const torch::jit::Graph &graph,
                     std::vector<glow::Placeholder *> &inputPlaceholders,
                     std::vector<glow::Placeholder *> &outputPlaceholders,
                     Error &error, const PyTorchLoaderSettings &settings,
                     std::set<size_t> *frozenInputIndices,
                     const at::ArrayRef<torch::jit::IValue> inputs,
                     const std::vector<InputMeta> &inputMeta = {});

  /// Takes a glow::Function \p F, a jit::Graph \p graph to load, and a
  /// graph \p inputs and placeholders \p parameters. Output parameters \p
  /// inputPlaceholders and \p outputPlaceholders are filled out.
  /// This is only used by loadJITGraphForOnnxTraining.
  PyTorchModelLoader(glow::Function &F, const torch::jit::Graph &graph,
                     const std::vector<at::Tensor> &parameters,
                     std::vector<glow::Placeholder *> &inputPlaceholders,
                     std::vector<glow::Placeholder *> &outputPlaceholders,
                     Error &error,
                     const at::ArrayRef<torch::jit::IValue> inputs);

  /// Save access to the mapping.
  static const MappingOfMemberFunctions &getSymbolsMapping();

  /// Add a new mapping from the PyTorch Value \p value to the Glow NodeValue
  /// \p nodeValue. Set \p wasFrozen to true if this comes from a from a frozen
  /// input.
  /// \returns error on failure.
  Error addValueMapping(const torch::jit::Value *value,
                        glow::NodeValue nodeValue, bool wasFrozen = false);

  /// Add a new mapping from the PyTorch Value \p value to the GlowIValue
  /// \p glowIValue. Set \p wasFrozen to true if this comes from a from a frozen
  /// input.
  /// \returns error on failure.
  Error addValueMapping(const torch::jit::Value *value,
                        glow::GlowIValue glowIValue, bool wasFrozen = false);

  /// Remove any ValueMapping associated with \p value.
  void removeValueMapping(const torch::jit::Value *value);

  /// Returns true if a Glow NodeValue has been created for a given PyTorch
  /// Value \p value.
  bool hasGlowNodeValueForValue(const torch::jit::Value *value) const;

  /// Returns true if a GlowIValue has been created for a given PyTorch
  /// Value \p value. If \p ignoreNones is true then this will return false even
  /// if a GlowIValue is mapped to this value if that GlowIValue's tag is None.
  bool hasGlowIValueForValue(const torch::jit::Value *value,
                             bool ignoreNones = false) const;

  /// Find the Glow NodeValue that maps to a given PyTorch value \p value.
  Expected<glow::NodeValue>
  getGlowNodeValueForValue(const torch::jit::Value *value);

  /// Find the GlowIValue that maps to a given PyTorch value \p value.
  Expected<glow::GlowIValue *>
  getGlowIValueForValue(const torch::jit::Value *value);

  /// For each Placeholder input to \p ptNode, if this input has been marked
  /// as being an input that should be frozen in MappingOfMemberFunctions,
  /// create a glow Constant for that Placeholder with the iValue from the stack
  /// of inputs for this loader. \returns a ValueMap containing just these new
  /// Constants.
  Error freezeWeights(const torch::jit::Node *ptNode);

  /// Load a given PyTorch Node \p ptNode. \returns
  /// error on failure.
  Error loadNode(const torch::jit::Node *ptNode);

  /// Load a PyTorch Constant node as a Glow Constant.
  /// \returns error on failure.
  Error loadConstant(const torch::jit::Node *ptNode);

  /// Load a PyTorch mul node.
  /// \returns error on failure.
  Error loadMul(const torch::jit::Node *ptNode);

  /// Load a PyTorch div node.
  /// \returns error on failure.
  Error loadDiv(const torch::jit::Node *ptNode);

  /// Load a PyTorch add node.
  /// \returns error on failure.
  Error loadAdd(const torch::jit::Node *ptNode);

  /// Load a PyTorch sub node.
  /// \returns error on failure.
  Error loadSub(const torch::jit::Node *ptNode);

  /// Load a PyTorch max node.
  /// \returns error on failure.
  Error loadMax(const torch::jit::Node *ptNode);

  /// Load a PyTorch relu node.
  /// \returns error on failure.
  Error loadRelu(const torch::jit::Node *ptNode);

  /// Load a PyTorch exp node.
  /// \returns error on failure.
  Error loadExp(const torch::jit::Node *ptNode);

  /// Load a PyTorch sqrt node.
  /// \returns error on failure.
  Error loadSqrt(const torch::jit::Node *ptNode);

  /// Load a PyTorch reciprocal node.
  Error loadReciprocal(const torch::jit::Node *ptNode);

  /// Load a PyTorch _convolution node.
  /// \returns error on failure.
  Error loadConvolution(const torch::jit::Node *ptNode);

  /// Load a PyTorch batch_norm node.
  /// \returns error on failure.
  Error loadBatchNorm(const torch::jit::Node *ptNode);

  /// Load a PyTorch quantization::add node.
  /// \return error on failure.
  Error loadQuantizedAdd(const torch::jit::Node *ptNode);

  /// Load a PyTorch quantize_linear node.
  /// \returns error on failure.
  Error loadQuantize(const torch::jit::Node *ptNode);

  /// Load a PyTorch dequantize node.
  /// \returns error on failure.
  Error loadDequantize(const torch::jit::Node *ptNode);

  /// Load a PyTorch max_pool2d node.
  /// \returns error on failure.
  Error loadMaxPool2d(const torch::jit::Node *ptNode);

  /// Load a PyTorch sigmoid node.
  /// \returns error on failure.
  Error loadSigmoid(const torch::jit::Node *ptNode);

  /// Load a PyTorch avg_pool2d node.
  /// \returns error on failure.
  Error loadAvgPool2d(const torch::jit::Node *ptNode);

  /// Load a PyTorch adaptive_avg_pool2d node.
  /// \returns error on failure.
  Error loadAdaptiveAvgPool2d(const torch::jit::Node *ptNode);

  /// Load a PyTorch t (transpose) node.
  /// \returns error on failure.
  Error loadTranspose(const torch::jit::Node *ptNode);

  /// Load a PyTorch min node.
  /// \returns error on failure.
  Error loadMin(const torch::jit::Node *ptNode);

  /// Load a PyTorch clamp node.
  /// \returns error on failure.
  Error loadClamp(const torch::jit::Node *ptNode);

  /// Load a PyTorch prelu node.
  /// \returns error on failure.
  Error loadPRelu(const torch::jit::Node *ptNode);

  /// Load a PyTorch SoftMax node.
  /// \returns error on failure.
  Error loadSoftMax(const torch::jit::Node *ptNode);

  /// Load a PyTorch flatten node.
  /// \returns error on failure.
  Error loadFlatten(const torch::jit::Node *ptNode);

  /// Load a PyTorch topK node.
  /// \returns error on failure.
  Error loadTopK(const torch::jit::Node *ptNode);

  /// Load a PyTorch aten::size node.
  /// \returns error on failure.
  Error loadSize(const torch::jit::Node *ptNode);

  /// Load a PyTorch prim::ListConstruct node.
  /// \returns error on failure.
  Error loadListConstruct(const torch::jit::Node *ptNode);

  /// Load a PyTorch aten::reshape node.
  /// \returns error on failure.
  Error loadReshape(const torch::jit::Node *ptNode);

  /// Load a PyTorch aten::mm node.
  /// \returns error on failure.
  Error loadMM(const torch::jit::Node *ptNode);

  /// Load a PyTorch aten::addmm node.
  /// \returns error on failure.
  Error loadAddMM(const torch::jit::Node *ptNode);

  /// Load a PyTorch aten::matmul node.
  /// \returns error on failure.
  Error loadMatMul(const torch::jit::Node *ptNode);
};

} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_PYTORCHMODELLOADER_H
