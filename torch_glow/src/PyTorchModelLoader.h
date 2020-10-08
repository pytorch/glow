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
};

/// The class is effectively a union of GlowIValues and NodeValue and is used to
/// represent things that can be mapped from PyTorch Values during model
/// loading, see ValueMappingType which provides tags for this class for
/// descriptions of how each of these things can be used during model loading.
class ValueMapping {
  /// Tag of which member is valid. Only one member is valid at a time.
  ValueMappingType mappingType_;

  // The original data type of this jit value in PyTorch.
  // Useful when we quantize an uint8 PyTorch tensor to int8 in Glow.
  c10::ScalarType correctType_;

  /// Members that store either a NodeValue or a pointer to a GlowIValue
  /// depending on what the PyTorch Value being mapped is.
  NodeValue nodeValue_;
  std::unique_ptr<GlowIValue> glowIValue_;

public:
  /// \returns the ValueMappingType representing the type that is mapped.
  ValueMappingType getMappingType() const;

  /// \return the correctType_ representing the original PyTorch data type.
  c10::ScalarType getCorrectType() const;

  /// Set the correctType_ to be \p dtype.
  /// \returns error on failure.
  void setCorrectType(c10::ScalarType dtype);

  /// Create a ValueMapping from a NodeValue \p nodeValue.
  ValueMapping(NodeValue nodeValue);

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
  std::vector<glow::sdim_t> dims;

  InputMeta(c10::ScalarType type_, std::vector<glow::sdim_t> &&dims_) {
    type = type_;
    dims = dims_;
  }
  InputMeta(c10::ScalarType type_, const std::vector<glow::sdim_t> &dims_) {
    type = type_;
    dims = dims_;
  }
};

/// Loads PyTorch JIT IR graphs as a Glow Function.
class PyTorchModelLoader {
public:
  /// Glow Function created outside this class. Made public so that it can be
  /// accessed by custom operator loaders.
  glow::Function &F_;

private:
  /// Map from input placeholders to their location on the input stack.
  std::unordered_map<glow::Placeholder *, size_t>
      inputPlaceholdersReverseIndex_;

  /// The reference PyTorch inputs used for loading. This is required for shape
  /// information.
  const at::ArrayRef<torch::jit::IValue> inputs_;

  /// Mapping from PyTorch Values to GlowIValues and Glow NodeValues created
  /// during loading.
  std::unordered_map<const torch::jit::Value *, ValueMapping> valueMap_;

  /// Reverse of valueMap_, mapping from Glow NodeValues to their
  /// corresponding PyTorch Values.
  std::unordered_map<glow::NodeValue, const torch::jit::Value *>
      valueMapReverse_;

  std::unordered_map<const torch::jit::Value *, torch::jit::IValue> qparamsMap_;

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

    MappingOfMemberFunctionsValue(std::vector<const char *> symbolsP,
                                  LoadFn loadFnP)
        : symbols(symbolsP), loadFn(loadFnP) {}
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

  /// Create and run a simple function with only one glowNode node,
  /// then map its result back to original graph.
  /// Used for constant propagation.
  /// \returns error on failure.
  /// \p glowNode is the glow node we would like to run,
  /// \p node is the torch jit node that generate \p glowNode,
  /// and \p nodeBeginPtr is the current glow node ptr of the glow node
  /// linklist.
  Error
  runAndRemapSingleNode(glow::Node &glowNode,
                        const torch::jit::Node *const node,
                        llvm::simple_ilist<glow::Node>::iterator nodeBeginPtr);

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
               std::vector<c10::ScalarType> &outputCorrectType,
               const PyTorchLoaderSettings &settings,
               const at::ArrayRef<torch::jit::IValue> inputs,
               const std::vector<InputMeta> &inputMeta);

private:
  /// Takes a glow::Function \p F, a jit::Graph \p graph to load, and a
  /// stack of \p inputs for the graph to be loaded. Parameter \p settings
  /// control the fusion details. Output parameters \p inputPlaceholders and
  /// \p outputPlaceholders are filled out.
  PyTorchModelLoader(glow::Function &F, const torch::jit::Graph &graph,
                     std::vector<glow::Placeholder *> &inputPlaceholders,
                     std::vector<glow::Placeholder *> &outputPlaceholders,
                     std::vector<c10::ScalarType> &outputCorrectType,
                     Error &error, const PyTorchLoaderSettings &settings,
                     const at::ArrayRef<torch::jit::IValue> inputs,
                     const std::vector<InputMeta> &inputMeta = {});

  /// Build mapping from jit symbols to function that loads nodes of that kind.
  static const MappingOfMemberFunctions buildSymbolsMapping();

  /// Save access to the mapping from jit symbols to function that loads nodes
  /// of that kind.
  static const MappingOfMemberFunctions &getSymbolsMapping();

  // The below methods are made public so that they can be accessed by custom
  // operator loaders.
public:
  /// Add a new mapping from the PyTorch Value \p value to the Glow NodeValue
  /// \p nodeValue.
  /// \returns error on failure.
  Error addValueMapping(const torch::jit::Value *value,
                        glow::NodeValue nodeValue);

  /// Add a new mapping from the PyTorch Value \p value and its original type in
  /// PyTorch \p correctType to the Glow NodeValue \p nodeValue. \returns error
  /// on failure.
  Error addValueMapping(const torch::jit::Value *value,
                        glow::NodeValue nodeValue, c10::ScalarType correctType);

  /// Add a new mapping from the PyTorch Value \p value to the GlowIValue
  /// \p glowIValue.
  /// \returns error on failure.
  Error addValueMapping(const torch::jit::Value *value,
                        glow::GlowIValue glowIValue);

  /// Add a new mapping from the PyTorch Value \p value and its original type in
  /// PyTorch \p correctType to the Glow IValue \p nodeValue.
  /// \returns error on failure.
  Error addValueMapping(const torch::jit::Value *value,
                        glow::GlowIValue glowIValue,
                        c10::ScalarType correctType);

  /// Get the correctType of \p src from valueMap_, and save it to \p dest.
  /// \returns error on failure.
  Error getCorrectTypeMapping(c10::ScalarType &dest,
                              const torch::jit::Value *src);

  /// Remove any ValueMapping associated with \p value.
  /// \returns error on failure.
  Error removeValueMapping(const torch::jit::Value *value);

  /// Returns true if a Glow NodeValue has been created for a given PyTorch
  /// Value \p value.
  bool hasGlowNodeValueForValue(const torch::jit::Value *value) const;

  /// Returns true if a GlowIValue has been created for a given PyTorch
  /// Value \p value. If \p ignoreNones is true then this will return false even
  /// if a GlowIValue is mapped to this value if that GlowIValue's tag is None.
  bool hasGlowIValueForValue(const torch::jit::Value *value,
                             bool ignoreNones = false) const;

  /// If a NodeValue is mapped to \p value then return it, otherwise look for a
  /// float or integer IValue mapped to \p value, create a Glow Constant by
  /// broadcasting that value to a tensor of size \p dims and return the result
  /// of that Constant.  If the optional flag \p makeFloat is true, generates a
  /// tensor with floating point elements (default if flag omitted).  Otherwise,
  /// generates a tensor with integer elements.
  Expected<glow::NodeValue>
  loadNodeValueOrBroadcastedIValue(const torch::jit::Value *value,
                                   llvm::ArrayRef<glow::dim_t> dims,
                                   bool makeFloat = true);

  /// If there is a NodeValue mapped to \p value then return it, otherwise
  /// create a Constant with type \p ty, name \p name, and value \p val
  /// broadcasted.
  template <typename T = float>
  glow::NodeValue
  loadNodeValueOrCreateBroadcastedConstant(const torch::jit::Value *value,
                                           llvm::StringRef name, const Type &ty,
                                           const T &val);

  /// Find the Glow NodeValue that maps to a given PyTorch value \p value.
  Expected<glow::NodeValue>
  getGlowNodeValueForValue(const torch::jit::Value *value);

  /// Find the GlowIValue that maps to a given PyTorch value \p value.
  Expected<glow::GlowIValue *>
  getGlowIValueForValue(const torch::jit::Value *value);

  /// Rescale a uint8 NodeValue \p input to the equivalent int8 NodeValue.
  glow::NodeValue rescaleUIntToInt(glow::NodeValue input);

  /// Rescale a int8 NodeValue \p input to the equivalent uint8 NodeValue.
  glow::NodeValue rescaleIntToUint(glow::NodeValue input);

  /// Given Node inputs and outputs, check the expected sizes. Negative size
  /// indicates that the size should be equal to or greater than that size (for
  /// example -2 means at least 2).
  template <typename T>
  static Error checkInputAndOutputSizes(const T &inputs, int64_t inputsSize,
                                        const T &outputs, int64_t outputsSize) {
    if (inputsSize >= 0) {
      RETURN_ERR_IF_NOT(inputs.size() == inputsSize,
                        glow::strFormat("Expected exactly %lu inputs, got %lu.",
                                        (size_t)inputsSize, inputs.size()));
    } else {
      inputsSize = inputsSize * -1;
      RETURN_ERR_IF_NOT(
          inputs.size() >= inputsSize,
          glow::strFormat("Expected at least %lu inputs, got %lu.",
                          (size_t)inputsSize, inputs.size()));
    }

    if (outputsSize >= 0) {
      RETURN_ERR_IF_NOT(
          outputs.size() == outputsSize,
          glow::strFormat("Expected exactly %lu outputs, got %lu.",
                          (size_t)outputsSize, outputs.size()));
    } else {
      outputsSize = outputsSize * -1;
      RETURN_ERR_IF_NOT(
          outputs.size() >= outputsSize,
          glow::strFormat("Expected at least %lu outputs, got %lu.",
                          (size_t)outputsSize, outputs.size()));
    }
    return Error::success();
  }

private:
  /// Load a quantized conv node from ptNode to qconv.
  /// a wrapper function of loadQuantizedConv and loadQuantizedConvRelu.
  /// Returns error on failure.
  Expected<NodeValue> loadQuantizedConvImpl(const torch::jit::Node *ptNode,
                                            const bool isRelu);

  /// Common function to load both 2D and 3D AvgPool nodes
  /// \returns error on failure.
  Expected<NodeValue> loadAvgPoolImpl(const torch::jit::Node *ptNode,
                                      int numDims);

  /// Load a PyTorch quantized batch_norm2d or batch_norm3d node.
  /// \returns error on failure.
  Expected<NodeValue> loadQuantizedBatchNormImpl(const torch::jit::Node *ptNode,
                                                 int numDims);

  // Load a PyTorch aten::embedding_bag node.
  // \returns error on failure.
  Error loadEmbeddingBag(const torch::jit::Node *ptNode);

  /// Load a _caffe2::BatchPermutation node.
  Error loadBatchPermutation(const torch::jit::Node *ptNode);

  // Load a PyTorch fb::embedding_bag_byte_rowwise_offsets node.
  // \returns error on failure.
  Error loadEmbeddingBagByteRowwiseOffsets(const torch::jit::Node *ptNode);

  // Load a PyTorch fb::embedding_bag_4bit_rowwise_offsets.
  // \returns error on failure.
  Error loadEmbeddingBag4BitRowwiseOffsets(const torch::jit::Node *ptNode);

  // Helper function that implements the loading logic for
  // fb::embedding_bag_4bit_rowwise_offsets and
  // fb::embedding_bag_byte_rowwise_offsets.
  // \returns error on failure.
  Error loadEmbeddingBagByteRowwiseOffsetsHelper(const torch::jit::Node *ptNode,
                                                 bool is4Bit = false);

  // Load a PyTorch _caffe2::RoIAlign op.
  // \returns error on failure.
  Error loadRoiAlign(const torch::jit::Node *ptNode);

  // Load a PyTorch _caffe2::RoIAlignRotated op.
  // \returns error on failure.
  Error loadRoiAlignRotated(const torch::jit::Node *ptNode);

  // Load a PyTorch _caffe2::BBoxTransform op.
  // \returns error on failure.
  Error loadBBoxTransform(const torch::jit::Node *ptNode);

  /// Load all PyTorch prim::GetAttr nodes in \p graph. This method uses the
  /// PyTorch Module hierarchy to map Values for all outputs of prim::GetAttr
  /// nodes. If the output type of a prim::GetAttr is a tensor, this will load
  /// it as a Glow constant, if it's an ivalue::Object it is ignored, and if
  /// it's any other kind of IValue, it is loaded as a GlowIvalue for use during
  /// the rest of model loading. \returns error on failure.
  Error loadAttributes(const torch::jit::Graph &graph,
                       const at::ArrayRef<torch::jit::IValue> inputs);

  /// Load each PyTorch Node in the Graph \p graph.
  /// \returns error on failure.
  Error loadNodes(const torch::jit::Graph &graph);

  /// Load a PyTorch Constant node as a Glow Constant.
  /// \returns error on failure.
  Error loadConstant(const torch::jit::Node *ptNode);

  /// Load a custom PyTorch op using a statically register CustomPyTorchOpLoader
  /// for that op.
  /// \returns error on failure.
  Error loadCustomOp(const torch::jit::Node *ptNode);

  /// Load a PyTorch type_as node.
  /// \returns error on failure.
  Error loadTypeAs(const torch::jit::Node *ptNode);

  /// Load a PyTorch contiguous node.
  /// \returns error on failure.
  Error loadContiguous(const torch::jit::Node *ptNode);

  /// NOP - aten::detach returns
  Error loadDetach(const torch::jit::Node *ptNode);

  /// Helper function for loading arithmetic nodes. \p name is of the name of
  /// the node in the Glow graph, \p lhs and \p rhs are the inputs to the
  /// arithetic node and template parameter \p GlowNode is the type of the node
  /// that should be created in the Glow graph. \returns the output of the
  /// loaded arithmetic node or an Error if any occurred.
  template <typename GlowNode>
  Expected<NodeValue> loadArithmeticNode(llvm::StringRef name,
                                         const torch::jit::Value *lhs,
                                         const torch::jit::Value *rhs);

  /// Load a PyTorch mul node.
  /// \returns error on failure.
  Error loadMul(const torch::jit::Node *ptNode);

  /// Load a PyTorch div node.
  /// \returns error on failure.
  Error loadDiv(const torch::jit::Node *ptNode);

  /// Load a PyTorch floor div node.
  /// \returns error on failure.
  Error loadFloorDiv(const torch::jit::Node *ptNode);

  /// Load a PyTorch add node.
  /// \returns error on failure.
  Error loadAdd(const torch::jit::Node *ptNode);

  /// Load a PyTorch arange node.
  /// \returns error on failure.
  Error loadArange(const torch::jit::Node *ptNode);

  /// Load a PyTorch sub node.
  /// \returns error on failure.
  Error loadSub(const torch::jit::Node *ptNode);

  /// Load a PyTorch rsub node.
  /// \returns error on failure.
  Error loadRsub(const torch::jit::Node *ptNode);

  /// Load a PyTorch max node.
  /// \returns error on failure.
  Error loadMax(const torch::jit::Node *ptNode);

  /// Load a PyTorch floor node.
  /// \returns error on failure.
  Error loadFloor(const torch::jit::Node *ptNode);

  /// Load a PyTorch ceil node.
  /// \returns error on failure.
  Error loadCeil(const torch::jit::Node *ptNode);

  /// Load a PyTorch relu node.
  /// \returns error on failure.
  Error loadRelu(const torch::jit::Node *ptNode);

  /// Load a PyTorch gelu node.
  /// \returns error on failure.
  Error loadGelu(const torch::jit::Node *ptNode);

  /// Load a PyTorch exp node.
  /// \returns error on failure.
  Error loadExp(const torch::jit::Node *ptNode);

  /// Load a PyTorch pow node.
  /// \returns error on failure.
  Error loadPow(const torch::jit::Node *ptNode);

  /// Load a PyTorch sqrt node.
  /// \returns error on failure.
  Error loadSqrt(const torch::jit::Node *ptNode);

  /// Load a PyTorch reciprocal node.
  /// \returns error on failure.
  Error loadReciprocal(const torch::jit::Node *ptNode);

  /// Load a PyTorch prim::cat node fused with a prim::ListConstruct into a
  /// prim::FusedConcat node.
  /// \returns error on failure.
  Error loadFusedConcat(const torch::jit::Node *ptNode);

  /// Load a PyTorch prim::stack node fused with a prim::ListConstruct into a
  /// glow:FusedStack node.
  /// \returns error on failure.
  Error loadFusedStack(const torch::jit::Node *ptNode);

  /// Load a PyTorch LSTM node.
  /// \returns error on failure.
  Error loadLSTM(const torch::jit::Node *ptNode);

  /// Load a PyTorch _convolution node.
  /// \returns error on failure.
  Error loadConvolution(const torch::jit::Node *ptNode);

  /// Load a PyTorch conv2d node.
  /// \returns error on failure.
  Error loadConv2D(const torch::jit::Node *ptNode);

  /// Load a PyTorch batch_norm node.
  /// \returns error on failure.
  Error loadBatchNorm(const torch::jit::Node *ptNode);

  /// Load a PyTorch vector_norm node.
  /// \returns error on failure.
  Error loadNorm(const torch::jit::Node *ptNode);

  /// Load a PyTorch quantized::batch_norm2d node.
  /// \returns error on failure.
  Error loadQuantizedBatchNorm2d(const torch::jit::Node *ptNode);

  /// Load a PyTorch quantized::batch_norm3d node.
  /// \returns error on failure.
  Error loadQuantizedBatchNorm3d(const torch::jit::Node *ptNode);

  /// Load a PyTorch quantized::batch_norm3d_relu node.
  /// \returns error on failure.
  Error loadQuantizedBatchNorm3dRelu(const torch::jit::Node *ptNode);

  /// Load a PyTorch aten::layer_norm node.
  /// \returns error on failure.
  Error loadLayerNorm(const torch::jit::Node *ptNode);

  /// Load a PyTorch dropout node.
  /// \returns error on failure.
  Error loadDropout(const torch::jit::Node *ptNode);

  /// Load a PyTorch quantized::add node.
  /// \return error on failure.
  Error loadQuantizedAdd(const torch::jit::Node *ptNode);

  /// Load a PyTorch quantized::add_relu node.
  /// \return error on failure.
  Error loadQuantizedAddRelu(const torch::jit::Node *ptNode);

  /// Load a PyTorch quantized::mul node.
  /// \return error on failure.
  Error loadQuantizedMul(const torch::jit::Node *ptNode);

  /// Load a glow::unpacked_quantized_conv node.
  // \return error on failure.
  Error loadQuantizedConvUnpacked(const torch::jit::Node *ptNode);

  Error loadQuantizedConvReluUnpacked(const torch::jit::Node *ptNode);

  Error loadQuantizedConvUnpackedImpl(const torch::jit::Node *ptNode,
                                      bool isRelu = false);

  /// Load a PyTorch _caffe2::RoIAlign op or if \p isRotated is true then a
  /// _caffe2::RoIAlignRotated op.
  Error loadRoiAlignImpl(const torch::jit::Node *ptNode, bool isRotated);

  /// Load a PyTorch quantized::conv2d node.
  // \return error on failure.
  Error loadQuantizedConv(const torch::jit::Node *ptNode);

  /// Load a PyTorch quantized::conv2d_relu node.
  // \return error on failure.
  Error loadQuantizedConvRelu(const torch::jit::Node *ptNode);

  /// Load a glow::unpacked_quantized_linear node.
  /// \return error on failure.
  Error loadQuantizedLinearUnpacked(const torch::jit::Node *ptNode);

  /// Load a PyTorch quantized::linear node.
  /// \return error on failure.
  Error loadQuantizedLinear(const torch::jit::Node *ptNode);

  /// Load a PyTorch quantize_per_tensor node.
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

  /// Load a PyTorch avg_pool3d node.
  /// \returns error on failure.
  Error loadAvgPool3d(const torch::jit::Node *ptNode);

  /// Load a PyTorch adaptive_avg_pool2d node.
  /// \returns error on failure.
  Error loadAdaptiveAvgPool2d(const torch::jit::Node *ptNode);

  /// Load a PyTorch aten::t (transpose) node.
  /// \returns error on failure.
  Error loadT(const torch::jit::Node *ptNode);

  /// Load a PyTorch aten::transpose node.
  /// \returns error on failure.
  Error loadTranspose(const torch::jit::Node *ptNode);

  /// Load a PyTorch mean node.
  /// \returns error on failure.
  Error loadMean(const torch::jit::Node *ptNode);

  /// Load a PyTorch min node.
  /// \returns error on failure.
  Error loadMin(const torch::jit::Node *ptNode);

  /// Load a PyTorch clamp node.
  /// \returns error on failure.
  Error loadClamp(const torch::jit::Node *ptNode);

  /// Load a PyTorch prelu node.
  /// \returns error on failure.
  Error loadPRelu(const torch::jit::Node *ptNode);

  /// Load a PyTorch slice node.
  /// \returns error on failure.
  Error loadSlice(const torch::jit::Node *ptNode);

  /// Load a PyTorch SoftMax node.
  /// \returns error on failure.
  Error loadSoftMax(const torch::jit::Node *ptNode);

  /// Load a PyTorch flatten node.
  /// \returns error on failure.
  Error loadFlatten(const torch::jit::Node *ptNode);

  /// Load a PyTorch aten::squeeze node.
  /// \returns error on failure.
  Error loadSqueeze(const torch::jit::Node *ptNode);

  /// Load a PyTorch aten::unsqueeze node.
  /// \returns error on failure.
  Error loadUnsqueeze(const torch::jit::Node *ptNode);

  /// Load a PyTorch aten::masked_fill node.
  /// \returns error on failure.
  Error loadMaskedFill(const torch::jit::Node *ptNode);

  /// Load a PyTorch topK node.
  /// \returns error on failure.
  Error loadTopK(const torch::jit::Node *ptNode);

  /// Load a PyTorch aten::size node.
  /// \returns error on failure.
  Error loadSize(const torch::jit::Node *ptNode);

  /// Load a PyTorch prim::ListConstruct node.
  /// \returns error on failure.
  Error loadListConstruct(const torch::jit::Node *ptNode);

  /// Load a PyTorch aten::Int node.
  /// \returns error on failure.
  Error loadInt(const torch::jit::Node *ptNode);

  /// Load a PyTorch prim::NumToTensor node.
  /// \returns error on failure.
  Error loadNumToTensor(const torch::jit::Node *ptNode);

  /// Load a PyTorch aten::reshape node.
  /// \returns error on failure.
  Error loadReshape(const torch::jit::Node *ptNode);

  /// Load a PyTorch aten::upsample_nearest3d node.
  /// \returns error on failure.
  Error loadUpsampleNearest3D(const torch::jit::Node *ptNode);

  /// Load a PyTorch aten::view node.
  /// \returns error on failure.
  Error loadView(const torch::jit::Node *ptNode);

  /// Load a PyTorch aten::mm node.
  /// \returns error on failure.
  Error loadMM(const torch::jit::Node *ptNode);

  /// Load a PyTorch aten::addmm node.
  /// \returns error on failure.
  Error loadAddMM(const torch::jit::Node *ptNode);

  /// Load a prim::constantChunk node.
  /// \returns error on failure.
  Error loadConstantChunk(const torch::jit::Node *ptNode);

  /// Helper function for loading a PyTorch aten::matmul node.
  Expected<glow::NodeValue> loadMatMulImpl(glow::NodeValue lhs,
                                           glow::NodeValue rhs);

  /// Load a PyTorch aten::matmul node.
  /// \returns error on failure.
  Error loadMatMul(const torch::jit::Node *ptNode);

  /// Load a PyTorch aten::bmm node.
  /// \returns error on failure.
  Error loadBmm(const torch::jit::Node *ptNode);

  /// Load a PyTorch aten::tanh node.
  /// \returns error on failure.
  Error loadTanh(const torch::jit::Node *ptNode);

  /// Load a glow::fused_linear node.
  /// \returns error on failure.
  Error loadGlowFusedLinear(const torch::jit::Node *ptNode);

  /// Load a PyTorch aten::permute node.
  /// \returns error on failure.
  Error loadPermute(const torch::jit::Node *ptNode);

  /// Load a PyTorch aten::to node.
  /// \returns error on failure.
  Error loadTo(const torch::jit::Node *ptNode);
};

} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_PYTORCHMODELLOADER_H
