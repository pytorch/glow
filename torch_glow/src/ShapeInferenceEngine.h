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

#ifndef GLOW_TORCH_GLOW_SRC_SHAPEINFERENCEENGINE_H
#define GLOW_TORCH_GLOW_SRC_SHAPEINFERENCEENGINE_H

#include "boost/variant.hpp"
#include <string>
#include <unordered_set>
#include <vector>

#include "glow/Support/Error.h"
#include <torch/script.h>

/// Given actual inputs and the glow graph, this class is responsible for
/// slicing the upperbounded outputs from Glow back to the actual output size
/// expected by PyTorch.
namespace glow {

typedef std::vector<int64_t> TensorShape;
typedef std::vector<std::vector<int64_t>> TensorListShape;
using ElemShape = boost::variant<TensorShape, TensorListShape>;

struct TensorOutput {
  TensorShape shapeOrIntValues;
  c10::ScalarType dtype;

  // This flag signals that tensor output contains real value rather than shape
  bool scalar = false;
};

struct TensorListOutput {
  TensorListShape shape;
  c10::ScalarType dtype;
  std::vector<c10::ScalarType> dtypeList;
};
using ElemOutput = boost::variant<TensorOutput, TensorListOutput>;

struct VariableMeta {
  /// For Tensor, Tensor[], store the shape in \p listOfShape[0]
  /// For tuple(Tensor, Tensor[], ...), store shapes in \p listOfShape
  std::vector<ElemShape> listOfShape;
  /// Store Int and IntList value
  std::vector<int64_t> intValue;
  /// Data type
  c10::ScalarType dtype = c10::ScalarType::Float;

  template <typename T> const T &shape() const {
    CHECK_GT(listOfShape.size(), 0);
    return boost::get<T>(listOfShape[0]);
  }
};

using MetaStack = std::vector<VariableMeta>;

class ShapeInferenceEngine {
public:
  ShapeInferenceEngine(const torch::jit::Graph &graph,
                       const at::ArrayRef<torch::jit::IValue> &inputs,
                       const std::string &fusionNodeSymbol = "ShapeInf",
                       const bool &compilationMode = false);

  /// Get all VariableMeta for outputs of the given graph.
  const MetaStack &getGraphOutputShape();

  /// Get the Variable Map which uses const torch::jit::Value * as a key,
  /// VariableMeta as a value.
  const std::unordered_map<const torch::jit::Value *, VariableMeta> &
  getVariableMap();

  /// This run the shape inference engine for the given \p graph and \p inputs.
  /// \returns error of failure.
  Error run();

  /// Collects the list of unsupported symbols present in a \p graph
  /// \returns a set of symbols
  std::unordered_set<std::string>
  findUnsupportedGraphSymbols(bool skipLastFusionNode = false);

  /// Print shapeMap_ as format:
  /// %5: [2 4]
  void printShapeMap();

private:
  /// Graph that needs to be run shape inference.
  const torch::jit::Graph &graph_;

  /// Actual inputs of the given \p graph.
  const at::ArrayRef<torch::jit::IValue> &inputs_;

  /// Glow fusion node symbol.
  const std::string fusionNodeSymbol_;

  const bool compilationMode_;

  /// This is a mapping which uses torch::jit::Value as a key, VariableMeta as a
  /// value. It is used for tracking the shape for tensors and values for
  /// integers in a graph.
  using ShapeMap = std::unordered_map<const torch::jit::Value *, VariableMeta>;
  ShapeMap shapeMap_;

  /// A set containing all ops that should be skipped during shape inference
  std::unordered_set<std::string> blockList_;

  /// Store shapes of all the outputs in a graph.
  MetaStack outputShape_;

  /// Offset flag for aten::embedding_bag and embedding_bag_byte_rowwise_offsets
  /// In Glow, \p hasEndOffset_ always true
  static bool const hasEndOffset_ = true;

  /// Run shape inference on a graph
  Error runGraph(const torch::jit::Graph &,
                 const at::ArrayRef<torch::jit::IValue> &);

  /// Run shape inference on a sub graph
  Error runSubGraph(const torch::jit::Graph &,
                    const at::ArrayRef<torch::jit::IValue> &);

  /// Collects the list of unsupported symbols present in a \p graph
  /// populates the provided set of symbol names
  void findUnsupportedGraphSymbols(const torch::jit::Graph &,
                                   std::unordered_set<std::string> &,
                                   bool skipLastFusionNode = false);

  /// \return true if the node's symbol is supported for shape inference
  static bool isSupportedNodeSymbol(const torch::jit::Node *);

  /// print graph for debugging purpose
  void printGraph(const torch::jit::Graph &graph, int64_t level);

  /// Put shape/type info of actual graph inputs into \p shapeMap_.
  Error getGraphInputShapeType(const torch::jit::Graph &,
                               const at::ArrayRef<torch::jit::IValue> &);

  /// Extract shape info of graph outputs from \p shapeMap_.
  Error generateGraphOutputShape();

  /// Extract shape info of node inputs from \p shapeMap_.
  bool getNodeInputShape(const torch::jit::Node *node, MetaStack &inputMetas);

  /// Infer shapes of node outputs
  Error shapeOnNode(const torch::jit::Node *node);

  struct ShapeInference {
    using InferenceFn0 = Expected<TensorOutput> (*)(const MetaStack &);
    using InferenceFn1 = Expected<TensorOutput> (*)(const torch::jit::Node *);
    using InferenceFn2 = Expected<TensorOutput> (*)(const MetaStack &,
                                                    const torch::jit::Node *);
    using InferenceFn3 = Expected<TensorListOutput> (*)(const MetaStack &);
    using InferenceFn4 =
        Expected<TensorListOutput> (*)(const torch::jit::Node *);
    using InferenceFn5 = Expected<TensorListOutput> (*)(
        const MetaStack &, const torch::jit::Node *);
    using InferenceFn6 = Expected<ElemOutput> (*)(const MetaStack &);

    using AddShapeFn0 = void (ShapeInferenceEngine::*)(const torch::jit::Node *,
                                                       TensorOutput &);
    using AddShapeFn1 = void (ShapeInferenceEngine::*)(const torch::jit::Node *,
                                                       TensorListOutput &);
    using AddShapeFn2 = void (ShapeInferenceEngine::*)(const torch::jit::Node *,
                                                       ElemOutput &);

    using InferenceFn =
        boost::variant<InferenceFn0, InferenceFn1, InferenceFn2, InferenceFn3,
                       InferenceFn4, InferenceFn5, InferenceFn6>;
    using AddShapeFn = boost::variant<AddShapeFn0, AddShapeFn1, AddShapeFn2>;

    ShapeInference(InferenceFn inferenceFn, AddShapeFn addShapeFn)
        : inferenceFn(inferenceFn), addShapeFn(addShapeFn) {}

    Error infer(ShapeInferenceEngine *engine, const MetaStack &meta,
                const torch::jit::Node *node) const;

    InferenceFn inferenceFn;
    AddShapeFn addShapeFn;
  };

  using SymbolToFunctionMap = std::unordered_map<std::string, ShapeInference>;

  /// Build mapping from jit symbols to inference functions
  static SymbolToFunctionMap buildShapeSymbolMapping();

  /// Retrieve the static copy of the jit symbols to shape inference functions
  static const SymbolToFunctionMap &getShapeSymbolMapping();

  void addShapeConstant(const torch::jit::Node *node, TensorOutput &output);

  void addShapeListConstruct(const torch::jit::Node *node,
                             TensorListOutput &output);

  void addShapeBag(const torch::jit::Node *node, TensorOutput &output);

  void addShapeChunk(const torch::jit::Node *node, TensorListOutput &output);

  void addShapeSlice(const torch::jit::Node *node, ElemOutput &output);

  void addShapeDefault(const torch::jit::Node *node, TensorOutput &output);

  void addShapeDefaultList(const torch::jit::Node *node,
                           TensorListOutput &output);

  static bool isScalarInt(const VariableMeta &vm);

  // Shape inference for prim::Constant
  static Expected<TensorOutput> primConstant(const torch::jit::Node *node);
  // Shape inference for aten::tanh, aten::relu, aten::sigmoid
  static Expected<TensorOutput> unaryOp(const MetaStack &variableMetas);
  // Shape inference for aten::add, aten::mul, aten::pow
  static Expected<TensorOutput> binaryOp(const MetaStack &variableMetas,
                                         const torch::jit::Node *ptNode);
  // Shape inference for aten::add
  static Expected<TensorOutput> add(const MetaStack &variableMetas,
                                    const torch::jit::Node *ptNode);
  // Shape inference for aten::mul
  static Expected<TensorOutput> mul(const MetaStack &variableMetas,
                                    const torch::jit::Node *ptNode);
  // Shape inference for aten::mm
  static Expected<TensorOutput> mm(const MetaStack &variableMetas);
  // Shape inference for aten::bmm
  static Expected<TensorOutput> bmm(const MetaStack &variableMetas);
  // Shape inference for aten::addmm
  static Expected<TensorOutput> addmm(const MetaStack &variableMetas,
                                      const torch::jit::Node *ptNode);
  // Shape inference for aten::t
  static Expected<TensorOutput> t(const MetaStack &variableMetas);
  // Shape inference for aten::sum, aten::mean
  static Expected<TensorOutput> reduceOp(const MetaStack &variableMetas);
  // Shape inference for prim::ConstantChunk
  static Expected<TensorListOutput>
  constantChunk(const MetaStack &variableMetas, const torch::jit::Node *node);
  // Shape inference for prim::FusedConcat
  static Expected<TensorOutput> fusedConcat(const MetaStack &variableMetas,
                                            const torch::jit::Node *node);
  static Expected<TensorOutput>
  fusedBroadcastConcat(const MetaStack &variableMetas,
                       const torch::jit::Node *node);
  // Shape inference for prim::ListConstruct
  static Expected<TensorListOutput>
  listConstruct(const MetaStack &variableMetas, const torch::jit::Node *node);
  // Shape inference for aten::permute
  static Expected<TensorOutput> permute(const MetaStack &variableMetas);
  // Shape inference for aten::reshape
  static Expected<TensorOutput> reshape(const MetaStack &variableMetas,
                                        const torch::jit::Node *node);
  // Shape inference for aten::slice
  static Expected<ElemOutput> slice(const MetaStack &variableMetas);
  // Shape inference for aten::cat
  static Expected<TensorOutput> cat(const MetaStack &variableMetas);
  // Shape inference for aten::transpose
  static Expected<TensorOutput> transpose(const MetaStack &variableMetas);
  // Shape inference for aten::flatten
  static Expected<TensorOutput> flatten(const MetaStack &variableMetas);
  // Shape inference for glow::fused_stack
  static Expected<TensorOutput> fusedStack(const MetaStack &variableMetas,
                                           const torch::jit::Node *node);
  static Expected<TensorOutput>
  fusedBroadcastStack(const MetaStack &variableMetas,
                      const torch::jit::Node *node);
  // Shape inference for glow::fused_split
  static Expected<TensorListOutput> fusedSplit(const MetaStack &variableMetas);
  // Shape inference for quantized::embedding_bag_byte_rowwise_offsets
  static Expected<TensorOutput>
  quantizedEmbeddingBagByteRowwiseOffsets(const MetaStack &variableMetas);
  // Shape inference for fb::embedding_bag_4bit_rowwise_offsets
  static Expected<TensorOutput>
  embeddingBag4BitRowwiseOffsets(const MetaStack &variableMetas);
  // Shape inference for quantized::linear
  static Expected<TensorOutput>
  glowUnpackedQuantizedLinear(const MetaStack &variableMetas);
  // Shape inference for aten::embedding_bag
  static Expected<TensorOutput> embeddingBag(const MetaStack &variableMetas);
  // Shape inference for fb::glowEmbedding_bag
  static Expected<TensorOutput>
  glowEmbeddingBag(const MetaStack &variableMetas);
  // Shape inference for fb::glow_embedding_bag_byte_rowwise_offsets
  static Expected<TensorOutput>
  quantizedGlowEmbeddingBagByteRowwiseOffsets(const MetaStack &variableMetas);
  // Shape inference for fb::glow_embedding_bag_4bit_rowwise_offsets
  static Expected<TensorOutput>
  quantizedGlowEmbeddingBag4BitRowwiseOffsets(const MetaStack &variableMetas);
  // Shape inference for fb::xl_embedding_bag
  static Expected<TensorOutput> xlEmbeddingBag(const MetaStack &variableMetas);
  static Expected<TensorOutput>
  quantizedXLEmbeddingBagByteRowwiseOffsets(const MetaStack &variableMetas);
  // Shape inference for fb::glow_embedding_bag_4bit_rowwise_offsets
  static Expected<TensorOutput>
  quantizedXLEmbeddingBag4BitRowwiseOffsets(const MetaStack &variableMetas);
  // Shape inference for aten::chuck
  static Expected<TensorListOutput> chunk(const MetaStack &variableMetas);
  // Shape inference for aten::stack
  static Expected<TensorOutput> stack(const MetaStack &variableMetas);
  // Shape inference for prim::listunpack
  static Expected<TensorListOutput> listUnpack(const MetaStack &variableMetas);
  // Shape inference for aten::to
  static Expected<TensorOutput> to(const MetaStack &variableMetas);
  // Shape inference for fb::lengths_to_offsets
  static Expected<TensorOutput>
  lengthsToOffsets(const MetaStack &variableMetas);
  // Shape inference for fb::Fused8BitRowwiseQuantizedToFloat
  static Expected<TensorOutput>
  fused8BitRowwiseQuantizedToFloat(const MetaStack &variableMetas);
  // Shape inference for prim::dtype
  static Expected<TensorOutput> primDtype(const MetaStack &variableMetas);
  // Shape inference for fb::fast_gather
  static Expected<TensorOutput> fastGather(const MetaStack &variableMetas);
  // Shape inference for fb::lengths_range
  static Expected<TensorOutput> lengthsRange(const MetaStack &variableMetas);
  // Shape inference for aten::quantize_per_tensor
  static Expected<TensorOutput>
  quantizePerTensor(const MetaStack &variableMetas);
  // Shape inference for aten::dequantize
  static Expected<TensorOutput> dequantize(const MetaStack &variableMetas);
  // Shape inference for quantized::mul
  static Expected<TensorOutput> quantizedMul(const MetaStack &variableMetas);
  // Shape inference for aten::matmul
  static Expected<TensorOutput> matmul(const MetaStack &variableMetas);
  // Shape inference for aten::layerNorm
  static Expected<TensorOutput> layerNorm(const MetaStack &variableMetas);
  // Shape inference for aten::linear
  static Expected<TensorOutput> linear(const MetaStack &variableMetas);
  // Shape inference for fb::compressed_indices_remap and
  // fb::xl_compressed_indices_remap
  static Expected<TensorListOutput>
  compressedIndicesRemap(const MetaStack &variableMetas);
  // Shape inference for quantized::embedding_bag_byte_unpack
  static Expected<TensorOutput>
  embeddingBagByteUnpack(const MetaStack &variableMetas);
  // Shape inference for fb::unsqueeze_n_times
  static Expected<TensorOutput> unsqueezeNTimes(const MetaStack &variableMetas);
  // Shape inference for fb::equally_split
  static Expected<TensorListOutput>
  equallySplit(const MetaStack &variableMetas);
  // Shape inference for aten::squeeze
  static Expected<TensorOutput> squeeze(const MetaStack &variableMetas);
  // Shape inference for aten::narrow
  static Expected<TensorOutput> narrow(const MetaStack &variableMetas);
  // Shape inference for fb::index_hash
  static Expected<TensorOutput> indexHash(const MetaStack &variableMetas);
  // Shape inference for fb::bucketize
  static Expected<TensorOutput> bucketize(const MetaStack &variableMetas);
  // Shape inference for fb::expand_dims
  static Expected<TensorOutput> expandDims(const MetaStack &variableMetas);
  // Shape inference for aten::split_with_sizes
  static Expected<TensorListOutput>
  splitWithSizes(const MetaStack &variableMetas);
  // Shape inference for aten::Int
  static Expected<TensorOutput> inferInt(const MetaStack &variableMetas);
  // Shape inference for prim::NumToTensor
  static Expected<TensorOutput> numToTensor(const MetaStack &variableMetas,
                                            const torch::jit::Node *node);
  // Shape inference for aten::size
  static Expected<TensorOutput> size(const MetaStack &variableMetas);
  // Shape inference for fb::scale_gradient
  static Expected<TensorOutput> scaleGradient(const MetaStack &variableMetas);
  // Shape inference for aten::repeat
  static Expected<TensorOutput> repeat(const MetaStack &variableMetas);
  // Shape inference for aten::softmax
  static Expected<TensorOutput> softmax(const MetaStack &variableMetas);
  // Shape inference for aten::unsqueeze
  static Expected<TensorOutput> unsqueeze(const MetaStack &variableMetas);
};

} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_SHAPEINFERENCEENGINE_H
