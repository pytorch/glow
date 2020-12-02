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
};

struct TensorListOutput {
  TensorListShape shape;
  c10::ScalarType dtype;
};

struct VariableMeta {
  /// For Tensor, Tensor[], store the shape in \p listOfShape[0]
  /// For tuple(Tensor, Tensor[], ...), store shapes in \p listOfShape
  std::vector<ElemShape> listOfShape;
  /// Store Int and IntList value
  std::vector<int64_t> intValue;
  /// Data type
  c10::ScalarType dtype = c10::ScalarType::Float;

  template <typename T> const T &shape() const {
    return boost::get<T>(listOfShape[0]);
  }
};

using MetaStack = std::vector<VariableMeta>;

class ShapeInferenceEngine {
public:
  ShapeInferenceEngine(const torch::jit::Graph &graph,
                       const at::ArrayRef<torch::jit::IValue> &inputs,
                       const std::string &fusionNodeSymbol = "ShapeInf");

  /// Get all VariableMeta for outputs of the given graph.
  const MetaStack &getGraphOutputShape();

  /// Get the Variable Map which uses const torch::jit::Value * as a key,
  /// VariableMeta as a value.
  const std::unordered_map<const torch::jit::Value *, VariableMeta> &
  getVariableMap();

  /// This run the shape inference engine for the given \p graph and \p inputs.
  /// \returns error of failure.
  Error run();

private:
  /// Graph that needs to be run shape inference.
  const torch::jit::Graph &graph_;

  /// Actual inputs of the given \p graph.
  const at::ArrayRef<torch::jit::IValue> &inputs_;

  /// Glow fusion node symbol.
  const std::string fusionNodeSymbol_;

  /// This is a mapping which uses torch::jit::Value as a key, VariableMeta as a
  /// value. It is used for tracking the shape for tensors and values for
  /// integers in a graph.
  std::unordered_map<const torch::jit::Value *, VariableMeta> shapeMap_;

  /// A set containing all ops that should be skipped during shape inference
  std::unordered_set<std::string> blockList_;

  /// Store shapes of all the outputs in a graph.
  MetaStack outputShape_;

  /// Offset flag for aten::embedding_bag and embedding_bag_byte_rowwise_offsets
  /// In Glow, \p hasEndOffset_ always true
  static bool const hasEndOffset_ = true;

  /// Run shape inference recursively
  Error runRecursively(const torch::jit::Graph &,
                       const at::ArrayRef<torch::jit::IValue> &);

  /// Print shapeMap_ as format:
  /// %5: [2 4]
  void printShapeMap();

  /// Put shape/type info of actual graph inputs into \p shapeMap_.
  Error getGraphInputShapeType(const torch::jit::Graph &,
                               const at::ArrayRef<torch::jit::IValue> &);

  /// Extract shape info of graph outputs from \p shapeMap_.
  Error generateGraphOutputShape();

  /// Extract shape info of node inputs from \p shapeMap_.
  void getNodeInputShape(const torch::jit::Node *node, MetaStack &inputMetas);

  /// Infer shapes of node outputs
  Error shapeOnNode(const torch::jit::Node *node);

  // Shape inference for prim::Constant
  static Expected<TensorOutput> primConstant(const torch::jit::Node *node);
  // Shape inference for aten::add, aten::mul, aten::pow
  static Expected<TensorOutput> binaryOp(const MetaStack &variableMetas);
  // Shape inference for aten::mm
  static Expected<TensorOutput> mm(const MetaStack &variableMetas);
  // Shape inference for aten::bmm
  static Expected<TensorOutput> bmm(const MetaStack &variableMetas);
  // Shape inference for aten::addmm
  static Expected<TensorOutput> addmm(const MetaStack &variableMetas);
  // Shape inference for aten::t
  static Expected<TensorOutput> t(const MetaStack &variableMetas);
  // Shape inference for prim::ConstantChunk
  static Expected<TensorListOutput>
  constantChunk(const MetaStack &variableMetas, int64_t chunks, int64_t dim);
  // Shape inference for prim::FusedConcat
  static Expected<TensorOutput> fusedConcat(const MetaStack &variableMetas,
                                            int64_t dim);
  // Shape inference for prim::ListConstruct
  static Expected<TensorListOutput>
  listConstruct(const MetaStack &variableMetas);
  // Shape inference for aten::permute
  static Expected<TensorOutput> permute(const MetaStack &variableMetas);
  // Shape inference for aten::reshape
  static Expected<TensorOutput> reshape(const MetaStack &variableMetas);
  // Shape inference for aten::slice
  static Expected<TensorOutput> slice(const MetaStack &variableMetas);
  // Shape inference for aten::cat
  static Expected<TensorOutput> cat(const MetaStack &variableMetas);
  // Shape inference for aten::transpose
  static Expected<TensorOutput> transpose(const MetaStack &variableMetas);
  // Shape inference for aten::flatten
  static Expected<TensorOutput> flatten(const MetaStack &variableMetas);
  // Shape inference for glow::fused_stack
  static Expected<TensorOutput> fusedStack(const MetaStack &variableMetas,
                                           int64_t dim);
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
  // Shape inference for prim::dtype
  static Expected<TensorOutput> primDtype(const MetaStack &variableMetas);
  // Shape inference for fb::fast_gather
  static Expected<TensorOutput> fastGather(const MetaStack &variableMetas);
  // Shape inference for fb::lengths_range
  static Expected<TensorOutput> lengthsRange(const MetaStack &variableMetas);
};

} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_SHAPEINFERENCEENGINE_H
