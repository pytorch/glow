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

#include <string>
#include <unordered_set>
#include <vector>

#include "glow/Support/Error.h"

/// Given actual inputs and the glow graph, this class is responsible for
/// slicing the upperbounded outputs from Glow back to the actual output size
/// expected by PyTorch.
namespace glow {

using ShapeStack = std::vector<std::vector<int64_t>>;

class ShapeInferenceEngine {
public:
  ShapeInferenceEngine(const torch::jit::Graph &graph,
                       const at::ArrayRef<torch::jit::IValue> &inputs);

  /// Get shapes of outputs of the given graph.
  ShapeStack getGraphOutputShape();

  /// This run the shape inference engine for the given \p graph and \p inputs.
  /// \returns error of failure.
  Error run();

private:
  /// Graph that needs to be run shape inference.
  const torch::jit::Graph &graph_;

  /// Actual inputs of the given \p graph.
  const at::ArrayRef<torch::jit::IValue> &inputs_;

  /// This is a mapping which uses torch::jit::Value as a key, Shape as a
  /// value. It is used for tracking the shape of each input or output in a
  /// graph.
  std::unordered_map<const torch::jit::Value *, std::vector<int64_t>> shapeMap_;

  /// Store shapes of all the outputs in a graph.
  ShapeStack outputShapeMeta_;

  /// Print shapeMap_.
  void printShapeMap();

  /// Put shape info of actual graph inputs into \p shapeMap_.
  void getGraphIntputShape();

  /// Extract shape info of graph outputs from \p shapeMap_.
  void generateGraphOutputShape();

  /// Extract shape info of node inputs from \p shapeMap_.
  void getNodeInputShape(const torch::jit::Node *node, ShapeStack &inputShape);

  // This function could only cover the operation which produces one output
  // case.
  // TODO: Since a node may have multiple outputs, this func will support
  // multiple outputs in the next diff
  std::vector<int64_t> &getNodeOutputShape(const torch::jit::Node *node);

  /// Infer shapes of node outputs
  Error shapeOnNode(const torch::jit::Node *node);

  // TODO: Support more Ops
  // Shape inference for prim::Constant
  static Expected<std::vector<int64_t>>
  primConstant(const torch::jit::Node *node);
  // Shape inference for aten::add
  static Expected<std::vector<int64_t>> add(const ShapeStack &shapeMeta);
  // Shape inference for aten::mm
  static Expected<std::vector<int64_t>> mm(const ShapeStack &shapeMeta);
  // Shape inference for aten::bmm
  static Expected<std::vector<int64_t>> bmm(const ShapeStack &shapeMeta);
};

} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_SHAPEINFERENCEENGINE_H
