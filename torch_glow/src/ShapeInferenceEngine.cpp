// Copyright 2004-present Facebook. All Rights Reserved.

#include <iostream>
#include <string>
#include <torch/script.h>
#include <unordered_set>
#include <vector>

#include "ShapeInferenceEngine.h"

#include "glow/Support/Error.h"
#include "glow/Support/Support.h"

namespace glow {

ShapeInferenceEngine::ShapeInferenceEngine(
    const torch::jit::Graph &graph, const at::ArrayRef<at::IValue> &inputs)
    : graph_(graph), inputs_(inputs){};

void ShapeInferenceEngine::getNodeInputShape(const torch::jit::Node *node,
                                             ShapeStack &inputShapes) {
  for (auto input : node->inputs()) {
    auto it = shapeMap_.find(input);
    CHECK(it != shapeMap_.end());
    inputShapes.emplace_back(shapeMap_[input]);
  }
}

/// This function could only cover the operation which produces one output
/// case.
/// TODO: Since a node may have multiple outputs, this func will support
/// multiple outputs later
std::vector<int64_t> &
ShapeInferenceEngine::getNodeOutputShape(const torch::jit::Node *node) {
  return shapeMap_[node->output()];
}

ShapeStack ShapeInferenceEngine::getGraphOutputShape() {
  return outputShapeMeta_;
}

Error ShapeInferenceEngine::shapeOnNode(const torch::jit::Node *node) {

  /// Get op symbol
  const auto kind = node->kind();

  /// Get shapes of inputs from shape mapping
  ShapeStack inputShapes;

  /// Get shapes of outputs from shape mapping
  /// Right now, all of the supported ops have single output. Multiple outputs
  /// cases will be covered later. The \p output_shape is for single output
  /// scenario. The \p output_shapes is for multiple output scenario.
  /// TODO: Cover more ops which will generate multiple outputs
  std::vector<int64_t> outputShape;
  ShapeStack outputShapes;

  getNodeInputShape(node, inputShapes);

  // Get output shape
  switch (kind) {
  case c10::prim::Constant: {
    ASSIGN_VALUE_OR_RETURN_ERR(outputShape, primConstant(node));
    break;
  }
  case c10::aten::sub:
  case c10::aten::add: {
    ASSIGN_VALUE_OR_RETURN_ERR(outputShape, add(inputShapes));
    break;
  }
  case c10::aten::mm: {
    ASSIGN_VALUE_OR_RETURN_ERR(outputShape, mm(inputShapes));
    break;
  }
  case c10::aten::bmm: {
    ASSIGN_VALUE_OR_RETURN_ERR(outputShape, bmm(inputShapes));
    break;
  }
  default: { return MAKE_ERR("Node is not supported"); }
  }

  // Put output shape into map
  if (node->outputs().size() == 1) {
    shapeMap_[node->output()] = outputShape;
  } else {
    for (int i = 0; i < outputShapes.size(); i++) {
      shapeMap_[node->output(i)] = outputShapes[i];
    }
  }
  return Error::success();
}

Error ShapeInferenceEngine::run() {

  if (inputs_.size() != graph_.inputs().size()) {
    return MAKE_ERR(
        "Number of inputs mismatch between Graph and actual inputs");
  }

  /// Put graph input into shape mapping
  getGraphIntputShape();

  for (auto *node : graph_.nodes()) {
    RETURN_IF_ERR(shapeOnNode(node));
  }

  /// Extract output from shape mapping
  generateGraphOutputShape();
  return Error::success();
}

void ShapeInferenceEngine::printShapeMap() {
  for (auto elem : shapeMap_) {
    std::cout << elem.first << " ";
    for (auto value : elem.second) {
      std::cout << value << " ";
    }
    std::cout << std::endl;
  }
}

void ShapeInferenceEngine::getGraphIntputShape() {
  for (auto i = 0; i < inputs_.size(); i++) {
    auto gInName = graph_.inputs()[i];
    shapeMap_[gInName] = {};

    auto input = inputs_[i];
    if (input.isTensor()) {
      auto ptTensor = input.toTensor();
      for (auto s : ptTensor.sizes()) {
        shapeMap_[gInName].emplace_back(s);
      }
    } else if (input.isBool() || input.isInt()) {
      shapeMap_[gInName] = {1};
    } else if (input.isIntList()) {
      auto ptIntList = input.toIntVector();
      shapeMap_[gInName] = {static_cast<long>(ptIntList.size()), 1};
    }
  }
}

void ShapeInferenceEngine::generateGraphOutputShape() {
  for (auto output : graph_.outputs()) {
    auto it = shapeMap_.find(output);
    CHECK(it != shapeMap_.end());
    outputShapeMeta_.emplace_back(it->second);
  }
}

/// The \p prim::Constant may have multiple types of output, eg.
/// int = prim::Constant[value=0]()
/// Float(1:1) = prim::Constant[value={0}]()
/// bool = prim::Constant[value=0]()
/// None = prim::Constant()
/// Tensor = prim::Constant[value= <Tensor>]()
/// TODO: Cover Tensor case.
Expected<std::vector<int64_t>>
ShapeInferenceEngine::primConstant(const torch::jit::Node *node) {

  if (node->inputs().size() != 0) {
    return MAKE_ERR("Expect zero input of prim::Constant Op.");
  }
  std::vector<int64_t> shape;
  at::TypePtr type = node->output()->type();

  if (type->isSubtypeOf(at::NumberType::get()) ||
      type->isSubtypeOf(at::BoolType::get())) {
    shape = {1};
  } else if (type->isSubtypeOf(at::NoneType::get())) {
    shape = {0};
  }
  return shape;
}

Expected<std::vector<int64_t>>
ShapeInferenceEngine::add(const ShapeStack &shapeMeta) {

  if (shapeMeta.size() != 2 && shapeMeta.size() != 3) {
    return MAKE_ERR("Expected two or three inputs shapes of this operation.");
  }

  std::vector<int64_t> t0 = shapeMeta[0];
  std::vector<int64_t> t1 = shapeMeta[1];

  auto d0 = t0.size();
  auto d1 = t1.size();
  size_t dim = std::max(d0, d1);
  std::vector<int64_t> shape(dim);

  for (auto i = 0; i < dim; i++) {
    auto j = -1 - i;
    if (i >= d0 || t0[d0 + j] == 1) {
      shape[dim + j] = t1[d1 + j];
    } else if (i >= d1 || t1[d1 + j] == 1) {
      shape[dim + j] = t0[d0 + j];
    } else {
      if (t1[d1 + j] != t0[d0 + j]) {
        return MAKE_ERR(
            strFormat("The size of tensor a (%zu) must match the size of "
                      "tensor b (%zu)at non-singleton dimension 1.",
                      t0[d0 + j], t1[d1 + j]));
      }

      shape[dim + j] = t1[d1 + j];
    }
  }
  return shape;
}

Expected<std::vector<int64_t>>
ShapeInferenceEngine::mm(const ShapeStack &shapeMeta) {

  if (shapeMeta.size() != 2) {
    return MAKE_ERR("Expected two inputs shapes of this operation.");
  }

  std::vector<int64_t> t0 = shapeMeta[0];
  std::vector<int64_t> t1 = shapeMeta[1];

  if (!(t1.size() == 2 && t0.size() == 2)) {
    return MAKE_ERR("Expected 2-dimensional tensor.");
  }

  if (t0[1] != t1[0]) {
    return MAKE_ERR(
        strFormat("The size of tensor a (%zu) at dimension 1 must match the "
                  "size of tensor b (%zu) at dimension 0.",
                  t0[1], t1[0]));
  }

  std::vector<int64_t> shape = {t0[0], t1[1]};
  return shape;
}

Expected<std::vector<int64_t>>
ShapeInferenceEngine::bmm(const ShapeStack &shapeMeta) {

  if (shapeMeta.size() != 2) {
    return MAKE_ERR("Expected two inputs shapes of this operation.");
  }

  std::vector<int64_t> t0 = shapeMeta[0];
  std::vector<int64_t> t1 = shapeMeta[1];

  if (!(t0.size() == 3 && t1.size() == 3)) {
    return MAKE_ERR("Expected 3-dimensional tensor.");
  }

  if (t0[0] != t1[0]) {
    return MAKE_ERR("Expected tensors to have same size at dimension 0");
  }

  if (t0[2] != t1[1]) {
    return MAKE_ERR(strFormat("The size of tensor a (%zu) at dimension 2 must"
                              "match the size of tensor b (%zu) at dimension 1",
                              t0[2], t1[1]));
  }
  std::vector<int64_t> shape = {t0[0], t0[1], t1[2]};
  return shape;
}

} // namespace glow
