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

#include "PyTorchModelLoader.h"

namespace {
glow::Type ptTypeToGlowType(const at::CompleteTensorType &ptType) {
  // TODO: get correct ElemKind
  std::vector<size_t> dims;
  for (auto &size : ptType.sizes()) {
    dims.push_back(size);
  }
  return glow::Type(glow::ElemKind::FloatTy, dims);
}
} // namespace

glow::Placeholder *PyTorchModelLoader::loadValue(const torch::jit::Value *val) {
  // TODO: do we need to care about optional IValues here?
  assert(val->isCompleteTensor());
  auto ptType = val->type()->cast<at::CompleteTensorType>();
  auto glowType = ptTypeToGlowType(*ptType.get());
  glow::Placeholder *ph =
      mod_.createPlaceholder(&glowType, "input", /*isTrainable*/ false);
  valueMap_[val] = ph->getOutput();
  return ph;
}

void PyTorchModelLoader::loadNode(const torch::jit::Node *ptNode) {
  glow::Node *glowNode = nullptr;

  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();

  switch (ptNode->kind()) {
  case at::aten::mul: {
    assert(inputs.size() == 2);
    glow::NodeValue rhs = valueMap_.at(inputs[0]);
    glow::NodeValue lhs = valueMap_.at(inputs[1]);
    glowNode = f_->createMul("mul", rhs, lhs);
    valueMap_[outputs[0]] = glowNode->getNthResult(glow::MulNode::ResultIdx);
    break;
  }
  case at::aten::div: {
    assert(inputs.size() == 2);
    glow::NodeValue rhs = valueMap_.at(inputs[0]);
    glow::NodeValue lhs = valueMap_.at(inputs[1]);
    glowNode = f_->createDiv("div", rhs, lhs);
    valueMap_[outputs[0]] = glowNode->getNthResult(glow::DivNode::ResultIdx);
    break;
  }
  default:
    assert(false && "unrecognized node");
  }
}

void PyTorchModelLoader::load() {
  assert(f_ == nullptr && "This model loader has already been used.");

  static std::atomic<size_t> nextFuncId{0};
  f_ =
      mod_.createFunction(glow::strFormat("PyTorchFunction_%lu", nextFuncId++));

  auto subgraphInputValues = subgraph_->inputs();

  assert(inputs_.size() == subgraphInputValues.size());

  for (size_t i = 0; i < subgraphInputValues.size(); ++i) {
    torch::jit::Value *inputValue = subgraphInputValues[i];
    const c10::IValue inputIValue = inputs_[i];
    inputValue->inferTypeFrom(inputIValue.toTensor());
    inputPlaceholders_.push_back(loadValue(inputValue));
  }

  // Nodes are topologically sorted.
  for (auto node : subgraph_->nodes()) {
    loadNode(node);
  }

  for (torch::jit::Value *output : subgraph_->outputs()) {
    glow::NodeValue outputNodeValue = valueMap_.at(output);
    auto *save = f_->createSave("save", outputNodeValue);
    outputPlaceholders_.push_back(save->getPlaceholder());
  }
}

PyTorchModelLoader::PyTorchModelLoader(glow::Module &mod,
                                       torch::jit::Graph *subgraph,
                                       at::ArrayRef<torch::jit::IValue> &inputs)
    : mod_(mod), subgraph_(subgraph), inputs_(inputs) {}

// static
bool PyTorchModelLoader::isNodeSupported(const torch::jit::Node *node) {
  switch (node->kind()) {
  case at::aten::mul:
  case at::aten::div:
    return true;
  default:
    return false;
  }
  return false;
}
