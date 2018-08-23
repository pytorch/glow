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

#include "glow/Importer/ProtobufLoader.h"

#include <cassert>
#include <string>

namespace glow {

bool isArrayConstant(llvm::ArrayRef<size_t> a) {
  for (size_t i = 1; i < a.size(); i++)
    if (a[0] != a[i])
      return false;
  return true;
}

Tensor *ProtobufLoader::getTensorByName(llvm::StringRef name) {
  assert(tensors_.count(name) &&
         "There is no tensor registered with this name.");
  return tensors_[name];
}

Variable *ProtobufLoader::getOutputByName(llvm::StringRef name) const {
  assert(outputVarsByName_.count(name) &&
         "There is no Variable registered with this name.");
  auto it = outputVarsByName_.find(name);
  assert(it != outputVarsByName_.end() &&
         "No external output Variable was registered with this name.");
  return it->second;
}

NodeValue
ProtobufLoader::getNodeValueByNameOrNullNodeValue(llvm::StringRef name) const {
  auto it = nodeValueByName_.find(name);
  if (it != nodeValueByName_.end()) {
    return it->second;
  }

  return NodeValue(nullptr);
}

NodeValue ProtobufLoader::getNodeValueByName(llvm::StringRef name) const {
  assert(hasNodeByName(name) && "No node under that name");
  auto node = getNodeValueByNameOrNullNodeValue(name);
  assert(node.getNode() && "Null is under that name??");
  return node;
}

Variable *ProtobufLoader::createAndRememberVariable(
    llvm::StringRef name, const Tensor &tensor, VisibilityKind visibilityKind) {
  assert(!hasNodeByName(name) && "Creating an already existing node?!");
  // Note: We do not support training from models loaded from protos, so
  // trainable is always set to false here.
  Variable *node = G_.getParent()->createVariable(name, tensor, visibilityKind,
                                                  /* trainable */ false);
  nodeValueByName_[name] = NodeValue(node, 0);

  return node;
}

NodeValue
ProtobufLoader::getNodeValueOrCreateVariableByName(llvm::StringRef name) {
  auto node = getNodeValueByNameOrNullNodeValue(name);
  if (node.getNode()) {
    return node;
  }

  Tensor *T = getTensorByName(name);
  return NodeValue(createAndRememberVariable(name, *T), 0);
}

Variable *ProtobufLoader::getVariableByName(llvm::StringRef name) const {
  assert(hasNodeByName(name) && "Variable was not created");
  auto *node = getNodeValueByName(name).getNode();

  assert(llvm::isa<Variable>(node) && "Node is not a variable");

  return llvm::cast<Variable>(node);
}

bool ProtobufLoader::hasNodeByName(llvm::StringRef name) const {
  return getNodeValueByNameOrNullNodeValue(name).getNode() != nullptr;
}

ProtobufLoader::ProtobufLoader(llvm::ArrayRef<const char *> tensorNames,
                               llvm::ArrayRef<Tensor *> tensors, Function &F)
    : G_(F) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  assert(tensorNames.size() == tensors.size() && "Invalid initialization list");
  for (unsigned i = 0; i < tensorNames.size(); i++) {
    assert(!hasNodeByName(tensorNames[i]) && "Input names have duplicate");
    createAndRememberVariable(tensorNames[i], *tensors[i],
                              VisibilityKind::Public);
  }
}

ProtobufLoader::~ProtobufLoader() {
  for (auto &it : tensors_) {
    delete it.second;
  }
}

}; // namespace glow
