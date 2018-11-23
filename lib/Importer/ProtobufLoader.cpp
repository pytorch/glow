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

llvm::Expected<Tensor *> ProtobufLoader::getTensorByName(llvm::StringRef name) {
  RETURN_ERR_IF_NOT(tensors_.count(name),
                    "There is no tensor registered with this name.");
  return tensors_[name];
}

llvm::Expected<Placeholder *>
ProtobufLoader::getOutputByName(llvm::StringRef name) const {
  RETURN_ERR_IF_NOT(outputVarsByName_.count(name),
                    "There is no Variable registered with this name.");
  auto it = outputVarsByName_.find(name);
  RETURN_ERR_IF_NOT(
      it != outputVarsByName_.end(),
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

llvm::Expected<NodeValue>
ProtobufLoader::getNodeValueByName(llvm::StringRef name) const {
  RETURN_ERR_IF_NOT(hasNodeByName(name), "No node under that name");
  auto node = getNodeValueByNameOrNullNodeValue(name);
  RETURN_ERR_IF_NOT(node.getNode(), "Null is under that name??");
  return node;
}

llvm::Expected<Constant *>
ProtobufLoader::createAndRegisterConstant(llvm::StringRef name,
                                          const Tensor &tensor) {
  RETURN_ERR_IF_NOT(!hasNodeByName(name), "Creating an already existing node");
  // Note: We do not support training from models loaded from protos, so
  // trainable is always set to false here.
  Constant *node = G_.getParent()->createConstant(name, tensor);
  nodeValueByName_[name] = NodeValue(node, 0);
  return node;
}

llvm::Expected<Placeholder *>
ProtobufLoader::createAndRegisterPlaceholder(llvm::StringRef name, TypeRef T) {
  RETURN_ERR_IF_NOT(!hasNodeByName(name), "Creating an already existing node");
  Placeholder *node = G_.getParent()->createPlaceholder(T, name, false);
  nodeValueByName_[name] = NodeValue(node, 0);
  return node;
}

llvm::Expected<NodeValue>
ProtobufLoader::getNodeValueOrCreateConstantByName(llvm::StringRef name) {
  auto node = getNodeValueByNameOrNullNodeValue(name);
  if (node.getNode()) {
    return node;
  }

  Tensor *T;
  ASSIGN_VALUE_OR_RETURN_ERR(T, getTensorByName(name));
  Constant *c;
  ASSIGN_VALUE_OR_RETURN_ERR(c, createAndRegisterConstant(name, *T));
  return NodeValue(c, 0);
}

bool ProtobufLoader::hasNodeByName(llvm::StringRef name) const {
  return getNodeValueByNameOrNullNodeValue(name).getNode() != nullptr;
}

ProtobufLoader::ProtobufLoader(llvm::ArrayRef<const char *> tensorNames,
                               llvm::ArrayRef<TypeRef> types, Function &F)
    : G_(F) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  assert(tensorNames.size() == types.size() && "Invalid initialization list");
  for (unsigned i = 0; i < tensorNames.size(); i++) {
    assert(!hasNodeByName(tensorNames[i]) && "Input names have duplicate");
    TEMP_UNWRAP(createAndRegisterPlaceholder(tensorNames[i], types[i]));
  }
}

ProtobufLoader::~ProtobufLoader() {
  for (auto &it : tensors_) {
    delete it.second;
  }
}

}; // namespace glow
