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

#include <string>

namespace glow {

bool isArrayConstant(llvm::ArrayRef<size_t> a) {
  for (size_t i = 1; i < a.size(); i++)
    if (a[0] != a[i])
      return false;
  return true;
}

Constant *ProtobufLoader::getConstantByNameOrNull(llvm::StringRef name) const {
  auto it = nodeValueByName_.find(name);
  if (it == nodeValueByName_.end()) {
    return nullptr;
  }
  auto *res = llvm::dyn_cast<Constant>(it->second.getNode());
  return res ? res : nullptr;
}

llvm::Expected<Constant *>
ProtobufLoader::getConstantByName(llvm::StringRef name) const {
  auto *ptr = getConstantByNameOrNull(name);
  RETURN_ERR_IF_NOT(
      ptr, strFormat("could not find constant with name %s", name.data()));
  return ptr;
}

bool ProtobufLoader::hasConstantByName(llvm::StringRef name) const {
  return getConstantByNameOrNull(name) != nullptr;
}

llvm::Expected<Placeholder *>
ProtobufLoader::getOutputByName(llvm::StringRef name) const {
  auto it = outputVarsByName_.find(name);
  RETURN_ERR_IF_NOT(
      it != outputVarsByName_.end(),
      llvm::Twine("No external output Variable was registered with name ", name)
          .str());
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
  RETURN_ERR_IF_NOT(hasNodeByName(name),
                    llvm::Twine("No node under name ", name).str());
  auto node = getNodeValueByNameOrNullNodeValue(name);
  RETURN_ERR_IF_NOT(node.getNode(), "Null is under that name??");
  return node;
}

llvm::Error ProtobufLoader::createAndRegisterConstant(llvm::StringRef name,
                                                      Tensor &&tensor) {
  auto it = nodeValueByName_.find(name);
  if (it != nodeValueByName_.end()) {
    if (llvm::dyn_cast<Placeholder>(it->second.getNode())) {
      // Placeholders take precedence over Constants.
      return llvm::Error::success();
    }
  }
  // Note: We do not support training from models loaded from protos, so
  // trainable is always set to false here.
  Constant *node = G_.getParent()->createConstant(name, std::move(tensor));
  nodeValueByName_[name] = node->getOutput();
  return llvm::Error::success();
}

void ProtobufLoader::deleteUnusedConstants() {
  std::vector<std::string> nodeValuesToRemove;
  for (auto &kv : nodeValueByName_) {
    auto *node = kv.second.getNode();
    if (auto *c = llvm::dyn_cast<Constant>(node)) {
      if (!c->hasUsers()) {
        nodeValuesToRemove.push_back(kv.getKey());
      }
    }
  }

  for (auto &name : nodeValuesToRemove) {
    auto it = nodeValueByName_.find(name);
    auto *c = llvm::dyn_cast<Constant>(it->second.getNode());
    DCHECK(c) << "NodeValue with name " << name
              << " was expected to have been a Constant";
    G_.getParent()->eraseConstant(c);
    nodeValueByName_.erase(it);
  }
}

llvm::Expected<Placeholder *>
ProtobufLoader::createAndRegisterPlaceholder(llvm::StringRef name, TypeRef T) {
  RETURN_ERR_IF_NOT(
      !hasNodeByName(name),
      llvm::Twine("Creating an already existing node ", name).str());
  Placeholder *node = G_.getParent()->createPlaceholder(T, name, false);
  nodeValueByName_[name] = node->getOutput();
  return node;
}

bool ProtobufLoader::hasNodeByName(llvm::StringRef name) const {
  return getNodeValueByNameOrNullNodeValue(name).getNode() != nullptr;
}

ProtobufLoader::ProtobufLoader(llvm::ArrayRef<const char *> tensorNames,
                               llvm::ArrayRef<TypeRef> types, Function &F,
                               llvm::Error *errPtr)
    : G_(F) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  // if errPtr already contains an error then don't continue with constructor
  if (errPtr && *errPtr) {
    return;
  }

  // Lambda to setup the ProtobufLoader and return any llvm::Errors that were
  // raised.
  auto setup = [&]() -> llvm::Error {
    RETURN_ERR_IF_NOT(tensorNames.size() == types.size(),
                      "Invalid initialization list");
    for (size_t i = 0, e = tensorNames.size(); i < e; i++) {
      RETURN_ERR_IF_NOT(!hasNodeByName(tensorNames[i]),
                        "Input names have duplicate");
      auto placeholderOrErr =
          createAndRegisterPlaceholder(tensorNames[i], types[i]);
      if (!placeholderOrErr) {
        return placeholderOrErr.takeError();
      }
    }
    return llvm::Error::success();
  };

  if (errPtr) {
    *errPtr = setup();
  } else {
    EXIT_ON_ERR(setup());
  }
}

}; // namespace glow
