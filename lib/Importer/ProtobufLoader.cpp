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

Node *ProtobufLoader::getNodeByNameOrNull(llvm::StringRef name) const {
  auto it = nodeByName_.find(name);
  if (it != nodeByName_.end()) {
    return it->second;
  }

  return nullptr;
}

Node *ProtobufLoader::getNodeByName(llvm::StringRef name) const {
  assert(hasNodeByName(name) && "No node under that name");
  auto *node = getNodeByNameOrNull(name);
  assert(node && "Null is under that name??");
  return node;
}

Node *ProtobufLoader::createVariable(llvm::StringRef name, Tensor &tensor,
                                     VisibilityKind visibilityKind,
                                     Variable::TrainKind trainKind) {
  auto *V = G_.getParent()->createVariable(
      tensor.getElementType(), tensor.dims(), name, visibilityKind, trainKind);
  V->copyFrom(&tensor);
  return V;
}

Node *ProtobufLoader::createAndRememberVariable(llvm::StringRef name,
                                                Tensor &tensor,
                                                VisibilityKind visibilityKind,
                                                Variable::TrainKind trainKind) {
  assert(!hasNodeByName(name) && "Creating an already existing node?!");
  auto *V = createVariable(name, tensor, visibilityKind, trainKind);
  nodeByName_[name] = V;
  return V;
}

Node *ProtobufLoader::getOrCreateVariableByName(llvm::StringRef name) {
  auto *node = getNodeByNameOrNull(name);
  if (node) {
    return node;
  }

  Tensor *T = getTensorByName(name);
  return createAndRememberVariable(name, *T);
}

bool ProtobufLoader::hasNodeByName(llvm::StringRef name) const {
  return getNodeByNameOrNull(name) != nullptr;
}

ProtobufLoader::ProtobufLoader(llvm::ArrayRef<const char *> names,
                               llvm::ArrayRef<Tensor *> tensors, Function &F)
    : G_(F) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  assert(names.size() == tensors.size() && "Invalid initialization list");
  for (unsigned i = 0; i < names.size(); i++) {
    assert(!hasNodeByName(names[i]) && "Input names have duplicate");
    createAndRememberVariable(names[i], *tensors[i], VisibilityKind::Public,
                              Variable::TrainKind::None);
  }
}

ProtobufLoader::~ProtobufLoader() {
  for (auto it : tensors_) {
    delete it.second;
  }
}

}; // namespace glow
