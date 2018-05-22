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

unsigned NCHW2NHWC[4] = {0u, 2u, 3u, 1u};
unsigned NHWC2NCHW[4] = {0u, 3u, 1u, 2u};

bool isArrayConstant(llvm::ArrayRef<size_t> a) {
  for (size_t i = 1; i < a.size(); i++)
    if (a[0] != a[i])
      return false;
  return true;
}

Tensor *ProtobufLoader::getTensorByName(const std::string &name) {
  assert(tensors_.count(name) &&
         "There is no tensor registered with this name.");
  return tensors_[name];
}

Node *ProtobufLoader::getNodeByName(const std::string &name) {
  auto it = nodeByName_.find(name);
  if (it != nodeByName_.end()) {
    return it->second;
  }

  llvm_unreachable("Could not find a node with this name.");
  return nullptr;
}

Node *ProtobufLoader::getOrCreateNodeByName(const std::string &name) {
  auto it = nodeByName_.find(name);
  if (it != nodeByName_.end()) {
    return it->second;
  }

  Tensor *T = getTensorByName(name);
  auto *V = G_.getParent()->createVariable(T->getElementType(), T->dims(), name,
                                           VisibilityKind::Private,
                                           Variable::TrainKind::Broadcast);
  V->copyFrom(T);
  nodeByName_[name] = V;
  return V;
}

bool ProtobufLoader::hasNodeByName(const std::string &name) const {
  auto it = nodeByName_.find(name);
  return (it != nodeByName_.end());
}

ProtobufLoader::ProtobufLoader(llvm::ArrayRef<const char *> names,
                               llvm::ArrayRef<Tensor *> tensors, Function &F)
    : G_(F) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  assert(names.size() == tensors.size() && "Invalid initialization list");
  for (unsigned i = 0; i < names.size(); i++) {
    auto *T = tensors[i];
    auto *V = G_.getParent()->createVariable(T->getElementType(), T->dims(),
                                             names[i], VisibilityKind::Public,
                                             Variable::TrainKind::None);
    V->copyFrom(T);
    nodeByName_[names[i]] = V;
  }
}

ProtobufLoader::~ProtobufLoader() {
  for (auto it : tensors_) {
    delete it.second;
  }
}

}; // namespace glow
