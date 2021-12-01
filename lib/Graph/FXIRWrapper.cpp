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

#include "glow/Graph/FXIRWrapper.h"
#include "glow/Graph/FXIRUtils.h"

namespace glow {
const FXWeight &FXIRWrapper::getWeight() const {
  return fx_mod_->at("weights");
}

const FXNodeList &FXIRWrapper::getNodes() const { return fx_mod_->at("nodes"); }

const char *FXIRWrapper::getConstantName(llvm::StringRef name) const {
  if (name.empty()) {
    return nullptr;
  }
  // If this is a Constant already, return the name back untouched.
  if (constants_.count(name)) {
    return name.data();
  }
  // Else return the name the getattr maps to if it exists.
  auto it = getattrs_.find(name);
  return it != getattrs_.end() ? it->second.c_str() : nullptr;
}

const std::string &FXIRWrapper::getInputNodeName(const FXNode &node,
                                                 bool optional) const {
  if (node.isNull()) {
    CHECK(optional) << "Non-optional node must be non-null";
    static const std::string empty;
    return empty;
  }

  CHECK(node.isObject()) << ": Expected Node object, but found "
                         << node.typeName() << ": " << node;
  CHECK(node.find("is_node") != node.items().end() && node["is_node"].asBool())
      << "Expected is_node";

  const auto &name = node["name"].getString();

  // Check if this name was for a getattr. If so, return the underlying Constant
  // name. Otherwise return the name unchanged.
  auto it = getattrs_.find(name);
  return it != getattrs_.end() ? it->second : name;
}

const void *FXIRWrapper::getConstant(llvm::StringRef name) {
  const char *baseName = getConstantName(name);
  if (!baseName) {
    return nullptr;
  }
  // There must be a constant with name baseName, so return it.
  auto it = constants_.find(baseName);
  CHECK(it != constants_.end())
      << "Should have found constant with name " << baseName;
  return it->second;
}

bool FXIRWrapper::constantExists(llvm::StringRef name) const {
  return constants_.count(name) != 0;
}

const FXNode &FXIRWrapper::getFXNodeByName(llvm::StringRef nodeName) const {
  auto it = namedNodes_.find(nodeName);
  CHECK(it != namedNodes_.end()) << " Node with name doesn't exist";
  return it->second;
}

const FXNode &
FXIRWrapper::getDestinationBufferForNode(const FXNode &node) const {

  CHECK(node.find("users") != node.items().end())
      << "users field doesn't exist in node " << node;
  CHECK_EQ(node["users"].size(), 1);
  auto users = node["users"].at(0);
  auto user_name = users["name"].getString();
  auto &dest_node = getFXNodeByName(user_name);
  auto &dest = (getNodeOpCode(dest_node) == "output") ? node : dest_node;
  return dest;
}

const Storage *FXIRWrapper::getStorageFromNodeName(llvm::StringRef name) const {
  auto it = mapNodeNameToStorage_.find(name);
  CHECK(it != mapNodeNameToStorage_.end())
      << " Storage with that name doesn't exist";
  return it->second;
}

const std::string
FXIRWrapper::getFXWeightNameFromGlowConstNodeName(llvm::StringRef name) const {
  auto it = mapGlowConstNodeToFXNodeName.find(name);
  CHECK(it != mapGlowConstNodeToFXNodeName.end())
      << " Glow const Node with that name doesn't exist";
  return it->second;
}

} // namespace glow
