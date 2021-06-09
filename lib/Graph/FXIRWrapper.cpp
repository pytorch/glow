// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#include "glow/Graph/FXIRWrapper.h"

namespace glow {
const FXWeight &FXIRWrapper::getWeight() { return fx_mod_->at("weights"); }

const FXNodeList &FXIRWrapper::getNodes() { return fx_mod_->at("nodes"); }

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

} // namespace glow
