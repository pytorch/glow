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

} // namespace glow
