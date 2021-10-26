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

#ifndef GLOW_GRAPH_FXIRWRAPPER_H
#define GLOW_GRAPH_FXIRWRAPPER_H

#include "glow/Graph/Graph.h"
#include "llvm/ADT/MapVector.h"

#include <folly/dynamic.h>

namespace glow {

using FXModule = folly::dynamic;
using FXNode = folly::dynamic;
using FXNodeList = folly::dynamic;
using FXWeight = folly::dynamic;

/// A thin wrapper around FX IR to provides common APIs. This is not mean to be
/// yet another IR, e.g., Glow high level graph IR.
class FXIRWrapper final : public IRContainer {
  /// Mapping from constant name to a void pointer points to where the constant
  /// is actually stored.
  const llvm::StringMap<const void *> &constants_;

  /// A reference to the owner of the function.
  Module *parent_;

  /// Mapping from getattrs to the underlying name of the constants they alias.
  llvm::StringMap<const std::string> getattrs_;

  /// A reference to the FXIR that this FXIRWrapper wraps.
  const FXModule *fx_mod_;

  /// Map nodes to their names.
  llvm::StringMap<const FXNode &> namedNodes_;

public:
  FXIRWrapper(const FXModule &FXIR, const std::string &submod,
              const llvm::StringMap<const void *> &constants,
              Module *glowModule, llvm::StringRef name = {})
      : IRContainer(name), constants_(constants), parent_(glowModule) {
    fx_mod_ = submod == "" ? &FXIR : &(FXIR["modules"][submod]);
    // Create mapping from getattrs to the underlying name of the constants they
    // alias.
    for (const auto &node : fx_mod_->at("nodes")) {
      const auto &opCode = node["op_code"].getString();
      if (opCode == "get_attr") {
        const auto &nodeName = node["name"].getString();
        bool inserted =
            getattrs_.try_emplace(nodeName, node["target"].getString()).second;
        CHECK(inserted) << "Already mapped a getattr by name " << nodeName
                        << " to its underlying Constant";
      }
      const auto &nodeName = node["name"].getString();
      namedNodes_.try_emplace(nodeName, node);
    }
  }

  ~FXIRWrapper() = default;

  /// A map to store (key) node name to  (value) placeholder/constant.
  llvm::StringMap<const Storage *> mapNodeNameToStorage_ = {};

  /// A map to store (key) Glow constant(storage) node name to FX IR weight node
  /// name. This is in spirit the reverse of the above map.
  /// Example use of this is to map constant node name from the symbol table to
  /// a FX weight node name.
  llvm::StringMap<std::string> mapGlowConstNodeToFXNodeName = {};

  IRKind getIRKind() const override { return IRKind::GlowFXIRKind; };

  static bool classof(const IRContainer *I) {
    return I->getIRKind() == IRKind::GlowFXIRKind;
  }

  static bool classof(const FXIRWrapper *F) { return true; }

  /// \returns the name of input \p node. Fatals if \p node is not
  /// specified as is_node. If \p optional then \returns an empty string if \p
  /// node is null.
  const std::string &getInputNodeName(const folly::dynamic &node,
                                      bool optional = false) const;

  /// \returns the underlying name of a Constant given provided \p name. If \p
  /// name is already the name of a Constant it is returned, else looks for
  /// getattr aliases to return the name of the actual underlying Constant.
  const char *getConstantName(llvm::StringRef name) const;

  /// \returns the underlying constant pointer given provided \p name. If \p
  /// name is already the name of a Constant, then the name is directly used,
  /// else looks for getattr aliases to find the name of the actual underlying
  /// Constant.
  const void *getConstant(llvm::StringRef name);

  /// \returns true if a constant with given \p name exists.
  bool constantExists(llvm::StringRef name) const;

  /// \returns the weights of the graph.
  const FXWeight &getWeight() const;

  /// \returns the nodes of the graph.
  const FXNodeList &getNodes() const;

  /// \returns parent module that owns this graph.
  Module *getParent() override { return parent_; }
  const Module *getParent() const override { return parent_; }

  /// \returns fx module.
  const FXModule &getFXModule() const { return *fx_mod_; }

  /// \returns the name of the node.
  const FXNode &getFXNodeByName(llvm::StringRef nodeName) const;

  const llvm::StringMap<const Storage *> &getMapNodeNameToStorage() const {
    return mapNodeNameToStorage_;
  }

  const llvm::StringMap<const void *> &getConstantsStringMap() const {
    return constants_;
  }

  /// For a given weights node, get the underlying Storage.
  const Storage *getStorageFromNodeName(llvm::StringRef name) const;

  /// Given the name of Glow Constant Node, return the FX Weight node name.
  const std::string
  getFXWeightNameFromGlowConstNodeName(llvm::StringRef name) const;

  /// When FXIR has the notion of memory/buffers. This function returns
  /// the memory buffer a node(operator) writes to.
  const FXNode &getDestinationBufferForNode(const FXNode &node) const;
};

} // namespace glow

#endif // GLOW_GRAPH_FXIRWRAPPER_H
