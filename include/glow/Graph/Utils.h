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
#ifndef GLOW_GRAPH_UTILS_H
#define GLOW_GRAPH_UTILS_H

#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"

#include "llvm/ADT/ArrayRef.h"

#include <unordered_set>
#include <vector>

namespace glow {

/// A helper class for ordering the nodes in a post-order order.
struct PostOrderVisitor : NodeWalker {
  /// A post-order list of nodes.
  std::vector<Node *> postOrder_;
  /// A set of visited nodes.
  std::unordered_set<const Node *> visited_;

public:
  bool shouldVisit(Node *parent, Node *N) override {
    // Don't revisit nodes that we've already processed.
    return !visited_.count(N);
  }

  bool shouldVisit(const Node *parent, const Node *N) override {
    // Don't revisit nodes that we've already processed.
    return !visited_.count(N);
  }

  explicit PostOrderVisitor() = default;

  void post(Node *parent, Node *N) override {
    visited_.insert(N);
    postOrder_.push_back(N);
  }

  /// \returns the order.
  llvm::ArrayRef<Node *> getPostOrder() { return postOrder_; }
};

/// A helper class for ordering Graph nodes in a post-order order.
class GraphPostOrderVisitor : public PostOrderVisitor {
  Function &G;
  void visit() {
    for (const auto *V : G.getParent()->getVars()) {
      V->visit(nullptr, this);
    }
    // Start visiting all root nodes, i.e. nodes that do not have any users.
    for (auto &N : G.getNodes()) {
      if (N.getNumUsers() == 0)
        N.visit(nullptr, this);
    }
  }

public:
  explicit GraphPostOrderVisitor(Function &G) : G(G) {}
  /// \returns the order.
  llvm::ArrayRef<Node *> getPostOrder() {
    if (postOrder_.empty())
      visit();
    return postOrder_;
  }
};

/// A helper class for ordering the nodes in a pre-order order.
struct PreOrderVisitor : NodeWalker {
  /// A pre-order list of nodes.
  std::vector<Node *> preOrder_;
  /// A set of visited nodes.
  std::unordered_set<const Node *> visited_;

public:
  bool shouldVisit(Node *parent, Node *N) override {
    // Don't revisit nodes that we've already processed.
    return !visited_.count(N);
  }

  bool shouldVisit(const Node *parent, const Node *N) override {
    // Don't revisit nodes that we've already processed.
    return !visited_.count(N);
  }

  explicit PreOrderVisitor() = default;

  void pre(Node *parent, Node *N) override {
    visited_.insert(N);
    preOrder_.push_back(N);
  }

  /// \returns the order.
  llvm::ArrayRef<Node *> getPreOrder() { return preOrder_; }
};

/// A helper class for ordering Graph nodes in a pre-order order.
class GraphPreOrderVisitor : public PreOrderVisitor {
  Function &G;
  void visit() {
    for (const auto *V : G.getParent()->getVars()) {
      V->visit(nullptr, this);
    }
    // Start visiting all root nodes, i.e. nodes that do not have any users.
    for (auto &N : G.getNodes()) {
      if (N.getNumUsers() == 0)
        N.visit(nullptr, this);
    }
  }

public:
  explicit GraphPreOrderVisitor(Function &G) : G(G) {}
  /// \returns the order.
  llvm::ArrayRef<Node *> getPreOrder() {
    if (preOrder_.empty())
      visit();
    return preOrder_;
  }
};

} // namespace glow

#endif // GLOW_GRAPH_UTILS_H
