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
  const Function &G;
  void visit() {
    for (const auto *V : G.getParent().getVars()) {
      V->visit(nullptr, this);
    }
    // Start visiting all root nodes, i.e. nodes that do not have any users.
    for (auto *N : G.getNodes()) {
      if (N->getNumUsers() == 0)
        N->visit(nullptr, this);
    }
  }

public:
  explicit GraphPostOrderVisitor(const Function &G) : G(G) {}
  /// \returns the order.
  llvm::ArrayRef<Node *> getPostOrder() {
    if (postOrder_.empty())
      visit();
    return postOrder_;
  }
};

} // namespace glow

#endif // GLOW_GRAPH_UTILS_H
