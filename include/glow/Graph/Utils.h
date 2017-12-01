#ifndef GLOW_GRAPH_UTILS_H
#define GLOW_GRAPH_UTILS_H

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
  std::unordered_set<Node *> visited;

public:
  bool shouldVisit(Node *parent, Node *N) override {
    // Don't revisit nodes that we've already processed.
    return !visited.count(N);
  }

  explicit PostOrderVisitor() = default;

  void post(Node *parent, Node *N) override {
    visited.insert(N);
    postOrder_.push_back(N);
  }

  /// \returns the order.
  llvm::ArrayRef<Node*> getPostOrder() { return postOrder_; }
};

} // namespace glow

#endif // GLOW_GRAPH_UTILS_H
