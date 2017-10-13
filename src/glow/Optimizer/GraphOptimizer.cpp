// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Optimizer/Optimizer.h"
#include "glow/Support/Casting.h"

#include <unordered_map>
#include <unordered_set>

using namespace glow;

/// A helper class for visiting the nodes and collecting a list of the used
/// nodes
struct UsageCollector : NodeVisitor {
  std::unordered_set<Node *> used{};

public:
  // Don't revisit visited nodes.
  bool shouldVisit(Node *parent, Node *N) override { return !used.count(N); }

  UsageCollector() = default;

  void pre(Node *parent, Node *N) override {
    // If this visit is an edge from some node, and not just the scan that
    // touches all of the nodes then mark this node as being used.
    if (parent) {
      used.insert(N);
    }
  }
};

/// Removes nodes that are not in the set \p usedSet, or are a ReturnNode.
/// \returns True if some node was removed.
static bool deleteUnusedNodes(const std::unordered_set<Node *> &usedSet,
                              std::list<Node *> &nodes) {
  bool changed = false;
  for (auto it = nodes.begin(), e = nodes.end(); it != e;) {
    bool used = usedSet.count(*it);
    if (used || isa<ReturnNode>(*it)) {
      it++;
      continue;
    }

    delete *it;
    it = nodes.erase(it);
    changed = true;
  }

  return changed;
}

/// Dead code elimination.
static void DCE(Graph &G) {
  auto &nodes = G.getNodes();
  UsageCollector usage;

  // Check which nodes are used as operands to other nodes.
  for (auto &N : nodes) {
    N->visit(nullptr, &usage);
  }

  // Remove unused nodes. Do not remove unused vars because they are the
  // interface to the user program.
  deleteUnusedNodes(usage.used, G.getNodes());
}

void glow::optimize(Graph &G, OptimizationMode mode) {
  if (mode == OptimizationMode::None) {
    return;
  }

  // Perform Dead Code Elimination.
  DCE(G);
}
