#ifndef GLOW_GRAPH_NODE_H
#define GLOW_GRAPH_NODE_H

#include "llvm/ADT/StringRef.h"

#include "glow/Base/Type.h"
#include "glow/Base/Traits.h"

namespace glow {

class NodeVisitor;

/// Represents a node in the compute graph.
class Node : public Kinded, public Typed, public Named {

public:
  Node(Kinded::Kind k, TypeRef Ty, llvm::StringRef name)
      : Kinded(k), Typed(Ty) {
    setName(name);
  }

  /// \returns a textual description of the node.
  virtual std::string getDebugDesc() const;

  /// This method implements the visitor pattern that scans the compute DAG top
  /// to bottom. The visitor \p visitor is sent by the parent node \p parent,
  /// or nullptr if this is the first node to be visited.
  virtual void visit(Node *parent, NodeVisitor *visitor){};

  virtual ~Node() = default;
};

class NodeVisitor {
public:
  /// This callback is called before visiting the children of \p N.
  virtual void pre(Node *parent, Node *N) {}

  /// This callback is called after visiting the children of \p N.
  virtual void post(Node *parent, Node *N) {}

  /// This callback is called before processing the graph. If the method returns
  /// false then we skip this node.
  virtual bool shouldVisit(Node *parent, Node *N) { return true; }
};

} // namespace glow

#endif // GLOW_GRAPH_NODE_H
