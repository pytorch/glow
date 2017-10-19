#ifndef GLOW_GRAPH_NODE_H
#define GLOW_GRAPH_NODE_H

#include "llvm/ADT/StringRef.h"

#include "glow/Base/Traits.h"
#include "glow/Base/Type.h"
#include "glow/IR/UseDef.h"

namespace glow {

class NodeVisitor;
class Node;

/// Node operands are the handles that wrap the pointer to the nodes that a node
/// references. They add functionality for maintaining the use-list.
struct NodeOperand {
private:
  Node *node_{nullptr};

public:
  /// Create a new operand and register the node we reference.
  explicit NodeOperand(Node *N) { setOperand(N); }
  /// Create a new operand and register it as a new user to the node.
  NodeOperand(const NodeOperand &that) { setOperand(that.get()); }
  /// When deleting an operand we need to unregister the operand from the
  /// use-list of the node it used to reference.
  ~NodeOperand() { setOperand(nullptr); }
  /// Sets the operand to point to \p N. This method registers the operand as a
  /// user of \p N.
  void setOperand(Node *N);

  /// \returns the underlying pointer.
  Node *get() const { return node_; }
  /// \returns the underlying pointer when casting.
  operator Node *() const { return node_; }
  /// Provide a smart-pointer interface.
  Node *operator->() const { return node_; }
};

/// A 'Use' is a use-list representation of a Node operand.
struct NodeUse {
  /// The operand site. This is the address of the operand that points to our
  /// node.
  NodeOperand *site_;

  NodeUse(NodeOperand *site) : site_(site) {}

  bool operator==(const NodeUse &other) const { return site_ == other.site_; }

  /// \returns the instruction that the use refers to.
  NodeOperand *get() const { return site_; }
  /// Sets the operand to a new value.
  void setOperand(Node *other);
};

/// Represents a node in the compute graph.
class Node : public Kinded,
             public Typed,
             public Named,
             public UseDef<Node, Node, NodeUse> {

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

  /// When the node is deleted we need to unregister all users. This allows us
  /// to deconstruct the graph in an arbitrary order.
  virtual ~Node() { replaceAllUsesOfWith(nullptr); }
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
