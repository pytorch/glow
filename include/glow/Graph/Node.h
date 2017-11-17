#ifndef GLOW_GRAPH_NODE_H
#define GLOW_GRAPH_NODE_H

#include "llvm/ADT/StringRef.h"

#include "glow/Base/Traits.h"
#include "glow/Base/Type.h"
#include "glow/IR/UseDef.h"

namespace glow {

class NodeVisitor;
class Node;

/// Unlike LLVM values, graph nodes may return multiple values as the result of
/// a computation. Gradient-calculating nodes, such as conv-grad return
/// multiple values. As such, each use of a node computation must indicate the
/// node that computes it as well as which return value to use from that node.
/// This pair of information is represented in this class. This class also
/// manages the node use-def chain, by registering and removing the address of
/// the value from the use-list. This data structure is similar to LLVM's
/// SDValue.
struct NodeValue {
private:
  /// A pointer the the node (owned by the graph).
  Node *node_{nullptr};
  /// Specifies the node result number to use.
  unsigned resNo_{0};

public:
  /// Create a new operand and register the node we reference.
  explicit NodeValue(Node *N) { setOperand(N); }
  /// Create a new operand and register it as a new user to the node.
  NodeValue(const NodeValue &that) { setOperand(that.getNode()); }
  /// When deleting an operand we need to unregister the operand from the
  /// use-list of the node it used to reference.
  ~NodeValue() { setOperand(nullptr); }
  /// Sets the operand to point to \p N. This method registers the operand as a
  /// user of \p N.
  void setOperand(Node *v);
  /// Get the index which selects a specific result in the SDNode
  unsigned getResNo() const { return resNo_; }
  /// \returns the underlying pointer.
  Node *getNode() const { return node_; }
  /// \returns the underlying pointer when casting.
  operator Node *() const { return node_; }
  /// Provide a smart-pointer interface.
  Node *operator->() const { return node_; }
  /// Return the TypeRef of the referenced return value.
  TypeRef getValueType() const;

  bool operator==(const NodeValue &O) const {
    return node_ == O.node_ && resNo_ == O.resNo_;
  }
};

/// A 'Use' is a use-list representation of a Node operand.
struct NodeUse {
  /// The operand site. This is the address of the operand that points to our
  /// node.
  NodeValue *site_;

  explicit NodeUse(NodeValue *site) : site_(site) {}

  bool operator==(const NodeUse &other) const { return site_ == other.site_; }

  /// \returns the instruction that the use refers to.
  NodeValue *get() const { return site_; }
  /// Sets the operand to a new value.
  void setOperand(Node *other);
};

/// Represents a node in the compute graph.
class Node : public Named,
             public Kinded,
             public Typed,
             public UseDef<Node, Node, NodeUse> {

public:
  Node(Kinded::Kind k, TypeRef Ty, llvm::StringRef name)
      : Named(name), Kinded(k), Typed(Ty) {}

  /// \returns a textual description of the node.
  virtual std::string getDebugDesc() const;

  /// This method implements the visitor pattern that scans the compute DAG top
  /// to bottom. The visitor \p visitor is sent by the parent node \p parent,
  /// or nullptr if this is the first node to be visited.
  virtual void visit(Node *parent, NodeVisitor *visitor) {}

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

  /// Dtor.
  virtual ~NodeVisitor() = default;
};

} // namespace glow

#endif // GLOW_GRAPH_NODE_H
