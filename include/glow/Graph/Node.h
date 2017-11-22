#ifndef GLOW_GRAPH_NODE_H
#define GLOW_GRAPH_NODE_H

#include "llvm/ADT/StringRef.h"

#include "glow/Base/Traits.h"
#include "glow/Base/Type.h"
#include "glow/IR/UseDef.h"

namespace glow {

class Node;
class NodeWalker;

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
  /// Create a new value and register the node we reference.
  NodeValue(Node *N);

  /// Create a new value for result \p resNo and register the node we reference.
  NodeValue(Node *N, unsigned resNo);

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
  TypeRef getType() const;

  /// Methods that forward to the result type (that must be valid):
  /// @{
  ElemKind getElementType() const;
  llvm::ArrayRef<size_t> dims() const;
  /// @}

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
class Node : public Named, public Kinded, public UseDef<Node, Node, NodeUse> {
public:
  /// This is the maximum number of results that a node may have.
  static constexpr unsigned max_node_resno = 4;

  /// The output types for the results of the node.
  std::array<TypeRef, max_node_resno> types_;
  /// The number of results that the node has.
  unsigned numRes_{0};

  Node(Kinded::Kind k, llvm::StringRef name) : Named(name), Kinded(k) {}

  /// \returns the number of results that the node has.
  unsigned getNumRes() { return numRes_; }

  /// \returns a textual description of the node.
  virtual std::string getDebugDesc() const;

  /// This method implements the visitor pattern that scans the compute DAG top
  /// to bottom. The visitor \p visitor is sent by the parent node \p parent,
  /// or nullptr if this is the first node to be visited.
  virtual void visit(Node *parent, NodeWalker *visitor) {}

  /// When the node is deleted we need to unregister all users. This allows us
  /// to deconstruct the graph in an arbitrary order.
  virtual ~Node() { replaceAllUsesOfWith(nullptr); }

  /// \returns the n'th result type of the node.
  TypeRef getType(unsigned idx = -1) const;

  /// Methods that forward to the result type (that must be valid):
  /// @{
  ElemKind getElementType(unsigned resNo = -1) const;
  llvm::ArrayRef<size_t> dims(unsigned resNo = -1) const;
  /// @}

protected:
  /// When constructing the node, add a new result of type \p T.
  void addResult(TypeRef T);
};

/// A walker that recursively visits a node and its children.
class NodeWalker {
public:
  /// This callback is called before visiting the children of \p N.
  virtual void pre(Node *parent, Node *N) {}

  /// This callback is called after visiting the children of \p N.
  virtual void post(Node *parent, Node *N) {}

  /// This callback is called before processing the graph. If the method returns
  /// false then we skip this node.
  virtual bool shouldVisit(Node *parent, Node *N) { return true; }

  /// Dtor.
  virtual ~NodeWalker() = default;
};

} // namespace glow

namespace llvm {
/// Allow casting NodeValue into Node*.
template <> struct simplify_type<glow::NodeValue> {
  typedef glow::Node *SimpleType;
  static SimpleType getSimplifiedValue(glow::NodeValue &val) {
    return val.getNode();
  }
};
/// Allow casting NodeValue into Node*.
template <> struct simplify_type<const glow::NodeValue> {
  typedef glow::Node *SimpleType;
  static SimpleType getSimplifiedValue(const glow::NodeValue &val) {
    return val.getNode();
  }
};
} // namespace llvm

#endif // GLOW_GRAPH_NODE_H
