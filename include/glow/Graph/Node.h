#ifndef GLOW_GRAPH_NODE_H
#define GLOW_GRAPH_NODE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include "glow/Base/Traits.h"
#include "glow/Base/Type.h"
#include "glow/IR/UseDef.h"

#include <list>

namespace glow {

class Node;
class NodeWalker;

/// Unlike LLVM values, graph nodes may return multiple values as the result of
/// a computation. Gradient-calculating nodes such as conv-grad return multiple
/// values. As such, each use of a node computation must indicate the node that
/// computes it as well as which return value to use from that node. This pair
/// of information is represented in this class. This class also manages the
/// node use-def chain, by registering and removing the address of the value
/// from the use-list. This data structure is similar to LLVM's SDValue.
struct NodeValue {
private:
  /// A pointer to the node (owned by the graph).
  Node *node_{nullptr};
  /// Specifies the node result number to use.
  unsigned resNo_{0};

public:
  /// Create a new value and register the node we reference.
  /*implicit*/ NodeValue(Node *N);

  /// Create a new value for result \p resNo and register the node we reference.
  NodeValue(Node *N, unsigned resNo);

  /// Create a new operand and register it as a new user to the node.
  NodeValue(const NodeValue &that) { setOperand(that.getNode(), that.resNo_); }

  /// Unregister old value, assign new NodeValue and register it.
  NodeValue &operator=(const NodeValue &that) {
    setOperand(that.getNode(), that.resNo_);
    return *this;
  }

  /// When deleting an operand we need to unregister the operand from the
  /// use-list of the node it used to reference.
  ~NodeValue() { setOperand(nullptr, 0); }
  /// Sets the operand to point to \p N. This method registers the operand as a
  /// user of \p N.
  void setOperand(Node *v, unsigned resNo);
  /// Get the index which selects a specific result in the SDNode
  unsigned getResNo() const { return resNo_; }
  /// \returns the underlying pointer.
  Node *getNode() const { return node_; }
  /// \returns the underlying pointer when casting.
  operator Node *() const { return node_; }

  /// Replace all of the uses of this value with \p v.
  void replaceAllUsesOfWith(NodeValue v);

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

  bool operator<(const NodeValue &O) const {
    if (node_ == O.node_)
      return resNo_ < O.resNo_;
    return (node_ < O.node_);
  }
};

/// A simple linear map that stores NodeValue without maintaining the reverse
/// reference that allows the RAUW operation.
class UnownedNodeValueMap {
public:
  /// A reference to some Node's result.
  using ValRef = std::pair<Node *, unsigned>;
  using Entry = std::pair<ValRef, ValRef>;

private:
  std::list<Entry> entries_;

public:
  /// \register the node \p from as mapping to \p to.
  void insert(NodeValue from, NodeValue to);

  /// \returns True if N is in the map.
  bool count(NodeValue from);

  /// \returns the node that \p from is mapped to.
  NodeValue get(NodeValue from);
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
  void setOperand(NodeValue &site);
};

/// Represents a node in the compute graph.
class Node : public Named,
             public Kinded,
             public UseDef<Node, NodeValue, NodeUse> {
protected:
  /// This is the maximum number of results that a node may have.
  static constexpr unsigned maxNodeResno_ = 6;

  /// The output types for the results of the node.
  std::array<TypeRef, maxNodeResno_> types_;
  /// The number of results that the node has.
  unsigned numRes_{0};
  /// A nullable reference to some tensor value that may predicate the execution
  /// of the current node.
  NodeValue predicate_{nullptr};

public:
  Node(Kinded::Kind k, llvm::StringRef name) : Named(name), Kinded(k) {}

  /// \returns the nullable predicate of the current node.
  const NodeValue &getPredicate() const;
  /// Assigns a nullable predicate to the current node.
  void setPredicate(const NodeValue &P);
  /// Checks if a predicate is assigned to the current node.
  bool hasPredicate() const;

  /// \returns the number of results that the node has.
  unsigned getNumResults() const { return numRes_; }
  /// \returns the \p idx result of the node.
  NodeValue getNthResult(unsigned idx);
  /// \returns the n'th result of the node.
  const NodeValue getNthResult(unsigned idx) const;

  /// Getters to access Node's inputs and outputs.
  unsigned getNumInputs() const;
  llvm::StringRef getInputName(unsigned idx) const;
  NodeValue &getNthInput(unsigned idx);
  const NodeValue &getNthInput(unsigned idx) const;
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const;

  /// \returns true if this input is being overwritten by the node.
  bool isOverwrittenNthInput(unsigned idx) const;

  /// \returns a textual description of the node.
  std::string getDebugDesc() const;

  /// \returns copy of the current node. Notice that the new node is not
  /// inserted into any DAG. The caller of this method should add it to some
  /// node-list.
  Node *clone() const;

  /// \returns true if the node is equal to the other node.
  bool isEqual(const Node &other) const;

  /// \returns a hash code of the node.
  llvm::hash_code getHash() const;

  /// This method implements the visitor pattern that scans the compute DAG top
  /// to bottom. The visitor \p visitor is sent by the parent node \p parent,
  /// or nullptr if this is the first node to be visited.
  void visit(Node *parent, NodeWalker *visitor);

  void visit(const Node *parent, NodeWalker *visitor) const;

  /// Verify node.
  void verify() const;

  /// Replace all uses of this node with null. This method is used by the
  /// destruction sequence. When the node is deleted we need to unregister all
  /// users. This allows us to deconstruct the graph in an arbitrary order.
  void releaseUsers() {
    NodeValue nop(nullptr);
    for (unsigned i = 0; i < getNumResults(); i++) {
      NodeValue(this, i).replaceAllUsesOfWith(nop);
    }
  }

  ~Node() { releaseUsers(); }

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
  virtual void pre(const Node *parent, const Node *N) {}

  /// This callback is called after visiting the children of \p N.
  virtual void post(Node *parent, Node *N) {}
  virtual void post(const Node *parent, const Node *N) {}

  /// This callback is called before processing the graph. If the method returns
  /// false then we skip this node.
  virtual bool shouldVisit(Node *parent, Node *N) { return true; }
  virtual bool shouldVisit(const Node *parent, const Node *N) { return true; }

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

// custom specialization of std::hash for NodeValue.
namespace std {
template <> struct hash<glow::NodeValue> {
  typedef glow::NodeValue argument_type;
  typedef std::size_t result_type;
  result_type operator()(argument_type const &s) const noexcept {
    auto name = s.getNode()->getName();
    result_type const h1(std::hash<std::string>{}(name.str()));
    result_type const h2(std::hash<unsigned>{}(s.getResNo()));
    return h1 ^ (h2 << 8);
  }
};
} // namespace std

#endif // GLOW_GRAPH_NODE_H
