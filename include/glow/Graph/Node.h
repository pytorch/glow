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
#ifndef GLOW_GRAPH_NODE_H
#define GLOW_GRAPH_NODE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/Support/Casting.h"

#include "glow/Base/Traits.h"
#include "glow/Base/Type.h"
#include "glow/Graph/UseDef.h"

#include <list>

namespace glow {

class Function;
class Node;
class NodeWalker;
struct NodeUse;

/// Unlike LLVM values, graph nodes may return multiple values as the result of
/// a computation. Gradient-calculating nodes such as conv-grad return multiple
/// values. As such, each use of a node computation must indicate the node that
/// computes it as well as which return value to use from that node. This pair
/// of information is represented in this class.

/// NodeValue is a simple POD struct that contains a reference to a node and a
/// result number.
struct NodeValue {
protected:
  /// A pointer to the node (owned by the graph).
  Node *node_{nullptr};
  /// Specifies the node result number to use.
  unsigned resNo_{0};

public:
  /// Create a new value.
  NodeValue() = default;
  /// Create a new value.
  /*implicit*/ NodeValue(Node *N);

  /// Create a new value for result \p resNo.
  NodeValue(Node *N, unsigned resNo);

  /// Create a new value from an existing one.
  NodeValue(const NodeValue &that) : node_(that.node_), resNo_{that.resNo_} {}

  /// Assignment.
  NodeValue &operator=(const NodeValue &that) {
    node_ = that.node_;
    resNo_ = that.resNo_;
    return *this;
  }

  /// Destructor.
  ~NodeValue() {}

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
  llvm::ArrayRef<uint64_t> dims() const;
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

/// A handle type for a NodeValue. This type should be used only by the
/// class members of Node classes when they need to refer to other nodes!
///
/// This class also manages the node use-def chain, by registering and
/// removing the address of the value from the use-list. This data structure
/// is similar to LLVM's SDValue. Only these NodeHandle instances are
/// registered as users of the nodes they refer to. The is different from the
/// usual NodeValue instances, which are not registered as users of the nodes
/// they refer to.
///
/// Instances of NodeHandle should always stay inside the Nodes they are
/// members of and should never leave it. E.g. they cannot be returned as
/// results of function calls, etc.
struct NodeHandle : NodeValue {
private:
  friend NodeUse;
  ///  Parent object which contains this handle.
  Node *parent_{nullptr};

public:
  /// Create a new value and register the node we reference
  /*implicit*/ NodeHandle(Node *parent, Node *N);

  /// Create a new value for result \p resNo and register the node we
  /// reference.
  NodeHandle(Node *parent, Node *N, unsigned resNo);

  /// Create a new operand and register it as a new user to the node.
  NodeHandle(Node *parent, const NodeValue &that)
      : NodeValue(nullptr), parent_(parent) {
    setOperand(that.getNode(), that.getResNo());
  }

  /// Create a new NodeHandle from an existing one and register it.
  NodeHandle(Node *parent, const NodeHandle &that)
      : NodeValue(nullptr), parent_(parent) {
    setOperand(that.getNode(), that.getResNo());
  }

  NodeHandle(const NodeHandle &that) : NodeHandle(that.parent_, that) {}

  /// Create an empty handle.
  NodeHandle() : NodeValue(nullptr), parent_(nullptr) {}

  /// When deleting an operand we need to unregister the operand from the
  /// use-list of the node it used to reference.
  ~NodeHandle() { setOperand(nullptr, 0); }

  /// Unregister old value, assign new NodeValue and register it.
  NodeHandle &operator=(const NodeHandle &that) {
    setOperand(that.getNode(), that.getResNo());
    return *this;
  }

  /// Unregister old value, assign new NodeValue and register it.
  NodeHandle &operator=(const NodeValue &that) {
    setOperand(that.getNode(), that.getResNo());
    return *this;
  }
  /// Sets the operand to point to \p N. This method registers the operand as
  /// a user of \p N.
  void setOperand(Node *v, unsigned resNo);

  /// Set the parent object.
  void setParent(Node *parent) {
    assert(!parent_ && "Offset was set already");
    parent_ = parent;
  }
};

/// A wrapper class to expose a vector of NodeHandles inside an
/// object as a vector of NodeValues. This is done to avoid leaking of
/// NodeHandles from Nodes into the user-code. This type can be used as a
/// return type of e.g. getInputs() and similar functions.
class NodeValueArrayRef {
  llvm::ArrayRef<NodeHandle> ref_;

public:
  using const_iterator = llvm::ArrayRef<NodeHandle>::const_iterator;

  NodeValueArrayRef(llvm::ArrayRef<NodeHandle> ref) : ref_(ref) {}
  NodeValueArrayRef(const std::vector<NodeHandle> &ref) : ref_(ref) {}
  const NodeValue &operator[](std::size_t idx) const { return ref_[idx]; }
  operator std::vector<NodeValue>() {
    return std::vector<NodeValue>(ref_.begin(), ref_.end());
  }
  size_t size() const { return ref_.size(); }
  bool empty() const { return ref_.empty(); }
  const_iterator begin() { return ref_.begin(); }
  const_iterator end() { return ref_.end(); }
  NodeValue front() { return *begin(); }
};

/// A 'Use' is a use-list representation of a Node operand.
struct NodeUse {
  /// The operand site. This is the address of the operand that points to our
  /// node.
  NodeHandle *site_;

  explicit NodeUse(NodeHandle *site) : site_(site) {}

  bool operator==(const NodeUse &other) const { return site_ == other.site_; }

  /// \returns the instruction that the use refers to.
  NodeHandle *get() const { return site_; }
  /// Get the node containing this use.
  const Node *getUser() const { return site_->parent_; }
  Node *getUser() { return site_->parent_; }
  /// Sets the operand to a new value.
  void setOperand(NodeHandle &site);
};

/// Represents a node in the compute graph.
class Node : public Named,
             public Kinded,
             public UseDef<Node, NodeHandle, NodeUse>,
             public llvm::ilist_node<Node> {
  friend llvm::ilist_traits<Node>;

protected:
  /// This is the maximum number of results that a node may have.
  static constexpr unsigned maxNodeResno_ = 6;

  /// The output types for the results of the node.
  std::array<TypeRef, maxNodeResno_> types_;
  /// The number of results that the node has.
  unsigned numRes_{0};
  /// A nullable reference to some tensor value that may predicate the execution
  /// of the current node.
  NodeHandle predicate_;

  /// Destroys a node and deallocates the memory. This method is typically
  /// implicitly invoked when a node is being removed from the intrusive list of
  /// nodes.
  static void destroyNode(Node *N);

  /// Link to the function holding this node.
  Function *parent_;

public:
  Node(Kinded::Kind k, llvm::StringRef name)
      : Named(name), Kinded(k), predicate_(this, nullptr), parent_(nullptr) {}

  /// \returns the nullable predicate of the current node.
  const NodeValue getPredicate() const;
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

  /// \returns the function holding this node.
  /// If that node does not belong to any function, this
  /// is nullptr.
  const Function *getParent() const { return parent_; }
  /// Set the link to the function that holds this node.
  void setParent(Function *parent) { parent_ = parent; }

  /// Getters/setters to access Node's inputs and outputs.
  unsigned getNumInputs() const;
  llvm::StringRef getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  const NodeValue getNthInput(unsigned idx) const;
  void setNthInput(unsigned idx, NodeValue val);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const;
  bool isArithmetic() const;

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

  /// \returns true if the node is equal to the other node.
  bool operator==(const Node &O) const { return isEqual(O); }

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
  TypeRef getType(unsigned idx) const;

  /// Methods that forward to the result type (that must be valid):
  /// @{
  ElemKind getElementType(unsigned resNo) const;
  llvm::ArrayRef<uint64_t> dims(unsigned resNo) const;
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
/// Allow casting NodeHandle into Node*.
template <> struct simplify_type<glow::NodeHandle> {
  typedef glow::Node *SimpleType;
  static SimpleType getSimplifiedValue(glow::NodeHandle &val) {
    return val.getNode();
  }
};
/// Allow casting const NodeHandle into Node*.
template <> struct simplify_type<const glow::NodeHandle> {
  typedef glow::Node *SimpleType;
  static SimpleType getSimplifiedValue(const glow::NodeHandle &val) {
    return val.getNode();
  }
};

//===----------------------------------------------------------------------===//
// ilist_traits for glow::Node
//===----------------------------------------------------------------------===//

template <>
struct ilist_traits<glow::Node> : public ilist_node_traits<glow::Node> {
  using Node = glow::Node;

  glow::Function *getContainingFunction();

private:
  using node_iterator = simple_ilist<Node>::iterator;

public:
  static void deleteNode(Node *N) { glow::Node::destroyNode(N); }

  void addNodeToList(Node *N);
  void removeNodeFromList(Node *N);
  void transferNodesFromList(ilist_traits<Node> &L2, node_iterator first,
                             node_iterator last);

private:
  void createNode(const Node &);
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
