/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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
#ifndef GLOW_GRAPH_NODEVALUE_H
#define GLOW_GRAPH_NODEVALUE_H

#include "glow/Base/Traits.h"
#include "glow/Base/Type.h"
#include "llvm/ADT/StringMap.h"

namespace glow {

class Function;
class Node;
class NodeWalker;
struct NodeUse;
template <bool is_const_iter> class NodeValueIteratorImpl;
using NodeValueIterator = NodeValueIteratorImpl<false>;
using NodeValueConstIterator = NodeValueIteratorImpl<true>;

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

  /// Replace all of the uses in \p F of this value with \p v. Types of the node
  /// value and \p v should be exactly the same.
  void replaceAllUsesOfWith(NodeValue v, const Function *F = nullptr,
                            Node *skipReplacement = nullptr) const;

  /// Replace all of the uses in \p F of this value with \p v. Types of the node
  /// value and \p v can be different.
  void typeUnsafeReplaceAllUsesOfWith(NodeValue v, const Function *F = nullptr,
                                      Node *skipReplacement = nullptr) const;

  /// Return the TypeRef of the referenced return value.
  TypeRef getType() const;
  /// Set the type of the referenced value.
  void setType(TypeRef ty);
  /// Set the type of the referenced value. Does not check that dims() match.
  void setTypeUnsafe(TypeRef ty);

  /// Methods that forward to the result type (that must be valid):
  /// @{
  ElemKind getElementType() const;
  llvm::ArrayRef<dim_t> dims() const;
  /// @}

  bool operator==(const NodeValue &O) const {
    return node_ == O.node_ && resNo_ == O.resNo_;
  }

  bool operator<(const NodeValue &O) const {
    if (node_ == O.node_)
      return resNo_ < O.resNo_;
    return (node_ < O.node_);
  }

  /// Check if this NodeValue has exactly one use.
  bool hasOneUse() const { return getNumUsers() == 1; }
  /// Get the number of users of this NodeValue.
  unsigned getNumUsers() const;

  /// Get the list of users of this NodeValue.
  llvm::iterator_range<NodeValueIterator> getUsers();
  llvm::iterator_range<NodeValueConstIterator> getUsers() const;

  /// Get the full node output name based on the node name and output number.
  /// The following format is used: nodename:outputNumber
  static std::string
  generateNodeOutputName(const std::string &nodeName, unsigned outputNumber = 0,
                         bool stripResNoFor0thInput = false) {
    return nodeName + ((stripResNoFor0thInput && outputNumber == 0)
                           ? ""
                           : ":" + std::to_string(outputNumber));
  }

  /// \returns a unique name for this NodeValue, where the name of the node is
  /// appended with a colon followed by \ref resNo_.
  /// If \p stripResNoFor0thInput then the result number for the 0th input will
  /// not be appended (i.e. no ":0" will be appended).
  std::string generateNodeOutputName(bool stripResNoFor0thInput = false) const;
};

/// Struct containing the output name string and node kind for use in the
/// LoweredInfoMap for keeping track of lowered node info.
struct NodeNameAndKind : public Named, public Kinded {
public:
  NodeNameAndKind(llvm::StringRef name, size_t resNo, Kinded::Kind k)
      : Named(NodeValue::generateNodeOutputName(name, resNo)), Kinded(k) {}
};

/// Overload < operator for NodeNameAndKind to allow for usage with std::set.
inline bool operator<(const NodeNameAndKind &x, const NodeNameAndKind &y) {
  return x.getName() < y.getName();
}

/// Overload == operator for NodeNameAndKind to allow for usage with std::set.
inline bool operator==(const NodeNameAndKind &x, const NodeNameAndKind &y) {
  return x.getName() == y.getName();
}

/// Used to keep track of the origin of lowered Nodes via output names as
/// determined by NodeValue::generateNodeOutputName(). For example if some
/// NodeValue X is lowered from some NodeValue Y, then the output name of X is a
/// key which maps to a set of names which contains the output name of Y.
using LoweredInfoMap = llvm::StringMap<std::set<NodeNameAndKind>>;

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

} // namespace glow

#endif // GLOW_GRAPH_NODEVALUE_H
