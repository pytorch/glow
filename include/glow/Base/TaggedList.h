/**
 * Copyright (c) 2018-present, Facebook, Inc.
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
#ifndef GLOW_BASE_TAGGEDLIST_H
#define GLOW_BASE_TAGGEDLIST_H

/// \file Intrusive doubly linked list with online node ordering.
///
/// This file implements a TaggedList<T> template which is very similar to
/// llvm::ilist<T> with the addition of online node ordering. This means that
/// iterators into a TaggedList<T> can be compared with the standard inequality
/// operators (<, <=, >, >=), and the iterator relations are consistent with the
/// list order.

#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <iterator>
#include <type_traits>

namespace glow {

template <typename T> class TaggedListNode;
template <typename T, typename Traits> class TaggedList;

/// Traits for a TaggedList<T>
///
/// Specialize these traits to customize list behavior.
///
/// The TaggedList will derive from the traits type.
template <typename T> struct TaggedListTraits {
  /// Delete a node.
  ///
  /// A TaggedList<T> never allocates nodes, but the erase() and clear() methods
  /// will delete nodes.
  static void deleteNode(T *node) { delete node; }

  /// Called before node is added to this list.
  void addNodeToList(T *) {}

  /// Called before node is removed from this list.
  void removeNodeFromList(T *) {}
};

/// Namespace of TaggedList implementation details
namespace tagged_list_details {

/// Base class for nodes that can be inserted in a `TaggedList`.
///
/// Users should derive from `TaggedListNode<T>`.
class NodeBase {
public:
  /// Is this the `end()` sentinel? All real nodes return false.
  bool isTaggedListSentinel() const { return nodeTag_ == 0; }

  /// Is this node inserted in a TaggedList?
  /// This also returns true for the end() sentinel.
  bool inTaggedList() const { return prevTaggedNode_; }

private:
  // Use long member names to avoid cluttering the namespace of sub-classes.
  NodeBase *prevTaggedNode_ = nullptr;
  NodeBase *nextTaggedNode_ = nullptr;
  std::uint32_t nodeTag_ = 0;

  friend class ListImpl;
  template <typename T, bool IsReverse, bool IsConst> friend class Iterator;
};

/// Traits for sorting out iterator types.
template <typename T, bool IsConst> struct IteratorTraits;
template <typename T> struct IteratorTraits<T, false> {
  using ValueType = T;
  using NodePtr = NodeBase *;
};
template <typename T> struct IteratorTraits<T, true> {
  using ValueType = const T;
  using NodePtr = const NodeBase *;
};

/// Iterator for a TaggedList<T>.
template <typename T, bool IsReverse, bool IsConst> class Iterator {
  using Traits = IteratorTraits<T, IsConst>;
  using NodePtr = typename Traits::NodePtr;

  template <typename T1, typename T2> friend class glow::TaggedList;
  friend class glow::TaggedListNode<T>;

  // Private constructor used by TaggedList<T> and TaggedListNode<T>.
  // Note that `n` will be down-casted to a T* unless it's the end() sentinel.
  Iterator(NodePtr n) : node_(n) {
    assert(n && n->inTaggedList() &&
           "TaggedList iterator must point to node in list");
  }

public:
  using difference_type = std::ptrdiff_t;
  using value_type = typename Traits::ValueType;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::bidirectional_iterator_tag;

  /// Create an iterator pointing at n.
  explicit Iterator(value_type *n) : node_(n) {
    assert(n && n->inTaggedList() &&
           "TaggedList iterator must point to node in list");
  }

  /// Copy constructor is templated so a const iterator can be initialized with
  /// a non-const iterator, but not the other way around.
  template <bool OrigIsConst>
  Iterator(
      const Iterator<T, IsReverse, OrigIsConst> &orig,
      typename std::enable_if<IsConst || !OrigIsConst, void *>::type = nullptr)
      : node_(orig.node_) {}

  /// Dereferencing with `*i`.
  value_type &operator*() {
    assert(!node_->isTaggedListSentinel() && "* can't dereference end()");
    return *static_cast<value_type *>(node_);
  }

  /// Dereferencing with `i->field`.
  value_type *operator->() {
    assert(!node_->isTaggedListSentinel() && "-> can't dereference end()");
    return static_cast<value_type *>(node_);
  }

  /// Comparing iterators for equality.
  friend bool operator==(const Iterator &LHS, const Iterator &RHS) {
    return LHS.node_ == RHS.node_;
  }
  friend bool operator!=(const Iterator &LHS, const Iterator &RHS) {
    return LHS.node_ != RHS.node_;
  }

  /// Comparing iterators for inequality.
  ///
  /// This is the defining feature of TaggedList -- It is possible to compare
  /// the relative positions of list nodes in constant time.
  ///
  /// Returns a node ordering key that is always monotonic.
  ///
  /// The key returned for an end() iterator is greater than any other key.
  uint32_t getNodeOrdering() const {
    // The end() sentinel has tag 0 which returns UINT32_MAX in either case.
    return IsReverse ? ~node_->nodeTag_ : node_->nodeTag_ - 1;
  }

  /// Pre-increment.
  Iterator &operator++() {
    assert(!node_->isTaggedListSentinel() && "can't ++end()");
    node_ = IsReverse ? node_->prevTaggedNode_ : node_->nextTaggedNode_;
    return *this;
  }

  /// Post-increment.
  Iterator operator++(int) {
    Iterator tmp = *this;
    ++*this;
    return tmp;
  }

  // Pre-decrement.
  Iterator &operator--() {
    node_ = IsReverse ? node_->nextTaggedNode_ : node_->prevTaggedNode_;
    assert(!node_->isTaggedListSentinel() && "can't --begin()");
    return *this;
  }

  /// Post-decrement.
  Iterator operator--(int) {
    Iterator tmp = *this;
    --*this;
    return tmp;
  }

private:
  // This is either pointing to a list element of type T or the sentinel
  // NodeBase object.
  NodePtr node_;
};

template <typename T, bool IsReverse, bool IsLHSConst, bool IsRHSConst>
bool operator<(const Iterator<T, IsReverse, IsLHSConst> &LHS,
               const Iterator<T, IsReverse, IsRHSConst> &RHS) {
  return LHS.getNodeOrdering() < RHS.getNodeOrdering();
}

template <typename T, bool IsReverse, bool IsLHSConst, bool IsRHSConst>
bool operator<=(const Iterator<T, IsReverse, IsLHSConst> &LHS,
                const Iterator<T, IsReverse, IsRHSConst> &RHS) {
  return LHS.getNodeOrdering() <= RHS.getNodeOrdering();
}

template <typename T, bool IsReverse, bool IsLHSConst, bool IsRHSConst>
bool operator>(const Iterator<T, IsReverse, IsLHSConst> &LHS,
               const Iterator<T, IsReverse, IsRHSConst> &RHS) {
  return LHS.getNodeOrdering() > RHS.getNodeOrdering();
}

template <typename T, bool IsReverse, bool IsLHSConst, bool IsRHSConst>
bool operator>=(const Iterator<T, IsReverse, IsLHSConst> &LHS,
                const Iterator<T, IsReverse, IsRHSConst> &RHS) {
  return LHS.getNodeOrdering() >= RHS.getNodeOrdering();
}

/// Template-free implementation of the `TaggedList`.
class ListImpl {
public:
  ListImpl() { clear(); }

  ListImpl(const ListImpl &) = delete;
  ListImpl(const ListImpl &&) = delete;

  /// Get the number of elements in the list.
  /// This is a constant time operation.
  std::size_t size() const { return size_; }

  NodeBase *begin() { return sentinel_.nextTaggedNode_; }
  NodeBase *end() { return &sentinel_; }
  NodeBase *rbegin() { return sentinel_.prevTaggedNode_; }
  NodeBase *rend() { return &sentinel_; }
  const NodeBase *begin() const { return sentinel_.nextTaggedNode_; }
  const NodeBase *end() const { return &sentinel_; }
  const NodeBase *rbegin() const { return sentinel_.prevTaggedNode_; }
  const NodeBase *rend() const { return &sentinel_; }

  /// Insert `node` before `next`.
  void insert(NodeBase *next, NodeBase *node) {
    size_++;
    NodeBase *prev = next->prevTaggedNode_;
    node->nextTaggedNode_ = next;
    node->prevTaggedNode_ = prev;
    next->prevTaggedNode_ = node;
    prev->nextTaggedNode_ = node;

    // Compute a tag half-way between prev and next.
    // Note that if `next` is the sentinel with tag 0, this will compute a tag
    // half-way between prev and UINT32_MAX. That is precisely what we want.
    std::uint32_t delta = (next->nodeTag_ - prev->nodeTag_) / 2;
    node->nodeTag_ = prev->nodeTag_ + delta;
    if (delta == 0)
      renumber(prev, node);
  }

  void remove(NodeBase *node) {
    size_--;
    NodeBase *prev = node->prevTaggedNode_;
    NodeBase *next = node->nextTaggedNode_;
    node->nextTaggedNode_ = nullptr;
    node->prevTaggedNode_ = nullptr;
    prev->nextTaggedNode_ = next;
    next->prevTaggedNode_ = prev;
  }

  void clear() {
    sentinel_.prevTaggedNode_ = sentinel_.nextTaggedNode_ = &sentinel_;
    size_ = 0;
  }

private:
  void renumber(NodeBase *lo, NodeBase *hi);
  NodeBase sentinel_;
  // Not a size_t because the list can't hold more than 2^32 nodes.
  std::uint32_t size_ = 0;
};

} // namespace tagged_list_details

/// Base class for intrusive list nodes of type T.
///
/// A linked list node of type T must be derived from TaggedListNode<T>.
template <typename T>
class TaggedListNode : public tagged_list_details::NodeBase {
public:
  // Get an iterator pointing at this node.
  tagged_list_details::Iterator<T, false, false> getIterator() {
    return tagged_list_details::Iterator<T, false, false>(this);
  }

  // Get a const iterator pointing at this node.
  tagged_list_details::Iterator<T, false, true> getIterator() const {
    return tagged_list_details::Iterator<T, false, true>(this);
  }

  // Get a reverse iterator pointing at this node.
  tagged_list_details::Iterator<T, true, false> getReverseIterator() {
    return tagged_list_details::Iterator<T, true, false>(this);
  }

  // Get a const reverse iterator pointing at this node.
  tagged_list_details::Iterator<T, true, true> getReverseIterator() const {
    return tagged_list_details::Iterator<T, true, true>(this);
  }
};

template <typename T, typename Traits = TaggedListTraits<T>>
class TaggedList : public Traits {
public:
  using value_type = T;
  using iterator = tagged_list_details::Iterator<T, false, false>;
  using const_iterator = tagged_list_details::Iterator<T, false, true>;
  using reverse_iterator = tagged_list_details::Iterator<T, true, false>;
  using const_reverse_iterator = tagged_list_details::Iterator<T, true, true>;

  ~TaggedList() { clear(); }

  /// Get the number of elements in the list.
  /// This is a constant time operation.
  size_t size() const { return impl_.size(); }

  /// Is the list empty?
  bool empty() const { return size() == 0; }

  iterator begin() { return impl_.begin(); }
  iterator end() { return impl_.end(); }
  reverse_iterator rbegin() { return impl_.rbegin(); }
  reverse_iterator rend() { return impl_.rend(); }
  const_iterator begin() const { return impl_.begin(); }
  const_iterator end() const { return impl_.end(); }
  const_reverse_iterator rbegin() const { return impl_.rbegin(); }
  const_reverse_iterator rend() const { return impl_.rend(); }

  /// Insert `node` before `next`.
  ///
  /// The node is not copied, and it can't already be on another list.
  ///
  /// Returns an iterator pointing to the inserted node.
  iterator insert(iterator next, T *node) {
    assert(node && !node->inTaggedList());
    this->addNodeToList(node);
    impl_.insert(next.node_, node);
    return iterator(node);
  }

  /// Remove `node` from this list without deleting it.
  ///
  /// Returns a pointer to the removed node.
  T *remove(T *node) {
    assert(node && node->inTaggedList());
    this->removeNodeFromList(node);
    impl_.remove(node);
    return node;
  }

  template <bool IsReverse>
  T *remove(tagged_list_details::Iterator<T, IsReverse, false> node) {
    return remove(&*node);
  }

  /// Remove `node` from the list and delete it.
  void erase(T *node) { this->deleteNode(remove(node)); }

  template <bool IsReverse>
  void erase(tagged_list_details::Iterator<T, IsReverse, false> node) {
    erase(&*node);
  }

  value_type &front() { return *begin(); }
  value_type &back() { return *rbegin(); }
  const value_type &front() const { return *begin(); }
  const value_type &back() const { return *rbegin(); }

  void push_front(T *node) { insert(begin(), node); }
  void push_back(T *node) { insert(end(), node); }

  void pop_front() { erase(begin()); }
  void pop_back() { erase(rbegin()); }

  /// Remove all nodes from the list without calling removeNodeFromList() or
  /// deleteNode().
  void clearAndLeakNodesUnsafely() { impl_.clear(); }

  /// Erase all nodes.
  void clear() {
    while (!empty()) {
      erase(begin());
    }
  }

private:
  tagged_list_details::ListImpl impl_;
};

} // namespace glow
#endif // GLOW_BASE_TAGGEDLIST_H
