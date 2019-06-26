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

#include "glow/Graph/NodeValue.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"

using namespace glow;

NodeValue::NodeValue(Node *N) {
  assert((!N || (N->getNumResults() == 1)) &&
         "Constructing a value for a multi-res node");
  node_ = N;
  resNo_ = 0;
}

NodeValue::NodeValue(Node *N, unsigned resNo) {
  assert(resNo < N->getNumResults() && "Invalid result number");
  node_ = N;
  resNo_ = resNo;
}

void NodeValue::replaceAllUsesOfWith(NodeValue v, const Function *F) const {
  if (v.getNode()) {
    assert(getType() == v.getType() && "Replacing value with the wrong type");
  }
  typeUnsafeReplaceAllUsesOfWith(v, F);
}

void NodeValue::typeUnsafeReplaceAllUsesOfWith(NodeValue v,
                                               const Function *F) const {
  // Copy the list of users in a temporary vector since that list (and the
  // underlying iterators) are going to be invalidated by the next loop.
  auto nodeValueUsers = getUsers();
  llvm::SmallVector<NodeUse, 4> usersVec(nodeValueUsers.begin(),
                                         nodeValueUsers.end());
  for (auto &U : usersVec) {
    NodeHandle *site = U.get();
    auto *userF = U.getUser()->getParent();
    // If the user is not in function F, don't touch it.
    if (F && userF != F)
      continue;
    assert(site->getNode() == node_ && "Invalid user");
    assert(site->getResNo() == getResNo() && "Invalid list of uses");

    // Log the change of node input(operand).
    if (Function *F = getNode()->getParent()) {
      F->getLogContext()->logNodeInputChange(*(U.getUser()), *this, v);
    }
    // Constant or Placeholder has no associated Function, we need to log the
    // input changes inside its user's Function.
    else if (getNode()->getKind() == Kinded::Kind::ConstantKind ||
             getNode()->getKind() == Kinded::Kind::PlaceholderKind) {
      userF->getLogContext()->logNodeInputChange(*(U.getUser()), *this, v);
    }

    site->setOperand(v.getNode(), v.getResNo());
  }
}

unsigned NodeValue::getNumUsers() const {
  auto range = getUsers();
  return std::distance(range.begin(), range.end());
}

llvm::iterator_range<NodeValueIterator> NodeValue::getUsers() {
  auto &unfilteredUsers = getNode()->getUsers();
  return llvm::make_range(NodeValueIterator(*this, unfilteredUsers.begin()),
                          NodeValueIterator(*this, unfilteredUsers.end()));
}

llvm::iterator_range<NodeValueConstIterator> NodeValue::getUsers() const {
  const auto &unfilteredUsers = getNode()->getUsers();
  return llvm::make_range(
      NodeValueConstIterator(*this, unfilteredUsers.begin()),
      NodeValueConstIterator(*this, unfilteredUsers.end()));
}

TypeRef NodeValue::getType() const { return node_->getType(resNo_); }
void NodeValue::setType(TypeRef ty) { node_->setType(resNo_, ty); }

ElemKind NodeValue::getElementType() const {
  return getType()->getElementType();
}

llvm::ArrayRef<size_t> NodeValue::dims() const { return getType()->dims(); }

NodeHandle::NodeHandle(Node *parent, Node *N) : NodeValue(N), parent_(parent) {
  setOperand(N, 0);
}

NodeHandle::NodeHandle(Node *parent, Node *N, unsigned resNo)
    : NodeValue(N, resNo), parent_(parent) {
  setOperand(N, resNo);
}

void NodeHandle::setOperand(Node *v, unsigned resNo) {
  if (node_ == v && resNo == resNo_) {
    return;
  }

  if (node_) {
    node_->removeUse(NodeUse(this));
    node_ = nullptr;
    resNo_ = 0;
  }

  if (v) {
    node_ = v;
    resNo_ = resNo;
    v->addUse(NodeUse(this));
  }
}

void NodeUse::setOperand(NodeHandle &other) {
  if (other && site_->getNode()) {
    assert(site_->getType() == other.getType() &&
           "Setting operand to a node with a different type");
  }
  site_->setOperand(other.getNode(), other.getResNo());
}
