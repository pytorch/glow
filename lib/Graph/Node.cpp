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

#include "glow/Base/Type.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/Support/Support.h"

using namespace glow;

void NodeUse::setOperand(NodeValue &other) {
  if (other && site_->getNode()) {
    assert(site_->getType() == other.getType() &&
           "Setting operand to a node with a different type");
  }
  site_->setOperand(other.getNode(), other.getResNo());
}

NodeValue::NodeValue(Node *N) {
  assert((!N || (N->getNumResults() == 1)) &&
         "Constructing a value for a multi-res node");
  setOperand(N, 0);
}

NodeValue::NodeValue(Node *N, unsigned resNo) {
  assert(resNo < N->getNumResults() && "Invalid result number");
  setOperand(N, resNo);
}

void NodeValue::setOperand(Node *v, unsigned resNo) {
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

void NodeValue::replaceAllUsesOfWith(NodeValue v) {
  if (v.getNode()) {
    assert(getType() == v.getType() && "Replacing value with the wrong type");
  }
  auto &users = node_->getUsers();
  llvm::SmallVector<NodeUse, 4> usersVec(users.begin(), users.end());
  for (auto &U : usersVec) {
    NodeValue *site = U.get();
    assert(site->getNode() == node_ && "Invalid user");
    if (site->getResNo() == getResNo()) {
      site->setOperand(v.getNode(), v.getResNo());
    }
  }
}

const NodeValue &Node::getPredicate() const { return predicate_; }

void Node::setPredicate(const NodeValue &P) { predicate_ = P; }

bool Node::hasPredicate() const { return predicate_.getNode(); }

TypeRef Node::getType(unsigned idx) const {
  assert(idx < numRes_ && "Result number does not exist.");
  return types_[idx];
}

ElemKind Node::getElementType(unsigned resNo) const {
  TypeRef TR = getType(resNo);
  return TR->getElementType();
}

llvm::ArrayRef<size_t> Node::dims(unsigned resNo) const {
  TypeRef TR = getType(resNo);
  return TR->dims();
}

void Node::addResult(TypeRef T) {
  assert(numRes_ < maxNodeResno_ && "Too many results");
  types_[numRes_++] = T;
}

bool Node::isEqual(const Node &other) const {
  if (this == &other)
    return true;

  if (getKind() != other.getKind())
    return false;

  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<const CLASS *>(this)->isEqual(                          \
        *static_cast<const CLASS *>(&other));
#include "AutoGenNodes.def"

#define DEF_INSTR(CLASS, NAME) case glow::Kinded::Kind::CLASS##Kind:
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME) DEF_INSTR(CLASS, NAME)
#define DEF_VALUE(CLASS, NAME) DEF_INSTR(CLASS, NAME)
#include "AutoGenInstr.def"

    llvm_unreachable(
        "Not reachable, values and instructions are not handled here");
  }
  return false;
}

namespace {
class HashNodeVisitor : public NodeVisitor<HashNodeVisitor, llvm::hash_code> {
  using hash_code = llvm::hash_code;
  using super = NodeVisitor;

public:
#define DEF_NODE(CLASS, NAME)                                                  \
  hash_code visit##CLASS(const CLASS *N) const { return N->getHash(); }
#include "AutoGenNodes.def"

  hash_code visit(const Node *N) const {
    return const_cast<HashNodeVisitor *>(this)->super::visit(
        const_cast<Node *>(N));
  }
};

} // namespace

llvm::hash_code Node::getHash() const { return HashNodeVisitor().visit(this); }

void Node::visit(Node *parent, NodeWalker *visitor) {
  if (hasPredicate()) {
    getPredicate()->visit(this, visitor);
  }

  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<CLASS *>(this)->visit(parent, visitor);
#include "AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}

TypeRef NodeValue::getType() const { return node_->getType(resNo_); }

ElemKind NodeValue::getElementType() const {
  return getType()->getElementType();
}

void UnownedNodeValueMap::insert(NodeValue from, NodeValue to) {
  entries_.push_front(
      {{from.getNode(), from.getResNo()}, {to.getNode(), to.getResNo()}});
}

NodeValue UnownedNodeValueMap::get(NodeValue from) {
  for (auto &E : entries_) {
    auto &F = E.first;
    auto &T = E.second;

    if (F.first == from.getNode() && F.second == from.getResNo()) {
      return NodeValue(T.first, T.second);
    }
  }

  llvm_unreachable("Invalid node");
  return NodeValue(nullptr, 0);
}

bool UnownedNodeValueMap::count(NodeValue from) {
  for (auto &E : entries_) {
    auto &F = E.first;
    if (F.first == from.getNode() && F.second == from.getResNo()) {
      return true;
    }
  }

  return false;
}

llvm::ArrayRef<size_t> NodeValue::dims() const { return getType()->dims(); }

//===----------------------------------------------------------------------===//
//                     Debug description methods
//===----------------------------------------------------------------------===//

unsigned Node::getNumInputs() const {
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<const CLASS *>(this)->getNumInputs();
#include "AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}
llvm::StringRef Node::getInputName(unsigned idx) const {
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<const CLASS *>(this)->getInputName(idx);
#include "AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}
NodeValue &Node::getNthInput(unsigned idx) {
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<CLASS *>(this)->getNthInput(idx);
#include "AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}

const NodeValue &Node::getNthInput(unsigned idx) const {
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<CLASS *>(const_cast<Node *>(this))->getNthInput(idx);
#include "AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}

NodeValue Node::getNthResult(unsigned idx) {
  assert(idx < getNumResults());
  return NodeValue(this, idx);
}

const NodeValue Node::getNthResult(unsigned idx) const {
  assert(idx < getNumResults());
  return NodeValue(const_cast<Node *>(this), idx);
}

llvm::StringRef Node::getOutputName(unsigned idx) const {
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<const CLASS *>(this)->getOutputName(idx);
#include "AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}

bool Node::hasSideEffects() const {
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<const CLASS *>(this)->hasSideEffects();
#include "AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}

// NOTE: This is used in conjunction with assuming the 1st input is LHS, and 2nd
// input is RHS. If adding a new Arithmetic inst, ensure this is the case.
bool Node::isArithmetic() const {
  switch (getKind()) {
  case glow::Kinded::Kind::AddNodeKind:
  case glow::Kinded::Kind::MulNodeKind:
  case glow::Kinded::Kind::SubNodeKind:
  case glow::Kinded::Kind::DivNodeKind:
  case glow::Kinded::Kind::MaxNodeKind:
  case glow::Kinded::Kind::MinNodeKind:
  case glow::Kinded::Kind::CmpLTENodeKind:
  case glow::Kinded::Kind::CmpEQNodeKind:
    return true;
  default:
    return false;
  }
}

bool Node::isOverwrittenNthInput(unsigned idx) const {
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<const CLASS *>(this)->isOverwrittenNthInput(idx);
#include "AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}

std::string Node::getDebugDesc() const {
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<const CLASS *>(this)->getDebugDesc();
#include "AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}

Node *Node::clone() const {
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<const CLASS *>(this)->clone();
#include "AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}

void Node::destroyNode(Node *N) {
  switch (N->getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind: {                                      \
    delete static_cast<CLASS *>(N);                                            \
    break;                                                                     \
  }
#include "AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}

//===----------------------------------------------------------------------===//
//                       Nodes verification
//===----------------------------------------------------------------------===//

void Node::verify() const {
  // Verify the shared members of the node.

  // Verify the predicate field.
  if (hasPredicate()) {
    auto pred = getPredicate();
    assert(pred.getNode() && "Invalid predicate");
    auto Ty = pred.getType();
    (void)Ty;
    assert(Ty->dims().size() == 1 && "Predicate must be a vector");
  }

  // Verify node-specific properties:
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<const CLASS *>(this)->verify();
#include "AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}
