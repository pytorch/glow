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

#include "glow/Base/Type.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/VerifierHelper.h"
#include "glow/Support/Support.h"

using namespace glow;

void Node::setPredicate(const NodeValue &P) { predicate_ = P; }

bool Node::hasPredicate() const { return predicate_.getNode(); }

TypeRef Node::getType(unsigned idx) const {
  assert(idx < getNumResults() && "Result number does not exist.");
  return types_[idx];
}

void Node::setType(unsigned idx, TypeRef ty) {
  assert(types_[idx]->dims() == ty->dims() &&
         "Better create a new node at this point");
  setTypeUnsafe(idx, ty);
}

void Node::setTypeUnsafe(unsigned idx, TypeRef ty) {
  assert(idx < getNumResults() && "Result number does not exist.");
  types_[idx] = ty;
}

ElemKind Node::getElementType(unsigned resNo) const {
  TypeRef TR = getType(resNo);
  return TR->getElementType();
}

llvm::ArrayRef<dim_t> Node::dims(unsigned resNo) const {
  TypeRef TR = getType(resNo);
  return TR->dims();
}

void Node::addResult(TypeRef T) { types_.push_back(T); }

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
#include "glow/AutoGenNodes.def"

#define DEF_INSTR(CLASS, NAME) case glow::Kinded::Kind::CLASS##Kind:
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME) DEF_INSTR(CLASS, NAME)
#define DEF_VALUE(CLASS, NAME) DEF_INSTR(CLASS, NAME)
#include "glow/AutoGenInstr.def"

    llvm_unreachable(
        "Not reachable, values and instructions are not handled here");
  }
  return false;
}

const NodeValue Node::getPredicate() const { return predicate_; }

namespace {
class HashNodeVisitor : public NodeVisitor<HashNodeVisitor, llvm::hash_code> {
  using hash_code = llvm::hash_code;
  using super = NodeVisitor;

public:
#define DEF_NODE(CLASS, NAME)                                                  \
  hash_code visit##CLASS(const CLASS *N) const { return N->getHash(); }
#include "glow/AutoGenNodes.def"

  hash_code visit(const Node *N) const {
    return const_cast<HashNodeVisitor *>(this)->super::visit(
        const_cast<Node *>(N));
  }
};

} // namespace

llvm::hash_code Node::getHash() const { return HashNodeVisitor().visit(this); }

void Node::visit(Node *parent, NodeWalker *visitor) {
  if (hasPredicate()) {
    getPredicate().getNode()->visit(this, visitor);
  }

  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<CLASS *>(this)->visit(parent, visitor);
#include "glow/AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}

//===----------------------------------------------------------------------===//
//                     Debug description methods
//===----------------------------------------------------------------------===//

unsigned Node::getNumInputs() const {
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<const CLASS *>(this)->getNumInputs();
#include "glow/AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}

std::string Node::getInputName(unsigned idx) const {
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<const CLASS *>(this)->getInputName(idx);
#include "glow/AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}

NodeValue Node::getNthInput(unsigned idx) {
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<CLASS *>(this)->getNthInput(idx);
#include "glow/AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}

const NodeValue Node::getNthInput(unsigned idx) const {
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<CLASS *>(const_cast<Node *>(this))->getNthInput(idx);
#include "glow/AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}

void Node::setNthInput(unsigned idx, NodeValue val) {
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    if (getParent()) {                                                         \
      getParent()->getLogContext()->logNodeInputChange(                        \
          *this, this->getNthInput(idx), val);                                 \
    }                                                                          \
    return static_cast<CLASS *>(this)->setNthInput(idx, val);
#include "glow/AutoGenNodes.def"
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

std::string Node::getOutputName(unsigned idx) const {
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<const CLASS *>(this)->getOutputName(idx);
#include "glow/AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}

bool Node::hasSideEffects() const {
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<const CLASS *>(this)->hasSideEffects();
#include "glow/AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}

bool Node::isCanonical() const {
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<const CLASS *>(this)->isCanonical();
#include "glow/AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}

bool Node::isDataParallel() const {
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<const CLASS *>(this)->isDataParallel();
#include "glow/AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}

// NOTE: This is used in conjunction with assuming the 1st input is LHS, and 2nd
// input is RHS, and 1st result is Result.
bool Node::isArithmetic() const {
  // Each case includes a static assert that the generated nodes that we
  // consider arithmetic have the expected format/order of LHS, RHS, Result.
#define ARITHMETIC_NODE_CASE(NODE_NAME_)                                       \
  static_assert((NODE_NAME_##Node::LHSIdx == ArithmeticNode::LHSIdx &&         \
                 NODE_NAME_##Node::RHSIdx == ArithmeticNode::RHSIdx &&         \
                 NODE_NAME_##Node::ResultIdx == ArithmeticNode::ResultIdx),    \
                #NODE_NAME_                                                    \
                "Node does not match expected arithmetic node format.");       \
  case glow::Kinded::Kind::NODE_NAME_##NodeKind:

  switch (getKind()) {
    ARITHMETIC_NODE_CASE(Add)
    [[fallthrough]];
    ARITHMETIC_NODE_CASE(Mul)
    [[fallthrough]];
    ARITHMETIC_NODE_CASE(Sub)
    [[fallthrough]];
    ARITHMETIC_NODE_CASE(Div)
    [[fallthrough]];
    ARITHMETIC_NODE_CASE(FloorDiv)
    [[fallthrough]];
    ARITHMETIC_NODE_CASE(Max)
    [[fallthrough]];
    ARITHMETIC_NODE_CASE(Min)
    [[fallthrough]];
    ARITHMETIC_NODE_CASE(CmpLTE)
    [[fallthrough]];
    ARITHMETIC_NODE_CASE(CmpLT)
    [[fallthrough]];
    ARITHMETIC_NODE_CASE(CmpEQ)
    [[fallthrough]];
    ARITHMETIC_NODE_CASE(Pow)
    [[fallthrough]];
    ARITHMETIC_NODE_CASE(Fmod)
    return true;
  default:
    return false;
  }
#undef ARITHMETIC_NODE_CASE
}

bool Node::isOverwrittenNthInput(unsigned idx) const {
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<const CLASS *>(this)->isOverwrittenNthInput(idx);
#include "glow/AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}

std::string Node::getDebugDesc() const {
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<const CLASS *>(this)->getDebugDesc();
#include "glow/AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}

void Node::dump(llvm::raw_ostream &out) const { out << this->getDebugDesc(); }

void Node::dump() const { dump(llvm::outs()); }

std::string Node::toString() const { return this->getDebugDesc(); }

size_t Node::getTotMemSize() const {
  size_t totMemSize = 0;
  for (unsigned idx = 0, e = getNumInputs(); idx < e; idx++) {
    totMemSize += getNthInput(idx).getType()->getSizeInBytes();
  }
  for (unsigned idx = 0, e = getNumResults(); idx < e; idx++) {
    totMemSize += getNthResult(idx).getType()->getSizeInBytes();
  }
  return totMemSize;
}

Node *Node::clone() const {
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<const CLASS *>(this)->clone();
#include "glow/AutoGenNodes.def"
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
#include "glow/AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}

namespace glow {

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Node &node) {
  node.dump(os);
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Node *node) {
  assert(node != nullptr && "Null Pointer.");
  node->dump(os);
  return os;
}
} // namespace glow

//===----------------------------------------------------------------------===//
//                       Nodes verification
//===----------------------------------------------------------------------===//

bool Node::verify() const {
  // Verify the shared members of the node.
  bool isValid = true;

  // Verify the predicate field.
  if (hasPredicate()) {
    auto pred = getPredicate();
    if (!expectCompareTrue("Invalid predicate", bool(pred.getNode()), true,
                           this)) {
      // The following code assumes pred is valid.
      return false;
    }
    auto Ty = pred.getType();
    isValid &= expectCompareTrue("Predicate must be a vector",
                                 Ty->dims().size(), size_t(1), this);
  }

  if (getParent()) {
    isValid &=
        expectCompareTrue("Node not present in its parent",
                          std::find(getParent()->getNodes().begin(),
                                    getParent()->getNodes().end(),
                                    *this) != getParent()->getNodes().end(),
                          true, this);
  }

  // Verify node-specific properties:
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    isValid &= static_cast<const CLASS *>(this)->verify();                     \
    break;
#include "glow/AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
  return isValid;
}

//===----------------------------------------------------------------------===//
// ilist_traits<glow::Node> Implementation
//===----------------------------------------------------------------------===//

// The trait object is embedded into a Function.  Use dirty hacks to
// reconstruct the Function from the 'self' pointer of the trait.
Function *llvm::ilist_traits<Node>::getContainingFunction() {
  size_t Offset(size_t(&((Function *)nullptr->*Function::getNodesMemberPtr())));
  iplist<Node> *Anchor(static_cast<iplist<Node> *>(this));
  return reinterpret_cast<Function *>(reinterpret_cast<char *>(Anchor) -
                                      Offset);
}

void llvm::ilist_traits<Node>::addNodeToList(Node *node) {
  assert(node->getParent() == nullptr && "Already in a list!");
  node->setParent(getContainingFunction());
}

void llvm::ilist_traits<Node>::removeNodeFromList(Node *node) {
  // When an instruction is removed from a function, clear the parent pointer.
  assert(node->getParent() && "Not in a list!");
  node->setParent(nullptr);
}

void llvm::ilist_traits<Node>::transferNodesFromList(
    llvm::ilist_traits<Node> &L2, node_iterator first, node_iterator last) {
  // If transferring nodes within the same Function, no reason to
  // update their parent pointers.
  Function *ThisParent = getContainingFunction();
  if (ThisParent == L2.getContainingFunction())
    return;

  // Update the parent fields in the nodes.
  for (; first != last; ++first)
    first->setParent(ThisParent);
}
