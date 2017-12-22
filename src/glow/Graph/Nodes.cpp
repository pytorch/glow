// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Nodes.h"
#include "glow/Base/Type.h"
#include "glow/Graph/Graph.h"
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
  assert((!N || (N->getNumRes() == 1)) &&
         "Constructing a value for a multi-res node");
  setOperand(N, 0);
}

NodeValue::NodeValue(Node *N, unsigned resNo) {
  assert(resNo < N->getNumRes() && "Invalid result number");
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

/// \returns the n'th result type of the node.
TypeRef Node::getType(unsigned idx) const {
  if (idx == (unsigned)-1) {
    assert(numRes_ == 1 && "Did not specify the result number for a node "
                           "with multiple results.");
    return types_[0];
  }
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
  assert(numRes_ < max_node_resno && "Too many results");
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
#define DEF_VALUE(CLASS, NAME) case glow::Kinded::Kind::CLASS##Kind:
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
  entries.push_front(
      {{from.getNode(), from.getResNo()}, {to.getNode(), to.getResNo()}});
}

NodeValue UnownedNodeValueMap::get(NodeValue from) {
  for (auto &E : entries) {
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
  for (auto &E : entries) {
    auto &F = E.first;
    if (F.first == from.getNode() && F.second == from.getResNo()) {
      return true;
    }
  }

  return false;
}

llvm::ArrayRef<size_t> NodeValue::dims() const { return getType()->dims(); }

const char *Variable::getInitKindStr(InitKind kind) {
  // extern: No initialization.
  // broadcast: Broadcast a single value to all elements.
  // xavier: Init the tensor with random values using the Xavier method.
  const char *names[] = {"extern", "broadcast", "xavier", nullptr};
  return names[static_cast<int>(kind)];
}

const char *Variable::getInitKindStr() const {
  return getInitKindStr(initKind_);
}

void Variable::initPayload() {
  payload_.reset(*getType());

  switch (getInitKind()) {
  case InitKind::Extern:
    break;

  case InitKind::Broadcast: {
    switch (payload_.getElementType()) {
    case ElemKind::FloatTy: {
      payload_.getHandle<float>().clear(val_);
      break;
    }
    case ElemKind::DoubleTy: {
      payload_.getHandle<double>().clear(val_);
      break;
    }
    case ElemKind::Int8Ty: {
      payload_.getHandle<int8_t>().clear(val_);
      break;
    };
    case ElemKind::Int32Ty: {
      payload_.getHandle<int32_t>().clear(val_);
      break;
    }
    case ElemKind::IndexTy: {
      payload_.getHandle<size_t>().clear(val_);
      break;
    }
    }
    break;
  }

  case InitKind::Xavier: {
    switch (payload_.getElementType()) {
    case ElemKind::FloatTy: {
      payload_.getHandle<float>().randomize(val_);
      break;
    }
    case ElemKind::DoubleTy: {
      payload_.getHandle<double>().randomize(val_);
      break;
    }
    case ElemKind::Int8Ty: {
      payload_.getHandle<int8_t>().randomize(val_);
      break;
    };
    case ElemKind::Int32Ty: {
      payload_.getHandle<int32_t>().randomize(val_);
      break;
    }
    case ElemKind::IndexTy: {
      payload_.getHandle<size_t>().randomize(val_);
      break;
    }
    }
    break;
  }
  }
}

/// Equality predicate for variables.
bool Variable::isEqual(const Variable &other) const {
  /// A variable should be equal only to itself!
  return this == &other;
}

llvm::hash_code Variable::getHash() const {
  return llvm::hash_combine(getName(), getInitKind(), getType(),
                            toBinary(val_));
}
//===----------------------------------------------------------------------===//
//                        Visitor methods
//===----------------------------------------------------------------------===//

void Variable::visit(Node *parent, NodeWalker *visitor) {
  if (!visitor->shouldVisit(parent, this)) {
    return;
  }
  visitor->pre(parent, this);
  visitor->post(parent, this);
}

//===----------------------------------------------------------------------===//
//                     Edge getters methods
//===----------------------------------------------------------------------===//
unsigned Variable::getNumInputs() const { return 0; }
llvm::StringRef Variable::getInputName(unsigned idx) const {
  llvm_unreachable("Invalid index");
}
NodeValue Variable::getInputNode(unsigned idx) const {
  llvm_unreachable("Invalid index");
}
llvm::StringRef Variable::getOutputName(unsigned idx) const {
  if (idx == 0) {
    return "Output";
  }
  llvm_unreachable("Invalid index");
}

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
NodeValue Node::getInputNode(unsigned idx) const {
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<const CLASS *>(this)->getInputNode(idx);
#include "AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }
}

llvm::StringRef Node::getOutputName(unsigned idx) {
  switch (getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return static_cast<const CLASS *>(this)->getOutputName(idx);
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

std::string Variable::getDebugDesc() const {
  DescriptionBuilder db(getKindName());
  db.addParam("name", quote(getName()))
      .addParam("output", *getType())
      .addParam("init", Variable::getInitKindStr(initKind_));
  if (initKind_ != Variable::InitKind::Extern) {
    db.addParam("val", val_);
  }
  db.addParam("users", getNumUsers());
  return db;
}

//===----------------------------------------------------------------------===//
//                     Node hashing support
//===----------------------------------------------------------------------===//

/// These hash functions are required for using llvm::hash_combine.
/// hash_value functions should be defined in the same namespace as
/// the types they apply to.
namespace glow {
/// Convert a float into an unsigned integer binary representation.
/// Do not use union-based tricks, because they introduce undefined behavior.
/// Instead, convert the float to an unsigned integer at the cost of
/// having more hash collisions.
size_t toBinary(float f) {
  /// First convert to an integer and then to an unsigned.
  /// Direct conversion from a float to an unsigned integer may result
  /// in an undefined behavior according to the C++ standard.
  return static_cast<size_t>(static_cast<int>(f));
}

/// FIXME: Provide a more meaningful implementation for Tensors.
llvm::hash_code hash_value(const glow::Tensor &T) { return 0; }

// Types are uniqued, so just a pointer can be used.
llvm::hash_code hash_value(const glow::Type *T) {
  return llvm::hash_value((void *)(T));
}

llvm::hash_code hash_value(glow::Node *N) { return N->getHash(); }

llvm::hash_code hash_value(const glow::NodeValue &NV) {
  return llvm::hash_combine(NV.getNode(), NV.getResNo());
}
} // namespace glow
