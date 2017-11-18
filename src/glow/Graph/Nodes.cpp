// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Nodes.h"
#include "glow/Base/Type.h"
#include "glow/Support/Support.h"

using namespace glow;

void NodeUse::setOperand(Node *other) {
  if (other && site_->getNode()) {
    assert(site_->getNode()->getType() == other->getType() &&
           "Setting operand to a node with a different type");
  }
  site_->setOperand(other);
}

NodeValue::NodeValue(Node *N) {
  resNo_ = 0;
  assert(N->getNumRes() == 1 && "Constructing a value for a multi-res node");
  setOperand(N);
}

NodeValue::NodeValue(Node *N, unsigned resNo) {
  assert(resNo < N->getNumRes() && "Invalid result number");
  setOperand(N);
}

void NodeValue::setOperand(Node *v) {
  if (node_ == v) {
    return;
  }

  if (node_) {
    node_->removeUse(NodeUse(this));
    node_ = nullptr;
  }

  if (v) {
    node_ = v;
    v->addUse(NodeUse(this));
  }
}

/// \returns the n'th result type of the node.
TypeRef Node::getType(unsigned idx) const {
  if (idx == -1) {
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

TypeRef NodeValue::getType() const { return node_->getType(resNo_); }

ElemKind NodeValue::getElementType() const {
  return getType()->getElementType();
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

//===----------------------------------------------------------------------===//
//                        Visitor methods
//===----------------------------------------------------------------------===//

void Variable::visit(Node *parent, NodeVisitor *visitor) {
  if (!visitor->shouldVisit(parent, this)) {
    return;
  }
  visitor->pre(parent, this);
  visitor->post(parent, this);
}

//===----------------------------------------------------------------------===//
//                     Debug description methods
//===----------------------------------------------------------------------===//
std::string Node::getDebugDesc() const { return "<node>"; }

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
