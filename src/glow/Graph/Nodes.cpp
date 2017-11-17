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
