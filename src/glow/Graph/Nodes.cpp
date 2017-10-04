// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Nodes.h"
#include "glow/IR/Instrs.h"
#include "glow/IR/Type.h"
#include "glow/Support/Support.h"

using namespace glow;

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

#define DEFINE_CLASS_VISITOR(CLASS_NAME)                                       \
  void CLASS_NAME::visit(Node *parent, NodeVisitor *visitor) {                 \
    if (!visitor->shouldVisit(parent, this))                                   \
      return;                                                                  \
    visitor->pre(parent, this);                                                \
    in_->visit(this, visitor);                                                 \
    visitor->post(parent, this);                                               \
  }

DEFINE_CLASS_VISITOR(ConvolutionNode)
DEFINE_CLASS_VISITOR(PoolNode)
DEFINE_CLASS_VISITOR(FullyConnectedNode)
DEFINE_CLASS_VISITOR(LocalResponseNormalizationNode)
DEFINE_CLASS_VISITOR(ReluNode)
DEFINE_CLASS_VISITOR(ReshapeNode)
DEFINE_CLASS_VISITOR(TransposeNode)
DEFINE_CLASS_VISITOR(SigmoidNode)
DEFINE_CLASS_VISITOR(TanhNode)
DEFINE_CLASS_VISITOR(RegressionNode)
DEFINE_CLASS_VISITOR(BatchNormalizationNode)

void ArithmeticNode::visit(Node *parent, NodeVisitor *visitor) {
  if (!visitor->shouldVisit(parent, this)) {
    return;
  }
  visitor->pre(parent, this);
  LHS_->visit(this, visitor);
  RHS_->visit(this, visitor);
  visitor->post(parent, this);
}

void SoftMaxNode::visit(Node *parent, NodeVisitor *visitor) {
  if (!visitor->shouldVisit(parent, this)) {
    return;
  }
  visitor->pre(parent, this);
  in_->visit(this, visitor);
  selected_->visit(this, visitor);
  visitor->post(parent, this);
}

void ConcatNode::visit(Node *parent, NodeVisitor *visitor) {
  if (!visitor->shouldVisit(parent, this)) {
    return;
  }
  visitor->pre(parent, this);
  for (auto &I : in_) {
    I->visit(this, visitor);
  }
  visitor->post(parent, this);
}

//===----------------------------------------------------------------------===//
//                     Debug description methods
//===----------------------------------------------------------------------===//
std::string Node::getDebugDesc() const { return "<node>"; }

std::string Variable::getDebugDesc() const {
  DescriptionBuilder db(getName());
  db.addParam("output", *getType());
  db.addParam("init", WeightVar::getInitKindStr(initKind_));
  db.addParam("val", val_);
  return db;
}

std::string ConvolutionNode::getDebugDesc() const {
  DescriptionBuilder db(getName());
  db.addParam("input", *in_->getType());
  db.addParam("output", *getType());
  db.addParam("filter", *filter_->getType());
  db.addParam("bias", *bias_->getType());
  db.addParam("kernel", kernel_);
  db.addParam("stride", stride_);
  db.addParam("pad", pad_);
  db.addParam("depth", depth_);
  return db;
}
std::string PoolNode::getDebugDesc() const {
  DescriptionBuilder db(getName());
  db.addParam("input", *in_->getType());
  db.addParam("output", *getType());
  db.addParam("kernel", kernel_);
  db.addParam("stride", stride_);
  db.addParam("pad", pad_);
  db.addParam("kind", kind_ == PoolInst::OpKind::Max ? "max" : "avg");
  return db;
}

std::string FullyConnectedNode::getDebugDesc() const {
  DescriptionBuilder db(getName());
  db.addParam("input", *in_->getType());
  db.addParam("output", *getType());
  db.addParam("filter", *filter_->getType());
  db.addParam("bias", *bias_->getType());
  db.addParam("depth", depth_);
  return db;
}

std::string LocalResponseNormalizationNode::getDebugDesc() const {
  DescriptionBuilder db(getName());
  db.addParam("input", *in_->getType());
  db.addParam("alpha", alpha_);
  db.addParam("beta", beta_);
  db.addParam("half window size", this->halfWindowSize_);
  db.addParam("scale", *scale_->getType());
  return db;
}

std::string ConcatNode::getDebugDesc() const {
  DescriptionBuilder db(getName());
  for (auto input : in_) {
    db.addParam("input", *input->getType());
  }
  db.addParam("output", *getType());
  db.addParam("dimension", dim_);
  return db;
}

std::string SoftMaxNode::getDebugDesc() const {
  DescriptionBuilder db(getName());
  db.addParam("input", *in_->getType());
  db.addParam("selected", *selected_->getType());
  return db;
}

std::string RegressionNode::getDebugDesc() const {
  DescriptionBuilder db(getName());
  db.addParam("input", *in_->getType());
  db.addParam("expected", *expected_->getType());
  return db;
}

std::string BatchNormalizationNode::getDebugDesc() const {
  DescriptionBuilder db(getName());
  db.addParam("input", *in_->getType());

  db.addParam("beta", *bias_->getType());
  db.addParam("gamma", *scale_->getType());

  db.addParam("channelIdx", channelIdx_);
  db.addParam("epsilon", epsilon_);
  db.addParam("momentum", momentum_);
  return db;
}

std::string ArithmeticNode::getDebugDesc() const {
  DescriptionBuilder db(getName());
  db.addParam("output", *getType());
  db.addParam("op", kind_ == ArithmeticInst::OpKind::Add ? "add" : "mul");
  return db;
}

#define DEFINE_CLASS_REPR(CLASS_NAME)                                          \
  std::string CLASS_NAME::getDebugDesc() const {                               \
    DescriptionBuilder db(getName());                                          \
    db.addParam("input", *in_->getType());                                     \
    return db;                                                                 \
  }

DEFINE_CLASS_REPR(ReluNode);
DEFINE_CLASS_REPR(ReshapeNode);
DEFINE_CLASS_REPR(TransposeNode);
DEFINE_CLASS_REPR(SigmoidNode);
DEFINE_CLASS_REPR(TanhNode);
