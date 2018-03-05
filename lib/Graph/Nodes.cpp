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

void Variable::initPayload() {
  payload_.reset(*getType());

  switch (getTrainKind()) {
  case TrainKind::None:
    break;

  case TrainKind::Broadcast: {
    switch (payload_.getElementType()) {
    case ElemKind::FloatTy: {
      payload_.getHandle<float>().clear(val_);
      break;
    }
    case ElemKind::Int8QTy: {
      payload_.getHandle<int8_t>().clear(val_);
      break;
    };
    case ElemKind::Int32QTy: {
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

  case TrainKind::Xavier: {
    switch (payload_.getElementType()) {
    case ElemKind::FloatTy: {
      payload_.getHandle<float>().initXavier(val_);
      break;
    }
    case ElemKind::Int8QTy: {
      payload_.getHandle<int8_t>().initXavier(val_);
      break;
    };
    case ElemKind::Int32QTy: {
      payload_.getHandle<int32_t>().initXavier(val_);
      break;
    }
    case ElemKind::IndexTy: {
      payload_.getHandle<size_t>().initXavier(val_);
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
  return llvm::hash_combine(getName(), getTrainKind(), getType(),
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

void Variable::visit(const Node *parent, NodeWalker *visitor) const {
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
NodeValue &Variable::getNthInput(unsigned idx) {
  llvm_unreachable("Invalid index");
}
llvm::StringRef Variable::getOutputName(unsigned idx) const {
  if (idx == 0) {
    return "Output";
  }
  llvm_unreachable("Invalid index");
}
bool Variable::hasSideEffects() const { return false; }

Node *Variable::clone() const {
  llvm_unreachable("variables can't be cloned.");
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

static const char *getVariableTrainKindStr(Variable::TrainKind kind) {
  const char *names[] = {"none", "broadcast", "xavier", nullptr};
  return names[static_cast<int>(kind)];
}

static const char *getVariableVisibilityKindStr(Variable::VisibilityKind kind) {
  const char *names[] = {"public", "private", nullptr};
  return names[static_cast<int>(kind)];
}

std::string Variable::getDebugDesc() const {
  DescriptionBuilder db(getKindName());
  db.addParam("name", quote(getName()))
      .addParam("output", *getType())
      .addParam("visibility", getVariableVisibilityKindStr(visibility_));
  if (train_ != Variable::TrainKind::None) {
    db.addParam("init", getVariableTrainKindStr(train_)).addParam("val", val_);
  }
  db.addParam("users", getNumUsers());
  return db;
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
    assert(Ty->dims().size() == 1 && Ty->dims()[0] == 1 &&
           "Predicate must be a boolean tensor");
    assert(Ty->getElementType() == ElemKind::IndexTy &&
           "Predicates are booleans");
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

/// Check that the type of the first operand matches the type of the second
/// operand.
static void checkSameType(NodeValue A, NodeValue B) {
  assert(A.getType() == B.getType() && "Invalid type");
}

/// Check that the shape of the first operand matches the shape of the second
/// operand.
static void checkSameShape(NodeValue A, NodeValue B) {
  assert(A.dims() == B.dims() && "Invalid shape");
}

static void checkType(NodeValue A, ElemKind expectedType) {
  assert(A.getElementType() == expectedType && "Invalid type");
}

static void checkSameDims(NodeValue A, NodeValue B) {
  assert(A.dims().equals(B.dims()) && "Dimensions mismatch");
}

static void verifyConvolution(NodeValue src, NodeValue dest, NodeValue filter,
                              NodeValue bias, size_t kernel, size_t stride,
                              size_t pad, size_t depth) {
  assert(src.getElementType() == dest.getElementType() && "Invalid Type");
  assert(src.getElementType() == filter.getElementType() && "Invalid Type");
  assert(src.getElementType() == bias.getElementType() && "Invalid Type");

  ShapeNHWC idim(src.getType()->dims());
  ShapeNHWC odim(dest.getType()->dims());

  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  auto outSz = calculateConvOutputDims(idim.h, idim.w, kernel, stride, pad);
  ShapeNHWC exp(idim.n, outSz.first, outSz.second, depth);
  (void)exp;
  assert(exp == odim && "Invalid output dimensions");

  auto filterDims = {depth, kernel, kernel, idim.c};
  assert(filter.getType()->dims().equals(filterDims) && "Invalid filter dims");
  (void)filterDims;

  auto biasDims = {depth};
  assert(bias.getType()->dims().equals(biasDims) && "Invalid bias dims");
  (void)biasDims;
}

static void verifyFullyConnected(NodeValue src, NodeValue weights,
                                 NodeValue bias, NodeValue dest) {
  assert(src.dims()[0] == dest.dims()[0] &&
         flattenCdr(src.dims()).second == weights.dims()[0] &&
         "Mismatch on expected source dimensions");

  assert(bias.dims()[0] == weights.dims()[1] &&
         weights.dims()[1] == dest.dims()[1] &&
         "Inconsistent bias/weights/dest sizes.");
}

static void verifyPool(NodeValue src, NodeValue dest, size_t kernel,
                       size_t stride, size_t pad) {
  ShapeNHWC idim = ShapeNHWC(src.getType()->dims());
  ShapeNHWC odim = ShapeNHWC(dest.getType()->dims());
  (void)odim;
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  auto outSz = calculateConvOutputDims(idim.h, idim.w, kernel, stride, pad);
  ShapeNHWC exp(idim.n, outSz.first, outSz.second, idim.c);
  (void)exp;
  assert(exp == odim && "Unexpected output dimensions");
}

static void verifyBatchNormalization(NodeValue src, NodeValue dest,
                                     NodeValue bias, NodeValue scale,
                                     NodeValue mean, NodeValue var,
                                     size_t channel) {
  checkSameType(dest, src);

  // Figure out how many channels are in the tensor.
  size_t channels = src.dims()[channel];

  auto exp = {channels};
  (void)exp;
  assert(bias.getType()->dims().equals(exp) && "Invalid bias dim");
  assert(scale.getType()->dims().equals(exp) && "Invalid scale dim");
  assert(mean.getType()->dims().equals(exp) && "Invalid mean dim");
  assert(var.getType()->dims().equals(exp) && "Invalid var dim");
}

static void verifySigmoid(NodeValue src, NodeValue dest) {
  checkSameType(src, dest);
}

static void verifyTanh(NodeValue src, NodeValue dest) {
  checkSameType(src, dest);
}

static void verifySoftMax(NodeValue src, NodeValue dest) {
  checkSameType(src, dest);
  assert(src.dims() == dest.dims() && "Invalid shape");
}

static void verifyCrossEntropyLoss(NodeValue P, NodeValue CE,
                                   NodeValue labels) {
  assert(P.getElementType() == CE->getElementType());
  assert(P.dims()[0] == labels.dims()[0] && "Invalid shape");
}

static void verifyLocalResponseNormalization(NodeValue src, NodeValue dest) {
  checkSameType(src, dest);
}

static void verifyArithmetic(NodeValue LHS, NodeValue RHS, NodeValue res) {
  checkSameShape(res, LHS);
  checkSameShape(LHS, RHS);
}

static void verifyRelu(NodeValue src, NodeValue dest) {
  checkSameType(src, dest);
}

static void verifyRegression(NodeValue src, NodeValue dest,
                             NodeValue expected) {
  checkSameType(src, dest);
  checkSameType(dest, expected);
}

void ConvolutionNode::verify() const {
  verifyConvolution(getInput(), getResult(), getFilter(), getBias(), Kernel_,
                    Stride_, Pad_, Depth_);
}

void ConvolutionGradNode::verify() const {
  verifyConvolution(getGradOfInputNamedInput(),
                    getGradOfOriginalOutputNamedResult(),
                    getGradOfInputNamedFilter(), getGradOfInputNamedBias(),
                    Kernel_, Stride_, Pad_, Depth_);
}

void PoolMaxNode::verify() const {
  verifyPool(getInput(), getResult(), Kernel_, Stride_, Pad_);
}

void PoolAvgNode::verify() const {
  verifyPool(getInput(), getResult(), Kernel_, Stride_, Pad_);
}

void PoolMaxGradNode::verify() const {
  verifyPool(getGradOfInputNamedInput(), getGradOfOriginalOutputNamedResult(),
             Kernel_, Stride_, Pad_);
}

void PoolAvgGradNode::verify() const {
  verifyPool(getGradOfInputNamedInput(), getGradOfOriginalOutputNamedResult(),
             Kernel_, Stride_, Pad_);
}

void BatchedMatMulNode::verify() const {
  auto dest = getResult();
  auto lhs = getLHS();
  auto rhs = getRHS();
  (void)dest;
  (void)lhs;
  (void)rhs;

  auto LDims = lhs.dims();
  auto RDims = rhs.dims();
  auto DDims = dest.dims();
  (void)DDims;
  assert(DDims.size() == 3);
  auto elem = dest.getType()->getElementType();
  (void)elem;
  assert(lhs.getType()->getElementType() == elem);
  assert(rhs.getType()->getElementType() == elem);

  size_t N, X, Y;
  std::tie(N, X, Y) = calculateMatMulOutputDims(LDims, RDims);

  assert(N == DDims[0] && "Invalid matrix dims");
  assert(X == DDims[1] && "Invalid matrix dims");
  assert(Y == DDims[2] && "Invalid matrix dims");
  (void)N;
  (void)X;
  (void)Y;
}

void SigmoidNode::verify() const { verifySigmoid(getInput(), getResult()); }

void SigmoidGradNode::verify() const {
  verifySigmoid(getGradOfInputNamedInput(),
                getGradOfOriginalOutputNamedResult());
}

void TanhNode::verify() const { verifyTanh(getInput(), getResult()); }

void TanhGradNode::verify() const {
  verifyTanh(getGradOfInputNamedInput(), getGradOfOriginalOutputNamedResult());
}

void SoftMaxNode::verify() const { verifySoftMax(getInput(), getResult()); }

void SoftMaxGradNode::verify() const {
  verifySoftMax(getGradOfInputNamedInput(),
                getGradOfOriginalOutputNamedResult());
}

void CrossEntropyLossNode::verify() const {
  verifyCrossEntropyLoss(getP(), getCE(), getLabels());
}

void CrossEntropyLossGradNode::verify() const {
  verifyCrossEntropyLoss(getGradOfInputNamedP(),
                         getGradOfOriginalOutputNamedCE(),
                         getGradOfInputNamedLabels());
}

void ReshapeNode::verify() const {
  assert(getResult().getType()->size() == getInput().getType()->size() &&
         "Reshape into a different size");
}

void TransposeNode::verify() const {
  auto dest = getResult();
  auto src = getInput();
  (void)dest;
  llvm::SmallVector<size_t, 6> shape;

  auto dims = src.dims();
  for (size_t i = 0; i < dims.size(); i++) {
    shape.push_back(dims[Shuffle_[i]]);
  }

  assert(dest.dims().equals(shape) && "Invalid transpose dims");
}

void BroadcastNode::verify() const {
  auto src = getInput();
  auto dest = getResult();
  auto shape = getShape();
  (void)src;
  (void)dest;
  (void)shape;

  assert(src.dims().size() <= dest.dims().size() &&
         "Source being broadcasted must have <= number dims of result shape.");
  assert(dest.dims().equals(shape) &&
         "New broadcasted shape does not match shape to broadcast to.");
}

void SplatNode::verify() const {}

void InsertTensorNode::verify() const {
  auto dest = getBig();
  auto src = getSmall();
  auto offsets = getStart();
  unsigned numDims = dest.dims().size();
  (void)numDims;
  (void)dest;
  (void)src;
  (void)offsets;
  assert(numDims == src.dims().size() && numDims == offsets.size() &&
         "Invalid number of dimensions");

  for (unsigned i = 0; i < numDims; i++) {
    assert(src.dims()[i] + offsets[i] <= dest.dims()[i] && "out of bounds");
  }
}

void SliceNode::verify() const {
  auto dest = getResult();
  auto src = getInput();
  auto offsets = getStart();
  unsigned numDims = dest.dims().size();
  (void)numDims;
  (void)dest;
  (void)src;
  (void)offsets;
  assert(numDims == src.dims().size() && numDims == offsets.size() &&
         "Invalid number of dimensions");

  for (unsigned i = 0; i < numDims; i++) {
    assert(dest.dims()[i] + offsets[i] <= src.dims()[i] && "out of bounds");
  }
}

void BatchNormalizationNode::verify() const {
  verifyBatchNormalization(getInput(), getResult(), getBias(), getScale(),
                           getMean(), getVar(), ChannelIdx_);
}

void BatchNormalizationGradNode::verify() const {
  verifyBatchNormalization(
      getGradOfInputNamedInput(), getGradOfOriginalOutputNamedResult(),
      getGradOfInputNamedBias(), getGradOfInputNamedScale(),
      getGradOfInputNamedMean(), getGradOfInputNamedVar(), ChannelIdx_);
}

void LocalResponseNormalizationNode::verify() const {
  verifyLocalResponseNormalization(getInput(), getResult());
}

void LocalResponseNormalizationGradNode::verify() const {
  verifyLocalResponseNormalization(getGradOfInputNamedInput(),
                                   getGradOfOriginalOutputNamedResult());
}

#define VERIFY_ARITHMETIC(NODE_NAME_)                                          \
  void NODE_NAME_##Node::verify() const {                                      \
    verifyArithmetic(getLHS(), getRHS(), getNthResult(0));                     \
  }
VERIFY_ARITHMETIC(Add);
VERIFY_ARITHMETIC(Mul);
VERIFY_ARITHMETIC(Sub);
VERIFY_ARITHMETIC(Div);
VERIFY_ARITHMETIC(Max);
VERIFY_ARITHMETIC(Min);
VERIFY_ARITHMETIC(CmpLTE);
#undef VERIFY_ARITHMETIC

#define VERIFY_ARITHMETIC(NODE_NAME_)                                          \
  void NODE_NAME_##Node::verify() const {                                      \
    verifyArithmetic(getGradOfInputNamedLHS(), getGradOfInputNamedRHS(),       \
                     getGradOfOriginalOutputNamedResult());                    \
  }
VERIFY_ARITHMETIC(AddGrad);
VERIFY_ARITHMETIC(MulGrad);
VERIFY_ARITHMETIC(SubGrad);
VERIFY_ARITHMETIC(DivGrad);
#undef VERIFY_ARITHMETIC

void BatchedAddNode::verify() const {
  auto batchShape = getBatch().dims();
  auto rhsShape = getSlice().dims();
  assert(batchShape.drop_front() == rhsShape && "Invalid shape");
  assert(getBatch().dims() == getResult().dims() && "Invalid dest type");
  (void)batchShape;
  (void)rhsShape;
  assert(getBatch().getType()->getElementType() ==
             getSlice().getType()->getElementType() &&
         "Mismatched element types");
}

void BatchedReduceAddNode::verify() const {
  assert(getBatch().dims().size() > 1 && "Invalid shape");
}

void SGDNode::verify() const {
  if (Momentum_ > 0.0) {
    assert(getGradient().getType() == getGsum().getType() &&
           "Invalid gsum type");
  }

  assert(getGradient().getType() == getWeight().getType() &&
         "Invalid weight or gradient type");
}

void QuantizationProfileNode::verify() const {
  // Make sure that input tensor is a floating point type.
  assert(getInput().getElementType() == ElemKind::FloatTy &&
         "Floating point type is expected");

  // Check computation info has proper size.
  assert(getComputationInfo().dims().size() == 1 &&
         "Computation info should be 1 dimensional");
  assert(getComputationInfo().dims()[0] == 2 &&
         "Computation info should contain Min and Max value only");
}

void QuantizeNode::verify() const {
  // Dest must be quantized.
  checkType(getResult(), ElemKind::Int8QTy);
  // Src must be float.
  checkType(getInput(), ElemKind::FloatTy);
  checkSameDims(getResult(), getInput());
}

void DequantizeNode::verify() const {
  // Dest must be float.
  checkType(getResult(), ElemKind::FloatTy);
  // Src must be quantized.
  checkType(getInput(), ElemKind::Int8QTy);
  checkSameDims(getResult(), getInput());
}

void RescaleQuantizedNode::verify() const {
  // Dest must be quantized.
  checkType(getResult(), ElemKind::Int8QTy);
  // Src must be quantized.
  checkType(getInput(), ElemKind::Int8QTy);
  checkSameDims(getResult(), getInput());
}

void TopKNode::verify() const {
  assert(getInput().getElementType() == ElemKind::FloatTy);
  assert(getValues().getElementType() == ElemKind::FloatTy);
  assert(getValues().dims() == getIndices().dims());
}

void GatherNode::verify() const {
  assert(getResult().getElementType() == getData().getElementType());
  assert(getIndices().getElementType() == ElemKind::IndexTy);
  assert(getResult().dims().size() ==
         getData().dims().size() + getIndices().dims().size() - 1);
}

void IntrinsicNode::verify() const {
  assert(getName().size() && "Name must not be empty");
}

void SaveNode::verify() const { checkSameType(getInput(), getOutput()); }

void PowNode::verify() const { checkSameType(getResult(), getBase()); }

void SelectNode::verify() const {
  checkSameType(getResult(), getCond());
  checkSameType(getResult(), getLHS());
  checkSameType(getResult(), getRHS());
}

void ReluNode::verify() const { verifyRelu(getResult(), getInput()); }

void ReluGradNode::verify() const {
  verifyRelu(getGradOfOriginalOutputNamedResult(), getInput());
}

void RegressionNode::verify() const {
  verifyRegression(getInput(), getResult(), getExpected());
}

void RegressionGradNode::verify() const {
  verifyRegression(getGradOfInputNamedInput(),
                   getGradOfOriginalOutputNamedResult(),
                   getGradOfInputNamedExpected());
}

void FullyConnectedNode::verify() const {
  verifyFullyConnected(getInput(), getWeights(), getBias(), getResult());
}

void FullyConnectedGradNode::verify() const {
  verifyFullyConnected(getGradOfInputNamedInput(), getGradOfInputNamedWeights(),
                       getGradOfInputNamedBias(),
                       getGradOfOriginalOutputNamedResult());
}

void ConcatNode::verify() const {
  auto inputs = getInputs();
  auto dimension = getDim();
  (void)inputs;
  (void)dimension;

  for (size_t i = 1; i < inputs.size(); i++) {
    for (size_t j = 0; j < inputs[0].dims().size(); j++) {
      if (j == dimension) {
        continue;
      }
      assert(inputs[0].dims()[j] == inputs[i].dims()[j]);
    }
  }

  for (size_t i = 0; i < inputs.size(); i++) {
    checkType(inputs[i], getResult()->getElementType());
    if (getResult()->getType()->isQuantizedType()) {
      assert(inputs[i]->getType()->getScale() ==
             getResult()->getType()->getScale());
      assert(inputs[i]->getType()->getOffset() ==
             getResult()->getType()->getOffset());
    }
  }
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
