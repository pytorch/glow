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
    case ElemKind::DoubleTy: {
      payload_.getHandle<double>().clear(val_);
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
    case ElemKind::DoubleTy: {
      payload_.getHandle<double>().initXavier(val_);
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
      .addParam("visibility", getVariableVisibilityKindStr(visibility_))
      .addParam("init", getVariableTrainKindStr(train_));
  if (train_ != Variable::TrainKind::None) {
    db.addParam("val", val_);
  }
  db.addParam("users", getNumUsers());
  return db;
}

//===----------------------------------------------------------------------===//
//                       Nodes verification
//===----------------------------------------------------------------------===//

void Node::verify() const {
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

static void checkType(NodeValue A, ElemKind expectedType) {
  assert(A.getElementType() == expectedType && "Invalid type");
}

static void checkSameDims(NodeValue A, NodeValue B) {
  assert(A.dims().equals(B.dims()) && "Dimensions mismatch");
}

static void verifyConvDims(ShapeNHWC idim, ShapeNHWC odim, size_t kernel,
                           size_t stride, size_t pad, size_t depth,
                           NodeValue filter, NodeValue bias) {
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

void ConvolutionNode::verify() const {
  auto dest = getResult();
  auto src = getInput();
  auto filter = getFilter();
  auto bias = getBias();

  assert(src.getElementType() == dest.getElementType() && "Invalid Type");
  assert(src.getElementType() == filter.getElementType() && "Invalid Type");
  assert(src.getElementType() == bias.getElementType() && "Invalid Type");

  ShapeNHWC idim(src.getType()->dims());
  ShapeNHWC odim(dest.getType()->dims());
  verifyConvDims(idim, odim, Kernel_, Stride_, Pad_, Depth_, filter, bias);
}

void PoolNode::verify() const {
  auto dest = getResult();
  auto src = getInput();
  ShapeNHWC idim = ShapeNHWC(src.getType()->dims());
  ShapeNHWC odim = ShapeNHWC(dest.getType()->dims());
  (void)odim;
  assert(idim.w >= Kernel_ && idim.h >= Kernel_ &&
         "buffer too small for selected stride");

  auto outSz = calculateConvOutputDims(idim.h, idim.w, Kernel_, Stride_, Pad_);
  ShapeNHWC exp(idim.n, outSz.first, outSz.second, idim.c);
  (void)exp;
  assert(exp == odim && "Unexpected output dimensions");
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
  assert(LDims.size() == 3);
  assert(RDims.size() == 3);
  assert(DDims.size() == 3);
  auto elem = dest.getType()->getElementType();
  (void)elem;
  assert(lhs.getType()->getElementType() == elem);
  assert(rhs.getType()->getElementType() == elem);

  auto outDims =
      calculateMatMulOutputDims(LDims[1], LDims[2], RDims[1], RDims[2]);

  size_t aN = LDims[0];
  size_t bN = RDims[0];
  size_t cN = DDims[0];

  size_t cx = DDims[1];
  size_t cy = DDims[2];

  assert(((aN == 1) || (bN == 1) || (aN == bN)) &&
         "Batch size must be broadcasted or identical");

  // Select the batch size. If the left operand is broadcast (value 1), select
  // the RHS.
  size_t N = (aN != 1 ? aN : bN);
  assert(N == cN);

  assert(outDims.first == cx && outDims.second == cy && "Invalid matrix dims");

  (void)aN;
  (void)bN;
  (void)cN;
  (void)cx;
  (void)cy;
  (void)N;
  (void)outDims;
}

void SigmoidNode::verify() const { checkSameType(getResult(), getInput()); }

void TanhNode::verify() const { checkSameType(getResult(), getInput()); }

void SoftMaxNode::verify() const {
  checkSameType(getResult(), getInput());
  assert(getResult().dims() == getInput().dims() && "Invalid shape");
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
  checkSameType(getResult(), getInput());

  // Figure out how many channels are in the tensor.
  size_t channels = getInput().dims()[ChannelIdx_];

  auto exp = {channels};
  (void)exp;
  assert(getBias().getType()->dims().equals(exp) && "Invalid bias dim");
  assert(getScale().getType()->dims().equals(exp) && "Invalid scale dim");
  assert(getMean().getType()->dims().equals(exp) && "Invalid mean dim");
  assert(getVar().getType()->dims().equals(exp) && "Invalid var dim");
}

void LocalResponseNormalizationNode::verify() const {
  checkSameType(getResult(), getInput());
}

void ArithmeticNode::verify() const {
  checkSameType(getNthResult(0), getLHS());
  checkSameType(getLHS(), getRHS());
}

void BatchedArithmeticNode::verify() const {
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

void BatchedReduceNode::verify() const {
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
  assert(getInput().getElementType() == ElemKind::FloatTy ||
         getInput().getElementType() == ElemKind::DoubleTy &&
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

void SelectNode::verify() const {
  checkSameType(getResult(), getCond());
  checkSameType(getResult(), getLHS());
  checkSameType(getResult(), getRHS());
}

// TODO: verify more kinds of nodes.
#define NOVERIFY(ClassName)                                                    \
  void ClassName::verify() const {}
NOVERIFY(ConvolutionGradNode)
NOVERIFY(PoolGradNode)
NOVERIFY(BatchNormalizationGradNode)
NOVERIFY(LocalResponseNormalizationGradNode)
NOVERIFY(SoftMaxGradNode)
NOVERIFY(ReluNode)
NOVERIFY(RegressionGradNode)
NOVERIFY(FullyConnectedNode)
NOVERIFY(FullyConnectedGradNode)
NOVERIFY(SigmoidGradNode)
NOVERIFY(ArithmeticGradNode)
NOVERIFY(ReluGradNode)
NOVERIFY(TanhGradNode)
NOVERIFY(RegressionNode)
NOVERIFY(ConcatNode)
#undef NOVERIFY

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
