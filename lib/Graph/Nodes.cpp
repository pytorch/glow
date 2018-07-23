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

#include "glow/Graph/Nodes.h"
#include "glow/Base/Type.h"
#include "glow/Graph/Graph.h"
#include "glow/Support/Support.h"

using namespace glow;

void Variable::initPayload(PseudoRNG &PRNG) {
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
      payload_.getHandle<float>().initXavier(val_, PRNG);
      break;
    }
    case ElemKind::Int8QTy: {
      payload_.getHandle<int8_t>().initXavier(val_, PRNG);
      break;
    };
    case ElemKind::Int32QTy: {
      payload_.getHandle<int32_t>().initXavier(val_, PRNG);
      break;
    }
    case ElemKind::IndexTy: {
      payload_.getHandle<size_t>().initXavier(val_, PRNG);
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

NodeValue Variable::getNthInput(unsigned idx) {
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

static const char *getVariableTrainKindStr(Variable::TrainKind kind) {
  const char *names[] = {"none", "broadcast", "xavier", nullptr};
  return names[static_cast<int>(kind)];
}

static const char *getVariableVisibilityKindStr(VisibilityKind kind) {
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

static void verifyConvolution(NodeValue src, NodeValue dest, NodeValue filter,
                              NodeValue bias, size_t kernel, size_t stride,
                              llvm::ArrayRef<size_t> pads, size_t group) {
  assert(src.getElementType() == dest.getElementType() && "Invalid Type");
  assert(src.getElementType() == filter.getElementType() && "Invalid Type");
  assert(src.getElementType() == bias.getElementType() && "Invalid Type");

  ShapeNHWC idim(src.getType()->dims());
  ShapeNHWC odim(dest.getType()->dims());
  PaddingTLBR pdim(pads);
  (void)pdim;
  assert((idim.w + pdim.left + pdim.right) >= kernel &&
         (idim.h + pdim.top + pdim.bottom) >= kernel &&
         "buffer too small for selected stride");

  assert(idim.c % group == 0 && "channels number must be divisible by groups");

  auto outSz =
      calculateConvPoolOutputDims(idim.h, idim.w, kernel, stride, pads);
  (void)outSz;
  assert(odim.n == idim.n && odim.h == outSz.first && odim.w == outSz.second &&
         odim.c % group == 0 && "Invalid output dimensions");

  auto filterDims = {odim.c, kernel, kernel, idim.c / group};
  assert(filter.getType()->dims().equals(filterDims) && "Invalid filter dims");
  (void)filterDims;

  auto biasDims = {odim.c};
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
                       size_t stride, llvm::ArrayRef<size_t> pads) {
  ShapeNHWC idim = ShapeNHWC(src.getType()->dims());
  ShapeNHWC odim = ShapeNHWC(dest.getType()->dims());
  (void)odim;
  PaddingTLBR pdim(pads);
  (void)pdim;
  assert((idim.w + pdim.left + pdim.right) >= kernel &&
         (idim.h + pdim.top + pdim.bottom) >= kernel &&
         "buffer too small for selected stride");

  auto outSz =
      calculateConvPoolOutputDims(idim.h, idim.w, kernel, stride, pads);
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
  assert(P.getElementType() == CE.getElementType());
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
                    Stride_, Pads_, Group_);
}

/// Verify that types of an input and its gradient are the same.
static void verifyInputAndGradInputTypes(NodeValue input, NodeValue gradInput) {
  assert(input.getType() == gradInput.getType() &&
         "Types of input and its gradient should be the same");
}

/// Verify that types of an output and its gradient are the same.
static void verifyOutputAndGradOutputTypes(NodeValue output,
                                           NodeValue gradOutput) {
  assert(output.getType() == gradOutput.getType() &&
         "Types of output and its gradient should be the same");
}

void ConvolutionGradNode::verify() const {
  verifyInputAndGradInputTypes(getInput(), getGradOfInputNamedInput());
  verifyInputAndGradInputTypes(getFilter(), getGradOfInputNamedFilter());
  verifyInputAndGradInputTypes(getBias(), getGradOfInputNamedBias());
  verifyOutputAndGradOutputTypes(getOriginalOutputForResult(),
                                 getGradOfOriginalOutputNamedResult());
  verifyConvolution(getGradOfInputNamedInput(),
                    getGradOfOriginalOutputNamedResult(),
                    getGradOfInputNamedFilter(), getGradOfInputNamedBias(),
                    Kernel_, Stride_, Pads_, Group_);
}

void PoolMaxNode::verify() const {
  verifyPool(getInput(), getResult(), Kernel_, Stride_, Pads_);
}

void PoolAvgNode::verify() const {
  verifyPool(getInput(), getResult(), Kernel_, Stride_, Pads_);
}

void PoolMaxGradNode::verify() const {
  verifyInputAndGradInputTypes(getInput(), getGradOfInputNamedInput());
  verifyOutputAndGradOutputTypes(getOriginalOutputForResult(),
                                 getGradOfOriginalOutputNamedResult());
  verifyPool(getGradOfInputNamedInput(), getGradOfOriginalOutputNamedResult(),
             Kernel_, Stride_, Pads_);
}

void PoolAvgGradNode::verify() const {
  verifyInputAndGradInputTypes(getInput(), getGradOfInputNamedInput());
  verifyOutputAndGradOutputTypes(getOriginalOutputForResult(),
                                 getGradOfOriginalOutputNamedResult());
  verifyPool(getGradOfInputNamedInput(), getGradOfOriginalOutputNamedResult(),
             Kernel_, Stride_, Pads_);
}

void MatMulNode::verify() const {
  auto lhs = getLHS();
  auto rhs = getRHS();
  auto dest = getResult();

  auto LDims = lhs.dims();
  auto RDims = rhs.dims();
  auto DDims = dest.dims();
  (void)LDims;
  (void)RDims;
  (void)DDims;
  assert(DDims.size() == 2);
  auto elem = dest.getType()->getElementType();
  (void)elem;
  assert(lhs.getType()->getElementType() == elem);
  assert(rhs.getType()->getElementType() == elem);

  assert(LDims[0] == DDims[0] && "Invalid matrix dims");
  assert(RDims[1] == DDims[1] && "Invalid matrix dims");
}

void SigmoidNode::verify() const { verifySigmoid(getInput(), getResult()); }

void SigmoidGradNode::verify() const {
  verifyInputAndGradInputTypes(getInput(), getGradOfInputNamedInput());
  verifyOutputAndGradOutputTypes(getOriginalOutputForResult(),
                                 getGradOfOriginalOutputNamedResult());
  verifySigmoid(getGradOfInputNamedInput(),
                getGradOfOriginalOutputNamedResult());
}

void TanhNode::verify() const { verifyTanh(getInput(), getResult()); }

void TanhGradNode::verify() const {
  verifyInputAndGradInputTypes(getInput(), getGradOfInputNamedInput());
  verifyOutputAndGradOutputTypes(getOriginalOutputForResult(),
                                 getGradOfOriginalOutputNamedResult());
  verifyTanh(getGradOfInputNamedInput(), getGradOfOriginalOutputNamedResult());
}

void SoftMaxNode::verify() const { verifySoftMax(getInput(), getResult()); }

void SoftMaxGradNode::verify() const {
  verifyInputAndGradInputTypes(getInput(), getGradOfInputNamedInput());
  verifyInputAndGradInputTypes(getSelected(), getGradOfInputNamedSelected());
  verifyOutputAndGradOutputTypes(getOriginalOutputForResult(),
                                 getGradOfOriginalOutputNamedResult());
  verifySoftMax(getGradOfInputNamedInput(),
                getGradOfOriginalOutputNamedResult());
}

void CrossEntropyLossNode::verify() const {
  verifyCrossEntropyLoss(getP(), getCE(), getLabels());
}

void CrossEntropyLossGradNode::verify() const {
  verifyInputAndGradInputTypes(getLabels(), getGradOfInputNamedLabels());
  verifyInputAndGradInputTypes(getP(), getGradOfInputNamedP());
  verifyOutputAndGradOutputTypes(getOriginalOutputForCE(),
                                 getGradOfOriginalOutputNamedCE());
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
  ShapeVector shape;

  auto dims = src.dims();
  for (size_t i = 0; i < dims.size(); i++) {
    shape.push_back(dims[Shuffle_[i]]);
  }

  assert(dest.dims().equals(shape) && "Invalid transpose dims");
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
  verifyInputAndGradInputTypes(getBias(), getGradOfInputNamedBias());
  verifyInputAndGradInputTypes(getInput(), getGradOfInputNamedInput());
  verifyInputAndGradInputTypes(getMean(), getGradOfInputNamedMean());
  verifyInputAndGradInputTypes(getScale(), getGradOfInputNamedScale());
  verifyInputAndGradInputTypes(getVar(), getGradOfInputNamedVar());
  verifyOutputAndGradOutputTypes(getOriginalOutputForResult(),
                                 getGradOfOriginalOutputNamedResult());
  verifyBatchNormalization(
      getGradOfInputNamedInput(), getGradOfOriginalOutputNamedResult(),
      getGradOfInputNamedBias(), getGradOfInputNamedScale(),
      getGradOfInputNamedMean(), getGradOfInputNamedVar(), ChannelIdx_);
}

void LocalResponseNormalizationNode::verify() const {
  verifyLocalResponseNormalization(getInput(), getResult());
}

void LocalResponseNormalizationGradNode::verify() const {
  verifyInputAndGradInputTypes(getInput(), getGradOfInputNamedInput());
  verifyOutputAndGradOutputTypes(getOriginalOutputForResult(),
                                 getGradOfOriginalOutputNamedResult());
  verifyLocalResponseNormalization(getGradOfInputNamedInput(),
                                   getGradOfOriginalOutputNamedResult());
}

#define VERIFY_ARITHMETIC(NODE_NAME_)                                          \
  void NODE_NAME_##Node::verify() const {                                      \
    verifyArithmetic(getLHS(), getRHS(), getResult());                         \
  }
VERIFY_ARITHMETIC(Add);
VERIFY_ARITHMETIC(Mul);
VERIFY_ARITHMETIC(Sub);
VERIFY_ARITHMETIC(Div);
VERIFY_ARITHMETIC(Max);
VERIFY_ARITHMETIC(Min);
VERIFY_ARITHMETIC(CmpEQ);
#undef VERIFY_ARITHMETIC

void CmpLTENode::verify() const {
  verifyArithmetic(getLHS(), getRHS(), getResult());
  if (getResult().getType()->isQuantizedType()) {
    // Quantization scale params for result must be (1.0, 0).
    assert(getResult().getType()->getScale() == 1.0);
    assert(getResult().getType()->getOffset() == 0);
  }
}

#define VERIFY_ARITHMETIC(NODE_NAME_)                                          \
  void NODE_NAME_##Node::verify() const {                                      \
    verifyInputAndGradInputTypes(getLHS(), getGradOfInputNamedLHS());          \
    verifyInputAndGradInputTypes(getRHS(), getGradOfInputNamedRHS());          \
    verifyOutputAndGradOutputTypes(getOriginalOutputForResult(),               \
                                   getGradOfOriginalOutputNamedResult());      \
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
  assert(getResult().getElementType() == getBatch().getElementType() &&
         "Mismatched element types");
  assert(getBatch().dims().size() > 1 && "Invalid shape");
}

void SparseLengthsSumNode::verify() const {
  assert(getResult().getElementType() == getData().getElementType() &&
         "Mismatched element types");
  assert(getIndices().getElementType() == ElemKind::IndexTy &&
         "Indices must have index type");
  assert(getLengths().getElementType() == ElemKind::IndexTy &&
         "Lengths must have index type");
  assert(getIndices().dims().size() == 1 && "Indices must be 1D vector");
  assert(getLengths().dims().size() == 1 && "Lengths must be 1D vector");
}

void SGDNode::verify() const {
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

void IntLookupTableNode::verify() const {
  assert(getInput().getElementType() == ElemKind::Int8QTy &&
         "Quantized input is expected");
  assert(getResult().getElementType() == ElemKind::Int8QTy &&
         "Quantized output is expected");

  assert(getMapping().dims().size() == 1 && "Mapping should be 1 dimensional");
  assert(getMapping().dims()[0] == 256 &&
         "Mapping should cover whole int8 range");
}

void QuantizeNode::verify() const {
  // Dest must be quantized.
  checkType(getResult(), ElemKind::Int8QTy);
  // Src must be float.
  checkType(getInput(), ElemKind::FloatTy);
  checkSameShape(getResult(), getInput());
}

void DequantizeNode::verify() const {
  // Dest must be float.
  checkType(getResult(), ElemKind::FloatTy);
  // Src must be quantized.
  checkType(getInput(), ElemKind::Int8QTy);
  checkSameShape(getResult(), getInput());
}

void RescaleQuantizedNode::verify() const {
  // Dest must be quantized.
  checkType(getResult(), ElemKind::Int8QTy);
  // Src must be quantized.
  checkType(getInput(), ElemKind::Int8QTy);
  checkSameShape(getResult(), getInput());
}

void TopKNode::verify() const {
  assert(getValues().dims() == getIndices().dims());
  if (getInput().getType()->isQuantizedType()) {
    // Quantization scales must be identical; no rescaling is allowed.
    assert(getValues().getType()->getScale() ==
           getInput().getType()->getScale());
    assert(getValues().getType()->getOffset() ==
           getInput().getType()->getOffset());
  }
}

void GatherNode::verify() const {
  assert(getResult().getElementType() == getData().getElementType());
  assert(getIndices().getElementType() == ElemKind::IndexTy);
  assert(getResult().dims().size() ==
         getData().dims().size() + getIndices().dims().size() - 1);
  if (getResult().getType()->isQuantizedType()) {
    // Quantization scales must be identical; no rescaling is allowed.
    assert(getResult().getType()->getScale() ==
           getData().getType()->getScale());
    assert(getResult().getType()->getOffset() ==
           getData().getType()->getOffset());
  }
}

void ScatterAssignNode::verify() const {
  const auto &slicesDims = getSlices().dims();
  const auto &dataDims = getData().dims();
  const auto &indicesDims = getIndices().dims();
  (void)slicesDims;
  (void)dataDims;
  (void)indicesDims;

  assert(indicesDims.size() == 1 &&
         "Indices should be a single dimensional vector.");
  assert(indicesDims[0] == slicesDims[0] &&
         "There should be an index for each slice.");
  assert(slicesDims.size() == dataDims.size() &&
         "Slices and data should have same number of dimensions.");

  if (dataDims.size() > 1) {
    for (size_t i = 1; i < dataDims.size(); i++) {
      assert(dataDims[i] == slicesDims[i] &&
             "Slice shape should equal data shape without first dimension.");
    }
  }
}

void SaveNode::verify() const { checkSameType(getInput(), getOutput()); }

void PowNode::verify() const { checkSameType(getResult(), getBase()); }

void LogNode::verify() const { checkSameType(getInput(), getResult()); }

void SelectNode::verify() const {
  assert(getResult().getElementType() == getCond().getElementType());
  assert(getResult().getElementType() == getLHS().getElementType());
  assert(getResult().getElementType() == getRHS().getElementType());
  checkSameShape(getResult(), getCond());
  checkSameShape(getResult(), getLHS());
  checkSameShape(getResult(), getRHS());
}

void ReluNode::verify() const { verifyRelu(getResult(), getInput()); }

void ReluGradNode::verify() const {
  verifyInputAndGradInputTypes(getInput(), getGradOfInputNamedInput());
  verifyOutputAndGradOutputTypes(getOriginalOutputForResult(),
                                 getGradOfOriginalOutputNamedResult());
  verifyRelu(getGradOfOriginalOutputNamedResult(), getInput());
}

void RegressionNode::verify() const {
  verifyRegression(getInput(), getResult(), getExpected());
}

void RegressionGradNode::verify() const {
  verifyInputAndGradInputTypes(getExpected(), getGradOfInputNamedExpected());
  verifyInputAndGradInputTypes(getInput(), getGradOfInputNamedInput());
  verifyOutputAndGradOutputTypes(getOriginalOutputForResult(),
                                 getGradOfOriginalOutputNamedResult());
  verifyRegression(getGradOfInputNamedInput(),
                   getGradOfOriginalOutputNamedResult(),
                   getGradOfInputNamedExpected());
}

void FullyConnectedNode::verify() const {
  verifyFullyConnected(getInput(), getWeights(), getBias(), getResult());
}

void FullyConnectedGradNode::verify() const {
  verifyInputAndGradInputTypes(getBias(), getGradOfInputNamedBias());
  verifyInputAndGradInputTypes(getInput(), getGradOfInputNamedInput());
  verifyInputAndGradInputTypes(getWeights(), getGradOfInputNamedWeights());
  verifyOutputAndGradOutputTypes(getOriginalOutputForResult(),
                                 getGradOfOriginalOutputNamedResult());
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
    checkType(inputs[i], getResult().getElementType());
    if (getResult().getType()->isQuantizedType()) {
      assert(inputs[i].getType()->getScale() ==
             getResult().getType()->getScale());
      assert(inputs[i].getType()->getOffset() ==
             getResult().getType()->getOffset());
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
size_t toBinary(float f) {
  // Convert floating-point binary representation to integer.  memcpy compiles
  // to a simple asm move on platforms we support.
  static_assert(sizeof(size_t) >= sizeof(float),
                "size_t is too small on this platform");
  size_t ret = 0;
  memcpy(&ret, &f, sizeof(float));
  return ret;
}

llvm::hash_code hash_value(const glow::Tensor &T) { return T.size(); }

// Types are uniqued, so just a pointer can be used.
llvm::hash_code hash_value(const glow::Type *T) {
  return llvm::hash_value((void *)(T));
}

llvm::hash_code hash_value(glow::Node *N) { return N->getHash(); }

llvm::hash_code hash_value(const glow::NodeValue &NV) {
  return llvm::hash_combine(NV.getNode(), NV.getResNo());
}

llvm::hash_code hash_value(const glow::NodeHandle &NV) {
  return llvm::hash_combine(NV.getNode(), NV.getResNo());
}
} // namespace glow
