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

bool Storage::isEqual(const Storage &other) const {
  /// A storage should be equal only to itself!
  return this == &other;
}

llvm::hash_code Constant::getHash() const {
  return llvm::hash_combine(getName(), getType());
}

llvm::hash_code Placeholder::getHash() const {
  return llvm::hash_combine(getName());
}
//===----------------------------------------------------------------------===//
//                        Visitor methods
//===----------------------------------------------------------------------===//

void Storage::visit(Node *parent, NodeWalker *visitor) {
  if (!visitor->shouldVisit(parent, this)) {
    return;
  }
  visitor->pre(parent, this);
  visitor->post(parent, this);
}

void Storage::visit(const Node *parent, NodeWalker *visitor) const {
  if (!visitor->shouldVisit(parent, this)) {
    return;
  }
  visitor->pre(parent, this);
  visitor->post(parent, this);
}

//===----------------------------------------------------------------------===//
//                     Edge getters methods
//===----------------------------------------------------------------------===//
unsigned Storage::getNumInputs() const { return 0; }

std::string Storage::getInputName(unsigned idx) const {
  llvm_unreachable("Invalid index");
}

NodeValue Storage::getNthInput(unsigned idx) {
  llvm_unreachable("Invalid index");
}

llvm::StringRef Storage::getOutputName(unsigned idx) const {
  if (idx == 0) {
    return "Output";
  }
  llvm_unreachable("Invalid index");
}

bool Storage::hasSideEffects() const { return false; }

Node *Storage::clone() const { llvm_unreachable("Storage can't be cloned."); }

//===----------------------------------------------------------------------===//
//                     Debug description methods
//===----------------------------------------------------------------------===//

std::string Constant::getDebugDesc() const {
  DescriptionBuilder db(getKindName());
  db.addParam("name", quote(getName()))
      .addParam("output", *getType())
      .addParam("users", getNumUsers());
  return db;
}

std::string Placeholder::getDebugDesc() const {
  DescriptionBuilder db(getKindName());
  db.addParam("name", quote(getName()))
      .addParam("output", *getType())
      .addParam("users", getNumUsers())
      .addParam("trainable", isTraining());
  return db;
}

//===----------------------------------------------------------------------===//
//                       Nodes verification
//===----------------------------------------------------------------------===//

/// Check that the type of the first operand \p A matches the type of the second
/// operand \p B.
static void checkSameType(NodeValue A, NodeValue B) {
  assert(A.getType() == B.getType() && "Invalid type");
}

/// Check that the shape of the first operand \p A matches the shape of the
/// second operand \p B.
static void checkSameShape(NodeValue A, NodeValue B) {
  assert(A.dims() == B.dims() && "Invalid shape");
}

/// Check that the element type of the operand \p A matches expected type \p
/// expected Type.
static void checkType(NodeValue A, ElemKind expectedType) {
  assert(A.getElementType() == expectedType && "Invalid type");
}

/// Check that the type of the first operand \p A matches the type of the second
/// operand \p B but ignore the actual shape. Use only element type and
/// quantization parameters in comparison.
static void checkTypeIgnoreShape(NodeValue A, NodeValue B) {
  assert(A.getElementType() == B.getElementType() && "Invalid element type");
  assert(A.getType()->isQuantizedType() == B.getType()->isQuantizedType() &&
         "Invalid mix of quantized and non quantized types");

  if (A.getType()->isQuantizedType()) {
    assert(A.getType()->getScale() == B.getType()->getScale() &&
           "Invalid scale");
    assert(A.getType()->getOffset() == B.getType()->getOffset() &&
           "Invalid offset");
  }
}

static void verifyConvolution(NodeValue src, NodeValue dest, NodeValue filter,
                              NodeValue bias,
                              llvm::ArrayRef<unsigned_t> kernels,
                              llvm::ArrayRef<unsigned_t> strides,
                              llvm::ArrayRef<unsigned_t> pads,
                              unsigned_t group) {
  assert(src.getElementType() == dest.getElementType() && "Invalid Type");
  assert(src.getElementType() == filter.getElementType() && "Invalid Type");
  assert(src.getElementType() == bias.getElementType() && "Invalid Type");

  ShapeNHWC idim(src.getType()->dims());
  ShapeNHWC odim(dest.getType()->dims());
  PaddingTLBR pdim(pads);
  (void)pdim;
  ShapeHW kdim(kernels);
  assert((idim.w + pdim.left + pdim.right) >= kdim.height &&
         (idim.h + pdim.top + pdim.bottom) >= kdim.width &&
         "buffer too small for selected stride");

  assert(idim.c % group == 0 && "channels number must be divisible by groups");

  auto outSz =
      calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides, pads);
  (void)outSz;
  assert(odim.n == idim.n && odim.h == outSz.first && odim.w == outSz.second &&
         odim.c % group == 0 && "Invalid output dimensions");

  auto filterDims = {odim.c, kdim.height, kdim.width, idim.c / (size_t)group};
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

static void verifyPool(NodeValue src, NodeValue dest,
                       llvm::ArrayRef<unsigned_t> kernels,
                       llvm::ArrayRef<unsigned_t> strides,
                       llvm::ArrayRef<unsigned_t> pads) {
  ShapeNHWC idim = ShapeNHWC(src.getType()->dims());
  ShapeNHWC odim = ShapeNHWC(dest.getType()->dims());
  (void)odim;
  PaddingTLBR pdim(pads);
  (void)pdim;
  ShapeHW kdim(kernels);
  assert((idim.w + pdim.left + pdim.right) >= kdim.height &&
         (idim.h + pdim.top + pdim.bottom) >= kdim.width &&
         "buffer too small for selected stride");

  auto outSz =
      calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides, pads);
  ShapeNHWC exp(idim.n, outSz.first, outSz.second, idim.c);
  (void)exp;
  assert(exp == odim && "Unexpected output dimensions");
  checkTypeIgnoreShape(src, dest);
}

static void verifyBatchNormalization(NodeValue src, NodeValue dest,
                                     NodeValue bias, NodeValue scale,
                                     NodeValue mean, NodeValue var,
                                     unsigned_t channel) {
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
  if (src.getType()->isQuantizedType()) {
    assert(src.getElementType() == dest.getElementType());
    checkSameShape(src, dest);
  } else {
    checkSameType(src, dest);
  }
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

static void verifyRelu(NodeValue result, NodeValue input) {
  if (input.getType()->isQuantizedType()) {
    assert(result.getType()->isQuantizedType());
    checkSameShape(result, input);
  } else {
    checkSameType(result, input);
  }
}

static void verifyRegression(NodeValue src, NodeValue dest,
                             NodeValue expected) {
  checkSameType(src, dest);
  checkSameType(dest, expected);
}

void ConvolutionNode::verify() const {
  verifyConvolution(getInput(), getResult(), getFilter(), getBias(), Kernels_,
                    Strides_, Pads_, Group_);
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

void Constant::verify() const {
  assert(*getType() == getPayload().getType() &&
         "Underlying tensor type doesn't match constant type");
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
                    Kernels_, Strides_, Pads_, Group_);
}

void ConvertToNode::verify() const {
  assert(getInput().dims() == getResult().dims() && "Shape must be the same");
  TypeRef srcTy = getInput().getType();
  (void)srcTy;
  TypeRef dstTy = getResult().getType();
  (void)dstTy;
  assert(srcTy != dstTy && "Nothing to convert");
  assert(!srcTy->isQuantizedType() && !dstTy->isQuantizedType() &&
         "Quantized conversion should use Dequantize, Quantize and Rescale");
}

void MaxPoolNode::verify() const {
  verifyPool(getInput(), getResult(), Kernels_, Strides_, Pads_);
}

void AvgPoolNode::verify() const {
  verifyPool(getInput(), getResult(), Kernels_, Strides_, Pads_);
}

void MaxPoolGradNode::verify() const {
  verifyInputAndGradInputTypes(getInput(), getGradOfInputNamedInput());
  verifyOutputAndGradOutputTypes(getOriginalOutputForResult(),
                                 getGradOfOriginalOutputNamedResult());
  verifyPool(getGradOfInputNamedInput(), getGradOfOriginalOutputNamedResult(),
             Kernels_, Strides_, Pads_);
}

void AvgPoolGradNode::verify() const {
  verifyInputAndGradInputTypes(getInput(), getGradOfInputNamedInput());
  verifyOutputAndGradOutputTypes(getOriginalOutputForResult(),
                                 getGradOfOriginalOutputNamedResult());
  verifyPool(getGradOfInputNamedInput(), getGradOfOriginalOutputNamedResult(),
             Kernels_, Strides_, Pads_);
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
  unsigned axis = getAxis();
  unsigned count = getCount();
  (void)numDims;
  (void)dest;
  (void)src;
  (void)offsets;
  assert(numDims == src.dims().size() && numDims == offsets.size() &&
         "Invalid number of dimensions");

  for (unsigned i = 0; i < numDims; i++) {
    assert(src.dims()[i] + offsets[i] <= dest.dims()[i] && "out of bounds");
  }

  assert(axis <= src.dims().size() && "Invalid axis.");
  for (size_t i = 0; i < src.dims().size(); i++) {
    size_t mul = (i == axis) ? count : 1;
    (void)mul;
    assert(src.dims()[i] * mul <= dest.dims()[i] &&
           "Small does not fit inside Big.");
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

void TileNode::verify() const {
  auto dest = getResult();
  auto src = getInput();
  unsigned axis = getAxis();
  unsigned count = getCount();
  (void)dest;
  assert(axis <= src.dims().size() && "Invalid axis.");
  for (size_t i = 0; i < src.dims().size(); i++) {
    size_t mul = (i == axis) ? count : 1;
    (void)mul;
    assert(src.dims()[i] * mul == dest.dims()[i] && "Incorrect output shape.");
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

void MeanVarNormalizationNode::verify() const {
  checkType(getMean(), ElemKind::FloatTy);
  checkSameType(getMean(), getVar());
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
VERIFY_ARITHMETIC(Pow);
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
  assert(getBatch().dims().size() > 0 && "Invalid shape");
}

void SparseLengthsWeightedSumNode::verify() const {
  assert(getResult().getElementType() == getData().getElementType() &&
         "Mismatched element types");
  assert(getWeights().getElementType() == getData().getElementType() &&
         "Mismatched element types");
  assert(getIndices().getElementType() == ElemKind::Int64ITy &&
         "Indices must have index type");
  assert(getLengths().getElementType() == ElemKind::Int64ITy &&
         "Lengths must have index type");
  assert(getIndices().dims().size() == 1 && "Indices must be 1D vector");
  assert(getLengths().dims().size() == 1 && "Lengths must be 1D vector");
  assert(getWeights().dims().size() == 1 && "Weights must be 1D vector");
  assert(getWeights().dims()[0] == getIndices().dims()[0] &&
         "Weights and Indices must have the same size");
}

void LengthsToRangesNode::verify() const {
  assert(getResult().getElementType() == getLengths().getElementType() &&
         "Mismatched element types");
  assert(getLengths().getElementType() == ElemKind::Int64ITy &&
         "Lengths must have index type");
  assert(getLengths().dims().size() == 1 && "Lengths must be 1D vector");
  assert(getResult().dims().size() == 2 && "Ranges must be 2D vector");
  assert(getResult().dims()[0] == getLengths().dims()[0] &&
         "Lengths and Ranges must have the same outer dimensions");
  assert(getResult().dims()[1] == 2 && "Inner dimension of Ranges must be 2");
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

void RowwiseQuantizedFullyConnectedNode::verify() const {
  auto src = getInput();
  auto weights = getWeights();
  auto scales = getScales();
  auto offsets = getOffsets();
  auto bias = getBias();
  auto dest = getResult();

  (void)src;
  (void)weights;
  (void)scales;
  (void)offsets;
  (void)bias;
  (void)dest;

  assert(src.dims().size() == 2 && weights.dims().size() == 2 &&
         dest.dims().size() == 2 &&
         "Input, weights and result should be 2D tensor");

  assert(offsets.dims().size() == 1 && scales.dims().size() == 1 &&
         bias.dims().size() == 1 &&
         "Offsets, scales and bias should be 1D tensor");

  assert(src.dims()[0] == dest.dims()[0] &&
         src.dims()[1] == weights.dims()[1] &&
         "Mismatch on expected source dimensions");

  assert(bias.dims()[0] == weights.dims()[0] &&
         weights.dims()[0] == dest.dims()[1] &&
         "Inconsistent bias/weights/dest sizes.");

  assert(scales.dims()[0] == offsets.dims()[0] &&
         scales.dims()[0] == weights.dims()[0] &&
         "Inconsistent weights/offsets/scales sizes");
}

void GatherNode::verify() const {
  assert(getResult().getElementType() == getData().getElementType());
  assert(getIndices().getElementType() == ElemKind::Int64ITy);
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

void LogNode::verify() const {
  if (getResult().getType()->isQuantizedType()) {
    checkSameShape(getInput(), getResult());
  } else {
    checkSameType(getInput(), getResult());
  }
}

void IsNaNNode::verify() const { checkSameShape(getInput(), getResult()); }

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

void SigmoidCrossEntropyWithLogitsNode::verify() const {
  assert(getResult().getElementType() == getLogits().getElementType());
  checkSameType(getLogits(), getTargets());
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
  assert(!inputs.empty() && "Empty concat?!");
  assert(inputs[0].dims().size() > dimension && "concat on invalid dimension");

  size_t nbDims = inputs[0].dims().size();
  for (size_t i = 1; i < inputs.size(); i++) {
    assert(nbDims == inputs[i].dims().size() &&
           "input #dims are incompatible between elements");
    for (size_t j = 0; j < nbDims; j++) {
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
