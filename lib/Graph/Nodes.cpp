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
#include "glow/Graph/VerifierHelper.h"
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

static bool verifyConvolution(NodeValue src, NodeValue dest, NodeValue filter,
                              NodeValue bias,
                              llvm::ArrayRef<unsigned_t> kernels,
                              llvm::ArrayRef<unsigned_t> strides,
                              llvm::ArrayRef<unsigned_t> pads, unsigned_t group,
                              unsigned_t dilation, bool checkBiasType = true) {
  const Node *parent = dest.getNode();
  bool isValid = checkType(src, dest.getElementType(), parent);
  isValid &= checkType(src, filter.getElementType(), parent);
  if (checkBiasType) {
    // Non quantization type check.
    if (src.getElementType() == ElemKind::FloatTy) {
      isValid &= checkType(bias, ElemKind::FloatTy, parent);
    }
    // Quantization type check.
    if (src.getElementType() == ElemKind::Int8QTy) {
      isValid &= checkType(bias, ElemKind::Int32QTy, parent);
    }
  }
  ShapeNHWC idim(src.getType()->dims());
  ShapeNHWC odim(dest.getType()->dims());
  PaddingTLBR pdim(pads);
  ShapeHW kdim(kernels);
  isValid &= expectCompareTrue("buffer height too small for selected stride",
                               idim.h + pdim.top + pdim.bottom, kdim.height,
                               parent, CompareOperatorGreaterEqual<size_t>());
  isValid &= expectCompareTrue("buffer width too small for selected stride",
                               idim.w + pdim.left + pdim.right, kdim.width,
                               parent, CompareOperatorGreaterEqual<size_t>());
  isValid &= expectCompareTrue("channels number must be divisible by groups",
                               idim.c % group, size_t(0), parent);

  auto outSz = calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides,
                                           pads, dilation);
  isValid &=
      expectCompareTrue("Invalid output dimension N", odim.n, idim.n, parent);
  isValid &= expectCompareTrue("Invalid output dimension H", odim.h,
                               outSz.first, parent);
  isValid &= expectCompareTrue("Invalid output dimension W", odim.w,
                               outSz.second, parent);
  isValid &= expectCompareTrue("Invalid output dimension C", odim.c % group,
                               size_t(0), parent);

  const size_t filterDims[] = {odim.c, kdim.height, kdim.width,
                               idim.c / (size_t)group};
  isValid &=
      expectCompareTrue("Invalid filter dimensions", filter.getType()->dims(),
                        llvm::makeArrayRef(filterDims), parent);
  const size_t biasDims[] = {odim.c};
  isValid &=
      expectCompareTrue("Invalid bias dimensions", bias.getType()->dims(),
                        llvm::makeArrayRef(biasDims), parent);
  return isValid;
}

static bool verifyConvolution3D(NodeValue src, NodeValue dest, NodeValue filter,
                                NodeValue bias,
                                llvm::ArrayRef<unsigned_t> kernels,
                                llvm::ArrayRef<unsigned_t> strides,
                                llvm::ArrayRef<unsigned_t> pads,
                                unsigned_t group) {

  const Node *parent = dest.getNode();
  bool isValid = checkType(src, dest.getElementType(), parent);
  isValid &= checkType(src, filter.getElementType(), parent);
  // Non quantization type check.
  if (src.getElementType() == ElemKind::FloatTy) {
    isValid &= checkType(bias, ElemKind::FloatTy, parent);
  }
  // Quantization type check.
  if (src.getElementType() == ElemKind::Int8QTy) {
    isValid &= checkType(bias, ElemKind::Int32QTy, parent);
  }
  ShapeNHWDC idim(src.getType()->dims());
  ShapeNHWDC odim(dest.getType()->dims());
  PaddingTLNBRF pdim(pads);
  ShapeHWD kdim(kernels);
  isValid &= expectCompareTrue("buffer height too small for selected stride",
                               idim.h + pdim.top + pdim.bottom, kdim.height,
                               parent, CompareOperatorGreaterEqual<size_t>());
  isValid &= expectCompareTrue("buffer width too small for selected stride",
                               idim.w + pdim.left + pdim.right, kdim.width,
                               parent, CompareOperatorGreaterEqual<size_t>());
  isValid &= expectCompareTrue("buffer time too small for selected stride",
                               idim.d + pdim.near + pdim.far, kdim.depth,
                               parent, CompareOperatorGreaterEqual<size_t>());
  isValid &= expectCompareTrue("channels number must be divisible by groups",
                               idim.c % group, size_t(0), parent);

  auto outSz = calculate3DConvPoolOutputDims(idim.h, idim.w, idim.d, kernels,
                                             strides, pads);
  isValid &=
      expectCompareTrue("Invalid output dimension N", odim.n, idim.n, parent);
  isValid &= expectCompareTrue("Invalid output dimension H", odim.h,
                               outSz.height, parent);
  isValid &= expectCompareTrue("Invalid output dimension W", odim.w,
                               outSz.width, parent);
  isValid &= expectCompareTrue("Invalid output dimension D", odim.d,
                               outSz.depth, parent);
  isValid &= expectCompareTrue("Invalid output dimension C", odim.c % group,
                               size_t(0), parent);

  const size_t filterDims[] = {odim.c, kdim.height, kdim.width, kdim.depth,
                               idim.c / (size_t)group};
  isValid &=
      expectCompareTrue("Invalid filter dimensions", filter.getType()->dims(),
                        llvm::makeArrayRef(filterDims), parent);
  const size_t biasDims[] = {odim.c};
  isValid &=
      expectCompareTrue("Invalid bias dimensions", bias.getType()->dims(),
                        llvm::makeArrayRef(biasDims), parent);
  return isValid;
}

static bool verifyFullyConnected(NodeValue src, NodeValue weights,
                                 NodeValue bias, NodeValue dest) {
  const Node *parent = dest.getNode();
  bool isValid = expectCompareTrue("FC input must be 2D", size_t(2),
                                   src.dims().size(), parent);
  isValid &= expectCompareTrue("FC weights must be 2D", size_t(2),
                               weights.dims().size(), parent);
  isValid &= expectCompareTrue("Mismatch between source and dest dimensions",
                               src.dims()[0], dest.dims()[0], parent);
  isValid &= expectCompareTrue("Mismatch between source and weight dimensions",
                               src.dims()[1], weights.dims()[0], parent);
  isValid &= expectCompareTrue("Inconsistent bias/dest sizes", bias.dims()[0],
                               weights.dims()[1], parent);
  isValid &= expectCompareTrue("Inconsistent weights/dest sizes",
                               weights.dims()[1], dest.dims()[1], parent);

  if (src.getElementType() == ElemKind::Int8QTy) {
    isValid &= checkType(bias, ElemKind::Int32QTy, parent);
  }
  return isValid;
}

static bool verifyPool(NodeValue src, NodeValue dest,
                       llvm::ArrayRef<unsigned_t> kernels,
                       llvm::ArrayRef<unsigned_t> strides,
                       llvm::ArrayRef<unsigned_t> pads, bool isAvgPool = true) {
  const Node *parent = dest.getNode();
  ShapeNHWC idim = ShapeNHWC(src.getType()->dims());
  ShapeNHWC odim = ShapeNHWC(dest.getType()->dims());
  PaddingTLBR pdim(pads);
  ShapeHW kdim(kernels);

  bool isValid =
      expectCompareTrue("buffer height too small for selected stride",
                        idim.h + pdim.top + pdim.bottom, kdim.height, parent,
                        CompareOperatorGreaterEqual<size_t>());
  isValid &= expectCompareTrue("buffer width too small for selected stride",
                               idim.w + pdim.left + pdim.right, kdim.width,
                               parent, CompareOperatorGreaterEqual<size_t>());

  auto outSz =
      calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides, pads);
  ShapeNHWC exp(idim.n, outSz.first, outSz.second, idim.c);
  isValid &=
      expectCompareTrue("Unexpected output dimensions", exp, odim, parent);

  // For quantized AvgPool, the scale and offset of its input and output could
  // be different. But for quantized MaxPool, the scale and offset of its input
  // and output should be the same.
  isValid &= checkSameIsQuantized(src.getType(), dest.getType(), parent);
  if (!isAvgPool) {
    isValid &= checkTypeIgnoreShape(src, dest, parent);
  }
  return isValid;
}

static bool verifyBatchNormalization(NodeValue src, NodeValue dest,
                                     NodeValue bias, NodeValue scale,
                                     NodeValue mean, NodeValue var,
                                     unsigned_t channel) {
  const Node *parent = dest.getNode();
  bool isValid = checkSameType(dest, src, parent);

  // Figure out how many channels are in the tensor.
  size_t channels = src.dims()[channel];

  const size_t expArray[] = {channels};
  auto exp = llvm::makeArrayRef(expArray);
  isValid &= expectCompareTrue("Invalid bias dimension", bias.getType()->dims(),
                               exp, parent);
  isValid &= expectCompareTrue("Invalid scale dimension",
                               scale.getType()->dims(), exp, parent);
  isValid &= expectCompareTrue("Invalid mean dimension", mean.getType()->dims(),
                               exp, parent);
  isValid &= expectCompareTrue("Invalid var dimension", var.getType()->dims(),
                               exp, parent);
  return isValid;
}

static bool verifySigmoid(NodeValue src, NodeValue dest) {
  const Node *parent = dest.getNode();
  bool isValid = checkSameIsQuantized(src.getType(), dest.getType(), parent);
  if (src.getType()->isQuantizedType()) {
    isValid &= checkType(src, dest.getElementType(), dest.getNode());
    isValid &= checkSameShape(src, dest, parent);
  } else {
    isValid &= checkSameType(src, dest, parent);
  }
  return isValid;
}

static bool verifyTanh(NodeValue src, NodeValue dest) {
  const Node *parent = dest.getNode();
  bool isValid = checkSameIsQuantized(src.getType(), dest.getType(), parent);
  if (src.getType()->isQuantizedType()) {
    isValid &= checkType(src, dest.getElementType(), dest.getNode());
    isValid &= checkSameShape(src, dest, parent);
  } else {
    isValid &= checkSameType(src, dest, parent);
  }
  return isValid;
}

static bool verifySoftMax(NodeValue src, NodeValue dest) {
  const Node *parent = dest.getNode();
  if (src.getType()->isQuantizedType()) {
    return checkType(src, dest.getElementType(), parent) &&
           checkSameShape(src, dest, parent);
  }
  return checkSameType(src, dest, parent);
}

static bool verifyCrossEntropyLoss(NodeValue P, NodeValue CE,
                                   NodeValue labels) {
  const Node *parent = CE.getNode();
  bool isValid = checkType(P, CE.getElementType(), parent);
  isValid &= expectCompareTrue("Mismatching shape", P.dims()[0],
                               labels.dims()[0], parent);
  return isValid;
}

static bool verifyLocalResponseNormalization(NodeValue src, NodeValue dest) {
  return checkSameType(src, dest, dest.getNode());
}

static bool verifyArithmetic(NodeValue LHS, NodeValue RHS, NodeValue res) {
  return checkSameShape(res, LHS, res.getNode()) &&
         checkSameShape(LHS, RHS, res.getNode());
}

static bool verifyRelu(NodeValue result, NodeValue input) {
  const Node *parent = result.getNode();
  if (input.getType()->isQuantizedType()) {
    return checkSameIsQuantized(input.getType(), result.getType(), parent) &&
           checkSameShape(result, input, parent);
  }
  return checkSameType(result, input, parent);
}

static bool verifyPRelu(NodeValue result, NodeValue input, NodeValue slope) {
  const Node *parent = result.getNode();
  if (input.getType()->isQuantizedType()) {
    return checkSameIsQuantized(input.getType(), result.getType(), parent) &&
           checkSameIsQuantized(input.getType(), slope.getType(), parent) &&
           checkSameShape(result, input, parent) &&
           checkSameShape(slope, input, parent);
  }
  return checkSameType(result, input, parent) &&
         checkSameType(slope, input, parent) &&
         checkSameShape(slope, input, parent);
}

static bool verifyRegression(NodeValue src, NodeValue dest,
                             NodeValue expected) {
  return checkSameType(src, dest, dest.getNode()) &&
         checkSameType(dest, expected, dest.getNode());
}

static bool verifySparseLengthsSum(NodeValue dest, NodeValue data,
                                   NodeValue indices, NodeValue lengths) {
  bool isValid = checkType(dest, data.getElementType(), dest.getNode());
  isValid &= checkType(indices, ElemKind::Int64ITy, dest.getNode());
  isValid &= checkType(lengths, ElemKind::Int32ITy, dest.getNode());
  isValid &=
      expectCompareTrue("Indices must be a 1D vector", indices.dims().size(),
                        size_t(1), dest.getNode());
  isValid &=
      expectCompareTrue("Lengths must be a 1D vector", lengths.dims().size(),
                        size_t(1), dest.getNode());
  return isValid;
}

static bool verifySparseLengthsWeightedSum(NodeValue dest, NodeValue data,
                                           NodeValue weights, NodeValue indices,
                                           NodeValue lengths) {
  bool isValid = checkType(dest, data.getElementType(), dest.getNode());
  isValid &= checkType(weights, data.getElementType(), dest.getNode());
  isValid &= checkType(indices, ElemKind::Int64ITy, dest.getNode());
  isValid &= checkType(lengths, ElemKind::Int32ITy, dest.getNode());
  isValid &=
      expectCompareTrue("Indices must be a 1D vector", indices.dims().size(),
                        size_t(1), dest.getNode());
  isValid &=
      expectCompareTrue("Lengths must be a 1D vector", lengths.dims().size(),
                        size_t(1), dest.getNode());
  isValid &=
      expectCompareTrue("Weights must be a 1D vector", weights.dims().size(),
                        size_t(1), dest.getNode());

  isValid &=
      expectCompareTrue("Weights and Indices must have the same size",
                        weights.dims()[0], indices.dims()[0], dest.getNode());
  return isValid;
}

bool PadNode::verify() const {
  // Pad is currently only supported for constant padding.
  return expectCompareTrue("only the 'constant' mode is currrently supported",
                           getMode() == PaddingMode::CONSTANT, true,
                           getResult().getNode());
}

bool ConvolutionNode::verify() const {
  return verifyConvolution(getInput(), getResult(), getFilter(), getBias(),
                           Kernels_, Strides_, Pads_, Group_, Dilation_);
}

bool ChannelwiseQuantizedConvolutionNode::verify() const {
  bool isValid = expectCompareTrue("Only groupwise quantization is supported.",
                                   getGroupwise(), true, this);

  if (!isValid) {
    return false;
  }

  isValid = verifyConvolution(getInput(), getResult(), getFilter(), getBias(),
                              Kernels_, Strides_, Pads_, Group_,
                              /* dilation */ 1, /* checkBiasType */ false);

  isValid &= checkType(getBias(), ElemKind::FloatTy, this);
  isValid &= checkType(getInput(), ElemKind::Int8QTy, this);

  // check qparam types
  isValid &= checkType(getOffsets(), ElemKind::Int32ITy, this);
  isValid &= checkType(getScales(), ElemKind::FloatTy, this);

  // check qparam dimensions
  isValid &= expectCompareTrue("Offsets must be a 1D vector",
                               getOffsets().dims().size(), size_t(1), this);
  isValid &= expectCompareTrue("Scales must be a 1D vector",
                               getScales().dims().size(), size_t(1), this);

  // check qparam sizes
  isValid &=
      expectCompareTrue("There must be one filter offset qparam per group",
                        getOffsets().dims()[0], size_t(getGroup()), this);
  isValid &=
      expectCompareTrue("There must be one filter scale qparam per group",
                        getScales().dims()[0], size_t(getGroup()), this);
  return isValid;
}

bool Convolution3DNode::verify() const {
  return verifyConvolution3D(getInput(), getResult(), getFilter(), getBias(),
                             Kernels_, Strides_, Pads_, Group_);
}

/// Verify that types of an input and its gradient are the same.
static bool verifyInputAndGradInputTypes(NodeValue input, NodeValue gradInput,
                                         const Node *parent) {
  return checkSameType(input, gradInput, parent);
}

/// Verify that types of an output and its gradient are the same.
static bool verifyOutputAndGradOutputTypes(NodeValue output,
                                           NodeValue gradOutput,
                                           const Node *parent) {
  return checkSameType(output, gradOutput, parent);
}

bool Constant::verify() const {
  return expectCompareTrue("Underlying tensor type doesn't match constant type",
                           *getType(), getPayload().getType(), this);
}

bool ConvolutionGradNode::verify() const {
  bool isValid = verifyInputAndGradInputTypes(getInput(),
                                              getGradOfInputNamedInput(), this);
  isValid &= verifyInputAndGradInputTypes(getFilter(),
                                          getGradOfInputNamedFilter(), this);
  isValid &=
      verifyInputAndGradInputTypes(getBias(), getGradOfInputNamedBias(), this);
  isValid &= verifyOutputAndGradOutputTypes(
      getOriginalOutputForResult(), getGradOfOriginalOutputNamedResult(), this);
  isValid &= verifyConvolution(
      getGradOfInputNamedInput(), getGradOfOriginalOutputNamedResult(),
      getGradOfInputNamedFilter(), getGradOfInputNamedBias(), Kernels_,
      Strides_, Pads_, Group_, Dilation_);
  return isValid;
}

bool Convolution3DGradNode::verify() const {
  bool isValid = verifyInputAndGradInputTypes(getInput(),
                                              getGradOfInputNamedInput(), this);
  isValid &= verifyInputAndGradInputTypes(getFilter(),
                                          getGradOfInputNamedFilter(), this);
  isValid &=
      verifyInputAndGradInputTypes(getBias(), getGradOfInputNamedBias(), this);
  isValid &= verifyOutputAndGradOutputTypes(
      getOriginalOutputForResult(), getGradOfOriginalOutputNamedResult(), this);
  isValid &= verifyConvolution3D(
      getGradOfInputNamedInput(), getGradOfOriginalOutputNamedResult(),
      getGradOfInputNamedFilter(), getGradOfInputNamedBias(), Kernels_,
      Strides_, Pads_, Group_);
  return isValid;
}

bool ConvertToNode::verify() const {
  bool isValid = checkSameShape(getInput(), getResult(), this);
  TypeRef srcTy = getInput().getType();
  TypeRef dstTy = getResult().getType();
  isValid &= expectCompareTrue(
      "Quantized conversion should use Dequantize, Quantize and Rescale",
      srcTy->isQuantizedType() || dstTy->isQuantizedType(), false, this);
  return isValid;
}

bool MaxPoolNode::verify() const {
  return verifyPool(getInput(), getResult(), Kernels_, Strides_, Pads_,
                    /* isAvgPool */ false);
}

bool AvgPoolNode::verify() const {
  return verifyPool(getInput(), getResult(), Kernels_, Strides_, Pads_);
}

bool MaxPoolGradNode::verify() const {
  bool isValid = verifyInputAndGradInputTypes(getInput(),
                                              getGradOfInputNamedInput(), this);
  isValid &= verifyOutputAndGradOutputTypes(
      getOriginalOutputForResult(), getGradOfOriginalOutputNamedResult(), this);
  isValid &= verifyPool(getGradOfInputNamedInput(),
                        getGradOfOriginalOutputNamedResult(), Kernels_,
                        Strides_, Pads_, /* isAvgPool */ false);
  return isValid;
}

bool AvgPoolGradNode::verify() const {
  bool isValid = verifyInputAndGradInputTypes(getInput(),
                                              getGradOfInputNamedInput(), this);
  isValid &= verifyOutputAndGradOutputTypes(
      getOriginalOutputForResult(), getGradOfOriginalOutputNamedResult(), this);
  isValid &= verifyPool(getGradOfInputNamedInput(),
                        getGradOfOriginalOutputNamedResult(), Kernels_,
                        Strides_, Pads_);
  return isValid;
}

bool MatMulNode::verify() const {
  auto lhs = getLHS();
  auto rhs = getRHS();
  auto dest = getResult();

  auto LDims = lhs.dims();
  auto RDims = rhs.dims();
  auto DDims = dest.dims();
  bool isValid = expectCompareTrue("Invalid MatMul dimensions", size_t(2),
                                   DDims.size(), this);

  auto elem = dest.getType()->getElementType();
  isValid &= checkType(lhs, elem, this);
  isValid &= checkType(rhs, elem, this);

  isValid &=
      expectCompareTrue("Invalid row dimensions", LDims[0], DDims[0], this);
  isValid &=
      expectCompareTrue("Invalid column dimensions", RDims[1], DDims[1], this);
  return isValid;
}

bool BatchMatMulNode::verify() const {
  auto LHS = getLHS();
  auto RHS = getRHS();
  auto dest = getResult();

  bool isValid = expectCompareTrue("LHS input must be 3 dimensional.",
                                   LHS.dims().size(), size_t(3), this);
  isValid &= expectCompareTrue("RHS input must be 3 dimensional.",
                               RHS.dims().size(), size_t(3), this);
  isValid &= expectCompareTrue("Result must be 3 dimensional.",
                               dest.dims().size(), size_t(3), this);
  isValid &= expectCompareTrue("LHS and RHS inputs must have same batch size.",
                               LHS.dims()[0], RHS.dims()[0], this);
  isValid &= expectCompareTrue("Result must have same batch size as inputs.",
                               LHS.dims()[0], dest.dims()[0], this);

  const size_t numBatches = LHS.dims()[0];
  const size_t N = LHS.dims()[1];
  const size_t M = LHS.dims()[2];
  const size_t P = RHS.dims()[2];
  isValid &= expectCompareTrue("Inputs have invalid dimensions.", RHS.dims()[1],
                               M, this);
  isValid &= expectCompareTrue("Result has invalid dimensions given inputs.",
                               dest.dims(), {numBatches, N, P}, this);

  auto elemType = dest.getType()->getElementType();
  isValid &= checkType(LHS, elemType, this);
  isValid &= checkType(RHS, elemType, this);

  return isValid;
}

bool SigmoidNode::verify() const {
  return verifySigmoid(getInput(), getResult());
}

bool SigmoidGradNode::verify() const {
  bool isValid = verifyInputAndGradInputTypes(getInput(),
                                              getGradOfInputNamedInput(), this);
  isValid &= verifyOutputAndGradOutputTypes(
      getOriginalOutputForResult(), getGradOfOriginalOutputNamedResult(), this);
  isValid &= verifySigmoid(getGradOfInputNamedInput(),
                           getGradOfOriginalOutputNamedResult());
  return isValid;
}

bool TanhNode::verify() const { return verifyTanh(getInput(), getResult()); }

bool TanhGradNode::verify() const {
  bool isValid = verifyInputAndGradInputTypes(getInput(),
                                              getGradOfInputNamedInput(), this);
  isValid &= verifyOutputAndGradOutputTypes(
      getOriginalOutputForResult(), getGradOfOriginalOutputNamedResult(), this);
  isValid &= verifyTanh(getGradOfInputNamedInput(),
                        getGradOfOriginalOutputNamedResult());
  return isValid;
}

bool ExpNode::verify() const {
  const Node *parent = getResult().getNode();
  bool isValid =
      checkSameIsQuantized(getInput().getType(), getResult().getType(), parent);
  if (getInput().getType()->isQuantizedType()) {
    return false;
  }
  isValid &= checkSameType(getInput(), getResult(), parent);
  isValid &= checkSameShape(getInput(), getResult(), parent);
  return isValid;
}

bool SoftMaxNode::verify() const {
  return verifySoftMax(getInput(), getResult());
}

bool SoftMaxGradNode::verify() const {
  bool isValid = verifyInputAndGradInputTypes(getInput(),
                                              getGradOfInputNamedInput(), this);
  isValid &= verifyInputAndGradInputTypes(getSelected(),
                                          getGradOfInputNamedSelected(), this);
  isValid &= verifyOutputAndGradOutputTypes(
      getOriginalOutputForResult(), getGradOfOriginalOutputNamedResult(), this);
  isValid &= verifySoftMax(getGradOfInputNamedInput(),
                           getGradOfOriginalOutputNamedResult());
  return isValid;
}

bool CrossEntropyLossNode::verify() const {
  return verifyCrossEntropyLoss(getP(), getCE(), getLabels());
}

bool CrossEntropyLossGradNode::verify() const {
  bool isValid = verifyInputAndGradInputTypes(
      getLabels(), getGradOfInputNamedLabels(), this);
  isValid &= verifyInputAndGradInputTypes(getP(), getGradOfInputNamedP(), this);
  isValid &= verifyOutputAndGradOutputTypes(
      getOriginalOutputForCE(), getGradOfOriginalOutputNamedCE(), this);
  isValid &= verifyCrossEntropyLoss(getGradOfInputNamedP(),
                                    getGradOfOriginalOutputNamedCE(),
                                    getGradOfInputNamedLabels());
  return isValid;
}

bool ReshapeNode::verify() const {
  bool isValid = expectCompareTrue("Reshape into a different size",
                                   getResult().getType()->size(),
                                   getInput().getType()->size(), this);
  isValid &= checkTypeIgnoreShape(getResult(), getInput(), this);
  return isValid;
}

bool TransposeNode::verify() const {
  auto dest = getResult();
  auto src = getInput();
  ShapeVector shape;

  auto dims = src.dims();
  for (size_t i = 0; i < dims.size(); i++) {
    shape.push_back(dims[Shuffle_[i]]);
  }

  bool isValid = expectCompareTrue("Invalid transpose dims", dest.dims(),
                                   llvm::makeArrayRef(shape), this);
  isValid &= checkTypeIgnoreShape(dest, src, this);
  return isValid;
}

bool ChannelShuffleNode::verify() const {
  bool isValid = expectCompareTrue("Channel shuffle into a different size.",
                                   getResult().getType()->size(),
                                   getInput().getType()->size(), this);
  isValid &= checkTypeIgnoreShape(getResult(), getInput(), this);
  return isValid;
}

bool SplatNode::verify() const { return true; }

bool TraceEventNode::verify() const { return true; }

bool InsertTensorNode::verify() const {
  auto dest = getBig();
  auto src = getSmall();
  auto offsets = getStart();
  size_t numDims = dest.dims().size();
  size_t axis = getAxis();
  size_t count = getCount();

  bool isValid = expectCompareTrue("Invalid number of dimensions", numDims,
                                   src.dims().size(), this);
  isValid &= expectCompareTrue("Invalid number of dimensions for offsets",
                               numDims, offsets.size(), this);

  if (!isValid) {
    // The following loop may be out-of-bound if the previous
    // comparisons failed.
    return false;
  }

  isValid &= checkType(dest, src.getType()->getElementType(), this);
  if (dest.getType()->isQuantizedType()) {
    isValid &= expectCompareTrue("Scales of Big and Small must match.",
                                 src.getType()->getScale(),
                                 dest.getType()->getScale(), this);
    isValid &= expectCompareTrue("Offsets of Big and Small must match.",
                                 src.getType()->getOffset(),
                                 dest.getType()->getOffset(), this);
  }

  for (unsigned i = 0; i < numDims; i++) {
    // TODO: We could come up with a mechanism to lazy compute that
    // string since it is going to be used only in case of an error.
    // However, this function is not performance critical so leave it
    // this way for now.
    std::string msg = std::to_string(i);
    msg = "out of bounds for index " + msg;
    isValid &= expectCompareTrue(msg.c_str(), src.dims()[i] + offsets[i],
                                 dest.dims()[i], this,
                                 CompareOperatorLessEqual<size_t>());
  }

  isValid &= expectCompareTrue("Invalid axis", axis, src.dims().size(), this,
                               CompareOperatorLessEqual<size_t>());
  for (size_t i = 0; i < src.dims().size(); i++) {
    size_t mul = (i == axis) ? count : 1;
    std::string msg = std::to_string(i);
    msg = "Small does not fit inside Big for index " + msg;
    isValid &=
        expectCompareTrue(msg.c_str(), src.dims()[i] * mul, dest.dims()[i],
                          this, CompareOperatorLessEqual<size_t>());
  }
  return isValid;
}

bool SliceNode::verify() const {
  auto dest = getResult();
  auto src = getInput();
  auto offsets = getStart();
  size_t numDims = dest.dims().size();
  bool isValid = expectCompareTrue("Invalid number of dimensions", numDims,
                                   src.dims().size(), this);
  isValid &= expectCompareTrue("Invalid number of dimensions", numDims,
                               offsets.size(), this);

  if (!isValid) {
    // The following loop may be out-of-bound if the previous
    // comparisons failed.
    return false;
  }

  for (unsigned i = 0; i < numDims; i++) {
    std::string msg = std::to_string(i);
    msg = "out of bounds for index " + msg;
    isValid &= expectCompareTrue(msg.c_str(), dest.dims()[i] + offsets[i],
                                 src.dims()[i], this,
                                 CompareOperatorLessEqual<size_t>());
  }
  isValid &= checkNotQuantizedOrSameParams(dest.getType(), src.getType(), this);
  return isValid;
}

bool TileNode::verify() const {
  auto dest = getResult();
  auto src = getInput();
  size_t axis = getAxis();
  unsigned count = getCount();

  bool isValid = expectCompareTrue("Invalid axis", axis, src.dims().size(),
                                   this, CompareOperatorLessEqual<size_t>());

  for (size_t i = 0; i < src.dims().size(); i++) {
    size_t mul = (i == axis) ? count : 1;
    std::string msg = std::to_string(i);
    msg = "Incorrect output shape for dim " + msg;
    isValid &= expectCompareTrue(msg.c_str(), src.dims()[i] * mul,
                                 dest.dims()[i], this);
  }
  return isValid;
}

bool BatchNormalizationNode::verify() const {
  return verifyBatchNormalization(getInput(), getResult(), getBias(),
                                  getScale(), getMean(), getVar(), ChannelIdx_);
}

bool BatchNormalizationGradNode::verify() const {
  bool isValid =
      verifyInputAndGradInputTypes(getBias(), getGradOfInputNamedBias(), this);
  isValid &= verifyInputAndGradInputTypes(getInput(),
                                          getGradOfInputNamedInput(), this);
  isValid &=
      verifyInputAndGradInputTypes(getMean(), getGradOfInputNamedMean(), this);
  isValid &= verifyInputAndGradInputTypes(getScale(),
                                          getGradOfInputNamedScale(), this);
  isValid &=
      verifyInputAndGradInputTypes(getVar(), getGradOfInputNamedVar(), this);
  isValid &= verifyOutputAndGradOutputTypes(
      getOriginalOutputForResult(), getGradOfOriginalOutputNamedResult(), this);
  isValid &= verifyBatchNormalization(
      getGradOfInputNamedInput(), getGradOfOriginalOutputNamedResult(),
      getGradOfInputNamedBias(), getGradOfInputNamedScale(),
      getGradOfInputNamedMean(), getGradOfInputNamedVar(), ChannelIdx_);
  return isValid;
}

bool MeanVarNormalizationNode::verify() const {
  return checkType(getMean(), ElemKind::FloatTy, this) &&
         checkSameType(getMean(), getVar(), this);
}

bool LocalResponseNormalizationNode::verify() const {
  return verifyLocalResponseNormalization(getInput(), getResult());
}

bool LocalResponseNormalizationGradNode::verify() const {
  bool isValid = verifyInputAndGradInputTypes(getInput(),
                                              getGradOfInputNamedInput(), this);
  isValid &= verifyOutputAndGradOutputTypes(
      getOriginalOutputForResult(), getGradOfOriginalOutputNamedResult(), this);
  isValid &= verifyLocalResponseNormalization(
      getGradOfInputNamedInput(), getGradOfOriginalOutputNamedResult());
  return isValid;
}

#define VERIFY_ARITHMETIC(NODE_NAME_)                                          \
  bool NODE_NAME_##Node::verify() const {                                      \
    return verifyArithmetic(getLHS(), getRHS(), getResult());                  \
  }
VERIFY_ARITHMETIC(Add);
VERIFY_ARITHMETIC(Mul);
VERIFY_ARITHMETIC(Sub);
VERIFY_ARITHMETIC(Div);
VERIFY_ARITHMETIC(Max);
VERIFY_ARITHMETIC(Min);
VERIFY_ARITHMETIC(Pow);
#undef VERIFY_ARITHMETIC

#define VERIFY_ARITHMETIC(NODE_NAME_)                                          \
  bool NODE_NAME_##Node::verify() const {                                      \
    bool isValid = verifyInputAndGradInputTypes(                               \
        getLHS(), getGradOfInputNamedLHS(), this);                             \
    isValid &= verifyInputAndGradInputTypes(getRHS(),                          \
                                            getGradOfInputNamedRHS(), this);   \
    isValid &= verifyOutputAndGradOutputTypes(                                 \
        getOriginalOutputForResult(), getGradOfOriginalOutputNamedResult(),    \
        this);                                                                 \
    isValid &=                                                                 \
        verifyArithmetic(getGradOfInputNamedLHS(), getGradOfInputNamedRHS(),   \
                         getGradOfOriginalOutputNamedResult());                \
    return isValid;                                                            \
  }
VERIFY_ARITHMETIC(AddGrad);
VERIFY_ARITHMETIC(MulGrad);
VERIFY_ARITHMETIC(SubGrad);
VERIFY_ARITHMETIC(DivGrad);
#undef VERIFY_ARITHMETIC

#define VERIFY_CMP(NODE_NAME_)                                                 \
  bool NODE_NAME_##Node::verify() const {                                      \
    bool isValid = checkSameShape(getLHS(), getRHS(), this);                   \
    isValid &= checkSameShape(getResult(), getLHS(), this);                    \
    isValid &= checkType(getLHS(), getRHS().getElementType(), this);           \
    isValid &= checkType(getResult(), ElemKind::BoolTy, this);                 \
    return isValid;                                                            \
  }

VERIFY_CMP(CmpLTE)
VERIFY_CMP(CmpEQ)
#undef VERIFY_CMP

bool BatchedAddNode::verify() const {
  auto batchShape = getBatch().dims();
  auto rhsShape = getSlice().dims();
  bool isValid = expectCompareTrue("Invalid shape", batchShape.drop_front(),
                                   rhsShape, this);
  isValid &= checkSameShape(getBatch(), getResult(), this);

  if (getBatch().getType()->isQuantizedType()) {
    expectCompareTrue("Mismatched slice element types",
                      getSlice().getType()->isQuantizedType(), true, this);
  } else {
    isValid &=
        checkType(getBatch(), getSlice().getType()->getElementType(), this);
  }
  return isValid;
}

bool BatchedReduceAddNode::verify() const {
  bool isValid = checkType(getResult(), getBatch().getElementType(), this);

  isValid &=
      expectCompareTrue("Invalid shape", getBatch().dims().size(), size_t(0),
                        this, CompareOperatorGreaterThan<size_t>());
  return isValid;
}

bool LengthsSumNode::verify() const {
  return expectCompareTrue("Lengths must be a 1D vector",
                           getLengths().dims().size(), size_t(1), this);
}

bool BatchedReduceMeanNode::verify() const {
  bool isValid = checkType(getResult(), getBatch().getElementType(), this);

  isValid &=
      expectCompareTrue("Invalid shape", getBatch().dims().size(), size_t(0),
                        this, CompareOperatorGreaterThan<size_t>());
  return isValid;
}

bool SparseLengthsSumNode::verify() const {
  return verifySparseLengthsSum(getResult(), getData(), getIndices(),
                                getLengths());
}

bool SparseLengthsWeightedSumNode::verify() const {
  return verifySparseLengthsWeightedSum(getResult(), getData(), getWeights(),
                                        getIndices(), getLengths());
}

bool SparseLengthsWeightedSumGradNode::verify() const {
  // Same checks as SparseLengthsWeightedSumNode.
  bool isValid =
      verifySparseLengthsWeightedSum(getOriginalOutputForResult(), getData(),
                                     getWeights(), getIndices(), getLengths());

  // Checks on gradient inputs/outputs.
  isValid &= checkSameType(getGradOfOriginalOutputNamedResult(),
                           getOriginalOutputForResult(), this);
  isValid &= checkSameType(getGradOfInputNamedData(), getData(), this);
  isValid &= checkSameType(getGradOfInputNamedWeights(), getWeights(), this);
  isValid &= checkSameType(getGradOfInputNamedIndices(), getIndices(), this);
  isValid &= checkSameType(getGradOfInputNamedLengths(), getLengths(), this);
  return isValid;
}

bool RowwiseQuantizedSparseLengthsWeightedSumNode::verify() const {
  bool isValid = checkType(getResult(), ElemKind::FloatTy, this);
  isValid &= checkType(getData(), ElemKind::Int8QTy, this);
  isValid &= checkType(getScales(), ElemKind::FloatTy, this);
  isValid &= checkType(getOffsets(), ElemKind::FloatTy, this);
  isValid &= checkType(getWeights(), ElemKind::FloatTy, this);
  isValid &= checkType(getIndices(), ElemKind::Int64ITy, this);
  isValid &= checkType(getLengths(), ElemKind::Int32ITy, this);
  isValid &= expectCompareTrue("Indices must be a 1D vector",
                               getIndices().dims().size(), size_t(1), this);
  isValid &= expectCompareTrue("Lengths must be a 1D vector",
                               getLengths().dims().size(), size_t(1), this);
  isValid &= expectCompareTrue("Weights must be a 1D vector",
                               getWeights().dims().size(), size_t(1), this);
  isValid &= expectCompareTrue("Scales must be a 1D vector",
                               getScales().dims().size(), size_t(1), this);
  isValid &= expectCompareTrue("Offsets must be a 1D vector",
                               getOffsets().dims().size(), size_t(1), this);
  isValid &=
      expectCompareTrue("Weights and Indices must have the same size",
                        getWeights().dims()[0], getIndices().dims()[0], this);
  isValid &= expectCompareTrue(
      "Scales and Data must have the same first dimension size",
      getData().dims()[0], getScales().dims()[0], this);
  isValid &= expectCompareTrue(
      "Offsets and Data must have the same first dimension size",
      getData().dims()[0], getOffsets().dims()[0], this);
  return isValid;
}

static bool verifyFusedRowwiseQuantizedSparseLengthsSum(NodeValue result,
                                                        NodeValue data,
                                                        NodeValue indices,
                                                        NodeValue lengths,
                                                        NodeValue weights) {
  const Node *parent = result.getNode();
  bool isValid = checkType(result, ElemKind::FloatTy, parent);
  isValid &= checkType(data, ElemKind::UInt8FusedQTy, parent);
  isValid &= checkType(indices, ElemKind::Int64ITy, parent);
  isValid &= checkType(lengths, ElemKind::Int32ITy, parent);
  isValid &= expectCompareTrue("Indices must be a 1D vector",
                               indices.dims().size(), size_t(1), parent);
  isValid &= expectCompareTrue("Lengths must be a 1D vector",
                               lengths.dims().size(), size_t(1), parent);
  isValid &= expectCompareTrue("Data must be 2 dimensional.",
                               data.dims().size(), size_t(2), parent);
  isValid &= expectCompareTrue("Data must have more than 8 columns.",
                               data.dims()[1], size_t(8), parent,
                               CompareOperatorGreaterEqual<size_t>());
  isValid &= expectCompareTrue("Result must be 2 dimensional.",
                               result.dims().size(), size_t(2), parent);

  if (weights.getNode()) {
    isValid &= checkType(weights, ElemKind::FloatTy, parent);
    isValid &= expectCompareTrue("Weights must be a 1D vector",
                                 weights.dims().size(), size_t(1), parent);
    isValid &= expectCompareTrue("Weights and Indices must have the same size",
                                 weights.dims()[0], indices.dims()[0], parent);
  }

  // Wrap this in isValid to prevent potential segfault if the result is
  // incorrectly shaped.
  if (isValid) {
    isValid &= expectCompareTrue(
        "Result output shape should have second dim as 8 less than Data.",
        result.dims()[1] + 8, data.dims()[1], parent);
  }
  return isValid;
}

bool FusedRowwiseQuantizedSparseLengthsWeightedSumNode::verify() const {
  return verifyFusedRowwiseQuantizedSparseLengthsSum(
      getResult(), getData(), getIndices(), getLengths(), getWeights());
}

bool FusedRowwiseQuantizedSparseLengthsSumNode::verify() const {
  return verifyFusedRowwiseQuantizedSparseLengthsSum(
      getResult(), getData(), getIndices(), getLengths(), nullptr);
}

bool LengthsToRangesNode::verify() const {
  bool isValid = checkType(getResult(), getLengths().getElementType(), this);
  isValid &= checkType(getLengths(), ElemKind::Int32ITy, this);
  isValid &= expectCompareTrue("Lengths must be a 1D vector",
                               getLengths().dims().size(), size_t(1), this);
  isValid &= expectCompareTrue("Ranges must be a 2D vector",
                               getResult().dims().size(), size_t(2), this);
  isValid &= expectCompareTrue(
      "Lengths and Ranges must have the same outer dimensions",
      getResult().dims()[0], getLengths().dims()[0], this);
  isValid &= expectCompareTrue("Inner dimension of Ranges must be 2",
                               getResult().dims()[1], size_t(2), this);
  return isValid;
}

bool LengthsRangeFillNode::verify() const {
  bool isValid = checkType(getLengths(), ElemKind::Int32ITy, this);
  isValid &= checkType(getResult(), getLengths().getElementType(), this);
  isValid &= expectCompareTrue("Lengths must be a 1D vector",
                               getLengths().dims().size(), size_t(1), this);
  isValid &= expectCompareTrue("Result must be a 1D vector",
                               getResult().dims().size(), size_t(1), this);
  return isValid;
}

bool SparseToDenseNode::verify() const {
  bool isValid = checkType(getResult(), getValues().getElementType(), this);
  isValid &= checkType(getIndices(), ElemKind::Int64ITy, this);
  isValid &= expectCompareTrue("Indices must be a 1D vector",
                               getIndices().dims().size(), size_t(1), this);
  isValid &=
      expectCompareTrue("Indices and Values must have the same first dimension",
                        getIndices().dims()[0], getValues().dims()[0], this);
  return isValid;
}

bool SparseToDenseMaskNode::verify() const {
  bool isValid = checkType(getResult(), getValues().getElementType(), this);
  isValid &= checkType(getResult(), getDefaultValue().getElementType(), this);
  isValid &= checkType(getIndices(), ElemKind::Int64ITy, this);
  isValid &= checkType(getLengths(), ElemKind::Int32ITy, this);
  isValid &= expectCompareTrue("Indices must be a 1D vector",
                               getIndices().dims().size(), size_t(1), this);
  isValid &= expectCompareTrue("Lengths must be a scalar or 1D vector",
                               getLengths().dims().size(), {0, 1}, this);
  isValid &=
      expectCompareTrue("Indices and Values must have the same first dimension",
                        getIndices().dims()[0], getValues().dims()[0], this);
  isValid &= expectCompareTrue(
      "Values[i] must have the same dimensions as DefaultValue",
      getValues().dims().slice(1), getDefaultValue().dims(), this);
  return isValid;
}

bool SGDNode::verify() const {
  return checkSameType(getGradient(), getWeight(), this);
}

bool QuantizationProfileNode::verify() const {
  // Make sure that input tensor is a floating point type.
  bool isValid = checkType(getInput(), ElemKind::FloatTy, this);

  // Check computation info has proper size.
  isValid &=
      expectCompareTrue("Computation info should be 1 dimensional",
                        getComputationInfo().dims().size(), size_t(1), this);
  isValid &= expectCompareTrue(
      "Computation info should contain Min and Max value only",
      getComputationInfo().dims()[0], size_t(2), this);
  return isValid;
}

bool IntLookupTableNode::verify() const {
  bool isValid =
      expectCompareTrue("Input should be quantized type",
                        getInput().getType()->isQuantizedType(), true, this);
  isValid &=
      expectCompareTrue("Result should be quantized type",
                        getResult().getType()->isQuantizedType(), true, this);
  isValid &= expectCompareTrue("Mapping should be 1 dimensional",
                               getMapping().dims().size(), size_t(1), this);
  isValid &= expectCompareTrue(
      "Mapping should cover whole quantized range", getMapping().dims()[0],
      (size_t)(256 * getResult().getType()->getElementSize()), this);
  return isValid;
}

bool QuantizeNode::verify() const {
  bool isValid =
      expectCompareTrue("Dest must be quantized",
                        getResult().getType()->isQuantizedType(), true, this);
  isValid &= expectCompareTrue("Src must be an FP type",
                               getInput().getType()->isFPType(), true, this);
  isValid &= checkSameShape(getResult(), getInput(), this);
  return isValid;
}

bool DequantizeNode::verify() const {
  bool isValid = expectCompareTrue(
      "Dest must be an FP type", getResult().getType()->isFPType(), true, this);
  isValid &=
      expectCompareTrue("Src must be quantized",
                        getInput().getType()->isQuantizedType(), true, this);
  isValid &= checkSameShape(getResult(), getInput(), this);
  return isValid;
}

bool RescaleQuantizedNode::verify() const {
  bool isValid =
      expectCompareTrue("Dest must be quantized",
                        getResult().getType()->isQuantizedType(), true, this);
  isValid &=
      checkType(getResult(), getInput().getType()->getElementType(), this);
  isValid &= checkSameShape(getResult(), getInput(), this);
  return isValid;
}

bool TopKNode::verify() const {
  bool isValid = checkSameShape(getValues(), getIndices(), this);
  isValid &= checkNotQuantizedOrSameParams(getInput().getType(),
                                           getValues().getType(), this);
  return isValid;
}

bool RowwiseQuantizedFullyConnectedNode::verify() const {
  auto src = getInput();
  auto weights = getWeights();
  auto scales = getScales();
  auto offsets = getOffsets();
  auto bias = getBias();
  auto dest = getResult();

  bool isValid = expectCompareTrue("Inputs should be 2D tensor",
                                   src.dims().size(), size_t(2), this);
  isValid &= expectCompareTrue("Weights should be 2D tensor",
                               weights.dims().size(), size_t(2), this);
  isValid &= expectCompareTrue("Result should be 2D tensor", dest.dims().size(),
                               size_t(2), this);
  isValid &= expectCompareTrue("Offsets should be 1D tensor",
                               offsets.dims().size(), size_t(1), this);
  isValid &= expectCompareTrue("Scales should be 1D tensor",
                               scales.dims().size(), size_t(1), this);
  isValid &= expectCompareTrue("Bias should be 1D tensor", bias.dims().size(),
                               size_t(1), this);

  isValid &= expectCompareTrue("Mismatch on expected source dimension 0",
                               src.dims()[0], dest.dims()[0], this);
  isValid &= expectCompareTrue("Mismatch on expected source dimension 1",
                               src.dims()[1], weights.dims()[1], this);

  isValid &= expectCompareTrue("Inconsistent bias/dest sizes", bias.dims()[0],
                               weights.dims()[0], this);
  isValid &= expectCompareTrue("Inconsistent weights/dest sizes",
                               weights.dims()[0], dest.dims()[1], this);

  isValid &= expectCompareTrue("Inconsistent scales/offsets sizes",
                               scales.dims()[0], offsets.dims()[0], this);
  isValid &= expectCompareTrue("Inconsistent scales/weights sizes",
                               scales.dims()[0], weights.dims()[0], this);
  return isValid;
}

bool GatherNode::verify() const {
  bool isValid = checkType(getResult(), getData().getElementType(), this);
  isValid &= checkType(
      getIndices(),
      llvm::ArrayRef<ElemKind>({ElemKind::Int64ITy, ElemKind::Int32ITy}), this);
  isValid &= expectCompareTrue(
      "Mismatching number of dimensions", getResult().dims().size(),
      getData().dims().size() + getIndices().dims().size() - 1, this);
  isValid &= checkNotQuantizedOrSameParams(getResult().getType(),
                                           getData().getType(), this);
  return isValid;
}

bool GatherRangesNode::verify() const {
  bool isValid = expectCompareTrue("Data must be 1D", getData().dims().size(),
                                   size_t(1), this);
  isValid &= expectCompareTrue("Ranges must be 3D", getRanges().dims().size(),
                               size_t(3), this);
  isValid &= expectCompareTrue("Last dimension of Ranges must be equal to 2",
                               getRanges().dims()[2], size_t(2), this);
  isValid &= expectCompareTrue("Output must be 1D", getOutput().dims().size(),
                               size_t(1), this);
  isValid &= expectCompareTrue("Lengths must be 1D", getLengths().dims().size(),
                               size_t(1), this);
  isValid &=
      expectCompareTrue("Number of examples must match number of lengths",
                        getRanges().dims()[0], getLengths().dims()[0], this);

  isValid &= checkTypeIgnoreShape(getOutput(), getData(), this);
  isValid &= checkTypeIgnoreShape(getRanges(), getLengths(), this);

  return isValid;
}

bool ScatterAssignNode::verify() const {
  const auto &slicesDims = getSlices().dims();
  const auto &dataDims = getData().dims();
  const auto &indicesDims = getIndices().dims();

  bool isValid =
      expectCompareTrue("Indices should be a single dimensional vector",
                        indicesDims.size(), size_t(1), this);
  isValid &= expectCompareTrue("There should be an index for each slice",
                               indicesDims[0], slicesDims[0], this);
  isValid &=
      expectCompareTrue("Slices and data should have same number of dimensions",
                        slicesDims.size(), dataDims.size(), this);

  if (dataDims.size() > 1) {
    if (!isValid) {
      // The following loop may be out-of-bound if the previous
      // comparisons failed.
      return false;
    }

    for (size_t i = 1; i < dataDims.size(); i++) {
      std::string msg = std::to_string(i);
      msg = "Slice shape should equal data shape for dim " + msg;
      isValid &=
          expectCompareTrue(msg.c_str(), dataDims[i], slicesDims[i], this);
    }
  }
  return isValid;
}

bool BatchOneHotNode::verify() const {
  const auto &dataDims = getData().dims();
  const auto &lengthsDims = getLengths().dims();
  const auto &valuesDims = getValues().dims();

  bool isValid = expectCompareTrue("Data should be a two dimensional matrix",
                                   dataDims.size(), size_t(2), this);

  isValid &= expectCompareTrue("Lengths should be a single dimensional vectors",
                               lengthsDims.size(), size_t(1), this);
  isValid &= checkType(getLengths(), ElemKind::Int32ITy, this);

  isValid &= expectCompareTrue("Values should be a single dimensional vectors",
                               valuesDims.size(), size_t(1), this);

  isValid &=
      expectCompareTrue("Size of Lengths should be equal to width of Data",
                        lengthsDims[0], dataDims[1], this);
  return isValid;
}

bool SpaceToDepthNode::verify() const {
  auto inputN = getInput();
  auto resultN = getResult();
  auto inputDims = inputN.dims();
  auto outputDims = resultN.dims();
  unsigned blockSize = getBlockSize();

  bool sameType = checkTypeIgnoreShape(inputN, resultN, this);
  bool dimTransform = inputDims[0] == outputDims[0] &&
                      inputDims[1] == outputDims[1] * blockSize &&
                      inputDims[2] == outputDims[2] * blockSize &&
                      inputDims[3] * blockSize * blockSize == outputDims[3];

  return sameType && dimTransform;
}

bool SaveNode::verify() const {
  return checkSameType(getInput(), getOutput(), this);
}

bool LogNode::verify() const {
  if (getResult().getType()->isQuantizedType()) {
    return checkSameShape(getInput(), getResult(), this);
  }
  return checkSameType(getInput(), getResult(), this);
}

bool IsNaNNode::verify() const {
  bool isValid = checkSameShape(getResult(), getInput(), this);
  isValid &= checkType(getResult(), ElemKind::BoolTy, this);
  return isValid;
}

bool ReplaceNaNNode::verify() const {
  return checkSameType(getResult(), getInput(), this);
}

bool SelectNode::verify() const {
  bool isValid = checkSameShape(getResult(), getLHS(), this);
  isValid &= checkSameShape(getResult(), getRHS(), this);
  isValid &= checkSameShape(getResult(), getCond(), this);
  isValid &= checkType(getLHS(), getRHS().getElementType(), this);
  isValid &= checkType(getLHS(), getResult().getElementType(), this);
  isValid &= checkType(getCond(), ElemKind::BoolTy, this);
  return isValid;
}

bool ReluNode::verify() const { return verifyRelu(getResult(), getInput()); }

bool ReluGradNode::verify() const {
  return verifyInputAndGradInputTypes(getInput(), getGradOfInputNamedInput(),
                                      this) &&
         verifyOutputAndGradOutputTypes(getOriginalOutputForResult(),
                                        getGradOfOriginalOutputNamedResult(),
                                        this) &&
         verifyRelu(getGradOfOriginalOutputNamedResult(), getInput());
}

bool PReluNode::verify() const {
  return verifyPRelu(getResult(), getInput(), getSlope());
}

bool RegressionNode::verify() const {
  return verifyRegression(getInput(), getResult(), getExpected());
}

bool RegressionGradNode::verify() const {
  return verifyInputAndGradInputTypes(getExpected(),
                                      getGradOfInputNamedExpected(), this) &&
         verifyInputAndGradInputTypes(getInput(), getGradOfInputNamedInput(),
                                      this) &&
         verifyOutputAndGradOutputTypes(getOriginalOutputForResult(),
                                        getGradOfOriginalOutputNamedResult(),
                                        this) &&
         verifyRegression(getGradOfInputNamedInput(),
                          getGradOfOriginalOutputNamedResult(),
                          getGradOfInputNamedExpected());
}

bool SigmoidCrossEntropyWithLogitsNode::verify() const {
  bool isValid = checkType(getResult(), getLogits().getElementType(), this);
  isValid &= checkSameType(getLogits(), getTargets(), this);
  return isValid;
}

bool FullyConnectedNode::verify() const {
  return verifyFullyConnected(getInput(), getWeights(), getBias(), getResult());
}

bool FullyConnectedGradNode::verify() const {
  return verifyInputAndGradInputTypes(getBias(), getGradOfInputNamedBias(),
                                      this) &&
         verifyInputAndGradInputTypes(getInput(), getGradOfInputNamedInput(),
                                      this) &&
         verifyInputAndGradInputTypes(getWeights(),
                                      getGradOfInputNamedWeights(), this) &&
         verifyOutputAndGradOutputTypes(getOriginalOutputForResult(),
                                        getGradOfOriginalOutputNamedResult(),
                                        this) &&
         verifyFullyConnected(
             getGradOfInputNamedInput(), getGradOfInputNamedWeights(),
             getGradOfInputNamedBias(), getGradOfOriginalOutputNamedResult());
}

bool ConcatNode::verify() const {
  auto inputs = getInputs();
  auto dimension = getDim();
  if (!expectCompareTrue("Empty concat?!", inputs.empty(), false, this)) {
    return false;
  }
  bool isValid = expectCompareTrue("concat on invalid dimension",
                                   inputs[0].dims().size(), size_t(dimension),
                                   this, CompareOperatorGreaterThan<size_t>());

  size_t nbDims = inputs[0].dims().size();
  for (size_t i = 1; i < inputs.size(); i++) {
    std::string istr = std::to_string(i);
    std::string msg =
        "input " + istr + "#dims are incompatible between elements";
    if (!expectCompareTrue(msg.c_str(), nbDims, inputs[i].dims().size(),
                           this)) {
      isValid = false;
      // The following loop depends on this condition being true.
      continue;
    }
    for (size_t j = 0; j < nbDims; j++) {
      if (j == dimension) {
        continue;
      }
      std::string innerMsg = std::to_string(j);
      innerMsg =
          "Mismatching dimension " + innerMsg + " for input 0 and " + istr;
      isValid &= expectCompareTrue(innerMsg.c_str(), inputs[0].dims()[j],
                                   inputs[i].dims()[j], this);
    }

    for (size_t i = 0; i < inputs.size(); i++) {
      isValid &= checkType(inputs[i], getResult().getElementType(), this);
      isValid &= checkNotQuantizedOrSameParams(getResult().getType(),
                                               inputs[i].getType(), this);
    }
  }
  return isValid;
}

bool ModuloNode::verify() const { return getDivisor() >= 1; }

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
