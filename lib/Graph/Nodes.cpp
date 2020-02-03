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
      .addParam("layout", getLayout())
      .addParam("output", *getType())
      .addParam("users", getNumUsers());
  return db;
}

std::string Placeholder::getDebugDesc() const {
  DescriptionBuilder db(getKindName());
  db.addParam("name", quote(getName()))
      .addParam("layout", getLayout())
      .addParam("output", *getType())
      .addParam("users", getNumUsers())
      .addParam("trainable", isTraining());
  return db;
}

//===----------------------------------------------------------------------===//
//                       Nodes verification
//===----------------------------------------------------------------------===//

static bool verifyConvFilter(const Node *parent, NodeValue filter,
                             const ShapeNHWC &idim, const ShapeNHWC &odim,
                             const ShapeHW &kdim, unsigned_t group) {
  const dim_t filterDims[] = {odim.c, kdim.height, kdim.width,
                              idim.c / (dim_t)group};
  return expectCompareTrue("Invalid filter dimensions",
                           filter.getType()->dims(),
                           llvm::makeArrayRef(filterDims), parent);
}

static bool verifyConvFilter(const Node *parent, NodeValue filter,
                             const ShapeNCHW &idim, const ShapeNCHW &odim,
                             const ShapeHW &kdim, unsigned_t group) {
  const dim_t filterDims[] = {odim.c, idim.c / (dim_t)group, kdim.height,
                              kdim.width};

  return expectCompareTrue("Invalid filter dimensions",
                           filter.getType()->dims(),
                           llvm::makeArrayRef(filterDims), parent);
}

template <typename Shape>
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
      isValid &=
          expectCompareTrue("Bias type should be Int8 or Int32 for Conv",
                            bias.getElementType() == ElemKind::Int8QTy ||
                                bias.getElementType() == ElemKind::Int32QTy,
                            true, parent);
    }
  }
  Shape idim(src.getType()->dims());
  Shape odim(dest.getType()->dims());
  PaddingTLBR pdim(pads);
  ShapeHW kdim(kernels);
  isValid &= expectCompareTrue("buffer height too small for selected stride",
                               idim.h + pdim.top + pdim.bottom, kdim.height,
                               parent, CompareOperatorGreaterEqual<dim_t>());
  isValid &= expectCompareTrue("buffer width too small for selected stride",
                               idim.w + pdim.left + pdim.right, kdim.width,
                               parent, CompareOperatorGreaterEqual<dim_t>());
  isValid &= expectCompareTrue("channels number must be divisible by groups",
                               idim.c % group, dim_t(0), parent);

  auto outSz = calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides,
                                           pads, dilation);
  isValid &=
      expectCompareTrue("Invalid output dimension N", odim.n, idim.n, parent);
  isValid &= expectCompareTrue("Invalid output dimension H", odim.h,
                               outSz.first, parent);
  isValid &= expectCompareTrue("Invalid output dimension W", odim.w,
                               outSz.second, parent);
  isValid &= expectCompareTrue("Invalid output dimension C", odim.c % group,
                               dim_t(0), parent);

  isValid &= verifyConvFilter(parent, filter, idim, odim, kdim, group);

  const dim_t biasDims[] = {odim.c};

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
    isValid &=
        expectCompareTrue("Bias type should be Int8 or Int32 for Conv3D",
                          bias.getElementType() == ElemKind::Int8QTy ||
                              bias.getElementType() == ElemKind::Int32QTy,
                          true, parent);
  }
  ShapeNHWDC idim(src.getType()->dims());
  ShapeNHWDC odim(dest.getType()->dims());
  PaddingTLNBRF pdim(pads);
  ShapeHWD kdim(kernels);
  isValid &= expectCompareTrue("buffer height too small for selected stride",
                               idim.h + pdim.top + pdim.bottom, kdim.height,
                               parent, CompareOperatorGreaterEqual<dim_t>());
  isValid &= expectCompareTrue("buffer width too small for selected stride",
                               idim.w + pdim.left + pdim.right, kdim.width,
                               parent, CompareOperatorGreaterEqual<dim_t>());
  isValid &= expectCompareTrue("buffer time too small for selected stride",
                               idim.d + pdim.near + pdim.far, kdim.depth,
                               parent, CompareOperatorGreaterEqual<dim_t>());
  isValid &= expectCompareTrue("channels number must be divisible by groups",
                               idim.c % group, dim_t(0), parent);

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
                               dim_t(0), parent);

  const dim_t filterDims[] = {odim.c, kdim.height, kdim.width, kdim.depth,
                              idim.c / group};
  isValid &=
      expectCompareTrue("Invalid filter dimensions", filter.getType()->dims(),
                        llvm::makeArrayRef(filterDims), parent);
  const dim_t biasDims[] = {odim.c};
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
    isValid &=
        expectCompareTrue("Bias type should be Int8 or Int32 for FC",
                          bias.getElementType() == ElemKind::Int8QTy ||
                              bias.getElementType() == ElemKind::Int32QTy,
                          true, parent);
  }
  return isValid;
}

template <typename Shape>
static bool verifyPool(NodeValue src, NodeValue dest,
                       llvm::ArrayRef<unsigned_t> kernels,
                       llvm::ArrayRef<unsigned_t> strides,
                       llvm::ArrayRef<unsigned_t> pads, bool isAvgPool = true) {
  const Node *parent = dest.getNode();
  Shape idim(src.getType()->dims());
  Shape odim(dest.getType()->dims());
  PaddingTLBR pdim(pads);
  ShapeHW kdim(kernels);

  bool isValid =
      expectCompareTrue("buffer height too small for selected stride",
                        idim.h + pdim.top + pdim.bottom, kdim.height, parent,
                        CompareOperatorGreaterEqual<dim_t>());
  isValid &= expectCompareTrue("buffer width too small for selected stride",
                               idim.w + pdim.left + pdim.right, kdim.width,
                               parent, CompareOperatorGreaterEqual<dim_t>());

  auto outSz =
      calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides, pads);
  Shape exp(idim);
  exp.h = outSz.first;
  exp.w = outSz.second;
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
  dim_t channels = src.dims()[channel];

  const dim_t expArray[] = {channels};
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
  isValid &= checkType(indices, IndexElemKind, dest.getNode());
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
  isValid &= checkType(indices, IndexElemKind, dest.getNode());
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

static bool verifyEmbeddingBag(NodeValue dest, NodeValue data,
                               NodeValue weights, NodeValue indices,
                               NodeValue offsets) {
  bool isValid = checkType(dest, data.getElementType(), dest.getNode());
  isValid &= checkType(weights, data.getElementType(), dest.getNode());
  isValid &= checkType(indices, IndexElemKind, dest.getNode());
  isValid &= checkType(offsets, IndexElemKind, dest.getNode());
  isValid &=
      expectCompareTrue("Indices must be a 1D vector", indices.dims().size(),
                        size_t(1), dest.getNode());
  isValid &=
      expectCompareTrue("Offsets must be a 1D vector", offsets.dims().size(),
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
  if (getLayout() == NHWC) {
    return verifyConvolution<ShapeNHWC>(getInput(), getResult(), getFilter(),
                                        getBias(), Kernels_, Strides_, Pads_,
                                        Group_, Dilation_);
  } else {
    return verifyConvolution<ShapeNCHW>(getInput(), getResult(), getFilter(),
                                        getBias(), Kernels_, Strides_, Pads_,
                                        Group_, Dilation_);
  }
}

bool ChannelwiseQuantizedConvolutionNode::verify() const {
  bool isValid =
      verifyConvolution<ShapeNHWC>(getInput(), getResult(), getFilter(),
                                   getBias(), Kernels_, Strides_, Pads_, Group_,
                                   /* dilation */ 1, /* checkBiasType */ false);

  isValid &= checkType(getBias(), ElemKind::Int32QTy, this);
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
  isValid &= expectCompareTrue(
      "There must be one filter offset qparam per output channel",
      getOffsets().dims()[0], dim_t(getResult().dims()[3]), this);
  isValid &= expectCompareTrue(
      "There must be one filter scale qparam per output channel",
      getScales().dims()[0], dim_t(getResult().dims()[3]), this);
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
  if (getLayout() == NHWC) {
    isValid &= verifyConvolution<ShapeNHWC>(
        getGradOfInputNamedInput(), getGradOfOriginalOutputNamedResult(),
        getGradOfInputNamedFilter(), getGradOfInputNamedBias(), Kernels_,
        Strides_, Pads_, Group_, Dilation_);
  } else {
    isValid &= verifyConvolution<ShapeNCHW>(
        getGradOfInputNamedInput(), getGradOfOriginalOutputNamedResult(),
        getGradOfInputNamedFilter(), getGradOfInputNamedBias(), Kernels_,
        Strides_, Pads_, Group_, Dilation_);
  }

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

/// \returns the number of columns of data from a fused \p type (i.e. not
/// considering the columns for per row scale/offsets).
static size_t getNumDataColumnsFromFused(TypeRef type) {
  size_t n = type->dims()[1];
  switch (type->getElementType()) {
  case ElemKind::UInt8FusedQTy:
    return n - 2 * sizeof(float);
  case ElemKind::UInt8FusedFP16QTy:
    return n - 2 * sizeof(float16_t);
  default:
    llvm_unreachable("Not supported Fused ElemKind");
  }
}

bool ConvertToNode::verify() const {
  TypeRef srcTy = getInput().getType();
  TypeRef dstTy = getResult().getType();
  const bool srcIsFused = isFusedQuantizedElemKind(srcTy->getElementType());
  const bool dstIsFused = isFusedQuantizedElemKind(dstTy->getElementType());

  bool isValid = expectCompareTrue(
      "Conversion of src and dst with mismatched fused property is not yet "
      "implemented",
      (srcIsFused && dstIsFused) || (!srcIsFused && !dstIsFused), true, this);

  if (srcIsFused && dstIsFused) {
    size_t srcNumCols = getNumDataColumnsFromFused(srcTy);
    size_t dstNumCols = getNumDataColumnsFromFused(dstTy);
    return expectCompareTrue("Shapes of data for fused kinds do not match",
                             srcNumCols, dstNumCols, this);
  }

  isValid &= checkSameShape(getInput(), getResult(), this);
  isValid &= expectCompareTrue(
      "Quantized conversion should use Dequantize, Quantize and Rescale",
      srcTy->isQuantizedType() || dstTy->isQuantizedType(), false, this);
  return isValid;
}

bool MaxPoolNode::verify() const {
  if (getLayout() == NHWC) {
    return verifyPool<ShapeNHWC>(getInput(), getResult(), Kernels_, Strides_,
                                 Pads_,
                                 /* isAvgPool */ false);
  } else {
    return verifyPool<ShapeNCHW>(getInput(), getResult(), Kernels_, Strides_,
                                 Pads_,
                                 /* isAvgPool */ false);
  }
}

bool AvgPoolNode::verify() const {
  if (getLayout() == NHWC) {
    return verifyPool<ShapeNHWC>(getInput(), getResult(), Kernels_, Strides_,
                                 Pads_);
  } else {
    return verifyPool<ShapeNCHW>(getInput(), getResult(), Kernels_, Strides_,
                                 Pads_);
  }
}

bool AdaptiveAvgPoolGradNode::verify() const {
  bool isValid = verifyInputAndGradInputTypes(getInput(),
                                              getGradOfInputNamedInput(), this);
  isValid &= verifyOutputAndGradOutputTypes(
      getOriginalOutputForResult(), getGradOfOriginalOutputNamedResult(), this);

  ShapeNHWC idim(getInput().getType()->dims());
  ShapeNHWC odim(getOriginalOutputForResult().getType()->dims());

  isValid &= expectCompareTrue(
      "expected the same number of channels for input and output", odim.c,
      idim.c, this);

  isValid &= expectCompareTrue(
      "expected the same number of batches for input and output", odim.n,
      idim.n, this);

  isValid &= expectCompareTrue("height too small for averaging area", odim.h,
                               idim.h, this, CompareOperatorLessEqual<dim_t>());

  isValid &= expectCompareTrue("width too small for averaging area", odim.w,
                               idim.w, this, CompareOperatorLessEqual<dim_t>());

  return isValid;
}

bool AdaptiveAvgPoolNode::verify() const {
  bool isValid = checkTypeIgnoreShape(getInput(), getResult(), this);

  TypeRef inTy = getInput().getType();
  TypeRef outTy = getResult().getType();

  isValid &= expectCompareTrue("Input should have 4 dimensions",
                               inTy->dims().size(), (size_t)4, this);

  isValid &= expectCompareTrue("Output should have 4 dimensions",
                               outTy->dims().size(), (size_t)4, this);

  if (!isValid) {
    return false;
  }

  isValid &= expectCompareTrue(
      "Output should have the same number of batches as the input",
      inTy->dims()[0], outTy->dims()[0], this);

  isValid &= expectCompareTrue(
      "Output should have the same number of channels as the input",
      inTy->dims()[3], outTy->dims()[3], this);

  isValid &= expectCompareTrue(
      "Output should not have more height than the input", inTy->dims()[1],
      outTy->dims()[1], this, CompareOperatorGreaterEqual<dim_t>());

  isValid &= expectCompareTrue(
      "Output should not have more width than the input", inTy->dims()[2],
      outTy->dims()[2], this, CompareOperatorGreaterEqual<dim_t>());

  return isValid;
}

bool MaxPoolGradNode::verify() const {
  bool isValid = verifyInputAndGradInputTypes(getInput(),
                                              getGradOfInputNamedInput(), this);
  isValid &= verifyOutputAndGradOutputTypes(
      getOriginalOutputForResult(), getGradOfOriginalOutputNamedResult(), this);

  if (getLayout() == NHWC) {
    isValid &= verifyPool<ShapeNHWC>(
        getGradOfInputNamedInput(), getGradOfOriginalOutputNamedResult(),
        Kernels_, Strides_, Pads_, /* isAvgPool */ false);
  } else {
    isValid &= verifyPool<ShapeNCHW>(
        getGradOfInputNamedInput(), getGradOfOriginalOutputNamedResult(),
        Kernels_, Strides_, Pads_, /* isAvgPool */ false);
  }
  return isValid;
}

bool AvgPoolGradNode::verify() const {
  bool isValid = verifyInputAndGradInputTypes(getInput(),
                                              getGradOfInputNamedInput(), this);
  isValid &= verifyOutputAndGradOutputTypes(
      getOriginalOutputForResult(), getGradOfOriginalOutputNamedResult(), this);

  if (getLayout() == NHWC) {
    isValid &= verifyPool<ShapeNHWC>(getGradOfInputNamedInput(),
                                     getGradOfOriginalOutputNamedResult(),
                                     Kernels_, Strides_, Pads_);
  } else {
    isValid &= verifyPool<ShapeNCHW>(getGradOfInputNamedInput(),
                                     getGradOfOriginalOutputNamedResult(),
                                     Kernels_, Strides_, Pads_);
  }

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

  const dim_t numBatches = LHS.dims()[0];
  const dim_t N = LHS.dims()[1];
  const dim_t M = LHS.dims()[2];
  const dim_t P = RHS.dims()[2];
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

bool BucketizeNode::verify() const {
  bool isValid = checkSameShape(getInput(), getResult(), this);
  isValid &= !getBoundaries().empty();
  isValid &= std::is_sorted(getBoundaries().begin(), getBoundaries().end());
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

bool FlipNode::verify() const {
  auto dest = getResult();
  auto src = getInput();
  dim_t axis = getAxis();
  bool isValid = checkSameType(src, dest, this);
  isValid &= expectCompareTrue("Invalid axis", axis, (dim_t)src.dims().size(),
                               this, CompareOperatorLess<dim_t>());
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

bool ClipNode::verify() const {
  return checkSameType(getInput(), getResult(), this);
}

bool InsertTensorNode::verify() const {
  auto dest = getBig();
  auto src = getSmall();
  auto offsets = getStart();
  dim_t numDims = dest.dims().size();
  dim_t axis = getAxis();
  dim_t count = getCount();

  bool isValid = expectCompareTrue("Invalid number of dimensions", numDims,
                                   (dim_t)src.dims().size(), this);
  isValid &= expectCompareTrue("Invalid number of dimensions for offsets",
                               numDims, (dim_t)offsets.size(), this);

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
                                 CompareOperatorLessEqual<dim_t>());
  }

  isValid &= expectCompareTrue("Invalid axis", axis, (dim_t)src.dims().size(),
                               this, CompareOperatorLessEqual<dim_t>());
  for (dim_t i = 0; i < src.dims().size(); i++) {
    dim_t mul = (i == axis) ? count : 1;
    std::string msg = std::to_string(i);
    msg = "Small does not fit inside Big for index " + msg;
    isValid &=
        expectCompareTrue(msg.c_str(), src.dims()[i] * mul, dest.dims()[i],
                          this, CompareOperatorLessEqual<dim_t>());
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
                                 CompareOperatorLessEqual<dim_t>());
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

  for (dim_t i = 0; i < src.dims().size(); i++) {
    dim_t mul = (i == axis) ? count : 1;
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

bool LayerNormalizationNode::verify() const {
  auto dest = getResult();
  auto src = getInput();
  auto scale = getScale();
  auto bias = getBias();

  bool isValid = true;

  // Check inputs and outputs match.
  isValid &= checkSameType(src, dest, this);

  // Check that the types of scale and bias match and that they have the same
  // ElemKind as input.
  isValid &= checkTypeIgnoreShape(scale, src, this);
  isValid &= checkSameType(scale, bias, this);

  // Check that the dims of scale and bias match the end of src.
  auto srcDims = src.getType()->dims();
  auto scaleDims = scale.getType()->dims();
  isValid &= expectCompareTrue("Expected input to have more dims than scale",
                               srcDims.size(), scaleDims.size(), this,
                               CompareOperatorGreaterThan<size_t>());
  for (size_t i = 0; i < scaleDims.size(); ++i) {
    size_t scaleI = scaleDims.size() - i - 1;
    size_t srcI = srcDims.size() - i - 1;
    isValid &=
        expectCompareTrue("Expected scale dims to match the end of src dims",
                          scaleDims[scaleI], srcDims[srcI], this);
  }

  return isValid;
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
VERIFY_CMP(CmpLT)
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

bool CumSumNode::verify() const {
  return checkSameType(getResult(), getInput(), this);
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

bool BatchedReduceMinNode::verify() const {
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

bool SparseLengthsSumGradNode::verify() const {
  // Same checks as SparseLengthsSumNode.
  bool isValid = verifySparseLengthsSum(getOriginalOutputForResult(), getData(),
                                        getIndices(), getLengths());

  // Checks on gradient inputs/outputs.
  isValid &= checkSameType(getGradOfOriginalOutputNamedResult(),
                           getOriginalOutputForResult(), this);
  isValid &= checkSameType(getGradOfInputNamedData(), getData(), this);
  isValid &= checkSameType(getGradOfInputNamedIndices(), getIndices(), this);
  isValid &= checkSameType(getGradOfInputNamedLengths(), getLengths(), this);
  return isValid;
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

bool EmbeddingBagNode::verify() const {
  return verifyEmbeddingBag(getResult(), getData(), getWeights(), getIndices(),
                            getOffsets());
}

bool RowwiseQuantizedSparseLengthsWeightedSumNode::verify() const {
  bool isValid = checkType(getData(), ElemKind::UInt8QTy, this);
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
  if (getUseFP16Accumulation()) {
    isValid &= expectCompareTrue(
        "Only use FP16 accumulation with FP16 version of Fused-RWQ-SLWS.",
        getResult().getType()->getElementType(), ElemKind::Float16Ty, this);
  }
  return isValid;
}

static bool verifyFusedRowwiseQuantizedSparseLengthsSum(
    NodeValue result, NodeValue data, NodeValue indices, NodeValue lengths,
    NodeValue weights, bool useFP16Accumulation,
    bool isEmbeddingBagByteRowwiseOffsets = false) {
  const Node *parent = result.getNode();
  bool isValid = expectCompareTrue(
      "Input data must be Fused Quantized type",
      isFusedQuantizedElemKind(data.getType()->getElementType()), true, parent);
  dim_t extraCols;
  if (data.getType()->getElementType() == ElemKind::UInt8FusedQTy) {
    extraCols = 2 * sizeof(float);
  } else {
    extraCols = 2 * sizeof(float16_t);
  }
  if (useFP16Accumulation) {
    isValid &= expectCompareTrue(
        "Only use FP16 accumulation with FP16 version of RWQ-SLWS.",
        result.getType()->getElementType(), ElemKind::Float16Ty, parent);
  }
  isValid &= checkType(indices, IndexElemKind, parent);
  // For EmbeddingBagByteRowwiseOffsets lengths are really offsets and should be
  // Int64ITy.
  if (isEmbeddingBagByteRowwiseOffsets) {
    isValid &= checkType(lengths, IndexElemKind, parent);
  } else {
    isValid &= checkType(lengths, ElemKind::Int32ITy, parent);
  }

  isValid &= expectCompareTrue("Indices must be a 1D vector",
                               indices.dims().size(), size_t(1), parent);
  isValid &= expectCompareTrue("Lengths must be a 1D vector",
                               lengths.dims().size(), size_t(1), parent);
  isValid &= expectCompareTrue("Data must be 2 dimensional.",
                               data.dims().size(), size_t(2), parent);
  isValid &= expectCompareTrue("Data must have extra columns for scale/offset.",
                               data.dims()[1], extraCols, parent,
                               CompareOperatorGreaterEqual<dim_t>());
  isValid &= expectCompareTrue("Result must be 2 dimensional.",
                               result.dims().size(), size_t(2), parent);

  if (weights.getNode()) {
    isValid &= expectCompareTrue("Weights must be a 1D vector",
                                 weights.dims().size(), size_t(1), parent);
    isValid &= expectCompareTrue("Weights and Indices must have the same size",
                                 weights.dims()[0], indices.dims()[0], parent);
  }

  // Wrap this in isValid to prevent potential segfault if the result is
  // incorrectly shaped.
  if (isValid) {
    // If using 4-bit quantization for embeddings then the input is packed into
    // two elements per byte.
    dim_t finalSize = result.dims()[1];
    if (data.getType()->getElementType() == ElemKind::UInt4FusedFP16QTy) {
      finalSize /= 2;
    }
    isValid &=
        expectCompareTrue("Result output shape should have second dim without "
                          "extra columns from scale/offset in Data.",
                          finalSize + extraCols, data.dims()[1], parent);
  }
  return isValid;
}

bool EmbeddingBagByteRowwiseOffsetsNode::verify() const {
  return verifyFusedRowwiseQuantizedSparseLengthsSum(
      getResult(), getData(), getIndices(), getOffsets(), getWeights(),
      getUseFP16Accumulation(), /*isEmbeddingBagByteRowwiseOffsets*/ true);
}

bool FusedRowwiseQuantizedSparseLengthsWeightedSumNode::verify() const {
  return verifyFusedRowwiseQuantizedSparseLengthsSum(
      getResult(), getData(), getIndices(), getLengths(), getWeights(),
      getUseFP16Accumulation());
}

bool FusedRowwiseQuantizedSparseLengthsSumNode::verify() const {
  return verifyFusedRowwiseQuantizedSparseLengthsSum(
      getResult(), getData(), getIndices(), getLengths(), nullptr,
      getUseFP16Accumulation());
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
                               getResult().dims()[1], dim_t(2), this);
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
  isValid &= checkType(getIndices(), IndexElemKind, this);
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
  isValid &= checkType(getIndices(), IndexElemKind, this);
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
      getComputationInfo().dims()[0], (dim_t)(2), this);
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
      (dim_t)(256 * getResult().getType()->getElementSize()), this);
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

bool ArgMaxNode::verify() const {
  bool isValid = true;

  // Check input type.
  isValid &=
      checkType(getArgmax(), llvm::ArrayRef<ElemKind>({IndexElemKind}), this);

  isValid &= expectCompareTrue("Input must be a 4D tensor",
                               getInput().dims().size(), size_t(4), this);

  // Check expected output type.
  bool keepdims = getKeepDims();
  const unsigned_t axis = getAxis();

  ShapeVector expDstDims;
  auto srcDim = getInput().dims();
  for (size_t i = 0; i < srcDim.size(); i++) {
    if (i == axis) {
      if (keepdims) {
        expDstDims.push_back(1);
      }
    } else {
      expDstDims.push_back(srcDim[i]);
    }
  }
  isValid &= expectCompareTrue("Invalid output dims", getArgmax().dims(),
                               llvm::makeArrayRef(expDstDims), this);
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
                               getRanges().dims()[2], dim_t(2), this);
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

bool ScatterDataNode::verify() const {
  const auto &slicesDims = getSlices().dims();
  const auto &dataDims = getData().dims();
  const auto &indicesDims = getIndices().dims();
  bool isValid = true;
  isValid &= expectCompareTrue("Type mismatch",
                               getSlices().getType()->getElementType(),
                               getData().getType()->getElementType(), this);
  if (!isValid) {
    return false;
  }
  // TODO: Do we need support for different quant params of copy?
  if (getSlices().getType()->isQuantizedType() && !getCumulative()) {
    isValid &=
        expectCompareTrue("Scale mismatch", getSlices().getType()->getScale(),
                          getData().getType()->getScale(), this);
    isValid &=
        expectCompareTrue("Offset mismatch", getSlices().getType()->getOffset(),
                          getData().getType()->getOffset(), this);
  }
  isValid &= expectCompareTrue("There should be an index for each slice",
                               indicesDims[0], slicesDims[0], this);
  isValid &= expectCompareTrue("Indices should be a 2D tensor",
                               indicesDims.size(), size_t(2), this);
  // The code below may crash if these conditions are not met.
  if (!isValid) {
    return false;
  }
  const size_t indexSize = indicesDims[1];
  isValid &= expectCompareTrue("Dimensions of Data should be equal to "
                               "dimensions of indices + dimensions of updates",
                               slicesDims.size() - 1 + indexSize,
                               dataDims.size(), this);
  if (dataDims.size() > 1) {
    for (size_t i = indexSize; i < dataDims.size(); i++) {
      std::string msg = std::to_string(i);
      msg = "Slice shape should equal data shape for dim " + msg;
      isValid &= expectCompareTrue(msg.c_str(), dataDims[i],
                                   slicesDims[i - indexSize + 1], this);
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

bool ResizeNearestNode::verify() const {
  auto input = getInput();
  auto result = getResult();
  auto inputDims = input.dims();
  auto outputDims = result.dims();

  bool isValid = checkTypeIgnoreShape(input, result, this);
  isValid &= expectCompareTrue("Input must be a 4D tensor", inputDims.size(),
                               size_t(4), this);
  isValid &= expectCompareTrue("Output must be a 4D tensor", outputDims.size(),
                               size_t(4), this);
  isValid &= expectCompareTrue("Batch size must be the same", inputDims[0],
                               outputDims[0], this);
  isValid &= expectCompareTrue("Depth must be the same", inputDims[3],
                               outputDims[0], this);
  isValid &= expectCompareTrue(
      "Unexpected output height",
      dim_t(std::floor(inputDims[1] * getHeightScale())), outputDims[1], this);
  isValid &= expectCompareTrue(
      "Unexpected output width",
      dim_t(std::floor(inputDims[2] * getWidthScale())), outputDims[2], this);
  isValid &=
      expectCompareTrue("Invalid height scale", getHeightScale(), float(0.0),
                        this, CompareOperatorGreaterThan<float>());
  isValid &=
      expectCompareTrue("Invalid width scale", getWidthScale(), float(0.0),
                        this, CompareOperatorGreaterThan<float>());

  return isValid;
}

bool NonMaxSuppressionNode::verify() const {
  NodeValue boxes = getBoxes();
  NodeValue scores = getScores();
  auto boxesDims = boxes.dims();
  auto scoresDims = scores.dims();
  bool isV4 = getIsTFVersion4();

  size_t scoresBoxDim = scores.dims().size() - 1;
  size_t scoresBatchDim = scores.dims().size() - 3;

  size_t boxesBoxDim = boxes.dims().size() - 2;
  size_t boxesBatchDim = boxes.dims().size() - 3;

  bool isValid = true;
  if (isV4) {
    isValid &= expectCompareTrue(
        "Number of boxes doesn't match number of confidence scores.",
        boxesDims[boxesBoxDim], scoresDims[scoresBoxDim], this,
        CompareOperatorEqual<dim_t>());
  }

  // checking layout matching. See ONNX spec for details.
  if (!isV4) {
    isValid &= expectCompareTrue(
        "Batch dimension doesn't match.", boxesDims[boxesBatchDim],
        scoresDims[scoresBatchDim], this, CompareOperatorEqual<dim_t>());

    isValid &= expectCompareTrue(
        "Number of boxes doesn't match number of confidence scores.",
        boxesDims[boxesBoxDim], scoresDims[scoresBoxDim], this,
        CompareOperatorEqual<dim_t>());
  }

  isValid &= checkType(boxes, scores.getElementType(), this);

  return isValid;
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

bool BatchBoxCoxNode::verify() const {
  auto result = getResult();
  auto data = getInput();
  auto lambda1 = getLambda1();
  auto lambda2 = getLambda2();
  bool isValid = checkSameType(lambda1, lambda2, this);
  isValid &= checkSameType(data, result, this);
  isValid &= checkType(data, lambda1.getElementType(), this);
  isValid &= checkType(data, {ElemKind::FloatTy, ElemKind::Float16Ty}, this);
  isValid &= expectCompareTrue("Input must be a 2D tensor", data.dims().size(),
                               size_t(2), this);
  isValid &= expectCompareTrue("Lambda1 must be a 1D vector",
                               lambda1.dims().size(), size_t(1), this);
  if (isValid) {
    isValid &= expectCompareTrue("Data dim 1 must equal lambda dim",
                                 data.dims()[1], lambda1.dims()[0], this);
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
/// Convert a collection of floats into a vector of
/// unsigned integer binary representation.
std::vector<size_t> toBinary(llvm::ArrayRef<float> vec) {
  std::vector<size_t> sizeVec(vec.size());
  std::for_each(vec.begin(), vec.end(), [&sizeVec](float f) -> void {
    sizeVec.push_back(toBinary(f));
  });
  return sizeVec;
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

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              FusedActivation fusedActivation) {
  switch (fusedActivation) {
  case FusedActivation::NONE:
    break;
  case FusedActivation::RELU:
    os << "RELU";
    break;
  case FusedActivation::SIGMOID:
    os << "SIGMOID";
    break;
  case FusedActivation::TANH:
    os << "TANH";
    break;
  }
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, ConvolutionLayout layout) {
  switch (layout) {
  case ConvolutionLayout::NCHW:
    os << "NCHW";
    break;
  case ConvolutionLayout::NHWC:
    os << "NHWC";
    break;
  }
  return os;
}
} // namespace glow
