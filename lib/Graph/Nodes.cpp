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

std::string Constant::getDebugDesc(bool skipUsers) const {
  DescriptionBuilder db(getKindName());
  db.addParam("Name", separateString(getName(), 100, "\n"))
      .addParam("Layout", getLayout())
      .addParam("Output", *getType());
  if (!skipUsers) {
    db.addParam("Users", getNumUsers());
  }
  return db;
}

std::string Placeholder::getDebugDesc(bool skipUsers) const {
  DescriptionBuilder db(getKindName());
  db.addParam("Name", separateString(getName(), 100, "\n"))
      .addParam("Layout", getLayout())
      .addParam("Output", *getType())
      .addParam("Trainable", isTraining())
      .addParam("Static", isStatic());
  if (!skipUsers) {
    db.addParam("Users", getNumUsers());
  }
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
                              llvm::ArrayRef<unsigned_t> dilation,
                              bool checkBiasType = true) {
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
          expectCompareTrue("Bias type should be float, Int8 or Int32 for Conv",
                            bias.getElementType() == ElemKind::FloatTy ||
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
  isValid &= expectCompareTrue("Dilation should have same length as Stride",
                               dilation.size(), strides.size(), parent);

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
        expectCompareTrue("Bias type should be Float, Int8 or Int32 for Conv3D",
                          bias.getElementType() == ElemKind::FloatTy ||
                              bias.getElementType() == ElemKind::Int8QTy ||
                              bias.getElementType() == ElemKind::Int32QTy,
                          true, parent);
  }
  ShapeNTHWC idim(src.getType()->dims());
  ShapeNTHWC odim(dest.getType()->dims());
  PaddingNFTBLR pdim(pads);
  ShapeTHW kdim(kernels);
  isValid &= expectCompareTrue("buffer height too small for selected stride",
                               idim.h + pdim.top + pdim.bottom, kdim.height,
                               parent, CompareOperatorGreaterEqual<dim_t>());
  isValid &= expectCompareTrue("buffer width too small for selected stride",
                               idim.w + pdim.left + pdim.right, kdim.width,
                               parent, CompareOperatorGreaterEqual<dim_t>());
  isValid &=
      expectCompareTrue("buffer time too small for selected stride",
                        idim.t + pdim.near + pdim.far, kdim.temporal_frames,
                        parent, CompareOperatorGreaterEqual<dim_t>());
  isValid &= expectCompareTrue("channels number must be divisible by groups",
                               idim.c % group, dim_t(0), parent);

  auto outSz = calculate3DConvPoolOutputDims(idim.t, idim.h, idim.w, kernels,
                                             strides, pads);
  isValid &=
      expectCompareTrue("Invalid output dimension N", odim.n, idim.n, parent);
  isValid &= expectCompareTrue("Invalid output dimension T", odim.t,
                               outSz.temporal_frames, parent);
  isValid &= expectCompareTrue("Invalid output dimension H", odim.h,
                               outSz.height, parent);
  isValid &= expectCompareTrue("Invalid output dimension W", odim.w,
                               outSz.width, parent);
  isValid &= expectCompareTrue("Invalid output dimension C", odim.c % group,
                               dim_t(0), parent);

  const dim_t filterDims[] = {odim.c, kdim.temporal_frames, kdim.height,
                              kdim.width, idim.c / group};
  isValid &=
      expectCompareTrue("Invalid filter dimensions", filter.getType()->dims(),
                        llvm::makeArrayRef(filterDims), parent);
  const dim_t biasDims[] = {odim.c};
  isValid &=
      expectCompareTrue("Invalid bias dimensions", bias.getType()->dims(),
                        llvm::makeArrayRef(biasDims), parent);
  return isValid;
}

static bool verifyConvTranspose(NodeValue src, NodeValue dest, NodeValue filter,
                                llvm::ArrayRef<unsigned_t> kernels,
                                llvm::ArrayRef<unsigned_t> strides,
                                llvm::ArrayRef<unsigned_t> pads,
                                unsigned_t group,
                                llvm::ArrayRef<unsigned_t> dilation) {
  const Node *parent = dest.getNode();
  bool isValid = checkType(src, dest.getElementType(), parent);
  isValid &= checkType(src, filter.getElementType(), parent);
  ShapeNHWC idim(src.getType()->dims());
  ShapeNHWC odim(dest.getType()->dims());
  PaddingTLBR pdim(pads);
  ShapeHW kdim(kernels);
  // TODO: any kernel size check in respect to input ? In contrast to Conv,
  // seems kernel can be any size.

  isValid &= expectCompareTrue("channels number must be divisible by groups",
                               idim.c % group, dim_t(0), parent);

  isValid &= expectCompareTrue("Stride should be less than kernel.",
                               strides[0] <= kernels[0], true, parent);

  isValid &= expectCompareTrue("Stride should be less than kernel.",
                               strides[1] <= kernels[1], true, parent);

  isValid &= expectCompareTrue("channels number must be divisible by groups",
                               idim.c % group, dim_t(0), parent);

  isValid &= expectCompareTrue("Dilation should have same length as Stride",
                               dilation.size(), strides.size(), parent);

  auto outSz = calculateConvTransposeOutputDims(idim.h, idim.w, kernels,
                                                strides, pads, dilation);
  (void)outSz;
  isValid &=
      expectCompareTrue("Invalid output dimension N", odim.n, idim.n, parent);

  isValid &=
      expectCompareTrue("Invalid output dimension HT", odim.h, outSz.first,
                        parent, CompareOperatorGreaterEqual<dim_t>());
  isValid &=
      expectCompareTrue("Invalid output dimension WT", odim.w, outSz.second,
                        parent, CompareOperatorGreaterEqual<dim_t>());

  isValid &= expectCompareTrue("Invalid output dimension CT", odim.c % group,
                               dim_t(0), parent);

  const dim_t filterDims[] = {odim.c / (dim_t)group, kdim.height, kdim.width,
                              idim.c};
  isValid &=
      expectCompareTrue("Invalid filter dimensions", filter.getType()->dims(),
                        llvm::makeArrayRef(filterDims), parent);
  return isValid;
}

static bool verifyFullyConnected(NodeValue src, NodeValue weights,
                                 NodeValue bias, NodeValue dest) {
  const Node *parent = dest.getNode();
  bool isValid = expectCompareTrue("FC input must be 2D", size_t(2),
                                   src.dims().size(), parent);
  isValid &= expectCompareTrue("FC weights must be 2D", size_t(2),
                               weights.dims().size(), parent);
  isValid &= expectCompareTrue("FC bias must be 1D", size_t(1),
                               bias.dims().size(), parent);
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
        expectCompareTrue("Bias type should be Int8, Int32 or FP32 for FC",
                          bias.getElementType() == ElemKind::Int8QTy ||
                              bias.getElementType() == ElemKind::Int32QTy ||
                              bias.getElementType() == ElemKind::FloatTy,
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

template <typename Shape>
static bool
verifyPool3D(NodeValue src, NodeValue dest, llvm::ArrayRef<unsigned_t> kernels,
             llvm::ArrayRef<unsigned_t> strides,
             llvm::ArrayRef<unsigned_t> pads, bool isAvgPool = true) {
  const Node *parent = dest.getNode();
  Shape idim(src.getType()->dims());
  Shape odim(dest.getType()->dims());
  PaddingTLNBRF pdim(pads);
  ShapeTHW kdim(kernels);

  bool isValid =
      expectCompareTrue("buffer height too small for selected stride",
                        idim.h + pdim.top + pdim.bottom, kdim.height, parent,
                        CompareOperatorGreaterEqual<dim_t>());
  isValid &= expectCompareTrue("buffer width too small for selected stride",
                               idim.w + pdim.left + pdim.right, kdim.width,
                               parent, CompareOperatorGreaterEqual<dim_t>());
  isValid &=
      expectCompareTrue("buffer temporal_frames are too small for "
                        "selected stride",
                        idim.t + pdim.near + pdim.far, kdim.temporal_frames,
                        parent, CompareOperatorGreaterEqual<dim_t>());

  auto outSz = calculate3DConvPoolOutputDims(idim.t, idim.h, idim.w, kernels,
                                             strides, pads);
  Shape exp(idim);
  exp.t = outSz.temporal_frames;
  exp.h = outSz.height;
  exp.w = outSz.width;
  isValid &=
      expectCompareTrue("Unexpected output dimensions", exp, odim, parent);

  // For quantized AvgPool, the scale and offset of its input and output could
  // be different. But for quantized MaxPool, the scale and offset of its input
  // and output should be the same.
  isValid =
      isValid && checkSameIsQuantized(src.getType(), dest.getType(), parent);
  if (!isAvgPool) {
    isValid = isValid && checkTypeIgnoreShape(src, dest, parent);
  }

  return isValid;
}

static bool verifyBatchNormalization(NodeValue src, NodeValue dest,
                                     NodeValue bias, NodeValue scale,
                                     NodeValue mean, NodeValue var,
                                     unsigned_t channel) {
  const Node *parent = dest.getNode();

  // Source and Dest can have different quantization params
  // but need to match in shape and element type.
  bool isValid = checkSameShape(dest, src, parent);
  isValid = isValid && checkType(dest, src.getElementType(), parent);

  isValid =
      isValid &&
      expectCompareTrue(
          "Require at least two input dims i.e., batch and channel dimensions",
          src.dims().size(), (size_t)1, parent,
          CompareOperatorGreaterThan<size_t>());

  // Figure out how many channels are in the tensor.
  dim_t channels = src.dims()[channel];

  const dim_t expArray[] = {channels};
  auto exp = llvm::makeArrayRef(expArray);
  isValid = isValid && expectCompareTrue("Invalid bias dimension",
                                         bias.getType()->dims(), exp, parent);
  isValid = isValid && expectCompareTrue("Invalid scale dimension",
                                         scale.getType()->dims(), exp, parent);
  isValid = isValid && expectCompareTrue("Invalid mean dimension",
                                         mean.getType()->dims(), exp, parent);
  isValid = isValid && expectCompareTrue("Invalid var dimension",
                                         var.getType()->dims(), exp, parent);
  return isValid;
}

static bool verifyInstanceNormalization(NodeValue src, NodeValue dest,
                                        NodeValue bias, NodeValue scale,
                                        unsigned_t channel) {
  const Node *parent = dest.getNode();
  bool isValid = true;
  if (src.getType()->isQuantizedType()) {
    isValid &= checkType(src, dest.getElementType(), dest.getNode());
    isValid &= checkSameShape(src, dest, parent);
  } else {
    isValid &= checkSameType(src, dest, parent);
  }

  isValid &= expectCompareTrue(
      "Require at least two input dims i.e., batch and channel dimensions",
      src.dims().size(), (size_t)1, parent,
      CompareOperatorGreaterThan<size_t>());

  // Figure out how many channels are in the tensor.
  dim_t channels = src.dims()[channel];

  const dim_t expArray[] = {channels};
  auto exp = llvm::makeArrayRef(expArray);
  isValid &= expectCompareTrue("Invalid bias dimension", bias.getType()->dims(),
                               exp, parent);
  isValid &= expectCompareTrue("Invalid scale dimension",
                               scale.getType()->dims(), exp, parent);

  return isValid;
}

static bool verifyActivation(NodeValue src, NodeValue dest) {
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

static bool verifyLogSoftMax(NodeValue src, NodeValue dest) {
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
  isValid &= checkType(indices, {ElemKind::Int64ITy, ElemKind::Int32ITy},
                       dest.getNode());
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
  isValid &= checkType(indices, {ElemKind::Int64ITy, ElemKind::Int32ITy},
                       dest.getNode());
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

static bool verifyEmbedding(NodeValue dest, NodeValue weights,
                            NodeValue indices) {
  bool isValid = checkType(dest, weights.getElementType(), dest.getNode());
  isValid &= checkType(
      indices,
      llvm::ArrayRef<ElemKind>({ElemKind::Int64ITy, ElemKind::Int32ITy}),
      dest.getNode());
  isValid &=
      expectCompareTrue("Weights must be a 2D tensor", weights.dims().size(),
                        size_t(2), weights.getNode());
  return isValid;
}

static bool verifyEmbeddingBag(NodeValue dest, NodeValue data,
                               NodeValue weights, NodeValue indices,
                               NodeValue offsets) {
  bool isValid = checkType(dest, data.getElementType(), dest.getNode());
  isValid &= checkType(weights, data.getElementType(), dest.getNode());
  isValid &= checkType(
      indices,
      llvm::ArrayRef<ElemKind>({ElemKind::Int64ITy, ElemKind::Int32ITy}),
      dest.getNode());
  isValid &= checkType(
      offsets,
      llvm::ArrayRef<ElemKind>({ElemKind::Int64ITy, ElemKind::Int32ITy}),
      dest.getNode());
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

bool HardSwishNode::verify() const {
  return checkSameType(getInput(), getResult(), this);
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
  auto input_dims = getInput().getType()->dims();
  bool isValid = false;
  bool isConv3D = (input_dims.size() == 5);
  if (isConv3D) {
    isValid = verifyConvolution3D(getInput(), getResult(), getFilter(),
                                  getBias(), Kernels_, Strides_, Pads_, Group_);

    if (!all_of(Dilation_.begin(), Dilation_.end(),
                [](unsigned_t i) { return i == 1; })) {
      report("For Conv3D dilation must be 1");
    }
  } else {
    isValid = verifyConvolution<ShapeNHWC>(
        getInput(), getResult(), getFilter(), getBias(), Kernels_, Strides_,
        Pads_, Group_, Dilation_, /* checkBiasType */ false);
  }

  isValid &= checkType(getResult(), ElemKind::Int8QTy, this);
  isValid &= checkType(getInput(), ElemKind::Int8QTy, this);
  isValid &= checkType(getFilter(), ElemKind::Int8QTy, this);
  isValid &= checkType(
      getBias(), {ElemKind::Int8QTy, ElemKind::Int32QTy, ElemKind::FloatTy},
      this);

  // Check qparam types.
  isValid &= checkType(getFilterOffsets(), ElemKind::Int32ITy, this);
  isValid &= checkType(getFilterScales(), ElemKind::FloatTy, this);
  isValid &= checkType(getBiasOffsets(), ElemKind::Int32ITy, this);
  isValid &= checkType(getBiasScales(), ElemKind::FloatTy, this);

  // Check qparam dimensions.
  isValid &=
      expectCompareTrue("Filter offsets must be a 1D vector",
                        getFilterOffsets().dims().size(), size_t(1), this);
  isValid &=
      expectCompareTrue("Filter scales must be a 1D vector",
                        getFilterScales().dims().size(), size_t(1), this);
  isValid &= expectCompareTrue("Bias offsets must be a 1D vector",
                               getBiasOffsets().dims().size(), size_t(1), this);
  isValid &= expectCompareTrue("Bias scales must be a 1D vector",
                               getBiasScales().dims().size(), size_t(1), this);

  // Check qparam sizes.
  isValid &= expectCompareTrue(
      "There must be one filter offset qparam per output channel",
      getFilterOffsets().dims()[0], dim_t(getResult().dims().back()), this);
  isValid &= expectCompareTrue(
      "There must be one filter scale qparam per output channel",
      getFilterScales().dims()[0], dim_t(getResult().dims().back()), this);
  isValid &= expectCompareTrue(
      "There must be one bias offset qparam per output channel",
      getBiasOffsets().dims()[0], dim_t(getResult().dims().back()), this);
  isValid &= expectCompareTrue(
      "There must be one bias scale qparam per output channel",
      getBiasScales().dims()[0], dim_t(getResult().dims().back()), this);

  return isValid;
}

bool Convolution3DNode::verify() const {
  return verifyConvolution3D(getInput(), getResult(), getFilter(), getBias(),
                             Kernels_, Strides_, Pads_, Group_);
}

bool ConvTransposeNode::verify() const {
  return verifyConvTranspose(getInput(), getResult(), getFilter(), Kernels_,
                             Strides_, Pads_, Group_, Dilation_);
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
  case ElemKind::UInt4FusedFP16QTy:
    return (n - 2 * sizeof(float16_t)) * 2;
  case ElemKind::UInt4FusedQTy:
    return (n - 2 * sizeof(float)) * 2;
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
  switch (getLayout()) {
  case NHWC:
    return verifyPool<ShapeNHWC>(getInput(), getResult(), Kernels_, Strides_,
                                 Pads_,
                                 /* isAvgPool */ false);
  case NCHW:
    return verifyPool<ShapeNCHW>(getInput(), getResult(), Kernels_, Strides_,
                                 Pads_,
                                 /* isAvgPool */ false);
  default: // MaxPool3D is unsupported
    return false;
  }
}

bool AvgPoolNode::verify() const {
  switch (getLayout()) {
  case NHWC:
    return verifyPool<ShapeNHWC>(getInput(), getResult(), Kernels_, Strides_,
                                 Pads_);
  case NCHW:
    return verifyPool<ShapeNCHW>(getInput(), getResult(), Kernels_, Strides_,
                                 Pads_);
  case NTHWC:
    return verifyPool3D<ShapeNTHWC>(getInput(), getResult(), Kernels_, Strides_,
                                    Pads_);
  case NCTHW:
    return verifyPool3D<ShapeNCTHW>(getInput(), getResult(), Kernels_, Strides_,
                                    Pads_);
  default:
    llvm_unreachable("Unsupported format");
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

  switch (getLayout()) {
  case NHWC:
    return isValid &&
           verifyPool<ShapeNHWC>(getGradOfInputNamedInput(),
                                 getGradOfOriginalOutputNamedResult(), Kernels_,
                                 Strides_, Pads_);
  case NCHW:
    return isValid &&
           verifyPool<ShapeNCHW>(getGradOfInputNamedInput(),
                                 getGradOfOriginalOutputNamedResult(), Kernels_,
                                 Strides_, Pads_);
  case NTHWC:
    return isValid &&
           verifyPool3D<ShapeNTHWC>(getGradOfInputNamedInput(),
                                    getGradOfOriginalOutputNamedResult(),
                                    Kernels_, Strides_, Pads_);
  case NCTHW:
    return isValid &&
           verifyPool3D<ShapeNCTHW>(getGradOfInputNamedInput(),
                                    getGradOfOriginalOutputNamedResult(),
                                    Kernels_, Strides_, Pads_);
  default:
    llvm_unreachable("Unsupported format");
  }
}

bool MatMulNode::verify() const {
  auto lhs = getLHS();
  auto rhs = getRHS();
  auto dest = getResult();

  auto LDims = lhs.dims();
  auto RDims = rhs.dims();
  auto DDims = dest.dims();
  bool isValid = expectCompareTrue("LHS input must be 2 dimensional.",
                                   LDims.size(), size_t(2), this);
  isValid &= expectCompareTrue("RHS input must be 2 dimensional.", RDims.size(),
                               size_t(2), this);
  isValid &= expectCompareTrue("Invalid MatMul dimensions", DDims.size(),
                               size_t(2), this);

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
  return verifyActivation(getInput(), getResult());
}

bool SigmoidGradNode::verify() const {
  bool isValid = verifyInputAndGradInputTypes(getInput(),
                                              getGradOfInputNamedInput(), this);
  isValid &= verifyOutputAndGradOutputTypes(
      getOriginalOutputForResult(), getGradOfOriginalOutputNamedResult(), this);
  isValid &= verifyActivation(getGradOfInputNamedInput(),
                              getGradOfOriginalOutputNamedResult());
  return isValid;
}

bool SoftPlusNode::verify() const {
  return verifyActivation(getInput(), getResult());
}

bool SwishNode::verify() const {
  return verifyActivation(getInput(), getResult());
}

bool TanhNode::verify() const {
  return verifyActivation(getInput(), getResult());
}

bool TanhGradNode::verify() const {
  bool isValid = verifyInputAndGradInputTypes(getInput(),
                                              getGradOfInputNamedInput(), this);
  isValid &= verifyOutputAndGradOutputTypes(
      getOriginalOutputForResult(), getGradOfOriginalOutputNamedResult(), this);
  isValid &= verifyActivation(getGradOfInputNamedInput(),
                              getGradOfOriginalOutputNamedResult());
  return isValid;
}

bool LogitNode::verify() const {
  const Node *parent = getResult().getNode();
  bool isValid = checkSameType(getInput(), getResult(), parent);
  isValid &= checkSameShape(getInput(), getResult(), parent);
  isValid &= expectCompareTrue(
      "Clamping parameter eps must be strictly positive", getEpsilon(), 0.0f,
      this, CompareOperatorGreaterThan<float>());
  isValid &=
      expectCompareTrue("Clamping parameter eps must be less than 0.5",
                        getEpsilon(), 0.5f, this, CompareOperatorLess<float>());
  return isValid;
}

bool ExpNode::verify() const {
  const Node *parent = getResult().getNode();
  bool isValid =
      checkSameIsQuantized(getInput().getType(), getResult().getType(), parent);

  if (getInput().getType()->isQuantizedType()) {
    isValid &= checkType(getInput(), getResult().getElementType(),
                         getResult().getNode());
    isValid &= checkSameShape(getInput(), getResult(), parent);
  } else {
    isValid &= checkSameType(getInput(), getResult(), parent);
  }

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

bool LogSoftMaxNode::verify() const {
  return verifyLogSoftMax(getInput(), getResult());
}

bool LogSoftMaxGradNode::verify() const {
  bool isValid = verifyInputAndGradInputTypes(getInput(),
                                              getGradOfInputNamedInput(), this);
  isValid &= ((verifyInputAndGradInputTypes(
                  getSelected(), getGradOfInputNamedSelected(), this))
                  ? 1
                  : 0);
  isValid &= ((verifyOutputAndGradOutputTypes(
                  getOriginalOutputForResult(),
                  getGradOfOriginalOutputNamedResult(), this))
                  ? 1
                  : 0);
  isValid &= ((verifyLogSoftMax(getGradOfInputNamedInput(),
                                getGradOfOriginalOutputNamedResult()))
                  ? 1
                  : 0);
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

bool TouchNode::verify() const { return true; }

bool TraceEventNode::verify() const { return true; }

bool ClipNode::verify() const {
  bool isValid =
      expectCompareTrue("Clip max must be greater than min", getMin(), getMax(),
                        this, CompareOperatorLess<float>());
  if (getInput().getType()->isQuantizedType()) {
    isValid &=
        checkSameIsQuantized(getInput().getType(), getResult().getType(), this);
    isValid &= checkSameShape(getInput(), getResult(), this);
  } else {
    isValid &= checkSameType(getInput(), getResult(), this);
  }
  return isValid;
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
  isValid &= checkTypeIgnoreShape(src, dest, this);
  return isValid;
}

bool BatchNormalizationNode::verify() const {
  return verifyBatchNormalization(getInput(), getResult(), getBias(),
                                  getScale(), getMean(), getVar(), ChannelIdx_);
}

bool InstanceNormalizationNode::verify() const {
  return verifyInstanceNormalization(getInput(), getResult(), getBias(),
                                     getScale(), ChannelIdx_);
}

bool LayerNormalizationNode::verify() const {
  auto dest = getResult();
  auto src = getInput();
  auto scale = getScale();
  auto bias = getBias();

  // Check input and output have same ElemKind.
  bool isValid = checkType(src, dest.getElementType(), this);

  // Check scale and bias have same ElemKind
  isValid &= checkType(bias, scale.getElementType(), this);

  // Check inputs/outputs and scale/bias match shapes.
  isValid &= checkSameShape(src, dest, this);
  isValid &= checkSameShape(scale, bias, this);

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

#define VERIFY_UNARY_LOGICAL(NODE_NAME_)                                       \
  bool NODE_NAME_##Node::verify() const {                                      \
    bool isValid = checkSameShape(getInput(), getResult(), this);              \
    isValid &= checkType(getInput(), ElemKind::BoolTy, this);                  \
    isValid &= checkType(getResult(), ElemKind::BoolTy, this);                 \
    return isValid;                                                            \
  }
VERIFY_UNARY_LOGICAL(Not)
#undef VERIFY_UNARY_LOGICAL

bool SignNode::verify() const {
  if (getResult().getType()->isQuantizedType()) {
    bool isValid = checkSameShape(getInput(), getResult(), this);
    isValid &=
        checkType(getResult(), getInput().getType()->getElementType(), this);
    return isValid;
  }
  return checkSameType(getInput(), getResult(), this);
}

#define VERIFY_BINARY_LOGICAL(NODE_NAME_)                                      \
  bool NODE_NAME_##Node::verify() const {                                      \
    bool isValid = checkSameShape(getLHS(), getResult(), this);                \
    isValid &= checkSameShape(getRHS(), getResult(), this);                    \
    isValid &= checkType(getLHS(), ElemKind::BoolTy, this);                    \
    isValid &= checkType(getRHS(), ElemKind::BoolTy, this);                    \
    isValid &= checkType(getResult(), ElemKind::BoolTy, this);                 \
    return isValid;                                                            \
  }
VERIFY_BINARY_LOGICAL(And)
VERIFY_BINARY_LOGICAL(Or)
VERIFY_BINARY_LOGICAL(Xor)
#undef VERIFY_BINARY_LOGICAL

#define VERIFY_BINARY(NODE_NAME_)                                              \
  bool NODE_NAME_##Node::verify() const {                                      \
    bool isValid = checkSameShape(getLHS(), getResult(), this);                \
    isValid &= checkSameShape(getRHS(), getResult(), this);                    \
    isValid &= checkSameType(getLHS(), getResult(), this);                     \
    isValid &= checkSameType(getRHS(), getResult(), this);                     \
    return isValid;                                                            \
  }
VERIFY_BINARY(BitwiseAnd)
VERIFY_BINARY(BitwiseOr)
VERIFY_BINARY(BitwiseXor)
#undef VERIFY_BINARY

#define VERIFY_UNARY_ARITHMETIC(NODE_NAME_)                                    \
  bool NODE_NAME_##Node::verify() const {                                      \
    return checkSameShape(getInput(), getResult(), this);                      \
  }
VERIFY_UNARY_ARITHMETIC(Abs);
VERIFY_UNARY_ARITHMETIC(Neg);
VERIFY_UNARY_ARITHMETIC(Floor);
VERIFY_UNARY_ARITHMETIC(Ceil);
VERIFY_UNARY_ARITHMETIC(Round);
VERIFY_UNARY_ARITHMETIC(Sqrt);
VERIFY_UNARY_ARITHMETIC(Rsqrt);
VERIFY_UNARY_ARITHMETIC(Reciprocal);
VERIFY_UNARY_ARITHMETIC(Sin);
VERIFY_UNARY_ARITHMETIC(Cos);
VERIFY_UNARY_ARITHMETIC(Erf);
VERIFY_UNARY_ARITHMETIC(Truncate);
VERIFY_UNARY_ARITHMETIC(BitwiseNot);
#undef VERIFY_UNARY_ARITHMETIC

#define VERIFY_ARITHMETIC(NODE_NAME_)                                          \
  bool NODE_NAME_##Node::verify() const {                                      \
    return verifyArithmetic(getLHS(), getRHS(), getResult());                  \
  }
VERIFY_ARITHMETIC(Add);
VERIFY_ARITHMETIC(Mul);
VERIFY_ARITHMETIC(Sub);
VERIFY_ARITHMETIC(Div);
VERIFY_ARITHMETIC(FloorDiv);
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

VERIFY_CMP(CmpEQ)
VERIFY_CMP(CmpNEQ)
VERIFY_CMP(CmpLT)
VERIFY_CMP(CmpLTE)
#undef VERIFY_CMP

//            Trigonometric Ops
#define VERIFY_TRIGONOMERTRIC_OPS(NODE_NAME_)                                  \
  bool NODE_NAME_##Node::verify() const {                                      \
    return checkSameShape(getInput(), getResult(), this);                      \
  }
VERIFY_TRIGONOMERTRIC_OPS(Acos);
VERIFY_TRIGONOMERTRIC_OPS(Asin);
VERIFY_TRIGONOMERTRIC_OPS(Atan);
#undef VERIFY_UNARY_ARITHMETIC

bool FmodNode::verify() const {
  auto res = getResult();
  auto LHS = getLHS();
  auto RHS = getRHS();
  return checkSameShape(res, LHS, res.getNode()) &&
         checkSameShape(LHS, RHS, res.getNode()) &&
         LHS.getElementType() != ElemKind::Int8QTy &&
         RHS.getElementType() != ElemKind::Int8QTy;
}

bool BatchedPairwiseDotProductNode::verify() const {
  auto inputs = getInputs();

  bool isValid = inputs.size() > 1;

  if (isValid) {
    auto firstInput = inputs[0];

    isValid &= firstInput.getElementType() == ElemKind::FloatTy;
    isValid &= firstInput.getType()->dims().size() == 2;

    for (auto &in : inputs) {
      isValid &= checkSameType(in, firstInput, this);
    }

    isValid &= getResult().getElementType() == ElemKind::FloatTy;
    isValid &=
        getResult().getType()->dims()[0] == firstInput.getType()->dims()[0];
    isValid &= getResult().getType()->dims()[1] ==
               inputs.size() * (inputs.size() - 1) / 2;
  }

  return isValid;
}

bool BatchedPairwiseDotProductGradNode::verify() const { return true; }

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

bool BatchedMulNode::verify() const {
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

bool BatchedReduceSumSquareNode::verify() const {
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

// Define verification for Reduction operations.
#define DEFINE_BATCHED_REDUCTION_VERIFICATION(name)                            \
  bool name##Node::verify() const {                                            \
    bool isValid = checkType(getResult(), getBatch().getElementType(), this);  \
    isValid &= expectCompareTrue("Invalid shape", getBatch().dims().size(),    \
                                 size_t(0), this,                              \
                                 CompareOperatorGreaterThan<size_t>());        \
    return isValid;                                                            \
  }

DEFINE_BATCHED_REDUCTION_VERIFICATION(BatchedReduceAdd)
DEFINE_BATCHED_REDUCTION_VERIFICATION(BatchedReduceMean)
DEFINE_BATCHED_REDUCTION_VERIFICATION(BatchedReduceMin)
DEFINE_BATCHED_REDUCTION_VERIFICATION(BatchedReduceMax)
DEFINE_BATCHED_REDUCTION_VERIFICATION(BatchedReduceProd)

#undef DEFINE_BATCHED_REDUCTION_VERIFICATION

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

bool EmbeddingNode::verify() const {
  return verifyEmbedding(getResult(), getWeights(), getIndices());
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
  if (data.getType()->getElementType() == ElemKind::UInt8FusedQTy ||
      data.getType()->getElementType() == ElemKind::UInt4FusedQTy) {
    extraCols = 2 * sizeof(float);
  } else {
    extraCols = 2 * sizeof(float16_t);
  }
  if (useFP16Accumulation) {
    isValid &= expectCompareTrue(
        "Only use FP16 accumulation with FP16 version of RWQ-SLWS.",
        result.getType()->getElementType(), ElemKind::Float16Ty, parent);
  }
  isValid &= checkType(
      indices,
      llvm::ArrayRef<ElemKind>({ElemKind::Int64ITy, ElemKind::Int32ITy}),
      parent);
  // For EmbeddingBagByteRowwiseOffsets lengths are really offsets and
  // can be either Int64ITy or Int64ITy.
  if (isEmbeddingBagByteRowwiseOffsets) {
    isValid &= checkType(
        lengths,
        llvm::ArrayRef<ElemKind>({ElemKind::Int64ITy, ElemKind::Int32ITy}),
        parent);
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
    if (data.getType()->getElementType() == ElemKind::UInt4FusedFP16QTy ||
        data.getType()->getElementType() == ElemKind::UInt4FusedQTy) {
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

bool BatchSparseToDenseNode::verify() const {
  bool isValid = checkType(getResult(), getValues().getElementType(), this);
  isValid &=
      checkType(getIndices(), {ElemKind::Int64ITy, ElemKind::Int32ITy}, this);
  isValid &=
      checkType(getLengths(), {ElemKind::Int64ITy, ElemKind::Int32ITy}, this);
  isValid &= expectCompareTrue("Lengths must be a 1D vector",
                               getLengths().dims().size(), size_t(1), this);
  isValid &= expectCompareTrue("Indices must be a 1D vector",
                               getIndices().dims().size(), size_t(1), this);
  isValid &= expectCompareTrue("Indices and Values must have the same shape",
                               getIndices().dims(), getValues().dims(), this);
  isValid &= expectCompareTrue(
      "The size of Lengths and batches in the result should be the same",
      getLengths().dims()[0], getResult().dims()[0], this);
  isValid &= expectCompareTrue(
      "The second dimension of the result should be equal to dense_last_dim",
      getDenseLastDim(), (unsigned)getResult().dims()[1], this);
  return isValid;
}

bool FillExamplesWithIndicatorNode::verify() const {
  bool isValid = checkType(getResult(), getData().getElementType(), this);
  isValid &= checkType(
      getIndicator(),
      {ElemKind::Int64ITy, ElemKind::Int32ITy, ElemKind::BoolTy}, this);
  isValid &= expectCompareTrue("Indicator must be a 1D vector",
                               getIndicator().dims().size(), size_t(1), this);
  isValid &= expectCompareTrue("Data must have at least one dimension",
                               getData().dims().size(), size_t(1), this,
                               CompareOperatorGreaterEqual<size_t>());
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

bool SparseLabelSplitNode::verify() const {
  bool isValid =
      checkType("Input and output values must be of the same type",
                getLabelValues(), getValues().getElementType(), this);
  isValid &= checkType("Lengths must be of type int32", getLengths(),
                       ElemKind::Int32ITy, this);
  isValid &= checkType("Indices must be of type int64", getIndices(),
                       ElemKind::Int64ITy, this);
  isValid &= checkType("ExampleIds must be of type int32", getExampleIds(),
                       ElemKind::Int32ITy, this);
  isValid &= checkType("GradientOffsetMap must be of type in32",
                       getGradientOffsetMap(), ElemKind::Int32ITy, this);
  isValid &= expectCompareTrue("Lengths must be a 1D vector",
                               getLengths().dims().size(), size_t(1), this);
  isValid &= expectCompareTrue("Indices must be a 1D vector",
                               getIndices().dims().size(), size_t(1), this);
  isValid &= expectCompareTrue("Values must be a 1D vector",
                               getValues().dims().size(), size_t(1), this);
  isValid &= expectCompareTrue("Indices and values must have the same shape",
                               getIndices().dims(), getValues().dims(), this);
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
      "Mapping should cover the whole input quantized range",
      getMapping().dims()[0],
      (dim_t)(getInput().getType()->getQuantizedValueCount()), this);
  isValid &= expectCompareTrue("Mapping and result type must be the same",
                               getMapping().getType()->getElementType(),
                               getResult().getType()->getElementType(), this);
  return isValid;
}

bool LookupTableNode::verify() const {
  bool isValid = true;
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
  if (getInput().getElementType() == ElemKind::UInt8FusedQTy) {
    isValid &= expectCompareTrue("Fused tensors should be 2D",
                                 getInput().dims().size(), size_t(2), this);
    isValid &= expectCompareTrue(
        "Expected space for per-row scale/offset", getInput().dims()[1],
        (dim_t)(2 * sizeof(float)), this, CompareOperatorGreaterThan<dim_t>());
  } else {
    isValid &= checkSameShape(getResult(), getInput(), this);
  }
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

bool CollectRpnProposalsNode::verify() const {
  auto result = getResult();
  auto rois = getRoisIn();
  auto probs = getRoisProbsIn();
  bool isValid = true;

  isValid &= expectCompareTrue("rpnPostNmsTopN should be greater than zero",
                               getRpnPostNmsTopN() > 0, true, this);

  isValid &= expectCompareTrue(
      "RPN min level should be less than or equal to RPN max level",
      getRpnMinLevel() <= getRpnMaxLevel(), true, this);

  dim_t rpnLevels = getRpnMaxLevel() - getRpnMinLevel() + 1;

  isValid &= expectCompareTrue("Invalid number of inputs",
                               rpnLevels == rois.size(), true, this);
  isValid &= expectCompareTrue("Invalid number of inputs",
                               rpnLevels == probs.size(), true, this);

  for (dim_t i = 0; i < rpnLevels; i++) {
    auto roi = rois[i];
    auto prob = probs[i];
    isValid &= checkType(result, roi.getElementType(), this);
    isValid &= checkType(result, prob.getElementType(), this);
    isValid &=
        expectCompareTrue("Rois and result must have same second dimension",
                          roi.dims()[1], result.dims()[1], this);
    isValid &= expectCompareTrue(
        "Rois and respective probability scores must have same first dimension",
        roi.dims()[0], prob.dims()[0], this);
  }

  isValid &=
      expectCompareTrue("Result is capped to rpnPostNmsTopN",
                        result.dims()[0] == getRpnPostNmsTopN(), true, this);

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

  // Check output type.
  isValid &= checkType(
      getResult(),
      llvm::ArrayRef<ElemKind>({ElemKind::Int64ITy, ElemKind::Int32ITy}), this);

  // Check output shape.
  ShapeVector expDstDims =
      reduceDims(getInput().dims(), {getAxis()}, getKeepDims());
  isValid &= expectCompareTrue("Invalid output dims", getResult().dims(),
                               llvm::makeArrayRef(expDstDims), this);
  return isValid;
}

bool ArgMinNode::verify() const {
  bool isValid = true;

  // Check output type.
  isValid &= checkType(
      getResult(),
      llvm::ArrayRef<ElemKind>({ElemKind::Int64ITy, ElemKind::Int32ITy}), this);

  // Check output shape.
  ShapeVector expDstDims =
      reduceDims(getInput().dims(), {getAxis()}, getKeepDims());
  isValid &= expectCompareTrue("Invalid output dims", getResult().dims(),
                               llvm::makeArrayRef(expDstDims), this);
  return isValid;
}

bool VectorNormNode::verify() const {
  bool isValid = true;

  isValid &= expectCompareTrue("Only support Frobenius, p should be 2", getP(),
                               (unsigned)2, this);
  // Check output shape.
  ShapeVector expDstDims = reduceDims(getInput().dims(), {getAxis()}, false);
  isValid &= expectCompareTrue("Invalid output dims", getResult().dims(),
                               llvm::makeArrayRef(expDstDims), this);
  return isValid;
}

bool GaussianFillNode::verify() const {
  auto dest = getResult();
  bool isValid = dest.getElementType() == ElemKind::Float16Ty;
  isValid &= checkSameShape(getInput(), dest, this);
  return isValid;
}

bool DynamicQuantizedFullyConnectedNode::verify() const {
  auto src = getInput();
  auto weights = getWeights();
  auto bias = getBias();
  auto dest = getResult();
  auto isPerBatchElement = getIsPerBatchElement();
  auto isSymmetric = getIsSymmetric();

  bool isValid = expectCompareTrue("Inputs should be 2D tensor",
                                   src.dims().size(), size_t(2), this);
  isValid &= expectCompareTrue(
      "Only per batch quantized input DynQuantizedFC is supported now",
      isPerBatchElement, true, this);
  isValid &= expectCompareTrue(
      "Only symmetric quantized DynQuantizedFC is supported now", isSymmetric,
      true, this);
  isValid &= expectCompareTrue("Weights should be 2D tensor",
                               weights.dims().size(), size_t(2), this);
  isValid &= expectCompareTrue("Result should be 2D tensor", dest.dims().size(),
                               size_t(2), this);
  isValid &= expectCompareTrue("Bias should be 1D tensor", bias.dims().size(),
                               size_t(1), this);

  isValid &= expectCompareTrue("Mismatch on expected source dimension 0",
                               src.dims()[0], dest.dims()[0], this);
  isValid &= expectCompareTrue("Mismatch on expected source dimension 1",
                               src.dims()[1], weights.dims()[0], this);

  isValid &= expectCompareTrue("Inconsistent bias/weights sizes",
                               bias.dims()[0], weights.dims()[1], this);
  isValid &= expectCompareTrue("Inconsistent bias/dest sizes", bias.dims()[0],
                               dest.dims()[1], this);

  return isValid;
}

bool DynamicRowwiseQuantizedFullyConnectedNode::verify() const {
  auto src = getInput();
  auto weights = getWeights();
  auto bias = getBias();
  auto dest = getResult();
  auto scales = getScales();
  auto offsets = getOffsets();
  auto isPerBatchElement = getIsPerBatchElement();
  auto isSymmetric = getIsSymmetric();

  bool isValid = expectCompareTrue("Inputs should be 2D tensor",
                                   src.dims().size(), size_t(2), this);
  isValid &= expectCompareTrue(
      "Only per batch quantized input DynQuantizedFC is supported now",
      isPerBatchElement, true, this);
  isValid &= expectCompareTrue(
      "Only symmetric quantized DynQuantizedFC is supported now", isSymmetric,
      true, this);
  isValid &= expectCompareTrue("Weights should be 2D tensor",
                               weights.dims().size(), size_t(2), this);
  isValid &= expectCompareTrue("Result should be 2D tensor", dest.dims().size(),
                               size_t(2), this);
  isValid &= expectCompareTrue("Bias should be 1D tensor", bias.dims().size(),
                               size_t(1), this);
  isValid &= expectCompareTrue("Offsets should be 1D tensor",
                               offsets.dims().size(), size_t(1), this);
  isValid &= expectCompareTrue("Scales should be 1D tensor",
                               scales.dims().size(), size_t(1), this);

  isValid &= expectCompareTrue("Mismatch on expected source dimension 0",
                               src.dims()[0], dest.dims()[0], this);
  isValid &= expectCompareTrue("Mismatch on expected source dimension 1",
                               src.dims()[1], weights.dims()[0], this);

  isValid &= expectCompareTrue("Inconsistent bias/weights sizes",
                               bias.dims()[0], weights.dims()[1], this);
  isValid &= expectCompareTrue("Inconsistent bias/dest sizes", bias.dims()[0],
                               dest.dims()[1], this);
  isValid &= expectCompareTrue("Inconsistent scales/offsets sizes",
                               scales.dims()[0], offsets.dims()[0], this);
  isValid &= expectCompareTrue("Inconsistent scales/weights sizes",
                               scales.dims()[0], weights.dims()[1], this);

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

bool GatherElementsNode::verify() const {
  bool isValid = checkType(getResult(), getData().getElementType(), this);
  isValid &= checkType(
      getIndices(),
      llvm::ArrayRef<ElemKind>({ElemKind::Int64ITy, ElemKind::Int32ITy}), this);
  isValid &= expectCompareTrue("Mismatching number of dimensions",
                               getResult().dims().size(),
                               getIndices().dims().size(), this);
  return isValid;
}

bool GatherNDNode::verify() const {
  bool isValid = checkType(getResult(), getData().getElementType(), this);
  isValid &= checkType(
      getIndices(),
      llvm::ArrayRef<ElemKind>({ElemKind::Int64ITy, ElemKind::Int32ITy}), this);
  isValid &= expectCompareTrue(
      "Mismatching number of dimensions", getResult().dims().size(),
      getData().dims().size() + getIndices().dims().size() -
          getIndices().dims().back() - 1 - getBatchDims(),
      this);
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
  auto scale = getScale();
  auto result = getResult();
  auto inputDims = input.dims();
  auto outputDims = result.dims();

  bool isValid = checkTypeIgnoreShape(input, result, this);
  isValid &=
      expectCompareTrue("Input size must be greater than 2", inputDims.size(),
                        size_t(2), this, CompareOperatorGreaterThan<size_t>());
  isValid &=
      expectCompareTrue("Output size must be greater than 2", outputDims.size(),
                        size_t(2), this, CompareOperatorGreaterThan<size_t>());
  isValid &= expectCompareTrue("Input size must be equal to the output size",
                               inputDims.size(), outputDims.size(), this);

  for (size_t i = 0, e = scale.size(); i < e; i++) {
    isValid &= expectCompareTrue("Unexpected output",
                                 dim_t(std::floor(inputDims[i] * scale[i])),
                                 outputDims[i], this);
    isValid &= expectCompareTrue("Invalid scale", scale[i], float(0.0), this,
                                 CompareOperatorGreaterThan<float>());
  }

  return isValid;
}

bool ResizeBilinearNode::verify() const {
  auto input = getInput();
  auto scale = getScale();
  auto result = getResult();
  auto inputDims = input.dims();
  auto outputDims = result.dims();

  bool isValid = checkTypeIgnoreShape(input, result, this);
  isValid &= expectCompareTrue("Input must be a 4D tensor", inputDims.size(),
                               size_t(4), this);
  isValid &= expectCompareTrue("Output must be a 4D tensor", outputDims.size(),
                               size_t(4), this);

  for (size_t i = 0, e = scale.size(); i < e; i++) {
    isValid &= expectCompareTrue("Unexpected output",
                                 dim_t(std::floor(inputDims[i] * scale[i])),
                                 outputDims[i], this);
    isValid &= expectCompareTrue("Invalid scale", scale[i], float(0.0), this,
                                 CompareOperatorGreaterThan<float>());
  }

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

bool TFLiteDetectionPostProcessNode::verify() const {
  NodeValue boxes = getBoxes();
  NodeValue scores = getScores();
  NodeValue anchors = getAnchors();

  auto boxesDims = boxes.dims();
  auto scoresDims = scores.dims();
  auto anchorsDims = anchors.dims();

  bool isValid = true;

  // Validate input tensor sizes.
  isValid &= expectCompareTrue("Input boxes must be a 3D tensor!",
                               boxesDims.size(), size_t(3), this);
  isValid &= expectCompareTrue("Input scores must be a 3D tensor!",
                               scoresDims.size(), size_t(3), this);
  isValid &= expectCompareTrue("Input anchors must be a 2D tensor!",
                               anchorsDims.size(), size_t(2), this);
  dim_t numBoxes = boxesDims[1];
  dim_t numTotClasses = scoresDims[2];
  isValid &= expectCompareTrue("Input boxes size invalid!", boxesDims[1],
                               numBoxes, this);
  isValid &= expectCompareTrue("Input boxes size invalid!", boxesDims[2],
                               dim_t(4), this);
  isValid &= expectCompareTrue("Input scores size invalid!", scoresDims[0],
                               boxesDims[0], this);
  isValid &= expectCompareTrue("Input scores size invalid!", scoresDims[1],
                               numBoxes, this);
  isValid &= expectCompareTrue("Input scores size invalid!", scoresDims[2],
                               numTotClasses, this);
  isValid &= expectCompareTrue("Input anchors size invalid!", anchorsDims[0],
                               numBoxes, this);
  isValid &= expectCompareTrue("Input anchors size invalid!", anchorsDims[1],
                               dim_t(4), this);

  // Validate parameters.
  isValid &=
      expectCompareTrue("Invalid IOU threshold!", getIouThreshold(), float(0.0),
                        this, CompareOperatorGreaterThan<float>());
  isValid &=
      expectCompareTrue("Invalid IOU threshold!", getIouThreshold(), float(1.0),
                        this, CompareOperatorLessEqual<float>());
  isValid &=
      expectCompareTrue("Invalid score threshold!", getScoreThreshold(),
                        float(0.0), this, CompareOperatorGreaterThan<float>());
  isValid &=
      expectCompareTrue("Invalid score threshold!", getScoreThreshold(),
                        float(1.0), this, CompareOperatorLessEqual<float>());
  isValid &=
      expectCompareTrue("Invalid number of classes!", dim_t(getNumClasses()),
                        numTotClasses, this, CompareOperatorLessEqual<dim_t>());
  isValid &=
      expectCompareTrue("Invalid max detections!", dim_t(getMaxDetections()),
                        dim_t(0), this, CompareOperatorGreaterThan<dim_t>());
  return isValid;
}

bool AudioSpectrogramNode::verify() const {
  NodeValue input = getInput();
  NodeValue spectrogram = getSpectrogram();
  auto inputLength = input.getType()->size();
  auto windowSize = getWindowSize();
  auto windowStride = getWindowStride();
  auto windowCount = std::floor((inputLength - windowSize) / windowStride) + 1;
  auto fftLen = 1 << (int)std::ceil(std::log2((double)windowSize));

  bool isValid = true;
  isValid &= expectCompareTrue("Input audio is too short for given window size",
                               dim_t(windowCount), dim_t(0), this,
                               CompareOperatorGreaterThan<dim_t>());
  isValid &= expectCompareTrue("Output spectrogram must be a 2D tensor",
                               spectrogram.dims().size(), size_t(2), this);
  isValid &= expectCompareTrue("Output spectrogram size is invalid",
                               spectrogram.dims()[0], dim_t(windowCount), this,
                               CompareOperatorEqual<dim_t>());
  isValid &= expectCompareTrue("Output spectrogram size is invalid",
                               spectrogram.dims()[1], dim_t(fftLen / 2 + 1),
                               this, CompareOperatorEqual<dim_t>());
  return isValid;
}

bool MFCCNode::verify() const {
  NodeValue spectrogram = getSpectrogram();
  NodeValue coefficients = getCoefficients();
  float sampleRate = getSampleRate();
  float lowerFrequency = getLowerFrequency();
  float upperFrequency = getUpperFrequency();
  auto filterBankCount = getFilterBankCount();
  auto numCoefficients = getNumCoefficients();
  auto fftLen = (spectrogram.dims()[1] - 1) * 2;
  int exp;

  bool isValid = true;
  isValid &= expectCompareTrue("Input spectrogram must be a 2D tensor",
                               spectrogram.dims().size(), size_t(2), this);
  isValid &= expectCompareTrue(
      "Input spectrogram size is invalid. Should be of the form 2^N/2+1.",
      std::abs(std::frexp((float)(fftLen), &exp)), float(0.5), this,
      CompareOperatorEqual<float>());
  isValid &= expectCompareTrue("Output coefficients must be a 2D tensor",
                               coefficients.dims().size(), size_t(2), this);
  isValid &= expectCompareTrue("Output coefficients size is invalid",
                               coefficients.dims()[1], dim_t(numCoefficients),
                               this, CompareOperatorEqual<dim_t>());
  isValid &= expectCompareTrue(
      "Number of windows should be same for both input and output",
      spectrogram.dims()[0], coefficients.dims()[0], this,
      CompareOperatorEqual<dim_t>());
  isValid &= expectCompareTrue("Lower frequency should be greater than 0",
                               lowerFrequency, float(0.0), this,
                               CompareOperatorGreaterThan<float>());
  isValid &= expectCompareTrue("Upper frequency should be greater than 0",
                               upperFrequency, float(0.0), this,
                               CompareOperatorGreaterThan<float>());
  isValid &= expectCompareTrue(
      "Upper frequency must be greater than lower frequency", upperFrequency,
      lowerFrequency, this, CompareOperatorGreaterThan<float>());
  isValid &= expectCompareTrue(
      "Upper frequency must be lower than half the sample rate", sampleRate,
      float(2.0 * upperFrequency), this, CompareOperatorGreaterThan<float>());
  isValid &= expectCompareTrue(
      "Number of coefficients should be smaller or equal than the filter bank",
      dim_t(filterBankCount), dim_t(numCoefficients), this,
      CompareOperatorGreaterEqual<dim_t>());
  return isValid;
}

bool ROIAlignNode::verify() const {
  auto featureMap = getFeatureMap();
  auto boxes = getBoxes();
  auto batchIndices = getBatchIndices();
  auto result = getResult();
  auto featureMapDims = featureMap.dims();
  auto boxesDims = boxes.dims();
  auto outputDims = result.dims();

  bool isValid = checkTypeIgnoreShape(featureMap, result, this);
  isValid &= checkTypeIgnoreShape(boxes, result, this);
  isValid &=
      checkType(featureMap, {ElemKind::FloatTy, ElemKind::Float16Ty}, this);
  isValid &= expectCompareTrue("FeatureMap must be a 4D tensor",
                               featureMapDims.size(), size_t(4), this);
  isValid &= expectCompareTrue("Boxes must be a 2D tensor", boxesDims.size(),
                               size_t(2), this);
  isValid &= expectCompareTrue("Output must be a 4D tensor", outputDims.size(),
                               size_t(4), this);
  // If batch size > 1 batch indices must be provided.
  if (featureMapDims[0] > 1) {
    // Caffe2 gets indices using boxes tensor
    bool indicesInBoxesTensor = boxesDims[1] == (getRotated() ? 6 : 5);
    // Onnx requires batchIndices to be valid
    if (!indicesInBoxesTensor) {
      auto batchIndicesDims = batchIndices.dims();
      isValid &= checkType(batchIndices,
                           {ElemKind::Int64ITy, ElemKind::Int32ITy}, this);
      isValid &= expectCompareTrue("BatchIndices must be a 1D tensor",
                                   batchIndicesDims.size(), size_t(1), this);
      isValid &=
          expectCompareTrue("BatchIndices must have same length as Boxes",
                            batchIndicesDims[0], boxesDims[0], this);
    }
  }
  return isValid;
}

bool BBoxTransformNode::verify() const {
  auto rois = getRois();
  auto deltas = getDeltas();
  auto imInfo = getImInfo();
  auto boxOut = getBoxOut();
  auto weights = getWeights();
  auto period = getAngleBoundHi() - getAngleBoundLo();

  auto roisDims = rois.dims();
  auto deltasDims = deltas.dims();
  auto imInfoDims = imInfo.dims();

  bool rotated = getRotated();
  bool angleBoundOn = getAngleBoundOn();
  // BoxDim is of the format
  // <x1, y1, x2, y2, [optional_angle]>
  dim_t expectedBoxDim = rotated ? 5 : 4;

  // Rois row is of the format
  // <[optinal_batch_index], x1, y1, x2, y2, [optional_angle]>
  bool validRoiDim =
      roisDims[1] == expectedBoxDim || roisDims[1] == expectedBoxDim + 1;

  bool isValid = checkTypeIgnoreShape(rois, boxOut, this);
  isValid &= checkSameType(deltas, boxOut, this);
  isValid &= checkTypeIgnoreShape(imInfo, boxOut, this);
  // ROIs can be float32 or float16.
  isValid &= checkType(rois, {ElemKind::FloatTy, ElemKind::Float16Ty}, this);
  isValid &= expectCompareTrue("Rois must be a 2D tensor", roisDims.size(),
                               size_t(2), this);
  isValid &=
      expectCompareTrue("Rois must have with equals boxDim or larger in 1",
                        validRoiDim, true, this);
  isValid &= expectCompareTrue("Deltas must be a 2D tensor", deltasDims.size(),
                               size_t(2), this);
  isValid &= expectCompareTrue("ImInfo must be a 2D tensor", imInfoDims.size(),
                               size_t(2), this);
  isValid &= expectCompareTrue("ImInfo must be a {batch_size, 3} tensor",
                               imInfoDims[1], dim_t(3), this);
  isValid &= expectCompareTrue("Rois and Deltas must have same 0 dimension",
                               roisDims[0], deltasDims[0], this);
  isValid &= expectCompareTrue(
      "Number of rois must be <= 2048 to be represented in FP16.", roisDims[0],
      dim_t(2048), this, CompareOperatorLessEqual<dim_t>());
  isValid &= expectCompareTrue("Deltas must be divisible by box dimensions",
                               deltasDims[1] % expectedBoxDim, dim_t(0), this);
  isValid &= expectCompareTrue("Weights must be a 1D vector of length 4",
                               weights.size(), size_t(4), this);
  if (roisDims[1] == expectedBoxDim) {
    isValid &= expectCompareTrue(
        "The batch size should be 1 if there's no batch index in rois",
        imInfoDims[0], dim_t(1), this);
  }
  if (rotated && angleBoundOn) {
    isValid &= expectCompareTrue(
        "The difference between angleBoundHi and angleBoundLo "
        "should be greater than 0 and divisible by 180",
        period > 0 && period % 180 == 0, true, this);
  }

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

bool NonZeroNode::verify() const {
  return checkType(getCond(), ElemKind::BoolTy, this) &&
         checkType(getResult(), ElemKind::Int32ITy, this);
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

bool GeluNode::verify() const {
  const Node *parent = getResult().getNode();
  return checkSameType(getResult(), getInput(), parent);
}

bool ReluGradNode::verify() const {
  return verifyInputAndGradInputTypes(getInput(), getGradOfInputNamedInput(),
                                      this) &&
         verifyOutputAndGradOutputTypes(getOriginalOutputForResult(),
                                        getGradOfOriginalOutputNamedResult(),
                                        this) &&
         verifyRelu(getGradOfOriginalOutputNamedResult(), getInput());
}

bool LeakyReluNode::verify() const {
  return verifyRelu(getResult(), getInput());
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

bool GemmNode::verify() const {
  NodeValue A = getA();
  NodeValue B = getB();
  NodeValue C = getC();
  NodeValue Y = getResult();
  bool transA = getTransposeA();
  bool transB = getTransposeB();
  const Node *parent = Y.getNode();

  // Check types.
  bool isValid = checkType(B, A.getElementType(), this);
  // Check for element kind of bias
  if (C.getNode()) {
    // Non quantization type check.
    if (A.getElementType() == ElemKind::FloatTy ||
        A.getElementType() == ElemKind::Float16Ty) {
      isValid &= checkType(C, A.getElementType(), parent);
    }
    // Quantization type check.
    if (A.getElementType() == ElemKind::Int8QTy) {
      isValid &= expectCompareTrue("Bias type should be Int8 or Int32 for Gemm",
                                   C.getElementType() == ElemKind::Int8QTy ||
                                       C.getElementType() == ElemKind::Int32QTy,
                                   true, parent);
    }
  }
  isValid &= checkType(Y, A.getElementType(), this);

  // Check shapes.
  isValid &=
      expectCompareTrue("Input A must be 2D", A.dims().size(), size_t(2), this);
  isValid &=
      expectCompareTrue("Input B must be 2D", B.dims().size(), size_t(2), this);
  if (C.getNode()) {
    isValid &=
        expectCompareTrue("Input C must be 1D or 2D", C.dims().size(),
                          size_t(2), this, CompareOperatorLessEqual<size_t>());
  }
  isValid &=
      expectCompareTrue("Output must be 2D", Y.dims().size(), size_t(2), this);
  std::vector<dim_t> dimsA = A.dims();
  std::vector<dim_t> dimsB = B.dims();
  if (transA) {
    dimsA[0] = A.dims()[1];
    dimsA[1] = A.dims()[0];
  }
  if (transB) {
    dimsB[0] = B.dims()[1];
    dimsB[1] = B.dims()[0];
  }
  isValid &= expectCompareTrue("Input A (transposed) dimension 0 size invalid",
                               dimsA[0], Y.dims()[0], this,
                               CompareOperatorEqual<dim_t>());
  isValid &= expectCompareTrue("Input A (transposed) dimension 1 size invalid",
                               dimsA[1], dimsB[0], this,
                               CompareOperatorEqual<dim_t>());
  isValid &= expectCompareTrue("Input B (transposed) dimension 1 size invalid",
                               dimsB[1], Y.dims()[1], this,
                               CompareOperatorEqual<dim_t>());
  if (C.getNode()) {
    if (C.dims().size() == 1) {
      isValid &=
          expectCompareTrue("Input C size invalid", C.dims()[0], Y.dims()[1],
                            this, CompareOperatorEqual<dim_t>());
    } else {
      isValid &=
          expectCompareTrue("Input C dimension 0 size invalid", C.dims()[0],
                            Y.dims()[0], this, CompareOperatorEqual<dim_t>());
      isValid &=
          expectCompareTrue("Input C dimension 1 size invalid", C.dims()[1],
                            Y.dims()[1], this, CompareOperatorEqual<dim_t>());
    }
  }
  return isValid;
}

bool LSTMUnitNode::verify() const {
  bool isValid = true;
  NodeValue C = getC();
  auto cDim = C.dims();
  NodeValue Input = getInput();
  auto inputDim = Input.dims();

  isValid &=
      expectCompareTrue("Input must be 2D", inputDim.size(), size_t(2), this);
  isValid &=
      expectCompareTrue("Cell State must be 2D", cDim.size(), size_t(2), this);
  isValid &= expectCompareTrue("Input dims[1] must be 4 * C dims[1]",
                               inputDim[1], 4 * cDim[1], this);
  isValid &=
      expectCompareTrue("Input dims[0] must be must be the same to C dims[0]",
                        inputDim[0], cDim[0], this);

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
  isValid &= checkType(
      data, {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty},
      this);
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

bool BroadcastNode::verify() const {
  const auto inputDims = getInput().dims();
  const auto axis = getAxis();
  const auto targetDims = getTargetDim();
  bool isValid = (axis + inputDims.size() <= targetDims.size());

  // Iterate over the new shape; if the original shape had a dimension here
  // (when considering the axis) then verify the dimension either matches the
  // new shape (no action taken) or == 1 (broadcast in that direction).
  for (dim_t i = 0; i < targetDims.size(); i++) {
    if (i >= axis && i < inputDims.size() + axis) {
      const int origIdx = i - axis;
      isValid &=
          (inputDims[origIdx] == targetDims[i] || inputDims[origIdx] == 1);
    }
  }
  isValid &= checkTypeIgnoreShape(getInput(), getResult(), this);

  return isValid;
}

bool ModuloNode::verify() const { return getDivisor() >= 1; }

bool ExternalFunctionCallNode::verify() const { return true; }

static bool verifyBatchedUnaryEmbeddingsBags(NodeValue dest, NodeValue weights,
                                             NodeValue indices,
                                             NodeValue offsets) {
  bool isValid = checkType(dest, weights.getElementType(), dest.getNode());
  isValid &= checkType(
      indices,
      llvm::ArrayRef<ElemKind>({ElemKind::Int64ITy, ElemKind::Int32ITy}),
      dest.getNode());
  isValid &= checkType(
      offsets,
      llvm::ArrayRef<ElemKind>({ElemKind::Int64ITy, ElemKind::Int32ITy}),
      dest.getNode());
  isValid &=
      expectCompareTrue("Indices must be a 1D vector", indices.dims().size(),
                        size_t(1), dest.getNode());
  isValid &=
      expectCompareTrue("Offsets must be a 1D vector", offsets.dims().size(),
                        size_t(1), dest.getNode());
  isValid &=
      expectCompareTrue("Weights must be a 3D vector", weights.dims().size(),
                        size_t(3), dest.getNode());
  return isValid;
}

bool BatchedUnaryEmbeddingsBagsNode::verify() const {
  return verifyBatchedUnaryEmbeddingsBags(getResult(), getWeights(),
                                          getIndices(), getOffsets());
}

static bool verifyIntNBitSplitEmbeddingBagsNode(NodeValue dest,
                                                NodeValue devWeights,
                                                NodeValue weightsOffsets,
                                                NodeValue weightsPlacements,
                                                NodeValue weightsTys) {
  bool isValid = checkType(dest, devWeights.getElementType(), dest.getNode());
  isValid &= checkType(devWeights, ElemKind::UInt8ITy, devWeights.getNode());
  isValid &= checkSameShape(weightsPlacements, weightsTys,
                            weightsPlacements.getNode());
  isValid &= checkType(
      weightsOffsets,
      llvm::ArrayRef<ElemKind>({ElemKind::Int64ITy, ElemKind::Int32ITy}),
      dest.getNode());
  return isValid;
}

bool IntNBitSplitEmbeddingBagsNode::verify() const {
  return verifyIntNBitSplitEmbeddingBagsNode(
      getResult(), getDevWeights(), getWeightsOffsets(), getWeightsPlacements(),
      getWeightsTys());
}

static bool verifyIntNBitSplitEmbeddingWeightedBagsNode(
    NodeValue dest, NodeValue devWeights, NodeValue weightsOffsets,
    NodeValue weightsPlacements, NodeValue weightsTys, NodeValue indices,
    NodeValue indiceWeights) {
  bool isValid = checkType(dest, devWeights.getElementType(), dest.getNode());
  isValid &= checkType(devWeights, ElemKind::UInt8ITy, devWeights.getNode());
  isValid &= checkSameShape(weightsPlacements, weightsTys,
                            weightsPlacements.getNode());
  isValid &= checkType(
      weightsOffsets,
      llvm::ArrayRef<ElemKind>({ElemKind::Int64ITy, ElemKind::Int32ITy}),
      dest.getNode());
  isValid &= checkSameShape(indices, indiceWeights, indiceWeights.getNode());
  return isValid;
}

bool IntNBitSplitEmbeddingWeightedBagsNode::verify() const {
  return verifyIntNBitSplitEmbeddingWeightedBagsNode(
      getResult(), getDevWeights(), getWeightsOffsets(), getWeightsPlacements(),
      getWeightsTys(), getIndices(), getIndiceWeight());
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
    os << "NONE";
    break;
  case FusedActivation::RELU:
    os << "RELU";
    break;
  case FusedActivation::CLIP:
    os << "CLIP";
    break;
  case FusedActivation::SIGMOID:
    os << "SIGMOID";
    break;
  case FusedActivation::TANH:
    os << "TANH";
    break;
  case FusedActivation::LEAKY_RELU:
    os << "LEAKY_RELU";
    break;
  }
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, LUTOperator lutOperator) {
  switch (lutOperator) {
  case LUTOperator::NONE:
    os << "NONE";
    break;
  case LUTOperator::RELU:
    os << "RELU";
    break;
  case LUTOperator::CLIP:
    os << "CLIP";
    break;
  case LUTOperator::SIGMOID:
    os << "SIGMOID";
    break;
  case LUTOperator::TANH:
    os << "TANH";
    break;
  case LUTOperator::LEAKY_RELU:
    os << "LEAKY_RELU";
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
  case ConvolutionLayout::NCTHW:
    os << "NCTHW";
    break;
  case ConvolutionLayout::NTHWC:
    os << "NTHWC";
    break;
  default:
    llvm_unreachable("Unknown format");
  }
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, LengthsMode lengthsMode) {
  switch (lengthsMode) {
  case LengthsMode::AllOne:
    os << "AllOne";
    break;
  case LengthsMode::Variable:
    os << "Variable";
    break;
  }
  return os;
}
} // namespace glow
