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

#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Base/Tensor.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "glow/Support/Error.h"

#include "llvm/Support/Casting.h"

#include "caffe2/proto/caffe2.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace glow;
using llvm::cast;

using ArgumentDictionaryTy =
    std::unordered_map<std::string, const caffe2::Argument *>;

namespace glow {
/// Template specialization of loadOperatorName for caffe2.
template <>
std::string
loadOperatorName<caffe2::OperatorDef>(const caffe2::OperatorDef &op) {
  if (op.name().length()) {
    return op.name();
  }
  if (op.output_size() > 0) {
    return op.output(0);
  }
  return op.type();
}

// FIXME: this is a temporary solution for the case when NonZero returns
// -2^31 as the boundary for the returned indices. For examples, currently
// we get this NonZero([0, 1, 1, 0, 0]) -> [1, 2, -2^31, 0, 0], because the
// shapes are static. This function makes sure that the output looks like
// [1, 2, -1, -1, -1] which is more convenient for now.
// The logic is we get [1, 2, -2^31, 0, 0], then we convert to [0, 0, 1, 0, 0]
// by finding negative element, then do cumsum so we get [0, 0, 1, 1, 1],
// then whenever we see 0, we use original value and when we see 1 we use -1,
// so it becomes [1, 2, -1, -1, -1].
Node *fixNonZero(Function *F, Module &mod, const std::string opName,
                 NodeValue node) {
  auto zeroes = F->createSplat(opName + ".fixNZ.zeroes", node.getType(), 0);
  auto floatTy = mod.uniqueType(ElemKind::Float16Ty, node.dims());
  auto minusOnesFloat =
      F->createSplat(opName + ".fixNZ.minusOnesFloat", floatTy, -1);
  auto zeroesFloat = F->createSplat(opName + ".fixNZ.zeroesFloat", floatTy, 0);
  auto onesFloat = F->createSplat(opName + ".fixNZ.onesFloat", floatTy, 1);
  auto nodeFloat = F->createConvertTo(opName + ".fixNZ.float", node, floatTy);

  // If there is a boundary, it will be marked as true.
  auto isNegBool = F->createCmpLT(opName + ".fixNZ.isNegBool", node, zeroes);
  auto isNegFloat = F->createSelect(opName + ".fixNZ.isNegFloat", isNegBool,
                                    onesFloat, zeroesFloat);
  auto isNegInt = F->createConvertTo(opName + ".fixNZ.isNegInt", isNegFloat,
                                     node.getType());

  // After applying cumsum every element before boundary will be 0
  // and starting from boundary will be 1.
  auto cumSum = F->createCumSum(opName + ".fixNZ.cumSum", isNegInt, 0);

  auto isAfterBoundary =
      F->createCmpGT(opName + ".fixNZ.isAfterBoundary", cumSum, zeroes);

  auto withMinusOnesFloat =
      F->createSelect(opName + ".fixNZ.withMinusOnesFloat", isAfterBoundary,
                      minusOnesFloat, nodeFloat);

  auto withMinusOnesInt = F->createConvertTo(
      opName + ".fixNZ.withMinusOnesInt", withMinusOnesFloat, node.getType());

  return withMinusOnesInt;
}
}; // namespace glow

/// Legacy padding modes supported in caffe2.  These are used by MaxPool
/// operators, and are defined in caffe2_legacy.proto in the caffe2 source
/// tree.
enum LegacyPaddingMode { NOTSET, VALID, SAME, CAFFE_LEGACY_POOLING, N_MODES };

/// Creates tensor \p T from the input \p in. Note, there is no data associated
/// with the Tensor. This method makes sure that the tensor is created with the
/// proper shape and element type.
Expected<LoadWeightResult>
Caffe2ModelLoader::createAndSetTensorType(const caffe2::TensorProto &in) {
  std::vector<dim_t> dim;
  for (auto d : in.dims()) {
    if (d == 0) {
      return MAKE_ERR("0 dimension is not supported");
    }
    dim.push_back(d);
  }

  LoadWeightResult result;
  result.t = glow::make_unique<Tensor>();

  if (in.data_type() == caffe2::TensorProto::FLOAT) {
    result.t->reset(ElemKind::FloatTy, dim);
  } else if (in.data_type() == caffe2::TensorProto::FLOAT16) {
    result.t->reset(ElemKind::Float16Ty, dim);
  } else if (in.data_type() == caffe2::TensorProto::INT32) {
    result.t->reset(ElemKind::Int32ITy, dim);
  } else if (in.data_type() == caffe2::TensorProto::INT64) {
    result.t->reset(ElemKind::Int64ITy, dim);
  } else if (in.data_type() == caffe2::TensorProto::UINT8) {
    result.t->reset(ElemKind::UInt8QTy, dim, 1.0, 0);
  } else if (in.data_type() == caffe2::TensorProto::INT8) {
    result.t->reset(ElemKind::Int8QTy, dim, 1.0, 0);
  } else {
    return MAKE_ERR(
        strFormat("FP32/16, Int32/64, Int8/Uint8 are supported. Got type"
                  " %s for tensor %s.",
                  caffe2::TensorProto_DataType_Name(in.data_type()).c_str(),
                  in.name().c_str()));
  }

  return Expected<LoadWeightResult>(std::move(result));
}

Expected<LoadWeightResult>
Caffe2ModelLoader::createAndSetTensorType(const caffe2::QTensorProto &in) {
  std::vector<dim_t> dim;
  for (auto d : in.dims()) {
    if (d == 0) {
      return MAKE_ERR("0 dimension qtensor is not supported");
    }
    dim.push_back(d);
  }

  if (in.axis() != 1) {
    return MAKE_ERR("axis must be 1");
  }

  dim_t qparams = static_cast<dim_t>(in.scales().size());

  RETURN_ERR_IF_NOT(qparams > 0, "No qparams found");

  RETURN_ERR_IF_NOT(in.biases().size() == in.scales().size(),
                    "Found a different number of biases and scales");

  LoadWeightResult result;
  result.t = glow::make_unique<Tensor>();

  float scale = 1.0;
  int32_t offset = 0;

  // If only one set of qparams is present then use them, otherwise load the
  // multiple sets of qparams as separate tensors and use the default qparams
  // for the main tensor result.t.
  // TODO: should we check is_multiparam?
  if (qparams == 1) {
    scale = in.scales(0);
    offset = in.biases(0);
  } else {
    RETURN_ERR_IF_NOT(!originNameToTQPMap_,
                      "Unsupported loading of uniqued qparams for vector of "
                      "scales/biases for " +
                          in.name());
    result.scales = glow::make_unique<Tensor>(ElemKind::FloatTy,
                                              llvm::makeArrayRef({qparams}));
    result.offsets = glow::make_unique<Tensor>(ElemKind::Int32ITy,
                                               llvm::makeArrayRef({qparams}));

    auto scalesH = result.scales->getHandle<float>();
    auto offsetsH = result.offsets->getHandle<int32_t>();
    for (size_t i = 0; i < qparams; ++i) {
      scalesH.raw(i) = in.scales(i);
      offsetsH.raw(i) = in.biases(i);
    }
  }

  if (in.data_type() == caffe2::TensorProto::INT8) {
    TypeRef outTy;
    ASSIGN_VALUE_OR_RETURN_ERR(
        outTy, ProtobufLoader::loadQuantTy(in.name(), ElemKind::Int8QTy, dim,
                                           scale, offset,
                                           /* shiftUInt8ToInt8 */ false));
    result.t->reset(*outTy);
  } else if (in.data_type() == caffe2::TensorProto::UINT8) {
    TypeRef outTy;
    ASSIGN_VALUE_OR_RETURN_ERR(
        outTy, ProtobufLoader::loadQuantTy(in.name(), ElemKind::Int8QTy, dim,
                                           scale, offset));
    result.t->reset(*outTy);
  } else if (in.data_type() == caffe2::TensorProto::INT32) {
    TypeRef outTy;
    ASSIGN_VALUE_OR_RETURN_ERR(
        outTy, ProtobufLoader::loadQuantTy(in.name(), ElemKind::Int32QTy, dim,
                                           scale, offset));
    result.t->reset(*outTy);
  } else {
    return MAKE_ERR("Only int8, uint8, and int32 qtensors are supported");
  }

  return Expected<LoadWeightResult>(std::move(result));
}

/// Translates the protocol buffer node \p op into a random access map.
template <typename T> static ArgumentDictionaryTy loadArgumentMap(const T &t) {
  ArgumentDictionaryTy dict;
  for (auto &arg : t.arg()) {
    dict[arg.name()] = &arg;
  }
  return dict;
}

static Expected<std::vector<unsigned_t>> getPads(ArgumentDictionaryTy &dict) {
  if (dict.count("pad")) {
    int pad;
    ASSIGN_VALUE_OR_RETURN_ERR(pad, loadInt(dict.at("pad")));
    std::vector<unsigned_t> pads(4, pad);
    return pads;
  }
  if (dict.count("pad_t")) {
    std::vector<unsigned_t> pads(4);
    ASSIGN_VALUE_OR_RETURN_ERR(pads[0], loadInt(dict.at("pad_t")));
    RETURN_ERR_IF_NOT(dict.count("pad_l"), "missing pad_l");
    ASSIGN_VALUE_OR_RETURN_ERR(pads[1], loadInt(dict.at("pad_l")));
    RETURN_ERR_IF_NOT(dict.count("pad_b"), "missing pad_b");
    ASSIGN_VALUE_OR_RETURN_ERR(pads[2], loadInt(dict.at("pad_b")));
    RETURN_ERR_IF_NOT(dict.count("pad_r"), "missing pad_r");
    ASSIGN_VALUE_OR_RETURN_ERR(pads[3], loadInt(dict.at("pad_r")));
    return pads;
  }
  if (dict.count("pads")) {
    std::vector<unsigned_t> shape;
    ASSIGN_VALUE_OR_RETURN_ERR(shape, getShape<unsigned_t>(dict["pads"]));
    return shape;
  }
  // Return default value 0 for pads.
  return std::vector<unsigned_t>{0, 0, 0, 0};
}

/// Translates the "order" field of dictionary \p dict into a channel number.
static Expected<unsigned_t> getChannel(ArgumentDictionaryTy &dict) {
  std::string order = "NCHW"; // default
  auto orderIt = dict.find("order");
  if (orderIt != dict.end()) {
    ASSIGN_VALUE_OR_RETURN_ERR(order, loadStr(orderIt->second));
  }
  if (order == "NHWC") {
    return 3;
  } else if (order == "NCHW") {
    return 1;
  }
  return MAKE_ERR("Invalid order field");
}

static Expected<std::vector<unsigned_t>> getSizeHW(ArgumentDictionaryTy &dict,
                                                   const std::string &name,
                                                   unsigned_t defaultValue) {
  if (dict.count(name)) {
    int value;
    ASSIGN_VALUE_OR_RETURN_ERR(value, loadInt(dict[name]));
    std::vector<unsigned_t> result(2, value);
    return result;
  }
  if (dict.count(name + "_h") && dict.count(name + "_w")) {
    std::vector<unsigned_t> result(2);
    ASSIGN_VALUE_OR_RETURN_ERR(result[0], loadInt(dict[name + "_h"]));
    ASSIGN_VALUE_OR_RETURN_ERR(result[1], loadInt(dict[name + "_w"]));
    return result;
  }
  if (dict.count(name + "s")) {
    return getShape<unsigned_t>(dict[name + "s"]);
  }
  return std::vector<unsigned_t>{defaultValue, defaultValue};
}

Expected<caffe2::NetDef>
Caffe2ModelLoader::loadProtoFile(const std::string &filename) {
  std::ifstream ff(filename, std::ios::in | std::ios::binary);
  RETURN_ERR_IF_NOT(ff,
                    strFormat("Can't find the model or network files for %s",
                              filename.c_str()));
  caffe2::NetDef net;

  bool parseNet = false;
  if (filename.find(".pbtxt") != std::string::npos) {
    std::string str((std::istreambuf_iterator<char>(ff)),
                    std::istreambuf_iterator<char>());
    parseNet = google::protobuf::TextFormat::ParseFromString(str, &net);
  } else {
    // Construct and configure a Coded Input Stream
    google::protobuf::io::IstreamInputStream filestr(&ff);
    google::protobuf::io::CodedInputStream codedstr(&filestr);
    // Don't warn about large file sizes.
#if GOOGLE_PROTOBUF_VERSION >= 3002000
    codedstr.SetTotalBytesLimit(MAX_PROTO_SIZE);
#else
    codedstr.SetTotalBytesLimit(MAX_PROTO_SIZE, MAX_PROTO_SIZE);
#endif
    parseNet = net.ParseFromCodedStream(&codedstr);
  }

  RETURN_ERR_IF_NOT(parseNet, "Failed to parse the network descriptor.");
  return net;
}

Expected<caffe2::NetDef> Caffe2ModelLoader::loadProto(const void *c2Model,
                                                      size_t c2ModelSize) {
  google::protobuf::io::ArrayInputStream arrayStream(c2Model, c2ModelSize);
  // Construct and configure a Coded Input Stream
  google::protobuf::io::CodedInputStream codedStream(&arrayStream);

  // Don't warn about large file sizes.
#if GOOGLE_PROTOBUF_VERSION >= 3002000
  codedStream.SetTotalBytesLimit(MAX_PROTO_SIZE);
#else
  codedStream.SetTotalBytesLimit(MAX_PROTO_SIZE, MAX_PROTO_SIZE);
#endif
  caffe2::NetDef MP;
  bool parseNet = MP.ParseFromCodedStream(&codedStream);
  RETURN_ERR_IF_NOT(parseNet, "Failed to parse NetDef");
  return MP;
}

Expected<bool> Caffe2ModelLoader::getBroadcast(ArgumentDictionaryTy &dict) {
  if (!dict.count("broadcast")) {
    return false;
  }
  int broadcast;
  ASSIGN_VALUE_OR_RETURN_ERR(broadcast, loadInt(dict.at("broadcast")));
  return broadcast == 1;
}

bool Caffe2ModelLoader::hasMultidirectionalBroadcast(
    const llvm::StringRef typeName) {
  (void)typeName;
  return false;
}

const std::string Caffe2ModelLoader::opErrMsg(const caffe2::OperatorDef &op,
                                              const std::string &errMsg) {
  const std::string &opName = loadOperatorName(op);
  return strFormat(" [Operator-'%s'] : %s ", opName.c_str(), errMsg.c_str());
}

// Caffe2 PRelu
// https://github.com/pytorch/pytorch/blob/master/caffe2/operators/prelu_op.cc
Error Caffe2ModelLoader::loadPRelu(const caffe2::OperatorDef &op,
                                   ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  NodeValue slope;
  ASSIGN_VALUE_OR_RETURN_ERR(slope, getNodeValueByName(op.input(1)));

  // Do broadcasting.
  auto targetDim = in.dims();
  // Set the axis assuming i/p is of NCHW format.
  int axis = 1;
  auto *finalSlope = G_->createBroadcast(opName, slope, targetDim, axis);
  auto *R = G_->createPRELU(opName, in, finalSlope);
  RETURN_IF_ERR(addNodeAsOutput(op, R));
  return Error::success();
}

Error Caffe2ModelLoader::loadSoftmax(const caffe2::OperatorDef &op,
                                     ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  RETURN_ERR_IF_NOT(
      in.dims().size() >= 2,
      opErrMsg(op,
               strFormat(
                   "SoftMax input dims must be >= 2, but found input dims %zu ",
                   in.dims().size())));

  // Create a constant to store labels to be used in SoftMaxGradNode.
  auto *selected = G_->createSplat(
      opName + ".selected",
      mod_.uniqueType(ElemKind::Int64ITy, {in.dims()[0], 1}), 0.f);

  int axis = 1;
  if (dict.count("axis")) {
    ASSIGN_VALUE_OR_RETURN_ERR(axis,
                               loadAxis<int>(dict["axis"], in.dims().size()));
  }

  auto *FN = G_->createFlatten(opName + ".reshapeInput", in, axis);
  auto *SM = G_->createSoftMax(opName, FN, selected);

  // The output should have the same shape as the original input.
  auto origInDims = in.getType()->dims();
  auto *RN = G_->createReshape(opName + ".reshapeOutput", SM, origInDims);
  RETURN_IF_ERR(addNodeAsOutput(op, RN));
  return Error::success();
}

Error Caffe2ModelLoader::loadConv(const caffe2::OperatorDef &op,
                                  ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  // Load the inputs:
  std::vector<unsigned_t> strides;
  ASSIGN_VALUE_OR_RETURN_ERR(strides, getSizeHW(dict, "stride", 1));
  std::vector<unsigned_t> pads;
  ASSIGN_VALUE_OR_RETURN_ERR(pads, getPads(dict));
  std::vector<unsigned_t> kernels;
  ASSIGN_VALUE_OR_RETURN_ERR(kernels, getSizeHW(dict, "kernel", 0));
  unsigned_t group = 1;
  if (dict.count("group")) {
    ASSIGN_VALUE_OR_RETURN_ERR(group, loadInt(dict["group"]));
  }
  std::string order = "NCHW";
  if (dict.count("order")) {
    ASSIGN_VALUE_OR_RETURN_ERR(order, loadStr(dict["order"]));
  }
  std::vector<unsigned_t> dilations;
  ASSIGN_VALUE_OR_RETURN_ERR(dilations,
                             getDilations(dict, std::vector<unsigned_t>{1, 1}));

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  NodeValue w;
  ASSIGN_VALUE_OR_RETURN_ERR(w, getConstantByName(op.input(1)));

  // Transpose the weights to the right format. Glow expects to read the
  // weights in the format CRSK.
  // C - output_depth, R - filter_height, S - filter_width, K - input_depth.
  // Caffe2 "Conv" op always stores the weight as CKRS.
  w = G_->createTranspose(w.getNode()->getName().str() + "_NHWC", w, NCHW2NHWC,
                          "NHWC");

  // The structure of the conv weights is: CRSK. We take the C, which is the
  // number of filters. We use this value to calculate the size of the bias
  // if it is not specified.
  dim_t depth = w.dims()[0];

  // We expect the input to be NHWC.
  NodeValue finalIn;
  if (order == "NCHW") {
    finalIn = G_->createTranspose(opName, in, NCHW2NHWC)->getResult();
  } else {
    finalIn = in;
  }

  TypeRef finalInType = finalIn.getType();

  // Calculate the size and allocate the output buffer.
  ShapeNHWC idim = ShapeNHWC(finalInType->dims());
  auto outSz = calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides,
                                           pads, dilations);
  std::array<dim_t, 4> outDims = {{idim.n, outSz.first, outSz.second, depth}};

  // Try to find a loaded bias constant.
  NodeValue bias(nullptr);
  if (op.input_size() > 2) {
    const auto &biasName = op.input(2);
    bias = getConstantByNameOrNull(biasName);
  }
  // Construct the bias constant if one wasn't found.
  if (!bias.getNode()) {
    TypeRef bTy = mod_.uniqueType(ElemKind::FloatTy, {depth});
    bias = G_->createSplat(opName + ".bias", bTy, 0.f);
  }

  TypeRef outTy = mod_.uniqueType(ElemKind::FloatTy, outDims);

  Node *node = G_->createConv(opName, finalIn, w, bias, outTy, kernels, strides,
                              pads, group, dilations);
  if (op.type() == "ConvRelu") {
    node = G_->createRELU(opName + ".relu", node);
  }
  if (order == "NCHW") {
    // Transpose the output back.
    node = G_->createTranspose(opName, node, NHWC2NCHW);
  }
  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return Error::success();
}

Error Caffe2ModelLoader::loadConvQuantized(const caffe2::OperatorDef &op,
                                           ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  // Load the inputs:
  std::vector<unsigned_t> strides;
  ASSIGN_VALUE_OR_RETURN_ERR(strides, getSizeHW(dict, "stride", 1));
  std::vector<unsigned_t> pads;
  ASSIGN_VALUE_OR_RETURN_ERR(pads, getPads(dict));
  std::vector<unsigned_t> kernels;
  ASSIGN_VALUE_OR_RETURN_ERR(kernels, getSizeHW(dict, "kernel", 0));
  unsigned_t group = 1;
  if (dict.count("group")) {
    ASSIGN_VALUE_OR_RETURN_ERR(group, loadInt(dict["group"]));
  }
  std::string order = "NCHW";
  if (dict.count("order")) {
    ASSIGN_VALUE_OR_RETURN_ERR(order, loadStr(dict["order"]));
  }
  bool quantizeGroupwise = false;
  if (dict.count("quantize_groupwise")) {
    ASSIGN_VALUE_OR_RETURN_ERR(quantizeGroupwise,
                               loadInt(dict["quantize_groupwise"]));
  }
  std::vector<unsigned_t> dilations;
  ASSIGN_VALUE_OR_RETURN_ERR(dilations,
                             getDilations(dict, std::vector<unsigned_t>{1, 1}));

  // Group quantization only applies if there is more than one group.
  quantizeGroupwise &= group > 1;

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  NodeValue w;
  ASSIGN_VALUE_OR_RETURN_ERR(w, getConstantByName(op.input(1)));

  // Transpose the weights to the right format. Glow expects to read the
  // weights in the format CRSK.
  // C - output_depth, R - filter_height, S - filter_width, K - input_depth.
  // For Caffe2 "Int8Conv" and "Int8ConvRelu", the weights always follows the
  // "order" arg.
  if (order != "NHWC") {
    w = G_->createTranspose(w.getNode()->getName().str() + "_NHWC", w,
                            NCHW2NHWC, "NHWC");
  }

  // The structure of the conv weights is: CRSK. We take the C, which is the
  // number of filters. We use this value to calculate the size of the bias
  // if it is not specified.
  dim_t depth = w.dims()[0];

  // We expect the input to be NHWC.
  NodeValue finalIn;
  if (order == "NCHW") {
    finalIn = G_->createTranspose(opName, in, NCHW2NHWC)->getResult();
  } else {
    finalIn = in;
  }

  TypeRef finalInType = finalIn.getType();

  // Calculate the size and allocate the output buffer.
  ShapeNHWC idim = ShapeNHWC(finalInType->dims());
  auto outSz = calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides,
                                           pads, dilations);
  std::array<dim_t, 4> outDims = {{idim.n, outSz.first, outSz.second, depth}};

  TypeRef outTy;

  RETURN_ERR_IF_NOT(dict.count("Y_zero_point"),
                    opErrMsg(op,
                             "ConvQuantized "
                             "missing zero point for quantized output type"));
  RETURN_ERR_IF_NOT(dict.count("Y_scale"),
                    opErrMsg(op, "ConvQuantized "
                                 "missing Y_scale for quantized output type"));

  // Try to find a loaded bias constant.
  NodeValue bias(nullptr);
  if (op.input_size() > 2) {
    const auto &biasName = op.input(2);
    bias = getConstantByNameOrNull(biasName);
  }
  // Construct the bias constant if one wasn't found.
  if (!bias.getNode()) {
    TypeRef bTy = mod_.uniqueType(ElemKind::Int32QTy, {depth}, 1.0, 0);
    bias = G_->createSplat(opName + "_conv.bias", bTy, 0.f);
  }

  RETURN_ERR_IF_NOT(
      bias.getType()->size() == depth,
      opErrMsg(op, strFormat("Loaded bias tensor of incorrect size %d ",
                             int(bias.getType()->size()))));

  // Construct output type
  ASSIGN_VALUE_OR_RETURN_ERR(
      outTy, loadQuantTy(opName, ElemKind::Int8QTy, outDims, dict));

  Node *node;

  if (quantizeGroupwise) {
    auto wScalesName = strFormat("%s_loaded_scales", op.input(1).c_str());
    auto wOffsetsName = strFormat("%s_loaded_offsets", op.input(1).c_str());
    Constant *wScales;
    Constant *wOffsets;
    ASSIGN_VALUE_OR_RETURN_ERR(wScales, getConstantByName(wScalesName));
    ASSIGN_VALUE_OR_RETURN_ERR(wOffsets, getConstantByName(wOffsetsName));

    // Quantize the filter automatically (only if it is float). The bias is NOT
    // quantized automatically and is left at the disposal of each Backend to
    // quantize it later using custom logic.
    node = G_->createChannelwiseQuantizedConv(
        opName, finalIn, w, bias, wScales, wOffsets, /* biasScales */ nullptr,
        /* biasOffsets */ nullptr, outTy, kernels, strides, pads, group,
        dilations, /* quantizeFilter */ true, /* quantizeBias */ false);
  } else {
    // If the bias isn't quantized for a non group quantized conv, quantize it.
    if (bias.getElementType() == ElemKind::FloatTy) {
      int32_t biasOffset = 0;
      float biasScale = finalInType->getScale() * w.getType()->getScale();

      auto biasTy = mod_.uniqueType(ElemKind::Int32QTy, bias.dims(), biasScale,
                                    biasOffset);
      bias = G_->createQuantize(opName + "_conv.bias", bias, biasTy);
    }

    node = G_->createConv(opName, finalIn, w, bias, outTy, kernels, strides,
                          pads, group, dilations);
  }

  if (op.type() == "Int8ConvRelu") {
    node = G_->createRELU(opName + ".relu", node);
  }

  if (order == "NCHW") {
    // Transpose the output back.
    node = G_->createTranspose(opName, node, NHWC2NCHW);
  }
  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return Error::success();
}

Error Caffe2ModelLoader::loadLayerNorm(const caffe2::OperatorDef &op,
                                       ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  unsigned_t axis = 1; // Caffe2 default.
  if (dict.count("axis")) {
    ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict["axis"]));
    ASSIGN_VALUE_OR_RETURN_ERR(axis,
                               getPositiveAxis<int>(axis, in.dims().size()));
  }

  // Feature shape is based on the input dims, from the axis to the end.
  ShapeVector featDims;
  for (dim_t i = axis, e = in.dims().size(); i < e; ++i) {
    featDims.push_back(in.dims()[i]);
  }
  TypeRef featTy = mod_.uniqueTypeWithNewShape(in.getType(), featDims);

  NodeValue weight, bias;
  if (op.input_size() > 1) {
    RETURN_ERR_IF_NOT(op.input_size() == 3,
                      opErrMsg(op, "Must have both weight and bias"));

    ASSIGN_VALUE_OR_RETURN_ERR(weight, getNodeValueByName(op.input(1)));
    RETURN_ERR_IF_NOT(weight.getType() == featTy,
                      opErrMsg(op, "Invalid weight shape"));

    ASSIGN_VALUE_OR_RETURN_ERR(bias, getNodeValueByName(op.input(2)));
    RETURN_ERR_IF_NOT(bias.getType() == featTy,
                      opErrMsg(op, "Invalid bias shape"));
  } else {
    // Caffe2 default to use weight 1 and bias 0.
    weight = G_->createSplat(opName + "_weight_ones", featTy, 1.0)->getResult();
    bias = G_->createSplat(opName + "_bias_zeros", featTy, 0.0)->getResult();
  }

  float eps = 0.001; // Caffe2 default.
  if (dict.count("epsilon")) {
    ASSIGN_VALUE_OR_RETURN_ERR(eps, loadFloat(dict["epsilon"]));
  }

  LayerNormalizationNode *node =
      G_->createLayerNormalization(opName, in.getType(), in, weight, bias, eps);

  // We only support one output for LayoutNorm. Ignoring the
  // rest of the outputs.
  RETURN_IF_ERR(addNodeAsOutput(op, node, /* numOutputs */ 1));

  return Error::success();
}

Expected<bool> Caffe2ModelLoader::foldOperator(const caffe2::OperatorDef &op) {
  const unsigned numInputs = op.input_size();
  const std::string &typeName = op.type();
  llvm::SmallVector<NodeValue, 4> inputs;
  inputs.reserve(numInputs);
  for (unsigned i = 0; i < numInputs; i++) {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(i)));
    inputs.push_back(in);
  }

  if (!isConstantFoldable(inputs, typeName)) {
    return false;
  }

  // Create a temporary lightweight loader to construct function representing
  // current Op, and then constant fold the function using Interp backend.
  Function *tmpF = mod_.createFunction("eval_const_fold__");
  Caffe2ModelLoader tmpLoader(*tmpF, nullptr);
  bool foldStatus =
      !ERR_TO_BOOL(constantFoldInLoader<Caffe2ModelLoader, caffe2::OperatorDef>(
                       tmpF, tmpLoader, this, op),
                   /* log */ false);
  mod_.eraseFunction(tmpF);
  return foldStatus;
}

Error Caffe2ModelLoader::loadConvTranspose(const caffe2::OperatorDef &op,
                                           ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  // Load the inputs:
  std::vector<unsigned_t> strides;
  ASSIGN_VALUE_OR_RETURN_ERR(strides, getSizeHW(dict, "stride", 1));
  std::vector<unsigned_t> pads;
  ASSIGN_VALUE_OR_RETURN_ERR(pads, getPads(dict));
  std::vector<unsigned_t> kernels;
  ASSIGN_VALUE_OR_RETURN_ERR(kernels, getSizeHW(dict, "kernel", 0));
  unsigned_t group = 1;
  if (dict.count("group")) {
    ASSIGN_VALUE_OR_RETURN_ERR(group, loadInt(dict["group"]));
  }
  std::string order = "NCHW";
  if (dict.count("order")) {
    ASSIGN_VALUE_OR_RETURN_ERR(order, loadStr(dict["order"]));
  }
  std::vector<unsigned_t> dilations;
  ASSIGN_VALUE_OR_RETURN_ERR(dilations,
                             getDilations(dict, std::vector<unsigned_t>{1, 1}));

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  NodeValue weight;
  ASSIGN_VALUE_OR_RETURN_ERR(weight, getConstantByName(op.input(1)));

  // Transpose the weights to the right format. Glow expects to read the
  // weights in the format CRSK.
  // C - output_depth, R - filter_height, S - filter_width, K - input_depth.
  // Caffe2 "ConvTranspose" op always stores the weight as KCRS.
  weight = G_->createTranspose(weight.getNode()->getName().str() + "_NHWC",
                               weight, CNHW2NHWC, "NHWC");

  // The structure of the conv weights is: CRSK. We take the C, which is the
  // number of filters. We use this value to calculate the size of the bias
  // if it is not specified.
  dim_t depth = weight.dims()[0];

  // We expect the input to be NHWC.
  NodeValue finalIn;
  if (order == "NCHW") {
    finalIn = G_->createTranspose(opName, in, NCHW2NHWC)->getResult();
  } else {
    finalIn = in;
  }

  TypeRef finalInType = finalIn.getType();

  // Calculate the size and allocate the output buffer.
  ShapeNHWC idim = ShapeNHWC(finalInType->dims());
  auto outSz = calculateConvTransposeOutputDims(idim.h, idim.w, kernels,
                                                strides, pads, dilations);
  std::array<dim_t, 4> outDims = {{idim.n, outSz.first, outSz.second, depth}};

  // Try to find a loaded bias constant.
  NodeValue bias(nullptr);
  if (op.input_size() > 2) {
    const auto &biasName = op.input(2);
    bias = getConstantByNameOrNull(biasName);
  }
  // Construct the bias constant if one wasn't found.
  if (!bias.getNode()) {
    TypeRef bTy = mod_.uniqueType(ElemKind::FloatTy, {depth});
    bias = G_->createSplat(opName + "_conv.bias", bTy, 0.f);
  }

  TypeRef outTy = mod_.uniqueType(ElemKind::FloatTy, outDims);

  Node *node =
      G_->createConvTranspose(opName, finalIn, weight, bias, outTy, kernels,
                              strides, pads, group, dilations);

  if (order == "NCHW") {
    // Transpose the output back.
    node = G_->createTranspose(opName, node, NHWC2NCHW);
  }
  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return Error::success();
}

Error Caffe2ModelLoader::loadOperator(const caffe2::OperatorDef &op) {
  ArgumentDictionaryTy dict = loadArgumentMap(op);
  const std::string &typeName = op.type();
  mod_.registerOriginalName(op.name());

  // Check if operator is supported in parent class, CommonOperatorLoader.
  bool loadCommonOperatorSuccess;
  ASSIGN_VALUE_OR_RETURN_ERR(loadCommonOperatorSuccess,
                             tryLoadCommonOperator(typeName, op, dict));
  if (loadCommonOperatorSuccess) {
    return Error::success();
  }
  const std::string &opName = loadOperatorName(op);

  if (typeName == "Gelu") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    Node *node = G_->createGelu(opName, in);

    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "Conv" || typeName == "ConvRelu") {
    return loadConv(op, dict);
  }

  if (typeName == "Softmax") {
    return loadSoftmax(op, dict);
  }

  if (typeName == "PRelu") {
    return loadPRelu(op, dict);
  }

  if (typeName == "ConvTranspose") {
    return loadConvTranspose(op, dict);
  }

  if (typeName == "Int8Conv" || typeName == "Int8ConvRelu") {
    return loadConvQuantized(op, dict);
  }

  if (typeName == "LayerNorm") {
    return loadLayerNorm(op, dict);
  }

  if (typeName == "Int8SumRelu") {
    RETURN_ERR_IF_NOT(op.input_size() == 2,
                      opErrMsg(op, "Only Sum of 2 inputs is supported."));
    RETURN_ERR_IF_NOT(
        dict.count("Y_zero_point"),
        opErrMsg(op, "missing zero point for quantized outout type"));
    RETURN_ERR_IF_NOT(
        dict.count("Y_scale"),
        opErrMsg(op, "missing Y_scale for quantized output type"));
    NodeValue in0;
    ASSIGN_VALUE_OR_RETURN_ERR(in0, getNodeValueByName(op.input(0)));
    NodeValue in1;
    ASSIGN_VALUE_OR_RETURN_ERR(in1, getNodeValueByName(op.input(1)));
    auto outDims = in0.getType()->dims();
    TypeRef outTy;
    ASSIGN_VALUE_OR_RETURN_ERR(
        outTy, loadQuantTy(opName, ElemKind::Int8QTy, outDims, dict));
    auto *add = G_->createAdd(opName + ".sum", outTy, in0, in1);
    auto *relu = G_->createRELU(opName + ".relu", add);
    RETURN_IF_ERR(addNodeAsOutput(op, relu));
    return Error::success();
  }

  if (typeName == "Int8Relu") {
    RETURN_ERR_IF_NOT(op.input_size() == 1,
                      opErrMsg(op, "Only one input is supported."));
    RETURN_ERR_IF_NOT(
        dict.count("Y_zero_point"),
        opErrMsg(op, "missing zero point for quantized outout type"));
    RETURN_ERR_IF_NOT(
        dict.count("Y_scale"),
        opErrMsg(op, "missing Y_scale for quantized output type"));
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto outDims = in.getType()->dims();
    TypeRef outTy;
    ASSIGN_VALUE_OR_RETURN_ERR(
        outTy, loadQuantTy(opName, ElemKind::Int8QTy, outDims, dict));
    auto *relu = G_->createRELU(opName, in, outTy);
    RETURN_IF_ERR(addNodeAsOutput(op, relu));
    return Error::success();
  }

  if (typeName == "Int8Quantize") {
    RETURN_ERR_IF_NOT(
        op.input_size() == 1,
        opErrMsg(op, "Glow only supports Int8Quantize with 1 input"));
    RETURN_ERR_IF_NOT(
        dict.count("Y_zero_point"),
        opErrMsg(op, "missing zero point for quantized output type"));
    RETURN_ERR_IF_NOT(
        dict.count("Y_scale"),
        opErrMsg(op, "missing Y_scale for quantized output type"));
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto outDims = in.getType()->dims();
    TypeRef outTy;
    ASSIGN_VALUE_OR_RETURN_ERR(
        outTy, loadQuantTy(opName, ElemKind::Int8QTy, outDims, dict));
    Node *N = G_->createQuantize(opName, in, outTy);
    RETURN_IF_ERR(addNodeAsOutput(op, N));
    return Error::success();
  }

  if (typeName == "Int8Dequantize") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto *node = G_->createDequantize(opName, in, ElemKind::FloatTy);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "MaxPool" || typeName == "AveragePool" ||
      typeName == "Int8MaxPool" || typeName == "Int8AveragePool") {
    // Load the inputs:
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    std::vector<unsigned_t> strides;
    ASSIGN_VALUE_OR_RETURN_ERR(strides, getSizeHW(dict, "stride", 1));
    std::vector<unsigned_t> kernels;
    ASSIGN_VALUE_OR_RETURN_ERR(kernels, getSizeHW(dict, "kernel", 0));
    std::vector<unsigned_t> pads;
    ASSIGN_VALUE_OR_RETURN_ERR(pads, getPads(dict));
    bool countIncludePads;
    ASSIGN_VALUE_OR_RETURN_ERR(
        countIncludePads, getCountIncludePads(dict, /* defaultValue */ true));
    std::string order = "NCHW";
    if (dict.count("order")) {
      ASSIGN_VALUE_OR_RETURN_ERR(order, loadStr(dict["order"]));
    }
    // We expect the input to be NHWC.
    NodeValue finalIn;
    if (order == "NCHW") {
      finalIn = G_->createTranspose(opName, in, NCHW2NHWC)->getResult();
    } else {
      finalIn = in;
    }

    // If 'global_pooling' is set then the operation will pool over the size
    // of the input by doing: kernels = {height, width}.
    if (dict.count("global_pooling")) {
      auto Ty = in.getType();
      kernels[0] = Ty->dims()[2];
      kernels[1] = Ty->dims()[3];
    }

    // Check the padding style.
    if (dict.count("legacy_pad")) {
      int mode;
      ASSIGN_VALUE_OR_RETURN_ERR(mode, loadInt(dict["legacy_pad"]));
      // Caffe1 (legacy) rounded-up and Caffe2 rounds down.
      // This style is deprecated according to caffe2's caffe2_legacy.proto
      // definition.
      if (static_cast<LegacyPaddingMode>(mode) ==
          LegacyPaddingMode::CAFFE_LEGACY_POOLING) {
        return MAKE_ERR(opErrMsg(op,
                                 "MaxPool nodes with legacy caffe padding are "
                                 "deprecated and not supported."));
      }
    }

    Node *node = nullptr;

    if (typeName == "Int8MaxPool" || typeName == "Int8AveragePool") {
      // Create the node with quantized type.
      RETURN_ERR_IF_NOT(
          dict.count("Y_zero_point"),
          opErrMsg(op, "missing zero point for quantized output type"));
      RETURN_ERR_IF_NOT(
          dict.count("Y_scale"),
          opErrMsg(op, "missing Y_scale for quantized output type"));

      TypeRef finalInType = finalIn.getType();
      ShapeNHWC idim = ShapeNHWC(finalInType->dims());
      auto outSz =
          calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides, pads);
      std::array<dim_t, 4> outDims = {
          {idim.n, outSz.first, outSz.second, idim.c}};
      if (typeName == "Int8MaxPool") {
        // Int8Maxpool output quantization should be same as the input, so
        // just ignore the given params.
        node = G_->createMaxPool(opName, finalIn, kernels, strides, pads);
      } else {
        TypeRef outTy;
        ASSIGN_VALUE_OR_RETURN_ERR(
            outTy, loadQuantTy(opName, ElemKind::Int8QTy, outDims, dict));
        node = G_->createAvgPool(opName, finalIn, outTy, kernels, strides, pads,
                                 NHWC, countIncludePads);
      }
    } else if (typeName == "MaxPool") {
      node = G_->createMaxPool(opName, finalIn, kernels, strides, pads);
    } else {
      node = G_->createAvgPool(opName, finalIn, kernels, strides, pads, NHWC,
                               countIncludePads);
    }
    if (order == "NCHW") {
      unsigned resIdx = 0;
      if (llvm::isa<MaxPoolNode>(node)) {
        resIdx = MaxPoolNode::ResultIdx;
      } else if (llvm::isa<AvgPoolNode>(node)) {
        resIdx = AvgPoolNode::ResultIdx;
      } else {
        return MAKE_ERR("Expected either Max or Avg Pool.");
      }
      // Transpose the output back.
      node = G_->createTranspose(opName, node->getNthResult(resIdx), NHWC2NCHW);
    }
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "SpatialBN") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    Constant *scale;
    ASSIGN_VALUE_OR_RETURN_ERR(scale, getConstantByName(op.input(1)));
    Constant *bias;
    ASSIGN_VALUE_OR_RETURN_ERR(bias, getConstantByName(op.input(2)));
    Constant *mean;
    ASSIGN_VALUE_OR_RETURN_ERR(mean, getConstantByName(op.input(3)));
    Constant *var;
    ASSIGN_VALUE_OR_RETURN_ERR(var, getConstantByName(op.input(4)));
    float epsilon = 1e-5f; // default
    auto epsilonIt = dict.find("epsilon");
    if (epsilonIt != dict.end()) {
      ASSIGN_VALUE_OR_RETURN_ERR(epsilon, loadFloat(epsilonIt->second));
    }

    unsigned_t channel;
    ASSIGN_VALUE_OR_RETURN_ERR(channel, getChannel(dict));
    auto *node = G_->createBatchNormalization(
        opName, in.getType(), in, bias, scale, mean, var, channel, epsilon);

    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "Bucketize") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    RETURN_ERR_IF_NOT(
        dict.count("boundaries"),
        opErrMsg(op, "Bucketize: Expected a boundaries member vector"));
    std::vector<float> boundaries;
    ASSIGN_VALUE_OR_RETURN_ERR(boundaries, getFloats(dict["boundaries"]));
    auto *node = G_->createBucketizeNode(opName, in, boundaries);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "ResizeNearest") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

    std::string order = "NCHW";
    if (dict.count("order")) {
      ASSIGN_VALUE_OR_RETURN_ERR(order, loadStr(dict["order"]));
    }
    // We expect the input to be NHWC.
    NodeValue finalIn;
    if (order == "NCHW") {
      finalIn = G_->createTranspose(opName, in, NCHW2NHWC)->getResult();
    } else {
      finalIn = in;
    }

    float heightScale;
    ASSIGN_VALUE_OR_RETURN_ERR(heightScale, loadFloat(dict["height_scale"]));
    float widthScale;
    ASSIGN_VALUE_OR_RETURN_ERR(widthScale, loadFloat(dict["width_scale"]));

    std::vector<float> scales;
    scales.push_back(1.0f);
    scales.push_back(heightScale);
    scales.push_back(widthScale);
    scales.push_back(1.0f);

    auto *node = G_->createResizeNearest(opName, finalIn, scales);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "Concat") {
    const unsigned numInputs = op.input_size();
    llvm::SmallVector<NodeValue, 4> inputs;
    inputs.reserve(numInputs);
    for (unsigned i = 0; i < numInputs; i++) {
      NodeValue in;
      ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(i)));
      inputs.push_back(std::move(in));
    }

    // If axis exists it takes priority over channel.
    unsigned_t channel;
    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(channel, loadInt(dict["axis"]));
    } else {
      ASSIGN_VALUE_OR_RETURN_ERR(channel, getChannel(dict));
    }

    unsigned_t addAxis = 0;
    if (dict.count("add_axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(addAxis, loadInt(dict["add_axis"]));
    }

    Node *node{nullptr};

    if (addAxis) {
      // When add axis is used, this means we have to add a new dimension
      // before the axis, instead of merging on the axis.
      std::vector<dim_t> outputDims = inputs[0].dims();

      if (channel < outputDims.size()) {
        unsigned i = 0;
        for (const auto &input : inputs) {
          RETURN_ERR_IF_NOT(
              outputDims[channel] == input.dims()[channel],
              opErrMsg(op,
                       strFormat("inputs need all to have the same dims for "
                                 "concat with add_axis: input 0 (%s) vs "
                                 "input %u (%s), %u vs %u, channel = %u",
                                 op.input(0).c_str(), i, op.input(i).c_str(),
                                 static_cast<unsigned>(outputDims[channel]),
                                 static_cast<unsigned>(input.dims()[channel]),
                                 channel)));
          ++i;
        }
        outputDims.insert(outputDims.begin() + channel, numInputs);
        node = G_->createConcat(opName, inputs, channel);
        node = G_->createReshape(opName, node, outputDims);
      } else if (channel == outputDims.size()) {
        // We convert inputs into 2D arrays with single columns, thus the
        // number of rows will be equal to the product of all original dims.
        // Every converted input will look like a vertical line of numbers.
        const auto flatVerticalShape = flattenCdr(inputs[0].dims(), channel);
        llvm::SmallVector<NodeValue, 4> verticalInputs;
        for (auto &input : inputs) {
          verticalInputs.push_back(G_->createReshape(
              opName, input,
              {flatVerticalShape.first, flatVerticalShape.second}));
        }

        // We glue together the vertical lines, so, the number of columns
        // becomes equal to the number of original inputs.
        node = G_->createConcat(opName, verticalInputs, 1);

        // Reshape to convert to desired shape.
        outputDims.push_back(numInputs);
        node = G_->createReshape(opName, node, outputDims);
      } else {
        return MAKE_ERR(opErrMsg(
            op, strFormat("Invalid input: channel (=%u) > number of dims (=%u)",
                          channel, static_cast<unsigned>(outputDims.size()))));
      }
    } else {
      // In normal case (i.e. when we are not adding a new dimension)
      // plain createConcat() would suffice.
      node = G_->createConcat(opName, inputs, channel);
    }

    // If we add the axis then node is a Reshape, otherwise it should be
    // Concat.
    RETURN_ERR_IF_NOT(
        llvm::isa<ConcatNode>(node) || llvm::isa<ReshapeNode>(node),
        opErrMsg(op,
                 "Internal error: Node should either be a Concat or Reshape."));
    NodeValue finalNode = llvm::isa<ConcatNode>(node)
                              ? NodeValue(node, ConcatNode::ResultIdx)
                              : NodeValue(node, ReshapeNode::ResultIdx);
    nodeValueByName_[op.output(0)] = finalNode;
    // Concat may have a second output in Caffe2 (split_info), but we don't
    // use it for inference
    return Error::success();
  }

  if (typeName == "FC" || typeName == "FCTransposed" || typeName == "Int8FC" ||
      typeName == "FbFCPacked") {
    RETURN_ERR_IF_NOT(op.input_size() == 3,
                      "Glow only suports FC with 3 inputs");
    // Load the inputs:
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

    auto originalInputDims = in.getType()->dims();

    size_t axis = 1;
    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict["axis"]));
    }

    // Load weights.
    unsigned_t axis_w = 1;
    if (dict.count("axis_w")) {
      ASSIGN_VALUE_OR_RETURN_ERR(axis_w, loadInt(dict["axis_w"]));
    }

    NodeValue W;
    if (hasConstantByName(op.input(1))) {
      ASSIGN_VALUE_OR_RETURN_ERR(W, getConstantByName(op.input(1)));
    } else {
      ASSIGN_VALUE_OR_RETURN_ERR(W, getNodeValueByName(op.input(1)));
    }

    // Caffe2 stores the transposed W matrix. In here we first coerce W to a
    // 2D matrix size if necessary and then transpose it back.
    auto wDims = flattenCdr(W.dims(), axis_w);
    if (W.dims().size() > 2) {
      W = G_->createReshape(W.getNode()->getName(), W,
                            {wDims.first, wDims.second});
    }

    if (typeName == "FC" || typeName == "Int8FC" || typeName == "FbFCPacked") {
      W = G_->createTranspose(W.getNode()->getName(), W, {1, 0});
    }

    NodeValue B;
    if (hasConstantByName(op.input(2))) {
      ASSIGN_VALUE_OR_RETURN_ERR(B, getConstantByName(op.input(2)));
    } else {
      ASSIGN_VALUE_OR_RETURN_ERR(B, getNodeValueByName(op.input(2)));
    }

    Node *node = nullptr;
    if (typeName == "Int8FC") {
      // Create a node with quantized type.
      auto outputDims = flattenCdr(in.dims(), axis);
      TypeRef outTy;
      ASSIGN_VALUE_OR_RETURN_ERR(
          outTy, loadQuantTy(opName, ElemKind::Int8QTy,
                             {outputDims.first, B.dims()[0]}, dict));
      int dequantizeOutput = 0;
      if (dict.count("dequantize_output")) {
        ASSIGN_VALUE_OR_RETURN_ERR(dequantizeOutput,
                                   loadInt(dict["dequantize_output"]));
      }
      if (dequantizeOutput == 1) {
        node = G_->createDynamicQuantizedFullyConnected(opName, in, W, B);
      } else {
        node = G_->createFullyConnected(opName, in, W, B, outTy, axis);
      }
    } else if (typeName == "FbFCPacked") {
      RETURN_ERR_IF_NOT(W.getElementType() == ElemKind::Float16Ty,
                        opErrMsg(op, "Expected float16 weights."));
      auto fp16InputType =
          mod_.uniqueType(ElemKind::Float16Ty, in.getType()->dims());
      in = G_->createConvertTo(opName + ".ConvertInput", in, fp16InputType);

      auto fp16BiasType = mod_.uniqueType(ElemKind::Float16Ty, B.dims());
      auto *fp16Bias =
          G_->createConvertTo(opName + ".ConvertBias", B, fp16BiasType);

      auto outputDims = flattenCdr(in.dims(), axis);
      TypeRef OT =
          mod_.uniqueType(ElemKind::Float16Ty, {outputDims.first, B.dims()[0]});
      auto fc = G_->createFullyConnected(opName, in, W, fp16Bias, OT, axis);
      auto outputType =
          mod_.uniqueType(ElemKind::FloatTy, fc->getResult().dims());
      node = G_->createConvertTo(opName + ".ConvertOutput", fc, outputType);
    } else {
      auto outputDims = flattenCdr(in.dims(), axis);
      TypeRef outputType =
          mod_.uniqueType(ElemKind::FloatTy, {outputDims.first, B.dims()[0]});
      node = G_->createFullyConnected(opName, in, W, B, outputType, axis);
    }

    // If number of original input dims is greater than 2, expand the output
    // dims back with the same axis.
    if (axis != 1) {
      llvm::SmallVector<dim_t, max_tensor_dimensions> reshapeDims;
      size_t totalReshapeSize = 1;
      for (size_t i = 0; i < axis; ++i) {
        auto d = originalInputDims[i];
        reshapeDims.push_back(d);
        totalReshapeSize *= static_cast<dim_t>(d);
      }

      size_t finalDim = typeName == "FCTransposed" ? wDims.second : wDims.first;

      reshapeDims.push_back(finalDim);
      totalReshapeSize *= finalDim;

      size_t totalOriginalOutputSize = node->getNthResult(0).getType()->size();
      RETURN_ERR_IF_NOT(
          totalReshapeSize == totalOriginalOutputSize,
          opErrMsg(op, strFormat("Cannot reshape from size %lu to size %lu",
                                 totalOriginalOutputSize, totalReshapeSize)));

      node = G_->createReshape(opName + ".fc.out", node, reshapeDims);
    }

    // Save the outputs:
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "ChannelShuffle") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

    size_t group;
    ASSIGN_VALUE_OR_RETURN_ERR(group, loadInt(dict["group"]));
    size_t kernel;
    ASSIGN_VALUE_OR_RETURN_ERR(kernel, loadInt(dict["kernel"]));

    Node *node = G_->createChannelShuffle(opName, in, group, kernel);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "Squeeze") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    std::vector<dim_t> dims;
    ASSIGN_VALUE_OR_RETURN_ERR(dims, getShape<dim_t>(dict["dims"]));
    Node *node = G_->createSqueeze(opName, in, dims);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "Log") {
    // Load the inputs:
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    // Create the log:
    auto *R = G_->createLog(opName, in);
    RETURN_IF_ERR(addNodeAsOutput(op, R));
    return Error::success();
  }

  if (typeName == "Swish") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto *S = G_->createSwish(opName, in);
    RETURN_IF_ERR(addNodeAsOutput(op, S));
    return Error::success();
  }

  if (typeName == "Logit") {
    // Load the input and (optional) epsilon clamping value:
    NodeValue input;
    ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));
    auto epsIt = dict.find("eps");
    // default: 1e-6 (as in Caffe2)
    float eps = 1E-6f;
    if (epsIt != dict.end()) {
      ASSIGN_VALUE_OR_RETURN_ERR(eps, loadFloat(epsIt->second));
    }

    auto *node = G_->createLogit(opName, input, eps);
    // Save the outputs:
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "EQ") {
    NodeValue in0;
    ASSIGN_VALUE_OR_RETURN_ERR(in0, getNodeValueByName(op.input(0)));
    NodeValue in1;
    ASSIGN_VALUE_OR_RETURN_ERR(in1, getNodeValueByName(op.input(1)));
    auto *node = G_->createCmpEQ(opName, in0, in1);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "Tile") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    unsigned_t tiles;
    ASSIGN_VALUE_OR_RETURN_ERR(tiles, loadInt(dict["tiles"]));
    unsigned_t axis;
    ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict["axis"]));

    auto *node = G_->createTile(opName, in, tiles, axis);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "Free") {
    // Glow frees memory automatically.
    return Error::success();
  }
  if (typeName == "StopGradient" || typeName == "ScaleGradient") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    // Currently Caffe2 importer only supports inference.
    RETURN_IF_ERR(addNodeAsOutput(op, in));
    return Error::success();
  }

  if (typeName == "Transpose") {
    RETURN_IF_ERR(loadTranspose(op, dict, "axes"));
    return Error::success();
  }

  if (typeName == "NCHW2NHWC") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto *node = G_->createTranspose(opName, in, NCHW2NHWC);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "CopyCPUToMKL" || typeName == "CopyMKLToCPU" ||
      typeName == "Copy" || typeName == "EnsureCPUOutput" ||
      typeName == "EnsureDense" || typeName == "Dropout") {
    // Glow does not support any of these ops now, so implement them as
    // no-ops. Note: Implement this as a no-op reshape because these ops may
    // have partition information, and we need a node to maintain the parent
    // Function partition it specified. This reshape will get eliminated later
    // on during graph optimizations.
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    ReshapeNode *RN = G_->createReshape(in.getNode()->getName(), in, in.dims());
    RETURN_IF_ERR(addNodeAsOutput(op, RN));
    return Error::success();
  }

  if (typeName == "Slice") {
    NodeValue data;
    ASSIGN_VALUE_OR_RETURN_ERR(data, getNodeValueByName(op.input(0)));

    std::vector<ssize_t> starts;
    ASSIGN_VALUE_OR_RETURN_ERR(starts, getShape<ssize_t>(dict["starts"]));
    std::vector<ssize_t> ends;
    ASSIGN_VALUE_OR_RETURN_ERR(ends, getShape<ssize_t>(dict["ends"]));

    std::vector<dim_t> newStarts, newEnds;
    RETURN_ERR_IF_NOT(
        starts.size() == ends.size(),
        opErrMsg(op, strFormat(
                         "Slice starts %lu and %lu ends must be the same size.",
                         starts.size(), ends.size())));
    for (size_t i = 0; i < starts.size(); i++) {
      ssize_t newStart = starts[i];
      if (newStart == -1) {
        newStart = data.dims()[i];
      }
      RETURN_ERR_IF_NOT(
          newStart >= 0,
          opErrMsg(op,
                   strFormat("Indices should never be negative, but found %lu ",
                             newStart)));
      newStarts.push_back(newStart);

      ssize_t newEnd = ends[i];
      if (newEnd == -1) {
        newEnd = data.dims()[i];
      }
      RETURN_ERR_IF_NOT(
          newEnd >= 0,
          opErrMsg(op,
                   strFormat("Indices should never be negative, but found %lu ",
                             newEnd)));
      newEnds.push_back(newEnd);
    }

    Node *SN = G_->createSlice(opName, data, newStarts, newEnds);
    RETURN_IF_ERR(addNodeAsOutput(op, SN));
    return Error::success();
  }

  if (typeName == "Clip") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    float cmin = std::numeric_limits<float>::lowest();
    if (dict.count("min")) {
      ASSIGN_VALUE_OR_RETURN_ERR(cmin, loadFloat(dict.find("min")->second));
    }

    float cmax = std::numeric_limits<float>::max();
    if (dict.count("max")) {
      ASSIGN_VALUE_OR_RETURN_ERR(cmax, loadFloat(dict.find("max")->second));
    }

    auto *node = G_->createClip(loadOperatorName(op), in, cmin, cmax);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "MatMul") {
    RETURN_IF_ERR(loadMatMul(op, dict));
    return Error::success();
  }

  if (typeName == "Cast") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    int to;
    ASSIGN_VALUE_OR_RETURN_ERR(to, loadInt(dict["to"]));

    switch (to) {
    case caffe2::TensorProto_DataType_FLOAT: {
      RETURN_ERR_IF_NOT(in.getElementType() == ElemKind::FloatTy,
                        opErrMsg(op, "Can only cast float to float."));
      break;
    }
    case caffe2::TensorProto_DataType_INT32: {
      RETURN_ERR_IF_NOT(in.getElementType() == ElemKind::Int32ITy,
                        opErrMsg(op, "Can only cast int32 to int32."));
      break;
    }
    case caffe2::TensorProto_DataType_INT64: {
      RETURN_ERR_IF_NOT(in.getElementType() == ElemKind::Int64ITy,
                        opErrMsg(op, "Can only cast int64 to int64."));
      break;
    }
    default:
      return MAKE_ERR(opErrMsg(op, "Unsupported Cast type."));
    }

    RETURN_IF_ERR(addNodeAsOutput(op, in));
    return Error::success();
  }

  if (typeName == "HalfToFloat") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto convertedType =
        mod_.uniqueType(ElemKind::FloatTy, in.getType()->dims());
    auto *R = G_->createConvertTo(opName + ".ConvertInput", in, convertedType);
    RETURN_IF_ERR(addNodeAsOutput(op, R));
    return Error::success();
  }

  if (typeName == "ScatterAssign") {
    NodeValue data;
    ASSIGN_VALUE_OR_RETURN_ERR(data, getNodeValueByName(op.input(0)));
    NodeValue indices;
    ASSIGN_VALUE_OR_RETURN_ERR(indices, getNodeValueByName(op.input(1)));
    NodeValue slices;
    ASSIGN_VALUE_OR_RETURN_ERR(slices, getNodeValueByName(op.input(2)));

    assert(indices.dims().size() == 1 && "Indices should be 1-dimensional!");
    NodeValue indices2D = G_->createReshape(opName + ".indices.2d", indices,
                                            {indices.dims()[0], 1});
    Node *SAN = G_->createScatterData(opName, data, indices2D, slices);
    RETURN_IF_ERR(addNodeAsOutput(op, SAN));
    return Error::success();
  }

  if (typeName == "ConstantFill" || typeName == "GivenTensorIntFill" ||
      typeName == "GivenTensorInt64Fill" || typeName == "GaussianFill" ||
      typeName == "UniformFill") {
    RETURN_IF_ERR(loadWeight(op));
    return Error::success();
  }

  if (typeName == "SigmoidCrossEntropyWithLogits") {
    NodeValue logits;
    ASSIGN_VALUE_OR_RETURN_ERR(logits, getNodeValueByName(op.input(0)));
    NodeValue targets;
    ASSIGN_VALUE_OR_RETURN_ERR(targets, getNodeValueByName(op.input(1)));
    Node *SCEL =
        G_->createSigmoidCrossEntropyWithLogits(opName, logits, targets);
    RETURN_IF_ERR(addNodeAsOutput(op, SCEL));
    return Error::success();
  }

  if (typeName == "ElementwiseLinear") {
    NodeValue X, w, b;

    // If the axis argument does not exist in the protobuf, the default
    // value should be 1.
    unsigned axis = 1;

    ASSIGN_VALUE_OR_RETURN_ERR(X, getNodeValueByName(op.input(0)));
    ASSIGN_VALUE_OR_RETURN_ERR(w, getNodeValueByName(op.input(1)));
    ASSIGN_VALUE_OR_RETURN_ERR(b, getNodeValueByName(op.input(2)));

    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict["axis"]));
    }

    Node *EL = G_->createElementwiseLinear(opName, X, w, b, axis);
    RETURN_IF_ERR(addNodeAsOutput(op, EL));
    return Error::success();
  }

  if (typeName == "AveragedLoss") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto *node = G_->createBatchedReduceMean(opName, in, 0);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "Mod") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    int64_t divisor;
    ASSIGN_VALUE_OR_RETURN_ERR(divisor, loadInt(dict["divisor"]));

    RETURN_ERR_IF_NOT(
        divisor >= 1,
        opErrMsg(op,
                 strFormat("Divisor must not be less than 1, but found %ld ",
                           divisor)));

    bool signFollowDivisor = false;
    if (dict.count("sign_follow_divisor")) {
      ASSIGN_VALUE_OR_RETURN_ERR(signFollowDivisor,
                                 loadInt(dict["sign_follow_divisor"]));
    }

    auto *node = G_->createModulo(opName, in, divisor, signFollowDivisor);
    RETURN_IF_ERR(addNodeAsOutput(op, node));

    return Error::success();
  }

  if (typeName == "Scale") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    float scale = 1.0;
    if (dict.count("scale")) {
      ASSIGN_VALUE_OR_RETURN_ERR(scale, loadFloat(dict["scale"]));
    }
    auto scaleType = mod_.uniqueType(ElemKind::FloatTy, {in.dims()});
    auto scales = G_->createSplat(opName + ".scales", scaleType, scale);
    Node *node = G_->createMul(opName, in, scales);

    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "SparseLengthsWeightedSum8BitsRowwise" ||
      typeName == "SparseLengthsSum8BitsRowwise" ||
      typeName == "SparseLengthsWeightedSumFused8BitRowwise" ||
      typeName == "SparseLengthsSumFused8BitRowwise" ||
      typeName == "SparseLengthsWeightedSumFused4BitRowwise" ||
      typeName == "SparseLengthsSumFused4BitRowwise") {
    const bool isWeighted =
        typeName == "SparseLengthsWeightedSum8BitsRowwise" ||
        typeName == "SparseLengthsWeightedSumFused8BitRowwise" ||
        typeName == "SparseLengthsWeightedSumFused4BitRowwise";
    const bool isFused =
        typeName == "SparseLengthsWeightedSumFused8BitRowwise" ||
        typeName == "SparseLengthsSumFused8BitRowwise" ||
        typeName == "SparseLengthsWeightedSumFused4BitRowwise" ||
        typeName == "SparseLengthsSumFused4BitRowwise";
    const bool is4Bit =
        typeName == "SparseLengthsWeightedSumFused4BitRowwise" ||
        typeName == "SparseLengthsSumFused4BitRowwise";
    // If weighted, then the weights are the second input and so we need to
    // shift indices/lengths/scalesBiases.
    size_t indicesIdx = 1;
    size_t lengthsIdx = 2;
    size_t scalesBiasesIdx = 3;
    if (isWeighted) {
      indicesIdx++;
      lengthsIdx++;
      scalesBiasesIdx++;
    }

    NodeValue data;
    ASSIGN_VALUE_OR_RETURN_ERR(data, getNodeValueByName(op.input(0)));
    NodeValue weights;
    if (isWeighted) {
      ASSIGN_VALUE_OR_RETURN_ERR(weights, getNodeValueByName(op.input(1)));
    }
    NodeValue indices;
    ASSIGN_VALUE_OR_RETURN_ERR(indices,
                               getNodeValueByName(op.input(indicesIdx)));
    NodeValue lengths;
    ASSIGN_VALUE_OR_RETURN_ERR(lengths,
                               getNodeValueByName(op.input(lengthsIdx)));
    Storage *dataS = llvm::dyn_cast<Storage>(data);

    const dim_t numRows = data.dims()[0];

    // Make sure all the shapes make sense.
    RETURN_ERR_IF_NOT(lengths.dims().size() == 1,
                      opErrMsg(op, "lengths must be a vector."));
    RETURN_ERR_IF_NOT(indices.dims().size() == 1,
                      opErrMsg(op, "indices must be a vector."));

    LengthsMode lengthsMode;
    ASSIGN_VALUE_OR_RETURN_ERR(lengthsMode, getLengthsMode(dict));

    float avgLength;
    ASSIGN_VALUE_OR_RETURN_ERR(avgLength, getAvgLength(dict));

    Node *node;
    if (isFused) {
      RETURN_IF_ERR(setFusedTy(dataS, is4Bit ? ElemKind::UInt4FusedFP16QTy
                                             : ElemKind::UInt8FusedQTy));

      // No other work to do, since the data is already loaded fused, so just
      // create the new node with its inputs.
      if (isWeighted) {
        node = G_->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
            opName, dataS, weights, indices, lengths,
            /* useFP16Accumulation */ false, lengthsMode, avgLength);
      } else {
        node = G_->createFusedRowwiseQuantizedSparseLengthsSum(
            opName, dataS, indices, lengths, /* useFP16Accumulation */ false,
            lengthsMode, avgLength);
      }

      if (is4Bit) {
        node = G_->createConvertTo(opName, node, ElemKind::FloatTy);
      }
    } else {
      NodeValue scalesBiases;
      ASSIGN_VALUE_OR_RETURN_ERR(scalesBiases,
                                 getNodeValueByName(op.input(scalesBiasesIdx)));

      Constant *scalesBiasesC = llvm::dyn_cast<Constant>(scalesBiases);
      RETURN_ERR_IF_NOT(scalesBiasesC,
                        opErrMsg(op, "scales_biases must be Constant."));
      RETURN_ERR_IF_NOT(scalesBiases.dims().size() == 2,
                        opErrMsg(op, "scale_bias has to be a matrix."));
      RETURN_ERR_IF_NOT(
          scalesBiases.dims()[0] == numRows,
          opErrMsg(
              op,
              strFormat("scale_bias must have the same number of rows as data, "
                        "but found scale_bias %d and rows %d ",
                        int(scalesBiases.dims()[0]), int(numRows))));
      RETURN_ERR_IF_NOT(
          scalesBiases.dims()[1] == 2,
          opErrMsg(op,
                   strFormat("Second dim of scale_bias has to be equal to 2 "
                             "but found %d ",
                             int(scalesBiases.dims()[1]))));

      // Now strip out the scales and biases into their own tensors.
      NodeValue sliceScales =
          G_->createSlice(scalesBiasesC->getName().str() + "_scale",
                          scalesBiasesC, {0, 0}, {numRows, 1});
      NodeValue sliceBiases =
          G_->createSlice(scalesBiasesC->getName().str() + "_bias",
                          scalesBiasesC, {0, 1}, {numRows, 2});
      sliceScales =
          G_->createReshape(sliceScales.getNode()->getName().str() + "_1D",
                            sliceScales, {numRows});
      sliceBiases =
          G_->createReshape(sliceBiases.getNode()->getName().str() + "_1D",
                            sliceBiases, {numRows});

      // Now create the actual node.
      if (isWeighted) {
        node = G_->createRowwiseQuantizedSparseLengthsWeightedSum(
            opName, dataS, sliceScales, sliceBiases, weights, indices, lengths,
            /* precision */ ElemKind::FloatTy,
            /* useFP16Accumulation */ false, lengthsMode, avgLength);
      } else {
        node = G_->createRowwiseQuantizedSparseLengthsSum(
            opName, dataS, sliceScales, sliceBiases, indices, lengths,
            /* precision */ ElemKind::FloatTy,
            /* useFP16Accumulation */ false, lengthsMode, avgLength);
      }
    }

    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "LengthsRangeFill") {
    NodeValue lengths;
    ASSIGN_VALUE_OR_RETURN_ERR(lengths, getNodeValueByName(op.input(0)));
    RETURN_ERR_IF_NOT(lengths.dims().size() == 1,
                      opErrMsg(op, "lengths must be a 1D vector."));

    auto maxOutputSizeIt = dict.find("maxOutputSize");
    RETURN_ERR_IF_NOT(
        maxOutputSizeIt != dict.end(),
        opErrMsg(op, "Require maxOutputSize when loading LengthsRangeFill."));
    unsigned_t maxOutputSize;
    ASSIGN_VALUE_OR_RETURN_ERR(maxOutputSize, loadInt(maxOutputSizeIt->second));

    auto *LRF = G_->createLengthsRangeFill(opName, lengths, maxOutputSize);
    RETURN_IF_ERR(addNodeAsOutput(op, LRF));

    return Error::success();
  }

  // TODO: add checks for number of inputs and argument values
  if (typeName == "ReduceBackSum") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    RETURN_ERR_IF_NOT(in.dims().size() >= 2,
                      opErrMsg(op, "Input should be at least 2D."));
    Node *node = G_->createBatchedReduceAdd(opName, in, in.dims().size() - 1);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "RMSNorm") {
    NodeValue X, gamma, beta;
    ASSIGN_VALUE_OR_RETURN_ERR(X, getNodeValueByName(op.input(0)));
    RETURN_ERR_IF_NOT(X.dims().size() == 2,
                      opErrMsg(op, "X should be a 2D tensor."));
    ASSIGN_VALUE_OR_RETURN_ERR(gamma, getNodeValueByName(op.input(1)));
    RETURN_ERR_IF_NOT(gamma.dims().size() == 1,
                      opErrMsg(op, "gamma should be a 1D tensor."));
    ASSIGN_VALUE_OR_RETURN_ERR(beta, getNodeValueByName(op.input(2)));
    RETURN_ERR_IF_NOT(beta.dims().size() == 1,
                      opErrMsg(op, "beta should be a 1D tensor."));

    float epsilon = .0f;
    if (dict.count("eps")) {
      ASSIGN_VALUE_OR_RETURN_ERR(epsilon, loadFloat(dict["eps"]));
    }

    auto nodes = G_->createRMSNorm(opName, X, gamma, beta, epsilon);
    nodeValueByName_[op.output(0)] = nodes[0];
    nodeValueByName_[op.output(1)] = nodes[1];
    return Error::success();
  }

  if (typeName == "Mean") {
    const unsigned numInputs = op.input_size();
    RETURN_ERR_IF_NOT(numInputs > 0,
                      opErrMsg(op, "Expect at least one input."));

    std::vector<NodeValue> inputs;
    inputs.reserve(numInputs);
    for (unsigned i = 0; i < numInputs; i++) {
      NodeValue in;
      ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(i)));
      inputs.push_back(std::move(in));
    }

    // Check that all inputs have the same shape
    const auto shape = inputs[0].dims();
    for (unsigned i = 1; i < numInputs; i++) {
      RETURN_ERR_IF_NOT(
          shape == inputs[i].dims(),
          opErrMsg(op,
                   "All inputs should have the same shape, violating input " +
                       op.input(i)));
    }

    if (numInputs == 1) {
      RETURN_IF_ERR(addNodeAsOutput(op, inputs[0]));
      return Error::success();
    }

    Node *node = G_->createConcat(opName + ".concat", inputs, 0);

    std::vector<dim_t> newShape{numInputs};
    newShape.insert(newShape.end(), shape.begin(), shape.end());
    node = G_->createReshape(opName + ".reshape", node, newShape);

    node = G_->createBatchedReduceMean(opName + ".reduceMean", node, 0);

    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "Negative") {
    RETURN_IF_ERR(loadNeg(op, dict));
    return Error::success();
  }

  if (typeName == "LpNorm") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

    int p = 2;
    if (dict.count("p")) {
      ASSIGN_VALUE_OR_RETURN_ERR(p, loadInt(dict["p"]));
      RETURN_ERR_IF_NOT(p == 1 || p == 2,
                        opErrMsg(op, "p should be either 1 or 2."));
    }
    bool average = false;
    if (dict.count("average")) {
      ASSIGN_VALUE_OR_RETURN_ERR(average, loadInt(dict["average"]));
    }
    RETURN_ERR_IF_NOT(!average, opErrMsg(op, "average is not supported."));

    Node *node = nullptr;
    if (p == 1) {
      node = G_->createAbs(opName, in);
    } else {
      node = G_->createPow(opName, in, 2);
    }

    const auto dims1D = flattenCdr(in.dims(), in.dims().size());
    node = G_->createReshape(opName + ".reshape1D", node, dims1D.first);

    auto outputType = mod_.uniqueType(in.getElementType(), {1});
    node = G_->createBatchedReduceAdd(opName + ".sum", outputType, node, 0);

    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "ArgMin") {
    NodeValue input;
    ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));
    int axis = 0;
    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict["axis"]));
    }
    bool keepDims = true;
    if (dict.count("keepdims")) {
      ASSIGN_VALUE_OR_RETURN_ERR(keepDims, loadInt(dict.at("keepdims")));
    }

    auto node = G_->createArgMin(opName, input, axis, keepDims);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "Sign") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

    Node *zeroes = G_->createSplat(opName + ".zeroes", in.getType(), 0.f);

    Node *isPos = G_->createCmpLT(opName + ".isPos", zeroes, in);
    Node *isNeg = G_->createCmpLT(opName + ".isNeg", in, zeroes);

    Node *posOnes = G_->createSplat(opName + ".posOnes", in.getType(), 1);
    Node *negOnes = G_->createSplat(opName + ".negOnes", in.getType(), -1);

    Node *node = G_->createSelect(opName + ".fillPos", isPos, posOnes, zeroes);
    node = G_->createSelect(opName + ".fillNeg", isNeg, negOnes, node);

    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "Softplus") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

    Node *node = G_->createSoftPlus(opName, in);

    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "TopK") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    RETURN_ERR_IF_NOT(
        op.input_size() <= 2,
        opErrMsg(
            op,
            strFormat(
                "TopK: Maximum number of inputs is 2, but found input size %d ",
                op.input_size())));
    unsigned_t k = 0;
    if (op.input_size() > 1) {
      Constant *kConst = getConstantByNameOrNull(op.input(1));
      RETURN_ERR_IF_NOT(
          kConst,
          opErrMsg(op, "TopK: Non-constant k is not supported by Glow."));
      RETURN_ERR_IF_NOT(
          kConst->getElementType() == ElemKind::Int64ITy,
          opErrMsg(op, strFormat(
                           "TopK: k input must be of type Int64, but found "
                           "input type '%s' ",
                           kConst->getType()->getElementName().str().c_str())));
      auto constH = kConst->getPayload().getHandle<int64_t>();
      k = constH.at({0});
    } else {
      ASSIGN_VALUE_OR_RETURN_ERR(k, loadInt(dict["k"]));
    }

    int lastDim = in.dims().size() - 1;
    int axis = lastDim;
    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(axis,
                                 loadAxis<int>(dict["axis"], in.dims().size()));
    }

    RETURN_ERR_IF_NOT(
        axis == lastDim,
        opErrMsg(
            op,
            strFormat(
                "TopK: Currently only support axis %d being last dimension %d ",
                axis, lastDim)));

    TopKNode *R = G_->createTopK(opName, in, k, ElemKind::Int32ITy);
    RETURN_IF_ERR(addNodeAsOutput(op, R));
    return Error::success();
  }

  if (typeName == "FillExamplesWithIndicator") {
    // Support FillExamplesWithIndicator
    NodeValue data;
    ASSIGN_VALUE_OR_RETURN_ERR(data, getNodeValueByName(op.input(0)));
    NodeValue indicator;
    ASSIGN_VALUE_OR_RETURN_ERR(indicator, getNodeValueByName(op.input(1)));
    // Validating input types and shapes
    RETURN_ERR_IF_NOT(
        indicator.getElementType() == ElemKind::Int32ITy ||
            indicator.getElementType() == ElemKind::Int64ITy,
        opErrMsg(op, "Indicator should be of int32 or int64 type."));
    RETURN_ERR_IF_NOT(indicator.dims().size() == 1,
                      opErrMsg(op, "Indicator should be 1D tensor."));
    dim_t dataReshapeDim = flattenCdr(data.dims()).second;
    ShapeVector outDims{indicator.dims()[0]};
    outDims.insert(outDims.end(), data.dims().begin() + 1, data.dims().end());
    auto outTy2D = mod_.uniqueTypeWithNewShape(
        data.getType(), {indicator.dims()[0], dataReshapeDim});

    auto data2D = G_->createReshape(opName + ".data2D", data,
                                    {data.dims()[0], dataReshapeDim});
    if (indicator.getElementType() == ElemKind::Int64ITy) {
      indicator = G_->createConvertTo(opName + ".int64ToInt32", indicator,
                                      ElemKind::Int32ITy);
    }
    // Select only takes boolean indicators, and converting from int to bool
    // must go from int -> float -> bool. Due to fp16 clipping, since only
    // int32 -> fp16 conversions are available, there is an initial conversion
    // from int64 to int32 if necessary.
    auto indicatorFloat = G_->createConvertTo(opName + ".intToFloat", indicator,
                                              ElemKind::FloatTy);
    auto indicatorBool = G_->createConvertTo(opName + ".floatToBool",
                                             indicatorFloat, ElemKind::BoolTy);
    auto nzIndices = G_->createNonZero(opName + ".nonzero", indicatorBool);

    auto nzIndicesFixed = fixNonZero(G_, mod_, opName, nzIndices);
    auto nonZeroCount = data.dims()[0];
    RETURN_ERR_IF_NOT(nonZeroCount <= nzIndicesFixed->getNthResult(0).dims()[0],
                      opErrMsg(op,
                               "The number of "
                               "non-zero elements in the indicator must be at "
                               "least that of the first dimension of data"));

    auto indices = G_->createSlice(opName + ".indices", nzIndicesFixed, {0, 0},
                                   {data.dims()[0], 1});

    auto zeros = G_->createSplat(opName + ".zeros", outTy2D, 0);

    auto res2D = G_->createScatterData(opName + ".scatterData", zeros, indices,
                                       data2D, true);
    auto node = G_->createReshape(opName + ".result", res2D, outDims);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "BatchSparseToDense") {
    // Support BatchSparseToDense for output second dim = 1 only
    NodeValue lengths;
    ASSIGN_VALUE_OR_RETURN_ERR(lengths, getNodeValueByName(op.input(0)));
    NodeValue indices;
    ASSIGN_VALUE_OR_RETURN_ERR(indices, getNodeValueByName(op.input(1)));
    NodeValue values;
    ASSIGN_VALUE_OR_RETURN_ERR(values, getNodeValueByName(op.input(2)));

    dim_t denseLastDim = 1;
    if (dict.count("dense_last_dim")) {
      ASSIGN_VALUE_OR_RETURN_ERR(denseLastDim,
                                 loadInt(dict.at("dense_last_dim")));
    }

    RETURN_ERR_IF_NOT(
        denseLastDim == 1,
        opErrMsg(op, "Only output second dimension = 1 supported"));
    // Validating input types and shapes
    RETURN_ERR_IF_NOT(
        lengths.getElementType() == ElemKind::Int32ITy ||
            lengths.getElementType() == ElemKind::Int64ITy,
        opErrMsg(op, "Lengths should be of int32 or int64 type."));
    RETURN_ERR_IF_NOT(lengths.dims().size() == 1,
                      opErrMsg(op, "Lengths should be 1D tensor."));
    RETURN_ERR_IF_NOT(
        indices.getElementType() == ElemKind::Int32ITy ||
            indices.getElementType() == ElemKind::Int64ITy,
        opErrMsg(op, "Indices should be of int32 or int64 type."));
    RETURN_ERR_IF_NOT(indices.dims().size() == 1,
                      opErrMsg(op, "Indices should be 1D tensor."));
    RETURN_ERR_IF_NOT(values.getElementType() == ElemKind::FloatTy,
                      opErrMsg(op, "Values should be of float type."));
    RETURN_ERR_IF_NOT(
        indices.dims()[0] == values.dims()[0],
        opErrMsg(op, "There should be the same number of values as indices."));

    float defaultValue = 0.0;
    if (dict.count("default_value")) {
      ASSIGN_VALUE_OR_RETURN_ERR(defaultValue,
                                 loadFloat(dict.at("default_value")));
    }
    // Select only takes boolean indicators, and converting from int to bool
    // must go from int -> float -> bool. Due to fp16 clipping, since only
    // int32 -> fp16 conversions are available, there is an initial conversion
    // from int64 to int32 if necessary.
    if (lengths.getElementType() == ElemKind::Int64ITy) {
      lengths = G_->createConvertTo(opName + ".int64ToInt32", lengths,
                                    ElemKind::Int32ITy);
    }
    auto lengthsIntToFloat =
        G_->createConvertTo(opName + ".intToFloat", lengths, ElemKind::FloatTy);
    auto lengthsFloatToBool = G_->createConvertTo(
        opName + ".floatToBool", lengthsIntToFloat, ElemKind::BoolTy);
    auto nonZeroIndices =
        G_->createNonZero(opName + ".nonzero", lengthsFloatToBool);
    auto nonZeroIndicesFixed = fixNonZero(G_, mod_, opName, nonZeroIndices);
    auto numIndices = indices.dims()[0];
    auto indicesSliced = G_->createSlice(
        opName + ".indicesSlice", nonZeroIndicesFixed, {0, 0}, {numIndices, 1});

    ShapeVector outDims{lengths.dims()[0], 1};
    auto dataTy = mod_.uniqueTypeWithNewShape(values.getType(), outDims);
    auto data = G_->createSplat(opName + ".data", dataTy, defaultValue);
    auto values2D =
        G_->createReshape(opName + ".reshape", values, {numIndices, 1});
    auto scatterData = G_->createScatterData(opName + ".scatterData", data,
                                             indicesSliced, values2D, true);

    RETURN_IF_ERR(addNodeAsOutput(op, scatterData));
    return Error::success();
  }

  if (typeName == "SparseLabelSplit") {
    NodeValue lengths;
    ASSIGN_VALUE_OR_RETURN_ERR(lengths, getNodeValueByName(op.input(0)));
    NodeValue indices;
    ASSIGN_VALUE_OR_RETURN_ERR(indices, getNodeValueByName(op.input(1)));
    NodeValue values;
    ASSIGN_VALUE_OR_RETURN_ERR(values, getNodeValueByName(op.input(2)));

    dim_t numLabels = 0;
    RETURN_ERR_IF_NOT(dict.count("num_labels"),
                      opErrMsg(op, "num_labels was not provided."));
    ASSIGN_VALUE_OR_RETURN_ERR(numLabels, loadInt(dict.at("num_labels")));

    bool keepGradientOffsetMap = false;
    if (dict.count("keep_gradient_offset_map")) {
      ASSIGN_VALUE_OR_RETURN_ERR(keepGradientOffsetMap,
                                 loadInt(dict.at("keep_gradient_offset_map")));
    }

    // Validating input types and shapes
    RETURN_ERR_IF_NOT(lengths.getElementType() == ElemKind::Int32ITy,
                      opErrMsg(op, "Lengths should be of int32 type."));
    RETURN_ERR_IF_NOT(lengths.dims().size() == 1 || lengths.dims().size() == 2,
                      opErrMsg(op, "Lengths should be 1D or 2D tensor."));
    RETURN_ERR_IF_NOT(indices.getElementType() == ElemKind::Int64ITy,
                      opErrMsg(op, "Indices should be of int64 type."));
    RETURN_ERR_IF_NOT(indices.dims().size() == 1 || indices.dims().size() == 2,
                      opErrMsg(op, "Indices should be 1D or 2D tensor."));
    RETURN_ERR_IF_NOT(values.getElementType() == ElemKind::FloatTy,
                      opErrMsg(op, "Values should be of float type."));
    RETURN_ERR_IF_NOT(values.dims().size() == 1 || values.dims().size() == 2,
                      opErrMsg(op, "Values should be 1D or 2D tensor."));
    RETURN_ERR_IF_NOT(
        indices.dims() == values.dims(),
        opErrMsg(op, "Indices and values should have the same shape."));

    // Optional conversion from 2D to 1D inputs
    if (lengths.dims().size() == 2) {
      RETURN_ERR_IF_NOT(
          lengths.dims()[1] == 1,
          opErrMsg(op, "Second dimension should be 1 in lengths."));
      lengths = G_->createReshape(opName + ".lengths1D", lengths,
                                  {lengths.dims()[0]});
    }
    if (indices.dims().size() == 2) {
      RETURN_ERR_IF_NOT(
          indices.dims()[1] == 1,
          opErrMsg(op, "Second dimension should be 1 in indices."));
      indices = G_->createReshape(opName + ".indices1D", indices,
                                  {indices.dims()[0]});
    }
    if (values.dims().size() == 2) {
      RETURN_ERR_IF_NOT(
          values.dims()[1] == 1,
          opErrMsg(op, "Second dimension should be 1 in values."));
      values =
          G_->createReshape(opName + ".values1D", values, {values.dims()[0]});
    }

    SparseLabelSplitNode *node =
        G_->createSparseLabelSplit(opName, lengths, indices, values, numLabels);

    std::vector<SliceNode *> labelValueSlices;
    G_->createSplit(opName + ".splitLabelValues",
                    node->getNthResult(SparseLabelSplitNode::LabelValuesIdx),
                    numLabels, 0, {}, labelValueSlices);

    std::vector<SliceNode *> exampleIdSlices;
    G_->createSplit(opName + ".splitExampleIds",
                    node->getNthResult(SparseLabelSplitNode::ExampleIdsIdx),
                    numLabels, 0, {}, exampleIdSlices);

    const auto numItems = indices.dims()[0] / numLabels;

    std::vector<Node *> labelValues;
    for (auto slice : labelValueSlices) {
      labelValues.push_back(
          G_->createReshape(opName + ".reshapeLabelValue", slice, {numItems}));
    }

    std::vector<Node *> exampleIds;
    for (auto slice : exampleIdSlices) {
      exampleIds.push_back(
          G_->createReshape(opName + ".reshapeExamplId", slice, {numItems}));
    }

    for (dim_t i = 0; i < numLabels; ++i) {
      nodeValueByName_[op.output(i)] = labelValues[i];
    }
    for (dim_t i = 0; i < numLabels; ++i) {
      nodeValueByName_[op.output(numLabels + i)] = exampleIds[i];
    }
    if (keepGradientOffsetMap) {
      nodeValueByName_[op.output(2 * numLabels)] =
          node->getNthResult(SparseLabelSplitNode::GradientOffsetMapIdx);
    }
    return Error::success();
  }

  if (typeName == "Log1p") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

    Node *ones = G_->createSplat(opName + ".ones", in.getType(), 1.0f);
    Node *add = G_->createAdd(opName + ".add", in, ones);
    Node *node = G_->createLog(opName + ".log", add);

    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "ReduceBackMean") {
    const unsigned numInputs = op.input_size();
    RETURN_ERR_IF_NOT(numInputs == 1,
                      opErrMsg(op, "Only single input is supported."));

    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    RETURN_ERR_IF_NOT(in.dims().size() >= 2,
                      opErrMsg(op, "Input should be at least 2D."));

    int numReduceDim = 1;
    if (dict.count("num_reduce_dim")) {
      ASSIGN_VALUE_OR_RETURN_ERR(numReduceDim, loadInt(dict["num_reduce_dim"]));
    }
    // TODO: check maybe we can support more dimensions to be reduced
    RETURN_ERR_IF_NOT(numReduceDim == 1,
                      opErrMsg(op, "Supporting reducing only one dimension."));

    Node *node = G_->createBatchedReduceMean(opName, in, in.dims().size() - 1);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  return MAKE_ERR(unexpectedNodeErrorMessage(op, "Unsupported operator."));
}

template <class TensorProtoType>
Error Caffe2ModelLoader::loadInputsWithTensorProtoType(
    const caffe2::NetDef &net,
    const std::unordered_set<std::string> &initializers,
    const TensorProtoType &in) {
  // Skip static weights
  if (getConstantByNameOrNull(in.name())) {
    return Error::success();
  }

  if (getStaticPlaceholderByNameOrNull(in.name())) {
    return Error::success();
  }

  LoadWeightResult loadRes;
  if (auto resOrErr = createAndSetTensorType(in)) {
    loadRes = std::move(*resOrErr);
  } else {
    RETURN_ERR(resOrErr.takeError());
  }

  bool multiQParamsLoaded = loadRes.scales || loadRes.offsets;
  RETURN_ERR_IF_NOT(
      (!multiQParamsLoaded || (loadRes.scales && loadRes.offsets)),
      "For tensors with separate qparams, both scales and offsets must be "
      "loaded");

  bool isInput = !initializers.count(in.name());
  if (isInput) {
    RETURN_ERR_IF_NOT(!clipQuantRangeToFP16_ ||
                          !loadRes.t->getType().isQuantizedType() ||
                          loadRes.t->getType().isFusedQuantizedType(),
                      "Do not support clipQuantRangeToFP16 with unfused "
                      "quantized input Placeholders: " +
                          in.name());
    Placeholder *placeholder;
    ASSIGN_VALUE_OR_RETURN_ERR(
        placeholder,
        createAndRegisterPlaceholder(in.name(), &loadRes.t->getType()));

    inputVarsByName_.try_emplace(in.name(), placeholder);

    if (multiQParamsLoaded) {
      auto offsetsName = strFormat("%s_loaded_offsets", in.name().c_str());
      auto scalesName = strFormat("%s_loaded_scales", in.name().c_str());
      Placeholder *offsetsPlaceholder;
      Placeholder *scalesPlaceholder;

      ASSIGN_VALUE_OR_RETURN_ERR(offsetsPlaceholder,
                                 createAndRegisterPlaceholder(
                                     offsetsName, &loadRes.offsets->getType()));
      inputVarsByName_.try_emplace(offsetsName, offsetsPlaceholder);

      ASSIGN_VALUE_OR_RETURN_ERR(
          scalesPlaceholder,
          createAndRegisterPlaceholder(scalesName, &loadRes.scales->getType()));
      inputVarsByName_.try_emplace(scalesName, scalesPlaceholder);
    }
  } else {
    RETURN_IF_ERR(createAndRegisterConstant(in.name(), std::move(*loadRes.t)));

    if (multiQParamsLoaded) {
      auto offsetsName = strFormat("%s_loaded_offsets", in.name().c_str());
      auto scalesName = strFormat("%s_loaded_scales", in.name().c_str());
      RETURN_IF_ERR(
          createAndRegisterConstant(offsetsName, std::move(*loadRes.offsets)));
      RETURN_IF_ERR(
          createAndRegisterConstant(scalesName, std::move(*loadRes.scales)));
    }
  }
  return Error::success();
}

Error Caffe2ModelLoader::loadInputs(
    const caffe2::NetDef &net,
    const std::unordered_set<std::string> &initializers) {
  const caffe2::Argument *arg = nullptr, *qarg = nullptr;
  for (auto i = 0, e = net.arg_size(); i < e && (!arg || !qarg); ++i) {
    if (net.arg(i).name() == "input_shape_info") {
      arg = &net.arg(i);
    } else if (net.arg(i).name() == "input_qshape_info") {
      qarg = &net.arg(i);
    }
  }

  // Load all regular tensor input
  if (arg) {
    for (const auto &in : arg->tensors()) {
      RETURN_IF_ERR(loadInputsWithTensorProtoType<caffe2::TensorProto>(
          net, initializers, in));
    }
  }

  // Load all quantized tensor input
  if (qarg) {
    for (const auto &in : qarg->qtensors()) {
      RETURN_IF_ERR(loadInputsWithTensorProtoType<caffe2::QTensorProto>(
          net, initializers, in));
    }
  }

  return Error::success();
}

Error Caffe2ModelLoader::loadNetwork(caffe2::NetDef &net) {
  // Make a claim on the unique name of all output Placeholders.
  for (int i = 0; i < net.external_output_size(); i++) {
    auto &outputName = net.external_output(i);
    mod_.registerStorageName(legalizeName(outputName));
  }

  /// Load the network operators:
  for (int i = 0; i < net.op_size(); i++) {
    auto &op = net.op(i);

    // Set up current partition to load into if relevant.
    if (partNameToFun_.size()) {
      auto &pName = op.device_option().node_name();
      auto it = partNameToFun_.find(pName);
      RETURN_ERR_IF_NOT(
          it != partNameToFun_.end(),
          strFormat("Did not find partition with name %s", pName.c_str()));
      G_ = it->second;
    }
    RETURN_ERR_IF_NOT(G_, "Internal Glow error; Graph was not valid.");

    if (constFoldInLoader_) {
      auto tryFold = foldOperator(op);
      if (!tryFold) {
        // Error during constant folding; load the op normally below.
        const std::string errStr = ERR_TO_STRING(tryFold.takeError());
        VLOG(1) << "Error while trying to ConstantFold " << loadOperatorName(op)
                << ": " << errStr;
      } else if (tryFold.get()) {
        // Folded successfully, so skip loading the op below.
        continue;
      }
    }
    RETURN_IF_ERR(loadOperator(op));
  }

  RETURN_ERR_IF_NOT(net.external_output_size(),
                    "Network needs external outputs defined.");

  for (int i = 0; i < net.external_output_size(); i++) {
    auto &outputName = net.external_output(i);
    NodeValue r;
    // We want to create the save node in the same Function as the original
    // NodeValue. Thus here we ignore the source function when getting the NV,
    // which avoids copying the NV to whatever G_ currently is via an
    // intermediate Placeholder.
    ASSIGN_VALUE_OR_RETURN_ERR(
        r, getNodeValueByName(outputName, /* ignoreSrcFun */ true));

    PlaceholderList &PHList = mod_.getPlaceholders();
    // Create a Placeholder with the previously claimed name.
    auto *PH =
        new Placeholder(legalizeName(outputName), mod_.uniqueType(*r.getType()),
                        false, ANY_LAYOUT);
    PHList.push_back(PH);
    // If r is storage then just use the current last Function to save, since
    // we're just saving directly from a Storage node anyway.
    Function *F = llvm::isa<Storage>(r) ? G_ : r.getNode()->getParent();
    assert(F && "F must be valid here.");
    auto *SN = F->createSave(outputName, r, PH);
    outputVarsByName_[outputName] = SN->getPlaceholder();
  }
  return Error::success();
}

/// Fills \p T with data from \p values.
template <typename ElemTy, typename RangeTy>
static Error fillTensor(Tensor &T, ElemKind kind, llvm::ArrayRef<dim_t> dim,
                        RangeTy values) {
  T.reset(kind, dim);
  auto TH = T.getHandle<ElemTy>();
  RETURN_ERR_IF_NOT((size_t)values.size() == T.size(),
                    llvm::formatv("Wrong number of values for GivenTensorFill "
                                  "({0} given, {1} expected)",
                                  values.size(), T.size())
                        .str());
  size_t i = 0;
  for (auto num : values) {
    TH.raw(i++) = num;
  }
  return Error::success();
}

Error Caffe2ModelLoader::loadWeight(const caffe2::OperatorDef &op) {
  ArgumentDictionaryTy dict = loadArgumentMap(op);
  const std::string &typeName = op.type();
  const std::string &opName = loadOperatorName(op);
  // Load tensors with values:
  if (typeName == "GivenTensorFill" || typeName == "GivenTensorFp16Fill" ||
      typeName == "GivenTensorIntFill" || typeName == "GivenTensorInt64Fill") {
    /*
     * op {
     *   output: "conv1_w"
     *   name: ""
     *   type: "GivenTensorFill"
     *   arg {
     *     name: "shape"
     *     ints: 96
     *     ints: 3
     *     ints: 11
     *     ints: 11
     *   }
     *   arg {
     *     name: "values"
     *     floats: -0.028315347
     *     ...
     *   }
     * }
     */

    // Note: Explicitly allow for an empty dim here, representing a scalar value
    // will be loaded below.
    std::vector<dim_t> dim;
    ASSIGN_VALUE_OR_RETURN_ERR(
        dim, getShape<dim_t>(dict["shape"], /* allowEmptyShape */ true));
    auto const &values = dict["values"];
    RETURN_ERR_IF_NOT(
        op.output_size() == 1,
        opErrMsg(
            op, strFormat(
                    "GivenTensorFill must have exactly 1 output, but found %d ",
                    op.output_size())));
    Tensor T;
    if (typeName == "GivenTensorFill") {
      RETURN_IF_ERR(
          fillTensor<float>(T, ElemKind::FloatTy, dim, values->floats()));
    } else if (typeName == "GivenTensorFp16Fill") {
      RETURN_IF_ERR(
          fillTensor<float16_t>(T, ElemKind::Float16Ty, dim, values->floats()));
    } else if (typeName == "GivenTensorIntFill") {
      RETURN_IF_ERR(
          fillTensor<int32_t>(T, ElemKind::Int32ITy, dim, values->ints()));
    } else if (typeName == "GivenTensorInt64Fill") {
      RETURN_IF_ERR(
          fillTensor<int64_t>(T, ElemKind::Int64ITy, dim, values->ints()));
    } else {
      return MAKE_ERR(
          strFormat("Unhandled tensor fill type: %s", typeName.c_str()));
    }
    RETURN_IF_ERR(createAndRegisterConstant(op.output().Get(0), std::move(T)));
    return Error::success();
  }

  if (typeName == "GivenTensorByteStringToUInt8Fill") {
    /*
      output: "data"
      type: "GivenTensorByteStringToUInt8Fill"
      arg {
      name: "shape"
      ints: 3
      ints: 10
      }
      arg {
      name: "values"
      s:
      "\000\377\152\232\115\072\000\000\200\077\000\377\050\132\215\073\063\063\023\100\000\377\314\063\232\073\000\000\220\100"
      }
     */

    for (auto &o : op.output()) {
      Tensor T;
      if (getConstantByNameOrNull(o)) {
        continue;
      }
      std::vector<dim_t> dim;
      ASSIGN_VALUE_OR_RETURN_ERR(dim, getShape<dim_t>(dict["shape"]));
      T.reset(ElemKind::UInt8QTy, dim, 0.0, 0);
      auto TH = T.getHandle<uint8_t>();
      RETURN_ERR_IF_NOT(
          dict["values"]->strings().size() == 1,
          "Expect single string input for GivenTensorByteStringToUInt8Fill");
      const std::string &str = dict["values"]->strings().Get(0);

      size_t pos;
      for (pos = 0; pos < str.size(); pos++) {
        TH.raw(pos) = (uint8_t)str[pos];
      }

      RETURN_ERR_IF_NOT(
          pos == T.size(),
          strFormat("The number of serialized values (%li) does not "
                    "match the size of the tensor (%li).",
                    pos, (size_t)T.size()));
      RETURN_IF_ERR(createAndRegisterConstant(o, std::move(T)));
    }
    return Error::success();
  }

  // Load quantized tensors:
  if (typeName == "Int8GivenTensorFill" ||
      typeName == "Int8GivenIntTensorFill") {
    /*
     output: "conv1_w"
     name: ""
     type: "Int8GivenTensorFill"
     arg {
     name: "shape"
     ints: 96
     ints: 3
     ints: 11
     ints: 11
     }
     arg {
     name: "values"
     s: "\x7f\x80\x80\x7"
     }
     arg {
     name: "Y_scale"
     f: 0.00044428
     }
     arg {
     name: "Y_zero_point"
     i: 127
     }
     */
    for (auto &o : op.output()) {
      Tensor T;
      if (getConstantByNameOrNull(o)) {
        continue;
      }

      std::vector<dim_t> dim;
      ASSIGN_VALUE_OR_RETURN_ERR(dim, getShape<dim_t>(dict["shape"]));

      RETURN_ERR_IF_NOT(dict.count("Y_zero_point"),
                        ("missing zero point for quantized output type"));
      RETURN_ERR_IF_NOT(dict.count("Y_scale"),
                        ("missing Y_scale for quantized output type"));

      float scale;
      ASSIGN_VALUE_OR_RETURN_ERR(scale, loadFloat(dict["Y_scale"]));
      (void)scale;
      int32_t offset;
      ASSIGN_VALUE_OR_RETURN_ERR(offset, loadInt(dict["Y_zero_point"]));
      (void)offset;
      size_t i = 0;
      if (typeName == "Int8GivenTensorFill") {
        // Although in Caffe2 quantized model, the weights is int8 quantized,
        // the weights is stored in uint8_t format due to that Caffe2 requires
        // the type of input and weights must be the same. Therefore, we need
        // to convert it to int8 by subtracting 128.
        TypeRef ty;
        ASSIGN_VALUE_OR_RETURN_ERR(
            ty, loadQuantTy(o, ElemKind::Int8QTy, dim, dict,
                            /* skipClipQuantRangeToFP16 */ true));
        T.reset(*ty);
        auto TH = T.getHandle<int8_t>();
        std::string str = dict["values"]->s();
        for (; i < str.size(); i++) {
          TH.raw(i) = ((uint8_t)(str.c_str()[i]) - UINT8_TO_INT8_SHIFT);
        }
      } else {
        TypeRef ty;
        ASSIGN_VALUE_OR_RETURN_ERR(
            ty, loadQuantTy(o, ElemKind::Int32QTy, dim, dict,
                            /* skipClipQuantRangeToFP16 */ true));
        T.reset(*ty);
        auto TH = T.getHandle<int32_t>();
        for (auto num : dict["values"]->ints()) {
          TH.raw(i++) = num;
        }
      }

      // If we're clipping quantized ranges tp FP16, then we need to rescale the
      // Tensor and update its type.
      if (clipQuantRangeToFP16_) {
        const ElemKind k = T.getType().getElementType();
        const auto qMinMax = getQuantizedValueRange(T.getType().getScale(),
                                                    T.getType().getOffset(), k);
        const float newMin = std::max(qMinMax.first, kMinFP16);
        const float newMax = std::min(qMinMax.second, kMaxFP16);
        if (newMin != qMinMax.first || newMax != qMinMax.second) {
          auto rescaledT = glow::make_unique<Tensor>();
          dispatchQuantizedImpl(rescaleQTensor, k, T, *rescaledT, newMin,
                                newMax);
          T = std::move(*rescaledT);
        }
      }

      RETURN_ERR_IF_NOT(
          i == T.size(),
          strFormat("The number of serialized values (%li) does not "
                    "match the size of the tensor (%li).",
                    i, (size_t)T.size()));

      RETURN_IF_ERR(createAndRegisterConstant(o, std::move(T)));
    }

    return Error::success();
  }

  // Load tensors with constant fill:
  if (typeName == "ConstantFill") {
    /*
     output: "data"
     name: ""
     type: "ConstantFill"
     arg {
     name: "shape"
     ints: 1
     }
     */

    const auto &name = op.output(0);
    // If the tensor is pre-populated by the user of this class then we don't
    // need to allocate a new tensor.
    if (getConstantByNameOrNull(name)) {
      return Error::success();
    }

    // The shape is set either the shape argument, or from another input
    // tensor. Shape takes priority over input.
    std::vector<dim_t> dims;
    if (dict.count("shape")) {
      ASSIGN_VALUE_OR_RETURN_ERR(dims, getShape<dim_t>(dict["shape"]));
    } else {
      RETURN_ERR_IF_NOT(op.input_size() > 0,
                        "If no shape provided, must have input shape.");

      bool inputAsShape = false;
      if (dict.count("input_as_shape")) {
        ASSIGN_VALUE_OR_RETURN_ERR(inputAsShape,
                                   loadInt(dict["input_as_shape"]));
      }

      if (inputAsShape) {
        // It must be registered as a Constant because it must be statically set
        // already, as shapes must be statically known.
        Constant *in;
        ASSIGN_VALUE_OR_RETURN_ERR(in, getConstantByName(op.input(0)));
        RETURN_ERR_IF_NOT(in->dims().size() == 1,
                          opErrMsg(op, "Input must be 1D tensor."));
        RETURN_ERR_IF_NOT(in->getElementType() == ElemKind::Int64ITy,
                          opErrMsg(op, "Input must be of int64 type."));
        const auto handle = in->getHandle<int64_t>();
        dims.reserve(in->dims().size());
        for (auto dim : handle) {
          dims.push_back(dim);
        }
      } else {
        NodeValue in;
        ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
        dims = in.dims();
      }
    }

    int to = caffe2::TensorProto_DataType_FLOAT;
    if (dict.count("dtype")) {
      ASSIGN_VALUE_OR_RETURN_ERR(to, loadInt(dict["dtype"]));
    }

    SplatNode *splatNode{nullptr};

    switch (to) {
    case caffe2::TensorProto_DataType_FLOAT: {
      float f = 0.0f;
      if ((dict.count("value") && dict["value"]->has_f())) {
        ASSIGN_VALUE_OR_RETURN_ERR(f, loadFloat(dict["value"]));
      }
      splatNode =
          G_->createSplat(opName, mod_.uniqueType(ElemKind::FloatTy, dims), f);
      break;
    }
    case caffe2::TensorProto_DataType_INT32: {
      int i = 0;
      if ((dict.count("value") && dict["value"]->has_i())) {
        ASSIGN_VALUE_OR_RETURN_ERR(i, loadInt(dict["value"]));
      }
      splatNode =
          G_->createSplat(opName, mod_.uniqueType(ElemKind::Int32ITy, dims), i);
      break;
    }
    case caffe2::TensorProto_DataType_INT64:
    case caffe2::TensorProto_DataType_BOOL: {
      int i = 0;
      if ((dict.count("value") && dict["value"]->has_i())) {
        ASSIGN_VALUE_OR_RETURN_ERR(i, loadInt(dict["value"]));
      }
      splatNode =
          G_->createSplat(opName, mod_.uniqueType(ElemKind::Int64ITy, dims), i);
      break;
    }
    default:
      return MAKE_ERR("Unsupported datatype for ConstantFill.");
    }

    RETURN_IF_ERR(addNodeAsOutput(op, splatNode));

    return Error::success();
  }

  if (typeName == "UniformFill") {
    /*
     output: "fc/w"
     name: ""
     type: "UniformFill"
     arg {
       name: "max"
       f: 0.25
     }
     arg {
       name: "shape"
       ints: 1
       ints: 16
     }
     arg {
       name: "min"
       f: -0.25
     }
    */
    const auto &name = op.output(0);
    Tensor T;
    std::vector<dim_t> dim;
    if (dict.count("shape")) {
      ASSIGN_VALUE_OR_RETURN_ERR(dim, getShape<dim_t>(dict["shape"]));
    } else {
      RETURN_ERR_IF_NOT(op.input_size() > 0,
                        "If no shape provided, must have input shape.");

      bool inputAsShape = false;
      if (dict.count("input_as_shape")) {
        ASSIGN_VALUE_OR_RETURN_ERR(inputAsShape,
                                   loadInt(dict["input_as_shape"]));
      }

      if (inputAsShape) {
        Constant *in;
        ASSIGN_VALUE_OR_RETURN_ERR(in, getConstantByName(op.input(0)));
        RETURN_ERR_IF_NOT(in->dims().size() == 1,
                          opErrMsg(op, "Input must be 1D tensor."));
        RETURN_ERR_IF_NOT(in->getElementType() == ElemKind::Int64ITy,
                          opErrMsg(op, "Input must be of int64 type."));
        const auto handle = in->getHandle<int64_t>();
        dim.reserve(in->dims().size());
        for (auto d : handle) {
          dim.push_back(d);
        }
      } else {
        NodeValue input;
        ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));
        dim = input.dims();
      }
    }
    T.reset(ElemKind::FloatTy, dim);
    auto TH = T.getHandle<>();
    float tensorMin;
    ASSIGN_VALUE_OR_RETURN_ERR(tensorMin, loadFloat(dict["min"]));
    float tensorMax;
    ASSIGN_VALUE_OR_RETURN_ERR(tensorMax, loadFloat(dict["max"]));

    DLOG(INFO)
        << "The model contains UniformFill operator, which generates random "
           "numbers. This could be source of discrepancy.";

    // Uniformly generate random numbers in [tensorMin; tensorMax).
    for (auto &elem : TH) {
      elem = mod_.getPRNG().nextRandReal(tensorMin, tensorMax);
    }

    RETURN_IF_ERR(createAndRegisterConstant(name, std::move(T)));

    return Error::success();
  }

  // Load tensors with constant fill:
  if (typeName == "GaussianFill") {
    /*
     output: "data"
     name: ""
     type: "GaussianFill"
     arg {
       name: "mean"
       f: 0.0
     }
     arg {
       name: "std"
       f: 1.0
     }
     arg {
       name: "shape"
       ints: 1
       ints: 16
     }
     */

    const auto &name = op.output(0);
    if (getConstantByNameOrNull(name)) {
      return Error::success();
    }

    // The shape of the output is set by shape, if provided. Otherwise, it is
    // set by the shape of the input or the shape indicated by input if
    // input_as_shape is true
    NodeValue input;
    std::vector<dim_t> dims;
    if (dict.count("shape")) {
      ASSIGN_VALUE_OR_RETURN_ERR(dims, getShape<dim_t>(dict["shape"]));
    } else {
      RETURN_ERR_IF_NOT(op.input_size() > 0,
                        "If no shape provided, must have input shape.");

      bool inputAsShape = false;
      if (dict.count("input_as_shape")) {
        ASSIGN_VALUE_OR_RETURN_ERR(inputAsShape,
                                   loadInt(dict["input_as_shape"]));
      }

      if (inputAsShape) {
        Constant *in;
        ASSIGN_VALUE_OR_RETURN_ERR(in, getConstantByName(op.input(0)));
        RETURN_ERR_IF_NOT(in->dims().size() == 1,
                          opErrMsg(op, "Input must be 1D tensor."));
        RETURN_ERR_IF_NOT(in->getElementType() == ElemKind::Int64ITy,
                          opErrMsg(op, "Input must be of int64 type."));
        const auto handle = in->getHandle<int64_t>();
        dims.reserve(in->dims().size());
        for (auto dim : handle) {
          dims.push_back(dim);
        }
      } else {
        ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));
        dims = input.dims();
      }

      if (dict.count("extra_shape")) {
        std::vector<dim_t> extra_shape;
        ASSIGN_VALUE_OR_RETURN_ERR(extra_shape,
                                   getShape<dim_t>(dict["extra_shape"]));
        dims.insert(dims.end(), extra_shape.begin(), extra_shape.end());
      }
    }
    if ((!input && !dims.empty()) || input.dims().vec() != dims) {
      input =
          G_->createSplat("in", mod_.uniqueType(ElemKind::FloatTy, dims), 0.);
    }
    float mean;
    ASSIGN_VALUE_OR_RETURN_ERR(mean, loadFloat(dict["mean"]));
    float scale;
    ASSIGN_VALUE_OR_RETURN_ERR(scale, loadFloat(dict["std"]));

    auto GF = G_->createGaussianFill(opName, input, mean, scale,
                                     std::random_device{}());
    auto outputType =
        mod_.uniqueType(ElemKind::FloatTy, GF->getResult().dims());
    auto node = G_->createConvertTo(opName + ".ConvertOutput", GF, outputType);
    RETURN_IF_ERR(addNodeAsOutput(op, node));

    return Error::success();
  }

  return MAKE_ERR(unexpectedNodeErrorMessage(op, "Unsupported weight kind"));
}

Error Caffe2ModelLoader::loadWeightsFromNet(caffe2::NetDef &net) {
  for (auto &op : net.op()) {
    RETURN_IF_ERR(loadWeight(op));
  }
  return Error::success();
}

Caffe2ModelLoader::Caffe2ModelLoader(Function &F, Error *errPtr)
    : CommonOperatorLoader({}, {}, &F, errPtr) {
  deleteUnusedConstants();
}

Caffe2ModelLoader::Caffe2ModelLoader(
    const std::string &netDescFilename, const std::string &netWeightFilename,
    llvm::ArrayRef<const char *> names, llvm::ArrayRef<TypeRef> types,
    Function &F, Error *errPtr, OriginNameToTQPMap *originNameToTQPMap,
    bool loadUniquedDummyQParams, bool zeroScaleFP16Clip,
    bool clipQuantRangeToFP16)
    : CommonOperatorLoader(names, types, &F, errPtr,
                           /* loadIntoExistingModule */ false,
                           originNameToTQPMap, loadUniquedDummyQParams,
                           zeroScaleFP16Clip, clipQuantRangeToFP16) {
  // if errPtr already contains an error then don't continue with constructor
  if (errPtr && *errPtr) {
    return;
  }

  // Lambda to setup the Caffe2ModelLoader and return any Errors that
  // were raised.
  auto setup = [&]() -> Error {
    // The caffe2 network descriptor that we are deserializing.
    caffe2::NetDef networkDef;
    ASSIGN_VALUE_OR_RETURN_ERR(networkDef, loadProtoFile(netDescFilename));

    // The caffe2 weights that we are deserializing.
    caffe2::NetDef weightsDef;
    ASSIGN_VALUE_OR_RETURN_ERR(weightsDef, loadProtoFile(netWeightFilename));

    RETURN_IF_ERR(loadWeightsFromNet(weightsDef));
    RETURN_IF_ERR(loadNetwork(networkDef));

    // This is to ensure that the same processing done with
    // the same network, even if order of operators is different.
    F.orderNodes();
    RETURN_ERR_IF_NOT(F.verify(), "Function verification failed.");

    deleteUnusedConstants();

    RETURN_IF_ERR(verifyDummyQParams());

    return Error::success();
  };

  if (errPtr) {
    *errPtr = setup();
  } else {
    EXIT_ON_ERR(setup());
  }
}

Error Caffe2ModelLoader::initWithModule(caffe2::NetDef &networkDef,
                                        llvm::StringRef funNamePrefix,
                                        runtime::PrePartitionedConfig *PPC) {
  // Look for any partitions that will be needed. If there is no
  // partition_info then we create a single Function to load into. Otherwise
  // we create multiple Functions and switch between them as we load each
  // operator.
  std::unordered_map<Function *, std::vector<runtime::DeviceIDTy>> funToIDs;
  std::unordered_map<Function *, BackendSpecificOptions> funToOpts;
  if (networkDef.partition_info_size() == 0) {
    G_ = mod_.createFunction(funNamePrefix);
  } else {
    for (int i = 0; i < networkDef.partition_info_size(); i++) {
      const std::string &pName = networkDef.partition_info(i).name();
      const std::string funName = funNamePrefix.str() + "_" + pName;
      Function *PF = mod_.createFunction(funName);
      partNameToFun_[pName] = PF;
      for (auto id : networkDef.partition_info(i).device_id()) {
        funToIDs[PF].push_back(id);
      }

      // Now set up device options for this partition.
      auto &optsMap = funToOpts[PF];
      for (auto &backendOpts : networkDef.partition_info(i).backend_options()) {
        const std::string &backendName = backendOpts.backend_name();
        for (auto &keyVal : backendOpts.option()) {
          optsMap[backendName + "_" + keyVal.key()] = keyVal.val();
        }
      }
    }
  }

  RETURN_IF_ERR(loadNetwork(networkDef));

  // Now setup the pre-partitioned config if relevant.
  if (partNameToFun_.size()) {
    RETURN_ERR_IF_NOT(
        PPC, "Partitioned model but no config to store meta information in.");
    PPC->funcName = funNamePrefix.str();

    PPC->funcs.reserve(partNameToFun_.size());
    PPC->logicalIDs.reserve(partNameToFun_.size());
    for (auto &SF : partNameToFun_) {
      Function *F = SF.getValue();
      // Remove unused Functions from the module and skip them.
      if (F->getNodes().size() == 0) {
        mod_.eraseFunction(SF.getValue());
        continue;
      }
      // This is to ensure that the same processing done with
      // the same network, even if order of operators is different.
      F->orderNodes();
      PPC->funcs.push_back(F);
      PPC->logicalIDs.emplace_back(funToIDs[F]);
      PPC->backendSpecificOpts.emplace_back(funToOpts[F]);
      // Replication counts not currently loaded through C2, so default to 1.
      PPC->replicationCounts.emplace_back(1);
      // Backend hints not currently loaded through C2, so use default.
      PPC->backendHints.emplace_back();
      RETURN_ERR_IF_NOT(F->verify(), "Function verification failed.");
    }
  }

  deleteUnusedConstants();

  RETURN_IF_ERR(verifyDummyQParams());

  return Error::success();
}

Caffe2ModelLoader::Caffe2ModelLoader(const std::string &netDescFilename,
                                     const std::string &netWeightFilename,
                                     llvm::ArrayRef<const char *> names,
                                     llvm::ArrayRef<TypeRef> types, Module &mod,
                                     llvm::StringRef funNamePrefix,
                                     runtime::PrePartitionedConfig *PPC,
                                     Error *errPtr)
    : CommonOperatorLoader(names, types, mod, errPtr) {
  // if errPtr already contains an error then don't continue with constructor
  if (errPtr && *errPtr) {
    return;
  }

  // Lambda to setup the Caffe2ModelLoader and return any Errors that
  // were raised.
  auto setup = [&]() -> Error {
    // The caffe2 network descriptor that we are deserializing.
    caffe2::NetDef networkDef;
    ASSIGN_VALUE_OR_RETURN_ERR(networkDef, loadProtoFile(netDescFilename));

    // The caffe2 weights that we are deserializing.
    caffe2::NetDef weightsDef;
    ASSIGN_VALUE_OR_RETURN_ERR(weightsDef, loadProtoFile(netWeightFilename));

    RETURN_IF_ERR(loadWeightsFromNet(weightsDef));

    return initWithModule(networkDef, funNamePrefix, PPC);
  };

  if (errPtr) {
    *errPtr = setup();
  } else {
    EXIT_ON_ERR(setup());
  }
}

Caffe2ModelLoader::Caffe2ModelLoader(
    const std::string &modelStr, uint32_t weightsCount,
    const onnxTensorDescriptorV1 *weightDescriptors, Module &dummyMod,
    Error *errPtr, OriginNameToTQPMap *originNameToTQPMap,
    bool clipQuantRangeToFP16)
    : CommonOperatorLoader(
          {}, {}, dummyMod, errPtr,
          /* loadIntoExistingModule */ false, originNameToTQPMap,
          /* loadUniquedDummyQParams */ false, /* replaceDummyTQPs */ false,
          /* zeroScaleFP16Clip */ false, clipQuantRangeToFP16) {
  if (errPtr && *errPtr) {
    return;
  }

  constFoldInLoader_ = false;

  // Lambda to setup the Caffe2ModelLoader and return any Errors that were
  // raised.
  auto setup = [&]() -> Error {
    caffe2::NetDef networkDef;
    RETURN_ERR_IF_NOT(
        google::protobuf::TextFormat::ParseFromString(modelStr, &networkDef),
        "Error loading model from string");

    ArgumentDictionaryTy dict = loadArgumentMap(networkDef);

    std::unordered_set<std::string> initializers;
    if (dict.count("initializers")) {
      const auto &strings = dict.at("initializers")->strings();
      for (const auto &s : strings) {
        initializers.insert(s);
      }
    }

    RETURN_IF_ERR(loadWeights(weightsCount, weightDescriptors));

    RETURN_IF_ERR(loadInputs(networkDef, initializers));

    // Identify primary input sequence
    std::unordered_set<std::string> weights;
    for (uint32_t i = 0; i < weightsCount; ++i) {
      weights.emplace(weightDescriptors[i].name);
    }

    runtime::PrePartitionedConfig dummyPPC;
    return initWithModule(networkDef, "dummy", &dummyPPC);
  };

  *errPtr = setup();
}

Caffe2ModelLoader::Caffe2ModelLoader(
    const void *model, uint32_t modelSize, uint32_t weightsCount,
    const onnxTensorDescriptorV1 *weightDescriptors, Module &mod,
    llvm::StringRef funNamePrefix, runtime::PrePartitionedConfig *PPC,
    Error *errPtr, bool constFoldInLoader,
    OriginNameToTQPMap *originNameToTQPMap, bool loadUniquedDummyQParams,
    bool zeroScaleFP16Clip, bool clipQuantRangeToFP16)
    : CommonOperatorLoader({}, {}, mod, errPtr,
                           /* loadIntoExistingModule */ false,
                           originNameToTQPMap, loadUniquedDummyQParams,
                           /* replaceDummyTQPs */ false, zeroScaleFP16Clip,
                           clipQuantRangeToFP16) {
  // if errPtr already contains an error then don't continue with constructor
  if (errPtr && *errPtr) {
    return;
  }

  // Always override the default for folding in this constructor.
  constFoldInLoader_ = constFoldInLoader;

  // Lambda to setup the Caffe2ModelLoader and return any Errors that were
  // raised.
  auto setup = [&]() -> Error {
    caffe2::NetDef networkDef;
    ASSIGN_VALUE_OR_RETURN_ERR(networkDef, loadProto(model, modelSize));

    ArgumentDictionaryTy dict = loadArgumentMap(networkDef);

    std::unordered_set<std::string> initializers;
    if (dict.count("initializers")) {
      const auto &strings = dict.at("initializers")->strings();
      for (const auto &s : strings) {
        initializers.insert(s);
      }
    }

    RETURN_IF_ERR(loadWeights(weightsCount, weightDescriptors));

    RETURN_IF_ERR(loadInputs(networkDef, initializers));

    // Identify primary input sequence
    std::unordered_set<std::string> weights;
    for (uint32_t i = 0; i < weightsCount; ++i) {
      weights.emplace(weightDescriptors[i].name);
    }
    for (const auto &input : networkDef.external_input()) {
      if (!weights.count(input)) {
        positionalInputNames_.emplace_back(input);
      }
    }
    for (const auto &output : networkDef.external_output()) {
      positionalOutputNames_.emplace_back(output);
    }

    return initWithModule(networkDef, funNamePrefix, PPC);
  };

  if (errPtr) {
    *errPtr = setup();
  } else {
    EXIT_ON_ERR(setup());
  }
}
