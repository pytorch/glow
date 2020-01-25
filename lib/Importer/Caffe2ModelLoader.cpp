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

/// For the quantized Caffe2 ops, the activations are quantized to uint_8.
/// In Glow, the activations are quantized to int_8. Therefore, for the offset
/// read from quantized caffe2 model, we need to subtract 128(i.e. INT8_MIN) to
/// make the activations becomes int8_t.
/// For Glow: -128 <= orig_fp32/scale_1 + offset_1 <= 127
/// For Caffe2: 0 <= orig_fp32/scale_2 + offset_2 <= 255
/// Therefore, we can make scale_1 == scale_2, and offset_1 = offset2 - 128
const int32_t OFFSETSHIFT = 128;

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
}; // namespace glow

/// Legacy padding modes supported in caffe2.  These are used by MaxPool
/// operators, and are defined in caffe2_legacy.proto in the caffe2 source
/// tree.
enum LegacyPaddingMode { NOTSET, VALID, SAME, CAFFE_LEGACY_POOLING, N_MODES };

namespace {
/// Creates tensor \p T from the input \p in. Note, there is no data associated
/// with the Tensor. This method makes sure that the tensor is created with the
/// proper shape and element type.
Expected<LoadWeightResult>
createAndSetTensorType(const caffe2::TensorProto &in) {
  std::vector<dim_t> dim;
  for (auto d : in.dims()) {
    if (d == 0) {
      RETURN_ERR("0 dimemsion is not supported");
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
    RETURN_ERR(
        strFormat("FP32/16, Int32/64, Int8/Uint8 are supported. Got type"
                  " %s for tensor %s.",
                  caffe2::TensorProto_DataType_Name(in.data_type()).c_str(),
                  in.name().c_str()));
  }

  return Expected<LoadWeightResult>(std::move(result));
}

Expected<LoadWeightResult>
createAndSetTensorType(const caffe2::QTensorProto &in) {
  std::vector<dim_t> dim;
  for (auto d : in.dims()) {
    if (d == 0) {
      RETURN_ERR("0 dimemsion qtensor is not supported");
    }
    dim.push_back(d);
  }

  if (in.axis() != 1) {
    RETURN_ERR("axis must be 1");
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
    result.t->reset(ElemKind::Int8QTy, dim, scale, offset);
  } else if (in.data_type() == caffe2::TensorProto::UINT8) {
    result.t->reset(ElemKind::Int8QTy, dim, scale, offset - OFFSETSHIFT);
  } else if (in.data_type() == caffe2::TensorProto::INT32) {
    result.t->reset(ElemKind::Int32QTy, dim, scale, offset);
  } else {
    RETURN_ERR("Only int8, uint8, and int32 qtensors are supported");
  }

  return Expected<LoadWeightResult>(std::move(result));
}
} // namespace

/// Translates the protocol buffer node \p op into a random access map.
static ArgumentDictionaryTy loadArgumentMap(const caffe2::OperatorDef &op) {
  ArgumentDictionaryTy dict;
  for (auto &arg : op.arg()) {
    dict[arg.name()] = &arg;
  }
  return dict;
}

static Expected<std::vector<unsigned_t>>
getPads(const ArgumentDictionaryTy &dict) {
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
    return getShape<unsigned_t>(dict.at("pads"));
  }
  // Return default value 0 for pads.
  return std::vector<unsigned_t>{0, 0, 0, 0};
}

/// Translates the "order" field of dictionary \p dict into a channel number.
static Expected<unsigned_t> getChannel(const ArgumentDictionaryTy &dict) {
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
  RETURN_ERR("Invalid order field");
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
    return getShape<unsigned_t>(dict.at(name + "s"));
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
    codedstr.SetTotalBytesLimit(MAX_PROTO_SIZE, MAX_PROTO_SIZE);
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
  codedStream.SetTotalBytesLimit(MAX_PROTO_SIZE, MAX_PROTO_SIZE);
  caffe2::NetDef MP;
  bool parseNet = MP.ParseFromCodedStream(&codedStream);
  RETURN_ERR_IF_NOT(parseNet, "Failed to parse NetDef");
  return MP;
}

Expected<bool>
Caffe2ModelLoader::getBroadcast(const ArgumentDictionaryTy &dict) {
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
  unsigned_t dilation = 1;
  if (dict.count("dilation")) {
    ASSIGN_VALUE_OR_RETURN_ERR(dilation, loadInt(dict["dilation"]));
  }

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  Constant *w;
  ASSIGN_VALUE_OR_RETURN_ERR(w, getConstantByName(op.input(1)));

  // Transpose the weights to the right format. Glow expects to read the
  // weights in the format CRSK.
  // C - output_depth, R - filter_height, S - filter_width, K - input_depth.
  // Caffe2 "Conv" op always stores the weight as CKRS.
  Tensor wT;
  w->getPayload().transpose(&wT, NCHW2NHWC);
  w = G_.getParent()->createConstant(w->getName(), std::move(wT), "NHWC");

  // The structure of the conv weights is: CRSK. We take the C, which is the
  // number of filters. We use this value to calculate the size of the bias
  // if it is not specified.
  dim_t depth = w->dims()[0];

  // We expect the input to be NHWC.
  NodeValue finalIn;
  if (order == "NCHW") {
    finalIn = G_.createTranspose(opName, in, NCHW2NHWC)->getResult();
  } else {
    finalIn = in;
  }

  TypeRef finalInType = finalIn.getType();

  // Calculate the size and allocate the output buffer.
  ShapeNHWC idim = ShapeNHWC(finalInType->dims());
  auto outSz = calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides,
                                           pads, dilation);
  std::array<dim_t, 4> outDims = {{idim.n, outSz.first, outSz.second, depth}};

  // Try to find a loaded bias constant.
  Constant *bias = nullptr;
  if (op.input_size() > 2) {
    const auto &biasName = op.input(2);
    bias = getConstantByNameOrNull(biasName);
  }
  // Construct the bias constant if one wasn't found.
  if (!bias) {
    Tensor b(ElemKind::FloatTy, {depth});
    b.zero();
    bias = G_.getParent()->createConstant("conv.bias", std::move(b));
  }

  TypeRef outTy = G_.getParent()->uniqueType(ElemKind::FloatTy, outDims);

  Node *node = G_.createConv(opName, finalIn, w, bias, outTy, kernels, strides,
                             pads, group, dilation);
  if (op.type() == "ConvRelu") {
    node = G_.createRELU(opName + ".relu", node);
  }
  if (order == "NCHW") {
    // Transpose the output back.
    node = G_.createTranspose(opName, node, NHWC2NCHW);
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
  unsigned_t dilation = 1;
  if (dict.count("dilation")) {
    ASSIGN_VALUE_OR_RETURN_ERR(dilation, loadInt(dict["dilation"]));
  }

  // Group quantization only applies if there is more than one group.
  quantizeGroupwise &= group > 1;

  if (quantizeGroupwise && dilation > 1) {
    RETURN_ERR("Dilation not supported for group quantized convolution");
  }

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  Constant *w;
  ASSIGN_VALUE_OR_RETURN_ERR(w, getConstantByName(op.input(1)));

  // Transpose the weights to the right format. Glow expects to read the
  // weights in the format CRSK.
  // C - output_depth, R - filter_height, S - filter_width, K - input_depth.
  // For Caffe2 "Int8Conv" and "Int8ConvRelu", the weights always follows the
  // "order" arg.
  if (order != "NHWC") {
    Tensor wT;
    w->getPayload().transpose(&wT, NCHW2NHWC);
    w = G_.getParent()->createConstant(w->getName(), std::move(wT), "NHWC");
  }

  // The structure of the conv weights is: CRSK. We take the C, which is the
  // number of filters. We use this value to calculate the size of the bias
  // if it is not specified.
  dim_t depth = w->dims()[0];

  // We expect the input to be NHWC.
  NodeValue finalIn;
  if (order == "NCHW") {
    finalIn = G_.createTranspose(opName, in, NCHW2NHWC)->getResult();
  } else {
    finalIn = in;
  }

  TypeRef finalInType = finalIn.getType();

  // Calculate the size and allocate the output buffer.
  ShapeNHWC idim = ShapeNHWC(finalInType->dims());
  auto outSz = calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides,
                                           pads, dilation);
  std::array<dim_t, 4> outDims = {{idim.n, outSz.first, outSz.second, depth}};

  TypeRef outTy;

  RETURN_ERR_IF_NOT(dict.count("Y_zero_point"),
                    "missing zero point for quantized output type");
  RETURN_ERR_IF_NOT(dict.count("Y_scale"),
                    "missing Y_scale for quantized output type");

  // Try to find a loaded bias constant.
  Constant *bias = nullptr;
  if (op.input_size() > 2) {
    const auto &biasName = op.input(2);
    bias = getConstantByNameOrNull(biasName);
  }
  // Construct the bias constant if one wasn't found.
  if (!bias) {
    Tensor b(ElemKind::Int32QTy, {depth}, 1.0, 0);
    b.zero();
    bias = G_.getParent()->createConstant("conv.bias", std::move(b));
  }

  RETURN_ERR_IF_NOT(bias->getPayload().size() == depth,
                    "Loaded bias tensor of incorrect size");

  // Construct output type
  float scale;
  ASSIGN_VALUE_OR_RETURN_ERR(scale, loadFloat(dict["Y_scale"]));
  int32_t offset;
  ASSIGN_VALUE_OR_RETURN_ERR(offset, loadInt(dict["Y_zero_point"]));
  outTy = G_.getParent()->uniqueType(ElemKind::Int8QTy, outDims, scale,
                                     offset - OFFSETSHIFT);

  Node *node;

  if (quantizeGroupwise) {
    auto wScalesName = strFormat("%s_loaded_scales", op.input(1).c_str());
    auto wOffsetsName = strFormat("%s_loaded_offsets", op.input(1).c_str());
    Constant *wScales;
    Constant *wOffsets;
    ASSIGN_VALUE_OR_RETURN_ERR(wScales, getConstantByName(wScalesName));
    ASSIGN_VALUE_OR_RETURN_ERR(wOffsets, getConstantByName(wOffsetsName));

    node = G_.createChannelwiseQuantizedConv(opName, finalIn, w, bias, wScales,
                                             wOffsets, outTy, kernels, strides,
                                             pads, group);
  } else {
    // If the bias isn't quantized for a non group quantized conv, quantize it.
    const Tensor &biasTensor = bias->getPayload();
    if (biasTensor.getElementType() == ElemKind::FloatTy) {
      TensorQuantizationParams tqp;
      // Bias quantization params chosen to match the scale of the output
      // without requiring shifting.
      tqp.offset = 0;
      tqp.scale =
          finalInType->getScale() * w->getPayload().getType().getScale();

      Tensor quantizedBiasTensor =
          quantization::quantizeTensor(biasTensor, tqp, ElemKind::Int32QTy);

      bias = G_.getParent()->createConstant("conv.bias", quantizedBiasTensor);
    }

    node = G_.createConv(opName, finalIn, w, bias, outTy, kernels, strides,
                         pads, group, dilation);
  }

  if (op.type() == "Int8ConvRelu") {
    node = G_.createRELU(opName + ".relu", node);
  }

  if (order == "NCHW") {
    // Transpose the output back.
    node = G_.createTranspose(opName, node, NHWC2NCHW);
  }
  RETURN_IF_ERR(addNodeAsOutput(op, node));
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
  Function *tmpF = G_.getParent()->createFunction("eval_const_fold__");
  Caffe2ModelLoader tmpLoader(*tmpF, nullptr);
  bool foldStatus =
      !ERR_TO_BOOL(constantFoldInLoader<Caffe2ModelLoader, caffe2::OperatorDef>(
                       tmpF, tmpLoader, this, op),
                   /* log */ false);
  G_.getParent()->eraseFunction(tmpF);
  return foldStatus;
}

Error Caffe2ModelLoader::loadOperator(const caffe2::OperatorDef &op) {
  ArgumentDictionaryTy dict = loadArgumentMap(op);
  const std::string &typeName = op.type();

  // Check if operator is supported in parent class, CommonOperatorLoader.
  bool loadCommonOperatorSuccess;
  ASSIGN_VALUE_OR_RETURN_ERR(loadCommonOperatorSuccess,
                             tryLoadCommonOperator(typeName, op, dict));
  if (loadCommonOperatorSuccess) {
    return Error::success();
  }
  const std::string &opName = loadOperatorName(op);

  if (typeName == "Conv" || typeName == "ConvRelu") {
    return loadConv(op, dict);
  }

  if (typeName == "Int8Conv" || typeName == "Int8ConvRelu") {
    return loadConvQuantized(op, dict);
  }

  if (typeName == "Int8SumRelu") {
    RETURN_ERR_IF_NOT(op.input_size() == 2,
                      "Only Sum of 2 inputs is supported.");
    RETURN_ERR_IF_NOT(dict.count("Y_zero_point"),
                      "missing zero point for quantized outout type");
    RETURN_ERR_IF_NOT(dict.count("Y_scale"),
                      "missing Y_scale for quantized output type");
    NodeValue in0;
    ASSIGN_VALUE_OR_RETURN_ERR(in0, getNodeValueByName(op.input(0)));
    NodeValue in1;
    ASSIGN_VALUE_OR_RETURN_ERR(in1, getNodeValueByName(op.input(1)));
    auto outDims = in0.getType()->dims();
    float yScale;
    ASSIGN_VALUE_OR_RETURN_ERR(yScale, loadFloat(dict["Y_scale"]));
    int yZeroPoint;
    ASSIGN_VALUE_OR_RETURN_ERR(yZeroPoint, loadInt(dict["Y_zero_point"]));
    auto outTy = G_.getParent()->uniqueType(ElemKind::Int8QTy, outDims, yScale,
                                            yZeroPoint - OFFSETSHIFT);
    auto *add = G_.createAdd(opName + ".sum", outTy, in0, in1);
    auto *relu = G_.createRELU(opName + ".relu", add);
    RETURN_IF_ERR(addNodeAsOutput(op, relu));
    return Error::success();
  }

  if (typeName == "Int8Relu") {
    RETURN_ERR_IF_NOT(op.input_size() == 1, "Only one input is supported.");
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto outDims = in.getType()->dims();
    float yScale;
    ASSIGN_VALUE_OR_RETURN_ERR(yScale, loadFloat(dict["Y_scale"]));
    int yZeroPoint;
    ASSIGN_VALUE_OR_RETURN_ERR(yZeroPoint, loadInt(dict["Y_zero_point"]));
    auto outTy = G_.getParent()->uniqueType(ElemKind::Int8QTy, outDims, yScale,
                                            yZeroPoint - OFFSETSHIFT);
    auto *relu = G_.createRELU(opName, in, outTy);
    RETURN_IF_ERR(addNodeAsOutput(op, relu));
    return Error::success();
  }

  if (typeName == "Int8Quantize") {
    RETURN_ERR_IF_NOT(dict.count("Y_zero_point"),
                      "missing zero point for quantized output type");
    RETURN_ERR_IF_NOT(dict.count("Y_scale"),
                      "missing Y_scale for quantized output type");
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto outDims = in.getType()->dims();
    float yScale;
    ASSIGN_VALUE_OR_RETURN_ERR(yScale, loadFloat(dict["Y_scale"]));
    int yZeroPoint;
    ASSIGN_VALUE_OR_RETURN_ERR(yZeroPoint, loadInt(dict["Y_zero_point"]));
    auto outTy = G_.getParent()->uniqueType(ElemKind::Int8QTy, outDims, yScale,
                                            yZeroPoint - OFFSETSHIFT);
    Node *N = G_.createQuantize(opName, in, outTy);
    RETURN_IF_ERR(addNodeAsOutput(op, N));
    return Error::success();
  }

  if (typeName == "Int8Dequantize") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto *node = G_.createDequantize(opName, in);
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
    std::string order = "NCHW";
    if (dict.count("order")) {
      ASSIGN_VALUE_OR_RETURN_ERR(order, loadStr(dict["order"]));
    }
    // We expect the input to be NHWC.
    NodeValue finalIn;
    if (order == "NCHW") {
      finalIn = G_.createTranspose(opName, in, NCHW2NHWC)->getResult();
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
        RETURN_ERR("MaxPool nodes with legacy caffe padding are "
                   "deprecated and not supported.");
      }
    }

    Node *node = nullptr;

    if (typeName == "Int8MaxPool" || typeName == "Int8AveragePool") {
      // Create the node with quantized type.
      RETURN_ERR_IF_NOT(dict.count("Y_zero_point"),
                        "missing zero point for quantized output type");
      RETURN_ERR_IF_NOT(dict.count("Y_scale"),
                        "missing Y_scale for quantized output type");

      TypeRef finalInType = finalIn.getType();
      ShapeNHWC idim = ShapeNHWC(finalInType->dims());
      auto outSz =
          calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides, pads);
      std::array<dim_t, 4> outDims = {
          {idim.n, outSz.first, outSz.second, idim.c}};
      if (typeName == "Int8MaxPool") {
        // Int8Maxpool output quantization should be same as the input, so
        // just ignore the given params.
        node = G_.createMaxPool(opName, finalIn, kernels, strides, pads);
      } else {
        float yScale;
        ASSIGN_VALUE_OR_RETURN_ERR(yScale, loadFloat(dict["Y_scale"]));
        int yZeroPoint;
        ASSIGN_VALUE_OR_RETURN_ERR(yZeroPoint, loadInt(dict["Y_zero_point"]));
        auto outTy = G_.getParent()->uniqueType(
            ElemKind::Int8QTy, outDims, yScale, yZeroPoint - OFFSETSHIFT);
        node = G_.createAvgPool(opName, finalIn, outTy, kernels, strides, pads);
      }
    } else if (typeName == "MaxPool") {
      node = G_.createMaxPool(opName, finalIn, kernels, strides, pads);
    } else {
      node = G_.createAvgPool(opName, finalIn, kernels, strides, pads);
    }
    if (order == "NCHW") {
      unsigned resIdx = 0;
      if (llvm::isa<MaxPoolNode>(node)) {
        resIdx = MaxPoolNode::ResultIdx;
      } else if (llvm::isa<AvgPoolNode>(node)) {
        resIdx = AvgPoolNode::ResultIdx;
      } else {
        RETURN_ERR("Expected either Max or Avg Pool.");
      }
      // Transpose the output back.
      node = G_.createTranspose(opName, node->getNthResult(resIdx), NHWC2NCHW);
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
    auto *node = G_.createBatchNormalization(opName, in, bias, scale, mean, var,
                                             channel, epsilon);

    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "Bucketize") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    RETURN_ERR_IF_NOT(dict.count("boundaries"),
                      "Bucketize: Expected a boundaries member vector");
    std::vector<float> boundaries = getFloats(dict["boundaries"]);
    auto *node = G_.createBucketizeNode(opName, in, boundaries);
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
      finalIn = G_.createTranspose(opName, in, NCHW2NHWC)->getResult();
    } else {
      finalIn = in;
    }

    float heightScale;
    ASSIGN_VALUE_OR_RETURN_ERR(heightScale, loadFloat(dict["height_scale"]));
    float widthScale;
    ASSIGN_VALUE_OR_RETURN_ERR(widthScale, loadFloat(dict["width_scale"]));

    auto *node =
        G_.createResizeNearest(opName, finalIn, heightScale, widthScale);
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
      inputs.push_back(in);
    }

    // If axis exists it takes priority over channel.
    unsigned_t channel;
    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(channel, loadInt(dict["axis"]));
    } else {
      ASSIGN_VALUE_OR_RETURN_ERR(channel, getChannel(dict));
    }

    Node *node = G_.createConcat(opName, inputs, channel);

    unsigned_t addAxis = 0;
    if (dict.count("add_axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(addAxis, loadInt(dict["add_axis"]));
    }

    if (addAxis) {
      // When add axis is used, this means we have to add a new dimension
      // before the axis, instead of merging on the axis.
      std::vector<dim_t> outputDims = inputs[0].dims();
      unsigned i = 0;
      for (const auto &input : inputs) {
        RETURN_ERR_IF_NOT(
            outputDims[channel] == input.dims()[channel],
            strFormat("inputs need all to have the same dims for "
                      "concat with add_axis: input 0 (%s) vs "
                      "input %u (%s), %u vs %u, channel = %u",
                      op.input(0).c_str(), i, op.input(i).c_str(),
                      static_cast<unsigned>(outputDims[channel]),
                      static_cast<unsigned>(input.dims()[channel]), channel));
        ++i;
      }
      outputDims.insert(outputDims.begin() + channel, numInputs);
      node = G_.createReshape(opName, node, outputDims);
    }

    // If we add the axis then node is a Reshape, otherwise it should be
    // Concat.
    RETURN_ERR_IF_NOT(
        llvm::isa<ConcatNode>(node) || llvm::isa<ReshapeNode>(node),
        "Internal error: Node should either be a Concat or Reshape.");
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

    Constant *W;
    ASSIGN_VALUE_OR_RETURN_ERR(W, getConstantByName(op.input(1)));

    // Caffe2 stores the transposed W matrix. In here we first coerce W to a
    // 2D matrix size if necessary and then transpose it back.
    auto wDims = flattenCdr(W->dims(), axis_w);
    if (W->dims().size() > 2) {
      Tensor tmp;
      if (typeName == "FC" || typeName == "FCTransposed") {
        tmp.reset(ElemKind::FloatTy, {wDims.first, wDims.second});
      } else if (typeName == "FbFCPacked") {
        tmp.reset(ElemKind::Float16Ty, {wDims.first, wDims.second});
      } else {
        tmp.reset(ElemKind::Int8QTy, {wDims.first, wDims.second},
                  W->getType()->getScale(), W->getType()->getOffset());
      }
      tmp.copyRawFrom(&W->getPayload());
      W = G_.getParent()->createConstant(W->getName(), tmp);
    }

    if (typeName == "FC" || typeName == "Int8FC" || typeName == "FbFCPacked") {
      Tensor tmp;
      W->getPayloadMutable().transpose(&tmp, {1, 0});
      W = G_.getParent()->createConstant(W->getName(), tmp);
    }

    Constant *B;
    ASSIGN_VALUE_OR_RETURN_ERR(B, getConstantByName(op.input(2)));

    Node *node = nullptr;
    if (typeName == "Int8FC") {
      // Create the node with quantized type.
      RETURN_ERR_IF_NOT(dict.count("Y_zero_point"),
                        "missing zero point for quantized output type");
      RETURN_ERR_IF_NOT(dict.count("Y_scale"),
                        "missing Y_scale for quantized output type");
      float yScale;
      ASSIGN_VALUE_OR_RETURN_ERR(yScale, loadFloat(dict["Y_scale"]));
      int yZeroPoint;
      ASSIGN_VALUE_OR_RETURN_ERR(yZeroPoint, loadInt(dict["Y_zero_point"]));
      auto outTy = G_.getParent()->uniqueType(
          ElemKind::Int8QTy, {in.getType()->dims()[0], B->getType()->dims()[0]},
          yScale, yZeroPoint - OFFSETSHIFT);
      node = G_.createFullyConnected(opName, in, W, B, outTy, axis);
    } else if (typeName == "FbFCPacked") {
      auto fp16InputType =
          G_.getParent()->uniqueType(ElemKind::Float16Ty, in.getType()->dims());
      in = G_.createConvertTo("ConvertInput", in, fp16InputType);

      auto fp16BiasType =
          G_.getParent()->uniqueType(ElemKind::Float16Ty, B->getType()->dims());
      auto *fp16Bias = G_.createConvertTo("ConvertBias", B, fp16BiasType);
      TypeRef OT = G_.getParent()->uniqueType(
          ElemKind::Float16Ty, {in.dims()[0], B->getType()->dims()[0]});

      auto fc = G_.createFullyConnected(opName, in, W, fp16Bias, OT, axis);
      auto outputType =
          G_.getParent()->uniqueType(ElemKind::FloatTy, fc->getResult().dims());
      node = G_.createConvertTo("ConvertOutput", fc, outputType);
    } else {
      node = G_.createFullyConnected(opName, in, W, B, axis);
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
      RETURN_ERR_IF_NOT(totalReshapeSize == totalOriginalOutputSize,
                        strFormat("Cannot reshape from size %lu to size %lu",
                                  totalOriginalOutputSize, totalReshapeSize));

      node = G_.createReshape("fc.out", node, reshapeDims);
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

    Node *node = G_.createChannelShuffle(opName, in, group, kernel);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "Squeeze") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto dims = getShape(dict["dims"]);
    Node *node = G_.createSqueeze(opName, in, dims);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "Log") {
    // Load the inputs:
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    // Create the log:
    auto *R = G_.createLog(opName, in);
    RETURN_IF_ERR(addNodeAsOutput(op, R));
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

    auto *node = G_.createLogit(opName, input, eps);
    // Save the outputs:
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "EQ") {
    NodeValue in0;
    ASSIGN_VALUE_OR_RETURN_ERR(in0, getNodeValueByName(op.input(0)));
    NodeValue in1;
    ASSIGN_VALUE_OR_RETURN_ERR(in1, getNodeValueByName(op.input(1)));
    auto *node = G_.createCmpEQ(opName, in0, in1);
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

    auto *node = G_.createTile(opName, in, tiles, axis);
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
    auto *node = G_.createTranspose(opName, in, NCHW2NHWC);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "CopyCPUToMKL" || typeName == "CopyMKLToCPU" ||
      typeName == "Copy" || typeName == "EnsureCPUOutput" ||
      typeName == "EnsureDense") {
    // Glow does not support any of these ops now, so implement them as
    // no-ops.
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    RETURN_IF_ERR(addNodeAsOutput(op, in));
    return Error::success();
  }

  if (typeName == "Slice") {
    NodeValue data;
    ASSIGN_VALUE_OR_RETURN_ERR(data, getNodeValueByName(op.input(0)));

    auto starts = getShape<ssize_t>(dict["starts"]);
    auto ends = getShape<ssize_t>(dict["ends"]);

    std::vector<dim_t> newStarts, newEnds;
    RETURN_ERR_IF_NOT(starts.size() == ends.size(),
                      "Slice starts and ends must be the same size.");
    for (size_t i = 0; i < starts.size(); i++) {
      ssize_t newStart = starts[i];
      if (newStart == -1) {
        newStart = data.dims()[i];
      }
      RETURN_ERR_IF_NOT(newStart >= 0, "Indices should never be negative.");
      newStarts.push_back(newStart);

      ssize_t newEnd = ends[i];
      if (newEnd == -1) {
        newEnd = data.dims()[i];
      }
      RETURN_ERR_IF_NOT(newEnd >= 0, "Indices should never be negative.");
      newEnds.push_back(newEnd);
    }

    Node *SN = G_.createSlice(opName, data, newStarts, newEnds);
    RETURN_IF_ERR(addNodeAsOutput(op, SN));
    return Error::success();
  }

  if (typeName == "MatMul") {
    RETURN_IF_ERR(loadBatchMatMul(op, dict, false));
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
                        "Can only cast float to float.");
      break;
    }
    case caffe2::TensorProto_DataType_INT32:
    case caffe2::TensorProto_DataType_INT64: {
      RETURN_ERR_IF_NOT(in.getElementType() == ElemKind::Int64ITy,
                        "Can only cast int to int.");
      break;
    }
    default:
      RETURN_ERR("Unsupported Cast type.");
    }

    RETURN_IF_ERR(addNodeAsOutput(op, in));
    return Error::success();
  }

  if (typeName == "HalfToFloat") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto convertedType =
        G_.getParent()->uniqueType(ElemKind::FloatTy, in.getType()->dims());
    auto *R = G_.createConvertTo("ConvertInput", in, convertedType);
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
    NodeValue indices2D =
        G_.createReshape("indices.2d", indices, {indices.dims()[0], 1});
    Node *SAN = G_.createScatterData(opName, data, indices2D, slices);
    RETURN_IF_ERR(addNodeAsOutput(op, SAN));
    return Error::success();
  }

  if (typeName == "ConstantFill" || typeName == "GivenTensorIntFill" ||
      typeName == "GivenTensorInt64Fill") {
    RETURN_IF_ERR(loadWeight(op));
    return Error::success();
  }

  if (typeName == "SigmoidCrossEntropyWithLogits") {
    NodeValue logits;
    ASSIGN_VALUE_OR_RETURN_ERR(logits, getNodeValueByName(op.input(0)));
    NodeValue targets;
    ASSIGN_VALUE_OR_RETURN_ERR(targets, getNodeValueByName(op.input(1)));
    Node *SCEL =
        G_.createSigmoidCrossEntropyWithLogits(opName, logits, targets);
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

    Node *EL = G_.createElementwiseLinear(opName, X, w, b, axis);
    RETURN_IF_ERR(addNodeAsOutput(op, EL));
    return Error::success();
  }

  if (typeName == "AveragedLoss") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto *node = G_.createBatchedReduceMean(opName, in, 0);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "Mod") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    int64_t divisor;
    ASSIGN_VALUE_OR_RETURN_ERR(divisor, loadInt(dict["divisor"]));

    RETURN_ERR_IF_NOT(divisor >= 1, "Divisor must not be less than 1.");

    bool signFollowDivisor = false;
    if (dict.count("sign_follow_divisor")) {
      ASSIGN_VALUE_OR_RETURN_ERR(signFollowDivisor,
                                 loadInt(dict["sign_follow_divisor"]));
    }

    auto *node = G_.createModulo(opName, in, divisor, signFollowDivisor);
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
    RETURN_ERR_IF_NOT(lengths.dims().size() == 1, "lengths must be a vector.");
    RETURN_ERR_IF_NOT(indices.dims().size() == 1, "indices must be a vector.");

    Node *node;
    if (isFused) {
      // There is no specific fused quantized type in Caffe2, so we will load
      // UInt8QTy. We then change it from UInt8QTy to UInt8FusedQTy or
      // UInt4FusedFP16QTy here if necessary -- another user could have already
      // changed it.
      if (dataS->getElementType() != ElemKind::UInt8FusedQTy) {
        RETURN_ERR_IF_NOT(dataS->getElementType() == ElemKind::UInt8QTy,
                          "Data must be UInt8QTy.");
        // Use dummy 0.0/0 as scale/offset, since the actual scales/offsets
        // are fused inline with the data.
        TypeRef fusedTy = G_.getParent()->uniqueType(
            is4Bit ? ElemKind::UInt4FusedFP16QTy : ElemKind::UInt8FusedQTy,
            dataS->dims(), 0.0, 0);
        dataS->setType(Storage::OutputIdx, fusedTy);
        // If the node is a Constant set the payload type as well.
        if (auto dataConstant = llvm::dyn_cast<Constant>(data)) {
          dataConstant->setPayloadType(fusedTy);
        }
      }

      // No other work to do, since the data is already loaded fused, so just
      // create the new node with its inputs.
      if (isWeighted) {
        node = G_.createFusedRowwiseQuantizedSparseLengthsWeightedSum(
            opName, dataS, weights, indices, lengths);
      } else {
        node = G_.createFusedRowwiseQuantizedSparseLengthsSum(opName, dataS,
                                                              indices, lengths);
      }

      if (is4Bit) {
        node = G_.createConvertTo(opName, node, ElemKind::FloatTy);
      }
    } else {
      NodeValue scalesBiases;
      ASSIGN_VALUE_OR_RETURN_ERR(scalesBiases,
                                 getNodeValueByName(op.input(scalesBiasesIdx)));

      Constant *scalesBiasesC = llvm::dyn_cast<Constant>(scalesBiases);
      RETURN_ERR_IF_NOT(scalesBiasesC, "scales_biases must be Constant.");
      RETURN_ERR_IF_NOT(scalesBiases.dims().size() == 2,
                        "scale_bias has to be a matrix.");
      RETURN_ERR_IF_NOT(
          scalesBiases.dims()[0] == numRows,
          "scale_bias must have the same number of rows as data.");
      RETURN_ERR_IF_NOT(scalesBiases.dims()[1] == 2,
                        "Second dim of scale_bias has to be equal to 2.");

      // Now strip out the scales and biases into their own tensors.
      Constant *dataScales = G_.getParent()->createConstant(
          ElemKind::FloatTy, {numRows}, "dataScales");
      Constant *dataOffsets = G_.getParent()->createConstant(
          ElemKind::FloatTy, {numRows}, "dataOffsets");

      auto dataScalesH = dataScales->getHandle<float>();
      auto dataOffsetsH = dataOffsets->getHandle<float>();
      auto scalesBiasesH = scalesBiasesC->getHandle<float>();
      for (dim_t i = 0, e = numRows; i < e; i++) {
        dataScalesH.at({i}) = scalesBiasesH.at({i, 0});
        dataOffsetsH.at({i}) = scalesBiasesH.at({i, 1});
      }

      // Now create the actual node.
      if (isWeighted) {
        node = G_.createRowwiseQuantizedSparseLengthsWeightedSum(
            opName, dataS, dataScales, dataOffsets, weights, indices, lengths);
      } else {
        node = G_.createRowwiseQuantizedSparseLengthsSum(
            opName, dataS, dataScales, dataOffsets, indices, lengths);
      }
    }

    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  if (typeName == "LengthsRangeFill") {
    NodeValue lengths;
    ASSIGN_VALUE_OR_RETURN_ERR(lengths, getNodeValueByName(op.input(0)));
    RETURN_ERR_IF_NOT(lengths.dims().size() == 1,
                      "lengths must be a 1D vector.");

    auto maxOutputSizeIt = dict.find("maxOutputSize");
    RETURN_ERR_IF_NOT(maxOutputSizeIt != dict.end(),
                      "Require maxOutputSize when loading LengthsRangeFill.");
    unsigned_t maxOutputSize;
    ASSIGN_VALUE_OR_RETURN_ERR(maxOutputSize, loadInt(maxOutputSizeIt->second));

    auto *LRF = G_.createLengthsRangeFill(opName, lengths, maxOutputSize);
    RETURN_IF_ERR(addNodeAsOutput(op, LRF));

    return Error::success();
  }

  RETURN_ERR(unexpectedNodeErrorMessage(op, "Unsupported operator."));
}

template <class TensorProtoType>
Error Caffe2ModelLoader::loadInputsWithTensorProtoType(
    const caffe2::NetDef &net, bool loadInputsAsPlaceholders,
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
    return resOrErr.takeError();
  }

  bool multiQParamsLoaded = loadRes.scales || loadRes.offsets;
  RETURN_ERR_IF_NOT(
      (!multiQParamsLoaded || (loadRes.scales && loadRes.offsets)),
      "For tensors with separate qparams, both scales and offsets must be "
      "loaded");

  if (loadInputsAsPlaceholders) {
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

Error Caffe2ModelLoader::loadInputs(const caffe2::NetDef &net,
                                    bool loadInputsAsPlaceholders) {
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
          net, loadInputsAsPlaceholders, in));
    }
  }

  // Load all quantized tensor input
  if (qarg) {
    for (const auto &in : qarg->qtensors()) {
      RETURN_IF_ERR(loadInputsWithTensorProtoType<caffe2::QTensorProto>(
          net, loadInputsAsPlaceholders, in));
    }
  }

  return Error::success();
}

Error Caffe2ModelLoader::loadNetwork(caffe2::NetDef &net) {
  // Make a claim on the unique name of all output Placeholders.
  for (int i = 0; i < net.external_output_size(); i++) {
    auto &outputName = net.external_output(i);
    G_.getParent()->registerStorageName(legalizeName(outputName));
  }

  /// Load the network operators:
  for (int i = 0; i < net.op_size(); i++) {
    auto &op = net.op(i);
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
    ASSIGN_VALUE_OR_RETURN_ERR(r, getNodeValueByName(outputName));

    PlaceholderList &PHList = G_.getParent()->getPlaceholders();
    // Create a Placeholder with the previously claimed name.
    auto *PH = new Placeholder(legalizeName(outputName),
                               G_.getParent()->uniqueType(*r.getType()), false,
                               ANY_LAYOUT);
    PHList.push_back(PH);
    auto *SN = G_.createSave("save_" + outputName, r, PH);
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

  // Load tensors with values:
  if (typeName == "GivenTensorFill" || typeName == "GivenTensorIntFill" ||
      typeName == "GivenTensorInt64Fill") {
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
    auto dim = getShape(dict["shape"]);
    auto const &values = dict["values"];
    RETURN_ERR_IF_NOT(op.output_size() == 1,
                      "GivenTensorFill must have exactly 1 output");
    Tensor T;
    if (typeName == "GivenTensorFill") {
      RETURN_IF_ERR(
          fillTensor<float>(T, ElemKind::FloatTy, dim, values->floats()));
    } else if (typeName == "GivenTensorIntFill") {
      RETURN_IF_ERR(
          fillTensor<int32_t>(T, ElemKind::Int32ITy, dim, values->ints()));
    } else if (typeName == "GivenTensorInt64Fill") {
      RETURN_IF_ERR(
          fillTensor<int64_t>(T, ElemKind::Int64ITy, dim, values->ints()));
    } else {
      RETURN_ERR(strFormat("Unhandled tensor fill type: %s", typeName.c_str()));
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
      auto dim = getShape(dict["shape"]);
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

      auto dim = getShape(dict["shape"]);

      RETURN_ERR_IF_NOT(dict.count("Y_zero_point"),
                        "missing zero point for quantized output type");
      RETURN_ERR_IF_NOT(dict.count("Y_scale"),
                        "missing Y_scale for quantized output type");

      float scale;
      ASSIGN_VALUE_OR_RETURN_ERR(scale, loadFloat(dict["Y_scale"]));
      int32_t offset;
      ASSIGN_VALUE_OR_RETURN_ERR(offset, loadInt(dict["Y_zero_point"]));
      size_t i = 0;
      if (typeName == "Int8GivenTensorFill") {
        // Although in Caffe2 quantized model, the weights is int8 quantized,
        // the weights is stored in uint8_t format due to that Caffe2 requires
        // the type of input and weights must be the same. Therefore, we need
        // to convert it to int8 by subtracting 128.
        T.reset(ElemKind::Int8QTy, dim, scale, offset - OFFSETSHIFT);
        auto TH = T.getHandle<int8_t>();
        std::string str = dict["values"]->s();
        for (; i < str.size(); i++) {
          TH.raw(i) = ((uint8_t)(str.c_str()[i]) - OFFSETSHIFT);
        }
      } else {
        T.reset(ElemKind::Int32QTy, dim, scale, offset);
        auto TH = T.getHandle<int32_t>();
        for (auto num : dict["values"]->ints()) {
          TH.raw(i++) = num;
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

    Tensor T;

    // The shape is set either the shape argument, or from another input
    // tensor. Shape takes priority over input.
    std::vector<dim_t> dims;
    if (dict.count("shape")) {
      dims = getShape(dict["shape"]);
    } else {
      RETURN_ERR_IF_NOT(op.input_size() > 0,
                        "If no shape provided, must have input shape.");
      // It must be registered as a Constant because it must be statically set
      // already, as shapes must be statically known.
      Constant *in;
      ASSIGN_VALUE_OR_RETURN_ERR(in, getConstantByName(op.input(0)));
      dims = in->dims();
    }

    int to = caffe2::TensorProto_DataType_FLOAT;
    if (dict.count("dtype")) {
      ASSIGN_VALUE_OR_RETURN_ERR(to, loadInt(dict["dtype"]));
    }

    switch (to) {
    case caffe2::TensorProto_DataType_FLOAT: {
      T.reset(ElemKind::FloatTy, dims);
      auto TH = T.getHandle<float>();
      float f = 0.0f;
      if ((dict.count("value") && dict["value"]->has_f())) {
        ASSIGN_VALUE_OR_RETURN_ERR(f, loadFloat(dict["value"]));
      }
      TH.clear(f);
      break;
    }
    case caffe2::TensorProto_DataType_INT32:
    case caffe2::TensorProto_DataType_INT64:
    case caffe2::TensorProto_DataType_BOOL: {
      T.reset(ElemKind::Int64ITy, dims);
      auto TH = T.getHandle<int64_t>();
      int i = 0;
      if ((dict.count("value") && dict["value"]->has_i())) {
        ASSIGN_VALUE_OR_RETURN_ERR(i, loadInt(dict["value"]));
      }
      TH.clear(i);
      break;
    }
    default:
      RETURN_ERR("Unsupported datatype for ConstantFill.");
    }

    RETURN_IF_ERR(createAndRegisterConstant(name, std::move(T)));

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
    auto dim = getShape(dict["shape"]);
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
      elem = G_.getParent()->getPRNG().nextRandReal(tensorMin, tensorMax);
    }

    RETURN_IF_ERR(createAndRegisterConstant(name, std::move(T)));

    return Error::success();
  }

  RETURN_ERR(unexpectedNodeErrorMessage(op, "Unsupported weight kind"));
}

Error Caffe2ModelLoader::loadWeightsFromNet(caffe2::NetDef &net) {
  for (auto &op : net.op()) {
    RETURN_IF_ERR(loadWeight(op));
  }
  return Error::success();
}

Caffe2ModelLoader::Caffe2ModelLoader(Function &F, Error *errPtr)
    : CommonOperatorLoader({}, {}, F, errPtr) {
  deleteUnusedConstants();
}

Caffe2ModelLoader::Caffe2ModelLoader(const std::string &netDescFilename,
                                     const std::string &netWeightFilename,
                                     llvm::ArrayRef<const char *> names,
                                     llvm::ArrayRef<TypeRef> types, Function &F,
                                     Error *errPtr)
    : CommonOperatorLoader(names, types, F, errPtr) {
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

    return Error::success();
  };

  if (errPtr) {
    *errPtr = setup();
  } else {
    EXIT_ON_ERR(setup());
  }
}

Caffe2ModelLoader::Caffe2ModelLoader(
    const void *model, uint32_t modelSize, uint32_t weightsCount,
    const onnxTensorDescriptorV1 *weightDescriptors, Function &F,
    bool loadInputsAsPlaceholders, Error *errPtr, bool constFoldInLoader)
    : CommonOperatorLoader({}, {}, F, errPtr) {
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

    RETURN_IF_ERR(loadWeights(weightsCount, weightDescriptors));

    RETURN_IF_ERR(loadInputs(networkDef, loadInputsAsPlaceholders));

    // TODO: in Caffe2ModelLoader, setOutputNodes is actually inside
    // loadNetwork, maybe we should make it a separate function?
    RETURN_IF_ERR(loadNetwork(networkDef));

    // This is to ensure that the same processing done with
    // the same network, even if order of operators is different.
    F.orderNodes();

    deleteUnusedConstants();

    return Error::success();
  };

  if (errPtr) {
    *errPtr = setup();
  } else {
    EXIT_ON_ERR(setup());
  }
}
