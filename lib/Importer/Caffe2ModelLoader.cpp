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
/// For Glow: -127 <= orig_fp32/scale_1 + offset_1 < 128
/// For Caffe2: 0 <= orig_fp32/scale_2 + offset_2 < 255
/// Therefore, we can make scale_1 == scale_2, and offset_1 = offset2 - 128
const int32_t OFFSETSHIFT = 128;

namespace {
/// Creates tensor \p T from the input \p in. Note, there is no data associated
/// with the Tensor. This method makes sure that the tensor is created with the
/// proper shape and element type.
llvm::Error setTensorType(const caffe2::TensorProto &in, Tensor *T) {
  std::vector<size_t> dim;
  for (auto d : in.dims()) {
    if (d == 0) {
      RETURN_ERR("0 dimemsion is not supported");
    }
    dim.push_back(d);
  }

  if (in.data_type() == caffe2::TensorProto::FLOAT) {
    T->reset(ElemKind::FloatTy, dim);
    return llvm::Error::success();
  } else if (in.data_type() == caffe2::TensorProto::INT64) {
    T->reset(ElemKind::Int64ITy, dim);
    return llvm::Error::success();
  } else if (in.data_type() == caffe2::TensorProto::INT32) {
    T->reset(ElemKind::Int32ITy, dim);
    return llvm::Error::success();
  } else {
    RETURN_ERR("Only float and index tensors are supported");
  }
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

static llvm::Expected<std::vector<unsigned_t>>
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
static llvm::Expected<unsigned_t> getChannel(const ArgumentDictionaryTy &dict) {
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

static llvm::Expected<std::vector<unsigned_t>>
getSizeHW(ArgumentDictionaryTy &dict, const std::string &name,
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

llvm::Expected<caffe2::NetDef>
Caffe2ModelLoader::loadProtoFile(const std::string &filename) {
  std::ifstream ff(filename, std::ios::in | std::ios::binary);
  RETURN_ERR_IF_NOT(ff, "Can't find the model or network files.");

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

llvm::Expected<caffe2::NetDef>
Caffe2ModelLoader::loadProto(const void *c2Model, size_t c2ModelSize) {
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

llvm::Expected<bool>
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

llvm::Error Caffe2ModelLoader::loadOperator(const caffe2::OperatorDef &op) {
  ArgumentDictionaryTy dict = loadArgumentMap(op);
  const std::string &typeName = op.type();

  // Check if operator is supported in parent class, CommonOperatorLoader.
  bool loadCommonOperatorSuccess;
  ASSIGN_VALUE_OR_RETURN_ERR(loadCommonOperatorSuccess,
                             tryLoadCommonOperator(typeName, op, dict));
  if (loadCommonOperatorSuccess) {
    return llvm::Error::success();
  }
  const std::string &opName = loadOperatorName(op);

  if (typeName == "Conv" || typeName == "Int8Conv" ||
      typeName == "Int8ConvRelu") {
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

    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    Tensor *w;
    ASSIGN_VALUE_OR_RETURN_ERR(w, getTensorByName(op.input(1)));

    // Transpose the weights to the right format. Glow expects to read the
    // weights in the format CRSK.
    // C - output_depth, R - filter_height, S - filter_width, K - input_depth.
    // Caffe2 "Conv" op always stores the weight as CKRS, while for "Int8Conv",
    // and "Int8ConvRelu", the weights always follows the "order" arg.
    Tensor wtag;
    if (typeName != "Conv" && order == "NHWC") {
      wtag.assign(w);
    } else {
      w->transpose(&wtag, NCHW2NHWC);
    }

    // The structure of the conv weigts is: NHWC. We take the C, which is the
    // number of filters. We use this value to calculate the size of the bias
    // if it is not specified.
    size_t depth = wtag.dims()[0];

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
    auto outSz =
        calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides, pads);
    std::array<size_t, 4> outDims = {
        {idim.n, outSz.first, outSz.second, depth}};

    TypeRef outTy;
    Constant *filter;
    Constant *bias;
    if (typeName == "Conv") {
      // Construct the Bias field.
      Tensor biasTensor(ElemKind::FloatTy, {depth});
      biasTensor.zero();

      // Check if we have a serialized bias vector.
      if (op.input_size() > 2) {
        const auto &biasTensorName = op.input(2);
        if (tensors_.count(biasTensorName)) {
          // Load the serialized bias vector.
          Tensor *b;
          ASSIGN_VALUE_OR_RETURN_ERR(b, getTensorByName(biasTensorName));
          biasTensor.assign(b);
        }
      }
      outTy = G_.getParent()->uniqueType(ElemKind::FloatTy, outDims);
      filter = G_.getParent()->createConstant("conv.filter", wtag);
      bias = G_.getParent()->createConstant("conv.bias", biasTensor);
    } else {
      RETURN_ERR_IF_NOT(dict.count("Y_zero_point"),
                        "missing zero point for quantized output type");
      RETURN_ERR_IF_NOT(dict.count("Y_scale"),
                        "missing Y_scale for quantized output type");
      // Construct the Bias field.
      Tensor biasTensor(ElemKind::Int32QTy, {depth}, 1.0, 0);
      biasTensor.zero();
      // Check if we have a serialized bias vector.
      if (op.input_size() > 2) {
        const auto &biasTensorName = op.input(2);
        if (tensors_.count(biasTensorName)) {
          // Load the serialized bias vector.
          Tensor *b;
          ASSIGN_VALUE_OR_RETURN_ERR(b, getTensorByName(biasTensorName));
          biasTensor.assign(b);
        }
      }
      float scale;
      ASSIGN_VALUE_OR_RETURN_ERR(scale, loadFloat(dict["Y_scale"]));
      int32_t offset;
      ASSIGN_VALUE_OR_RETURN_ERR(offset, loadInt(dict["Y_zero_point"]));
      outTy = G_.getParent()->uniqueType(ElemKind::Int8QTy, outDims, scale,
                                         offset - OFFSETSHIFT);

      // Construct the quantized Filter and bias field.
      filter = G_.getParent()->createConstant(
          ElemKind::Int8QTy, wtag.dims(), wtag.getType().getScale(),
          wtag.getType().getOffset(), "conv.filter");
      filter->assign(&wtag);
      bias = G_.getParent()->createConstant(
          ElemKind::Int32QTy, biasTensor.dims(),
          biasTensor.getType().getScale(), biasTensor.getType().getOffset(),
          "conv.bias");
      bias->assign(&biasTensor);
    }

    Node *node = G_.createConv(opName, finalIn, filter, bias, outTy, kernels,
                               strides, pads, group);

    if (order == "NCHW") {
      // Transpose the output back.
      node = G_.createTranspose(opName, node, NHWC2NCHW);
    }
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  if (typeName == "Int8SumRelu") {
    RETURN_ERR_IF_NOT(op.input_size() == 2,
                      "Only Sum of 2 inputs is supported.");
    RETURN_ERR_IF_NOT(dict.count("Y_zero_point"),
                      "missing zero point for quantized outout type");
    RETURN_ERR_IF_NOT(dict.count("Y_scale"),
                      "missing Y_scale for quantized output type");
    NodeValue in0;
    ASSIGN_VALUE_OR_RETURN_ERR(in0,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    NodeValue in1;
    ASSIGN_VALUE_OR_RETURN_ERR(in1,
                               getNodeValueOrCreateConstantByName(op.input(1)));
    auto outDims = in0.getType()->dims();
    float yScale;
    ASSIGN_VALUE_OR_RETURN_ERR(yScale, loadFloat(dict["Y_scale"]));
    int yZeroPoint;
    ASSIGN_VALUE_OR_RETURN_ERR(yZeroPoint, loadInt(dict["Y_zero_point"]));
    auto outTy = G_.getParent()->uniqueType(ElemKind::Int8QTy, outDims, yScale,
                                            yZeroPoint - OFFSETSHIFT);
    auto *node = G_.createAdd(opName, outTy, in0, in1);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  if (typeName == "Int8Quantize") {
    RETURN_ERR_IF_NOT(dict.count("Y_zero_point"),
                      "missing zero point for quantized output type");
    RETURN_ERR_IF_NOT(dict.count("Y_scale"),
                      "missing Y_scale for quantized output type");
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    auto outDims = in.getType()->dims();
    float yScale;
    ASSIGN_VALUE_OR_RETURN_ERR(yScale, loadFloat(dict["Y_scale"]));
    int yZeroPoint;
    ASSIGN_VALUE_OR_RETURN_ERR(yZeroPoint, loadInt(dict["Y_zero_point"]));
    auto outTy = G_.getParent()->uniqueType(ElemKind::Int8QTy, outDims, yScale,
                                            yZeroPoint - OFFSETSHIFT);
    Node *N = G_.createQuantize(opName, in, outTy);
    RETURN_IF_ERR(addNodeAsOutput(op, N));
    return llvm::Error::success();
  }

  if (typeName == "Int8Dequantize") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    auto *node = G_.createDequantize(opName, in);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  if (typeName == "MaxPool" || typeName == "AveragePool" ||
      typeName == "Int8MaxPool" || typeName == "Int8AveragePool") {
    // Load the inputs:
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
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

    // If 'global_pooling' is set then the operation will pool over the size of
    // the input by doing: kernels = {height, width}.
    if (dict.count("global_pooling")) {
      auto Ty = in.getType();
      kernels[0] = Ty->dims()[2];
      kernels[1] = Ty->dims()[3];
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
      std::array<size_t, 4> outDims = {
          {idim.n, outSz.first, outSz.second, idim.c}};
      if (typeName == "Int8MaxPool") {
        // Int8Maxpool output quantization should be same as the input, so just
        // ignore the given params.
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
      // Transpose the output back.
      node = G_.createTranspose(opName, node, NHWC2NCHW);
    }
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  if (typeName == "SpatialBN") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    Tensor *scale;
    ASSIGN_VALUE_OR_RETURN_ERR(scale, getTensorByName(op.input(1)));
    Tensor *bias;
    ASSIGN_VALUE_OR_RETURN_ERR(bias, getTensorByName(op.input(2)));
    Tensor *mean;
    ASSIGN_VALUE_OR_RETURN_ERR(mean, getTensorByName(op.input(3)));
    Tensor *var;
    ASSIGN_VALUE_OR_RETURN_ERR(var, getTensorByName(op.input(4)));
    float epsilon = 1e-5f; // default
    auto epsilonIt = dict.find("epsilon");
    if (epsilonIt != dict.end()) {
      ASSIGN_VALUE_OR_RETURN_ERR(epsilon, loadFloat(epsilonIt->second));
    }

    unsigned_t channel;
    ASSIGN_VALUE_OR_RETURN_ERR(channel, getChannel(dict));
    auto *scaleV = G_.getParent()->createConstant("scale", *scale);
    auto *biasV = G_.getParent()->createConstant("bias", *bias);
    auto *meanV = G_.getParent()->createConstant("mean", *mean);
    auto *varV = G_.getParent()->createConstant("var", *var);
    auto *node = G_.createBatchNormalization(opName, in, biasV, scaleV, meanV,
                                             varV, channel, epsilon);

    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  if (typeName == "Concat") {
    const unsigned numInputs = op.input_size();
    llvm::SmallVector<NodeValue, 4> inputs;
    inputs.reserve(numInputs);
    for (unsigned i = 0; i < numInputs; i++) {
      NodeValue in;
      ASSIGN_VALUE_OR_RETURN_ERR(
          in, getNodeValueOrCreateConstantByName(op.input(i)));
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
      // When add axis is used, this means we have to add a new dimension before
      // the axis, instead of merging on the axis.
      std::vector<size_t> outputDims = inputs[0].dims();
      for (const auto &input : inputs) {
        RETURN_ERR_IF_NOT(
            outputDims[channel] == input.dims()[channel],
            "inputs need all to have the same dims for concat with add_axis");
      }
      outputDims.insert(outputDims.begin() + channel, numInputs);
      node = G_.createReshape(opName, node, outputDims);
    }

    // If we add the axis then node is a Reshape, otherwise it should be Concat.
    RETURN_ERR_IF_NOT(
        llvm::isa<ConcatNode>(node) || llvm::isa<ReshapeNode>(node),
        "Internal error: Node should either be a Concat or Reshape.");
    NodeValue finalNode = llvm::isa<ConcatNode>(node)
                              ? NodeValue(node, ConcatNode::ResultIdx)
                              : NodeValue(node, ReshapeNode::ResultIdx);
    nodeValueByName_[op.output(0)] = finalNode;
    // Concat may have a second output in Caffe2 (split_info), but we don't use
    // it for inference
    return llvm::Error::success();
  }

  if (typeName == "FC" || typeName == "FCTransposed" || typeName == "Int8FC") {
    // Load the inputs:
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    if (in.getType()->dims().size() > 2) {
      size_t axis = 1;
      if (dict.count("axis")) {
        ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict["axis"]));
      }

      in = G_.createFlatten("fc.in", in, axis);
    }

    // Load weights.
    Tensor *w;
    ASSIGN_VALUE_OR_RETURN_ERR(w, getTensorByName(op.input(1)));
    Tensor *b;
    ASSIGN_VALUE_OR_RETURN_ERR(b, getTensorByName(op.input(2)));
    unsigned_t axis_w = 1;
    if (dict.count("axis_w")) {
      ASSIGN_VALUE_OR_RETURN_ERR(axis_w, loadInt(dict["axis_w"]));
    }

    // Caffe2 stores the transposed W matrix. In here we first coerce W to a 2D
    // matrix size if necessay and then transpose it back.
    Tensor tmp;
    if (w->dims().size() > 2) {
      auto wDims = flattenCdr(w->dims(), axis_w);
      if (typeName == "FC" || typeName == "FCTransposed") {
        tmp.reset(ElemKind::FloatTy, {wDims.first, wDims.second});
      } else {
        tmp.reset(ElemKind::Int8QTy, {wDims.first, wDims.second},
                  w->getType().getScale(), w->getType().getOffset());
      }
      tmp.copyRawFrom(w);
      w = &tmp;
    }

    Tensor wtag;
    if (typeName == "FC" || typeName == "Int8FC") {
      w->transpose(&wtag, {1, 0});
    } else {
      wtag.assign(w);
    }

    auto W =
        G_.getParent()->addConstant(new Constant("weights", std::move(wtag)));
    auto B = G_.getParent()->addConstant(new Constant("biases", std::move(*b)));

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
      node = G_.createFullyConnected(opName, in, W, B, outTy);
    } else {
      node = G_.createFullyConnected(opName, in, W, B);
    }

    // Save the outputs:
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  if (typeName == "ChannelShuffle") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));

    size_t group;
    ASSIGN_VALUE_OR_RETURN_ERR(group, loadInt(dict["group"]));
    size_t kernel;
    ASSIGN_VALUE_OR_RETURN_ERR(kernel, loadInt(dict["kernel"]));

    Node *node = G_.createChannelShuffle(opName, in, group, kernel);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  if (typeName == "Squeeze") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    auto dims = getShape(dict["dims"]);
    Node *node = G_.createSqueeze(opName, in, dims);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  if (typeName == "Log") {
    // Load the inputs:
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    // Create the log:
    auto *R = G_.createLog(opName, in);
    RETURN_IF_ERR(addNodeAsOutput(op, R));
    return llvm::Error::success();
  }

  if (typeName == "Logit") {
    // Load the input and (optional) epsilon clamping value:
    NodeValue input;
    ASSIGN_VALUE_OR_RETURN_ERR(input,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    auto epsIt = dict.find("eps");
    // default: 1e-6 (as in Caffe2)
    float eps = 1E-6f;
    if (epsIt != dict.end()) {
      ASSIGN_VALUE_OR_RETURN_ERR(eps, loadFloat(epsIt->second));
    }

    auto *node = G_.createLogit(opName, input, eps);
    // Save the outputs:
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  if (typeName == "EQ") {
    NodeValue in0;
    ASSIGN_VALUE_OR_RETURN_ERR(in0,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    NodeValue in1;
    ASSIGN_VALUE_OR_RETURN_ERR(in1,
                               getNodeValueOrCreateConstantByName(op.input(1)));
    auto *node = G_.createCmpEQ(opName, in0, in1);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  if (typeName == "Tile") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    unsigned_t tiles;
    ASSIGN_VALUE_OR_RETURN_ERR(tiles, loadInt(dict["tiles"]));
    unsigned_t axis;
    ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict["axis"]));

    auto *node = G_.createTile(opName, in, tiles, axis);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  if (typeName == "Free") {
    // Glow frees memory automatically.
    return llvm::Error::success();
  }
  if (typeName == "StopGradient") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    // Currently Caffe2 importer only supports inference.
    RETURN_IF_ERR(addNodeAsOutput(op, in));
    return llvm::Error::success();
  }

  if (typeName == "Transpose") {
    RETURN_IF_ERR(loadTranspose(op, dict, "axes"));
    return llvm::Error::success();
  }

  if (typeName == "NCHW2NHWC") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    auto *node = G_.createTranspose(opName, in, NCHW2NHWC);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  if (typeName == "CopyCPUToMKL" || typeName == "CopyMKLToCPU" ||
      typeName == "Copy" || typeName == "EnsureCPUOutput" ||
      typeName == "EnsureDense") {
    // Glow does not support any of these ops now, so implement them as no-ops.
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    RETURN_IF_ERR(addNodeAsOutput(op, in));
    return llvm::Error::success();
  }

  if (typeName == "Slice") {
    NodeValue data;
    ASSIGN_VALUE_OR_RETURN_ERR(data,
                               getNodeValueOrCreateConstantByName(op.input(0)));

    auto starts = getShape<ssize_t>(dict["starts"]);
    auto ends = getShape<ssize_t>(dict["ends"]);

    std::vector<size_t> newStarts, newEnds;
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
    return llvm::Error::success();
  }

  if (typeName == "MatMul") {
    RETURN_IF_ERR(loadBatchMatMul(op, dict, false));
    return llvm::Error::success();
  }

  if (typeName == "Cast") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
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
    return llvm::Error::success();
  }

  if (typeName == "ScatterAssign") {
    NodeValue data;
    ASSIGN_VALUE_OR_RETURN_ERR(data,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    NodeValue indices;
    ASSIGN_VALUE_OR_RETURN_ERR(indices,
                               getNodeValueOrCreateConstantByName(op.input(1)));
    NodeValue slices;
    ASSIGN_VALUE_OR_RETURN_ERR(slices,
                               getNodeValueOrCreateConstantByName(op.input(2)));

    Node *SAN = G_.createScatterAssign(opName, data, indices, slices);
    RETURN_IF_ERR(addNodeAsOutput(op, SAN));
    return llvm::Error::success();
  }

  if (typeName == "ConstantFill" || typeName == "GivenTensorIntFill" ||
      typeName == "GivenTensorInt64Fill") {
    RETURN_IF_ERR(loadWeight(op));
    return llvm::Error::success();
  }

  if (typeName == "SigmoidCrossEntropyWithLogits") {
    NodeValue logits;
    ASSIGN_VALUE_OR_RETURN_ERR(logits,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    NodeValue targets;
    ASSIGN_VALUE_OR_RETURN_ERR(targets,
                               getNodeValueOrCreateConstantByName(op.input(1)));
    Node *SCEL =
        G_.createSigmoidCrossEntropyWithLogits(opName, logits, targets);
    RETURN_IF_ERR(addNodeAsOutput(op, SCEL));
    return llvm::Error::success();
  }

  if (typeName == "ElementwiseLinear") {
    NodeValue X, w, b;

    // If the axis argument does not exist in the protobuf, the default
    // value should be 1.
    unsigned axis = 1;

    ASSIGN_VALUE_OR_RETURN_ERR(X,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    ASSIGN_VALUE_OR_RETURN_ERR(w,
                               getNodeValueOrCreateConstantByName(op.input(1)));
    ASSIGN_VALUE_OR_RETURN_ERR(b,
                               getNodeValueOrCreateConstantByName(op.input(2)));

    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict["axis"]));
    }

    Node *EL = G_.createElementwiseLinear(opName, X, w, b, axis);
    RETURN_IF_ERR(addNodeAsOutput(op, EL));
    return llvm::Error::success();
  }

  if (typeName == "AveragedLoss") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    auto *node = G_.createBatchedReduceMean(opName, in, 0);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  if (typeName == "Mod") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
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

    return llvm::Error::success();
  }

  if (typeName == "SparseLengthsWeightedSum8BitsRowwise" ||
      typeName == "SparseLengthsSum8BitsRowwise" ||
      typeName == "SparseLengthsWeightedSumFused8BitRowwise" ||
      typeName == "SparseLengthsSumFused8BitRowwise") {
    const bool isWeighted =
        typeName == "SparseLengthsWeightedSum8BitsRowwise" ||
        typeName == "SparseLengthsWeightedSumFused8BitRowwise";
    const bool isFused =
        typeName == "SparseLengthsWeightedSumFused8BitRowwise" ||
        typeName == "SparseLengthsSumFused8BitRowwise";
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
    ASSIGN_VALUE_OR_RETURN_ERR(data,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    NodeValue weights;
    if (isWeighted) {
      ASSIGN_VALUE_OR_RETURN_ERR(
          weights, getNodeValueOrCreateConstantByName(op.input(1)));
    }
    NodeValue indices;
    ASSIGN_VALUE_OR_RETURN_ERR(
        indices, getNodeValueOrCreateConstantByName(op.input(indicesIdx)));
    NodeValue lengths;
    ASSIGN_VALUE_OR_RETURN_ERR(
        lengths, getNodeValueOrCreateConstantByName(op.input(lengthsIdx)));
    Constant *dataC = llvm::dyn_cast<Constant>(data);

    const size_t numRows = data.dims()[0];

    // Make sure all the shapes make sense.
    RETURN_ERR_IF_NOT(lengths.dims().size() == 1, "lengths must be a vector.");
    RETURN_ERR_IF_NOT(indices.dims().size() == 1, "indices must be a vector.");

    Node *node;
    if (isFused) {
      // There is no specific fused quantized type in Caffe2, so we will load
      // Int8QTy. We then change it from Int8QTy to UInt8FusedQTy here if
      // necessary -- another user could have already changed it.
      if (dataC->getElementType() != ElemKind::UInt8FusedQTy) {
        RETURN_ERR_IF_NOT(dataC->getElementType() == ElemKind::Int8QTy,
                          "Data must be Int8QTy.");
        // Use dummy 0.0/0 as scale/offset, since the actual scales/offsets are
        // fused inline with the data.
        TypeRef fusedTy = G_.getParent()->uniqueType(ElemKind::UInt8FusedQTy,
                                                     dataC->dims(), 0.0, 0);
        dataC->setType(Storage::OutputIdx, fusedTy);
        dataC->getPayload().setType(fusedTy);

        // We also need to update the data to be unsigned instead of signed.
        auto dataCH = dataC->getHandle<uint8_t>();
        for (size_t i = 0, e = dataCH.size(); i < e; i++) {
          dataCH.raw(i) = dataCH.raw(i) + OFFSETSHIFT;
        }
      }

      // No other work to do, since the data is already loaded fused, so just
      // create the new node with its inputs.
      if (isWeighted) {
        node = G_.createFusedRowwiseQuantizedSparseLengthsWeightedSum(
            opName, dataC, weights, indices, lengths);
      } else {
        node = G_.createFusedRowwiseQuantizedSparseLengthsSum(opName, dataC,
                                                              indices, lengths);
      }
    } else {
      NodeValue scalesBiases;
      ASSIGN_VALUE_OR_RETURN_ERR(
          scalesBiases,
          getNodeValueOrCreateConstantByName(op.input(scalesBiasesIdx)));

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
      for (size_t i = 0, e = numRows; i < e; i++) {
        dataScalesH.at({i}) = scalesBiasesH.at({i, 0});
        dataOffsetsH.at({i}) = scalesBiasesH.at({i, 1});
      }

      // Now create the actual node.
      if (isWeighted) {
        node = G_.createRowwiseQuantizedSparseLengthsWeightedSum(
            opName, dataC, dataScales, dataOffsets, weights, indices, lengths);
      } else {
        node = G_.createRowwiseQuantizedSparseLengthsSum(
            opName, dataC, dataScales, dataOffsets, indices, lengths);
      }
    }

    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  RETURN_ERR(unexpectedNodeErrorMessage(op, "Unsupported operator."));
}

llvm::Error Caffe2ModelLoader::loadInputs(const caffe2::NetDef &net,
                                          bool loadInputsAsPlaceholders) {
  const caffe2::Argument *arg = nullptr;
  for (auto i = 0, e = net.arg_size(); i < e; ++i) {
    if (net.arg(i).name() == "input_shape_info") {
      arg = &net.arg(i);
      break;
    }
  }
  if (arg) {
    for (const auto &in : arg->tensors()) {
      // Skip static weights
      if (tensors_.count(in.name())) {
        continue;
      }

      if (loadInputsAsPlaceholders) {
        Tensor T;
        RETURN_IF_ERR(setTensorType(in, &T));

        Placeholder *placeholder;
        ASSIGN_VALUE_OR_RETURN_ERR(
            placeholder, createAndRegisterPlaceholder(in.name(), &T.getType()));
        nameToInputVars_.try_emplace(in.name(), placeholder);
      } else {
        std::unique_ptr<Tensor> T(new Tensor());
        RETURN_IF_ERR(setTensorType(in, T.get()));
        tensors_[in.name()] = std::move(T);
      }
    }
  }
  return llvm::Error::success();
}

llvm::Error Caffe2ModelLoader::loadNetwork(caffe2::NetDef &net) {
  /// Load the network operators:
  for (int i = 0; i < net.op_size(); i++) {
    auto &op = net.op(i);
    RETURN_IF_ERR(loadOperator(op));
  }

  RETURN_ERR_IF_NOT(net.external_output_size(),
                    "Network needs external outputs defined.");

  for (int i = 0; i < net.external_output_size(); i++) {
    auto &outputName = net.external_output(i);
    NodeValue r;
    ASSIGN_VALUE_OR_RETURN_ERR(r, getNodeValueByName(outputName));
    auto *SN = G_.createSave("save_" + outputName, r);
    outputVarsByName_[outputName] = SN->getPlaceholder();
  }
  return llvm::Error::success();
}

/// Fills \p T with data from \p values.
template <typename ElemTy, typename RangeTy>
static llvm::Error fillTensor(Tensor &T, ElemKind kind,
                              llvm::ArrayRef<size_t> dim, RangeTy values) {
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
  return llvm::Error::success();
}

llvm::Error Caffe2ModelLoader::loadWeight(const caffe2::OperatorDef &op) {
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
    std::unique_ptr<Tensor> T(new Tensor());
    if (typeName == "GivenTensorFill") {
      RETURN_IF_ERR(
          fillTensor<float>(*T, ElemKind::FloatTy, dim, values->floats()));
    } else if (typeName == "GivenTensorIntFill") {
      RETURN_IF_ERR(
          fillTensor<int32_t>(*T, ElemKind::Int32ITy, dim, values->ints()));
    } else if (typeName == "GivenTensorInt64Fill") {
      RETURN_IF_ERR(
          fillTensor<int64_t>(*T, ElemKind::Int64ITy, dim, values->ints()));
    } else {
      GLOW_UNREACHABLE("Unhandled GivenTensorFill type");
    }
    tensors_[op.output().Get(0)] = std::move(T);
    return llvm::Error::success();
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
      std::unique_ptr<Tensor> T(new Tensor());
      if (tensors_.count(o)) {
        continue;
      }
      auto dim = getShape(dict["shape"]);

      T->reset(ElemKind::Int8QTy, dim, 0.0, 0);
      auto TH = T->getHandle<int8_t>();
      assert(dict["values"]->strings().size() == 1 &&
             "Expect single string input for GivenTensorByteStringToUInt8Fill");
      const std::string str = dict["values"]->strings().Get(0);

      // We're loading unsigned data into Int8QTy, so we use OFFSETSHIFT to
      // convert to signed.
      size_t i;
      for (i = 0; i < str.size(); i++) {
        TH.raw(i) = (uint8_t)str.c_str()[i] - OFFSETSHIFT;
      }
      RETURN_ERR_IF_NOT(i == T->size(),
                        "The number of serialized values does not "
                        "match the size of the tensor.");
      tensors_[o] = std::move(T);
    }
    return llvm::Error::success();
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
      std::unique_ptr<Tensor> T(new Tensor());
      if (tensors_.count(o)) {
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
        // the type of input and weights must be the same. Therefore, we need to
        // convert it to int8 by subtracting 128.
        T->reset(ElemKind::Int8QTy, dim, scale, offset - OFFSETSHIFT);
        auto TH = T->getHandle<int8_t>();
        std::string str = dict["values"]->s();
        for (; i < str.size(); i++) {
          TH.raw(i) = ((uint8_t)(str.c_str()[i]) - OFFSETSHIFT);
        }
      } else {
        T->reset(ElemKind::Int32QTy, dim, scale, offset);
        auto TH = T->getHandle<int32_t>();
        for (auto num : dict["values"]->ints()) {
          TH.raw(i++) = num;
        }
      }
      RETURN_ERR_IF_NOT(i == T->size(),
                        "The number of serialized values does not "
                        "match the size of the tensor.");

      tensors_[o] = std::move(T);
    }

    return llvm::Error::success();
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
    if (tensors_.count(name)) {
      return llvm::Error::success();
    }

    std::unique_ptr<Tensor> T(new Tensor());

    // The shape is set either the shape argument, or from another input
    // tensor. Shape takes priority over input.
    std::vector<size_t> dims;
    if (dict.count("shape")) {
      dims = getShape(dict["shape"]);
    } else {
      RETURN_ERR_IF_NOT(op.input_size() > 0,
                        "If no shape provided, must have input shape.");
      // It must be registered as a tensor because it must be statically set
      // already, as shapes must be statically known.
      Tensor *in;
      ASSIGN_VALUE_OR_RETURN_ERR(in, getTensorByName(op.input(0)));
      dims = in->dims();
    }

    int to = caffe2::TensorProto_DataType_FLOAT;
    if (dict.count("dtype")) {
      ASSIGN_VALUE_OR_RETURN_ERR(to, loadInt(dict["dtype"]));
    }

    switch (to) {
    case caffe2::TensorProto_DataType_FLOAT: {
      T->reset(ElemKind::FloatTy, dims);
      auto TH = T->getHandle<float>();
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
      T->reset(ElemKind::Int64ITy, dims);
      auto TH = T->getHandle<int64_t>();
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

    tensors_[name] = std::move(T);
    return llvm::Error::success();
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
    std::unique_ptr<Tensor> T(new Tensor());
    auto dim = getShape(dict["shape"]);
    T->reset(ElemKind::FloatTy, dim);
    auto TH = T->getHandle<>();
    float tensorMin;
    ASSIGN_VALUE_OR_RETURN_ERR(tensorMin, loadFloat(dict["min"]));
    float tensorMax;
    ASSIGN_VALUE_OR_RETURN_ERR(tensorMax, loadFloat(dict["max"]));

#ifndef NDEBUG
    llvm::outs() << "The model contains UniformFill operator, which generates"
                 << " random numbers. This could be source of discrepancy.\n";
#endif // NDEBUG
    // Uniformly generate random numbers in [tensorMin; tensorMax).
    for (auto &elem : TH) {
      elem = G_.getParent()->getPRNG().nextRandReal(tensorMin, tensorMax);
    }

    tensors_[name] = std::move(T);
    return llvm::Error::success();
  }

  RETURN_ERR(unexpectedNodeErrorMessage(op, "Unsupported weight kind"));
}

llvm::Error Caffe2ModelLoader::loadWeightsFromNet(caffe2::NetDef &net) {
  for (auto &op : net.op()) {
    RETURN_IF_ERR(loadWeight(op));
  }
  return llvm::Error::success();
}

Caffe2ModelLoader::Caffe2ModelLoader(Function &F, llvm::Error *errPtr)
    : CommonOperatorLoader({}, {}, F, errPtr) {}

Caffe2ModelLoader::Caffe2ModelLoader(const std::string &netDescFilename,
                                     const std::string &netWeightFilename,
                                     llvm::ArrayRef<const char *> names,
                                     llvm::ArrayRef<TypeRef> types, Function &F,
                                     llvm::Error *errPtr)
    : CommonOperatorLoader(names, types, F, errPtr) {
  // if errPtr already contains an error then don't continue with constructor
  if (errPtr && *errPtr) {
    return;
  }

  // Lambda to setup the Caffe2ModelLoader and return any llvm::Errors that were
  // raised.
  auto setup = [&]() -> llvm::Error {
    // The caffe2 network descriptor that we are deserializing.
    caffe2::NetDef networkDef;
    ASSIGN_VALUE_OR_RETURN_ERR(networkDef, loadProtoFile(netDescFilename));

    // The caffe2 weights that we are deserializing.
    caffe2::NetDef weightsDef;
    ASSIGN_VALUE_OR_RETURN_ERR(weightsDef, loadProtoFile(netWeightFilename));

    RETURN_IF_ERR(loadWeightsFromNet(weightsDef));
    RETURN_IF_ERR(loadNetwork(networkDef));

    RETURN_ERR_IF_NOT(F.verify(), "Function verification failed.");

    return llvm::Error::success();
  };

  if (errPtr) {
    *errPtr = setup();
  } else {
    EXIT_ON_ERR(setup());
  }
}
