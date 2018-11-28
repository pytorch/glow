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

#include "llvm/Support/Casting.h"

#include "caffe2/proto/caffe2.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <cassert>
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

/// Translates the protocol buffer node \p op into a random access map.
static ArgumentDictionaryTy loadArgumentMap(const caffe2::OperatorDef &op) {
  ArgumentDictionaryTy dict;
  for (auto i = 0, e = op.arg_size(); i < e; i++) {
    const caffe2::Argument &arg = op.arg(i);
    dict[arg.name()] = &arg;
  }
  return dict;
}

static std::vector<unsigned_t> getPads(const ArgumentDictionaryTy &dict) {
  if (dict.count("pad")) {
    int pad = loadInt(dict.at("pad"));
    std::vector<unsigned_t> pads(4, pad);
    return pads;
  }
  if (dict.count("pad_t")) {
    std::vector<unsigned_t> pads(4);
    pads[0] = loadInt(dict.at("pad_t"));
    assert(dict.count("pad_l") && "missing pad_l");
    pads[1] = loadInt(dict.at("pad_l"));
    assert(dict.count("pad_b") && "missing pad_b");
    pads[2] = loadInt(dict.at("pad_b"));
    assert(dict.count("pad_r") && "missing pad_r");
    pads[3] = loadInt(dict.at("pad_r"));
    return pads;
  }
  if (dict.count("pads")) {
    return getShape<unsigned_t>(dict.at("pads"));
  }
  // Return default value 0 for pads.
  return {0, 0, 0, 0};
}

/// Translates the "order" field of dictionary \p dict into a channel number.
static unsigned_t getChannel(const ArgumentDictionaryTy &dict) {
  std::string order = "NCHW"; // default
  auto orderIt = dict.find("order");
  if (orderIt != dict.end()) {
    order = loadStr(orderIt->second);
  }
  if (order == "NHWC") {
    return 3;
  } else if (order == "NCHW") {
    return 1;
  }
  GLOW_ASSERT(false && "Invalid order field");
}

static std::vector<unsigned_t> getSizeHW(ArgumentDictionaryTy &dict,
                                         const std::string &name,
                                         unsigned_t defaultValue) {
  if (dict.count(name)) {
    int value = loadInt(dict[name]);
    std::vector<unsigned_t> result(2, value);
    return result;
  }
  if (dict.count(name + "_h") && dict.count(name + "_w")) {
    std::vector<unsigned_t> result(2);
    result[0] = loadInt(dict[name + "_h"]);
    result[1] = loadInt(dict[name + "_w"]);
    return result;
  }
  if (dict.count(name + "s")) {
    return getShape<unsigned_t>(dict.at(name + "s"));
  }
  return {defaultValue, defaultValue};
}

bool Caffe2ModelLoader::loadProtoFile(caffe2::NetDef &net,
                                      const std::string &filename) {
  std::ifstream ff(filename, std::ios::in | std::ios::binary);
  GLOW_ASSERT(ff && "Can't find the model or network files.");

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

  GLOW_ASSERT(parseNet && "Failed to parse the network descriptor.");
  return true;
}

bool Caffe2ModelLoader::getBroadcast(const ArgumentDictionaryTy &dict) {
  return dict.count("broadcast") && (loadInt(dict.at("broadcast")) == 1);
}

void Caffe2ModelLoader::loadOperator(const caffe2::OperatorDef &op) {
  ArgumentDictionaryTy dict = loadArgumentMap(op);
  const std::string &typeName = op.type();

  // Check if operator is supported in parent class, CommonOperatorLoader.
  if (tryLoadCommonOperator(typeName, op, dict)) {
    // If operator is supported, CommonOperatorLoader loaded it to the Graph.
    return;
  }

  const std::string &opName = loadOperatorName(op);

  if (typeName == "Conv" || typeName == "Int8Conv" ||
      typeName == "Int8ConvRelu") {
    // Load the inputs:
    std::vector<unsigned_t> strides = getSizeHW(dict, "stride", 1);
    std::vector<unsigned_t> pads = getPads(dict);
    std::vector<unsigned_t> kernels = getSizeHW(dict, "kernel", 0);
    unsigned_t group = dict.count("group") ? loadInt(dict["group"]) : 1;
    std::string order = dict.count("order") ? loadStr(dict["order"]) : "NCHW";

    auto in = getNodeValueOrCreateConstantByName(op.input(0));
    Tensor *w = getTensorByName(op.input(1));

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
    Node *tr;
    if (order == "NCHW") {
      tr = G_.createTranspose(opName, in, NCHW2NHWC);
    } else {
      tr = in;
    }

    // Calculate the size and allocate the output buffer.
    ShapeNHWC idim = ShapeNHWC(tr->getType(0)->dims());
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
          Tensor *b = getTensorByName(biasTensorName);
          biasTensor.assign(b);
        }
      }
      outTy = G_.getParent()->uniqueType(ElemKind::FloatTy, outDims);
      filter = G_.getParent()->createConstant("conv.filter", wtag);
      bias = G_.getParent()->createConstant("conv.bias", biasTensor);
    } else {
      assert(dict.count("Y_zero_point") &&
             "missing zero point for quantized output type");
      assert(dict.count("Y_scale") &&
             "missing Y_scale for quantized output type");
      // Construct the Bias field.
      Tensor biasTensor(ElemKind::Int32QTy, {depth}, 1.0, 0);
      biasTensor.zero();
      // Check if we have a serialized bias vector.
      if (op.input_size() > 2) {
        const auto &biasTensorName = op.input(2);
        if (tensors_.count(biasTensorName)) {
          // Load the serialized bias vector.
          Tensor *b = getTensorByName(biasTensorName);
          biasTensor.assign(b);
        }
      }
      float scale = loadFloat(dict["Y_scale"]);
      int32_t offset = loadInt(dict["Y_zero_point"]);
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

    Node *node = G_.createConv(opName, tr, filter, bias, outTy, kernels,
                               strides, pads, group);

    if (order == "NCHW") {
      // Transpose the output back.
      node = G_.createTranspose(opName, node, NHWC2NCHW);
    }
    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "Int8SumRelu") {
    assert(op.input_size() == 2 && "Only Sum of 2 inputs is supported.");
    assert(dict.count("Y_zero_point") &&
           "missing zero point for quantized outout type");
    assert(dict.count("Y_scale") &&
           "missing Y_scale for quantized output type");
    auto in0 = getNodeValueOrCreateConstantByName(op.input(0));
    auto in1 = getNodeValueOrCreateConstantByName(op.input(1));
    auto outDims = in0.getType()->dims();
    auto outTy = G_.getParent()->uniqueType(
        ElemKind::Int8QTy, outDims, loadFloat(dict["Y_scale"]),
        loadInt(dict["Y_zero_point"]) - OFFSETSHIFT);
    auto *node = G_.createAdd(opName, outTy, in0, in1);
    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "Int8Quantize") {
    assert(dict.count("Y_zero_point") &&
           "missing zero point for quantized output type");
    assert(dict.count("Y_scale") &&
           "missing Y_scale for quantized output type");
    auto in = getNodeValueOrCreateConstantByName(op.input(0));
    auto outDims = in.getType()->dims();
    auto outTy = G_.getParent()->uniqueType(
        ElemKind::Int8QTy, outDims, loadFloat(dict["Y_scale"]),
        loadInt(dict["Y_zero_point"]) - OFFSETSHIFT);
    Node *N = G_.createQuantize(opName, in, outTy);
    addNodeAsOutput(op, N);
    return;
  }

  if (typeName == "Int8Dequantize") {
    auto in = getNodeValueOrCreateConstantByName(op.input(0));
    auto *node = G_.createDequantize(opName, in);
    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "MaxPool" || typeName == "AveragePool" ||
      typeName == "Int8MaxPool" || typeName == "Int8AveragePool") {
    // Load the inputs:
    auto in = getNodeValueOrCreateConstantByName(op.input(0));
    std::vector<unsigned_t> strides = getSizeHW(dict, "stride", 1);
    std::vector<unsigned_t> kernels = getSizeHW(dict, "kernel", 0);
    std::vector<unsigned_t> pads = getPads(dict);
    std::string order = dict.count("order") ? loadStr(dict["order"]) : "NCHW";
    // We expect the input to be NHWC.
    Node *tr;
    if (order == "NCHW") {
      tr = G_.createTranspose(opName, in, NCHW2NHWC);
    } else {
      tr = in;
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
      assert(dict.count("Y_zero_point") &&
             "missing zero point for quantized output type");
      assert(dict.count("Y_scale") &&
             "missing Y_scale for quantized output type");
      ShapeNHWC idim = ShapeNHWC(tr->getType(0)->dims());
      auto outSz =
          calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides, pads);
      std::array<size_t, 4> outDims = {
          {idim.n, outSz.first, outSz.second, idim.c}};
      if (typeName == "Int8MaxPool") {
        // Int8Maxpool output quantization should be same as the input, so just
        // ignore the given params.
        node = G_.createMaxPool(opName, tr, kernels, strides, pads);
      } else {
        auto outTy = G_.getParent()->uniqueType(
            ElemKind::Int8QTy, outDims, loadFloat(dict["Y_scale"]),
            loadInt(dict["Y_zero_point"]) - OFFSETSHIFT);
        node = G_.createAvgPool(opName, tr, outTy, kernels, strides, pads);
      }
    } else if (typeName == "MaxPool") {
      node = G_.createMaxPool(opName, tr, kernels, strides, pads);
    } else {
      node = G_.createAvgPool(opName, tr, kernels, strides, pads);
    }
    if (order == "NCHW") {
      // Transpose the output back.
      node = G_.createTranspose(opName, node, NHWC2NCHW);
    }
    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "SpatialBN") {
    auto in = getNodeValueOrCreateConstantByName(op.input(0));
    auto *scale = getTensorByName(op.input(1));
    auto *bias = getTensorByName(op.input(2));
    auto *mean = getTensorByName(op.input(3));
    auto *var = getTensorByName(op.input(4));
    float epsilon = 1e-5f; // default
    auto epsilonIt = dict.find("epsilon");
    if (epsilonIt != dict.end()) {
      epsilon = loadFloat(epsilonIt->second);
    }

    auto channel = getChannel(dict);
    auto *scaleV = G_.getParent()->createConstant("scale", *scale);
    auto *biasV = G_.getParent()->createConstant("bias", *bias);
    auto *meanV = G_.getParent()->createConstant("mean", *mean);
    auto *varV = G_.getParent()->createConstant("var", *var);
    auto *node = G_.createBatchNormalization(opName, in, biasV, scaleV, meanV,
                                             varV, channel, epsilon);

    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "Concat") {
    const unsigned numInputs = op.input_size();
    llvm::SmallVector<NodeValue, 4> inputs;
    inputs.reserve(numInputs);
    for (unsigned i = 0; i < numInputs; i++) {
      inputs.push_back(getNodeValueOrCreateConstantByName(op.input(i)));
    }

    // If axis exists it takes priority over channel.
    unsigned_t channel =
        dict.count("axis") ? loadInt(dict["axis"]) : getChannel(dict);

    Node *node = G_.createConcat(opName, inputs, channel);

    unsigned_t addAxis = dict.count("add_axis") ? loadInt(dict["add_axis"]) : 0;

    if (addAxis) {
      // When add axis is used, this means we have to add a new dimension before
      // the axis, instead of merging on the axis.
      std::vector<size_t> outputDims = inputs[0].dims();
      for (const auto &input : inputs) {
        GLOW_ASSERT(
            outputDims[channel] == input.dims()[channel] &&
            "inputs need all to have the same dims for concat with add_axis");
      }
      outputDims.insert(outputDims.begin() + channel, numInputs);
      node = G_.createReshape(opName, node, outputDims);
    }
    // Concat has multiple outputs in Caffe2, but I believe the other output
    // (split_info) is not used for inference.
    nodeValueByName_[op.output(0)] = NodeValue(node, 0);
    return;
  }

  if (typeName == "FC" || typeName == "FCTransposed" || typeName == "Int8FC") {
    // Load the inputs:
    auto in = getNodeValueOrCreateConstantByName(op.input(0));
    if (in.getType()->dims().size() > 2) {
      size_t axis = dict.count("axis") ? loadInt(dict["axis"]) : 1;
      in = G_.createFlatten("fc.in", in, axis);
    }

    // Load weights.
    Tensor *w = getTensorByName(op.input(1));
    Tensor *b = getTensorByName(op.input(2));
    unsigned_t axis_w = dict.count("axis_w") ? loadInt(dict["axis_w"]) : 1;

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
      assert(dict.count("Y_zero_point") &&
             "missing zero point for quantized output type");
      assert(dict.count("Y_scale") &&
             "missing Y_scale for quantized output type");
      auto outTy = G_.getParent()->uniqueType(
          ElemKind::Int8QTy, {in.getType()->dims()[0], B->getType()->dims()[0]},
          loadFloat(dict["Y_scale"]),
          loadInt(dict["Y_zero_point"]) - OFFSETSHIFT);
      node = G_.createFullyConnected(opName, in, W, B, outTy);
    } else {
      node = G_.createFullyConnected(opName, in, W, B);
    }

    // Save the outputs:
    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "ChannelShuffle") {
    auto in = getNodeValueOrCreateConstantByName(op.input(0));

    size_t group = loadInt(dict["group"]);
    size_t kernel = loadInt(dict["kernel"]);

    Node *node = G_.createChannelShuffle(opName, in, group, kernel);
    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "Squeeze") {
    auto in = getNodeValueOrCreateConstantByName(op.input(0));
    auto dims = getShape(dict["dims"]);
    Node *node = G_.createSqueeze(opName, in, dims);
    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "Log") {
    // Load the inputs:
    auto in = getNodeValueOrCreateConstantByName(op.input(0));
    // Create the log:
    auto *R = G_.createLog(opName, in);
    addNodeAsOutput(op, R);
    return;
  }

  if (typeName == "Logit") {
    // Load the input and (optional) epsilon clamping value:
    auto input = getNodeValueOrCreateConstantByName(op.input(0));
    auto epsIt = dict.find("eps");
    // default: 1e-6 (as in Caffe2)
    auto eps = epsIt != dict.end() ? loadFloat(epsIt->second) : 1E-6f;
    auto *node = G_.createLogit(opName, input, eps);
    // Save the outputs:
    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "EQ") {
    auto in0 = getNodeValueOrCreateConstantByName(op.input(0));
    auto in1 = getNodeValueOrCreateConstantByName(op.input(1));
    auto *node = G_.createCmpEQ(opName, in0, in1);
    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "Tile") {
    auto in = getNodeValueOrCreateConstantByName(op.input(0));
    unsigned_t tiles = loadInt(dict["tiles"]);
    unsigned_t axis = loadInt(dict["axis"]);

    auto *node = G_.createTile(opName, in, tiles, axis);
    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "Free") {
    // Glow frees memory automatically.
    return;
  }
  if (typeName == "StopGradient") {
    auto in = getNodeValueOrCreateConstantByName(op.input(0));
    // Currently Caffe2 importer only supports inference.
    addNodeAsOutput(op, in);
    return;
  }

  if (typeName == "Transpose") {
    return loadTranspose(op, dict, "axes");
  }

  if (typeName == "NCHW2NHWC") {
    auto in = getNodeValueOrCreateConstantByName(op.input(0));
    auto *node = G_.createTranspose(opName, in, NCHW2NHWC);
    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "CopyCPUToMKL" || typeName == "CopyMKLToCPU" ||
      typeName == "Copy" || typeName == "EnsureCPUOutput" ||
      typeName == "EnsureDense") {
    // Glow does not support any of these ops now, so implement them as no-ops.
    auto in = getNodeValueOrCreateConstantByName(op.input(0));
    addNodeAsOutput(op, in);
    return;
  }

  if (typeName == "Slice") {
    auto data = getNodeValueOrCreateConstantByName(op.input(0));

    auto starts = getShape<ssize_t>(dict["starts"]);
    auto ends = getShape<ssize_t>(dict["ends"]);

    std::vector<size_t> newStarts, newEnds;
    assert(starts.size() == ends.size());
    for (size_t i = 0; i < starts.size(); i++) {
      ssize_t newStart = starts[i];
      if (newStart == -1) {
        newStart = data.dims()[i];
      }
      assert(newStart >= 0 && "Indices should never be negative.");
      newStarts.push_back(newStart);

      ssize_t newEnd = ends[i];
      if (newEnd == -1) {
        newEnd = data.dims()[i];
      }
      assert(newEnd >= 0 && "Indices should never be negative.");
      newEnds.push_back(newEnd);
    }

    Node *SN = G_.createSlice(opName, data, newStarts, newEnds);
    addNodeAsOutput(op, SN);
    return;
  }

  if (typeName == "MatMul") {
    loadBatchMatMul(op, dict, false);
    return;
  }

  if (typeName == "Cast") {
    auto in = getNodeValueOrCreateConstantByName(op.input(0));
    int to = loadInt(dict["to"]);

    switch (to) {
    case caffe2::TensorProto_DataType_FLOAT: {
      assert(in.getElementType() == ElemKind::FloatTy &&
             "Can only cast float to float.");
      break;
    }
    case caffe2::TensorProto_DataType_INT32:
    case caffe2::TensorProto_DataType_INT64: {
      assert(in.getElementType() == ElemKind::Int64ITy &&
             "Can only cast int to int.");
      break;
    }
    default:
      llvm_unreachable("Unsupported Cast type.");
    }

    addNodeAsOutput(op, in);
    return;
  }

  if (typeName == "ScatterAssign") {
    auto data = getNodeValueOrCreateConstantByName(op.input(0));
    auto indices = getNodeValueOrCreateConstantByName(op.input(1));
    auto slices = getNodeValueOrCreateConstantByName(op.input(2));

    Node *SAN = G_.createScatterAssign(opName, data, indices, slices);
    addNodeAsOutput(op, SAN);
    return;
  }

  if (typeName == "ConstantFill" || typeName == "GivenTensorIntFill" ||
      typeName == "GivenTensorInt64Fill") {
    loadWeight(op);
    return;
  }

  if (typeName == "SigmoidCrossEntropyWithLogits") {
    auto logits = getNodeValueOrCreateConstantByName(op.input(0));
    auto targets = getNodeValueOrCreateConstantByName(op.input(1));
    Node *SCEL =
        G_.createSigmoidCrossEntropyWithLogits(opName, logits, targets);
    addNodeAsOutput(op, SCEL);
    return;
  }

  if (typeName == "AveragedLoss") {
    auto in = getNodeValueOrCreateConstantByName(op.input(0));
    auto *node = G_.createBatchedReduceMean(opName, in, 0);
    addNodeAsOutput(op, node);
    return;
  }

  unexpectedNodeError(op, "Unsupported operator.");
}

void Caffe2ModelLoader::loadNetwork(caffe2::NetDef &net) {
  /// Load the network operators:
  for (int i = 0; i < net.op_size(); i++) {
    auto &op = net.op(i);
    loadOperator(op);
  }

  assert(net.external_output_size() &&
         "Network needs external outputs defined.");

  for (int i = 0; i < net.external_output_size(); i++) {
    auto &outputName = net.external_output(i);
    auto r = getNodeValueByName(outputName);
    auto *SN = G_.createSave("save_" + outputName, r);
    outputVarsByName_[outputName] = SN->getPlaceholder();
  }
}

void Caffe2ModelLoader::loadWeight(const caffe2::OperatorDef &op) {
  ArgumentDictionaryTy dict = loadArgumentMap(op);
  const std::string &typeName = op.type();

  /// Load tensors with values:
  if (typeName == "GivenTensorFill" || typeName == "GivenTensorIntFill" ||
      typeName == "GivenTensorInt64Fill") {
    /*
     output: "conv1_w"
     name: ""
     type: "GivenTensorFill"
     arg {
     name: "shape"
     ints: 96
     ints: 3
     ints: 11
     ints: 11
     }
     arg {
     name: "values"
     floats: -0.028315347
     */

    auto *T = new Tensor();
    for (auto &o : op.output()) {
      tensors_[o] = T;
    }

    auto dim = getShape(dict["shape"]);

    size_t i = 0;
#define LOAD_TENSOR_FILL(TYPE_NAME, NATIVE_TYPE, PROTO_TYPE_NAME)              \
  T->reset(ElemKind::TYPE_NAME, dim);                                          \
  auto TH = T->getHandle<NATIVE_TYPE>();                                       \
  for (auto num : dict["values"]->PROTO_TYPE_NAME()) {                         \
    TH.raw(i++) = num;                                                         \
  }

    if (dict["values"]->floats_size()) {
      assert(typeName != "GivenTensorIntFill" &&
             typeName != "GivenTensorInt64Fill");
      LOAD_TENSOR_FILL(FloatTy, float, floats);
    } else if (dict["values"]->ints_size()) {
      if (typeName == "GivenTensorIntFill") {
        LOAD_TENSOR_FILL(Int32ITy, int32_t, ints);
      } else if (typeName == "GivenTensorInt64Fill" ||
                 typeName == "GivenTensorFill") {
        LOAD_TENSOR_FILL(Int64ITy, int64_t, ints);
      } else {
        unexpectedNodeError(op, "Unsupported data type for " + typeName);
      }
    } else {
      unexpectedNodeError(op, "Unsupported data type for " + typeName);
    }
#undef LOAD_TENSOR_FILL

    assert(i == T->size() && "The number of serialized values does not "
                             "match the size of the tensor.");
    return;
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
    auto *T = new Tensor();
    for (auto &o : op.output()) {
      if (tensors_.count(o))
        continue;
      tensors_[o] = T;
    }

    auto dim = getShape(dict["shape"]);

    assert(dict.count("Y_zero_point") &&
           "missing zero point for quantized output type");
    assert(dict.count("Y_scale") &&
           "missing Y_scale for quantized output type");

    float scale = loadFloat(dict["Y_scale"]);
    int32_t offset = loadInt(dict["Y_zero_point"]);
    size_t i = 0;
    if (typeName == "Int8GivenTensorFill") {
      // Although in Caffe2 quantized model, the weights is int8 quantized,
      // the weights is stored in uint8_t format due to that Caffe2 requires the
      // type of input and weights must be the same. Therefore, we need to
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
    assert(i == T->size() && "The number of serialized values does not "
                             "match the size of the tensor.");
    return;
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
      return;
    }

    auto *T = new Tensor();
    tensors_[name] = T;

    // The shape is set either the shape argument, or from another input
    // tensor. Shape takes priority over input.
    std::vector<size_t> dims;
    if (dict.count("shape")) {
      dims = getShape(dict["shape"]);
    } else {
      assert(op.input_size() > 0 &&
             "If no shape provided, must have input shape.");
      // It must be registered as a tensor because it must be statically set
      // already, as shapes must be statically known.
      auto *in = getTensorByName(op.input(0));
      dims = in->dims();
    }

    int to = dict.count("dtype") ? loadInt(dict["dtype"])
                                 : caffe2::TensorProto_DataType_FLOAT;

    switch (to) {
    case caffe2::TensorProto_DataType_FLOAT: {
      T->reset(ElemKind::FloatTy, dims);
      auto TH = T->getHandle<float>();
      auto f = (dict.count("value") && dict["value"]->has_f())
                   ? loadFloat(dict["value"])
                   : 0.0f;
      TH.clear(f);
      break;
    }
    case caffe2::TensorProto_DataType_INT32:
    case caffe2::TensorProto_DataType_INT64:
    case caffe2::TensorProto_DataType_BOOL: {
      T->reset(ElemKind::Int64ITy, dims);
      auto TH = T->getHandle<int64_t>();
      auto i = (dict.count("value") && dict["value"]->has_i())
                   ? loadInt(dict["value"])
                   : 0;
      TH.clear(i);
      break;
    }
    default:
      llvm_unreachable("Unsupported datatype for ConstantFill.");
    }

    return;
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
    auto *T = new Tensor();
    tensors_[name] = T;
    auto dim = getShape(dict["shape"]);
    T->reset(ElemKind::FloatTy, dim);
    auto TH = T->getHandle<>();
    float tensorMin = loadFloat(dict["min"]);
    float tensorMax = loadFloat(dict["max"]);

#ifndef NDEBUG
    llvm::outs() << "The model contains UniformFill operator, which generates"
                 << " random numbers. This could be source of discrepancy.\n";
#endif // NDEBUG
    // Uniformly generate random numbers in [tensorMin; tensorMax).
    for (size_t i = 0, e = T->size(); i != e; i++) {
      TH.raw(i) = G_.getParent()->getPRNG().nextRandReal(tensorMin, tensorMax);
    }
    return;
  }

  unexpectedNodeError(op, "Unsupported weight kind");
}

void Caffe2ModelLoader::loadWeights(caffe2::NetDef &net) {
  for (auto &op : net.op()) {
    loadWeight(op);
  }
}

Caffe2ModelLoader::Caffe2ModelLoader(const std::string &netDescFilename,
                                     const std::string &netWeightFilename,
                                     llvm::ArrayRef<const char *> names,
                                     llvm::ArrayRef<TypeRef> types, Function &F)
    : CommonOperatorLoader(names, types, F) {
  // The caffe2 weights that we are deserializing.
  caffe2::NetDef weightsDef;
  // The caffe2 network descriptor that we are deserializing.
  caffe2::NetDef networkDef;

  loadProtoFile(networkDef, netDescFilename);
  loadProtoFile(weightsDef, netWeightFilename);
  loadWeights(weightsDef);
  loadNetwork(networkDef);
}
