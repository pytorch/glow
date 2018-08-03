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

#include "glow/Importer/Caffe2.h"
#include "glow/Base/Tensor.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"

#include "llvm/Support/Casting.h"

#include "glow/caffe.pb.h"
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

/// Translates the protocol buffer node \p op into a random access map.
static ArgumentDictionaryTy loadArgumentMap(const caffe2::OperatorDef &op) {
  ArgumentDictionaryTy dict;
  for (auto i = 0, e = op.arg_size(); i < e; i++) {
    const caffe2::Argument &arg = op.arg(i);
    dict[arg.name()] = &arg;
  }
  return dict;
}

static std::vector<size_t> getPads(const ArgumentDictionaryTy &dict) {
  if (dict.count("pad")) {
    int pad = loadInt(dict.at("pad"));
    std::vector<size_t> pads(4, pad);
    return pads;
  }
  if (dict.count("pad_t")) {
    std::vector<size_t> pads(4);
    pads[0] = loadInt(dict.at("pad_t"));
    assert(dict.count("pad_l") && "missing pad_l");
    pads[1] = loadInt(dict.at("pad_l"));
    assert(dict.count("pad_b") && "missing pad_b");
    pads[2] = loadInt(dict.at("pad_b"));
    assert(dict.count("pad_r") && "missing pad_r");
    pads[3] = loadInt(dict.at("pad_r"));
    return pads;
  }
  // Return default value 0 for pads.
  return {0, 0, 0, 0};
}

/// Translates the "order" field of dictionary \p dict into a channel number.
static unsigned getChannel(const ArgumentDictionaryTy &dict) {
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

static std::vector<size_t> getSizeHW(ArgumentDictionaryTy &dict,
                                     const std::string &name,
                                     unsigned defaultValue) {
  if (dict.count(name)) {
    int value = loadInt(dict[name]);
    std::vector<size_t> result(2, value);
    return result;
  }
  if (dict.count(name + "_h") && dict.count(name + "_w")) {
    std::vector<size_t> result(2);
    result[0] = loadInt(dict[name + "_h"]);
    result[1] = loadInt(dict[name + "_w"]);
    return result;
  }
  if (dict.count(name + "s")) {
    return getShape(dict.at(name + "s"));
  }
  return {defaultValue, defaultValue};
}

bool caffe2ModelLoader::loadProtoFile(caffe2::NetDef &net,
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

bool caffe2ModelLoader::getBroadcast(const ArgumentDictionaryTy &dict) {
  return dict.count("broadcast") && (loadInt(dict.at("broadcast")) == 1);
}

void caffe2ModelLoader::loadOperator(const caffe2::OperatorDef &op) {
  ArgumentDictionaryTy dict = loadArgumentMap(op);
  const std::string &typeName = op.type();

  // Check if operator is supported in parent class, CommonOperatorLoader.
  if (tryLoadCommonOperator(typeName, op, dict)) {
    // If operator is supported, CommonOperatorLoader loaded it to the Graph.
    return;
  }

  const std::string &opName = loadOperatorName(op);

  if (typeName == "Conv") {
    // Load the inputs:
    std::vector<size_t> strides = getSizeHW(dict, "stride", 1);
    std::vector<size_t> pads = getPads(dict);
    std::vector<size_t> kernels = getSizeHW(dict, "kernel", 0);
    unsigned group = dict.count("group") ? loadInt(dict["group"]) : 1;

    auto in = getNodeValueOrCreateVariableByName(op.input(0));
    Tensor *w = getTensorByName(op.input(1));

    // Transpose the weights to the right format. Glow expects to read the
    // weights in the format CRSK. Caffe2 stores the operators as KCRS.
    // C - output_depth, R - filter_height, S - filter_width, K - input_depth.
    Tensor wtag;
    w->transpose(&wtag, NCHW2NHWC);

    // The structure of the conv weigts is: NHWC. We take the C, which is the
    // number of filters. We use this value to calculate the size of the bias
    // if it is not specified.
    size_t depth = wtag.dims()[0];

    // Construct the Filter field.
    auto *filter = G_.getParent()->createVariable("conv.filter", wtag);

    // Construct the Bias field.
    Tensor biasTensor(ElemKind::FloatTy, {depth});
    biasTensor.zero();

    // Check if we have a serialized bias vector.
    if (op.input_size() > 2) {
      auto &biasTensorName = op.input(2);
      if (tensors_.count(biasTensorName)) {
        // Load the serialized bias vector.
        Tensor *b = getTensorByName(biasTensorName);
        biasTensor.copyFrom(b);
      }
    }
    auto *bias = G_.getParent()->createVariable("conv.bias", biasTensor);

    // Caffe passes the input as NCHW, and we expect the input to be NHWC.
    auto *tr = G_.createTranspose(opName, in, NCHW2NHWC);

    // Calculate the size and allocate the output buffer.
    ShapeNHWC idim = ShapeNHWC(tr->getResult().dims());
    auto outSz =
        calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides, pads);
    std::array<size_t, 4> outDims = {
        {idim.n, outSz.first, outSz.second, depth}};
    auto outTy = G_.getParent()->uniqueType(ElemKind::FloatTy, outDims);

    auto *node = G_.createConv(opName, tr, filter, bias, outTy, kernels,
                               strides, pads, group);

    // Transpose the output back.
    auto *N = G_.createTranspose(opName, node, NHWC2NCHW);
    addNodeAsOutput(op, N);
    return;
  }

  if (typeName == "MaxPool" || typeName == "AveragePool") {
    // Load the inputs:
    auto in = getNodeValueOrCreateVariableByName(op.input(0));
    std::vector<size_t> strides = getSizeHW(dict, "stride", 1);
    std::vector<size_t> kernels = getSizeHW(dict, "kernel", 0);
    std::vector<size_t> pads = getPads(dict);

    auto *tr = G_.createTranspose(opName, in, NCHW2NHWC);

    // If 'global_pooling' is set then the operation will pool over the size of
    // the input by doing: kernels = {height, width}.
    if (dict.count("global_pooling")) {
      auto Ty = in->getType(0);
      kernels[0] = Ty->dims()[2];
      kernels[1] = Ty->dims()[3];
    }

    Node *node = nullptr;
    if (typeName == "MaxPool") {
      node = G_.createMaxPool(opName, tr, kernels, strides, pads);
    } else {
      node = G_.createAvgPool(opName, tr, kernels, strides, pads);
    }
    auto *N = G_.createTranspose(opName, node, NHWC2NCHW);
    addNodeAsOutput(op, N);
    return;
  }

  if (typeName == "SpatialBN") {
    auto in = getNodeValueOrCreateVariableByName(op.input(0));
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
    auto *node = G_.createBatchNormalization(opName, in, channel, epsilon);

    // Load the weights.
    cast<Variable>(node->getScale())->copyFrom(scale);
    cast<Variable>(node->getBias())->copyFrom(bias);
    cast<Variable>(node->getMean())->copyFrom(mean);
    cast<Variable>(node->getVar())->copyFrom(var);

    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "Concat") {
    const unsigned numInputs = op.input_size();
    llvm::SmallVector<NodeValue, 4> inputs;
    inputs.reserve(numInputs);
    for (unsigned i = 0; i < numInputs; i++) {
      inputs.push_back(getNodeValueOrCreateVariableByName(op.input(i)));
    }

    // If axis exists it takes priority over channel.
    unsigned channel =
        dict.count("axis") ? loadInt(dict["axis"]) : getChannel(dict);

    Node *node = G_.createConcat(opName, inputs, channel);

    // Concat has multiple outputs in Caffe2, but I believe the other output
    // (split_info) is not used for inference.
    nodeValueByName_[op.output(0)] = NodeValue(node, 0);
    return;
  }

  if (typeName == "FC") {
    // Load the inputs:
    auto in = getNodeValueOrCreateVariableByName(op.input(0));
    if (in.getType()->dims().size() > 2) {
      size_t axis = dict.count("axis") ? loadInt(dict["axis"]) : 1;
      in = G_.createFlatten("fc.in", in, axis);
    }

    // Load weights.
    Tensor *w = getTensorByName(op.input(1));
    Tensor *b = getTensorByName(op.input(2));
    size_t axis_w = dict.count("axis_w") ? loadInt(dict["axis_w"]) : 1;

    // Caffe2 stores the transposed W matrix. In here we first coerce W to a 2D
    // matrix size if necessay and then transpose it back.
    Tensor wtag;
    if (w->dims().size() > 2) {
      auto wDims = flattenCdr(w->dims(), axis_w);
      Tensor tmp(ElemKind::FloatTy, {wDims.first, wDims.second});
      tmp.copyRawFrom(w);
      tmp.transpose(&wtag, {1, 0});
    } else
      w->transpose(&wtag, {1, 0});

    auto W = G_.getParent()->addVar(
        new Variable("weights", VisibilityKind::Private, std::move(wtag)));
    auto B = G_.getParent()->addVar(
        new Variable("biases", VisibilityKind::Private, std::move(*b)));
    auto *node = G_.createFullyConnected(opName, in, W, B);

    // Save the outputs:
    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "ChannelShuffle") {
    auto in = getNodeValueOrCreateVariableByName(op.input(0));

    size_t group = loadInt(dict["group"]);
    size_t kernel = loadInt(dict["kernel"]);

    Node *node = G_.createChannelShuffle(opName, in, group, kernel);
    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "Squeeze") {
    auto in = getNodeValueOrCreateVariableByName(op.input(0));
    auto dims = getShape(dict["dims"]);
    Node *node = G_.createSqueeze(opName, in, dims);
    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "Gather" || typeName == "BatchGather") {
    auto data = getNodeValueOrCreateVariableByName(op.input(0));
    auto indices = getNodeValueOrCreateVariableByName(op.input(1));
    size_t batchDims = typeName == "Gather" ? 0 : 1;

    Node *GN = G_.createGather(opName, data, indices, batchDims);
    addNodeAsOutput(op, GN);
    return;
  }

  if (typeName == "Log") {
    // Load the inputs:
    auto in = getNodeValueOrCreateVariableByName(op.input(0));
    // Create the log:
    auto *R = G_.createLog(opName, in);
    addNodeAsOutput(op, R);
    return;
  }

  if (typeName == "EQ") {
    auto in0 = getNodeValueOrCreateVariableByName(op.input(0));
    auto in1 = getNodeValueOrCreateVariableByName(op.input(1));
    auto *node = G_.createCmpEQ(opName, in0, in1);
    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "Tile") {
    auto in = getNodeValueOrCreateVariableByName(op.input(0));
    unsigned tiles = loadInt(dict["tiles"]);
    unsigned axis = loadInt(dict["axis"]);

    auto *node = G_.createTile(opName, in, tiles, axis);
    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "Free") {
    // Glow frees memory automatically.
    return;
  }
  if (typeName == "StopGradient") {
    auto in = getNodeValueOrCreateVariableByName(op.input(0));
    // Currently Caffe2 importer only supports inference.
    addNodeAsOutput(op, in);
    return;
  }

  if (typeName == "Transpose") {
    return loadTranspose(op, dict, "axes");
  }

  if (typeName == "SparseLengthsSum") {
    auto in0 = getNodeValueOrCreateVariableByName(op.input(0));
    auto in1 = getNodeValueOrCreateVariableByName(op.input(1));
    auto in2 = getNodeValueOrCreateVariableByName(op.input(2));
    auto *node = G_.createSparseLengthsSum(opName, in0, in1, in2);
    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "ExpandDims") {
    auto in = getNodeValueOrCreateVariableByName(op.input(0));
    auto dims = getShape(dict["dims"]);
    Node *node = G_.createExpandDims(opName, in, dims);
    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "CopyCPUToMKL" || typeName == "CopyMKLToCPU") {
    // Glow does not support MKL now, just pass these two ops.
    auto in = getNodeValueOrCreateVariableByName(op.input(0));
    addNodeAsOutput(op, in);
    return;
  }

  if (typeName == "Slice") {
    auto data = getNodeValueOrCreateVariableByName(op.input(0));

    auto starts = getShape<ssize_t>(dict["starts"]);
    auto ends = getShape<ssize_t>(dict["ends"]);

    std::vector<size_t> newStarts, newEnds;
    assert(starts.size() == ends.size());
    for (size_t i = 0; i < starts.size(); i++) {
      ssize_t newStart = starts[i];
      if (newStart == -1) {
        newStart = data->dims(0)[i];
      }
      assert(newStart >= 0 && "Indices should never be negative.");
      newStarts.push_back(newStart);

      ssize_t newEnd = ends[i];
      if (newEnd == -1) {
        newEnd = data->dims(0)[i];
      }
      assert(newEnd >= 0 && "Indices should never be negative.");
      newEnds.push_back(newEnd);
    }

    Node *SN = G_.createSlice(opName, data, newStarts, newEnds);
    addNodeAsOutput(op, SN);
    return;
  }

  if (typeName == "MatMul" || typeName == "BatchMatMul") {
    auto LHS = getNodeValueOrCreateVariableByName(op.input(0));
    auto RHS = getNodeValueOrCreateVariableByName(op.input(1));

    bool transLHS = dict.count("trans_a") && (loadInt(dict["trans_a"]) == 1);
    (void)transLHS;
    assert(!transLHS && "Don't support transpose lhs for now.");
    bool transRHS = dict.count("trans_b") && (loadInt(dict["trans_b"]) == 1);
    if (transRHS) {
      RHS = G_.createTranspose("RHS.transpose", RHS, {1, 0});
    }

    Node *node = nullptr;

    // BatchMatMul sometimes is actually just a matmul, depending on dimensions
    // of inputs. Thus, only do batch matmul if LHS is 3-dimensional.
    if (typeName == "BatchMatMul" && LHS->dims(0).size() == 3) {
      node = G_.createBatchMatMul(opName, LHS, RHS);
    } else {
      node = G_.createMatMul(opName, LHS, RHS);
    }

    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "Cast") {
    auto in = getNodeValueOrCreateVariableByName(op.input(0));
    assert(in.getElementType() == ElemKind::FloatTy &&
           "Only float to float cast is currently supported.");
    int to = loadInt(dict["to"]);
    (void)to;
    assert(to == caffe2::TensorProto_DataType_FLOAT &&
           "Only float to float cast is currently supported.");
    addNodeAsOutput(op, in);
    return;
  }

  if (typeName == "ScatterAssign") {
    auto data = getNodeValueOrCreateVariableByName(op.input(0));
    auto indices = getNodeValueOrCreateVariableByName(op.input(1));
    auto slices = getNodeValueOrCreateVariableByName(op.input(2));

    Node *SAN = G_.createScatterAssign(opName, data, indices, slices);
    addNodeAsOutput(op, SAN);
    return;
  }

  if (typeName == "ConstantFill" || typeName == "GivenTensorIntFill" ||
      typeName == "GivenTensorInt64Fill") {
    loadWeight(op);
    return;
  }

  unexpectedNodeError(op, "Unsupported operator.");
}

void caffe2ModelLoader::loadNetwork(caffe2::NetDef &net) {
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
    outputsByName_[outputName] = G_.createSave("save_" + outputName, r);
  }
}

void caffe2ModelLoader::loadWeight(const caffe2::OperatorDef &op) {
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
    if (dict["values"]->floats_size()) {
      assert(typeName != "GivenTensorIntFill" &&
             typeName != "GivenTensorInt64Fill");
      T->reset(ElemKind::FloatTy, dim);
      auto TH = T->getHandle<>();
      for (auto num : dict["values"]->floats()) {
        TH.raw(i++) = num;
      }
    } else if (dict["values"]->ints_size()) {
      T->reset(ElemKind::IndexTy, dim);
      auto TH = T->getHandle<size_t>();
      for (auto num : dict["values"]->ints()) {
        assert(0 <= num && num < (1LL << 32) &&
               "Only uint32 integers are supported");
        TH.raw(i++) = num;
      }
    } else {
      unexpectedNodeError(op, "Unsupported data type for GivenTensorFill.");
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

    auto dim = getShape(dict["shape"]);
    T->reset(ElemKind::FloatTy, dim);
    auto TH = T->getHandle<>();
    TH.clear();
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

void caffe2ModelLoader::loadWeights(caffe2::NetDef &net) {
  for (auto &op : net.op()) {
    loadWeight(op);
  }
}

caffe2ModelLoader::caffe2ModelLoader(const std::string &netDescFilename,
                                     const std::string &netWeightFilename,
                                     llvm::ArrayRef<const char *> names,
                                     llvm::ArrayRef<Tensor *> tensors,
                                     Function &F)
    : CommonOperatorLoader(names, tensors, F) {
  // The caffe2 weights that we are deserializing.
  caffe2::NetDef weightsDef;
  // The caffe2 network descriptor that we are deserializing.
  caffe2::NetDef networkDef;

  loadProtoFile(networkDef, netDescFilename);
  loadProtoFile(weightsDef, netWeightFilename);
  loadWeights(weightsDef);
  loadNetwork(networkDef);
}
