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

#include "caffe.pb.h"
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

unsigned getSizeHW(ArgumentDictionaryTy &dict, const std::string &name,
                   unsigned defaultValue) {
  if (dict.count(name)) {
    return loadInt(dict[name]);
  }
  if (dict.count(name + "_h") && dict.count(name + "_w")) {
    assert(loadInt(dict[name + "_h"]) == loadInt(dict[name + "_w"]) &&
           "Unsupported size: _h and _w must be equal.");
    return loadInt(dict[name + "_h"]);
  }
  return defaultValue;
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
    int stride = getSizeHW(dict, "stride", 1);
    std::vector<size_t> pads = getPads(dict);
    unsigned kernel = getSizeHW(dict, "kernel", 0);
    unsigned group = dict.count("group") ? loadInt(dict["group"]) : 1;

    auto *in = getOrCreateVariableByName(op.input(0));
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
    auto *filter = createVariable("conv.filter", wtag);

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
    auto *bias = createVariable("conv.bias", biasTensor);

    // Caffe passes the input as NCHW, and we expect the input to be NHWC.
    auto *tr = G_.createTranspose(opName, in, NCHW2NHWC);

    // Calculate the size and allocate the output buffer.
    ShapeNHWC idim = ShapeNHWC(tr->getResult().dims());
    auto outSz =
        calculateConvPoolOutputDims(idim.h, idim.w, kernel, stride, pads);
    std::array<size_t, 4> outDims = {
        {idim.n, outSz.first, outSz.second, depth}};
    auto outTy = G_.getParent()->uniqueType(ElemKind::FloatTy, outDims);

    auto *node = G_.createConv(opName, tr, filter, bias, outTy, kernel, stride,
                               pads, group);

    // Transpose the output back.
    auto *N = G_.createTranspose(opName, node, NHWC2NCHW);
    addNodeAsOutput(op, N);
    return;
  }

  if (typeName == "MaxPool" || typeName == "AveragePool") {
    // Load the inputs:
    auto *in = getOrCreateVariableByName(op.input(0));
    int stride = getSizeHW(dict, "stride", 1);
    unsigned kernel = getSizeHW(dict, "kernel", 0);
    std::vector<size_t> pads = getPads(dict);

    auto *tr = G_.createTranspose(opName, in, NCHW2NHWC);

    // If 'global_pooling' is set then the operation will pool over the size of
    // the input by doing: kernel = height/width.
    if (dict.count("global_pooling")) {
      auto Ty = in->getType(0);
      kernel = Ty->dims()[3];
    }

    Node *node = nullptr;
    if (typeName == "MaxPool") {
      node = G_.createPoolMax(opName, tr, kernel, stride, pads);
    } else {
      node = G_.createPoolAvg(opName, tr, kernel, stride, pads);
    }
    auto *N = G_.createTranspose(opName, node, NHWC2NCHW);
    addNodeAsOutput(op, N);
    return;
  }

  if (typeName == "Dropout") {
    auto *in = getOrCreateVariableByName(op.input(0));
    // Save the identity operation:
    addNodeAsOutput(op, in);
    return;
  }

  if (typeName == "SpatialBN") {
    auto *in = getOrCreateVariableByName(op.input(0));
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
      inputs.push_back(getOrCreateVariableByName(op.input(i)));
    }

    auto channel = getChannel(dict);
    Node *node = G_.createConcat(opName, inputs, channel);

    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "FC") {
    // Load the inputs:
    auto *in = getOrCreateVariableByName(op.input(0));
    if (in->getType(0)->dims().size() > 2) {
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
    auto *in = getOrCreateVariableByName(op.input(0));

    size_t group = loadInt(dict["group"]);
    size_t kernel = loadInt(dict["kernel"]);

    Node *node = G_.createChannelShuffle(opName, in, group, kernel);
    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "Squeeze") {
    auto *in = getOrCreateVariableByName(op.input(0));
    auto dims = getShape(dict["dims"]);
    Node *node = G_.createSqueeze(opName, in, dims);
    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "Gather") {
    auto *data = getOrCreateVariableByName(op.input(0));
    auto *indices = getOrCreateVariableByName(op.input(1));

    Node *GN = G_.createGather(opName, data, indices);
    addNodeAsOutput(op, GN);
    return;
  }

  if (typeName == "Log") {
    // Load the inputs:
    auto *in = getOrCreateVariableByName(op.input(0));
    // Create the log:
    auto *R = G_.createLog(opName, in);
    addNodeAsOutput(op, R);
    return;
  }

  if (typeName == "EQ") {
    auto *in0 = getOrCreateVariableByName(op.input(0));
    auto *in1 = getOrCreateVariableByName(op.input(1));
    auto *node = G_.createCmpEQ(opName, in0, in1);
    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "Tile") {
    auto *in = getOrCreateVariableByName(op.input(0));
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

  if (typeName == "Transpose") {
    return loadTranspose(op, dict, "axes");
  }

  if (typeName == "SparseLengthsSum") {
    auto *in0 = getOrCreateVariableByName(op.input(0));
    auto *in1 = getOrCreateVariableByName(op.input(1));
    auto *in2 = getOrCreateVariableByName(op.input(2));
    auto *node = G_.createSparseLengthsSum(opName, in0, in1, in2);
    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "ExpandDims") {
    auto *in = getOrCreateVariableByName(op.input(0));
    auto dims = getShape(dict["dims"]);
    Node *node = G_.createExpandDims(opName, in, dims);
    addNodeAsOutput(op, node);
    return;
  }

  if (typeName == "CopyCPUToMKL" || typeName == "CopyMKLToCPU") {
    // Glow does not support MKL now, just pass these two ops.
    auto *in = getOrCreateVariableByName(op.input(0));
    addNodeAsOutput(op, in);
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
    auto *r = getNodeByName(outputName);
    outputsByName_[outputName] = G_.createSave("save_" + outputName, r);
  }
}

void caffe2ModelLoader::loadWeights(caffe2::NetDef &net) {
  for (auto &op : net.op()) {
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
      continue;
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
        continue;
      }

      auto *T = new Tensor();
      tensors_[name] = T;

      auto dim = getShape(dict["shape"]);
      T->reset(ElemKind::FloatTy, dim);
      auto TH = T->getHandle<>();
      TH.clear();
      continue;
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
        TH.raw(i) =
            G_.getParent()->getPRNG().nextRandReal(tensorMin, tensorMax);
      }
      continue;
    }

    unexpectedNodeError(op, "Unsupported weight kind");
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
