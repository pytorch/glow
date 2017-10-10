// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Importer/Caffe2.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "glow/Interpreter/Interpreter.h"
#include "glow/Network/Tensor.h"

#include "caffe.pb.h"
#include <google/protobuf/text_format.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace glow;

auto NCHW2NHWC = {0u, 2u, 3u, 1u};
auto NHWC2NCHW = {0u, 3u, 1u, 2u};

using ArgumentDictionaryTy =
    std::unordered_map<std::string, const caffe2::Argument *>;

/// Prints a single serialized protocol buffer node. This method is useful for
/// debugging the network and printing errors.
template <typename T>
void unexpectedNodeError(const T &node, const std::string &message) {
  std::string str;
  google::protobuf::TextFormat::PrintToString(node, &str);
  std::cout << message << "\n" << str << "\n";
}

/// Reads a single integer.
static int loadInt(const caffe2::Argument *arg) {
  assert(arg->has_i() && "Node has no Int value");
  return arg->i();
}

/// Reads a single float.
static float loadFloat(const caffe2::Argument *arg) {
  assert(arg->has_f() && "Node has no float value");
  return arg->f();
}

/// Reads a single string.
static const std::string &loadStr(const caffe2::Argument *arg) {
  assert(arg->has_s() && "Node has no str value");
  return arg->s();
}

/// Loda the 'shape' record into a vector of sizes.
std::vector<size_t> getShape(const ::caffe2::Argument *arg) {
  std::vector<size_t> dim;
  for (auto i : arg->ints()) {
    dim.push_back(i);
  }
  return dim;
}

/// Translates the protocol buffer node \p op into a random access map.
static ArgumentDictionaryTy loadArgumenrMap(const caffe2::OperatorDef &op) {
  ArgumentDictionaryTy dict;
  for (auto i = 0, e = op.arg_size(); i < e; i++) {
    const caffe2::Argument &arg = op.arg(i);
    dict[arg.name()] = &arg;
  }
  return dict;
}

bool caffe2ModelLoader::loadProtoFile(caffe2::NetDef &net,
                                      const std::string &filename) {
  std::fstream ff(filename, std::ios::in | std::ios::binary);
  assert(ff && "Can't find the model or network files.");

  bool parseNet = false;
  if (filename.find(".pbtxt") != std::string::npos) {
    std::string str((std::istreambuf_iterator<char>(ff)),
                    std::istreambuf_iterator<char>());
    parseNet = google::protobuf::TextFormat::ParseFromString(str, &net);
  } else {
    parseNet = net.ParseFromIstream(&ff);
  }

  assert(parseNet && "Failed to parse the network descriptor.");
  return true;
}

Tensor *caffe2ModelLoader::getTensorByName(const std::string &name) {
  assert(tensors_.count(name) &&
         "There is no tensor registered with this name.");
  return tensors_[name];
}

Node *caffe2ModelLoader::getNodeByName(const std::string &name) {
  auto it = nodeByName_.find(name);
  if (it != nodeByName_.end()) {
    return it->second;
  }

  assert(false && "Could not find a node with this name.");
  glow_unreachable();
}

Node *caffe2ModelLoader::getOrCreateNodeByName(const std::string &name) {
  auto it = nodeByName_.find(name);
  if (it != nodeByName_.end()) {
    return it->second;
  }

  Tensor *T = getTensorByName(name);
  auto *V = G_.createVariable(T->getElementType(), T->dims(), name,
                              WeightVar::InitKind::Broadcast);
  nodeByName_[name] = V;
  return V;
}

void caffe2ModelLoader::loadOperator(const caffe2::OperatorDef &op) {
  ArgumentDictionaryTy dict = loadArgumenrMap(op);

  const std::string &typeName = op.type();

  if (typeName == "Relu") {
    // Load the inputs:
    auto *in = getOrCreateNodeByName(op.input(0));
    // Create the RELU:
    auto *R = G_.createRELU(op.name(), in);
    // Save the outputs:
    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = R;
    }
    return;
  }

  if (typeName == "Conv") {
    // Load the inputs:
    int stride = loadInt(dict["stride"]);
    int pad = dict.count("pad") ? loadInt(dict["pad"]) : 0;
    int kernel = loadInt(dict["kernel"]);

    auto *in = getOrCreateNodeByName(op.input(0));
    Tensor *w = getTensorByName(op.input(1));

    // Transpose the weights to the right format. Glow expects to read the
    // weights in the format CRSK. Caffe2 stores the operators as KCRS.
    // C - output_depth, R - filter_height, S - filter_width, K - input_depth.

    // TODO: need to test this code with a large batch size to verify that the
    // conversion is correct.
    Tensor *wtag = new Tensor();
    w->getHandle<FloatTy>().transpose(wtag, {0, 2, 3, 1});
    tensors_[op.name() + "_conv_filter"] = wtag;

    // The structure of the conv weigts is: NHWC. We take the C, which is the
    // number of filters. We use this value to calculate the size of the bias
    // if it is not specified.
    size_t numFilters = wtag->dims()[0];

    // Load the bias vector:
    Tensor *b = nullptr;

    // Check if we have a serialized bias vector.
    if (op.input_size() > 2) {
      auto &biasTensorName = op.input(2);
      if (tensors_.count(biasTensorName)) {
        b = getTensorByName(biasTensorName);
      }
    }

    // If we don't have a bias vector then create one that matches the weight
    // size and fill it with zeros.
    if (!b) {
      b = new Tensor(ElemKind::FloatTy, {numFilters});
      b->getHandle<FloatTy>().clear();
      tensors_[op.name() + "_conv_bias"] = b;
    }

    auto *tr = G_.createTranspose(op.name(), in, NCHW2NHWC);
    auto *node = G_.createConv(op.name(), tr, numFilters, kernel, stride, pad);

    // Load the weights into the operator.
    registerVariableInit(node->getFilter(), wtag);
    registerVariableInit(node->getBias(), b);

    auto *N = G_.createTranspose(op.name(), node, NHWC2NCHW);
    // Save the outputs:
    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = N;
    }
    return;
  }

  if (typeName == "MaxPool" || typeName == "AveragePool") {
    using OpKind = PoolInst::OpKind;
    OpKind opk = (typeName == "MaxPool") ? OpKind::Max : OpKind::Avg;

    // Load the inputs:
    auto *in = getOrCreateNodeByName(op.input(0));
    int stride = loadInt(dict["stride"]);
    int pad = dict.count("pad") ? loadInt(dict["pad"]) : 0;
    int kernel = loadInt(dict["kernel"]);

    auto *tr = G_.createTranspose(op.name(), in, NCHW2NHWC);
    auto *node = G_.createPool(op.name(), tr, opk, kernel, stride, pad);
    auto *N = G_.createTranspose(op.name(), node, NHWC2NCHW);

    // Save the outputs:
    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = N;
    }
    return;
  }

  if (typeName == "Dropout") {
    auto *in = getOrCreateNodeByName(op.input(0));
    // Save the identity operation:
    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = in;
    }
    return;
  }

  if (typeName == "SpatialBN") {
    auto *in = getOrCreateNodeByName(op.input(0));
    auto *scale = getTensorByName(op.input(1));
    auto *bias = getTensorByName(op.input(2));
    auto *mean = getTensorByName(op.input(3));
    auto *var = getTensorByName(op.input(4));
    float epsilon = loadFloat(dict["epsilon"]);

    unsigned channel = 0;
    auto order = loadStr(dict["order"]);
    if (order == "NHWC") {
      channel = 3;
    } else if (order == "NCHW") {
      channel = 1;
    } else {
      assert(false && "Invalid order field");
    }

    auto *node = G_.createBatchNormalization(op.name(), in, channel, epsilon);

    // Load the weights.
    registerVariableInit(node->getScale(), scale);
    registerVariableInit(node->getBias(), bias);
    registerVariableInit(node->getMean(), mean);
    registerVariableInit(node->getVar(), var);

    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = node;
    }
    return;
  }
  if (typeName == "Sum") {
    auto *in0 = getOrCreateNodeByName(op.input(0));
    auto *in1 = getOrCreateNodeByName(op.input(1));
    auto *node =
        G_.createArithmetic(op.name(), in0, in1, ArithmeticInst::OpKind::Add);
    // Save the outputs:
    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = node;
    }
    return;
  }

  if (typeName == "Softmax") {
    auto *softmaxExpected = getOrCreateNodeByName("softmax_expected");

    // Load the inputs:
    auto *in = getOrCreateNodeByName(op.input(0));

    auto *node = G_.createSoftMax(op.name(), in, softmaxExpected);
    // Save the outputs:
    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = node;
    }
    return;
  }
  if (typeName == "FC") {
    // Load the inputs:
    auto *in = getOrCreateNodeByName(op.input(0));
    Tensor *w = getTensorByName(op.input(1));
    Tensor *b = getTensorByName(op.input(2));
    auto *FC = G_.createFullyConnected(op.name(), in, b->size());

    // Load weights.
    registerVariableInit(FC->getFilter(), w);
    registerVariableInit(FC->getBias(), b);

    // Save the outputs:
    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = FC;
    }
    return;
  }

  unexpectedNodeError(op, "Could not load the operator.");
}

void caffe2ModelLoader::loadNetwork(caffe2::NetDef &net) {
  /// Load the network operators:
  for (int i = 0; i < net.op_size(); i++) {
    auto &op = net.op(i);
    loadOperator(op);
  }

  root_ = getNodeByName(net.external_output(0));
}

void caffe2ModelLoader::loadWeights(caffe2::NetDef &net) {
  for (auto &op : net.op()) {
    ArgumentDictionaryTy dict = loadArgumenrMap(op);

    /// Load tensors with values:
    if (op.type() == "GivenTensorFill") {
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
      T->reset(ElemKind::FloatTy, dim);
      auto TH = T->getHandle<FloatTy>();
      size_t i = 0;
      for (auto f : dict["values"]->floats()) {
        TH.raw(i++) = f;
      }

      assert(i == TH.size() && "The number of serialized values does not "
                               "match the size of the tensor.");
      continue;
    }

    // Load tensors with constant fill:
    if (op.type() == "ConstantFill") {
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
      auto TH = T->getHandle<FloatTy>();
      TH.clear();
      continue;
    }

    unexpectedNodeError(op, "Unknown node found.");
  }
}

caffe2ModelLoader::caffe2ModelLoader(const std::string &netDescFilename,
                                     const std::string &netWeightFilename,
                                     llvm::ArrayRef<const char *> names,
                                     llvm::ArrayRef<Tensor *> tensors,
                                     Interpreter &IP)
    : IP_(IP), G_(IP_.getGraph()) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  // The caffe2 weights that we are deserializing.
  caffe2::NetDef weightsDef;
  // The caffe2 network descriptor that we are deserializing.
  caffe2::NetDef networkDef;

  assert(names.size() == tensors.size() && "Invalid initialization list");
  for (unsigned i = 0; i < names.size(); i++) {
    tensors_[names[i]] = tensors[i];
    getOrCreateNodeByName(names[i]);
  }

  loadProtoFile(networkDef, netDescFilename);
  loadProtoFile(weightsDef, netWeightFilename);
  loadWeights(weightsDef);
  loadNetwork(networkDef);

  // Save the result of the last operator into a weight.
  root_ = G_.createReturn("ret", root_);

  // Emit IR for the graph.
  G_.generateIR();

  // Load the value of the variables.
  for (auto p : variableInit_) {
    WeightVar *N = cast<WeightVar>(G_.getIRForNode(p.first));
    N->setInitKind(WeightVar::InitKind::Extern);
    IP.initValue(N, p.second);
  }
}
