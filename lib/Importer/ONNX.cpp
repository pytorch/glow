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

#include "glow/Importer/ONNX.h"
#include "glow/Base/Tensor.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "onnx.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace glow;
using llvm::cast;

static auto NCHW2NHWC = {0u, 2u, 3u, 1u};
static auto NHWC2NCHW = {0u, 3u, 1u, 2u};

using ArgumentDictionaryTy =
    std::unordered_map<std::string, const onnx::AttributeProto *>;

/// Prints a single serialized protocol buffer node. This method is useful for
/// debugging the network and printing errors.
template <typename T>
void unexpectedNodeError(const T &node, llvm::StringRef message) {
  std::string str;
  google::protobuf::TextFormat::PrintToString(node, &str);
  llvm::outs() << message << "\n" << str << "\n";
}

/// Reads a single integer.
static int loadInt(const onnx::AttributeProto *arg) {
  assert(arg->has_i() && "Node has no Int value");
  return arg->i();
}

/// Reads a single float.
static float loadFloat(const onnx::AttributeProto *arg) {
  assert(arg->has_f() && "Node has no float value");
  return arg->f();
}

/// Reads a single string.
static const std::string &loadStr(const onnx::AttributeProto *arg) {
  assert(arg->has_s() && "Node has no str value");
  return arg->s();
}

/// Load the 'shape' record into a vector of sizes.
std::vector<size_t> getShape(const onnx::AttributeProto *arg) {
  std::vector<size_t> dim;
  for (auto i : arg->ints()) {
    dim.push_back(i);
  }
  return dim;
}

bool arrayIsConstant(const llvm::ArrayRef<size_t> a) {
  for (size_t i = 1; i < a.size(); i++)
    if (a[0] != a[i])
      return false;
  return true;
}

size_t getConstantArrayHead(const onnx::AttributeProto *arg) {
  auto dim = getShape(arg);
  assert(arrayIsConstant(dim) &&
         "Only equal values along each dimensions are supported");
  return dim[0];
}

/// Translates the protocol buffer node \p op into a random access map.
static ArgumentDictionaryTy loadArgumentMap(const onnx::NodeProto &op) {
  ArgumentDictionaryTy dict;
  for (auto i = 0, e = op.attribute_size(); i < e; i++) {
    const onnx::AttributeProto &arg = op.attribute(i);
    dict[arg.name()] = &arg;
  }
  return dict;
}

bool ONNXModelLoader::loadProtoFile(onnx::GraphProto &net,
                                    const std::string &filename) {
  std::ifstream ff(filename, std::ios::in | std::ios::binary);
  GLOW_ASSERT(ff && "Can't find the model or network files.");

  // Construct and configure a Coded Input Stream
  google::protobuf::io::IstreamInputStream filestr(&ff);
  google::protobuf::io::CodedInputStream codedstr(&filestr);
  // Don't warn about large file sizes.
  codedstr.SetTotalBytesLimit(1e+9, 1e+9);
  onnx::ModelProto MP;
  bool parseNet = MP.ParseFromCodedStream(&codedstr);
  net = MP.graph();

  GLOW_ASSERT(parseNet && "Failed to parse the network descriptor.");
  return true;
}

size_t getPad(ArgumentDictionaryTy &dict) {
  if (dict.count("pads")) {
    return getConstantArrayHead(dict["pads"]);
  } else if (dict.count("auto_pad")) {
    auto padStr = loadStr(dict["auto_pad"]);
    if (padStr == "VALID")
      return 0;
    assert(false && "only auto_pad==VALID is supported");
  }
  return 0;
}

Tensor *ONNXModelLoader::getTensorByName(const std::string &name) {
  assert(tensors_.count(name) &&
         "There is no tensor registered with this name.");
  return tensors_[name];
}

Node *ONNXModelLoader::getNodeByName(const std::string &name) {
  auto it = nodeByName_.find(name);
  if (it != nodeByName_.end()) {
    return it->second;
  }

  llvm_unreachable("Could not find a node with this name.");
}

Node *ONNXModelLoader::getOrCreateNodeByName(const std::string &name) {
  auto it = nodeByName_.find(name);
  if (it != nodeByName_.end()) {
    return it->second;
  }

  Tensor *T = getTensorByName(name);
  auto *V = G_.getParent()->createVariable(T->getElementType(), T->dims(), name,
                                           VisibilityKind::Private,
                                           Variable::TrainKind::Broadcast);
  V->copyFrom(T);
  nodeByName_[name] = V;
  return V;
}

bool ONNXModelLoader::hasNodeByName(const std::string &name) const {
  auto it = nodeByName_.find(name);
  return (it != nodeByName_.end());
}

void loadTensor(const onnx::TensorProto &in, Tensor *T) {
  std::vector<size_t> dim;
  for (auto d : in.dims()) {
    dim.push_back(d);
  }

  if (in.data_type() == onnx::TensorProto::FLOAT) {
    T->reset(ElemKind::FloatTy, dim);

    if (in.float_data_size() > 0) {
      auto TH = T->getHandle<>();
      size_t i = 0;
      for (auto f : in.float_data()) {
        TH.raw(i++) = f;
      }
    } else if (in.has_raw_data()) {
      std::istringstream inStream(in.raw_data(), std::stringstream::binary);
      inStream.read((char *)T->getRawDataPointer<float>(),
                    T->size() * sizeof(float));
    } else {
      assert(false && "Unsupported Tensor format.");
    }
  } else if (in.data_type() == onnx::TensorProto::INT64) {
    // TODO: either switch IndexTy to be 64 bit, or switch to another type here
    T->reset(ElemKind::IndexTy, dim);

    if (in.int64_data_size() > 0) {
      auto TH = T->getHandle<>();
      size_t i = 0;
      for (auto f : in.int64_data()) {
        TH.raw(i++) = f;
      }
    } else if (in.has_raw_data()) {
      std::istringstream inStream(in.raw_data(), std::stringstream::binary);
      inStream.read((char *)T->getRawDataPointer<size_t>(),
                    T->size() * sizeof(int64_t));
    } else {
      assert(false && "Unsupported Tensor format.");
    }
  } else {
    assert(false && "Only float and index tensors are supported");
  }
}

void ONNXModelLoader::loadOperator(const onnx::NodeProto &op) {
  ArgumentDictionaryTy dict = loadArgumentMap(op);

  const std::string &typeName = op.op_type();
  const std::string &opName = op.name().length() ? op.name() : op.output(0);

  // Load tensors with values:
  if (typeName == "Constant") {
    /*
      output: "Parameter6"
      name: "Parameter6"
      op_type: "Constant"
      attribute {
        name: "value"
        t {
          dims: 8
          data_type: FLOAT
          float_data: -0.161539719
          float_data: -0.433835655
          float_data: 0.091641359
          float_data: -0.0168522168
          float_data: -0.0650264397
          float_data: -0.131737873
          float_data: 0.0204175506
          float_data: -0.121110231
        }
        type: TENSOR
      }
      doc_string: ""
      domain: ""
    */

    const auto &name = op.output(0);
    // If the tensor is pre-populated by the user of this class then we don't
    // need to allocate a new tensor.
    if (tensors_.count(name)) {
      return;
    }

    assert(dict["value"]->type() == onnx::AttributeProto::TENSOR &&
           "Only Tensor type constants are supported.");

    auto *T = new Tensor();
    loadTensor(dict["value"]->t(), T);
    tensors_[name] = T;
    return;
  }

  if (typeName == "Relu") {
    // Load the inputs:
    auto *in = getOrCreateNodeByName(op.input(0));
    // Create the RELU:
    auto *R = G_.createRELU(opName, in);
    // Save the outputs:
    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = R;
    }
    return;
  }

  if (typeName == "Conv") {
    // Load the inputs:
    int stride =
        dict.count("strides") ? getConstantArrayHead(dict["strides"]) : 1;
    int pad = getPad(dict);
    unsigned group = dict.count("group") ? loadInt(dict["group"]) : 1;

    auto *in = getOrCreateNodeByName(op.input(0));
    Tensor *w = getTensorByName(op.input(1));

    // Transpose the weights to the right format. Glow expects to read the
    // weights in the format CRSK. ONNX stores the operators as KCRS.
    // C - output_depth, R - filter_height, S - filter_width, K - input_depth.
    Tensor wtag;
    w->transpose(&wtag, {0, 2, 3, 1});

    // The structure of the conv weigts is: NHWC. We take the C, which is the
    // number of filters. We use this value to calculate the size of the bias
    // if it is not specified.
    size_t depth = wtag.dims()[0];

    // Construct the Filter field.
    Variable *filter =
        G_.getParent()->createVariable(&wtag.getType(), "conv.filter");
    filter->getPayload().copyFrom(&wtag);

    unsigned kernel;
    if (dict.count("kernel_shape")) {
      kernel = getConstantArrayHead(dict["kernel_shape"]);
    } else {
      assert(filter->dims()[1] == filter->dims()[2] &&
             "Only square kernels are supported");
      kernel = filter->dims()[1];
    }

    // Construct the Bias field.
    Variable *bias =
        G_.getParent()->createVariable(ElemKind::FloatTy, {depth}, "conv.bias");
    bias->getPayload().zero();

    // Check if we have a serialized bias vector.
    if (op.input_size() > 2) {
      auto &biasTensorName = op.input(2);
      if (tensors_.count(biasTensorName)) {
        // Load the serialized bias vector.
        Tensor *b = getTensorByName(biasTensorName);
        bias->copyFrom(b);
      }
    }

    // ONNX passes the input as NCHW, and we expect the input to be NHWC.
    auto *tr = G_.createTranspose(opName, in, NCHW2NHWC);

    // Calculate the size and allocate the output buffer.
    ShapeNHWC idim = ShapeNHWC(tr->dims());
    auto outSz = calculateConvOutputDims(idim.h, idim.w, kernel, stride, pad);
    std::array<size_t, 4> outDims = {
        {idim.n, outSz.first, outSz.second, depth}};
    auto outTy = G_.getParent()->uniqueType(ElemKind::FloatTy, outDims);

    auto *node = G_.createConv(opName, tr, filter, bias, outTy, kernel, stride,
                               pad, group);

    // Transpose the output back.
    auto *N = G_.createTranspose(opName, node, NHWC2NCHW);
    // Save the outputs:
    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = N;
    }
    return;
  }

  if (typeName == "MaxPool" || typeName == "AveragePool") {
    // Load the inputs:
    auto *in = getOrCreateNodeByName(op.input(0));
    int stride =
        dict.count("strides") ? getConstantArrayHead(dict["strides"]) : 1;
    size_t kernel = getConstantArrayHead(dict["kernel_shape"]);

    int pad = getPad(dict);

    auto *tr = G_.createTranspose(opName, in, NCHW2NHWC);

    // If 'global_pooling' is set then the operation will pool over the size of
    // the input by doing: kernel = height/width.
    if (dict.count("global_pooling")) {
      auto Ty = in->getType();
      kernel = Ty->dims()[3];
    }

    Node *node = nullptr;
    if (typeName == "MaxPool") {
      node = G_.createPoolMax(opName, tr, kernel, stride, pad);
    } else {
      node = G_.createPoolAvg(opName, tr, kernel, stride, pad);
    }
    auto *N = G_.createTranspose(opName, node, NHWC2NCHW);

    // Save the outputs:
    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = N;
    }
    return;
  }

  if (typeName == "GlobalAveragePool") {
    // Load the inputs:
    auto *in = getOrCreateNodeByName(op.input(0));
    int stride =
      dict.count("strides") ? getConstantArrayHead(dict["strides"]) : 1;

    GLOW_ASSERT(in->dims()[2] == in->dims()[3] && "For the image, height == weight is required");

    size_t kernel = in->dims()[2];
    int pad = getPad(dict);
    auto *tr = G_.createTranspose(opName, in, NCHW2NHWC);
    Node *node = G_.createPoolAvg(opName, tr, kernel, stride, pad);
    auto *N = G_.createTranspose(opName, node, NHWC2NCHW);
    // Save the outputs:
    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = N;
    }
    return;
  }

  if (typeName == "Squeeze") {
    auto *in = getOrCreateNodeByName(op.input(0));
    auto axes = getShape(dict["axes"]);
    Node *node = G_.createSqueeze(opName, in, axes);

    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = node;
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

  if (typeName == "BatchNormalization") {
    auto *in = getOrCreateNodeByName(op.input(0));
    auto *scale = getTensorByName(op.input(1));
    auto *bias = getTensorByName(op.input(2));
    auto *mean = getTensorByName(op.input(3));
    auto *var = getTensorByName(op.input(4));
    float epsilon = 1e-5f; // default
    auto epsilonIt = dict.find("epsilon");
    if (epsilonIt != dict.end()) {
      epsilon = loadFloat(epsilonIt->second);
    }

    auto *node = G_.createBatchNormalization(opName, in, 1, epsilon);

    // Load the weights.
    cast<Variable>(node->getScale())->copyFrom(scale);
    cast<Variable>(node->getBias())->copyFrom(bias);
    cast<Variable>(node->getMean())->copyFrom(mean);
    cast<Variable>(node->getVar())->copyFrom(var);

    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = node;
    }
    return;
  }

  if (typeName == "Concat") {
    const unsigned numInputs = op.input_size();
    llvm::SmallVector<Node *, 4> inputs;
    inputs.reserve(numInputs);
    for (unsigned i = 0; i < numInputs; i++) {
      inputs.push_back(getOrCreateNodeByName(op.input(i)));
    }

    auto axis = loadInt(dict["axis"]);
    Node *node = G_.createConcat(opName, inputs, axis);

    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = node;
    }
    return;
  }

  if (typeName == "Sum") {
    // TODO: support variadic arguments
    assert(op.input_size() == 2 && "Only Sum of 2 inputs is supported.");
    auto *in0 = getOrCreateNodeByName(op.input(0));
    auto *in1 = getOrCreateNodeByName(op.input(1));
    auto *node = G_.createAdd(opName, in0, in1);
    // Save the outputs:
    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = node;
    }
    return;
  }

  if (typeName == "Softmax") {
    auto *softmaxExpected = getOrCreateNodeByName("softmax_expected");

    // Load the inputs:
    Node *in = getOrCreateNodeByName(op.input(0));

    // ONNX allows shapes like <N x 10 x 1 x 1 >. Flatten the inputs to the
    // softmax function. This is similar to a bitcast operation.
    auto flatten = flattenCdr(in->getType()->dims());
    in = G_.createReshape("reshape", in, {flatten.first, flatten.second});

    auto *node = G_.createSoftMax(opName, in, softmaxExpected);
    // Save the outputs:
    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = node;
    }
    return;
  }

  if (typeName == "FC") {
    // Load the inputs:
    auto *in = getOrCreateNodeByName(op.input(0));
    // Load weights.
    Tensor *w = getTensorByName(op.input(1));
    Tensor *b = getTensorByName(op.input(2));

    // ONNX stores the transposed W matrix. In here we transpose W back.
    Tensor wtag;
    w->transpose(&wtag, {1, 0});

    auto W = G_.getParent()->addVar(
        new Variable("weights", VisibilityKind::Private, std::move(wtag)));
    auto B = G_.getParent()->addVar(
        new Variable("biases", VisibilityKind::Private, std::move(*b)));
    auto *FC = G_.createFullyConnected(opName, in, W, B);

    // Save the outputs:
    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = FC;
    }
    return;
  }

  if (typeName == "LRN") {
    auto *in = getOrCreateNodeByName(op.input(0));

    size_t size = loadInt(dict["size"]);
    float alpha = loadFloat(dict["alpha"]);
    float beta = loadFloat(dict["beta"]);
    float k = loadFloat(dict["bias"]);

    auto *tr = G_.createTranspose(opName, in, NCHW2NHWC);

    auto *node = G_.createLocalResponseNormalization(opName, tr, size / 2,
                                                     alpha, beta, k);

    auto *N = G_.createTranspose(opName, node, NHWC2NCHW);

    // Save the outputs:
    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = N;
    }
    return;
  }

  if (typeName == "Mul" || typeName == "Add") {
    auto *in0 = getOrCreateNodeByName(op.input(0));
    auto *in1 = getOrCreateNodeByName(op.input(1));

    int broadcast = loadInt(dict["broadcast"]);

    Node *finalIn1 = nullptr;
    if (broadcast == 1) {
      int axis = loadInt(dict["axis"]);
      // In ONNX, if axis == -1 then it sets the axis so that the
      // trailing-most dimensions are aligned like this.
      if (axis == -1) {
        axis = in0->dims().size() - in1->dims().size();
      }
      finalIn1 = G_.createBroadcast(opName, in1, in0->dims(), axis);
    } else {
      finalIn1 = in1;
    }

    Node *node = nullptr;
    if (typeName == "Mul") {
      node = G_.createMul(opName, in0, finalIn1);
    } else {
      node = G_.createAdd(opName, in0, finalIn1);
    }

    // Save the outputs:
    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = node;
    }
    return;
  }

  if (typeName == "Gemm") {
    Node *A = getOrCreateNodeByName(op.input(0));
    Node *B = getOrCreateNodeByName(op.input(1));
    Node *C = getOrCreateNodeByName(op.input(2));

    bool broadcastC = dict.count("broadcast") && loadInt(dict["broadcast"]);
    bool transA = dict.count("transA") && loadInt(dict["transA"]);
    bool transB = dict.count("transB") && loadInt(dict["transB"]);
    // TODO: support alpha * A * B + beta * C

    if (transA)
      A = G_.createTranspose(opName, A, {1, 0});
    if (transB)
      B = G_.createTranspose(opName, B, {1, 0});

    Node *mul = G_.createMatMul(opName, A, B);
    if (broadcastC) {
      int axis = mul->dims().size() - C->dims().size();
      C = G_.createBroadcast(opName, C, mul->dims(), axis);
    }

    Node *node = G_.createAdd(opName, mul, C);

    // Save the outputs:
    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = node;
    }
    return;
  }

  if (typeName == "Reshape") {
    auto *in = getOrCreateNodeByName(op.input(0));

    std::vector<size_t> newDim;
    if (dict.count("shape")) {
      newDim = getShape(dict["shape"]);
    } else {
      auto *T = getTensorByName(op.input(1));
      auto TH = T->getHandle<size_t>();
      for (size_t i = 0, e = T->size(); i != e; i++) {
        newDim.push_back(TH.raw(i));
      }
    }

    auto *node = G_.createReshape(opName, in, newDim);

    // Save the outputs:
    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = node;
    }
    return;
  }

  unexpectedNodeError(op, "Unsupported operator.");
}

void ONNXModelLoader::loadInitializers(onnx::GraphProto &net) {
  /// Load the network initializaers:
  for (const auto &in : net.initializer()) {
    Tensor *T = new Tensor();
    loadTensor(in, T);
    tensors_[in.name()] = T;
  }
}

void ONNXModelLoader::loadNetwork(onnx::GraphProto &net) {
  /// Load the network operators:
  for (int i = 0; i < net.node_size(); i++) {
    auto &op = net.node(i);
    loadOperator(op);
  }

  assert(net.output_size() && "Network needs external outputs defined.");
  auto *r = getNodeByName(net.output(0).name());
  root_ = G_.createSave("output", r);
}

ONNXModelLoader::ONNXModelLoader(const std::string &modelDescFilename,
                                 llvm::ArrayRef<const char *> names,
                                 llvm::ArrayRef<Tensor *> tensors, Function &G)
    : G_(G) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  // The ONNX model that we are deserializing.
  onnx::GraphProto modelDef;

  assert(names.size() == tensors.size() && "Invalid initialization list");
  for (unsigned i = 0; i < names.size(); i++) {
    auto *T = tensors[i];
    auto *V = G_.getParent()->createVariable(T->getElementType(), T->dims(),
                                             names[i], VisibilityKind::Public,
                                             Variable::TrainKind::None);
    V->copyFrom(T);
    nodeByName_[names[i]] = V;
  }

  loadProtoFile(modelDef, modelDescFilename);
  loadInitializers(modelDef);
  loadNetwork(modelDef);
}

ONNXModelLoader::~ONNXModelLoader() {
  for (auto it : tensors_) {
    delete it.second;
  }
}
