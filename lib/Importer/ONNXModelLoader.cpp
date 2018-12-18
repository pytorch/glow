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

#include "glow/Importer/ONNXModelLoader.h"
#include "glow/Base/Tensor.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "onnx/onnx_pb.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace glow;
using llvm::cast;

namespace {
/// Creates tensor \p T from the input \p in. Note, there is no data associated
/// with the Tensor. This method makes sure that the tensor is created with the
/// proper shape and element type.
llvm::Error setTensorType(const ONNX_NAMESPACE::TypeProto &in, Tensor *T) {
  std::vector<size_t> dim;
  for (auto d : in.tensor_type().shape().dim()) {
    dim.push_back(d.dim_value());
  }

  if (in.tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto::FLOAT) {
    T->reset(ElemKind::FloatTy, dim);
    return llvm::Error::success();
  } else if (in.tensor_type().elem_type() ==
             ONNX_NAMESPACE::TensorProto::INT64) {
    T->reset(ElemKind::Int64ITy, dim);
    return llvm::Error::success();
  } else if (in.tensor_type().elem_type() ==
             ONNX_NAMESPACE::TensorProto::INT32) {
    T->reset(ElemKind::Int32ITy, dim);
    return llvm::Error::success();
  } else {
    RETURN_ERR("Only float and index tensors are supported");
  }
}
} // namespace

using ArgumentDictionaryTy =
    std::unordered_map<std::string, const ONNX_NAMESPACE::AttributeProto *>;

/// Translates the protocol buffer node \p op into a random access map.
static ArgumentDictionaryTy
loadArgumentMap(const ONNX_NAMESPACE::NodeProto &op) {
  ArgumentDictionaryTy dict;
  for (auto &arg : op.attribute()) {
    dict[arg.name()] = &arg;
  }
  return dict;
}

llvm::Error ONNXModelLoader::loadInputs(ONNX_NAMESPACE::GraphProto &net,
                                        bool loadInputsAsPlaceholders) {
  for (const auto &in : net.input()) {
    // Skip static weights.
    if (tensors_.count(in.name())) {
      continue;
    }

    if (loadInputsAsPlaceholders) {
      Tensor T;
      RETURN_IF_ERR(setTensorType(in.type(), &T));

      Placeholder *placeholder;
      ASSIGN_VALUE_OR_RETURN_ERR(
          placeholder, createAndRegisterPlaceholder(in.name(), &T.getType()));
      onnxNameToInputVars_.try_emplace(in.name(), placeholder);
    } else {
      std::unique_ptr<Tensor> T(new Tensor());
      RETURN_IF_ERR(setTensorType(in.type(), T.get()));
      tensors_[in.name()] = std::move(T);
    }
  }
  return llvm::Error::success();
}

llvm::Expected<bool>
ONNXModelLoader::getBroadcast(const ArgumentDictionaryTy &dict) {
  // Starting with opset 7, broadcasting is implicit and doesn't require any
  // attribute.
  if (opsetVersion_ > 6) {
    return true;
  }
  if (!dict.count("broadcast")) {
    return false;
  }

  int broadcast;
  ASSIGN_VALUE_OR_RETURN_ERR(broadcast, loadInt(dict.at("broadcast")));
  return broadcast == 1;
}

bool ONNXModelLoader::hasMultidirectionalBroadcast(
    const llvm::StringRef typeName) {
  // Before opset 7, broadcasting was unidirectional.
  if (opsetVersion_ > 6) {
    if ((typeName == "Add") || (typeName == "Sub") || (typeName == "Mul") ||
        (typeName == "Div")) {
      return true;
    }
    // TODO: some other operators also support multidirectional broadcasting.
  }
  return false;
}

llvm::Error ONNXModelLoader::setVersion(ONNX_NAMESPACE::ModelProto MP) {
  irVersion_ = MP.ir_version();
  opsetVersion_ = 0;
  RETURN_ERR_IF_NOT(
      irVersion_ >= 3,
      "This ONNX model with ir_version < 3 is too old to be supported.",
      GlowErr::EC::MODEL_LOADER_UNSUPPORTED_ONNX_VERSION);
  for (const auto &imp : MP.opset_import()) {
    if (!imp.has_domain() || imp.domain() == "") {
      opsetVersion_ = imp.version();
      break;
    }
  }
  RETURN_ERR_IF_NOT(opsetVersion_ > 0,
                    "The opset of this ONNX model is not supported.");
  return llvm::Error::success();
}

llvm::Expected<ONNX_NAMESPACE::ModelProto>
ONNXModelLoader::loadProto(google::protobuf::io::ZeroCopyInputStream &iStream) {
  // Construct and configure a Coded Input Stream
  google::protobuf::io::CodedInputStream codedStream(&iStream);

  // Don't warn about large file sizes.
  codedStream.SetTotalBytesLimit(MAX_PROTO_SIZE, MAX_PROTO_SIZE);
  ONNX_NAMESPACE::ModelProto MP;
  bool parseNet = MP.ParseFromCodedStream(&codedStream);
  RETURN_ERR_IF_NOT(parseNet, "Failed to parse ModelProto",
                    GlowErr::EC::MODEL_LOADER_INVALID_PROTOBUF);
  return MP;
}

llvm::Expected<ONNX_NAMESPACE::ModelProto>
ONNXModelLoader::loadProto(const void *onnxModel, size_t onnxModelSize) {
  google::protobuf::io::ArrayInputStream arrayStream(onnxModel, onnxModelSize);
  return loadProto(arrayStream);
}

llvm::Expected<ONNX_NAMESPACE::ModelProto>
ONNXModelLoader::loadProto(const std::string &filename) {
  std::ifstream ff(filename, std::ios::in | std::ios::binary);
  RETURN_ERR_IF_NOT(ff, "Can't find the model or network files.",
                    GlowErr::EC::MODEL_LOADER_INVALID_PROTOBUF);

  // TODO: intend to find a way to reuse the following function later
  // for the text format onnx model:
  // bool ONNXModelLoader::loadProto(ONNX_NAMESPACE::GraphProto &net,
  //  google::protobuf::io::ZeroCopyInputStream &iStream)
  if (filename.find(".onnxtxt") != std::string::npos) {
    std::string str((std::istreambuf_iterator<char>(ff)),
                    std::istreambuf_iterator<char>());
    ONNX_NAMESPACE::ModelProto MP;
    bool parseNet = google::protobuf::TextFormat::ParseFromString(str, &MP);

    RETURN_ERR_IF_NOT(parseNet, "Failed to parse ModelProto",
                      GlowErr::EC::MODEL_LOADER_INVALID_PROTOBUF);
    return MP;
  }

  google::protobuf::io::IstreamInputStream fileStream(&ff);
  return loadProto(fileStream);
}

namespace {
/// Helper type for pads.
using Pads = std::vector<unsigned_t>;
} // namespace

llvm::Expected<Pads> getPads(const ArgumentDictionaryTy &dict) {
  if (dict.count("pads")) {
    return getShape<unsigned_t>(dict.at("pads"));
  }
  if (dict.count("auto_pad")) {
    std::string padStr;
    ASSIGN_VALUE_OR_RETURN_ERR(padStr, loadStr(dict.at("auto_pad")));
    if (padStr == "VALID") {
      // Return default value 0 for pads.
      return Pads({0, 0, 0, 0});
    }
    RETURN_ERR("only auto_pad==VALID is supported");
  }
  // Return default value 0 for pads.
  return Pads({0, 0, 0, 0});
}

/// Loads tensor \p T from the input \p in.
static llvm::Error loadTensor(const ONNX_NAMESPACE::TensorProto &in,
                              Tensor *T) {
  std::vector<size_t> dim;
  for (auto d : in.dims()) {
    dim.push_back(d);
  }

  if (in.data_type() == ONNX_NAMESPACE::TensorProto::FLOAT) {
    T->reset(ElemKind::FloatTy, dim);

    if (in.float_data_size() > 0) {
      auto TH = T->getHandle<>();
      size_t i = 0;
      for (auto f : in.float_data()) {
        TH.raw(i++) = f;
      }
    } else if (in.has_raw_data()) {
      std::istringstream inStream(in.raw_data(), std::stringstream::binary);
      inStream.read(T->getUnsafePtr(), T->size() * sizeof(float));
    } else {
      RETURN_ERR("Unsupported Tensor format.",
                 GlowErr::EC::MODEL_LOADER_UNSUPPORTED_DATATYPE);
    }
  } else if (in.data_type() == ONNX_NAMESPACE::TensorProto::INT64) {
    T->reset(ElemKind::Int64ITy, dim);

    if (in.int64_data_size() > 0) {
      auto TH = T->getHandle<int64_t>();
      size_t i = 0;
      for (auto f : in.int64_data()) {
        TH.raw(i++) = f;
      }
    } else if (in.has_raw_data()) {
      std::istringstream inStream(in.raw_data(), std::stringstream::binary);
      inStream.read(T->getUnsafePtr(), T->size() * sizeof(int64_t));
    } else {
      RETURN_ERR("Unsupported Tensor format.",
                 GlowErr::EC::MODEL_LOADER_UNSUPPORTED_DATATYPE);
    }
  } else if (in.data_type() == ONNX_NAMESPACE::TensorProto::INT32) {
    // There are few cases when we will have int32 tensors. For example, the
    // second output of Concat from Caffe2 concat op is int32
    T->reset(ElemKind::Int32ITy, dim);

    if (in.int32_data_size() > 0) {
      auto TH = T->getHandle<int32_t>();
      size_t i = 0;
      for (auto f : in.int32_data()) {
        TH.raw(i++) = f;
      }
    } else if (in.has_raw_data()) {
      std::istringstream inStream(in.raw_data(), std::stringstream::binary);
      inStream.read(T->getUnsafePtr(), T->size() * sizeof(int32_t));
    } else {
      RETURN_ERR("Unsupported Tensor format.",
                 GlowErr::EC::MODEL_LOADER_UNSUPPORTED_DATATYPE);
    }
  } else {
    RETURN_ERR("Only float and index tensors are supported",
               GlowErr::EC::MODEL_LOADER_UNSUPPORTED_DATATYPE);
  }
  return llvm::Error::success();
}

llvm::Error ONNXModelLoader::loadConstant(const ONNX_NAMESPACE::NodeProto &op,
                                          const ArgumentDictionaryTy &dict) {
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
    return llvm::Error::success();
  }

  RETURN_ERR_IF_NOT(dict.at("value")->type() ==
                        ONNX_NAMESPACE::AttributeProto::TENSOR,
                    "Only Tensor type constants are supported.",
                    GlowErr::EC::MODEL_LOADER_UNSUPPORTED_DATATYPE);

  std::unique_ptr<Tensor> T(new Tensor());
  RETURN_IF_ERR(loadTensor(dict.at("value")->t(), T.get()));
  tensors_[name] = std::move(T);

  return llvm::Error::success();
}

llvm::Error ONNXModelLoader::loadSlice(const ONNX_NAMESPACE::NodeProto &op,
                                       const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);
  NodeValue data;
  ASSIGN_VALUE_OR_RETURN_ERR(data,
                             getNodeValueOrCreateConstantByName(op.input(0)));
  auto dims = data.dims();
  auto numDims = dims.size();

  // Attributes 'starts' and 'ends' are mandatory and must be consistent.
  RETURN_ERR_IF_NOT(dict.count("starts"),
                    "Slice: attribute 'starts' is mandatory.");
  RETURN_ERR_IF_NOT(dict.count("ends"),
                    "Slice: attribute 'ends' is mandatory.");
  auto starts = getShape<ssize_t>(dict.at("starts"));
  auto ends = getShape<ssize_t>(dict.at("ends"));
  RETURN_ERR_IF_NOT(
      (starts.size() == ends.size()),
      "Slice: 'starts' and 'ends' arrays must have the same size.");

  // Attribute 'axes' is optional.
  std::vector<ssize_t> axes;
  if (dict.count("axes")) {
    // The ONNX spec is unclear so we consider that the 'axes' array may have
    // any size. The constraints are:
    // - the element value must be in range [0, numDims),
    // - 'starts' & 'ends' arrays must have the same size as the 'axes' array.
    // In case an axis is specified multiple times in 'axes', the later
    // parameters will simply overwrite the previous ones.
    axes = getShape<ssize_t>(dict.at("axes"));
  } else {
    for (size_t i = 0; i < numDims; i++) {
      axes.push_back(ssize_t(i));
    }
  }

  // The ONNX description is unclear and doesn't describe what to do when a
  // an axis index is not given in the axes array. An interpretation is that
  // for such an axis, the entire range is taken. Then, we initialize
  // newStarts and newEnds with the full range for all axes.
  std::vector<size_t> newStarts(numDims);
  std::vector<size_t> newEnds(numDims);
  for (size_t i = 0; i < numDims; i++) {
    newStarts[i] = 0;
    newEnds[i] = dims[i];
  }

  // Determine the coordinates of the sub-tensor to extract.
  RETURN_ERR_IF_NOT(axes.size() == starts.size(),
                    "'axes' and 'starts' must be the same size.");
  RETURN_ERR_IF_NOT(starts.size() == ends.size(),
                    "'starts' and 'ends' must be the same size.");
  for (size_t i = 0; i < axes.size(); i++) {
    ssize_t newStart = starts[i];
    ssize_t newEnd = ends[i];
    ssize_t axisId = axes[i];
    RETURN_ERR_IF_NOT((axisId >= 0) && (axisId < ssize_t(numDims)),
                      "Axes indexes must be within the input tensor range.");

    // ONNX: "If the value passed to start or end is larger than the n (the
    // number of elements in this dimension), it represents n".
    if (newStart > ssize_t(dims[axisId])) {
      newStart = ssize_t(dims[axisId]);
    }
    if (newEnd > ssize_t(dims[axisId])) {
      newEnd = ssize_t(dims[axisId]);
    }

    // The ONNX description is unclear and the numpy definition is more
    // accurate.
    // - ONNX: "Similar to numpy. [...]. If a negative value is passed for any
    // of the start or end indices, it represent number of elements before the
    // end of that dimension."
    // - Numpy: "Negative indices are interpreted as counting from the end of
    // the array (i.e., if n_i < 0, it means n_i + d_i)."
    if (newStart < 0) {
      newStart = ssize_t(dims[axisId]) + newStart;
      RETURN_ERR_IF_NOT(newStart >= 0,
                        "Slice: final start index should never be negative.");
    }
    if (newEnd < 0) {
      newEnd = ssize_t(dims[axisId]) + newEnd;
      RETURN_ERR_IF_NOT(newEnd >= 0,
                        "Slice: final end index should never be negative.");
    }

    newStarts[axisId] = size_t(newStart);
    newEnds[axisId] = size_t(newEnd);
  }

  // Create the IR node.
  Node *SN = G_.createSlice(opName, data, newStarts, newEnds);
  addNodeAsOutput(op, SN);

  return llvm::Error::success();
}

llvm::Error ONNXModelLoader::loadConv(const ONNX_NAMESPACE::NodeProto &op,
                                      const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);
  // Load the attributes
  std::vector<unsigned_t> strides(2, 1);
  if (dict.count("strides")) {
    strides = getShape<unsigned_t>(dict.at("strides"));
  }
  unsigned_t group = 1;
  if (dict.count("group")) {
    ASSIGN_VALUE_OR_RETURN_ERR(group, loadInt(dict.at("group")));
  }

  // Pads : {pad_top, pad_left, pad_bottom, pad_right}
  Pads pads;
  ASSIGN_VALUE_OR_RETURN_ERR(pads, getPads(dict));

  // Load the inputs
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in,
                             getNodeValueOrCreateConstantByName(op.input(0)));
  NodeValue filterValue;
  ASSIGN_VALUE_OR_RETURN_ERR(filterValue,
                             getNodeValueOrCreateConstantByName(op.input(1)));

  // Transpose the filter to the right format. Glow expects to read the
  // weights in the format CRSK. ONNX stores the operators as KCRS.
  // C - output_depth, R - filter_height, S - filter_width, K - input_depth.
  TransposeNode *filterTransposeNode =
      G_.createTranspose(opName, filterValue, NCHW2NHWC);

  // The structure of the conv weigts is: NHWC. We take the C, which is the
  // number of filters. We use this value to calculate the size of the bias
  // if it is not specified.
  const NodeValue filterTransposedValue = filterTransposeNode->getResult();
  size_t depth = filterTransposedValue.dims()[0];

  // Get the kernel shape from the input.
  std::vector<unsigned_t> kernelShape(2);
  kernelShape[0] = filterTransposedValue.dims()[1];
  kernelShape[1] = filterTransposedValue.dims()[2];

  // Extra check when the 'kernel_shape' attribute exists.
  // The 'kernel_shape' attribute is redundant not mandatory.
  if (dict.count("kernel_shape")) {
    std::vector<unsigned_t> kernelShapeAttribute =
        getShape<unsigned_t>(dict.at("kernel_shape"));
    RETURN_ERR_IF_NOT(
        (kernelShape[0] == kernelShapeAttribute[0] &&
         kernelShape[1] == kernelShapeAttribute[1]),
        "The 'kernel_shape' attribute is not consistent with the actual "
        "convolution kernel shape.");
    (void)kernelShapeAttribute; // Avoids compilation warning in release mode.
  }

  // Construct the Bias field.
  Tensor biasTensor(ElemKind::FloatTy, {depth});
  biasTensor.zero();

  // Check if we have a serialized bias vector.
  if (op.input_size() > 2) {
    auto &biasTensorName = op.input(2);
    if (tensors_.count(biasTensorName)) {
      // Load the serialized bias vector.
      Tensor *b;
      ASSIGN_VALUE_OR_RETURN_ERR(b, getTensorByName(biasTensorName));
      biasTensor.assign(b);
    }
  }
  auto *bias = G_.getParent()->createConstant("conv.bias", biasTensor);

  // ONNX passes the input as NCHW, and we expect the input to be NHWC.
  auto *tr = G_.createTranspose(opName, in, NCHW2NHWC);

  // Calculate the size and allocate the output buffer.
  ShapeNHWC idim = ShapeNHWC(tr->getResult().dims());
  auto outSz =
      calculateConvPoolOutputDims(idim.h, idim.w, kernelShape, strides, pads);
  std::array<size_t, 4> outDims = {{idim.n, outSz.first, outSz.second, depth}};
  auto outTy = G_.getParent()->uniqueType(ElemKind::FloatTy, outDims);

  auto *node = G_.createConv(opName, tr, filterTransposeNode, bias, outTy,
                             kernelShape, strides, pads, group);

  // Transpose the output back.
  auto *N = G_.createTranspose(opName, node, NHWC2NCHW);
  addNodeAsOutput(op, N);

  return llvm::Error::success();
}

llvm::Error ONNXModelLoader::loadPool(const ONNX_NAMESPACE::NodeProto &op,
                                      const ArgumentDictionaryTy &dict,
                                      llvm::StringRef typeName) {
  const std::string &opName = loadOperatorName(op);

  // Glow doesn't support argmax output yet.
  if (op.output_size() > 1) {
    RETURN_ERR("Glow doesn't support argmax output yet.",
               GlowErr::EC::MODEL_LOADER_UNSUPPORTED_OPERATOR);
  }
  // Load the inputs:
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in,
                             getNodeValueOrCreateConstantByName(op.input(0)));
  std::vector<unsigned_t> strides(2, 1);
  if (dict.count("strides")) {
    strides = getShape<unsigned_t>(dict.at("strides"));
  }
  std::vector<unsigned_t> kernels =
      getShape<unsigned_t>(dict.at("kernel_shape"));

  Pads pads;
  ASSIGN_VALUE_OR_RETURN_ERR(pads, getPads(dict));

  if (in.dims().size() != 4 || kernels.size() != 2) {
    // Glow only handles 2D pooling currently.
    RETURN_ERR("Glow only handles 2D pooling currently.",
               GlowErr::EC::MODEL_LOADER_UNSUPPORTED_SHAPE);
  }

  auto *tr = G_.createTranspose(opName, in, NCHW2NHWC);

  // If 'global_pooling' is set then the operation will pool over the size of
  // the input by doing: kernel = height/width.
  if (dict.count("global_pooling")) {
    auto Ty = in.getType();
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
  return llvm::Error::success();
}

llvm::Error
ONNXModelLoader::loadGlobalAveragePool(const ONNX_NAMESPACE::NodeProto &op,
                                       const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  // Load the inputs:
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in,
                             getNodeValueOrCreateConstantByName(op.input(0)));
  std::vector<unsigned_t> strides(2, 1);
  if (dict.count("strides")) {
    strides = getShape<unsigned_t>(dict.at("strides"));
  }

  std::vector<unsigned_t> kernels(2);
  kernels[0] = in.dims()[2];
  kernels[1] = in.dims()[3];

  Pads pads;
  ASSIGN_VALUE_OR_RETURN_ERR(pads, getPads(dict));

  auto *tr = G_.createTranspose(opName, in, NCHW2NHWC);
  Node *node = G_.createAvgPool(opName, tr, kernels, strides, pads);
  auto *N = G_.createTranspose(opName, node, NHWC2NCHW);
  addNodeAsOutput(op, N);
  return llvm::Error::success();
}

llvm::Error ONNXModelLoader::loadSqueeze(const ONNX_NAMESPACE::NodeProto &op,
                                         const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in,
                             getNodeValueOrCreateConstantByName(op.input(0)));
  auto axes = getShape(dict.at("axes"));
  Node *node = G_.createSqueeze(opName, in, axes);
  addNodeAsOutput(op, node);
  return llvm::Error::success();
}

llvm::Error ONNXModelLoader::loadUnsqueeze(const ONNX_NAMESPACE::NodeProto &op,
                                           const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in,
                             getNodeValueOrCreateConstantByName(op.input(0)));
  auto axes = getShape(dict.at("axes"));
  Node *node = G_.createExpandDims(opName, in, axes);
  addNodeAsOutput(op, node);
  return llvm::Error::success();
}

llvm::Error
ONNXModelLoader::loadBatchNormalization(const ONNX_NAMESPACE::NodeProto &op,
                                        const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

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

  auto *scaleV = G_.getParent()->createConstant("scale", *scale);
  auto *biasV = G_.getParent()->createConstant("bias", *bias);
  auto *meanV = G_.getParent()->createConstant("mean", *mean);
  auto *varV = G_.getParent()->createConstant("var", *var);
  auto *node = G_.createBatchNormalization(opName, in, biasV, scaleV, meanV,
                                           varV, 1, epsilon);

  addNodeAsOutput(op, node);
  return llvm::Error::success();
}

llvm::Error ONNXModelLoader::loadConcat(const ONNX_NAMESPACE::NodeProto &op,
                                        const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  const unsigned numInputs = op.input_size();
  llvm::SmallVector<NodeValue, 4> inputs;
  inputs.reserve(numInputs);
  for (unsigned i = 0; i < numInputs; i++) {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(i)));
    inputs.push_back(in);
  }

  int axis;
  ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict.at("axis")));
  Node *node = G_.createConcat(opName, inputs, axis);

  addNodeAsOutput(op, node);
  return llvm::Error::success();
}

llvm::Error
ONNXModelLoader::loadFCTransposed(const ONNX_NAMESPACE::NodeProto &op,
                                  const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in,
                             getNodeValueOrCreateConstantByName(op.input(0)));
  if (in.getType()->dims().size() > 2) {
    size_t axis = 1;
    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict.at("axis")));
    }
    in = G_.createFlatten("fc.in", in, axis);
  }

  Tensor *w;
  ASSIGN_VALUE_OR_RETURN_ERR(w, getTensorByName(op.input(1)));
  Tensor *b;
  ASSIGN_VALUE_OR_RETURN_ERR(b, getTensorByName(op.input(2)));
  unsigned_t axis_w = 1;
  if (dict.count("axis_w")) {
    ASSIGN_VALUE_OR_RETURN_ERR(axis_w, loadInt(dict.at("axis_w")));
  }

  // w is stored already transposed. No need to additionally transpose it.
  Tensor tmp;
  if (w->dims().size() > 2) {
    auto wDims = flattenCdr(w->dims(), axis_w);
    tmp.reset(ElemKind::FloatTy, {wDims.first, wDims.second});
    tmp.copyRawFrom(w);
    w = &tmp;
  }

  auto W = G_.getParent()->addConstant(new Constant("weights", std::move(*w)));
  auto B = G_.getParent()->addConstant(new Constant("biases", std::move(*b)));
  auto *node = G_.createFullyConnected(opName, in, W, B);

  addNodeAsOutput(op, node);
  return llvm::Error::success();
}

llvm::Error ONNXModelLoader::loadGemm(const ONNX_NAMESPACE::NodeProto &op,
                                      const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue A;
  ASSIGN_VALUE_OR_RETURN_ERR(A,
                             getNodeValueOrCreateConstantByName(op.input(0)));
  NodeValue B;
  ASSIGN_VALUE_OR_RETURN_ERR(B,
                             getNodeValueOrCreateConstantByName(op.input(1)));
  NodeValue C;
  ASSIGN_VALUE_OR_RETURN_ERR(C,
                             getNodeValueOrCreateConstantByName(op.input(2)));

  bool broadcastC;
  ASSIGN_VALUE_OR_RETURN_ERR(broadcastC, getBroadcast(dict));
  bool transA = false;
  if (dict.count("transA")) {
    ASSIGN_VALUE_OR_RETURN_ERR(transA, loadInt(dict.at("transA")));
  }
  bool transB = false;
  if (dict.count("transB")) {
    ASSIGN_VALUE_OR_RETURN_ERR(transB, loadInt(dict.at("transB")));
  }
  // TODO: support alpha * A * B + beta * C

  if (transA)
    A = G_.createTranspose(opName, A, {1, 0});
  if (transB)
    B = G_.createTranspose(opName, B, {1, 0});

  MatMulNode *mul = G_.createMatMul(opName, A, B);
  if (broadcastC) {
    int axis = mul->getResult().dims().size() - C.dims().size();
    C = G_.createBroadcast(opName, C, mul->getResult().dims(), axis);
  }

  Node *node = G_.createAdd(opName, mul, C);
  addNodeAsOutput(op, node);
  return llvm::Error::success();
}

llvm::Error ONNXModelLoader::loadMatMul(const ONNX_NAMESPACE::NodeProto &op,
                                        const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue LHS;
  ASSIGN_VALUE_OR_RETURN_ERR(LHS,
                             getNodeValueOrCreateConstantByName(op.input(0)));
  NodeValue RHS;
  ASSIGN_VALUE_OR_RETURN_ERR(RHS,
                             getNodeValueOrCreateConstantByName(op.input(1)));

  Node *node = G_.createMatMul(opName, LHS, RHS);
  addNodeAsOutput(op, node);
  return llvm::Error::success();
}

llvm::Error ONNXModelLoader::loadPad(const ONNX_NAMESPACE::NodeProto &op,
                                     const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  // Input
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input,
                             getNodeValueOrCreateConstantByName(op.input(0)));
  auto inputDims = input.dims();
  auto numDims = inputDims.size();

  // Padding properties.
  unsigned_t mode = PaddingMode::CONSTANT; // default is constant.
  if (dict.count("mode")) {
    std::string modeStr;
    ASSIGN_VALUE_OR_RETURN_ERR(modeStr, loadStr(dict.at("mode")));
    if (modeStr == "constant") {
      mode = PaddingMode::CONSTANT;
    } else if (modeStr == "reflect") {
      mode = PaddingMode::REFLECT;
    } else if (modeStr == "edge") {
      mode = PaddingMode::EDGE;
    } else {
      RETURN_ERR("Pad: Invalid mode",
                 GlowErr::EC::MODEL_LOADER_UNSUPPORTED_ATTRIBUTE);
    }
  }
  float value = 0.f; // Default
  if (dict.count("value")) {
    ASSIGN_VALUE_OR_RETURN_ERR(value, loadFloat(dict.at("value")));
  }

  // Pads are mandatory.
  RETURN_ERR_IF_NOT(dict.count("pads"),
                    "Pad: The 'pads' property is mandatory");
  auto pads = getShape<int>(dict.at("pads"));
  RETURN_ERR_IF_NOT(
      (pads.size() == 2 * numDims),
      "Pad: the 'pads' array must contain 2 values per dimensions");

  // Compute the output type.
  std::vector<size_t> outDims(numDims);
  for (unsigned_t i = 0; i < numDims; i++) {
    auto new_dim = inputDims[i] + pads[i] + pads[i + numDims];
    RETURN_ERR_IF_NOT(new_dim > 0,
                      "The padding can't remove all elements of a dimension");
    outDims[i] = new_dim;
  }
  auto outTy = G_.getParent()->uniqueType(ElemKind::FloatTy, outDims);

  // Create the IR node.
  Node *N = G_.createPad(opName, input, outTy, mode, pads, value);
  addNodeAsOutput(op, N);

  return llvm::Error::success();
}

llvm::Error ONNXModelLoader::loadOperator(const ONNX_NAMESPACE::NodeProto &op) {
  ArgumentDictionaryTy dict = loadArgumentMap(op);
  const std::string &typeName = op.op_type();

  // Check if operator is supported in parent class, CommonOperatorLoader.
  bool tryLoadCommonOperatorResult;
  ASSIGN_VALUE_OR_RETURN_ERR(tryLoadCommonOperatorResult,
                             tryLoadCommonOperator(typeName, op, dict));
  if (tryLoadCommonOperatorResult) {
    return llvm::Error::success();
  }

  if (typeName == "Constant") {
    return loadConstant(op, dict);
  }
  if (typeName == "Slice") {
    return loadSlice(op, dict);
  }
  if (typeName == "Conv") {
    return loadConv(op, dict);
  }
  if (typeName == "MaxPool" || typeName == "AveragePool") {
    return loadPool(op, dict, typeName);
  }
  if (typeName == "GlobalAveragePool") {
    return loadGlobalAveragePool(op, dict);
  }
  if (typeName == "Squeeze") {
    return loadSqueeze(op, dict);
  }
  if (typeName == "Unsqueeze") {
    return loadUnsqueeze(op, dict);
  }
  if (typeName == "BatchNormalization") {
    return loadBatchNormalization(op, dict);
  }
  if (typeName == "Concat") {
    return loadConcat(op, dict);
  }
  if (typeName == "FCTransposed") {
    return loadFCTransposed(op, dict);
  }
  if (typeName == "Gemm") {
    return loadGemm(op, dict);
  }
  if (typeName == "Transpose") {
    return loadTranspose(op, dict, "perm");
  }
  if (typeName == "MatMul") {
    return loadMatMul(op, dict);
  }
  if (typeName == "Pad") {
    return loadPad(op, dict);
  }

  RETURN_ERR("Failed to load operator.",
             GlowErr::EC::MODEL_LOADER_UNSUPPORTED_OPERATOR);
}

llvm::Error ONNXModelLoader::loadInitializers(ONNX_NAMESPACE::GraphProto &net) {
  // Load the network initializaers:
  for (const auto &in : net.initializer()) {
    std::unique_ptr<Tensor> T(new Tensor());
    RETURN_IF_ERR(loadTensor(in, T.get()));
    tensors_[in.name()] = std::move(T);
  }
  return llvm::Error::success();
}

llvm::Error ONNXModelLoader::setOutputNodes(ONNX_NAMESPACE::GraphProto &net) {
  if (net.output_size() == 0) {
    RETURN_ERR("Net output size must be greater than 0");
  }

  for (int i = 0; i < net.output_size(); i++) {
    const auto &outputName = net.output(i).name();
    NodeValue r;
    ASSIGN_VALUE_OR_RETURN_ERR(r,
                               getNodeValueOrCreateConstantByName(outputName));
    SaveNode *SN = G_.createSave("save_" + outputName, r);
    outputVarsByName_[outputName] = SN->getPlaceholder();
  }

  return llvm::Error::success();
}

llvm::Error ONNXModelLoader::loadNetwork(ONNX_NAMESPACE::GraphProto &net) {
  /// Load the network operators:
  for (int i = 0; i < net.node_size(); i++) {
    auto &op = net.node(i);
    RETURN_IF_ERR(loadOperator(op));
  }

  return llvm::Error::success();
}

ONNXModelLoader::ONNXModelLoader(Function &F, llvm::Error *errPtr)
    : CommonOperatorLoader({}, {}, F, errPtr) {}

llvm::Error
ONNXModelLoader::checkInputs(ONNX_NAMESPACE::GraphProto &net,
                             llvm::ArrayRef<const char *> tensorNames,
                             llvm::ArrayRef<TypeRef> types) {
  for (size_t i = 0; i < tensorNames.size(); i++) {
    // Look if a corresponding input exists.
    for (int j = 0; j < net.input_size(); j++) {
      const ONNX_NAMESPACE::ValueInfoProto &valueInfo = net.input(j);
      const std::string &inputName = valueInfo.name();

      if (inputName != tensorNames[i]) {
        continue;
      }

      llvm::ArrayRef<size_t> dims = types[i]->dims();
      const ONNX_NAMESPACE::TensorShapeProto &shape =
          valueInfo.type().tensor_type().shape();
      (void)shape;

      // Check if the number of dimensions is consistent.
      RETURN_ERR_IF_NOT(dims.size() == (size_t)shape.dim_size(),
                        "Mismatch between input image and ONNX input shape");
      // Allow batch dimensions to be different.
      for (size_t k = 1; k < dims.size(); k++) {
        RETURN_ERR_IF_NOT(dims[k] == (size_t)shape.dim(k).dim_value(),
                          "Mismatch between input image and ONNX input shape");
      }
    }
  }
  return llvm::Error::success();
}

ONNXModelLoader::ONNXModelLoader(const std::string &modelDescFilename,
                                 llvm::ArrayRef<const char *> tensorNames,
                                 llvm::ArrayRef<TypeRef> types, Function &F,
                                 llvm::Error *errPtr)
    : CommonOperatorLoader(tensorNames, types, F, errPtr) {
  // if errPtr already contains an error then don't continue with constructor
  if (errPtr && *errPtr) {
    return;
  }

  // Lambda to setup the ONNXModelLoader and return any llvm::Errors that were
  // raised.
  auto setup = [&]() -> llvm::Error {
    // The ONNX model that we are deserializing.
    ONNX_NAMESPACE::ModelProto modelDef;
    ASSIGN_VALUE_OR_RETURN_ERR(modelDef, loadProto(modelDescFilename));

    RETURN_IF_ERR(setVersion(modelDef));

    ONNX_NAMESPACE::GraphProto graphDef = modelDef.graph();
    RETURN_IF_ERR(checkInputs(graphDef, tensorNames, types));

    RETURN_IF_ERR(loadInitializers(graphDef));
    RETURN_IF_ERR(loadNetwork(graphDef));

    RETURN_IF_ERR(setOutputNodes(graphDef));

    RETURN_ERR_IF_NOT(F.verify(), "Function verification failed.");

    return llvm::Error::success();
  };

  if (errPtr) {
    *errPtr = setup();
  } else {
    EXIT_ON_ERR(setup());
  }
}
