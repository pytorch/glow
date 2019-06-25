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
    if (getConstantByNameOrNull(in.name())) {
      continue;
    }

    if (loadInputsAsPlaceholders) {
      Tensor T;
      RETURN_IF_ERR(setTensorType(in.type(), &T));

      Placeholder *placeholder;
      ASSIGN_VALUE_OR_RETURN_ERR(
          placeholder, createAndRegisterPlaceholder(in.name(), &T.getType()));
      inputVarsByName_.try_emplace(in.name(), placeholder);
    } else {
      Tensor T;
      RETURN_IF_ERR(setTensorType(in.type(), &T));
      RETURN_IF_ERR(createAndRegisterConstant(in.name(), std::move(T)));
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

llvm::Expected<ElemKind> ONNXModelLoader::convertTensorProtoDataType(
    ONNX_NAMESPACE::TensorProto_DataType t) {
  switch (t) {
  case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
    return ElemKind::FloatTy;
  case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
    return ElemKind::Float16Ty;
  case ONNX_NAMESPACE::TensorProto_DataType_INT32:
    return ElemKind::Int32ITy;
  case ONNX_NAMESPACE::TensorProto_DataType_INT64:
    return ElemKind::Int64ITy;
  default:;
  }
  RETURN_ERR("Non supported ONNX type");
}

llvm::Error ONNXModelLoader::setVersion(ONNX_NAMESPACE::ModelProto MP) {
  irVersion_ = MP.ir_version();
  opsetVersion_ = 0;
  RETURN_ERR_IF_NOT(
      irVersion_ >= 3,
      "This ONNX model with ir_version < 3 is too old to be supported.",
      GlowErr::ErrorCode::MODEL_LOADER_UNSUPPORTED_ONNX_VERSION);
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
                    GlowErr::ErrorCode::MODEL_LOADER_INVALID_PROTOBUF);
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
                    GlowErr::ErrorCode::MODEL_LOADER_INVALID_PROTOBUF);

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
                      GlowErr::ErrorCode::MODEL_LOADER_INVALID_PROTOBUF);
    return MP;
  }

  google::protobuf::io::IstreamInputStream fileStream(&ff);
  return loadProto(fileStream);
}

namespace {
/// Helper type for pads.
using Pads = std::vector<unsigned_t>;
} // namespace

/// Get the Pads value based on setting for auto_pad.
/// \p kdim : kernel sizes (HW)
/// \p sdim: stride sizes (HW)
/// \p idim: input sizes (HW)
llvm::Expected<Pads> getPads(const ArgumentDictionaryTy &dict,
                             llvm::ArrayRef<unsigned_t> kdim,
                             llvm::ArrayRef<unsigned_t> sdim,
                             llvm::ArrayRef<unsigned_t> idim) {
  if (dict.count("pads")) {
    return getShape<unsigned_t>(dict.at("pads"));
  }
  if (dict.count("auto_pad")) {
    std::string padStr;
    ASSIGN_VALUE_OR_RETURN_ERR(padStr, loadStr(dict.at("auto_pad")));
    if (padStr == "VALID") {
      // Return default value 0 for pads.
      return Pads({0, 0, 0, 0});
    } else if (padStr == "SAME_UPPER" || padStr == "SAME_LOWER") {
      unsigned_t top, left, bottom, right;
      // From https://arxiv.org/pdf/1603.07285.pdf 2.4,
      // o = floor((i + 2*p - k)/s) + 1
      // Also, from https://github.com/onnx/onnx/blob/master/docs/Operators.md
      // output_spatial_shape[i] =
      //     ceil(input_spatial_shape[i] / strides_spatial_shape[i])
      // pad_shape[i] =
      //     (output_spatial_shape[i] - 1) * strides_spatial_shape[i]
      //         + kernel_spatial_shape[i] - input_spatial_shape[i]
      // Use the smallest padding possible out of the possible options.
      llvm::SmallVector<unsigned_t, 2> pdim(2); // Total Paddding, HW.
      for (size_t i = 0, e = pdim.size(); i < e; i++) {
        pdim[i] = sdim[i] * (idim[i] - 1) + kdim[i] - idim[i];
      }
      if (padStr == "SAME_UPPER") {
        // SAME_UPPPER: if odd number for pdim[i], use extra padding at the end.
        top = pdim[0] / 2;
        bottom = top + (pdim[0] & 0x1);
        left = pdim[1] / 2;
        right = left + (pdim[1] & 0x1);
      } else {
        // SAME_LOWER: if odd number for pdim[i], use extra padding at the
        // beginning.
        bottom = pdim[0] / 2;
        top = bottom + (pdim[0] & 0x1);
        right = pdim[1] / 2;
        left = right + (pdim[1] & 0x1);
      }
      return Pads({top, left, bottom, right});
    }
    RETURN_ERR("only auto_pad==VALID, SAME_UPPER and SAME_LOWER are supported");
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
                 GlowErr::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);
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
                 GlowErr::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);
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
                 GlowErr::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);
    }
  } else if (in.data_type() == ONNX_NAMESPACE::TensorProto::BOOL) {
    T->reset(ElemKind::BoolTy, dim);
    if (in.has_raw_data()) {
      std::istringstream inStream(in.raw_data(), std::stringstream::binary);
      inStream.read(T->getUnsafePtr(), T->size() * sizeof(bool));
    } else {
      RETURN_ERR("Unsupported Tensor format.",
                 GlowErr::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);
    }
  } else {
    RETURN_ERR("Only float and index tensors are supported",
               GlowErr::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);
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
  if (getConstantByNameOrNull(name)) {
    return llvm::Error::success();
  }

  RETURN_ERR_IF_NOT(dict.at("value")->type() ==
                        ONNX_NAMESPACE::AttributeProto::TENSOR,
                    "Only Tensor type constants are supported.",
                    GlowErr::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);

  Tensor T;
  RETURN_IF_ERR(loadTensor(dict.at("value")->t(), &T));
  RETURN_IF_ERR(createAndRegisterConstant(name, std::move(T)));

  return llvm::Error::success();
}

/// Retrieves data from a constant Tensor and stores it in a vector.
template <typename T>
static void helperSetter(Constant *constT, std::vector<ssize_t> &vec) {
  auto constH = constT->getPayload().getHandle<T>();
  for (size_t i = 0; i < constH.size(); ++i) {
    vec.push_back(constH.at({i}));
  }
}

llvm::Error ONNXModelLoader::loadSlice(const ONNX_NAMESPACE::NodeProto &op,
                                       const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);
  NodeValue data;
  ASSIGN_VALUE_OR_RETURN_ERR(data, getNodeValueByName(op.input(0)));
  auto dims = data.dims();
  auto numDims = dims.size();

  std::vector<ssize_t> starts;
  std::vector<ssize_t> ends;
  // Attribute 'axes' is optional.
  std::vector<ssize_t> axes;
  if (this->opsetVersion_ >= 10) {
    Constant *startsC = getConstantByNameOrNull(op.input(1));
    Constant *endsC = getConstantByNameOrNull(op.input(2));

    RETURN_ERR_IF_NOT(startsC, "Starts Tensor is not Constant.");
    RETURN_ERR_IF_NOT(endsC, "Ends Tensor is not Constant.");

    if (startsC->getElementType() == ElemKind::Int64ITy) {
      helperSetter<int64_t>(startsC, starts);
    } else if (startsC->getElementType() == ElemKind::Int32ITy) {
      helperSetter<int32_t>(startsC, starts);
    } else {
      RETURN_ERR_IF_NOT(false, "Starts Tensor has unsupported type.");
    }

    if (endsC->getElementType() == ElemKind::Int64ITy) {
      helperSetter<int64_t>(endsC, ends);
    } else if (endsC->getElementType() == ElemKind::Int32ITy) {
      helperSetter<int32_t>(endsC, ends);
    } else {
      RETURN_ERR_IF_NOT(false, "Ends Tensor has unsupported type.");
    }

    if (op.input_size() > 3) {
      Constant *axesC = getConstantByNameOrNull(op.input(3));

      RETURN_ERR_IF_NOT(startsC, "Axes Tensor is not Constant.");

      if (axesC->getElementType() == ElemKind::Int64ITy) {
        helperSetter<int64_t>(axesC, axes);
      } else if (axesC->getElementType() == ElemKind::Int32ITy) {
        helperSetter<int32_t>(axesC, axes);
      } else {
        RETURN_ERR_IF_NOT(false, "Axes Tensor has unsupported type.");
      }

      RETURN_ERR_IF_NOT(op.input_size() == 5,
                        "Steps is not currently supported.");
    }
  } else {
    // Attributes 'starts' and 'ends' are mandatory and must be consistent.
    RETURN_ERR_IF_NOT(dict.count("starts"),
                      "Slice: attribute 'starts' is mandatory.");
    RETURN_ERR_IF_NOT(dict.count("ends"),
                      "Slice: attribute 'ends' is mandatory.");
    starts = getShape<ssize_t>(dict.at("starts"));
    ends = getShape<ssize_t>(dict.at("ends"));

    if (dict.count("axes")) {
      // The ONNX spec is unclear so we consider that the 'axes' array may have
      // any size. The constraints are:
      // - the element value must be in range [0, numDims),
      // - 'starts' & 'ends' arrays must have the same size as the 'axes' array.
      // In case an axis is specified multiple times in 'axes', the later
      // parameters will simply overwrite the previous ones.
      axes = getShape<ssize_t>(dict.at("axes"));
    }
  }
  RETURN_ERR_IF_NOT(
      (starts.size() == ends.size()),
      "Slice: 'starts' and 'ends' arrays must have the same size.");

  if (axes.empty()) {
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
  RETURN_IF_ERR(addNodeAsOutput(op, SN));

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

  unsigned_t dilation = 1;
  if (dict.count("dilations")) {
    std::vector<unsigned_t> dilations(2, 1);
    dilations = getShape<unsigned_t>(dict.at("dilations"));
    RETURN_ERR_IF_NOT(dilations.size() == 2,
                      "Conv: dilations must be specified for 2 axes.");
    RETURN_ERR_IF_NOT(dilations[1] == dilations[0],
                      "Conv: different dilation values along different axes "
                      "are not supported currently. values must be same.");
    dilation = dilations[0];
  }

  // Load the inputs
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  NodeValue filterValue;
  ASSIGN_VALUE_OR_RETURN_ERR(filterValue, getNodeValueByName(op.input(1)));

  // Transpose the filter to the right format. Glow expects to read the
  // weights in the format CRSK. ONNX stores the operators as KCRS.
  // C - output_depth, R - filter_height, S - filter_width, K - input_depth.
  TransposeNode *filterTransposeNode =
      G_.createTranspose(opName, filterValue, NCHW2NHWC);

  // The structure of the conv weights is: CRSK. We take the C, which is the
  // number of filters. We use this value to calculate the size of the bias
  // if it is not specified.
  const NodeValue filterTransposedValue = filterTransposeNode->getResult();
  size_t depth = filterTransposedValue.dims()[0];

  // Get the kernel shape from the input.
  llvm::SmallVector<unsigned_t, 2> kernelShape(2);
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
  Constant *bias = nullptr;

  // Check if we have a serialized bias vector.
  if (op.input_size() > 2) {
    auto &biasTensorName = op.input(2);
    // Load the serialized bias vector.
    bias = getConstantByNameOrNull(biasTensorName);
  }

  // If a serialized bias wasn't found then create a zero bias.
  if (!bias) {
    Tensor biasTensor(ElemKind::FloatTy, {depth});
    biasTensor.zero();
    bias = G_.getParent()->createConstant("conv.bias", std::move(biasTensor));
  }

  // ONNX passes the input as NCHW, and we expect the input to be NHWC.
  auto *tr = G_.createTranspose(opName, in, NCHW2NHWC);

  // Calculate the size and allocate the output buffer.
  ShapeNHWC idim = ShapeNHWC(tr->getResult().dims());

  llvm::SmallVector<unsigned_t, 2> idimHW(2);
  idimHW[0] = in.dims()[2];
  idimHW[1] = in.dims()[3];

  // Pads : {pad_top, pad_left, pad_bottom, pad_right}
  Pads pads;
  ASSIGN_VALUE_OR_RETURN_ERR(pads, getPads(dict, kernelShape, strides, idimHW));

  auto outSz = calculateConvPoolOutputDims(idim.h, idim.w, kernelShape, strides,
                                           pads, dilation);
  std::array<size_t, 4> outDims = {{idim.n, outSz.first, outSz.second, depth}};
  auto outTy = G_.getParent()->uniqueType(ElemKind::FloatTy, outDims);

  auto *node = G_.createConv(opName, tr, filterTransposeNode, bias, outTy,
                             kernelShape, strides, pads, group, dilation);

  // Transpose the output back.
  auto *N = G_.createTranspose(opName, node, NHWC2NCHW);
  RETURN_IF_ERR(addNodeAsOutput(op, N));

  return llvm::Error::success();
}

llvm::Error ONNXModelLoader::loadPool(const ONNX_NAMESPACE::NodeProto &op,
                                      const ArgumentDictionaryTy &dict,
                                      llvm::StringRef typeName) {
  const std::string &opName = loadOperatorName(op);

  // Glow doesn't support argmax output yet.
  if (op.output_size() > 1) {
    RETURN_ERR("Glow doesn't support argmax output yet.",
               GlowErr::ErrorCode::MODEL_LOADER_UNSUPPORTED_OPERATOR);
  }
  // Load the inputs:
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  std::vector<unsigned_t> strides(2, 1);
  if (dict.count("strides")) {
    strides = getShape<unsigned_t>(dict.at("strides"));
  }
  std::vector<unsigned_t> kernels =
      getShape<unsigned_t>(dict.at("kernel_shape"));

  if (in.dims().size() != 4 || kernels.size() != 2) {
    // Glow only handles 2D pooling currently.
    RETURN_ERR("Glow only handles 2D pooling currently.",
               GlowErr::ErrorCode::MODEL_LOADER_UNSUPPORTED_SHAPE);
  }

  auto *tr = G_.createTranspose(opName, in, NCHW2NHWC);

  // If 'global_pooling' is set then the operation will pool over the size of
  // the input by doing: kernel = height/width.
  if (dict.count("global_pooling")) {
    auto Ty = in.getType();
    kernels[0] = Ty->dims()[2];
    kernels[1] = Ty->dims()[3];
  }

  // NHWC
  llvm::SmallVector<unsigned_t, 2> idimHW(2);
  idimHW[0] = in.dims()[1];
  idimHW[1] = in.dims()[2];

  Pads pads;
  ASSIGN_VALUE_OR_RETURN_ERR(pads, getPads(dict, kernels, strides, idimHW));

  Node *node = nullptr;
  if (typeName == "MaxPool") {
    node = G_.createMaxPool(opName, tr, kernels, strides, pads);
  } else {
    node = G_.createAvgPool(opName, tr, kernels, strides, pads);
  }
  auto *N = G_.createTranspose(opName, node, NHWC2NCHW);
  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return llvm::Error::success();
}

llvm::Error
ONNXModelLoader::loadGlobalAveragePool(const ONNX_NAMESPACE::NodeProto &op,
                                       const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  // Load the inputs:
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  std::vector<unsigned_t> strides(2, 1);
  if (dict.count("strides")) {
    strides = getShape<unsigned_t>(dict.at("strides"));
  }

  llvm::SmallVector<unsigned_t, 2> kernels(2);
  kernels[0] = in.dims()[2];
  kernels[1] = in.dims()[3];

  Pads pads;
  ASSIGN_VALUE_OR_RETURN_ERR(
      pads, getPads(dict, kernels, strides, kernels /* input sizes*/));

  auto *tr = G_.createTranspose(opName, in, NCHW2NHWC);
  Node *node = G_.createAvgPool(opName, tr, kernels, strides, pads);
  auto *N = G_.createTranspose(opName, node, NHWC2NCHW);
  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return llvm::Error::success();
}

llvm::Error ONNXModelLoader::loadSqueeze(const ONNX_NAMESPACE::NodeProto &op,
                                         const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  auto axes = getShape(dict.at("axes"));
  Node *node = G_.createSqueeze(opName, in, axes);
  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return llvm::Error::success();
}

llvm::Error ONNXModelLoader::loadUnsqueeze(const ONNX_NAMESPACE::NodeProto &op,
                                           const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  auto axes = getShape(dict.at("axes"));
  Node *node = G_.createExpandDims(opName, in, axes);
  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return llvm::Error::success();
}

llvm::Error
ONNXModelLoader::loadBatchNormalization(const ONNX_NAMESPACE::NodeProto &op,
                                        const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

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

  auto *node = G_.createBatchNormalization(opName, in, bias, scale, mean, var,
                                           1, epsilon);

  // BatchNormalization has 4 optional outputs that are not supported by glow.
  // Then: 1/ In case the optional outputs are present and used by other
  // operations of the model, then the import should fail. 2/ In case the
  // optional outputs are declared but not used, the import should succeed. By
  // registering only the mandatory output, we make sure the import will fail if
  // the non supported features are actually requested by the ONNX model.
  RETURN_IF_ERR(addNodeAsOutput(op, node, 1));

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
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(i)));
    inputs.push_back(in);
  }

  int axis;
  ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict.at("axis")));
  Node *node = G_.createConcat(opName, inputs, axis);

  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return llvm::Error::success();
}

llvm::Error
ONNXModelLoader::loadFCTransposed(const ONNX_NAMESPACE::NodeProto &op,
                                  const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  if (in.getType()->dims().size() > 2) {
    size_t axis = 1;
    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict.at("axis")));
    }
    in = G_.createFlatten("fc.in", in, axis);
  }

  unsigned_t axis_w = 1;
  if (dict.count("axis_w")) {
    ASSIGN_VALUE_OR_RETURN_ERR(axis_w, loadInt(dict.at("axis_w")));
  }

  Constant *W;
  ASSIGN_VALUE_OR_RETURN_ERR(W, getConstantByName(op.input(1)));

  // w is stored already transposed. No need to additionally transpose it.
  if (W->dims().size() > 2) {
    Tensor tmp;
    auto wDims = flattenCdr(W->dims(), axis_w);
    tmp.reset(ElemKind::FloatTy, {wDims.first, wDims.second});
    tmp.copyRawFrom(&W->getPayload());
    W = G_.getParent()->createConstant(W->getName(), tmp);
  }

  Constant *B;
  ASSIGN_VALUE_OR_RETURN_ERR(B, getConstantByName(op.input(2)));

  auto *node = G_.createFullyConnected(opName, in, W, B);

  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return llvm::Error::success();
}

llvm::Error ONNXModelLoader::loadGemm(const ONNX_NAMESPACE::NodeProto &op,
                                      const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue A;
  ASSIGN_VALUE_OR_RETURN_ERR(A, getNodeValueByName(op.input(0)));
  NodeValue B;
  ASSIGN_VALUE_OR_RETURN_ERR(B, getNodeValueByName(op.input(1)));
  NodeValue C;
  ASSIGN_VALUE_OR_RETURN_ERR(C, getNodeValueByName(op.input(2)));

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
  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return llvm::Error::success();
}

llvm::Error ONNXModelLoader::loadMatMul(const ONNX_NAMESPACE::NodeProto &op,
                                        const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue LHS;
  ASSIGN_VALUE_OR_RETURN_ERR(LHS, getNodeValueByName(op.input(0)));
  NodeValue RHS;
  ASSIGN_VALUE_OR_RETURN_ERR(RHS, getNodeValueByName(op.input(1)));

  Node *node = G_.createMatMul(opName, LHS, RHS);
  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return llvm::Error::success();
}

llvm::Error ONNXModelLoader::loadLeakyRelu(const ONNX_NAMESPACE::NodeProto &op,
                                           const ArgumentDictionaryTy &dict) {
  // Input Type.
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));
  ElemKind inputType = input.getType()->getElementType();

  // Only supports float types.
  RETURN_ERR_IF_NOT((inputType == ElemKind::FloatTy) ||
                        (inputType == ElemKind::Float16Ty),
                    "Unsupported Type for LeakyRelu");

  // ONNX spec says default is 0.01, but doesn't explicitly say it's optional.
  // like for others. The default example just omits alpha.
  float alphaVal = 0.01f;
  if (dict.count("alpha")) {
    ASSIGN_VALUE_OR_RETURN_ERR(alphaVal, loadFloat(dict.at("alpha")));
  }

  // Create the node.
  auto splatType = G_.getParent()->uniqueType(ElemKind::FloatTy, input.dims());
  const std::string &opName = loadOperatorName(op);
  Node *splatN = G_.createSplat(opName + "Alpha", splatType, alphaVal);
  Node *N = G_.createPRELU(opName, input, splatN);

  RETURN_IF_ERR(addNodeAsOutput(op, N));

  return llvm::Error::success();
}

llvm::Error ONNXModelLoader::loadPad(const ONNX_NAMESPACE::NodeProto &op,
                                     const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  // Input
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));
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
                 GlowErr::ErrorCode::MODEL_LOADER_UNSUPPORTED_ATTRIBUTE);
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
  RETURN_IF_ERR(addNodeAsOutput(op, N));

  return llvm::Error::success();
}

llvm::Error ONNXModelLoader::loadCast(const ONNX_NAMESPACE::NodeProto &op,
                                      const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  // Input type
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));
  ElemKind inputKind = input.getType()->getElementType();

  // Target type.
  ElemKind targetKind;
  RETURN_ERR_IF_NOT(dict.count("to"), "Cast: missing 'to' attribute");
  int toONNXTypeValue;
  ASSIGN_VALUE_OR_RETURN_ERR(toONNXTypeValue, loadInt(dict.at("to")));
  RETURN_ERR_IF_NOT(
      ONNX_NAMESPACE::TensorProto_DataType_IsValid(toONNXTypeValue),
      "Cast: invalid target type",
      GlowErr::ErrorCode::MODEL_LOADER_INVALID_PROTOBUF);
  ASSIGN_VALUE_OR_RETURN_ERR(
      targetKind, convertTensorProtoDataType(
                      ONNX_NAMESPACE::TensorProto_DataType(toONNXTypeValue)));

  // Only support non quantized types.
  RETURN_ERR_IF_NOT((!isQuantizedElemKind(inputKind)) &&
                        (!isQuantizedElemKind(targetKind)),
                    "Unsupported Cast");

  // Create the node.
  auto inputDims = input.dims();
  auto outTy = G_.getParent()->uniqueType(targetKind, inputDims);

  // Create the IR node.
  Node *N = G_.createConvertTo(opName, input, outTy);
  RETURN_IF_ERR(addNodeAsOutput(op, N));

  return llvm::Error::success();
}

llvm::Error
ONNXModelLoader::loadSpaceToDepth(const ONNX_NAMESPACE::NodeProto &op,
                                  const ArgumentDictionaryTy &dict) {

  // Input Type
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));

  int blockSize = 0;
  if (dict.count("blocksize")) {
    ASSIGN_VALUE_OR_RETURN_ERR(blockSize, loadInt(dict.at("blocksize")));
  } else {
    RETURN_ERR("SpaceToDepth: missing 'blocksize' attribute");
  }

  // Create the node.
  std::string opName = loadOperatorName(op);
  auto *tr = G_.createTranspose(opName, input, NCHW2NHWC);
  Node *nd = G_.createSpaceToDepth(opName, tr, blockSize);
  auto *N = G_.createTranspose(opName, nd, NHWC2NCHW);

  RETURN_IF_ERR(addNodeAsOutput(op, N));

  return llvm::Error::success();
}

llvm::Error
ONNXModelLoader::loadConstantOfShape(const ONNX_NAMESPACE::NodeProto &op,
                                     const ArgumentDictionaryTy &dict) {
  Tensor T(ElemKind::FloatTy, {1});
  T.getHandle().raw(0) = 0.0;

  if (dict.count("value")) {
    RETURN_IF_ERR(loadTensor(dict.at("value")->t(), &T));
    RETURN_ERR_IF_NOT(T.dims().size() == 1, "Value must be a 1D vector.");
    RETURN_ERR_IF_NOT(T.getType().getElementType() == ElemKind::FloatTy ||
                          T.getType().getElementType() == ElemKind::Int64ITy ||
                          T.getType().getElementType() == ElemKind::Int32ITy,
                      T.getType().getElementName().str() +
                          " type Value is not supported.");
  }

  TypeRef ty;
  Node *SN = nullptr;
  if (op.input_size() > 0) {
    Constant *in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getConstantByName(op.input(0)));
    // Must be 1D tensor of int64_t.
    RETURN_ERR_IF_NOT(in->dims().size() == 1, "Input must be a 1D vector.");
    RETURN_ERR_IF_NOT(in->getType()->getElementType() == ElemKind::Int64ITy,
                      "Input element type must be Int64ITy.");
    // Convert 1D tensor of int64_t into llvm::ArrayRef<size_t>.
    auto TH = in->getPayload().getHandle<int64_t>();
    llvm::ArrayRef<size_t> outputDims = {(const size_t *)TH.begin(),
                                         (const size_t *)TH.end()};

    ty = G_.getParent()->uniqueType(T.getType().getElementType(), outputDims);
    switch (T.getType().getElementType()) {
    case ElemKind::Int64ITy: {
      int64_t v = T.getHandle<int64_t>().raw(0);
      RETURN_ERR_IF_NOT(
          v == static_cast<int64_t>(static_cast<float>(v)),
          "This ConstantOfShape implementation may cause losses for value " +
              std::to_string(v) + " .");
      SN = G_.createSplat(loadOperatorName(op), ty, v);
      break;
    }
    case ElemKind::Int32ITy: {
      int32_t v = T.getHandle<int32_t>().raw(0);
      RETURN_ERR_IF_NOT(
          v == static_cast<int32_t>(static_cast<float>(v)),
          "This ConstantOfShape implementation may cause losses for value " +
              std::to_string(v) + " .");
      SN = G_.createSplat(loadOperatorName(op), ty, v);
      break;
    }
    default:
      SN = G_.createSplat(loadOperatorName(op), ty, T.getHandle().raw(0));
    }
  } else {
    ty = G_.getParent()->uniqueType(ElemKind::FloatTy, {1});
    SN = G_.createSplat(loadOperatorName(op), ty, T.getHandle().raw(0));
  }
  RETURN_IF_ERR(addNodeAsOutput(op, SN));
  return llvm::Error::success();
}

llvm::Error ONNXModelLoader::loadTile(const ONNX_NAMESPACE::NodeProto &op,
                                      const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);
  NodeValue in, repeats;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  ASSIGN_VALUE_OR_RETURN_ERR(repeats, getNodeValueByName(op.input(1)));
  if (!llvm::isa<Constant>(repeats)) {
    RETURN_ERR("Only constant Repeats is supported!");
  }

  if (repeats.dims().size() != 1) {
    RETURN_ERR("Repeats must be a single-dimensional tensor!");
  }

  if (repeats.dims()[0] != in.dims().size()) {
    RETURN_ERR("Repeats should have one value for each dimension of input!");
  }
  auto rh = llvm::cast<Constant>(repeats)->getPayload().getHandle<int64_t>();
  Node *N = in;
  for (size_t i = 0; i < in.dims().size(); i++) {
    auto tiles = rh.raw(i);
    if (tiles != 1) {
      std::string name = opName + "." + std::to_string(i);
      N = G_.createTile(name, N, tiles, /*axis*/ i);
    }
  }

  RETURN_IF_ERR(addNodeAsOutput(op, N));
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
  if (typeName == "Cast") {
    return loadCast(op, dict);
  }
  if (typeName == "LeakyRelu") {
    return loadLeakyRelu(op, dict);
  }
  if (typeName == "SpaceToDepth") {
    return loadSpaceToDepth(op, dict);
  }
  if (typeName == "ConstantOfShape") {
    return loadConstantOfShape(op, dict);
  }
  if (typeName == "Tile") {
    return loadTile(op, dict);
  }

  RETURN_ERR("Failed to load operator " + typeName + " .",
             GlowErr::ErrorCode::MODEL_LOADER_UNSUPPORTED_OPERATOR);
}

llvm::Error ONNXModelLoader::loadInitializers(ONNX_NAMESPACE::GraphProto &net) {
  // Load the network initializaers:
  for (const auto &in : net.initializer()) {
    Tensor T;
    RETURN_IF_ERR(loadTensor(in, &T));
    RETURN_IF_ERR(createAndRegisterConstant(in.name(), std::move(T)));
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
    ASSIGN_VALUE_OR_RETURN_ERR(r, getNodeValueByName(outputName));
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
    : CommonOperatorLoader({}, {}, F, errPtr) {
  deleteUnusedConstants();
}

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

    if (tensorNames.empty() && types.empty()) {
      // Detect inputs without initializers and create placeholders.
      RETURN_IF_ERR(loadInputs(graphDef, /* loadInputsAsPlaceholders */ true));
    }

    RETURN_IF_ERR(loadNetwork(graphDef));

    RETURN_IF_ERR(setOutputNodes(graphDef));

    RETURN_ERR_IF_NOT(F.verify(), "Function verification failed.");

    deleteUnusedConstants();

    return llvm::Error::success();
  };

  if (errPtr) {
    *errPtr = setup();
  } else {
    EXIT_ON_ERR(setup());
  }
}

ONNXModelLoader::ONNXModelLoader(
    const void *model, uint32_t modelSize, uint32_t weightsCount,
    const onnxTensorDescriptorV1 *weightDescriptors, Function &F,
    bool loadInputsAsPlaceholders, llvm::Error *errPtr)
    : CommonOperatorLoader({}, {}, F, errPtr) {
  // if errPtr already contains an error then don't continue with constructor
  if (errPtr && *errPtr) {
    return;
  }

  // Lambda to setup the ONNXModelLoader and return any llvm::Errors that were
  // raised.
  auto setup = [&]() -> llvm::Error {
    ONNX_NAMESPACE::ModelProto modelDef;
    ASSIGN_VALUE_OR_RETURN_ERR(modelDef, loadProto(model, modelSize));

    RETURN_IF_ERR(setVersion(modelDef));

    RETURN_IF_ERR(loadWeights(weightsCount, weightDescriptors));

    ONNX_NAMESPACE::GraphProto graphDef = modelDef.graph();

    RETURN_IF_ERR(loadInputs(graphDef, loadInputsAsPlaceholders));

    RETURN_IF_ERR(loadInitializers(graphDef));

    RETURN_IF_ERR(loadNetwork(graphDef));

    RETURN_IF_ERR(setOutputNodes(graphDef));

    deleteUnusedConstants();

    return llvm::Error::success();
  };

  if (errPtr) {
    *errPtr = setup();
  } else {
    EXIT_ON_ERR(setup());
  }
}
