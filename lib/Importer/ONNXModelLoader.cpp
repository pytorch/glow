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

#include "glow/Importer/ONNXModelLoader.h"
#include "glow/Base/Tensor.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/Support/ZipUtils.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"

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

llvm::cl::OptionCategory onnxModelLoaderCat("ONNX Model Loader Options");

std::vector<std::string> onnxDefineSymbol;
llvm::cl::list<std::string, std::vector<std::string>> onnxDefineSymbolOpt(
    "onnx-define-symbol", llvm::cl::ZeroOrMore,
    llvm::cl::location(onnxDefineSymbol),
    llvm::cl::desc(
        "Define (replace) the undefined symbols from the tensor descriptions\n"
        "in the ONNX model with actual integer sizes. The undefined symbols \n"
        "are marked in the proto description with the 'dim_param' field. For\n"
        "example, if the model contains a tensor with the size described as \n"
        "'None' x 3 x 224 x 224, the symbol 'None' can be replaced with an  \n"
        "actual integer size (for example 1) by using the following command \n"
        "line option:                                                       \n"
        "    -onnx-define-symbol=None,1                                     \n"
        "Multiple symbols can be defined using this option, for example:    \n"
        "    -onnx-define-symbol=<symbol_name1>,<symbol_value1>             \n"
        "    -onnx-define-symbol=<symbol_name2>,<symbol_value2>             \n"
        "    ..................................................\n"),
    llvm::cl::value_desc("name,value"), llvm::cl::cat(onnxModelLoaderCat));

/// Parse the command line option and get the user defined map of symbols.
/// The command line option has the format <symbol_name>,<symbol_value>.
Expected<std::unordered_map<std::string, dim_t>> getSymbolMap() {
  std::unordered_map<std::string, dim_t> symbolMap;
  for (const auto &str : onnxDefineSymbol) {
    auto strPair = llvm::StringRef(str).split(',');
    llvm::StringRef name = strPair.first;
    RETURN_ERR_IF_NOT(name.size() > 0, "ONNX defined symbol name is empty.");
    dim_t value;
    RETURN_ERR_IF_NOT(!strPair.second.getAsInteger(0, value),
                      strFormat("ONNX defined symbol value '%s' is invalid.",
                                strPair.second.data()));
    symbolMap[name.str()] = value;
  }
  return symbolMap;
}

/// Get the shape of a TensorShapeProto given by \p shapeProto and return the
/// dimensions in the vector \p dim passed by reference.
Expected<std::vector<dim_t>>
getProtoShape(const ONNX_NAMESPACE::TensorShapeProto &shapeProto) {
  std::vector<dim_t> dim;
  for (auto d : shapeProto.dim()) {
    if (d.has_dim_value()) {
      // Proto shape has an explicit size given by the "dim_value" field.
      dim.push_back(d.dim_value());
    } else if (d.has_dim_param()) {
      // Proto shape has a symbolic size given by the "dim_param" field. Search
      // the symbol in the user defined map of symbols. If the symbol is not
      // found then raise an error.
      auto symbolName = d.dim_param();
      std::unordered_map<std::string, dim_t> symbolMap;
      ASSIGN_VALUE_OR_RETURN_ERR(symbolMap, getSymbolMap());
      if (symbolMap.count(symbolName)) {
        dim.push_back(symbolMap[symbolName]);
      } else {
        RETURN_ERR(strFormat(
            "ONNX model symbol '%s' is undefined. Define the symbol with the "
            "following command line option: -onnx-define-symbol=%s,<value>.",
            symbolName.c_str(), symbolName.c_str()));
      }
    } else {
      // Proto shape has no "dim_value" and no "dim_param" field.
      RETURN_ERR("Tensor shape proto has no 'dim_value' or 'dim_param' field!");
    }
  }
  return dim;
}

Error onnxTensorDataTypeToElemKind(int32_t onnxType, ElemKind *elemTy) {
  if (onnxType == ONNX_NAMESPACE::TensorProto::FLOAT) {
    *elemTy = ElemKind::FloatTy;
    return Error::success();
  } else if (onnxType == ONNX_NAMESPACE::TensorProto::FLOAT16) {
    *elemTy = ElemKind::Float16Ty;
    return Error::success();
  } else if (onnxType == ONNX_NAMESPACE::TensorProto::INT64) {
    *elemTy = ElemKind::Int64ITy;
    return Error::success();
  } else if (onnxType == ONNX_NAMESPACE::TensorProto::INT32) {
    *elemTy = ElemKind::Int32ITy;
    return Error::success();
  } else if (onnxType == ONNX_NAMESPACE::TensorProto::UINT8) {
    *elemTy = ElemKind::UInt8FusedQTy;
    return Error::success();
  } else if (ONNX_NAMESPACE::TensorProto::BOOL) {
    *elemTy = ElemKind::BoolTy;
    return Error::success();
  } else {
    RETURN_ERR(strFormat(
        "Don't know how to convert ONNX tensor data type %d to ElemKind",
        onnxType));
  }
}

/// Creates tensor \p T from the input \p in. Note, there is no data associated
/// with the Tensor. This method makes sure that the tensor is created with the
/// proper shape and element type.
Error setTensorType(const ONNX_NAMESPACE::TypeProto &in, Tensor *T) {
  std::vector<dim_t> dim;
  ASSIGN_VALUE_OR_RETURN_ERR(dim, getProtoShape(in.tensor_type().shape()));

  ElemKind kind = ElemKind::FloatTy;
  RETURN_IF_ERR(
      onnxTensorDataTypeToElemKind(in.tensor_type().elem_type(), &kind));
  if (kind == ElemKind::UInt8FusedQTy || kind == ElemKind::UInt4FusedFP16QTy) {
    T->reset(kind, dim, 0.0, 0);
  } else {
    T->reset(kind, dim);
  }
  return Error::success();
}

/// Given a docstring encoding \p str of a type and its dimension \p
/// dims, parses the string and \returns a Glow Type from it or Error if parsing
/// failed. Expected format of str is either "ElemKind" or
/// "ElemKind:scale:offset".
Expected<Type> parseTypeFromDocString(const std::string &str,
                                      llvm::ArrayRef<dim_t> dims) {
  size_t begin = 0;

  float scale = 1.0;
  int32_t offset = 0;

  // Find Elemkind string
  size_t end = str.find(':', begin);

  // If a ':' isn't found then assume the whole string is ElemKind (for
  // backwards compatibility reasons) otherwise look for scale and offset
  // strings.
  std::string elemKindStr;
  if (end == std::string::npos) {
    elemKindStr = str.substr(0, str.size());
  } else {
    elemKindStr = str.substr(begin, end - begin);

    // Get scale string.
    begin = end + 1;
    end = str.find(':', begin);
    if (end == std::string::npos) {
      return MAKE_ERR("scale not found");
    }
    std::string scaleStr = str.substr(begin, end - begin);

    // Get offset string.
    begin = end + 1;
    end = str.size();
    if (end - begin == 0) {
      return MAKE_ERR("offset not found");
    }

    std::string offsetStr = str.substr(begin, end - begin);

    scale = std::stof(scaleStr);
    offset = std::stoi(offsetStr);
  }

  ElemKind elemKind = Type::getElementKindFromName(elemKindStr);

  if (isQuantizedElemKind(elemKind)) {
    return Type(elemKind, dims, scale, offset);
  } else {
    return Type(elemKind, dims);
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

void glow::setOnnxDefineSymbol(const std::vector<std::string> &strs) {
  onnxDefineSymbol = strs;
}

/// Loads tensor \p T from the input \p in.
Error glow::loadTensor(const ONNX_NAMESPACE::TensorProto &in, Tensor *T) {
  std::vector<dim_t> dim;
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
                 ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);
    }
  } else if (in.data_type() == ONNX_NAMESPACE::TensorProto::FLOAT16) {
    T->reset(ElemKind::Float16Ty, dim);
    if (in.has_raw_data()) {
      std::istringstream inStream(in.raw_data(), std::stringstream::binary);
      inStream.read(T->getUnsafePtr(), T->size() * (sizeof(float) / 2));
    } else {
      RETURN_ERR("Unsupported Tensor format.",
                 ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);
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
                 ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);
    }
  } else if (in.data_type() == ONNX_NAMESPACE::TensorProto::INT8) {
    Type ty;
    ASSIGN_VALUE_OR_RETURN_ERR(ty,
                               parseTypeFromDocString(in.doc_string(), dim));
    T->reset(ty);

    if (in.has_raw_data()) {
      std::istringstream inStream(in.raw_data(), std::stringstream::binary);
      inStream.read(T->getUnsafePtr(), T->size() * sizeof(int8_t));
    } else {
      RETURN_ERR("Unsupported Tensor format.",
                 ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);
    }
  } else if (in.data_type() == ONNX_NAMESPACE::TensorProto::INT32) {
    if (in.has_doc_string()) {
      Type ty;
      ASSIGN_VALUE_OR_RETURN_ERR(ty,
                                 parseTypeFromDocString(in.doc_string(), dim));
      T->reset(ty);
    } else {
      // There are few cases when we will have int32 tensors. For example, the
      // second output of Concat from Caffe2 concat op is int32
      T->reset(ElemKind::Int32ITy, dim);
    }

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
                 ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);
    }
  } else if (in.data_type() == ONNX_NAMESPACE::TensorProto::UINT8) {
    Type ty;
    ASSIGN_VALUE_OR_RETURN_ERR(ty,
                               parseTypeFromDocString(in.doc_string(), dim));
    T->reset(ty);

    if (in.has_raw_data()) {
      std::istringstream inStream(in.raw_data(), std::stringstream::binary);
      inStream.read(T->getUnsafePtr(), T->size() * sizeof(uint8_t));
    } else {
      RETURN_ERR("Unsupported Tensor format.",
                 ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);
    }
  } else if (in.data_type() == ONNX_NAMESPACE::TensorProto::BOOL) {
    T->reset(ElemKind::BoolTy, dim);
    if (in.has_raw_data()) {
      std::istringstream inStream(in.raw_data(), std::stringstream::binary);
      inStream.read(T->getUnsafePtr(), T->size() * sizeof(bool));
    } else {
      RETURN_ERR("Unsupported Tensor format.",
                 ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);
    }
  } else {
    RETURN_ERR(strFormat("Unsupported tensor data type: %u",
                         static_cast<unsigned>(in.data_type())),
               ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);
  }
  return Error::success();
}

Error ONNXModelLoader::loadInputs(ONNX_NAMESPACE::GraphProto &net,
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
          placeholder,
          createAndRegisterPlaceholder(in.name(), &T.getType(),
                                       staticInputs_.count(in.name())));
      inputVarsByName_.try_emplace(in.name(), placeholder);
    } else {
      Tensor T;
      RETURN_IF_ERR(setTensorType(in.type(), &T));
      RETURN_IF_ERR(createAndRegisterConstant(in.name(), std::move(T)));
    }
  }
  return Error::success();
}

Expected<bool> ONNXModelLoader::getBroadcast(const ArgumentDictionaryTy &dict) {
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

Expected<ElemKind> ONNXModelLoader::convertTensorProtoDataType(
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

Error ONNXModelLoader::setVersion(ONNX_NAMESPACE::ModelProto MP) {
  irVersion_ = MP.ir_version();
  opsetVersion_ = 0;
  RETURN_ERR_IF_NOT(
      irVersion_ >= 3,
      "This ONNX model with ir_version < 3 is too old to be supported.",
      ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_ONNX_VERSION);
  for (const auto &imp : MP.opset_import()) {
    if (!imp.has_domain() || imp.domain() == "") {
      opsetVersion_ = imp.version();
      break;
    }
  }
  RETURN_ERR_IF_NOT(opsetVersion_ > 0,
                    "The opset of this ONNX model is not supported.");
  return Error::success();
}

Expected<ONNX_NAMESPACE::ModelProto>
ONNXModelLoader::loadProto(google::protobuf::io::ZeroCopyInputStream &iStream) {
  // Construct and configure a Coded Input Stream
  google::protobuf::io::CodedInputStream codedStream(&iStream);

  // Don't warn about large file sizes.
  codedStream.SetTotalBytesLimit(MAX_PROTO_SIZE, MAX_PROTO_SIZE);
  ONNX_NAMESPACE::ModelProto MP;
  bool parseNet = MP.ParseFromCodedStream(&codedStream);
  RETURN_ERR_IF_NOT(parseNet, "Failed to parse ModelProto",
                    ErrorValue::ErrorCode::MODEL_LOADER_INVALID_PROTOBUF);
  return MP;
}

Expected<ONNX_NAMESPACE::ModelProto>
ONNXModelLoader::loadProto(const void *onnxModel, size_t onnxModelSize) {
  google::protobuf::io::ArrayInputStream arrayStream(onnxModel, onnxModelSize);
  return loadProto(arrayStream);
}

Expected<ONNX_NAMESPACE::ModelProto>
ONNXModelLoader::loadProto(const std::string &filename) {
  std::ifstream ff(filename, std::ios::in | std::ios::binary);
  RETURN_ERR_IF_NOT(ff,
                    strFormat("Can't find the model or network files for %s.",
                              filename.c_str()),
                    ErrorValue::ErrorCode::MODEL_LOADER_INVALID_PROTOBUF);

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
                      ErrorValue::ErrorCode::MODEL_LOADER_INVALID_PROTOBUF);
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
Expected<Pads> getPads(const ArgumentDictionaryTy &dict,
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

Error ONNXModelLoader::loadConstant(const ONNX_NAMESPACE::NodeProto &op,
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
    return Error::success();
  }

  RETURN_ERR_IF_NOT(dict.at("value")->type() ==
                        ONNX_NAMESPACE::AttributeProto::TENSOR,
                    "Only Tensor type constants are supported.",
                    ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);

  Tensor T;
  RETURN_IF_ERR(loadTensor(dict.at("value")->t(), &T));
  RETURN_IF_ERR(createAndRegisterConstant(name, std::move(T)));

  return Error::success();
}

/// Retrieves data from a constant Tensor and stores it in a vector.
template <typename T>
static void helperSetter(Constant *constT, std::vector<ssize_t> &vec) {
  auto constH = constT->getPayload().getHandle<T>();
  for (dim_t i = 0; i < constH.size(); ++i) {
    vec.push_back(constH.at({i}));
  }
}

Error ONNXModelLoader::loadSlice(const ONNX_NAMESPACE::NodeProto &op,
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
  std::vector<dim_t> newStarts(numDims);
  std::vector<dim_t> newEnds(numDims);
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

  return Error::success();
}

Error ONNXModelLoader::loadConv(const ONNX_NAMESPACE::NodeProto &op,
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
  dim_t depth = filterTransposedValue.dims()[0];

  // Get the kernel shape from the input.
  llvm::SmallVector<unsigned_t, 2> kernelShape(2);
  kernelShape[0] = filterTransposedValue.dims()[1];
  kernelShape[1] = filterTransposedValue.dims()[2];

  // Extra check when the 'kernel_shape' attribute exists.
  // The 'kernel_shape' attribute is redundant not mandatory.
  if (dict.count("kernel_shape")) {
    auto kernelShapeAttribute = getShape<unsigned_t>(dict.at("kernel_shape"));
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
    ASSIGN_VALUE_OR_RETURN_ERR(bias, getConstantByName(biasTensorName));
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
  std::array<dim_t, 4> outDims = {{idim.n, outSz.first, outSz.second, depth}};
  auto outTy = G_.getParent()->uniqueType(ElemKind::FloatTy, outDims);

  auto *node = G_.createConv(opName, tr, filterTransposeNode, bias, outTy,
                             kernelShape, strides, pads, group, dilation);

  // Transpose the output back.
  auto *N = G_.createTranspose(opName, node, NHWC2NCHW);
  RETURN_IF_ERR(addNodeAsOutput(op, N));

  return Error::success();
}

Error ONNXModelLoader::loadTensorwiseQuantizedConvolution(
    const ONNX_NAMESPACE::NodeProto &op, const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));
  NodeValue filterValue;
  ASSIGN_VALUE_OR_RETURN_ERR(filterValue, getNodeValueByName(op.input(1)));
  NodeValue biasValue;
  ASSIGN_VALUE_OR_RETURN_ERR(biasValue, getNodeValueByName(op.input(2)));

  auto kernels = getShape<unsigned_t>(dict.at("kernel_shape"));
  auto strides = getShape<unsigned_t>(dict.at("strides"));
  auto pads = getShape<unsigned_t>(dict.at("pads"));
  unsigned_t groups;
  ASSIGN_VALUE_OR_RETURN_ERR(groups, loadInt(dict.at("group")));
  unsigned_t dilation;
  ASSIGN_VALUE_OR_RETURN_ERR(dilation, loadInt(dict.at("group")));

  float outScale;
  ASSIGN_VALUE_OR_RETURN_ERR(outScale, loadFloat(dict.at("out_scale")));
  int32_t outOffset;
  ASSIGN_VALUE_OR_RETURN_ERR(outOffset, loadInt(dict.at("out_offset")));

  ShapeNHWC idim(input.dims());
  auto outSz =
      calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides, pads);
  std::array<dim_t, 4> outDims = {
      {idim.n, outSz.first, outSz.second, biasValue.dims()[0]}};
  auto outTy = G_.getParent()->uniqueType(ElemKind::Int8QTy, outDims, outScale,
                                          outOffset);

  auto *node = G_.createConv(opName, input, filterValue, biasValue, outTy,
                             kernels, strides, pads, groups);

  return addNodeAsOutput(op, node);
}

Error ONNXModelLoader::loadChannelwiseQuantizedConvolution(
    const ONNX_NAMESPACE::NodeProto &op, const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));
  NodeValue filterValue;
  ASSIGN_VALUE_OR_RETURN_ERR(filterValue, getNodeValueByName(op.input(1)));
  NodeValue biasValue;
  ASSIGN_VALUE_OR_RETURN_ERR(biasValue, getNodeValueByName(op.input(2)));
  NodeValue scalesValue;
  ASSIGN_VALUE_OR_RETURN_ERR(scalesValue, getNodeValueByName(op.input(3)));
  NodeValue offsetsValue;
  ASSIGN_VALUE_OR_RETURN_ERR(offsetsValue, getNodeValueByName(op.input(4)));

  auto kernels = getShape<unsigned_t>(dict.at("kernel_shape"));
  auto strides = getShape<unsigned_t>(dict.at("strides"));
  auto pads = getShape<unsigned_t>(dict.at("pads"));
  unsigned_t groups;
  ASSIGN_VALUE_OR_RETURN_ERR(groups, loadInt(dict.at("group")));

  float outScale;
  ASSIGN_VALUE_OR_RETURN_ERR(outScale, loadFloat(dict.at("out_scale")));
  int32_t outOffset;
  ASSIGN_VALUE_OR_RETURN_ERR(outOffset, loadInt(dict.at("out_offset")));

  ShapeNHWC idim(input.dims());
  auto outSz =
      calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides, pads);
  std::array<dim_t, 4> outDims = {
      {idim.n, outSz.first, outSz.second, biasValue.dims()[0]}};
  auto outTy = G_.getParent()->uniqueType(ElemKind::Int8QTy, outDims, outScale,
                                          outOffset);

  auto *node = G_.createChannelwiseQuantizedConv(
      opName, input, filterValue, biasValue, scalesValue, offsetsValue, outTy,
      kernels, strides, pads, groups);

  return addNodeAsOutput(op, node);
}

Error ONNXModelLoader::loadPool(const ONNX_NAMESPACE::NodeProto &op,
                                const ArgumentDictionaryTy &dict,
                                llvm::StringRef typeName) {
  const std::string &opName = loadOperatorName(op);

  // Load the inputs:
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  std::vector<unsigned_t> strides(2, 1);
  if (dict.count("strides")) {
    strides = getShape<unsigned_t>(dict.at("strides"));
  }
  auto kernels = getShape<unsigned_t>(dict.at("kernel_shape"));

  if (in.dims().size() != 4 || kernels.size() != 2) {
    // Glow only handles 2D pooling currently.
    RETURN_ERR("Glow only handles 2D pooling currently.",
               ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_SHAPE);
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
  if (op.output_size() > 1) {
    if (typeName != "MaxPool") {
      RETURN_ERR("Argmax output is only supported for MaxPool!",
                 ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_OPERATOR);
    }

    node = G_.createMaxPool(opName, tr, kernels, strides, pads);
    auto *res = G_.createTranspose(opName, NodeValue(node, 0), NHWC2NCHW);
    auto *argmax = G_.createTranspose(opName, NodeValue(node, 1), NHWC2NCHW);
    RETURN_IF_ERR(assignNodeOutputs(op, {res, argmax}));
  } else {
    size_t idx = 0;
    if (typeName == "MaxPool") {
      node = G_.createMaxPool(opName, tr, kernels, strides, pads);
      idx = MaxPoolNode::ResultIdx;
    } else {
      node = G_.createAvgPool(opName, tr, kernels, strides, pads);
      idx = AvgPoolNode::ResultIdx;
    }
    auto *N = G_.createTranspose(opName, NodeValue(node, idx), NHWC2NCHW);
    RETURN_IF_ERR(addNodeAsOutput(op, N));
  }
  return Error::success();
}

Error ONNXModelLoader::loadArgMax(const ONNX_NAMESPACE::NodeProto &op,
                                  const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  size_t axis = 0;
  if (dict.count("axis")) {
    ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict.at("axis")));
  }
  bool keepDims = true;
  if (dict.count("keepDims")) {
    ASSIGN_VALUE_OR_RETURN_ERR(keepDims, loadInt(dict.at("keepDims")));
  }
  Node *node = G_.createArgMax(opName, in, axis, keepDims);
  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return Error::success();
}

Error ONNXModelLoader::loadGlobalAveragePool(
    const ONNX_NAMESPACE::NodeProto &op, const ArgumentDictionaryTy &dict) {
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
  return Error::success();
}

Error ONNXModelLoader::loadSqueeze(const ONNX_NAMESPACE::NodeProto &op,
                                   const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  auto axes = getShape(dict.at("axes"));
  Node *node = G_.createSqueeze(opName, in, axes);
  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return Error::success();
}

Error ONNXModelLoader::loadUnsqueeze(const ONNX_NAMESPACE::NodeProto &op,
                                     const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  auto axes = getShape(dict.at("axes"));
  Node *node = G_.createExpandDims(opName, in, axes);
  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return Error::success();
}

Error ONNXModelLoader::loadBatchNormalization(
    const ONNX_NAMESPACE::NodeProto &op, const ArgumentDictionaryTy &dict) {
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

  return Error::success();
}

Error ONNXModelLoader::loadConcat(const ONNX_NAMESPACE::NodeProto &op,
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
  return Error::success();
}

Error ONNXModelLoader::loadFCTransposed(const ONNX_NAMESPACE::NodeProto &op,
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
  return Error::success();
}

Error ONNXModelLoader::loadGemm(const ONNX_NAMESPACE::NodeProto &op,
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
  return Error::success();
}

Error ONNXModelLoader::loadMatMul(const ONNX_NAMESPACE::NodeProto &op,
                                  const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue LHS;
  ASSIGN_VALUE_OR_RETURN_ERR(LHS, getNodeValueByName(op.input(0)));
  NodeValue RHS;
  ASSIGN_VALUE_OR_RETURN_ERR(RHS, getNodeValueByName(op.input(1)));

  Node *node = G_.createMatMul(opName, LHS, RHS);
  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return Error::success();
}

Error ONNXModelLoader::loadLeakyRelu(const ONNX_NAMESPACE::NodeProto &op,
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

  return Error::success();
}

Error ONNXModelLoader::loadPad(const ONNX_NAMESPACE::NodeProto &op,
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
                 ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_ATTRIBUTE);
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
  std::vector<dim_t> outDims(numDims);
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

  return Error::success();
}

Error ONNXModelLoader::loadCast(const ONNX_NAMESPACE::NodeProto &op,
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
      ErrorValue::ErrorCode::MODEL_LOADER_INVALID_PROTOBUF);
  ASSIGN_VALUE_OR_RETURN_ERR(
      targetKind, convertTensorProtoDataType(
                      ONNX_NAMESPACE::TensorProto_DataType(toONNXTypeValue)));

  // Only support non quantized types.
  RETURN_ERR_IF_NOT((!isQuantizedElemKind(inputKind)) &&
                        (!isQuantizedElemKind(targetKind)),
                    "Unsupported Cast");

  // Create the IR node.
  Node *N = G_.createConvertTo(opName, input, targetKind);
  RETURN_IF_ERR(addNodeAsOutput(op, N));

  return Error::success();
}

Error ONNXModelLoader::loadSpaceToDepth(const ONNX_NAMESPACE::NodeProto &op,
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

  return Error::success();
}

Error ONNXModelLoader::loadConstantOfShape(const ONNX_NAMESPACE::NodeProto &op,
                                           const ArgumentDictionaryTy &dict,
                                           bool isSplat) {
  Tensor T(ElemKind::FloatTy, {1});
  T.getHandle().raw(0) = 0.0;

  if (dict.count("value")) {
    RETURN_IF_ERR(loadTensor(dict.at("value")->t(), &T));
    if (!isSplat) {
      // Validate tensor only for ConstantOfShape operator.
      RETURN_ERR_IF_NOT(T.dims().size() == 1, "Value must be a 1D vector.");
      RETURN_ERR_IF_NOT(
          T.getType().getElementType() == ElemKind::FloatTy ||
              T.getType().getElementType() == ElemKind::Int64ITy ||
              T.getType().getElementType() == ElemKind::Int32ITy,
          T.getType().getElementName().str() + " type Value is not supported.");
    }
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
    // Convert 1D tensor of int64_t into llvm::ArrayRef<dim_t>.
    auto TH = in->getPayload().getHandle<int64_t>();
    auto begin = &TH.raw(0);
    auto end = begin + TH.actualSize();
    ShapeVector outputDims(begin, end);

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
    ty = G_.getParent()->uniqueType(T.getType().getElementType(), T.dims());
    SN = G_.createSplat(loadOperatorName(op), ty, T.getHandle().raw(0));
  }
  RETURN_IF_ERR(addNodeAsOutput(op, SN));
  return Error::success();
}

Error ONNXModelLoader::loadTile(const ONNX_NAMESPACE::NodeProto &op,
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
  return Error::success();
}

Expected<bool>
ONNXModelLoader::foldOperator(const ONNX_NAMESPACE::NodeProto &op) {
  const unsigned numInputs = op.input_size();
  const std::string &typeName = op.op_type();
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
  ONNXModelLoader tmpLoader(*tmpF);
  tmpLoader.opsetVersion_ = opsetVersion_;
  bool foldStatus = !ERR_TO_BOOL(
      constantFoldInLoader<ONNXModelLoader, ONNX_NAMESPACE::NodeProto>(
          tmpF, tmpLoader, this, op),
      /* log */ false);
  G_.getParent()->eraseFunction(tmpF);
  return foldStatus;
}

Error ONNXModelLoader::loadWhere(const ONNX_NAMESPACE::NodeProto &op,
                                 const ArgumentDictionaryTy &dict) {
  NodeValue cNV;
  ASSIGN_VALUE_OR_RETURN_ERR(cNV, getNodeValueByName(op.input(0)));
  NodeValue xNV;
  ASSIGN_VALUE_OR_RETURN_ERR(xNV, getNodeValueByName(op.input(1)));
  NodeValue yNV;
  ASSIGN_VALUE_OR_RETURN_ERR(yNV, getNodeValueByName(op.input(2)));

  std::string opName = loadOperatorName(op);

  // Passing -1 for multi directional broadcast, axis will be computed
  // automatically.
  Node *N = G_.createNodeWithBroadcast<SelectNode>(opName, -1, cNV, xNV, yNV);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

/// Utility function to get the RNN, GRU or LSTM direction from the proto
/// description. If not provided, the default direction is 'forward'.
static Expected<Function::RnnDirection>
getRnnDirection(const ONNX_NAMESPACE::NodeProto &op,
                const ArgumentDictionaryTy &dict) {
  Function::RnnDirection direction = Function::RnnDirection::Forward;
  if (dict.count("direction")) {
    std::string directionStr;
    ASSIGN_VALUE_OR_RETURN_ERR(directionStr, loadStr(dict.at("direction")));
    if (directionStr == "forward") {
      direction = Function::RnnDirection::Forward;
    } else if (directionStr == "reverse") {
      direction = Function::RnnDirection::Reverse;
    } else if (directionStr == "bidirectional") {
      direction = Function::RnnDirection::Bidirectional;
    } else {
      RETURN_ERR("ONNX " + op.op_type() + " 'direction' attribute is invalid!",
                 ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_ATTRIBUTE);
    }
  }
  return direction;
}

/// Relu activation function definition.
static Function::RnnActivation RnnActivationRelu(Function &F) {
  return [&F](llvm::StringRef name, Node *input) {
    return F.createRELU(name, input);
  };
}

/// Tanh activation function definition.
static Function::RnnActivation RnnActivationTanh(Function &F) {
  return [&F](llvm::StringRef name, Node *input) {
    return F.createTanh(name, input);
  };
}

/// Sigmoid activation function definition.
static Function::RnnActivation RnnActivationSigmoid(Function &F) {
  return [&F](llvm::StringRef name, Node *input) {
    return F.createSigmoid(name, input);
  };
}

/// Utility function to get the RNN, GRU or LSTM activation functions from the
/// proto description. The activation function array is assumed to be already
/// initialized with the default values upon entering this function so that the
/// purpose of this function is to overwrite the specific default values.
/// Currenlty only Sigmoid, Tahn and ReLU activations are supported.
static Error
getRnnActivations(const ONNX_NAMESPACE::NodeProto &op,
                  const ArgumentDictionaryTy &dict, Function &F,
                  std::vector<Function::RnnActivation> &activations) {

  // Activation alpha not supported (Optional)(Default:activation dependent).
  RETURN_ERR_IF_NOT(!dict.count("activation_alpha"),
                    "ONNX " + op.op_type() +
                        " 'activation_alpha' attribute not supported!");

  // Activation beta not supported (Optional)(Default:activation dependent).
  RETURN_ERR_IF_NOT(!dict.count("activation_beta"),
                    "ONNX " + op.op_type() +
                        " 'activation_beta' attribute not supported!");

  // Get activations.
  if (dict.count("activations") && dict.at("activations")->strings_size()) {
    size_t actNum = dict.at("activations")->strings_size();
    size_t actNumExpected = activations.size();
    RETURN_ERR_IF_NOT(actNum == actNumExpected,
                      strFormat("ONNX %s 'activations' attribute has invalid "
                                "number of functions! Expected number is %d!",
                                op.op_type().c_str(), (int)actNumExpected));
    for (size_t actIdx = 0; actIdx < actNum; actIdx++) {
      std::string actStr = dict.at("activations")->strings().Get(actIdx);
      if (actStr == "Relu") {
        activations[actIdx] = RnnActivationRelu(F);
      } else if (actStr == "Tanh") {
        activations[actIdx] = RnnActivationTanh(F);
      } else if (actStr == "Sigmoid") {
        activations[actIdx] = RnnActivationSigmoid(F);
      } else {
        RETURN_ERR("ONNX " + op.op_type() + " activation '" + actStr +
                       "' not supported!",
                   ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_ATTRIBUTE);
      }
    }
  }
  return Error::success();
}

// Limitations:
// - Activation clipping not supported.
// - Variable sequence length not supported.
Error ONNXModelLoader::loadRNN(const ONNX_NAMESPACE::NodeProto &op,
                               const ArgumentDictionaryTy &dict) {

  const std::string &opName = loadOperatorName(op);

  // ------------------------- Attributes -------------------------------------
  // Get direction (Optional)(Default:forward).
  Function::RnnDirection direction;
  ASSIGN_VALUE_OR_RETURN_ERR(direction, getRnnDirection(op, dict));
  dim_t numDirections =
      (direction == Function::RnnDirection::Bidirectional) ? 2 : 1;

  // Get activations as lambdas (Optional)(Default:f=Tanh).
  std::vector<Function::RnnActivation> activations;
  if (direction == Function::RnnDirection::Bidirectional) {
    activations = {RnnActivationTanh(G_), RnnActivationTanh(G_)};
  } else {
    activations = {RnnActivationTanh(G_)};
  }
  RETURN_IF_ERR(getRnnActivations(op, dict, G_, activations));

  // Activation clipping not supported (Optional)(Default: 0 for no clipping).
  RETURN_ERR_IF_NOT(!dict.count("clip"),
                    "ONNX RNN 'clip' attribute not supported!");

  // Get hidden size (Required).
  dim_t hiddenSize;
  RETURN_ERR_IF_NOT(dict.count("hidden_size"),
                    "ONNX RNN 'hidden_size' attribute is required!");
  ASSIGN_VALUE_OR_RETURN_ERR(hiddenSize, loadInt(dict.at("hidden_size")));

  // --------------------------- Inputs ---------------------------------------
  const int numInputs = op.input_size();
  RETURN_ERR_IF_NOT((3 <= numInputs) && (numInputs <= 6),
                    "ONNX RNN should have minimum 3 and maximum 6 inputs!");

  // Input0: X (Required).
  NodeValue X;
  ASSIGN_VALUE_OR_RETURN_ERR(X, getNodeValueByName(op.input(0)));

  // Input1: W (Required).
  NodeValue W;
  ASSIGN_VALUE_OR_RETURN_ERR(W, getNodeValueByName(op.input(1)));

  // Input2: R (Required).
  NodeValue R;
  ASSIGN_VALUE_OR_RETURN_ERR(R, getNodeValueByName(op.input(2)));

  // Input3: B (Optional).
  NodeValue B = nullptr;
  if (numInputs > 3 && !op.input(3).empty()) {
    ASSIGN_VALUE_OR_RETURN_ERR(B, getNodeValueByName(op.input(3)));
  }

  // Input4: sequence_lens (Optional).
  if (numInputs > 4) {
    RETURN_ERR_IF_NOT(op.input(4).empty(),
                      "ONNX RNN 'sequence_lens' attribute not supported!");
  }

  // Input5: initial_h (Optional).
  NodeValue initial_h = nullptr;
  if (numInputs > 5 && !op.input(5).empty()) {
    ASSIGN_VALUE_OR_RETURN_ERR(initial_h, getNodeValueByName(op.input(5)));
  }

  // -------------------------- Outputs ---------------------------------------
  // We always create placeholders for the RNN state variable Y_h for the
  // following reasons:
  // - expose the RNN state in the graph interface for accessibility (set
  //   desired state, reset state, watch the state being updated automatically).
  // - since the RNN cells are unrolled (no graph loop primitive available
  //   at this point), the optimal way to use the RNN within a model would be
  //   to have it defined with only 1 time step and have the loop in the top
  //   of the application while the RNN state will be automatically updated
  //   from one iteration (time step) to the next through the placeholders.

  // Derived parameters.
  RETURN_ERR_IF_NOT(X.dims().size() == 3,
                    "ONNX RNN input 'X' should have 3 dimensions!");
  dim_t batchSize = X.dims()[1];

  // Create Y_h (hidden state) output placeholder.
  Placeholder *Y_h_ph;
  TypeRef Htype = G_.getParent()->uniqueTypeWithNewShape(
      X.getType(), {numDirections, batchSize, hiddenSize});
  std::string Hname = opName + ".Y_h";
  ASSIGN_VALUE_OR_RETURN_ERR(Y_h_ph,
                             createAndRegisterPlaceholder(Hname, Htype));
  inputVarsByName_.try_emplace(Hname, Y_h_ph);

  // If RNN input state is explicitly provided then used it. If not, then
  // use the RNN state placeholder.
  NodeValue Y_h_init = initial_h.getNode() ? initial_h : Y_h_ph;

  // Create ONNX RNN.
  NodeValue Y, Y_h;
  G_.createOnnxRNN(opName, X, W, R, B, Y_h_init, Y, Y_h, hiddenSize, direction,
                   activations);

  // Save RNN state in the state placeholder.
  G_.createSave(opName + ".Y_h.save", Y_h, Y_h_ph);

  // Add node.
  const int numOutputs = op.output_size();
  if (numOutputs == 1) {
    RETURN_IF_ERR(addNodeAsOutput(op, Y));
  } else if (numOutputs == 2) {
    RETURN_IF_ERR(assignNodeOutputs(op, {Y, Y_h}));
  } else {
    RETURN_ERR("ONNX RNN should have minimum 1 and maximum 2 outputs!");
  }
  return Error::success();
}

// Limitations:
// - Activation clipping not supported.
// - Variable sequence length not supported.
Error ONNXModelLoader::loadGRU(const ONNX_NAMESPACE::NodeProto &op,
                               const ArgumentDictionaryTy &dict) {

  const std::string &opName = loadOperatorName(op);

  // ------------------------- Attributes -------------------------------------
  // Get direction (Optional)(Default:forward).
  Function::RnnDirection direction;
  ASSIGN_VALUE_OR_RETURN_ERR(direction, getRnnDirection(op, dict));
  dim_t numDirections =
      (direction == Function::RnnDirection::Bidirectional) ? 2 : 1;

  // Get activations as lambdas (Optional)(Default:f=Sigmoid, g=Tanh).
  std::vector<Function::RnnActivation> activations;
  if (direction == Function::RnnDirection::Bidirectional) {
    activations = {RnnActivationSigmoid(G_), RnnActivationTanh(G_),
                   RnnActivationSigmoid(G_), RnnActivationTanh(G_)};
  } else {
    activations = {RnnActivationSigmoid(G_), RnnActivationTanh(G_)};
  }
  RETURN_IF_ERR(getRnnActivations(op, dict, G_, activations));

  // Activation clipping not supported (Optional)(Default: 0 for no clipping).
  RETURN_ERR_IF_NOT(!dict.count("clip"),
                    "ONNX GRU 'clip' attribute not supported!");

  // Get hidden size (Required).
  dim_t hiddenSize;
  RETURN_ERR_IF_NOT(dict.count("hidden_size"),
                    "ONNX GRU 'hidden_size' attribute is required!");
  ASSIGN_VALUE_OR_RETURN_ERR(hiddenSize, loadInt(dict.at("hidden_size")));

  // Get linear_before_reset (Optional)(Default:0).
  int linearBeforeReset = 0;
  if (dict.count("linear_before_reset") &&
      dict.at("linear_before_reset")->has_i()) {
    linearBeforeReset = dict.at("linear_before_reset")->i();
  }

  // --------------------------- Inputs ---------------------------------------
  const int numInputs = op.input_size();
  RETURN_ERR_IF_NOT((3 <= numInputs) && (numInputs <= 6),
                    "ONNX GRU should have minimum 3 and maximum 6 inputs!");

  // Input0: X (Required).
  NodeValue X;
  ASSIGN_VALUE_OR_RETURN_ERR(X, getNodeValueByName(op.input(0)));

  // Input1: W (Required).
  NodeValue W;
  ASSIGN_VALUE_OR_RETURN_ERR(W, getNodeValueByName(op.input(1)));

  // Input2: R (Required).
  NodeValue R;
  ASSIGN_VALUE_OR_RETURN_ERR(R, getNodeValueByName(op.input(2)));

  // Input3: B (Optional).
  NodeValue B = nullptr;
  if (numInputs > 3 && !op.input(3).empty()) {
    ASSIGN_VALUE_OR_RETURN_ERR(B, getNodeValueByName(op.input(3)));
  }

  // Input4: sequence_lens (Optional).
  if (numInputs > 4) {
    RETURN_ERR_IF_NOT(op.input(4).empty(),
                      "ONNX GRU 'sequence_lens' attribute not supported!");
  }

  // Input5: initial_h (Optional).
  NodeValue initial_h = nullptr;
  if (numInputs > 5 && !op.input(5).empty()) {
    ASSIGN_VALUE_OR_RETURN_ERR(initial_h, getNodeValueByName(op.input(5)));
  }

  // -------------------------- Outputs ---------------------------------------
  // We always create placeholders for the GRU state variable Y_h for the
  // following reasons:
  // - expose the GRU state in the graph interface for accessibility (set
  //   desired state, reset state, watch the state being updated automatically).
  // - since the GRU cells are unrolled (no graph loop primitive available
  //   at this point), the optimal way to use the GRU within a model would be
  //   to have it defined with only 1 time step and have the loop in the top
  //   of the application while the GRU state will be automatically updated
  //   from one iteration (time step) to the next through the placeholders.

  // Derived parameters.
  RETURN_ERR_IF_NOT(X.dims().size() == 3,
                    "ONNX GRU input 'X' should have 3 dimensions!");
  dim_t batchSize = X.dims()[1];

  // Create Y_h (hidden state) output placeholder.
  Placeholder *Y_h_ph;
  TypeRef Htype = G_.getParent()->uniqueTypeWithNewShape(
      X.getType(), {numDirections, batchSize, hiddenSize});
  std::string Hname = opName + ".Y_h";
  ASSIGN_VALUE_OR_RETURN_ERR(Y_h_ph,
                             createAndRegisterPlaceholder(Hname, Htype));
  inputVarsByName_.try_emplace(Hname, Y_h_ph);

  // If GRU input state is explicitly provided then used it. If not, then
  // use the GRU state placeholder.
  NodeValue Y_h_init = initial_h.getNode() ? initial_h : Y_h_ph;

  // Create ONNX GRU.
  NodeValue Y, Y_h;
  G_.createOnnxGRU(opName, X, W, R, B, Y_h_init, Y, Y_h, hiddenSize, direction,
                   activations, (bool)linearBeforeReset);

  // Save GRU state in the state placeholder.
  G_.createSave(opName + ".Y_h.save", Y_h, Y_h_ph);

  // Add node.
  const int numOutputs = op.output_size();
  if (numOutputs == 1) {
    RETURN_IF_ERR(addNodeAsOutput(op, Y));
  } else if (numOutputs == 2) {
    RETURN_IF_ERR(assignNodeOutputs(op, {Y, Y_h}));
  } else {
    RETURN_ERR("ONNX GRU should have minimum 1 and maximum 2 outputs!");
  }
  return Error::success();
}

// Limitations:
// - Activation clipping not supported.
// - Variable sequence length not supported.
Error ONNXModelLoader::loadLSTM(const ONNX_NAMESPACE::NodeProto &op,
                                const ArgumentDictionaryTy &dict) {

  const std::string &opName = loadOperatorName(op);

  // ------------------------- Attributes -------------------------------------
  // Get direction (Optional)(Default:forward).
  Function::RnnDirection direction;
  ASSIGN_VALUE_OR_RETURN_ERR(direction, getRnnDirection(op, dict));
  dim_t numDirections =
      (direction == Function::RnnDirection::Bidirectional) ? 2 : 1;

  // Get activations as lambdas (Optional)(Default:f=Sigmoid, g=Tanh, h=Tanh).
  std::vector<Function::RnnActivation> activations;
  if (direction == Function::RnnDirection::Bidirectional) {
    activations = {RnnActivationSigmoid(G_), RnnActivationTanh(G_),
                   RnnActivationTanh(G_),    RnnActivationSigmoid(G_),
                   RnnActivationTanh(G_),    RnnActivationTanh(G_)};
  } else {
    activations = {RnnActivationSigmoid(G_), RnnActivationTanh(G_),
                   RnnActivationTanh(G_)};
  }
  RETURN_IF_ERR(getRnnActivations(op, dict, G_, activations));

  // Activation clipping not supported (Optional)(Default: 0 for no clipping).
  RETURN_ERR_IF_NOT(!dict.count("clip"),
                    "ONNX LSTM 'clip' attribute not supported!");

  // Get hidden size (Required).
  dim_t hiddenSize;
  RETURN_ERR_IF_NOT(dict.count("hidden_size"),
                    "ONNX LSTM 'hidden_size' attribute is required!");
  ASSIGN_VALUE_OR_RETURN_ERR(hiddenSize, loadInt(dict.at("hidden_size")));

  // Get input forget (Optional)(Default:0).
  int inputForget = 0;
  if (dict.count("input_forget") && dict.at("input_forget")->has_i()) {
    inputForget = dict.at("input_forget")->i();
  }

  // --------------------------- Inputs ---------------------------------------
  const int numInputs = op.input_size();
  RETURN_ERR_IF_NOT((3 <= numInputs) && (numInputs <= 8),
                    "ONNX LSTM should have minimum 3 and maximum 8 inputs!");

  // Input0: X (Required).
  NodeValue X;
  ASSIGN_VALUE_OR_RETURN_ERR(X, getNodeValueByName(op.input(0)));

  // Input1: W (Required).
  NodeValue W;
  ASSIGN_VALUE_OR_RETURN_ERR(W, getNodeValueByName(op.input(1)));

  // Input2: R (Required).
  NodeValue R;
  ASSIGN_VALUE_OR_RETURN_ERR(R, getNodeValueByName(op.input(2)));

  // Input3: B (Optional).
  NodeValue B = nullptr;
  if (numInputs > 3 && !op.input(3).empty()) {
    ASSIGN_VALUE_OR_RETURN_ERR(B, getNodeValueByName(op.input(3)));
  }

  // Input4: sequence_lens (Optional).
  if (numInputs > 4) {
    RETURN_ERR_IF_NOT(op.input(4).empty(),
                      "ONNX LSTM 'sequence_lens' attribute not supported!");
  }

  // Input5: initial_h (Optional).
  NodeValue initial_h = nullptr;
  if (numInputs > 5 && !op.input(5).empty()) {
    ASSIGN_VALUE_OR_RETURN_ERR(initial_h, getNodeValueByName(op.input(5)));
  }

  // Input6: initial_c (Optional).
  NodeValue initial_c = nullptr;
  if (numInputs > 6 && !op.input(6).empty()) {
    ASSIGN_VALUE_OR_RETURN_ERR(initial_c, getNodeValueByName(op.input(6)));
  }

  // Input7: P (Optional).
  NodeValue P = nullptr;
  if (numInputs > 7 && !op.input(7).empty()) {
    ASSIGN_VALUE_OR_RETURN_ERR(P, getNodeValueByName(op.input(7)));
  }

  // -------------------------- Outputs ---------------------------------------
  // We always create placeholders for the LSTM state variables (Y_h and Y_c)
  // for the following reasons:
  // - expose the LSTM state in the graph interface for accessibility (set
  //   desired state, reset state, watch the state being updated automatically).
  // - since the LSTM cells are unrolled (no graph loop primitive available
  //   at this point), the optimal way to use the LSTM within a model would be
  //   to have it defined with only 1 time step and have the loop in the top
  //   of the application while the LSTM state will be automatically updated
  //   from one iteration (time step) to the next through the placeholders.

  // Derived parameters.
  RETURN_ERR_IF_NOT(X.dims().size() == 3,
                    "ONNX LSTM input 'X' should have 3 dimensions!");
  dim_t batchSize = X.dims()[1];

  // Create Y_h (hidden state) output placeholder.
  Placeholder *Y_h_ph;
  TypeRef Htype = G_.getParent()->uniqueTypeWithNewShape(
      X.getType(), {numDirections, batchSize, hiddenSize});
  std::string Hname = opName + ".Y_h";
  ASSIGN_VALUE_OR_RETURN_ERR(Y_h_ph,
                             createAndRegisterPlaceholder(Hname, Htype));
  inputVarsByName_.try_emplace(Hname, Y_h_ph);

  // Create Y_c (cell state) output placeholder.
  Placeholder *Y_c_ph;
  TypeRef Ctype = G_.getParent()->uniqueTypeWithNewShape(
      X.getType(), {numDirections, batchSize, hiddenSize});
  std::string Cname = opName + ".Y_c";
  ASSIGN_VALUE_OR_RETURN_ERR(Y_c_ph,
                             createAndRegisterPlaceholder(Cname, Ctype));
  inputVarsByName_.try_emplace(Cname, Y_c_ph);

  // If LSTM input states are explicitly provided then used them. If not, then
  // use the LSTM state placeholders.
  NodeValue Y_h_init = initial_h.getNode() ? initial_h : Y_h_ph;
  NodeValue Y_c_init = initial_c.getNode() ? initial_c : Y_c_ph;

  // Create ONNX LSTM.
  NodeValue Y, Y_h, Y_c;
  G_.createOnnxLSTM(opName, X, W, R, B, Y_h_init, Y_c_init, P, Y, Y_h, Y_c,
                    hiddenSize, direction, activations, (bool)inputForget);

  // Save LSTM state in the state placeholders.
  G_.createSave(opName + ".Y_h.save", Y_h, Y_h_ph);
  G_.createSave(opName + ".Y_c.save", Y_c, Y_c_ph);

  // Add node.
  const int numOutputs = op.output_size();
  if (numOutputs == 1) {
    RETURN_IF_ERR(addNodeAsOutput(op, Y));
  } else if (numOutputs == 2) {
    RETURN_IF_ERR(assignNodeOutputs(op, {Y, Y_h}));
  } else if (numOutputs == 3) {
    RETURN_IF_ERR(assignNodeOutputs(op, {Y, Y_h, Y_c}));
  } else {
    RETURN_ERR("ONNX LSTM should have minimum 1 and maximum 3 outputs!");
  }
  return Error::success();
}

Error ONNXModelLoader::loadCmpEQ(const ONNX_NAMESPACE::NodeProto &op,
                                 const ArgumentDictionaryTy &dict) {
  NodeValue LHS;
  ASSIGN_VALUE_OR_RETURN_ERR(LHS, getNodeValueByName(op.input(0)));
  NodeValue RHS;
  ASSIGN_VALUE_OR_RETURN_ERR(RHS, getNodeValueByName(op.input(1)));

  Node *N = G_.createCmpEQ(loadOperatorName(op), LHS, RHS);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadCmpLTE(const ONNX_NAMESPACE::NodeProto &op,
                                  const ArgumentDictionaryTy &dict) {
  NodeValue LHS;
  ASSIGN_VALUE_OR_RETURN_ERR(LHS, getNodeValueByName(op.input(0)));
  NodeValue RHS;
  ASSIGN_VALUE_OR_RETURN_ERR(RHS, getNodeValueByName(op.input(1)));

  Node *N = G_.createCmpLTE(loadOperatorName(op), LHS, RHS);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadSelect(const ONNX_NAMESPACE::NodeProto &op,
                                  const ArgumentDictionaryTy &dict) {
  NodeValue Cond;
  ASSIGN_VALUE_OR_RETURN_ERR(Cond, getNodeValueByName(op.input(0)));
  NodeValue LHS;
  ASSIGN_VALUE_OR_RETURN_ERR(LHS, getNodeValueByName(op.input(1)));
  NodeValue RHS;
  ASSIGN_VALUE_OR_RETURN_ERR(RHS, getNodeValueByName(op.input(2)));

  auto shape = getShape<dim_t>(dict.at("shape"));

  auto outTy = G_.getParent()->uniqueType(LHS.getElementType(), shape);
  Node *N = G_.createSelect(loadOperatorName(op), outTy, Cond, LHS, RHS);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadQuantize(const ONNX_NAMESPACE::NodeProto &op,
                                    const ArgumentDictionaryTy &dict) {
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  float scale;
  ASSIGN_VALUE_OR_RETURN_ERR(scale, loadFloat(dict.at("scale")));
  unsigned_t offset;
  ASSIGN_VALUE_OR_RETURN_ERR(offset, loadInt(dict.at("offset")));
  std::string elemKindStr;
  ASSIGN_VALUE_OR_RETURN_ERR(elemKindStr, loadStr(dict.at("elem_kind")));

  ElemKind elemKind = Type::getElementKindFromName(elemKindStr);

  auto outDims = in.getType()->dims();
  auto outTy = G_.getParent()->uniqueType(elemKind, outDims, scale, offset);
  Node *N = G_.createQuantize(loadOperatorName(op), in, outTy);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadConvertTo(const ONNX_NAMESPACE::NodeProto &op,
                                     const ArgumentDictionaryTy &dict) {
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  const auto *attr = dict.at("shape");
  RETURN_ERR_IF_NOT(attr->has_t(),
                    "ConvertTo should have t() field as \"shape\"");
  const auto &t = attr->t();
  std::vector<dim_t> shape;
  for (const auto d : t.dims()) {
    shape.push_back(d);
  }

  auto type = ElemKind::FloatTy;
  RETURN_IF_ERR(onnxTensorDataTypeToElemKind(t.data_type(), &type));
  auto outTy = G_.getParent()->uniqueType(type, shape);
  Node *N = G_.createConvertTo(loadOperatorName(op), in, outTy);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadDequantize(const ONNX_NAMESPACE::NodeProto &op,
                                      const ArgumentDictionaryTy &dict) {
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  Node *N = G_.createDequantize(loadOperatorName(op), in);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadRegression(const ONNX_NAMESPACE::NodeProto &op,
                                      const ArgumentDictionaryTy &dict) {
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  NodeValue expected;
  ASSIGN_VALUE_OR_RETURN_ERR(expected, getNodeValueByName(op.input(1)));

  Node *N = G_.createRegression(loadOperatorName(op), in, expected);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadBatchedAdd(const ONNX_NAMESPACE::NodeProto &op,
                                      const ArgumentDictionaryTy &dict) {
  NodeValue batch;
  ASSIGN_VALUE_OR_RETURN_ERR(batch, getNodeValueByName(op.input(0)));
  NodeValue sample;
  ASSIGN_VALUE_OR_RETURN_ERR(sample, getNodeValueByName(op.input(1)));

  Node *N = G_.createBatchedAdd(loadOperatorName(op), batch, sample);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadCumSum(const ONNX_NAMESPACE::NodeProto &op,
                                  const ArgumentDictionaryTy &dict) {
  if (op.input_size() > 1) {
    Expected<NodeValue> axis = getNodeValueByName(op.input(1));
    if (axis) {
      if (auto *AC = llvm::dyn_cast<Constant>(axis->getNode())) {
        RETURN_ERR_IF_NOT(AC->getPayload().dims().size() == 1,
                          "CumSum axis must be 0-D");
        RETURN_ERR_IF_NOT(AC->getPayload().dims()[0] == 1,
                          "CumSum axis must be 0-D");
        RETURN_ERR_IF_NOT(AC->getHandle<int32_t>().at(0) == 0,
                          "CumSum only supports axis == 0");
      } else {
        RETURN_ERR("Axis must be Constant");
      }

      // Axis default is 0, which is fine.
    }
  }

  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));
  bool exclusive = false;
  if (dict.count("exclusive")) {
    ASSIGN_VALUE_OR_RETURN_ERR(exclusive, loadInt(dict.at("exclusive")));
  }

  bool reverse = false;
  if (dict.count("reverse")) {
    ASSIGN_VALUE_OR_RETURN_ERR(reverse, loadInt(dict.at("reverse")));
  }

  Node *N = G_.createCumSum(loadOperatorName(op), input, exclusive, reverse);
  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadScatterAssign(const ONNX_NAMESPACE::NodeProto &op,
                                         const ArgumentDictionaryTy &dict) {
  NodeValue data;
  ASSIGN_VALUE_OR_RETURN_ERR(data, getNodeValueByName(op.input(0)));
  NodeValue indices;
  ASSIGN_VALUE_OR_RETURN_ERR(indices, getNodeValueByName(op.input(1)));
  NodeValue slices;
  ASSIGN_VALUE_OR_RETURN_ERR(slices, getNodeValueByName(op.input(2)));

  Node *N = G_.createScatterData(loadOperatorName(op), data, indices, slices);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadIntLookupTable(const ONNX_NAMESPACE::NodeProto &op,
                                          const ArgumentDictionaryTy &dict) {
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  auto values = getShape<int8_t>(dict.at("values"));
  auto shape = getShape<dim_t>(dict.at("shape"));

  auto outTy = G_.getParent()->uniqueType(in.getElementType(), shape);
  Node *N = G_.createIntLookupTable(loadOperatorName(op), in, values, outTy);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadLengthsRangeFill(const ONNX_NAMESPACE::NodeProto &op,
                                            const ArgumentDictionaryTy &dict) {
  NodeValue lengths;
  ASSIGN_VALUE_OR_RETURN_ERR(lengths, getNodeValueByName(op.input(0)));
  unsigned_t size;
  ASSIGN_VALUE_OR_RETURN_ERR(size, loadInt(dict.at("size")));

  Node *N = G_.createLengthsRangeFill(loadOperatorName(op), lengths, size);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadRescaleQuantized(const ONNX_NAMESPACE::NodeProto &op,
                                            const ArgumentDictionaryTy &dict) {
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  float scale;
  ASSIGN_VALUE_OR_RETURN_ERR(scale, loadFloat(dict.at("scale")));
  unsigned_t offset;
  ASSIGN_VALUE_OR_RETURN_ERR(offset, loadInt(dict.at("offset")));

  auto inTy = in.getType();
  auto outTy = G_.getParent()->uniqueType(inTy->getElementType(), inTy->dims(),
                                          scale, offset);

  Node *N = G_.createRescaleQuantized(loadOperatorName(op), in, outTy);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadRowwiseQuantizedSparseLengthsWeightedSum(
    const ONNX_NAMESPACE::NodeProto &op, const ArgumentDictionaryTy &dict) {
  Constant *data;
  ASSIGN_VALUE_OR_RETURN_ERR(data, getConstantByName(op.input(0)));
  Constant *scales;
  ASSIGN_VALUE_OR_RETURN_ERR(scales, getConstantByName(op.input(1)));
  Constant *offsets;
  ASSIGN_VALUE_OR_RETURN_ERR(offsets, getConstantByName(op.input(2)));
  NodeValue weights;
  ASSIGN_VALUE_OR_RETURN_ERR(weights, getNodeValueByName(op.input(3)));
  NodeValue indices;
  ASSIGN_VALUE_OR_RETURN_ERR(indices, getNodeValueByName(op.input(4)));
  NodeValue lengths;
  ASSIGN_VALUE_OR_RETURN_ERR(lengths, getNodeValueByName(op.input(5)));

  Node *N = G_.createRowwiseQuantizedSparseLengthsWeightedSum(
      loadOperatorName(op), data, scales, offsets, weights, indices, lengths);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadFusedRowwiseQuantizedSparseLengthsWeightedSum(
    const ONNX_NAMESPACE::NodeProto &op, const ArgumentDictionaryTy &dict) {
  NodeValue data;
  ASSIGN_VALUE_OR_RETURN_ERR(data, getNodeValueByName(op.input(0)));
  NodeValue weights;
  ASSIGN_VALUE_OR_RETURN_ERR(weights, getNodeValueByName(op.input(1)));
  NodeValue indices;
  ASSIGN_VALUE_OR_RETURN_ERR(indices, getNodeValueByName(op.input(2)));
  NodeValue lengths;
  ASSIGN_VALUE_OR_RETURN_ERR(lengths, getNodeValueByName(op.input(3)));

  Node *N = G_.createFusedRowwiseQuantizedSparseLengthsWeightedSum(
      loadOperatorName(op), data, weights, indices, lengths);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadFusedRowwiseQuantizedSparseLengthsSum(
    const ONNX_NAMESPACE::NodeProto &op,
    const ArgumentDictionaryTy & /* unused */) {
  NodeValue data;
  ASSIGN_VALUE_OR_RETURN_ERR(data, getNodeValueByName(op.input(0)));
  NodeValue indices;
  ASSIGN_VALUE_OR_RETURN_ERR(indices, getNodeValueByName(op.input(1)));
  NodeValue lengths;
  ASSIGN_VALUE_OR_RETURN_ERR(lengths, getNodeValueByName(op.input(2)));

  Storage *dataS = llvm::dyn_cast<Storage>(data);
  Node *N = G_.createFusedRowwiseQuantizedSparseLengthsSum(
      loadOperatorName(op), dataS, indices, lengths);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadFullyConnected(const ONNX_NAMESPACE::NodeProto &op,
                                          const ArgumentDictionaryTy &dict) {
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  Constant *W;
  ASSIGN_VALUE_OR_RETURN_ERR(W, getConstantByName(op.input(1)));
  Constant *B = getConstantByNameOrNull(op.input(2));
  NodeValue b;
  if (!B) {
    ASSIGN_VALUE_OR_RETURN_ERR(b, getNodeValueByName(op.input(2)));
  }

  unsigned_t axis = 1;
  if (dict.count("axis")) {
    ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict.at("axis")));
  }

  Node *N =
      G_.createFullyConnected(loadOperatorName(op), in, W, B ? B : b, axis);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadNonMaxSuppression(
    const ONNX_NAMESPACE::NodeProto &op, const ArgumentDictionaryTy &dict,
    bool isV4) {
  NodeValue boxesNV;
  ASSIGN_VALUE_OR_RETURN_ERR(boxesNV, getNodeValueByName(op.input(0)));
  NodeValue scoresNV;
  ASSIGN_VALUE_OR_RETURN_ERR(scoresNV, getNodeValueByName(op.input(1)));
  Constant *maxOutputBoxesPerClassC = getConstantByNameOrNull(op.input(2));
  Constant *iouThresholdC = getConstantByNameOrNull(op.input(3));
  Constant *scoreThresholdC = getConstantByNameOrNull(op.input(4));

  // Defaults to 0 which is the same representation as TF.
  unsigned centerPointBox = 0;
  if (dict.count("center_point_box")) {
    ASSIGN_VALUE_OR_RETURN_ERR(centerPointBox,
                               loadInt(dict.at("center_point_box")));
  }

  int32_t padToMaxOutputSize = 0;
  if (isV4) {
    if (dict.count("pad_to_max_output_size")) {
      ASSIGN_VALUE_OR_RETURN_ERR(padToMaxOutputSize,
                                 loadInt(dict.at("pad_to_max_output_size")));
    }

    // Does it make sense within GLOW context to have no padding? Since Size has
    // to be compile time constant.
    RETURN_ERR_IF_NOT(padToMaxOutputSize == 1,
                      "NonMaxSuppressionV4 does not support non-padding mode.");
  }

  unsigned maxOutputBoxesPerClass = 0;
  float iouThreshold = 0.0f;
  float scoreThreshold = 0.0f;

  if (maxOutputBoxesPerClassC) {
    if (maxOutputBoxesPerClassC->getPayload().getElementType() ==
        ElemKind::Int64ITy) {
      maxOutputBoxesPerClass =
          maxOutputBoxesPerClassC->getPayload().getHandle<int64_t>().raw(0);
    } else if (maxOutputBoxesPerClassC->getPayload().getElementType() ==
               ElemKind::Int32ITy) {
      maxOutputBoxesPerClass =
          maxOutputBoxesPerClassC->getPayload().getHandle<int32_t>().raw(0);
    } else {
      RETURN_ERR("Unsupported type for maxoutputboxesperclass.");
    }
  } else {
    RETURN_ERR("NMS: maxOutputBoxesPerClass is not a contant tensor.");
  }

  if (iouThresholdC) {
    iouThreshold = iouThresholdC->getPayload().getHandle<float>().raw(0);
  } else {
    RETURN_ERR("NMS: iouThreshold is not a contant tensor.");
  }

  if (scoreThresholdC) {
    scoreThreshold = scoreThresholdC->getPayload().getHandle<float>().raw(0);
  } else {
    RETURN_ERR("NMS: scoreThrehold is not a contant tensor.");
  }

  // Create Node.
  std::string opName = loadOperatorName(op);
  Node *N = nullptr;

  if (isV4) {
    N = G_.createNonMaxSuppressionV4(opName, boxesNV, scoresNV, centerPointBox,
                                     maxOutputBoxesPerClass, iouThreshold,
                                     scoreThreshold);
  } else {
    N = G_.createNonMaxSuppressionONNX(opName, boxesNV, scoresNV,
                                       centerPointBox, maxOutputBoxesPerClass,
                                       iouThreshold, scoreThreshold);
  }
  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadSplat(const ONNX_NAMESPACE::NodeProto &op,
                                 const ArgumentDictionaryTy &dict) {
  return loadConstantOfShape(op, dict, true /* isSplat */);
}

Error ONNXModelLoader::loadInsertTensor(const ONNX_NAMESPACE::NodeProto &op,
                                        const ArgumentDictionaryTy &dict) {
  NodeValue big;
  ASSIGN_VALUE_OR_RETURN_ERR(big, getNodeValueByName(op.input(0)));
  NodeValue small;
  ASSIGN_VALUE_OR_RETURN_ERR(small, getNodeValueByName(op.input(1)));

  auto start = getShape<dim_t>(dict.at("start"));

  unsigned_t count = 1;
  if (dict.count("count")) {
    ASSIGN_VALUE_OR_RETURN_ERR(count, loadInt(dict.at("count")));
  }

  unsigned_t axis = 0;
  if (dict.count("axis")) {
    ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict.at("axis")));
  }

  Node *N = G_.createInsertTensor(loadOperatorName(op), big, small, start,
                                  count, axis);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadIdentity(const ONNX_NAMESPACE::NodeProto &op,
                                    const ArgumentDictionaryTy &dict) {
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  RETURN_IF_ERR(addNodeAsOutput(op, in));
  return Error::success();
}

Error ONNXModelLoader::loadAdaptiveAvgPool(const ONNX_NAMESPACE::NodeProto &op,
                                           const ArgumentDictionaryTy &dict) {
  // Glow expects inputs to be in NHWC but ONNX keeps them in NCHW so we
  // transpose them.
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));

  input = G_.createTranspose("adaptive_avg_pool2d_input_transposed", input,
                             NCHW2NHWC);

  // OutputSize defaults to size of input if not provided.
  std::vector<dim_t> outputSize;
  if (dict.count("output_size")) {
    outputSize = getShape<dim_t>(dict.at("output_size"));
  } else {
    outputSize = {input.dims()[2], input.dims()[3]};
  }

  auto idim = glow::ShapeNHWC(input.dims());
  auto outTy = G_.getParent()->uniqueTypeWithNewShape(
      input.getType(), {idim.n, outputSize[0], outputSize[1], idim.c});

  Node *N = G_.createAdaptiveAvgPool("adaptive_avg_pool2d", input, outTy);
  N = G_.createTranspose("adaptive_avg_pool2d_output_transposed", N, NHWC2NCHW);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadFlip(const ONNX_NAMESPACE::NodeProto &op,
                                const ArgumentDictionaryTy &dict) {
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));

  unsigned_t axis = 0;
  if (dict.count("axis")) {
    ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict.at("axis")));
  }

  Node *N = G_.createFlip("flip", input, axis);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadRowwiseQuantizedFullyConnected(
    const ONNX_NAMESPACE::NodeProto &op, const ArgumentDictionaryTy &dict) {
  // TODO
  RETURN_ERR("Not implemented.");
}

Error ONNXModelLoader::loadOperator(const ONNX_NAMESPACE::NodeProto &op) {
  ArgumentDictionaryTy dict = loadArgumentMap(op);
  const std::string &typeName = op.op_type();

  // Check if operator is supported in parent class, CommonOperatorLoader.
  bool tryLoadCommonOperatorResult;
  ASSIGN_VALUE_OR_RETURN_ERR(tryLoadCommonOperatorResult,
                             tryLoadCommonOperator(typeName, op, dict));
  if (tryLoadCommonOperatorResult) {
    return Error::success();
  }

  if (typeName == "Constant") {
    return loadConstant(op, dict);
  }
  if (typeName == "Slice") {
    return loadSlice(op, dict);
  }
  if (typeName == "Conv") {
    // If the Conv operator has quantized inputs, use
    // loadTensorwiseQuantizedConvolution.
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    return in.getType()->isQuantizedType()
               ? loadTensorwiseQuantizedConvolution(op, dict)
               : loadConv(op, dict);
  }
  if (typeName == "ChannelwiseQuantizedConvolution") {
    return loadChannelwiseQuantizedConvolution(op, dict);
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
    return loadConstantOfShape(op, dict, false /* isSplat */);
  }
  if (typeName == "Tile") {
    return loadTile(op, dict);
  }
  if (typeName == "Where") {
    return loadWhere(op, dict);
  }
  if (typeName == "RNN") {
    return loadRNN(op, dict);
  }
  if (typeName == "GRU") {
    return loadGRU(op, dict);
  }
  if (typeName == "LSTM") {
    return loadLSTM(op, dict);
  }
  // Glow specific operators
  if (typeName == "CmpEQ") {
    return loadCmpEQ(op, dict);
  }
  if (typeName == "CmpLTE") {
    return loadCmpLTE(op, dict);
  }
  if (typeName == "Select") {
    return loadSelect(op, dict);
  }
  if (typeName == "Quantize") {
    return loadQuantize(op, dict);
  }
  if (typeName == "ConvertTo") {
    return loadConvertTo(op, dict);
  }
  if (typeName == "Dequantize") {
    return loadDequantize(op, dict);
  }
  if (typeName == "Regression") {
    return loadRegression(op, dict);
  }
  if (typeName == "BatchedAdd") {
    return loadBatchedAdd(op, dict);
  }
  if (typeName == "CumSum") {
    return loadCumSum(op, dict);
  }
  if (typeName == "ScatterAssign") {
    return loadScatterAssign(op, dict);
  }
  if (typeName == "IntLookupTable") {
    return loadIntLookupTable(op, dict);
  }
  if (typeName == "LengthsRangeFill") {
    return loadLengthsRangeFill(op, dict);
  }
  if (typeName == "RescaleQuantized") {
    return loadRescaleQuantized(op, dict);
  }
  if (typeName == "RowwiseQuantizedSparseLengthsWeightedSum") {
    return loadRowwiseQuantizedSparseLengthsWeightedSum(op, dict);
  }
  if (typeName == "FusedRowwiseQuantizedSparseLengthsWeightedSum") {
    return loadFusedRowwiseQuantizedSparseLengthsWeightedSum(op, dict);
  }
  if (typeName == "FusedRowwiseQuantizedSparseLengthsSum") {
    return loadFusedRowwiseQuantizedSparseLengthsSum(op, dict);
  }
  if (typeName == "FullyConnected") {
    return loadFullyConnected(op, dict);
  }
  if (typeName == "RowwiseQuantizedFullyConnected") {
    return loadRowwiseQuantizedFullyConnected(op, dict);
  }
  if (typeName == "Splat") {
    return loadSplat(op, dict);
  }
  if (typeName == "InsertTensor") {
    return loadInsertTensor(op, dict);
  }
  if (typeName == "ArgMax") {
    return loadArgMax(op, dict);
  }
  if (typeName == "NonMaxSuppressionV4") {
    return loadNonMaxSuppression(op, dict, true);
  }
  if (typeName == "NonMaxSuppression") {
    return loadNonMaxSuppression(op, dict, false);
  }
  if (typeName == "AdaptiveAvgPool") {
    return loadAdaptiveAvgPool(op, dict);
  }
  if (typeName == "Flip") {
    return loadFlip(op, dict);
  }
  if (typeName == "Identity") {
    return loadIdentity(op, dict);
  }

  RETURN_ERR("Failed to load operator " + typeName + " .",
             ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_OPERATOR);
}

Error ONNXModelLoader::loadInitializers(ONNX_NAMESPACE::GraphProto &net) {
  // Load the network initializers:
  for (const auto &in : net.initializer()) {
    Tensor T;
    RETURN_IF_ERR(loadTensor(in, &T));
    RETURN_IF_ERR(createAndRegisterConstant(in.name(), std::move(T)));
  }
  return Error::success();
}

Error ONNXModelLoader::setOutputNodes(ONNX_NAMESPACE::GraphProto &net) {
  if (net.output_size() == 0) {
    RETURN_ERR("Net output size must be greater than 0");
  }

  for (int i = 0; i < net.output_size(); i++) {
    const auto &outputName = net.output(i).name();
    NodeValue r;
    ASSIGN_VALUE_OR_RETURN_ERR(r, getNodeValueByName(outputName));

    Placeholder *placeholder =
        G_.getParent()->createPlaceholder(r.getType(), outputName, false);
    SaveNode *SN = G_.createSave("save_" + outputName, r, placeholder);
    outputVarsByName_[outputName] = SN->getPlaceholder();
  }

  return Error::success();
}

Error ONNXModelLoader::loadNetwork(ONNX_NAMESPACE::GraphProto &net) {
  /// Load the network operators:
  for (int i = 0; i < net.node_size(); i++) {
    auto &op = net.node(i);
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

  return Error::success();
}

ONNXModelLoader::ONNXModelLoader(Function &F, Error *errPtr)
    : CommonOperatorLoader({}, {}, F, errPtr) {
  deleteUnusedConstants();
}

Error ONNXModelLoader::collectStaticInputs(ONNX_NAMESPACE::GraphProto &net) {
  for (int i = 0; i < net.input_size(); i++) {
    const ONNX_NAMESPACE::ValueInfoProto &valueInfo = net.input(i);
    const std::string &inputName = valueInfo.name();
    if (valueInfo.has_doc_string() && valueInfo.doc_string() == "offline") {
      staticInputs_.emplace(inputName);
    }
  }
  return Error::success();
}

Error ONNXModelLoader::checkInputs(ONNX_NAMESPACE::GraphProto &net,
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

      // Get tensor shape.
      llvm::ArrayRef<dim_t> dims = types[i]->dims();

      // Get proto shape.
      std::vector<dim_t> dimsProto;
      ASSIGN_VALUE_OR_RETURN_ERR(
          dimsProto, getProtoShape(valueInfo.type().tensor_type().shape()));

      // Check if the number of dimensions is consistent.
      RETURN_ERR_IF_NOT(dims.size() == dimsProto.size(),
                        "Mismatch between input image and ONNX input shape");

      // Allow batch dimensions to be different.
      for (size_t k = 1; k < dims.size(); k++) {
        RETURN_ERR_IF_NOT(dims[k] == dimsProto[k],
                          "Mismatch between input image and ONNX input shape");
      }

      if (valueInfo.has_doc_string() && valueInfo.doc_string() == "offline") {
        staticInputs_.emplace(inputName);
      }
    }
  }
  return Error::success();
}

ONNXModelLoader::ONNXModelLoader(const std::string &modelDescFilename,
                                 llvm::ArrayRef<const char *> tensorNames,
                                 llvm::ArrayRef<TypeRef> types, Function &F,
                                 Error *errPtr, bool zipMode)
    : CommonOperatorLoader(tensorNames, types, F, errPtr) {
  // if errPtr already contains an error then don't continue with constructor
  if (errPtr && *errPtr) {
    return;
  }

  // Lambda to setup the ONNXModelLoader and return any Errors that were
  // raised.
  auto setup = [&]() -> Error {
    // The ONNX model that we are deserializing.
    ONNX_NAMESPACE::ModelProto modelDef;
    if (zipMode) {
      ZipReader zip(modelDescFilename);
      std::string buffer;
      buffer = zip.getRecord("model");
      modelDef.ParseFromString(buffer);
      size_t numWeights = 0;
      auto numWeightsStr = zip.getRecord("weights");
      numWeights = atoi(numWeightsStr.c_str());
      for (size_t i = 0; i < numWeights; ++i) {
        std::stringstream ss;
        ss << "weight_" << i;
        buffer = zip.getRecord(ss.str());
        auto *t = modelDef.mutable_graph()->add_initializer();
        t->ParseFromString(buffer);
      }
    } else {
      ASSIGN_VALUE_OR_RETURN_ERR(modelDef, loadProto(modelDescFilename));
    }

    RETURN_IF_ERR(setVersion(modelDef));

    ONNX_NAMESPACE::GraphProto graphDef = modelDef.graph();
    RETURN_IF_ERR(checkInputs(graphDef, tensorNames, types));
    RETURN_IF_ERR(collectStaticInputs(graphDef));

    RETURN_IF_ERR(loadInitializers(graphDef));

    if (tensorNames.empty() && types.empty()) {
      // Detect inputs without initializers and create placeholders.
      RETURN_IF_ERR(loadInputs(graphDef, /* loadInputsAsPlaceholders */ true));
    }

    RETURN_IF_ERR(loadNetwork(graphDef));

    RETURN_IF_ERR(setOutputNodes(graphDef));

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

ONNXModelLoader::ONNXModelLoader(
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

  // Lambda to setup the ONNXModelLoader and return any Errors that were
  // raised.
  auto setup = [&]() -> Error {
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

    return Error::success();
  };

  if (errPtr) {
    *errPtr = setup();
  } else {
    EXIT_ON_ERR(setup());
  }
}
