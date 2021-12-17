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
#include "glow/Flags/Flags.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Support/Support.h"
#include "glow/Support/ZipUtils.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/tokenizer.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace glow;
using namespace glow::runtime;
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

llvm::cl::opt<bool> onnxExportRnnStatesOpt(
    "onnx-export-rnn-states", llvm::cl::init(false), llvm::cl::Optional,
    llvm::cl::desc(
        "Option to export the states of the ONNX RNN operators (for example \n"
        "RNN, GRU, LSTM) as graph placeholders regardless of whether the    \n"
        "states are explicitly set or not in the graph. The placeholders are\n"
        "also providing an automatic way for tracking the RNN states since  \n"
        "the states are updated automatically with the new RNN states after \n"
        "each inference. Default is false."),
    llvm::cl::cat(onnxModelLoaderCat));

llvm::cl::opt<unsigned> loopUnrollLimit(
    "loop-unroll-limit",
    llvm::cl::desc("Maximum unrollable iterations for the Loop operator"),
    llvm::cl::Optional, llvm::cl::init(20), llvm::cl::cat(onnxModelLoaderCat));

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
      if (symbolMap.count(symbolName) && symbolMap[symbolName] > 0) {
        dim.push_back(symbolMap[symbolName]);
      } else {
        return MAKE_ERR(strFormat(
            "ONNX model symbol '%s' is undefined. Define the symbol with the "
            "following command line option: -onnx-define-symbol=%s,<value> and "
            "each 'dim_value' of tensor shape proto must be greater than 0.",
            symbolName.c_str(), symbolName.c_str()));
      }
    } else {
      // Proto shape has no "dim_value" and no "dim_param" field.
      return MAKE_ERR(
          "Tensor shape proto has no 'dim_value' or 'dim_param' field!");
    }
  }
  return dim;
}

/// Given some \p onnxType, sets \p elemTy to a corresponding Glow
/// ElemKind. \returns whether an ElemKind was successfully selected.
Error onnxTensorDataTypeToElemKind(int32_t onnxType, ElemKind *elemTy) {
  if (onnxType == ONNX_NAMESPACE::TensorProto::FLOAT) {
    *elemTy = ElemKind::FloatTy;
    return Error::success();
  } else if (onnxType == ONNX_NAMESPACE::TensorProto::FLOAT16) {
    *elemTy = ElemKind::Float16Ty;
    return Error::success();
  } else if (onnxType == ONNX_NAMESPACE::TensorProto::BFLOAT16) {
    *elemTy = ElemKind::BFloat16Ty;
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
  } else if (onnxType == ONNX_NAMESPACE::TensorProto::INT8) {
    *elemTy = ElemKind::Int8QTy;
    return Error::success();
  } else if (onnxType == ONNX_NAMESPACE::TensorProto::INT16) {
    *elemTy = ElemKind::Int16QTy;
    return Error::success();
  } else if (onnxType == ONNX_NAMESPACE::TensorProto::BOOL) {
    *elemTy = ElemKind::BoolTy;
    return Error::success();
  } else {
    return MAKE_ERR(strFormat(
        "Don't know how to convert ONNX tensor data type %d to ElemKind",
        onnxType));
  }
}

/// Finds an attribute from the doc_string and \returns it. If it does not exist
/// then \returns Error. The expected structure here is that each attribute
/// starts with startChar and is separated from its value by a sepChar.
Expected<std::string> getAttrFromDocString(const std::string &attr,
                                           const std::string &docStr) {
  const std::string attrAndSep = attr + sepChar;
  size_t begin = 0;
  while (true) {
    begin = docStr.find(startChar, begin);
    if (begin == std::string::npos) {
      return MAKE_ERR(strFormat("Didn't find PH attribute '%s'", attr.c_str()));
    }

    // Note: +1 here and following line to account for the leading startChar.
    if (!docStr.compare(begin + 1, attrAndSep.size(), attrAndSep)) {
      // If we found the attribute then set begin to just after attrAndSep.
      begin += attrAndSep.size() + 1;
      break;
    }
    // Move past the current non-matching attribute to try the next attribute.
    begin = begin + attrAndSep.size();
  }

  return docStr.substr(begin, docStr.find(startChar, begin) - begin);
}

Expected<std::pair<bool, std::string>>
getTrainableLayoutPairFromDocString(const std::string &docString,
                                    bool useGlowCustomOps) {
  std::string layout = ANY_LAYOUT;
  std::string isTrainableStr = "0";
  if (useGlowCustomOps) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        isTrainableStr, getAttrFromDocString(trainableSignifier, docString));
    ASSIGN_VALUE_OR_RETURN_ERR(
        layout, getAttrFromDocString(layoutSignifier, docString));
  }
  return std::make_pair(isTrainableStr != "0", layout);
}

Expected<std::pair<float, int32_t>>
getQuantParamsFromDocString(const std::string &docStr) {
  std::string scaleStr;
  ASSIGN_VALUE_OR_RETURN_ERR(scaleStr,
                             getAttrFromDocString(qScaleSignifier, docStr));
  float scale = std::strtof(scaleStr.c_str(), NULL);

  std::string offsetStr;
  ASSIGN_VALUE_OR_RETURN_ERR(offsetStr,
                             getAttrFromDocString(qOffsetSignifier, docStr));
  int32_t offset;
  ASSIGN_VALUE_OR_RETURN_ERR(offset, getIntFromStr(offsetStr));
  return std::make_pair(scale, offset);
}

ShapeVector getStridesFromDocString(const std::string &docStr) {
  ShapeVector strides;
  std::string stridesStr;
  auto stridesOrError = getAttrFromDocString(stridesSignifier, docStr);
  if (ERR_TO_BOOL(stridesOrError.takeError(), /* log */ false)) {
    return strides;
  }
  stridesStr = std::move(stridesOrError.get());
  if (stridesStr.empty()) {
    return strides;
  }
  // Parse comma-delimited stride values.
  llvm::SmallVector<llvm::StringRef, max_tensor_dimensions> stridesStrSplit;
  llvm::StringRef stridesStrRef = llvm::StringRef(stridesStr);
  stridesStrRef.split(stridesStrSplit, ',');
  for (const auto &stride : stridesStrSplit) {
    strides.emplace_back(std::stoi(stride.str()));
  }
  return strides;
}

/// Used for retrieving an attribute of type \p T from \p attr. Some
/// specializations used \p loader if necessary.
template <bool IsInteger, typename T> struct AttributeRetriever {
  static Expected<T> get(const ONNX_NAMESPACE::AttributeProto *attr,
                         const ProtobufLoader &loader);
};

/// Specialization for std::vector<float>.
template <> struct AttributeRetriever<false, std::vector<float>> {
  static Expected<std::vector<float>>
  get(const ONNX_NAMESPACE::AttributeProto *attr,
      const ProtobufLoader & /* unused */) {
    return getFloats(attr);
  }
};

/// Specialization for std::vector<NodeValue>.
template <> struct AttributeRetriever<false, std::vector<NodeValue>> {
  static Expected<std::vector<NodeValue>>
  get(const ONNX_NAMESPACE::AttributeProto *attr, ProtobufLoader &loader) {
    // Retrieve the names from the proto which map to NodeValues.
    std::vector<std::string> strs;
    ASSIGN_VALUE_OR_RETURN_ERR(strs, getStrings(attr));

    // Get NodeValues corresponding to these names from the loader.
    std::vector<NodeValue> NVs;
    for (const auto &str : strs) {
      NodeValue NV;
      ASSIGN_VALUE_OR_RETURN_ERR(NV, loader.getNodeValueByName(str));
      NVs.push_back(NV);
    }
    return NVs;
  }
};

/// Specialization for NodeValue.
template <> struct AttributeRetriever<false, NodeValue> {
  static Expected<NodeValue> get(const ONNX_NAMESPACE::AttributeProto *attr,
                                 ProtobufLoader &loader) {
    // Retrieve the name from the proto, which is mapped to a NodeValue.
    std::string str;
    ASSIGN_VALUE_OR_RETURN_ERR(str, loadStr(attr));

    // Get/return the corresponding NodeValue for this name from the loader.
    NodeValue NV;
    ASSIGN_VALUE_OR_RETURN_ERR(NV, loader.getNodeValueByName(str));
    return NV;
  }
};

/// Specialization for std::vector<T>. Fall back for integer types.
template <typename T> struct AttributeRetriever<false, std::vector<T>> {
  static Expected<std::vector<T>>
  get(const ONNX_NAMESPACE::AttributeProto *attr,
      const ProtobufLoader & /* unused */) {
    return getShape<T>(attr, /* allowEmptyShape */ true);
  }
};

/// Specialization for integer types.
template <typename T> struct AttributeRetriever<true, T> {
  static Expected<T> get(const ONNX_NAMESPACE::AttributeProto *attr,
                         const ProtobufLoader & /* unused */) {
    return loadInt(attr);
  }
};

/// Specialization for LengthsMode.
template <> struct AttributeRetriever<false, LengthsMode> {
  static Expected<LengthsMode> get(const ONNX_NAMESPACE::AttributeProto *attr,
                                   const ProtobufLoader & /* unused */) {
    std::string str;
    ASSIGN_VALUE_OR_RETURN_ERR(str, loadStr(attr));
    if (str == "AllOne") {
      return LengthsMode::AllOne;
    } else if (str == "Variable") {
      return LengthsMode::Variable;
    } else {
      return MAKE_ERR("Invalid LengthsMode");
    }
  }
};

/// Specialization for FusedActivation.
template <> struct AttributeRetriever<false, FusedActivation> {
  static Expected<FusedActivation>
  get(const ONNX_NAMESPACE::AttributeProto *attr,
      const ProtobufLoader & /* unused */) {
    std::string str;
    ASSIGN_VALUE_OR_RETURN_ERR(str, loadStr(attr));
    if (str == "NONE") {
      return FusedActivation::NONE;
    } else if (str == "RELU") {
      return FusedActivation::RELU;
    } else if (str == "CLIP") {
      return FusedActivation::CLIP;
    } else if (str == "TANH") {
      return FusedActivation::TANH;
    } else if (str == "SIGMOID") {
      return FusedActivation::SIGMOID;
    } else if (str == "LEAKY_RELU") {
      return FusedActivation::LEAKY_RELU;
    } else {
      return MAKE_ERR("Invalid FusedActivation");
    }
  }
};

/// Specialization for FusedActivation.
template <> struct AttributeRetriever<false, LUTOperator> {
  static Expected<LUTOperator> get(const ONNX_NAMESPACE::AttributeProto *attr,
                                   const ProtobufLoader & /* unused */) {
    std::string str;
    ASSIGN_VALUE_OR_RETURN_ERR(str, loadStr(attr));
    if (str == "NONE") {
      return LUTOperator::NONE;
    } else if (str == "RELU") {
      return LUTOperator::RELU;
    } else if (str == "CLIP") {
      return LUTOperator::CLIP;
    } else if (str == "TANH") {
      return LUTOperator::TANH;
    } else if (str == "SIGMOID") {
      return LUTOperator::SIGMOID;
    } else if (str == "LEAKY_RELU") {
      return LUTOperator::LEAKY_RELU;
    } else {
      return MAKE_ERR("Invalid LUTOperator");
    }
  }
};

/// Specialization for ConvolutionLayout.
template <> struct AttributeRetriever<false, ConvolutionLayout> {
  static Expected<ConvolutionLayout>
  get(const ONNX_NAMESPACE::AttributeProto *attr,
      const ProtobufLoader & /* unused */) {
    std::string str;
    ASSIGN_VALUE_OR_RETURN_ERR(str, loadStr(attr));
    if (str == "NHWC") {
      return ConvolutionLayout::NHWC;
    } else if (str == "NCHW") {
      return ConvolutionLayout::NCHW;
    } else if (str == "NTHWC") {
      return ConvolutionLayout::NTHWC;
    } else if (str == "NCTHW") {
      return ConvolutionLayout::NCTHW;
    } else {
      return MAKE_ERR("Invalid ConvolutionLayout");
    }
  }
};

/// Specialization for PaddingMode.
template <> struct AttributeRetriever<false, PaddingMode> {
  static Expected<PaddingMode> get(const ONNX_NAMESPACE::AttributeProto *attr,
                                   const ProtobufLoader & /* unused */) {
    std::string str;
    ASSIGN_VALUE_OR_RETURN_ERR(str, loadStr(attr));
    if (str == "CONSTANT") {
      return PaddingMode::CONSTANT;
    } else if (str == "REFLECT") {
      return PaddingMode::REFLECT;
    } else if (str == "EDGE") {
      return PaddingMode::EDGE;
    } else {
      return MAKE_ERR("Invalid PaddingMode");
    }
  }
};

/// Specialization for SplitEmbeddingPoolingMode.
template <> struct AttributeRetriever<false, SplitEmbeddingPoolingMode> {
  static Expected<SplitEmbeddingPoolingMode>
  get(const ONNX_NAMESPACE::AttributeProto *attr,
      const ProtobufLoader & /* unused */) {
    std::string poolingMode;
    ASSIGN_VALUE_OR_RETURN_ERR(poolingMode, loadStr(attr));
    if (poolingMode == "0") {
      return SplitEmbeddingPoolingMode::EP_SUM;
    } else if (poolingMode == "1") {
      return SplitEmbeddingPoolingMode::EP_MEAN;
    } else if (poolingMode == "2") {
      return SplitEmbeddingPoolingMode::EP_NONE;
    } else {
      return MAKE_ERR("Invalid SplitEmbeddingPoolingMode");
    }
  }
};

/// Specialization for SplitEmbeddingSparseType.
template <> struct AttributeRetriever<false, SplitEmbeddingSparseType> {
  static Expected<SplitEmbeddingSparseType>
  get(const ONNX_NAMESPACE::AttributeProto *attr,
      const ProtobufLoader & /* unused */) {
    std::string str;
    ASSIGN_VALUE_OR_RETURN_ERR(str, loadStr(attr));
    if (str == "0") {
      return SplitEmbeddingSparseType::EST_FLOAT;
    } else if (str == "1") {
      return SplitEmbeddingSparseType::EST_FLOAT16;
    } else if (str == "2") {
      return SplitEmbeddingSparseType::EST_INT8;
    } else if (str == "3") {
      return SplitEmbeddingSparseType::EST_INT4;
    } else if (str == "4") {
      return SplitEmbeddingSparseType::EST_INT2;
    } else {
      return MAKE_ERR("Invalid SplitEmbeddingSparseType");
    }
  }
};

/// Specialization for float.
template <> struct AttributeRetriever<false, float> {
  static Expected<float> get(const ONNX_NAMESPACE::AttributeProto *attr,
                             const ProtobufLoader & /* unused */) {
    return loadFloat(attr);
  }
};

/// Specialization for std::string.
template <> struct AttributeRetriever<false, std::string> {
  static Expected<std::string> get(const ONNX_NAMESPACE::AttributeProto *attr,
                                   const ProtobufLoader & /* unused */) {
    return loadStr(attr);
  }
};

/// Forwards to the correct AttributeRetriever specialization.
template <typename T>
Expected<T> loadAttribute(const ONNX_NAMESPACE::AttributeProto *attr,
                          ProtobufLoader &loader) {
  RETURN_ERR_IF_NOT(attr, "No such attribute");
  return AttributeRetriever<std::numeric_limits<T>::is_integer, T>::get(attr,
                                                                        loader);
}

} // namespace

using ArgumentDictionaryTy =
    std::unordered_map<std::string, const ONNX_NAMESPACE::AttributeProto *>;

/// \returns a type based on \p ty, but using the provided \p strides.
static Type getTypeWithCustomStrides(const Type &ty,
                                     llvm::ArrayRef<dim_t> strides) {
  if (strides.empty()) {
    return ty;
  }
  return Type::newStrides(ty, strides);
}

/// \returns a type in module \p mod, based on \p ty, but using the provided \p
/// strides.
static TypeRef getTypeWithCustomStrides(glow::Module &mod, const TypeRef ty,
                                        llvm::ArrayRef<dim_t> strides) {
  if (strides.empty()) {
    return ty;
  }
  return mod.uniqueTypeWithNewStrides(ty, ty->dims(), strides);
}

/// Given a docstring encoding \p str of a type and its dimension \p
/// dims, parses the string and \returns a Glow Type from it or Error if
/// parsing failed. Expected format of str is either elemKindSignifier or
/// "ElemKind:scale:offset".
Expected<Type> parseTypeFromDocString(const std::string &str,
                                      llvm::ArrayRef<dim_t> dims,
                                      bool useGlowCustomOps) {
  float scale = 1.0;
  int32_t offset = 0;
  ElemKind elemKind = ElemKind::FloatTy;
  ShapeVector strides;

  if (useGlowCustomOps) {
    std::string elemKindStr;
    ASSIGN_VALUE_OR_RETURN_ERR(elemKindStr,
                               getAttrFromDocString(elemKindSignifier, str));
    elemKind = Type::getElementKindFromName(elemKindStr);

    if (isQuantizedElemKind(elemKind)) {
      std::pair<float, int32_t> scaleOffsetPair;
      ASSIGN_VALUE_OR_RETURN_ERR(scaleOffsetPair,
                                 getQuantParamsFromDocString(str));
      std::tie(scale, offset) = scaleOffsetPair;
    }
    strides = getStridesFromDocString(str);
  } else {
    size_t begin = 0;

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

    elemKind = Type::getElementKindFromName(elemKindStr);
  }

  Type ty;
  if (isQuantizedElemKind(elemKind)) {
    ty = Type(elemKind, dims, scale, offset);
  } else {
    ty = Type(elemKind, dims);
  }
  return getTypeWithCustomStrides(ty, strides);
}

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

ONNX_NAMESPACE::GraphProto glow::parseOnnxFile(const std::string &fileName) {
  ::ONNX_NAMESPACE::GraphProto graphProto;
  std::ifstream inputFileStream(fileName, std::ios::in | std::ios::binary);
  CHECK(inputFileStream) << "Can't find the input file for " << fileName;
  google::protobuf::io::IstreamInputStream protobufFileStream(&inputFileStream);
  google::protobuf::io::CodedInputStream codedStream(&protobufFileStream);
#if GOOGLE_PROTOBUF_VERSION >= 3002000
  codedStream.SetTotalBytesLimit(MAX_PROTO_SIZE);
#else
  codedStream.SetTotalBytesLimit(MAX_PROTO_SIZE, MAX_PROTO_SIZE);
#endif
  bool parsedSuccessfully = graphProto.ParseFromCodedStream(&codedStream);
  CHECK(parsedSuccessfully) << "Failed to parse GraphProto";
  return graphProto;
}

void glow::fillPlaceholders(const ONNX_NAMESPACE::GraphProto &inputGroup,
                            PlaceholderBindings *bindings,
                            std::vector<Tensor> *partialTensorPayloads,
                            bool usingGlowCustomOps) {
  for (const auto &tensorProto : inputGroup.initializer()) {
    const std::string glowLegalizedName =
        glow::legalizeName(tensorProto.name());
    auto *tensor =
        bindings->get(bindings->getPlaceholderByNameSlow(glowLegalizedName));
    CHECK(tensor) << "Missing " << tensorProto.name()
                  << ", Glow legalized name " << glowLegalizedName;
    size_t fullSize = tensor->getSizeInBytes();
    const auto fullType = tensor->getType();
    auto error = loadTensor(tensorProto, tensor, usingGlowCustomOps);
    bool hasError = ERR_TO_BOOL(std::move(error));
    CHECK(!hasError) << "Cannot load input tensor";
    size_t loadedSize = tensor->getSizeInBytes();
    if (loadedSize != fullSize) {
      if (partialTensorPayloads) {
        VLOG(1) << "Loading " << tensorProto.name() << ", Glow legalized name "
                << glowLegalizedName << " as a partial tensor: partial size="
                << tensor->getType().toString()
                << " full size=" << fullType.toString();
        Tensor fullTensor(tensor->getUnsafePtr(), &fullType,
                          tensor->getSizeInBytes());
        // 'fullTensor' doesn't own the underlying data. 'tensor' does. So
        // we want to keep the original tensor object around until inference
        // is finished.
        partialTensorPayloads->emplace_back(std::move(*tensor));
        *tensor = std::move(fullTensor);
      } else {
        // pad with 0
        VLOG(1) << "Loading and padding " << tensorProto.name()
                << ", Glow legalized name " << glowLegalizedName
                << " as a partial tensor: partial size="
                << tensor->getType().toString()
                << " full size=" << fullType.toString();
        Tensor fullTensor(&fullType);
        std::memcpy(fullTensor.getUnsafePtr(), tensor->getUnsafePtr(),
                    tensor->getSizeInBytes());
        std::memset(fullTensor.getUnsafePtr() + tensor->getSizeInBytes(), 0,
                    fullTensor.getSizeInBytes() - tensor->getSizeInBytes());
        *tensor = std::move(fullTensor);
      }
    }
  }
}

void glow::fillPlaceholders(const std::string &fileName,
                            PlaceholderBindings *bindings,
                            std::vector<Tensor> *partialTensorPayloads,
                            bool usingGlowCustomOps) {
  const ONNX_NAMESPACE::GraphProto &inputGroup = parseOnnxFile(fileName);
  fillPlaceholders(inputGroup, bindings, partialTensorPayloads,
                   usingGlowCustomOps);
}

/// Loads tensor \p T from the input \p in.
Error glow::loadTensor(const ONNX_NAMESPACE::TensorProto &in, Tensor *T,
                       bool useGlowCustomOps, const std::string &data) {
  std::vector<dim_t> dim;
  for (auto d : in.dims()) {
    dim.push_back(d);
  }

  ShapeVector strides;
  if (in.has_doc_string()) {
    strides = getStridesFromDocString(in.doc_string());
  }

  if (in.data_type() == ONNX_NAMESPACE::TensorProto::FLOAT) {
    Type ty(ElemKind::FloatTy, dim);
    T->reset(getTypeWithCustomStrides(ty, strides));

    if (in.float_data_size() > 0) {
      auto TH = T->getHandle<>();
      size_t i = 0;
      for (auto f : in.float_data()) {
        TH.raw(i++) = f;
      }
    } else if (in.has_raw_data() || !data.empty()) {
      std::istringstream inStream(data.empty() ? in.raw_data() : data,
                                  std::stringstream::binary);
      inStream.read(T->getUnsafePtr(), T->actualSize() * sizeof(float));
    } else {
      return MAKE_ERR("Unsupported Tensor format for FLOAT, name: " + in.name(),
                      ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);
    }
  } else if (in.data_type() == ONNX_NAMESPACE::TensorProto::FLOAT16) {
    Type ty(ElemKind::Float16Ty, dim);
    T->reset(getTypeWithCustomStrides(ty, strides));
    if (in.has_raw_data() || !data.empty()) {
      std::istringstream inStream(data.empty() ? in.raw_data() : data,
                                  std::stringstream::binary);
      inStream.read(T->getUnsafePtr(), T->actualSize() * (sizeof(float) / 2));
    } else {
      return MAKE_ERR("Unsupported Tensor format for FLOAT16, name: " +
                          in.name(),
                      ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);
    }
  } else if (in.data_type() == ONNX_NAMESPACE::TensorProto::BFLOAT16) {
    Type ty(ElemKind::BFloat16Ty, dim);
    T->reset(getTypeWithCustomStrides(ty, strides));
    if (in.has_raw_data() || !data.empty()) {
      std::istringstream inStream(data.empty() ? in.raw_data() : data,
                                  std::stringstream::binary);
      inStream.read(T->getUnsafePtr(), T->actualSize() * (sizeof(float) / 2));
    } else {
      return MAKE_ERR("Unsupported Tensor format for BFLOAT16, name: " +
                          in.name(),
                      ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);
    }
  } else if (in.data_type() == ONNX_NAMESPACE::TensorProto::INT64) {
    Type ty(ElemKind::Int64ITy, dim);
    T->reset(getTypeWithCustomStrides(ty, strides));

    if (in.int64_data_size() > 0) {
      auto TH = T->getHandle<int64_t>();
      size_t i = 0;
      for (auto f : in.int64_data()) {
        TH.raw(i++) = f;
      }
    } else if (in.has_raw_data() || !data.empty()) {
      std::istringstream inStream(data.empty() ? in.raw_data() : data,
                                  std::stringstream::binary);
      inStream.read(T->getUnsafePtr(), T->actualSize() * sizeof(int64_t));
    } else {
      return MAKE_ERR("Unsupported Tensor format for INT64, name: " + in.name(),
                      ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);
    }
  } else if (in.data_type() == ONNX_NAMESPACE::TensorProto::INT8) {
    if (in.has_doc_string()) {
      Type ty;
      ASSIGN_VALUE_OR_RETURN_ERR(
          ty, parseTypeFromDocString(in.doc_string(), dim, useGlowCustomOps));
      T->reset(ty);
    } else {
      // Onnx uses Int8 data type through operators like QuantizeLinear.
      // Also data is passed through raw_data since TensorProto
      // does not have data field for int8_t or uint8_t.
      // scale is set to 1 and offset is set to 0 since both scale and offset
      // themselves are operators inputs.
      Type ty(ElemKind::Int8QTy, dim, 1 /* scale*/, 0 /* offset*/);
      T->reset(getTypeWithCustomStrides(ty, strides));
    }
    if (in.has_raw_data() || !data.empty()) {
      std::istringstream inStream(data.empty() ? in.raw_data() : data,
                                  std::stringstream::binary);
      inStream.read(T->getUnsafePtr(), T->actualSize() * sizeof(int8_t));
    } else {
      return MAKE_ERR("Unsupported Tensor format for INT8, name: " + in.name(),
                      ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);
    }
  } else if (in.data_type() == ONNX_NAMESPACE::TensorProto::INT16) {
    Type ty;
    ASSIGN_VALUE_OR_RETURN_ERR(
        ty, parseTypeFromDocString(in.doc_string(), dim, useGlowCustomOps));
    T->reset(ty);

    if (in.has_raw_data() || !data.empty()) {
      std::istringstream inStream(data.empty() ? in.raw_data() : data,
                                  std::stringstream::binary);
      inStream.read(T->getUnsafePtr(), T->actualSize() * sizeof(int16_t));
    } else {
      return MAKE_ERR("Unsupported Tensor format for INT16, name: " + in.name(),
                      ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);
    }
  } else if (in.data_type() == ONNX_NAMESPACE::TensorProto::INT32) {
    if (in.has_doc_string()) {
      Type ty;
      ASSIGN_VALUE_OR_RETURN_ERR(
          ty, parseTypeFromDocString(in.doc_string(), dim, useGlowCustomOps));
      T->reset(ty);
    } else {
      // There are few cases when we will have int32 tensors. For example, the
      // second output of Concat from Caffe2 concat op is int32
      Type ty(ElemKind::Int32ITy, dim);
      T->reset(getTypeWithCustomStrides(ty, strides));
    }

    if (in.int32_data_size() > 0) {
      auto TH = T->getHandle<int32_t>();
      size_t i = 0;
      for (auto f : in.int32_data()) {
        TH.raw(i++) = f;
      }
    } else if (in.has_raw_data() || !data.empty()) {
      std::istringstream inStream(data.empty() ? in.raw_data() : data,
                                  std::stringstream::binary);
      inStream.read(T->getUnsafePtr(), T->actualSize() * sizeof(int32_t));
    } else {
      return MAKE_ERR("Unsupported Tensor format for INT32, name: " + in.name(),
                      ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);
    }
  } else if (in.data_type() == ONNX_NAMESPACE::TensorProto::UINT8) {
    if (in.has_doc_string()) {
      Type ty;
      ASSIGN_VALUE_OR_RETURN_ERR(
          ty, parseTypeFromDocString(in.doc_string(), dim, useGlowCustomOps));
      T->reset(ty);
    } else {
      // Onnx uses Int8 data type through operators like QuantizeLinear.
      // Also data is passed through raw_data since TensorProto
      // does not have data field for int8_t or uint8_t.
      // scale is set to 1 and offset is set to 0 since both scale and offset
      // themselves are operators inputs.
      Type ty(ElemKind::UInt8QTy, dim, 1 /* scale*/, 0 /* offset*/);
      T->reset(getTypeWithCustomStrides(ty, strides));
    }

    if (in.has_raw_data() || !data.empty()) {
      std::istringstream inStream(data.empty() ? in.raw_data() : data,
                                  std::stringstream::binary);
      inStream.read(T->getUnsafePtr(), T->actualSize() * sizeof(uint8_t));
    } else {
      return MAKE_ERR("Unsupported Tensor format for UINT8, name: " + in.name(),
                      ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);
    }
  } else if (in.data_type() == ONNX_NAMESPACE::TensorProto::BOOL) {
    Type ty(ElemKind::BoolTy, dim);
    T->reset(getTypeWithCustomStrides(ty, strides));
    if (in.has_raw_data() || !data.empty()) {
      std::istringstream inStream(data.empty() ? in.raw_data() : data,
                                  std::stringstream::binary);
      inStream.read(T->getUnsafePtr(), T->actualSize() * sizeof(bool));
    } else if (in.int32_data_size() > 0) {
      // Some ONNX models use int32_data to initialize bool type (e.g., when
      // converted from Keras).
      auto TH = T->getHandle<bool>();
      size_t i = 0;
      for (auto f : in.int32_data()) {
        TH.raw(i++) = (bool)f;
      }
    } else {
      return MAKE_ERR("Unsupported Tensor format for BOOL, name: " + in.name(),
                      ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);
    }
  } else {
    return MAKE_ERR(strFormat("Unsupported tensor data type: %u",
                              static_cast<unsigned>(in.data_type())),
                    ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);
  }
  return Error::success();
}

Expected<Type>
ONNXModelLoader::getTensorType(const ONNX_NAMESPACE::TensorProto &in) {
  std::vector<dim_t> dim;
  for (auto d : in.dims()) {
    dim.push_back(d);
  }

  switch (in.data_type()) {
  case ONNX_NAMESPACE::TensorProto::FLOAT:
    return Type(ElemKind::FloatTy, dim);

  case ONNX_NAMESPACE::TensorProto::FLOAT16:
    return Type(ElemKind::Float16Ty, dim);

  case ONNX_NAMESPACE::TensorProto::BFLOAT16:
    return Type(ElemKind::BFloat16Ty, dim);

  case ONNX_NAMESPACE::TensorProto::INT64:
    return Type(ElemKind::Int64ITy, dim);

  case ONNX_NAMESPACE::TensorProto::UINT8:
  case ONNX_NAMESPACE::TensorProto::INT8:
  case ONNX_NAMESPACE::TensorProto::INT16:
    return parseTypeFromDocString(in.doc_string(), dim, useGlowCustomOps_);

  case ONNX_NAMESPACE::TensorProto::INT32:
    if (in.has_doc_string()) {
      return parseTypeFromDocString(in.doc_string(), dim, useGlowCustomOps_);
    }
    return Type(ElemKind::Int32ITy, dim);

  case ONNX_NAMESPACE::TensorProto::BOOL:
    return Type(ElemKind::BoolTy, dim);

  default:
    return MAKE_ERR(strFormat("Unsupported tensor data type: %u",
                              static_cast<unsigned>(in.data_type())),
                    ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);
  }
  llvm_unreachable("Unsupported tensor data type");
}

Expected<Type>
ONNXModelLoader::getTensorType(const ONNX_NAMESPACE::ValueInfoProto &in) {
  auto type = in.type();

  std::vector<dim_t> dim;
  ASSIGN_VALUE_OR_RETURN_ERR(dim, getProtoShape(type.tensor_type().shape()));

  ElemKind kind = ElemKind::FloatTy;
  float scale = 1.0;
  int32_t offset = 0;
  ShapeVector strides;
  if (useGlowCustomOps_) {
    std::string elemKindStr;
    ASSIGN_VALUE_OR_RETURN_ERR(
        elemKindStr, getAttrFromDocString(elemKindSignifier, in.doc_string()));
    kind = Type::getElementKindFromName(elemKindStr);
    if (isQuantizedElemKind(kind)) {
      std::pair<float, int32_t> scaleOffsetPair;
      ASSIGN_VALUE_OR_RETURN_ERR(scaleOffsetPair,
                                 getQuantParamsFromDocString(in.doc_string()));
      std::tie(scale, offset) = scaleOffsetPair;
    }
    strides = getStridesFromDocString(in.doc_string());
  } else {
    // Retrieve the ElemKind from the ONNX type, including considerations for
    // whether the datatype is quantized.
    RETURN_IF_ERR(
        onnxTensorDataTypeToElemKind(type.tensor_type().elem_type(), &kind));
  }

  // If quantized then retrieve the scale and offset if provided (may not be for
  // fused quantized types since they're ignored anyway).
  if (isQuantizedElemKind(kind)) {
    assert(useGlowCustomOps_ &&
           "Quantized loading not fully supported without custom Glow ops.");
    return getTypeWithCustomStrides(Type(kind, dim, scale, offset), strides);
  }
  return getTypeWithCustomStrides(Type(kind, dim), strides);
}

Error ONNXModelLoader::getInputsNamesAndTypes(
    std::vector<std::string> &inTensorNames, std::vector<Type> &inTypes,
    const std::string &filename) {
  // Creating GraphProto from modelfile
  ONNX_NAMESPACE::ModelProto modelDef;

  ASSIGN_VALUE_OR_RETURN_ERR(modelDef, loadProto(filename, false, nullptr));

  ONNX_NAMESPACE::GraphProto graphDef = modelDef.graph();

  // GraphDef.input can have both inputs and intermediate tensors whereas
  // initializers have info about intermediate tensors. Thus the difference
  // betweem two is taken
  std::vector<std::string> inputs;
  for (auto &in : graphDef.input()) {
    inputs.push_back(in.name());
  }

  for (const auto &in : graphDef.initializer()) {
    auto position = std::find(inputs.begin(), inputs.end(), in.name());
    if (position != inputs.end()) {
      inputs.erase(position);
    }
  }

  for (const auto &finalIn : inputs) {
    for (const auto &in : graphDef.input()) {
      if (finalIn.compare(in.name()) != 0) {
        continue;
      }
      inTensorNames.push_back(in.name());
      const ONNX_NAMESPACE::ValueInfoProto &valueInfo = in;
      auto type = valueInfo.type();

      std::vector<dim_t> dim;
      ASSIGN_VALUE_OR_RETURN_ERR(dim,
                                 getProtoShape(type.tensor_type().shape()));

      ElemKind kind = ElemKind::FloatTy;
      RETURN_IF_ERR(
          onnxTensorDataTypeToElemKind(type.tensor_type().elem_type(), &kind));

      inTypes.emplace_back(kind, dim);
    }
  }

  return Error::success();
}

Error ONNXModelLoader::verifyPreexistingStorage(const Storage *S,
                                                const std::string &name,
                                                const Type &ty,
                                                const std::string &layout,
                                                const bool trainable) {
  RETURN_ERR_IF_NOT(S, "Storage did not exist in Module: " + name);
  if (replaceDummyTQPs_ && ty.isQuantizedType() &&
      ty.getScale() == dummyScale) {
    TensorQuantizationParams TQP;
    ASSIGN_VALUE_OR_RETURN_ERR(TQP, getUpdatedTQP(ty.getOffset()));
    // If we are replacing dummy TQPs with updated ones, then do verification
    // based on the updated type and not the base dummy type we found.
    Type updatedTy(ty.getElementType(), ty.dims(), TQP.scale, TQP.offset);
    RETURN_ERR_IF_NOT(S->getType()->isEqual(updatedTy),
                      "Incorrect type for quant param updated existing  " +
                          S->getDebugDesc() + " " + "Expected type " +
                          updatedTy.toString());
  } else {
    RETURN_ERR_IF_NOT(S->getType()->isEqual(ty),
                      "Incorrect type for existing  " + S->getDebugDesc() +
                          " " + "Expected type " + ty.toString());
  }
  if (const Placeholder *PH = llvm::dyn_cast<Placeholder>(S)) {
    RETURN_ERR_IF_NOT(trainable == PH->isTraining(),
                      "Incorrect trainability for existing Storage " + name);
  }
  RETURN_ERR_IF_NOT(layout == S->getLayout(),
                    "Incorrect layout for existing Storage " + name);
  return Error::success();
}

Error ONNXModelLoader::loadInputs(ONNX_NAMESPACE::GraphProto &net,
                                  bool loadInputsAsPlaceholdersForOnnx) {
  for (const auto &in : net.input()) {
    // Skip static weights.
    if (getConstantByNameOrNull(in.name())) {
      continue;
    }

    const std::string &docString = in.doc_string();

    Type ty;
    ASSIGN_VALUE_OR_RETURN_ERR(ty, getTensorType(in));

    if (replaceDummyTQPs_ && ty.isQuantizedType() &&
        ty.getScale() == dummyScale) {
      TensorQuantizationParams TQP;
      ASSIGN_VALUE_OR_RETURN_ERR(TQP, getUpdatedTQP(ty.getOffset()));
      ty = Type(ty.getElementType(), ty.dims(), TQP.scale, TQP.offset);
    }

    std::pair<bool, std::string> trainableLayoutPair;
    ASSIGN_VALUE_OR_RETURN_ERR(
        trainableLayoutPair,
        getTrainableLayoutPairFromDocString(docString, useGlowCustomOps_));

    // If we already have the existing module then we may already have the input
    // Placeholder. If so, verify it has the correct type.
    if (loadIntoExistingModule_) {
      RETURN_ERR_IF_NOT(
          loadInputsAsPlaceholdersForOnnx,
          "Must load inputs as Placeholders when using existing Module.");
      if (Placeholder *PH = mod_.getPlaceholderByNameSlow(in.name())) {
        // Set Fused types of Placeholders if they were expected to be
        // fused. Necessary because Caffe2/ONNX protos do not have fused types
        // explicitly, so will be loaded initially as int8.
        if (ty.isFusedQuantizedType()) {
          RETURN_IF_ERR(setFusedTy(PH, mod_.uniqueType(ty)));
        }
        RETURN_IF_ERR(verifyPreexistingStorage(PH, in.name(), ty,
                                               trainableLayoutPair.second,
                                               trainableLayoutPair.first));
        nodeValueByName_[in.name()] = PH->getOutput();
        continue;
      }
    }

    // We must not have the input created yet, so do so.
    if (loadInputsAsPlaceholdersForOnnx) {
      RETURN_ERR_IF_NOT(!clipQuantRangeToFP16_ || !ty.isQuantizedType() ||
                            ty.isFusedQuantizedType(),
                        "Do not support clipQuantRangeToFP16 with unfused "
                        "quantized input Placeholders: " +
                            in.name());
      Placeholder *inPH;
      ASSIGN_VALUE_OR_RETURN_ERR(
          inPH, createAndRegisterPlaceholder(in.name(), mod_.uniqueType(ty),
                                             staticInputs_.count(in.name()),
                                             trainableLayoutPair.first,
                                             trainableLayoutPair.second));
      auto loaderNameOrErr =
          getAttrFromDocString(loaderNameSignifier, docString);
      const std::string &loaderName =
          !ERR_TO_BOOL(loaderNameOrErr.takeError(), /* log */ false)
              ? loaderNameOrErr.get()
              : in.name();
      RETURN_ERR_IF_NOT(inputVarsByName_.try_emplace(loaderName, inPH).second,
                        "Already had input placeholder by name " + loaderName);
    } else {
      Tensor T(ty);
      RETURN_IF_ERR(createAndRegisterConstant(in.name(), std::move(T)));
    }
  }
  return Error::success();
}

Expected<bool> ONNXModelLoader::getBroadcast(ArgumentDictionaryTy &dict) {
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
    // List of ops that support multidirectional broadcast can be found at
    // https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
    if ((typeName == "Add") || (typeName == "Sub") || (typeName == "Mul") ||
        (typeName == "Div") || (typeName == "Equal") ||
        (typeName == "Greater") || (typeName == "Less") ||
        (typeName == "Max") || (typeName == "Mean") || (typeName == "Min") ||
        (typeName == "Or") || (typeName == "Pow") || (typeName == "Sum") ||
        (typeName == "Xor")) {
      return true;
    }
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
  case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
    return ElemKind::BFloat16Ty;
  case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
    return ElemKind::Float64Ty;
  case ONNX_NAMESPACE::TensorProto_DataType_INT32:
    return ElemKind::Int32ITy;
  case ONNX_NAMESPACE::TensorProto_DataType_INT64:
    return ElemKind::Int64ITy;
  case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
    return ElemKind::BoolTy;
  default:;
  }
  return MAKE_ERR("Non supported ONNX type");
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
#if GOOGLE_PROTOBUF_VERSION >= 3002000
  codedStream.SetTotalBytesLimit(MAX_PROTO_SIZE);
#else
  codedStream.SetTotalBytesLimit(MAX_PROTO_SIZE, MAX_PROTO_SIZE);
#endif
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
ONNXModelLoader::loadProto(const std::string &filename, bool zipMode,
                           const std::string *inputStringPtr) {
  /// A helper class to report parsing errors when loading model protos.
  class OnnxTxtErrCollector : public google::protobuf::io::ErrorCollector {
  public:
    OnnxTxtErrCollector() = default;
    ~OnnxTxtErrCollector() override = default;
    void AddError(int line, int column, const std::string &message) override {
      llvm::errs() << strFormat("ONNX parsing error at [%d, %d]: ", line,
                                column)
                   << message << "\n";
    }
    void AddWarning(int line, int column, const std::string &message) override {
      llvm::errs() << strFormat("ONNX parsing warning at [%d, %d]: ", line,
                                column)
                   << message << "\n";
    }
  };

  // Create a parser object and attach an error collector to it.
  OnnxTxtErrCollector errorCollector;
  google::protobuf::TextFormat::Parser parser;
  parser.RecordErrorsTo(&errorCollector);
  bool parseNet;
  ONNX_NAMESPACE::ModelProto MP;

  if (zipMode) {
    RETURN_ERR_IF_NOT(
        inputStringPtr == nullptr,
        "OnnxModelLoader load from string for zip mode not supported");
    ZipReader zip(filename);
    std::string buffer = zip.getRecord("model");
    // Try to parse as a protocol buffer first.
    parseNet = MP.ParseFromString(buffer);
    // Try to parse a textual representation first.
    if (!parseNet) {
      // If it is not a protocol buffer, try to parse as a text format.
      parseNet = parser.ParseFromString(buffer, &MP);
    }
    if (!parseNet) {
      RETURN_ERR_IF_NOT(false, "Failed to parse ModelProto",
                        ErrorValue::ErrorCode::MODEL_LOADER_INVALID_PROTOBUF);
    }
    size_t numWeights = 0;
    auto numWeightsStr = zip.getRecord("weights");
    numWeights = atoi(numWeightsStr.c_str());
    for (size_t i = 0; i < numWeights; ++i) {
      std::stringstream ss;
      ss << "weight_" << i;
      buffer = zip.getRecord(ss.str());
      auto *t = MP.mutable_graph()->add_initializer();
      t->ParseFromString(buffer);
    }
    return MP;
  }

  std::ifstream ff(filename, std::ios::in | std::ios::binary);
  if (inputStringPtr == nullptr) {
    RETURN_ERR_IF_NOT(ff,
                      strFormat("Can't find the model or network files for %s.",
                                filename.c_str()),
                      ErrorValue::ErrorCode::MODEL_LOADER_INVALID_PROTOBUF);
  }

  // TODO: intend to find a way to reuse the following function later
  // for the text format onnx model:
  // bool ONNXModelLoader::loadProto(ONNX_NAMESPACE::GraphProto &net,
  //  google::protobuf::io::ZeroCopyInputStream &iStream)
  if (filename.find(".onnxtxt") != std::string::npos) {
    if (inputStringPtr == nullptr) {
      // Reserve a reasonably big buffer to make sure that reading of models
      // with serialized constants is fast enough.
      const std::streamsize bufferSize = 1024 * 1024 * 16;
      std::vector<char> buffer(bufferSize);
      ff.rdbuf()->pubsetbuf(buffer.data(), bufferSize);
      std::stringstream ss;
      ss << ff.rdbuf();
      std::string str = ss.str();
      parseNet = parser.ParseFromString(str, &MP);
    } else {
      parseNet = parser.ParseFromString(*inputStringPtr, &MP);
    }

    RETURN_ERR_IF_NOT(parseNet, "Failed to parse ModelProto",
                      ErrorValue::ErrorCode::MODEL_LOADER_INVALID_PROTOBUF);
    return MP;
  }

  if (inputStringPtr != nullptr) {
    std::istringstream iss(*inputStringPtr);
    google::protobuf::io::IstreamInputStream stringStream(&iss);
    return loadProto(stringStream);
  }
  google::protobuf::io::IstreamInputStream fileStream(&ff);
  return loadProto(fileStream);
}

/// Given an input \p val , ceil value is computed for a given datatype T
template <typename T> T ceil(float val) {
  return (val - (T)val) > 0 ? (T)(val + 1) : (T)val;
}

namespace {
/// Helper type for pads.
using Pads = std::vector<unsigned_t>;
} // namespace

/// Get the Pads value based on setting for auto_pad.
/// \p kdim : kernel sizes (HW)
/// \p sdim: stride sizes (HW)
/// \p idim: input sizes (HW)
Expected<Pads> getPads(ArgumentDictionaryTy &dict,
                       llvm::ArrayRef<unsigned_t> kdim,
                       llvm::ArrayRef<unsigned_t> sdim,
                       llvm::ArrayRef<unsigned_t> idim) {
  // TODO: ONNX spec disallows using "pads" and "auto_pad" together. However,
  // the implementation allows mixing them and onnxruntime gives pads priority.
  if (dict.count("pads")) {
    if (dict.at("pads")->ints_size() == 2) { // For maxPool1D
      return Pads({0, (unsigned_t)dict.at("pads")->ints(0), 0,
                   (unsigned_t)dict.at("pads")->ints(1)});
    }
    return getShape<unsigned_t>(dict["pads"]);
  }

  // Set the default zero pads. Number of pads is dependent on input dims
  auto zeroPads = [idim]() {
    if (idim.size() == 3) {
      return Pads({0, 0, 0, 0, 0, 0});
    } else {
      return Pads({0, 0, 0, 0});
    }
  };

  if (dict.count("auto_pad")) {
    std::string padStr;
    ASSIGN_VALUE_OR_RETURN_ERR(padStr, loadStr(dict.at("auto_pad")));
    if (padStr == "VALID") {
      // Return default value 0 for pads.
      return zeroPads();
    } else if (padStr == "SAME_UPPER" || padStr == "SAME_LOWER") {
      unsigned_t near, far, top, left, bottom, right;
      // From https://arxiv.org/pdf/1603.07285.pdf 2.4,
      // o = floor((i + 2*p - k)/s) + 1
      // Also, from https://github.com/onnx/onnx/blob/master/docs/Operators.md
      // output_spatial_shape[i] =
      //     ceil(input_spatial_shape[i] / strides_spatial_shape[i])
      // pad_shape[i] =
      //     (output_spatial_shape[i] - 1) * strides_spatial_shape[i]
      //         + kernel_spatial_shape[i] - input_spatial_shape[i]
      // Use the smallest padding possible out of the possible options.
      unsigned_t odim;
      if (idim.size() == 2) {
        llvm::SmallVector<unsigned_t, 2> pdim(2); // Total Paddding, HW.
        for (size_t i = 0, e = pdim.size(); i < e; i++) {
          odim = ceil<unsigned_t>((float)idim[i] / (float)sdim[i]);
          pdim[i] = sdim[i] * (odim - 1) + kdim[i] - idim[i];
        }
        if (padStr == "SAME_UPPER") {
          // SAME_UPPPER: if odd number for pdim[i], use extra padding at the
          // end.
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
      } else if (idim.size() == 3) {
        llvm::SmallVector<unsigned_t, 3> pdim(3); // Total Paddding, HW.
        for (size_t i = 0, e = pdim.size(); i < e; i++) {
          odim = ceil<unsigned_t>((float)idim[i] / (float)sdim[i]);
          pdim[i] = sdim[i] * (odim - 1) + kdim[i] - idim[i];
        }
        if (padStr == "SAME_UPPER") {
          // SAME_UPPPER: if odd number for pdim[i], use extra padding at the
          // end.
          near = pdim[0] / 2;
          far = near + (pdim[0] & 0x1);
          top = pdim[1] / 2;
          bottom = top + (pdim[1] & 0x1);
          left = pdim[2] / 2;
          right = left + (pdim[2] & 0x1);
        } else {
          // SAME_LOWER: if odd number for pdim[i], use extra padding at the
          // beginning.
          far = pdim[0] / 2;
          near = far + (pdim[0] & 0x1);
          bottom = pdim[1] / 2;
          top = bottom + (pdim[1] & 0x1);
          right = pdim[2] / 2;
          left = right + (pdim[2] & 0x1);
        }
        return Pads({near, top, left, far, bottom, right});
      } else {
        return MAKE_ERR("getPads only works for 2D or 3D");
      }
    } else if (padStr == "NOTSET") {
      // We use explicit pads (if not given we assume its all zeros).
      if (dict.count("pads")) {
        if (dict.at("pads")->ints_size() == 2) { // For maxPool1D
          return Pads({0, (unsigned_t)dict.at("pads")->ints(0), 0,
                       (unsigned_t)dict.at("pads")->ints(1)});
        }
        return getShape<unsigned_t>(dict["pads"]);
      } else {
        return zeroPads();
      }
    }
    return MAKE_ERR("Only auto_pad==VALID, SAME_UPPER, SAME_LOWER and NOTSET "
                    "are supported");
  }
  // Return default value 0 for pads.
  return zeroPads();
}

/// Get the Pads value based on setting for auto_pad.
/// \p kdim : kernel sizes (HW)
/// \p sdim: stride sizes (HW)
/// \p idim: input sizes (HW)
static Expected<Pads> getConvTransposePadsfromOutput(
    ArgumentDictionaryTy &dict, llvm::ArrayRef<unsigned_t> kdim,
    llvm::ArrayRef<unsigned_t> sdim, llvm::ArrayRef<unsigned_t> dilation,
    llvm::ArrayRef<unsigned_t> idim, llvm::ArrayRef<unsigned_t> odim) {

  llvm::SmallVector<unsigned_t, 2> pdim(2); // Total Paddding, HW.
  for (size_t i = 0, e = pdim.size(); i < e; i++) {
    pdim[i] = sdim[i] * (idim[i] - 1) /* + output_padding[0]*/ +
              ((kdim[i] - 1) * dilation[i] + 1) - odim[i];
  }

  unsigned_t top, left, bottom, right;

  if (dict.count("auto_pad")) {
    std::string padStr;
    ASSIGN_VALUE_OR_RETURN_ERR(padStr, loadStr(dict.at("auto_pad")));
    if (padStr == "SAME_UPPER") {
      // SAME_UPPER ONNX formula:
      // if odd number for pdim[i], use extra padding at the end.
      //   pads[start_i] = total_padding[i] - (total_padding[i]/2);
      //   pads[end_i] = (total_padding[i]/2).
      top = pdim[0] - pdim[0] / 2;
      bottom = pdim[0] / 2;
      left = pdim[1] - pdim[1] / 2;
      right = pdim[1] / 2;
      return Pads({top, left, bottom, right});
    }
  }
  // !SAME_UPPER ONNX formula:
  //   pads[start_i] = total_padding[i]/2;
  //   pads[end_i] = total_padding[i] - (total_padding[i]/2)
  top = pdim[0] / 2;
  bottom = pdim[0] - pdim[0] / 2;
  left = pdim[1] / 2;
  right = pdim[1] - pdim[1] / 2;
  return Pads({top, left, bottom, right});
}

const std::string ONNXModelLoader::opErrMsg(const ONNX_NAMESPACE::NodeProto &op,
                                            const std::string &errMsg) {
  const std::string &opName = loadOperatorName(op);
  return strFormat(" [Operator-'%s', opset_version-%d, ir_version-%d] : %s ",
                   opName.c_str(), int(opsetVersion_), int(irVersion_),
                   errMsg.c_str());
}

Error ONNXModelLoader::loadConstant(const ONNX_NAMESPACE::NodeProto &op,
                                    ArgumentDictionaryTy &dict) {
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

  const auto &type = dict.at("value")->type();
  RETURN_ERR_IF_NOT((type == ONNX_NAMESPACE::AttributeProto::TENSOR ||
                     type == ONNX_NAMESPACE::AttributeProto::INTS),
                    "Only Tensor type constants are supported.",
                    ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE);

  Tensor T;
  if (type == ONNX_NAMESPACE::AttributeProto::TENSOR) {
    RETURN_IF_ERR(loadTensor(dict.at("value")->t(), &T, useGlowCustomOps_));
  } else {
    std::vector<int64_t> ints;
    ASSIGN_VALUE_OR_RETURN_ERR(ints, getShape<int64_t>(dict["value"]));
    T = Tensor(ElemKind::Int64ITy, {(dim_t)ints.size()});
    auto TH = T.getHandle<int64_t>();
    for (dim_t i = 0, e = ints.size(); i < e; ++i) {
      TH.at({i}) = ints[i];
    }
  }
  RETURN_IF_ERR(createAndRegisterConstant(name, std::move(T)));

  return Error::success();
}

/// Retrieves data from a constant Tensor and stores it in a vector.
template <typename T, typename datatype = ssize_t>
static void helperSetter(Constant *constT, std::vector<datatype> &vec) {
  auto constH = constT->getPayload().getHandle<T>();
  for (dim_t i = 0; i < constH.size(); ++i) {
    vec.push_back(constH.at({i}));
  }
}

template <typename T>
Error ONNXModelLoader::getRange(const ONNX_NAMESPACE::NodeProto &op,
                                Constant *constT) {
  T start = constT->getPayload().getHandle<T>().raw(0);

  ASSIGN_VALUE_OR_RETURN_ERR(constT, getConstantByName(op.input(1)));
  T limit = constT->getPayload().getHandle<T>().raw(0);

  ASSIGN_VALUE_OR_RETURN_ERR(constT, getConstantByName(op.input(2)));
  T delta = constT->getPayload().getHandle<T>().raw(0);

  std::vector<T> rangeValues;
  if (limit > start) {
    RETURN_ERR_IF_NOT(delta > 0, "delta should be positive");
    auto i = start;
    while (i < limit) {
      rangeValues.push_back(i);
      i += delta;
    }
  } else if (limit < start) {
    RETURN_ERR_IF_NOT(delta < 0, "delta should be negative");
    auto i = start;
    while (i > limit) {
      rangeValues.push_back(i);
      i += delta;
    }
  } else {
    return MAKE_ERR("limit and start value should be different");
  }

  Tensor rangeTensor(constT->getElementType(),
                     {static_cast<unsigned int>(rangeValues.size())});
  rangeTensor.getHandle<T>() = rangeValues;
  RETURN_IF_ERR(
      createAndRegisterConstant(op.output(0), std::move(rangeTensor)));
  return Error::success();
}

Error ONNXModelLoader::loadRange(const ONNX_NAMESPACE::NodeProto &op,
                                 ArgumentDictionaryTy &dict) {
  Constant *constT;
  ASSIGN_VALUE_OR_RETURN_ERR(constT, getConstantByName(op.input(0)));
  auto glowType = constT->getElementType();
  if (glowType == ElemKind::Int64ITy) {
    return getRange<int64_t>(op, constT);
  } else if (glowType == ElemKind::Int32ITy) {
    return getRange<int32_t>(op, constT);
  } else if (glowType == ElemKind::FloatTy) {
    return getRange<float>(op, constT);
  } else {
    return MAKE_ERR("Data type not supported");
  }
}

Error ONNXModelLoader::loadPRelu(const ONNX_NAMESPACE::NodeProto &op,
                                 ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  NodeValue slope;
  ASSIGN_VALUE_OR_RETURN_ERR(slope, getNodeValueByName(op.input(1)));

  // Do broadcasting.
  auto targetDim = in.dims();
  // Sets the axis of each inputs so that the trailing-most dimensions of
  // input tensors and the target shape are aligned.
  int axis = targetDim.size() - slope.dims().size();
  auto *finalSlope = G_->createBroadcast(opName, slope, targetDim, axis);
  auto *R = G_->createPRELU(opName, in, finalSlope);
  RETURN_IF_ERR(addNodeAsOutput(op, R));
  return Error::success();
}

Error ONNXModelLoader::loadSlice(const ONNX_NAMESPACE::NodeProto &op,
                                 ArgumentDictionaryTy &dict) {
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

    RETURN_ERR_IF_NOT(startsC, opErrMsg(op, "Starts Tensor is not Constant."));
    RETURN_ERR_IF_NOT(endsC, opErrMsg(op, "Ends Tensor is not Constant."));

    if (startsC->getElementType() == ElemKind::Int64ITy) {
      helperSetter<int64_t>(startsC, starts);
    } else if (startsC->getElementType() == ElemKind::Int32ITy) {
      helperSetter<int32_t>(startsC, starts);
    } else {
      RETURN_ERR_IF_NOT(
          false,
          opErrMsg(
              op,
              strFormat("Slice Starts Tensor has unsupported type '%s' ",
                        startsC->getType()->getElementName().str().c_str())));
    }

    if (endsC->getElementType() == ElemKind::Int64ITy) {
      helperSetter<int64_t>(endsC, ends);
    } else if (endsC->getElementType() == ElemKind::Int32ITy) {
      helperSetter<int32_t>(endsC, ends);
    } else {
      RETURN_ERR_IF_NOT(
          false,
          opErrMsg(
              op, strFormat("Slice Ends Tensor has unsupported type '%s' ",
                            endsC->getType()->getElementName().str().c_str())));
    }

    if (op.input_size() > 3) {
      Constant *axesC = getConstantByNameOrNull(op.input(3));

      RETURN_ERR_IF_NOT(axesC, opErrMsg(op, "Axes Tensor is not Constant."));

      if (axesC->getElementType() == ElemKind::Int64ITy) {
        helperSetter<int64_t>(axesC, axes);
      } else if (axesC->getElementType() == ElemKind::Int32ITy) {
        helperSetter<int32_t>(axesC, axes);
      } else {
        RETURN_ERR_IF_NOT(
            false,
            opErrMsg(
                op,
                strFormat("Slice Axes Tensor has unsupported type '%s' ",
                          axesC->getType()->getElementName().str().c_str())));
      }

      if (op.input_size() > 4) {
        std::vector<ssize_t> step;
        Constant *stepC = getConstantByNameOrNull(op.input(4));

        RETURN_ERR_IF_NOT(stepC, opErrMsg(op, "Step tensor is not Constant."));

        if (stepC->getElementType() == ElemKind::Int64ITy) {
          helperSetter<int64_t>(stepC, step);
        } else if (stepC->getElementType() == ElemKind::Int32ITy) {
          helperSetter<int32_t>(stepC, step);
        } else {
          RETURN_ERR_IF_NOT(
              false,
              opErrMsg(
                  op,
                  strFormat("Step Tensor has unsupported type '%s'",
                            stepC->getType()->getElementName().str().c_str())));
        }

        // Step is interpreted 1 as default.
        for (size_t i = 0; i < step.size(); i++) {
          RETURN_ERR_IF_NOT(step[i] == 1,
                            opErrMsg(op, "step!=1 is currently not supported"));
        }
      }
    }
  } else {
    // Attributes 'starts' and 'ends' are mandatory and must be consistent.
    ASSIGN_VALUE_OR_RETURN_ERR(starts, getShape<ssize_t>(dict["starts"]));
    ASSIGN_VALUE_OR_RETURN_ERR(ends, getShape<ssize_t>(dict["ends"]));

    if (dict.count("axes")) {
      // The ONNX spec is unclear so we consider that the 'axes' array may have
      // any size. The constraints are:
      // - the element value must be in range [0, numDims),
      // - 'starts' & 'ends' arrays must have the same size as the 'axes' array.
      // In case an axis is specified multiple times in 'axes', the later
      // parameters will simply overwrite the previous ones.
      ASSIGN_VALUE_OR_RETURN_ERR(
          axes, loadAxes<ssize_t>(dict["axes"], data.dims().size()));
    }
  }
  RETURN_ERR_IF_NOT(
      (starts.size() == ends.size()),
      opErrMsg(
          op,
          strFormat("Slice: 'starts' and 'ends' arrays must have the same size."
                    " but found starts %zu and ends %zu sizes ",
                    starts.size(), ends.size())));

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
                    opErrMsg(op, strFormat("'axes' %zu and 'starts' %zu must"
                                           "be the same size.",
                                           axes.size(), starts.size())));
  for (size_t i = 0; i < axes.size(); i++) {
    ssize_t newStart = starts[i];
    ssize_t newEnd = ends[i];
    ssize_t axisId = axes[i];
    RETURN_ERR_IF_NOT(
        (axisId >= 0) && (axisId < ssize_t(numDims)),
        opErrMsg(op, "Axes indexes must be within the input tensor range."));

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
      RETURN_ERR_IF_NOT(
          newStart >= 0,
          opErrMsg(op, strFormat("Slice: final start index %zu should "
                                 " never be negative.",
                                 newStart)));
    }
    if (newEnd < 0) {
      newEnd = ssize_t(dims[axisId]) + newEnd;
      RETURN_ERR_IF_NOT(
          newEnd >= 0,
          opErrMsg(op, strFormat("Slice: final end index %zu should "
                                 " never be negative.",
                                 newEnd)));
    }

    newStarts[axisId] = size_t(newStart);
    newEnds[axisId] = size_t(newEnd);
  }

  // Create the IR node.
  Node *SN = G_->createSlice(opName, data, newStarts, newEnds);
  RETURN_IF_ERR(addNodeAsOutput(op, SN));

  return Error::success();
}

Error ONNXModelLoader::loadTrigonometricOps(const std::string &typeName,
                                            const ONNX_NAMESPACE::NodeProto &op,
                                            ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  Node *N;
  if (typeName == "Sin") {
    N = G_->createSin(opName, in);
  } else {
    N = G_->createCos(opName, in);
  }
  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadErf(const ONNX_NAMESPACE::NodeProto &op,
                               const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  Node *N = G_->createErf(opName, in);
  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadConv(const ONNX_NAMESPACE::NodeProto &op,
                                ArgumentDictionaryTy &dict) {
  // Load the inputs
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  if (in.dims().size() == 3) {
    return loadConv1D(op, dict);
  } else if (in.dims().size() == 4) {
    return loadConv2D(op, dict);
  } else if (in.dims().size() == 5) {
    return loadConv3D(op, dict);
  } else {
    return MAKE_ERR(strFormat(
        "Only 1D (3 dims), 2D (4 dims) and 3D (5 dims) convolution are "
        "supported by the ONNX loader but there are %zu input dims.",
        in.dims().size()));
  }
}

Error ONNXModelLoader::loadConv1D(const ONNX_NAMESPACE::NodeProto &op,
                                  ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);
  // Load the attributes
  std::vector<glow::unsigned_t> strides(2, 1);

  strides[1] = dict.count("strides") ? dict.at("strides")->ints(0) : 1;
  strides[0] = 1;

  unsigned_t group = 1;
  if (dict.count("group")) {
    ASSIGN_VALUE_OR_RETURN_ERR(group, loadInt(dict.at("group")));
  }

  std::vector<unsigned_t> dilations;
  ASSIGN_VALUE_OR_RETURN_ERR(dilations,
                             getDilations(dict, std::vector<unsigned_t>{1, 1}));
  // Expand dilations to length 2 since Glow treat conv1D as conv2D
  if (dilations.size() == 1) {
    dilations.push_back(dilations[0]);
  }

  // Load the inputs
  NodeValue in;
  // input == NCW ---> NCHW
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  in = G_->createExpandDims(opName, in, 2);
  // filtervalue == CKS ---> CKRS
  NodeValue filterValue;
  ASSIGN_VALUE_OR_RETURN_ERR(filterValue, getNodeValueByName(op.input(1)));
  filterValue = G_->createExpandDims(opName, filterValue, 2);
  // Transpose the filter to the right format. Glow expects to read the
  // weights in the format CRSK. ONNX stores the operators as CKRS.
  // C - output_depth, R - filter_height, S - filter_width, K - input_depth.
  // filtervalue == CKRS ---> CRSK
  TransposeNode *filterTransposeNode =
      G_->createTranspose(opName, filterValue, NCHW2NHWC);
  // The structure of the conv weights is: CRSK. We take the C, which is the
  // number of filters. We use this value to calculate the size of the bias
  // if it is not specified.
  const NodeValue filterTransposedValue = filterTransposeNode->getResult();
  dim_t depth = filterTransposedValue.dims()[0];

  // Construct the Bias field.
  NodeValue B;
  // Check if we have a serialized bias vector.
  if (op.input_size() > 2) {
    auto &biasTensorName = op.input(2);
    // Load the serialized bias vector as NodeValue.
    ASSIGN_VALUE_OR_RETURN_ERR(B, getNodeValueByName(biasTensorName));
  }

  // If a serialized bias wasn't found then create a zero bias.
  if (op.input_size() == 2) {
    auto biasTy = mod_.uniqueTypeWithNewShape(in.getType(), {depth});
    Tensor biasTensor(biasTy);
    biasTensor.zero();
    B = mod_.createConstant("conv.bias", std::move(biasTensor));
  }

  // ONNX passes the input as NCHW, and we expect the input to be NHWC.
  auto *tr = G_->createTranspose(opName, in, NCHW2NHWC);
  // Calculate the size and allocate the output buffer.
  ShapeNHWC idim = ShapeNHWC(tr->getResult().dims());
  llvm::SmallVector<unsigned_t, 2> idimHW(2);
  idimHW[0] = in.dims()[2];
  idimHW[1] = in.dims()[3];

  // Pads : {pad_top, pad_left, pad_bottom, pad_right}
  Pads pads;
  // Get the kernel shape.
  llvm::SmallVector<unsigned_t, 2> kernelShape(2);
  kernelShape[0] = filterTransposedValue.dims()[1];
  kernelShape[1] = filterTransposedValue.dims()[2];

  ASSIGN_VALUE_OR_RETURN_ERR(pads, getPads(dict, kernelShape, strides, idimHW));
  auto outSz = calculateConvPoolOutputDims(idim.h, idim.w, kernelShape, strides,
                                           pads, dilations);
  std::array<dim_t, 4> outDims = {{idim.n, outSz.first, outSz.second, depth}};
  auto outTy = mod_.uniqueTypeWithNewShape(in.getType(), outDims);
  auto *node = G_->createConv(opName, tr, filterTransposeNode, B, outTy,
                              kernelShape, strides, pads, group, dilations);

  auto *N = G_->createSqueeze(opName, node, 1 /*axes*/);
  // Transpose the output back
  auto *RR = G_->createTranspose(opName, N, {0, 2, 1});
  RETURN_IF_ERR(addNodeAsOutput(op, RR));
  return Error::success();
}

Error ONNXModelLoader::loadConv2D(const ONNX_NAMESPACE::NodeProto &op,
                                  ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  // Load the inputs
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  NodeValue filterValue;
  ASSIGN_VALUE_OR_RETURN_ERR(filterValue, getNodeValueByName(op.input(1)));

  // Load the attributes
  std::vector<unsigned_t> strides(2, 1);
  if (dict.count("strides")) {
    ASSIGN_VALUE_OR_RETURN_ERR(strides, getShape<unsigned_t>(dict["strides"]));
  }
  unsigned_t group = 1;
  if (dict.count("group")) {
    ASSIGN_VALUE_OR_RETURN_ERR(group, loadInt(dict.at("group")));
  }

  std::vector<unsigned_t> dilations;
  ASSIGN_VALUE_OR_RETURN_ERR(dilations,
                             getDilations(dict, std::vector<unsigned_t>{1, 1}));
  RETURN_ERR_IF_NOT(
      dilations.size() == 2,
      opErrMsg(op, strFormat("2D Conv dilations must be specified for 2 axes "
                             " found axes %zu",
                             dilations.size())));

  // Transpose the filter to the right format. Glow expects to read the
  // weights in the format CRSK. ONNX stores the operators as KCRS.
  // C - output_depth, R - filter_height, S - filter_width, K - input_depth.
  TransposeNode *filterTransposeNode =
      G_->createTranspose(opName, filterValue, NCHW2NHWC);

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
    std::vector<unsigned_t> kernelShapeAttribute;
    ASSIGN_VALUE_OR_RETURN_ERR(kernelShapeAttribute,
                               getShape<unsigned_t>(dict["kernel_shape"]));
    RETURN_ERR_IF_NOT((kernelShape[0] == kernelShapeAttribute[0] &&
                       kernelShape[1] == kernelShapeAttribute[1]),
                      opErrMsg(op, "Conv The 'kernel_shape' attribute is not "
                                   "consistent with the actual "
                                   "convolution kernel shape."));
    (void)kernelShapeAttribute; // Avoids compilation warning in release mode.
  }

  // Construct the Bias field.
  NodeValue B;
  // Check if we have a serialized bias vector.
  if (op.input_size() > 2) {
    auto &biasTensorName = op.input(2);
    // Load the serialized bias vector as NodeValue.
    ASSIGN_VALUE_OR_RETURN_ERR(B, getNodeValueByName(biasTensorName));
  }

  // If a serialized bias wasn't found then create a zero bias.
  if (op.input_size() == 2) {
    auto biasTy = mod_.uniqueTypeWithNewShape(in.getType(), {depth});
    Tensor biasTensor(biasTy);
    biasTensor.zero();
    B = mod_.createConstant("conv.bias", std::move(biasTensor));
  }

  // ONNX passes the input as NCHW, and we expect the input to be NHWC.
  auto *tr = G_->createTranspose(opName, in, NCHW2NHWC);

  // Calculate the size and allocate the output buffer.
  ShapeNHWC idim = ShapeNHWC(tr->getResult().dims());

  llvm::SmallVector<unsigned_t, 2> idimHW(2);
  idimHW[0] = in.dims()[2];
  idimHW[1] = in.dims()[3];

  // Pads : {pad_top, pad_left, pad_bottom, pad_right}
  Pads pads;
  ASSIGN_VALUE_OR_RETURN_ERR(pads, getPads(dict, kernelShape, strides, idimHW));

  auto outSz = calculateConvPoolOutputDims(idim.h, idim.w, kernelShape, strides,
                                           pads, dilations);
  std::array<dim_t, 4> outDims = {{idim.n, outSz.first, outSz.second, depth}};
  auto outTy = mod_.uniqueTypeWithNewShape(in.getType(), outDims);

  auto *node = G_->createConv(opName, tr, filterTransposeNode, B, outTy,
                              kernelShape, strides, pads, group, dilations);

  // Transpose the output back.
  auto *N = G_->createTranspose(opName, node, NHWC2NCHW);

  RETURN_IF_ERR(addNodeAsOutput(op, N));

  return Error::success();
}

Error ONNXModelLoader::loadConv3D(const ONNX_NAMESPACE::NodeProto &op,
                                  ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  // Load the inputs
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  NodeValue filterValue;
  ASSIGN_VALUE_OR_RETURN_ERR(filterValue, getNodeValueByName(op.input(1)));

  // Load the attributes
  std::vector<unsigned_t> strides(3, 1);
  if (dict.count("strides")) {
    ASSIGN_VALUE_OR_RETURN_ERR(strides, getShape<unsigned_t>(dict["strides"]));
  }
  unsigned_t group = 1;
  if (dict.count("group")) {
    ASSIGN_VALUE_OR_RETURN_ERR(group, loadInt(dict.at("group")));
  }

  std::vector<unsigned_t> dilations(3, 1);
  if (dict.count("dilations")) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        dilations, getDilations(dict, std::vector<unsigned_t>{1, 1, 1}));
    RETURN_ERR_IF_NOT(
        dilations.size() == 3,
        opErrMsg(op, strFormat("3D Conv dilations must be specified for 3 axes "
                               "found %zu axes",
                               dilations.size())));
    RETURN_ERR_IF_NOT(
        dilations[0] == 1 && dilations[1] == 1 && dilations[2] == 1,
        opErrMsg(op, strFormat("3D Conv dilations currently only support "
                               "the default value of [1, 1, 1] but the model "
                               "contains [%u, %u, %u]",
                               dilations[0], dilations[1], dilations[2])));
  }

  // Transpose the filter to the right format. Glow expects to read the
  // weights in the format CRSK. ONNX stores the operators as KCRS.
  // C - output_depth, R - filter_height, S - filter_width, K - input_depth.
  TransposeNode *filterTransposeNode =
      G_->createTranspose(opName, filterValue, NCTHW2NTHWC);

  // The structure of the conv weights is: CRSK. We take the C, which is the
  // number of filters. We use this value to calculate the size of the bias
  // if it is not specified.
  const NodeValue filterTransposedValue = filterTransposeNode->getResult();
  dim_t depth = filterTransposedValue.dims()[0];

  // Get the kernel shape from the input.
  llvm::SmallVector<unsigned_t, 3> kernelShape(3);
  kernelShape[0] = filterTransposedValue.dims()[1];
  kernelShape[1] = filterTransposedValue.dims()[2];
  kernelShape[2] = filterTransposedValue.dims()[3];

  // Extra check when the 'kernel_shape' attribute exists.
  // The 'kernel_shape' attribute is redundant not mandatory.
  if (dict.count("kernel_shape")) {
    std::vector<unsigned_t> kernelShapeAttribute;
    ASSIGN_VALUE_OR_RETURN_ERR(kernelShapeAttribute,
                               getShape<unsigned_t>(dict["kernel_shape"]));
    RETURN_ERR_IF_NOT(
        (kernelShape[0] == kernelShapeAttribute[0] &&
         kernelShape[1] == kernelShapeAttribute[1] &&
         kernelShape[2] == kernelShapeAttribute[2]),
        opErrMsg(
            op,
            strFormat(
                "The 'kernel_shape' attribute [%d, %d, %d] is not consistent "
                "with the actual convolution kernel shape [%d, %d, %d].",
                kernelShapeAttribute[0], kernelShapeAttribute[1],
                kernelShapeAttribute[2], kernelShape[0], kernelShape[1],
                kernelShape[2])));
    (void)kernelShapeAttribute; // Avoids compilation warning in release mode.
  }

  // Construct the Bias field.
  NodeValue B;
  // Check if we have a serialized bias vector.
  if (op.input_size() > 2) {
    auto &biasTensorName = op.input(2);
    // Load the serialized bias vector as NodeValue.
    ASSIGN_VALUE_OR_RETURN_ERR(B, getNodeValueByName(biasTensorName));
  }

  // If a serialized bias wasn't found then create a zero bias.
  if (op.input_size() == 2) {
    auto biasTy = mod_.uniqueTypeWithNewShape(in.getType(), {depth});
    Tensor biasTensor(biasTy);
    biasTensor.zero();
    B = mod_.createConstant("conv.bias", std::move(biasTensor));
  }

  // ONNX passes the input as NCTHW, and we expect the input to be NTHWC.
  auto *tr = G_->createTranspose(opName, in, NCTHW2NTHWC);

  // Calculate the size and allocate the output buffer.
  ShapeNTHWC idim(tr->getResult().dims());

  llvm::SmallVector<unsigned_t, 3> idimTHW(3);
  idimTHW = {static_cast<glow::unsigned_t>(idim.t),
             static_cast<glow::unsigned_t>(idim.h),
             static_cast<glow::unsigned_t>(idim.w)};

  // Pads : {pad_near, pad_top, pad_left, pad_far, pad_bottom, pad_right}
  Pads tmpPads;
  ASSIGN_VALUE_OR_RETURN_ERR(tmpPads,
                             getPads(dict, kernelShape, strides, idimTHW));

  // Transpose padding from NTLFBR, which is the ONNX specified layout, to
  // NFTBLR, which is the glow internal layout per the interpreter
  // implementation.

  Pads pads = {tmpPads[0], tmpPads[3], tmpPads[1],
               tmpPads[4], tmpPads[2], tmpPads[5]};

  auto outSz = calculate3DConvPoolOutputDims(idim.t, idim.h, idim.w,
                                             kernelShape, strides, pads);
  std::array<dim_t, 5> outDims = {
      {idim.n, outSz.temporal_frames, outSz.height, outSz.width, depth}};
  auto outTy = mod_.uniqueTypeWithNewShape(in.getType(), outDims);

  auto *node = G_->createConv3D(opName, tr, filterTransposeNode, B, outTy,
                                kernelShape, strides, pads, group);

  // Transpose the output back.
  auto *N = G_->createTranspose(opName, node, NTHWC2NCTHW);

  RETURN_IF_ERR(addNodeAsOutput(op, N));

  return Error::success();
}

Error ONNXModelLoader::loadTensorwiseQuantizedConvolution(
    const ONNX_NAMESPACE::NodeProto &op, ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));
  NodeValue filterValue;
  ASSIGN_VALUE_OR_RETURN_ERR(filterValue, getNodeValueByName(op.input(1)));
  NodeValue biasValue;
  ASSIGN_VALUE_OR_RETURN_ERR(biasValue, getNodeValueByName(op.input(2)));

  std::vector<unsigned_t> kernels;
  ASSIGN_VALUE_OR_RETURN_ERR(kernels,
                             getShape<unsigned_t>(dict["kernel_shape"]));
  std::vector<unsigned_t> strides;
  ASSIGN_VALUE_OR_RETURN_ERR(strides, getShape<unsigned_t>(dict["strides"]));
  std::vector<unsigned_t> pads;
  ASSIGN_VALUE_OR_RETURN_ERR(pads, getShape<unsigned_t>(dict["pads"]));

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
  auto outTy = mod_.uniqueType(ElemKind::Int8QTy, outDims, outScale, outOffset);

  auto *node = G_->createConv(opName, input, filterValue, biasValue, outTy,
                              kernels, strides, pads, groups);

  return addNodeAsOutput(op, node);
}

Error ONNXModelLoader::loadChannelwiseQuantizedConvolution(
    const ONNX_NAMESPACE::NodeProto &op, ArgumentDictionaryTy &dict) {
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

  std::vector<unsigned_t> kernels;
  ASSIGN_VALUE_OR_RETURN_ERR(kernels,
                             getShape<unsigned_t>(dict["kernel_shape"]));
  std::vector<unsigned_t> strides;
  ASSIGN_VALUE_OR_RETURN_ERR(strides, getShape<unsigned_t>(dict["strides"]));
  std::vector<unsigned_t> pads;
  ASSIGN_VALUE_OR_RETURN_ERR(pads, getShape<unsigned_t>(dict["pads"]));

  unsigned_t group;
  ASSIGN_VALUE_OR_RETURN_ERR(group, loadInt(dict.at("group")));

  std::vector<unsigned_t> dilations;
  ASSIGN_VALUE_OR_RETURN_ERR(dilations,
                             getDilations(dict, std::vector<unsigned_t>{1, 1}));

  float outScale;
  ASSIGN_VALUE_OR_RETURN_ERR(outScale, loadFloat(dict.at("out_scale")));
  int32_t outOffset;
  ASSIGN_VALUE_OR_RETURN_ERR(outOffset, loadInt(dict.at("out_offset")));

  ShapeNHWC idim(input.dims());
  auto outSz =
      calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides, pads);
  std::array<dim_t, 4> outDims = {
      {idim.n, outSz.first, outSz.second, biasValue.dims()[0]}};
  auto outTy = mod_.uniqueType(ElemKind::Int8QTy, outDims, outScale, outOffset);

  // Quantize the filter automatically (only if it is float). The bias is NOT
  // quantized automatically and is left at the disposal of each Backend to
  // quantize it later using custom logic.
  auto *node = G_->createChannelwiseQuantizedConv(
      opName, input, filterValue, biasValue, scalesValue, offsetsValue,
      /* biasScales */ nullptr, /* biasOffsets */ nullptr, outTy, kernels,
      strides, pads, group, dilations, /* quantizeFilter */ true,
      /* quantizeBias */ false);

  return addNodeAsOutput(op, node);
}

Error ONNXModelLoader::loadConvTranspose(const ONNX_NAMESPACE::NodeProto &op,
                                         ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);
  // Load the attributes
  std::vector<unsigned_t> strides(2, 1);
  if (dict.count("strides")) {
    ASSIGN_VALUE_OR_RETURN_ERR(strides, getShape<unsigned_t>(dict["strides"]));
  }
  unsigned_t group = 1;
  if (dict.count("group")) {
    ASSIGN_VALUE_OR_RETURN_ERR(group, loadInt(dict.at("group")));
  }

  std::vector<unsigned_t> dilations;
  ASSIGN_VALUE_OR_RETURN_ERR(dilations,
                             getDilations(dict, std::vector<unsigned_t>{1, 1}));
  RETURN_ERR_IF_NOT(dilations.size() == 2,
                    opErrMsg(op, strFormat("ConvTranspose dilations must be "
                                           "specified for 2 axes, found %zu ",
                                           dilations.size())));

  // Load the inputs
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  NodeValue filterValue;
  ASSIGN_VALUE_OR_RETURN_ERR(filterValue, getNodeValueByName(op.input(1)));

  // Transpose the filter to the right format. Glow expects to read the
  // weights in the format CRSK. ONNX stores the operators as KCRS.
  // C - output_depth, R - filter_height, S - filter_width, K - input_depth.
  TransposeNode *filterTransposeNode =
      G_->createTranspose(opName, filterValue, CNHW2NHWC /* flip matrix */);

  // The structure of the conv weigts is: NHWC. We take the C, which is the
  // number of filters. We use this value to calculate the size of the bias
  // if it is not specified.
  const NodeValue filterTransposedValue = filterTransposeNode->getResult();
  dim_t depth = filterTransposedValue.dims()[0] * group;

  // Get the kernel shape from the input.
  llvm::SmallVector<unsigned_t, 2> kernels(2);
  kernels[0] = filterTransposedValue.dims()[1];
  kernels[1] = filterTransposedValue.dims()[2];

  // Extra check when the 'kernel_shape' attribute exists.
  // The 'kernel_shape' attribute is redundant not mandatory.
  if (dict.count("kernel_shape")) {
    std::vector<unsigned_t> kernelShapeAttribute;
    ASSIGN_VALUE_OR_RETURN_ERR(kernelShapeAttribute,
                               getShape<unsigned_t>(dict["kernel_shape"]));
    RETURN_ERR_IF_NOT(
        (kernels[0] == kernelShapeAttribute[0] &&
         kernels[1] == kernelShapeAttribute[1]),
        opErrMsg(
            op,
            "The 'kernel_shape' attribute is not consistent with the actual "
            "convolution kernel shape."));
    (void)kernelShapeAttribute; // Avoids compilation warning in release mode.
  }

  // Construct the Bias field.
  NodeValue B;
  // Check if we have a serialized bias vector.
  if (op.input_size() > 2) {
    auto &biasTensorName = op.input(2);
    // Load the serialized bias vector as constant or NodeValue.
    ASSIGN_VALUE_OR_RETURN_ERR(B, getNodeValueByName(biasTensorName));
  }

  // If a serialized bias wasn't found then create a zero bias.
  if (op.input_size() == 2) {
    auto biasTy = mod_.uniqueTypeWithNewShape(in.getType(), {depth});
    Tensor biasTensor(biasTy);
    biasTensor.zero();
    B = mod_.createConstant("conv.bias", std::move(biasTensor));
  }

  // ONNX passes the input as NCHW, and we expect the input to be NHWC.
  auto *tr = G_->createTranspose(opName, in, NCHW2NHWC);

  // Calculate the size and allocate the output buffer.
  ShapeNHWC idim = ShapeNHWC(tr->getResult().dims());

  llvm::SmallVector<unsigned_t, 2> idimHW(2);
  idimHW[0] = in.dims()[2];
  idimHW[1] = in.dims()[3];

  // Pads : {pad_top, pad_left, pad_bottom, pad_right}
  Pads pads;

  // Conv transpose output size (HxW) is either specified or calculated.
  std::pair<dim_t, dim_t> outSz;

  // Per spec, if output_shape is specified, pads are ignored.
  if (dict.count("output_shape")) {
    std::vector<unsigned_t> outShape;
    ASSIGN_VALUE_OR_RETURN_ERR(outShape,
                               getShape<unsigned_t>(dict["output_shape"]));
    ASSIGN_VALUE_OR_RETURN_ERR(
        pads, getConvTransposePadsfromOutput(dict, kernels, strides, dilations,
                                             idimHW, outShape));
    outSz = {outShape[0], outShape[1]};

    std::pair<dim_t, dim_t> outSzTest = calculateConvTransposeOutputDims(
        idim.h, idim.w, kernels, strides, pads, dilations);
    RETURN_ERR_IF_NOT(
        (outShape[0] == outSzTest.first),
        opErrMsg(op, strFormat("ConvTranspose Expected %d /calculated %d "
                               "pads don't match ",
                               int(outShape[0]), int(outSzTest.first))));
    RETURN_ERR_IF_NOT(
        (outShape[1] == outSzTest.second),
        opErrMsg(op, strFormat("ConvTranspose Expected %d /calculated %d "
                               "pads don't match ",
                               int(outShape[1]), int(outSzTest.second))));
  } else {
    if (dict.count("output_padding")) {
      std::vector<dim_t> outPad;
      ASSIGN_VALUE_OR_RETURN_ERR(outPad,
                                 getShape<dim_t>(dict["output_padding"]));
      if (std::equal(outPad.begin() + 1, outPad.end(), outPad.begin()) &&
          outPad[0] != 0) {
        LOG(FATAL)
            << "ConvTranspose argument 'output_padding' is not supported.";
      }
    }
    ASSIGN_VALUE_OR_RETURN_ERR(pads, getPads(dict, kernels, strides, idimHW));
    outSz = calculateConvTransposeOutputDims(idim.h, idim.w, kernels, strides,
                                             pads, dilations);
  }
  std::array<dim_t, 4> outDims = {{idim.n, outSz.first, outSz.second, depth}};
  auto outTy = mod_.uniqueTypeWithNewShape(in.getType(), outDims);

  auto *node =
      G_->createConvTranspose(opName, tr, filterTransposeNode, B, outTy,
                              kernels, strides, pads, group, dilations);

  // Transpose the output back.
  auto *N = G_->createTranspose(opName, node, NHWC2NCHW);
  RETURN_IF_ERR(addNodeAsOutput(op, N));

  return Error::success();
}

Error ONNXModelLoader::loadPool(const ONNX_NAMESPACE::NodeProto &op,
                                ArgumentDictionaryTy &dict,
                                llvm::StringRef typeName) {
  const std::string &opName = loadOperatorName(op);

  // Load the inputs:
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  std::vector<unsigned_t> strides(2, 1);

  size_t inDim = in.dims().size();

  std::vector<unsigned_t> kernelsShape;
  ASSIGN_VALUE_OR_RETURN_ERR(kernelsShape,
                             getShape<unsigned_t>(dict["kernel_shape"]));

  size_t kerDim = kernelsShape.size();

  std::vector<unsigned_t> kernels = {1, kernelsShape[kerDim - 1]};

  bool countIncludePads;
  ASSIGN_VALUE_OR_RETURN_ERR(
      countIncludePads, getCountIncludePads(dict, /* defaultValue */ false));

  // For maxPool1D inDim = 3
  if (inDim == 3) {
    in = G_->createExpandDims(opName, in, 2);
    if (kerDim != 1) {
      return MAKE_ERR(
          opErrMsg(op, strFormat("Glow handles 1D pooling with kernel "
                                 "dimenstion size 1, but found %d ",
                                 int(kerDim))),
          ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_SHAPE);
    } else {
      if (dict.count("strides")) {
        strides[1] = dict.at("strides")->ints(0);
        strides[0] = 1;
      }
    }
  }

  if (kerDim == 2) { // For maxPool2D
    kernels[0] = kernelsShape[0];
    if (dict.count("strides")) {
      ASSIGN_VALUE_OR_RETURN_ERR(strides,
                                 getShape<unsigned_t>(dict["strides"]));
    }
  }

  if (in.dims().size() != 4 || kernels.size() != 2) {
    // Glow only handles 2D pooling currently.
    return MAKE_ERR(opErrMsg(op, "Glow only handles 2D pooling currently."),
                    ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_SHAPE);
  }

  auto *tr = G_->createTranspose(opName, in, NCHW2NHWC);

  // If 'global_pooling' is set then the operation will pool over the size of
  // the input by doing: kernel = height/width.
  if (dict.count("global_pooling")) {
    auto Ty = in.getType();
    kernels[0] = Ty->dims()[2];
    kernels[1] = Ty->dims()[3];
  }

  // NHWC
  llvm::SmallVector<unsigned_t, 2> idimHW(2);
  idimHW[0] = in.dims()[2]; // As per NCHW format
  idimHW[1] = in.dims()[3];

  Pads pads;
  ASSIGN_VALUE_OR_RETURN_ERR(pads, getPads(dict, kernels, strides, idimHW));

  Node *node = nullptr;
  if (op.output_size() > 1) {
    if (typeName != "MaxPool") {
      return MAKE_ERR(
          opErrMsg(op, "Pool Argmax output is only supported for MaxPool!"),
          ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_OPERATOR);
    }

    node = G_->createMaxPool(opName, tr, kernels, strides, pads);
    auto *res = G_->createTranspose(opName, NodeValue(node, 0), NHWC2NCHW);
    auto *argmax = G_->createTranspose(opName, NodeValue(node, 1), NHWC2NCHW);
    RETURN_IF_ERR(assignNodeOutputs(op, {res, argmax}));
  } else {
    size_t idx = 0;
    if (typeName == "MaxPool") {
      node = G_->createMaxPool(opName, tr, kernels, strides, pads);
      idx = MaxPoolNode::ResultIdx;
    } else {
      node = G_->createAvgPool(opName, tr, kernels, strides, pads, NHWC,
                               countIncludePads);
      idx = AvgPoolNode::ResultIdx;
    }

    Node *N = nullptr;
    if (inDim == 3) { // For maxPool1D
      auto *R = G_->createSqueeze(opName, NodeValue(node, idx), 1);
      N = G_->createTranspose(opName, R, {0, 2, 1});
    } else {
      N = G_->createTranspose(opName, NodeValue(node, idx), NHWC2NCHW);
    }

    RETURN_IF_ERR(addNodeAsOutput(op, N));
  }
  return Error::success();
}

Error ONNXModelLoader::loadTensorwiseQuantizedPool(
    const ONNX_NAMESPACE::NodeProto &op, ArgumentDictionaryTy &dict,
    llvm::StringRef typeName) {
  const std::string &opName = loadOperatorName(op);

  // Load the inputs:
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  std::vector<unsigned_t> kernels;
  ASSIGN_VALUE_OR_RETURN_ERR(kernels,
                             getShape<unsigned_t>(dict["kernel_shape"]));
  std::vector<unsigned_t> strides;
  ASSIGN_VALUE_OR_RETURN_ERR(strides, getShape<unsigned_t>(dict["strides"]));

  if (in.dims().size() != 4 || kernels.size() != 2) {
    // Glow only handles 2D pooling currently.
    return MAKE_ERR(
        opErrMsg(op, strFormat("TensorwiseQuantizedPool Glow only handles 2D "
                               "pooling currently, but found kernel %zu ",
                               kernels.size())),
        ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_SHAPE);
  }

  bool countIncludePads;
  ASSIGN_VALUE_OR_RETURN_ERR(
      countIncludePads, getCountIncludePads(dict, /* defaultValue */ false));

  // NHWC
  llvm::SmallVector<unsigned_t, 2> idimHW(2);
  idimHW[0] = in.dims()[1];
  idimHW[1] = in.dims()[2];

  Pads pads;
  ASSIGN_VALUE_OR_RETURN_ERR(pads, getPads(dict, kernels, strides, idimHW));

  if (op.output_size() > 1) {
    if (typeName != "MaxPool") {
      return MAKE_ERR(opErrMsg(op,
                               "TensorwiseQuantizedPool Argmax output is only "
                               "supported for MaxPool!"),
                      ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_OPERATOR);
    }

    Node *maxpool = G_->createMaxPool(opName, in, kernels, strides, pads);
    auto res = maxpool->getNthResult(MaxPoolNode::ResultIdx);
    auto argmax = maxpool->getNthResult(MaxPoolNode::ArgmaxIdx);
    RETURN_IF_ERR(assignNodeOutputs(op, {res, argmax}));
  } else {
    Node *poolNode;
    if (typeName == "MaxPool") {
      poolNode = G_->createMaxPool(opName, in, kernels, strides, pads);
    } else {
      poolNode = G_->createAvgPool(opName, in, kernels, strides, pads, NHWC,
                                   countIncludePads);
    }
    RETURN_IF_ERR(addNodeAsOutput(op, poolNode));
  }
  return Error::success();
}

Error ONNXModelLoader::loadArgMinMax(const ONNX_NAMESPACE::NodeProto &op,
                                     ArgumentDictionaryTy &dict, bool isMin) {
  const std::string &opName = loadOperatorName(op);

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  size_t axis = 0;
  if (dict.count("axis")) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        axis, loadAxis<size_t>(dict.at("axis"), in.dims().size()));
  }
  bool keepDims = true;
  if (dict.count("keepdims")) {
    ASSIGN_VALUE_OR_RETURN_ERR(keepDims, loadInt(dict.at("keepdims")));
  }
  Node *node;
  if (isMin) {
    node = G_->createArgMin(opName, in, axis, keepDims);
  } else {
    node = G_->createArgMax(opName, in, axis, keepDims);
  }
  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return Error::success();
}

Error ONNXModelLoader::loadUpsample(const ONNX_NAMESPACE::NodeProto &op,
                                    ArgumentDictionaryTy &dict) {

  RETURN_ERR_IF_NOT(
      (opsetVersion_ < 10) && (opsetVersion_ > 6),
      opErrMsg(op, "Version mismatch issue found, Upsample operator is "
                   "supported for opset_version between 7 and 9"
                   "use resize operator if opset_version > 9"));

  const std::string &opName = loadOperatorName(op);
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  // Default mode of upsample operator is "nearest"
  std::string mode("nearest");
  if (dict.count("mode")) {
    ASSIGN_VALUE_OR_RETURN_ERR(mode, loadStr(dict.at("mode")));
  }

  /// Only Nearest Mode is supported
  RETURN_ERR_IF_NOT(mode.compare("nearest") == 0,
                    opErrMsg(op, strFormat("UpSample Operator has nearest mode "
                                           "support only, found mode '%s' ",
                                           mode.c_str())));

  /// Scale is always float as per onnx documentation
  std::vector<float> scales;

  if (opsetVersion_ == 7) {
    if (dict.count("scales")) {
      /// As per onnx documentation this is a required field
      /// and if not present then onnx.checker.check_model file check to fail
      ASSIGN_VALUE_OR_RETURN_ERR(scales, getFloats(dict["scales"]));
    } else {
      return MAKE_ERR(opErrMsg(op,
                               "UpSample Scales field is not present, expected "
                               "for Upsample opset_version==7"));
    }
  }

  if (opsetVersion_ > 7) {
    Constant *scale;
    ASSIGN_VALUE_OR_RETURN_ERR(scale, getConstantByName(op.input(1)));
    if (scale->getElementType() != ElemKind::FloatTy) {
      return MAKE_ERR(opErrMsg(op,
                               "UpSample Scales Tensor should have float type "
                               "for opset_version > 7"));
    }
    auto constH = scale->getPayload().getHandle<float>();
    for (dim_t i = 0; i < constH.size(); ++i) {
      scales.push_back(constH.at({i}));
    }
  }

  /// Scales tensor format is NHWC for supported modes other than nearest.
  /// For nearest mode scale can be 4D or 5D.
  RETURN_ERR_IF_NOT(
      (scales.size() >= 4 && scales.size() <= 5 && mode == "nearest") ||
          scales.size() == 4,
      opErrMsg(
          op, strFormat(
                  "UpSample Scales dimension invalid. Mode: %s Scale Size: %zu",
                  mode.c_str(), scales.size())));

  for (auto &val : scales) {
    RETURN_ERR_IF_NOT(
        val >= 1,
        opErrMsg(op, strFormat("UpSample Scales value can only be "
                               " greater than or equal to 1, but found %d",
                               int(val))));
  }

  switch (scales.size()) {
  case 4: {
    vectorReorder(scales, {NHWC2NCHW});
    auto *intr = G_->createTranspose(opName, in, NCHW2NHWC);
    auto *node = G_->createResizeNearest(opName, intr, scales);
    auto *N = G_->createTranspose(opName, node, NHWC2NCHW);
    RETURN_IF_ERR(addNodeAsOutput(op, N));
    return Error::success();
  }
  case 5: {
    vectorReorder(scales, {NTHWC2NCTHW});

    auto *intr = G_->createTranspose(opName, in, NCTHW2NTHWC);
    auto *node = G_->createResizeNearest(opName, intr, scales);
    auto *N = G_->createTranspose(opName, node, NTHWC2NCTHW);
    RETURN_IF_ERR(addNodeAsOutput(op, N));
    return Error::success();
  }
  default:
    RETURN_ERR_IF_NOT(
        false, opErrMsg(op, strFormat("UpSample Scales dimension invalid")));
  }
}

Error ONNXModelLoader::loadResize(const ONNX_NAMESPACE::NodeProto &op,
                                  const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  std::string modeStr;
  ASSIGN_VALUE_OR_RETURN_ERR(modeStr, loadStr(dict.at("mode")));

  Constant *scalesC = nullptr;

  // Either scales or outDims will be populated (V11 can do either, V10 scales
  // only)
  std::vector<float> scales;
  std::vector<dim_t> outDims;

  int32_t scalesIdx = (this->opsetVersion_ >= 11) ? 2 : 1;
  scalesC = getConstantByNameOrNull(op.input(scalesIdx));
  RETURN_ERR_IF_NOT(
      scalesC,
      opErrMsg(op, strFormat("Resize Scales Tensor '%s' is not Constant.",
                             op.input(scalesIdx).c_str())));
  if (scalesC->getElementType() != ElemKind::FloatTy) {
    return MAKE_ERR(opErrMsg(
        op, strFormat(
                "Resize Scales Tensor should have float type, but found '%s' ",
                scalesC->getType()->getElementName().str().c_str())));
  }

  // For ONNX Resize v11, support attributes that are compatible with v10:
  // exclude_outside = 0
  // extrapolation_value = 0.0
  // nearest_mode = floor
  // coordinate_transformation_mode = asymmetric
  // mode = nearest, (bi)linear
  if (this->opsetVersion_ >= 11) {
    int32_t excludeOutside = 0;
    // attribute: exclude_outside.
    if (dict.count("exclude_outside")) {
      ASSIGN_VALUE_OR_RETURN_ERR(excludeOutside,
                                 loadInt(dict.at("exclude_outside")));
    }
    RETURN_ERR_IF_NOT(excludeOutside == 0,
                      opErrMsg(op, strFormat("ONNX Resize exclude outside "
                                             " not supported.")));
    // attribute: extrapolation_value.
    float extrapolationValue = 0.0;
    if (dict.count("extrapolation_value")) {
      ASSIGN_VALUE_OR_RETURN_ERR(extrapolationValue,
                                 loadFloat(dict.at("extrapolation_value")));
    }
    RETURN_ERR_IF_NOT(
        extrapolationValue == 0.0,
        opErrMsg(op, strFormat("Resize extrapolation value 0 supported only, "
                               "but found value %f",
                               extrapolationValue)));
    // attribute: nearest_mode.
    std::string nearestMode = "round_prefer_floor";
    if (dict.count("nearest_mode")) {
      ASSIGN_VALUE_OR_RETURN_ERR(nearestMode, loadStr(dict.at("nearest_mode")));
    }
    if (modeStr == "nearest" && nearestMode != "floor") {
      return MAKE_ERR(
          opErrMsg(op, strFormat("Resize 'floor' and 'nearest' mode "
                                 "supported only, but found mode '%s' ",
                                 modeStr.c_str())));
    }
    // attribute: coordinate_transformation_mode.
    std::string coordTransformMode = "half_pixel";
    if (dict.count("coordinate_transformation_mode")) {
      ASSIGN_VALUE_OR_RETURN_ERR(
          coordTransformMode,
          loadStr(dict.at("coordinate_transformation_mode")));
    }
    RETURN_ERR_IF_NOT(
        coordTransformMode == "asymmetric",
        opErrMsg(op, strFormat("Resize 'asymmetric' coordinate transformation "
                               "mode supported only, but found %s",
                               coordTransformMode.c_str())));

    // If no scales tensor, sizes tensor should be valid.
    if (scalesC->getPayload().getHandle().size() == 0) {
      Constant *sizesC;
      ASSIGN_VALUE_OR_RETURN_ERR(sizesC, getConstantByName(op.input(3)));
      RETURN_ERR_IF_NOT(sizesC,
                        opErrMsg(op, strFormat("Resize Sizes Tensor '%s'"
                                               " is not Constant.",
                                               op.input(3).c_str())));

      // Must be 1D tensor of int64_t.
      RETURN_ERR_IF_NOT(
          sizesC->dims().size() == 1,
          opErrMsg(op, strFormat("Resize Input must be a 1D vector."
                                 " but found vector size %zu ",
                                 sizesC->dims().size())));
      RETURN_ERR_IF_NOT(
          sizesC->getType()->getElementType() == ElemKind::Int64ITy,
          opErrMsg(op, strFormat(
                           "Resize Input element type must be Int64ITy, but "
                           "found type '%s' ",
                           sizesC->getType()->getElementName().str().c_str())));

      auto sizesH = sizesC->getPayload().getHandle<int64_t>();
      RETURN_ERR_IF_NOT(
          in.dims().size() == sizesH.size(),
          opErrMsg(
              op,
              strFormat("Data input %s and sizes input %s must match in size.",
                        std::to_string(in.dims().size()).c_str(),
                        std::to_string(sizesH.size()).c_str())));
      // Now fill the output tensor
      for (dim_t i = 0; i < sizesH.size(); ++i) {
        outDims.push_back(sizesH.at({i}));
      }
    } else {
      RETURN_ERR_IF_NOT(
          op.input_size() == 3,
          opErrMsg(op, "Resize 'sizes' not valid with 'scales' input"));
    }
  } // v11 processing.

  NodeValue outtr = nullptr;
  auto scalesH = scalesC->getPayload().getHandle();

  // Check is scales is not empty - if yes, use it.
  if (scalesH.size()) {
    for (dim_t i = 0; i < scalesH.size(); ++i) {
      scales.push_back(scalesH.at({i}));
    }

    // Scales tensor format is NHWC for supported modes other than nearest.
    // For nearest mode scale can be 3D, 4D, 5D, or 6D.
    RETURN_ERR_IF_NOT(
        (scales.size() >= 3 && scales.size() <= 6 && modeStr == "nearest") ||
            scales.size() == 4,
        opErrMsg(
            op, strFormat(
                    "Resize Scales dimension invalid. Mode: %s Scale Size: %zu",
                    modeStr.c_str(), scales.size())));

    for (auto &val : scales) {
      RETURN_ERR_IF_NOT(
          val > 0,
          opErrMsg(
              op,
              strFormat(
                  "Resize Scale value must be greater than zero, but found %d",
                  int(val))));
    }

    if (modeStr == "nearest") {
      outtr = G_->createResizeNearest(opName, in, scales);
    } else if (modeStr == "bilinear" || modeStr == "linear") {
      vectorReorder(scales, {NHWC2NCHW});
      auto *intr = G_->createTranspose(opName, in, NCHW2NHWC);
      auto RN = G_->createResizeBilinear(opName, intr, scales);
      outtr = G_->createTranspose(opName, RN, NHWC2NCHW);
    } else {
      return MAKE_ERR(
          opErrMsg(op, strFormat("Resize Supports nearest or bilinear "
                                 "interpolation only, but found mode as '%s' ",
                                 modeStr.c_str())),
          ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_ATTRIBUTE);
    }
  } else if (outDims.size()) {
    if (modeStr == "nearest") {
      auto outTy = G_->getParent()->uniqueTypeWithNewShape(
          in.getType(), llvm::ArrayRef<dim_t>(outDims));
      outtr = G_->createResizeNearest(opName, in, outTy);
    } else if (modeStr == "bilinear" || modeStr == "linear") {
      vectorReorder(outDims, {NHWC2NCHW});
      auto outTy = G_->getParent()->uniqueTypeWithNewShape(
          in.getType(), llvm::ArrayRef<dim_t>(outDims));
      auto *intr = G_->createTranspose(opName, in, NCHW2NHWC);
      auto RN = G_->createResizeBilinear(opName, intr, outTy);
      outtr = G_->createTranspose(opName, RN, NHWC2NCHW);
    } else {
      return MAKE_ERR(
          opErrMsg(op, strFormat(
                           "Supporting nearest or (bi)linear interpolation only"
                           " but found mode '%s' ",
                           modeStr.c_str())),
          ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_ATTRIBUTE);
    }
  } else {
    return MAKE_ERR(opErrMsg(op, "Resize Neither scales or sizes are set."),
                    ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_ATTRIBUTE);
  }

  RETURN_IF_ERR(addNodeAsOutput(op, outtr));
  return Error::success();
}

Error ONNXModelLoader::loadGlobalAveragePool(
    const ONNX_NAMESPACE::NodeProto &op, ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  // Load the inputs:
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  std::vector<unsigned_t> strides(2, 1);
  if (dict.count("strides")) {
    ASSIGN_VALUE_OR_RETURN_ERR(strides, getShape<unsigned_t>(dict["strides"]));
  }

  llvm::SmallVector<unsigned_t, 2> kernels(2);
  kernels[0] = in.dims()[2];
  kernels[1] = in.dims()[3];

  Pads pads;
  ASSIGN_VALUE_OR_RETURN_ERR(
      pads, getPads(dict, kernels, strides, kernels /* input sizes*/));

  bool countIncludePads;
  ASSIGN_VALUE_OR_RETURN_ERR(
      countIncludePads, getCountIncludePads(dict, /* defaultValue */ false));

  auto *tr = G_->createTranspose(opName, in, NCHW2NHWC);
  Node *node = G_->createAvgPool(opName, tr, kernels, strides, pads, NHWC,
                                 countIncludePads);
  auto *N = G_->createTranspose(opName, node, NHWC2NCHW);
  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadSqueeze(const ONNX_NAMESPACE::NodeProto &op,
                                   ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  std::vector<dim_t> axes;
  ASSIGN_VALUE_OR_RETURN_ERR(axes,
                             loadAxes<dim_t>(dict["axes"], in.dims().size()));
  Node *node = G_->createSqueeze(opName, in, axes);
  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return Error::success();
}

Error ONNXModelLoader::loadUnsqueeze(const ONNX_NAMESPACE::NodeProto &op,
                                     ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  // Compute output rank.
  std::vector<int> axesTemp;
  ASSIGN_VALUE_OR_RETURN_ERR(axesTemp, getShape<int>(dict["axes"]));
  int outputRank = in.dims().size() + axesTemp.size();

  // Read again the axes and use the output rank to wrap negative axes.
  std::vector<dim_t> axes;
  ASSIGN_VALUE_OR_RETURN_ERR(axes, loadAxes<dim_t>(dict["axes"], outputRank));

  Node *node = G_->createExpandDims(opName, in, axes);
  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return Error::success();
}

Error ONNXModelLoader::loadBatchNormalization(
    const ONNX_NAMESPACE::NodeProto &op, ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  NodeValue scale;
  ASSIGN_VALUE_OR_RETURN_ERR(scale, getNodeValueByName(op.input(1)));
  NodeValue bias;
  ASSIGN_VALUE_OR_RETURN_ERR(bias, getNodeValueByName(op.input(2)));
  NodeValue mean;
  ASSIGN_VALUE_OR_RETURN_ERR(mean, getNodeValueByName(op.input(3)));
  NodeValue var;
  ASSIGN_VALUE_OR_RETURN_ERR(var, getNodeValueByName(op.input(4)));
  float epsilon = 1e-5f; // default
  auto epsilonIt = dict.find("epsilon");
  if (epsilonIt != dict.end()) {
    ASSIGN_VALUE_OR_RETURN_ERR(epsilon, loadFloat(epsilonIt->second));
  }

  auto *node = G_->createBatchNormalization(opName, in.getType(), in, bias,
                                            scale, mean, var, 1, epsilon);

  // BatchNormalization has 4 optional outputs that are not supported by glow.
  // Then: 1/ In case the optional outputs are present and used by other
  // operations of the model, then the import should fail. 2/ In case the
  // optional outputs are declared but not used, the import should succeed. By
  // registering only the mandatory output, we make sure the import will fail if
  // the non supported features are actually requested by the ONNX model.
  RETURN_IF_ERR(addNodeAsOutput(op, node, 1));

  return Error::success();
}

Error ONNXModelLoader::loadInstanceNormalization(
    const ONNX_NAMESPACE::NodeProto &op, ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  NodeValue scale;
  ASSIGN_VALUE_OR_RETURN_ERR(scale, getNodeValueByName(op.input(1)));
  NodeValue bias;
  ASSIGN_VALUE_OR_RETURN_ERR(bias, getNodeValueByName(op.input(2)));

  float epsilon = 1e-5f; // default
  auto epsilonIt = dict.find("epsilon");
  if (epsilonIt != dict.end()) {
    ASSIGN_VALUE_OR_RETURN_ERR(epsilon, loadFloat(epsilonIt->second));
  }

  auto *node =
      G_->createInstanceNormalization(opName, in, bias, scale, 1, epsilon);
  RETURN_IF_ERR(addNodeAsOutput(op, node, 1));

  return Error::success();
}

Error ONNXModelLoader::loadConcat(const ONNX_NAMESPACE::NodeProto &op,
                                  ArgumentDictionaryTy &dict) {
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
  ASSIGN_VALUE_OR_RETURN_ERR(
      axis, loadAxis<int>(dict.at("axis"), inputs.back().dims().size()));

  Node *node = G_->createConcat(opName, inputs, axis);

  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return Error::success();
}

Error ONNXModelLoader::loadFCTransposed(const ONNX_NAMESPACE::NodeProto &op,
                                        ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  if (in.getType()->dims().size() > 2) {
    size_t axis = 1;
    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(
          axis, loadAxis<size_t>(dict.at("axis"), in.dims().size()));
    }
    in = G_->createFlatten(opName + ".fc.in", in, axis);
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
    W = mod_.createConstant(W->getName(), tmp);
  }

  Constant *B;
  ASSIGN_VALUE_OR_RETURN_ERR(B, getConstantByName(op.input(2)));

  auto *node = G_->createFullyConnected(opName, in, W, B);

  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return Error::success();
}

Error ONNXModelLoader::loadGemm(const ONNX_NAMESPACE::NodeProto &op,
                                ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);
  NodeValue A;
  ASSIGN_VALUE_OR_RETURN_ERR(A, getNodeValueByName(op.input(0)));
  NodeValue B;
  ASSIGN_VALUE_OR_RETURN_ERR(B, getNodeValueByName(op.input(1)));
  NodeValue C = nullptr;
  if (op.input_size() > 2 && !op.input(2).empty()) {
    ASSIGN_VALUE_OR_RETURN_ERR(C, getNodeValueByName(op.input(2)));
  }

  float alpha = 1.0;
  if (dict.count("alpha")) {
    ASSIGN_VALUE_OR_RETURN_ERR(alpha, loadFloat(dict.at("alpha")));
  }

  float beta = 1.0;
  if (dict.count("beta")) {
    ASSIGN_VALUE_OR_RETURN_ERR(beta, loadFloat(dict.at("beta")));
  }

  bool transA = false;
  if (dict.count("transA")) {
    ASSIGN_VALUE_OR_RETURN_ERR(transA, loadInt(dict.at("transA")));
  }

  bool transB = false;
  if (dict.count("transB")) {
    ASSIGN_VALUE_OR_RETURN_ERR(transB, loadInt(dict.at("transB")));
  }

  Node *node = G_->createGemm(opName, A, B, C, alpha, beta, transA, transB);

  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return Error::success();
}

Error ONNXModelLoader::loadMatMul(const ONNX_NAMESPACE::NodeProto &op,
                                  ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue LHS;
  ASSIGN_VALUE_OR_RETURN_ERR(LHS, getNodeValueByName(op.input(0)));
  NodeValue RHS;
  ASSIGN_VALUE_OR_RETURN_ERR(RHS, getNodeValueByName(op.input(1)));

  /// For dimension greater than 2 use batchedMatMul
  if (LHS.dims().size() > 2) {
    Node *node = G_->createBatchMatMul(opName, LHS, RHS);
    const size_t numDimsLHS = LHS.dims().size();
    if (numDimsLHS > 3) {
      const size_t numDimsRHS = RHS.dims().size();
      std::vector<dim_t> finalShape;
      for (auto d : LHS.dims()) {
        finalShape.push_back(d);
      }
      finalShape[numDimsLHS - 1] = RHS.dims()[numDimsRHS - 1];
      node = G_->createReshape(opName, node, finalShape);
    }
    RETURN_IF_ERR(addNodeAsOutput(op, node));
  } else {
    Node *node = G_->createMatMul(opName, LHS, RHS);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
  }
  return Error::success();
}

Error ONNXModelLoader::loadHardSigmoid(const ONNX_NAMESPACE::NodeProto &op,
                                       ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  // Input Type.
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));

  float alphaVal = 0.2f;
  if (dict.count("alpha")) {
    ASSIGN_VALUE_OR_RETURN_ERR(alphaVal, loadFloat(dict.at("alpha")));
  }
  float betaVal = 0.5f;
  if (dict.count("beta")) {
    ASSIGN_VALUE_OR_RETURN_ERR(betaVal, loadFloat(dict.at("beta")));
  }

  // Create the node.
  Node *N = G_->createHardSigmoid(opName, input, alphaVal, betaVal);
  RETURN_IF_ERR(addNodeAsOutput(op, N));

  return Error::success();
}

Error ONNXModelLoader::loadLeakyRelu(const ONNX_NAMESPACE::NodeProto &op,
                                     ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  // Input Type.
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));

  // ONNX spec says default is 0.01, but doesn't explicitly say it's optional.
  // like for others. The default example just omits alpha.
  float alphaVal = 0.01f;
  if (dict.count("alpha")) {
    ASSIGN_VALUE_OR_RETURN_ERR(alphaVal, loadFloat(dict.at("alpha")));
  }

  // Create the node.
  Node *N = G_->createLeakyRELU(opName, input, alphaVal);
  RETURN_IF_ERR(addNodeAsOutput(op, N));

  return Error::success();
}

Error ONNXModelLoader::loadPad(const ONNX_NAMESPACE::NodeProto &op,
                               ArgumentDictionaryTy &dict) {
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
      return MAKE_ERR(
          "Pad: Invalid mode",
          ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_ATTRIBUTE);
    }
  }

  std::vector<int> pads;

  if (this->opsetVersion_ > 10) {
    // Get pads through input(1) from opset v11.
    RETURN_ERR_IF_NOT(
        op.input_size() > 1,
        "Pad: The 'pads' is mandatory as input(1) from opsetv11.");
    Constant *padC;
    ASSIGN_VALUE_OR_RETURN_ERR(padC, getConstantByName(op.input(1)));
    RETURN_ERR_IF_NOT(padC, "Support only constant pad");
    helperSetter<int64_t, int>(padC, pads);
  } else {
    RETURN_ERR_IF_NOT(dict.count("pads"),
                      "Pad: The 'pads' property is mandatory");
    ASSIGN_VALUE_OR_RETURN_ERR(pads, getShape<int>(dict["pads"]));
  }

  RETURN_ERR_IF_NOT(
      (pads.size() == 2 * numDims),
      opErrMsg(op, " The 'pads' array must contain 2 values per dimensions"));

  float value = 0.f;
  if (this->opsetVersion_ > 10) {
    if (op.input_size() > 2) {
      Constant *valueC;
      ASSIGN_VALUE_OR_RETURN_ERR(valueC, getConstantByName(op.input(2)));
      RETURN_ERR_IF_NOT(valueC, "Support only constant value in Pad");
      RETURN_ERR_IF_NOT(valueC->getElementType() == ElemKind::FloatTy,
                        "Value in Pad should be float type.");
      value = valueC->getPayload().getHandle().raw(0);
    }
  } else {
    if (dict.count("value")) {
      ASSIGN_VALUE_OR_RETURN_ERR(value, loadFloat(dict.at("value")));
    }
  }

  // Compute the output type.
  std::vector<dim_t> outDims(numDims);
  for (unsigned_t i = 0; i < numDims; i++) {
    auto new_dim = inputDims[i] + pads[i] + pads[i + numDims];
    RETURN_ERR_IF_NOT(
        new_dim > 0,
        opErrMsg(op, "The padding can't remove all elements of a dimension"));
    outDims[i] = new_dim;
  }
  auto outTy = mod_.uniqueType(ElemKind::FloatTy, outDims);

  // Create the IR node.
  Node *N = G_->createPad(opName, input, outTy, mode, pads, value);
  RETURN_IF_ERR(addNodeAsOutput(op, N));

  return Error::success();
}

Error ONNXModelLoader::loadCast(const ONNX_NAMESPACE::NodeProto &op,
                                ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  // Input type
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));
  ElemKind inputKind = input.getType()->getElementType();

  // Target type.
  ElemKind targetKind;
  RETURN_ERR_IF_NOT(dict.count("to"),
                    opErrMsg(op, "Cast missing 'to' attribute"));
  int toONNXTypeValue;
  ASSIGN_VALUE_OR_RETURN_ERR(toONNXTypeValue, loadInt(dict.at("to")));
  RETURN_ERR_IF_NOT(
      ONNX_NAMESPACE::TensorProto_DataType_IsValid(toONNXTypeValue),
      opErrMsg(op, "Cast invalid target type"),
      ErrorValue::ErrorCode::MODEL_LOADER_INVALID_PROTOBUF);
  ASSIGN_VALUE_OR_RETURN_ERR(
      targetKind, convertTensorProtoDataType(
                      ONNX_NAMESPACE::TensorProto_DataType(toONNXTypeValue)));

  // Only support non quantized types.
  RETURN_ERR_IF_NOT(
      (!isQuantizedElemKind(inputKind)) && (!isQuantizedElemKind(targetKind)),
      opErrMsg(op,
               "Cast Unsupported types (Supports only non quantized types)"));

  // Create the IR node.
  Node *N = G_->createConvertTo(opName, input, targetKind);
  RETURN_IF_ERR(addNodeAsOutput(op, N));

  return Error::success();
}

Error ONNXModelLoader::loadDepthToSpace(const ONNX_NAMESPACE::NodeProto &op,
                                        const ArgumentDictionaryTy &dict) {
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));

  dim_t blockSize = 0;
  if (dict.count("blocksize")) {
    ASSIGN_VALUE_OR_RETURN_ERR(blockSize, loadInt(dict.at("blocksize")));
  } else {
    return MAKE_ERR("DepthToSpace: missing 'blocksize' attribute");
  }

  std::string mode = "DCR";
  if (dict.count("mode")) {
    ASSIGN_VALUE_OR_RETURN_ERR(mode, loadStr(dict.at("mode")));
  }

  auto inputDim = input.dims();
  RETURN_ERR_IF_NOT(inputDim.size() == 4,
                    "DepthToSpace: dimension size of 4 is expected.");
  RETURN_ERR_IF_NOT(inputDim[1] % blockSize == 0,
                    "DepthToSpace: depth should be divisible by block size.");

  std::string opName = loadOperatorName(op);
  auto *TR1 = G_->createTranspose(opName, input, NCHW2NHWC);
  auto *D2S = G_->createDepthToSpace(opName, TR1, blockSize, mode == "CRD");
  auto *TR2 = G_->createTranspose(opName, D2S, NHWC2NCHW);

  RETURN_IF_ERR(addNodeAsOutput(op, TR2));
  return Error::success();
}

Error ONNXModelLoader::loadSpaceToDepth(const ONNX_NAMESPACE::NodeProto &op,
                                        ArgumentDictionaryTy &dict) {

  // Input Type
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));

  int blockSize = 0;
  if (dict.count("blocksize")) {
    ASSIGN_VALUE_OR_RETURN_ERR(blockSize, loadInt(dict.at("blocksize")));
  } else {
    return MAKE_ERR(
        opErrMsg(op, "SpaceToDepth: missing 'blocksize' attribute"));
  }

  // Create the node.
  std::string opName = loadOperatorName(op);
  auto *tr = G_->createTranspose(opName, input, NCHW2NHWC);
  Node *nd = G_->createSpaceToDepth(opName, tr, blockSize);
  auto *N = G_->createTranspose(opName, nd, NHWC2NCHW);

  RETURN_IF_ERR(addNodeAsOutput(op, N));

  return Error::success();
}

Error ONNXModelLoader::loadReduceL2(const ONNX_NAMESPACE::NodeProto &op,
                                    const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  in = G_->createMul(opName, in, in);

  // ReduceAdd.
  std::vector<unsigned_t> shapeAxes = {};
  if (dict.count("axes")) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        shapeAxes, loadAxes<unsigned_t>(dict.at("axes"), in.dims().size()));
    std::sort(shapeAxes.begin(), shapeAxes.end());
    if (shapeAxes.size() > 1) {
      auto it = std::unique(shapeAxes.begin(), shapeAxes.end());
      if (it != shapeAxes.end())
        return MAKE_ERR(opErrMsg(op, "ReduceL2 Axes values are not unique."),
                        ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_SHAPE);
    }
  } else {
    shapeAxes.resize(in.dims().size());
    std::iota(shapeAxes.begin(), shapeAxes.end(), 0);
  }

  bool keepDims = true;
  if (dict.count("keepdims")) {
    int keepdims;
    ASSIGN_VALUE_OR_RETURN_ERR(keepdims, loadInt(dict.at("keepdims")));
    keepDims = (bool)keepdims;
  }

  // Reduceadd works only for single axis as of now.
  for (auto it = shapeAxes.rbegin(), e = shapeAxes.rend(); it != e; ++it) {
    in = G_->createBatchedReduceAdd(opName, in, llvm::makeArrayRef(*it));
    if (keepDims) {
      in = G_->createExpandDims(opName, in, *it);
    }
  }

  in = G_->createPow(opName, in, 0.5f);
  RETURN_IF_ERR(addNodeAsOutput(op, in));
  return Error::success();
}

Error ONNXModelLoader::loadConstantOfShape(const ONNX_NAMESPACE::NodeProto &op,
                                           ArgumentDictionaryTy &dict,
                                           bool isSplat) {
  Tensor T(ElemKind::FloatTy, {1});
  T.getHandle().raw(0) = 0.0;

  if (dict.count("value")) {
    RETURN_IF_ERR(loadTensor(dict.at("value")->t(), &T, useGlowCustomOps_));
    if (!isSplat) {
      // Validate tensor only for ConstantOfShape operator.
      RETURN_ERR_IF_NOT(
          T.dims().size() == 1,
          opErrMsg(op, strFormat("ConstantOfShape Value must be "
                                 "a 1D vector, but found size %zu ",
                                 T.dims().size())));
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
    RETURN_ERR_IF_NOT(
        in->dims().size() == 1,
        opErrMsg(
            op,
            strFormat(
                "ConstantOfShape Input must be a 1D vector, but found size %d ",
                int(in->dims().size()))));
    RETURN_ERR_IF_NOT(
        in->getType()->getElementType() == ElemKind::Int64ITy,
        opErrMsg(op, "ConstantOfShape Input element type must be Int64ITy."));
    // Convert 1D tensor of int64_t into llvm::ArrayRef<dim_t>.
    auto TH = in->getPayload().getHandle<int64_t>();
    auto begin = &TH.raw(0);
    auto end = begin + TH.actualSize();
    ShapeVector outputDims(begin, end);

    ty = mod_.uniqueType(T.getType().getElementType(), outputDims);
    switch (T.getType().getElementType()) {
    case ElemKind::Int64ITy: {
      int64_t v = T.getHandle<int64_t>().raw(0);
      RETURN_ERR_IF_NOT(
          v == static_cast<int64_t>(static_cast<float>(v)),
          opErrMsg(
              op, "ConstantOfShape implementation may cause losses for value " +
                      std::to_string(v) + " ."));
      SN = G_->createSplat(loadOperatorName(op), ty, v);
      break;
    }
    case ElemKind::Int32ITy: {
      int32_t v = T.getHandle<int32_t>().raw(0);
      RETURN_ERR_IF_NOT(
          v == static_cast<int32_t>(static_cast<float>(v)),
          opErrMsg(
              op, "ConstantOfShape implementation may cause losses for value " +
                      std::to_string(v) + " ."));
      SN = G_->createSplat(loadOperatorName(op), ty, v);
      break;
    }
    default:
      SN = G_->createSplat(loadOperatorName(op), ty, T.getHandle().raw(0));
    }
  } else {
    ty = mod_.uniqueType(T.getType().getElementType(), T.dims());
    SN = G_->createSplat(loadOperatorName(op), ty, T.getHandle().raw(0));
  }
  RETURN_IF_ERR(addNodeAsOutput(op, SN));
  return Error::success();
}

Error ONNXModelLoader::loadTile(const ONNX_NAMESPACE::NodeProto &op,
                                ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);
  NodeValue in, repeats;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  ASSIGN_VALUE_OR_RETURN_ERR(repeats, getNodeValueByName(op.input(1)));
  if (!llvm::isa<Constant>(repeats)) {
    return MAKE_ERR(opErrMsg(op, "Tile Only constant Repeats is supported!"));
  }

  if (repeats.dims().size() != 1) {
    return MAKE_ERR(
        opErrMsg(op, "Tile Repeats must be a single-dimensional tensor!"));
  }

  if (repeats.dims()[0] != in.dims().size()) {
    return MAKE_ERR(opErrMsg(
        op, "Tile Repeats should have one value for each dimension of input!"));
  }
  auto rh = llvm::cast<Constant>(repeats)->getPayload().getHandle<int64_t>();
  Node *N = in;
  for (size_t i = 0; i < in.dims().size(); i++) {
    auto tiles = rh.raw(i);
    if (tiles != 1) {
      std::string name = opName + "." + std::to_string(i);
      N = G_->createTile(name, N, tiles, /*axis*/ i);
    }
  }

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadExpand(const ONNX_NAMESPACE::NodeProto &op,
                                  const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);
  NodeValue in;
  Constant *repeats;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  ASSIGN_VALUE_OR_RETURN_ERR(repeats, getConstantByName(op.input(1)));

  std::vector<dim_t> tiles;
  helperSetter<int64_t, dim_t>(repeats, tiles);
  auto inputDimSize = (size_t)in.dims().size();
  auto repeatSize = (size_t)tiles.size();
  if (repeatSize > inputDimSize) {
    for (size_t i = 0, e = repeatSize - inputDimSize; i < e; i++) {
      in = G_->createExpandDims(opName + "_" + std::to_string(i), in, i);
    }
  }

  Node *N = in;
  for (size_t i = 0, e = tiles.size(); i < e; i++) {
    // Two corresponding dimension must have the same value,
    // or one of them is equal to 1.
    if (in.dims()[i] != 1 && tiles[i] != in.dims()[i] && tiles[i] != 1) {
      return MAKE_ERR(opErrMsg(op, "Expand Invalid repeat value"));
    }
    if (tiles[i] != in.dims()[i] && tiles[i] != 1) {
      std::string name = opName + "_" + std::to_string(i);
      N = G_->createTile(name, N, tiles[i], /*axis*/ i);
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
    // If the name of the input is empty then consider it to be unspecified,
    // which is valid for optional inputs, so simply skip. If it is necessary
    // for loading the op, then when we later try to load the proper error will
    // be propagated upward.
    if (op.input(i).empty()) {
      continue;
    }
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(i)));
    inputs.push_back(in);
  }

  if (!isConstantFoldable(inputs, typeName)) {
    return false;
  }

  // Create a temporary lightweight loader to construct function representing
  // current Op, and then constant fold the function using Interp backend.
  Function *tmpF = mod_.createFunction("eval_const_fold__");
  ONNXModelLoader tmpLoader(*tmpF);
  tmpLoader.opsetVersion_ = opsetVersion_;
  bool foldStatus = !ERR_TO_BOOL(
      constantFoldInLoader<ONNXModelLoader, ONNX_NAMESPACE::NodeProto>(
          tmpF, tmpLoader, this, op),
      /* log */ false);
  mod_.eraseFunction(tmpF);
  return foldStatus;
}

Error ONNXModelLoader::loadWhere(const ONNX_NAMESPACE::NodeProto &op,
                                 ArgumentDictionaryTy &dict) {
  NodeValue cNV;
  ASSIGN_VALUE_OR_RETURN_ERR(cNV, getNodeValueByName(op.input(0)));
  NodeValue xNV;
  ASSIGN_VALUE_OR_RETURN_ERR(xNV, getNodeValueByName(op.input(1)));
  NodeValue yNV;
  ASSIGN_VALUE_OR_RETURN_ERR(yNV, getNodeValueByName(op.input(2)));

  std::string opName = loadOperatorName(op);

  // Passing -1 for multi directional broadcast, axis will be computed
  // automatically.
  Node *N = G_->createNodeWithBroadcast<SelectNode>(opName, -1, cNV, xNV, yNV);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

/// Utility function to get the RNN, GRU or LSTM direction from the proto
/// description. If not provided, the default direction is 'forward'.
static Expected<Function::RnnDirection>
getRnnDirection(const ONNX_NAMESPACE::NodeProto &op,
                ArgumentDictionaryTy &dict) {
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
      return MAKE_ERR(
          "ONNX " + op.op_type() + " 'direction' attribute is invalid!",
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
                  ArgumentDictionaryTy &dict, Function *F,
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
        activations[actIdx] = RnnActivationRelu(*F);
      } else if (actStr == "Tanh") {
        activations[actIdx] = RnnActivationTanh(*F);
      } else if (actStr == "Sigmoid") {
        activations[actIdx] = RnnActivationSigmoid(*F);
      } else {
        return MAKE_ERR(
            "ONNX " + op.op_type() + " activation '" + actStr +
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
                               ArgumentDictionaryTy &dict) {

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
    activations = {RnnActivationTanh(*G_), RnnActivationTanh(*G_)};
  } else {
    activations = {RnnActivationTanh(*G_)};
  }
  RETURN_IF_ERR(getRnnActivations(op, dict, G_, activations));

  // Activation clipping not supported (Optional)(Default: 0 for no clipping).
  RETURN_ERR_IF_NOT(!dict.count("clip"),
                    opErrMsg(op, "ONNX RNN 'clip' attribute not supported!"));

  // Get hidden size (Required).
  dim_t hiddenSize;
  RETURN_ERR_IF_NOT(
      dict.count("hidden_size"),
      opErrMsg(op, "ONNX RNN 'hidden_size' attribute is required!"));
  ASSIGN_VALUE_OR_RETURN_ERR(hiddenSize, loadInt(dict.at("hidden_size")));

  // --------------------------- Inputs ---------------------------------------
  const int numInputs = op.input_size();
  RETURN_ERR_IF_NOT((3 <= numInputs) && (numInputs <= 6),
                    opErrMsg(op, strFormat("ONNX RNN should have minimum 3 and "
                                           "maximum 6 inputs, but found %d ",
                                           numInputs)));

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
  if (numInputs > 4 && !op.input(4).empty()) {
    LOG(WARNING) << "sequence_lens ignored, will be inferred from shape of "
                    "ONNX RNN input.";
  }

  // Input5: initial_h (Optional).
  NodeValue initial_h = nullptr;
  if (numInputs > 5 && !op.input(5).empty()) {
    ASSIGN_VALUE_OR_RETURN_ERR(initial_h, getNodeValueByName(op.input(5)));
  }

  // -------------------------- Outputs ---------------------------------------
  // We allow creating placeholders for the RNN state variable Y_h for the
  // following reasons:
  // - expose the RNN state in the graph interface for accessibility (set
  //   desired state, reset state, watch the state being updated automatically).
  // - since the RNN cells are unrolled (no graph loop primitive available
  //   at this point), the optimal way to use the RNN within a model would be
  //   to have it defined with only 1 time step and have the loop in the top
  //   of the application while the RNN state will be automatically updated
  //   from one iteration (time step) to the next through the placeholders.

  // Derived parameters.
  RETURN_ERR_IF_NOT(
      X.dims().size() == 3,
      opErrMsg(op, "ONNX RNN input 'X' should have 3 dimensions!"));
  dim_t batchSize = X.dims()[1];

  // Create Y_h (hidden state) output placeholder.
  Placeholder *Y_h_ph = nullptr;
  if (onnxExportRnnStatesOpt) {
    TypeRef Htype = mod_.uniqueTypeWithNewShape(
        X.getType(), {numDirections, batchSize, hiddenSize});
    std::string Hname = opName + ".Y_h";
    ASSIGN_VALUE_OR_RETURN_ERR(Y_h_ph,
                               createAndRegisterPlaceholder(Hname, Htype));
    inputVarsByName_.try_emplace(Hname, Y_h_ph);
  }

  // Set RNN input state.
  NodeValue Y_h_init = onnxExportRnnStatesOpt ? Y_h_ph : initial_h;

  // Create ONNX RNN.
  NodeValue Y, Y_h;
  G_->createOnnxRNN(opName, X, W, R, B, Y_h_init, Y, Y_h, hiddenSize, direction,
                    activations);

  // Save RNN output state.
  if (onnxExportRnnStatesOpt) {
    G_->createSave(opName + ".Y_h.save", Y_h, Y_h_ph);
  }

  // Add node.
  const int numOutputs = op.output_size();
  if (numOutputs == 1) {
    RETURN_IF_ERR(addNodeAsOutput(op, Y));
  } else if (numOutputs == 2) {
    RETURN_IF_ERR(assignNodeOutputs(op, {Y, Y_h}));
  } else {
    return MAKE_ERR(opErrMsg(op, strFormat("ONNX RNN should have minimum 1 and "
                                           "maximum 2 outputs, but found %d ",
                                           numOutputs)));
  }
  return Error::success();
}

// Limitations:
// - Activation clipping not supported.
// - Variable sequence length not supported.
Error ONNXModelLoader::loadGRU(const ONNX_NAMESPACE::NodeProto &op,
                               ArgumentDictionaryTy &dict) {

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
    activations = {RnnActivationSigmoid(*G_), RnnActivationTanh(*G_),
                   RnnActivationSigmoid(*G_), RnnActivationTanh(*G_)};
  } else {
    activations = {RnnActivationSigmoid(*G_), RnnActivationTanh(*G_)};
  }
  RETURN_IF_ERR(getRnnActivations(op, dict, G_, activations));

  // Activation clipping not supported (Optional)(Default: 0 for no clipping).
  RETURN_ERR_IF_NOT(!dict.count("clip"),
                    opErrMsg(op, "ONNX GRU 'clip' attribute not supported!"));

  // Get hidden size (Required).
  dim_t hiddenSize;
  RETURN_ERR_IF_NOT(
      dict.count("hidden_size"),
      opErrMsg(op, "ONNX GRU 'hidden_size' attribute is required!"));
  ASSIGN_VALUE_OR_RETURN_ERR(hiddenSize, loadInt(dict.at("hidden_size")));
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
                    opErrMsg(op, strFormat("ONNX GRU should have minimum 3 and "
                                           "maximum 6 inputs, but found %d ",
                                           numInputs)));

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
  if (numInputs > 4 && !op.input(4).empty()) {
    LOG(WARNING) << "sequence_lens ignored, will be inferred from shape of "
                    "ONNX GRU input.";
  }

  // Input5: initial_h (Optional).
  NodeValue initial_h = nullptr;
  if (numInputs > 5 && !op.input(5).empty()) {
    ASSIGN_VALUE_OR_RETURN_ERR(initial_h, getNodeValueByName(op.input(5)));
  }

  // -------------------------- Outputs ---------------------------------------
  // We allow creating placeholders for the GRU state variable Y_h for the
  // following reasons:
  // - expose the GRU state in the graph interface for accessibility (set
  //   desired state, reset state, watch the state being updated automatically).
  // - since the GRU cells are unrolled (no graph loop primitive available
  //   at this point), the optimal way to use the GRU within a model would be
  //   to have it defined with only 1 time step and have the loop in the top
  //   of the application while the GRU state will be automatically updated
  //   from one iteration (time step) to the next through the placeholders.

  // Derived parameters.
  RETURN_ERR_IF_NOT(
      X.dims().size() == 3,
      opErrMsg(op, "ONNX GRU input 'X' should have 3 dimensions!"));
  dim_t batchSize = X.dims()[1];

  // Create Y_h (hidden state) output placeholder.
  Placeholder *Y_h_ph = nullptr;
  if (onnxExportRnnStatesOpt) {
    TypeRef Htype = mod_.uniqueTypeWithNewShape(
        X.getType(), {numDirections, batchSize, hiddenSize});
    std::string Hname = opName + ".Y_h";
    ASSIGN_VALUE_OR_RETURN_ERR(Y_h_ph,
                               createAndRegisterPlaceholder(Hname, Htype));
    inputVarsByName_.try_emplace(Hname, Y_h_ph);
  }

  // Set GRU input state.
  NodeValue Y_h_init = onnxExportRnnStatesOpt ? Y_h_ph : initial_h;

  // Create ONNX GRU.
  NodeValue Y, Y_h;
  G_->createOnnxGRU(opName, X, W, R, B, Y_h_init, Y, Y_h, hiddenSize, direction,
                    activations, (bool)linearBeforeReset);

  // Save GRU output state.
  if (onnxExportRnnStatesOpt) {
    G_->createSave(opName + ".Y_h.save", Y_h, Y_h_ph);
  }

  // Add node.
  const int numOutputs = op.output_size();
  if (numOutputs == 1) {
    RETURN_IF_ERR(addNodeAsOutput(op, Y));
  } else if (numOutputs == 2) {
    RETURN_IF_ERR(assignNodeOutputs(op, {Y, Y_h}));
  } else {
    return MAKE_ERR(opErrMsg(op, strFormat("ONNX GRU should have minimum 1 and "
                                           "maximum 2 outputs, but found %d ",
                                           numOutputs)));
  }
  return Error::success();
}

// Limitations:
// - Activation clipping not supported.
// - Variable sequence length not supported.
Error ONNXModelLoader::loadLSTM(const ONNX_NAMESPACE::NodeProto &op,
                                ArgumentDictionaryTy &dict) {

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
    activations = {RnnActivationSigmoid(*G_), RnnActivationTanh(*G_),
                   RnnActivationTanh(*G_),    RnnActivationSigmoid(*G_),
                   RnnActivationTanh(*G_),    RnnActivationTanh(*G_)};
  } else {
    activations = {RnnActivationSigmoid(*G_), RnnActivationTanh(*G_),
                   RnnActivationTanh(*G_)};
  }
  RETURN_IF_ERR(getRnnActivations(op, dict, G_, activations));

  // Activation clipping not supported (Optional)(Default: 0 for no clipping).
  RETURN_ERR_IF_NOT(!dict.count("clip"),
                    opErrMsg(op, "ONNX LSTM 'clip' attribute not supported!"));

  // Get hidden size (Required).
  dim_t hiddenSize;
  RETURN_ERR_IF_NOT(
      dict.count("hidden_size"),
      opErrMsg(op, "ONNX LSTM 'hidden_size' attribute is required!"));
  ASSIGN_VALUE_OR_RETURN_ERR(hiddenSize, loadInt(dict.at("hidden_size")));

  // Get input forget (Optional)(Default:0).
  int inputForget = 0;
  if (dict.count("input_forget") && dict.at("input_forget")->has_i()) {
    inputForget = dict.at("input_forget")->i();
  }

  // --------------------------- Inputs ---------------------------------------
  const int numInputs = op.input_size();
  RETURN_ERR_IF_NOT(
      (3 <= numInputs) && (numInputs <= 8),
      opErrMsg(op, strFormat("ONNX LSTM should have minimum 3 and maximum 8 "
                             "inputs, but found %d ",
                             numInputs)));

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
  if (numInputs > 4 && !op.input(4).empty()) {
    LOG(WARNING) << "sequence_lens ignored, will be inferred from shape of "
                    "ONNX LSTM input.";
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
  // We allow creating placeholders for the LSTM state variables (Y_h and Y_c)
  // for the following reasons:
  // - expose the LSTM state in the graph interface for accessibility (set
  //   desired state, reset state, watch the state being updated automatically).
  // - since the LSTM cells are unrolled (no graph loop primitive available
  //   at this point), the optimal way to use the LSTM within a model would be
  //   to have it defined with only 1 time step and have the loop in the top
  //   of the application while the LSTM state will be automatically updated
  //   from one iteration (time step) to the next through the placeholders.

  // Derived parameters.
  RETURN_ERR_IF_NOT(
      X.dims().size() == 3,
      opErrMsg(op, "ONNX LSTM input 'X' should have 3 dimensions!"));
  dim_t batchSize = X.dims()[1];

  // Create Y_h (hidden state) output placeholder.
  Placeholder *Y_h_ph = nullptr;
  if (onnxExportRnnStatesOpt) {
    TypeRef Htype = mod_.uniqueTypeWithNewShape(
        X.getType(), {numDirections, batchSize, hiddenSize});
    std::string Hname = opName + ".Y_h";
    ASSIGN_VALUE_OR_RETURN_ERR(Y_h_ph,
                               createAndRegisterPlaceholder(Hname, Htype));
    inputVarsByName_.try_emplace(Hname, Y_h_ph);
  }

  // Create Y_c (cell state) output placeholder.
  Placeholder *Y_c_ph = nullptr;
  if (onnxExportRnnStatesOpt) {
    TypeRef Ctype = mod_.uniqueTypeWithNewShape(
        X.getType(), {numDirections, batchSize, hiddenSize});
    std::string Cname = opName + ".Y_c";
    ASSIGN_VALUE_OR_RETURN_ERR(Y_c_ph,
                               createAndRegisterPlaceholder(Cname, Ctype));
    inputVarsByName_.try_emplace(Cname, Y_c_ph);
  }

  // Set LSTM input states.
  NodeValue Y_h_init = onnxExportRnnStatesOpt ? Y_h_ph : initial_h;
  NodeValue Y_c_init = onnxExportRnnStatesOpt ? Y_c_ph : initial_c;

  // Create ONNX LSTM.
  NodeValue Y, Y_h, Y_c;
  G_->createOnnxLSTM(opName, X, W, R, B, Y_h_init, Y_c_init, P, Y, Y_h, Y_c,
                     hiddenSize, direction, activations, (bool)inputForget);

  // Save LSTM output states.
  if (onnxExportRnnStatesOpt) {
    G_->createSave(opName + ".Y_h.save", Y_h, Y_h_ph);
    G_->createSave(opName + ".Y_c.save", Y_c, Y_c_ph);
  }

  // Add node.
  const int numOutputs = op.output_size();
  if (numOutputs == 1) {
    RETURN_IF_ERR(addNodeAsOutput(op, Y));
  } else if (numOutputs == 2) {
    RETURN_IF_ERR(assignNodeOutputs(op, {Y, Y_h}));
  } else if (numOutputs == 3) {
    RETURN_IF_ERR(assignNodeOutputs(op, {Y, Y_h, Y_c}));
  } else {
    return MAKE_ERR(
        opErrMsg(op, strFormat("ONNX LSTM should have minimum 1 and "
                               "maximum 3 outputs, but found %d ",
                               numOutputs)));
  }
  return Error::success();
}

Error ONNXModelLoader::loadClip(const ONNX_NAMESPACE::NodeProto &op,
                                const ArgumentDictionaryTy &dict) {
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  float cmin = std::numeric_limits<float>::lowest();
  if (opsetVersion_ > 10 && op.input_size() > 1 && !op.input(1).empty()) {
    // min value is optional and might not be supplied.
    Constant *minC = getConstantByNameOrNull(op.input(1));
    RETURN_ERR_IF_NOT(minC, "Expect constant for min value in Clip operator.");
    cmin = minC->getPayload().getHandle().raw(0);
  } else if (dict.count("min")) {
    ASSIGN_VALUE_OR_RETURN_ERR(cmin, loadFloat(dict.find("min")->second));
  }

  // Windows headers define `max` macro, so have to wrap the function name in
  // parenthesis to avoid compilation error.
  float cmax = (std::numeric_limits<float>::max)();
  if (opsetVersion_ > 10 && op.input_size() > 2 && !op.input(2).empty()) {
    // max value is optional and might not be supplied.
    Constant *maxC = getConstantByNameOrNull(op.input(2));
    RETURN_ERR_IF_NOT(maxC, "Expect constant for max value in Clip operator.");
    cmax = maxC->getPayload().getHandle().raw(0);
  } else if (dict.count("max")) {
    ASSIGN_VALUE_OR_RETURN_ERR(cmax, loadFloat(dict.find("max")->second));
  }

  auto *node = G_->createClip(loadOperatorName(op), in, cmin, cmax);
  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return Error::success();
}

Error ONNXModelLoader::loadCmpEQ(const ONNX_NAMESPACE::NodeProto &op,
                                 ArgumentDictionaryTy &dict) {
  NodeValue LHS;
  ASSIGN_VALUE_OR_RETURN_ERR(LHS, getNodeValueByName(op.input(0)));
  NodeValue RHS;
  ASSIGN_VALUE_OR_RETURN_ERR(RHS, getNodeValueByName(op.input(1)));

  Node *N = G_->createNodeWithBroadcast<CmpEQNode>(loadOperatorName(op),
                                                   /* axis */ -1, LHS, RHS);
  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadCmpLTE(const ONNX_NAMESPACE::NodeProto &op,
                                  ArgumentDictionaryTy &dict) {
  NodeValue LHS;
  ASSIGN_VALUE_OR_RETURN_ERR(LHS, getNodeValueByName(op.input(0)));
  NodeValue RHS;
  ASSIGN_VALUE_OR_RETURN_ERR(RHS, getNodeValueByName(op.input(1)));

  Node *N = G_->createNodeWithBroadcast<CmpLTENode>(loadOperatorName(op),
                                                    /* axis */ -1, LHS, RHS);
  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

/// Takes a list of NodeValues \p inputs and broadcasts them to a common shape
/// \p broadcastShape based on the maximum value along each dimension.
static Error getShapeForBroadcast(llvm::ArrayRef<NodeValue> inputs,
                                  std::vector<dim_t> &broadcastShape) {
  std::vector<uint32_t> numDims;
  for (auto &N : inputs) {
    numDims.push_back(N.dims().size());
  }
  const uint32_t outputNumDims =
      *std::max_element(numDims.begin(), numDims.end());
  for (uint32_t i = 0; i < outputNumDims; i++) {
    std::vector<dim_t> dims;
    for (uint32_t j = 0; j < inputs.size(); j++) {
      auto vals = inputs[j].dims();
      if (vals.size() > i) {
        dims.push_back(vals[vals.size() - 1 - i]);
      }
    }
    broadcastShape.insert(broadcastShape.begin(),
                          *std::max_element(dims.begin(), dims.end()));
  }
  return Error::success();
}

Error ONNXModelLoader::loadMean(const ONNX_NAMESPACE::NodeProto &op,
                                ArgumentDictionaryTy &dict) {
  size_t numInputTensors = op.input_size();
  if (numInputTensors == 1) {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    RETURN_IF_ERR(addNodeAsOutput(op, in));
  } else {
    const std::string &opName = loadOperatorName(op);
    llvm::SmallVector<NodeValue, 4> inputTensors;
    inputTensors.reserve(numInputTensors);
    for (unsigned i = 0; i < numInputTensors; i++) {
      NodeValue in;
      ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(i)));
      inputTensors.push_back(in);
    }
    std::vector<dim_t> broadcastShape;
    RETURN_IF_ERR(getShapeForBroadcast(inputTensors, broadcastShape));
    for (unsigned i = 0; i < numInputTensors; i++) {
      auto &in = inputTensors[i];
      int axis = broadcastShape.size() - in.dims().size();
      in = G_->createBroadcast(opName, in, broadcastShape, axis);
      in = G_->createExpandDims(opName, in, {0});
    }
    ConcatNode *concat = G_->createConcat(opName, inputTensors, /* axis */ 0);
    Node *N = G_->createBatchedReduceMean(opName, concat, /* axis */ {0});
    RETURN_IF_ERR(addNodeAsOutput(op, N));
  }
  return Error::success();
}

Error ONNXModelLoader::loadSelect(const ONNX_NAMESPACE::NodeProto &op,
                                  ArgumentDictionaryTy &dict) {
  NodeValue Cond;
  ASSIGN_VALUE_OR_RETURN_ERR(Cond, getNodeValueByName(op.input(0)));
  NodeValue LHS;
  ASSIGN_VALUE_OR_RETURN_ERR(LHS, getNodeValueByName(op.input(1)));
  NodeValue RHS;
  ASSIGN_VALUE_OR_RETURN_ERR(RHS, getNodeValueByName(op.input(2)));

  std::vector<dim_t> shape;
  ASSIGN_VALUE_OR_RETURN_ERR(shape, getShape<dim_t>(dict["shape"]));

  auto outTy = mod_.uniqueType(LHS.getElementType(), shape);
  Node *N = G_->createSelect(loadOperatorName(op), outTy, Cond, LHS, RHS);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadNonZero(const ONNX_NAMESPACE::NodeProto &op,
                                   const ArgumentDictionaryTy &dict) {
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));

  Constant *C = getConstantByNameOrNull(op.input(0));
  RETURN_ERR_IF_NOT(C,
                    opErrMsg(op, "NonZero Only constant shape is supported!"));

  // output tensor.
  Tensor outT;

  // Fold NonZero operator.
  auto foldNonZero = [&C, &outT](auto dummy) -> Error {
    auto inH = C->getPayload().getHandle<decltype(dummy)>();
    auto dims = C->dims();

    // First pass over the input is used to find the number of non-zero elements
    // so we can create output tensor that will be filled in the 2nd pass.
    dim_t nonZeroCnt = 0;
    for (dim_t idx = 0, e = inH.size(); idx < e; idx++) {
      nonZeroCnt += (inH.raw(idx) != 0) ? 1 : 0;
    }

    // No need to support zero Tensor (empty output); we support constant input
    // only and such input likely means it's an invalid model or the part of
    // graph can be removed.
    RETURN_ERR_IF_NOT(nonZeroCnt > 0,
                      "Non-Zero input with all zeroes is not supported.");

    // Create output tensor. First dimension is the rank of input tensor, second
    // dimension is the number of non-zero elements.
    outT.reset(ElemKind::Int64ITy, {(dim_t)dims.size(), nonZeroCnt});

    // Strides for each dimensions, needed to calculate NonZero output.
    std::vector<dim_t> strides;
    strides.resize(dims.size());
    strides[dims.size() - 1] = 1;
    if (dims.size() > 1) {
      for (int i = dims.size() - 2; i >= 0; i--) {
        strides[i] = dims[i + 1] * strides[i + 1];
      }
    }

    // Second pass over the input is used to fill the output tensor. For each
    // non-zero element we fill all the dimensions, at position determined
    // by the non-zero element's index when zero elements are ignored.
    auto outH = outT.getHandle<int64_t>();
    for (dim_t idx = 0, pos = 0, e = inH.size(); idx < e; idx++) {
      if (inH.raw(idx) != 0) {
        for (dim_t dim = 0; dim < dims.size(); dim++) {
          outH.at({dim, pos}) = (idx / strides[dim]) % dims[dim];
        }
        pos++;
      }
    }
    return Error::success();
  };

  std::string err;
  if (C->getElementType() == ElemKind::FloatTy) {
    RETURN_IF_ERR(foldNonZero((float)0));
  } else if (C->getElementType() == ElemKind::Int64ITy) {
    RETURN_IF_ERR(foldNonZero((int64_t)0));
  } else if (C->getElementType() == ElemKind::Int32ITy) {
    RETURN_IF_ERR(foldNonZero((int32_t)0));
  } else {
    return MAKE_ERR(
        opErrMsg(op, "NonZero: Unsupported input type for NonZero operator."
                     "(Supports Float, Int32 and Int64)"));
  }

  Constant *outC = G_->getParent()->createConstant("nonZero", std::move(outT));
  RETURN_IF_ERR(addNodeAsOutput(op, outC));
  return Error::success();
}

Error ONNXModelLoader::loadQuantize(const ONNX_NAMESPACE::NodeProto &op,
                                    ArgumentDictionaryTy &dict) {
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
  auto outTy = mod_.uniqueType(elemKind, outDims, scale, offset);
  Node *N = G_->createQuantize(loadOperatorName(op), in, outTy);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadQuantizeLinear(const ONNX_NAMESPACE::NodeProto &op,
                                          ArgumentDictionaryTy &dict) {
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  // Onnx documentation expects input to be Float or Int32.
  if (!(in.getElementType() == ElemKind::FloatTy ||
        in.getElementType() == ElemKind::Int32ITy)) {
    return MAKE_ERR(
        opErrMsg(op, "QuantizeLinear supports input to be Float or Int32."));
  }

  // Only scale with constant is supported.
  Constant *scale;
  ASSIGN_VALUE_OR_RETURN_ERR(scale, getConstantByName(op.input(1)));

  // Glow supports only per layer scale.
  if (!((scale->getType()->dims().size() == 1 &&
         scale->getType()->dims()[0] == 1) ||
        scale->getType()->dims().size() == 0)) {
    return MAKE_ERR(opErrMsg(op, "QuantizeLinear: y_scale scalar value is only"
                                 " supported."));
  }

  float scaleValue = scale->getPayload().getHandle<float>().raw(0);

  // Default values as per Onnx documentation.
  int32_t offsetValue = 0;
  auto type = ElemKind::UInt8QTy;

  // Check if we have a offset vector.
  if (op.input_size() > 2) {
    auto &offsetTensorName = op.input(2);
    // Only offset with constant is supported.
    Constant *offset = nullptr;
    // Load the serialized offset vector.
    ASSIGN_VALUE_OR_RETURN_ERR(offset, getConstantByName(offsetTensorName));
    if (!((offset->getType()->dims().size() == 1 &&
           offset->getType()->dims()[0] == 1) ||
          offset->getType()->dims().size() == 0)) {
      return MAKE_ERR(
          opErrMsg(op, "QuantizeLinear: y_zero_point scalar value is only"
                       " supported."));
    }

    type = offset->getElementType();
    // Only uint8 and int8 values are supported as per onnx.
    if (type == ElemKind::UInt8QTy) {
      offsetValue = static_cast<int32_t>(
          offset->getPayload().getHandle<uint8_t>().raw(0));
    } else if (type == ElemKind::Int8QTy) {
      offsetValue =
          static_cast<int32_t>(offset->getPayload().getHandle<int8_t>().raw(0));
    } else {
      // This condition is hit when there is onnx graph creation issue or
      // constant is not created correctly.
      return MAKE_ERR(
          opErrMsg(op, "QuantizeLinear: Supports only uint8 or int8 data in"
                       " y_zero_point"));
    }
  }

  auto outDims = in.getType()->dims();
  auto outTy = mod_.uniqueType(type, outDims, scaleValue, offsetValue);
  Node *N = G_->createQuantize(loadOperatorName(op), in, outTy);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadConvertTo(const ONNX_NAMESPACE::NodeProto &op,
                                     ArgumentDictionaryTy &dict) {
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  const auto *attr = dict.at("shape");
  RETURN_ERR_IF_NOT(
      attr->has_t(),
      opErrMsg(op, "ConvertTo should have t() field as \"shape\""));
  const auto &t = attr->t();
  std::vector<dim_t> shape;
  for (const auto d : t.dims()) {
    shape.push_back(d);
  }

  auto type = ElemKind::FloatTy;
  RETURN_IF_ERR(onnxTensorDataTypeToElemKind(t.data_type(), &type));
  auto outTy = mod_.uniqueType(type, shape);
  Node *N = G_->createConvertTo(loadOperatorName(op), in, outTy);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadDequantize(const ONNX_NAMESPACE::NodeProto &op,
                                      ArgumentDictionaryTy &dict) {
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  Node *N = G_->createDequantize(loadOperatorName(op), in, ElemKind::FloatTy);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadRegression(const ONNX_NAMESPACE::NodeProto &op,
                                      ArgumentDictionaryTy &dict) {
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  NodeValue expected;
  ASSIGN_VALUE_OR_RETURN_ERR(expected, getNodeValueByName(op.input(1)));

  Node *N = G_->createRegression(loadOperatorName(op), in, expected);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadBatchedAdd(const ONNX_NAMESPACE::NodeProto &op,
                                      ArgumentDictionaryTy &dict) {
  NodeValue batch;
  ASSIGN_VALUE_OR_RETURN_ERR(batch, getNodeValueByName(op.input(0)));
  NodeValue sample;
  ASSIGN_VALUE_OR_RETURN_ERR(sample, getNodeValueByName(op.input(1)));

  Node *N = G_->createBatchedAdd(loadOperatorName(op), batch, sample);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadCumSum(const ONNX_NAMESPACE::NodeProto &op,
                                  ArgumentDictionaryTy &dict) {
  if (op.input_size() > 1) {
    Expected<NodeValue> axis = getNodeValueByName(op.input(1));
    if (axis) {
      if (auto *AC = llvm::dyn_cast<Constant>(axis->getNode())) {
        RETURN_ERR_IF_NOT(AC->getPayload().dims().size() == 1,
                          opErrMsg(op, "CumSum axis must be 0-D"));
        RETURN_ERR_IF_NOT(AC->getPayload().dims()[0] == 1,
                          opErrMsg(op, "CumSum axis must be 0-D"));
        RETURN_ERR_IF_NOT(AC->getHandle<int32_t>().at(0) == 0,
                          opErrMsg(op, "CumSum only supports axis == 0"));
      } else {
        return MAKE_ERR(opErrMsg(op, "Axis must be Constant"));
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

  // TODO: add axis/dim support
  Node *N =
      G_->createCumSum(loadOperatorName(op), input, 0, exclusive, reverse);
  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadScatterAssign(const ONNX_NAMESPACE::NodeProto &op,
                                         ArgumentDictionaryTy &dict) {
  NodeValue data;
  ASSIGN_VALUE_OR_RETURN_ERR(data, getNodeValueByName(op.input(0)));
  NodeValue indices;
  ASSIGN_VALUE_OR_RETURN_ERR(indices, getNodeValueByName(op.input(1)));
  NodeValue slices;
  ASSIGN_VALUE_OR_RETURN_ERR(slices, getNodeValueByName(op.input(2)));

  Node *N = G_->createScatterData(loadOperatorName(op), data, indices, slices);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadIntLookupTable(const ONNX_NAMESPACE::NodeProto &op,
                                          ArgumentDictionaryTy &dict) {
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  if (in.getType()->getElementType() == ElemKind::Int8QTy) {
    std::vector<int8_t> values;
    ASSIGN_VALUE_OR_RETURN_ERR(values, getShape<int8_t>(dict["values"]));
    std::vector<dim_t> shape;
    ASSIGN_VALUE_OR_RETURN_ERR(shape, getShape<dim_t>(dict["shape"]));

    auto outTy = mod_.uniqueType(in.getElementType(), shape);
    Node *N = G_->createIntLookupTable<int8_t>(loadOperatorName(op), in, values,
                                               outTy);

    RETURN_IF_ERR(addNodeAsOutput(op, N));
    return Error::success();
  } else if (in.getType()->getElementType() == ElemKind::Int16QTy) {
    std::vector<int16_t> values;
    ASSIGN_VALUE_OR_RETURN_ERR(values, getShape<int16_t>(dict["values"]));
    std::vector<dim_t> shape;
    ASSIGN_VALUE_OR_RETURN_ERR(shape, getShape<dim_t>(dict["shape"]));

    auto outTy = mod_.uniqueType(in.getElementType(), shape);
    Node *N = G_->createIntLookupTable<int16_t>(loadOperatorName(op), in,
                                                values, outTy);

    RETURN_IF_ERR(addNodeAsOutput(op, N));
    return Error::success();
  } else {
    return MAKE_ERR(strFormat("Lookup table type '%s' not supported!",
                              in.getType()->getElementName().str().c_str()));
  }
}

Error ONNXModelLoader::loadLengthsRangeFill(const ONNX_NAMESPACE::NodeProto &op,
                                            ArgumentDictionaryTy &dict) {
  NodeValue lengths;
  ASSIGN_VALUE_OR_RETURN_ERR(lengths, getNodeValueByName(op.input(0)));
  unsigned_t size;
  ASSIGN_VALUE_OR_RETURN_ERR(size, loadInt(dict.at("size")));

  Node *N = G_->createLengthsRangeFill(loadOperatorName(op), lengths, size);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadRescaleQuantized(const ONNX_NAMESPACE::NodeProto &op,
                                            ArgumentDictionaryTy &dict) {
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  float scale;
  ASSIGN_VALUE_OR_RETURN_ERR(scale, loadFloat(dict.at("scale")));
  unsigned_t offset;
  ASSIGN_VALUE_OR_RETURN_ERR(offset, loadInt(dict.at("offset")));

  auto inTy = in.getType();
  auto outTy =
      mod_.uniqueType(inTy->getElementType(), inTy->dims(), scale, offset);

  Node *N = G_->createRescaleQuantized(loadOperatorName(op), in, outTy);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadRowwiseQuantizedSparseLengthsWeightedSum(
    const ONNX_NAMESPACE::NodeProto &op, ArgumentDictionaryTy &dict) {
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
  LengthsMode lengthsMode;
  ASSIGN_VALUE_OR_RETURN_ERR(lengthsMode, getLengthsMode(dict));

  Node *N = G_->createRowwiseQuantizedSparseLengthsWeightedSum(
      loadOperatorName(op), data, scales, offsets, weights, indices, lengths,
      /* precision */ ElemKind::FloatTy, /* useFP16Accumulation */ false,
      lengthsMode);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadFusedRowwiseQuantizedSparseLengthsWeightedSum(
    const ONNX_NAMESPACE::NodeProto &op, ArgumentDictionaryTy &dict) {
  NodeValue data;
  ASSIGN_VALUE_OR_RETURN_ERR(data, getNodeValueByName(op.input(0)));
  NodeValue weights;
  ASSIGN_VALUE_OR_RETURN_ERR(weights, getNodeValueByName(op.input(1)));
  NodeValue indices;
  ASSIGN_VALUE_OR_RETURN_ERR(indices, getNodeValueByName(op.input(2)));
  NodeValue lengths;
  ASSIGN_VALUE_OR_RETURN_ERR(lengths, getNodeValueByName(op.input(3)));
  LengthsMode lengthsMode;
  ASSIGN_VALUE_OR_RETURN_ERR(lengthsMode, getLengthsMode(dict));

  Node *N = G_->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
      loadOperatorName(op), data, weights, indices, lengths,
      /* useFP16Accumulation */ false, lengthsMode);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadFusedRowwiseQuantizedSparseLengthsSum(
    const ONNX_NAMESPACE::NodeProto &op, ArgumentDictionaryTy &dict) {
  NodeValue data;
  ASSIGN_VALUE_OR_RETURN_ERR(data, getNodeValueByName(op.input(0)));
  NodeValue indices;
  ASSIGN_VALUE_OR_RETURN_ERR(indices, getNodeValueByName(op.input(1)));
  NodeValue lengths;
  ASSIGN_VALUE_OR_RETURN_ERR(lengths, getNodeValueByName(op.input(2)));
  LengthsMode lengthsMode;
  ASSIGN_VALUE_OR_RETURN_ERR(lengthsMode, getLengthsMode(dict));

  Storage *dataS = llvm::dyn_cast<Storage>(data);
  Node *N = G_->createFusedRowwiseQuantizedSparseLengthsSum(
      loadOperatorName(op), dataS, indices, lengths,
      /* useFP16Accumulation */ false, lengthsMode);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadFullyConnected(const ONNX_NAMESPACE::NodeProto &op,
                                          ArgumentDictionaryTy &dict) {
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  NodeValue w;
  ASSIGN_VALUE_OR_RETURN_ERR(w, getNodeValueByName(op.input(1)));
  NodeValue b;
  ASSIGN_VALUE_OR_RETURN_ERR(b, getNodeValueByName(op.input(2)));

  unsigned_t axis = 1;
  if (dict.count("axis")) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        axis, loadAxis<unsigned_t>(dict.at("axis"), in.dims().size()));
  }

  Node *N = G_->createFullyConnected(loadOperatorName(op), in, w, b, axis);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadRowwiseQuantizedFullyConnected(
    const ONNX_NAMESPACE::NodeProto &op, ArgumentDictionaryTy &dict) {
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));

  NodeValue weights;
  ASSIGN_VALUE_OR_RETURN_ERR(weights, getNodeValueByName(op.input(1)));
  auto *weightsC = llvm::dyn_cast<Constant>(weights.getNode());

  NodeValue scales;
  ASSIGN_VALUE_OR_RETURN_ERR(scales, getNodeValueByName(op.input(2)));
  auto *scalesC = llvm::dyn_cast<Constant>(scales.getNode());

  NodeValue offsets;
  ASSIGN_VALUE_OR_RETURN_ERR(offsets, getNodeValueByName(op.input(3)));
  auto *offsetsC = llvm::dyn_cast<Constant>(offsets.getNode());

  NodeValue bias;
  ASSIGN_VALUE_OR_RETURN_ERR(bias, getNodeValueByName(op.input(4)));
  auto *biasC = llvm::dyn_cast<Constant>(bias.getNode());

  float outScale;
  ASSIGN_VALUE_OR_RETURN_ERR(outScale, loadFloat(dict.at("out_scale")));
  int32_t outOffset;
  ASSIGN_VALUE_OR_RETURN_ERR(outOffset, loadInt(dict.at("out_offset")));

  auto outTy =
      mod_.uniqueType(ElemKind::Int8QTy, {input.dims()[0], weights.dims()[0]},
                      outScale, outOffset);

  Node *N = G_->createRowwiseQuantizedFullyConnected(
      loadOperatorName(op) + ".rowwise_quantized_fc", input, weightsC, scalesC,
      offsetsC, biasC, outTy);

  return addNodeAsOutput(op, N);
}

Error ONNXModelLoader::loadNonMaxSuppression(
    const ONNX_NAMESPACE::NodeProto &op, ArgumentDictionaryTy &dict,
    bool isV4) {
  NodeValue boxesNV;
  ASSIGN_VALUE_OR_RETURN_ERR(boxesNV, getNodeValueByName(op.input(0)));
  NodeValue scoresNV;
  ASSIGN_VALUE_OR_RETURN_ERR(scoresNV, getNodeValueByName(op.input(1)));
  unsigned maxOutputBoxesPerClass = 0;
  Constant *maxOutputBoxesPerClassC = nullptr;
  if (op.input_size() > 2 && !op.input(2).empty()) {
    maxOutputBoxesPerClassC = getConstantByNameOrNull(op.input(2));
    RETURN_ERR_IF_NOT(maxOutputBoxesPerClassC,
                      "NMS: maxOutputBoxesPerClass is not a constant tensor.");
    if (maxOutputBoxesPerClassC->getPayload().getElementType() ==
        ElemKind::Int64ITy) {
      maxOutputBoxesPerClass =
          maxOutputBoxesPerClassC->getPayload().getHandle<int64_t>().raw(0);
    } else if (maxOutputBoxesPerClassC->getPayload().getElementType() ==
               ElemKind::Int32ITy) {
      maxOutputBoxesPerClass =
          maxOutputBoxesPerClassC->getPayload().getHandle<int32_t>().raw(0);
    } else {
      return MAKE_ERR("NMS: Unsupported type for maxoutputboxesperclass.");
    }
  }
  float iouThreshold = 0.0f;
  Constant *iouThresholdC = nullptr;
  if (op.input_size() > 3 && !op.input(3).empty()) {
    iouThresholdC = getConstantByNameOrNull(op.input(3));
    RETURN_ERR_IF_NOT(iouThresholdC,
                      "NMS: iouThreshold is not a constant tensor.");
    iouThreshold = iouThresholdC->getPayload().getHandle<float>().raw(0);
  }
  float scoreThreshold = 0.0f;
  Constant *scoreThresholdC = nullptr;
  if (op.input_size() > 4 && !op.input(4).empty()) {
    scoreThresholdC = getConstantByNameOrNull(op.input(4));
    RETURN_ERR_IF_NOT(scoreThresholdC,
                      "NMS: scoreThreshold is not a constant tensor.");
    scoreThreshold = scoreThresholdC->getPayload().getHandle<float>().raw(0);
  }

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
    RETURN_ERR_IF_NOT(
        padToMaxOutputSize == 1,
        opErrMsg(op, "NonMaxSuppressionV4 does not support non-padding mode."));
  }

  // Create Node.
  std::string opName = loadOperatorName(op);
  Node *N = nullptr;

  if (isV4) {
    N = G_->createNonMaxSuppressionV4(opName, boxesNV, scoresNV, centerPointBox,
                                      maxOutputBoxesPerClass, iouThreshold,
                                      scoreThreshold);
  } else {
    N = G_->createNonMaxSuppressionONNX(opName, boxesNV, scoresNV,
                                        centerPointBox, maxOutputBoxesPerClass,
                                        iouThreshold, scoreThreshold);
  }
  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadSplat(const ONNX_NAMESPACE::NodeProto &op,
                                 ArgumentDictionaryTy &dict) {
  return loadConstantOfShape(op, dict, true /* isSplat */);
}

Error ONNXModelLoader::loadInsertTensor(const ONNX_NAMESPACE::NodeProto &op,
                                        ArgumentDictionaryTy &dict) {
  NodeValue big;
  ASSIGN_VALUE_OR_RETURN_ERR(big, getNodeValueByName(op.input(0)));
  NodeValue small;
  ASSIGN_VALUE_OR_RETURN_ERR(small, getNodeValueByName(op.input(1)));

  std::vector<dim_t> start;
  ASSIGN_VALUE_OR_RETURN_ERR(start, getShape<dim_t>(dict["start"]));

  unsigned_t count = 1;
  if (dict.count("count")) {
    ASSIGN_VALUE_OR_RETURN_ERR(count, loadInt(dict.at("count")));
  }

  unsigned_t axis = 0;
  if (dict.count("axis")) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        axis, loadAxis<unsigned_t>(dict.at("axis"), big.dims().size()));
  }

  Node *N = G_->createInsertTensor(loadOperatorName(op), big, small, start,
                                   count, axis);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadIdentity(const ONNX_NAMESPACE::NodeProto &op,
                                    ArgumentDictionaryTy &dict) {
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  RETURN_IF_ERR(addNodeAsOutput(op, in));
  return Error::success();
}

Error ONNXModelLoader::loadAdaptiveAvgPool(const ONNX_NAMESPACE::NodeProto &op,
                                           ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));

  std::vector<unsigned_t> outputShape;
  ASSIGN_VALUE_OR_RETURN_ERR(outputShape,
                             getShape<unsigned_t>(dict["output_size"]));

  ShapeNHWC idim(input.dims());

  auto outTy = mod_.uniqueTypeWithNewShape(
      input.getType(), {idim.n, outputShape[0], outputShape[1], idim.c});

  Node *N = G_->createAdaptiveAvgPool(opName, input, outTy);

  return addNodeAsOutput(op, N);
}

Error ONNXModelLoader::loadFlip(const ONNX_NAMESPACE::NodeProto &op,
                                ArgumentDictionaryTy &dict) {
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));

  unsigned_t axis = 0;
  if (dict.count("axis")) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        axis, loadAxis<unsigned_t>(dict.at("axis"), input.dims().size()));
  }

  Node *N = G_->createFlip(loadOperatorName(op) + ".flip", input, axis);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadAudioSpectrogram(const ONNX_NAMESPACE::NodeProto &op,
                                            ArgumentDictionaryTy &dict) {
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));

  // Get window size (Required).
  int64_t windowSize;
  RETURN_ERR_IF_NOT(
      dict.count("window_size"),
      "ONNX AudioSpectrogram 'window_size' attribute is required!");
  ASSIGN_VALUE_OR_RETURN_ERR(windowSize, loadInt(dict.at("window_size")));

  // Get window stride (Required).
  int64_t windowStride;
  RETURN_ERR_IF_NOT(dict.count("stride"),
                    "ONNX AudioSpectrogram 'stride' attribute is required!");
  ASSIGN_VALUE_OR_RETURN_ERR(windowStride, loadInt(dict.at("stride")));

  // Get magnitude squared flag (Optional)(Default: 1).
  int magnitudeSquared = 1;
  if (dict.count("magnitude_squared") &&
      dict.at("magnitude_squared")->has_i()) {
    magnitudeSquared = dict.at("magnitude_squared")->i();
  }

  Node *N = G_->createAudioSpectrogram(loadOperatorName(op), input, windowSize,
                                       windowStride, (bool)magnitudeSquared);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadROIAlign(const ONNX_NAMESPACE::NodeProto &op,
                                    ArgumentDictionaryTy &dict) {
  NodeValue featureMap;
  ASSIGN_VALUE_OR_RETURN_ERR(featureMap, getNodeValueByName(op.input(0)));
  NodeValue boxes;
  ASSIGN_VALUE_OR_RETURN_ERR(boxes, getNodeValueByName(op.input(1)));
  NodeValue batchIndices;
  ASSIGN_VALUE_OR_RETURN_ERR(batchIndices, getNodeValueByName(op.input(2)));

  PoolingMode mode = PoolingMode::AVG;
  if (dict.count("mode")) {
    std::string modeStr;
    ASSIGN_VALUE_OR_RETURN_ERR(modeStr, loadStr(dict.at("mode")));
    if (modeStr == "avg") {
      mode = PoolingMode::AVG;
    } else if (modeStr == "max") {
      mode = PoolingMode::MAX;
    } else {
      return MAKE_ERR(strFormat("Invalid PoolingMode: %s", modeStr.c_str()));
    }
  }

  bool rotated = false;
  if (dict.count("rotated")) {
    ASSIGN_VALUE_OR_RETURN_ERR(rotated, loadInt(dict.at("rotated")));
  }

  bool aligned = false;
  if (dict.count("aligned")) {
    ASSIGN_VALUE_OR_RETURN_ERR(aligned, loadInt(dict.at("aligned")));
  }

  uint32_t outputHeight = 1;
  if (dict.count("output_height")) {
    ASSIGN_VALUE_OR_RETURN_ERR(outputHeight, loadInt(dict.at("output_height")));
  }

  uint32_t outputWidth = 1;
  if (dict.count("output_width")) {
    ASSIGN_VALUE_OR_RETURN_ERR(outputWidth, loadInt(dict.at("output_width")));
  }

  uint32_t samplingRatio = 0;
  if (dict.count("sampling_ratio")) {
    ASSIGN_VALUE_OR_RETURN_ERR(samplingRatio,
                               loadInt(dict.at("sampling_ratio")));
  }

  float spatialScale = 1.0;
  if (dict.count("spatial_scale")) {
    ASSIGN_VALUE_OR_RETURN_ERR(spatialScale,
                               loadFloat(dict.at("spatial_scale")));
  }

  const std::string &opName = loadOperatorName(op);
  featureMap = G_->createTranspose(opName, featureMap, NCHW2NHWC);
  Node *N = G_->createROIAlign(opName, featureMap, boxes, batchIndices,
                               outputHeight, outputWidth, samplingRatio,
                               spatialScale, aligned, rotated, mode);
  N = G_->createTranspose(opName, N, NHWC2NCHW);
  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadMFCC(const ONNX_NAMESPACE::NodeProto &op,
                                ArgumentDictionaryTy &dict) {
  NodeValue spectrogram;
  ASSIGN_VALUE_OR_RETURN_ERR(spectrogram, getNodeValueByName(op.input(0)));

  // Get sample rate [Hz] (Required).
  float sampleRate;
  RETURN_ERR_IF_NOT(dict.count("sample_rate"),
                    "ONNX MFCC 'sample_rate' attribute is required!");
  ASSIGN_VALUE_OR_RETURN_ERR(sampleRate, loadFloat(dict.at("sample_rate")));

  // Get lower frequency [Hz] (Required).
  float lowerFrequency;
  RETURN_ERR_IF_NOT(dict.count("lower_frequency_limit"),
                    "ONNX MFCC 'lower_frequency_limit' attribute is required!");
  ASSIGN_VALUE_OR_RETURN_ERR(lowerFrequency,
                             loadFloat(dict.at("lower_frequency_limit")));

  // Get upper frequency [Hz] (Required).
  float upperFrequency;
  RETURN_ERR_IF_NOT(dict.count("upper_frequency_limit"),
                    "ONNX MFCC 'upper_frequency_limit' attribute is required!");
  ASSIGN_VALUE_OR_RETURN_ERR(upperFrequency,
                             loadFloat(dict.at("upper_frequency_limit")));

  // Get filter bank count (Required).
  int64_t filterBankCount;
  RETURN_ERR_IF_NOT(
      dict.count("filterbank_channel_count"),
      "ONNX MFCC 'filterbank_channel_count' attribute is required!");
  ASSIGN_VALUE_OR_RETURN_ERR(filterBankCount,
                             loadInt(dict.at("filterbank_channel_count")));

  // Get number of coefficients (Required).
  int64_t numCoefficients;
  RETURN_ERR_IF_NOT(dict.count("dct_coefficient_count"),
                    "ONNX MFCC 'dct_coefficient_count' attribute is required!");
  ASSIGN_VALUE_OR_RETURN_ERR(numCoefficients,
                             loadInt(dict.at("dct_coefficient_count")));

  Node *N = G_->createMFCC(loadOperatorName(op), spectrogram, sampleRate,
                           lowerFrequency, upperFrequency, filterBankCount,
                           numCoefficients);

  RETURN_IF_ERR(addNodeAsOutput(op, N));
  return Error::success();
}

Error ONNXModelLoader::loadAsin(const ONNX_NAMESPACE::NodeProto &op,
                                const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  auto outTy = mod_.uniqueType(*(in.getType()));
  Node *node = G_->createAsin(opName, outTy, in);
  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return Error::success();
}

Error ONNXModelLoader::loadAcos(const ONNX_NAMESPACE::NodeProto &op,
                                const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  auto outTy = mod_.uniqueType(*(in.getType()));
  Node *node = G_->createAcos(opName, outTy, in);
  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return Error::success();
}

Error ONNXModelLoader::loadAtan(const ONNX_NAMESPACE::NodeProto &op,
                                const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  auto outTy = mod_.uniqueType(*(in.getType()));
  Node *node = G_->createAtan(opName, outTy, in);
  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return Error::success();
}

Error ONNXModelLoader::loadSign(const ONNX_NAMESPACE::NodeProto &op,
                                const ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  Node *node = G_->createSign(opName, in);
  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return Error::success();
}

Error ONNXModelLoader::loadSoftmax(const ONNX_NAMESPACE::NodeProto &op,
                                   const ArgumentDictionaryTy &dict) {

  const std::string &opName = loadOperatorName(op);
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  RETURN_ERR_IF_NOT(in.dims().size() >= 2, "SoftMax input dims must be >= 2");

  // Create a constant to store labels to be used in SoftMaxGradNode.
  auto selected =
      mod_.createConstant(ElemKind::Int64ITy, {in.dims()[0], 1}, "selected");

  if (opsetVersion_ == 13) {
    int axis = in.dims().size() - 1;
    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(
          axis, loadAxis<int>(dict.at("axis"), in.dims().size()));
    }
    RETURN_ERR_IF_NOT(in.dims().size() == 4, "SoftMax 13 input dims must be 4");
    // Compute the shuffle layout  based on axis input.
    std::vector<unsigned_t> shuffle;
    std::vector<unsigned_t> shuffleBack;
    switch (axis) {
    case 0:
      shuffle = {1u, 2u, 3u, 0u};
      shuffleBack = {3u, 0u, 1u, 2u};
      break;

    case 1:
      shuffle = {0u, 2u, 3u, 1u};
      shuffleBack = {0u, 3u, 1u, 2u};
      break;

    case 2:
      shuffle = {0u, 1u, 3u, 2u};
      shuffleBack = {0u, 1u, 3u, 2u};
      break;

    case 3:
      shuffle = {0u, 1u, 2u, 3u};
      shuffleBack = {0u, 1u, 2u, 3u};
      break;

    default:
      return MAKE_ERR("SoftMax Axis must be <=3");
      break;
    }
    auto *NH = G_->createTranspose(opName, in, shuffle);
    auto *FN = G_->createFlattenV1("reshapeInput", NH, axis);
    auto *SM = G_->createSoftMax(opName, FN, selected);

    // The output should have the same shape as the original input.
    auto origInDims = NH->getResult().dims();
    auto *RN = G_->createReshape("reshapeOutput", SM, origInDims);
    auto *NC = G_->createTranspose(opName, RN, shuffleBack);
    RETURN_IF_ERR(addNodeAsOutput(op, NC));
  } else {
    // ONNX allows shapes like <N x 10 x 1 x 1 >. Flatten the inputs to the
    // softmax function. This is basimilar to a bitcast operation.
    int axis = 1;
    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(
          axis, loadAxis<int>(dict.at("axis"), in.dims().size()));
    }
    auto *FN = G_->createFlatten("reshapeInput", in, axis);
    auto *SM = G_->createSoftMax(opName, FN, selected);

    // The output should have the same shape as the original input.
    auto origInDims = in.getType()->dims();
    auto *RN = G_->createReshape("reshapeOutput", SM, origInDims);
    RETURN_IF_ERR(addNodeAsOutput(op, RN));
  }
  return Error::success();
}

Error ONNXModelLoader::loadLogSoftmax(const ONNX_NAMESPACE::NodeProto &op,
                                      const ArgumentDictionaryTy &dict) {

  const std::string &opName = loadOperatorName(op);
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

  RETURN_ERR_IF_NOT(in.dims().size() >= 2,
                    "LogSoftMax input dims must be >= 2");

  // Create a constant to store labels to be used in SoftMaxGradNode.
  auto selected =
      mod_.createConstant(ElemKind::Int64ITy, {in.dims()[0], 1}, "selected");

  if (opsetVersion_ == 13) {
    int axis = in.dims().size() - 1;
    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(
          axis, loadAxis<int>(dict.at("axis"), in.dims().size()));
    }
    RETURN_ERR_IF_NOT(in.dims().size() == 4,
                      "LogSoftMax 13 input dims must be 4");
    // Compute the shuffle layout  based on axis input.
    std::vector<unsigned_t> shuffle;
    std::vector<unsigned_t> shuffleBack;
    switch (axis) {
    case 0:
      shuffle = {1u, 2u, 3u, 0u};
      shuffleBack = {3u, 0u, 1u, 2u};
      break;

    case 1:
      shuffle = {0u, 2u, 3u, 1u};
      shuffleBack = {0u, 3u, 1u, 2u};
      break;

    case 2:
      shuffle = {0u, 1u, 3u, 2u};
      shuffleBack = {0u, 1u, 3u, 2u};
      break;

    case 3:
      shuffle = {0u, 1u, 2u, 3u};
      shuffleBack = {0u, 1u, 2u, 3u};
      break;

    default:
      return MAKE_ERR("LogSoftMax Axis must be <=3");
      break;
    }
    auto *NH = G_->createTranspose(opName, in, shuffle);
    auto *FN = G_->createFlattenV1("reshapeInput", NH, axis);
    auto *SM = G_->createLogSoftMax(opName, FN, selected);

    // The output should have the same shape as the original input.
    auto origInDims = NH->getResult().dims();
    auto *RN = G_->createReshape("reshapeOutput", SM, origInDims);
    auto *NC = G_->createTranspose(opName, RN, shuffleBack);
    RETURN_IF_ERR(addNodeAsOutput(op, NC));
  } else {
    // ONNX allows shapes like <N x 10 x 1 x 1 >. Flatten the inputs to the
    // logsoftmax function. This is basimilar to a bitcast operation.
    int axis = 1;
    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(
          axis, loadAxis<int>(dict.at("axis"), in.dims().size()));
    }
    auto *FN = G_->createFlatten("reshapeInput", in, axis);
    auto *SM = G_->createLogSoftMax(opName, FN, selected);

    // The output should have the same shape as the original input.
    auto origInDims = in.getType()->dims();
    auto *RN = G_->createReshape("reshapeOutput", SM, origInDims);
    RETURN_IF_ERR(addNodeAsOutput(op, RN));
  }
  return Error::success();
}

Error ONNXModelLoader::loadScatterData(const ONNX_NAMESPACE::NodeProto &op,
                                       const ArgumentDictionaryTy &dict) {

  const std::string &opName = loadOperatorName(op);
  NodeValue data;
  ASSIGN_VALUE_OR_RETURN_ERR(data, getNodeValueByName(op.input(0)));
  NodeValue indices;
  ASSIGN_VALUE_OR_RETURN_ERR(indices, getNodeValueByName(op.input(1)));
  NodeValue values;
  ASSIGN_VALUE_OR_RETURN_ERR(values, getNodeValueByName(op.input(2)));

  RETURN_ERR_IF_NOT(indices.dims().size() == 2,
                    opErrMsg(op, "Indices must be a 2D tensor!"));
  RETURN_ERR_IF_NOT(indices.dims()[0] == values.dims()[0],
                    opErrMsg(op, "Indices and values must have same lengths!"));

  bool cumulative = false;
  if (dict.count("cumulative")) {
    ASSIGN_VALUE_OR_RETURN_ERR(cumulative, loadInt(dict.at("cumulative")));
  }

  Node *node = G_->createScatterData(opName, data, indices, values, cumulative);
  RETURN_IF_ERR(addNodeAsOutput(op, node));
  return Error::success();
}

Error ONNXModelLoader::loadTopK(const ONNX_NAMESPACE::NodeProto &op,
                                ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);
  NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
  RETURN_ERR_IF_NOT(
      op.input_size() <= 2,
      opErrMsg(
          op,
          strFormat(
              "TopK: Maximum number of inputs is 2, but found input size %d ",
              op.input_size())));
  unsigned_t k = 0;
  if (op.input_size() > 1) {
    Constant *kConst = getConstantByNameOrNull(op.input(1));
    RETURN_ERR_IF_NOT(
        kConst, opErrMsg(op, "TopK: Non-constant k is not supported by Glow."));
    RETURN_ERR_IF_NOT(
        kConst->getElementType() == ElemKind::Int64ITy,
        opErrMsg(op,
                 strFormat("TopK: k input must be of type Int64, but found "
                           "input type '%s' ",
                           kConst->getType()->getElementName().str().c_str())));
    auto constH = kConst->getPayload().getHandle<int64_t>();
    k = constH.at({0});
  } else {
    ASSIGN_VALUE_OR_RETURN_ERR(k, loadInt(dict["k"]));
  }

  int lastDim = in.dims().size() - 1;
  int axis = lastDim;
  if (dict.count("axis")) {
    ASSIGN_VALUE_OR_RETURN_ERR(axis,
                               loadAxis<int>(dict["axis"], in.dims().size()));
  }

  RETURN_ERR_IF_NOT(
      axis == lastDim,
      opErrMsg(
          op,
          strFormat(
              "TopK: Currently only support axis %d being last dimension %d ",
              axis, lastDim)));

  auto *R = G_->createTopK(opName, in, k);
  RETURN_IF_ERR(addNodeAsOutput(op, R));
  return Error::success();
}

Error ONNXModelLoader::loadLoop(const ONNX_NAMESPACE::NodeProto &op,
                                const ArgumentDictionaryTy &dict) {
  int64_t maxTripCount;
  bool ignoreMaxTripCount = op.input(0).empty();
  if (!ignoreMaxTripCount) {
    Constant *M = getConstantByNameOrNull(op.input(0));
    RETURN_ERR_IF_NOT(M, "Loop operator M input must be a constant.");
    RETURN_ERR_IF_NOT(M->getElementType() == ElemKind::Int64ITy,
                      "Loop operator M input must be int64.");
    maxTripCount = (M->getPayload().getHandle<int64_t>()).raw(0);

    RETURN_ERR_IF_NOT(
        maxTripCount >= 0,
        strFormat("Loop operator trip count (%ld) must be positive",
                  maxTripCount));
  }

  bool condOrig = false;
  bool ignoreCond = op.input(1).empty();
  if (!ignoreCond) {
    Constant *cond = getConstantByNameOrNull(op.input(1));
    RETURN_ERR_IF_NOT(cond, "Loop operator cond input must be a constant.");
    RETURN_ERR_IF_NOT(cond->getElementType() == ElemKind::BoolTy,
                      "Loop operator cond input must be bool.");
    condOrig = (cond->getPayload().getHandle<bool>()).raw(0);
  }

  RETURN_ERR_IF_NOT(dict.count("body"), "Loop body not found.");
  auto body = dict.at("body")->g();

  // 2 + N (i.e., maximum trip-count, cond, and N)
  const int numLoopInputs = op.input_size();
  // N + K (final N loop carried dependency values then K scan_outputs)
  const int numLoopOutputs = op.output_size();
  // 2 + N (i.e., iteration_num, condition, and N loop carried dependencies)
  const int numBodyInputs = body.input_size();
  // 1 + N + K (i.e., condition, N loop carried dependencies, and K
  // scan_outputs)
  const int numBodyOutputs = body.output_size();

  RETURN_ERR_IF_NOT(numLoopInputs >= 2 && numLoopInputs == numBodyInputs &&
                        numLoopOutputs == numBodyOutputs - 1 &&
                        numLoopOutputs >= 1,
                    "Mismatched inputs/outputs of Loop and subgraph.");

  const int numK = numBodyOutputs - numBodyInputs + 1;
  const int numN = numLoopInputs - 2;

  CHECK_GE(numN, 0) << "Invalid number of v_initial in Loop operator : "
                    << numN;
  CHECK_GE(numK, 0) << "Invalid number of scan_outputs in Loop operator : "
                    << numK;

  // Handle a loop with no iterations.
  if ((!ignoreMaxTripCount && maxTripCount == 0) ||
      (!ignoreCond && !condOrig)) {
    // No need to load the subgraph, just connect loop's N v_initial to
    // N v_final and empty Tensor for K scan_outputs.
    llvm::SmallVector<NodeValue, 4> outputs;
    outputs.reserve(numLoopOutputs);
    // Connect N v_initial to N v_final.
    for (int i = 0; i < numN; ++i) {
      NodeValue in;
      ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(i + 2)));
      outputs.push_back(in);
    }
    // Connect empty Tensors for K scan_outputs.
    for (int k = 0; k < numK; ++k) {
      int ki = (body.input_size() - 1) + k;
      auto scan_output = body.output(ki);
      Type ty;
      ASSIGN_VALUE_OR_RETURN_ERR(ty, getTensorType(scan_output));
      Tensor T(ty);
      T.zero();
      Constant *c = G_->getParent()->createConstant("empty", std::move(T));
      outputs.push_back(G_->createExpandDims("unsqueeze_K", c, {0}));
    }
    RETURN_IF_ERR(assignNodeOutputs(op, outputs));
    return Error::success();
  }

  // Now, there is at least one iteration.
  llvm::SmallVector<NodeValue, 4> inputs;
  inputs.reserve(numBodyInputs);

  // Collect names of N loop carried dependencies from Loop op. It will be used
  // to connect Loop's inputs with subgraph's inputs for the first iteration.
  llvm::SmallVector<llvm::StringRef, 4> namesOfLoopNs;
  namesOfLoopNs.reserve(numN);
  for (int i = 0; i < numN; ++i) {
    namesOfLoopNs.push_back(op.input(i + 2));
  }

  // Collect output node names for the next iteration. It will be used when
  // connecting subgraph's outputs with subgraph's input between iteration. Add
  // names just for N loop carried dependencies. No need to add names for K
  // scan_outputs because they are not input of subgraph.
  llvm::SmallVector<llvm::StringRef, 4> namesOfBodyOutputNs;
  namesOfBodyOutputNs.reserve(numN);
  for (int i = 0; i < numN; ++i) {
    namesOfBodyOutputNs.push_back(body.output(i + 1).name());
  }

  // Collect names of K scan_outputs.
  llvm::SmallVector<llvm::SmallVector<NodeValue, 2>, 2> scanOutputKs;
  scanOutputKs.reserve(numK);
  for (int k = 0; k < numK; ++k) {
    llvm::SmallVector<NodeValue, 2> scanOutputIter;
    scanOutputIter.reserve(loopUnrollLimit);
    scanOutputKs.push_back(scanOutputIter);
  }

  auto getDummyCond = [&]() -> Expected<NodeValue> {
    RETURN_ERR_IF_NOT(ignoreCond, "Unexpected empty name in cond in Loop");
    Tensor dummyCondT(ElemKind::BoolTy, {1});
    dummyCondT.zero();
    Constant *dummyCondNode =
        G_->getParent()->createConstant("dumpCond", std::move(dummyCondT));
    return dummyCondNode->getOutput();
  };

  auto getIterationNumConst = [&](int64_t val) -> Constant * {
    Tensor T(ElemKind::Int64ITy, {1});
    T.getHandle<int64_t>() = {val};
    Constant *C = G_->getParent()->createConstant("const", std::move(T));
    return C;
  };

  auto prepareNextIteration =
      [&](int64_t iterationNum, NodeValue condNode,
          llvm::ArrayRef<llvm::StringRef> outputNames) -> Error {
    inputs.clear();
    // Set iteration_num.
    inputs.push_back(getIterationNumConst(iterationNum));
    // Set condition.
    inputs.push_back(condNode);
    // Set N loop carried dependencies.
    for (auto oName : outputNames) {
      NodeValue in;
      ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(oName));
      inputs.push_back(in);
    }
    RETURN_IF_ERR(assignGraphInputs(body, inputs));
    return Error::success();
  };

  auto accumulateScanOutputs = [&]() -> Error {
    // Accumulate scan-outputs values.
    for (int k = 0; k < numK; ++k) {
      int ki = (body.input_size() - 1) + k;
      auto scan_output = body.output(ki);
      NodeValue outK;
      ASSIGN_VALUE_OR_RETURN_ERR(outK, getNodeValueByName(scan_output.name()));
      Node *unsqueezedOutK = G_->createExpandDims("unsqueezed_K", outK, {0});
      scanOutputKs[k].push_back(unsqueezedOutK);
    }
    return Error::success();
  };

  auto loadSubgraph = [&](ONNX_NAMESPACE::GraphProto &graphDef) -> Error {
    RETURN_IF_ERR(loadInitializers(graphDef));
    RETURN_IF_ERR(loadNetwork(graphDef, false));
    return Error::success();
  };

  auto canUnrollNextIter = [&](int64_t iterNum) -> Expected<bool> {
    if (!ignoreMaxTripCount && iterNum >= maxTripCount) {
      return false;
    }
    Constant *condOut = getConstantByNameOrNull(body.output(0).name());
    RETURN_ERR_IF_NOT(condOut,
                      "Loop exit condition is unpredictable to be unrolled.");
    bool cond = (condOut->getPayload().getHandle<bool>()).raw(0) || ignoreCond;
    if (!cond) {
      return false;
    }
    RETURN_ERR_IF_NOT(
        iterNum < loopUnrollLimit,
        strFormat("Exceed the unroll limit (%u) while unrolling Loop operator.",
                  loopUnrollLimit.getValue()));
    return cond;
  };

  // Unroll the first iteration by connecting Loop's inputs to subgraph's
  // inputs.
  int64_t iterNum = 0;
  NodeValue condNode;
  ASSIGN_VALUE_OR_RETURN_ERR(condNode, op.input(1).empty()
                                           ? getDummyCond()
                                           : getNodeValueByName(op.input(1)));
  RETURN_IF_ERR(prepareNextIteration(iterNum, condNode, namesOfLoopNs));
  RETURN_IF_ERR(loadSubgraph(body));
  RETURN_IF_ERR(accumulateScanOutputs());
  ++iterNum;
  auto condCheck = canUnrollNextIter(iterNum);

  RETURN_ERR_IF_NOT(condCheck, ERR_TO_STRING(condCheck.takeError()));

  // Unroll remaining iterations by connecting outputs of previous iteration
  // with inputs of current iteration.
  while (condCheck && condCheck.get()) {
    NodeValue condOutNode;
    ASSIGN_VALUE_OR_RETURN_ERR(condOutNode,
                               getNodeValueByName(body.output(0).name()));
    RETURN_IF_ERR(
        prepareNextIteration(iterNum, condOutNode, namesOfBodyOutputNs));
    RETURN_IF_ERR(loadSubgraph(body));
    RETURN_IF_ERR(accumulateScanOutputs());
    ++iterNum;
    condCheck = canUnrollNextIter(iterNum);
    RETURN_ERR_IF_NOT(condCheck, ERR_TO_STRING(condCheck.takeError()));
  }

  // Hook final subgraph outputs to loop outputs.
  llvm::SmallVector<NodeValue, 4> outputs;
  outputs.reserve(numLoopOutputs);
  // Set outputs for N loop carried dependency values.
  for (int i = 0; i < numN; ++i) {
    NodeValue bodyout;
    ASSIGN_VALUE_OR_RETURN_ERR(bodyout,
                               getNodeValueByName(body.output(i + 1).name()));
    outputs.push_back(bodyout);
  }
  // Set outputs for K scan_outputs.
  for (int k = 0; k < numK; ++k) {
    outputs.push_back(G_->createConcat("concat", scanOutputKs[k], 0));
  }
  RETURN_IF_ERR(assignNodeOutputs(op, outputs));
  return Error::success();
}

Expected<TypeRef>
ONNXModelLoader::loadTypeFromAttributes(unsigned resNo,
                                        ArgumentDictionaryTy &dict) {
  // Load ElemKind.
  std::string elemKindStr;
  ASSIGN_VALUE_OR_RETURN_ERR(
      elemKindStr, loadStr(dict[getTypeAttrID(resNo, elemKindSignifier)]));
  const ElemKind k = Type::getElementKindFromName(elemKindStr);

  // Load Shape. Note that we allow for empty shapes here because 0 dimensional
  // shapes are allowed (representing scalars).
  std::vector<dim_t> shape;
  ASSIGN_VALUE_OR_RETURN_ERR(
      shape, getShape<dim_t>(dict[getTypeAttrID(resNo, shapeSignifier)],
                             /* allowEmptyShape */ true));

  // Load strides. Note that we allow for empty strides here because 0
  // dimensional shapes are allowed (representing scalars).
  std::vector<dim_t> strides;
  auto stridesKey = getTypeAttrID(resNo, stridesSignifier);
  if (dict.count(stridesKey)) {
    ASSIGN_VALUE_OR_RETURN_ERR(strides,
                               getShape<dim_t>(dict[stridesKey],
                                               /* allowEmptyShape */ true));
  }

  // Create and return uniqued non-quantized Type.
  if (!isQuantizedElemKind(k)) {
    return getTypeWithCustomStrides(mod_, mod_.uniqueType(k, shape), strides);
  }

  // Must be quantized kind, so get scale/offset and create and return uniqued
  // quantized Type.
  float scale;
  ASSIGN_VALUE_OR_RETURN_ERR(
      scale, loadFloat(dict[getTypeAttrID(resNo, qScaleSignifier)]));
  int32_t offset;
  ASSIGN_VALUE_OR_RETURN_ERR(
      offset, loadInt(dict[getTypeAttrID(resNo, qOffsetSignifier)]));

  // If we have a scale of dummyScale, then this must be a dummy pair of
  // scale/offset. Look up the actual scale/offset to use as previously loaded,
  // using the offset as the key to updatedTQPs_. Skip fused kinds because
  // scales are already dummies.
  if (replaceDummyTQPs_ && scale == dummyScale &&
      !isFusedQuantizedElemKind(k)) {
    TensorQuantizationParams TQP;
    ASSIGN_VALUE_OR_RETURN_ERR(TQP, getUpdatedTQP(offset));
    scale = TQP.scale;
    offset = TQP.offset;
  }

  return getTypeWithCustomStrides(
      mod_, mod_.uniqueType(k, shape, scale, offset), strides);
}

Expected<Node *>
ONNXModelLoader::tryLoadGlowCustomOp(llvm::StringRef typeName,
                                     const ONNX_NAMESPACE::NodeProto &op,
                                     ArgumentDictionaryTy &dict) {
  const std::string &opName = loadOperatorName(op);

// Try all automatically generated import cases.
#include "glow/AutoGenNodesImport.h"

  // If we get here then no case handled the op, so return nullptr.
  return nullptr;
}

/// Load Node options for \p loadedNode from \p dict and set in \p nodeInfo.
/// These are specified in the format "NodeOpt_BACKENDNAME_OPTIONNAME".
static Error loadPerNodeOptions(const Node *loadedNode,
                                BackendSpecificNodeInfo &nodeInfo,
                                ArgumentDictionaryTy &dict) {
  // Look through all attributes in the dict for ones that have NodeOpt_ prefix.
  for (const auto &attrPair : dict) {
    // Split across the first '_' and check if it has the "NodeOpt" prefix.
    auto splitPair = llvm::StringRef(attrPair.first).split('_');
    if (splitPair.first == attrPair.first && splitPair.first == "") {
      // No '_' found, so continue.
      continue;
    }
    if (splitPair.first != nodeOptSignifier) {
      // Prefix is not "NodeOpt_", so continue.
      continue;
    }

    // Must have a NodeOpt, so check it has strings and load them into nodeInfo.
    const ONNX_NAMESPACE::AttributeProto *attr = attrPair.second;
    RETURN_ERR_IF_NOT(attr->strings_size() > 0,
                      strFormat("%s in %s has no strings",
                                attrPair.first.c_str(),
                                loadedNode->getName().data()));
    std::vector<std::string> &attrVals =
        nodeInfo[loadedNode->getParent()][loadedNode][splitPair.second];
    for (const std::string &s : attr->strings()) {
      attrVals.push_back(s);
    }
  }
  return Error::success();
}

Error ONNXModelLoader::loadIf(const ONNX_NAMESPACE::NodeProto &op,
                              const ArgumentDictionaryTy &dict) {
  Constant *condition = getConstantByNameOrNull(op.input(0));
  RETURN_ERR_IF_NOT(condition, "Only constant condition is supported!");
  RETURN_ERR_IF_NOT(condition->getElementType() == ElemKind::BoolTy,
                    "Condition must be boolean!");

  RETURN_ERR_IF_NOT(dict.count("then_branch"), "Then branch not found!");
  RETURN_ERR_IF_NOT(dict.count("else_branch"), "Else branch not found!");
  RETURN_ERR_IF_NOT(dict.at("then_branch")->type() ==
                        ONNX_NAMESPACE::AttributeProto::GRAPH,
                    "Only Subgraph branches are supported.");
  RETURN_ERR_IF_NOT(dict.at("else_branch")->type() ==
                        ONNX_NAMESPACE::AttributeProto::GRAPH,
                    "Only Subgraph branches are supported.");

  auto loadSubgraph = [&](ONNX_NAMESPACE::GraphProto &graphDef) -> Error {
    RETURN_IF_ERR(loadInitializers(graphDef));
    RETURN_IF_ERR(loadNetwork(graphDef, false));
    return Error::success();
  };

  if (condition->getPayload().getHandle<bool>().isZero()) {
    auto ifFalse = dict.at("else_branch")->g();
    RETURN_ERR_IF_NOT(ifFalse.output_size() == 1,
                      "Only single output 'else' subgraph is supported.");
    RETURN_IF_ERR(loadSubgraph(ifFalse));
    NodeValue ifFalseVal;
    ASSIGN_VALUE_OR_RETURN_ERR(ifFalseVal,
                               getNodeValueByName(ifFalse.output(0).name()));
    RETURN_IF_ERR(addNodeAsOutput(op, ifFalseVal));
  } else {
    auto ifTrue = dict.at("then_branch")->g();
    RETURN_ERR_IF_NOT(ifTrue.output_size() == 1,
                      "Only single output 'then' subgraph is supported.");
    RETURN_IF_ERR(loadSubgraph(ifTrue));
    NodeValue ifTrueVal;
    ASSIGN_VALUE_OR_RETURN_ERR(ifTrueVal,
                               getNodeValueByName(ifTrue.output(0).name()));
    RETURN_IF_ERR(addNodeAsOutput(op, ifTrueVal));
  }
  return Error::success();
}

Error ONNXModelLoader::loadOperator(const ONNX_NAMESPACE::NodeProto &op) {
  ArgumentDictionaryTy dict = loadArgumentMap(op);
  const std::string &typeName = op.op_type();
  mod_.registerOriginalName(op.name());

  if (useGlowCustomOps_) {
    Node *loadedNode;
    ASSIGN_VALUE_OR_RETURN_ERR(loadedNode,
                               tryLoadGlowCustomOp(typeName, op, dict));
    if (loadedNode) {
      if (!perNodeOpts_) {
        return Error::success();
      }
      return loadPerNodeOptions(loadedNode, *perNodeOpts_, dict);
    }

    // These are handled earlier when loading initializers and inputs and so can
    // be safely ignored here.
    if (typeName == constFoldSubgraphNodeName ||
        typeName == staticPHDummyNodeName) {
      return Error::success();
    }

    // Identity is the only official ONNX op used with useGlowCustomOps. Let it
    // fall through to logic to handle below, otherwise return error.
    if (typeName != "Identity") {
      return MAKE_ERR("Failed to load operator " + typeName + " .",
                      ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_OPERATOR);
    }
  }

  // Check if operator is supported in parent class, CommonOperatorLoader.
  bool tryLoadCommonOperatorResult;
  ASSIGN_VALUE_OR_RETURN_ERR(tryLoadCommonOperatorResult,
                             tryLoadCommonOperator(typeName, op, dict));
  if (tryLoadCommonOperatorResult) {
    return Error::success();
  }
  if (typeName == "Loop") {
    return loadLoop(op, dict);
  }

  if (typeName == "Constant") {
    return loadConstant(op, dict);
  }
  if (typeName == "Range") {
    return loadRange(op, dict);
  }
  if (typeName == "PRelu") {
    return loadPRelu(op, dict);
  }
  if (typeName == "Slice") {
    return loadSlice(op, dict);
  }
  if (typeName == "Sin" || typeName == "Cos") {
    return loadTrigonometricOps(typeName, op, dict);
  }
  if (typeName == "Erf") {
    return loadErf(op, dict);
  }
  if (typeName == "Conv") {
    // If the Conv operator has quantized inputs and
    // dict contains the scale and offset params, use
    // loadTensorwiseQuantizedConvolution.
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    return in.getType()->isQuantizedType() && dict.count("out_scale") &&
                   dict.count("out_offset")
               ? loadTensorwiseQuantizedConvolution(op, dict)
               : loadConv(op, dict);
  }
  if (typeName == "ChannelwiseQuantizedConvolution") {
    return loadChannelwiseQuantizedConvolution(op, dict);
  }
  if (typeName == "MaxPool" || typeName == "AveragePool") {
    // If the pool operator has quantized inputs, use
    // loadTensorwiseQuantizedPool.
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    return in.getType()->isQuantizedType() && dict.count("out_scale") &&
                   dict.count("out_offset")
               ? loadTensorwiseQuantizedPool(op, dict, typeName)
               : loadPool(op, dict, typeName);
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
  if (typeName == "InstanceNormalization") {
    return loadInstanceNormalization(op, dict);
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
  if (typeName == "ReduceSumSquare") {
    return loadReduceOp(typeName, op, dict);
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
  if (typeName == "HardSigmoid") {
    return loadHardSigmoid(op, dict);
  }
  if (typeName == "LeakyRelu") {
    return loadLeakyRelu(op, dict);
  }
  if (typeName == "SpaceToDepth") {
    return loadSpaceToDepth(op, dict);
  }
  if (typeName == "DepthToSpace") {
    return loadDepthToSpace(op, dict);
  }
  if (typeName == "ReduceL2") {
    return loadReduceL2(op, dict);
  }
  if (typeName == "ConstantOfShape") {
    return loadConstantOfShape(op, dict, false /* isSplat */);
  }
  if (typeName == "Tile") {
    return loadTile(op, dict);
  }
  if (typeName == "Expand") {
    return loadExpand(op, dict);
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
  if (typeName == "Clip") {
    return loadClip(op, dict);
  }
  if (typeName == "Equal") {
    return loadCmpEQ(op, dict);
  }
  if (typeName == "CmpLTE") {
    return loadCmpLTE(op, dict);
  }
  if (typeName == "Mean") {
    return loadMean(op, dict);
  }
  if (typeName == "Select") {
    return loadSelect(op, dict);
  }
  if (typeName == "Quantize") {
    return loadQuantize(op, dict);
  }
  if (typeName == "QuantizeLinear") {
    return loadQuantizeLinear(op, dict);
  }
  if (typeName == "ConvertTo") {
    return loadConvertTo(op, dict);
  }
  if ((typeName == "Dequantize") || (typeName == "DequantizeLinear")) {
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
  if ((typeName == "ScatterAssign") || (typeName == "ScatterND")) {
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
  if (typeName == "ArgMin") {
    return loadArgMinMax(op, dict, true);
  }
  if (typeName == "ArgMax") {
    return loadArgMinMax(op, dict, false);
  }
  if (typeName == "NonMaxSuppressionV4") {
    return loadNonMaxSuppression(op, dict, true);
  }
  if (typeName == "NonMaxSuppression") {
    return loadNonMaxSuppression(op, dict, false);
  }
  if (typeName == "ConvTranspose") {
    return loadConvTranspose(op, dict);
  }
  if (typeName == "If") {
    return loadIf(op, dict);
  }
  if (typeName == "AdaptiveAvgPool") {
    return loadAdaptiveAvgPool(op, dict);
  }
  if (typeName == "Flip") {
    return loadFlip(op, dict);
  }
  if (typeName == "AudioSpectrogram") {
    return loadAudioSpectrogram(op, dict);
  }
  if (typeName == "RoiAlign") {
    return loadROIAlign(op, dict);
  }
  if (typeName == "MFCC") {
    return loadMFCC(op, dict);
  }
  if (typeName == "Identity") {
    return loadIdentity(op, dict);
  }
  if (typeName == "Upsample") {
    return loadUpsample(op, dict);
  }
  if (typeName == "Resize") {
    return loadResize(op, dict);
  }
  if (typeName == "NonZero") {
    return loadNonZero(op, dict);
  }
  if (typeName == "Acos") {
    return loadAcos(op, dict);
  }
  if (typeName == "Asin") {
    return loadAsin(op, dict);
  }
  if (typeName == "Atan") {
    return loadAtan(op, dict);
  }
  if (typeName == "Sign") {
    return loadSign(op, dict);
  }
  if (typeName == "Softmax") {
    return loadSoftmax(op, dict);
  }
  if (typeName == "LogSoftmax") {
    return loadLogSoftmax(op, dict);
  }
  if (typeName == "ScatterData") {
    return loadScatterData(op, dict);
  }
  if (typeName == "TopK") {
    return loadTopK(op, dict);
  }

  return MAKE_ERR("Failed to load operator " + typeName + " .",
                  ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_OPERATOR);
}

void ONNXModelLoader::deleteConstFoldFunctions() {
  for (Function *constFoldF : constFoldFuns_) {
    mod_.eraseFunction(constFoldF);
  }
}

Expected<Constant *>
ONNXModelLoader::runDeserializedConstFold(llvm::StringRef initializerName,
                                          llvm::StringRef outputName) {
  NodeValue NV;
  ASSIGN_VALUE_OR_RETURN_ERR(NV, getNodeValueByName(outputName));

  // Force folding single splats, because we're folding a constant folding
  // subgraph, and so we know the exported model already decided to fold it
  // (normally the backend decides whether to fold it or not).
  std::vector<Constant *> constResults =
      constantFold(NV.getNode(), /* foldSingleSplats */ true);
  RETURN_ERR_IF_NOT(constResults.size() > 0,
                    strFormat("Constant folding did not occur for %s",
                              NV.getNode()->getName().data()));
  RETURN_ERR_IF_NOT(NV.getResNo() < constResults.size(),
                    strFormat("Needed result %u from const folding results, "
                              "but only got %lu results",
                              NV.getResNo(), constResults.size()));
  Constant *foldedC = constResults[NV.getResNo()];

  // Now we have the final Constant we want and it exists in the module. Set its
  // name to the actual initializer it came with if not already named that.
  if (foldedC->getName() != initializerName) {
    RETURN_ERR_IF_NOT(
        mod_.getConstantByName(initializerName) == nullptr,
        strFormat("Already had a Constant by name %s", initializerName.data()));
    foldedC->setName(initializerName);
  }
  RETURN_ERR_IF_NOT(
      nodeValueByName_.count(initializerName) == 0,
      strFormat("Should not have been a Constant by name %s registered yet",
                initializerName.data()));
  nodeValueByName_[initializerName] = foldedC->getOutput();

  return foldedC;
}

Expected<Constant *> ONNXModelLoader::replaySerializedConstFold(
    const ONNX_NAMESPACE::TensorProto &in, ONNX_NAMESPACE::GraphProto &net) {
  // Check if ins has a constant folding node associated with it.
  const char *constFoldNodeName = nullptr;
  int resNo = -1;
  for (const auto &keyVal : in.external_data()) {
    if (keyVal.key() == "ConstFoldNodeName") {
      constFoldNodeName = keyVal.value().data();
      continue;
    }
    if (keyVal.key() == "ConstFoldResNo") {
      ASSIGN_VALUE_OR_RETURN_ERR(resNo, getIntFromStr(keyVal.value()));
      continue;
    }
  }
  if (!constFoldNodeName) {
    return nullptr;
  }
  RETURN_ERR_IF_NOT(resNo >= 0,
                    "Require ConstFoldResNo for Glow__ConstFoldSubgraph");

  // Look through the ops in the graph to find the Node we need by name.
  ONNX_NAMESPACE::NodeProto *op = nullptr;
  for (int i = 0; i < net.node_size(); i++) {
    auto *curOp = net.mutable_node(i);
    if (loadOperatorName(*curOp) == constFoldNodeName) {
      op = curOp;
      break;
    }
  }
  RETURN_ERR_IF_NOT(
      op, strFormat("Did not find Node by name %s", constFoldNodeName));
  RETURN_ERR_IF_NOT(
      op->op_type() == constFoldSubgraphNodeName,
      strFormat("Node %s has type %s but expected Glow__ConstFoldSubgraph",
                constFoldNodeName, op->op_type().data()));

  // Now look through the Node's attributes to find the subgraph.
  ONNX_NAMESPACE::GraphProto *subgraph = nullptr;
  for (auto &arg : *op->mutable_attribute()) {
    if (arg.name() == "ConstFoldSubgraph") {
      subgraph = arg.mutable_g();
      break;
    }
  }

  RETURN_ERR_IF_NOT(subgraph, strFormat("Expected associated subgraph for %s",
                                        constFoldNodeName));

  // We have the constant folding subgraph proto and need to load it to run it.
  const bool functionAlreadyLoaded = mod_.hasFunction(constFoldNodeName);
  Function *constFoldF = functionAlreadyLoaded
                             ? mod_.getFunction(constFoldNodeName)
                             : mod_.createFunction(constFoldNodeName);
  const auto insert = constFoldFuns_.insert(constFoldF);
  RETURN_ERR_IF_NOT(!(functionAlreadyLoaded && insert.second),
                    strFormat("Function %s should only be processed once",
                              constFoldNodeName));

  // Temporarily swap in state for the constant folding Function.
  Function *origF = G_;
  G_ = constFoldF;
  llvm::StringMap<Function *> partNameToFunBackup;
  std::swap(partNameToFun_, partNameToFunBackup);
  // Make sure to restore original state of the loader when exiting this scope.
  ScopeGuard restoreOrigStateGuard([&]() {
    G_ = origF;
    std::swap(partNameToFun_, partNameToFunBackup);
  });

  // Deserialize the Function if not already done.
  if (!functionAlreadyLoaded) {
    RETURN_IF_ERR(loadNetwork(*subgraph, /* loadingConstFoldSubgraph */ true));
  }

  // Now that we have the Function deserialized, actually run and return the
  // resulting Constant that is foldled.
  RETURN_ERR_IF_NOT(subgraph->output_size() > resNo,
                    strFormat("ConstFoldResNo %d invalid output idx.", resNo));
  return runDeserializedConstFold(in.name(), subgraph->output(resNo).name());
}

Error ONNXModelLoader::loadInitializers(ONNX_NAMESPACE::GraphProto &net) {
  // Load the network initializers:
  for (const auto &in : net.initializer()) {
    // Replay any constant folding that occurred from previous optimization if
    // necessary. foldedC will be left as nullptr if no constant folding occurs.
    Constant *foldedC;
    ASSIGN_VALUE_OR_RETURN_ERR(foldedC, replaySerializedConstFold(in, net));

    std::string layout = ANY_LAYOUT;
    if (useGlowCustomOps_) {
      ASSIGN_VALUE_OR_RETURN_ERR(
          layout, getAttrFromDocString(layoutSignifier, in.doc_string()));
    }

    // If we already an existing module then expect to find Constants already
    // existing for each initializer.
    if (foldedC || loadIntoExistingModule_) {
      Constant *C = foldedC ? foldedC : mod_.getConstantByName(in.name());
      Type ty;
      ASSIGN_VALUE_OR_RETURN_ERR(ty, getTensorType(in));

      // If the expected type is fused, and we are processing an initializer
      // with payload that already exists in the Module, then set the type to
      // fused here. This is because Caffe2 and ONNX (non-Glow-custom) protos do
      // not support fused ElemKinds, so we should explicitly set them as we do
      // during Caffe2ModelLoading.
      if (!foldedC && loadIntoExistingModule_ && ty.isFusedQuantizedType()) {
        RETURN_IF_ERR(setFusedTy(C, mod_.uniqueType(ty)));
      }

      RETURN_IF_ERR(verifyPreexistingStorage(C, in.name(), ty, layout));
      nodeValueByName_[in.name()] = C->getOutput();
      continue;
    }

    // If we are loading into an existing module then we would expect this
    // initializer doesn't have any data associated with it.
    Tensor T;
    RETURN_IF_ERR(loadTensor(in, &T, useGlowCustomOps_));
    RETURN_IF_ERR(createAndRegisterConstant(in.name(), std::move(T), layout));
  }

  return Error::success();
}

Error ONNXModelLoader::setOutputNodes(ONNX_NAMESPACE::GraphProto &net) {
  if (net.output_size() == 0) {
    return MAKE_ERR("Net output size must be greater than 0");
  }

  for (int i = 0; i < net.output_size(); i++) {
    const auto &outputName = net.output(i).name();
    NodeValue r;
    ASSIGN_VALUE_OR_RETURN_ERR(r, getNodeValueByName(outputName));

    const std::string &docString = net.output(i).doc_string();

    Expected<std::string> saveName =
        getAttrFromDocString(saveNameSignifier, docString);

    const bool hasSpecifiedSaveName =
        !ERR_TO_BOOL(saveName.takeError(), /* log */ false);
    const std::string &saveNodeName =
        hasSpecifiedSaveName ? saveName.get() : outputName;

    std::pair<bool, std::string> trainableLayoutPair;
    ASSIGN_VALUE_OR_RETURN_ERR(
        trainableLayoutPair,
        getTrainableLayoutPairFromDocString(docString, useGlowCustomOps_));

    // If loadIntoExistingModule_ then it's reasonable for there to be a savePH
    // already. If not then there shouldn't be one.
    Placeholder *savePH = mod_.getPlaceholderByNameSlow(outputName);
    if (!savePH) {
      savePH = mod_.createPlaceholder(r.getType(), outputName,
                                      trainableLayoutPair.first,
                                      trainableLayoutPair.second);
    } else {
      RETURN_ERR_IF_NOT(loadIntoExistingModule_,
                        "Found pre-existing PH by name " + outputName);
      RETURN_IF_ERR(verifyPreexistingStorage(savePH, outputName, *r.getType(),
                                             trainableLayoutPair.second,
                                             trainableLayoutPair.first));
    }
    G_->createSave(saveNodeName, r, savePH, hasSpecifiedSaveName);

    auto loaderNameOrErr = getAttrFromDocString(loaderNameSignifier, docString);
    const std::string &loaderName =
        !ERR_TO_BOOL(loaderNameOrErr.takeError(), /* log */ false)
            ? loaderNameOrErr.get()
            : outputName;
    RETURN_ERR_IF_NOT(outputVarsByName_.try_emplace(loaderName, savePH).second,
                      "Already had output placeholder by name " + loaderName);
  }

  return Error::success();
}

Error ONNXModelLoader::loadNetwork(ONNX_NAMESPACE::GraphProto &net,
                                   bool loadingConstFoldSubgraph) {
  /// Load the network operators:
  for (int i = 0; i < net.node_size(); i++) {
    auto &op = net.node(i);

    // Always ignore these since they're dummy nodes used to just carry meta
    // info that is processed via setupOrigStaticTypeMap().
    if (op.op_type() == staticPHDummyNodeName) {
      continue;
    }

    // Set up current partition to load into if relevant.
    if (partNameToFun_.size() && !loadingConstFoldSubgraph &&
        op.op_type() != constFoldSubgraphNodeName) {
      const ONNX_NAMESPACE::AttributeProto *pNameAttr = nullptr;
      for (auto &arg : op.attribute()) {
        if (arg.name() == "partitionName") {
          pNameAttr = &arg;
          break;
        }
      }
      RETURN_ERR_IF_NOT(pNameAttr, "partitionName not found for " + op.name());
      std::string pName;
      ASSIGN_VALUE_OR_RETURN_ERR(pName, loadStr(pNameAttr));
      auto it = partNameToFun_.find(pName);
      RETURN_ERR_IF_NOT(it != partNameToFun_.end(),
                        "Did not find partition with name " + pName);
      G_ = it->second;
    }
    RETURN_ERR_IF_NOT(G_, "Internal Glow error; Graph was not valid.");

    if (constFoldInLoader_) {
      auto tryFold = foldOperator(op);
      if (!tryFold) {
        // Error during constant folding; load the op normally below.
        const std::string errStr =
            ERR_TO_STRING(tryFold.takeError(), /* warning */ true);
        VLOG(1) << "Issue while trying to ConstantFold " << loadOperatorName(op)
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
    : CommonOperatorLoader({}, {}, &F, errPtr) {
  deleteUnusedConstants();
}

static Error checkStaticPH(const ONNX_NAMESPACE::ValueInfoProto &valueInfo,
                           std::unordered_set<std::string> &staticInputs,
                           bool useGlowCustomOps) {
  const std::string &inputName = valueInfo.name();
  if (useGlowCustomOps) {
    std::string isStatic;
    ASSIGN_VALUE_OR_RETURN_ERR(
        isStatic,
        getAttrFromDocString(staticSignifier, valueInfo.doc_string()));
    if (isStatic == "1") {
      staticInputs.emplace(inputName);
    }
  } else if (valueInfo.has_doc_string() &&
             valueInfo.doc_string() == staticSignifier) {
    staticInputs.emplace(inputName);
  }
  return Error::success();
}

Error ONNXModelLoader::collectStaticInputs(ONNX_NAMESPACE::GraphProto &net) {
  for (int i = 0; i < net.input_size(); i++) {
    RETURN_IF_ERR(
        checkStaticPH(net.input(i), staticInputs_, useGlowCustomOps_));
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

      RETURN_IF_ERR(checkStaticPH(valueInfo, staticInputs_, useGlowCustomOps_));
    }
  }
  return Error::success();
}

Error ONNXModelLoader::setupOrigStaticTypeMap(ONNX_NAMESPACE::GraphProto &net) {
  if (!staticPlaceholderTypes_) {
    return Error::success();
  }

  for (int i = 0; i < net.node_size(); i++) {
    auto &op = net.node(i);
    ArgumentDictionaryTy dict = loadArgumentMap(op);
    if (op.op_type() != staticPHDummyNodeName) {
      continue;
    }
    RETURN_ERR_IF_NOT(staticInputs_.count(op.name()),
                      "Expected static input for " + op.name());
    TypeRef OT;
    ASSIGN_VALUE_OR_RETURN_ERR(
        OT, loadTypeFromAttributes(Storage::OutputIdx, dict));
    staticPlaceholderTypes_->emplace(op.name(), *OT);
  }
  RETURN_ERR_IF_NOT(
      staticPlaceholderTypes_->size() == staticInputs_.size(),
      strFormat(
          "Expected to find types for all static Placeholders. %lu vs. %lu",
          staticPlaceholderTypes_->size(), staticInputs_.size()));
  return Error::success();
}

Error ONNXModelLoader::assignGraphInputs(const ONNX_NAMESPACE::GraphProto &net,
                                         llvm::ArrayRef<NodeValue> NVs) {
  RETURN_ERR_IF_NOT((dim_t)NVs.size() == (dim_t)net.input_size(),
                    "Input size mismatch.");
  for (size_t i = 0; i < NVs.size(); i++) {
    nodeValueByName_[net.input(i).name()] = NVs[i];
  }
  return Error::success();
}

Error ONNXModelLoader::loadModel(ONNX_NAMESPACE::ModelProto &modelDef,
                                 llvm::ArrayRef<const char *> tensorNames,
                                 llvm::ArrayRef<TypeRef> types,
                                 const Backend *B,
                                 bool loadInputsAsPlaceholdersForOnnx) {
  useGlowCustomOps_ = modelDef.producer_name() == "GlowONNXModelWriter";

  RETURN_IF_ERR(setVersion(modelDef));

  ONNX_NAMESPACE::GraphProto graphDef = modelDef.graph();
  RETURN_IF_ERR(checkInputs(graphDef, tensorNames, types));
  RETURN_IF_ERR(collectStaticInputs(graphDef));
  RETURN_IF_ERR(setupOrigStaticTypeMap(graphDef));

  RETURN_IF_ERR(loadInitializers(graphDef));

  if (tensorNames.empty() && types.empty()) {
    // Detect inputs without initializers and create placeholders.
    RETURN_IF_ERR(loadInputs(graphDef, loadInputsAsPlaceholdersForOnnx));
  }

  RETURN_IF_ERR(loadNetwork(graphDef, /* loadingConstFoldSubgraph */ false));

  RETURN_IF_ERR(setOutputNodes(graphDef));

  RETURN_ERR_IF_NOT(G_->verify(B), "Function verification failed.");

  deleteUnusedConstants();

  deleteConstFoldFunctions();

  RETURN_IF_ERR(verifyDummyQParams());

  return Error::success();
}

ONNXModelLoader::ONNXModelLoader(const std::string &modelDescFilename,
                                 llvm::ArrayRef<const char *> tensorNames,
                                 llvm::ArrayRef<TypeRef> types, Function &F,
                                 Error *errPtr, bool zipMode,
                                 BackendSpecificNodeInfo *perNodeOpts,
                                 bool disableConstFoldInLoader,
                                 bool loadIntoExistingModule, const Backend *B,
                                 const std::string *inputStringPtr)
    : CommonOperatorLoader(tensorNames, types, &F, errPtr,
                           loadIntoExistingModule),
      perNodeOpts_(perNodeOpts), staticPlaceholderTypes_(nullptr) {
  // if errPtr already contains an error then don't continue with constructor
  if (errPtr && *errPtr) {
    return;
  }

  if (disableConstFoldInLoader) {
    constFoldInLoader_ = false;
  }

  auto setup = [&]() -> Error {
    ONNX_NAMESPACE::ModelProto modelDef;
    ASSIGN_VALUE_OR_RETURN_ERR(
        modelDef, loadProto(modelDescFilename, zipMode, inputStringPtr));
    RETURN_IF_ERR(loadModel(modelDef, tensorNames, types, B,
                            /* loadInputsAsPlaceholdersForOnnx */ true));
    return Error::success();
  };

  if (errPtr) {
    *errPtr = setup();
  } else {
    EXIT_ON_ERR(setup());
  }
}

/// \returns a metadata prop found at \p key in \p modelDef.
static const char *getMetadataProp(const ONNX_NAMESPACE::ModelProto &modelDef,
                                   llvm::StringRef key) {
  for (const auto &keyVal : modelDef.metadata_props()) {
    if (keyVal.key() == key) {
      return keyVal.value().data();
    }
  }
  return nullptr;
}

static Expected<int32_t>
getIntMetadataProp(const ONNX_NAMESPACE::ModelProto &modelDef,
                   llvm::StringRef key) {
  const char *intStr = getMetadataProp(modelDef, key);
  RETURN_ERR_IF_NOT(intStr, "Did not find value for " + std::string(key));
  int32_t intVal;
  ASSIGN_VALUE_OR_RETURN_ERR(intVal, getIntFromStr(intStr));
  return intVal;
}

Error ONNXModelLoader::setupPartitions(ONNX_NAMESPACE::ModelProto &modelDef,
                                       PrePartitionedConfig &PPC,
                                       llvm::StringRef rootName,
                                       int numPartitions) {
  PPC.funcName = rootName.str();
  PPC.resizeAndReserve(numPartitions);

  for (int i = 0; i < numPartitions; i++) {
    const std::string partIdPrefix = getPartitionIdPrefix(i);
    const char *pName = getMetadataProp(modelDef, partIdPrefix + nameSignifier);
    RETURN_ERR_IF_NOT(pName, "Didn't find expected partition name");

    // Load the partition name and create a Function with the same name.
    Function *PF = nullptr;
    if (loadIntoExistingModule_ && mod_.hasFunction(pName)) {
      PF = mod_.getFunction(pName);
      RETURN_ERR_IF_NOT(PF->getNodes().size() == 0,
                        "Function must be empty to load into.");
    } else {
      PF = mod_.createFunction(pName);
    }
    partNameToFun_[pName] = PF;
    PPC.funcs.push_back(PF);

    // Load all logical devices for the partition.
    int32_t numLogicalDevices;
    ASSIGN_VALUE_OR_RETURN_ERR(
        numLogicalDevices,
        getIntMetadataProp(modelDef,
                           partIdPrefix + numLogicalDevicesSignifier));
    for (int j = 0; j < numLogicalDevices; j++) {
      DeviceIDTy ID;
      ASSIGN_VALUE_OR_RETURN_ERR(
          ID, getIntMetadataProp(modelDef,
                                 partIdPrefix + getLogicalDeviceSignfier(j)));
      PPC.logicalIDs[i].push_back(ID);
    }

    // Get backend name.
    const char *backendName =
        getMetadataProp(modelDef, partIdPrefix + backendNameSignifier);
    RETURN_ERR_IF_NOT(backendName, "Didn't find Backend name");
    PPC.backendNames.emplace_back(backendName);

    // Get backendHints.executionUnits. Note that we don't support serializing
    // SRAMPrioritization, so it's left empty.
    unsigned execUnits;
    ASSIGN_VALUE_OR_RETURN_ERR(
        execUnits,
        getIntMetadataProp(modelDef, partIdPrefix + executionUnitsSignifier));
    PPC.backendHints.push_back({execUnits, /* SRAMPrioritization */ {}});

    // Load all backend-specific options.
    int32_t numBackendSpecificOpts;
    ASSIGN_VALUE_OR_RETURN_ERR(
        numBackendSpecificOpts,
        getIntMetadataProp(modelDef,
                           partIdPrefix + numBackendSpecificOptsSignifier));
    for (int j = 0; j < numBackendSpecificOpts; j++) {
      const char *optKey = getMetadataProp(
          modelDef, partIdPrefix + getBackendSpecificOptKeySignifier(j));
      RETURN_ERR_IF_NOT(optKey,
                        "Didn't find expected backend-specific option key");
      const char *optVal = getMetadataProp(
          modelDef, partIdPrefix + getBackendSpecificOptValSignifier(j));
      RETURN_ERR_IF_NOT(optVal,
                        "Didn't find expected backend-specific option val");
      PPC.backendSpecificOpts[i][optKey] = optVal;
    }

    // Get replicationCount.
    int32_t replicationCount;
    ASSIGN_VALUE_OR_RETURN_ERR(
        replicationCount,
        getIntMetadataProp(modelDef, partIdPrefix + replicationCountSignifier));
    PPC.replicationCounts.push_back(replicationCount);
  }

  return Error::success();
}

void ONNXModelLoader::setupPositionalIO(
    const ONNX_NAMESPACE::GraphProto &graph) {
  for (const auto &in : graph.input()) {
    if (staticInputs_.count(in.name())) {
      continue;
    }
    auto loaderNameOrErr =
        getAttrFromDocString(loaderNameSignifier, in.doc_string());
    if (ERR_TO_BOOL(loaderNameOrErr.takeError(), /* log */ false)) {
      positionalInputNames_.clear();
      break;
    }
    positionalInputNames_.emplace_back(loaderNameOrErr.get());
  }

  for (const auto &out : graph.output()) {
    auto loaderNameOrErr =
        getAttrFromDocString(loaderNameSignifier, out.doc_string());
    if (ERR_TO_BOOL(loaderNameOrErr.takeError(), /* log */ false)) {
      positionalOutputNames_.clear();
      break;
    }
    positionalOutputNames_.emplace_back(loaderNameOrErr.get());
  }
}

Error ONNXModelLoader::setupUpdatedTQPMap(
    ONNX_NAMESPACE::ModelProto &modelDef, uint32_t weightsCount,
    const onnxTensorDescriptorV1 *weightDescriptors) {
  // Check if we have the two strings in metadata props we need to do TQP
  // updating, and if not print a warning/return early.
  const char *originNameToUniqueOffsetMappingStr =
      getMetadataProp(modelDef, originNameToUniqueOffsetMappingSignifier);
  if (!originNameToUniqueOffsetMappingStr) {
    LOG(WARNING) << "Did not find \""
                 << originNameToUniqueOffsetMappingSignifier
                 << "\" in ONNX model, skipping setting updated TQP map.";
    return Error::success();
  }

  const char *qParamC2ProtoStr = getMetadataProp(modelDef, "C2_with_q_params");
  if (!qParamC2ProtoStr) {
    LOG(WARNING) << "Did not find \"C2_with_q_params\" in ONNX model, skipping "
                    "setting updated TQP map.";
    return Error::success();
  }

  // Now load the qParamC2ProtoStr into a temporary dummy Caffe2ModelLoader,
  // which fills in the originNameToTQPMap based on that model and the weights.
  OriginNameToTQPMap originNameToTQPMap;
  Error err(Error::success());
  Module dummyMod;
  Caffe2ModelLoader tmpLoader(qParamC2ProtoStr, weightsCount, weightDescriptors,
                              dummyMod, &err, &originNameToTQPMap,
                              clipQuantRangeToFP16_);
  RETURN_IF_ERR(err);

  // Now parse the originNameToUniqueOffsetMappingStr to find the original C2
  // name : unique offset pairs. These are formatted like
  // "name_op_a@0@@name_op_b@1@@", with @@ separating each pair, and @
  // separating name from unique offset.
  llvm::SmallVector<llvm::StringRef, 128> nameOffsetSplits;
  llvm::StringRef strRef = llvm::StringRef(originNameToUniqueOffsetMappingStr);
  strRef.split(nameOffsetSplits, offsetEndSig, /* MaxSplit */ -1,
               /* KeepEmpty */ false);

  // Store the mapping into updatedTQPs_, where each unique offset is used as
  // the index into updatedTQPs_ to the actual TQP to use. I.e. we essentially
  // already have c2_name -> offset, and c2_name -> TQP, so we change this to
  // offset -> TQP.
  updatedTQPs_.resize(nameOffsetSplits.size());
  for (auto &nameOffsetSplit : nameOffsetSplits) {
    auto nameOffsetPair = nameOffsetSplit.split(offsetSepSig);
    int32_t idx;
    ASSIGN_VALUE_OR_RETURN_ERR(idx, getIntFromStr(nameOffsetPair.second));
    RETURN_ERR_IF_NOT(idx < int32_t(updatedTQPs_.size()),
                      strFormat("Provided offset index %d not inside size "
                                "of updatedTQPs_ %lu",
                                idx, updatedTQPs_.size()));

    auto it = originNameToTQPMap.find(nameOffsetPair.first.str());
    RETURN_ERR_IF_NOT(it != originNameToTQPMap.end(),
                      strFormat("Did not find matching TQP for %s",
                                nameOffsetPair.first.str().data()));
    updatedTQPs_[idx] = it->second;
  }
  return Error::success();
}

ONNXModelLoader::ONNXModelLoader(
    const std::string &modelDescFilename,
    llvm::ArrayRef<const char *> tensorNames, llvm::ArrayRef<TypeRef> types,
    Module &mod, llvm::StringRef funName, PrePartitionedConfig *PPC,
    Error *errPtr, bool zipMode, BackendSpecificNodeInfo *perNodeOpts,
    bool loadIntoExistingModule, bool disableConstFoldInLoader,
    const Backend *B, const std::string *inputStringPtr)
    : CommonOperatorLoader(tensorNames, types, mod, errPtr,
                           loadIntoExistingModule),
      perNodeOpts_(perNodeOpts), staticPlaceholderTypes_(nullptr) {
  // if errPtr already contains an error then don't continue with constructor
  if (errPtr && *errPtr) {
    return;
  }

  if (disableConstFoldInLoader) {
    constFoldInLoader_ = false;
  }

  auto setup = [&]() -> Error {
    ONNX_NAMESPACE::ModelProto modelDef;
    ASSIGN_VALUE_OR_RETURN_ERR(
        modelDef, loadProto(modelDescFilename, zipMode, inputStringPtr));

    auto numPartitionsOrErr = getIntMetadataProp(modelDef, "numPartitions");
    if (!numPartitionsOrErr) {
      ERR_TO_VOID(numPartitionsOrErr.takeError(), /*log*/ false);
      G_ = mod_.createFunction(funName);
    } else {
      RETURN_ERR_IF_NOT(PPC, "No PrePartitionConfig to load partitions into");
      RETURN_IF_ERR(
          setupPartitions(modelDef, *PPC, funName, *numPartitionsOrErr));
    }

    RETURN_IF_ERR(loadModel(modelDef, tensorNames, types, B,
                            /* loadInputsAsPlaceholdersForOnnx */ true));

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
    bool loadInputsAsPlaceholdersForOnnx, Error *errPtr, bool constFoldInLoader,
    BackendSpecificNodeInfo *perNodeOpts)
    : CommonOperatorLoader({}, {}, &F, errPtr, true), perNodeOpts_(perNodeOpts),
      staticPlaceholderTypes_(nullptr) {
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

    RETURN_IF_ERR(loadWeights(weightsCount, weightDescriptors));

    RETURN_IF_ERR(loadModel(modelDef, {}, {}, /* B */ nullptr,
                            loadInputsAsPlaceholdersForOnnx));

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
    const onnxTensorDescriptorV1 *weightDescriptors, Module &mod,
    llvm::StringRef funName, PrePartitionedConfig *PPC,
    bool loadInputsAsPlaceholdersForOnnx, Error *errPtr, bool constFoldInLoader,
    BackendSpecificNodeInfo *perNodeOpts,
    std::map<std::string, Type> *staticPlaceholderTypes, bool replaceDummyTQPs,
    bool clipQuantRangeToFP16)
    : CommonOperatorLoader({}, {}, mod, errPtr,
                           /* loadIntoExistingModule */ true,
                           /* originNameToTQPMap */ nullptr,
                           /* loadUniquedDummyQParams */ false,
                           replaceDummyTQPs, /* zeroScaleFP16Clip */ false,
                           clipQuantRangeToFP16),
      perNodeOpts_(perNodeOpts),
      staticPlaceholderTypes_(staticPlaceholderTypes) {
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

    // If we're going to be replacing dummy TQPs then setup the updated TQP map,
    // which is used later on when loading each op.
    if (replaceDummyTQPs_) {
      // Check if the model has specified that clipQuantRangeToFP16 should be
      // overridden via the clipQuantRangeToFP16Key metadata prop. Do this
      // before updating TQP map, because this will affect the qparams loaded.
      for (const auto &keyVal : modelDef.metadata_props()) {
        if (keyVal.key() == clipQuantRangeToFP16Key && keyVal.value() == "1") {
          LOG(INFO) << "ONNXModelLoader found enabled "
                    << clipQuantRangeToFP16Key;
          clipQuantRangeToFP16_ = true;
          break;
        }
      }

      RETURN_IF_ERR(
          setupUpdatedTQPMap(modelDef, weightsCount, weightDescriptors));
    }

    RETURN_IF_ERR(loadWeights(weightsCount, weightDescriptors));

    auto numPartitionsOrErr = getIntMetadataProp(modelDef, "numPartitions");
    if (!numPartitionsOrErr) {
      ERR_TO_VOID(numPartitionsOrErr.takeError(), /*log*/ false);
      G_ = mod_.createFunction(funName);
    } else {
      RETURN_ERR_IF_NOT(PPC, "No PrePartitionConfig to load partitions into");
      RETURN_IF_ERR(
          setupPartitions(modelDef, *PPC, funName, *numPartitionsOrErr));
    }

    RETURN_IF_ERR(loadModel(modelDef, {}, {}, /* B */ nullptr,
                            loadInputsAsPlaceholdersForOnnx));

    if (loadInputsAsPlaceholdersForOnnx) {
      setupPositionalIO(modelDef.graph());
    }

    return Error::success();
  };

  if (errPtr) {
    *errPtr = setup();
  } else {
    EXIT_ON_ERR(setup());
  }
}
