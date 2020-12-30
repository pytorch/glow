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

#include "glow/Graph/OperationInfo.h"

#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"

// Operation Descriptor.
// Helper for parsing Op config given as yaml.
struct glow::OpDesc {
  std::string typeName_;
  std::string packageName_;
  std::vector<ParamDesc> parameters_;
  std::vector<NodeIODesc> inputs_;
  std::vector<NodeIODesc> outputs_;
  std::vector<ImplDesc> implementations_;
  std::string pathToFunctionLibrary_;
};

// Parser utilities for OperationInfo.
// Yaml config file is parsed and read into "Desc" (ParamDesc, ImplDesc, etc.)
// structs. Info objects (ParamInfo, ImplInfo, etc.) are then created from these
// Desc objects.
namespace llvm {
namespace yaml {

template <> struct ScalarEnumerationTraits<CustomOpDataType> {
  static void enumeration(IO &io, CustomOpDataType &dtype) {
    io.enumCase(dtype, "float", CustomOpDataType::DTFloat32);
    io.enumCase(dtype, "int", CustomOpDataType::DTIInt32);
    io.enumCase(dtype, "bool", CustomOpDataType::DTBool);
    io.enumCase(dtype, "string", CustomOpDataType::DTString);
  }
};

template <> struct MappingTraits<glow::ParamDesc> {
  static void mapping(IO &io, glow::ParamDesc &data) {
    io.mapRequired("name", data.name_);
    io.mapRequired("dataType", data.dataType_);
    io.mapOptional("scalar", data.isScalar_, false);
    io.mapOptional("size", data.size_);
  }

  static StringRef validate(IO &io, glow::ParamDesc &data) {
    if (data.isScalar_ && data.size_ != 0) {
      return StringRef("Found non-zero size for scalar Parameter " +
                       data.name_);
    }
    return StringRef();
  }
};

template <> struct MappingTraits<glow::NodeIODesc> {
  static void mapping(IO &io, glow::NodeIODesc &data) {
    io.mapRequired("name", data.name_);
    io.mapRequired("maxDims", data.maxDims_);
    io.mapOptional("constant", data.isConstant_, false);
  }
};

template <> struct MappingTraits<glow::ImplDesc> {
  static void mapping(IO &io, glow::ImplDesc &data) {
    io.mapRequired("backend", data.backendName_);
    io.mapRequired("type", data.type_);
    io.mapOptional("config", data.config_);
    io.mapRequired("impl", data.path_);
  }

  static StringRef validate(IO &io, glow::ImplDesc &data) {
    if (data.path_.empty()) {
      return StringRef("Implementation not provided for backend " +
                       data.backendName_ + " and imp type " + data.type_);
    }
    return StringRef();
  }
};

template <> struct MappingTraits<glow::OpDesc> {
  static void mapping(IO &io, glow::OpDesc &data) {
    io.mapRequired("type", data.typeName_);
    io.mapRequired("package", data.packageName_);
    io.mapRequired("parameters", data.parameters_);
    io.mapRequired("inputs", data.inputs_);
    io.mapRequired("outputs", data.outputs_);
    io.mapRequired("implementations", data.implementations_);
    io.mapRequired("functionsLibrary", data.pathToFunctionLibrary_);
  }
};

} // end namespace yaml
} // end namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(glow::ParamDesc);
LLVM_YAML_IS_SEQUENCE_VECTOR(glow::NodeIODesc);
LLVM_YAML_IS_SEQUENCE_VECTOR(glow::ImplDesc);

namespace glow {

ImplementationInfo::ImplementationInfo(const ImplementationInfo &data) {
  backendName_ = data.backendName_;
  type_ = data.type_;
  config_ = data.config_;
  path_ = data.path_;
  if (!path_.empty()) {
    impl_ = (void *)(&path_);
  } else {
    impl_ = data.impl_;
  }
}

OperationInfo::OperationInfo(const OpDesc &data) {
  typeName_ = data.typeName_;
  packageName_ = data.packageName_;
  parameters_.insert(parameters_.begin(), data.parameters_.begin(),
                     data.parameters_.end());
  inputs_.insert(inputs_.begin(), data.inputs_.begin(), data.inputs_.end());
  outputs_.insert(outputs_.begin(), data.outputs_.begin(), data.outputs_.end());
  implementations_.insert(implementations_.begin(),
                          data.implementations_.begin(),
                          data.implementations_.end());
  pathToFunctionLibrary_ = data.pathToFunctionLibrary_;
}

std::vector<std::string> OperationInfo::getParamNames() const {
  std::vector<std::string> parameterNames;
  for (auto p : parameters_) {
    parameterNames.push_back(p.getName());
  }
  return parameterNames;
}

static Error deserializeOpInfoFromYaml(llvm::StringRef fileName,
                                       std::vector<OpDesc> &opDescs) {
  // Open YAML input stream.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> text =
      llvm::MemoryBuffer::getFileAsStream(fileName);
  RETURN_ERR_IF_NOT(!text.getError(),
                    "Unable to open file with name: " + fileName.str());
  std::unique_ptr<llvm::MemoryBuffer> buffer = std::move(*text);
  llvm::yaml::Input yin(buffer->getBuffer());

  // Read Documents into OpDesc.
  do {
    OpDesc desc;
    yin >> desc;
    RETURN_ERR_IF_NOT(!yin.error(),
                      "Error reading yaml document " + fileName.str());
    opDescs.emplace_back(desc);
  } while (yin.nextDocument());

  return Error::success();
}

// Parses \p fileName to read OperationInfo.
// Adds OperationInfo into \p opinfos.
Error deserializeOpInfoFromYaml(llvm::StringRef fileName,
                                std::vector<OperationInfo> &opinfos) {
  std::vector<OpDesc> ops;
  RETURN_IF_ERR(deserializeOpInfoFromYaml(fileName, ops));
  // Get OpInfo from OpDesc.
  for (const auto &op : ops) {
    OperationInfo opInfo(op);
    opinfos.emplace_back(opInfo);
  }

  return Error::success();
}

} // namespace glow
