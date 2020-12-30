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

#ifndef GLOW_GRAPH_OPERATIONINFO_H
#define GLOW_GRAPH_OPERATIONINFO_H

// std lib headers
#include <string>
#include <vector>

// llvm headers
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

// glow headers
#include "glow/Base/Traits.h"
#include "glow/Base/Type.h"
#include "glow/CustomOp/CustomOpTypes.h"
#include "glow/Support/Error.h"

namespace glow {

// Param Descriptior.
struct ParamDesc {
  std::string name_{};
  CustomOpDataType dataType_;
  bool isScalar_{}; // Scalar or Vector
  size_t size_{};   // size of the Vector, size should be 0 for scalar
};

/// Holds Information on the Parameters/Attributes of an Operation.
class ParamInfo : private ParamDesc {
public:
  ParamInfo() = delete;
  ParamInfo(llvm::StringRef name, CustomOpDataType dataType,
            bool isScalar = false, size_t size = 0)
      : ParamDesc{name, dataType, isScalar, size} {}

  ParamInfo(const ParamDesc &pdata) : ParamDesc(pdata) {}

  bool isScalar() const { return isScalar_; }
  bool isArray() const { return !isScalar_; }
  size_t getSize() const { return size_; }
  CustomOpDataType getDataType() const { return dataType_; }
  std::string getName() const { return name_; }
};

// NodeIO Descriptior.
struct NodeIODesc {
  std::string name_;
  size_t maxDims_; // specify the max number of dimensions for this input/output
  bool isConstant_; // if the input is constant, not applicable for outputs
};

// Holds Information on the inputs/outputs of the Operation.
class NodeIOInfo : private NodeIODesc {
public:
  NodeIOInfo() = delete;
  NodeIOInfo(llvm::StringRef name, size_t maxDims, bool isConstant = false)
      : NodeIODesc{name, maxDims, isConstant} {}
  NodeIOInfo(const NodeIODesc &data) : NodeIODesc(data) {}
  bool isConstant() const { return isConstant_; }
  size_t getMaxDims() const { return maxDims_; }
  std::string getName() const { return name_; }
};

// Implementation Desc.
struct ImplDesc {
  /// Name of the backend.
  std::string backendName_{};
  /// Type of implementation.
  std::string type_{};
  /// Config. TODO: Change to vector.
  std::string config_{};
  /// Implementation.
  std::string path_{};
};

// Holds Information on Implementation of the Operation.
class ImplementationInfo : private ImplDesc {
  // Implementation.
  void *impl_{};

public:
  ImplementationInfo() = delete;

  ImplementationInfo(llvm::StringRef backendName, llvm::StringRef id,
                     void *impl, const std::string &config = std::string())
      : ImplDesc{backendName, id, config}, impl_{impl} {}

  ImplementationInfo(const ImplDesc &data)
      : ImplDesc(data), impl_{(void *)(&path_)} {}

  ImplementationInfo(const ImplementationInfo &data);

  std::string getBackendName() const { return backendName_; }

  std::string getType() const { return type_; }

  std::string getConfig() const { return config_; }

  void *getImplementation() const { return impl_; }
};

// Operation Descriptor.
// Helper for parsing Op config given as yaml.
struct OpDesc;

class OperationInfo {
  /// Name of type of Op, e.g. "ConvND".
  std::string typeName_;

  /// Package Name of the Op, e.g. "Research".
  std::string packageName_;

  /// Parameter Information of the Op.
  std::vector<ParamInfo> parameters_;

  /// Ordered Input Information of the Op.
  std::vector<NodeIOInfo> inputs_;

  /// Ordered Output Information of the Op.
  std::vector<NodeIOInfo> outputs_;

  // ImplementationInfo for all backends and data types.
  std::vector<ImplementationInfo> implementations_;

  // Path on disk to the shared library with functions for the operation.
  std::string pathToFunctionLibrary_;

public:
  OperationInfo() = delete;

  // Inputs and Outputs Information must be ordered.
  OperationInfo(llvm::StringRef typeName, llvm::StringRef packageName,
                llvm::ArrayRef<ParamInfo> params,
                llvm::ArrayRef<NodeIOInfo> inputs,
                llvm::ArrayRef<NodeIOInfo> outputs,
                llvm::ArrayRef<ImplementationInfo> implementations,
                llvm::StringRef pathToFunctionLibrary)
      : typeName_{typeName}, packageName_{packageName}, parameters_{params},
        inputs_{inputs}, outputs_{outputs}, implementations_{implementations},
        pathToFunctionLibrary_{pathToFunctionLibrary} {}

  OperationInfo(const OpDesc &data);

  llvm::StringRef getTypeName() const { return typeName_; }

  llvm::StringRef getPackageName() const { return packageName_; }

  llvm::ArrayRef<ParamInfo> getParamInfo() const { return parameters_; }

  // Returns vector of parameters names in the order that was registered.
  std::vector<std::string> getParamNames() const;

  // Input and output information is ordered.
  llvm::ArrayRef<NodeIOInfo> getInputInfo() const { return inputs_; }

  llvm::ArrayRef<NodeIOInfo> getOutputInfo() const { return outputs_; }

  llvm::ArrayRef<ImplementationInfo> getImplementations() const {
    return implementations_;
  }
  llvm::StringRef getFunctionLibraryPath() const {
    return pathToFunctionLibrary_;
  }
};

// Parses \p fileName to read OperationInfo.
// Adds OperationInfo into \p opinfos.
Error deserializeOpInfoFromYaml(llvm::StringRef fileName,
                                std::vector<OperationInfo> &opinfos);

} // namespace glow

#endif // GLOW_GRAPH_OPERATIONINFO_H