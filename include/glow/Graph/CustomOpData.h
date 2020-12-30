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

#ifndef GLOW_GRAPH_CUSTOM_OP_DATA_H
#define GLOW_GRAPH_CUSTOM_OP_DATA_H

#include <string>
#include <unordered_map>

#include "glow/Graph/OpRepository.h"
#include "glow/Graph/OperationInfo.h"

#include "llvm/ADT/StringRef.h"

namespace glow {

/*--------------------------------------------------------------------------*/
//                    Parameter data
/*--------------------------------------------------------------------------*/

/// Data gathered from the model for custom ops must be populated in
/// CustomOpData class. This data is specific to the particular instance of a
/// node.
class CustomOpData final {
public:
  CustomOpData() {}
  CustomOpData(std::string opName, std::string opTypeName,
               std::string opPackageName,
               llvm::ArrayRef<ParamInfo> paramInfoVec) {
    opName_ = opName;
    opTypeName_ = opTypeName;
    opPackageName_ = opPackageName;
    for (int i = 0; i < paramInfoVec.size(); i++) {
      const ParamInfo *pInfo = &paramInfoVec[i];
      std::string name = pInfo->getName();
      paramNames_.push_back(name);
      mapNameToInfo_[name] = pInfo;
    }
  }

  /// Add parameter, it's data type and its value to the CustomOpData.
  void addParam(std::string name, float data);
  void addParam(std::string name, int data);
  void addParam(std::string name, std::string data);
  void addParam(std::string name, std::vector<float> data);
  void addParam(std::string name, std::vector<int> data);

  /// Return true if parameter information is registered in CustomOpData for a
  /// param with \p name and attributes \p type and \p isScalar.
  bool hasParamInfo(std::string &name, CustomOpDataType type, bool isScalar);

  // Get the parameter value from CustomOpData by using parameter name.
  float getFloatParam(std::string name);
  std::vector<float> getFloatVecParam(std::string name);
  int32_t getIntParam(std::string name);
  std::vector<int32_t> getIntVecParam(std::string name);
  std::string getStringParam(std::string name);

  // TODO figure out a way to do type check also.
  bool hasParam(std::string name);

  /// Utility functions.
  const llvm::StringRef getName() const;
  const llvm::StringRef getTypeName() const;
  const llvm::StringRef getPackageName() const;
  const std::unordered_map<std::string, const ParamInfo *> &
  getNameToInfoMap() const;
  const std::vector<std::string> &getParamNames() const;
  const size_t getNumParameters() const;

  // Verification functions.
  void setVerificationFunction(customOpVerify_t verifyFunction);
  customOpVerify_t getVerificationFunction() const;
  llvm::hash_code getHash() const;

private:
  /// This data is specific to a particular node in the graph.
  std::string opName_;
  std::string opTypeName_;
  std::string opPackageName_;

  /// Map of parameter name and corresponding data. Data can be of different
  /// types like float, int, vector<float> etc.. The type of data is specified
  /// in by the ParamInfo during operation registration. Depending on the type
  /// of data appropriate template is used to store and retrieve data. Parameter
  /// name is the name used in \p ParamInfo during registration.
  std::unordered_map<std::string, float> floatParams_;
  std::unordered_map<std::string, std::vector<float>> floatVecParams_;
  std::unordered_map<std::string, int> intParams_;
  std::unordered_map<std::string, std::vector<int>> intVecParams_;
  std::unordered_map<std::string, std::string> stringParams_;

  /// Vector of paramater names.
  std::vector<std::string> paramNames_;

  /// Map of parameter name and pointer to ParameterInfo present in
  /// OpRepository.
  std::unordered_map<std::string, const ParamInfo *> mapNameToInfo_;

  // Node verification function for the custom op.
  customOpVerify_t verifyFunction_;
};
} // namespace glow

#endif // GLOW_GRAPH_CUSTOM_OP_DATA_H
