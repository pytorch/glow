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

#include "glow/Graph/CustomOpData.h"
#include "glow/Base/Traits.h"
#include "glow/Base/Type.h"
#include "glow/Graph/CustomOpUtils.h"

using namespace glow;

namespace glow {
/// Check if the CustomOpNode has a parameter with name \p name.
bool CustomOpData::hasParam(std::string name) {
  bool exists = false;
  exists |= floatParams_.find(name) != floatParams_.end();
  exists |= intParams_.find(name) != intParams_.end();
  exists |= floatVecParams_.find(name) != floatVecParams_.end();
  exists |= intVecParams_.find(name) != intVecParams_.end();
  exists |= stringParams_.find(name) != stringParams_.end();
  return exists;
}

// Check if the parameter is registered.
bool CustomOpData::hasParamInfo(std::string &name, CustomOpDataType type,
                                bool isScalar) {
  bool registered = false;

  if (mapNameToInfo_.find(name) == mapNameToInfo_.end()) {
    LOG(ERROR) << "Cannot add unregistered parameter: " << name << "\n";
    return registered;
  }

  const ParamInfo *pInfo = mapNameToInfo_.at(name);
  if (type != pInfo->getDataType() || isScalar != pInfo->isScalar()) {
    LOG(ERROR) << "Registered parameter type is not " << toString(type)
               << (isScalar ? "" : " vector") << " for " << name
               << ". Cannnot add it. \n";
    return registered;
  }

  registered = true;
  return registered;
}

#define ADD_PARAM_TO(MAP, TYPE, CUSTOM_OP_DTYPE, IS_SCALAR)                    \
  void CustomOpData::addParam(std::string name, TYPE data) {                   \
    if (!hasParamInfo(name, CUSTOM_OP_DTYPE, IS_SCALAR)) {                     \
      return;                                                                  \
    }                                                                          \
    assert(MAP.find(name) == MAP.end() &&                                      \
           "Parameter with this name already exists!\n");                      \
    MAP[name] = data;                                                          \
  }

/// Add parameter value \p data with name \p name to the CustomOpData class.
ADD_PARAM_TO(floatParams_, float, CustomOpDataType::DTFloat32, true)
ADD_PARAM_TO(intParams_, int, CustomOpDataType::DTIInt32, true)
ADD_PARAM_TO(stringParams_, std::string, CustomOpDataType::DTString, true)
ADD_PARAM_TO(floatVecParams_, std::vector<float>, CustomOpDataType::DTFloat32,
             false)
ADD_PARAM_TO(intVecParams_, std::vector<int32_t>, CustomOpDataType::DTIInt32,
             false)

#define GET_PARAM_FROM(MAP, FUNC_NAME, TYPE, STR)                              \
  TYPE CustomOpData::FUNC_NAME(std::string PARAM_NAME) {                       \
    assert(MAP.find(PARAM_NAME) != MAP.end() && STR);                          \
    return MAP[PARAM_NAME];                                                    \
  }

/// Returns parameter with name \p name.
GET_PARAM_FROM(
    floatParams_, getFloatParam, float,
    "Float type parameter not found in CustomOpData for this node.!\n")
GET_PARAM_FROM(
    floatVecParams_, getFloatVecParam, std::vector<float>,
    "Float vector type parameter not found in CustomOpData for this node.!\n")
GET_PARAM_FROM(intParams_, getIntParam, int,
               "Int type parameter not found in CustomOpData for this node.!\n")
GET_PARAM_FROM(
    intVecParams_, getIntVecParam, std::vector<int>,
    "Int vector type parameter not found in CustomOpData for this node.!\n")
GET_PARAM_FROM(
    stringParams_, getStringParam, std::string,
    "String type parameter not found in CustomOpData for this node.!\n")

/// Utility functions.
const llvm::StringRef CustomOpData::getName() const { return opName_; }
const llvm::StringRef CustomOpData::getTypeName() const { return opTypeName_; }
const llvm::StringRef CustomOpData::getPackageName() const {
  return opPackageName_;
}
const std::vector<std::string> &CustomOpData::getParamNames() const {
  return paramNames_;
}
const std::unordered_map<std::string, const ParamInfo *> &
CustomOpData::getNameToInfoMap() const {
  return mapNameToInfo_;
}
const size_t CustomOpData::getNumParameters() const {
  return mapNameToInfo_.size();
}
// Verification function.
void CustomOpData::setVerificationFunction(customOpVerify_t verifyFunction) {
  verifyFunction_ = verifyFunction;
}

customOpVerify_t CustomOpData::getVerificationFunction() const {
  return verifyFunction_;
}

llvm::hash_code CustomOpData::getHash() const {
  return llvm::hash_combine(getName(), getTypeName(), getPackageName());
}

} // namespace glow
