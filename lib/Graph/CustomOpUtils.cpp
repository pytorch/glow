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

#include "glow/Graph/CustomOpUtils.h"

namespace glow {

// Maps CustomOpDataType to glow ElemKind.
static ElemKind customOpDataTypeToElemKind(CustomOpDataType dtype) {
  ElemKind elemK;
  switch (dtype) {
  case DTFloat32:
    elemK = ElemKind::FloatTy;
    break;
  case DTFloat16:
    elemK = ElemKind::Float16Ty;
    break;
  case DTQInt8:
    elemK = ElemKind::Int8QTy;
    break;
  case DTQUInt8:
    elemK = ElemKind::UInt8QTy;
    break;
  case DTIInt32:
    elemK = ElemKind::Int32ITy;
    break;
  case DTIInt64:
    elemK = ElemKind::Int64ITy;
    break;
  default:
    llvm_unreachable("Type not supported yet");
  }
  return elemK;
}

CustomOpIOTensor glowTypeToCustomOpIOTensor(const TypeRef type) {
  CustomOpIOTensor iot;

  // Copy dims.
  auto dims = type->dims();
  iot.rank = dims.size();
  iot.dims = (int32_t *)malloc(iot.rank * sizeof(int32_t));

  for (int i = 0; i < iot.rank; i++)
    iot.dims[i] = dims[i];

  switch (type->getElementType()) {
  case ElemKind::FloatTy:
    iot.dtype = CustomOpDataType::DTFloat32;
    break;
  case ElemKind::Float16Ty:
    iot.dtype = CustomOpDataType::DTFloat16;
    break;
  case ElemKind::Int8QTy:
    iot.dtype = DTQInt8;
    break;
  case ElemKind::UInt8QTy:
    iot.dtype = DTQUInt8;
    break;
  case ElemKind::Int32ITy:
    iot.dtype = CustomOpDataType::DTIInt32;
    break;
  case ElemKind::Int64ITy:
    iot.dtype = CustomOpDataType::DTIInt64;
    break;
  default:
    llvm_unreachable("Type not supported yet");
  }

  // Explicitly set to nullptr when only holding type information.
  iot.data = nullptr;

  return iot;
}

CustomOpIOTensor glowTensorToCustomOpIOTensor(Tensor *tensor) {
  CustomOpIOTensor iot = glowTypeToCustomOpIOTensor(&tensor->getType());
  iot.data = tensor->getUnsafePtr();
  return iot;
}

CustomOpIOTensor initializeCustomOpIOTensor(const int32_t &rank,
                                            const CustomOpDataType &dtype,
                                            void *data) {
  CustomOpIOTensor iot;
  iot.dtype = dtype;
  iot.rank = rank;
  iot.dims = (int32_t *)malloc(rank * sizeof(int32_t));
  for (int i = 0; i < rank; i++) {
    iot.dims[i] = 0;
  }
  iot.data = data;
  return iot;
}

// Returns Glow Type for a given \p CustomOpIOTensor.
Type customOpIOTensorToglowType(const CustomOpIOTensor &iotensor) {
  ElemKind elemTy = customOpDataTypeToElemKind(iotensor.dtype);
  std::vector<dim_t> dims(iotensor.rank);
  for (int i = 0; i < dims.size(); i++) {
    dims[i] = iotensor.dims[i];
  }
  Type type(elemTy, dims);
  return type;
}

void freeCustomOpIOTensor(CustomOpIOTensor &iot) {
  // No need to free iot.data, it is not handled by us.
  if (iot.dims)
    free(iot.dims);
  iot.dims = nullptr;
}

// Populates CustomOpParam struct with param data and returns a vector.
//
// Note: The returned vector should only be used for the duration of external
// API call. At the end, freeCustomOpParams() must be called to avoid memory
// leaks. Copying and storing the vector elsewhere may lead to double free
// related crashes.
std::vector<CustomOpParam> getCustomOpParams(CustomOpData &metadata) {
  std::vector<CustomOpParam> params;
  auto infoMap = metadata.getNameToInfoMap();

  for (const auto &name : metadata.getParamNames()) {
    auto info = infoMap[name];
    CustomOpParam param;
    param.name = name.c_str();
    param.dtype = info->getDataType();
    switch (param.dtype) {
    case CustomOpDataType::DTFloat32: {
      if (info->isScalar()) {
        param.size = 0;
        param.data = malloc(sizeof(float));
        *((float *)param.data) = metadata.getFloatParam(name);
      } else {
        std::vector<float> parr = metadata.getFloatVecParam(name);
        param.size = parr.size();
        param.data = malloc(parr.size() * sizeof(float));
        memcpy(param.data, parr.data(), parr.size() * sizeof(float));
      }
      break;
    }
    case CustomOpDataType::DTIInt32: {
      if (info->isScalar()) {
        param.size = 0;
        param.data = malloc(sizeof(int));
        *((int *)param.data) = metadata.getIntParam(name);
      } else {
        std::vector<int> parr = metadata.getIntVecParam(name);
        param.size = parr.size();
        param.data = malloc(parr.size() * sizeof(int));
        memcpy(param.data, parr.data(), parr.size() * sizeof(int));
      }
      break;
    }
    case CustomOpDataType::DTString: {
      if (info->isScalar()) {
        param.size = 0;
        std::string paramVal = metadata.getStringParam(name);
        int cpySize = sizeof(char) * (paramVal.size() + 1);
        param.data = malloc(cpySize);
        memcpy(param.data, paramVal.c_str(), cpySize);
      } else {
        llvm_unreachable("Param Type - string vector is not supported");
      }
      break;
    }
    default:
      llvm_unreachable("Param Type not supported");
      break;
    } // switch
    params.push_back(param);
  }
  return params;
}

// Calls free() on malloced param.data pointer.
void freeCustomOpParams(std::vector<CustomOpParam> &params) {
  for (auto param : params) {
    if (param.data)
      free(param.data);
  }
}

std::string toString(CustomOpDataType dtype) {
  std::string dtypeString{};
  switch (dtype) {
  case DTFloat32:
    dtypeString = "float";
    break;
  case DTFloat16:
    dtypeString = "float16";
    break;
  case DTQInt8:
    dtypeString = "i8";
    break;
  case DTQUInt8:
    dtypeString = "ui8";
    break;
  case DTIInt32:
    dtypeString = "int";
    break;
  case DTIInt64:
    dtypeString = "int64";
    break;
  case DTBool:
    dtypeString = "bool";
    break;
  case DTString:
    dtypeString = "string";
    break;
  default:
    llvm_unreachable("Type not supported yet");
  } // end switch

  return dtypeString;
}

} // end namespace glow
