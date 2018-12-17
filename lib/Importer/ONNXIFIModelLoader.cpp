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

#include "glow/Importer/ONNXIFIModelLoader.h"

#include "onnx/onnx_pb.h"

namespace glow {

/// Creates tensor \p T from the input \p in. Note, there is no data associated
/// with the Tensor. This method makes sure that the tensor is created with the
/// proper shape and element type.
static llvm::Error setTensorType(const ONNX_NAMESPACE::TypeProto &in,
                                 Tensor *T) {
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

llvm::Error ONNXIFIModelLoader::loadInputs(ONNX_NAMESPACE::GraphProto &net,
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

/// Loads tensor \p T from the input \p in.
static llvm::Error loadWeight(const onnxTensorDescriptorV1 &in, Tensor *T) {
  // Only support CPU memory tensors.
  if (in.memoryType != ONNXIFI_MEMORY_TYPE_CPU) {
    RETURN_ERR("Only support CPU memory tensors.");
  }

  std::vector<size_t> dims;
  for (unsigned i = 0; i < in.dimensions; ++i) {
    dims.push_back(in.shape[i]);
  }

  if (in.dataType == ONNXIFI_DATATYPE_FLOAT32) {
    T->reset(ElemKind::FloatTy, dims);

    auto TH = T->getHandle<>();
    float *data = (float *)in.buffer;
    for (size_t i = 0; i < TH.size(); ++i) {
      TH.raw(i) = data[i];
    }
  } else if (in.dataType == ONNXIFI_DATATYPE_UINT64 ||
             in.dataType == ONNXIFI_DATATYPE_INT64) {
    const bool inDataSigned = in.dataType == ONNXIFI_DATATYPE_INT64;
    (void)inDataSigned;
    T->reset(ElemKind::Int64ITy, dims);

    auto TH = T->getHandle<int64_t>();
    int64_t *data = (int64_t *)in.buffer;
    for (size_t i = 0; i < TH.size(); ++i) {
      RETURN_ERR_IF_NOT(
          (inDataSigned || data[i] >= 0),
          "Disallow overflow of loaded UINT64 data into Int64ITy.");
      TH.raw(i) = data[i];
    }
  } else if (in.dataType == ONNXIFI_DATATYPE_INT32) {
    T->reset(ElemKind::Int32ITy, dims);

    auto TH = T->getHandle<int32_t>();
    int32_t *data = (int32_t *)in.buffer;
    for (size_t i = 0; i < TH.size(); ++i) {
      TH.raw(i) = data[i];
    }
  } else {
    RETURN_ERR("Only float and index tensors are supported.");
  }

  return llvm::Error::success();
}

llvm::Error ONNXIFIModelLoader::loadWeights(
    uint32_t weightsCount, const onnxTensorDescriptorV1 *weightDescriptors) {
  for (uint32_t i = 0; i < weightsCount; ++i) {
    std::unique_ptr<Tensor> T(new Tensor());
    RETURN_IF_ERR(loadWeight(weightDescriptors[i], T.get()));
    tensors_[weightDescriptors[i].name] = std::move(T);
  }

  return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<ONNXIFIModelLoader>>
ONNXIFIModelLoader::parse(const void *onnxModel, uint32_t onnxModelSize,
                          uint32_t weightsCount,
                          const onnxTensorDescriptorV1 *weightDescriptors,
                          Function &F, bool loadInputsAsPlaceholders) {
  llvm::Error loaderConstructionErr = llvm::Error::success();
  std::unique_ptr<ONNXIFIModelLoader> loader(
      new ONNXIFIModelLoader(F, &loaderConstructionErr));
  if (loaderConstructionErr) {
    return std::move(loaderConstructionErr);
  }

  ONNX_NAMESPACE::ModelProto modelDef;
  ASSIGN_VALUE_OR_RETURN_ERR(modelDef,
                             loader->loadProto(onnxModel, onnxModelSize));

  RETURN_IF_ERR(loader->setVersion(modelDef));

  RETURN_IF_ERR(loader->loadWeights(weightsCount, weightDescriptors));

  ONNX_NAMESPACE::GraphProto graphDef = modelDef.graph();

  RETURN_IF_ERR(loader->loadInputs(graphDef, loadInputsAsPlaceholders));

  RETURN_IF_ERR(loader->loadInitializers(graphDef));

  RETURN_IF_ERR(loader->loadNetwork(graphDef));

  RETURN_IF_ERR(loader->setOutputNodes(graphDef));

  return llvm::Expected<std::unique_ptr<ONNXIFIModelLoader>>(std::move(loader));
}
} // namespace glow
