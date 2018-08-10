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

#include "glow/Importer/ONNXIFILoader.h"

#include "onnx/onnx.pb.h"

namespace glow {
namespace onnxifi {

/// Creates tensor \p T from the input \p in. Note, there is no data associated
/// with the Tensor. This method makes sure that the tensor is created with the
/// proper shape and element type.
static void setTensorType(const ONNX_NAMESPACE::TypeProto &in, Tensor *T) {
  std::vector<size_t> dim;
  for (auto d : in.tensor_type().shape().dim()) {
    dim.push_back(d.dim_value());
  }

  if (in.tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto::FLOAT) {
    T->reset(ElemKind::FloatTy, dim);
  } else if (in.tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto::INT64) {
    // TODO: either switch IndexTy to be 64 bit, or switch to another type here.
    T->reset(ElemKind::IndexTy, dim);
  } else {
    assert(false && "Only float and index tensors are supported");
  }
}

void ModelLoader::loadInputs(ONNX_NAMESPACE::GraphProto &net) {
  for (const auto &in : net.input()) {
    Tensor *T = new Tensor();
    setTensorType(in.type(), T);
    tensors_[in.name()] = T;
  }
}

/// Loads tensor \p T from the input \p in.
static bool loadWeight(const onnxTensorDescriptorV1 &in, Tensor *T) {
  std::cout << "memory type: " << in.memoryType << " CPU: " << ONNXIFI_MEMORY_TYPE_CPU << std::endl;
  std::cout << "trying to load dimensions:" << in.dimensions << std::endl;
  // Only support CPU memory tensors.
  if (in.memoryType && in.memoryType != ONNXIFI_MEMORY_TYPE_CPU) {
    return false;
  }

  
  std::vector<size_t> dims;
  for (unsigned i = 0; i < in.dimensions; ++i) {
    dims.push_back(in.shape[i]);
  }

  if (in.dataType == ONNXIFI_DATATYPE_FLOAT32) {
    std::cout << "trying to load tensor, dims: " << in.dimensions <<  std::endl;
    T->reset(ElemKind::FloatTy, dims);

    auto TH = T->getHandle<>();
    float *data = (float *)in.buffer;
    for (size_t i = 0; i < TH.size(); ++i) {
      TH.raw(i) = data[i];
    }
  } else if (in.dataType == ONNXIFI_DATATYPE_UINT64) {
    // TODO: either switch IndexTy to be 64 bit, or switch to another type here.
    T->reset(ElemKind::IndexTy, dims);

    auto TH = T->getHandle<size_t>();
    int64_t *data = (int64_t *)in.buffer;
    for (size_t i = 0; i < TH.size(); ++i) {
      TH.raw(i) = data[i];
    }
  } else {
    llvm_unreachable("Only float and index tensors are supported");
  }

  return false;
}

bool ModelLoader::loadWeights(uint32_t weightsCount,
                              const onnxTensorDescriptorV1 *weightDescriptors) {
  for (uint32_t i = 0; i < weightsCount; ++i) {
    std::cout << "loading tensor" << std::endl;
    Tensor *T = new Tensor();

    std::cout << "loading weight" << std::endl;
    if (!loadWeight(weightDescriptors[i], T)) {
      std::cout << "failed to load the weight" << std::endl;
      return false;
    }

    std::cout << "set the weight descriptor" << std::endl;
    tensors_[weightDescriptors[i].name] = T;
  }

  return true;
}

std::unique_ptr<ModelLoader> ModelLoader::parse(
    const void *onnxModel, uint32_t onnxModelSize, uint32_t weightsCount,
    const onnxTensorDescriptorV1 *weightDescriptors, Function &F) {
  std::unique_ptr<ModelLoader> loader(new ModelLoader(F));

  //ONNX_NAMESPACE::GraphProto modelDef;
  //std::cout << "before loading proto" << std::endl;
  //if (!loader->loadProto(modelDef, onnxModel, onnxModelSize)) {
  //  return nullptr;
  //}
  
  std::cout << "before loading weights, weightsCount" << weightsCount << std::endl;
  if (!loader->loadWeights(weightsCount, weightDescriptors)) {
    return nullptr;
  }

  std::cout << "before loading inputs" << std::endl;
  /*loader->loadInputs(modelDef);

  std::cout << "before loading outputs" << std::endl;
  if (!loader->setOutputNodes(modelDef)) {
    return nullptr;
  }

  std::cout << "before loading network" << std::endl;
  if (!loader->loadNetwork(modelDef)) {
    return nullptr;
  }*/

  return loader;
}

std::unique_ptr<ModelLoader>
ModelLoader::parse(const void *onnxModel, size_t onnxModelSize, Function &F) {
  std::unique_ptr<ModelLoader> loader(new ModelLoader(F));

  ONNX_NAMESPACE::GraphProto modelDef;
  if (!loader->loadProto(modelDef, onnxModel, onnxModelSize)) {
    return nullptr;
  }

  loader->loadInputs(modelDef);
  if (!loader->loadNetwork(modelDef)) {
    return nullptr;
  }

  return loader;
}

} // namespace onnxifi
} // namespace glow
