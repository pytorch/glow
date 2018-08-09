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

#include "onnx.pb.h"

namespace glow {
namespace onnxifi {

/// Creates tensor \p T from the input \p in. Note, there is no data associated
/// with the Tensor. This method makes sure that the tensor is created with the
/// proper shape and element type.
static void setTensorType(const onnx::TypeProto &in, Tensor *T) {
  std::vector<uint64_t> dim;
  for (auto d : in.tensor_type().shape().dim()) {
    dim.push_back(d.dim_value());
  }

  if (in.tensor_type().elem_type() == onnx::TensorProto::FLOAT) {
    T->reset(ElemKind::FloatTy, dim);
  } else if (in.tensor_type().elem_type() == onnx::TensorProto::INT64) {
    // TODO: either switch IndexTy to be 64 bit, or switch to another type here.
    T->reset(ElemKind::IndexTy, dim);
  } else {
    assert(false && "Only float and index tensors are supported");
  }
}

void ModelLoader::loadInputs(onnx::GraphProto &net) {
  for (const auto &in : net.input()) {
    Tensor *T = new Tensor();
    setTensorType(in.type(), T);
    tensors_[in.name()] = T;
  }
}

/// Loads tensor \p T from the input \p in.
static bool loadWeight(const onnxTensorDescriptorV1 &in, Tensor *T) {
  // Only support CPU memory tensors.
  if (in.memoryType != ONNXIFI_MEMORY_TYPE_CPU) {
    return false;
  }

  std::vector<uint64_t> dims;
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
  } else if (in.dataType == ONNXIFI_DATATYPE_UINT64) {
    // TODO: either switch IndexTy to be 64 bit, or switch to another type here.
    T->reset(ElemKind::IndexTy, dims);

    auto TH = T->getHandle<uint64_t>();
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
    Tensor *T = new Tensor();

    if (!loadWeight(weightDescriptors[i], T)) {
      return false;
    }

    tensors_[weightDescriptors[i].name] = T;
  }

  return true;
}

std::unique_ptr<ModelLoader> ModelLoader::parse(
    const void *onnxModel, uint32_t onnxModelSize, uint32_t weightsCount,
    const onnxTensorDescriptorV1 *weightDescriptors, Function &F) {
  std::unique_ptr<ModelLoader> loader(new ModelLoader(F));

  onnx::GraphProto modelDef;
  if (!loader->loadProto(modelDef, onnxModel, onnxModelSize)) {
    return nullptr;
  }

  if (!loader->loadWeights(weightsCount, weightDescriptors)) {
    return nullptr;
  }
  loader->loadInputs(modelDef);

  if (!loader->loadNetwork(modelDef)) {
    return nullptr;
  }

  return loader;
}

std::unique_ptr<ModelLoader>
ModelLoader::parse(const void *onnxModel, size_t onnxModelSize, Function &F) {
  std::unique_ptr<ModelLoader> loader(new ModelLoader(F));

  onnx::GraphProto modelDef;
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
