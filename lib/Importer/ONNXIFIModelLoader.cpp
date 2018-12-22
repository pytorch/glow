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
#include "glow/Importer/Caffe2ModelLoader.h"

#include "caffe2/proto/caffe2.pb.h"
#include "onnx/onnx_pb.h"

namespace glow {

llvm::Expected<std::unique_ptr<ONNXIFIModelLoader>> ONNXIFIModelLoader::parse(
    const void *model, uint32_t modelSize, uint32_t weightsCount,
    const onnxTensorDescriptorV1 *weightDescriptors, Function &F,
    bool loadInputsAsPlaceholders, bool use_onnx) {
  llvm::Error loaderConstructionErr = llvm::Error::success();
  std::unique_ptr<ONNXIFIModelLoader> loader(
      new ONNXIFIModelLoader(&loaderConstructionErr));
  if (loaderConstructionErr) {
    return std::move(loaderConstructionErr);
  }

  if (use_onnx) {
    std::unique_ptr<ONNXModelLoader> onnxLoader(
        new ONNXModelLoader(F, &loaderConstructionErr));
    if (loaderConstructionErr) {
      return std::move(loaderConstructionErr);
    }
    ONNX_NAMESPACE::ModelProto modelDef;
    ASSIGN_VALUE_OR_RETURN_ERR(modelDef,
                               onnxLoader->loadProto(model, modelSize));

    RETURN_IF_ERR(onnxLoader->setVersion(modelDef));

    RETURN_IF_ERR(onnxLoader->loadWeights(weightsCount, weightDescriptors));

    ONNX_NAMESPACE::GraphProto graphDef = modelDef.graph();

    RETURN_IF_ERR(onnxLoader->loadInputs(graphDef, loadInputsAsPlaceholders));

    RETURN_IF_ERR(onnxLoader->loadInitializers(graphDef));

    RETURN_IF_ERR(onnxLoader->loadNetwork(graphDef));

    RETURN_IF_ERR(onnxLoader->setOutputNodes(graphDef));

    loader->onnxNameToInputVars_ = onnxLoader->getInputVarsMapping();

    // Keep hold of the context
    loader->core_ = std::move(onnxLoader);
  } else {
    // Use Caffe2 Model loader
    std::unique_ptr<Caffe2ModelLoader> c2Loader(
        new Caffe2ModelLoader(F, &loaderConstructionErr));
    if (loaderConstructionErr) {
      return std::move(loaderConstructionErr);
    }

    caffe2::NetDef networkDef;
    ASSIGN_VALUE_OR_RETURN_ERR(networkDef,
                               c2Loader->loadProto(model, modelSize));

    RETURN_IF_ERR(c2Loader->loadWeights(weightsCount, weightDescriptors));

    RETURN_IF_ERR(c2Loader->loadInputs(networkDef, loadInputsAsPlaceholders));

    // TODO: in Caffe2ModelLoader, setOutputNodes is actually inside
    // loadNetwork, maybe we should make it a separate function?
    RETURN_IF_ERR(c2Loader->loadNetwork(networkDef));

    loader->onnxNameToInputVars_ = c2Loader->getInputVarsMapping();

    // Keep hold of the context
    loader->core_ = std::move(c2Loader);
  }
  return llvm::Expected<std::unique_ptr<ONNXIFIModelLoader>>(std::move(loader));
}
} // namespace glow
