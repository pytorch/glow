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

#include "glow/Importer/ONNXIFIModelLoader.h"
#include "glow/Importer/Caffe2ModelLoader.h"

#include "caffe2/proto/caffe2.pb.h"
#include "onnx/onnx_pb.h"

namespace glow {

Expected<std::unique_ptr<ONNXIFIModelLoader>> ONNXIFIModelLoader::parse(
    const void *model, uint32_t modelSize, uint32_t weightsCount,
    const onnxTensorDescriptorV1 *weightDescriptors, Module &mod,
    llvm::StringRef netName, CompilationContext &cctx,
    std::map<std::string, Type> *staticPlaceholderTypes,
    bool loadInputsAsPlaceholdersForOnnx, bool use_onnx,
    bool constFoldInLoader) {

  std::unique_ptr<ONNXIFIModelLoader> loader(new ONNXIFIModelLoader());
  Error loaderConstructionErr = Error::empty();

  if (use_onnx) {
    // If we're loading an ONNX model then we will always be replacing dummy
    // TQPs if they're found.
    cctx.precisionConfig.replaceDummyTQPs = true;
    std::unique_ptr<ONNXModelLoader> onnxLoader(new ONNXModelLoader(
        model, modelSize, weightsCount, weightDescriptors, mod, netName,
        cctx.prepartitionedConfig, loadInputsAsPlaceholdersForOnnx,
        &loaderConstructionErr, constFoldInLoader,
        &cctx.backendOpts.backendSpecificNodeInfo, staticPlaceholderTypes,
        cctx.precisionConfig.replaceDummyTQPs));
    if (loaderConstructionErr) {
      return std::move(loaderConstructionErr);
    }
    // Keep hold of the context
    loader->core_ = std::move(onnxLoader);
  } else {
    // Use Caffe2 Model loader
    std::unique_ptr<Caffe2ModelLoader> c2Loader(new Caffe2ModelLoader(
        model, modelSize, weightsCount, weightDescriptors, mod, netName,
        cctx.prepartitionedConfig, &loaderConstructionErr, constFoldInLoader,
        cctx.precisionConfig.originNameToTQPMap,
        cctx.precisionConfig.loadUniquedDummyQParams,
        cctx.precisionConfig.zeroScaleFP16Clip));
    if (loaderConstructionErr) {
      return std::move(loaderConstructionErr);
    }
    // Keep hold of the context
    loader->core_ = std::move(c2Loader);
  }

  return Expected<std::unique_ptr<ONNXIFIModelLoader>>(std::move(loader));
}
} // namespace glow
