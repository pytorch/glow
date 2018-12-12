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

#ifndef GLOW_IMPORTER_ONNXIFIMODELLOADER_H
#define GLOW_IMPORTER_ONNXIFIMODELLOADER_H

#include "onnx/onnxifi.h"

#include "glow/Importer/ONNXModelLoader.h"

#include "llvm/ADT/StringMap.h"

namespace glow {

class ONNXIFIModelLoader : public ONNXModelLoader {
private:
  /// If \p errPtr is not null then if an error occurs it will get assigned
  /// there otherwise if an error occurs it will abort.
  ONNXIFIModelLoader(Function &F, llvm::Error *errPtr = nullptr)
      : ONNXModelLoader(F, errPtr) {}

  /// Load the inputs from the GraphProto. If \p loadInputsAsPlaceholders is
  /// true then this will load each graph input as a placeholder otherwise it
  /// will create an empty tensor for each input.
  llvm::Error loadInputs(ONNX_NAMESPACE::GraphProto &net,
                         bool loadInputsAsPlaceholders);

  /// Load pre-trained weights from \p weightDescriptors.
  llvm::Error loadWeights(uint32_t weightsCount,
                          const onnxTensorDescriptorV1 *weightDescriptors);

  /// Mapping between ONNX names for inputs and actual Glow input vars.
  llvm::StringMap<Placeholder *> onnxNameToInputVars_;

public:
  /// \returns mapping between ONNX names and actual Glow input vars.
  const llvm::StringMap<Placeholder *> &getInputVarsMapping() const {
    return onnxNameToInputVars_;
  }

  /// \returns mapping between ONNX names and actual Glow output nodes.
  const llvm::StringMap<Placeholder *> &getOutputVarsMapping() const {
    return outputVarsByName_;
  }

  /// \returns a unique_ptr<ONNXIFIModelLoader> if \p onnxModel can be
  /// parsed and static weights can be loaded from the \p wightDescriptors.
  /// \returns Error otherwise. \p loadInputsAsPlaceholders is passed to
  /// loadInputs to determine whether inputs are loaded as Placeholders or
  /// Tensors. Loading as Tensors is useful for when the graph being loaded is
  /// actually a small slice of a larger graph.
  static llvm::Expected<std::unique_ptr<ONNXIFIModelLoader>>
  parse(const void *onnxModel, uint32_t onnxModelSize, uint32_t weightsCount,
        const onnxTensorDescriptorV1 *weightDescriptors, Function &F,
        bool loadInputsAsPlaceholders = true);
};

} // namespace glow

#endif // GLOW_IMPORTER_ONNXIFIMODELLOADER_H
