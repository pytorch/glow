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
  ONNXIFIModelLoader(Function &F) : ONNXModelLoader(F) {}

  /// Load the inputs from the GraphProto. This is useful when the
  /// initializers are not available.
  llvm::Error loadInputs(ONNX_NAMESPACE::GraphProto &net);

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

  /// \returns a ONNXIFIModelLoader if \p onnxModel can be
  /// parsed and static weights can be loaded from the \p wightDescriptors.
  /// \returns Error otherwise.
  static llvm::Expected<ONNXIFIModelLoader>
  parse(const void *onnxModel, uint32_t onnxModelSize, uint32_t weightsCount,
        const onnxTensorDescriptorV1 *weightDescriptors, Function &F);

  /// \returns empty std::vector if any of the ONNX operators from
  /// the \p onnxModel is not supported by the ONNX model parser.
  /// \returns std::vector of Glow operation kind and element kind otherwise.
  ///          It represents a mapping between ONNX nodes and Glow operations.
  ///
  /// \param onnxModel contains a single ONNX operator.
  static std::vector<std::pair<Kinded::Kind, ElemKind>>
  parseOperators(const void *onnxModel, size_t onnxModelSize);
};

} // namespace glow

#endif // GLOW_IMPORTER_ONNXIFIMODELLOADER_H
