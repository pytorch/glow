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

#ifndef GLOW_IMPORTER_ONNXIFILOADER_H
#define GLOW_IMPORTER_ONNXIFILOADER_H

#include "onnx/onnxifi.h"

#include "glow/Importer/ONNX.h"

#include "llvm/ADT/StringMap.h"

namespace glow {
namespace onnxifi {

class ModelLoader : public ONNXModelLoader {
private:
  ModelLoader(Function &F) : ONNXModelLoader(F) {}

  /// Load the inputs from the GraphProto. This is useful when the
  /// initializers are not available.
  void loadInputs(ONNX_NAMESPACE::GraphProto &net);

  /// Load pre-trained weights from \p weightDescriptors.
  bool loadWeights(uint32_t weightsCount,
                   const onnxTensorDescriptorV1 *weightDescriptors);

  /// Mapping between ONNX names for inputs and actual Glow input vars.
  llvm::StringMap<Variable *> onnxNameToInputVars_;

public:
  /// \returns mapping between ONNX names and actual Glow input vars.
  const llvm::StringMap<Variable *> &getInputVarsMapping() const {
    return onnxNameToInputVars_;
  }

  /// \returns mapping between ONNX names and actual Glow output nodes.
  const llvm::StringMap<Variable *> &getOutputVarsMapping() const {
    return outputVarsByName_;
  }

  /// \returns unique pointer to ModelLoader if \p onnxModel can be parsed
  /// and static weights can be loaded from the \p wightDescriptors.
  /// \returns nullptr otherwise.
  static std::unique_ptr<ModelLoader>
  parse(const void *onnxModel, uint32_t onnxModelSize, uint32_t weightsCount,
        const onnxTensorDescriptorV1 *weightDescriptors, Function &F);

  /// \returns nullptr if ONNX operator from the \p onnxModel is not
  /// supported by the ONNX model parser.
  /// \returns unique ptr to operation kind and element kind otherwise.
  ///
  /// \param onnxModel contains a single ONNX operator.
  static std::unique_ptr<std::pair<Kinded::Kind, ElemKind>>
  parseOperator(const void *onnxModel, size_t onnxModelSize);
};

} // namespace onnxifi
} // namespace glow

#endif // GLOW_IMPORTER_ONNXIFILOADER_H
