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

#ifndef GLOW_IMPORTER_ONNXIFIMODELLOADER_H
#define GLOW_IMPORTER_ONNXIFIMODELLOADER_H

#include "foxi/onnxifi.h"

#include "glow/Importer/ONNXModelLoader.h"

#include "llvm/ADT/StringMap.h"

namespace glow {

class ONNXIFIModelLoader {
private:
  /// Default constructor.
  explicit ONNXIFIModelLoader() {}

  /// The real loader. It can be ONNXModelLoader or Caffe2ModelLoader
  std::unique_ptr<ProtobufLoader> core_{nullptr};

public:
  /// \returns mapping between ONNX names and actual Glow input vars.
  const llvm::StringMap<Placeholder *> &getInputVarsMapping() const {
    return core_->getInputVarsMapping();
  }

  /// \returns mapping between ONNX names and actual Glow output nodes.
  const llvm::StringMap<Placeholder *> &getOutputVarsMapping() const {
    return core_->getOutputVarsMapping();
  }

  /// \returns a unique_ptr<ONNXIFIModelLoader> if \p onnxModel can be
  /// parsed and static weights can be loaded from the \p wightDescriptors.
  /// \returns Error otherwise. \p loadInputsAsPlaceholders is passed to
  /// loadInputs to determine whether graph inputs are loaded as Placeholders or
  /// Tensors. Loading inputs as Tensors is useful for when weights are not
  /// provided such as when the graph being loaded is actually a small patch of
  /// a larger graph because the graph inputs in this case may represent
  /// internal values for the larger graph. \p constFoldInLoader is used to
  /// determine whether to try constant folding at load time.
  static Expected<std::unique_ptr<ONNXIFIModelLoader>>
  parse(const void *onnxModel, uint32_t onnxModelSize, uint32_t weightsCount,
        const onnxTensorDescriptorV1 *weightDescriptors, Function &F,
        bool loadInputsAsPlaceholders = true, bool use_onnx = true,
        bool constFoldInLoader = true);
};

} // namespace glow

#endif // GLOW_IMPORTER_ONNXIFIMODELLOADER_H
