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

  /// \returns vector of primary input names based on their position
  const std::vector<std::string> &getPositionalInputNames() const {
    return core_->getPositionalInputNames();
  }

  /// \returns vector of primary output names based on their position
  const std::vector<std::string> &getPositionalOutputNames() const {
    return core_->getPositionalOutputNames();
  }

  /// \returns a unique_ptr<ONNXIFIModelLoader> if \p onnxModel can be
  /// parsed and static weights can be loaded from the \p weightDescriptors.
  /// \returns Error otherwise. \p loadInputsAsPlaceholdersForOnnx is passed to
  /// loadInputs to determine whether graph inputs are loaded as Placeholders or
  /// Tensors. Loading inputs as Tensors is useful for when weights are not
  /// provided such as when the graph being loaded is actually a small patch of
  /// a larger graph because the graph inputs in this case may represent
  /// internal values for the larger graph. \p constFoldInLoader is used to
  /// determine whether to try constant folding at load time. \p mod will be
  /// filled wth one or more Functions built. If the model is pre-partitioned,
  /// then prepartitionedConfig from \p cctx will be filled with relevant
  /// configuration for partitioning, and all Functions created will be named
  /// with prefix \p netName. Otherwise prepartitionedConfig from \p cctx is
  /// ignored, and \p netName is used as the name of the single Function that is
  /// created inside \p mod. backendOpts.backendSpecificNodeInfo from \p cctx
  /// is filled with any info loaded from the proto, relevant for custom ONNX
  /// Glow models only. \p staticPlaceholderTypes will be filled with types to
  /// use for static Placeholders if the proto being parsed is a custom ONNX
  /// Glow model and contains such information; otherwise it's left unchanged.
  static Expected<std::unique_ptr<ONNXIFIModelLoader>>
  parse(const void *onnxModel, uint32_t onnxModelSize, uint32_t weightsCount,
        const onnxTensorDescriptorV1 *weightDescriptors, Module &mod,
        llvm::StringRef netName, CompilationContext &cctx,
        std::map<std::string, Type> *staticPlaceholderTypes,
        bool loadInputsAsPlaceholdersForOnnx = true, bool use_onnx = true,
        bool constFoldInLoader = true);
};

} // namespace glow

#endif // GLOW_IMPORTER_ONNXIFIMODELLOADER_H
