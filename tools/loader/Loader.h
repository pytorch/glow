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
#ifndef GLOW_TOOLS_LOADER_LOADER_H
#define GLOW_TOOLS_LOADER_LOADER_H

#include "glow/ExecutionEngine/ExecutionEngine.h"

namespace glow {

class Tensor;

/// \return true if emit bundle mode is enabled.
bool emittingBundle();

/// \return true if profiling the graph.
bool profilingGraph();

/// Driver class for loading, compiling, and running inference for ONNX and
/// Caffe2 models.
class Loader {
  /// Caffe2 network file name.
  std::string caffe2NetDescFilename_;
  /// Caffe2 weights file name.
  std::string caffe2NetWeightFilename_;
  /// ONNX model file name.
  std::string onnxModelFilename_;
  /// Execution engine for compiling and running.
  ExecutionEngine EE_{};
  /// Function containing the model.
  Function *F_{nullptr};
  /// A map between quantization profiling names of NodeValues that were lowered
  /// from each other. Maps to a set of names of NodeValues and their NodeKinds
  /// that were replaced by the NodeValue (whose output name is the key) that
  /// replaced them.
  LoweredInfoMap loweredMap_;

public:
  /// Getter for the Function.
  Function *getFunction() { return F_; }
  /// Getter for the Module.
  Module *getModule() { return F_->getParent(); }
  /// Getter for the Caffe2 network file name.
  llvm::StringRef getCaffe2NetDescFilename() { return caffe2NetDescFilename_; }
  /// Getter for the Caffe2 weights file name.
  llvm::StringRef getCaffe2NetWeightFilename() {
    return caffe2NetWeightFilename_;
  }
  /// Getter for the ONNX model file name.
  llvm::StringRef getOnnxModelFilename() { return onnxModelFilename_; }
  /// Getter for the model path.
  /// \pre (modelPathOpt.size() == 1)
  llvm::StringRef getModelOptPath();

  /// Compiles the Function F_. Handles quantization, emitting bundles, and
  /// dumping debug information. \p bindings bind specific
  /// placeholders to concrete tensors. The concrete tensors include
  /// quantization profile guided information.
  void compile(PlaceholderBindings &bindings);

  /// Runs inference, unless emit bundle mode is enabled. \p bindings
  /// binds specific placeholders to concrete tensors. The concrete
  /// tensors include quantization profile guided information.
  void runInference(PlaceholderBindings &bindings, size_t batchSize = 1);

  /// Generates and serializes the quantization infos after gathering a profile
  /// by running inference one or more times. \p bindings
  /// binds specific placeholders to concrete tensors. The concrete tensors
  /// include quantization profile guided information.
  void generateAndSerializeQuantizationInfos(PlaceholderBindings &bindings);

  /// Create the Loader driver object, and parse/verify the command line
  /// parameters.
  Loader(int argc, char **argv);
};

} // namespace glow

#endif // GLOW_TOOLS_LOADER_LOADER_H
