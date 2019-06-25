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
#ifndef GLOW_OPTIMIZER_GRAPHOPTIMIZER_COMPILATIONCONTEXT_H
#define GLOW_OPTIMIZER_GRAPHOPTIMIZER_COMPILATIONCONTEXT_H

#include "glow/Backends/BackendOptions.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Quantization/Base/Base.h"
#include "glow/Support/Error.h"

namespace glow {

/// Configuration for different precision modes.
struct PrecisionConfiguration {
  /// Enum for what kind of transformation should be done for Quantization.
  enum class QuantizationMode {
    None,     /// Perform no transformations for quantization.
    Quantize, /// Quantize the graph using previously gathered statistics.
    Profile,  /// Add profiling nodes for quantization statistics gathering.
  } quantMode{QuantizationMode::None};

  /// Configuration for Quantization.
  quantization::QuantizationConfiguration quantConfig;

  /// Whether to convert the FloatTy to Float16Ty in the Function.
  bool convertToFP16{false};

  /// Used during Quantization and convertToFP16 to keep the original precision
  /// of specific node kinds (i.e. quantization/FP16 conversion would be skipped
  /// for any node kinds found here). Used during profiling to prevent nodes
  /// from being lowered before instrumenting the graph (e.g. do not lower group
  /// convolutions for profiling; see `-do-not-lower-nodes-for-profiling` in
  /// docs/Quantization.md).
  KindSet precisionModeKindSet;
};

using QuantizationMode = PrecisionConfiguration::QuantizationMode;

/// Context for compilation.
struct CompilationContext {
  /// Used during Profiling.
  PlaceholderBindings *bindings{nullptr};

  /// Used during Quantization and Profiling.
  LoweredInfoMap *loweredInfoMap{nullptr};

  /// Select whether in Training or Inference mode.
  enum class CompilationMode {
    Train, /// Compile the graph in preperation for training.
    Infer, /// Compile the graph for inference. Notice that this operation
           /// changes the graph in a way that is not reversible.
  } compMode{CompilationMode::Infer};

  /// Options for the Backend to use.
  BackendOptions backendOpts;

  /// Configuration for different precision modes.
  PrecisionConfiguration precisionConfig;

  CompilationContext(PlaceholderBindings *bindings_ = nullptr,
                     LoweredInfoMap *loweredInfoMap_ = nullptr)
      : bindings(bindings_), loweredInfoMap(loweredInfoMap_) {}

  /// \returns an error if the CompilationContext is malformed for whatever
  /// configuration it is set up for, otherwise returns success.
  llvm::Error verify() const {
    switch (precisionConfig.quantMode) {
    case QuantizationMode::Profile:
      RETURN_ERR_IF_NOT(bindings, GlowErr::ErrorCode::COMPILE_CONTEXT_MALFORMED,
                        "In Profiling mode, but bindings was not set.\n");

      RETURN_ERR_IF_NOT(loweredInfoMap,
                        GlowErr::ErrorCode::COMPILE_CONTEXT_MALFORMED,
                        "In Profiling mode, but loweredInfoMap was not set.\n");

      RETURN_ERR_IF_NOT(!precisionConfig.convertToFP16,
                        GlowErr::ErrorCode::COMPILE_CONTEXT_MALFORMED,
                        "Converting to FP16 while profiling is unsupported.\n");
      break;

    case QuantizationMode::Quantize:
      RETURN_ERR_IF_NOT(
          loweredInfoMap, GlowErr::ErrorCode::COMPILE_CONTEXT_MALFORMED,
          "In Quantization mode, but loweredInfoMap was not set.\n");
      break;

    case QuantizationMode::None:
      break;
    }

    return llvm::Error::success();
  }
};

using CompilationMode = CompilationContext::CompilationMode;

}; // namespace glow

#endif // GLOW_OPTIMIZER_GRAPHOPTIMIZER_COMPILATIONCONTEXT_H
