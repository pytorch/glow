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
#ifndef GLOW_OPTIMIZER_GRAPHOPTIMIZER_COMPILATIONCONTEXT_H
#define GLOW_OPTIMIZER_GRAPHOPTIMIZER_COMPILATIONCONTEXT_H

#include "glow/Backends/BackendOptions.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Quantization/Base/Base.h"
#include "glow/Support/Error.h"

namespace glow {
namespace runtime {
struct PartitionConfig;
class DeferredWeightLoader;
} // namespace runtime

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

  /// Whether to convert UInt8FusedQTy to UInt8FusedFP16QTy in the Function.
  bool convertFusedToFP16{false};

  /// If convertToFP16, whether to convert input Placeholders.
  bool convertPlaceholdersToFP16{false};

  /// If convertToFP16, whether to convert Constants.
  bool convertConstantsToFP16{false};

  /// If convertToFP16, whether to clip out-of-range FP values to the min/max of
  /// fp16.
  bool clipFP16{false};

  /// If clipFP16, whether to skip clipping inputs of Nodes.
  bool clipFP16SkipInputs{false};

  /// Whether to force FP16 accumulation for the SLS family of ops.
  bool forceFP16AccumSLS{true};

  /// Used during Quantization and convertToFP16 to keep the original precision
  /// of specific node kinds (i.e. quantization/FP16 conversion would be skipped
  /// for any node kinds found here). Used during profiling to prevent nodes
  /// from being lowered before instrumenting the graph (e.g. do not lower group
  /// convolutions for profiling; see `-do-not-lower-nodes-for-profiling` in
  /// docs/Quantization.md).
  KindSet precisionModeKindSet;

  /// Whether to use the precisionModeKindSet as a whitelist instead of the
  /// default blacklist. Currently only supported for convertToFP16.
  bool useSetAsWhitelist{false};
};

using QuantizationMode = PrecisionConfiguration::QuantizationMode;

/// Options relevant to optimizations during compilation.
struct OptimizationOptions {
  /// Only lower, i.e. skip optimizations and precision transformations. Used
  /// for testing.
  llvm::SmallSet<Function *, 1> onlyLowerFuns;

  /// If true, perform compile-time computation of constant operations.
  bool enableConstantFolding{true};

  /// If true, this will merge ConvertTo and Quantize nodes into inputs and
  /// outputs of the Function. This means modifying the types of Placeholders
  /// and SaveNodes if they have a corresponding ElemKind conversion (ConvertTo,
  /// Quantize, Dequantize nodes). Note that this must be accompanied by
  /// modifying the Tensors backing Placeholders at runtime.
  bool foldElemKindConversionIntoIO{false};

  /// If true this will fold convertTo and Quantize nodes into only static
  /// placeholders. The conversion of the Tensors will be handled by the
  /// provisioner.
  bool foldStaticPlaceholderConversions{false};

  /// If true, this will direct the partitioner to use SparseNN partitioning
  /// scheme
  bool useSparseNNPartitioningScheme{false};

  /// The number of cards over which to split SLS tables when using SparseNN
  /// partitioning scheme
  unsigned int sparseNNPartitioningSchemeNumCards{1};

  /// The number of bytes to allocate per card for SLS tables when using
  /// the SparseNN partitioning scheme
  unsigned int sparseNNPartitioningSchemeSLSTableKBytesPerCard{0};

  /// The number of cores to assign to SLS partition when using SparseNN
  /// partitioning scheme
  unsigned int sparseNNPartitioningSchemeNumCoresSLS{1};

  /// The number of cores to assign to non-SLS partition when using SparseNN
  /// partitioning scheme
  unsigned int sparseNNPartitioningSchemeNumCoresOther{1};
};

/// Context for compilation.
struct CompilationContext {
  /// Used during Profiling.
  PlaceholderBindings *bindings{nullptr};

  /// Allows the user to specify user defined partitioning.
  runtime::PartitionConfig *partitionConfig{nullptr};

  /// Used during Quantization and Profiling.
  LoweredInfoMap *loweredInfoMap{nullptr};

  /// Select whether in Training or Inference mode.
  enum class CompilationMode {
    Train, /// Compile the graph in preperation for training.
    Infer, /// Compile the graph for inference. Notice that this operation
           /// changes the graph in a way that is not reversible.
    NumCompilationModes, /// Used to count the number of CompilationModes.
  } compMode{CompilationMode::Infer};

  /// Options for the Backend to use.
  BackendOptions backendOpts;

  /// Options for the optimizations to use.
  OptimizationOptions optimizationOpts;

  /// Configuration for different precision modes.
  PrecisionConfiguration precisionConfig;

  /// How to annotate the compilation log filename.
  std::string compilationLogPrefix{"glow"};

  /// Pointer to deferredWeightLoader object, this is used for large model
  /// support.
  runtime::DeferredWeightLoader *deferredWeightLoader{nullptr};

  // Whether to print out issues/logging during compilation. Used for example to
  // disable printing issues encountered during ConstantFolding.
  bool verboseCompile{true};

  CompilationContext(PlaceholderBindings *bindings_ = nullptr,
                     LoweredInfoMap *loweredInfoMap_ = nullptr)
      : bindings(bindings_), loweredInfoMap(loweredInfoMap_) {}

  /// \returns an error if the CompilationContext is malformed for whatever
  /// configuration it is set up for, otherwise returns success.
  Error verify() const {
    RETURN_ERR_IF_NOT(!precisionConfig.useSetAsWhitelist ||
                          precisionConfig.convertToFP16,
                      "Can only use the precisionModeKindSet as a whitelist in "
                      "convertToFP16 mode.");

    switch (precisionConfig.quantMode) {
    case QuantizationMode::Profile:
      RETURN_ERR_IF_NOT(bindings,
                        ErrorValue::ErrorCode::COMPILE_CONTEXT_MALFORMED,
                        "In Profiling mode, but bindings was not set.\n");

      RETURN_ERR_IF_NOT(loweredInfoMap,
                        ErrorValue::ErrorCode::COMPILE_CONTEXT_MALFORMED,
                        "In Profiling mode, but loweredInfoMap was not set.\n");

      RETURN_ERR_IF_NOT(!precisionConfig.convertToFP16,
                        ErrorValue::ErrorCode::COMPILE_CONTEXT_MALFORMED,
                        "Converting to FP16 while profiling is unsupported.\n");
      break;

    case QuantizationMode::Quantize:
      RETURN_ERR_IF_NOT(
          loweredInfoMap, ErrorValue::ErrorCode::COMPILE_CONTEXT_MALFORMED,
          "In Quantization mode, but loweredInfoMap was not set.\n");
      break;

    case QuantizationMode::None:
      break;
    }

    return Error::success();
  }
};

using CompilationMode = CompilationContext::CompilationMode;

}; // namespace glow

#endif // GLOW_OPTIMIZER_GRAPHOPTIMIZER_COMPILATIONCONTEXT_H
