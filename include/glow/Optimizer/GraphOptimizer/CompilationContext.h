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
struct PrePartitionedConfig;
class DeferredWeightLoader;
} // namespace runtime

/// Map from Placeholders to their original name and index in the proto that
/// loaded them. Used to keep around info from when we import a proto to then
/// exporting it later on.
using LoadedPlaceholderNameMap =
    std::unordered_map<const Placeholder *, std::pair<std::string, unsigned>>;

/// Map from the name of the original op that some quantization parameters was
/// loaded from to those associated quantization parameters.
using OriginNameToTQPMap =
    std::unordered_map<std::string, TensorQuantizationParams>;

/// Configuration for different precision modes.
struct PrecisionConfiguration {
  /// Enum for what kind of transformation should be done for Quantization.
  enum class QuantizationMode {
    None,     /// Perform no transformations for quantization.
    Quantize, /// Quantize the graph using previously gathered statistics.
    Profile,  /// Add profiling nodes for quantization statistics gathering.
  } quantMode{QuantizationMode::None};

  /// Configuration for Profiling.
  quantization::ProfilingConfiguration profConfig;

  /// Configuration for Quantization.
  quantization::QuantizationConfiguration quantConfig;

  /// Enum for what kind of float16 format should be used.
  enum class Float16Format {
    None,     /// No float16 format should be used.
    FP16,     /// FP16 format for float16 should be used.
    BFloat16, /// FP16 format for float16 should be used.
  } float16Format{
      Float16Format::FP16}; /// If convertToFp16, float16 format to be used.

  /// Whether to convert the FloatTy to Float16Ty in the Function.
  bool convertToFP16{false};

  /// Whether to convert UInt8FusedQTy to UInt8FusedFP16QTy in the Function.
  bool convertFusedToFP16{false};

  /// Whether to convert UInt4FusedFP16QTy to UInt8FusedQTy in the Function.
  bool convert4BitFusedTo8Bit{false};

  /// Whether to convert UInt8FusedFP16QTy to UInt8FusedQTy in the Function.
  bool convert8BitFusedToFP32{false};

  /// Whether to convert UInt4FusedFP16QTy to UInt4FusedQTy in the Function.
  bool convert4BitFusedToFP32{false};

  /// Whether to convert indices in FusedRowwiseSLWS to Int64ITy.
  bool convertIndicesToInt64{false};

  /// If convertToFP16, whether to convert input Placeholders.
  bool convertPlaceholdersToFP16{false};

  /// If convertToFP16, whether to convert Constants.
  bool convertConstantsToFP16{false};

  /// If convertToFp16, whether to skip convert bias from fp32 to fp16 in FC
  bool skipBiasFp32tofp16Convert{false};

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

  /// Pointer to a map of loader names to loaded quant params.
  OriginNameToTQPMap *originNameToTQPMap{nullptr};

  /// If true, then discard original quantization params that are loaded, to
  /// instead track origin of quantization params in \ref originNameToTQPMap.
  bool loadUniquedDummyQParams{false};

  /// If true, when scales for qparams are loaded, they are clipped to
  /// kMinScaleFP16 if below kMinScaleFP16.
  bool zeroScaleFP16Clip{false};

  /// If true, then the model that is loaded is expected to have been originally
  /// serialized with dummy quantization parameters, and was replaced with
  /// actual quantization parameters when loaded in this compilation context.
  bool replaceDummyTQPs{false};

  /// If true, then we can safely assume that all qparams (even dummy qparams)
  /// are clipped inside the FP16 range.
  bool clipQuantRangeToFP16{false};

  /// Converts a float16 \p format into an ElemKind.
  static ElemKind getElementType(Float16Format format) {
    switch (format) {
    case Float16Format::FP16:
      return ElemKind::Float16Ty;
    case Float16Format::BFloat16:
      return ElemKind::BFloat16Ty;
    default:
      llvm_unreachable("Unknown float16 format");
    }
  }
};

using QuantizationMode = PrecisionConfiguration::QuantizationMode;

/// Options relevant to optimizations during compilation.
struct OptimizationOptions {
  /// Only lower, i.e. skip optimizations and precision transformations. Used
  /// for testing.
  llvm::SmallSet<Function *, 1> onlyLowerFuns;

  /// If true, perform compile-time computation of constant operations.
  bool enableConstantFolding{true};

  /// If true, perform compile-time deduplication of Constants.
  bool enableConstantDeduplication{true};

  /// For all Splats in the Function being optimized, if they are used by any
  /// Nodes listed in this set, then they will be materialized into Constants
  /// during Constant Folding.
  KindSet materializeSplatsUsedBySet;

  /// If true, before any Function optimization, all the Constants will be
  /// temporarily replaced by Placeholders, preventing the Constants from being
  /// modified during the normal optimization pipeline. The original Constants
  /// will be put back in place automatically afterward, and then Constant
  /// Folding will be run.
  bool delayAndRecordConstantModification{false};

  /// A set used to hold all temporary PHs that were swapped in for real PHs
  /// when delayAndRecordConstantModification is set.
  std::unordered_set<Placeholder *> tempPHsForConstants;

  /// If true, then there will be no error checking for backend support during
  /// the optimization pipeline. Expected that the caller will check if desired
  /// later on.
  bool skipBackendSupportCheck{false};

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

  /// If true, SparseNN partitioning scheme will add extra concats to the
  /// SLS partition for more efficient inter-partition transfers
  bool sparseNNPartitioningAddSLSConcats{false};

  /// If true, SparseNN partitioning scheme will balance SLS tables across
  /// cards using a performance model
  bool sparseNNPartitioningBalancePerfModel{false};

  /// If true, SparseNN partitioning scheme will move Layer Normalization
  /// nodes immediately following SLS into SLS partitions
  bool sparseNNPartitioningPairLNWithSLS{false};

  /// If true, SparseNN partitioning scheme will move Tile
  /// nodes immediately following SLS for user embeddings into SLS partitions
  bool sparseNNPartitioningPairTileWithSLS{false};

  /// SparseNN partitioning scheme will move nodes specified
  /// in a comma-separated string which immediately follow SLS nodes into SLS
  /// partitions. For example, to move Tanh and Concat, use "Tanh,Concat".
  std::string sparseNNPartitioningPairSLSWith{""};

  // If "Concat" and "Tanh" are specified in sparseNNPartitioningPairSLSWith,
  // this will split large Concats going into a Tanh sink to the specified size
  // before moving them into SLS partitions
  unsigned int sparseNNPartitioningConcatSplitSize{1};

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

  /// The algorithm used for Placement tagging in DAG Optimizer
  std::string DAGOptimizerPlacementTaggingAlgorithm;

  /// The algorithm used for Parallelization tagging in DAG Optimizer
  std::string DAGOptimizerParallelizationTaggingAlgorithm;

  /// The number of parallel chunks used in DAG Optimizer parallelization
  int32_t DAGOptimizerNumParallelChunks;

  /// If it is true (false), perform (not perform) ASAP op placement in DAG
  /// optimization; If it is not set, use acc perf GFlag APLASAPPlacement to
  /// determine whether to perform ASAP op placement or not
  llvm::Optional<bool> enableAPLASAPPlacement;

  /// If true does int64 to int32 type demotion if backend supports for specific
  /// nodes.
  bool enableTypeDemotion{true};

  /// If true, optimizations are allowed to change quantization scale/offset.
  bool enableQuantParamChanges{true};

  /// If true, ConcatNodes will not be merged during the optimizer.
  bool skipConcatMerging{false};

  /// If true, will sink tanh below concat
  bool sinkTanhBelowConcat{false};

  /// Default ctor.
  OptimizationOptions() {
    // By default, always materialize Splats used by ConvolutionNodes, as
    // optimizations such as BatchNorm fusion depend on it.
    materializeSplatsUsedBySet.insert(Kinded::Kind::ConvolutionNodeKind);
  }
};

/// Meta information produced during the compilation. Whereas the compile
/// options should be interpreted as input variables for the compilation, the
/// below structure is output information produced by the compilation process.
struct CompilationInfo {
  /// The hash of the graph before the lowering stage.
  llvm::hash_code graphPreLowerHash{0};
};

/// Context for compilation.
struct CompilationContext {
  /// Used during Profiling.
  PlaceholderBindings *bindings{nullptr};

  /// Allows the user to specify user defined partitioning.
  runtime::PartitionConfig *partitionConfig{nullptr};

  /// Allows a loader to store a pre-partitioned config.
  runtime::PrePartitionedConfig *prepartitionedConfig{nullptr};

  /// If true the HostManager will try to use all available devices on the host.
  bool saturateHost{false};

  /// If greater than zero, this is the number of available devices that are
  /// used when saturateHost is enabled.
  /// If saturateKDevices is zero and saturateHost is enabled, all available
  /// devices will be saturated.
  unsigned saturateKDevices{0};

  /// Number of max active requests per instance of this network.
  unsigned maxActiveRequestsPerInstance{48};

  /// Used during Quantization and Profiling.
  LoweredInfoMap *loweredInfoMap{nullptr};

  /// Set up during model loading to map from Placeholders in the Module to the
  /// symbolic name they were loaded with from the input model.
  LoadedPlaceholderNameMap loadedPHNames;

  /// Select whether in Training or Inference mode.
  enum class CompilationMode {
    Train, /// Compile the graph in preparation for training.
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

  /// Information produced during compilation.
  CompilationInfo info;

  /// How to annotate the compilation log filename.
  std::string compilationLogPrefix{"glow"};

  /// Pointer to deferredWeightLoader object, this is used for large model
  /// support.
  runtime::DeferredWeightLoader *deferredWeightLoader{nullptr};

  /// Whether to print out issues/logging during compilation. Used for example
  /// to disable printing issues encountered during ConstantFolding.
  bool verboseCompile{true};

  /// Call dumpDag on each Function passed to the backend for compilation.
  bool dumpFinalGraph = false;

  /// Path where the dumped graphs should go, default "./".
  std::string dumpGraphPath = "./";

  /// Whether to skip stripping the module.
  bool skipModuleStrip{false};

  /// Enables Peer to Peer Tensor optimization.
  bool enableP2P{false};

  /// Enables Device Resident Tensor optimization.
  bool enableDRT{false};

  /// Number of times a function should be replicated on a device. This is
  /// enabled for single partition networks. For advanced replication setups use
  /// user-defined partitioning.
  unsigned replicationCount{1};

  /// Whether to serialize the DAG that has been optimized and partitioned.
  bool serializeCompiledDAG{false};

  /// Whether to return the Glow AOT serialized ONNX model as a string;
  /// If false, dump the model as an ONNX model file in local;
  /// If true, return the model string to glowAOTSerializationModelStrPtr;
  /// This is for Glow AOT compilation
  bool returnGlowSerializedModelStr{false};

  /// Placeholder for the returned Glow AOT serialized ONNX model string
  std::shared_ptr<std::string> glowAOTSerializationModelStrPtr{nullptr};

  /// Whether to use Zip mode to serialize the DAG that has been optimized and
  /// partitioned.
  bool useZipModeForSerializeCompiledDAG{false};

  /// Whether to save constant data into the serialized DAG.
  bool saveConstantInSerializeCompiledDAG{false};

  /// Whether to call the DAG optimizer after the DAG is created in HostManager.
  bool callDAGOptimizer{false};

  /// Whether to use AOT mode for DAG optimizer.
  bool useDAGOptimizerAOTMode{false};

  /// Whether we're loading a model that has been AOT optimized.
  bool loadingAOTModel{false};

  /// Whether to skip provisioning, e.g. if we're doing AOT optimization.
  bool skipProvisioning{false};

  /// Static placeholder type info used for AOT optimization.
  std::map<std::string, Type> staticPlaceholderTypesForAOT;

  /// Map from function name to its corresponding compiled serialized functions;
  /// Used in deserialization.
  std::unordered_map<std::string, std::shared_ptr<std::vector<char>>>
      nameToFunctions;

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

    RETURN_ERR_IF_NOT(!(optimizationOpts.foldElemKindConversionIntoIO &&
                        optimizationOpts.delayAndRecordConstantModification),
                      "Cannot currently perform elem kind merging into PHs "
                      "when also preventing constant modification.");

    RETURN_ERR_IF_NOT(
        !(serializeCompiledDAG && skipProvisioning &&
          !optimizationOpts.delayAndRecordConstantModification &&
          !saveConstantInSerializeCompiledDAG),
        "When serializing the compiled DAG while skipping provisioning, C2 "
        "must also enable delayAndRecordConstantModification. PyTorch does not "
        "enable delayAndRecordConstantModification in this case, but "
        "saveConstantInSerializeCompiledDAG should be enabled");

    RETURN_ERR_IF_NOT(
        !precisionConfig.loadUniquedDummyQParams ||
            precisionConfig.originNameToTQPMap,
        "If loading unique dummy QParams, must have valid originNameToTQPMap");

    RETURN_ERR_IF_NOT(!precisionConfig.clipQuantRangeToFP16 ||
                          precisionConfig.convertToFP16,
                      "Assuming quant ranges are clipped to fp16 should only "
                      "be done along with fp16 conversion.");

    return Error::success();
  }
};

using CompilationMode = CompilationContext::CompilationMode;

}; // namespace glow

#endif // GLOW_OPTIMIZER_GRAPHOPTIMIZER_COMPILATIONCONTEXT_H
