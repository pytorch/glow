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

#ifndef GLOW_TORCH_GLOW_SRC_COMMON_H
#define GLOW_TORCH_GLOW_SRC_COMMON_H

#include "ShapeInferenceEngine.h"
#include "glow/Base/Tensor.h"
#include "glow/Base/Type.h"
#include "glow/Importer/CommonOperatorLoader.h"
#include "glow/Runtime/HostManager/HostManager.h"

#include <torch/csrc/jit/ir/ir.h>
#include <unordered_map>

DECLARE_bool(dumpFinalGlowGraph);

namespace glow {

struct InputMetaStack;

using BatchShapesMapType = std::unordered_map<
    int, std::unordered_map<const torch::jit::Value *, VariableMeta>>;

void dumpOperatorStats(const torch::jit::Graph &graph);

/// Various settings to be used by code that loads PyTorch models. There should
/// only be one of these and it should be obtained by calling
/// getPyTorchLoaderSettings().
struct PyTorchLoaderSettings {
private:
  void initSettings();

public:
  PyTorchLoaderSettings();
  ~PyTorchLoaderSettings() {}

  std::string toString() const;

  /// Whether or not run the custom pass that fuses jit nodes into a glow node.
  bool fusionPassEnabled = false;

  /// Dump Glow dot graph to file after model loading is finished.
  bool dumpGlowDag = false;

  /// A list of symbols for nodes that will be ignored by the Glow fuser and
  /// thus will not be fused to Glow.
  std::unordered_set<torch::jit::Symbol> opBlocklist;

  /// The minimum size of a glow fusion groups in terms of number of PyTorch
  /// nodes. 0 indicates no minimum size.
  size_t minFusionGroupSize = 0;

  /// The maximum total number of nodes which are allowed to merge when
  /// fusing groups. The resulting group may be larger than this limit
  /// however as additional nodes may be inserted during the merge.
  size_t maxFusionMergeSize = 0;

  /// Index (inclusive) of the first node in the JIT graph to fuse. Ignored if
  /// negative.
  /// NOTE: this should only be used for debugging.
  int64_t fusionStartIndex = -1;

  /// Index (exclusive) of the last node in the JIT graph to fuse. Ignored if
  /// negative.
  /// NOTE: this should only be used for debugging.
  int64_t fusionEndIndex = -1;

  /// Convert fp32 opts to fp16 ops during Glow compilation.
  bool convertToFP16 = false;

  bool skipBiasFp32tofp16Convert = false;

  /// Convert fp32 fused opts to fp16 ops during Glow compilation.
  bool convertFusedToFP16 = false;

  /// Print all JIT node indexes for debugging use.
  bool printJITIndex = false;

  /// Ignore the rounding argument to aten::div
  /// TODO: Handle this case with FloorDiv
  bool ignoreDivRoundingArgs = false;

  /// Add clip operators after each fp16 ops during Glow compilation.
  bool clipFP16 = false;

  /// Force glow to skip clipping fp16 Node inputs to min/max
  bool clipFP16SkipInputs = true;

  /// Enable fp16 conversion for Placeholders
  bool convertPlaceholdersToFP16 = true;

  /// Enable fp16 conversion for Constants
  bool convertConstantsToFP16 = true;

  /// Force all SLS/SLWS ops to use FP16 accumulation.
  bool forceFP16AccumSLS = true;

  /// Dump Glow dot graph to file after Glow compilation is finished.
  bool dumpFinalGlowGraph = false;

  /// Enable tracing inside of Glow.
  bool enableGlowTracing = false;

  /// Enable the auto removal of muation in JIT graph, i.e, inline ops.
  bool enableRemoveMutation = true;

  /// Enable the dumping of compiled serialized model for backend.
  bool enableSerialize = false;

  /// Enable the extracting already serialized model to backend.
  bool enableDeserialize = false;

  /// Disable all tensor layout verifying for all nodes.
  bool disableLayoutVerifying = true;

  /// Dump statistics about the operators and fusion support in the graph.
  bool dumpOperatorInventory = false;

  /// Number of traces per json trace file dump.
  size_t numTracesPerDump = 1;

  /// Replication count of a graph on a device.
  size_t replicationCount = 1;

  /// Backend-specific options to be put into the CompilationContext and passed
  /// to the Glow backend.
  std::map<std::string, std::string> backendSpecificOpts;

  /// Whether or not to write the loaded Glow function and inputs and outputs to
  /// and from the function to file as ONNX graphs.
  bool writeToOnnx = false;

  /// Whether or not to use zip mode when writing graphs to ONNX files
  bool onnxZipMode = false;

  /// Whether or not to write onnx files to /tmp/ directory.
  /// Must be true in twshared hosts.
  bool writeOnnxToTmp = false;

  /// The JSON file name that stores Glow deserialization specs
  std::string serializationSpecFileName = "serializationSpec.json";

  /// Optional prefix for naming of onnx files (otherwise an internal id)
  std::string onnxFileNamePrefix = "";

  /// Whether or not to do a numerical comparions of Glow and jit outputs
  bool jitVsGlowCompare = false;

  /// Name of a YAML file containing backend specific options.
  std::string backendOptionsFile;

  /// Whether not to set the saturateHost flag (use all available device) when
  /// adding networks to HostManager.
  bool saturateHost = false;

  /// If saturateHost is enabled and saturateKDevices is greater than zero,
  /// this is the number of devices that will be saturated.
  /// If saturateKDevices is zero and saturateHost is enabled, all available
  /// devices will be saturated.
  unsigned saturateKDevices{0};

  /// If true then randomize the Constants in the Function loaded by
  /// PyTorchModelLoader.
  bool randomizeConstants = false;

  /// If true then writing to Onnx without randomizing the constants is allowed.
  /// Otherwise, program will be abort if trying to write to Onnx without
  /// randomizing the constants.
  bool writeWithoutRandomize = false;

  /// Name of the Glow backend to use.
  std::string backendName = "Interpreter";

  /// Number of Glow devices to use.
  int32_t numDevices = -1;

  /// Whether to scan devices to get available ones in torch backend
  bool scanDevices = false;

  // Whether to run shape inference of meta input
  bool runShapeInference = false;

  /// Run Fusion flow within to_glow compile function
  bool enableDebugFuser = false;

  /// Whether to enforce module conversion to set include_last_offset for all
  /// embedding-bag-like operators. This is default to true since it is
  /// currently a requirement if we want to support partial inputs
  bool setIncludeLastOffsets = true;

  /// Call Glow's Function verifier after loading each JIT node to catch any
  /// Glow graph errors as soon as possible during loading. This is disabled by
  /// default because it can slow down model loading.
  bool debugContinuouslyVerifyDuringModelLoading = false;

  /// Index of input to extract batch size
  /// NOTE: this should only be used for development testing.
  int32_t nominalBatchIdx = -1;

  /// Indices of available backend devices on the machine.
  std::vector<int32_t> availableDevices;

  /// Whether to dump out failed inputs and reference outputs to onnx files.
  bool dumpFailedInputsToOnnxFiles = false;

  /// Whether to defer compilation to runtime
  /// NOTE: this might cause real request timeout, should only be used for
  ///       development testing.
  bool lazyCompile = false;

  /// Whether to enable device tracing from HostManger. NOTE: this must be set
  /// before network compilation.
  bool enableDeviceTracing = false;

  /// Whethere to enable DAG optimizer
  bool use_dag_optimizer = false;
  /// Additional parameters to DAG optimizer
  /// Keep algorithm default empty as only some of the algorithms may be
  /// required.
  std::string apl_parallelization_alg = "";
  std::string apl_placement_alg = "";
  int32_t apl_num_parallel_chunks = 2;

  /// Whether to use max size compilation.
  bool useMaxSizeCompilation = false;

  // Serialize GlowIR into ONNX txt file during warmCache, this file can be
  // use for future model loading, which a part of AOT compilation
  bool saveGlowIRIntoONNX = false;

  // Load GlowIR by deserializing ONNX txt file during warmCache
  bool loadGlowIRFromONNX = false;

  // Skip provisioning (currently only happens in Glow serialization, in which
  // we do model partition, DAG optimization, and then serialize the optimized
  // and partitioned model into ONNX model file. During this process, we do not
  // need to do provisioning)
  bool skipProvisioning = false;

  /// Set the number of predecessor Nodes to be printed from an error node.
  int32_t debugLayers = 5;

  // Sink tanh below concat
  bool sinkTanhBelowConcat = false;

  // Sparse NN Partitioning Scheme Constants, refer to OptimizationOptions in
  // CompilationContext for details
  bool useSparseNNPartitioningScheme = false;
  bool sparseNNPartitioningAddSLSConcats = false;
  bool sparseNNPartitioningBalancePerfModel = false;
  bool sparseNNPartitioningPairLNWithSLS = false;
  bool sparseNNPartitioningPairTileWithSLS = false;
  std::string sparseNNPartitioningPairSLSWith = "";
  int32_t sparseNNPartitioningConcatSplitSize = 1;
  int32_t sparseNNPartitioningSchemeNumCards = 1;
  int64_t sparseNNPartitioningSchemeSLSTableKBytesPerCard = 1;
  int32_t SparseNNPartitioningSchemeNumCoresSLS = 1;
  int32_t SparseNNPartitioningSchemeNumCoresOther = 1;

  // Enables Peer to Peer Tensor optimization
  bool enableP2P = false;

  // Enables Device Resident Tensor optimization
  bool enableDRT = false;

  // Enable conversion of fp16 scale and bias of embedding tables to fp32.
  bool convert8BitFusedToFP32 = false;

  // Enable conversion of fp16 scale and bias of embedding tables to fp32.
  bool convert4BitFusedToFP32 = false;
};

struct ModelCompilationConfigOverride {
  c10::optional<bool> useDagOptimizer;
  c10::optional<int32_t> aplNumParallelChunks;
  c10::optional<bool> aplAsapPlacement;
};

/// Represents different possible output types from to_glow modules.
enum class GraphOutputType {
  TENSORS,      // Single tensor or multiple tensors
  TENSOR_TUPLE, // Single tuple of tensors
  TENSOR_LIST,  // Single list of tensors
};

using PostFusionProcessFn =
    std::function<void(std::shared_ptr<torch::jit::Graph> graph)>;

/// Given a PyTorch ScalarType \p ty, \returns a matching Glow ElemKind.
ElemKind scalarTypeToElemKind(c10::ScalarType ty);

// Given a Glow ElemKind \p ty, \returns a matching PyTorch ScalarType.
c10::ScalarType elemKindToScalarType(glow::ElemKind ty);

/// Given a c10 typekind \p ty, \returns a matching Glow ElemKind.
ElemKind typeKindToElemKind(c10::TypeKind ty);

/// Get a snapshot of the current global PyTorchLoaderSettings singleton
PyTorchLoaderSettings getGlobalPyTorchLoaderSettingsSnapshot();

/// Get a mutable reference to the current global PyTorchLoaderSettings
/// singleton, this should almost never be used outside of binding.cpp.
PyTorchLoaderSettings &getGlobalPyTorchLoaderSettingsMutable();

/// \returns the HostManager singleton used to run all PyTorch graphs with for
/// the Glow backend specified by \p settings. The HostManager will have the
/// number of devices specified by settings. If a previous HostManager is
/// actively being used with the same backend but a different number of devices
/// then this is an error. If the specified number of devices is -1 then the
/// active HostManager for the given backend will be returned, if no active
/// HostManager is found then a HostManager with 1 device will be returned.
std::shared_ptr<runtime::HostManager>
getHostManager(const PyTorchLoaderSettings &settings);

/// \returns the PyTorch symbol to be used for the PyTorch node which represents
/// the subgraph that Glow will compile and run. \p g is the PyTorch graph to
/// lower, and if specified, will be used to generate unique symbol
c10::Symbol getGlowSymbol(std::shared_ptr<torch::jit::Graph> g = nullptr);

/// Given a PyTorch TensorType \p ptType, \returns a matching Glow Type.
glow::Type ptTypeToGlowType(const c10::TensorType &ptType);

/// Given a PyTorch Tensor \p ptTensor and a PyTorch scalar type \p dtype,
/// returns a new tensor which is \p ptTensor converted to \p dtype.
at::Tensor convertQuantizedToDtype(const at::Tensor ptTensor,
                                   c10::ScalarType dtype);

/// Given a PyTorch Tensor \p ptTensor, \returns an unowned Glow Tensor with a
/// matching type backed by the same memory as ptTensor.
glow::Tensor ptTensorToGlowTensor(const at::Tensor &ptTensor);

/// Given a Glow Type \p glowType, \returns an empty PyTorch Tensor with a
/// matching type.
at::Tensor glowTypeToEmptyPTTensor(const glow::Type &glowType);

/// Lower a pytorch \p module to glow before execution. \p inputMetaStr is the
/// raw string containing the meta data of the glow fuser node input.
/// \p glowAOTSerializationSpecStrPtr and \p glowAOTSerializationModelStrPtr are
/// used in offline Glow AOT compilation (i.e., Glow serialization), while
/// \p serializationSpec and \p onnxModelFile are used for online serving (i.e.,
/// Glow deserialization)
void glowAOTFusion(
    torch::jit::Module &module, const std::string &inputMetaStr,
    runtime::DeferredWeightLoader *loader,
    const PyTorchLoaderSettings &settings, std::string method_name = "forward",
    const std::unordered_map<int, std::string> &batchShapes = {},
    std::shared_ptr<std::string> glowAOTSerializationSpecStrPtr = nullptr,
    std::shared_ptr<std::string> glowAOTSerializationModelStrPtr = nullptr,
    const std::string &serializationSpec = "",
    const std::string &onnxModelFile = "",
    c10::optional<PostFusionProcessFn> postFusionProcessFn = {},
    const c10::optional<ModelCompilationConfigOverride>
        &modelCompilationConfigOverride = c10::nullopt);

/// Lower a pytorch \p module to glow before execution. \p inputMeta is a
/// vector containing the meta data of the model inputs.
/// \p glowAOTSerializationSpecStrPtr and \p glowAOTSerializationModelStrPtr are
/// used in offline Glow AOT compilation (i.e., Glow serialization), while
/// \p serializationSpec and \p onnxModelFile are used for online serving (i.e.,
/// Glow deserialization)
void glowAOTFusionWithShapeInference(
    torch::jit::Module &module, const glow::InputMetaStack &metaStack,
    runtime::DeferredWeightLoader *loader,
    const PyTorchLoaderSettings &settings, std::string method_name = "forward",
    const std::unordered_map<int, std::string> &batchShapes = {},
    std::shared_ptr<std::string> glowAOTSerializationSpecStrPtr = nullptr,
    std::shared_ptr<std::string> glowAOTSerializationModelStrPtr = nullptr,
    const std::string &serializationSpec = "",
    const std::string &onnxModelFile = "",
    c10::optional<PostFusionProcessFn> postFusionProcessFn = {},
    const c10::optional<ModelCompilationConfigOverride>
        &modelCompilationConfigOverride = c10::nullopt);

/// Enable overriding signal handlers while exeucting torch_glow code. This
/// should only be used in Python to enable easier debugging and not in
/// production C++ multithreaded environments. \p enable is used to enable or
/// disable overriding if set to false.
void enableSignalHandlerOverrides(bool enable = true);

/// \returns whether or not signal handler overriding is enabled.
bool signalHandlerOverridesEnabled();

Expected<std::string> getTempFileLoc(const std::string &name,
                                     const std::string &suffix);

BatchShapesMapType parseBatchShapeMapFromInputMeta(
    const std::shared_ptr<struct torch::jit::Graph> &graph,
    const std::unordered_map<int, std::string> &batchShapes,
    const std::string &baseSymbol);

} // namespace glow
#endif // GLOW_TORCH_GLOW_SRC_COMMON_H
