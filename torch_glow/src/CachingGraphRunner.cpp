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

#include "CachingGraphRunner.h"

#include "ShapeInferenceEngine.h"

#include "glow/Base/Type.h"
#include "glow/Exporter/ONNXModelWriter.h"
#include "glow/Flags/Flags.h"
#include "glow/Importer/ONNXModelLoader.h"
#include "glow/Runtime/DeferredWeightLoader.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "glow/Runtime/TraceExporter.h"
#include "glow/Support/Support.h"

#include <mutex>

namespace glow {

namespace {

/// Initialize the Glow compilation context \p cctx from glow::flags
/// This is backward compatible with existing PyTorchLoaderSettings, who
/// will overwrite any overlapping settings in this function.
Error initializeCompilationContextFromGlowFlags(
    glow::CompilationContext &cctx) {
  auto &precConfig = cctx.precisionConfig;
  if (glow::flags::ConvertToFP16) {
    precConfig.convertToFP16 = glow::flags::ConvertToFP16;
    LOG(INFO) << "Conversion to fp16 enabled";
  }
  if (glow::flags::SkipBiasFp32tofp16Convert) {
    precConfig.skipBiasFp32tofp16Convert =
        glow::flags::SkipBiasFp32tofp16Convert;
    LOG(INFO) << "Skip fp16 convert for bias";
  }
  if (glow::flags::ConvertPlaceholdersToFP16) {
    precConfig.convertPlaceholdersToFP16 =
        glow::flags::ConvertPlaceholdersToFP16;
    LOG(INFO) << "Conversion of Placeholders to fp16 enabled";
  }
  if (glow::flags::ConvertConstantsToFP16) {
    precConfig.convertConstantsToFP16 = glow::flags::ConvertConstantsToFP16;
    LOG(INFO) << "Conversion of Constants to fp16 enabled";
  }
  if (glow::flags::ConvertFusedScaleOffsetToFP16) {
    precConfig.convertFusedToFP16 = glow::flags::ConvertFusedScaleOffsetToFP16;
    LOG(INFO) << "Conversion of fused scales/offsets to fp16 enabled";
  }
  if (glow::flags::ClipToFP16) {
    precConfig.clipFP16 = glow::flags::ClipToFP16;
    LOG(INFO) << "Clipping to fp16 enabled";
  }
  if (glow::flags::SkipInputsOnClipToFP16) {
    precConfig.clipFP16SkipInputs = glow::flags::SkipInputsOnClipToFP16;
    LOG(INFO) << "Skipping clipping for fp16 Node inputs fp16";
  }
  if (glow::flags::ForceSLSToFP16Accum) {
    precConfig.forceFP16AccumSLS = glow::flags::ForceSLSToFP16Accum;
    LOG(INFO) << "Forcing all SLS/SLWS ops to use FP16 accumulation enabled";
  }
  if (!glow::flags::EnableQuantParamChanges) {
    cctx.optimizationOpts.enableQuantParamChanges = false;
    LOG(INFO) << "Disabling quantization param changes during optimizations";
  }
  if (glow::flags::DumpCompilationLog) {
    cctx.compilationLogPrefix = "torch-glow";
  }

  // glow_sparsenn_partitioning_add_sls_concats
  // (SparseNNPartitioningAddSLSConcats) enables addition of concats to create a
  // bigger tensor out of many smaller tensors that needs to be communicated
  // with other partitions so that communication is efficient. This doesn't work
  // if one of the tensor is [1, x] (coming from user embeddings and it's [1, x]
  // due to inbatch broadcast, i.e., broadcast happens on accelerator card) and
  // other (coming from ad embeddings) is [32, y]. However, [1, x] is followed
  // by tile in glow graph to make it [32, x]. This diff D27781184 (
  // glow_sparsenn_partitioning_pair_tile_with_sls, i.e.,
  // SparseNNPartitioningPairTileWithSLS) pulls in the tile operator on user
  // embeddings to sls partition as well so that sls_concat can now work since
  // post-tile tensors become [32, x] and [32, y]. Thus,
  // SparseNNPartitioningPairTileWithSLS is required for
  // SparseNNPartitioningAddSLSConcats to work.
  if (glow::flags::SparseNNPartitioningAddSLSConcats) {
    LOG(INFO)
        << "Enabling glow_sparsenn_partitioning_pair_tile_with_sls because "
           "glow_sparsenn_partitioning_add_sls_concats is enabled ";
    cctx.optimizationOpts.sparseNNPartitioningPairTileWithSLS = true;
  }

  if (glow::flags::UseDAGOptimizer) {
    LOG(INFO) << "Enabling DAG optimizer and related options (server AOT)";
    cctx.callDAGOptimizer = true;
    cctx.optimizationOpts.DAGOptimizerNumParallelChunks =
        glow::flags::DAGOptimizerNumParallelChunks;
    cctx.optimizationOpts.DAGOptimizerParallelizationTaggingAlgorithm =
        glow::flags::DAGOptimizerParallelizationTaggingAlgorithm;
    cctx.optimizationOpts.DAGOptimizerPlacementTaggingAlgorithm =
        glow::flags::DAGOptimizerPlacementTaggingAlgorithm;
  }

  if (glow::flags::UseSparseNNPartitioningScheme) {

    cctx.optimizationOpts.useSparseNNPartitioningScheme = true;
    cctx.optimizationOpts.sparseNNPartitioningAddSLSConcats =
        glow::flags::SparseNNPartitioningAddSLSConcats;
    cctx.optimizationOpts.sparseNNPartitioningBalancePerfModel =
        glow::flags::SparseNNPartitioningBalancePerfModel;
    cctx.optimizationOpts.sparseNNPartitioningPairLNWithSLS =
        glow::flags::SparseNNPartitioningPairLNWithSLS;
    cctx.optimizationOpts.sparseNNPartitioningPairTileWithSLS =
        glow::flags::SparseNNPartitioningPairTileWithSLS;
    cctx.optimizationOpts.sparseNNPartitioningPairSLSWith =
        glow::flags::SparseNNPartitioningPairSLSWith;
    cctx.optimizationOpts.sparseNNPartitioningConcatSplitSize =
        glow::flags::SparseNNPartitioningConcatSplitSize;
    cctx.optimizationOpts.sparseNNPartitioningSchemeNumCards =
        glow::flags::SparseNNPartitioningSchemeNumCards;
    cctx.optimizationOpts.sparseNNPartitioningSchemeSLSTableKBytesPerCard =
        glow::flags::SparseNNPartitioningSchemeSLSTableKBytesPerCard;
    cctx.optimizationOpts.sparseNNPartitioningSchemeNumCoresSLS =
        glow::flags::SparseNNPartitioningSchemeNumCoresSLS;
    cctx.optimizationOpts.sparseNNPartitioningSchemeNumCoresOther =
        glow::flags::SparseNNPartitioningSchemeNumCoresOther;
    LOG(INFO) << "Using SLS partitioning scheme";
  }
  cctx.saturateHost = glow::flags::SaturateHost;

  if (!glow::flags::processBackendSpecificOpts(
          cctx.backendOpts.backendSpecificOpts,
          glow::flags::BackendSpecificOpts)) {
    MAKE_ERR("Failed glow::flags::processBackendSpecificOpts");
  }

  if (glow::runtime::flags::EnableP2P) {
    LOG(INFO) << "Glow P2P Enabled";
    cctx.enableP2P = true;
  }
  if (glow::runtime::flags::EnableDRT) {
    LOG(INFO) << "Glow DRT Enabled";
    cctx.enableDRT = true;
  }
  return Error::success();
}

/// Initialize the Glow compilation context \p cctx with \p settings
void initializeCompilationContextFromSettings(
    glow::CompilationContext &cctx, const PyTorchLoaderSettings &settings) {
  if (!cctx.precisionConfig.convertToFP16 && settings.convertToFP16) {
    cctx.precisionConfig.convertToFP16 = settings.convertToFP16;
    LOG(INFO) << "Conversion to fp16 enabled";
  }
  if (!cctx.precisionConfig.skipBiasFp32tofp16Convert &&
      settings.skipBiasFp32tofp16Convert) {
    cctx.precisionConfig.skipBiasFp32tofp16Convert =
        settings.skipBiasFp32tofp16Convert;
    LOG(INFO) << "Skipping bias fp32 -> fp16 conversion enabled";
  }
  if (!cctx.precisionConfig.convertPlaceholdersToFP16 &&
      settings.convertPlaceholdersToFP16) {
    cctx.precisionConfig.convertPlaceholdersToFP16 =
        settings.convertFusedToFP16;
    LOG(INFO) << "Conversion of Placeholders to fp16 enabled";
  }
  if (!cctx.precisionConfig.convertConstantsToFP16 &&
      settings.convertConstantsToFP16) {
    cctx.precisionConfig.convertConstantsToFP16 =
        settings.convertConstantsToFP16;
    LOG(INFO) << "Conversion of Constants to fp16 enabled";
  }
  if (!cctx.precisionConfig.convertFusedToFP16 && settings.convertFusedToFP16) {
    cctx.precisionConfig.convertFusedToFP16 = settings.convertFusedToFP16;
    LOG(INFO) << "Conversion of fused scales/offsets to fp16 enabled";
  }
  if (!cctx.precisionConfig.clipFP16 && settings.clipFP16) {
    cctx.precisionConfig.clipFP16 = settings.clipFP16;
    LOG(INFO) << "Clipping to fp16 enabled";
  }
  if (!cctx.precisionConfig.clipFP16SkipInputs && settings.clipFP16SkipInputs) {
    cctx.precisionConfig.clipFP16SkipInputs = settings.clipFP16SkipInputs;
    LOG(INFO) << "Skipping clipping for fp16 Node inputs fp16";
  }
  if (!cctx.precisionConfig.forceFP16AccumSLS && settings.forceFP16AccumSLS) {
    cctx.precisionConfig.forceFP16AccumSLS = settings.forceFP16AccumSLS;
    LOG(INFO) << "Forcing all SLS/SLWS ops to use FP16 accumulation enabled";
  }
  if (!cctx.precisionConfig.convert8BitFusedToFP32 &&
      settings.convert8BitFusedToFP32) {
    LOG(INFO) << "Enabling conversion of FP16 scale and bias to FP32 for 8bit "
                 "EmbeddingBagByteRowwiseOffset";
    cctx.precisionConfig.convert8BitFusedToFP32 =
        settings.convert8BitFusedToFP32;
  }
  if (!cctx.precisionConfig.convert4BitFusedToFP32 &&
      settings.convert4BitFusedToFP32) {
    LOG(INFO) << "Enabling conversion of FP16 scale and bias to FP32 for 4bit "
                 "EmbeddingBagByteRowwiseOffset";
    cctx.precisionConfig.convert4BitFusedToFP32 =
        settings.convert4BitFusedToFP32;
  }
  if (!glow::flags::DisableLayoutVerifying && settings.disableLayoutVerifying) {
    glow::flags::DisableLayoutVerifying = true;
    LOG(INFO) << "Skipping all layout verifying";
  }

  // If we want to enable serialize, we have to not free compiled stream in
  // provisoner.
  if (settings.enableSerialize) {
    glow::flags::DisableFreeCompilationResource = true;
    LOG(INFO)
        << "Free compilation resource after compiling on backend is disabled";
  }

  if (settings.dumpFinalGlowGraph) {
    cctx.dumpFinalGraph = settings.dumpFinalGlowGraph;
  }
  if (settings.saturateHost) {
    cctx.saturateHost = settings.saturateHost;
  }

  if (settings.saturateKDevices > 0) {
    cctx.saturateKDevices = settings.saturateKDevices;
  }

  if (settings.use_dag_optimizer) {
    cctx.callDAGOptimizer = true;
    if (!settings.apl_placement_alg.empty()) {
      cctx.optimizationOpts.DAGOptimizerPlacementTaggingAlgorithm =
          settings.apl_placement_alg;
    }
    if (!settings.apl_parallelization_alg.empty()) {
      cctx.optimizationOpts.DAGOptimizerParallelizationTaggingAlgorithm =
          settings.apl_parallelization_alg;
      cctx.optimizationOpts.DAGOptimizerNumParallelChunks =
          settings.apl_num_parallel_chunks;
    }
  }

  if (!settings.backendSpecificOpts.empty()) {
    cctx.backendOpts.backendSpecificOpts = settings.backendSpecificOpts;
  }

  cctx.replicationCount = settings.replicationCount;

  if (settings.skipProvisioning) {
    LOG(INFO) << "Will skip provisioning (likely due to AOT opt).";
    cctx.skipProvisioning = true;
  }

  if (settings.sinkTanhBelowConcat) {
    LOG(INFO) << "Sinking tanh below concat";
    cctx.optimizationOpts.sinkTanhBelowConcat = true;
  }

  if (settings.useSparseNNPartitioningScheme) {
    cctx.optimizationOpts.useSparseNNPartitioningScheme = true;
    cctx.optimizationOpts.sparseNNPartitioningAddSLSConcats =
        settings.sparseNNPartitioningAddSLSConcats;
    cctx.optimizationOpts.sparseNNPartitioningBalancePerfModel =
        settings.sparseNNPartitioningBalancePerfModel;
    cctx.optimizationOpts.sparseNNPartitioningPairLNWithSLS =
        settings.sparseNNPartitioningPairLNWithSLS;
    cctx.optimizationOpts.sparseNNPartitioningPairTileWithSLS =
        settings.sparseNNPartitioningPairTileWithSLS;
    cctx.optimizationOpts.sparseNNPartitioningPairSLSWith =
        settings.sparseNNPartitioningPairSLSWith;
    cctx.optimizationOpts.sparseNNPartitioningSchemeNumCards =
        settings.sparseNNPartitioningSchemeNumCards;
    cctx.optimizationOpts.sparseNNPartitioningSchemeSLSTableKBytesPerCard =
        settings.sparseNNPartitioningSchemeSLSTableKBytesPerCard;
    cctx.optimizationOpts.sparseNNPartitioningSchemeNumCoresSLS =
        settings.SparseNNPartitioningSchemeNumCoresSLS;
    cctx.optimizationOpts.sparseNNPartitioningSchemeNumCoresOther =
        settings.SparseNNPartitioningSchemeNumCoresOther;
  }

  if (settings.enableP2P) {
    LOG(INFO) << "Glow P2P Enabled";
    cctx.enableP2P = true;
  }

  if (settings.enableDRT) {
    LOG(INFO) << "Glow DRT Enabled";
    cctx.enableDRT = true;
  }
}

/// Initialize the Glow compilation context \p cctx with \p
/// ModelCompilationConfigOverride configOverride
void initializeCompilationContextFromModelCompilationConfigOverride(
    glow::CompilationContext &cctx,
    const glow::ModelCompilationConfigOverride &configOverride) {
  if (configOverride.useDagOptimizer.has_value()) {
    cctx.callDAGOptimizer = configOverride.useDagOptimizer.value();
  }

  if (configOverride.aplNumParallelChunks.has_value()) {
    cctx.optimizationOpts.DAGOptimizerNumParallelChunks =
        configOverride.aplNumParallelChunks.value();
  }

  if (configOverride.aplAsapPlacement.has_value()) {
    cctx.optimizationOpts.enableAPLASAPPlacement =
        configOverride.aplAsapPlacement.value();
  }
}

/// This function slice the input Tensor according to the expected shape in the
/// zero dimension.
/// TODO: Multi-dimension slicing will be supported later.
at::Tensor sliceTensor(at::Tensor &t, const TensorShape &shape) {
  CHECK_GT(shape.size(), 0);
  return at::native::slice(t, 0, 0, shape[0]);
}

/// The following two methods account for the auto FP32->FP16 conversion in Glow
/// for placeholder \p type match in AOT
ElemKind getConvertElemTypeForAOT(const Type &type,
                                  const CompilationContext &cctx) {
  auto elementType = type.getElementType();
  if (cctx.precisionConfig.convertToFP16 && elementType == ElemKind::FloatTy) {
    elementType = ElemKind::Float16Ty;
  } else if (cctx.precisionConfig.convertFusedToFP16 &&
             elementType == ElemKind::UInt8FusedQTy) {
    elementType = ElemKind::UInt8FusedFP16QTy;
  }
  return elementType;
}

std::vector<unsigned long>
getConvertDimVecForAOT(const Type &type, const CompilationContext &cctx) {
  auto dims = type.dims();
  auto dimVec = dims.vec();
  if (cctx.precisionConfig.convertFusedToFP16 &&
      type.getElementType() == ElemKind::UInt8FusedQTy) {
    assert(dimVec.size() == 2);
    dimVec[1] = dimVec[1] - 4;
  }
  return dimVec;
}

/// This function is the preparation of Glow serialization. It sets \p
/// GlowDeserializationSpec for AOT model loading and sets cctx to let
/// HostManager serialize lowerred Glow IR into onnx file
Error setupGlowDeserializationSpecAndCctx(
    const PyTorchLoaderSettings &settings,
    const std::shared_ptr<CachingGraphRunner::PerGlowGraphInfo> &info,
    CompilationContext &cctx, Function *f, GlowDeserializationSpec &spec,
    std::shared_ptr<std::string> glowAOTSerializationModelStrPtr) {
  auto glowPyTorchLoaderSettings = spec.pytorchLoaderSettings;
  glowPyTorchLoaderSettings->overrideSettings(settings);

  spec.functionName = info->functionName;
  auto &inputPHNames = spec.inputPHNames;
  auto &inputPHTypes = spec.inputPHTypes;
  auto &staticPHNames = spec.staticPHNames;
  auto &staticPHTypes = spec.staticPHTypes;
  auto &outputPHNames = spec.outputPHNames;
  size_t inputIdx = 0;
  for (const auto &ph : info->inputPlaceholders) {
    inputPHNames.emplace_back(ph->getName().data());
    inputPHTypes.emplace_back(ph->getType()->toString());
    cctx.loadedPHNames.emplace(ph,
                               std::make_pair(ph->getName().data(), inputIdx));
    ++inputIdx;
  }
  std::map<std::string, glow::Type> staticPlaceholderTypes;
  for (const auto &ph : f->findPlaceholders()) {
    if (ph->isStatic()) {
      auto type = *ph->getType();
      /// Account for the auto FP32->FP16 conversion in Glow
      /// for placeholder \p type match
      auto convertedElemType = getConvertElemTypeForAOT(type, cctx);
      auto convertedDimVec = getConvertDimVecForAOT(type, cctx);
      auto convertedDims = llvm::ArrayRef<unsigned long>(convertedDimVec);
      auto convertedType =
          type.isQuantizedType()
              ? f->getParent()->uniqueType(convertedElemType, convertedDims,
                                           type.getScale(), type.getOffset())
              : f->getParent()->uniqueType(convertedElemType, convertedDims);
      // Here staticPlaceholderTypes is used for serializing Glow IR in
      // hostManager, which is post-precision conversion. staticPHTypes on the
      // other hand is used in Glow deserialization, which requires the input
      // tensor types (i.e., pre-precision conversion)
      staticPlaceholderTypes[std::string(ph->getName())] = *convertedType;
      staticPHNames.emplace_back(ph->getName().data());
      staticPHTypes.emplace_back(type.toString());
    }
  }
  size_t outputIdx = 0;
  for (const auto &ph : info->outputPlaceholders) {
    outputPHNames.emplace_back(ph->getName().data());
    cctx.loadedPHNames.emplace(ph,
                               std::make_pair(ph->getName().data(), outputIdx));
    ++outputIdx;
  }
  cctx.serializeCompiledDAG = true;
  cctx.saveConstantInSerializeCompiledDAG = true;
  cctx.staticPlaceholderTypesForAOT = staticPlaceholderTypes;
  cctx.returnGlowSerializedModelStr = true;
  cctx.glowAOTSerializationModelStrPtr = glowAOTSerializationModelStrPtr;
  // We currently save all the non-embedding weights in the ONNX file
  // and thus do not delay/record constant modification. Since AOT
  // compilation is performed for every training snapshot, we do not
  // need to support updating quantization params for AOT.
  RETURN_ERR_IF_NOT(
      !cctx.optimizationOpts.delayAndRecordConstantModification,
      "delayAndRecordConstantModification should be false when loading "
      "PyTorch models in Glow");

  return Error::success();
}

/// This function serialize Glow deserialization spec in JSON format
/// The JSON file contains
///     1. PyTorchLoaderSettings;
///     2. Glow function name;
///     3. Input placeholder names & types;
///     4. Static placeholder names & types;
///     5. Output placeholder names;
/// Remarks: (1) ONNXModelLoader initialization requires both input and
///          static PHs and types as the inputTensors and inputTypes;
///          (2) Glow deserialization will reset PH static status in
///           GlowIR, we need to set them static manually during
///           deserialization
///          (3) Input&Output PH names are used for reconstructing
///           PerGlowGraphInfo
Error saveGlowDeserializationSpec(
    GlowDeserializationSpec &spec,
    std::shared_ptr<std::string> glowAOTSerializationSpecStrPtr) {
  std::string serializedSpec;
  ASSIGN_VALUE_OR_RETURN_ERR(serializedSpec, spec.toJson());
  *glowAOTSerializationSpecStrPtr = std::move(serializedSpec);
  return Error::success();
}

glow::Expected<std::string> getOnnxFilePath(const std::string &filePrefix,
                                            bool writeOnnxToTmp,
                                            const char *extension = ".onnx") {
  if (writeOnnxToTmp) {
    std::string filepath;
    ASSIGN_VALUE_OR_RETURN_ERR(filepath, getTempFileLoc(filePrefix, extension));
    return filepath;
  } else {
    return filePrefix + extension;
  }
}

} // namespace

void CachingGraphRunner::aggregateAndDumpTraces(TraceContext *traceContext,
                                                bool flush) {
  size_t numTracesPerDump = defaultSettings_.numTracesPerDump;
  bool doDump = false;
  std::string filename;
  {
    std::unique_lock<std::mutex> lock(tracesMutex_);

    if (traceContext) {
      mergedTraceContext_->merge(traceContext);
      numTraces_++;
    } else if (mergedTraceContext_->getTraceEvents().empty()) {
      return;
    }
    size_t numTraces = numTraces_;

    // If numTracesPerDump <= 0, it means we don't merge unless there is a flush
    if (flush || (numTracesPerDump > 0 && numTraces % numTracesPerDump == 0)) {
      // Initial way of differentiating the dump files when there are multiple
      // graph runners
      // TODO(allwu): find a better way to generate trace file names
      size_t hash = reinterpret_cast<size_t>(this);
      size_t dumpNum = numTraceDumps_++;
      filename =
          strFormat("glow-trace-%04lx-%zu.json", hash % (1 << 16), dumpNum);
      doDump = true;
    }
  }
  if (doDump) {
    mergedTraceContext_->dump(filename);
    mergedTraceContext_ = glow::make_unique<TraceContext>(TraceLevel::STANDARD);
  }
}

std::unique_ptr<
    std::unordered_map<std::string, std::unique_ptr<BlockStreamBase>>>
CachingGraphRunner::getAllSerializedFunctionsMap() {
  return hostManager_->getAllSerializedFunctions();
}

Expected<std::shared_ptr<CachingGraphRunner::PerGlowGraphInfo>>
CachingGraphRunner::loadImpl(torch::jit::Stack &stack,
                             const PyTorchLoaderSettings &settings,
                             TraceContext *traceContext) {
  TRACE_EVENT_SCOPE(traceContext, TraceLevel::RUNTIME, "torch_glow::loadImpl");
  RECORD_USER_SCOPE("torch_glow::loadImpl");
  const auto inputs = torch::jit::last(stack, graph_->inputs().size());

  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME,
                    "InputMetaStack_creation");
  InputMetaStack metaStack;
  {
    RECORD_USER_SCOPE("InputMetaStack_creation");
    ASSIGN_VALUE_OR_RETURN_ERR(
        metaStack, inputMetaStackFromStack(stack, /*ignoreObjects*/ true));
  }
  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "InputMetaStack_creation");

  // If we already have a Glow function compiled for this graph with and the
  // given inputs then use that.
  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME,
                    "perGlowGraphInfoMap__lookup");
  std::unique_lock<std::shared_timed_mutex> wlock(graphInfoMapMutex);
  size_t hash = getGraphMapKeyFromInputStack(metaStack);
  {
    auto it = perGlowGraphInfoMap_.find(hash);
    if (it != perGlowGraphInfoMap_.end()) {
      return it->second;
    }
  }
  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME,
                  "perGlowGraphInfoMap__lookup");

  LOG(INFO) << "Compiling graph for inputs:" << std::endl << metaStack.print();

  PyTorchLoaderSettings loadSettings = settings;
  if (settings.lazyCompile) {
    auto it = pyTorchLoaderSettingsMap_.find(metaStack.hash());
    if (it != pyTorchLoaderSettingsMap_.end()) {
      LOG(INFO) << "Loading compilation settping for hash:" << metaStack.hash();
      loadSettings = it->second;
    }
  }

  auto info = std::make_shared<PerGlowGraphInfo>(
      strFormat("pt_function_%lu_%lu", size_t(this), metaStack.hash()),
      loadSettings);

  std::unique_ptr<Module> module = glow::make_unique<Module>();
  Function *f = module->createFunction(info->functionName);

  glow::CompilationContext cctx;
  RETURN_IF_ERR(initializeCompilationContextFromGlowFlags(cctx));
  initializeCompilationContextFromSettings(cctx, loadSettings);

  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "loadJITGraph");
  {
    RECORD_USER_SCOPE("loadJITGraph");
    RETURN_IF_ERR(PyTorchModelLoader::loadJITGraph(
        *f, *graph_, info->inputPlaceholders, info->outputPlaceholders,
        outputCorrectTypes_, loadSettings, inputs, {}));
  }
  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "loadJITGraph");

  info->inputSanitizers = runtime::getInputSanitizers(*f);

  if (loadSettings.convertToFP16) {
    cctx.precisionConfig.precisionModeKindSet.insert(
        Kinded::Kind::ChannelwiseQuantizedConvolutionNodeKind);
    cctx.precisionConfig.precisionModeKindSet.insert(
        Kinded::Kind::RowwiseQuantizedFullyConnectedNodeKind);
  }
  cctx.replicationCount = loadSettings.replicationCount;
  cctx.backendOpts.backendSpecificOpts = loadSettings.backendSpecificOpts;

  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "addNetwork");
  {
    RECORD_USER_SCOPE("addNetwork");
    // If --load-backend-specific-opts was passed from python, add it to the
    // compile context so the host manager knows to load backend options from
    // yaml.

    if (!loadSettings.backendOptionsFile.empty()) {
      std::pair<std::string, std::string> loadBackendSpecificOpts(
          "loadBackendSpecificOptions", loadSettings.backendOptionsFile);
      cctx.backendOpts.backendSpecificOpts.insert(loadBackendSpecificOpts);
    }

    RETURN_IF_ERR(hostManager_->addNetwork(std::move(module), cctx));
  }
  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "addNetwork");

  auto ret = perGlowGraphInfoMap_.emplace(hash, info);
  RETURN_ERR_IF_NOT(ret.second,
                    strFormat("Tried to store duplicate Glow graph for %s",
                              metaStack.print().c_str()));

  return info;
}

Expected<MetaStack *>
CachingGraphRunner::loadShape(const c10::ArrayRef<c10::IValue> &inputs,
                              TraceContext *traceContext) {

  TRACE_EVENT_SCOPE(traceContext, TraceLevel::RUNTIME, "torch_glow::loadShape");
  RECORD_USER_SCOPE("torch_glow::loadShape");
  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME,
                    "computeShapeInputMetaStack");
  InputMetaStack metaStack;
  {
    RECORD_USER_SCOPE("computeShapeInputMetaStack");
    ASSIGN_VALUE_OR_RETURN_ERR(metaStack,
                               inputMetaStackFromStack(inputs, true));
  }
  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME,
                  "computeShapeInputMetaStack");

  // If we already have a shape info for this graph output with and the
  // given inputs then use that.
  size_t hash = getGraphMapKeyFromInputStack(metaStack);
  {
    std::lock_guard<std::mutex> graphShapeLock(glowGraphShapeMapMutex_);
    auto it = perGlowGraphShapeMap_.find(hash);
    if (it != perGlowGraphShapeMap_.end()) {
      return &(it->second);
    }
  }

  VLOG(1) << "Compiling graph with tensor shape:\n" << metaStack.print();

  // If we don't have a shape info for this graph output with and the
  // given inputs then run shape inference, then push into the map.
  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "runShapeInference");
  MetaStack outputShape;
  {
    RECORD_USER_SCOPE("runShapeInference");

    ShapeInferenceEngine shapeG(*graph_, inputs);
    RETURN_IF_ERR(shapeG.run());
    outputShape = shapeG.getGraphOutputShape();
  }
  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "runShapeInference");

  {
    std::lock_guard<std::mutex> graphShapeLock(glowGraphShapeMapMutex_);
    auto ret = perGlowGraphShapeMap_.emplace(hash, outputShape);
    return &(ret.first->second);
  }
}

int64_t CachingGraphRunner::runOnJit(torch::jit::Stack &stack) {
  static std::mutex runJitLock;
  std::lock_guard<std::mutex> guard(runJitLock);
  bool temp = getGlobalPyTorchLoaderSettingsMutable().fusionPassEnabled;
  getGlobalPyTorchLoaderSettingsMutable().fusionPassEnabled = false;
  int64_t startTime;
  startTime = TraceEvent::now();
  ptGraphExecutor_.run(stack);
  int64_t runTime = TraceEvent::now() - startTime;
  getGlobalPyTorchLoaderSettingsMutable().fusionPassEnabled = temp;
  return runTime;
}

struct TensorCompareResult {
  double relErr;
  double maxErr;
  double maxRelErr;
};

template <typename Ty>
TensorCompareResult compareTensors(glow::Tensor &RefT, glow::Tensor &CmpT) {
  TensorCompareResult result = {INFINITY, INFINITY, INFINITY};
  if (CmpT.getHandle<Ty>().size() != RefT.getHandle<Ty>().size()) {
    LOG(ERROR) << "Dimension mismatch: "
               << "\tReference dims: " << RefT.getHandle().getType().dims()
               << "\tGlow dims: " << CmpT.getHandle().getType().dims()
               << std::endl;
    return result;
  }
  double totalErrSq = 0.0;
  double totalMagSq = 0.0;
  double maxErr = 0.0;
  double maxRelErr = 0.0;
  for (dim_t idx = 0; idx < RefT.getHandle().size(); idx++) {
    double refVal = (double)RefT.getHandle<Ty>().raw(idx);
    double cmpVal = (double)CmpT.getHandle<Ty>().raw(idx);
    double diff = refVal - cmpVal;
    double mag = refVal * refVal;
    double eltRelErr = (fabs(refVal)) > 0.0 ? fabs(diff) / fabs(refVal) : 0.0;
    totalErrSq += diff * diff;
    totalMagSq += mag;
    maxErr = (fabs(diff) > maxErr) ? fabs(diff) : maxErr;
    maxRelErr = (eltRelErr > maxRelErr) ? eltRelErr : maxRelErr;
  }
  result.relErr = (totalMagSq > 0.0) ? std::sqrt(totalErrSq / totalMagSq) : 0.0;
  result.maxErr = maxErr;
  result.maxRelErr = maxRelErr;
  return result;
}

/// Create an onnx graph for the tensors in \p glowTensors with names from \p
/// placeholders and write the graph to \p filePrefix
static Error
writeGlowTensorsToOnnx(const std::string &filePrefix,
                       const PyTorchLoaderSettings &settings,
                       const std::vector<glow::Placeholder *> &placeholders,
                       const std::vector<glow::Tensor> &glowTensors) {
  DCHECK_EQ(placeholders.size(), glowTensors.size());

  ONNX_NAMESPACE::GraphProto onnxGraph;
  for (size_t i = 0; i < placeholders.size(); ++i) {
    const auto *ph = placeholders[i];
    if (ph->getNumUsers() == 0) {
      LOG(INFO) << "Tensor onnxification not required. Not being used: "
                << ph->getName().str() << "\n";
      continue;
    }

    auto *onnxT = onnxGraph.add_initializer();
    const auto &t = glowTensors[i];
    onnxT->set_name(ph->getName().str());
    size_t unpaddedSize = t.getUnpaddedSizeInBytes();
    size_t tensorSize = t.getSizeInBytes();
    if (unpaddedSize == tensorSize) {
      ONNXModelWriter::writeTensor(t, onnxT,
                                   /*useGlowCustomOps*/ true);
    } else {
      // If the tensor is a partial tensor, then save only the part
      // that has data.
      auto ty = t.getType();
      auto dims = ty.dims().vec();
      DCHECK_GT(dims.size(), 0);
      dims[0] = dims[0] * unpaddedSize / tensorSize;
      const auto &resized = t.getUnowned(dims);
      ONNXModelWriter::writeTensor(resized, onnxT,
                                   /*useGlowCustomOps*/ true);
    }
  }

  std::string filename;
  ASSIGN_VALUE_OR_RETURN_ERR(
      filename, getOnnxFilePath(filePrefix, settings.writeOnnxToTmp));
  std::ofstream of(filename, std::ios::binary);
  if (!of) {
    return MAKE_ERR(
        strFormat("Cannot create onnx tensor file %s", filename.c_str()));
  }

  std::string buffer;
  onnxGraph.SerializeToString(&buffer);
  of << buffer;

  return Error::success();
}

/// Get outputs from \p stack which contains PyTorch tensors from running on JIT
/// GraphExector and create a onnx file for those outputs at \p filePrefix.
static Error
writeJITOutputsToOnnxFile(const std::string &filePrefix,
                          const torch::jit::Stack &stack,
                          const CachingGraphRunner::PerGlowGraphInfo &info) {
  // pull outputs off the stack, create corresponding vector of Glow tensors
  std::vector<glow::Tensor> glowTensorOutputs;
  std::vector<torch::Tensor> ptTensorOutputs;
  size_t numOutputs = info.outputPlaceholders.size();
  for (size_t i = 0; i < numOutputs; ++i) {
    auto &jitOutput = torch::jit::peek(stack, i, numOutputs);
    auto jitPtTensor = jitOutput.toTensor().contiguous();
    glow::Tensor jitGlowT = ptTensorToGlowTensor(jitPtTensor);
    glowTensorOutputs.push_back(std::move(jitGlowT));
    ptTensorOutputs.push_back(std::move(jitPtTensor));
  }

  // write outputs to file
  RETURN_IF_ERR(writeGlowTensorsToOnnx(
      filePrefix, info.settings, info.outputPlaceholders, glowTensorOutputs));

  return Error::success();
}

Error CachingGraphRunner::writeJitIOToOnnxFile(
    const std::string &inputFilePrefix, const std::string &outputFilePrefix,
    const torch::jit::Stack &stack) {

  if (!defaultSettings_.dumpFailedInputsToOnnxFiles) {
    return Error::success();
  }

  std::shared_ptr<PerGlowGraphInfo> info;
  ASSIGN_VALUE_OR_RETURN_ERR(info, findGraphInfoForStack(stack));

  // Write inputs
  size_t numInputs = graph_->inputs().size();
  const auto inputs = torch::jit::last(stack, numInputs);

  std::vector<glow::Tensor> glowTensorInputs;
  std::vector<torch::Tensor> ptTensorInputs;

  if (auto tensorsOrErr =
          processPyTorchInputs(inputs, info->inputPlaceholders)) {
    glowTensorInputs = std::move(tensorsOrErr->first);
    ptTensorInputs = std::move(tensorsOrErr->second);
  } else {
    RETURN_ERR(tensorsOrErr.takeError());
  }

  RETURN_IF_ERR(writeGlowTensorsToOnnx(inputFilePrefix, info->settings,
                                       info->inputPlaceholders,
                                       glowTensorInputs));

  // Write InputMetaStack to file so we know the type of the inputs
  InputMetaStack metaStack;
  ASSIGN_VALUE_OR_RETURN_ERR(metaStack, inputMetaStackFromStack(inputs));

  std::string metaStackFilename;
  ASSIGN_VALUE_OR_RETURN_ERR(
      metaStackFilename,
      getOnnxFilePath(inputFilePrefix, info->settings.writeOnnxToTmp, ".txt"));
  std::ofstream metaStackOF(metaStackFilename, std::ios::binary);
  if (!metaStackOF) {
    return MAKE_ERR(strFormat("Cannot create metastack text file %s",
                              metaStackFilename.c_str()));
  }

  metaStackOF << metaStack.print();

  // Run the stack on JIT to get outputs then write them to file
  torch::jit::Stack copyStack;
  // We will use original graph for runOnJit, which means the first input
  // should be module.
  if (origGraph_ != nullptr) {
    copyStack.push_back(module_);
  }
  for (auto &ival : stack) {
    if (ival.isTensor()) {
      copyStack.push_back(ival.deepcopy());
    } else {
      copyStack.push_back(ival);
    }
  }
  runOnJit(copyStack);

  // Write outputs
  RETURN_IF_ERR(writeJITOutputsToOnnxFile(outputFilePrefix, copyStack, *info));

  return Error::success();
}

Expected<std::pair<glow::Tensor, torch::Tensor>>
CachingGraphRunner::convertPyTorchInputToGlowInput(
    torch::Tensor &&ptTensor, const glow::Placeholder *ph) {
  glow::Tensor glowTensor;

  glow::TypeRef ty = ph->getType();

  if (ptTensor.is_quantized()) {
    ptTensor = convertQuantizedToDtype(ptTensor, at::kQInt8);
  }

  // If the tensor is an int64 tensor but should be an int32 tensor in Glow,
  // convert it.
  if (ptTensor.scalar_type() == at::kLong &&
      ty->getElementType() == ElemKind::Int32ITy) {
    ptTensor = ptTensor.to(at::kInt);
  }

  // Make sure the runtime pytorch tensor type matches the placeholder.
  // Note this needs to be placed after convertQuantizedToDtype to
  // correctly handle quantized types.
  if (ty->getElementType() != scalarTypeToElemKind(ptTensor.scalar_type())) {
    std::stringstream ss;
    ss << "Found type mismatch for input \"" << ph->getName().str() << "\""
       << ": pytorch tensor is " << ptTensor.toString() << ", ph type is "
       << ty->toString();
    return MAKE_ERR(ss.str());
  }

  if (!ptTensor.is_contiguous()) {
    ptTensor = ptTensor.contiguous();
  }

  // Check Tensor size, making sure enough memory is allocated
  if (ptTensor.numel() > ty->size()) {
    std::stringstream ss;
    ss << "Input tensor is too large: " << ptTensor.numel() << " vs "
       << ty->size() << ": " << ph->getName().str();
    return MAKE_ERR(ss.str());
  }

  if (ty->dims().size() == ptTensor.ndimension() &&
      std::equal(ty->dims().begin(), ty->dims().end(),
                 ptTensor.sizes().begin())) {
    glowTensor = glow::Tensor(ptTensor.data_ptr(), ty);
  } else if (ptTensor.data_ptr() && ptTensor.numel() > 0 &&
             backend_.supportsPartialTensors()) {
    // This is a partial tensor, to create padded unown tensor
    glowTensor = glow::Tensor(ptTensor.data_ptr(), ty, ptTensor.nbytes());
  } else if (ptTensor.numel() == 0) {
    // Handles zero-size input tensor
    // Here zeroLengthSequence_ is pre-allocated if warmCache is called
    assert(zeroLengthSequence_.getUnsafePtr());
    glowTensor = glow::Tensor((void *)zeroLengthSequence_.getUnsafePtr(), ty);
  } else {
    // For backends that does not support partial tensor, manually pad
    // zeros
    auto inputTensorOpt = tensorPool_.get(ty);
    if (!inputTensorOpt.hasValue()) {
      std::stringstream ss;
      ss << "Tensorpool tensor not found for input " << ptTensor.name();
      return MAKE_ERR(ss.str());
    }
    // We want fresh DeviceResidencyInfo for this fresh Tensor.
    glow::Tensor inputTensor(std::move(inputTensorOpt.getValue()));
    inputTensor.resetDeviceInfo();
    if (ptTensor.data_ptr()) {
      memcpy(inputTensor.getUnsafePtr(), ptTensor.data_ptr(),
             ptTensor.nbytes());
      // Pad remaining space with zeroes.
      memset(inputTensor.getUnsafePtr() + ptTensor.nbytes(), 0,
             inputTensor.getSizeInBytes() - ptTensor.nbytes());
    } else {
      inputTensor.zero();
    }
    glowTensor = std::move(inputTensor);
  }

  std::pair<glow::Tensor, torch::Tensor> tensors = {std::move(glowTensor),
                                                    std::move(ptTensor)};
  return tensors;
}

Expected<std::pair<std::vector<glow::Tensor>, std::vector<torch::Tensor>>>
CachingGraphRunner::processPyTorchInputs(
    at::ArrayRef<at::IValue> inputs,
    const std::vector<Placeholder *> &inputPlaceholders) {
  size_t numInputs = inputs.size();

  std::vector<glow::Tensor> glowTensorInputs;
  std::vector<torch::Tensor> ptTensorInputs;
  glowTensorInputs.reserve(numInputs);
  ptTensorInputs.reserve(numInputs);

  // We only hold placeholders for tensor inputs so indexing them is
  // different than indexing all inputs.
  size_t placeholderI = 0;

  for (const auto &input : inputs) {
    if (!input.isTensor()) {
      continue;
    }

    glow::Placeholder *ph = inputPlaceholders[placeholderI++];

    auto ptTensorOrig = input.toTensor();

    std::pair<glow::Tensor, torch::Tensor> tensors;
    ASSIGN_VALUE_OR_RETURN_ERR(
        tensors, convertPyTorchInputToGlowInput(std::move(ptTensorOrig), ph));

    glowTensorInputs.push_back(std::move(tensors.first));

    // Save the PyTorch tensor in case it owns memory we need for inference
    ptTensorInputs.push_back(std::move(tensors.second));
  }

  RETURN_ERR_IF_NOT(inputPlaceholders.size() == glowTensorInputs.size(),
                    strFormat("Expected %d Tensor inputs but found %d",
                              int(inputPlaceholders.size()),
                              int(glowTensorInputs.size())));

  std::pair<std::vector<glow::Tensor>, std::vector<torch::Tensor>> tensors = {
      std::move(glowTensorInputs), std::move(ptTensorInputs)};

  return tensors;
}

Error CachingGraphRunner::runImpl(const PerGlowGraphInfo &info,
                                  torch::jit::Stack &stack,
                                  std::unique_ptr<ExecutionContext> &ctx) {
  size_t runId = numRuns_++;

  int64_t jitRunningTime = 0;
  const PyTorchLoaderSettings &settings = info.settings;

  // Save all of the PyTorch input tensors in case they were allocated here for
  // things like making an input contiguous.
  std::vector<torch::Tensor> ptTensorInputs;

  // Run the subgraph using JIT for comparison with Glow.
  torch::jit::Stack copyStack;
  std::string onnxFileNamePrefix;
  if (settings.writeToOnnx && settings.onnxFileNamePrefix.empty()) {
    onnxFileNamePrefix = info.functionName;
  } else if (settings.writeToOnnx) {
    onnxFileNamePrefix = settings.onnxFileNamePrefix;
  }

  if (settings.jitVsGlowCompare) {

    // We will use original graph for runOnJit, which means the first input
    // should be module.
    if (origGraph_ != nullptr) {
      copyStack.push_back(module_);
    }
    for (auto &ival : stack) {
      if (ival.isTensor()) {
        copyStack.push_back(ival.deepcopy());
      } else {
        copyStack.push_back(ival);
      }
    }
    jitRunningTime = runOnJit(copyStack);
  }

  TraceContext *traceContext = ctx->getTraceContext();
  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "torch_glow::runImpl");
  {
    int64_t runImplTime = TraceEvent::now();
    RECORD_USER_SCOPE(strFormat("torch_glow::runImpl_%s TS:%li",
                                info.functionName.c_str(), runImplTime));

    auto *bindings = ctx->getPlaceholderBindings();

    TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "adjustInputs");

    size_t numInputs = graph_->inputs().size();
    const auto inputs = torch::jit::last(stack, numInputs);

    {
      RECORD_USER_SCOPE("adjustInputs");

      std::vector<glow::Tensor> glowTensorInputs;
      if (auto tensorsOrErr =
              processPyTorchInputs(inputs, info.inputPlaceholders)) {
        glowTensorInputs = std::move(tensorsOrErr->first);
        ptTensorInputs = std::move(tensorsOrErr->second);
      } else {
        RETURN_ERR(tensorsOrErr.takeError());
      }

      // Write input tensors to file
      if (settings.writeToOnnx) {
        RETURN_IF_ERR(writeGlowTensorsToOnnx(
            strFormat("%s_input_%zu", onnxFileNamePrefix.c_str(), runId),
            settings, info.inputPlaceholders, glowTensorInputs));
      }

      // Populate PlaceholderBindings
      for (size_t i = 0; i < glowTensorInputs.size(); ++i) {
        bindings->insert(info.inputPlaceholders[i],
                         std::move(glowTensorInputs[i]));
      }
    } // end adjustInputs

    TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "adjustInputs");

    TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "setupOutput");
    std::vector<at::IValue> outputs;
    {
      RECORD_USER_SCOPE("setupOutput");

      for (auto *ph : info.outputPlaceholders) {
        std::vector<int64_t> sizes;
        for (auto size : ph->dims()) {
          sizes.push_back(static_cast<int64_t>(size));
        }

        auto ptT = glowTypeToEmptyPTTensor(*ph->getType());

        glow::Tensor t(ptT.data_ptr(), ph->getType());

        outputs.push_back(std::move(ptT));
        bindings->insert(ph, std::move(t));
      }
    }
    TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "setupOutput");

    TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME,
                      "runInputSanitization");
    RETURN_IF_ERR(sanitizeInputs(info.inputSanitizers, *bindings));
    TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "runInputSanitization");

    TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "runNetwork");
    int64_t glowRunStartTime, glowRunningTime;
    {
      RECORD_USER_SCOPE("runNetwork");

      glowRunStartTime = TraceEvent::now();
      RETURN_IF_ERR(hostManager_->runNetworkBlocking(info.functionName, ctx));
      glowRunningTime = TraceEvent::now() - glowRunStartTime;

      // Reset the traceContext again in case it was changed during run.
      traceContext = ctx->getTraceContext();
    }
    TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "runNetwork");
    TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "setOutputs");
    {
      RECORD_USER_SCOPE("setOutputs");

      MetaStack *ptrOutputShape;
      if (defaultSettings_.runShapeInference) {
        /// load shape. If already existed, extracted directly, if not, run
        /// shape inference
        ASSIGN_VALUE_OR_RETURN_ERR(ptrOutputShape,
                                   loadShape(inputs, traceContext));
        if (outputs.size() != (*ptrOutputShape).size()) {
          return MAKE_ERR("Fail to infer shape for outputs");
        }
      }

      torch::jit::drop(stack, numInputs);
      std::vector<glow::Tensor> convertedGlowTensors;
      for (int i = 0; i < outputs.size(); i++) {
        auto &output = outputs[i];
        auto ptTensor = output.toTensor();

        // Convert the output to the correct dtype if necessary.
        if (ptTensor.is_quantized()) {
          at::ScalarType dtype = outputCorrectTypes_[i];
          if (dtype == at::ScalarType::QUInt8 ||
              dtype == at::ScalarType::QInt8) {
            ptTensor = convertQuantizedToDtype(ptTensor, dtype);
          } else {
            return MAKE_ERR(
                strFormat("Fail to propagate quantized dtype to output"));
          }
        }

        if (ptTensor.dtype() == at::kInt &&
            outputCorrectTypes_[i] == at::kLong) {
          ptTensor = ptTensor.to(at::kLong);
        }

        // Write the output from Glow to ONNX if necessary.
        if (settings.writeToOnnx) {
          glow::Tensor glowT = ptTensorToGlowTensor(ptTensor);
          convertedGlowTensors.push_back(std::move(glowT));
        }

        // Run shape inference and slice out the correct size of the Glow
        // output if necessary.
        if (settings.runShapeInference) {
          if (i < (*ptrOutputShape).size()) {
            // Assuming all the outputs are tensors for now
            // TODO Add support for other output types, e.g., tensor[]
            const TensorShape &expectedShape =
                (*ptrOutputShape)[i].shape<TensorShape>();
            ptTensor = sliceTensor(ptTensor, expectedShape);
          }
        }

        // Run comparison between Glow and JIT outputs
        if (settings.jitVsGlowCompare) {
          glow::Tensor glowT = ptTensorToGlowTensor(ptTensor);
          auto &jitOutput = torch::jit::peek(copyStack, i, outputs.size());
          auto jitPtTensor = jitOutput.toTensor().contiguous();
          glow::Tensor jitGlowT = ptTensorToGlowTensor(jitPtTensor);
          auto tensorElemType = jitGlowT.getType().getElementType();
          if (tensorElemType == glow::ElemKind::FloatTy) {
            TensorCompareResult res = compareTensors<float_t>(jitGlowT, glowT);
            LOG(INFO) << "Correctness check | Function: " << info.functionName
                      << "\tTensor: "
                      << std::string(info.outputPlaceholders[i]->getName())
                      << "\tRelError: " << res.relErr
                      << "\tMaxErr: " << res.maxErr
                      << "\tMaxRelErr: " << res.maxRelErr << std::endl;
          } else {
            LOG(INFO) << "Correctness Check | Function: " << info.functionName
                      << "\tTensor: "
                      << std::string(info.outputPlaceholders[i]->getName())
                      << "\tUnsupported type: "
                      << std::string(glow::Type::getElementName(tensorElemType))
                      << std::endl;
          }
        }
        stack.push_back(at::IValue(std::move(ptTensor)));
      }

      if (settings.jitVsGlowCompare) {
        LOG(INFO) << "Perf comparison | Function: " << info.functionName
                  << "\tGlow run time: " << glowRunningTime
                  << " us\tCPU run time: " << jitRunningTime << " us"
                  << std::endl;
      }

      if (settings.writeToOnnx) {
        // Write Glow outputs to file
        RETURN_IF_ERR(writeGlowTensorsToOnnx(
            strFormat("%s_glow_output_%zu", onnxFileNamePrefix.c_str(), runId),
            info.settings, info.outputPlaceholders, convertedGlowTensors));

        // Convert JIT outputs to Glow outputs and write to file
        if (settings.jitVsGlowCompare) {
          RETURN_IF_ERR(writeJITOutputsToOnnxFile(
              strFormat("%s_pytorch_output_%zu", onnxFileNamePrefix.c_str(),
                        runId),
              copyStack, info));
        }
      }
    }
    TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "setOutputs");
  }
  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "torch_glow::runImpl");

  return Error::success();
} // namespace glow

Error CachingGraphRunner::run(torch::jit::Stack &stack) {
  if (useRunOnly_) {
    return runOnly(stack);
  }
  std::unique_ptr<ExecutionContext> ctx = glow::make_unique<ExecutionContext>();

  TraceContext *traceContext = nullptr;
  if (defaultSettings_.enableGlowTracing ||
      TraceExporterRegistry::getInstance()->shouldTrace()) {
    ctx->setTraceContext(glow::make_unique<TraceContext>(TraceLevel::STANDARD));
    traceContext = ctx->getTraceContext();
    traceContext->setThreadName("torch_glow");
  }

  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "torch_glow::run");
  detail::GlowError err = detail::GlowError::empty();
  {
    // XXX remove these once we integrate with pytorch profiler using
    // Kineto
    RECORD_USER_SCOPE("torch_glow::run");

    std::shared_ptr<PerGlowGraphInfo> info;
    ASSIGN_VALUE_OR_RETURN_ERR(info,
                               loadImpl(stack, defaultSettings_, traceContext));
    err = runImpl(*DCHECK_NOTNULL(info.get()), stack, ctx);

    // Reset the traceContext again in case it was changed during run.
    traceContext = ctx->getTraceContext();
  }
  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "torch_glow::run");

  TraceExporterRegistry::getInstance()->exportTrace(traceContext);
  if (defaultSettings_.enableGlowTracing) {
    aggregateAndDumpTraces(traceContext);
  }

  return err;
}

Expected<std::shared_ptr<CachingGraphRunner::PerGlowGraphInfo>>
CachingGraphRunner::findGraphInfoForStack(const torch::jit::Stack &stack) {
  if (useMaxSizeCompilation_) {
    std::unique_lock<std::shared_timed_mutex> wlock(graphInfoMapMutex);
    if (perGlowGraphInfoMap_.size() != 1) {
      return MAKE_ERR(strFormat(
          "There should be one and only one compiled graph, but got %lu",
          perGlowGraphInfoMap_.size()));
    }
    return perGlowGraphInfoMap_.begin()->second;
  } else {
    const auto relevantInputs =
        torch::jit::last(stack, graph_->inputs().size());
    InputMetaStack metaStack;
    ASSIGN_VALUE_OR_RETURN_ERR(metaStack,
                               inputMetaStackFromStack(relevantInputs));
    std::unique_lock<std::shared_timed_mutex> wlock(graphInfoMapMutex);
    size_t hash = getGraphMapKeyFromInputStack(metaStack);
    auto it = perGlowGraphInfoMap_.find(hash);
    if (it == perGlowGraphInfoMap_.end()) {
      std::ostringstream ss;
      ss << "No compiled graph found for input stack:\n"
         << metaStack.print() << "\n";
      ss << "There are " << perGlowGraphInfoMap_.size()
         << " input sets with compiled graphs, they are:\n";
      for (const auto &kv : perGlowGraphInfoMap_) {
        ss << kv << "\n";
      }
      return MAKE_ERR(ss.str());
    }
    return it->second;
  }
}

Error CachingGraphRunner::runOnly(torch::jit::Stack &stack) {
  std::shared_ptr<PerGlowGraphInfo> info;
  ASSIGN_VALUE_OR_RETURN_ERR(info, findGraphInfoForStack(stack));

  const PyTorchLoaderSettings &settings = info->settings;

  std::unique_ptr<ExecutionContext> ctx = glow::make_unique<ExecutionContext>();
  TraceContext *traceContext = nullptr;
  if (settings.enableGlowTracing ||
      TraceExporterRegistry::getInstance()->shouldTrace()) {
    ctx->setTraceContext(glow::make_unique<TraceContext>(TraceLevel::STANDARD));
    traceContext = ctx->getTraceContext();
    traceContext->setThreadName("torch_glow");
  }
  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "torch_glow::runOnly");
  detail::GlowError err = detail::GlowError::empty();
  {
    RECORD_USER_SCOPE("torch_glow::runOnly");
    err = runImpl(*DCHECK_NOTNULL(info.get()), stack, ctx);

    // Reset the traceContext again in case it was changed during run.
    traceContext = ctx->getTraceContext();
  }
  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "torch_glow::runOnly");

  TraceExporterRegistry::getInstance()->exportTrace(traceContext);
  if (settings.enableGlowTracing) {
    aggregateAndDumpTraces(traceContext);
  }
  return err;
}

Error CachingGraphRunner::warmCache(
    const std::vector<InputMetaStack> &metaStacks,
    const PyTorchLoaderSettings &settings,
    runtime::DeferredWeightLoader *loader, bool useMaxSizeCompilation,
    bool useDeserialize,
    std::shared_ptr<std::unordered_map<std::string, std::vector<char>>>
        nameToFunctions,
    std::shared_ptr<std::string> glowAOTSerializationSpecStrPtr,
    std::shared_ptr<std::string> glowAOTSerializationModelStrPtr,
    const std::string &serializationSpec, const std::string &onnxModelFile,
    const c10::optional<ModelCompilationConfigOverride>
        &modelCompilationConfigOverride) {
  if (!hostManager_) {
    return MAKE_ERR("Host manager is null!");
  }

  if (!graph_) {
    return MAKE_ERR("No graph found!");
  }

  std::unique_ptr<TraceContext> traceContext;
  if (settings.enableGlowTracing) {
    traceContext = std::make_unique<TraceContext>(TraceLevel::STANDARD);
    traceContext->setThreadName("torch_glow");
  }

  useMaxSizeCompilation_ = useMaxSizeCompilation;
  std::unique_ptr<Module> glowModule = std::make_unique<Module>();

  glow::CompilationContext cctx;
  RETURN_IF_ERR(initializeCompilationContextFromGlowFlags(cctx));
  initializeCompilationContextFromSettings(cctx, settings);
  if (modelCompilationConfigOverride.has_value()) {
    initializeCompilationContextFromModelCompilationConfigOverride(
        cctx, modelCompilationConfigOverride.value());
  }
  cctx.glowAOTSerializationModelStrPtr = glowAOTSerializationModelStrPtr;

  {
    if (settings.lazyCompile) {
      for (const auto &metaStack : metaStacks) {
        size_t hash = getGraphMapKeyFromInputStack(metaStack);

        LOG(INFO) << "Caching compilation setting for hash: " << hash;
        pyTorchLoaderSettingsMap_.emplace(hash, settings);
      }

      return Error::success();
    }

    TRACE_EVENT_BEGIN(traceContext.get(), TraceLevel::RUNTIME,
                      "torch_glow::warmCache");
    RECORD_USER_SCOPE("torch_glow::warmCache");

    runtime::PrePartitionedConfig PPC;

    for (const auto &metaStack : metaStacks) {
      size_t hash = getGraphMapKeyFromInputStack(metaStack);

      {
        std::unique_lock<std::shared_timed_mutex> wlock(graphInfoMapMutex);
        if (perGlowGraphInfoMap_.find(hash) != perGlowGraphInfoMap_.end()) {
          return MAKE_ERR(strFormat("There is already a compiled graph for %s",
                                    metaStack.print().c_str()));
        }
      }

      // HostManager is shared across CachingGraphRunner instances so Function
      // names should be unique so this is included in the name.
      auto info = std::make_shared<PerGlowGraphInfo>(
          strFormat("pt_function_%lu_%lu", size_t(this), hash), settings);
      std::string functionNameHash = strFormat("%lu", hash);

      // If this function has already been compiled, the compiled stream is
      // already stored in nameToFunction, and deserialize is enabled, we use
      // the compiled stream instead of compiling it again.
      if (useDeserialize &&
          nameToFunctions->find(functionNameHash) != nameToFunctions->end()) {
        cctx.nameToFunctions.emplace(std::make_pair(
            info->functionName,
            std::make_shared<std::vector<char>>(
                nameToFunctions->find(functionNameHash)->second)));
        cctx.backendOpts.useDeserialize = true;
      }

      // Type table for deferred weight loader
      std::map<std::string, Type> staticPlaceholderTypes;
      {
        TRACE_EVENT_BEGIN(traceContext.get(), TraceLevel::RUNTIME,
                          "loadJITGraph");
        RECORD_USER_SCOPE("loadJITGraph");
        if (settings.loadGlowIRFromONNX) {
          if (serializationSpec.empty()) {
            return MAKE_ERR("Missing serialization spec file when doing Glow "
                            "deserialization");
          }
          if (onnxModelFile.empty()) {
            return MAKE_ERR(
                "Missing onnx model file when doing Glow deserialization");
          }
          GlowDeserializationSpec spec;
          RETURN_IF_ERR(spec.fromJson(serializationSpec));
          std::vector<const char *> tensorNames;
          for (const auto &name : spec.inputPHNames) {
            tensorNames.emplace_back(name.data());
          }
          for (const auto &name : spec.staticPHNames) {
            tensorNames.emplace_back(name.data());
          }
          std::vector<Type> types;
          for (const auto &typeStr : spec.inputPHTypes) {
            types.emplace_back(Type::fromString(typeStr));
          }
          for (const auto &typeStr : spec.staticPHTypes) {
            types.emplace_back(Type::fromString(typeStr));
          }
          std::vector<TypeRef> typeRefs;
          for (const auto &type : types) {
            typeRefs.emplace_back(&type);
          }
          Error err = Error::empty();
          cctx.prepartitionedConfig = &PPC;
          // We use a dummy file name "model.onnxtxt" to trigger the loadproto
          // logic in ONNXModelLoader; Note that the file name must contain
          // ".onnxtxt" to trigger the logic
          static_cast<void>(ONNXModelLoader(
              "model.onnxtxt", tensorNames, typeRefs, *glowModule,
              spec.functionName, &PPC, &err, /* zipMode */ false,
              /* perNodeOpts */ nullptr,
              /* loadIntoExistingModule */ false,
              /* disableConstFoldInLoader */ false, /* B */ nullptr,
              /* inputStringPtr */ &onnxModelFile));
          RETURN_IF_ERR(err);
          VLOG(1) << "Successfully deserialized Glow IR from onnx model file";

          for (auto &ph : glowModule->getPlaceholders()) {
            // Set PH static if it is in the original GlowIR
            if (std::find(spec.staticPHNames.begin(), spec.staticPHNames.end(),
                          ph->getName().data()) != spec.staticPHNames.end()) {
              auto type = *ph->getType();
              /// Account for the auto FP32->FP16 conversion in Glow
              /// for placeholder \p type match
              auto convertedElemType = getConvertElemTypeForAOT(type, cctx);
              auto convertedDimVec = getConvertDimVecForAOT(type, cctx);
              auto convertedDims =
                  llvm::ArrayRef<unsigned long>(convertedDimVec);
              auto convertedType =
                  type.isQuantizedType()
                      ? glowModule->uniqueType(convertedElemType, convertedDims,
                                               type.getScale(),
                                               type.getOffset())
                      : glowModule->uniqueType(convertedElemType,
                                               convertedDims);
              ph->setType(0, convertedType);
              ph->setStatic(true);

              // Record the types of the actual placeholders before conversion,
              // so that we can replay the conversion in Provisioner
              staticPlaceholderTypes[std::string(ph->getName())] = type;
            }
            // Reconstruct PerGlowGraphInfo
            if (std::find(spec.inputPHNames.begin(), spec.inputPHNames.end(),
                          ph->getName().data()) != spec.inputPHNames.end()) {
              info->inputPlaceholders.push_back(ph);
            }
            if (std::find(spec.outputPHNames.begin(), spec.outputPHNames.end(),
                          ph->getName().data()) != spec.outputPHNames.end()) {
              info->outputPlaceholders.push_back(ph);
            }
          }
          info->functionName = spec.functionName;
          cctx.loadingAOTModel = true;
        } else {
          Function *f = glowModule->createFunction(info->functionName);
          RETURN_IF_ERR(PyTorchModelLoader::loadJITGraph(
              *f, *graph_, info->inputPlaceholders, info->outputPlaceholders,
              outputCorrectTypes_, info->settings, {}, metaStack));
          info->inputSanitizers = runtime::getInputSanitizers(*f);
          // Prepare GlowDeserializationSpec and cctx for serializing Glow IR
          if (settings.saveGlowIRIntoONNX) {
            GlowDeserializationSpec spec;
            RETURN_IF_ERR(setupGlowDeserializationSpecAndCctx(
                settings, info, cctx, f, spec,
                glowAOTSerializationModelStrPtr));
            RETURN_IF_ERR(saveGlowDeserializationSpec(
                spec, glowAOTSerializationSpecStrPtr));
          }

          for (auto *PH : glowModule->getPlaceholders()) {
            if (PH->isStatic()) {
              staticPlaceholderTypes[std::string(PH->getName())] =
                  *PH->getType();
            }
          }
        }

        TRACE_EVENT_END(traceContext.get(), TraceLevel::RUNTIME,
                        "loadJITGraph");
      }

      // Obtain maxSeqLength from metaStack
      // This step can also be done with a user input, but the overhead of the
      // following code should not be significant
      for (auto meta : metaStack.inputMetas) {
        maxSeqLength_ = std::max(
            maxSeqLength_,
            (size_t)std::accumulate(meta.dims.begin(), meta.dims.end(), 1,
                                    std::multiplies<glow::dim_t>()));
      }

      // Allocate zeroLengthSequence with maximum size
      // Similar to the impl in Onnxifi/Base.cpp
      glow::Type zt(ElemKind::Int64ITy, {maxSeqLength_});
      zeroLengthSequence_.reset(zt);
      zeroLengthSequence_.zero();

      if (loader) {
        loader->setTypeInfo(std::move(staticPlaceholderTypes));
        cctx.deferredWeightLoader = loader;

        // Signal that we want to fold convertTo and Quantize into static
        // Placeholders. Also want to do this for AOT optimization even if we
        // don't have a deferred blob reader present.
        cctx.optimizationOpts.foldStaticPlaceholderConversions = true;
      }

      // There should be only one element in the map when model is precompiled.
      perGlowGraphInfoMap_.emplace(hash, info);
    }

    {
      TRACE_EVENT_BEGIN(traceContext.get(), TraceLevel::RUNTIME, "addNetwork");
      RECORD_USER_SCOPE("addNetwork");
      RETURN_IF_ERR(hostManager_->addNetwork(std::move(glowModule), cctx));
      TRACE_EVENT_END(traceContext.get(), TraceLevel::RUNTIME, "addNetwork");
    }

    TRACE_EVENT_END(traceContext.get(), TraceLevel::RUNTIME,
                    "torch_glow::warmCache");
  }
  if (settings.enableGlowTracing) {
    aggregateAndDumpTraces(traceContext.get());
  }
  return Error::success();
}

CachingGraphRunner::CachingGraphRunner(
    std::shared_ptr<torch::jit::Graph> graph,
    std::shared_ptr<runtime::HostManager> hostManager,
    PyTorchLoaderSettings defaultSettings, bool useRunOnly,
    std::shared_ptr<torch::jit::Graph> origGraph, c10::IValue module)
    : graph_(graph), origGraph_(origGraph), ptGraphExecutor_(graph, "forward"),
      module_(module), hostManager_(hostManager),
      backend_(*EXIT_ON_ERR(hostManager->getBackend())),
      defaultSettings_(std::move(defaultSettings)), useRunOnly_(useRunOnly) {

  if (origGraph_ != nullptr) {
    ptGraphExecutor_ = torch::jit::GraphExecutor(origGraph_, "forward");
  } else {
    ptGraphExecutor_ = torch::jit::GraphExecutor(graph_, "forward");
  }
  mergedTraceContext_ = glow::make_unique<TraceContext>(TraceLevel::STANDARD);
}

CachingGraphRunner::~CachingGraphRunner() {
  // Dump trace for the last time if there are remaining
  aggregateAndDumpTraces(nullptr, true);

  // Remove Glow functions saved in HostManager when being destroyed.
  std::unique_lock<std::shared_timed_mutex> wlock(graphInfoMapMutex);
  for (auto &kv : perGlowGraphInfoMap_) {
    ERR_TO_BOOL(hostManager_->removeNetwork(kv.second->functionName));
  }
}

Error CachingGraphRunner::warmupGraphOutputShapeMap(
    const c10::ArrayRef<torch::jit::Value *> &graphOutputValues,
    const BatchShapesMapType &graphShapeMetaMap) {
  for (auto &it : graphShapeMetaMap) {
    auto &shapeMap = it.second;
    auto batchSize = it.first;
    MetaStack outputShape;
    for (auto outputValue : graphOutputValues) {
      auto itr = shapeMap.find(outputValue);
      if (itr == shapeMap.end()) {
        std::ostringstream ss;
        ss << "Node output " << outputValue->debugName()
           << " Not found in the shape map!";
        return MAKE_ERR(ss.str());
      }
      outputShape.emplace_back(itr->second);
    }
    perGlowGraphShapeMap_.emplace(batchSize, outputShape);
  }
  return Error::success();
}

Error CachingGraphRunner::setNominalInputIndex(
    const c10::ArrayRef<torch::jit::Value *> &graphInputValues,
    const BatchShapesMapType &graphShapeMetaMap) {
  int nominalInputIndex = -1;
  if (graphShapeMetaMap.size() == 0) {
    std::ostringstream ss;
    ss << "Input graph shap meta map is empty.";
    return MAKE_ERR(ss.str());
  }
  for (size_t i = 0; i < graphInputValues.size(); ++i) {
    const torch::jit::Value *inputValue = graphInputValues[i];
    bool matchIndex = true;
    for (auto &itz : graphShapeMetaMap) {
      auto &shapeMap = itz.second;
      auto batchSize = itz.first;
      auto itr = shapeMap.find(inputValue);
      if (itr == shapeMap.end()) {
        std::ostringstream ss;
        ss << "Node input " << inputValue->debugName()
           << " not found in the shape map!";
        return MAKE_ERR(ss.str());
      }
      auto &tensorShape = itr->second.shape<TensorShape>();
      if (tensorShape.size() < 2 || tensorShape[0] != batchSize) {
        matchIndex = false;
        break;
      }
    }
    if (matchIndex) {
      nominalInputIndex = i;
      break;
    }
  }
  if (nominalInputIndex != -1) {
    nominalInputIndex_ = nominalInputIndex;
    LOG(INFO) << "Finish Setting nomnial input index: " << nominalInputIndex;
    return Error::success();
  } else {
    std::ostringstream ss;
    ss << "No valid nominalInputIndex is found.";
    return MAKE_ERR(ss.str());
  }
}

int CachingGraphRunner::getNominalInputIndex() { return nominalInputIndex_; }

size_t CachingGraphRunner::getGraphMapKeyFromInputStack(
    const InputMetaStack &metaStack) {
  size_t hash;
  if (defaultSettings_.nominalBatchIdx >= 0) {
    hash = metaStack.optimizedHash(defaultSettings_.nominalBatchIdx);
  } else if (nominalInputIndex_ >= 0) {
    hash = metaStack.optimizedHash(nominalInputIndex_);
  } else {
    hash = metaStack.hash();
  }
  return hash;
}

} // namespace glow
