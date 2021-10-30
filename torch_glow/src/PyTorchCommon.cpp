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

#include <fstream>
#include <string>

#include "ATen/core/interned_strings.h"
#include "PyTorchCommon.h"

#include "FuseKnownPatterns.h"
#include "GlowFuser.h"
#include "PyTorchModelLoader.h"
#include "Registration.h"
#include "ShapeInferenceEngine.h"

#include "glow/Flags/Flags.h"
#include "glow/Runtime/ErrorReporter.h"
#include "llvm/Support/FileSystem.h"

#include "torch/csrc/jit/passes/canonicalize_graph_fuser_ops.h"
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/operator_options.h>

#include "torch/csrc/jit/ir/irparser.h"

#include <torch/script.h>

DEFINE_string(torch_glow_backend, "Interpreter",
              "Glow backend used for torchifi");
DEFINE_int32(torch_glow_num_devices, -1, "Number of devices for Glow backend");
DEFINE_bool(torch_glow_scan_devices, false,
            "Control if scanning available devices in torch glow backend");

DEFINE_bool(saturateHost, false, "See PyTorchLoaderSettings");

DEFINE_int32(torch_glow_min_fusion_group_size, 1,
             "Minimum number of nodes in the glow fusion group");
DEFINE_bool(printJITIndex, false, "Enable printing of jit node indexes");
// TODO: Handle this case with FloorDiv
DEFINE_bool(ignoreDivRoundingArgs, false,
            "Ignore the rounding argument to aten::div");
DEFINE_bool(dumpGlowDag, false, "See PyTorchLoaderSettings");
DEFINE_bool(jitVsGlowCompare, false, "Enable per-group error check");
DEFINE_bool(dumpFinalGlowGraph, false, "See PyTorchLoaderSettings");
DEFINE_bool(enableGlowTracing, false, "See PyTorchLoaderSettings");
DEFINE_int32(numTracesPerDump, 1, "See PyTorchLoaderSettings");

// settings for model precision conversion
DEFINE_bool(convertToFP16, false, "See PyTorchLoaderSettings");
DEFINE_bool(skipBiasFp32tofp16Convert, false, "See PyTorchLoaderSettings");
DEFINE_bool(convertFusedToFP16, false, "See PyTorchLoaderSettings");
DEFINE_bool(clipFP16, false, "See PyTorchLoaderSettings");
DEFINE_bool(clipFP16SkipInputs, true, "See PyTorchLoaderSettings");
DEFINE_bool(convertPlaceholdersToFP16, true, "See PyTorchLoaderSettings");
DEFINE_bool(convertConstantsToFP16, true, "See PyTorchLoaderSettings");
DEFINE_bool(forceFP16AccumSLS, true, "See PyTorchLoaderSettings");

DEFINE_string(opBlacklist, "", "See PyTorchLoaderSettings");
DEFINE_int32(replicationCount, 1, "Number of replications on each device");
DEFINE_bool(writeToOnnx, false, "See PyTorchLoaderSettings");
DEFINE_bool(onnxZipMode, false, "See PyTorchLoaderSettings");
DEFINE_bool(writeOnnxToTmp, false, "See PyTorchLoaderSettings");
DEFINE_int32(maxActiveRequests, 250,
             "Max number of active requests before HostManager starts queuing");
DEFINE_bool(dumpOperatorInventory, false,
            "Dump jit operator inventory after glow lowering.");
DEFINE_bool(randomizeConstants, false, "See PyTorchLoaderSettings");
DEFINE_bool(writeWithoutRandomize, false, "See PyTorchLoaderSettings");
DEFINE_bool(runShapeInference, false, "See PyTorchLoaderSettings");
DEFINE_int32(fusionStartIndex, -1, "See PyTorchLoaderSettings");
DEFINE_int32(fusionEndIndex, -1, "See PyTorchLoaderSettings");
DEFINE_bool(setIncludeLastOffsets, true, "See PyTorchLoaderSettings");
DEFINE_bool(inferShapeForCompilation, false,
            "Infer shape for the entire model for compilation");
DEFINE_bool(enableRemoveMutation, true, "See PyTorchLoaderSettings");
DEFINE_bool(enableDeserialize, false, "See PyTorchLoaderSettings");
DEFINE_string(backendSpecificOpts, "",
              "Comma separated list of key=value for building the "
              "BackendSpecificOptions map in BackendOptions in "
              "CompilationContext.");
DEFINE_bool(debugContinuouslyVerifyDuringModelLoading, false,
            "See PyTorchLoaderSettings");
DEFINE_int32(nominalBatchIdx, -1, "See PyTorchLoaderSettings");
DEFINE_bool(dumpFailedInputsToOnnxFiles, false, "See PyTorchLoaderSettings");
DEFINE_bool(lazyCompile, false, "see PyTorchLoaderSettings");
DEFINE_bool(enableDeviceTracing, false, "See PyTorchLoaderSettings");
DEFINE_int32(debugLayers, 5, "See PyTorchLoaderSettings");

DEFINE_bool(saveGlowIRIntoONNX, false, "See PyTorchLoaderSettings");
DEFINE_bool(loadGlowIRFromONNX, false, "See PyTorchLoaderSettings");

DEFINE_bool(useSparseNNPartitioningScheme, false, "See PyTorchLoaderSettings");
DEFINE_bool(sparseNNPartitioningAddSLSConcats, false,
            "See PyTorchLoaderSettings");
DEFINE_bool(sparseNNPartitioningBalancePerfModel, false,
            "See PyTorchLoaderSettings");
DEFINE_bool(sparseNNPartitioningPairLNWithSLS, false,
            "See PyTorchLoaderSettings");
DEFINE_bool(sparseNNPartitioningPairTileWithSLS, false,
            "See PyTorchLoaderSettings");
DEFINE_string(sparseNNPartitioningPairSLSWith, "", "See PyTorchLoaderSettings");
DEFINE_int32(sparseNNPartitioningConcatSplitSize, 1,
             "See PyTorchLoaderSettings");
DEFINE_int32(sparseNNPartitioningSchemeNumCards, 1,
             "See PyTorchLoaderSettings");
DEFINE_int64(sparseNNPartitioningSchemeSLSTableKBytesPerCard, 1,
             "See PyTorchLoaderSettings");
DEFINE_int32(SparseNNPartitioningSchemeNumCoresSLS, 1,
             "See PyTorchLoaderSettings");
DEFINE_int32(SparseNNPartitioningSchemeNumCoresOther, 1,
             "See PyTorchLoaderSettings");

namespace glow {
namespace {

static int setGraphExecutorToLegacy() {
  // use legacy GraphExecutor for Glow
  torch::jit::getExecutorMode() = false;
  torch::jit::getProfilingMode() = false;
  return 0;
}

static const int USE_LEGACY_GE = setGraphExecutorToLegacy();

PyTorchLoaderSettings &getPyTorchLoaderSettingsInternalOnly() {
  static PyTorchLoaderSettings settings;
  return settings;
}

} // namespace

void dumpOperatorStats(const torch::jit::Graph &graph) {
  std::map<torch::jit::NodeKind, int> opCounter;
  for (const auto *node : graph.nodes()) {
    opCounter[node->kind()]++;
  }
  std::ostringstream ss;
  ss << "Dump of operator/node stats for graph:\n";
  ss << folly::stringPrintf("%45s %13s \n", "Node Kind", "Count");
  for (const auto &[kind, count] : opCounter) {
    ss << folly::stringPrintf("%45s %13d \n", kind.toQualString(), count);
  }
  LOG(INFO) << ss.str();
}

std::shared_ptr<runtime::HostManager>
getHostManager(const PyTorchLoaderSettings &settings) {
  static std::mutex m_;
  std::unique_lock<std::mutex> lock(m_);
  static std::unordered_map<std::string, std::weak_ptr<runtime::HostManager>>
      map_;

  std::shared_ptr<runtime::HostManager> hostManager;
  auto it = map_.find(settings.backendName);
  if (it != map_.end()) {
    hostManager = it->second.lock();
  }

  // If HostManager was found, check that it's valid, otherwise create a new
  // HostManager
  if (hostManager) {
    if (settings.numDevices != -1) {
      CHECK_EQ(hostManager->numDevices(), settings.numDevices)
          << "Tried to create a new HostManager for backend \""
          << settings.backendName
          << "\" but there is already an existing HostManager in use for that "
             "Backend but with a different number of devices";
    }
  } else {
    std::vector<std::unique_ptr<runtime::DeviceConfig>> deviceConfigs;
    // If scan devices flag is set, we should scan devices that's available
    if (settings.scanDevices) {
      deviceConfigs = runtime::DeviceManager::generateDeviceConfigs(
          settings.backendName, true /*scanDevices*/);
    } else {
      // If number of devices isn't specified then just use 1 device.
      for (int32_t i = 0, e = settings.numDevices < 0 ? 1 : settings.numDevices;
           i < e; i++) {
        auto config =
            std::make_unique<runtime::DeviceConfig>(settings.backendName);
        config->deviceID = i;
        deviceConfigs.push_back(std::move(config));
      }
    }

    glow::runtime::HostConfig hostConfig;

    hostConfig.maxActiveRequests = glow::flags::MaxActiveRequests;
    hostConfig.maxQueueSize = glow::flags::MaxQueueSize;
    hostConfig.executorThreads = glow::flags::ExecutorThreads;

    // now overwrite existing config if torch_glow gflag is present
    hostConfig.maxActiveRequests = FLAGS_maxActiveRequests;

    hostManager = std::make_shared<runtime::HostManager>(
        std::move(deviceConfigs), hostConfig);

    if (settings.enableDeviceTracing) {
      if (!hostManager->getTraceContext()) {
        hostManager->setTraceContext(
            glow::make_unique<TraceContext>(TraceLevel::STANDARD));
      }
      ERR_TO_VOID(hostManager->startDeviceTrace());
    } else {
      ERR_TO_VOID(hostManager->stopDeviceTrace());
    }

    map_[settings.backendName] = hostManager;
  }

  // Update the available devices list if necessary
  if (!settings.availableDevices.empty()) {
    std::vector<size_t> availableDevices(settings.availableDevices.begin(),
                                         settings.availableDevices.end());
    hostManager->setAvailableDevices(availableDevices);
  }

  return hostManager;
}

/// Given a Glow ElemKind \p ty, \returns a matching PyTorch ScalarType.
c10::ScalarType elemKindToScalarType(glow::ElemKind ty) {
  switch (ty) {
  case ElemKind::FloatTy:
    return at::kFloat;
  case ElemKind::Float16Ty:
    return at::kHalf;
  case ElemKind::BFloat16Ty:
    return at::kBFloat16;
  case ElemKind::Int32ITy:
    return at::kInt;
  case ElemKind::Int64ITy:
    return at::kLong;
  case ElemKind::BoolTy:
    return at::kBool;
  case ElemKind::Int8QTy:
    return at::kQInt8;
  case ElemKind::UInt8QTy:
    LOG(DFATAL) << "UInt8QTy is not supported yet.";
    return at::kQUInt8;
  case ElemKind::Float64Ty:
    return at::kDouble;
  case ElemKind::UInt8FusedQTy:
  case ElemKind::UInt8FusedFP16QTy:
  case ElemKind::UInt4FusedFP16QTy:
  case ElemKind::UInt4FusedQTy:
  case ElemKind::UInt8ITy:
  case ElemKind::Int16QTy:
  case ElemKind::Int32QTy:
  case ElemKind::Int64QTy:
    LOG(DFATAL) << "Not supported yet.";
    return at::kLong;
  }
  LOG(DFATAL) << "Cannot reach here.";
}

/// Given a PyTorch ScalarType \p ty, \returns a matching Glow ElemKind.
glow::ElemKind scalarTypeToElemKind(c10::ScalarType ty) {
  if (ty == at::kFloat) {
    return ElemKind::FloatTy;
  } else if (ty == at::kHalf) {
    return ElemKind::Float16Ty;
  } else if (ty == at::kInt) {
    return ElemKind::Int32ITy;
  } else if (ty == at::kLong) {
    return ElemKind::Int64ITy;
  } else if (ty == at::kBool) {
    return ElemKind::BoolTy;
  } else if (ty == at::kByte) {
    // We should have an 8-byte non-quantized integer type eventually
    // Currently usage of Bool is fine
    return ElemKind::BoolTy;
  } else if (ty == at::kQInt8) {
    return ElemKind::Int8QTy;
  } else if (ty == at::kQUInt8) {
    return ElemKind::UInt8QTy;
  } else if (ty == at::kDouble) {
    return ElemKind::Float64Ty;
  } else {
    LOG(DFATAL) << "ScalarType " << c10::toString(ty)
                << " is not supported yet. Using int64 instead";
    return ElemKind::Int64ITy;
  }
}

/// Given a c10 typekind \p ty, \returns a matching Glow ElemKind.
ElemKind typeKindToElemKind(c10::TypeKind ty) {
  if (ty == c10::TypeKind::FloatType) {
    return ElemKind::FloatTy;
  } else if (ty == c10::TypeKind::IntType) {
    return ElemKind::Int32ITy;
  } else if (ty == c10::TypeKind::BoolType) {
    return ElemKind::BoolTy;
  } else {
    LOG(DFATAL) << "Not supported yet.";
    return ElemKind::Int64ITy;
  }
}

/// Split string \p s on character \p k and eliminate spaces.
static std::vector<std::string> splitString(const std::string &s,
                                            const char k = ',') {
  std::vector<std::string> substrings;
  size_t start = 0;
  bool lastWasSplit = true;
  for (size_t i = 0; i < s.size(); i++) {
    if (lastWasSplit && s[i] == ' ') {
      start = i + 1;
      continue;
    }
    lastWasSplit = false;
    if (s[i] == k) {
      substrings.push_back(s.substr(start, i - start));
      start = i + 1;
      lastWasSplit = true;
    }
  }

  if (start < s.size() - 1) {
    substrings.push_back(s.substr(start, s.size() - start));
  }

  return substrings;
}

void PyTorchLoaderSettings::initSettings() {
  minFusionGroupSize = FLAGS_torch_glow_min_fusion_group_size;
  dumpGlowDag = FLAGS_dumpGlowDag;
  jitVsGlowCompare = FLAGS_jitVsGlowCompare;
  printJITIndex = FLAGS_printJITIndex;
  ignoreDivRoundingArgs = FLAGS_ignoreDivRoundingArgs;
  dumpFinalGlowGraph = FLAGS_dumpFinalGlowGraph;
  enableGlowTracing = FLAGS_enableGlowTracing;
  numTracesPerDump = FLAGS_numTracesPerDump;
  saturateHost = FLAGS_saturateHost;
  convertToFP16 = FLAGS_convertToFP16;
  skipBiasFp32tofp16Convert = FLAGS_skipBiasFp32tofp16Convert;
  convertFusedToFP16 = FLAGS_convertFusedToFP16;
  clipFP16 = FLAGS_clipFP16;
  clipFP16SkipInputs = FLAGS_clipFP16SkipInputs;
  convertPlaceholdersToFP16 = FLAGS_convertPlaceholdersToFP16;
  convertConstantsToFP16 = FLAGS_convertConstantsToFP16;
  forceFP16AccumSLS = FLAGS_forceFP16AccumSLS;
  replicationCount = FLAGS_replicationCount;
  writeToOnnx = FLAGS_writeToOnnx;
  onnxZipMode = FLAGS_onnxZipMode;
  dumpFailedInputsToOnnxFiles = FLAGS_dumpFailedInputsToOnnxFiles;
  enableDeviceTracing = FLAGS_enableDeviceTracing;
  writeOnnxToTmp = FLAGS_writeOnnxToTmp;
  randomizeConstants = FLAGS_randomizeConstants;
  dumpOperatorInventory = FLAGS_dumpOperatorInventory;
  writeWithoutRandomize = FLAGS_writeWithoutRandomize;
  backendName = FLAGS_torch_glow_backend;
  numDevices = FLAGS_torch_glow_num_devices;
  scanDevices = FLAGS_torch_glow_scan_devices;
  runShapeInference = FLAGS_runShapeInference;
  fusionStartIndex = FLAGS_fusionStartIndex;
  fusionEndIndex = FLAGS_fusionEndIndex;
  setIncludeLastOffsets = FLAGS_setIncludeLastOffsets;
  enableDeserialize = FLAGS_enableDeserialize;
  enableRemoveMutation = FLAGS_enableRemoveMutation;
  debugContinuouslyVerifyDuringModelLoading =
      FLAGS_debugContinuouslyVerifyDuringModelLoading;
  nominalBatchIdx = FLAGS_nominalBatchIdx;
  lazyCompile = FLAGS_lazyCompile;
  use_dag_optimizer = glow::flags::UseDAGOptimizer;
  apl_parallelization_alg =
      glow::flags::DAGOptimizerParallelizationTaggingAlgorithm;
  apl_num_parallel_chunks = glow::flags::DAGOptimizerNumParallelChunks;
  saveGlowIRIntoONNX = FLAGS_saveGlowIRIntoONNX;
  loadGlowIRFromONNX = FLAGS_loadGlowIRFromONNX;
  skipProvisioning = glow::flags::SkipProvisioning || saveGlowIRIntoONNX;
  sinkTanhBelowConcat = glow::flags::SinkTanhBelowConcat;
  useSparseNNPartitioningScheme = FLAGS_useSparseNNPartitioningScheme;
  sparseNNPartitioningAddSLSConcats = FLAGS_sparseNNPartitioningAddSLSConcats;
  sparseNNPartitioningBalancePerfModel =
      FLAGS_sparseNNPartitioningBalancePerfModel;
  sparseNNPartitioningPairLNWithSLS = FLAGS_sparseNNPartitioningPairLNWithSLS;
  sparseNNPartitioningPairTileWithSLS =
      FLAGS_sparseNNPartitioningPairTileWithSLS;
  sparseNNPartitioningPairSLSWith = FLAGS_sparseNNPartitioningPairSLSWith;
  sparseNNPartitioningConcatSplitSize =
      FLAGS_sparseNNPartitioningConcatSplitSize;
  sparseNNPartitioningSchemeNumCards = FLAGS_sparseNNPartitioningSchemeNumCards;
  sparseNNPartitioningSchemeSLSTableKBytesPerCard =
      FLAGS_sparseNNPartitioningSchemeSLSTableKBytesPerCard;
  SparseNNPartitioningSchemeNumCoresSLS =
      FLAGS_SparseNNPartitioningSchemeNumCoresSLS;
  SparseNNPartitioningSchemeNumCoresOther =
      FLAGS_SparseNNPartitioningSchemeNumCoresOther;
  debugLayers = FLAGS_debugLayers;

  if (!FLAGS_opBlacklist.empty()) {
    auto kindStrings = splitString(FLAGS_opBlacklist);
    for (const auto &kindString : kindStrings) {
      opBlocklist.insert(torch::jit::Symbol::fromQualString(kindString));
    }
  }

  glow::flags::processBackendSpecificOpts(backendSpecificOpts,
                                          FLAGS_backendSpecificOpts);
}

PyTorchLoaderSettings::PyTorchLoaderSettings() { initSettings(); }

PyTorchLoaderSettings getGlobalPyTorchLoaderSettingsSnapshot() {
  return getPyTorchLoaderSettingsInternalOnly();
}
PyTorchLoaderSettings &getGlobalPyTorchLoaderSettingsMutable() {
  return getPyTorchLoaderSettingsInternalOnly();
}

std::string PyTorchLoaderSettings::toString() const {
#define INSERT_BOOL_TO_STREAM(value, stream)                                   \
  (stream) << #value << ": " << ((value) ? "true" : "false") << std::endl;

#define INSERT_VALUE_TO_STREAM(value, stream)                                  \
  (stream) << #value << ": " << value << std::endl;

  std::stringstream s;
  s << std::endl;
  INSERT_BOOL_TO_STREAM(convertToFP16, s);
  INSERT_BOOL_TO_STREAM(skipBiasFp32tofp16Convert, s);
  INSERT_BOOL_TO_STREAM(convertFusedToFP16, s);
  INSERT_BOOL_TO_STREAM(clipFP16, s);
  INSERT_BOOL_TO_STREAM(clipFP16SkipInputs, s);
  INSERT_BOOL_TO_STREAM(convertPlaceholdersToFP16, s);
  INSERT_BOOL_TO_STREAM(convertConstantsToFP16, s);
  INSERT_BOOL_TO_STREAM(forceFP16AccumSLS, s);
  INSERT_BOOL_TO_STREAM(saturateHost, s);
  INSERT_VALUE_TO_STREAM(backendOptionsFile, s);
  INSERT_VALUE_TO_STREAM(replicationCount, s);
  INSERT_BOOL_TO_STREAM(fusionPassEnabled, s);
  INSERT_BOOL_TO_STREAM(dumpGlowDag, s);
  INSERT_BOOL_TO_STREAM(dumpOperatorInventory, s);
  INSERT_VALUE_TO_STREAM(minFusionGroupSize, s);
  INSERT_VALUE_TO_STREAM(maxFusionMergeSize, s);
  INSERT_VALUE_TO_STREAM(fusionStartIndex, s);
  INSERT_BOOL_TO_STREAM(enableRemoveMutation, s);
  INSERT_BOOL_TO_STREAM(enableDeserialize, s);
  INSERT_VALUE_TO_STREAM(fusionEndIndex, s);
  INSERT_BOOL_TO_STREAM(dumpFinalGlowGraph, s);
  INSERT_BOOL_TO_STREAM(enableGlowTracing, s);
  INSERT_VALUE_TO_STREAM(numTracesPerDump, s);
  INSERT_BOOL_TO_STREAM(writeToOnnx, s);
  INSERT_BOOL_TO_STREAM(onnxZipMode, s);
  INSERT_BOOL_TO_STREAM(writeOnnxToTmp, s);
  INSERT_BOOL_TO_STREAM(jitVsGlowCompare, s);
  INSERT_BOOL_TO_STREAM(randomizeConstants, s);
  INSERT_BOOL_TO_STREAM(writeWithoutRandomize, s);
  INSERT_VALUE_TO_STREAM(backendName, s);
  INSERT_VALUE_TO_STREAM(numDevices, s);
  INSERT_BOOL_TO_STREAM(scanDevices, s);
  INSERT_BOOL_TO_STREAM(runShapeInference, s);
  INSERT_BOOL_TO_STREAM(setIncludeLastOffsets, s);
  INSERT_BOOL_TO_STREAM(enableDebugFuser, s);
  INSERT_BOOL_TO_STREAM(debugContinuouslyVerifyDuringModelLoading, s);
  INSERT_BOOL_TO_STREAM(dumpFailedInputsToOnnxFiles, s);
  INSERT_BOOL_TO_STREAM(lazyCompile, s);
  INSERT_BOOL_TO_STREAM(enableDeviceTracing, s);
  INSERT_VALUE_TO_STREAM(debugLayers, s);
  INSERT_BOOL_TO_STREAM(useMaxSizeCompilation, s);

  if (opBlocklist.size() > 0) {
    s << "opBlocklist: [";
    for (const auto &op : opBlocklist) {
      s << op.toQualString() << ",";
    }
    s << "]" << std::endl;
  }
  if (backendSpecificOpts.size() > 0) {
    s << "backendSpecificOpts: [";
    for (const auto &kv : backendSpecificOpts) {
      s << kv.first << "=" << kv.second << ",";
    }
    s << "]" << std::endl;
  }
  return s.str();

#undef INSERT_VALUE_TO_STREAM
#undef INSERT_BOOL_TO_STREAM
}

const c10::Symbol &getGlowSymbol() {
  static c10::Symbol glowSymbol =
      at::Symbol::fromQualString("glow::FusionGroup");
  return glowSymbol;
}

c10::Symbol getGlowSymbol(std::shared_ptr<torch::jit::Graph> g) {
  std::string symbol = "glow::FusionGroup";
  if (g) {
    symbol += strFormat("_%lu", reinterpret_cast<uint64_t>(g.get()));
  }
  return at::Symbol::fromQualString(symbol);
}

glow::Type ptTypeToGlowType(const c10::TensorType &ptType) {
  DCHECK(ptType.scalarType().has_value())
      << "TensorType has no associated scalar type.";
  const auto concreteSizes = ptType.sizes().concrete_sizes().value();
  std::vector<glow::dim_t> dims;
  for (const auto &size : concreteSizes) {
    dims.push_back(static_cast<glow::dim_t>(size));
  }

  auto scalarType = ptType.scalarType().value();
  return glow::Type(scalarTypeToElemKind(scalarType), dims);
}

glow::Type ptTypeToGlowType(const c10::TensorType &ptType, float scale,
                            int32_t zero_point) {
  DCHECK(ptType.scalarType().has_value())
      << "TensorType has no associated scalar type.";
  const auto concreteSizes = ptType.sizes().concrete_sizes().value();
  std::vector<glow::dim_t> dims;
  for (const auto &size : concreteSizes) {
    dims.push_back(static_cast<glow::dim_t>(size));
  }

  auto scalarType = ptType.scalarType().value();
  return glow::Type(scalarTypeToElemKind(scalarType), dims, scale, zero_point);
}

at::Tensor convertQuantizedToDtype(at::Tensor ptTensor, c10::ScalarType dtype) {
  if (dtype != at::kQInt8 && dtype != at::kQUInt8) {
    LOG(DFATAL) << "Can only convert to int8 or uint8";
  }

  if (!ptTensor.is_quantized()) {
    LOG(DFATAL) << "Only support perform convert in quantized tensor.";
  }

  if (ptTensor.qscheme() != at::kPerTensorAffine) {
    LOG(DFATAL)
        << "Only support perform convert for per tensor quantized tensor.";
  }

  // dtype is ptTensor type, do nothing
  if (ptTensor.scalar_type() == dtype) {
    return ptTensor;
  }

  int offsetShift = 0;
  c10::ScalarType targetDQType;

  // We need to manually cast ptTensor to targetDQType, then make it quantized
  // tensor. In PyTorch, int8 is char and uint8 is byte.
  if (dtype == at::kQUInt8 && ptTensor.scalar_type() == at::kQInt8) {
    offsetShift = UINT8_TO_INT8_SHIFT;
    targetDQType = at::kByte;
  } else if (dtype == at::kQInt8 && ptTensor.scalar_type() == at::kQUInt8) {
    offsetShift = -UINT8_TO_INT8_SHIFT;
    targetDQType = at::kChar;
  } else {
    LOG(FATAL) << "Can not reach here.";
  }

  auto scale = static_cast<float>(ptTensor.q_scale());
  auto offset = static_cast<int32_t>(ptTensor.q_zero_point());
  auto ptNewTensor = ptTensor.int_repr().to(targetDQType).add(offsetShift);
  auto ptNewQTensor = at::_make_per_tensor_quantized_tensor(
      ptNewTensor, scale, offset + offsetShift);
  return ptNewQTensor;
}

at::Tensor glowTypeToEmptyPTTensor(const glow::Type &glowType) {
  std::vector<int64_t> sizes;
  for (const auto dim : glowType.dims()) {
    sizes.push_back(dim);
  }
  if (glowType.isQuantizedType()) {
    auto scale = glowType.getScale();
    auto offset = glowType.getOffset();
    return at::_empty_affine_quantized(
        sizes,
        at::TensorOptions().dtype(
            elemKindToScalarType(glowType.getElementType())),
        scale, offset);
  } else {
    return at::empty(sizes, at::TensorOptions().dtype(elemKindToScalarType(
                                glowType.getElementType())));
  }
}

glow::Tensor ptTensorToGlowTensor(const at::Tensor &ptTensor) {
  CHECK(ptTensor.is_contiguous());
  if (ptTensor.is_quantized()) {
    float scale = 1.0;
    int32_t offset = 0;
    if (ptTensor.qscheme() == at::kPerChannelAffine) {
      // If it is channel wise quantized, which means
      // this tensor is the weight of quantized linear or conv
      // Then we dont deal with the qparams here,
      // and only set up soome dummy scale & offset by using the first
      // elements's scale & offset.
      scale = ptTensor.q_per_channel_scales()[0].item<float>();
      offset = ptTensor.q_per_channel_zero_points()[0].item<int32_t>();
    } else if (ptTensor.qscheme() == at::kPerTensorAffine) {
      scale = static_cast<float>(ptTensor.q_scale());
      offset = static_cast<int32_t>(ptTensor.q_zero_point());
    } else {
      LOG(DFATAL)
          << "PyTorch tensor with unsupported quantization scheme detected.";
    }
    auto glowType =
        ptTypeToGlowType(*c10::TensorType::create(ptTensor), scale, offset);
    glow::Tensor glowTensor(ptTensor.data_ptr(), &glowType);

    // If tensor is of UInt8QTy kind, convert it to Int8QTy.
    if (glowTensor.getElementType() == ElemKind::UInt8QTy) {
      auto handle = glowTensor.getHandle<uint8_t>();
      glow::Tensor newTensor(glow::ElemKind::Int8QTy, glowTensor.dims(),
                             glowTensor.getType().getScale(),
                             glowTensor.getType().getOffset() -
                                 UINT8_TO_INT8_SHIFT);
      auto newHandle = newTensor.getHandle<int8_t>();
      for (size_t i = 0; i < handle.size(); i++) {
        newHandle.raw(i) =
            static_cast<int8_t>((int32_t)handle.raw(i) - UINT8_TO_INT8_SHIFT);
      }
      return newTensor;
    }
    return glowTensor;
  } else if (ptTensor.scalar_type() == at::kDouble) {
    at::Tensor atTensor = ptTensor.to(at::kFloat);
    auto glowType = ptTypeToGlowType(*c10::TensorType::create(atTensor));
    return glow::Tensor(atTensor.data_ptr(), &glowType).clone();
  } else {
    auto glowType = ptTypeToGlowType(*c10::TensorType::create(ptTensor));
    return glow::Tensor(ptTensor.data_ptr(), &glowType);
  }
}

// Preprocess jit module to prepare for lowering. Here we leverage JIT freeze
// API to cleanup the IR after IR rewrites.
void modelPreprocessing(torch::jit::Module &model,
                        const std::string &method_name) {
  auto graph =
      toGraphFunction(model.get_method(method_name).function()).graph();

  torch::jit::CanonicalizeOps(graph);
  detail::rewriteQuantizedLinear(graph);

  model = torch::jit::freeze_module(model);
}

// Similar to glowAOTFusion() however supports multiple Glow subgraphs and
// runners. We'd still need both since in some cases we may not be able to infer
// the entire model and would leverage glowAOTFusion() to run the partially
// lowered model.
void glowAOTFusionWithShapeInference(
    torch::jit::Module &model, const InputMetaStack &metaStack,
    runtime::DeferredWeightLoader *loader,
    const PyTorchLoaderSettings &settings, std::string method_name,
    const std::unordered_map<int, std::string> &batchShapes,
    std::shared_ptr<std::string> glowAOTSerializationSpecStrPtr,
    std::shared_ptr<std::string> glowAOTSerializationModelStrPtr,
    const std::string &serializationSpec, const std::string &onnxModelFile,
    c10::optional<PostFusionProcessFn> postFusionProcessFn,
    const c10::optional<ModelCompilationConfigOverride>
        &modelCompilationConfigOverride) {
  auto graph =
      toGraphFunction(model.get_method(method_name).function()).graph();

  // create some fake inputs to run shape inference.
  // Usually users provide one set of inputs for the entire
  // model and expect the model can be lowered. However there
  // are cases where we cannot lower the entire model.
  // There could be multiple fused graphs and the inputs to
  // each fused graph could be different from the metaStack user
  // provided. Therefore we leverage shape inference to populate
  // shape and type information over the entire model so we
  // could lower whatever we want.
  std::vector<torch::jit::IValue> inputs;
  for (const auto &i : metaStack.inputMetas) {
    inputs.emplace_back(
        torch::empty(i.dims, torch::TensorOptions().dtype(i.type)));
  }

  const at::ArrayRef<torch::jit::IValue> inputRefs(inputs);

  // The base symbol of all.
  std::string baseSymbol = glow::getGlowSymbol(nullptr).toQualString();

  // There could be multiple glow fusion nodes created.
  glow::glowCustomFuse(graph, settings);

  // If anything needs to be adjusted after graph fusion,
  // then this callback can do it.
  if (postFusionProcessFn) {
    (*postFusionProcessFn)(graph);
  }

  ShapeInferenceEngine shapeInf(*graph, inputRefs, baseSymbol, true);
  auto e = shapeInf.run();
  if (e) {
    LOG(ERROR) << ERR_TO_STRING(std::move(e));
  }

  const auto &shapeMap = shapeInf.getVariableMap();

  BatchShapesMapType batchShapesMap;
  if (batchShapes.size() > 0) {
    batchShapesMap =
        parseBatchShapeMapFromInputMeta(graph, batchShapes, baseSymbol);
    LOG(INFO) << "Finish populating batch shape map with batch size: "
              << batchShapes.size();
  }

  // this is a fuser subgraph to lower
  std::shared_ptr<torch::jit::Graph> subgraph;
  // Create one cachingGraphRunner for each fused graph.
  for (auto *node : graph->nodes()) {
    std::string kind = node->kind().toQualString();

    if (kind == baseSymbol) { // Found a match
      assert(node->hasAttribute(torch::jit::attr::Subgraph));
      subgraph = node->g(torch::jit::attr::Subgraph);
      // Find the index of this fusion node
      int idx = findIndex(node);

      // create the graph runner and warm its cache, this graph runner will be
      // picked up during operator registration
      // All Glow fusion nodes would have the same kind and there isn't a good
      // native way to differentiate them at runtime. Therefore we scan the
      // graph containing Glow fusion nodes and index each of them. The index
      // would be used as part of the key to find corresponding
      // cachingGraphRunner.
      auto runner = glow::setGraphRunnerForKey(
          kind + std::to_string(idx), [subgraph, settings] {
            return std::make_unique<glow::CachingGraphRunner>(
                subgraph, getHostManager(settings), settings,
                /*useRunOnly*/ true);
          });

      InputMetaStack metaStackForCompilation;
      auto graphInputValues = subgraph->inputs();

      for (size_t i = 0; i < graphInputValues.size(); ++i) {
        const torch::jit::Value *inputValue = graphInputValues[i];
        auto itr = shapeMap.find(inputValue);
        if (itr == shapeMap.end()) {
          LOG(ERROR) << "Node " << node->kind().toQualString() << " input " << i
                     << " Not found in the shape map!";
        }
        // Only support tensor input for now
        // TODO Add support for other input types, e.g., tensor[]
        metaStackForCompilation.inputMetas.emplace_back(
            itr->second.dtype, itr->second.shape<TensorShape>());
      }

      REPORT_AND_EXIT_ON_ERR(runner->warmCache(
          {metaStackForCompilation}, settings, loader,
          /*useMaxSizeCompilation*/ true, /*useDeserialize*/ false,
          /*nameToFunctions*/ nullptr, glowAOTSerializationSpecStrPtr,
          glowAOTSerializationModelStrPtr, serializationSpec, onnxModelFile,
          modelCompilationConfigOverride));

      if (batchShapesMap.size() > 0) {
        auto graphOutputValues = subgraph->outputs();
        e = runner->warmupGraphOutputShapeMap(graphOutputValues,
                                              batchShapesMap);
        if (e) {
          LOG(ERROR) << ERR_TO_STRING(std::move(e));
        } else {
          LOG(INFO) << "Finish warming up shape map: " << batchShapesMap.size();
        }
        e = runner->setNominalInputIndex(graphInputValues, batchShapesMap);
        if (e) {
          LOG(ERROR) << ERR_TO_STRING(std::move(e));
        } else {
          LOG(INFO) << "Finish Setting up the nomialInputIndex";
        }
      }
    }
  }
  if (!subgraph) {
    // at least one
    LOG(ERROR) << "Cannot create a Glow fusion subgraph";
  }
}

void glowAOTFusion(torch::jit::Module &model, const std::string &inputMetaStr,
                   runtime::DeferredWeightLoader *loader,
                   const PyTorchLoaderSettings &settings,
                   std::string method_name,
                   const std::unordered_map<int, std::string> &batchShapes,
                   std::shared_ptr<std::string> glowAOTSerializationSpecStrPtr,
                   std::shared_ptr<std::string> glowAOTSerializationModelStrPtr,
                   const std::string &serializationSpec,
                   const std::string &onnxModelFile,
                   c10::optional<PostFusionProcessFn> postFusionProcessFn,
                   const c10::optional<ModelCompilationConfigOverride>
                       &modelCompilationConfigOverride) {
  InputMetaStack metaStack = glow::loadInputMeta(inputMetaStr);

  modelPreprocessing(model, method_name);

  // In Glow AOT serialization (i.e., settings.saveGlowIRIntoONNX = true), we
  // always enable inferShapeForCompilation
  if (FLAGS_inferShapeForCompilation || settings.saveGlowIRIntoONNX) {
    return glowAOTFusionWithShapeInference(
        model, metaStack, loader, settings, method_name, batchShapes,
        glowAOTSerializationSpecStrPtr, glowAOTSerializationModelStrPtr,
        serializationSpec, onnxModelFile, postFusionProcessFn,
        modelCompilationConfigOverride);
  }

  // We assume the model is flattened and only one graph will be lowered. In the
  // future we may need to support multiple graphs.
  auto graph =
      toGraphFunction(model.get_method(method_name).function()).graph();

  c10::Symbol symbol = glow::getGlowSymbol(graph);
  glow::registerGlowOp(symbol);
  glow::glowCustomFuse(graph, settings, symbol);

  // If anything needs to be adjusted after graph fusion,
  // then this callback can do it.
  if (postFusionProcessFn) {
    (*postFusionProcessFn)(graph);
  }

  // this is the fuser subgraph to lower
  std::shared_ptr<torch::jit::Graph> subgraph;
  for (auto *node : graph->nodes()) {
    if (node->kind().toQualString() == symbol.toQualString()) {
      assert(node->hasAttribute(torch::jit::attr::Subgraph));
      subgraph = node->g(torch::jit::attr::Subgraph);
      break;
    }
  }
  if (!subgraph) {
    MAKE_ERR("Cannot create a Glow fusion subgraph");
  }

  // create the graph runner and warm its cache, this graph runner will be
  // picked up during operator registration
  auto runner =
      glow::setGraphRunnerForKey(symbol.toQualString(), [subgraph, settings] {
        return std::make_unique<glow::CachingGraphRunner>(
            subgraph, getHostManager(settings), settings, /*useRunOnly*/ true);
      });

  auto e = runner->warmCache(
      {metaStack}, settings, loader,
      /*useMaxSizeCompilation*/ true, /*useDeserialize*/ false,
      /*nameToFunctions*/ nullptr, glowAOTSerializationSpecStrPtr,
      glowAOTSerializationModelStrPtr, serializationSpec, onnxModelFile,
      modelCompilationConfigOverride);
  if (e) {
    // If the graph is already compiled previously, warmCache() will report
    // an error but it is fine with our execution. So here we extract the
    // error only.
    LOG(ERROR) << ERR_TO_STRING(std::move(e));
  }
}

static bool &_signalHandlerOverridesEnabled() {
  static bool enabled = false;
  return enabled;
}

void enableSignalHandlerOverrides(bool enable) {
  _signalHandlerOverridesEnabled() = enable;
}

bool signalHandlerOverridesEnabled() {
  return _signalHandlerOverridesEnabled();
}

/// Get a temporary file location given \p name and \p suffix.
Expected<std::string> getTempFileLoc(const std::string &name,
                                     const std::string &suffix) {
  llvm::SmallString<64> path;
  auto tempFileRes =
      llvm::sys::fs::createTemporaryFile("export", name + suffix, path);
  RETURN_ERR_IF_NOT(tempFileRes.value() == 0,
                    "Failed to create temp file to write into.");
  return std::string(path.c_str());
}

BatchShapesMapType parseBatchShapeMapFromInputMeta(
    const std::shared_ptr<struct torch::jit::Graph> &graph,
    const std::unordered_map<int, std::string> &batchShapes,
    const std::string &baseSymbol) {
  BatchShapesMapType batchShapesMap;
  for (auto &it : batchShapes) {
    InputMetaStack inputMetaStack = glow::loadInputMeta(it.second);
    std::vector<torch::jit::IValue> inputs;
    for (const auto &i : inputMetaStack.inputMetas) {
      inputs.emplace_back(
          torch::empty(i.dims, torch::TensorOptions().dtype(i.type)));
    }
    const at::ArrayRef<torch::jit::IValue> inputRefs(inputs);
    ShapeInferenceEngine shapeInf(*graph, inputRefs, baseSymbol, true);
    auto e = shapeInf.run();
    if (e) {
      LOG(ERROR) << ERR_TO_STRING(std::move(e));
    }
    const auto &shapeMap = shapeInf.getVariableMap();
    batchShapesMap[it.first] = shapeMap;
  }
  return batchShapesMap;
}

} // namespace glow
