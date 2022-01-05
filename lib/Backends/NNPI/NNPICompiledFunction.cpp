/*
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

#include "NNPICompiledFunction.h"
#include "DebugMacros.h"
#include "Importer.h"
#include "NNPI.h"
#include "NNPIOptions.h"
#include "nnpi_transformer.h"

#include "glow/Backend/BackendUtils.h"
#include "glow/Flags/Flags.h"

#include "CustomKernels/GetNNPIKernels.h"

#include <sstream>

#include "llvm/ADT/StringSet.h"

using namespace glow;

NNPIDeviceNetworkConfig glow::parseDeviceNetworkConfig(
    const NNPICompilationOptions &compilationOptions) {
  NNPIDeviceNetworkConfig cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.pnpHints.ringFrequencyPrio = compilationOptions.ringPrio;
  cfg.pnpHints.iceBOFrequencyPrio[0] = compilationOptions.iceBOPrio0;
  cfg.pnpHints.iceBOFrequencyPrio[1] = compilationOptions.iceBOPrio1;
  cfg.pnpHints.iceBOFrequencyPrio[2] = compilationOptions.iceBOPrio2;
  cfg.pnpHints.iceBOFrequencyPrio[3] = compilationOptions.iceBOPrio3;
  cfg.pnpHints.iceBOFrequencyPrio[4] = compilationOptions.iceBOPrio4;
  cfg.pnpHints.iceBOFrequencyPrio[5] = compilationOptions.iceBOPrio5;
  cfg.pnpHints.DDRBandwidth = compilationOptions.ddrBandwidth;
  return cfg;
}

Error NNPICompiledFunction::updateCompilationConfigFromOptions(
    NNPICompilationOptions &compilationOptions, bool requiresDSPKernels) {
  if (compilationOptions.showVars) {
    LOG(INFO) << compilationOptions.dumpStatus();
  }

  std::string dspKernelsFile;

  if (dspKernelsFile.empty() &&
      !compilationOptions.customDspKernelsFile.get().empty()) {
    dspKernelsFile = compilationOptions.customDspKernelsFile.get();
    if (!dspKernelsFile.empty()) {
      LOG(INFO) << "Found DSP library from "
                   "NNPICompilationOptions: "
                << dspKernelsFile;
    }
  }

  // If a kernels file was already provided then use that instead of fetching
  // custom kernels so that the explicitly specified kernels file is honored.
  if (dspKernelsFile.empty() && requiresDSPKernels) {
    dspKernelsFile = GetNNPIKernels::getCompiledDSPKernelsFilePath();
    if (!dspKernelsFile.empty()) {
      LOG(INFO) << "Found DSP library from "
                   "NNPIBackend::getDSPKernelsPrivate: "
                << dspKernelsFile;
    }
  }

  if (dspKernelsFile.empty() && requiresDSPKernels) {
    // If kernels file was not provided then check if pointer to lib content
    // was provided in compilation options.
    if (compilationOptions.customDspKernelsSize) {
      config_.sizeCustomDspLib = compilationOptions.customDspKernelsSize.get();
      config_.customDspLib = reinterpret_cast<uint8_t *>(
          compilationOptions.customDspKernelsLibPtr.get());
      LOG(INFO) << "Loading DSP library from NNPICompilationOptions with size "
                << compilationOptions.customDspKernelsSize;
    } else {
      return MAKE_ERR(
          "Neither DSP kernels file found, nor pointer to lib provided."
          "Atleast one of them is needed to run function containing DSP "
          "kernels");
    }
  } else {
    std::strncpy(config_.customDspKernelsFile, dspKernelsFile.c_str(),
                 dspKernelsFile.size());
  }

  // Handle device version.
  ASSIGN_VALUE_OR_RETURN_ERR(
      config_.deviceType,
      NNPIOptions::getDeviceVersion(compilationOptions.inferOnDevice,
                                    compilationOptions.deviceVersion));

  if (compilationOptions.iceCores > 0) {
    config_.numCoresToUse = static_cast<uint32_t>(compilationOptions.iceCores);
  }
  if (!compilationOptions.debugCompileConfigFile.get().empty()) {
    strncpy(config_.debugConfigFile,
            compilationOptions.debugCompileConfigFile.get().c_str(),
            sizeof(config_.debugConfigFile));
  }

  config_.disableSLSOffloadToIA = compilationOptions.disableSLSOffloadToIA;
  config_.enableLightweightCompilation = compilationOptions.lightCompilation;
  config_.dumpDotFiles = compilationOptions.dumpDotFiles;

  config_.forceWeightsOutOfLLC = compilationOptions.forceWeightsOutOfLLC;
  config_.enableFCDynamicQuantizationAllSA =
      compilationOptions.enableFCDynamicQuantizationAllSA;
  config_.disableSlsAllLenOneCalcAtRunTime =
      compilationOptions.disableSlsAllLenOneCalcAtRunTime;
#if NNPI_MAJOR_VERSION >= 1 && NNPI_MINOR_VERSION >= 1
  config_.enableESUnifyAdditionalPass =
      compilationOptions.enableESUnifyAdditionalPass;

  config_.enableLayerSplitter = compilationOptions.enableLayerSplitter;
  config_.enableConvSpatialSplitter =
      compilationOptions.enableConvSpatialSplitter;
  config_.enableConvBatchSplitter = compilationOptions.enableConvBatchSplitter;
  config_.disableWeightsInPool = compilationOptions.disableWeightsInPool;
#endif

#if NNPI_MAJOR_VERSION >= 1 && NNPI_MINOR_VERSION >= 7
  config_.dumpIntermediate = compilationOptions.dumpIntermediate;
  config_.numParallelDeciderCompilation =
      compilationOptions.numDeciderCompilation;
  config_.weightsThresholdForWeightSharing =
      compilationOptions.thresholdDisableWeightsPool;
#endif // NNPI >= 1.7
  return Error::success();
}

/// Inserts into \p opts a mapping from \p optKey to \p optVal. If \p optKey
/// already existed in \p opts then logs a warning about overriding it for \p F.
static void insertOptLogOverride(const Function &F,
                                 BackendSpecificOptions &opts,
                                 const std::string &optKey,
                                 const std::string &optVal) {
  auto it = opts.find(optKey);
  if (it != opts.end()) {
    LOG(WARNING) << optKey << " was set to " << it->second << " for network "
                 << F.getName().data() << "; overriding it with " << optVal;
  }
  opts[optKey] = optVal;
}

Error NNPICompiledFunction::setupCompilationHints(
    const Function *F, const BackendSpecificNodeInfo &backendSpecificNodeInfo) {
  // If there's no info to validate for this Function then return early.
  auto funNodeInfoIt = backendSpecificNodeInfo.find(F);
  if (funNodeInfoIt == backendSpecificNodeInfo.end()) {
    return Error::success();
  }
  auto &currFunInfo = funNodeInfoIt->second;

  // Gather all Node names to more easily/efficiently validate extraEdges.
  llvm::StringSet<> allNodeNames;
  for (const Node &N : F->getNodes()) {
    allNodeNames.insert(N.getName().str());
    if (const ConcatNode *CN = llvm::dyn_cast<ConcatNode>(&N)) {
      for (dim_t i = 0, e = CN->getNumInputs(); i < e; i++) {
        allNodeNames.insert(N.getName().str() + "@copy_" + std::to_string(i));
      }
    }
  }

  // Create hints here before copying into config_, as we don't know how many
  // there are a priori.
  std::vector<NNPICompilationHint> hints;
  for (const auto &nodeInfoPair : currFunInfo) {
    const Node *N = nodeInfoPair.first;
    RETURN_ERR_IF_NOT(N->getParent() == F,
                      "Node mapped to this Function in backendSpecificNodeInfo "
                      "has incorrect parent.");

    const auto &nodeInfo = nodeInfoPair.second;

    // Read core assignments
    auto coreAssignmentsIt = nodeInfo.find(coreAssignmentsKey);
    auto coreAssignmentsSuffixIt = nodeInfo.find(coreAssignmentsSuffixKey);
    if (coreAssignmentsIt != nodeInfo.end()) {
      auto coreAssignments = coreAssignmentsIt->second;
      if (coreAssignmentsSuffixIt != nodeInfo.end()) {
        auto coreAssignmentsSuffix = coreAssignmentsSuffixIt->second;
        RETURN_ERR_IF_NOT(
            coreAssignments.size() == coreAssignmentsSuffix.size(),
            strFormat("Node %s coreAssignmentsSuffix has length "
                      "%zu, but coreAssignments has length %zu",
                      N->getName().data(), coreAssignmentsSuffix.size(),
                      coreAssignments.size()));
      }

      for (dim_t i = 0; i < coreAssignments.size(); i++) {
        int core;
        ASSIGN_VALUE_OR_RETURN_ERR(core, getIntFromStr(coreAssignments[i]));
        RETURN_ERR_IF_NOT(core >= 0 && core <= 11,
                          "Core assignment must be [0-11]");
        NNPICompilationHint hint;
        hint.type = NNPI_HINT_ICE_CORE_PLACEMENT;
        std::string opName = N->getName().str();
        if (coreAssignmentsSuffixIt != nodeInfo.end()) {
          auto coreAssignmentsSuffix = coreAssignmentsSuffixIt->second;
          if (i < coreAssignmentsSuffix.size()) {
            opName = opName + coreAssignmentsSuffix[i];
          }
        }
        strncpy(hint.iceCorePlacement.opName, opName.c_str(),
                sizeof(NNPIObjectName));
        hint.iceCorePlacement.iceCore = core;
        hints.emplace_back(hint);
      }
    }

    // Read tensor assignments
    auto tensorAssignmentNamesIt = nodeInfo.find(tensorAssignmentNamesKey);
    auto tensorAssignmentValuesIt = nodeInfo.find(tensorAssignmentValuesKey);
    if ((tensorAssignmentNamesIt != nodeInfo.end()) &&
        (tensorAssignmentValuesIt != nodeInfo.end())) {
      auto tensorAssignmentNames = tensorAssignmentNamesIt->second;
      auto tensorAssignmentValues = tensorAssignmentValuesIt->second;
      RETURN_ERR_IF_NOT(
          tensorAssignmentNames.size() == tensorAssignmentValues.size(),
          strFormat("Node %s tensorAssignmentsNames has length %zu, but "
                    "tensorAssignmentValues has length %zu",
                    N->getName().data(), tensorAssignmentNames.size(),
                    tensorAssignmentValues.size()));

      for (dim_t i = 0; i < tensorAssignmentNames.size(); i++) {
        const std::string &memoryLevel = tensorAssignmentValues[i];
        RETURN_ERR_IF_NOT((memoryLevel == "SRAM") || (memoryLevel == "LLC") ||
                              (memoryLevel == "DRAM"),
                          strFormat("Memory level must be either SRAM, LLC, or "
                                    "DRAM. Unknown level: %s",
                                    memoryLevel.data()));

        const std::string &tensorName = tensorAssignmentNames[i];

        NNPICompilationHint hint;
        hint.type = NNPI_HINT_TENSOR_PLACEMENT;
        strncpy(hint.tensorPlacement.tensorName, tensorName.c_str(),
                sizeof(NNPIObjectName));
        hint.tensorPlacement.allocationType =
            (memoryLevel == "SRAM")  ? NNPI_ALLOCATION_SRAM
            : (memoryLevel == "LLC") ? NNPI_ALLOCATION_LLC
                                     : NNPI_ALLOCATION_DRAM;
        // positive priority is required to enforce the hints
        hint.tensorPlacement.priority = 1.0f;
        hints.emplace_back(hint);
      }
    }

    // Read extra edges
    auto extraEdgesTargetNameIt = nodeInfo.find(extraEdgesTargetNameKey);
    auto extraEdgesTargetSuffixIt = nodeInfo.find(extraEdgesTargetSuffixKey);
    auto extraEdgesSourceSuffixIt = nodeInfo.find(extraEdgesSourceSuffixKey);

    if (extraEdgesTargetNameIt != nodeInfo.end()) {
      auto extraEdgesTargetName = extraEdgesTargetNameIt->second;
      if (extraEdgesTargetSuffixIt != nodeInfo.end()) {
        auto extraEdgesTargetSuffix = extraEdgesTargetSuffixIt->second;
        RETURN_ERR_IF_NOT(
            extraEdgesTargetName.size() == extraEdgesTargetSuffix.size(),
            strFormat("Node %s extraEdgesTargetSuffix has length %zu, but "
                      "extraEdgesTargetName has length %zu",
                      N->getName().data(), extraEdgesTargetSuffix.size(),
                      extraEdgesTargetName.size()));
      }

      if (extraEdgesSourceSuffixIt != nodeInfo.end()) {
        auto extraEdgesSourceSuffix = extraEdgesSourceSuffixIt->second;
        RETURN_ERR_IF_NOT(
            extraEdgesTargetName.size() == extraEdgesSourceSuffix.size(),
            strFormat("Node %s extraEdgesSourceSuffix has length %zu, but "
                      "extraEdgesTargetName has length %zu",
                      N->getName().data(), extraEdgesSourceSuffix.size(),
                      extraEdgesTargetName.size()));
      }

      for (dim_t i = 0; i < extraEdgesTargetName.size(); i++) {
        std::string opName = std::string(N->getName());
        if (extraEdgesSourceSuffixIt != nodeInfo.end()) {
          auto extraEdgesSourceSuffix = extraEdgesSourceSuffixIt->second;
          opName = opName + extraEdgesSourceSuffix[i];
        }

        std::string edgeName = extraEdgesTargetName[i];
        if (allNodeNames.count(extraEdgesTargetName[i]) == 0) {
          LOG(WARNING) << "Discarding edge targeting non-existent node "
                       << extraEdgesTargetName[i];
          continue;
        }
        if (extraEdgesTargetSuffixIt != nodeInfo.end()) {
          auto extraEdgesTargetSuffix = extraEdgesTargetSuffixIt->second;
          edgeName = edgeName + extraEdgesTargetSuffix[i];
        }

        NNPICompilationHint hint;
        hint.type = NNPI_HINT_OP_DEPENDENCY;
        strncpy(hint.opDependency.opName, opName.c_str(),
                sizeof(NNPIObjectName));
        strncpy(hint.opDependency.dependsOnOpName, edgeName.c_str(),
                sizeof(NNPIObjectName));
        hints.emplace_back(hint);
      }
    }
  }

  config_.numCompilationHints = hints.size();
  const size_t hintsBytes = hints.size() * sizeof(NNPICompilationHint);
  config_.compilationHints = (NNPICompilationHint *)malloc(hintsBytes);
  if (hintsBytes > 0) {
    memcpy(config_.compilationHints, hints.data(), hintsBytes);
  }

  return Error::success();
}

Error NNPICompiledFunction::compile(Function *F, const BackendOptions &opts) {
  BackendOptions newOpts = opts;
  if (opts.backendHints.executionUnits) {
    insertOptLogOverride(*F, newOpts.backendSpecificOpts, "NNPI_IceCores",
                         std::to_string(opts.backendHints.executionUnits));
  }

  if (glow::nnpi::flags::DumpCompilerData) {
    const std::string icetFName =
        std::string("icet_file_") + F->getName().str();
    insertOptLogOverride(*F, newOpts.backendSpecificOpts, "NNPI_CompiledFile",
                         icetFName);
  }

  if (glow::nnpi::flags::UsePerPartitionIcetConfig) {
    const std::string icetConfigFName =
        std::string("icet_config_") + F->getName().str() + std::string(".json");
    insertOptLogOverride(*F, newOpts.backendSpecificOpts,
                         "NNPI_CompilationDebugConfigFile", icetConfigFName);
  }

  for (const auto &keyValPair : newOpts.backendSpecificOpts) {
    LOG(INFO) << "Backend-specific option " << keyValPair.first << " set to "
              << keyValPair.second << " for network " << F->getName().data();
  }

  compilationOptions_ = NNPICompilationOptions(newOpts.backendSpecificOpts);

  if (compilationOptions_.compileOutputPostfix) {
    compilationFileName_ = compilationOptions_.compiledFile.get() + "_" +
                           std::string(F->getName());
  } else {
    compilationFileName_ = compilationOptions_.compiledFile.get();
  }
  LOG_IF_NOT_RETURN_LLVMERROR(
      compilationFileName_.length() < NNPI_MAX_STRING_LEN, "Bad filename");

  NNPIImporter importer(compilationOptions_);
  // requiresDSPKernels set by importFunction.
  bool requiresDSPKernels = false;
  network_ = importer.importFunction(F, newOpts, requiresDSPKernels);
  iaExtensionPaths_ = importer.getIAExtensionPaths();
  iaExtensionLibs_ = importer.getIAExtensionLibs();

  LOG_IF_INVALID_HANDLE_RETURN_LLVMERROR(network_, "Failed to import function");
  // Setting the network name.
  std::string networkName = compilationFileName_;
  if (compilationFileName_.empty()) {
    networkName = F->getName().str();
  }
  ASSERT_LOG_NNPI_ERROR(nnpiNetworkSetName(network_, networkName.c_str()),
                        "Failed to set NNPI network name");

  // Apply optimizations.
  NNPIOptimizationConfig optConf;
  std::memset(&optConf, 0, sizeof(NNPIOptimizationConfig));
  optConf.lstmReconstruction = 1;
  optConf.reorderTransposeConvert = 1;
  if (!compilationOptions_.disableConstFolding) {
    optConf.constantFolding = 1;
  }

  DBG_MEM_USAGE("NNPICompiledFunction call optimize <<");
  LOG_NNPI_IF_ERROR_RETURN_LLVMERROR(nnpiNetworkOptimize(network_, &optConf),
                                     "Failed NNPI API Optimize");
  DBG_MEM_USAGE("NNPICompiledFunction call get compilation config <<");
  LOG_NNPI_IF_ERROR_RETURN_LLVMERROR(nnpiGetDefaultCompilationConfig(&config_),
                                     "Failed NNPI API Read Config");

  RETURN_IF_ERR(updateCompilationConfigFromOptions(compilationOptions_,
                                                   requiresDSPKernels));

  RETURN_IF_ERR(setupCompilationHints(F, newOpts.backendSpecificNodeInfo));

  // Collect input/output names.
  {
    size_t numInputs, numOutputs;
    NNPIObjectName name;
    NNPITensorDesc desc;
    LOG_NNPI_IF_ERROR_RETURN_LLVMERROR(
        nnpiNetworkGetInputNum(network_, &numInputs),
        "Failed to query NNPI network inputs");
    for (size_t i = 0; i < numInputs; i++) {
      LOG_NNPI_IF_ERROR_RETURN_LLVMERROR(
          nnpiNetworkGetInputDesc(network_, i, name, &desc),
          "Failed to query NNPI network inputs");
      inputNames_.push_back(name);
    }
    LOG_NNPI_IF_ERROR_RETURN_LLVMERROR(
        nnpiNetworkGetOutputNum(network_, &numOutputs),
        "Failed to query NNPI network outputs");
    for (size_t i = 0; i < numOutputs; i++) {
      LOG_NNPI_IF_ERROR_RETURN_LLVMERROR(
          nnpiNetworkGetOutputDesc(network_, i, name, &desc),
          "Failed to query NNPI network outputs");
      outputNames_.push_back(name);
    }
  }

  if (compilationOptions_.useIceT || compilationOptions_.inferOnDevice) {
    if (compilationFileName_.empty()) // Compile to memory.
    {
      NNPIStream outFileStream;
      outFileStream.userData = &compiledStream_;
      outFileStream.readCallback = NULL;
      outFileStream.seekCallback = NULL;
      outFileStream.writeCallback = [](const void *ptr, uint64_t size,
                                       uint64_t count,
                                       void *userData) -> uint64_t {
        DBG_MEM_USAGE("NNPICompiledFunction before appending: "
                      << ((size * count) / 1024));
        BlockStream *ss = reinterpret_cast<BlockStream *>(userData);
        size_t wSize = ss->write(static_cast<const char *>(ptr), size * count);
        if (wSize < size * count) {
          return 0;
        } else {
          DBG_MEM_USAGE("NNPICompiledFunction stream appended: "
                        << ((size * count) / 1024) << " KB current size: "
                        << ss->getSize() / 1024 << " KB ");
          return size * count;
        }
      };
      DBG_MEM_USAGE("NNPICompiledFunction call get compile <<");
      if (!opts.useDeserialize) {

        LOG_NNPI_IF_ERROR_RETURN_LLVMERROR(
            nnpiNetworkCompileToStream(network_, &config_, &outFileStream,
                                       NULL),
            "Failed NNPI Compile");
      }
      DBG_MEM_USAGE("NNPICompiledFunction done compile <<");
    } else // Compile to file.
    {
      LOG_NNPI_IF_ERROR_RETURN_LLVMERROR(
          nnpiNetworkCompileToFile(network_, &config_,
                                   compilationFileName_.c_str(), NULL),
          "Failed NNPI Compile");
    }

    // Update compilation info after NNPI compilation.
    if (compilationOptions_.dumpCompilationInfo ||
        compilationOptions_.lightCompilation ||
        flags::DumpBackendSpecificIRJSON) {
      if (!updateCompilationInfo()) {
        // Only issuing a warning (soft fail)
        LOG(WARNING) << "Failed to update NNPI compilation info";
      } else if (compilationOptions_.dumpCompilationInfo) {
        LONG_LOG(INFO, compilationInfo_.dump(networkName));
      }
    }

    if (compilationOptions_.inferOnDevice) {
      DBG_MEM_USAGE("NNPICompiledFunction destroy network");
      // NNPINetwork is not needed anymore on the inferfence api path.
      // Once the complied stream is loaded, query on the network can be done
      // using the host network instead.
      LOG_NNPI_IF_ERROR(nnpiNetworkDestroy(network_),
                        "Failed NNPI Network Destroy");
      network_ = NNPI_INVALID_NNPIHANDLE;
      DBG_MEM_USAGE("NNPICompiledFunction destroy network done");
    }
  }

  // Determine and save what inputs can be treated as partial. Need to do this
  // while we still have access to F.
  for (auto const &P : F->getParent()->getPlaceholders()) {
    if (!usedInFunction(P, F)) {
      continue;
    }
    if (allowsPartialInput(P, F)) {
      partialInputs_.insert(P);
    } else if (requiresPadding(P, F)) {
      paddedInputs_.insert(P);
    }
    if (P->isStatic()) {
      staticInputs_.insert(P);
    }
  }

  // Update device network config.
  devNetConfig_ = parseDeviceNetworkConfig(compilationOptions_);

  return Error::success();
}

NNPICompiledFunction::NNPICompiledFunction(Function *F)
    : CompiledFunction(runtime::RuntimeBundle::create(*F)),
      compilationOptions_({}) {
  std::memset(&config_, 0, sizeof(config_));
  std::memset(&devNetConfig_, 0, sizeof(devNetConfig_));
};

NNPICompiledFunction::~NNPICompiledFunction() {
  if (network_ != NNPI_INVALID_NNPIHANDLE) {
    LOG_NNPI_IF_ERROR(nnpiNetworkDestroy(network_),
                      "Failed NNPI Network Destroy");
  }
  if (config_.compilationHints) {
    free(config_.compilationHints);
  }
}

BlockStream &NNPICompiledFunction::lockCompiledStream() {
  DBG_MEM_USAGE("lockCompiledStream");
  compiledStreamMutex_.lock();
  return compiledStream_;
}

void NNPICompiledFunction::unlockCompiledStream() {
  DBG_MEM_USAGE("unlockCompiledStream");
  compiledStream_.resetRead();
  compiledStreamMutex_.unlock();
}

void NNPICompiledFunction::freeCompilationResources() {
  DBG_MEM_USAGE("[Before] freeCompilationResources ");
  lockCompiledStream();
  compiledStream_.releaseMemory();
  unlockCompiledStream();
  DBG_MEM_USAGE("[After] freeCompilationResources ");
}

bool NNPICompiledFunction::updateCompilationInfo() {
  // Clear existing info.
  compilationInfo_.clear();

  if (network_ == NNPI_INVALID_NNPIHANDLE) {
    LOG(ERROR) << "Invalid NNPINetwork";
    return false;
  }

  // Collect operators.
  uint64_t numOps = 0;
  LOG_NNPI_IF_ERROR_RETURN_FALSE(nnpiNetworkGetOpNum(network_, &numOps),
                                 "Failed to get num ops");
  for (uint64_t op = 0; op < numOps; op++) {
    NNPIOpInfo opInfo;
    LOG_NNPI_IF_ERROR_RETURN_FALSE(nnpiNetworkGetOpInfo(network_, op, &opInfo),
                                   "Failed to get op info");
    NNPICompiledOp compiledOp;
    compiledOp.name = std::string(opInfo.name);
    compiledOp.type = std::string(opInfo.type);
    compiledOp.coreIndex = opInfo.coreIndex;
    compiledOp.iceBo = opInfo.iceBo;
    compiledOp.execType = opInfo.executionType;
    for (uint32_t t = 0; t < opInfo.numTensors; t++) {
      NNPITensorInfo tensorInfo;
      LOG_NNPI_IF_ERROR_RETURN_FALSE(
          nnpiNetworkGetOpTensorInfo(network_, op, t, &tensorInfo),
          "Failed to get tensor info");
      NNPICompiledTensor compiledTensor;
      compiledTensor.name = std::string(tensorInfo.name);
      compiledTensor.type = std::string(tensorInfo.type);
      compiledTensor.allocType = tensorInfo.allocation;
      for (uint32_t d = 0; d < tensorInfo.numAllocations; d++) {
        compiledTensor.possibleAlloc.push_back(
            tensorInfo.possibleAllocation[d]);
      }
      for (uint32_t d = 0; d < tensorInfo.numDims; d++) {
        compiledTensor.shape.push_back(tensorInfo.dims[d]);
      }
      switch (tensorInfo.usage) {
      case NNPI_TENSOR_USAGE_INPUT:
        compiledOp.inputs.push_back(compiledTensor);
        break;
      case NNPI_TENSOR_USAGE_OUTPUT:
        compiledOp.outputs.push_back(compiledTensor);
        break;
      default:
        LOG(WARNING) << "Invalid tensor usage";
        break;
      }
    }
    compilationInfo_.ops.insert({compiledOp.name, compiledOp});
  }

  // Collect dependencies.
  uint64_t numDeps = 0;
  LOG_NNPI_IF_ERROR_RETURN_FALSE(
      nnpiNetworkGetOpDependenciesNum(network_, &numDeps),
      "Failed to get num dependencies");

  for (uint64_t dep = 0; dep < numDeps; dep++) {
    NNPIObjectName src;
    NNPIObjectName dst;
    LOG_NNPI_IF_ERROR_RETURN_FALSE(
        nnpiNetworkGetOpDependency(network_, dep, src, dst),
        "Failed to get op dependency");
    compilationInfo_.opDependencies.push_back(
        {std::string(src), std::string(dst)});
  }

  return true;
}

static const char *dumpAllocType(const NNPI_ALLOCATION_TYPE &allocType) {
  switch (allocType) {
  case NNPI_ALLOCATION_DEFAULT:
    return "Default";
  case NNPI_ALLOCATION_DRAM:
    return "DRAM";
  case NNPI_ALLOCATION_ECC_DRAM:
    return "ECC DRAM";
  case NNPI_ALLOCATION_LLC:
  case NNPI_ALLOCATION_LLC_CLOS0:
  case NNPI_ALLOCATION_LLC_CLOS1:
  case NNPI_ALLOCATION_LLC_CLOS2:
  case NNPI_ALLOCATION_LLC_CLOS3:
    return "LLC";
  case NNPI_ALLOCATION_SRAM:
    return "SRAM";
  case NNPI_ALLOCATION_INTERNAL:
    return "Internal";
  default:
    return "Unknown";
  }
}

std::string NNPICompiledTensor::dump() const {
  std::stringstream stream;
  stream << "name: " << name << ", type: " << type << " (";
  for (const auto &d : shape) {
    stream << d << ",";
  }
  if (shape.size() > 0) {
    stream.seekp(-1, stream.cur);
  }
  stream << "), allocation: ";
  stream << dumpAllocType(allocType);
  return stream.str();
}

std::string NNPICompiledOp::dump() const {
  std::stringstream stream;
  stream << "  [Op] name: " << name << ", type: " << type << ", exec: ";
  switch (execType) {
  case NNPI_EXECUTION_IA:
    stream << "IA";
    break;
  case NNPI_EXECUTION_DSP:
    stream << "DSP";
    break;
  case NNPI_EXECUTION_DELPHI:
    stream << "Delphi";
    break;
  case NNPI_EXECUTION_DSE:
    stream << "DSE";
    break;
  case NNPI_EXECUTION_COMBINED:
    stream << "Combined";
    break;
  case NNPI_EXECUTION_NOT_SET:
    stream << "NotSet";
    break;
  default:
    stream << "Unknown";
    break;
  }
  stream << ", core: " << coreIndex << ", iceBo: " << iceBo << "\n";
  for (const auto &in : inputs) {
    stream << "    [Input] " << in.dump() << "\n";
  }
  for (const auto &out : outputs) {
    stream << "    [Output] " << out.dump() << "\n";
  }

  return stream.str();
}

std::string NNPICompilationInfo::dump(const std::string &functionName) const {
  std::stringstream stream;
  stream << "[Start] NNPI Compilation Info for function: \"" << functionName
         << "\":\n";
  for (const auto &op : ops) {
    stream << op.second.dump();
  }
  for (const auto &dep : opDependencies) {
    stream << "  [Dep] " << dep.first << " -> " << dep.second << "\n";
  }
  stream << "[End] NNPI Compilation Info for function: \"" << functionName
         << "\":\n";

  return stream.str();
}

static const std::string tensorToJSON(const NNPICompiledTensor &tensor) {
  std::stringstream fs;
  fs << "{" << std::endl;
  fs << "\"name\" : \"" << tensor.name << "\"," << std::endl;
  fs << "\"type\" : \"" << tensor.type << "\"," << std::endl;
  fs << "\"alloc\" : \"" << dumpAllocType(tensor.allocType) << "\","
     << std::endl;
  fs << "\"possible_alloc\" : ";
  fs << "[" << std::endl;
  for (auto it = tensor.possibleAlloc.begin(); it != tensor.possibleAlloc.end();
       it++) {
    if (it != tensor.possibleAlloc.begin()) {
      fs << "," << std::endl;
    }
    fs << "\"" << dumpAllocType(*it) << "\" ";
  }
  fs << "]," << std::endl;
  fs << std::endl;
  fs << "\"size\" : " << std::endl;
  fs << "[" << std::endl;
  for (auto it = tensor.shape.begin(); it != tensor.shape.end(); it++) {
    if (it != tensor.shape.begin()) {
      fs << "," << std::endl;
    }
    fs << *it << std::endl;
  }
  fs << "]" << std::endl;
  fs << "}" << std::endl;
  return fs.str();
}

static const std::string
tensorListToJSON(const std::vector<NNPICompiledTensor> &tensors,
                 const std::string &label) {
  std::stringstream fs;
  fs << "  \"" << label << "\": [ " << std::endl;
  for (auto it = tensors.begin(); it != tensors.end(); it++) {
    if (it != tensors.begin()) {
      fs << "," << std::endl;
    }
    fs << tensorToJSON(*it);
  }
  fs << "]" << std::endl;
  return fs.str();
}

static const std::string
opsToJSON(const std::map<std::string, NNPICompiledOp> &ops) {
  std::stringstream fs;
  fs << "  \"ops\": { " << std::endl;
  for (auto it = ops.begin(); it != ops.end(); it++) {
    if (it != ops.begin()) {
      fs << "," << std::endl;
    }
    fs << " \"" << it->second.name << "\": " << std::endl;
    fs << "{" << std::endl;
    fs << tensorListToJSON(it->second.inputs, "inputs");
    fs << "," << std::endl;
    fs << tensorListToJSON(it->second.outputs, "outputs");
    fs << "," << std::endl;
    fs << " \"core\": " << it->second.coreIndex << "," << std::endl;
    fs << " \"type\": \"" << it->second.type << "\"" << std::endl;
    fs << "}" << std::endl;
  }
  fs << "  }" << std::endl;
  return fs.str();
}

static const std::string
edgesToJSON(const std::vector<std::pair<std::string, std::string>> &edges) {
  std::stringstream fs;
  fs << "  \"edges\": [ " << std::endl;
  for (auto it = edges.begin(); it != edges.end(); it++) {
    if (it != edges.begin()) {
      fs << "," << std::endl;
    }
    fs << " \"" << it->first << "\",\"" << it->second << "\"";
  }
  fs << "  ]" << std::endl;
  return fs.str();
}

const std::string NNPICompiledFunction::toJSON() const {
  std::stringstream fs;
  fs << "{" << std::endl;
  fs << opsToJSON(compilationInfo_.ops) << std::endl;
  fs << "," << std::endl;
  fs << edgesToJSON(compilationInfo_.opDependencies) << std::endl;
  fs << "}" << std::endl;
  return fs.str();
}

std::unique_ptr<BlockStreamBase> NNPICompiledFunction::serialize() {
  compiledStream_.resetRead();
  return std::make_unique<BlockStream>(compiledStream_);
}

Error NNPICompiledFunction::deserialize(
    const std::vector<char> &serializedData) {
  compiledStream_.releaseMemory();
  const char *buffer = reinterpret_cast<const char *>(serializedData.data());
  compiledStream_.write(buffer, serializedData.size());
  return Error::success();
}
