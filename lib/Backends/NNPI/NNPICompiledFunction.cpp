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

#include <fstream>

#include "llvm/ADT/StringSet.h"

using namespace glow;

namespace glow {
namespace onnxifi {
extern bool GlowDumpNNPICompilerData;
extern bool GlowUsePerPartitionIcetConfig;

} // namespace onnxifi
} // namespace glow

/// Looks for device stepping and sets it if possible in \p compilationOptions.
static void trySetDeviceVersion(NNPICompilationOptions &compilationOptions) {
  std::ifstream inFile;
  constexpr char stepLoc[] = "/sys/class/nnpi/nnpi0/card_stepping";
  inFile.open(stepLoc);
  if (!inFile.good() || inFile.eof()) {
    LOG(INFO) << strFormat("Could not find device steppping at %s\n", stepLoc);
    return;
  }

  // Only value in the file should be a single int for which step we're using.
  std::string stepping;
  getline(inFile, stepping);
  inFile.close();

  auto devVerOrErr = getIntFromStr(stepping);
  if (ERR_TO_BOOL(devVerOrErr.takeError(), /* log */ false)) {
    LOG(INFO) << strFormat("Invalid value for stepping at %s: '%s'\n", stepLoc,
                           stepping.data());
    return;
  }
  // Stepping is off by one vs. deviceVersion.
  compilationOptions.deviceVersion.setVal(*devVerOrErr + 1);
}

/// Update device network config from the compilation config
static NNPIDeviceNetworkConfig parseDeviceNetworkConfig(
    const glow::NNPICompilationOptions &compilationOptions) {
  NNPIDeviceNetworkConfig cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.pnpHints.ringFrequencyPrio = compilationOptions.ringPrio;
  cfg.pnpHints.iceBOFrequencyPrio[0] = compilationOptions.iceBOPrio0;
  cfg.pnpHints.iceBOFrequencyPrio[1] = compilationOptions.iceBOPrio1;
  cfg.pnpHints.iceBOFrequencyPrio[2] = compilationOptions.iceBOPrio2;
  cfg.pnpHints.iceBOFrequencyPrio[3] = compilationOptions.iceBOPrio3;
  cfg.pnpHints.iceBOFrequencyPrio[4] = compilationOptions.iceBOPrio4;
  cfg.pnpHints.iceBOFrequencyPrio[5] = compilationOptions.iceBOPrio5;
  cfg.pnpHints.IAFrequencyPrio[0] = compilationOptions.iaPrio0;
  cfg.pnpHints.IAFrequencyPrio[1] = compilationOptions.iaPrio1;
  cfg.pnpHints.DDRBandwidth = compilationOptions.ddrBandwidth;
  return cfg;
}

Error NNPICompiledFunction::updateCompilationConfigFromOptions(
    NNPICompilationOptions &compilationOptions) {
  if (compilationOptions.showVars) {
    LOG(INFO) << compilationOptions.dumpStatus();
  }
  if (!compilationOptions.customDspKernelsFile.get().empty()) {
    std::strncpy(config_.customDspKernelsFile,
                 compilationOptions.customDspKernelsFile.get().c_str(),
                 sizeof(config_.customDspKernelsFile));
  }

  // Handle device version.
  if (compilationOptions.inferOnDevice &&
      compilationOptions.deviceVersion == -1) {
    trySetDeviceVersion(compilationOptions);
  }

  if (compilationOptions.deviceVersion > 0) {
    switch (compilationOptions.deviceVersion) {
    case 1:
      config_.deviceType = NNPI_1000_A;
      break;
    case 2:
      config_.deviceType = NNPI_1000_B;
      break;
    case 3:
      config_.deviceType = NNPI_1000_C;
      break;
    default:
      LOG_IF_NOT_RETURN_LLVMERROR(
          false, "INVALID NNPI_DEVICE_VERSION, valid values are 1,2,3");
    }
  }

  if (compilationOptions.iceCores > 0) {
    config_.numCoresToUse = static_cast<uint32_t>(compilationOptions.iceCores);
  }
  if (!compilationOptions.debugCompileConfigFile.get().empty()) {
    strncpy(config_.debugConfigFile,
            compilationOptions.debugCompileConfigFile.get().c_str(),
            sizeof(config_.debugConfigFile));
  }

  config_.disableSLSOnIA = compilationOptions.disableSLSOnIA;
  config_.enableLightweightCompilation = compilationOptions.lightCompilation;
  config_.dumpDotFiles = compilationOptions.dumpDotFiles;

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
    for (const auto &keyOptsPair : nodeInfoPair.second) {
      const llvm::StringRef &key = keyOptsPair.getKey();
      const std::vector<std::string> &opts = keyOptsPair.getValue();

      RETURN_ERR_IF_NOT(key != numParallelChunksKey,
                        "Should have processed and removed all " +
                            std::string(numParallelChunksKey));

      RETURN_ERR_IF_NOT(key != parallelTransformKindKey,
                        "Should have processed and removed all " +
                            std::string(parallelTransformKindKey));

      RETURN_ERR_IF_NOT(N->getName().size() < sizeof(NNPIObjectName),
                        "Name lengths must fit inside NNPIObjectName.");

      if (key == coreAssignmentsKey) {
        if (const ConcatNode *CN = llvm::dyn_cast<ConcatNode>(N)) {
          RETURN_ERR_IF_NOT(opts.size() == CN->getInputs().size(),
                            "Should have same number of " +
                                std::string(coreAssignmentsKey) + " (" +
                                std::to_string(opts.size()) +
                                ") as inputs to " + N->getName().str() + " (" +
                                std::to_string(CN->getInputs().size()) + ")");
        } else {
          RETURN_ERR_IF_NOT(
              opts.size() == 1,
              strFormat("Should have only a single coreAssignment for %s",
                        N->getName().data()));
        }
        for (size_t i = 0, e = opts.size(); i < e; i++) {
          int core;
          ASSIGN_VALUE_OR_RETURN_ERR(core, getIntFromStr(opts[i]));
          RETURN_ERR_IF_NOT(core >= 0 && core <= 11,
                            "Core assignment must be [0-11]");
          NNPICompilationHint hint;
          hint.type = NNPI_HINT_ICE_CORE_PLACEMENT;
          const std::string opName =
              N->getName().str() +
              (opts.size() == 1 ? "" : ("@copy_" + std::to_string(i)));
          strncpy(hint.iceCorePlacement.opName, opName.c_str(),
                  sizeof(NNPIObjectName));
          hint.iceCorePlacement.iceCore = core;
          hints.emplace_back(hint);
        }
      }

      if (key == extraEdgesKey) {
        for (const std::string &edgeNameInp : opts) {
          ExtraEdgeSplitPair sourceEdgePair;
          std::string opName = std::string(N->getName());
          std::string edgeName = edgeNameInp;
          ASSIGN_VALUE_OR_RETURN_ERR(sourceEdgePair,
                                     getExtraEdgeSourceSplitPair(edgeNameInp));
          if (sourceEdgePair.hasSplit) {
            RETURN_ERR_IF_NOT(
                llvm::isa<ConcatNode>(N),
                "Extra edge " + edgeName +
                    " has a source split, but was not sourced from either a "
                    "parallelized node, or a Concat node.");
            opName =
                opName + "@copy_" + std::to_string(sourceEdgePair.splitNum);
            edgeName = sourceEdgePair.label;
          }

          RETURN_ERR_IF_NOT(allNodeNames.count(edgeName),
                            "Extra edge " + edgeName +
                                " is not mapped to a current Node name.");
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
  }

  config_.numCompilationHints = hints.size();
  const size_t hintsBytes = hints.size() * sizeof(NNPICompilationHint);
  config_.compilationHints = (NNPICompilationHint *)malloc(hintsBytes);
  memcpy(config_.compilationHints, hints.data(), hintsBytes);

  return Error::success();
}

Error NNPICompiledFunction::compile(Function *F, const BackendOptions &opts) {
  BackendOptions newOpts = opts;
  if (opts.backendHints.executionUnits) {
    insertOptLogOverride(*F, newOpts.backendSpecificOpts, "NNPI_IceCores",
                         std::to_string(opts.backendHints.executionUnits));
  }

  if (glow::onnxifi::GlowDumpNNPICompilerData) {
    const std::string icetFName =
        std::string("icet_file_") + F->getName().str();
    insertOptLogOverride(*F, newOpts.backendSpecificOpts, "NNPI_CompiledFile",
                         icetFName);
  }

  if (glow::onnxifi::GlowUsePerPartitionIcetConfig) {
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
  network_ = importer.importFunction(F, newOpts);
  iaExtensionPaths_ = importer.getIAExtensionPaths();

  LOG_IF_INVALID_HANDLE_RETURN_LLVMERROR(network_, "Failed to import function");
  // Setting the network name.
  std::string networkName = compilationFileName_;
  if (compilationFileName_.empty()) {
    networkName = F->getName();
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

  auto error = updateCompilationConfigFromOptions(compilationOptions_);
  if (error) {
    return error;
  }

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
    static std::mutex compileMutex;
    std::lock_guard<std::mutex> guard(compileMutex);
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
      LOG_NNPI_IF_ERROR_RETURN_LLVMERROR(
          nnpiNetworkCompileToStream(network_, &config_, &outFileStream, NULL),
          "Failed NNPI Compile");

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
        compilationOptions_.lightCompilation) {
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
  switch (allocType) {
  case NNPI_ALLOCATION_DEFAULT:
    stream << "Default";
    break;
  case NNPI_ALLOCATION_DRAM:
    stream << "DRAM";
    break;
  case NNPI_ALLOCATION_ECC_DRAM:
    stream << "ECC DRAM";
    break;
  case NNPI_ALLOCATION_LLC:
  case NNPI_ALLOCATION_LLC_CLOS0:
  case NNPI_ALLOCATION_LLC_CLOS1:
  case NNPI_ALLOCATION_LLC_CLOS2:
  case NNPI_ALLOCATION_LLC_CLOS3:
    stream << "LLC";
    break;
  case NNPI_ALLOCATION_SRAM:
    stream << "SRAM";
    break;
  case NNPI_ALLOCATION_INTERNAL:
    stream << "Internal";
    break;
  default:
    stream << "Unknown";
    break;
  }
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
