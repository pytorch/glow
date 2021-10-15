/*
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

#include "HostManagerOnnxifi.h"

#include "glow/Flags/Flags.h"
#include "glow/Runtime/DeferredWeightLoader.h"
#include "glow/Runtime/ErrorReporter.h"
#include "glow/Runtime/RequestData.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"

namespace glow {
namespace onnxifi {

static llvm::cl::opt<int32_t, true>
    GlowNumDevicesOpt("glow-num-devices",
                      llvm::cl::desc("Number of devices for Glow backend"),
                      llvm::cl::location(::glow::flags::NumDevices));

static llvm::cl::opt<bool, true>
    GlowDumpDebugTracesOpt("glow-dump-debug-traces",
                           llvm::cl::desc("Dump a trace of each run to /tmp"),
                           llvm::cl::location(glow::flags::DumpDebugTraces));

static llvm::cl::opt<bool, true> GlowSaturateHostOpt(
    "glow-saturate-host",
    llvm::cl::desc("Try to use all available devices on the host"),
    llvm::cl::location(glow::flags::SaturateHost));

static llvm::cl::opt<int32_t, true> GlowSparseNNPartitioningSchemeNumCardsOpt(
    "glow_snn_partitioning_num_cards",
    llvm::cl::desc("Number of cards for SparseNNPartitioningScheme"),
    llvm::cl::location(glow::flags::SparseNNPartitioningSchemeNumCards));

static llvm::cl::opt<int64_t, true>
    GlowSparseNNPartitioningSchemeSLSTableKBytesPerCardOpt(
        "glow_snn_partitioning_kbytes_per_card",
        llvm::cl::desc("SLS KBytes per card for SparseNNPartitioningScheme"),
        llvm::cl::location(
            glow::flags::SparseNNPartitioningSchemeSLSTableKBytesPerCard));

static llvm::cl::opt<int32_t, true>
    GlowSparseNNPartitioningSchemeNumCoresSLSOpt(
        "glow_snn_partitioning_num_cores_sls",
        llvm::cl::desc(
            "Number of cores for SLS for SparseNNPartitioningScheme"),
        llvm::cl::location(glow::flags::SparseNNPartitioningSchemeNumCoresSLS));

static llvm::cl::opt<int32_t, true>
    GlowSparseNNPartitioningSchemeNumCoresOtherOpt(
        "glow_snn_partitioning_num_cores_other",
        llvm::cl::desc(
            "Number of cores for other for SparseNNPartitioningScheme"),
        llvm::cl::location(
            glow::flags::SparseNNPartitioningSchemeNumCoresOther));

static llvm::cl::opt<bool, true> GlowUseSparseNNPartitioningSchemeOpt(
    "glow_use_sparsenn_partitioning_scheme",
    llvm::cl::desc("Whether to use SparseNNPartitioningScheme"),
    llvm::cl::location(glow::flags::UseSparseNNPartitioningScheme));

static llvm::cl::opt<bool, true> GlowSparseNNPartitioningAddSLSConcatsOpt(
    "glow_sparsenn_partitioning_add_sls_concats",
    llvm::cl::desc("Add extra concats inside of SLS partitions for more "
                   "efficient inter-partitition transfers"),
    llvm::cl::location(glow::flags::SparseNNPartitioningAddSLSConcats));

static llvm::cl::opt<bool, true> GlowSparseNNPartitioningBalancePerfModelOpt(
    "glow_sparsenn_partitioning_balance_perf_model",
    llvm::cl::desc("Balance SLS tables across cards using a perf model"),
    llvm::cl::location(glow::flags::SparseNNPartitioningBalancePerfModel));

static llvm::cl::opt<bool, true> GlowSparseNNPartitioningPairLNWithSLSOpt(
    "glow_sparsenn_partitioning_pair_ln_with_sls",
    llvm::cl::desc("Place layer normalization nodes immediately following SLS "
                   "into SLS partition"),
    llvm::cl::location(glow::flags::SparseNNPartitioningPairLNWithSLS));
static llvm::cl::opt<bool, true> GlowSparseNNPartitioningPairTileWithSLSOpt(
    "glow_sparsenn_partitioning_pair_tile_with_sls",
    llvm::cl::desc("Place Tile nodes immediately following SLS "
                   "for user embeddings into SLS partition"),
    llvm::cl::location(glow::flags::SparseNNPartitioningPairTileWithSLS));

static llvm::cl::opt<std::string, true> GlowSparseNNPartitioningPairSLSWithOpt(
    "glow_sparsenn_partitioning_pair_sls_with",
    llvm::cl::desc("Place specified nodes immediately following SLS "
                   "into SLS partition"),
    llvm::cl::location(glow::flags::SparseNNPartitioningPairSLSWith));

static llvm::cl::opt<int32_t, true> GlowSparseNNPartitioningConcatSplitSizeOpt(
    "glow_sparsenn_partitioning_concat_split_size",
    llvm::cl::desc("Split concat going into tanh sink into smaller concats of "
                   "specified size to move into SLS partition"),
    llvm::cl::location(glow::flags::SparseNNPartitioningConcatSplitSize));

std::unique_ptr<runtime::HostManager>
HostManagerBackend::createHostManager(llvm::StringRef backendName) {
  std::vector<std::unique_ptr<runtime::DeviceConfig>> configs;
  // If GlowNumDevices is set specify that many devices, otherwise use all
  // discovered devices.
  if (glow::flags::NumDevices) {
    for (int i = 0; i < glow::flags::NumDevices; i++) {
      auto config = glow::make_unique<runtime::DeviceConfig>(backendName);
      config->deviceID = i;
      configs.push_back(std::move(config));
    }
  } else {
    configs = runtime::DeviceManager::generateDeviceConfigs(
        backendName, glow::flags::ScanDevices);
  }

  runtime::HostConfig hostConfig;
  hostConfig.maxActiveRequests = glow::flags::MaxActiveRequests;
  hostConfig.maxQueueSize = glow::flags::MaxQueueSize;
  hostConfig.executorThreads = glow::flags::ExecutorThreads;

  return glow::make_unique<runtime::HostManager>(std::move(configs),
                                                 hostConfig);
}

void HostManagerBackend::runNetwork(const Graph *graph,
                                    std::unique_ptr<ExecutionContext> context,
                                    runtime::ResultCBTy callback,
                                    uint64_t priority) {
  DCHECK(callback != nullptr);

  auto hostManagerGraph = static_cast<const HostManagerGraph *>(graph);
  hostManager_->runNetwork(hostManagerGraph->getName(), std::move(context),
                           std::move(callback), priority);
}

onnxStatus HostManagerBackend::addNetwork(
    std::unique_ptr<Module> module, void *deferredBlobReader,
    CompilationContext &cctx,
    std::map<std::string, Type> &&staticPlaceholderTypes) {
  PrecisionConfiguration &precConfig = cctx.precisionConfig;
  cctx.maxActiveRequestsPerInstance = glow::flags::MaxActiveRequestsPerInstance;

  if (glow::flags::SkipProvisioning || deferredBlobReader) {
    // Generate a map of type date for all static placeholders. Do this
    // regardless of whether we have deferredBlobReader because we don't have
    // one for AOT but we still want to use this info for serialization.
    if (staticPlaceholderTypes.size() == 0) {
      for (auto *PH : module->getPlaceholders()) {
        if (PH->isStatic()) {
          staticPlaceholderTypes[std::string(PH->getName())] = *PH->getType();
        }
      }
    }

    // Signal that we want to fold convertTo and Quantize into static
    // Placeholders. Also want to do this for AOT optimization even if we don't
    // have a deferred blob reader present.
    cctx.optimizationOpts.foldStaticPlaceholderConversions = true;
  }

  // Copy the types into the cctx so that we have access to them regardless of
  // whether there is a deferredBlobReader.
  cctx.staticPlaceholderTypesForAOT = staticPlaceholderTypes;

  if (deferredBlobReader) {
    // Initialize loader and set field in cctx.
    auto loader = runtime::DeferredLoader()->getLoader();
    if (!loader) {
      LOG(INFO) << "Blob reader provided but no loader registered!";
      return ONNXIFI_STATUS_INTERNAL_ERROR;
    }

    loader->setTypeInfo(std::move(staticPlaceholderTypes));
    auto err = loader->setSrc(deferredBlobReader);
    if (ERR_TO_BOOL(std::move(err))) {
      return ONNXIFI_STATUS_INTERNAL_ERROR;
    }

    cctx.deferredWeightLoader = loader;
  }

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
  if (glow::flags::ConvertFusedScaleOffsetToFP32) {
    precConfig.convert4BitFusedToFP32 =
        glow::flags::ConvertFusedScaleOffsetToFP32;
    precConfig.convert8BitFusedToFP32 =
        glow::flags::ConvertFusedScaleOffsetToFP32;
    LOG(INFO) << "Conversion of fused scales/offsets to fp32 enabled";
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
    cctx.compilationLogPrefix = "glow-onnxifi";
  }
  if (glow::flags::SinkTanhBelowConcat) {
    cctx.optimizationOpts.sinkTanhBelowConcat =
        glow::flags::SinkTanhBelowConcat;
    LOG(INFO) << "Sinking tanh below concat";
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
  }
  if (glow::flags::DumpGraph) {
    cctx.dumpFinalGraph = true;
    cctx.dumpGraphPath = glow::flags::DumpGraphPath;
  }
  if (glow::flags::UseDAGOptimizer) {
    LOG(INFO) << "Will call the DAG optimizer.";
    cctx.callDAGOptimizer = true;
    cctx.optimizationOpts.DAGOptimizerPlacementTaggingAlgorithm =
        glow::flags::DAGOptimizerPlacementTaggingAlgorithm;
    cctx.optimizationOpts.DAGOptimizerParallelizationTaggingAlgorithm =
        glow::flags::DAGOptimizerParallelizationTaggingAlgorithm;
    cctx.optimizationOpts.DAGOptimizerNumParallelChunks =
        glow::flags::DAGOptimizerNumParallelChunks;
  }
  if (glow::flags::SkipProvisioning) {
    LOG(INFO) << "Will skip provisioning (likely due to AOT opt).";
    cctx.skipProvisioning = true;
  }
  if (glow::onnxifi::flags::SaveDAG) {
    LOG(INFO) << "Serializing DAG after optimization and partitioning.";
    cctx.serializeCompiledDAG = true;
  }
  if (glow::flags::DelayAndRecordConstantModification) {
    LOG(INFO) << "Delaying constant modification until after optimizations, "
                 "including recording constant folding for DAG serialization.";
    cctx.optimizationOpts.delayAndRecordConstantModification = true;
  }
  cctx.saturateHost = glow::flags::SaturateHost;

  if (!glow::flags::processBackendSpecificOpts(
          cctx.backendOpts.backendSpecificOpts,
          glow::flags::BackendSpecificOpts)) {
    return ONNXIFI_STATUS_INTERNAL_ERROR;
  }
  if (glow::runtime::flags::EnableP2P) {
    LOG(INFO) << "Glow P2P Enabled";
    cctx.enableP2P = true;
  }
  if (glow::runtime::flags::EnableDRT) {
    LOG(INFO) << "Glow DRT Enabled";
    cctx.enableDRT = true;
  }

  auto err = hostManager_->addNetwork(std::move(module), cctx);

  if (err) {
    std::string msg = err.peekErrorValue()->logToString();
    auto reporters = ErrorReporterRegistry::ErrorReporters();
    if (reporters) {
      reporters->report(msg);
    }
    const std::string errMsg =
        "Non-recoverable device error when adding network: " + msg;
    if (cctx.skipProvisioning) {
      LOG(ERROR) << errMsg;
      throw std::invalid_argument(strFormat(
          "Error during AOT optimization (non-provisioned addNetwork):\n%s\n",
          errMsg.c_str()));
    } else if (err.peekErrorValue()->getErrorCode() ==
               ErrorValue::ErrorCode::RUNTIME_DEFERRED_WEIGHT_ERROR) {
      // If a deferred weight error occurs, log the error but do not fatal so we
      // can try again.
      LOG(ERROR) << errMsg;
      return ONNXIFI_STATUS_INTERNAL_ERROR;
    } else {
      LOG(FATAL) << errMsg;
    }
  }

  return ONNXIFI_STATUS_SUCCESS;
}

onnxStatus HostManagerBackend::removeNetwork(const Graph *graph) {
  auto hostManagerGraph = static_cast<const HostManagerGraph *>(graph);
  auto error = hostManager_->removeNetwork(hostManagerGraph->getName());

  if (ERR_TO_BOOL(std::move(error))) {
    return ONNXIFI_STATUS_INTERNAL_ERROR;
  }

  return ONNXIFI_STATUS_SUCCESS;
}

onnxStatus HostManagerGraph::initGraph(
    const void *onnxModel, size_t onnxModelSize, uint32_t weightCount,
    const onnxTensorDescriptorV1 *weightDescriptors, uint32_t maxSeqLength,
    void *deferedBlobReader, bool loadingGlowAOT) {

  netName_ = strFormat("onnxifi_function_%lu", makeUniqueGraphId());

  std::unique_ptr<Module> module = glow::make_unique<Module>();
  CompilationContext cctx;
  runtime::PrePartitionedConfig PPC;
  cctx.prepartitionedConfig = &PPC;
  OriginNameToTQPMap originNameToTQPMap;
  if (glow::flags::UseTrackedDummyQuantParams) {
    cctx.precisionConfig.originNameToTQPMap = &originNameToTQPMap;
    cctx.precisionConfig.loadUniquedDummyQParams = true;
  }
  cctx.precisionConfig.clipQuantRangeToFP16 = glow::flags::ClipQuantRangeToFP16;
  cctx.precisionConfig.zeroScaleFP16Clip = glow::flags::ClipZeroScaleFP16;
  std::map<std::string, Type> staticPlaceholderTypes;

  std::unique_ptr<ONNXIFIModelLoader> loader;
  auto loaderOrErr = ONNXIFIModelLoader::parse(
      onnxModel, onnxModelSize, weightCount, weightDescriptors, *module,
      netName_, cctx, &staticPlaceholderTypes,
      true /*loadInputsAsPlaceholdersForOnnx*/, backendPtr_->getUseOnnx(),
      /* constFoldInLoader */ false);
  if (loaderOrErr) {
    loader = std::move(*loaderOrErr);
  } else {
    LOG(ERROR) << "Error when loading model: "
               << ERR_TO_STRING(loaderOrErr.takeError());
    return ONNXIFI_STATUS_INVALID_MODEL;
  }

  if (!bindPlaceholders(*loader, &cctx.loadedPHNames)) {
    return ONNXIFI_STATUS_INVALID_MODEL;
  }
  setZeroLengthSequence(maxSeqLength);
  // Make sure the pool is ready to go.
  for (auto &obj : onnxInputToPlaceholder_) {
    tensorPool_.reserve(obj.second->getType(), 10);
  }

  if (glow::onnxifi::flags::SaveModel) {
    for (Function *F : module->getFunctions()) {
      saveOnnxifiModel(F);
    }
  }

  if (glow::flags::DumpInitialLoadedGraph) {
    for (Function *F : module->getFunctions()) {
      auto fname = strFormat("initial_graph__%s.dot", F->getName().data());
      LOG(INFO) << "Dumping initially loaded graph to " << fname;
      F->dumpDAG(fname);
    }
  }

  if (loadingGlowAOT) {
    LOG(INFO) << "Loading a Glow AOT optimized model.";
    cctx.loadingAOTModel = true;
  }

  return static_cast<HostManagerBackend *>(backendPtr_)
      ->addNetwork(std::move(module), deferedBlobReader, cctx,
                   std::move(staticPlaceholderTypes));
}

namespace {
void dumpTraces(TraceContext *traceContext) {
  CHECK(traceContext);
  llvm::SmallString<64> path;
  auto tempFileRes =
      llvm::sys::fs::createTemporaryFile("glow-trace", "json", path);
  if (tempFileRes.value() != 0) {
    LOG(ERROR) << "Failed to create temp file for Glow trace events: "
               << tempFileRes;
  } else {
    traceContext->dump(path);
  }
}

} // namespace

onnxStatus HostManagerGraph::run(std::unique_ptr<ExecutionContext> ctx,
                                 EventPtr outputEvent,
                                 onnxTraceEventList *traceEvents) {
  auto threadId = threads::getThreadId();
  auto startTime = TraceEvent::now();

  auto *data = ::glow::runtime::RequestData::get();
  std::map<std::string, std::string> attributes;
  if (data) {
    attributes["app level request id"] =
        llvm::formatv("{0}", data->appLevelRequestId);
  }

  backendPtr_->runNetwork(
      this, std::move(ctx),
      [outputEvent, traceEvents, threadId, startTime,
       attributes = std::move(attributes),
       this](runtime::RunIdentifierTy runId, Error err,
             std::unique_ptr<ExecutionContext> ctx) mutable {
        TRACE_EVENT_SCOPE(ctx->getTraceContext(), TraceLevel::RUNTIME,
                          "Onnxifi::callback");

        if (err) {
          if (err.peekErrorValue() && err.peekErrorValue()->isFatalError()) {
            std::string msg = err.peekErrorValue()->logToString();
            auto reporters = ErrorReporterRegistry::ErrorReporters();
            if (reporters) {
              reporters->report(msg);
            }
            LOG(FATAL) << "Non-recoverable device error when running network: "
                       << msg;
          }
          outputEvent->setMessage(ERR_TO_STRING(std::move(err)));
          outputEvent->signal(ONNXIFI_STATUS_INTERNAL_ERROR);
          return;
        }

        // End the current trace event before we convert TraceEvents to the
        // ONNX format.
        TRACE_EVENT_SCOPE_END();

        auto *traceContext = ctx->getTraceContext();
        if (traceContext) {
          // We want to log the async start event with the original caller's
          // threadId. This way, chrome UI will put the async event next to
          // the caller thread.
          traceContext->logTraceEvent("glow e2e", TraceLevel::RUNTIME,
                                      TraceEvent::BeginType, startTime,
                                      attributes, threadId, runId);
          traceContext->logTraceEvent("glow e2e", TraceLevel::RUNTIME,
                                      TraceEvent::EndType, TraceEvent::now(),
                                      attributes, threadId, runId);
          setTraceEvents(traceEvents, traceContext);
        }

        // Signal to caller that the inference is completed.
        outputEvent->signal(ONNXIFI_STATUS_SUCCESS);

        if (traceContext && glow::flags::DumpDebugTraces) {
          // Dumping traces to a file can take a while. So avoid tracesMutex_
          // while we call dumpTraces.
          std::unique_ptr<TraceContext> toDump;
          {
            std::unique_lock<std::mutex> lock(tracesMutex_);
            if (!mergedTraceContext_) {
              mergedTraceContext_ =
                  glow::make_unique<TraceContext>(TraceLevel::STANDARD);
            }
            mergedTraceContext_->merge(traceContext);

            if (++numTracesToDump_ >= glow::flags::NumDebugTracesPerDump) {
              numTracesToDump_ = 0;
              toDump.reset(mergedTraceContext_.release());
            }
          }

          if (toDump) {
            dumpTraces(toDump.get());
          }
        }
      });

  return ONNXIFI_STATUS_SUCCESS;
}

HostManagerGraph::~HostManagerGraph() {
  // Remove network from the Backend
  backendPtr_->removeNetwork(this);

  if (glow::flags::DumpDebugTraces) {
    std::unique_lock<std::mutex> lock(tracesMutex_);
    if (mergedTraceContext_ && numTracesToDump_ > 0) {
      dumpTraces(mergedTraceContext_.get());
    }
  }
}

size_t HostManagerGraph::makeUniqueGraphId() {
  static std::atomic<size_t> nextId{0};
  return nextId++;
}

} // namespace onnxifi
} // namespace glow
