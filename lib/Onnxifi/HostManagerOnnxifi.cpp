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
                      llvm::cl::location(GlowNumDevices));

static llvm::cl::opt<bool, true>
    GlowDumpDebugTracesOpt("glow-dump-debug-traces",
                           llvm::cl::desc("Dump a trace of each run to /tmp"),
                           llvm::cl::location(GlowDumpDebugTraces));

static llvm::cl::opt<bool, true> GlowSaturateHostOpt(
    "glow-saturate-host",
    llvm::cl::desc("Try to use all available devices on the host"),
    llvm::cl::location(GlowSaturateHost));

static llvm::cl::opt<int32_t, true> GlowSparseNNPartitioningSchemeNumCardsOpt(
    "glow_snn_partitioning_num_cards",
    llvm::cl::desc("Number of cards for SparseNNPartitioningScheme"),
    llvm::cl::location(GlowSparseNNPartitioningSchemeNumCards));

static llvm::cl::opt<int64_t, true>
    GlowSparseNNPartitioningSchemeSLSTableKBytesPerCardOpt(
        "glow_snn_partitioning_kbytes_per_card",
        llvm::cl::desc("SLS KBytes per card for SparseNNPartitioningScheme"),
        llvm::cl::location(
            GlowSparseNNPartitioningSchemeSLSTableKBytesPerCard));

static llvm::cl::opt<int32_t, true>
    GlowSparseNNPartitioningSchemeNumCoresSLSOpt(
        "glow_snn_partitioning_num_cores_sls",
        llvm::cl::desc(
            "Number of cores for SLS for SparseNNPartitioningScheme"),
        llvm::cl::location(GlowSparseNNPartitioningSchemeNumCoresSLS));

static llvm::cl::opt<int32_t, true>
    GlowSparseNNPartitioningSchemeNumCoresOtherOpt(
        "glow_snn_partitioning_num_cores_other",
        llvm::cl::desc(
            "Number of cores for other for SparseNNPartitioningScheme"),
        llvm::cl::location(GlowSparseNNPartitioningSchemeNumCoresOther));

static llvm::cl::opt<bool, true> GlowUseSparseNNPartitioningSchemeOpt(
    "glow_use_sparsenn_partitioning_scheme",
    llvm::cl::desc("Whether to use SparseNNPartitioningScheme"),
    llvm::cl::location(GlowUseSparseNNPartitioningScheme));

static llvm::cl::opt<bool, true> GlowSparseNNPartitioningAddSLSConcatsOpt(
    "glow_sparsenn_partitioning_add_sls_concats",
    llvm::cl::desc("Add extra concats inside of SLS partitions for more "
                   "efficient inter-partitition transfers"),
    llvm::cl::location(GlowSparseNNPartitioningAddSLSConcats));

static llvm::cl::opt<bool, true> GlowSparseNNPartitioningBalancePerfModelOpt(
    "glow_sparsenn_partitioning_balance_perf_model",
    llvm::cl::desc("Balance SLS tables across cards using a perf model"),
    llvm::cl::location(GlowSparseNNPartitioningBalancePerfModel));

static llvm::cl::opt<bool, true> GlowSparseNNPartitioningPairLNWithSLSOpt(
    "glow_sparsenn_partitioning_pair_ln_with_sls",
    llvm::cl::desc("Place layer normalization nodes immediately following SLS "
                   "into SLS partition"),
    llvm::cl::location(GlowSparseNNPartitioningPairLNWithSLS));

std::unique_ptr<runtime::HostManager>
HostManagerBackend::createHostManager(llvm::StringRef backendName) {
  std::vector<std::unique_ptr<runtime::DeviceConfig>> configs;
  // If GlowNumDevices is set specify that many devices, otherwise use all
  // discovered devices.
  if (GlowNumDevices) {
    for (int i = 0; i < GlowNumDevices; i++) {
      auto config = glow::make_unique<runtime::DeviceConfig>(backendName);
      config->deviceID = i;
      configs.push_back(std::move(config));
    }
  } else {
    configs = runtime::DeviceManager::generateDeviceConfigs(backendName);
  }

  runtime::HostConfig hostConfig;
  hostConfig.maxActiveRequests = GlowMaxActiveRequests;
  hostConfig.maxQueueSize = GlowMaxQueueSize;
  hostConfig.executorThreads = GlowExecutorThreads;

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
  cctx.maxActiveRequestsPerInstance = GlowMaxActiveRequestsPerInstance;

  if (GlowUseDAGOptimizerAOT || deferredBlobReader) {
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

  if (GlowFP16) {
    precConfig.convertToFP16 = GlowFP16;
    LOG(INFO) << "Conversion to fp16 enabled";
  }
  if (GlowFP16Placeholders) {
    precConfig.convertPlaceholdersToFP16 = GlowFP16Placeholders;
    LOG(INFO) << "Conversion of Placeholders to fp16 enabled";
  }
  if (GlowFP16Constants) {
    precConfig.convertConstantsToFP16 = GlowFP16Constants;
    LOG(INFO) << "Conversion of Constants to fp16 enabled";
  }
  if (GlowFusedScaleOffsetFP16) {
    precConfig.convertFusedToFP16 = GlowFusedScaleOffsetFP16;
    LOG(INFO) << "Conversion of fused scales/offsets to fp16 enabled";
  }
  if (GlowClipFP16) {
    precConfig.clipFP16 = GlowClipFP16;
    LOG(INFO) << "Clipping to fp16 enabled";
  }
  if (GlowClipFP16SkipInputs) {
    precConfig.clipFP16SkipInputs = GlowClipFP16SkipInputs;
    LOG(INFO) << "Skipping clipping for fp16 Node inputs fp16";
  }
  if (GlowForceSLSAccumFP16) {
    precConfig.forceFP16AccumSLS = GlowForceSLSAccumFP16;
    LOG(INFO) << "Forcing all SLS/SLWS ops to use FP16 accumulation enabled";
  }
  if (!GlowEnableQuantParamChanges) {
    cctx.optimizationOpts.enableQuantParamChanges = false;
    LOG(INFO) << "Disabling quantization param changes during optimizations";
  }
  if (GlowDumpCompilationLog) {
    cctx.compilationLogPrefix = "glow-onnxifi";
  }
  if (GlowUseSparseNNPartitioningScheme) {
    cctx.optimizationOpts.useSparseNNPartitioningScheme = true;
    cctx.optimizationOpts.sparseNNPartitioningAddSLSConcats =
        GlowSparseNNPartitioningAddSLSConcats;
    cctx.optimizationOpts.sparseNNPartitioningBalancePerfModel =
        GlowSparseNNPartitioningBalancePerfModel;
    cctx.optimizationOpts.sparseNNPartitioningPairLNWithSLS =
        GlowSparseNNPartitioningPairLNWithSLS;
    cctx.optimizationOpts.sparseNNPartitioningSchemeNumCards =
        GlowSparseNNPartitioningSchemeNumCards;
    cctx.optimizationOpts.sparseNNPartitioningSchemeSLSTableKBytesPerCard =
        GlowSparseNNPartitioningSchemeSLSTableKBytesPerCard;
    cctx.optimizationOpts.sparseNNPartitioningSchemeNumCoresSLS =
        GlowSparseNNPartitioningSchemeNumCoresSLS;
    cctx.optimizationOpts.sparseNNPartitioningSchemeNumCoresOther =
        GlowSparseNNPartitioningSchemeNumCoresOther;
  }
  if (GlowDumpGraph) {
    cctx.dumpFinalGraph = true;
    cctx.dumpGraphPath = GlowDumpGraphPath;
  }
  if (GlowUseDAGOptimizer) {
    LOG(INFO) << "Will call the DAG optimizer.";
    cctx.callDAGOptimizer = true;
    cctx.optimizationOpts.DAGOptimizerPlacementTaggingAlgorithm =
        GlowDAGOptimizerPlacementTaggingAlgorithm;
    cctx.optimizationOpts.DAGOptimizerParallelizationTaggingAlgorithm =
        GlowDAGOptimizerParallelizationTaggingAlgorithm;
    cctx.optimizationOpts.DAGOptimizerNumParallelChunks =
        GlowDAGOptimizerNumParallelChunks;
    if (GlowUseDAGOptimizerAOT) {
      LOG(INFO) << "Using AOT mode for DAG optimizer.";
      cctx.useDAGOptimizerAOTMode = true;
    }
  }
  if (GlowSaveOnnxifiDAG) {
    LOG(INFO) << "Serializing DAG after optimization and partitioning.";
    cctx.serializeCompiledDAG = true;
  }
  if (GlowDelayAndRecordConstantModification) {
    LOG(INFO) << "Delaying constant modification until after optimizations, "
                 "including recording constant folding for DAG serialization.";
    cctx.optimizationOpts.delayAndRecordConstantModification = true;
  }
  cctx.saturateHost = GlowSaturateHost;

  auto err = hostManager_->addNetwork(std::move(module), cctx);

  if (ERR_TO_BOOL(std::move(err))) {
    return ONNXIFI_STATUS_INTERNAL_ERROR;
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
  if (GlowUseTrackedDummyQuantParams) {
    cctx.precisionConfig.originNameToTQPMap = &originNameToTQPMap;
    cctx.precisionConfig.loadUniquedDummyQParams = true;
  }
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

  if (GlowSaveOnnxifiModel) {
    for (Function *F : module->getFunctions()) {
      saveOnnxifiModel(F);
    }
  }

  if (GlowDumpInitialLoadedGraph) {
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
            LOG(FATAL) << "Non-recoverable device error: " << msg;
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

        if (traceContext && GlowDumpDebugTraces) {
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

            if (++numTracesToDump_ >= GlowNumDebugTracesPerDump) {
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

  if (GlowDumpDebugTraces) {
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
