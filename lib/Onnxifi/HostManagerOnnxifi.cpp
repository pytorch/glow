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
#include "glow/Runtime/DeferredWeightLoader.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"

namespace glow {
extern bool GlowDumpCompilationLog;
namespace onnxifi {

extern bool GlowSaveOnnxifiModel;
;
int32_t GlowNumDevices = 0;
int32_t GlowSparseNNPartitioningSchemeNumCards = 1;
int64_t GlowSparseNNPartitioningSchemeSLSTableKBytesPerCard = 0;
int32_t GlowSparseNNPartitioningSchemeNumCoresSLS = 1;
int32_t GlowSparseNNPartitioningSchemeNumCoresOther = 1;
bool GlowDumpDebugTraces = false;
bool GlowSaturateHost = false;
bool GlowFP16 = false;
bool GlowFusedScaleOffsetFP16 = false;
bool GlowForceSLSAccumFP16 = false;
bool GlowClipFP16 = false;
bool GlowUseSparseNNPartitioningScheme = false;

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
  return glow::make_unique<runtime::HostManager>(std::move(configs));
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

onnxStatus HostManagerBackend::addNetwork(std::unique_ptr<Module> module,
                                          void *deferredBlobReader) {
  CompilationContext cctx;
  PrecisionConfiguration &precConfig = cctx.precisionConfig;

  if (deferredBlobReader) {
    // Initialize loader and set field in cctx.
    auto loader = runtime::DeferredLoader()->getLoader();
    if (!loader) {
      LOG(INFO) << "Blob reader provided but no loader registered!";
      return ONNXIFI_STATUS_INTERNAL_ERROR;
    }

    // Generate a map of type date for all static placeholders.
    std::map<std::string, Type> staticPlaceholderTypes;
    for (auto PH : module->getPlaceholders()) {
      if (PH->isStatic()) {
        staticPlaceholderTypes[std::string(PH->getName())] = *PH->getType();
      }
    }
    loader->setTypeInfo(std::move(staticPlaceholderTypes));
    loader->setSrc(deferredBlobReader);
    cctx.deferredWeightLoader = loader;
    // Signal that we want to fold convertTo and Quantize into static
    // Placeholders.
    cctx.optimizationOpts.foldStaticPlaceholderConversions = true;
  }

  if (GlowFP16) {
    precConfig.convertToFP16 = GlowFP16;
    LOG(INFO) << "Conversion to fp16 enabled";
  }
  if (GlowFusedScaleOffsetFP16) {
    precConfig.convertFusedToFP16 = GlowFusedScaleOffsetFP16;
    LOG(INFO) << "Conversion of fused scales/offsets to fp16 enabled";
  }
  if (GlowClipFP16) {
    precConfig.clipFP16 = GlowClipFP16;
    LOG(INFO) << "Clipping to fp16 enabled";
  }
  if (GlowForceSLSAccumFP16) {
    precConfig.forceFP16AccumSLS = GlowForceSLSAccumFP16;
    LOG(INFO) << "Forcing all SLS/SLWS ops to use FP16 accumulation enabled";
  }
  if (GlowDumpCompilationLog) {
    cctx.compilationLogPrefix = "glow-onnxifi";
  }
  if (GlowUseSparseNNPartitioningScheme) {
    cctx.optimizationOpts.useSparseNNPartitioningScheme = true;
    cctx.optimizationOpts.sparseNNPartitioningSchemeNumCards =
        GlowSparseNNPartitioningSchemeNumCards;
    cctx.optimizationOpts.sparseNNPartitioningSchemeSLSTableKBytesPerCard =
        GlowSparseNNPartitioningSchemeSLSTableKBytesPerCard;
    cctx.optimizationOpts.sparseNNPartitioningSchemeNumCoresSLS =
        GlowSparseNNPartitioningSchemeNumCoresSLS;
    cctx.optimizationOpts.sparseNNPartitioningSchemeNumCoresOther =
        GlowSparseNNPartitioningSchemeNumCoresOther;
  }

  auto err =
      hostManager_->addNetwork(std::move(module), cctx, GlowSaturateHost);

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

onnxStatus
HostManagerGraph::initGraph(const void *onnxModel, size_t onnxModelSize,
                            uint32_t weightCount,
                            const onnxTensorDescriptorV1 *weightDescriptors,
                            uint32_t maxSeqLength, void *deferedBlobReader) {

  netName_ = strFormat("onnxifi_function_%lu", makeUniqueGraphId());

  std::unique_ptr<Module> module = glow::make_unique<Module>();
  Function *function = module->createFunction(netName_);

  // TODO: make better error reporting.
  std::unique_ptr<ONNXIFIModelLoader> loader =
      EXIT_ON_ERR(ONNXIFIModelLoader::parse(
          onnxModel, onnxModelSize, weightCount, weightDescriptors, *function,
          true /*loadInputsAsPlaceholders*/, backendPtr_->getUseOnnx()));

  onnxInputToPlaceholder_ = loader->getInputVarsMapping();
  onnxOutputToPlaceholder_ = loader->getOutputVarsMapping();

  setZeroLengthSequence(maxSeqLength);
  // Make sure the pool is ready to go.
  for (auto &obj : onnxInputToPlaceholder_) {
    tensorPool_.reserve(obj.second->getType(), 10);
  }

  if (GlowSaveOnnxifiModel) {
    saveOnnxifiModel(function);
  }

  return static_cast<HostManagerBackend *>(backendPtr_)
      ->addNetwork(std::move(module), deferedBlobReader);
}

onnxStatus HostManagerGraph::run(std::unique_ptr<ExecutionContext> ctx,
                                 EventPtr outputEvent,
                                 onnxTraceEventList *traceEvents) {
  backendPtr_->runNetwork(
      this, std::move(ctx),
      [outputEvent, traceEvents](runtime::RunIdentifierTy runId, Error err,
                                 std::unique_ptr<ExecutionContext> ctx) {
        TRACE_EVENT_SCOPE(ctx->getTraceContext(), TraceLevel::RUNTIME,
                          "Onnxifi::callback");
        // If an Error occurred then log it in ERR_TO_BOOL and signal the output
        // event.
        if (ERR_TO_BOOL(std::move(err))) {
          outputEvent->signal(ONNXIFI_STATUS_INTERNAL_ERROR);
          return;
        }

        // End the current trace event before we convert TraceEvents to the ONNX
        // format.
        TRACE_EVENT_SCOPE_END();

        if (auto *traceContext = ctx->getTraceContext()) {
          setTraceEvents(traceEvents, traceContext);

          if (GlowDumpDebugTraces) {
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
        }

        outputEvent->signal(ONNXIFI_STATUS_SUCCESS);
      });

  return ONNXIFI_STATUS_SUCCESS;
}

HostManagerGraph::~HostManagerGraph() {
  // Remove network from the Backend
  backendPtr_->removeNetwork(this);
}

size_t HostManagerGraph::makeUniqueGraphId() {
  static std::atomic<size_t> nextId{0};
  return nextId++;
}

} // namespace onnxifi
} // namespace glow
