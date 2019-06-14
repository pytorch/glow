/*
 * Copyright (c) 2017-present, Facebook, Inc.
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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"

namespace glow {
namespace onnxifi {

int32_t GlowNumDevices = 1;
bool GlowDumpDebugTraces = false;

static llvm::cl::opt<int32_t, true>
    GlowNumDevicesOpt("glow-num-devices",
                      llvm::cl::desc("Number of devices for Glow backend"),
                      llvm::cl::location(GlowNumDevices));

static llvm::cl::opt<bool, true>
    GlowDumpDebugTracesOpt("glow-dump-debug-traces",
                           llvm::cl::desc("Dump a trace of each run to /tmp"),
                           llvm::cl::location(GlowDumpDebugTraces));

std::unique_ptr<runtime::HostManager>
HostManagerBackend::createHostManager(llvm::StringRef backendName) {
  std::vector<std::unique_ptr<runtime::DeviceConfig>> configs;
  for (int i = 0; i < GlowNumDevices; i++) {
    configs.push_back(llvm::make_unique<runtime::DeviceConfig>(backendName));
  }
  return llvm::make_unique<runtime::HostManager>(std::move(configs));
}

void HostManagerBackend::runNetwork(const Graph *graph,
                                    std::unique_ptr<ExecutionContext> context,
                                    runtime::ResultCBTy callback) {
  auto hostManagerGraph = static_cast<const HostManagerGraph *>(graph);
  hostManager_->runNetwork(hostManagerGraph->getName(), std::move(context),
                           std::move(callback));
}

onnxStatus HostManagerBackend::addNetwork(std::unique_ptr<Module> module) {
  CompilationContext cctx;
  auto err = hostManager_->addNetwork(std::move(module), cctx);

  if (errToBool(std::move(err))) {
    return ONNXIFI_STATUS_INTERNAL_ERROR;
  }

  return ONNXIFI_STATUS_SUCCESS;
}

onnxStatus HostManagerBackend::removeNetwork(const Graph *graph) {
  auto hostManagerGraph = static_cast<const HostManagerGraph *>(graph);
  auto error = hostManager_->removeNetwork(hostManagerGraph->getName());

  if (errorToBool(std::move(error))) {
    return ONNXIFI_STATUS_INTERNAL_ERROR;
  }

  return ONNXIFI_STATUS_SUCCESS;
}

onnxStatus
HostManagerGraph::initGraph(const void *onnxModel, size_t onnxModelSize,
                            uint32_t weightCount,
                            const onnxTensorDescriptorV1 *weightDescriptors) {

  netName_ = strFormat("onnxifi_function_%lu", makeUniqueGraphId());

  std::unique_ptr<Module> module = llvm::make_unique<Module>();
  Function *function = module->createFunction(netName_);

  // TODO: make better error reporting.
  std::unique_ptr<ONNXIFIModelLoader> loader =
      TEMP_EXIT_ON_ERR(ONNXIFIModelLoader::parse(
          onnxModel, onnxModelSize, weightCount, weightDescriptors, *function,
          true /*loadInputsAsPlaceholders*/, backendPtr_->getUseOnnx()));

  onnxInputToPlaceholder_ = loader->getInputVarsMapping();
  onnxOutputToPlaceholder_ = loader->getOutputVarsMapping();

  // Make sure the pool is ready to go.
  for (auto &obj : onnxInputToPlaceholder_) {
    tensorPool_.reserve(obj.second->getType(), 10);
  }

  return static_cast<HostManagerBackend *>(backendPtr_)
      ->addNetwork(std::move(module));
}

onnxStatus HostManagerGraph::run(std::unique_ptr<ExecutionContext> ctx,
                                 EventPtr outputEvent,
                                 onnxTraceEventList *traceEvents) {
  backendPtr_->runNetwork(
      this, std::move(ctx),
      [outputEvent, traceEvents](runtime::RunIdentifierTy runId,
                                 llvm::Error err,
                                 std::unique_ptr<ExecutionContext> ctx) {
        TRACE_EVENT_SCOPE(ctx->getTraceContext(), TraceLevel::RUNTIME,
                          "Onnxifi::callback");
        // If an Error occurred then log it in errToBool and signal the output
        // event.
        if (errToBool(std::move(err))) {
          outputEvent->signal();
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

        outputEvent->signal();
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
