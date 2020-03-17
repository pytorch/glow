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

#include "InferenceContext.h"
#include "DebugMacros.h"
#include "Importer.h"
#include "NNPI.h"
#include "NNPIDeviceManager.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <iomanip>
#include <sstream>

namespace glow {
namespace runtime {

InferenceContext::InferenceContext()
    : nnpiNetwork_(NNPI_INVALID_NNPIHANDLE), device_(NNPI_INVALID_NNPIHANDLE),
      inferCmd_(NNPI_INVALID_NNPIHANDLE), commandList_(NNPI_INVALID_NNPIHANDLE),
      deviceTracing_(nullptr), deviceOptions_(nullptr) {}

InferenceContext::~InferenceContext() {
  if (deviceOptions_ && deviceOptions_->inferOnDevice) {
    if (commandList_ != NNPI_INVALID_NNPIHANDLE) {
      LOG_NNPI_INF_IF_ERROR(nnpiCommandListDestroy(commandList_),
                            "Failed to destroy NNPI command list");
    }
    LOG_NNPI_INF_IF_ERROR(nnpiInferCommandDestroy(inferCmd_),
                          "Failed to destroy NNPI inference command");
  }
}

bool InferenceContext::init(
    // For ICE-Ref path.
    NNPINetwork network, NNPICompilationConfig config,
    // For ICE-T path.
    NNPIHostNetwork hostNetwork, NNPIDeviceNetwork deviceNetwork,
    NNPIAdapter adapter, NNPIDeviceContext device,
    const std::unordered_set<const Placeholder *> &partialInputs,
    const std::unordered_set<const Placeholder *> &staticInputs,
    std::shared_ptr<NNPIDeviceTracing> deviceTracing,
    StaticPlaceholderMap *staticPlaceholderMap,
    std::shared_ptr<NNPIDeviceOptions> deviceOptions,
    const std::string &functionName, unsigned deviceId) {
  deviceOptions_ = deviceOptions;
  deviceId_ = deviceId;
  nnpiNetwork_ = network;
  device_ = device;
  compilationConfig_ = config;
  partialInputs_ = &partialInputs;
  deviceTracing_ = deviceTracing;
  functionName_ = functionName;

  // Initialize trace context titles with device ID.
  std::stringstream deviceInfo;
  deviceInfo << "[Device #" << deviceId_ << "] ";
  traceBackendExecuteContextName_ = deviceInfo.str() + TRACING_BACKEND_EXECUTE;
  tracePreProcessContextName_ = deviceInfo.str() + TRACING_PRE_PROCESS;
  traceInferenceContextName_ = deviceInfo.str() + TRACING_INFERENCE;
  tracePostProcessContextName_ = deviceInfo.str() + TRACING_POST_PROCESS;

  LOG_AND_RETURN_IF(ERROR, staticPlaceholderMap == nullptr,
                    "InferenceContext Init was called with an invalid "
                    "staticPlaceholderMap",
                    false);

  /// Map from names to their Placeholders.
  std::unordered_map<std::string, const Placeholder *> staticPlaceholders;
  for (auto staticInput : staticInputs) {
    staticPlaceholders[staticInput->getName().str()] = staticInput;
    staticInputs_.insert(staticInput);
  }

  if (!deviceOptions_->inferOnDevice) {
    size_t numInputs, numOutputs;
    NNPIObjectName name;
    NNPITensorDesc desc;
    LOG_NNPI_IF_ERROR_RETURN_FALSE(
        nnpiNetworkGetInputNum(nnpiNetwork_, &numInputs),
        "Failed to query NNPI network inputs");
    for (size_t i = 0; i < numInputs; i++) {
      LOG_NNPI_IF_ERROR_RETURN_FALSE(
          nnpiNetworkGetInputDesc(nnpiNetwork_, i, name, &desc),
          "Failed to query NNPI network inputs");
      LOG_AND_RETURN_IF(
          ERROR, !deviceOptions_->useIceT && staticPlaceholders.count(name),
          "ICE-Ref doesn't support static inputs", false);
      inputResources_.emplace_back(std::make_shared<NNPIResource>());
      NNPIResourceDesc rDesc;
      LOG_AND_RETURN_IF(
          ERROR, !NNPIResource::UpdateResourceDescFromTensorDesc(&rDesc, &desc),
          "Failed to update ResourceDesc", false);
      LOG_AND_RETURN_IF(ERROR,
                        !inputResources_.back()->init(
                            name, deviceOptions_, adapter, device_, &rDesc,
                            NNPIResource::ResourceUsage::InputResource),
                        "Failed to init input resource", false);
    }
    LOG_NNPI_IF_ERROR_RETURN_FALSE(
        nnpiNetworkGetOutputNum(nnpiNetwork_, &numOutputs),
        "Failed to query NNPI network outputs");
    for (size_t i = 0; i < numOutputs; i++) {
      LOG_NNPI_IF_ERROR_RETURN_FALSE(
          nnpiNetworkGetOutputDesc(nnpiNetwork_, i, name, &desc),
          "Failed to query NNPI network outputs");
      LOG_AND_RETURN_IF(
          ERROR, !deviceOptions_->useIceT && staticPlaceholders.count(name),
          "ICE-Ref doesn't support static outputs", false);
      outputResources_.emplace_back(std::make_shared<NNPIResource>());
      NNPIResourceDesc rDesc;
      LOG_AND_RETURN_IF(
          ERROR, !NNPIResource::UpdateResourceDescFromTensorDesc(&rDesc, &desc),
          "Failed to update ResourceDesc", false);
      LOG_AND_RETURN_IF(ERROR,
                        !outputResources_.back()->init(
                            name, deviceOptions_, adapter, device_, &rDesc,
                            NNPIResource::ResourceUsage::OutputResource),
                        "Failed to init input resource", false);
    }

    return true; // Nothing else to be done here for ice-ref.
  }

  // Query input/output resources.
  uint32_t numInputs, numOutputs;
  LOG_NNPI_INF_IF_ERROR_RETURN_FALSE(
      nnpiHostNetworkGetInputNum(hostNetwork, &numInputs),
      "Failed to query NNPI network inputs");
  LOG_NNPI_INF_IF_ERROR_RETURN_FALSE(
      nnpiHostNetworkGetOutputNum(hostNetwork, &numOutputs),
      "Failed to query NNPI network outputs");

  // Create resources for inputs.
  for (uint32_t i = 0; i < numInputs; i++) {
    NNPIObjectName name;
    NNPIResourceDesc desc;
    LOG_NNPI_INF_IF_ERROR_RETURN_FALSE(
        nnpiHostNetworkGetInputDesc(hostNetwork, i, name, &desc),
        "Failed to query NNPI host network input");
    memset(&desc.hostAttrib, 0, sizeof(desc.hostAttrib));
    memset(&desc.deviceAttrib, 0, sizeof(desc.deviceAttrib));

    const auto isStaticInput = staticPlaceholders.count(name);
    if (isStaticInput) {
      // Treat as a static input.
      auto PH = staticPlaceholders.at(name);
      if (staticPlaceholderMap->count(PH) &&
          staticPlaceholderMap->at(PH).lock()) {
        // Static placeholder already exists.
        inputResources_.push_back(staticPlaceholderMap->at(PH).lock());
      } else {
        // Create a new static placeholder.
        inputResources_.emplace_back(std::make_shared<NNPIResource>());
        LOG_AND_RETURN_IF(ERROR,
                          !inputResources_.back()->init(
                              name, deviceOptions_, adapter, device_, &desc,
                              NNPIResource::ResourceUsage::StaticInputResource),
                          "Failed to init static input resource", false);
        staticPlaceholderMap->insert({PH, inputResources_.back()});
      }
    } else {
      // Regular input resource - create it here.
      inputResources_.emplace_back(std::make_shared<NNPIResource>());
      LOG_AND_RETURN_IF(ERROR,
                        !inputResources_.back()->init(
                            name, deviceOptions_, adapter, device_, &desc,
                            NNPIResource::ResourceUsage::InputResource),
                        "Failed to init input resource", false);
      inputResources_.back()->SetCmdListIdx(
          static_cast<uint32_t>(inputResources_.size()));
    }
  }

  // Create resources for outputs.
  for (uint32_t i = 0; i < numOutputs; i++) {
    {
      NNPIObjectName name;
      NNPIResourceDesc desc;
      LOG_NNPI_INF_IF_ERROR_RETURN_FALSE(
          nnpiHostNetworkGetOutputDesc(hostNetwork, i, name, &desc),
          "Failed to query NNPI host network output");
      memset(&desc.hostAttrib, 0, sizeof(desc.hostAttrib));
      memset(&desc.deviceAttrib, 0, sizeof(desc.deviceAttrib));
      outputResources_.emplace_back(std::make_shared<NNPIResource>());
      LOG_AND_RETURN_IF(ERROR,
                        !outputResources_.back()->init(
                            name, deviceOptions_, adapter, device_, &desc,
                            NNPIResource::ResourceUsage::OutputResource),
                        "Failed to init output resource", false);
      outputResources_.back()->SetCmdListIdx(
          static_cast<uint32_t>(outputResources_.size()));
    }
  }
  DBG_MEM_USAGE("Created input and output host resources");

  // Create infer command.
  NNPIDeviceResource inputHandles[numInputs];
  NNPIDeviceResource outputHandles[numOutputs];
  for (uint32_t i = 0; i < numInputs; i++) {
    inputHandles[i] = inputResources_.at(i)->GetDeviceResource();
  }
  for (uint32_t i = 0; i < numOutputs; i++) {
    outputHandles[i] = outputResources_.at(i)->GetDeviceResource();
  }
  LOG_NNPI_INF_IF_ERROR_RETURN_FALSE(
      nnpiInferCommandCreate(deviceNetwork, inputHandles, numInputs,
                             outputHandles, numOutputs, &inferCmd_),
      "Failed to create NNPI inference command");

  if (deviceOptions_->enabledCommandLists > 0) {
    // collect copy commands for the list (some resources may not need copying).
    std::vector<NNPICommandHandle> commands;
    std::vector<NNPICopyCommand> inputCopyCmds, outputCopyCmds;
    for (auto &res : inputResources_) {
      auto copyCmd = res->GetCopyCommand();
      if (copyCmd) {
        res->SetCmdListIdx(static_cast<uint32_t>(commands.size()));
        NNPICommandHandle cmd;
        cmd.type = NNPI_COMMAND_TYPE_COPY;
        cmd.copyCommand = copyCmd;
        commands.push_back(cmd);
      }
    }
    {
      NNPICommandHandle cmd;
      cmd.type = NNPI_COMMAND_TYPE_INFER;
      cmd.inferCommand = inferCmd_;
      commands.push_back(cmd);
    }
    for (auto &res : outputResources_) {
      auto copyCmd = res->GetCopyCommand();
      if (copyCmd) {
        res->SetCmdListIdx(static_cast<uint32_t>(commands.size()));
        NNPICommandHandle cmd;
        cmd.type = NNPI_COMMAND_TYPE_COPY;
        cmd.copyCommand = copyCmd;
        commands.push_back(cmd);
      }
    }

    // Create command list.
    LOG_NNPI_INF_IF_ERROR_RETURN_FALSE(
        nnpiCommandListCreate(&(commands[0]),
                              static_cast<uint32_t>(commands.size()), nullptr,
                              0, &commandList_),
        "Failed to create NNPI command list");

    // Preallocate enough configs to be used for partials later on.
    cmdConfigs_.resize(commands.size());
    // Preallocate enough errors to be used later durin inference.
    cmdListErrors_.resize(commands.size());
  }

  return true;
}

void InferenceContext::execute(RunIdentifierTy runId,
                               std::unique_ptr<ExecutionContext> ctx,
                               runtime::ResultCBTy resultCB) {
  TRACE_EVENT_SCOPE(ctx->getTraceContext(), TraceLevel::REQUEST,
                    traceBackendExecuteContextName_);
  if (ctx->getTraceContext()) {
    ctx->getTraceContext()->setThreadName(
        llvm::formatv("Inf ctx - device: {0}: {1}", deviceId_, functionName_)
            .str());
  }
  if (deviceTracing_) {
    deviceTracing_->start(ctx->getTraceContext(), device_);
  }

  // Pre inference input preparation.
  PlaceholderBindings &bindings = *ctx->getPlaceholderBindings();

  // Initialize placeholder lists in the same orders as inputResources_ and
  // outputResources_.
  if (netInputPlaceholders_.empty()) {
    for (const auto &in : inputResources_) {
      if (in->GetUsage() == NNPIResource::ResourceUsage::StaticInputResource) {
        continue;
      }
      auto *placeholder = bindings.getPlaceholderByName(in->GetName());
      if (!placeholder) {
        netInputPlaceholders_.clear();
        LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(ERROR, placeholder,
                                             "Can't find tensor for input",
                                             runId, ctx, resultCB);
      }

      netInputPlaceholders_.push_back(placeholder);
    }
  }
  if (netOutputPlaceholders_.empty()) {
    for (const auto &out : outputResources_) {
      auto *placeholder = bindings.getPlaceholderByName(out->GetName());
      if (!placeholder) {
        netOutputPlaceholders_.clear();
        LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(ERROR, placeholder,
                                             "Can't find tensor for input",
                                             runId, ctx, resultCB);
      }
      netOutputPlaceholders_.push_back(placeholder);
    }
  }

  std::unordered_set<Tensor *> partialTensorInputs;
  for (auto &pht : bindings.pairs()) {
    if (partialInputs_->count(pht.first)) {
      partialTensorInputs.insert(pht.second);
    }
  }
  TRACE_EVENT_BEGIN(ctx->getTraceContext(), TraceLevel::COPY,
                    tracePreProcessContextName_);

  // Pre-inference
  std::vector<void *> rawInputs, rawOutputs;
  unsigned idx = 0;
  for (const auto &in : inputResources_) {
    if (in->GetUsage() != NNPIResource::ResourceUsage::StaticInputResource) {
      auto *t = bindings.get(netInputPlaceholders_[idx++]);
      LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(
          ERROR, t, "Can't find tensor for input", runId, ctx, resultCB);
      LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(
          ERROR,
          in->PreInference(t, partialTensorInputs.count(t)) ==
              NNPI_INF_NO_ERROR,
          "Failed pre-inference for input", runId, ctx, resultCB);
    }
    rawInputs.push_back(in->GetHostPtr());
  }

  // Inference.
  if (deviceOptions_->inferOnDevice) {
    if (deviceOptions_->enabledCommandLists < 1) {
      // No command lists (schedule individual commands).
      TRACE_EVENT_END(ctx->getTraceContext(), TraceLevel::COPY,
                      tracePreProcessContextName_);
      TRACE_EVENT_BEGIN(ctx->getTraceContext(), TraceLevel::OPERATOR,
                        traceInferenceContextName_);
      // Queue inference.
      LOG_AND_CALLBACK_EXECUTE_NNPI_INF_IF_ERROR(
          nnpiInferCommandQueue(inferCmd_, 0), "Failed to queue infer command.",
          runId, ctx, resultCB);

      // Queue output copies
      for (auto &res : outputResources_) {
        auto cmd = res->GetCopyCommand();
        if (cmd) {
          // todo: assert no partial output
          LOG_AND_CALLBACK_EXECUTE_NNPI_INF_IF_ERROR(
              nnpiCopyCommandQueue(cmd, nullptr),
              "Failed to queue output copy command.", runId, ctx, resultCB);
        }
      }
    } else { // Use commands lists.
      // Prepare updates for partial copies.
      uint32_t usedConfigs = 0;
      for (auto &res : inputResources_) {
        const auto partialSize = res->GetPartialSize();
        if (partialSize > 0) {
          cmdConfigs_[usedConfigs].index = res->GetCmdListIdx();
          cmdConfigs_[usedConfigs].type = NNPI_COMMAND_TYPE_COPY;
          cmdConfigs_[usedConfigs].copyConfig.size = partialSize;
          usedConfigs++;
        }
      }

      TRACE_EVENT_END(ctx->getTraceContext(), TraceLevel::COPY,
                      tracePreProcessContextName_);
      TRACE_EVENT_BEGIN(ctx->getTraceContext(), TraceLevel::OPERATOR,
                        traceInferenceContextName_);
      // Queue Command list
      LOG_AND_CALLBACK_EXECUTE_NNPI_INF_IF_ERROR(
          nnpiCommandListQueue(commandList_, &(cmdConfigs_.at(0)), usedConfigs),
          "Failed to queue command list.", runId, ctx, resultCB);

      // Wait on completion and error handling.
      uint32_t numErrors(0);
      // First wait for the command list to complete.
      NNPIInferenceErrorCode res =
          nnpiCommandListWait(commandList_, UINT32_MAX, NULL, 0, &numErrors);
      if (res != NNPI_INF_NO_ERROR) {
        LOG_NNPI_INF_IF_ERROR(res, "Failed to wait on command list");
      } else if (numErrors > 0) {
        LOG(ERROR) << "Errors returned from command list";

        // Errors were generate so we allocate error objects to hold the data.
        NNPICommandListError commandErrors[numErrors];
        LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(
            ERROR, commandErrors, "Failed to allocate command error array",
            runId, ctx, resultCB);
        memset(commandErrors, 0, sizeof(NNPICommandListError) * numErrors);

        // Then query all errors using another wait call (should return
        // immediately).
        NNPIInferenceErrorCode tmpRes = nnpiCommandListWait(
            commandList_, UINT32_MAX, commandErrors, numErrors, &numErrors);
        if (tmpRes != NNPI_INF_NO_ERROR) {
          LOG_NNPI_INF_IF_ERROR(tmpRes,
                                "Failed to wait on command list to get errors");
        } else {
          for (uint32_t i = 0; i < numErrors; i++) {
            LOG(ERROR) << NNPI_INF_ERROR_MSG(commandErrors[i].err,
                                             commandErrors[i].desc);
          }
        }
      }
      if (res != NNPI_INF_NO_ERROR || numErrors > 0) {
        LOG_AND_CALLBACK_EXECUTE_NNPI_INF_IF_ERROR(
            nnpiCommandListClearErrors(commandList_),
            "Failed to clear command list errors", runId, ctx, resultCB);
        LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(
            ERROR, false /* fail */, "Errors found in command list execution",
            runId, ctx, resultCB);
      }
    }
  } else if (!deviceOptions_->useIceT) {
    // Infer on ice-ref.

    for (auto &out : outputResources_) {
      // Collect output ptrs for ICE-Ref
      rawOutputs.push_back(out->GetHostPtr());
    }

    TRACE_EVENT_END(ctx->getTraceContext(), TraceLevel::COPY,
                    TRACING_PRE_PROCESS);
    TRACE_EVENT_BEGIN(ctx->getTraceContext(), TraceLevel::OPERATOR,
                      TRACING_INFERENCE);
    LOG_AND_CALLBACK_EXECUTE_NNPI_IF_ERROR(
        nnpiNetworkInferOnHost(nnpiNetwork_, &(rawInputs[0]), rawInputs.size(),
                               &(rawOutputs[0]), rawOutputs.size(),
                               &compilationConfig_, NNPI_INVALID_NNPIHANDLE),
        "Failed NNPI infer (ICE-Ref)", runId, ctx, resultCB);
  } else //! UseInferenceAPI && UseIceT (compile without running).
  {
    // Nothing else to do here.
  }

  TRACE_EVENT_END(ctx->getTraceContext(), TraceLevel::OPERATOR,
                  traceInferenceContextName_);
  TRACE_EVENT_BEGIN(ctx->getTraceContext(), TraceLevel::COPY,
                    tracePostProcessContextName_);

  // Post inference output handling.
  for (unsigned i = 0, e = outputResources_.size(); i < e; ++i) {
    auto *t = bindings.get(netOutputPlaceholders_[i]);
    LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(
        ERROR, t, "Can't find tensor for output", runId, ctx, resultCB);
    LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(
        ERROR, outputResources_[i]->PostInference(t) == NNPI_INF_NO_ERROR,
        "Failed in output PostInference", runId, ctx, resultCB);
  }

  TRACE_EVENT_END(ctx->getTraceContext(), TraceLevel::COPY,
                  tracePostProcessContextName_);
  if (deviceTracing_) {
    deviceTracing_->stopAndUpdate(ctx->getTraceContext(), device_);
  }
  TRACE_EVENT_SCOPE_END(); // we move context in the line below

  // Invoke CB.
  resultCB(runId, Error::success(), std::move(ctx));
}

} // namespace runtime
} // namespace glow
