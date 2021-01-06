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
#include "NNPIAdapterContainer.h"
#include "NNPIDeviceManager.h"
#include "glow/Flags/Flags.h"
#include "glow/Runtime/RequestData.h"
#include "llvm/Support/raw_ostream.h"
#include <folly/Random.h>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace glow {
namespace runtime {

InferenceContext::InferenceContext()
    : nnpiNetwork_(NNPI_INVALID_NNPIHANDLE), device_(NNPI_INVALID_NNPIHANDLE),
      inferCmd_(NNPI_INVALID_NNPIHANDLE), commandList_(NNPI_INVALID_NNPIHANDLE),
      deviceOptions_(nullptr) {}

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
    const ResourceDescVec &inputs, const ResourceDescVec &outputs,
    // For ICE-Ref path.
    NNPINetwork network, NNPICompilationConfig config,
    // For ICE-T path.
    NNPIDeviceNetwork deviceNetwork, NNPIAdapterContainer *adapter,
    NNPIDeviceContext device,
    const std::unordered_set<const Placeholder *> &partialInputs,
    const std::vector<ValidateSLSInfo> &validateSLSInputs,
    const std::unordered_set<const Placeholder *> &paddedInputs,
    const std::unordered_set<const Placeholder *> &staticInputs,
    StaticPlaceholderMap *staticPlaceholderMap,
    std::shared_ptr<NNPIDeviceOptions> deviceOptions,
    const std::string &functionName, unsigned deviceId,
    PlaceholderUsageMap *phUsage) {
  deviceOptions_ = deviceOptions;
  deviceId_ = deviceId;
  nnpiNetwork_ = network;
  device_ = device;
  compilationConfig_ = config;
  partialInputs_ = &partialInputs;
  validateSLSInputs_ = &validateSLSInputs;
  paddedInputs_ = &paddedInputs;
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

  const size_t numInputs = inputs.size();
  const size_t numOutputs = outputs.size();

  if (!deviceOptions_->inferOnDevice) {
    // No P2P/DRT for ICE-Ref (everything is on the host).
    for (auto &in : inputs) {
      const auto *name = in.first.c_str();
      const auto &desc = in.second;
      const bool isStaticInput = staticPlaceholders.count(name) != 0;
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
          LOG_AND_RETURN_IF(
              ERROR,
              !inputResources_.back()->init(
                  name, deviceOptions_, adapter, device_, &desc,
                  NNPIResource::ResourceUsage::StaticInputResource),
              "Failed to init static input resource", false);
          staticPlaceholderMap->insert({PH, inputResources_.back()});
        }
      } else {
        inputResources_.emplace_back(std::make_shared<NNPIResource>());
        LOG_AND_RETURN_IF(ERROR,
                          !inputResources_.back()->init(
                              name, deviceOptions_, adapter, device_, &desc,
                              NNPIResource::ResourceUsage::InputResource),
                          "Failed to init input resource", false);
      }
    }
    for (auto &out : outputs) {
      const auto *name = out.first.c_str();
      const auto &desc = out.second;
      LOG_AND_RETURN_IF(
          ERROR, !deviceOptions_->useIceT && staticPlaceholders.count(name),
          "ICE-Ref doesn't support static outputs", false);
      outputResources_.emplace_back(std::make_shared<NNPIResource>());
      LOG_AND_RETURN_IF(ERROR,
                        !outputResources_.back()->init(
                            name, deviceOptions_, adapter, device_, &desc,
                            NNPIResource::ResourceUsage::OutputResource),
                        "Failed to init input resource", false);
    }

    return true; // Nothing else to be done here for ice-ref.
  }

  // Create resources for inputs.
  for (auto &in : inputs) {
    const auto *name = in.first.c_str();
    const auto &desc = in.second;

    LOG_AND_RETURN_IF(ERROR,
                      phUsage &&
                          ((phUsage->count(name) == 0) ||
                           (phUsage->at(name).devices.count(device_) == 0)),
                      "Invalid placheholder usage for input resource", false);

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
      // Dynamic input resource - create it here.
      NNPIResource::ResourceUsage usage = NNPIResource::ResourceUsage::None;
      if (!phUsage || (phUsage->at(name).numWriters == 0)) {
        usage = NNPIResource::ResourceUsage::InputResource; // Net input
      } else { // Some other context is writing to this placeholder --> P2P/DRT
        switch (phUsage->at(name).devices.size()) {
        case 1: // DRT
          usage = NNPIResource::ResourceUsage::DRTInput;
          break;
        case 2: // P2P
          usage = NNPIResource::ResourceUsage::P2PInput;
          break;
        default:
          LOG_AND_RETURN_IF(ERROR, true,
                            "Invalid number of devices accessing a resource",
                            false);
        }
      }
      inputResources_.emplace_back(std::make_shared<NNPIResource>());
      LOG_AND_RETURN_IF(ERROR,
                        !inputResources_.back()->init(name, deviceOptions_,
                                                      adapter, device_, &desc,
                                                      usage, phUsage),
                        "Failed to init input resource", false);
    }

    // Update placeholder usage.
    if (phUsage) {
      phUsage->at(name).readers.push_back(inputResources_.back());
    }
  }

  // Create resources for outputs.
  for (auto &out : outputs) {
    const auto *name = out.first.c_str();
    const auto &desc = out.second;
    LOG_AND_RETURN_IF(ERROR,
                      phUsage &&
                          ((phUsage->count(name) == 0) ||
                           (phUsage->at(name).devices.count(device_) == 0)),
                      "Invalid placheholder usage for output resource", false);
    NNPIResource::ResourceUsage usage = NNPIResource::ResourceUsage::None;
    if (!phUsage || (phUsage->at(name).numReaders == 0)) {
      usage = NNPIResource::ResourceUsage::OutputResource; // Net output
    } else { // Some other context is writing to this placeholder --> P2P/DRT
      switch (phUsage->at(name).devices.size()) {
      case 1: // DRT
        usage = NNPIResource::ResourceUsage::DRTOutput;
        break;
      case 2: // P2P
        usage = NNPIResource::ResourceUsage::P2POutput;
        break;
      default:
        LOG_AND_RETURN_IF(ERROR, true,
                          "Invalid number of devices accessing a resource",
                          false);
      }
    }
    outputResources_.emplace_back(std::make_shared<NNPIResource>());
    LOG_AND_RETURN_IF(ERROR,
                      !outputResources_.back()->init(name, deviceOptions_,
                                                     adapter, device_, &desc,
                                                     usage, phUsage),
                      "Failed to init output resource", false);

    // Update placeholder usage.
    if (phUsage) {
      phUsage->at(name).writers.push_back(outputResources_.back());
    }
  }
  DBG_MEM_USAGE("Created input and output host resources");

  // Create infer command.
  NNPIDeviceResource inputHandles[numInputs];
  NNPIDeviceResource outputHandles[numOutputs];
  for (uint32_t i = 0; i < numInputs; i++) {
    inputHandles[i] = inputResources_.at(i)->getDeviceResource();
  }
  for (uint32_t i = 0; i < numOutputs; i++) {
    outputHandles[i] = outputResources_.at(i)->getDeviceResource();
  }
  LOG_NNPI_INF_IF_ERROR_RETURN_FALSE(
      nnpiInferCommandCreate(deviceNetwork, inputHandles, numInputs,
                             outputHandles, numOutputs, &inferCmd_),
      "Failed to create NNPI inference command for " + functionName_);

  // Collect copy commands for the list (some resources may not need copying).
  std::vector<NNPICommandHandle> commands;
  std::vector<NNPICopyCommand> inputCopyCmds, outputCopyCmds;
  for (auto &res : inputResources_) {
    auto copyCmd = res->getCopyCommand();
    if (copyCmd) {
      res->setCmdListIdx(static_cast<uint32_t>(commands.size()));
      NNPICommandHandle cmd;
      cmd.type = NNPI_COMMAND_TYPE_COPY;
      cmd.copyCommand = copyCmd;
      commands.push_back(cmd);
    }
  }
  if (deviceOptions_->disableCommands < 1) {
    NNPICommandHandle cmd;
    cmd.type = NNPI_COMMAND_TYPE_INFER;
    cmd.inferCommand = inferCmd_;
    commands.push_back(cmd);
  }
  for (auto &res : outputResources_) {
    auto copyCmd = res->getCopyCommand();
    if (copyCmd) {
      res->setCmdListIdx(static_cast<uint32_t>(commands.size()));
      NNPICommandHandle cmd;
      cmd.type = NNPI_COMMAND_TYPE_COPY;
      cmd.copyCommand = copyCmd;
      commands.push_back(cmd);
    }
  }

  // Create command list.
  LOG_NNPI_INF_IF_ERROR_RETURN_FALSE(
      nnpiCommandListCreate(&(commands[0]),
                            static_cast<uint32_t>(commands.size()), nullptr, 0,
                            &commandList_),
      "Failed to create NNPI command list");

  // Preallocate enough configs to be used for partials later on.
  cmdConfigs_.resize(commands.size());

  if (deviceOptions_->dumpRuntime) {
    dumpRuntime();
  }

  return true;
}

void InferenceContext::execute(RunIdentifierTy runId,
                               std::unique_ptr<ExecutionContext> ctx,
                               runtime::ResultCBTy resultCB) {
  if (deviceOptions_->disableCommands > 2) {
    // Return immediately without doing anything.
    resultCB(runId, Error::success(), std::move(ctx));
    return;
  }

  std::string traceBackendExecuteStr =
      llvm::formatv("{0} {1:x}", traceBackendExecuteContextName_, ctx.get());

  std::map<std::string, std::string> attributes;

  auto *data = ::glow::runtime::RequestData::get();
  if (data) {
    attributes["app level request id"] =
        llvm::formatv("{0}", data->appLevelRequestId);
  }

  TRACE_EVENT_SCOPE_NAMED(ctx->getTraceContext(), TraceLevel::REQUEST,
                          traceBackendExecuteStr, traceBlock);
  for (const auto &iter : attributes) {
    traceBlock.addArg(iter.first, iter.second);
  }

  if (ctx->getTraceContext()) {
    ctx->getTraceContext()->setThreadName(
        llvm::formatv("Inf ctx - device: {0}: {1}", deviceId_, functionName_)
            .str());
  }

  // Pre inference input preparation.
  PlaceholderBindings &bindings = *ctx->getPlaceholderBindings();

  // Size of Inputs/Outputs
  uint64_t outputSize{0}, inputSize{0};

  // Initialize placeholder lists in the same orders as inputResources_ and
  // outputResources_.
  if (netInputPlaceholders_.empty()) {
    for (const auto &in : inputResources_) {
      if (in->getUsage() == NNPIResource::ResourceUsage::StaticInputResource) {
        continue;
      }
      auto *placeholder = bindings.getPlaceholderByNameSlow(in->getName());
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
      auto *placeholder = bindings.getPlaceholderByNameSlow(out->getName());
      if (!placeholder) {
        netOutputPlaceholders_.clear();
        LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(ERROR, placeholder,
                                             "Can't find tensor for input",
                                             runId, ctx, resultCB);
      }
      netOutputPlaceholders_.push_back(placeholder);
    }
  }

  TRACE_EVENT_BEGIN_ATTR(ctx->getTraceContext(), TraceLevel::COPY,
                         tracePreProcessContextName_, attributes);

  // Sanitize inputs before running inference
  LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(
      ERROR, sanitize(bindings), "Failed santization.", runId, ctx, resultCB);

  // Pre-inference
  std::vector<void *> rawInputs, rawOutputs;
  unsigned idx = 0;
  for (const auto &in : inputResources_) {
    if (in->getUsage() != NNPIResource::ResourceUsage::StaticInputResource) {
      auto *ph = netInputPlaceholders_[idx++];
      auto *t = bindings.get(ph);
      inputSize += t->getUnpaddedSizeInBytes();
      LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(
          ERROR, t, "Can't find tensor for input", runId, ctx, resultCB);
      LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(
          ERROR,
          in->preInference(t, partialInputs_->count(ph),
                           paddedInputs_->count(ph)) == NNPI_INF_NO_ERROR,
          "Failed pre-inference for input", runId, ctx, resultCB);
    }
    rawInputs.push_back(in->getHostPtr());
  }
  auto *requestData = ::glow::runtime::RequestData::get();
  if (requestData) {
    requestData->inputSize += inputSize;
  }
  std::string inferContext = traceInferenceContextName_;
  // Inference.
  if (deviceOptions_->inferOnDevice) {
    // Prepare updates for partial copies.
    uint32_t usedConfigs = 0;
    for (auto &res : inputResources_) {
      const auto partialSize = res->getPartialSize();
      if (partialSize > -1) {
        memset(&(cmdConfigs_[usedConfigs]), 0, sizeof(NNPICommandConfig));
        cmdConfigs_[usedConfigs].index = res->getCmdListIdx();
        cmdConfigs_[usedConfigs].type = NNPI_COMMAND_TYPE_COPY;
        cmdConfigs_[usedConfigs].copyConfig.size = partialSize;
        if (partialSize == 0) {
          cmdConfigs_[usedConfigs].copyConfig.disableCopy = 1;
        }
        usedConfigs++;
      }
    }
    inferContext = llvm::formatv("{0} {1:x}", inferContext, commandList_);
    TRACE_EVENT_END(ctx->getTraceContext(), TraceLevel::COPY,
                    tracePreProcessContextName_);
    TRACE_EVENT_BEGIN_ATTR(ctx->getTraceContext(), TraceLevel::OPERATOR,
                           inferContext, attributes);

    if (deviceOptions_->disableCommands < 2) {
      // Queue Command list
      int64_t issueTime = TraceEvent::now();
      FATAL_CALLBACK_IF_NOT(
          nnpiCommandListQueue(commandList_, &(cmdConfigs_.at(0)),
                               usedConfigs) == NNPI_INF_NO_ERROR,
          "Failed to queue command list.", runId, ctx, resultCB);

      // Wait on completion and error handling.
      uint32_t numErrors(0);
      // First wait for the command list to complete.
      NNPIInferenceErrorCode res = nnpiCommandListWait(
          commandList_, deviceOptions_->inferTimeoutUs, NULL, 0, &numErrors);
      uint64_t completeTime = TraceEvent::now();
      // Set batchDeviceTimestamps.
      auto *requestDataRun = ::glow::runtime::RequestData::get();
      if (requestDataRun) {
        auto duration = completeTime - issueTime;
        requestDataRun->deviceRuntime.fetch_add(duration);
      }
      if (res != NNPI_INF_NO_ERROR) {
        std::stringstream ss;
        ss << "Nonrecoverable card error: ";
        ss << NNPI_INF_ERROR_MSG(res, "Failed to wait on command list");
        FATAL_CALLBACK_IF_NOT(false /* fatal */, ss.str(), runId, ctx,
                              resultCB);
      } else if (numErrors > 0) {
        LOG(ERROR) << "Errors returned from command list";

        // Errors were generate so we allocate error objects to hold the data.
        NNPICommandListError commandErrors[numErrors];
        FATAL_CALLBACK_IF_NOT(
            commandErrors,
            "Nonrecoverable card error happened but we failed "
            "to allocate command error array to get more details",
            runId, ctx, resultCB);
        memset(commandErrors, 0, sizeof(NNPICommandListError) * numErrors);

        // Then query all errors using another wait call (should return
        // immediately).
        NNPIInferenceErrorCode tmpRes =
            nnpiCommandListWait(commandList_, deviceOptions_->inferTimeoutUs,
                                commandErrors, numErrors, &numErrors);
        if (tmpRes != NNPI_INF_NO_ERROR) {
          FATAL_CALLBACK_IF_NOT(false /* fatal */,
                                "Nonrecoverable card error happened but we "
                                "failed to wait on command list to get errors",
                                runId, ctx, resultCB);
        } else {
          std::stringstream ss;
          ss << "Nonrecoverable card error: ";
          for (uint32_t i = 0; i < numErrors; i++) {
            ss << NNPI_INF_ERROR_MSG(commandErrors[i].err,
                                     commandErrors[i].desc)
               << "; ";
          }
          FATAL_CALLBACK_IF_NOT(false /* fatal */, ss.str(), runId, ctx,
                                resultCB);
        }
      }
      if (res != NNPI_INF_NO_ERROR || numErrors > 0) {
        FATAL_CALLBACK_IF_NOT(nnpiCommandListClearErrors(commandList_),
                              "Unknown nonrecoverable card error happend and "
                              "we failed to clear command list errors",
                              runId, ctx, resultCB);
        FATAL_CALLBACK_IF_NOT(false /* fail */,
                              "Unknown nonrecoverable card error happen", runId,
                              ctx, resultCB);
      }
    }
  } else if (!deviceOptions_->useIceT) {
    // Infer on ice-ref.

    // Ice-ref not re-entrant - To be removed once ICE-29869 is implemented
    static std::mutex icerefMutex;
    std::lock_guard<std::mutex> guard(icerefMutex);

    for (auto &out : outputResources_) {
      // Collect output ptrs for ICE-Ref
      rawOutputs.push_back(out->getHostPtr());
    }
    TRACE_EVENT_END(ctx->getTraceContext(), TraceLevel::COPY,
                    tracePreProcessContextName_);
    TRACE_EVENT_BEGIN_ATTR(ctx->getTraceContext(), TraceLevel::OPERATOR,
                           inferContext, attributes);
    LOG_AND_CALLBACK_EXECUTE_NNPI_IF_ERROR(
        nnpiNetworkInferOnHost(nnpiNetwork_, &(rawInputs[0]), rawInputs.size(),
                               &(rawOutputs[0]), rawOutputs.size(),
                               &compilationConfig_, NNPI_INVALID_NNPIHANDLE),
        "Failed NNPI infer (ICE-Ref)", runId, ctx, resultCB);
  } else //! UseInferenceAPI && UseIceT (compile without running).
  {
    // Nothing else to do here.
  }

  TRACE_EVENT_END(ctx->getTraceContext(), TraceLevel::OPERATOR, inferContext);
  TRACE_EVENT_BEGIN_ATTR(ctx->getTraceContext(), TraceLevel::COPY,
                         tracePostProcessContextName_, attributes);
  // Post inference output handling.
  for (unsigned i = 0, e = outputResources_.size(); i < e; ++i) {
    auto *t = bindings.get(netOutputPlaceholders_[i]);
    outputSize += t->getSizeInBytes();
    LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(
        ERROR, t, "Can't find tensor for output", runId, ctx, resultCB);
    LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(
        ERROR, outputResources_[i]->postInference(t) == NNPI_INF_NO_ERROR,
        "Failed in output postInference", runId, ctx, resultCB);
  }
  if (requestData) {
    requestData->outputSize += outputSize;
  }

  TRACE_EVENT_END(ctx->getTraceContext(), TraceLevel::COPY,
                  tracePostProcessContextName_);
  TRACE_EVENT_SCOPE_END_NAMED(traceBlock); // we move context in the line below

  // Invoke CB.
  resultCB(runId, Error::success(), std::move(ctx));
}

void InferenceContext::dumpRuntime() const {
  for (auto &in : inputResources_) {
    std::string resourceType;
    unsigned color = 0;
    switch (in->getUsage()) {
    case NNPIResource::ResourceUsage::InputResource:
      resourceType = "Input";
      color = 2;

      // Add host resource node
      DotWriter::addNode(std::to_string(in->getHostResource()),
                         std::string("Host Resource\\lHandle: ") +
                             DotWriter::getHexStr(in->getHostResource()),
                         1, "Host");
      // Add copy command h2c
      DotWriter::addEdge(std::to_string(in->getHostResource()),
                         std::to_string(in->getDeviceResource()));
      break;
    case NNPIResource::ResourceUsage::StaticInputResource:
      resourceType = "Static Input";
      color = 3;
      break;
    case NNPIResource::ResourceUsage::P2PInput:
      resourceType = "P2P Input";
      color = 4;
      break;
    case NNPIResource::ResourceUsage::DRTInput:
      resourceType = "DRT In/Out";
      color = 5;
      break;
    default:; // do nothing
    }

    // add device resource node
    DotWriter::addNode(std::to_string(in->getDeviceResource()),
                       resourceType + std::string("\\lHandle: ") +
                           DotWriter::getHexStr(in->getDeviceResource()),
                       color, std::to_string(in->getDevice()));

    // connect to function
    DotWriter::addEdge(std::to_string(in->getDeviceResource()),
                       functionName_ + ":" + in->getName());
  }
  for (auto &out : outputResources_) {
    std::string resourceType;
    unsigned color = 0;
    switch (out->getUsage()) {
    case NNPIResource::ResourceUsage::OutputResource:
      resourceType = "Output";
      color = 2;

      // Add host resource node
      DotWriter::addNode(std::to_string(out->getHostResource()),
                         std::string("Host Resource\\lHandle: ") +
                             DotWriter::getHexStr(out->getHostResource()),
                         1, "Host");
      // Add copy command c2h
      DotWriter::addEdge(std::to_string(out->getDeviceResource()),
                         std::to_string(out->getHostResource()));
      break;
    case NNPIResource::ResourceUsage::P2POutput:
      resourceType = "P2P Output";
      color = 4;
      // Add copy command c2c
      DotWriter::addEdge(std::to_string(out->getDeviceResource()),
                         std::to_string(out->getP2PDeviceResource()));
      break;
    case NNPIResource::ResourceUsage::DRTOutput:
      resourceType = "DRT In/Out";
      color = 5;
      break;
    default:; // do nothing
    }

    // add device resource node
    DotWriter::addNode(std::to_string(out->getDeviceResource()),
                       resourceType + std::string("\\lHandle: ") +
                           DotWriter::getHexStr(out->getDeviceResource()),
                       color, std::to_string(out->getDevice()));

    // connect to function
    DotWriter::addEdge(functionName_ + ":" + out->getName(),
                       std::to_string(out->getDeviceResource()));
  }
}

template <class T>
static bool sanitizeSLS(Handle<int64_t> &indices, Handle<T> &lenOrOffset,
                        size_t tableHeight, bool isOffset) {
  size_t totalLensSum;
  size_t indicesLen = indices.getRealNumElements();
  size_t lenOrOffsetLen = lenOrOffset.getRealNumElements();
  // indices in [0, n)
  for (auto i = 0; i < indicesLen; i++) {
    LOG_AND_RETURN_IF(ERROR,
                      indices.raw(i) < 0 || indices.raw(i) >= tableHeight,
                      "index out of range " + to_string(indices.raw(i)) +
                              " at idx " + to_string(i) + " tableHeight "
                          << to_string(tableHeight),
                      false);
  }
  // last offset = len(indices)
  if (isOffset) {
    for (auto i = 0; i < lenOrOffsetLen - 1; i++) {
      LOG_AND_RETURN_IF(ERROR, lenOrOffset.raw(i) >= lenOrOffset.raw(i + 1),
                        "offset should be monotonically increasing " +
                            to_string(lenOrOffset.raw(i)) + " " +
                            to_string(lenOrOffset.raw(i + 1)),
                        false);
    }
    totalLensSum = lenOrOffset.raw(lenOrOffsetLen - 1);
  } else {
    // sum(lens) = len(indices)
    totalLensSum = std::accumulate(lenOrOffset.begin(), lenOrOffset.end(), 0U);
  }

  LOG_AND_RETURN_IF(
      ERROR, indicesLen != totalLensSum,
      "santize failed, indices length " + std::to_string(indicesLen) +
          " different from sum of lengths " + std::to_string(totalLensSum),
      false);

  return true;
}

bool InferenceContext::sanitize(PlaceholderBindings &bindings) {
  if (flags::SanitizeInputsPercent == 0 ||
      folly::Random::rand32() % 100 > flags::SanitizeInputsPercent) {
    return true;
  }

  LOG_EVERY_N(INFO, 1000) << "===== Sanitizing " << validateSLSInputs_->size()
                          << " set of inputs at "
                          << flags::SanitizeInputsPercent << " probability";
  for (const auto &sls : *validateSLSInputs_) {
    size_t indicesLen, weightsLen, totalLensSum = 0;
    auto *indices = bindings.get(sls.indices);

    // Either a constant or some node internal to the function, skip
    if (indices == nullptr) {
      continue;
    }

    // The indices tensor is not partial
    if (indices->getSizeInBytes() == indices->getUnpaddedSizeInBytes()) {
      continue;
    }
    indicesLen = indices->getRealNumElements();
    if (sls.weights) {
      auto *weights = bindings.get(sls.weights);
      // If this is a weigthed one and the placeholder is real (not a constant
      // or internal to the function, then sanitize
      if (weights != nullptr) {
        weightsLen = weights->getRealNumElements();
        LOG_AND_RETURN_IF(
            ERROR, indicesLen != weightsLen,
            "santize failed, indices length: " + std::to_string(indicesLen) +
                " different from weights length: " + std::to_string(weightsLen),
            false);
      }
    }
    auto I = indices->getHandle<int64_t>();
    bool ret;
    if (sls.isEmbeddingBag) {
      // EmbeddingBag case
      auto *offsets = bindings.get(sls.offsets);
      // Either a constant or some node internal to the function, skip
      if (offsets == nullptr) {
        continue;
      }

      if (offsets->getElementType() == ElemKind::Int32ITy) {
        auto O = offsets->getHandle<int32_t>();
        ret = sanitizeSLS<int32_t>(I, O, sls.tableHeight, sls.isEmbeddingBag);
      } else if (offsets->getElementType() == ElemKind::Int64ITy) {
        auto O = offsets->getHandle<int64_t>();
        ret = sanitizeSLS<int64_t>(I, O, sls.tableHeight,

                                   sls.isEmbeddingBag);
      } else {
        LOG_AND_RETURN_IF(ERROR, true, "unsupported element type", false);
      }
    } else {
      // SLS case
      auto *lengths = bindings.get(const_cast<Placeholder *>(sls.lengths));
      // Either a constant or some node internal to the function, skip
      if (lengths == nullptr) {
        continue;
      }

      if (lengths->getElementType() == ElemKind::Int32ITy) {
        auto L = lengths->getHandle<int32_t>();
        ret = sanitizeSLS<int32_t>(I, L, sls.tableHeight,

                                   sls.isEmbeddingBag);
      } else if (lengths->getElementType() == ElemKind::Int64ITy) {
        auto L = lengths->getHandle<int64_t>();
        ret = sanitizeSLS<int64_t>(I, L, sls.tableHeight,

                                   sls.isEmbeddingBag);
      } else {
        LOG_AND_RETURN_IF(ERROR, true, "unsupported element type", false);
      }
    }
    LOG_AND_RETURN_IF(ERROR, !ret, "failed SLS sanitization", false);
  }

  return true;
}

} // namespace runtime
} // namespace glow
