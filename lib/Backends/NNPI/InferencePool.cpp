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

#include "InferencePool.h"
#include "DebugMacros.h"
#include "Importer.h"
#include "NNPI.h"
#include "NNPIDeviceManager.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <iomanip>
#include <sstream>

#ifdef USE_AVX
#include <immintrin.h>
static inline void convertI64ToI32(int64_t const *i64InputOrg,
                                   int32_t *i32OutputOrg, uint32_t elements) {
  uint8_t i64Input = reinterpret_cast<const uint8_t *>(i64InputOrg);
  uint8_t i32OutputOrg = reinterpret_cast<uint8_t *>(i32OutputOrg);
  const __mmask8 masks[9] = {
      0b0, 0b1, 0b11, 0b111, 0b1111, 0b11111, 0b111111, 0b1111111, 0b11111111,
  };
  constexpr uint32_t vecSize = (sizeof(__m512i) / sizeof(int64_t));
  const uint32_t fullIterations = (elements / vecSize);
  const uint32_t tailElements = (elements % vecSize);

  for (uint32_t i = 0; i < fullIterations; i++) {
    __m512i i64vec = _mm512_maskz_loadu_epi64(masks[vecSize], i64Input);
    _mm512_mask_cvtepi64_storeu_epi32(i32Output, masks[vecSize], i64vec);
    i64Input += (sizeof(int64_t) * vecSize);
    i32Output += (sizeof(int32_t) * vecSize);
  }
  if (tailElements > 0) {
    __m512i i64vec = _mm512_maskz_loadu_epi64(masks[tailElements], i64Input);
    _mm512_mask_cvtepi64_storeu_epi32(i32Output, masks[tailElements], i64vec);
  }
}
#else
static inline void convertI64ToI32(int64_t const *i64Input, int32_t *i32Output,
                                   uint32_t elements) {
  for (size_t i = 0; i < elements; i++) {
    i32Output[i] = static_cast<int32_t>(i64Input[i]);
  }
}
#endif // USE_AVX

static std::string convertHexTextFormat(void *data, size_t size) {
  std::stringstream ss;

  uint8_t *buf = reinterpret_cast<uint8_t *>(data);

  for (size_t cl = 0; cl < size; cl += 64) {
    size_t size_to_read = std::min((size_t)64, size - cl);
    for (size_t i = 0; i < size_to_read; i++) {
      ss << std::setfill('0') << std::setw(2) << std::uppercase << std::hex
         << (uint32_t)(buf[cl + (size_to_read - 1 - i)]);
    }

    ss << "\n";
  }

  return ss.str();
}

static void dumpToFile(const std::string &filename, void *data, size_t size) {
  std::fstream fs(filename, std::ios::out | std::ios::binary);

  if (fs.is_open()) {
    auto res = convertHexTextFormat(data, size);
    fs << res;
    fs.close();
  } else {
    LOG(ERROR) << "cannot open the file \"" << filename << "\" for writing";
  }
}

namespace glow {
namespace runtime {

InferenceThreadEnv::InferenceThreadEnv()
    : nnpiNetwork_(NNPI_INVALID_NNPIHANDLE), device_(NNPI_INVALID_NNPIHANDLE),
      inferCmd_(NNPI_INVALID_NNPIHANDLE), commandList_(NNPI_INVALID_NNPIHANDLE),
      commandErrors_(nullptr), numCommands_(0), deviceOptions_(nullptr) {}

InferenceThreadEnv::~InferenceThreadEnv() {
  if (deviceOptions_ && deviceOptions_->inferOnDevice) {
    if (commandErrors_ != nullptr) {
      delete[] commandErrors_;
    }
    if (commandList_ != NNPI_INVALID_NNPIHANDLE) {
      LOG_NNPI_INF_ERROR(nnpiCommandListDestroy(commandList_),
                         "Failed to destroy NNPI command list");
    }
    LOG_NNPI_INF_ERROR(nnpiInferCommandDestroy(inferCmd_),
                       "Failed to destroy NNPI inference command");
    for (auto &cmd : inputCopyCmds_) {
      LOG_NNPI_INF_ERROR(nnpiCopyCommandDestroy(cmd),
                         "Failed to destroy NNPI copy command");
    }
    for (auto &cmd : outputCopyCmds_) {
      LOG_NNPI_INF_ERROR(nnpiCopyCommandDestroy(cmd),
                         "Failed to destroy NNPI copy command");
    }
    for (auto &nr : hostInputs_) {
      LOG_NNPI_INF_ERROR(nnpiHostResourceDestroy(nr.handle),
                         "Failed to destroy NNPI host resource");
    }
    for (auto &nr : hostOutputs_) {
      LOG_NNPI_INF_ERROR(nnpiHostResourceDestroy(nr.handle),
                         "Failed to destroy NNPI host resource");
    }
    for (auto &nr : allocatedDeviceInputs_) {
      LOG_NNPI_INF_ERROR(nnpiDeviceResourceDestroy(nr.handle),
                         "Failed to destroy NNPI device resource");
    }
    for (auto &ph : staticInputs_) {
      LOG_IF_NOT(ERROR, staticPlaceholderContainer_->ReleaseDeviceResource(ph))
          << "Failed to release device resource for " << ph->getName().str();
    }
    for (auto &nr : deviceOutputs_) {
      LOG_NNPI_INF_ERROR(nnpiDeviceResourceDestroy(nr.handle),
                         "Failed to destroy NNPI device resource");
    }
  }
}

void InferenceThreadEnv::execute(RunIdentifierTy runId,
                                 std::unique_ptr<ExecutionContext> ctx,
                                 runtime::ResultCBTy resultCB) {
  TRACE_EVENT_SCOPE(ctx->getTraceContext(), TraceLevel::REQUEST,
                    TRACING_BACKEND_EXECUTE);
  if (ctx->getTraceContext()) {
    ctx->getTraceContext()->setThreadName("InferenceThreadEnv");
  }

  // Pre inference input preparation.
  PlaceholderBindings &bindings = *ctx->getPlaceholderBindings();
  ioTensors_.clear();
  uint32_t usedConfigs = 0;

  std::unordered_set<Tensor *> partialTensorInputs;
  for (auto &pht : bindings.pairs()) {
    ioTensors_.emplace(pht.first->getName(), pht.second);
    if (partialInputs_->count(pht.first)) {
      partialTensorInputs.insert(pht.second);
    }
  }
  TRACE_EVENT_BEGIN(ctx->getTraceContext(), TraceLevel::COPY,
                    TRACING_PRE_PROCESS);
  // Handle inputs & outputs (+convert).
  for (auto &in : netInputs_) {
    LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(ERROR, ioTensors_.count(in.first),
                                         "Can't find tensor for input", runId,
                                         ctx, resultCB);
    auto *t = ioTensors_.at(in.first);
    char *bufferPtr = t->getUnsafePtr();

    // Check if we need to allocate a new temporary buffer to handle this
    // input.
    const bool downcastInt64 = t->getElementType() == glow::ElemKind::Int64ITy;
    size_t paddedSize = t->getSizeInBytes();
    size_t unpaddedSize = t->getUnpaddedSizeInBytes();
    if (downcastInt64) {
      paddedSize /= 2;
      unpaddedSize /= 2;
    }
    const bool padUnhandledPartial =
        paddedSize != unpaddedSize && !partialTensorInputs.count(t);
    if (padUnhandledPartial || downcastInt64) {
      char *tmp = new char[padUnhandledPartial ? paddedSize : unpaddedSize];
      tmpBuffers_.insert(tmp);

      // Copy over the original data. Downcast int64 tensors to int32 if
      // needed.
      if (downcastInt64) {
        const int64_t *pInput = reinterpret_cast<int64_t *>(bufferPtr);
        int32_t *tmp32 = reinterpret_cast<int32_t *>(tmp);
        convertI64ToI32(pInput, tmp32, unpaddedSize / sizeof(int32_t));
      } else {
        // If we don't need to downcast then this must be an unhandled
        // partial, so copy over the original data before we pad below.
        memcpy(tmp, bufferPtr, unpaddedSize);
      }
      bufferPtr = tmp;

      // If this tensor cannot be treated as partial but it is partial, then
      // we need to zero out anything that's leftover in the extra padding, as
      // it is still going to be used.
      if (padUnhandledPartial) {
        // Zero out rest of the new buffer.
        memset(bufferPtr + unpaddedSize, 0, paddedSize - unpaddedSize);
      }
    }

    rawInputs_.push_back(bufferPtr);

    if (deviceOptions_->dumpIOtoFiles) {
      dumpToFile("input_" + in.first + ".txt", rawInputs_.back(), unpaddedSize);
    }
  }
  for (auto &out : netOutputs_) {
    LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(ERROR, ioTensors_.count(out.first),
                                         "Can't find tensor for output", runId,
                                         ctx, resultCB);
    auto *t = ioTensors_.at(out.first);

    switch (t->getElementType()) {
    case glow::ElemKind::Int64ITy: {
      // Create int32 buffer for size_t tensors.
      int32_t *tmp = new int32_t[t->getUnpaddedSizeInBytes() / sizeof(int64_t)];
      rawOutputs_.push_back(tmp);
      tmpBuffers_.insert(reinterpret_cast<char *>(tmp));
    } break;
    default:
      rawOutputs_.push_back(t->getUnsafePtr());
    }
  }
  // Prepare inputs.
  if (deviceOptions_->inferOnDevice) {
    if (deviceTracing_ != nullptr) {
      deviceTracing_->start(ctx->getTraceContext(), runId);
    }

    // Copy data to host resource and preprocess int64.
    // Queue copy commands.
    // For every input: lock host, copy data (+convert), unlock.
    LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(
        ERROR, hostInputs_.size() == rawInputs_.size(), "Bad inputs", runId,
        ctx, resultCB);
    for (size_t i = 0, e = hostInputs_.size(); i < e; i++) {
      void *pHostInput(hostInputs_[i].hostPtr);
      LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(ERROR, pHostInput,
                                           "Invalid host input address", runId,
                                           ctx, resultCB);
      LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(
          ERROR, ioTensors_.count(hostInputs_[i].name), "Input not found",
          runId, ctx, resultCB);
      auto *t = ioTensors_.at(hostInputs_[i].name);

      size_t bufferSize = t->getUnpaddedSizeInBytes();
      size_t fullBufferSize = t->getSizeInBytes();

      if (t->getElementType() == glow::ElemKind::Int64ITy) {
        // Type int64 is converted to int32 (half number of bytes).
        bufferSize /= 2;
        fullBufferSize /= 2;
      }

      size_t actualBufferSize =
          (bufferSize != fullBufferSize && partialTensorInputs.count(t))
              ? bufferSize
              : fullBufferSize;
      std::memcpy(pHostInput, rawInputs_[i], actualBufferSize);

      if (deviceOptions_->enabledCommandLists < 1) {
        // Queue a copy command.
        if (actualBufferSize != fullBufferSize) {
          NNPICopyCommandConfig cfg;
          memset(&cfg, 0, sizeof(NNPICopyCommandConfig));
          cfg.size = actualBufferSize;
          LOG_AND_CALLBACK_EXECUTE_NNPI_INF_ERROR(
              nnpiCopyCommandQueue(inputCopyCmds_[i], &cfg),
              "Failed to queue input copy command.", runId, ctx, resultCB);
        } else {
          LOG_AND_CALLBACK_EXECUTE_NNPI_INF_ERROR(
              nnpiCopyCommandQueue(inputCopyCmds_[i], nullptr),
              "Failed to queue input copy command.", runId, ctx, resultCB);
        }
      } else if (deviceOptions_->enabledCommandLists > 2) {
        if (actualBufferSize != fullBufferSize) {
          // Handle copy command config with a command list.
          NNPICommandConfig &cfg = cmdConfigs_.at(usedConfigs);
          memset(&cfg, 0, sizeof(NNPICommandConfig));
          cfg.index = i;
          cfg.type = NNPI_COMMAND_TYPE_COPY;
          cfg.copyConfig.size = actualBufferSize;
          usedConfigs++;
        }
      }
    }
  }
  // Handle inference.
  if (deviceOptions_->inferOnDevice) {
    if (deviceOptions_->enabledCommandLists < 1) {
      TRACE_EVENT_END(ctx->getTraceContext(), TraceLevel::COPY,
                      TRACING_PRE_PROCESS);
      TRACE_EVENT_BEGIN(ctx->getTraceContext(), TraceLevel::OPERATOR,
                        TRACING_INFERENCE);
      // Queue inference.
      LOG_AND_CALLBACK_EXECUTE_NNPI_INF_ERROR(
          nnpiInferCommandQueue(inferCmd_, 0), "Failed to queue infer command.",
          runId, ctx, resultCB);

      // Check for partial copies and queue output copies
      for (size_t i = 0, e = hostOutputs_.size(); i < e; i++) {
        auto *t = ioTensors_.at(hostOutputs_[i].name);
        size_t bufferSize = t->getUnpaddedSizeInBytes();
        size_t fullBufferSize = t->getSizeInBytes();
        NNPICopyCommandConfig copyConfig;
        memset(&copyConfig, 0, sizeof(NNPICopyCommandConfig));
        if (bufferSize != fullBufferSize) {
          copyConfig.size = bufferSize;
          if (t->getElementType() == glow::ElemKind::Int64ITy) {
            // Type int64 is converted to int32 (half number of bytes).
            copyConfig.size /= 2;
          }
          LOG_AND_CALLBACK_EXECUTE_NNPI_INF_ERROR(
              nnpiCopyCommandQueue(outputCopyCmds_[i], &copyConfig),
              "Failed to queue output copy command.", runId, ctx, resultCB);
        } else {
          LOG_AND_CALLBACK_EXECUTE_NNPI_INF_ERROR(
              nnpiCopyCommandQueue(outputCopyCmds_[i], nullptr),
              "Failed to queue output copy command.", runId, ctx, resultCB);
        }
        if (i == 0) {
          if (deviceTracing_ != nullptr) {
            deviceTracing_->startCopyTime();
          }
        }
      }

    } else {
      if (deviceOptions_->enabledCommandLists > 2) {
        // Collect updates for outputs and queue with all config updates.
        for (size_t i = 0, e = hostOutputs_.size(); i < e; i++) {
          auto *t = ioTensors_.at(hostOutputs_[i].name);
          size_t bufferSize = t->getUnpaddedSizeInBytes();
          size_t fullBufferSize = t->getSizeInBytes();
          size_t actualBufferSize =
              (bufferSize != fullBufferSize && partialTensorInputs.count(t))
                  ? bufferSize
                  : fullBufferSize;

          NNPICopyCommandConfig copyConfig;
          memset(&copyConfig, 0, sizeof(NNPICopyCommandConfig));
          if (bufferSize != fullBufferSize) {
            // Add another config update.
            NNPICommandConfig &cfg = cmdConfigs_.at(usedConfigs);
            memset(&cfg, 0, sizeof(NNPICommandConfig));
            cfg.index =
                i + hostInputs_.size() + 1 /* 1 for the inference command */;
            cfg.type = NNPI_COMMAND_TYPE_COPY;
            cfg.copyConfig.size = actualBufferSize;
            if (t->getElementType() == glow::ElemKind::Int64ITy) {
              // Type int64 is converted to int32 (half number of bytes).
              cfg.copyConfig.size /= 2;
            }
            usedConfigs++;
          }
        }
        TRACE_EVENT_END(ctx->getTraceContext(), TraceLevel::COPY,
                        TRACING_PRE_PROCESS);
        TRACE_EVENT_BEGIN(ctx->getTraceContext(), TraceLevel::OPERATOR,
                          TRACING_INFERENCE);
        // Then queue command list with config updates.
        LOG_AND_CALLBACK_EXECUTE_NNPI_INF_ERROR(
            nnpiCommandListQueue(commandList_, &(cmdConfigs_.at(0)),
                                 usedConfigs),
            "Failed to queue command list.", runId, ctx, resultCB);
        if (deviceTracing_ != nullptr) {
          deviceTracing_->startCopyTime();
        }
      } else {
        TRACE_EVENT_END(ctx->getTraceContext(), TraceLevel::COPY,
                        TRACING_PRE_PROCESS);
        TRACE_EVENT_BEGIN(ctx->getTraceContext(), TraceLevel::OPERATOR,
                          TRACING_INFERENCE);
        LOG_AND_CALLBACK_EXECUTE_NNPI_INF_ERROR(
            nnpiCommandListQueue(commandList_, nullptr, 0),
            "Failed to queue command list.", runId, ctx, resultCB);
        if (deviceTracing_ != nullptr) {
          deviceTracing_->startCopyTime();
        }
      }
    }
  } else if (!deviceOptions_->useIceT) {
    // Infer on ice-ref.
    TRACE_EVENT_END(ctx->getTraceContext(), TraceLevel::COPY,
                    TRACING_PRE_PROCESS);
    TRACE_EVENT_BEGIN(ctx->getTraceContext(), TraceLevel::OPERATOR,
                      TRACING_INFERENCE);
    LOG_AND_CALLBACK_EXECUTE_NNPI_ERROR(
        nnpiNetworkInferOnHost(nnpiNetwork_, &(rawInputs_[0]),
                               rawInputs_.size(), &(rawOutputs_[0]),
                               rawOutputs_.size(), &compilationConfig_,
                               NNPI_INVALID_NNPIHANDLE),
        "Failed NNPI infer (ICE-Ref)", runId, ctx, resultCB);
    TRACE_EVENT_END(ctx->getTraceContext(), TraceLevel::OPERATOR,
                    TRACING_INFERENCE);
    TRACE_EVENT_BEGIN(ctx->getTraceContext(), TraceLevel::COPY,
                      TRACING_POST_PROCESS);
    // Convert outputs.
    size_t currOut = 0;
    for (auto &out : netOutputs_) {
      LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(ERROR, ioTensors_.count(out.first),
                                           "Output not found", runId, ctx,
                                           resultCB);
      auto *t = ioTensors_.at(out.first);

      switch (t->getElementType()) {
      case glow::ElemKind::Int64ITy: {
        // Convert int32 outputs to size_t.
        int64_t *pOutput = reinterpret_cast<int64_t *>(t->getUnsafePtr());
        int32_t *tmp = reinterpret_cast<int32_t *>(rawOutputs_[currOut]);
        for (size_t i = 0, e = t->size(); i < e; i++) {
          pOutput[i] = static_cast<int64_t>(tmp[i]);
        }
      } break;
      default:; // Do nothing.
      }
      currOut++;
    }
  } else //! UseInferenceAPI && UseIceT.
  {
    TRACE_EVENT_END(ctx->getTraceContext(), TraceLevel::OPERATOR,
                    TRACING_INFERENCE);
    TRACE_EVENT_BEGIN(ctx->getTraceContext(), TraceLevel::COPY,
                      TRACING_POST_PROCESS);
    // Just zero the outputs (no inference).
    for (auto &out : netOutputs_) {
      LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(ERROR, ioTensors_.count(out.first),
                                           "Output not found", runId, ctx,
                                           resultCB);
      auto *t = ioTensors_.at(out.first);
      t->zero();
    }
  }
  // Post inference output handling and wait for outputs.
  if (deviceOptions_->inferOnDevice) {
    // For every output, lock and copy data (+convert), unlock
    // then copy to output tensors.
    LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(
        ERROR, hostOutputs_.size() == rawOutputs_.size(), "Bad outputs", runId,
        ctx, resultCB);
    if (deviceOptions_->enabledCommandLists > 1) {
      uint32_t numErrors(0);
      NNPIInferenceErrorCode res = nnpiCommandListWait(
          commandList_, UINT32_MAX, commandErrors_, numCommands_, &numErrors);
      TRACE_EVENT_END(ctx->getTraceContext(), TraceLevel::OPERATOR,
                      TRACING_INFERENCE);
      TRACE_EVENT_BEGIN(ctx->getTraceContext(), TraceLevel::COPY,
                        TRACING_POST_PROCESS);
      if (res != NNPI_INF_NO_ERROR) {
        LOG_NNPI_INF_ERROR(res, "Failed to wait on command list");
      } else {
        for (uint32_t i = 0; i < numErrors; i++) {
          LOG(ERROR) << NNPI_INF_ERROR_MSG(commandErrors_[i].err,
                                           commandErrors_[i].desc);
        }
      }
      if (res != NNPI_INF_NO_ERROR || numErrors > 0) {
        LOG_AND_CALLBACK_EXECUTE_NNPI_INF_ERROR(
            nnpiCommandListClearErrors(commandList_),
            "Failed to clear command list errors", runId, ctx, resultCB);
        LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(
            ERROR, false /* fail */, "Errors found in command list execution",
            runId, ctx, resultCB);
      }
    }
    for (size_t i = 0, e = hostOutputs_.size(); i < e; i++) {
      void *pHostOutput(hostOutputs_[i].hostPtr);

      // Lock output host resource.
      if (deviceOptions_->enabledCommandLists < 2) {
        LOG_AND_CALLBACK_EXECUTE_NNPI_INF_ERROR(
            nnpiHostResourceLock(hostOutputs_[i].handle, NNPI_LOCK_FOR_READ,
                                 UINT32_MAX, &pHostOutput),
            "Failed to lock host resource", runId, ctx, resultCB);
        if (i == 0) {
          // Infer must be done and post proccessing begins after first lock is
          // obtained.
          TRACE_EVENT_END(ctx->getTraceContext(), TraceLevel::OPERATOR,
                          TRACING_INFERENCE);
          TRACE_EVENT_BEGIN(ctx->getTraceContext(), TraceLevel::COPY,
                            TRACING_POST_PROCESS);
        }
        LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(ERROR, pHostOutput, "Bad output",
                                             runId, ctx, resultCB);
      }

      // Copy data to output tensor.
      LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(
          ERROR, ioTensors_.count(hostOutputs_[i].name), "Can't find output",
          runId, ctx, resultCB);
      auto *t = ioTensors_.at(hostOutputs_[i].name);
      const bool downcastInt64 =
          t->getElementType() == glow::ElemKind::Int64ITy;
      const size_t bufferSize = t->getUnpaddedSizeInBytes();
      if (deviceOptions_->dumpIOtoFiles) {
        const size_t unpaddedSize = downcastInt64 ? bufferSize / 2 : bufferSize;
        dumpToFile(std::string("output_") + hostOutputs_[i].name + ".txt",
                   pHostOutput, unpaddedSize);
      }

      if (downcastInt64) {
        const size_t outputSize = bufferSize / t->getType().getElementSize();
        int64_t *pOutput = reinterpret_cast<int64_t *>(t->getUnsafePtr());
        int32_t *tmp = reinterpret_cast<int32_t *>(pHostOutput);
        for (size_t j = 0; j < outputSize; j++) {
          pOutput[j] = static_cast<int64_t>(tmp[j]);
        }
      } else {
        std::memcpy(t->getUnsafePtr(), pHostOutput, bufferSize);
      }

      if (deviceOptions_->enabledCommandLists < 2) {
        // Unlock host resource.
        LOG_AND_CALLBACK_EXECUTE_NNPI_INF_ERROR(
            nnpiHostResourceUnlock(hostOutputs_[i].handle),
            "Failed to unlock host resource", runId, ctx, resultCB);
      }
    }
    if (deviceTracing_ != nullptr) {
      deviceTracing_->stopAndUpdate(ctx->getTraceContext(), runId);
    }
  }

  for (auto *p : tmpBuffers_) {
    delete[](p);
  }
  tmpBuffers_.clear();
  rawInputs_.clear();
  rawOutputs_.clear();
  ioTensors_.clear();
  TRACE_EVENT_END(ctx->getTraceContext(), TraceLevel::COPY,
                  TRACING_POST_PROCESS);

  TRACE_EVENT_SCOPE_END(); // we move context in the line below

  // Invoke CB.
  resultCB(runId, Error::success(), std::move(ctx));
}

bool InferenceThreadEnv::init(
    // For ICE-Ref path.
    NNPINetwork network, NNPICompilationConfig config,
    // For ICE-T path.
    NNPIHostNetwork hostNetwork, NNPIDeviceNetwork deviceNetwork,
    NNPIAdapter adapter, NNPIDeviceContext device,
    const std::unordered_set<const Placeholder *> &partialInputs,
    const std::unordered_set<const Placeholder *> &staticInputs,
    std::shared_ptr<NNPIDeviceTracing> deviceTracing,
    NNPIStaticPlaceholderContainer *staticPlaceholderContainer,
    const NNPIDeviceOptions &deviceOptions) {
  deviceOptions_ = std::make_shared<NNPIDeviceOptions>(deviceOptions);

  nnpiNetwork_ = network;
  device_ = device;
  compilationConfig_ = config;
  partialInputs_ = &partialInputs;
  deviceTracing_ = deviceTracing;

  LOG_AND_RETURN_IF(ERROR, staticPlaceholderContainer == nullptr,
                    "InferenceThreadEnv Init was called with non-initialized "
                    "staticPlaceholderContainer",
                    false);
  staticPlaceholderContainer_ = staticPlaceholderContainer;

  /// Map from names to their Placeholders.
  std::unordered_map<std::string, const Placeholder *> staticPlaceholders;
  for (auto staticInput : staticInputs) {
    staticPlaceholders[staticInput->getName().str()] = staticInput;
    staticInputs_.emplace(staticInput);
  }

  if (!deviceOptions_->inferOnDevice) {
    size_t numInputs, numOutputs;
    NNPIObjectName name;
    NNPITensorDesc desc;
    LOG_NNPI_ERROR_RETURN_FALSE(
        nnpiNetworkGetInputNum(nnpiNetwork_, &numInputs),
        "Failed to query NNPI network inputs");
    for (size_t i = 0; i < numInputs; i++) {
      LOG_NNPI_ERROR_RETURN_FALSE(
          nnpiNetworkGetInputDesc(nnpiNetwork_, i, name, &desc),
          "Failed to query NNPI network inputs");
      LOG_AND_RETURN_IF(
          ERROR, !deviceOptions_->useIceT && staticPlaceholders.count(name),
          "ICE-Ref doesn't support static inputs", false);
      netInputs_.push_back({name, desc});
    }
    LOG_NNPI_ERROR_RETURN_FALSE(
        nnpiNetworkGetOutputNum(nnpiNetwork_, &numOutputs),
        "Failed to query NNPI network outputs");
    for (size_t i = 0; i < numOutputs; i++) {
      LOG_NNPI_ERROR_RETURN_FALSE(
          nnpiNetworkGetOutputDesc(nnpiNetwork_, i, name, &desc),
          "Failed to query NNPI network outputs");
      LOG_AND_RETURN_IF(
          ERROR, !deviceOptions_->useIceT && staticPlaceholders.count(name),
          "ICE-Ref doesn't support static outputs", false);
      netOutputs_.push_back({name, desc});
    }

    return true; // Nothing to be done here for ice-ref.
  }

  // Query input/output resources.
  uint32_t numInputs, numOutputs;
  LOG_NNPI_INF_ERROR_RETURN_FALSE(
      nnpiHostNetworkGetInputNum(hostNetwork, &numInputs),
      "Failed to query NNPI network inputs");
  LOG_NNPI_INF_ERROR_RETURN_FALSE(
      nnpiHostNetworkGetOutputNum(hostNetwork, &numOutputs),
      "Failed to query NNPI network outputs");

  // Create resources and copy commands.
  for (uint32_t i = 0; i < numInputs; i++) {
    NamedResource nr;
    LOG_NNPI_INF_ERROR_RETURN_FALSE(
        nnpiHostNetworkGetInputDesc(hostNetwork, i, nr.name, &nr.desc),
        "Failed to query NNPI host network input");
    // Host resource.
    NNPIHostResource hInput = 0;

    const auto isStaticInput = staticPlaceholders.count(nr.name);
    if (!isStaticInput) {
      netInputs_.push_back({nr.name, NNPITensorDesc()});
      LOG_NNPI_INF_ERROR_RETURN_FALSE(
          nnpiHostResourceCreate(adapter, &nr.desc, &nr.handle),
          "Failed to create NNPI host resource");

      // Lock/Unlock host resource and keep host address.
      LOG_NNPI_INF_ERROR_RETURN_FALSE(
          nnpiHostResourceLock(nr.handle, NNPI_LOCK_FOR_WRITE, UINT32_MAX,
                               &nr.hostPtr),
          "Failed to lock host resource");
      LOG_NNPI_INF_ERROR_RETURN_FALSE(nnpiHostResourceUnlock(nr.handle),
                                      "Failed to unlock host resource");

      hostInputs_.push_back(nr);
      hInput = nr.handle; // Save before overwriting with device handle.

      // Device resource.
      LOG_NNPI_INF_ERROR_RETURN_FALSE(
          nnpiDeviceResourceCreate(device_, &nr.desc, &nr.handle),
          "Failed to create NNPI device resource");
      allocatedDeviceInputs_.push_back(nr);
    } else {
      const auto PH = staticPlaceholders.at(nr.name);
      nr = staticPlaceholderContainer_->AcquireDeviceResource(PH, nr);
      // Exception for internal testing (ICE-24091).
      if (deviceOptions_->internalTesting.get().empty()) {
        LOG_AND_RETURN_IF(ERROR, nr.handle == NNPI_INVALID_NNPIHANDLE,
                          "Failed to acquire device resource", false);
      }
    }

    deviceInputs_.push_back(nr);

    if (!isStaticInput) {
      // Copy command.
      NNPICopyCommand copyInputCmd(NNPI_INVALID_NNPIHANDLE);
      LOG_NNPI_INF_ERROR_RETURN_FALSE(
          nnpiCopyCommandCreateHostToDevice(device_, nr.handle, hInput,
                                            &copyInputCmd),
          "Failed to create NNPI copy command");
      inputCopyCmds_.push_back(copyInputCmd);
    }
  }

  DBG_MEM_USAGE("Created input host resources and copy commands");
  // Create output resources and copy commands.
  for (uint32_t i = 0; i < numOutputs; i++) {
    // Host resource.
    NamedResource nr;

    LOG_NNPI_INF_ERROR_RETURN_FALSE(
        nnpiHostNetworkGetOutputDesc(hostNetwork, i, nr.name, &nr.desc),
        "Failed to query NNPI host network output");
    netOutputs_.push_back({nr.name, NNPITensorDesc()});

    const auto isStaticOutput = staticPlaceholders.count(nr.name);
    ASSERT_WITH_MSG(isStaticOutput == false, "Static outputs are illegal");

    LOG_NNPI_INF_ERROR_RETURN_FALSE(
        nnpiHostResourceCreate(adapter, &nr.desc, &nr.handle),
        "Failed to create NNPI host resource");

    // Lock/Unlock host resource and keep host address.
    LOG_NNPI_INF_ERROR_RETURN_FALSE(
        nnpiHostResourceLock(nr.handle, NNPI_LOCK_FOR_WRITE, UINT32_MAX,
                             &nr.hostPtr),
        "Failed to lock host resource");
    LOG_NNPI_INF_ERROR_RETURN_FALSE(nnpiHostResourceUnlock(nr.handle),
                                    "Failed to unlock host resource");

    hostOutputs_.push_back(nr);
    NNPIHostResource hOutput =
        nr.handle; // Save before overwriting with device handle.

    // Device resource.
    LOG_NNPI_INF_ERROR_RETURN_FALSE(
        nnpiDeviceResourceCreate(device_, &nr.desc, &nr.handle),
        "Failed to create NNPI device resource");
    deviceOutputs_.push_back(nr);

    // Copy command.
    NNPICopyCommand copyOutputCmd(NNPI_INVALID_NNPIHANDLE);
    LOG_NNPI_INF_ERROR_RETURN_FALSE(
        nnpiCopyCommandCreateDeviceToHost(device_, hOutput, nr.handle,
                                          &copyOutputCmd),
        "Failed to create NNPI copy command");
    outputCopyCmds_.push_back(copyOutputCmd);
  }

  // Create infer command.
  NNPIDeviceResource inputHandles[numInputs];
  NNPIDeviceResource outputHandles[numOutputs];
  for (uint32_t i = 0; i < numInputs; i++) {
    inputHandles[i] = deviceInputs_[i].handle;
  }
  for (uint32_t i = 0; i < numOutputs; i++) {
    outputHandles[i] = deviceOutputs_[i].handle;
  }

  LOG_NNPI_INF_ERROR_RETURN_FALSE(
      nnpiInferCommandCreate(deviceNetwork, inputHandles, numInputs,
                             outputHandles, numOutputs, &inferCmd_),
      "Failed to create NNPI inference command");

  if (deviceOptions.enabledCommandLists > 0) {
    numCommands_ = inputCopyCmds_.size() + outputCopyCmds_.size() +
                   1 /*1 for inference command*/;

    // Prepare empty command configurations.
    NNPICommandConfig cfg;
    memset(&cfg, 0, sizeof(NNPICommandConfig));
    for (size_t i = 0; i < numCommands_; i++) {
      cmdConfigs_.push_back(cfg);
    }

    // Create command list.
    NNPICommandHandle commands[numCommands_];
    LOG_AND_RETURN_IF_NOT(ERROR, commands,
                          "Failed to allocate command handle array", false);
    uint32_t cmdIdx = 0;
    for (auto cmd : inputCopyCmds_) {
      commands[cmdIdx].type = NNPI_COMMAND_TYPE_COPY;
      commands[cmdIdx].copyCommand = cmd;
      cmdIdx++;
    }
    commands[cmdIdx].type = NNPI_COMMAND_TYPE_INFER;
    commands[cmdIdx].inferCommand = inferCmd_;
    cmdIdx++;
    for (auto cmd : outputCopyCmds_) {
      commands[cmdIdx].type = NNPI_COMMAND_TYPE_COPY;
      commands[cmdIdx].copyCommand = cmd;
      cmdIdx++;
    }

    LOG_NNPI_INF_ERROR_RETURN_FALSE(nnpiCommandListCreate(commands,
                                                          numCommands_, nullptr,
                                                          0, &commandList_),
                                    "Failed to create NNPI command list");

    commandErrors_ = new NNPICommandListError[numCommands_];
    LOG_AND_RETURN_IF_NOT(ERROR, commandErrors_,
                          "Failed to allocate command error array", false);
    memset(commandErrors_, 0, sizeof(NNPICommandListError) * numCommands_);
  }

  return true;
}

InferencePoolEnv::InferencePoolEnv()
    : numWorkers_(0), hostNetwork_(NNPI_INVALID_NNPIHANDLE),
      deviceOptions_(nullptr) {}

InferencePoolEnv::~InferencePoolEnv() {
  if (deviceOptions_ && deviceOptions_->inferOnDevice) {
    if (hostNetwork_ != NNPI_INVALID_NNPIHANDLE) {
      LOG_NNPI_INF_ERROR(nnpiHostNetworkDestroy(hostNetwork_),
                         "Failed to destroy NNPI host network");
      hostNetwork_ = NNPI_INVALID_NNPIHANDLE;
    }
    if (deviceNetwork_ != NNPI_INVALID_NNPIHANDLE) {
      LOG_NNPI_INF_ERROR(nnpiDeviceNetworkDestroy(deviceNetwork_),
                         "Failed to destroy NNPI device network");
      deviceNetwork_ = NNPI_INVALID_NNPIHANDLE;
    }
  }
}

Error InferencePoolEnv::init(
    unsigned numWorkers, NNPIAdapter adapter, NNPIDeviceContext device,
    std::shared_ptr<NNPIDeviceTracing> deviceTracing,
    CompiledFunction *compiledFunction,
    NNPIStaticPlaceholderContainer *staticPlaceholderContainer,
    const NNPIDeviceOptions &deviceOptions) {
  deviceOptions_ = std::make_shared<NNPIDeviceOptions>(deviceOptions);
  if (workersPool_) {
    return MAKE_ERR("InferencePool already initialized!");
  }
  numWorkers_ = numWorkers;
  workersPool_ = std::make_unique<ThreadPool>(numWorkers_);
  deviceTracing_ = deviceTracing;

  threadEnvs_.resize(numWorkers_);
  if (threadEnvs_.size() != numWorkers_) {
    return MAKE_ERR("InferencePool failed to initialize thread env");
  }

  // Create host network.
  auto *nnpiFunction = static_cast<NNPICompiledFunction *>(compiledFunction);
  if (deviceOptions_->inferOnDevice) {
    // Create NNPI host network (load compiled binary).
    auto filename = nnpiFunction->getCompilationFilename();
    if (filename.empty()) // Create network from memory.
    {
      NNPIHostStream inputStream;
      inputStream.userData = &(nnpiFunction->lockCompiledStream());
      inputStream.readCallback = [](void *ptr, uint64_t size, uint64_t count,
                                    void *userData) -> uint64_t {
        BlockStream *ss = reinterpret_cast<BlockStream *>(userData);
        size_t readSize = ss->read(static_cast<char *>(ptr), size * count);
        return readSize;
      };
      inputStream.writeCallback = NULL;
      inputStream.seekCallback = NULL;
      DBG_MEM_USAGE("call nnpiHostNetworkCreateFromStream");
      LOG_NNPI_INF_ERROR_RETURN_LLVMERROR(
          nnpiHostNetworkCreateFromStream(adapter, &inputStream, &hostNetwork_),
          "Failed to create NNPI host network");
      DBG_MEM_USAGE("done nnpiHostNetworkCreateFromStream");
      nnpiFunction->unlockCompiledStream();
    } else // Create network from file.
    {
      filename += ".zip";
      LOG_NNPI_INF_ERROR_RETURN_LLVMERROR(
          nnpiHostNetworkCreateFromFile(adapter, filename.c_str(),
                                        &hostNetwork_),
          "Failed to create NNPI host network");
    }

    DBG_MEM_USAGE("call nnpiDeviceNetworkCreate");
    // Create NNPI device network (deploy to device).
    LOG_NNPI_INF_ERROR_RETURN_LLVMERROR(
        nnpiDeviceNetworkCreate(device, hostNetwork_, nullptr, &deviceNetwork_),
        "Failed to create NNPI device network");
    DBG_MEM_USAGE("done nnpiDeviceNetworkCreate");
    if (nnpiFunction->getCompilationOptions().reserveResources) {
      LOG_NNPI_INF_ERROR_RETURN_LLVMERROR(
          nnpiDeviceNetworkReserveExecResources(deviceNetwork_, UINT32_MAX),
          "Failed to reserve resources for device network");
    }
  }

  // Initialize all thread envs.
  for (auto &tEnv : threadEnvs_) {
    auto success = tEnv.init(nnpiFunction->getCompiledNetworkHandle(),
                             nnpiFunction->getCompilationConfig(), hostNetwork_,
                             deviceNetwork_, adapter, device,
                             nnpiFunction->getPartialInputs(),
                             nnpiFunction->getStaticInputs(), deviceTracing_,
                             staticPlaceholderContainer, deviceOptions);
    if (!success) {
      return MAKE_ERR("Failed to initialize thread env");
    }
  }
  if (deviceOptions_->inferOnDevice &&
      hostNetwork_ != NNPI_INVALID_NNPIHANDLE) {
    DBG_MEM_USAGE("call nnpiHostNetworkDestroy");
    LOG_NNPI_INF_ERROR(nnpiHostNetworkDestroy(hostNetwork_),
                       "Failed to destroy NNPI host network");
    hostNetwork_ = NNPI_INVALID_NNPIHANDLE;
    DBG_MEM_USAGE("done nnpiHostNetworkDestroy");
  }
  return Error::success();
}

void InferencePoolEnv::stop(bool block) { workersPool_->stop(block); }

void InferencePoolEnv::execute(RunIdentifierTy runId,
                               std::unique_ptr<ExecutionContext> ctx,
                               runtime::ResultCBTy resultCB) {
  unsigned id = (workerIndex_++) % numWorkers_;
  workersPool_->submit([this, env = &(threadEnvs_.at(id)), runId,
                        ctx = std::move(ctx),
                        resultCB = std::move(resultCB)]() mutable {
    env->execute(runId, std::move(ctx), resultCB);
  });
}

} // namespace runtime
} // namespace glow
