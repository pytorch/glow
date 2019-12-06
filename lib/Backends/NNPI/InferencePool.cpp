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
#include "llvm/Support/raw_ostream.h"

namespace glow {
namespace runtime {

InferenceThreadEnv::InferenceThreadEnv()
    : nnpiNetwork_(NNPI_INVALID_NNPIHANDLE), device_(NNPI_INVALID_NNPIHANDLE),
      inferCmd_(NNPI_INVALID_NNPIHANDLE) {}

InferenceThreadEnv::~InferenceThreadEnv() {
  if (UseInferenceAPI()) {
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
    for (auto &nr : deviceInputs_) {
      LOG_NNPI_INF_ERROR(nnpiDeviceResourceDestroy(nr.handle),
                         "Failed to destroy NNPI device resource");
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
  TRACE_EVENT_SCOPE(ctx->getTraceContext(), TraceLevel::REQUEST, "execute");
  if (ctx->getTraceContext()) {
    ctx->getTraceContext()->setThreadName("InferenceThreadEnv");
  }

  // Pre inference input preparation.
  PlaceholderBindings &bindings = *ctx->getPlaceholderBindings();
  ioTensors_.clear();

  std::unordered_set<Tensor *> partialTensorInputs;
  for (auto &pht : bindings.pairs()) {
    ioTensors_.emplace(pht.first->getName(), pht.second);
    if (partialInputs_->count(pht.first)) {
      partialTensorInputs.insert(pht.second);
    }
  }

  // Handle inputs & outputs (+convert).
  for (auto &in : netInputs_) {
    LOG_AND_FAIL_CALLBACK_IF_NOT(ERROR, ioTensors_.count(in.first),
                                 "Can't find tensor for input", runId, ctx,
                                 resultCB);
    auto *t = ioTensors_.at(in.first);
    char *bufferPtr = t->getUnsafePtr();

    // Check if we need to allocate a new temporary buffer to handle this input.
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

      // Copy over the original data. Downcast int64 tensors to int32 if needed.
      if (downcastInt64) {
        const int64_t *pInput = reinterpret_cast<int64_t *>(bufferPtr);
        int32_t *tmp32 = reinterpret_cast<int32_t *>(tmp);
        const size_t hostElems = unpaddedSize / sizeof(int32_t);
        for (size_t i = 0; i < hostElems; i++) {
          tmp32[i] = static_cast<int32_t>(pInput[i]);
        }
      } else {
        // If we don't need to downcast then this must be an unhandled partial,
        // so copy over the original data before we pad below.
        memcpy(tmp, bufferPtr, unpaddedSize);
      }
      bufferPtr = tmp;

      // If this tensor cannot be treated as partial but it is partial, then we
      // need to zero out anything that's leftover in the extra padding, as it
      // is still going to be used.
      if (padUnhandledPartial) {
        // Zero out rest of the new buffer.
        memset(bufferPtr + unpaddedSize, 0, paddedSize - unpaddedSize);
      }
    }

    rawInputs_.push_back(bufferPtr);
  }

  for (auto &out : netOutputs_) {
    LOG_AND_FAIL_CALLBACK_IF_NOT(ERROR, ioTensors_.count(out.first),
                                 "Can't find tensor for output", runId, ctx,
                                 resultCB);
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

  {
    TRACE_EVENT_SCOPE(ctx->getTraceContext(), TraceLevel::REQUEST,
                      UseInferenceAPI() ? "queuing" : "running");
    if (UseInferenceAPI()) {
      if (deviceTracing_ != nullptr) {
        deviceTracing_->start(ctx->getTraceContext(), runId);
      }
      // Copy data to host resource and preprocess int64.
      // Queue copy commands.
      // For every input: lock host, copy data (+convert), unlock.
      LOG_AND_FAIL_CALLBACK_IF_NOT(ERROR,
                                   hostInputs_.size() == rawInputs_.size(),
                                   "Bad inputs", runId, ctx, resultCB);
      for (size_t i = 0, e = hostInputs_.size(); i < e; i++) {
        void *pHostInput(nullptr);
        // Lock input host resource.
        LOG_AND_CALLBACK_NNPI_INF_ERROR(
            nnpiHostResourceLock(hostInputs_[i].handle, NNPI_LOCK_FOR_WRITE,
                                 UINT32_MAX, &pHostInput),
            "Failed to lock host resource", runId, ctx, resultCB);
        LOG_AND_FAIL_CALLBACK_IF_NOT(ERROR, pHostInput, "Bad input", runId, ctx,
                                     resultCB);

        LOG_AND_FAIL_CALLBACK_IF_NOT(ERROR,
                                     ioTensors_.count(hostInputs_[i].name),
                                     "Input not found", runId, ctx, resultCB);
        auto *t = ioTensors_.at(hostInputs_[i].name);

        size_t bufferSize = t->getUnpaddedSizeInBytes();
        size_t fullBufferSize = t->getSizeInBytes();

        if (t->getElementType() == glow::ElemKind::Int64ITy) {
          // Type int64 is converted to int32 (half number of bytes).
          bufferSize /= 2;
          fullBufferSize /= 2;
        }
        NNPICopyCommandConfig *copyConfig = nullptr;
        if (bufferSize != fullBufferSize && partialTensorInputs.count(t)) {
          copyConfig = &inputCopyCmdConfigs_[i];
          copyConfig->size = bufferSize;
          std::memcpy(pHostInput, rawInputs_[i], bufferSize);
        } else {
          std::memcpy(pHostInput, rawInputs_[i], fullBufferSize);
        }
        // Unlock host resource.
        LOG_AND_CALLBACK_NNPI_INF_ERROR(
            nnpiHostResourceUnlock(hostInputs_[i].handle),
            "Failed to unlock host resource", runId, ctx, resultCB);

        LOG_AND_CALLBACK_NNPI_INF_ERROR(
            nnpiCopyCommandQueue(inputCopyCmds_[i], copyConfig),
            "Failed to queue input copy command.", runId, ctx, resultCB);
      }
      if (deviceTracing_ != nullptr) {
        deviceTracing_->startCopyTime();
      }
    }

    // Inference.
    if (UseInferenceAPI()) {
      LOG_AND_CALLBACK_NNPI_INF_ERROR(nnpiInferCommandQueue(inferCmd_, 0),
                                      "Failed to queue infer command.", runId,
                                      ctx, resultCB);
      TRACE_EVENT_SCOPE(ctx->getTraceContext(), TraceLevel::REQUEST,
                        "infer_queue");
      for (size_t i = 0, e = hostOutputs_.size(); i < e; i++) {
        auto *t = ioTensors_.at(hostOutputs_[i].name);
        size_t bufferSize = t->getUnpaddedSizeInBytes();
        size_t fullBufferSize = t->getSizeInBytes();
        NNPICopyCommandConfig *copyConfig = nullptr;
        if (bufferSize != fullBufferSize) {
          copyConfig = &outputCopyCmdConfigs_[i];
          copyConfig->size = bufferSize;
          if (t->getElementType() == glow::ElemKind::Int64ITy) {
            // Type int64 is converted to int32 (half number of bytes).
            copyConfig->size /= 2;
          }
        }
        LOG_AND_CALLBACK_NNPI_INF_ERROR(
            nnpiCopyCommandQueue(outputCopyCmds_[i], copyConfig),
            "Failed to queue output copy command.", runId, ctx, resultCB);
      }
    } else if (!UseIceT()) {
      // Infer on ice-ref.
      LOG_AND_CALLBACK_NNPI_ERROR(
          nnpiNetworkInferOnHost(nnpiNetwork_, &(rawInputs_[0]),
                                 rawInputs_.size(), &(rawOutputs_[0]),
                                 rawOutputs_.size(), &compilationConfig_,
                                 NNPI_INVALID_NNPIHANDLE),
          "Failed NNPI infer (ICE-Ref)", runId, ctx, resultCB);

      // Convert outputs.
      size_t currOut = 0;
      for (auto &out : netOutputs_) {
        LOG_AND_FAIL_CALLBACK_IF_NOT(ERROR, ioTensors_.count(out.first),
                                     "Output not found", runId, ctx, resultCB);
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
      for (auto &out : netOutputs_) {
        LOG_AND_FAIL_CALLBACK_IF_NOT(ERROR, ioTensors_.count(out.first),
                                     "Output not found", runId, ctx, resultCB);
        auto *t = ioTensors_.at(out.first);
        t->zero();
      }
    }
  }

  {
    TRACE_EVENT_SCOPE(ctx->getTraceContext(), TraceLevel::REQUEST,
                      UseInferenceAPI() ? "running on device"
                                        : "copying output");
    // Post inference output handling.
    if (UseInferenceAPI()) {
      TRACE_EVENT_SCOPE(ctx->getTraceContext(), TraceLevel::REQUEST,
                        "copy_output");
      // For every output, lock and copy data (+convert), unlock
      // then copy to output tensors.
      LOG_AND_FAIL_CALLBACK_IF_NOT(ERROR,
                                   hostOutputs_.size() == rawOutputs_.size(),
                                   "Bad outputs", runId, ctx, resultCB);
      for (size_t i = 0, e = hostOutputs_.size(); i < e; i++) {
        void *pHostOutput(nullptr);

        // Lock output host resource.
        LOG_AND_CALLBACK_NNPI_INF_ERROR(
            nnpiHostResourceLock(hostOutputs_[i].handle, NNPI_LOCK_FOR_READ,
                                 UINT32_MAX, &pHostOutput),
            "Failed to lock host resource", runId, ctx, resultCB);
        LOG_AND_FAIL_CALLBACK_IF_NOT(ERROR, pHostOutput, "Bad output", runId,
                                     ctx, resultCB);

        // Copy data to output tensor.
        LOG_AND_FAIL_CALLBACK_IF_NOT(ERROR,
                                     ioTensors_.count(hostOutputs_[i].name),
                                     "Can't find output", runId, ctx, resultCB);
        auto *t = ioTensors_.at(hostOutputs_[i].name);
        size_t bufferSize = t->getUnpaddedSizeInBytes();

        if (t->getElementType() == glow::ElemKind::Int64ITy) {
          const size_t outputSize = bufferSize / t->getType().getElementSize();
          int64_t *pOutput = reinterpret_cast<int64_t *>(t->getUnsafePtr());
          int32_t *tmp = reinterpret_cast<int32_t *>(pHostOutput);
          for (size_t j = 0; j < outputSize; j++) {
            pOutput[j] = static_cast<int64_t>(tmp[j]);
          }
        } else {
          std::memcpy(t->getUnsafePtr(), pHostOutput, bufferSize);
        }

        // Unlock host resource.
        LOG_AND_CALLBACK_NNPI_INF_ERROR(
            nnpiHostResourceUnlock(hostOutputs_[i].handle),
            "Failed to unlock host resource", runId, ctx, resultCB);
        if (deviceTracing_ != nullptr) {
          deviceTracing_->stopAndUpdate(ctx->getTraceContext(), runId);
        }
      }
    }
  }

  for (auto *p : tmpBuffers_) {
    delete[](p);
  }
  tmpBuffers_.clear();
  rawInputs_.clear();
  rawOutputs_.clear();
  ioTensors_.clear();

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
    std::shared_ptr<NNPIDeviceTracing> deviceTracing) {

  nnpiNetwork_ = network;
  device_ = device;
  compilationConfig_ = config;
  partialInputs_ = &partialInputs;
  deviceTracing_ = deviceTracing;

  if (!UseInferenceAPI()) {
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
      netInputs_.push_back({name, desc});
    }
    LOG_NNPI_ERROR_RETURN_FALSE(
        nnpiNetworkGetOutputNum(nnpiNetwork_, &numOutputs),
        "Failed to query NNPI network outputs");
    for (size_t i = 0; i < numOutputs; i++) {
      LOG_NNPI_ERROR_RETURN_FALSE(
          nnpiNetworkGetOutputDesc(nnpiNetwork_, i, name, &desc),
          "Failed to query NNPI network outputs");
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
    // Host resource.
    NamedResource nr;
    LOG_NNPI_INF_ERROR_RETURN_FALSE(
        nnpiHostNetworkGetInputDesc(hostNetwork, i, nr.name, &nr.desc),
        "Failed to query NNPI host network input");
    netInputs_.push_back({nr.name, NNPITensorDesc()});

    LOG_NNPI_INF_ERROR_RETURN_FALSE(
        nnpiHostResourceCreate(adapter, &nr.desc, &nr.handle),
        "Failed to create NNPI host resource");
    hostInputs_.push_back(nr);
    NNPIHostResource hInput =
        nr.handle; // Save before overwriting with device handle.

    // Device resource.
    LOG_NNPI_INF_ERROR_RETURN_FALSE(
        nnpiDeviceResourceCreate(device_, &nr.desc, &nr.handle),
        "Failed to create NNPI device resource");
    deviceInputs_.push_back(nr);

    // Copy command.
    NNPICopyCommand copyInputCmd(NNPI_INVALID_NNPIHANDLE);
    LOG_NNPI_INF_ERROR_RETURN_FALSE(
        nnpiCopyCommandCreateHostToDevice(device_, nr.handle, hInput,
                                          &copyInputCmd),
        "Failed to create NNPI copy command");
    inputCopyCmds_.push_back(copyInputCmd);

    // Allocate copy command config struct (size will be assigned later for
    // partial copy).
    NNPICopyCommandConfig copyConfig;
    memset(&copyConfig, 0, sizeof(NNPICopyCommandConfig));
    inputCopyCmdConfigs_.push_back(copyConfig);
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

    LOG_NNPI_INF_ERROR_RETURN_FALSE(
        nnpiHostResourceCreate(adapter, &nr.desc, &nr.handle),
        "Failed to create NNPI host resource");
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

    // Allocate copy command config struct (size will be assigned later for
    // partial copy).
    NNPICopyCommandConfig copyConfig;
    memset(&copyConfig, 0, sizeof(NNPICopyCommandConfig));
    outputCopyCmdConfigs_.push_back(copyConfig);
  }

  // Create infer command.
  NNPIDeviceResource *inputHandles = new NNPIDeviceResource[numInputs];
  NNPIDeviceResource *outputHandles = new NNPIDeviceResource[numOutputs];
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

  delete[] inputHandles;
  delete[] outputHandles;

  return true;
}

InferencePoolEnv::InferencePoolEnv()
    : numWorkers_(0), hostNetwork_(NNPI_INVALID_NNPIHANDLE) {}

InferencePoolEnv::~InferencePoolEnv() {
  if (UseInferenceAPI()) {
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

Error InferencePoolEnv::init(unsigned numWorkers, NNPIAdapter adapter,
                             NNPIDeviceContext device,
                             std::shared_ptr<NNPIDeviceTracing> deviceTracing,
                             CompiledFunction *compiledFunction) {
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
  if (UseInferenceAPI()) {
    // Create NNPI host network (load compiled binary).
    auto filename = ICETFilename();
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
  }

  // Initialize all thread envs.
  for (auto &tEnv : threadEnvs_) {
    auto success = tEnv.init(nnpiFunction->getCompiledNetworkHandle(),
                             nnpiFunction->getCompilationConfig(), hostNetwork_,
                             deviceNetwork_, adapter, device,
                             nnpiFunction->getPartialInputs(), deviceTracing_);
    if (!success) {
      return MAKE_ERR("Failed to initialize thread env");
    }
  }
  if (UseInferenceAPI() && hostNetwork_ != NNPI_INVALID_NNPIHANDLE) {
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
