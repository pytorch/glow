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

namespace glow {
namespace runtime {

InferencePoolEnv::InferencePoolEnv()
    : numWorkers_(0), hostNetwork_(NNPI_INVALID_NNPIHANDLE),
      deviceOptions_(nullptr) {}

InferencePoolEnv::~InferencePoolEnv() {
  if (deviceOptions_ && deviceOptions_->inferOnDevice) {
    if (hostNetwork_ != NNPI_INVALID_NNPIHANDLE) {
      LOG_NNPI_INF_IF_ERROR(nnpiHostNetworkDestroy(hostNetwork_),
                            "Failed to destroy NNPI host network");
      hostNetwork_ = NNPI_INVALID_NNPIHANDLE;
    }
    if (deviceNetwork_ != NNPI_INVALID_NNPIHANDLE) {
      LOG_NNPI_INF_IF_ERROR(nnpiDeviceNetworkDestroy(deviceNetwork_),
                            "Failed to destroy NNPI device network");
      deviceNetwork_ = NNPI_INVALID_NNPIHANDLE;
    }
  }
}

Error InferencePoolEnv::init(unsigned numWorkers, NNPIAdapter adapter,
                             NNPIDeviceContext device,
                             std::shared_ptr<NNPIDeviceTracing> deviceTracing,
                             CompiledFunction *compiledFunction,
                             StaticPlaceholderMap *staticPlaceholderMap,
                             std::shared_ptr<NNPIDeviceOptions> deviceOptions,
                             const std::string &functionName,
                             unsigned deviceId) {
  deviceOptions_ = deviceOptions;
  deviceId_ = deviceId;
  if (workersPool_) {
    return MAKE_ERR("InferencePool already initialized!");
  }
  numWorkers_ = numWorkers;
  workersPool_ = glow::make_unique<folly::CPUThreadPoolExecutor>(
      numWorkers_, std::make_shared<folly::NamedThreadFactory>("NNPI-worker"));
  deviceTracing_ = deviceTracing;

  inferenceContexts_.resize(numWorkers_);
  freeContexts_.resize(numWorkers_);
  if (inferenceContexts_.size() != numWorkers_) {
    return MAKE_ERR("InferencePool failed to create inference contexts");
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
      LOG_NNPI_INF_IF_ERROR_RETURN_LLVMERROR(
          nnpiHostNetworkCreateFromStream(adapter, &inputStream, &hostNetwork_),
          "Failed to create NNPI host network");
      DBG_MEM_USAGE("done nnpiHostNetworkCreateFromStream");
      nnpiFunction->unlockCompiledStream();
    } else // Create network from file.
    {
      filename += ".zip";
      LOG_NNPI_INF_IF_ERROR_RETURN_LLVMERROR(
          nnpiHostNetworkCreateFromFile(adapter, filename.c_str(),
                                        &hostNetwork_),
          "Failed to create NNPI host network");
    }

    DBG_MEM_USAGE("call nnpiDeviceNetworkCreate");
    // Create NNPI device network (deploy to device).
    LOG_NNPI_INF_IF_ERROR_RETURN_LLVMERROR(
        nnpiDeviceNetworkCreate(device, hostNetwork_, nullptr, &deviceNetwork_),
        "Failed to create NNPI device network");
    DBG_MEM_USAGE("done nnpiDeviceNetworkCreate");
    if (nnpiFunction->getCompilationOptions().reserveResources) {
      LOG_NNPI_INF_IF_ERROR_RETURN_LLVMERROR(
          nnpiDeviceNetworkReserveExecResources(deviceNetwork_, UINT32_MAX),
          "Failed to reserve resources for device network");
    }
  }

  for (auto &infCtx : inferenceContexts_) {
    auto success = infCtx.init(
        nnpiFunction->getCompiledNetworkHandle(),
        nnpiFunction->getCompilationConfig(), hostNetwork_, deviceNetwork_,
        adapter, device, nnpiFunction->getPartialInputs(),
        nnpiFunction->getStaticInputs(), deviceTracing_, staticPlaceholderMap,
        deviceOptions, functionName, deviceId_);
    if (!success) {
      return MAKE_ERR("Failed to initialize inferece context");
    }
    freeContexts_.push_back(&infCtx);
  }

  if (deviceOptions_->inferOnDevice &&
      hostNetwork_ != NNPI_INVALID_NNPIHANDLE) {
    DBG_MEM_USAGE("call nnpiHostNetworkDestroy");
    LOG_NNPI_INF_IF_ERROR(nnpiHostNetworkDestroy(hostNetwork_),
                          "Failed to destroy NNPI host network");
    hostNetwork_ = NNPI_INVALID_NNPIHANDLE;
    DBG_MEM_USAGE("done nnpiHostNetworkDestroy");
  }
  return Error::success();
}

void InferencePoolEnv::stop(bool block) {
  workersPool_->stop();
  if (block) {
    workersPool_->join();
  }
}

void InferencePoolEnv::execute(RunIdentifierTy runId,
                               std::unique_ptr<ExecutionContext> ctx,
                               runtime::ResultCBTy resultCB) {
  workersPool_->add([this, runId, ctx = std::move(ctx),
                     resultCB = std::move(resultCB)]() mutable {
    InferenceContext *infCtx = nullptr;
    {
      const std::lock_guard<std::mutex> lock(freeContextsLock_);
      CHECK(!freeContexts_.empty());
      infCtx = *freeContexts_.rbegin();
      freeContexts_.pop_back();
    }
    infCtx->execute(runId, std::move(ctx), resultCB);
    {
      const std::lock_guard<std::mutex> lock(freeContextsLock_);
      freeContexts_.push_back(infCtx);
    }
  });
}

} // namespace runtime
} // namespace glow
