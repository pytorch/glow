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
#include "NNPIUtils.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <iomanip>
#include <sstream>

namespace glow {
namespace runtime {

static bool isEmptyDeviceNetworkConfig(const NNPIDeviceNetworkConfig &cfg) {
  if (cfg.disableECC != 0) {
    return false;
  }

  if (cfg.pnpHints.ringFrequencyPrio != 0.f) {
    return false;
  }

  const int numIceBO = sizeof(cfg.pnpHints.iceBOFrequencyPrio) /
                       sizeof(cfg.pnpHints.iceBOFrequencyPrio[0]);
  for (int i = 0; i < numIceBO; i++) {
    if (cfg.pnpHints.iceBOFrequencyPrio[i] != 0.f) {
      return false;
    }
  }

  if (cfg.pnpHints.DDRBandwidth != 0.f) {
    return false;
  }
  return true;
}

InferencePoolEnv::InferencePoolEnv()
    : deviceOptions_(nullptr), nnpiCompiledFunction_(nullptr),
      staticPlaceholderMap_(nullptr) {}

InferencePoolEnv::~InferencePoolEnv() {
  if (deviceOptions_ && deviceOptions_->inferOnDevice) {
    if (deviceNetwork_ != NNPI_INVALID_NNPIHANDLE) {
      LOG_NNPI_INF_IF_ERROR(nnpiDeviceNetworkDestroy(deviceNetwork_),
                            "Failed to destroy NNPI device network");
      deviceNetwork_ = NNPI_INVALID_NNPIHANDLE;
    }
  }
}

Error InferencePoolEnv::init(NNPIAdapterContainer *adapter,
                             NNPIDeviceContext device,
                             CompiledFunction *compiledFunction,
                             StaticPlaceholderMap *staticPlaceholderMap,
                             std::shared_ptr<NNPIDeviceOptions> deviceOptions,
                             const std::string &functionName,
                             unsigned deviceId) {
  deviceOptions_ = deviceOptions;
  deviceId_ = deviceId;
  functionName_ = functionName;
  device_ = device;
  pAdapter_ = adapter;
  if (workersPool_) {
    return MAKE_ERR(
        strFormat("InferencePool already initialized for function %s!",
                  functionName.c_str()));
  }

  nnpiCompiledFunction_ = static_cast<NNPICompiledFunction *>(compiledFunction);
  size_t optionsNumWorkers =
      nnpiCompiledFunction_->getCompilationOptions().numWorkers;
  // Ice-ref not re-entrant for the same nnpiNetwork.
  size_t numWorkers = deviceOptions_->inferOnDevice ? optionsNumWorkers : 1;
  workersPool_ = glow::make_unique<folly::CPUThreadPoolExecutor>(
      numWorkers, std::make_shared<folly::NamedThreadFactory>("NNPI-worker"));
  staticPlaceholderMap_ = staticPlaceholderMap;

  inferenceContexts_.resize(numWorkers);
  freeContexts_.resize(numWorkers);
  if (inferenceContexts_.size() != numWorkers) {
    return MAKE_ERR(strFormat(
        "InferencePool failed to create inference contexts for function %s",
        functionName.c_str()));
  }

  // Create host network.
  NNPIHostNetwork hostNetwork(NNPI_INVALID_NNPIHANDLE);
  if (deviceOptions_->inferOnDevice) {
    // Load IA extenstions.
    for (auto &extensionPath : nnpiCompiledFunction_->getIAExtensionPaths()) {
      NNPIExtension ext;
      LOG_NNPI_INF_IF_ERROR_RETURN_LLVMERROR(
          nnpiExtensionCreate(extensionPath.c_str(), &ext),
          strFormat("Failed to create NNPI IA Extension object for function %s",
                    functionName.c_str()));
      LOG_NNPI_INF_IF_ERROR_RETURN_LLVMERROR(
          nnpiDeviceContextLoadExtension(device, ext),
          strFormat("Failed to load NNPI IA Extension object for function %s",
                    functionName.c_str()));
    }

#if NNPI_INF_MAJOR_VERSION >= 1 || NNPI_INF_MINOR_VERSION >= 11
    for (auto &extensionLib : nnpiCompiledFunction_->getIAExtensionLibs()) {
      NNPIExtension ext;
      LOG_NNPI_INF_IF_ERROR_RETURN_LLVMERROR(
          nnpiExtensionCreateLoadBin(extensionLib.first.c_str(),
                                     extensionLib.second.data(),
                                     extensionLib.second.size(), &ext),
          strFormat("Failed to create NNPI IA Extension object for function %s",
                    functionName.c_str()));
      LOG_NNPI_INF_IF_ERROR_RETURN_LLVMERROR(
          nnpiDeviceContextLoadExtension(device, ext),
          strFormat("Failed to load NNPI IA Extension object for function %s",
                    functionName.c_str()));
    }
#endif // NNPI INF >= 0.11

    // Create NNPI host network (load compiled binary).
    auto filename = nnpiCompiledFunction_->getCompilationFilename();
    if (filename.empty()) // Create network from memory.
    {
      NNPIHostStream inputStream;
      inputStream.userData = &(nnpiCompiledFunction_->lockCompiledStream());
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
          nnpiHostNetworkCreateFromStream(pAdapter_->getHandle(), &inputStream,
                                          &hostNetwork),
          strFormat("Failed to create NNPI host network for function %s",
                    functionName.c_str()));
      DBG_MEM_USAGE("done nnpiHostNetworkCreateFromStream");
      nnpiCompiledFunction_->unlockCompiledStream();
    } else // Create network from file.
    {
      filename += ".zip";
      LOG_NNPI_INF_IF_ERROR_RETURN_LLVMERROR(
          nnpiHostNetworkCreateFromFile(pAdapter_->getHandle(),
                                        filename.c_str(), &hostNetwork),
          strFormat("Failed to create NNPI host network for function %s",
                    functionName.c_str()));
    }

    DBG_MEM_USAGE("call nnpiDeviceNetworkCreate");
    NNPIDeviceNetworkConfig cfg =
        nnpiCompiledFunction_->getDeviceNetworkConfig();
    NNPIDeviceNetworkConfig *pCfg = nullptr;
    if (!isEmptyDeviceNetworkConfig(cfg)) {
      pCfg = &cfg;
      LOG(INFO) << "DeviceNetwork PnP: "
                << "\n";
      LOG(INFO) << "  Ring: " << cfg.pnpHints.ringFrequencyPrio << "\n";
      LOG(INFO) << "  ICEBO 0: " << cfg.pnpHints.iceBOFrequencyPrio[0] << "\n";
      LOG(INFO) << "  ICEBO 1: " << cfg.pnpHints.iceBOFrequencyPrio[1] << "\n";
      LOG(INFO) << "  ICEBO 2: " << cfg.pnpHints.iceBOFrequencyPrio[2] << "\n";
      LOG(INFO) << "  ICEBO 3: " << cfg.pnpHints.iceBOFrequencyPrio[3] << "\n";
      LOG(INFO) << "  ICEBO 4: " << cfg.pnpHints.iceBOFrequencyPrio[4] << "\n";
      LOG(INFO) << "  ICEBO 5: " << cfg.pnpHints.iceBOFrequencyPrio[5] << "\n";
      LOG(INFO) << "  DDR: " << cfg.pnpHints.DDRBandwidth << "\n";
      LOG(INFO)
          << "  Resource reservation: "
          << nnpiCompiledFunction_->getCompilationOptions().reserveResources
          << "\n";
    }

    // Create NNPI device network (deploy to device).
    LOG_NNPI_INF_IF_ERROR_RETURN_FATAL_LLVMERROR(
        nnpiDeviceNetworkCreate(device, hostNetwork, pCfg, &deviceNetwork_),
        strFormat("Failed to create NNPI device network for function %s",
                  functionName.c_str()));
    DBG_MEM_USAGE("done nnpiDeviceNetworkCreate");
    if (nnpiCompiledFunction_->getCompilationOptions().reserveResources) {
      LOG_NNPI_INF_IF_ERROR_RETURN_LLVMERROR(
          nnpiDeviceNetworkReserveExecResources(deviceNetwork_, UINT32_MAX),
          strFormat(
              "Failed to reserve resources for device network for function %s",
              functionName.c_str()));
    }

    // Collect input/output descriptors from host network
    uint32_t numInputs, numOutputs;
    LOG_NNPI_INF_IF_ERROR_RETURN_LLVMERROR(
        nnpiHostNetworkGetInputNum(hostNetwork, &numInputs),
        strFormat("Failed to query NNPI network inputs for function %s",
                  functionName.c_str()));
    LOG_NNPI_INF_IF_ERROR_RETURN_LLVMERROR(
        nnpiHostNetworkGetOutputNum(hostNetwork, &numOutputs),
        strFormat("Failed to query NNPI network outputs for function %s",
                  functionName.c_str()));
    NNPIObjectName name;
    NNPIResourceDesc desc;
    for (uint32_t i = 0; i < numInputs; i++) {
      NNPIObjectName name;
      NNPIResourceDesc desc;
      LOG_NNPI_INF_IF_ERROR_RETURN_LLVMERROR(
          nnpiHostNetworkGetInputDesc(hostNetwork, i, name, &desc),
          strFormat("Failed to query NNPI host network input for function %s",
                    functionName.c_str()));
      memset(&desc.hostAttrib, 0, sizeof(desc.hostAttrib));
      memset(&desc.deviceAttrib, 0, sizeof(desc.deviceAttrib));
      inputDesc_.push_back({name, desc});
    }
    for (uint32_t i = 0; i < numOutputs; i++) {

      LOG_NNPI_INF_IF_ERROR_RETURN_LLVMERROR(
          nnpiHostNetworkGetOutputDesc(hostNetwork, i, name, &desc),
          strFormat("Failed to query NNPI host network output for function %s",
                    functionName.c_str()));
      memset(&desc.hostAttrib, 0, sizeof(desc.hostAttrib));
      memset(&desc.deviceAttrib, 0, sizeof(desc.deviceAttrib));
      outputDesc_.push_back({name, desc});
    }
  } else {
    // Collect input/output descriptors from nnpi network (for ICE-Ref)
    size_t numInputs, numOutputs;
    NNPIObjectName name;
    NNPITensorDesc desc;
    auto nnpiNetwork = nnpiCompiledFunction_->getCompiledNetworkHandle();
    LOG_NNPI_IF_ERROR_RETURN_LLVMERROR(
        nnpiNetworkGetInputNum(nnpiNetwork, &numInputs),
        strFormat("Failed to query NNPI network inputs for function %s",
                  functionName.c_str()));
    LOG_NNPI_IF_ERROR_RETURN_LLVMERROR(
        nnpiNetworkGetOutputNum(nnpiNetwork, &numOutputs),
        strFormat("Failed to query NNPI network outputs for function %s",
                  functionName.c_str()));

    for (size_t i = 0; i < numInputs; i++) {
      LOG_NNPI_IF_ERROR_RETURN_LLVMERROR(
          nnpiNetworkGetInputDesc(nnpiNetwork, i, name, &desc),
          strFormat("Failed to query NNPI network input for function %s",
                    functionName.c_str()));
      NNPIResourceDesc rDesc;
      LOG_IF_NOT_RETURN_LLVMERROR(
          NNPIResource::updateResourceDescFromTensorDesc(&rDesc, &desc),
          strFormat("Failed to update ResourceDesc for function %s",
                    functionName.c_str()));
      inputDesc_.push_back({name, rDesc});
    }
    for (size_t i = 0; i < numOutputs; i++) {
      LOG_NNPI_IF_ERROR_RETURN_LLVMERROR(
          nnpiNetworkGetOutputDesc(nnpiNetwork, i, name, &desc),
          strFormat("Failed to query NNPI network output for function %s",
                    functionName.c_str()));
      NNPIResourceDesc rDesc;
      LOG_IF_NOT_RETURN_LLVMERROR(
          NNPIResource::updateResourceDescFromTensorDesc(&rDesc, &desc),
          strFormat("Failed to update ResourceDesc for function %s",
                    functionName.c_str()));
      outputDesc_.push_back({name, rDesc});
    }
  }

  for (auto &infCtx : inferenceContexts_) {
    auto success = infCtx.init(
        inputDesc_, outputDesc_,
        nnpiCompiledFunction_->getCompiledNetworkHandle(),
        nnpiCompiledFunction_->getCompilationConfig(), deviceNetwork_, adapter,
        device, nnpiCompiledFunction_->getPartialInputs(),
        nnpiCompiledFunction_->getPaddedInputs(),
        nnpiCompiledFunction_->getStaticInputs(), staticPlaceholderMap_,
        deviceOptions_, functionName_, deviceId_);
    if (!success) {
      return MAKE_ERR(
          strFormat("Failed to initialize inferece context for function %s",
                    functionName.c_str()));
    }
    freeContexts_.push_back(&infCtx);
  }

  if (deviceOptions_->inferOnDevice && hostNetwork != NNPI_INVALID_NNPIHANDLE) {
    DBG_MEM_USAGE("call nnpiHostNetworkDestroy");
    LOG_NNPI_INF_IF_ERROR(
        nnpiHostNetworkDestroy(hostNetwork),
        strFormat("Failed to destroy NNPI host network for function %s",
                  functionName.c_str()));
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
    NNPIDeviceBindings *bindings =
        dynamic_cast<NNPIDeviceBindings *>(ctx->getDeviceBindings());
    if (bindings) {
      // TODO: verify with garret we don't need to lock here - i.e. host manager
      // can't invoke the same context twice in parallel.
      auto infCtx = bindings->getInferenceContext();
      CHECK(infCtx);
      infCtx->execute(runId, std::move(ctx), resultCB);
      return;
    }

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

InferenceContext *
InferencePoolEnv::createDetachedInferenceContext(PlaceholderUsageMap &phUsage) {
  if (deviceOptions_->dumpRuntime) {
    // Add function node to graph dump.
    std::ostringstream label;
    label << "{{";
    for (auto input : nnpiCompiledFunction_->getInputNames()) {
      label << "<" << input << ">" << input << "|";
    }
    label.seekp(-1, label.cur); // remove the trailing '|'
    label << "}|{"
          << "Function\\lname : " << functionName_ << "}|{";
    for (auto output : nnpiCompiledFunction_->getOutputNames()) {
      label << "<" << output << ">" << output << "|";
    }
    label.seekp(-1, label.cur); // remove the trailing '|'
    label << "}}";
    DotWriter::addNode(functionName_, label.str(), 0, std::to_string(device_));
  }

  InferenceContext *infCtx = new InferenceContext();

  if (!infCtx->init(
          inputDesc_, outputDesc_,
          nnpiCompiledFunction_->getCompiledNetworkHandle(),
          nnpiCompiledFunction_->getCompilationConfig(), deviceNetwork_,
          pAdapter_, device_, nnpiCompiledFunction_->getPartialInputs(),
          nnpiCompiledFunction_->getPaddedInputs(),
          nnpiCompiledFunction_->getStaticInputs(), staticPlaceholderMap_,
          deviceOptions_, functionName_, deviceId_, &phUsage)) {
    delete infCtx;
    ASSERT_WITH_MSG(infCtx, "Failed to initialize detached inference context");
    return nullptr;
  }

  return infCtx;
}

} // namespace runtime
} // namespace glow
