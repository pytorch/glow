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

#include "NNPIDeviceManager.h"
#include "Importer.h"
#include "InferencePool.h"
#include "NNPI.h"
#include "NNPIAdapterContainer.h"
#include "NNPICompiledFunction.h"
#include "NNPITracing.h"
#include "NNPIUtils.h"
#include "glow/Flags/Flags.h"
#include "glow/Support/Error.h"
#include "nnpi_inference.h"
#include "nnpi_transformer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "DebugMacros.h"
#include "glow/Support/Error.h"

namespace glow {
namespace runtime {

DeviceManager *createNNPIDeviceManager(const DeviceConfig &config,
                                       NNPIAdapterContainer *adapter) {
  std::shared_ptr<NNPIDeviceOptions> deviceOptions =
      std::make_shared<NNPIDeviceOptions>(config.parameters);
  if (deviceOptions->inferOnDevice &&
      adapter->getHandle() == NNPI_INVALID_NNPIHANDLE) {
    LOG(ERROR) << "Adapter allocation failed";
    return nullptr;
  }
  return new NNPIDeviceManager(config, deviceOptions, adapter);
}

// 1K bytes.
static constexpr uint64_t KB = 1000;

//////////////////////////////////////////////////////////////////////////
std::atomic<RunIdentifierTy> NNPIDeviceManager::runIdentifier_;

NNPIDeviceManager::NNPIDeviceManager(
    const DeviceConfig &config,
    std::shared_ptr<NNPIDeviceOptions> deviceOptions,
    NNPIAdapterContainer *adapter)
    : DeviceManager(config), deviceId_(config_.deviceID), pAdapter_(adapter),
      device_(NNPI_INVALID_NNPIHANDLE), deviceOptions_(deviceOptions) {
  if (deviceOptions_->showVars) {
    LOG(INFO) << deviceOptions_->dumpStatus();
  }
  if (deviceOptions_->deviceId >= 0) {
    deviceId_ = static_cast<unsigned>(deviceOptions_->deviceId);
  }
  if (GlowNNPITimeout != 0) {
    deviceOptions_->inferTimeout = GlowNNPITimeout;
  }
}

NNPIDeviceManager::~NNPIDeviceManager() {
  std::unordered_set<std::string> functionsNames;
  for (auto func : functions_) {
    functionsNames.emplace(func.first);
  }

  for (auto func : functionsNames) {
    evictNetwork(func, [](std::string name, Error err) {
      LOG_IF(ERROR, ERR_TO_BOOL(std::move(err)))
          << "Failed to evict network during NNPIDeviceManager destructor";
    });
  }

  // Verify all static placeholders have no external refs.
  for (auto &res : staticPlaceholders_) {
    LOG_ERROR_IF_NOT(res.second.use_count() == 0)
        << "Static placeholder has pending refs";
  }

  // Verify all static placeholders have no external refs.
  for (auto &res : staticPlaceholders_) {
    LOG_ERROR_IF_NOT(res.second.use_count() == 0)
        << "Static placeholder has pending refs";
  }

  if (device_ != NNPI_INVALID_NNPIHANDLE) {
    LOG_NNPI_INF_IF_ERROR(nnpiDeviceContextDestroy(device_),
                          "Failed to destroy NNPI device context");
    device_ = NNPI_INVALID_NNPIHANDLE;
  }
}

Error NNPIDeviceManager::init() {
  LOG_IF_NOT_RETURN_LLVMERROR(device_ == NNPI_INVALID_NNPIHANDLE,
                              "Invalid NNPI device");

  NNPITransformerInfo info;
  CHECK_EQ(nnpiTransformerGetInfo(&info), NNPI_NO_ERROR);
  LOG(INFO) << "NNPI Transformer Version " << info.majorVersion << "."
            << info.minorVersion << "." << info.patchVersion << "."
            << info.minorPatchVersion;

  if (deviceOptions_->inferOnDevice) {
    // Create NNPI device.
    LOG_NNPI_INF_IF_ERROR_RETURN_LLVMERROR(
        nnpiDeviceContextCreate(pAdapter_->getHandle(), deviceId_, &device_),
        "Failed to create NNPI Device");
    NNPIDeviceInfo deviceInfo;
    LOG_NNPI_INF_IF_ERROR_RETURN_LLVMERROR(
        nnpiDeviceGetInfo(deviceId_, &deviceInfo),
        "Failed to get NNPI Device Info");
    maxMemoryBytes_ =
        static_cast<uint64_t>(deviceInfo.totalUnprotectedMemory) * KB;
    LOG(INFO) << "NNPI Driver Version "
              << static_cast<int>(deviceInfo.driverVersion.major) << "."
              << static_cast<int>(deviceInfo.driverVersion.minor) << "."
              << static_cast<int>(deviceInfo.driverVersion.dot);
    LOG(INFO) << "NNPI Firmware Version "
              << static_cast<int>(deviceInfo.fwVersion.major) << "."
              << static_cast<int>(deviceInfo.fwVersion.minor) << "."
              << static_cast<int>(deviceInfo.fwVersion.dot);
  }
  if (GlowNNPIMemory > 0) {
    maxMemoryBytes_ = static_cast<uint64_t>(GlowNNPIMemory) * KB;
  } else if (deviceOptions_->deviceMemory > 0) {
    maxMemoryBytes_ = static_cast<uint64_t>(deviceOptions_->deviceMemory) * KB;
  }

  runIdentifier_ = 0;
  return Error::success();
}

void NNPIDeviceManager::addNetwork(const Module *module,
                                   FunctionMapTy functions, ReadyCBTy readyCB) {
  std::unique_lock<std::mutex> lock(functionMapMutex_);
  // First check for uniqueness of the function name.
  for (const auto &func : functions) {
    if (functions_.count(func.first) != 0) {
      lock.unlock();
      readyCB(
          module,
          MAKE_ERR(
              llvm::formatv(
                  "Failed to add network: already have a function called {0}",
                  func.first)
                  .str()));
      return;
    }

    if (func.second->getCompileBackendName() != "NNPI") {
      lock.unlock();
      readyCB(module, MAKE_ERR(llvm::formatv("Failed to add network: function "
                                             "{0} is not a NNPIFunction",
                                             func.first)
                                   .str()));
      return;
    }
  }

  if (usedMemoryBytes_ + functionCost_ > maxMemoryBytes_) {
    lock.unlock();
    readyCB(module,
            MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_OUT_OF_DEVICE_MEMORY,
                     "Failed to add network: not enough memory"));
    return;
  }

  // Add to the function name lookup map.
  for (const auto &func : functions) {
    functions_.emplace(func.first, func.second);
    usedMemoryBytes_ += functionCost_; // TODO:: static moduleSize.
    auto err = inferencePools_[func.first].init(
        pAdapter_, device_, func.second, &staticPlaceholders_, deviceOptions_,
        func.first, deviceId_);
    if (err) {
      functions_.erase(func.first);
      lock.unlock();
      readyCB(module, std::move(err));
      return;
    }
  }

  LOG_IF(WARNING, usedMemoryBytes_ > maxMemoryBytes_)
      << "Using more memory than expected";

  // Fire the ready CB.
  lock.unlock();
  readyCB(module, Error::success());
}
void NNPIDeviceManager::evictNetwork(std::string functionName,

                                     EvictFunctionCBTy evictCB) {
  std::unique_lock<std::mutex> lock(functionMapMutex_);
  Error err = Error::success();

  if (functions_.erase(functionName)) {
    usedMemoryBytes_ -= functionCost_; // TODO: static moduleSize.
    inferencePools_.at(functionName)
        .stop(true); // First stop existing threads on this network.
    inferencePools_.erase(functionName);
  } else {
    err =
        MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_NET_NOT_FOUND,
                 llvm::formatv("Could not find function with name {0} to evict",
                               functionName)
                     .str());
  }

  // Remove unused static placeholders
  std::unordered_set<const Placeholder *> unusedPlaceholders;
  for (auto &sph : staticPlaceholders_) {
    if (sph.second.use_count() == 0) {
      unusedPlaceholders.insert(sph.first);
    }
  }
  for (auto *ph : unusedPlaceholders) {
    staticPlaceholders_.erase(ph);
  }

  lock.unlock();

  if (evictCB) {
    evictCB(functionName, std::move(err));
  } else {
    llvm::errs() << ERR_TO_STRING(std::move(err));
  }
}

RunIdentifierTy
NNPIDeviceManager::runFunction(std::string functionName,
                               std::unique_ptr<ExecutionContext> ctx,
                               runtime::ResultCBTy resultCB) {
  RunIdentifierTy runId = runIdentifier_++;

  // Get thread env.
  auto infEnv = inferencePools_.find(functionName);
  if (infEnv == inferencePools_.end()) {
    resultCB(runId, MAKE_ERR("Function isn't ready on the device"),
             std::move(ctx));
    return runId;
  }
  infEnv->second.execute(runId, std::move(ctx), std::move(resultCB));
  return runId;
}

Error NNPIDeviceManager::stop(bool block) {
  for (auto &env : inferencePools_) {
    env.second.stop(block);
  }
  return Error::success();
}
uint64_t NNPIDeviceManager::getMaximumMemory() const { return maxMemoryBytes_; }
uint64_t NNPIDeviceManager::getAvailableMemory() const {
  if (GlowNNPIMemory == 0 && deviceOptions_->deviceMemory == 0 &&
      deviceOptions_->inferOnDevice) {
    NNPIDeviceStatus devStatus;
    NNPIInferenceErrorCode res = nnpiDeviceGetStatus(deviceId_, &devStatus);
    if (res != NNPI_INF_NO_ERROR) {
      LOG_NNPI_INF_IF_ERROR(res, "Failed to read available memory from device.")
      return 0;
    }
    const auto availableMem =
        static_cast<uint64_t>(devStatus.availableUnprotectedMemory) * KB;
    if (availableMem == 0) {
      LOG(WARNING) << "NNPI Device " << deviceId_
                   << " available memory: " << availableMem;
    }
    return availableMem;
  }
  auto freeMemory = getMaximumMemory();
  for (const auto &p : functions_) {
    const auto &fn = p.second;
    const auto &bundle = fn->getRuntimeBundle();
    freeMemory -= bundle.getConstantWeightSize();
    freeMemory -= bundle.getMutableWeightSize();
  }
  return freeMemory;
}
bool NNPIDeviceManager::isMemoryAvailable(uint64_t estimate) const {
  return estimate <= getAvailableMemory(); // This is just an estimate and not
                                           // necessarily accurate.
}

void NNPIDeviceManager::transferStaticPlaceholderToDevice(
    Placeholder *PH, Tensor *T, std::function<void(Error)> resultCB) {
  LOG_AND_FAIL_CALLBACK_IF_NOT(
      staticPlaceholders_.count(PH) != 0,
      "Static placeholder does not exist on the device", resultCB);

  auto nnpiResource = staticPlaceholders_.at(PH).lock();
  LOG_AND_FAIL_CALLBACK_IF_NOT(
      nnpiResource != nullptr,
      "Static placeholder no longer exists on the device", resultCB);

  nnpiResource->updateDeviceResourceFromTensor(T, resultCB);
};

Error NNPIDeviceManager::startDeviceTrace(TraceContext *traceContext) {
  if (!NNPIDeviceTracing::getForDevice(deviceId_)->start(
          traceContext, device_, true /* Software traces are always enabled. */,
          deviceOptions_->hardwareTraces,
          deviceOptions_->softwareTracesMaxBuffer,
          deviceOptions_->hardwareTracesMaxBuffer,
          deviceOptions_->rawTracesDumpPath)) {
    return MAKE_ERR("Failed to start NNPI device trace.");
  }
  return Error::success();
}

Error NNPIDeviceManager::stopDeviceTrace(TraceContext *traceContext) {
  if (!NNPIDeviceTracing::getForDevice(deviceId_)->stopAndUpdate(traceContext,
                                                                 device_)) {
    return MAKE_ERR("Failed to stop NNPI device trace.");
  }
  return Error::success();
}

Error NNPIDeviceManager::bindContext(std::string functionName,
                                     ExecutionContext *ctx,
                                     PlaceholderUsageMap &phUsage) {
  if (deviceOptions_->dumpRuntime) {
    DotWriter::addSubGraph(std::to_string(device_),
                           std::string("Device ") + std::to_string(deviceId_) +
                               " (" + DotWriter::getHexStr(device_) + ")");
  }

  // Create inference context.
  ASSERT_WITH_MSG(inferencePools_.count(functionName),
                  "Invalid function name.");
  std::shared_ptr<InferenceContext> infCtx(
      inferencePools_.at(functionName).createDetachedInferenceContext(phUsage));
  ASSERT_WITH_MSG(
      infCtx, "Failed to create detached context; NNPIDeviceManager status: " +
                  getStatusStr() +
                  "; with NNPIDeviceOptions: " + deviceOptions_->dumpStatus());

  // Set the inference context into NNPIDeviceBinding and store in the ExCtx.
  ctx->setDeviceBindings(std::make_unique<NNPIDeviceBindings>(infCtx));
  return Error::success();
}

void NNPIDeviceManager::addPlaceholderUsageCount(std::string functionName,
                                                 PlaceholderUsageMap &phUsage) {
  if (functions_.count(functionName)) {
    NNPICompiledFunction *func =
        dynamic_cast<NNPICompiledFunction *>(functions_.at(functionName));
    ASSERT_WITH_MSG(func, "Invalid function.");
    for (auto inputName : func->getInputNames()) {
      phUsage[inputName].numReaders++;
      phUsage[inputName].devices.insert(device_);
    }
    for (auto outputName : func->getOutputNames()) {
      phUsage[outputName].numWriters++;
      phUsage[outputName].devices.insert(device_);
    }
  }
}

void *NNPIDeviceManager::allocateDeviceIOBuffer(dim_t size) {
  if (deviceOptions_->inferOnDevice && !deviceOptions_->disableDeviceIOBuffer) {
    return pAdapter_->allocateHostResource(size);
  } else {
    return DeviceManager::allocateDeviceIOBuffer(size);
  }
}

void NNPIDeviceManager::freeAllocatedDeviceIOBuffer(void *buffer) {
  if (deviceOptions_->inferOnDevice && !deviceOptions_->disableDeviceIOBuffer) {
    return pAdapter_->freeHostResource(buffer);
  } else {
    return DeviceManager::freeAllocatedDeviceIOBuffer(buffer);
  }
}

std::string NNPIDeviceManager::getStatusStr() const {
  std::stringstream stream;
  stream << "MaximumMemory: \"" << getMaximumMemory() << '"';
  stream << ", AvailableMemory: \"" << getAvailableMemory() << '"';
  stream << ", DeviceID: \"" << deviceId_ << '"';
  stream << ", Functions: {";
  for (const auto &func : functions_) {
    stream << func.first << ",";
  }
  stream << "}, ";
  stream << ", FunctionCost: \"" << functionCost_ << '"';
  stream << ", RunIdentifier: \"" << runIdentifier_ << '"';
  return stream.str();
}

} // namespace runtime
} // namespace glow
