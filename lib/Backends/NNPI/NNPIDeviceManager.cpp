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
#include "NNPICompiledFunction.h"
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

unsigned GlowNNPIMemory = 0;

static llvm::cl::opt<unsigned, /* ExternalStorage */ true>
    GlowNNPIMemoryOpt("glow-nnpi-memory",
                      llvm::cl::desc("Override the amount of DRAM to allocate "
                                     "per NNPI device, in kilobytes"),
                      llvm::cl::location(GlowNNPIMemory));

DeviceManager *createNNPIDeviceManager(const DeviceConfig &config) {
  return new NNPIDeviceManager(config);
}

// 1K bytes.
static constexpr uint64_t KB = 1 << 10;

//////////////////////////////////////////////////////////////////////////
std::atomic<RunIdentifierTy> NNPIDeviceManager::runIdentifier_;

NNPIDeviceManager::NNPIDeviceManager(const DeviceConfig &config,
                                     unsigned numInferenceWorkers)
    : DeviceManager(config), numWorkersPerFunction_(numInferenceWorkers),
      deviceId_(config_.deviceID), adapter_(NNPI_INVALID_NNPIHANDLE),
      device_(NNPI_INVALID_NNPIHANDLE), deviceOptions_(config_.parameters) {
  if (deviceOptions_.showVars) {
    LOG(INFO) << deviceOptions_.dumpStatus();
  }
  if (deviceOptions_.deviceID >= 0) {
    deviceId_ = static_cast<unsigned>(deviceOptions_.deviceID);
  }

  if (!numWorkersPerFunction_) {
    numWorkersPerFunction_ =
        deviceOptions_.inferOnDevice
            ? 2
            : 1; // Ice-ref not re-entrant for the same nnpiNetwork.
  }

  if (deviceOptions_.numWorkers > 0) {
    numWorkersPerFunction_ = deviceOptions_.numWorkers;
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

  if (device_ != NNPI_INVALID_NNPIHANDLE ||
      !deviceOptions_.internalTesting.get().empty()) {
    LOG_NNPI_INF_ERROR(nnpiDeviceContextDestroy(device_),
                       "Failed to destroy NNPI device context");
    device_ = NNPI_INVALID_NNPIHANDLE;
  }

  if (adapter_ != NNPI_INVALID_NNPIHANDLE ||
      !deviceOptions_.internalTesting.get().empty()) {
    LOG_NNPI_INF_ERROR(nnpiAdapterDestroy(adapter_),
                       "Failed to destroy NNPI adapter");
    adapter_ = NNPI_INVALID_NNPIHANDLE;
  }
}

Error NNPIDeviceManager::init() {
  if (!deviceOptions_.internalTesting.get().empty()) {
    LOG_IF_NOT_RETURN_LLVMERROR(adapter_ == NNPI_INVALID_NNPIHANDLE,
                                "Invalid NNPI adapter");
    LOG_IF_NOT_RETURN_LLVMERROR(device_ == NNPI_INVALID_NNPIHANDLE,
                                "Invalid NNPI device");
  }

  NNPITransformerInfo info;
  CHECK_EQ(nnpiTransformerGetInfo(&info), NNPI_NO_ERROR);
  LOG(INFO) << "NNPI Transformer Version " << info.majorVersion << "."
            << info.minorVersion << "." << info.patchVersion;

  if (deviceOptions_.inferOnDevice) {
    // Create NNPI adapter.
    LOG_NNPI_INF_ERROR_RETURN_LLVMERROR(nnpiAdapterCreate(nullptr, &adapter_),
                                        "Failed to create NNPI Adapter");

    // Create NNPI device.
    LOG_NNPI_INF_ERROR_RETURN_LLVMERROR(
        nnpiDeviceContextCreate(adapter_, deviceId_, &device_),
        "Failed to create NNPI Device");
    LOG_IF_NOT_RETURN_LLVMERROR(
        staticPlaceholderContainer_.SetDevice(
            device_, !deviceOptions_.internalTesting.get().empty()),
        "setting device for StaticPlaceholderContainer failed");
    if (deviceOptions_.enabledDeviceTracing) {
      deviceTracing_ = NNPIDeviceTracing::getForDevice(deviceId_);
    }
    NNPIDeviceInfo deviceInfo;
    LOG_NNPI_INF_ERROR_RETURN_LLVMERROR(
        nnpiDeviceGetInfo(deviceId_, &deviceInfo),
        "Failed to get NNPI Device Info");
    maxMemoryBytes_ =
        static_cast<uint64_t>(deviceInfo.totalUnprotectedMemory) * KB;
    LOG(INFO) << "NNPI Driver Version "
              << static_cast<int>(deviceInfo.driverVersion.major) << "."
              << static_cast<int>(deviceInfo.driverVersion.minor) << "."
              << static_cast<int>(deviceInfo.driverVersion.dot);
    LOG(INFO) << "NNPI Fireware Version "
              << static_cast<int>(deviceInfo.fwVersion.major) << "."
              << static_cast<int>(deviceInfo.fwVersion.minor) << "."
              << static_cast<int>(deviceInfo.fwVersion.dot);
  }
  if (GlowNNPIMemory > 0) {
    maxMemoryBytes_ = static_cast<uint64_t>(GlowNNPIMemory) * KB;
  } else if (deviceOptions_.deviceMemory > 0) {
    maxMemoryBytes_ = static_cast<uint64_t>(deviceOptions_.deviceMemory) * KB;
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
    auto err = inferenceEnvs_[func.first].init(
        numWorkersPerFunction_, adapter_, device_, deviceTracing_, func.second,
        &staticPlaceholderContainer_, deviceOptions_);
    if (err) {
      functions_.erase(func.first);
      lock.unlock();
      readyCB(module, std::move(err));
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
    inferenceEnvs_.at(functionName)
        .stop(true); // First stop existing threads on this network.
    inferenceEnvs_.erase(functionName);
  } else {
    err =
        MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_NET_NOT_FOUND,
                 llvm::formatv("Could not find function with name {0} to evict",
                               functionName)
                     .str());
  }
  lock.unlock();

  if (evictCB) {
    evictCB(functionName, std::move(err));
  } else {
    llvm::errs() << errorToString(std::move(err));
  }
}

RunIdentifierTy
NNPIDeviceManager::runFunction(std::string functionName,
                               std::unique_ptr<ExecutionContext> ctx,
                               runtime::ResultCBTy resultCB) {
  RunIdentifierTy runId = runIdentifier_++;

  /// NNPI DeviceManager doesn't support Device Resident Tensors.
  ctx->getPlaceholderBindings()->ensureOnHost();

  // Get thread env.
  auto infEnv = inferenceEnvs_.find(functionName);
  if (infEnv == inferenceEnvs_.end()) {
    resultCB(runId, MAKE_ERR("Function isn't ready on the device"),
             std::move(ctx));
    return runId;
  }
  infEnv->second.execute(runId, std::move(ctx), std::move(resultCB));
  return runId;
}

Error NNPIDeviceManager::stop(bool block) {
  for (auto &env : inferenceEnvs_) {
    env.second.stop(block);
  }
  return Error::success();
}
uint64_t NNPIDeviceManager::getMaximumMemory() const { return maxMemoryBytes_; }
uint64_t NNPIDeviceManager::getAvailableMemory() const {
  if (GlowNNPIMemory == 0 && deviceOptions_.deviceMemory == 0 &&
      deviceOptions_.inferOnDevice) {
    NNPIDeviceStatus devStatus;
    NNPIInferenceErrorCode res = nnpiDeviceGetStatus(deviceId_, &devStatus);
    if (res != NNPI_INF_NO_ERROR) {
      LOG_NNPI_INF_ERROR(res, "Failed to read available memory from device.")
      return 0;
    }
    return static_cast<uint64_t>(devStatus.availableUnprotectedMemory) * KB;
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

  NNPIHostResource hInput;
  NamedResource nr;
  nr = staticPlaceholderContainer_.AcquireDeviceResource(PH, nr);
  if (deviceOptions_.internalTesting.get().empty()) {
    LOG_AND_FAIL_CALLBACK_IF_NOT(nr.handle != NNPI_INVALID_NNPIHANDLE,
                                 "Failed to acquire device resource", resultCB);
  }

  LOG_AND_CALLBACK_NNPI_INF_ERROR(
      nnpiHostResourceCreate(adapter_, &nr.desc, &hInput),
      "Failed to create NNPI host resource", resultCB);

  void *pHostInput(nullptr);
  LOG_AND_CALLBACK_NNPI_INF_ERROR(
      nnpiHostResourceLock(hInput, NNPI_LOCK_FOR_WRITE, UINT32_MAX,
                           &pHostInput),
      "Failed to create NNPI host resource", resultCB);

  size_t bufferSize = T->getUnpaddedSizeInBytes();
  size_t fullBufferSize = T->getSizeInBytes();

  switch (T->getElementType()) {
  case glow::ElemKind::Int64ITy: {
    // Convert int64_t tensors to int32.
    int64_t *pInput = reinterpret_cast<int64_t *>(T->getUnsafePtr());
    const size_t unpaddedSize = bufferSize / sizeof(int64_t);
    int32_t *tmp = new int32_t[unpaddedSize];
    for (size_t i = 0; i < unpaddedSize; i++) {
      tmp[i] = static_cast<int32_t>(pInput[i]);
    }
    bufferSize /= 2;
    fullBufferSize /= 2;
    std::memcpy(pHostInput, tmp, bufferSize);
    delete[](tmp);
  } break;
  default:
    std::memcpy(pHostInput, T->getUnsafePtr(), bufferSize);
  }

  LOG_AND_CALLBACK_NNPI_INF_ERROR(nnpiHostResourceUnlock(hInput),
                                  "Failed to unlock host resource", resultCB);

  NNPICopyCommand copyInputCmd(NNPI_INVALID_NNPIHANDLE);
  LOG_AND_CALLBACK_NNPI_INF_ERROR(
      nnpiCopyCommandCreateHostToDevice(device_, nr.handle, hInput,
                                        &copyInputCmd),
      "Failed to create NNPI copy command", resultCB);

  if (deviceOptions_.enabledCommandLists > 1) {
    // Create command list.
    NNPICommandList cmdList = NNPI_INVALID_NNPIHANDLE;
    NNPICommandHandle cmdHnd;
    cmdHnd.type = NNPI_COMMAND_TYPE_COPY;
    cmdHnd.copyCommand = copyInputCmd;
    LOG_AND_CALLBACK_NNPI_INF_ERROR(
        nnpiCommandListCreate(&cmdHnd, 1, nullptr, 0, &cmdList),
        "Failed to create NNPI command list", resultCB);

    // Queue command list.
    LOG_AND_CALLBACK_NNPI_INF_ERROR(nnpiCommandListQueue(cmdList, nullptr, 0),
                                    "Failed to queue command list.", resultCB);

    // Wait for completion.
    uint32_t numErrors(0);
    NNPICommandListError singleError;
    memset(&singleError, 0, sizeof(NNPICommandListError));
    NNPIInferenceErrorCode res =
        nnpiCommandListWait(cmdList, UINT32_MAX, &singleError, 1, &numErrors);

    if (res != NNPI_INF_NO_ERROR) {
      LOG_NNPI_INF_ERROR(res, "Failed to wait on command list");
    } else {
      if (numErrors > 0) {
        LOG(ERROR) << NNPI_INF_ERROR_MSG(singleError.err, singleError.desc);
      }
    }
    if (res != NNPI_INF_NO_ERROR || numErrors > 0) {
      LOG_AND_CALLBACK_NNPI_INF_ERROR(
          res, "Errors detected during command list", resultCB);
    }

    // Destroy command list.
    LOG_NNPI_INF_ERROR(nnpiCommandListDestroy(cmdList),
                       "Failed to destroy NNPI command list");
  } else {
    LOG_AND_CALLBACK_NNPI_INF_ERROR(nnpiCopyCommandQueue(copyInputCmd, nullptr),
                                    "Failed to queue input copy command.",
                                    resultCB);
    LOG_AND_CALLBACK_NNPI_INF_ERROR(
        nnpiHostResourceLock(hInput, NNPI_LOCK_FOR_WRITE, UINT32_MAX,
                             &pHostInput),
        "Failed to lock host resource during static Placeholder transfer",
        resultCB);
    LOG_AND_CALLBACK_NNPI_INF_ERROR(nnpiHostResourceUnlock(hInput),
                                    "Failed to unlock host resource", resultCB);
  }

  LOG_AND_CALLBACK_NNPI_INF_ERROR(nnpiCopyCommandDestroy(copyInputCmd),
                                  "Failed to destroy NNPI copy command",
                                  resultCB);

  LOG_AND_CALLBACK_NNPI_INF_ERROR(nnpiHostResourceDestroy(hInput),
                                  "Failed to destroy NNPI host resource",
                                  resultCB);

  LOG_AND_FAIL_CALLBACK_IF_NOT(
      staticPlaceholderContainer_.ReleaseDeviceResource(PH),
      "Failed to release device resource", resultCB);

  resultCB(Error::success());
};

NNPIStaticPlaceholderContainer::~NNPIStaticPlaceholderContainer() {
  LOG_IF_NOT(ERROR, staticPlaceholdersDeviceResource_.size() == 0)
      << "NNPIStaticPlaceholderContainer contains allocated refs for device "
         "resource";
  for (auto item : staticPlaceholdersDeviceResource_) {
    auto PH = item.first;
    EraseAndDestroyDeviceResource_(PH);
  }
}

bool NNPIStaticPlaceholderContainer::SetDevice(NNPIDeviceContext device,
                                               bool inferOnRuntime) {
  // Exception for internal testing (ICE-24091)
  if (!inferOnRuntime) {
    LOG_AND_RETURN_IF(ERROR, device == NNPI_INVALID_NNPIHANDLE,
                      "NNPIStaticPlaceholderContainer received invalid device",
                      false);
  }
  device_ = device;
  return true;
}

NamedResource
NNPIStaticPlaceholderContainer::AcquireDeviceResource(const Placeholder *PH,
                                                      const NamedResource &nr) {
  if (staticPlaceholdersDeviceResource_.count(PH) == 0) {
    NamedResourceWithRef nrf = nr;
    LOG_NNPI_INF_ERROR(
        nnpiDeviceResourceCreate(device_, &nrf.desc, &nrf.handle),
        "Failed to create NNPI device resource");
    staticPlaceholdersDeviceResource_[PH] = nrf;
  }
  auto nrf = staticPlaceholdersDeviceResource_.at(PH);

  nrf.refCount += 1;
  staticPlaceholdersDeviceResource_[PH] = nrf;
  return nrf;
}

bool NNPIStaticPlaceholderContainer::EraseAndDestroyDeviceResource_(
    const Placeholder *PH) {
  LOG_AND_RETURN_IF_NOT(ERROR, staticPlaceholdersDeviceResource_.count(PH),
                        "Resource with name:" + PH->getName().str() +
                            " wasn't initialized as static Placeholder",
                        false)
  auto &nrf = staticPlaceholdersDeviceResource_.at(PH);
  LOG_NNPI_INF_ERROR_RETURN_FALSE(nnpiDeviceResourceDestroy(nrf.handle),
                                  "Failed to destroy NNPI device resource");
  nrf.handle = NNPI_INVALID_NNPIHANDLE;
  staticPlaceholdersDeviceResource_.erase(PH);
  return true;
}

bool NNPIStaticPlaceholderContainer::ReleaseDeviceResource(
    const Placeholder *PH) {
  LOG_AND_RETURN_IF_NOT(ERROR, staticPlaceholdersDeviceResource_.count(PH),
                        "Resource with name:" + PH->getName().str() +
                            " wasn't initialized as static Placeholder",
                        false)

  auto &nrf = staticPlaceholdersDeviceResource_.at(PH);
  LOG_IF_NOT(ERROR, nrf.refCount > 0)
      << "ref count for resource with name:" << PH->getName().str()
      << " is already 0";
  if (nrf.refCount > 0) {
    nrf.refCount -= 1;
  }

  if (nrf.refCount == 0) {
    return EraseAndDestroyDeviceResource_(PH);
  }
  return true;
}

} // namespace runtime
} // namespace glow
