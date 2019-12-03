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

unsigned GlowNNPIMemory = 16 << 20; // 16 GB.

static llvm::cl::opt<unsigned, /* ExternalStorage */ true>
    GlowNNPIMemoryOpt("glow-nnpi-memory",
                      llvm::cl::desc("Override the amount of DRAM to allocate "
                                     "per NNPI device, in kilobytes"),
                      llvm::cl::location(GlowNNPIMemory));

DeviceManager *createNNPIDeviceManager(const DeviceConfig &config) {
  return new NNPIDeviceManager(config);
}

//////////////////////////////////////////////////////////////////////////
std::atomic<RunIdentifierTy> NNPIDeviceManager::runIdentifier_;

NNPIDeviceManager::NNPIDeviceManager(const DeviceConfig &config,
                                     unsigned numInferenceWorkers)
    : DeviceManager(config), numWorkersPerFunction_(numInferenceWorkers),
      deviceId_(config_.deviceID), adapter_(NNPI_INVALID_NNPIHANDLE),
      device_(NNPI_INVALID_NNPIHANDLE) {
  auto it = config_.parameters.find("DeviceID");
  if (it != config_.parameters.end()) {
    // Todo: check device id is appropriate for the machine (check adapter for
    // active devices).
    deviceId_ = std::stoul(it->second);
  }
  const auto envDeviceId = EnvDeviceID();
  if (envDeviceId.length() > 0) {
    // Override if exists in environment variable.
    deviceId_ = std::stoul(envDeviceId);
  }

  if (!numWorkersPerFunction_) {
    numWorkersPerFunction_ =
        UseInferenceAPI()
            ? 2
            : 1; // Ice-ref not re-entrant for the same nnpiNetwork.
  }
  const auto envWorkers = EnvNumWorkers();
  if (envWorkers > 0) {
    numWorkersPerFunction_ = envWorkers;
  }
}

NNPIDeviceManager::~NNPIDeviceManager() {
  if (device_ != NNPI_INVALID_NNPIHANDLE) {
    LOG_NNPI_INF_ERROR(nnpiDeviceContextDestroy(device_),
                       "Failed to destroy NNPI device context");
    device_ = NNPI_INVALID_NNPIHANDLE;
  }

  if (adapter_ != NNPI_INVALID_NNPIHANDLE) {
    LOG_NNPI_INF_ERROR(nnpiAdapterDestroy(adapter_),
                       "Failed to destroy NNPI adapter");
    adapter_ = NNPI_INVALID_NNPIHANDLE;
  }
}

Error NNPIDeviceManager::init() {
  LOG_IF_NOT_RETURN_LLVMERROR(adapter_ == NNPI_INVALID_NNPIHANDLE,
                              "Bad NNPI adapter");
  LOG_IF_NOT_RETURN_LLVMERROR(device_ == NNPI_INVALID_NNPIHANDLE,
                              "Bad NNPI device");

  NNPITransformerInfo info;
  CHECK_EQ(nnpiTransformerGetInfo(&info), NNPI_NO_ERROR);
  LOG(INFO) << "NNPI Transformer Version " << info.majorVersion << "."
            << info.minorVersion << "." << info.patchVersion;

  if (UseInferenceAPI()) {
    // Create NNPI adapter.
    LOG_NNPI_INF_ERROR_RETURN_LLVMERROR(nnpiAdapterCreate(nullptr, &adapter_),
                                        "Failed to create NNPI Adapter");

    // Create NNPI device.
    LOG_NNPI_INF_ERROR_RETURN_LLVMERROR(
        nnpiDeviceContextCreate(adapter_, deviceId_, &device_),
        "Failed to create NNPI Device");
    if (EnabledDeviceTracing()) {
      deviceTracing_ = NNPIDeviceTracing::getForDevice(deviceId_);
    }
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
        numWorkersPerFunction_, adapter_, device_, deviceTracing_, func.second);
    if (err) {
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
uint64_t NNPIDeviceManager::getMaximumMemory() const {
  return uint64_t{GlowNNPIMemory} * 1024;
  // Todo: use nnpiDeviceGetInfo.
}
uint64_t NNPIDeviceManager::getAvailableMemory() const {
  auto freeMemory = getMaximumMemory();
  for (const auto &p : functions_) {
    const auto &fn = p.second;
    const auto &bundle = fn->getRuntimeBundle();
    freeMemory -= bundle.getConstantWeightSize();
    freeMemory -= bundle.getMutableWeightSize();
  }
  return freeMemory;
  // Todo: use nnpiDeviceGetStatus.
}
bool NNPIDeviceManager::isMemoryAvailable(uint64_t estimate) const {
  return estimate <= getAvailableMemory(); // This is just an estimate and not
                                           // necessarily accurate.
}
} // namespace runtime
} // namespace glow
