/**
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

#include "glow/Runtime/HostManager/HostManager.h"
#include "glow/Backends/DeviceManager.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Partitioner/Partitioner.h"
#include "glow/Runtime/Executor/Executor.h"
#include "glow/Runtime/Provisioner/Provisioner.h"
#include "glow/Runtime/RuntimeTypes.h"

#include <future>
#include <queue>

using namespace glow;
using namespace runtime;

HostManager::HostManager(const std::vector<DeviceManagerConfig> &configs) {
  // TODO: move all initialization out of constructor.
  TEMP_EXIT_ON_ERR(init(configs));
}

llvm::Error HostManager::init(const std::vector<DeviceManagerConfig> &configs) {
  DeviceIDTy deviceCount = 0;

  if (configs.size() > 0) {
    backend_.reset(createBackend(configs[0].backendKind));
  }
  for (auto &config : configs) {
    devices_[deviceCount] = std::unique_ptr<DeviceManager>(
        DeviceManager::createDeviceManager(config.backendKind, nullptr));

    RETURN_IF_ERR(devices_[deviceCount]->init());

    deviceCount++;
  }
  provisioner_.reset(new Provisioner(devices_));
  executor_.reset(createExecutor(devices_));

  return llvm::Error::success();
}

HostManager::~HostManager() { llvm::toString(clearHost()); }

llvm::Error HostManager::addNetwork(Module *M) {
  std::lock_guard<std::mutex> networkLock(networkLock_);
  std::vector<DeviceInfo> deviceInfo;
  for (auto &device : devices_) {
    DeviceInfo info = DeviceInfo();
    info.availableMemory = device.second->getAvailableMemory();
    deviceInfo.push_back(info);
  }
  // Optimize functions before passing to partitioner.
  // Currently hardcoding inference.
  if (backend_) {
    CompilationOptions opts;
    opts.mode = CompilationMode::Infer;
    for (auto F : M->getFunctions()) {
      backend_->optimizeFunction(F, opts);
    }
  }
  auto partitioner = Partitioner(M, deviceInfo);
  auto nodeList = std::move(partitioner.Partition());

  RETURN_IF_ERR(provisioner_->provision(nodeList, *M));

  for (auto &node : nodeList) {
    networks_.emplace((node.root)->name, std::move(node));
  }

  return llvm::Error::success();
}

void HostManager::removeNetwork(llvm::StringRef networkName) {
  std::lock_guard<std::mutex> networkLock(networkLock_);
  auto networkIterator = networks_.find(networkName);
  if (networkIterator == networks_.end()) {
    return;
  }
  networks_.erase(networkIterator);
}

bool HostManager::networkAdded(llvm::StringRef networkName) {
  std::lock_guard<std::mutex> networkLock(networkLock_);
  return networks_.find(networkName) != networks_.end();
}

llvm::Error HostManager::clearHost() {
  // shutdown the executor, blocking on any current inflight and prevent new
  // requests from being serviced.
  executor_->shutdown();
  assert(activeRequestCount_ == 0 &&
         "All requests should be finished when shutting down HostManager.");

  std::lock_guard<std::mutex> networkLock(networkLock_);
  OneErrOnly errContainer;
  for (auto &it : devices_) {
    errContainer.set(it.second->stop());
  }

  for (auto &network : networks_) {
    for (auto &node : network.second.nodes) {
      devices_[node->deviceID]->evictNetwork(node->name, /*evictCB=*/nullptr);
    }
  }
  networks_.clear();
  return errContainer.get();
}

RunIdentifierTy
HostManager::runNetwork(llvm::StringRef networkName,
                        std::unique_ptr<ExecutionContext> context,
                        ResultCBTy callback) {

  auto currentRun = totalRequestCount_++;
  std::lock_guard<std::mutex> networkLock(networkLock_);
  if (networks_.find(networkName) == networks_.end()) {
    callback(
        currentRun,
        MAKE_ERR(GlowErr::ErrorCode::RUNTIME_NET_NOT_FOUND,
                 llvm::formatv("Function {} not found", networkName).str()),
        std::move(context));
    return currentRun;
  }
  if (activeRequestCount_ >= activeRequestLimit_) {
    callback(currentRun,
             MAKE_ERR(GlowErr::ErrorCode::RUNTIME_REQUEST_REFUSED,
                      "The number of allowed requests has been exceeded."),
             std::move(context));
    return currentRun;
  }
  activeRequestCount_++;
  executor_->run(networks_[networkName].root.get(), std::move(context),
                 currentRun,
                 [&activeRequest = this->activeRequestCount_,
                  callback](RunIdentifierTy runID, llvm::Error err,
                            std::unique_ptr<ExecutionContext> context) {
                   --activeRequest;
                   callback(runID, std::move(err), std::move(context));
                 });
  return currentRun;
}
