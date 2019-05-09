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
#include "glow/Optimizer/Optimizer.h"
#include "glow/Partitioner/Partitioner.h"
#include "glow/Runtime/Executor/Executor.h"
#include "glow/Runtime/Provisioner/Provisioner.h"
#include "glow/Runtime/RuntimeTypes.h"

#include <future>
#include <queue>

using namespace glow;
using namespace runtime;

HostManager::HostManager(std::vector<std::unique_ptr<DeviceConfig>> configs) {
  // TODO: move all initialization out of constructor.
  TEMP_EXIT_ON_ERR(init(std::move(configs)));
}

llvm::Error
HostManager::init(std::vector<std::unique_ptr<DeviceConfig>> configs) {
  DeviceIDTy deviceCount = 0;

  if (configs.size() > 0) {
    backend_.reset(createBackend(configs[0]->getBackendKind()));
  }

  for (auto &config : configs) {
    if (!config->hasName()) {
      config->setName(backend_->getBackendName() + std::to_string(deviceCount));
    }

    auto backendKind = config->getBackendKind();
    devices_[deviceCount] = std::unique_ptr<DeviceManager>(
        DeviceManager::createDeviceManager(backendKind, std::move(config)));

    RETURN_IF_ERR(devices_[deviceCount]->init());

    deviceCount++;
  }
  provisioner_.reset(new Provisioner(devices_));
  executor_.reset(createExecutor(devices_));

  return llvm::Error::success();
}

HostManager::~HostManager() { llvm::toString(clearHost()); }

llvm::Error HostManager::addNetwork(std::unique_ptr<Module> module,
                                    bool saturateHost) {
  std::lock_guard<std::mutex> networkLock(networkLock_);
  auto functions = module->getFunctions();
  for (auto &F : functions) {
    std::string name = F->getName();
    auto it = networks_.find(name);
    if (it != networks_.end()) {
      return MAKE_ERR(GlowErr::ErrorCode::RUNTIME_ERROR,
                      "Failed to add network: already have a function called " +
                          name);
    }
  }
  std::vector<DeviceInfo> deviceInfo;
  for (auto &device : devices_) {
    DeviceInfo info = DeviceInfo();
    info.availableMemory = device.second->getAvailableMemory();
    deviceInfo.push_back(info);
  }
  // Optimize functions before passing to partitioner.
  // Currently hardcoding inference.
  if (backend_) {
    CompilationContext cctx;
    cctx.mode = CompilationMode::Infer;
    for (auto F : module->getFunctions()) {
      RETURN_IF_ERR(::glow::optimizeFunction(F, *backend_, cctx));
    }
  }
  auto partitioner = Partitioner(module.get(), deviceInfo, saturateHost);
  RETURN_IF_ERR(partitioner.Partition());
  auto nodeList = std::move(partitioner.getPartitionResult());

  RETURN_IF_ERR(provisioner_->provision(nodeList, *module));

  // Clear functions and constants contents from the module then put it a
  // shared_ptr to be shared between all of the networks created from each
  // function in the module.
  module->strip();
  auto sharedModule = std::shared_ptr<Module>(std::move(module));

  for (auto &node : nodeList) {
    auto &networkData = networks_[(node.root)->name];
    networkData.dag = std::move(node);
    networkData.module = sharedModule;
  }

  return llvm::Error::success();
}

void HostManager::removeNetwork(llvm::StringRef networkName) {
  std::lock_guard<std::mutex> networkLock(networkLock_);
  auto networkIterator = networks_.find(networkName);
  if (networkIterator == networks_.end()) {
    return;
  }

  // TODO: should return an error here rather than asserting:
  // https://github.com/pytorch/glow/issues/2855
  assert(networkIterator->second.refcount == 0 &&
         "Cannot remove a network since there are outstanding runs");

  auto &nodes = networkIterator->second.dag.nodes;
  for (auto &node : nodes) {
    for (auto device : node->deviceIDs) {
      std::promise<void> removeNetwork;
      llvm::Error removeErr = llvm::Error::success();
      auto done = removeNetwork.get_future();
      devices_[device]->evictNetwork(
          node->name,
          [&removeNetwork, &removeErr](std::string name, llvm::Error err) {
            removeErr = std::move(err);
            removeNetwork.set_value();
          });
      done.get();
      errToBool(std::move(removeErr));
    }
    // Also remove compiledFunction from Provisioner.
    provisioner_->removeFunction(node->name);
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
    for (auto &node : network.second.dag.nodes) {
      for (auto device : node->deviceIDs) {
        devices_[device]->evictNetwork(node->name);
      }
    }
  }
  networks_.clear();
  return errContainer.get();
}

RunIdentifierTy
HostManager::runNetwork(llvm::StringRef networkName,
                        std::unique_ptr<ExecutionContext> context,
                        ResultCBTy callback) {
  TRACE_EVENT_SCOPE(context->getTraceContext(), "HostManager::runNetwork");
  auto currentRun = totalRequestCount_++;

  NetworkData *network = nullptr;
  {
    std::lock_guard<std::mutex> networkLock(networkLock_);
    auto it = networks_.find(networkName);
    if (it != networks_.end()) {
      network = &it->second;
      network->refcount++;
    }
  }

  if (network == nullptr) {
    callback(
        currentRun,
        MAKE_ERR(GlowErr::ErrorCode::RUNTIME_NET_NOT_FOUND,
                 llvm::formatv("Function {0} not found", networkName).str()),
        std::move(context));
    return currentRun;
  }

  size_t activeRequestCount = activeRequestCount_++;
  if (activeRequestCount >= activeRequestLimit_) {
    activeRequestCount_--;
    network->refcount--;
    callback(
        currentRun,
        MAKE_ERR(GlowErr::ErrorCode::RUNTIME_REQUEST_REFUSED,
                 strFormat("The number of allowed requests has been exceeded. "
                           "active requests: %lu allowed requests: %u",
                           activeRequestCount, activeRequestLimit_)),
        std::move(context));
    return currentRun;
  }

  executor_->run(
      networks_[networkName].dag.root.get(), std::move(context), currentRun,
      [this, callback,
       name = networkName.str()](RunIdentifierTy runID, llvm::Error err,
                                 std::unique_ptr<ExecutionContext> context) {
        --activeRequestCount_;
        {
          std::lock_guard<std::mutex> networkLock(networkLock_);
          auto it = networks_.find(name);
          if (it != networks_.end()) {
            it->second.refcount--;
          }
        }
        TRACE_EVENT_INSTANT(context->getTraceContext(), "finish_" + name);
        callback(runID, std::move(err), std::move(context));
      });
  return currentRun;
}
